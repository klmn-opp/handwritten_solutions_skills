"""Finite real-patch diffusion experiment, entirely outside production Runtime."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    config = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "steps": 6000,
        "batch": 8,
        "lr": 0.0002,
        "seed": 3014,
        "generation_seeds": [2012, 2013],
        "reverse_steps": 64,
        "architecture": "local-full-resolution-ink-diffusion/1",
        "endpoint_probability": 0.02,
        "input_patch": 128,
        "loss_patch": 64,
        "receptive_field_per_step": 63,
        "frozen_codec": False,
        "real_resampled_targets_only": True,
        "font_supervision": False,
        "font_condition": True,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01014_PROTOCOL.md")),
    }
    run = create("local-stochastic-ink", config=config, seed=config["seed"])
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.local_diffusion_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: Path):
    import importlib.metadata
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.adversarial_ink_model import word_patches
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.local_diffusion_model import (
        LocalInkDiffusion,
        generate_diffusion,
        noise_schedule,
    )

    r = json.loads(request.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    assert digest(data / "run.json") == r["data_sha256"]
    assert digest(data / "reports/arrays.npz") == r["arrays_sha256"]
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    train = idx["train"]
    evaluated = train + idx["development"] + idx["novel"]
    assert not set(train) & set(idx["development"] + idx["novel"])
    assert all(rows[i]["dataset_split"] == "train" for i in train)
    raw = np.load(data / "reports/arrays.npz")
    reference = torch.tensor(raw["reference"][train, None], device="cuda")
    targets = torch.tensor(raw["target"][train, None], device="cuda")
    points = [np.argwhere(x > 0.5) for x in raw["reference"][train]]
    all_ref = torch.tensor(raw["reference"][:, None], device="cuda")
    model = LocalInkDiffusion().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    noise_rng = torch.Generator(device="cuda").manual_seed(r["seed"])
    counts = np.zeros(len(train), dtype=np.int64)
    traces, curve = [], []
    write(run / "reports/indices.json", {**idx, "evaluated": evaluated})
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(len(train), r["batch"], replace=True)
        np.add.at(counts, ii, 1)
        centers = [(int(i), *map(int, points[i][rng.integers(len(points[i]))])) for i in ii]
        traces.append(centers)
        ref = word_patches(reference, centers, 64)
        target = word_patches(targets, centers, 64)
        timepoint = 0.005 + 0.99 * torch.rand(len(ii), device="cuda", generator=noise_rng)
        timepoint[
            torch.rand(len(ii), device="cuda", generator=noise_rng) < r["endpoint_probability"]
        ] = 1
        alpha, sigma = noise_schedule(timepoint)
        noise = torch.randn(target.shape, device="cuda", generator=noise_rng)
        clean = 2 * target - 1
        velocity = alpha * noise - sigma * clean
        prediction = model(alpha * clean + sigma * noise, ref, timepoint)
        loss = ((prediction - velocity).square() * (1 + 4 * target))[..., 32:96, 32:96].mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diffusion objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append(
                {"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - begin}
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 1000 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "noise_rng": noise_rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs, metrics = {}, []
    with torch.inference_mode():
        for seed in r["generation_seeds"]:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            chunks = []
            for k in range(0, len(evaluated), 4):
                ref = F.pad(all_ref[evaluated[k : k + 4]], (32, 32, 32, 32))
                noise = torch.randn(ref.shape, device="cuda", generator=gen)
                out = generate_diffusion(model, ref, noise, r["reverse_steps"])
                chunks.append(out[:, 0, 32:-32, 32:-32].cpu().numpy())
            images = np.concatenate(chunks)
            outputs[str(seed)] = images
            for j, i in enumerate(evaluated):
                ink = images[j]
                mask, truth = ink > 0.5, raw["target"][i] > 0.5
                dest = run / f"rendered/pages/seed{seed}-{i:04d}.png"
                Image.fromarray(np.uint8((1 - ink) * 255)).save(dest)
                metrics.append(
                    {
                        "index": i,
                        "seed": seed,
                        "text": rows[i]["text"],
                        "image": str(dest),
                        "sha256": digest(dest),
                        "group": "novel"
                        if i in idx["novel"]
                        else "development"
                        if i in idx["development"]
                        else "train",
                        "dice": None
                        if i in idx["novel"]
                        else float(2 * (mask & truth).sum() / max(1, mask.sum() + truth.sum())),
                        "blank": not bool(mask.any()),
                        "edge": bool(
                            mask[:2].any()
                            or mask[-2:].any()
                            or mask[:, :2].any()
                            or mask[:, -2:].any()
                        ),
                        "target_read_for_output": False,
                    }
                )
    np.savez_compressed(run / "reports/generated.npz", **outputs)
    np.savez_compressed(run / "logs/patch-trace.npz", trace=np.array(traces, dtype=np.int32))
    write(run / "reports/generation-metrics.json", metrics)
    write(
        run / "reports/sampling-counts.json",
        {"indices": train, "counts": counts.tolist(), "development_used": 0, "novel_used": 0},
    )
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "generator_updates": r["steps"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / "logs/checkpoint-06000.pt"),
            "blank": sum(q["blank"] for q in metrics),
            "edge": sum(q["edge"] for q in metrics),
            "C_used": False,
            "font_conditioned": True,
            "stochastic_output": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["launch", "worker"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(launch(args.path))
