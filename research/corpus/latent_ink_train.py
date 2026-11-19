"""Finite, isolated CUDA codec/latent learning with immutable evidence receipts."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write

CAPACITY = Path("workspace/runs/20261119T141712403006Z-real-ink-diffusion-training-ba3c97a2")
DATA = Path("workspace/runs/20261119T141744249007Z-real-ink-expanded-data-0c499af3")
FORBIDDEN = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"


def launch(mode: str, *, codec_run: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if mode not in {"diagnose", "codec", "latent"}:
        raise ValueError("Only preregistered capacity phases are enabled")
    verify_run(DATA)
    verify_run(CAPACITY)
    cp = CAPACITY / "logs/checkpoint-02000.pt"
    if mode == "latent":
        if codec_run is None:
            raise ValueError("Frozen learned codec required")
        verify_run(codec_run)
        result = json.loads((codec_run / "reports/result.json").read_text())
        if result["train_dice"] < 0.90 or result["train_blank"]:
            raise ValueError("Codec capacity gate failed")
        cp = codec_run / "logs/checkpoint-02000.pt"
    r = {
        "mode": mode,
        "data": str(DATA.resolve()),
        "data_sha256": digest(DATA / "run.json"),
        "arrays_sha256": digest(DATA / "reports/arrays.npz"),
        "capacity_indices": str((CAPACITY / "reports/indices.json").resolve()),
        "capacity_indices_sha256": digest(CAPACITY / "reports/indices.json"),
        "checkpoint": str(cp.resolve()),
        "checkpoint_sha256": digest(cp),
        "codec_run": str(codec_run.resolve()) if codec_run else None,
        "steps": {"diagnose": 0, "codec": 2000, "latent": 3000}[mode],
        "lr": 0.001 if mode == "codec" else 0.0003,
        "batch": 8,
        "seed": 2011,
        "generation_seeds": [2012, 2013],
        "reverse_steps": 64,
        "C_used": False,
        "font_is_condition_only": True,
        "ocr_teacher": False,
        "protocol_sha256": digest(Path("docs/V01011_PROTOCOL.md")),
    }
    run = create("latent-ink-" + mode, config=r, seed=r["seed"])
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.latent_ink_train",
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


def save_outputs(run, arrays, rows, indices, generated, *, target_read):
    import numpy as np
    from PIL import Image

    metrics = []
    for seed, images in generated.items():
        for j, i in enumerate(indices):
            ink = images[j]
            mask = ink > 0.5
            truth = arrays["target"][i] > 0.5
            dest = run / f"rendered/pages/{seed}-{i:04d}.png"
            Image.fromarray(np.uint8((1 - ink) * 255)).save(dest)
            metrics.append(
                {
                    "seed": seed,
                    "index": int(i),
                    "text": rows[i]["text"],
                    "split": rows[i]["dataset_split"],
                    "dice": float(2 * (mask & truth).sum() / max(1, mask.sum() + truth.sum())),
                    "blank": not bool(mask.any()),
                    "border": int(
                        mask[:2].sum() + mask[-2:].sum() + mask[:, :2].sum() + mask[:, -2:].sum()
                    ),
                    "image": str(dest),
                    "sha256": digest(dest),
                    "target_read_for_output": target_read.get(seed)
                    if isinstance(target_read, dict)
                    else target_read,
                }
            )
    np.savez_compressed(run / "reports/generated.npz", **generated)
    write(run / "reports/generation-metrics.json", metrics)
    return metrics


def worker(request: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.ink_diffusion_model import InkDenoiser, schedule
    from research.corpus.ink_diffusion_model import generate as old_generate
    from research.corpus.latent_ink_model import (
        InkCodec,
        LatentInk,
        bounded_code,
        generate,
        reconstruction_loss,
    )

    r = json.loads(request.read_text())
    setup(r["seed"])
    run = Path(r["run"])
    data = Path(r["data"])
    if (
        digest(data / "run.json") != r["data_sha256"]
        or digest(data / "reports/arrays.npz") != r["arrays_sha256"]
    ):
        raise ValueError("Changed data")
    if digest(Path(r["capacity_indices"])) != r["capacity_indices_sha256"]:
        raise ValueError("Changed fixed capacity indices")
    rows = json.loads((data / "reports/samples.json").read_text())
    arrays = np.load(data / "reports/arrays.npz")
    if any(q["source_hash"] == FORBIDDEN for q in rows):
        raise ValueError("C forbidden")
    if set(arrays["split"]) - {"train", "validation"}:
        raise ValueError("Unknown split")
    ids = np.array(json.loads(Path(r["capacity_indices"]).read_text())["train"])
    if len(ids) != 24 or any(rows[i]["dataset_split"] != "train" for i in ids):
        raise ValueError("Fixed24 Train required")
    if digest(Path(r["checkpoint"])) != r["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    targets = torch.tensor(arrays["target"][:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    if r["mode"] == "diagnose":
        model = InkDenoiser().cuda()
        model.load_state_dict(
            torch.load(r["checkpoint"], map_location="cuda", weights_only=False)["model"]
        )
        model.eval()
        fields = torch.tensor(arrays["field"][:, None], device="cuda")
        records = []
        generated = {}
        noise = torch.randn_like(fields[ids])
        begin = time.perf_counter()
        with torch.inference_mode():
            for value in [0.0, 0.1, 0.5, 0.9, 1.0]:
                t = torch.full((len(ids),), value, device="cuda")
                a, s = schedule(t)
                x = a * fields[ids] + s * noise
                v = model(x, refs[ids], tokens[ids], t)
                truth = a * noise - s * fields[ids]
                pred = (a * x - s * v).clamp(-1, 1)
                ink = torch.sigmoid(pred * (8 / 0.35))
                edge = fields[ids].abs() < 0.75
                err = (v - truth).square()
                records.append(
                    {
                        "t": value,
                        "mse": float(err.mean()),
                        "near_ink_mse": float(err[edge].mean()),
                        "far_background_mse": float(err[~edge].mean()),
                        "target_supplied_in_noisy_input": value < 1,
                    }
                )
                generated[f"oracle-t{value:g}"] = ink[:, 0].cpu().numpy()
            for steps in [64, 128]:
                gen = torch.Generator(device="cuda").manual_seed(2012)
                chunks = []
                for k in range(0, len(ids), 4):
                    ii = ids[k : k + 4]
                    ink, _ = old_generate(
                        model,
                        refs[ii],
                        tokens[ii],
                        torch.randn(refs[ii].shape, device="cuda", generator=gen),
                        steps,
                    )
                    chunks.append(ink[:, 0].cpu().numpy())
                generated[f"prior-{steps}"] = np.concatenate(chunks)
        metrics = save_outputs(
            run,
            arrays,
            rows,
            ids,
            generated,
            target_read={
                name: name.startswith("oracle-t") and float(name[8:]) < 1 for name in generated
            },
        )
        write(run / "reports/noise-diagnostics.json", records)
        write(run / "reports/indices.json", {"train": ids.tolist(), "evaluated": ids.tolist()})
        write(
            run / "reports/result.json",
            {
                "actual_parameter_training": False,
                "updates": 0,
                "seconds": time.perf_counter() - begin,
                "C_used": False,
                "noise_diagnostics": records,
                "mean_dice_by_group": {
                    name: float(np.mean([q["dice"] for q in metrics if q["seed"] == name]))
                    for name in generated
                },
            },
        )
        return
    codec = InkCodec().cuda()
    if r["mode"] == "latent":
        codec.load_state_dict(
            torch.load(r["checkpoint"], map_location="cuda", weights_only=False)["model"]
        )
        codec.eval()
        codec.requires_grad_(False)
        with torch.no_grad():
            code = torch.cat([codec.encode(targets[k : k + 16]) for k in range(0, len(rows), 16)])
            refcode = torch.cat([codec.encode(refs[k : k + 16]) for k in range(0, len(rows), 16)])
            mean = code[ids].mean((0, 2, 3), keepdim=True)
            std = code[ids].std((0, 2, 3), keepdim=True).clamp(min=0.1)
            code = (code - mean) / std
            refcode = (refcode - mean) / std
        model = LatentInk().cuda()
        eval_ids = ids
    else:
        model = codec
        mean = std = None
        eval_ids = np.array(sorted(set(ids) | set(np.where(arrays["split"] == "validation")[0])))
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    curve = []
    checkpoints = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(run / "reports/indices.json", {"train": ids.tolist(), "evaluated": eval_ids.tolist()})
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(ids, r["batch"], replace=True)
        extra = {}
        if r["mode"] == "codec":
            loss = reconstruction_loss(codec(targets[ii]), targets[ii])
        else:
            t = 0.001 + 0.998 * torch.rand(len(ii), device="cuda")
            t[torch.rand_like(t) < 0.1] = 1
            a, s = schedule(t)
            noise = torch.randn_like(code[ii])
            x = a * code[ii] + s * noise
            velocity = a * noise - s * code[ii]
            pred = model(x, refcode[ii], tokens[ii], t)
            weight = 1 + 3 * F.avg_pool2d(targets[ii], 8)
            v_loss = ((pred - velocity).square() * weight).mean()
            clean = bounded_code(a * x - s * pred, mean, std)
            image_loss = reconstruction_loss(codec.decode(clean * std + mean), targets[ii])
            loss = v_loss + 0.2 * image_loss
            extra = {
                "v_loss": float(v_loss.detach()),
                "real_ink_auxiliary": float(image_loss.detach()),
            }
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "seconds": time.perf_counter() - begin,
                    **extra,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 500 == 0:
            dst = run / f"logs/checkpoint-{step:05d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "mean": mean,
                    "std": std,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                dst,
            )
            checkpoints.append({"step": step, "path": str(dst), "sha256": digest(dst)})
    model.eval()
    generated = {}
    with torch.inference_mode():
        if r["mode"] == "codec":
            generated["oracle-reconstruction"] = np.concatenate(
                [
                    codec(targets[eval_ids[k : k + 4]]).sigmoid()[:, 0].cpu().numpy()
                    for k in range(0, len(eval_ids), 4)
                ]
            )
        else:
            for seed in r["generation_seeds"]:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                chunks = []
                for k in range(0, len(eval_ids), 4):
                    ii = eval_ids[k : k + 4]
                    noise = torch.randn(refcode[ii].shape, device="cuda", generator=gen)
                    ink, _ = generate(
                        model, codec, refcode[ii], tokens[ii], noise, mean, std, r["reverse_steps"]
                    )
                    chunks.append(ink[:, 0].cpu().numpy())
                generated[str(seed)] = np.concatenate(chunks)
    metrics = save_outputs(run, arrays, rows, eval_ids, generated, target_read=r["mode"] == "codec")
    train = [q for q in metrics if q["index"] in ids]
    write(run / "reports/checkpoints.json", checkpoints)
    torch.cuda.synchronize()
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "updates": r["steps"],
            "train_regions": len(ids),
            "train_labels": len({rows[i]["text"] for i in ids}),
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_absolute_change": sum(
                (v - initial[k]).abs().sum().item() for k, v in model.state_dict().items()
            ),
            "train_dice": float(np.mean([q["dice"] for q in train])),
            "train_blank": sum(q["blank"] for q in train),
            "target_required_for_output": r["mode"] == "codec",
            "font_output_copy_or_warp": False,
            "seconds": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "checkpoint_sha256": checkpoints[-1]["sha256"],
            "C_used": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["worker", "diagnose", "codec", "latent"])
    p.add_argument("path", type=Path, nargs="?")
    args = p.parse_args()
    if args.command == "worker":
        worker(args.path)
    else:
        print(launch(args.command, codec_run=args.path))
