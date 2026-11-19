"""Two finite warm-start arms; old models, private data and CPU runtime stay frozen."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write
from research.corpus.latent_ink_train import DATA, FORBIDDEN, save_outputs

BASE = Path("workspace/runs/20261119T170441302262Z-latent-ink-latent-b9aec932")
CODEC = Path("workspace/runs/20261119T170016941957Z-latent-ink-codec-ffb2c521")


def launch(arm: str) -> Path:
    import numpy as np

    from hwsol.workspace import verify_run
    from research.corpus.experiment import create
    from research.corpus.thin_ink_data import component_weights, token_positions

    if arm not in {"detail", "local"}:
        raise ValueError("Only preregistered detail/local arms allowed")
    for path in [BASE, CODEC, DATA]:
        verify_run(path)
    config = {
        "arm": arm,
        "data": str(DATA.resolve()),
        "data_sha256": digest(DATA / "run.json"),
        "arrays_sha256": digest(DATA / "reports/arrays.npz"),
        "base": str((BASE / "logs/checkpoint-03000.pt").resolve()),
        "base_sha256": digest(BASE / "logs/checkpoint-03000.pt"),
        "codec": str((CODEC / "logs/checkpoint-02000.pt").resolve()),
        "codec_sha256": digest(CODEC / "logs/checkpoint-02000.pt"),
        "indices": str((BASE / "reports/indices.json").resolve()),
        "indices_sha256": digest(BASE / "reports/indices.json"),
        "steps": 3000,
        "batch": 8,
        "lr": 0.0001,
        "seed": 2112,
        "generation_seeds": [2012, 2013],
        "reverse_steps": 64,
        "pure_noise_probability": 0.2,
        "detail_auxiliary_weight": 0.35,
        "fresh_optimizer": True,
        "base_training_updates": 3000,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01012_PROTOCOL.md")),
    }
    run = create("thin-ink-" + arm, config=config, seed=config["seed"])
    print(run.path, flush=True)
    try:
        ids = json.loads((BASE / "reports/indices.json").read_text())["train"]
        rows = json.loads((DATA / "reports/samples.json").read_text())
        arrays = np.load(DATA / "reports/arrays.npz")
        conditioning = run.path / "reports/conditioning.npz"
        np.savez_compressed(
            conditioning,
            positions=np.stack([token_positions(rows[i]) for i in ids]),
            component_weights=np.stack([component_weights(arrays["target"][i]) for i in ids])[
                :, None
            ],
        )
        write(
            run.path / "reports/request.json",
            {**config, "run": str(run.path), "conditioning_sha256": digest(conditioning)},
        )
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.thin_ink_train",
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


def worker(path: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.ink_diffusion_model import schedule
    from research.corpus.latent_ink_model import InkCodec, bounded_code, generate
    from research.corpus.thin_ink_model import LocalLatentInk, ink_detail_loss

    r = json.loads(path.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    for file, expected in [
        (data / "run.json", r["data_sha256"]),
        (data / "reports/arrays.npz", r["arrays_sha256"]),
        (Path(r["base"]), r["base_sha256"]),
        (Path(r["codec"]), r["codec_sha256"]),
        (Path(r["indices"]), r["indices_sha256"]),
    ]:
        if digest(file) != expected:
            raise ValueError("Changed frozen input")
    arrays = np.load(data / "reports/arrays.npz")
    rows = json.loads((data / "reports/samples.json").read_text())
    ids = np.array(json.loads(Path(r["indices"]).read_text())["train"])
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError("Same fixed 24 required")
    if any(q["source_hash"] == FORBIDDEN for q in rows):
        raise ValueError("C forbidden")
    if any(rows[i]["dataset_split"] != "train" for i in ids):
        raise ValueError("Train only")
    selected = [rows[i] for i in ids]
    targets = torch.tensor(arrays["target"][ids, None], device="cuda")
    refs = torch.tensor(arrays["reference"][ids, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in selected]), device="cuda")
    if digest(run / "reports/conditioning.npz") != r["conditioning_sha256"]:
        raise ValueError("Changed derived conditioning")
    conditioning = np.load(run / "reports/conditioning.npz")
    positions = torch.tensor(conditioning["positions"], device="cuda")
    components = torch.tensor(conditioning["component_weights"], device="cuda")
    codec = InkCodec().cuda().eval().requires_grad_(False)
    codec.load_state_dict(torch.load(r["codec"], map_location="cuda", weights_only=False)["model"])
    base = torch.load(r["base"], map_location="cuda", weights_only=False)
    mean, std = base["mean"], base["std"]
    model = LocalLatentInk(local=r["arm"] == "local").cuda()
    missing = model.load_state_dict(base["model"], strict=False)
    expected = ["local_projection.weight"] if model.local else []
    if missing.missing_keys != expected or missing.unexpected_keys:
        raise ValueError("Unexpected warm-start mismatch")
    with torch.no_grad():
        code = (
            torch.cat([codec.encode(targets[k : k + 16]) for k in range(0, 24, 16)]) - mean
        ) / std
        refcode = (
            torch.cat([codec.encode(refs[k : k + 16]) for k in range(0, 24, 16)]) - mean
        ) / std
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    noise_rng = torch.Generator(device="cuda").manual_seed(r["seed"])
    curve, checkpoints = [], []
    write(run / "reports/indices.json", {"train": ids.tolist(), "evaluated": ids.tolist()})
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(24, r["batch"], replace=True)
        t = 0.001 + 0.998 * torch.rand(len(ii), device="cuda", generator=noise_rng)
        t[torch.rand(len(ii), device="cuda", generator=noise_rng) < r["pure_noise_probability"]] = 1
        a, s = schedule(t)
        noise = torch.randn(code[ii].shape, device="cuda", generator=noise_rng)
        x = a * code[ii] + s * noise
        pred = model(x, refcode[ii], tokens[ii], t, positions[ii])
        velocity = a * noise - s * code[ii]
        weight = 1 + 3 * F.avg_pool2d(targets[ii], 8)
        v_loss = ((pred - velocity).square() * weight).mean()
        clean = bounded_code(a * x - s * pred, mean, std)
        detail = ink_detail_loss(codec.decode(clean * std + mean), targets[ii], components[ii])
        loss = v_loss + r["detail_auxiliary_weight"] * detail
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
                    "v_loss": float(v_loss.detach()),
                    "detail_loss": float(detail.detach()),
                    "seconds": time.perf_counter() - begin,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 500 == 0:
            dest = run / f"logs/checkpoint-{step:05d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "mean": mean,
                    "std": std,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "noise_rng": noise_rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                dest,
            )
            checkpoints.append({"step": step, "path": str(dest), "sha256": digest(dest)})
    model.eval()
    generated = {}
    with torch.inference_mode():
        for seed in r["generation_seeds"]:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            chunks = []
            for k in range(0, 24, 4):

                def conditioned(x, reference, text, time, pos=positions[k : k + 4]):
                    return model(x, reference, text, time, pos)

                out, _ = generate(
                    conditioned,
                    codec,
                    refcode[k : k + 4],
                    tokens[k : k + 4],
                    torch.randn(refcode[k : k + 4].shape, device="cuda", generator=gen),
                    mean,
                    std,
                    r["reverse_steps"],
                )
                chunks.append(out[:, 0].cpu().numpy())
            generated[str(seed)] = np.concatenate(chunks)
    metrics = save_outputs(run, arrays, rows, ids, generated, target_read=False)
    write(run / "reports/checkpoints.json", checkpoints)
    torch.cuda.synchronize()
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "updates": r["steps"],
            "warmstart_updates": 3000,
            "train_regions": 24,
            "train_labels": len(set(q["text"] for q in selected)),
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_absolute_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "train_dice": float(np.mean([q["dice"] for q in metrics])),
            "train_blank": sum(q["blank"] for q in metrics),
            "target_required_for_output": False,
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
    p.add_argument("command", choices=["detail", "local", "worker"])
    p.add_argument("path", type=Path, nargs="?")
    args = p.parse_args()
    if args.command == "worker":
        worker(args.path)
    else:
        print(launch(args.command))
