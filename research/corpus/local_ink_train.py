"""Finite learning from resampled real words; prior words and raw scans stay immutable."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "steps": 4000,
        "batch": 8,
        "lr": 0.0003,
        "seed": 2814,
        "architecture": "canonical-context-local-ink/1",
        "receptive_field": 63,
        "deterministic_inference": True,
        "output_copies_font": False,
        "real_targets_resampled": True,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01014_PROTOCOL.md")),
    }
    run = create("local-real-ink", config=r, seed=r["seed"])
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.local_ink_train",
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


def worker(request: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.latent_ink_model import reconstruction_loss
    from research.corpus.local_ink_model import LocalInk

    r = json.loads(request.read_text())
    setup(r["seed"])
    run = Path(r["run"])
    data = Path(r["data"])
    assert (
        digest(data / "run.json") == r["data_sha256"]
        and digest(data / "reports/arrays.npz") == r["arrays_sha256"]
    )
    idx = json.loads((data / "reports/indices.json").read_text())
    train = idx["train"]
    evaluated = train + idx["development"] + idx["novel"]
    rows = json.loads((data / "reports/samples.json").read_text())
    a = np.load(data / "reports/arrays.npz")
    raw = a["target"]
    refs = torch.tensor(a["reference"][:, None], device="cuda")
    targets = torch.tensor(raw[train, None], device="cuda")
    assert len(train) >= 36 and not set(train) & set(idx["development"] + idx["novel"])
    assert all(rows[i]["dataset_split"] == "train" for i in train)
    model = LocalInk().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    counts = np.zeros(len(train), dtype=np.int64)
    curve = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(run / "reports/indices.json", {**idx, "evaluated": evaluated})
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(len(train), r["batch"], replace=True)
        np.add.at(counts, ii, 1)
        jj = np.array(train)[ii]
        logits = model(refs[jj])
        base = reconstruction_loss(logits, targets[ii])
        p = logits.sigmoid()
        gt = targets[ii]
        edges = ((p[..., 1:] - p[..., :-1]) - (gt[..., 1:] - gt[..., :-1])).abs().mean() + (
            (p[..., 1:, :] - p[..., :-1, :]) - (gt[..., 1:, :] - gt[..., :-1, :])
        ).abs().mean()
        loss = base + edges
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite local ink loss")
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
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    images = []
    with torch.inference_mode():
        for k in range(0, len(evaluated), 4):
            images.append(model(refs[evaluated[k : k + 4]]).sigmoid()[:, 0].cpu().numpy())
    images = np.concatenate(images)
    metrics = []
    for j, i in enumerate(evaluated):
        im = images[j]
        pred = im > 0.5
        truth = raw[i] > 0.5
        dest = run / f"rendered/pages/local-{i:04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
        metrics.append(
            {
                "index": i,
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
                else float(2 * (pred & truth).sum() / max(1, pred.sum() + truth.sum())),
                "blank": not bool(pred.any()),
                "edge": bool(
                    pred[:2].any() or pred[-2:].any() or pred[:, :2].any() or pred[:, -2:].any()
                ),
                "target_read_for_output": False,
            }
        )
    np.savez_compressed(run / "reports/generated.npz", ink=images)
    write(run / "reports/generation-metrics.json", metrics)
    write(
        run / "reports/sampling-counts.json",
        {"indices": train, "counts": counts.tolist(), "development_used": 0, "novel_used": 0},
    )
    write(
        run / "reports/result.json",
        {
            "new_updates": r["steps"],
            "actual_parameter_training": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "mean_dice": {
                group: float(np.mean([q["dice"] for q in metrics if q["group"] == group]))
                for group in ["train", "development"]
            },
            "blank": sum(q["blank"] for q in metrics),
            "edge": sum(q["edge"] for q in metrics),
            "seconds": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / "logs/checkpoint-04000.pt"),
            "deterministic_inference": True,
            "neural_word_latent_variation": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "worker"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(launch(args.path))
