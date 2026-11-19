"""Bounded CUDA training of semantic ink fields; no target image at inference."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path):
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "architecture": "semantic-neighbour-ink-field/1",
        "seed": 3015,
        "steps": 6000,
        "lr": 0.0005,
        "batch": 4,
        "points": 2048,
        "supervision": "personal-font-derived auxiliary, zero real scans",
        "input": "text identities, neighbours, layout bounds; NO image/outline",
        "stochastic_inference": False,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01015_PROTOCOL.md")),
    }
    run = create("semantic-ink-font-capacity", seed=r["seed"], config=r)
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    Path("/tmp/context-v01015-font").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.semantic_ink_train",
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
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_ink_model import SemanticInk

    r = json.loads(request.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    assert digest(data / "run.json") == r["data_sha256"]
    assert digest(data / "reports/arrays.npz") == r["arrays_sha256"]
    arrays = np.load(data / "reports/arrays.npz")
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    train = idx["train"]
    assert not set(train) & set(idx["development"] + idx["novel"])
    assert all(rows[i]["split"] == "train" for i in train)
    x = {k: torch.tensor(arrays[k], device="cuda") for k in arrays.files}
    model = SemanticInk().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    foreground = {i: np.flatnonzero(arrays["ink"][i].ravel() > 0.1) for i in train}
    counts = np.zeros(len(rows), np.int64)
    curve = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(train, r["batch"])
        np.add.at(counts, ii, 1)
        pixel = []
        for i in ii:
            # Oversample real target foreground only DURING training, never generation.
            a = rng.choice(foreground[i], r["points"] // 2)
            b = rng.integers(0, 128 * 512, r["points"] - len(a))
            pixel.append(np.concatenate([a, b]))
        pi = torch.tensor(np.array(pixel), device="cuda")
        points = torch.stack([pi % 512, pi // 512], -1).float()
        logits, distance = model(x["tokens"][ii], x["boxes"][ii], points)
        gt = x["ink"][ii].flatten(1).gather(1, pi)
        sdf = x["sdf"][ii].flatten(1).gather(1, pi)
        loss = (
            F.binary_cross_entropy_with_logits(logits, gt, reduction="none") * (1 + 3 * gt)
        ).mean()
        loss = loss + 0.2 * (distance - sdf).abs().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite semantic ink loss")
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
    images, metrics = [], []
    evaluated = train[:12] + idx["development"] + idx["novel"]
    for i in evaluated:
        im = model.image(x["tokens"][i : i + 1], x["boxes"][i : i + 1])[0].cpu().numpy()
        path = run / f"rendered/pages/semantic-{i:04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(path)
        p, gt = im > 0.5, arrays["ink"][i] > 0.5
        images.append(im)
        metrics.append(
            {
                "index": i,
                "text": rows[i]["text"],
                "split": rows[i]["split"],
                "image": str(path),
                "sha256": digest(path),
                "dice_to_font": float(2 * (p & gt).sum() / max(1, p.sum() + gt.sum())),
                "blank": not bool(p.any()),
                "edge": bool(p[:2].any() or p[-2:].any() or p[:, :2].any() or p[:, -2:].any()),
                "input_reference_image": False,
                "target_read_for_output": False,
            }
        )
    np.savez_compressed(run / "reports/generated.npz", ink=np.array(images))
    write(run / "reports/generation-metrics.json", metrics)
    write(run / "reports/sampling-counts.json", {"indices": train, "counts": counts.tolist()})
    write(
        run / "reports/result.json",
        {
            "new_updates": r["steps"],
            "actual_parameter_training": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds_including_generation": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / "logs/checkpoint-06000.pt"),
            "mean_dice_to_font": float(np.mean([q["dice_to_font"] for q in metrics])),
            "blank": sum(q["blank"] for q in metrics),
            "edge": sum(q["edge"] for q in metrics),
            "real_scan_training": False,
            "learned_random_variation": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "worker"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        launch(args.path)
    else:
        worker(args.path)
