"""Finite real-word supervised correction with a frozen local neural backbone."""

import json
import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write


def complete_indices(rows: list[dict]) -> list[int]:
    result = []
    for row in rows:
        if row["known_local_positions"] == row["semantic_characters"]:
            result.append(row["index"])
    if len(set(result)) != len(result) or not result:
        raise ValueError("Unique complete original-word indices required")
    return result


def launch(audit: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(audit)
    old = json.loads((audit / "reports/request.json").read_text())
    rows = json.loads((audit / "reports/outputs.json").read_text())
    indices = complete_indices(rows)
    assert len(indices) == 13 and len(rows) == 18
    cfg = {
        "audit": str(audit),
        "audit_sha256": digest(audit / "run.json"),
        "checkpoint": old["checkpoint"],
        "checkpoint_sha256": old["checkpoint_sha256"],
        "data": old["data"],
        "data_sha256": old["data_sha256"],
        "indices": indices,
        "seed": 4522,
        "steps": 1000,
        "learning_rate": 0.001,
        "batch_size": 4,
        "background_penalty": 0.001,
        "architecture": "bounded-word-ink-context/1",
        "backbone_updates": 0,
        "new_independent_holdout": False,
        "C_used": False,
        "protocol_sha256": digest("docs/V01022_PROTOCOL.md"),
    }
    run = create("bounded-real-word-context-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-context-model").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    __name__,
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: str) -> None:
    import importlib.metadata
    import time

    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.word_context_model import WordContextInk

    cfg = json.loads(Path(request).read_text())
    run, audit = Path(cfg["run"]), Path(cfg["audit"])
    assert digest(audit / "run.json") == cfg["audit_sha256"]
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    setup(cfg["seed"])
    with np.load(audit / "reports/reconstructions.npz") as a:
        base = torch.tensor(a["ink"], device="cuda")[:, None]
    with np.load(audit / "solutions/original-words.npz") as a:
        target = torch.tensor(a["ink"], device="cuda")[:, None]
    model = WordContextInk().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    curve, counts = [], np.zeros(len(base), np.int64)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        ids = rng.choice(cfg["indices"], size=cfg["batch_size"], replace=True)
        np.add.at(counts, ids, 1)
        output, delta = model(base[ids], return_delta=True)
        bce = (
            F.binary_cross_entropy(output.clamp(1e-6, 1 - 1e-6), target[ids], reduction="none")
            * (1 + 3 * target[ids])
        ).mean()
        penalty = (delta.square() * (base[ids] < 0.01)).mean()
        loss = bce + cfg["background_penalty"] * penalty
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite context loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "bce": float(bce.detach()),
                "background_penalty": float(penalty.detach()),
                "seconds": time.perf_counter() - start,
            }
            curve.append(row)
            write(run / "reports/curve.json", curve)
            print(json.dumps(row), flush=True)
        if step % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "config": cfg,
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs, images = [], []
    rows = json.loads((audit / "reports/outputs.json").read_text())
    for i, row in enumerate(rows):
        im = model.image(base[i : i + 1])[0, 0].cpu().numpy()
        images.append(im)
        path = run / f"rendered/pages/context-{i:02d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(path)
        tt, pp = target[i, 0].cpu().numpy() > 0.5, im > 0.5
        outputs.append(
            {
                "index": i,
                "text": row["text"],
                "image": str(path),
                "sha256": digest(path),
                "used_context_train": i in cfg["indices"],
                "baseline_dice": row["dice"],
                "dice": float(2 * (tt & pp).sum() / max(1, tt.sum() + pp.sum())),
                "no_independent_validation": True,
            }
        )
    np.savez_compressed(run / "reports/reconstructions.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "new_model_updates": cfg["steps"],
            "frozen_backbone_updates": 0,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "outputs": len(outputs),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{cfg['steps']:05d}.pt"),
            "quality_passed": False,
            "C_used": False,
            "trained_real_words": len(cfg["indices"]),
            "spatial_adapter_not_end_to_end_line_generator": True,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
