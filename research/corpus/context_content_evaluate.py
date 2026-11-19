"""Freeze all content candidates before reading newly sealed regional evaluation data."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(paths: dict[str, str]) -> Path:
    from hwsol.profiles.io import atomic_json
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    candidates = {}
    for name, p in paths.items():
        path = Path(p)
        verify_run(path)
        trained = json.loads((path / "reports/training.json").read_text())
        checkpoint = Path(trained["checkpoint"])
        candidates[name] = {
            "run": str(path.resolve()),
            "run_sha256": digest(path / "run.json"),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest(checkpoint),
        }
    run = create(
        "context-content-freeze",
        config={
            "candidates": candidates,
            "selection": "all final fixed-step checkpoints; no winner",
            "holdout_opened": False,
            "protocol_sha256": digest("docs/V0101_PROTOCOL.md"),
        },
    )
    atomic_json(run.path / "reports/candidates.json", candidates)
    run.finish("held", result={"models": len(candidates), "holdout_opened": False})
    return run.path


def evaluate(frozen: Path, views: Path, holdout: Path) -> Path:
    import numpy as np
    from PIL import Image

    from hwsol.profiles.io import atomic_json
    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_dataset import validate_rows
    from research.corpus.context_training_data import reference_image, whole_ink
    from research.corpus.experiment import create

    for p in [frozen, views, holdout]:
        verify_run(p)
    manifest = json.loads((views / "reports/manifest.json").read_text())
    dataset = Path(manifest["dataset"])
    assignments = json.loads((dataset / "reports/manifest.json").read_text())["split_assignments"]
    rows = json.loads((views / "reports/samples.json").read_text())
    held = validate_rows(holdout, assignments, allowed={"holdout"})
    rows += held
    candidates = json.loads((frozen / "reports/candidates.json").read_text())
    old = Path(
        "workspace/runs/20261119T014658153649Z-context-region-training-aaff0d9b/logs/checkpoint-0800.pt"
    )
    candidates["old-context"] = {
        "checkpoint": str(old.resolve()),
        "checkpoint_sha256": digest(old),
        "architecture": "context-ink-region/1",
    }
    run = create(
        "context-content-evaluation",
        seed=1105,
        config={
            "frozen": str(frozen),
            "frozen_sha256": digest(frozen / "run.json"),
            "views": str(views),
            "holdout": str(holdout),
            "holdout_sha256": digest(holdout / "run.json"),
            "fresh_document": False,
            "models": candidates,
            "train_updates": 0,
            "teacher_metric_not_independent": True,
        },
    )
    try:
        font = profile_store().resolve("personal@v1").font
        refs = []
        targets = []
        for row in rows:
            if digest(row["crop"]) != row["crop_sha256"]:
                raise ValueError("Crop mutated")
            ref, _ = whole_ink(reference_image(row["normalized_label"], font))
            target, _ = whole_ink(Image.open(row["crop"]))
            refs.append(ref)
            targets.append(target)
        np.savez_compressed(
            run.path / "reports/inputs.npz",
            reference=np.array(refs),
            target=np.array(targets),
            role=np.array([int(len(r["normalized_label"].split()) > 1) for r in rows]),
        )
        atomic_json(run.path / "reports/samples.json", rows)
        request = {
            "run": str(run.path),
            "models": candidates,
            "inputs_sha256": digest(run.path / "reports/inputs.npz"),
        }
        atomic_json(run.path / "reports/request.json", request)
        with (run.path / "logs/inference.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.context_content_evaluate",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        run.finish(
            "held",
            result={
                "regions": len(rows),
                "holdout_regions": len(held),
                "models": len(candidates),
                "stochastic_diversity_claimed": False,
                "production_eligible": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(path: Path):
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.context_content_model import ContentInk
    from research.corpus.context_model import ContextInk
    from research.corpus.reference_generator import ink_from_sdf

    r = json.loads(path.read_text())
    run = Path(r["run"])
    assert digest(run / "reports/inputs.npz") == r["inputs_sha256"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    data = np.load(run / "reports/inputs.npz")
    refs = torch.tensor(data["reference"][:, None], device="cuda")
    roles = torch.tensor(data["role"], device="cuda")
    results = {}
    for name, c in r["models"].items():
        assert digest(c["checkpoint"]) == c["checkpoint_sha256"]
        state = torch.load(c["checkpoint"], map_location="cuda", weights_only=True)
        old = c.get("architecture") == "context-ink-region/1"
        model = (ContextInk() if old else ContentInk()).cuda()
        model.load_state_dict(state["model"])
        model.eval()
        start = time.perf_counter()
        out = []
        with torch.inference_mode():
            for i in range(0, len(refs), 8):
                x = refs[i : i + 8]
                prediction = (
                    model.generate(x, roles[i : i + 8], torch.zeros(len(x), 8, device="cuda"))
                    if old
                    else model(x)
                )
                out.append(ink_from_sdf(prediction))
        torch.cuda.synchronize()
        images = torch.cat(out)[:, 0].cpu().numpy()
        np.savez_compressed(run / f"reports/generated-{name}.npz", ink=images)
        for i, ink in enumerate(images):
            Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(
                run / f"rendered/pages/{name}-{i:03d}-s0.png"
            )
        distances = np.mean(
            abs(images - data["target"]) * (1 + 4 * images + 4 * data["target"]), axis=(1, 2)
        )
        results[name] = {
            "seconds": time.perf_counter() - start,
            "samples": len(images),
            "raw_shape_per_region": distances.tolist(),
            "blank": int(np.sum(np.max(images, axis=(1, 2)) < 0.5)),
            "device": torch.cuda.get_device_name(),
            "peak_bytes": torch.cuda.max_memory_allocated(),
        }
        del model
    (run / "reports/inference.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker", "evaluate", "freeze"])
    p.add_argument("path", type=Path)
    p.add_argument("--views", type=Path)
    p.add_argument("--holdout", type=Path)
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.path)
    elif a.mode == "freeze":
        print(freeze(json.loads(a.path.read_text())))
    else:
        print(evaluate(a.path, a.views, a.holdout))
