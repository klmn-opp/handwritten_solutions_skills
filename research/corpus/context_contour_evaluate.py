"""Freeze and replay constrained contour candidates; record every comparison source."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def evaluate(views: Path, candidates: dict[str, str]) -> Path:
    import numpy as np

    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_training_data import reference_image, whole_ink
    from research.corpus.experiment import create

    verify_run(views)
    models = {}
    for name, location in candidates.items():
        p = Path(location)
        verify_run(p)
        r = json.loads((p / "reports/training.json").read_text())
        cfg = json.loads((p / "reports/request.json").read_text())
        models[name] = {
            "run": str(p),
            "run_sha256": digest(p / "run.json"),
            "checkpoint": r["checkpoint"],
            "checkpoint_sha256": r["checkpoint_sha256"],
            "context": cfg["mode"] in {"context", "flow"},
            "bound": cfg["bound_pixels"],
            "architecture": r["architecture"],
            "steps": r["executed_updates"],
        }
    old = Path(
        "workspace/runs/20261119T035904738321Z-context-clean-ink-training-5758634f/logs/checkpoint-1500.pt"
    ).resolve()
    run = create(
        "context-contour-frozen-evaluation",
        config={
            "views": str(views),
            "views_sha256": digest(views / "run.json"),
            "models": models,
            "selection": "both predeclared final checkpoints",
            "old_mass": str(old),
            "old_mass_sha256": digest(old),
            "holdout_opened": False,
            "evaluation_scope": "development-only",
        },
    )
    try:
        rows = json.loads((views / "reports/samples.json").read_text())
        font = profile_store().resolve("personal@v1").font
        refs = np.array([whole_ink(reference_image(r["normalized_label"], font))[0] for r in rows])
        np.savez_compressed(run.path / "reports/legacy-reference.npz", reference=refs)
        cfg = {
            "run": str(run.path),
            "models": models,
            "views": str(views),
            "views_sha256": digest(views / "run.json"),
            "arrays_sha256": digest(views / "reports/arrays.npz"),
            "old_mass": str(old),
            "old_mass_sha256": digest(old),
            "legacy_sha256": digest(run.path / "reports/legacy-reference.npz"),
        }
        write(run.path / "reports/request.json", cfg)
        with (run.path / "logs/inference.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.context_contour_evaluate",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        write(run.path / "reports/samples.json", rows)
        # PNG outputs above come from actual frozen-checkpoint inference.
        run.finish(
            "held",
            result={
                "regions": len(rows),
                "frozen_models": len(models),
                "development_only": True,
                "holdout_opened": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def worker(path: Path):
    import os
    import time

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.context_content_model import ContentInk
    from research.corpus.context_contour_model import ContourInk
    from research.corpus.context_flow_model import FlowInk
    from research.corpus.reference_generator import ink_from_sdf

    r = json.loads(path.read_text())
    run = Path(r["run"])
    views = Path(r["views"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    assert digest(views / "reports/arrays.npz") == r["arrays_sha256"]
    assert digest(run / "reports/legacy-reference.npz") == r["legacy_sha256"]
    data = np.load(views / "reports/arrays.npz")
    legacy = np.load(run / "reports/legacy-reference.npz")["reference"]

    def distance(ink):
        m = np.uint8(ink > 0.5)
        return (
            cv2.distanceTransform(m, cv2.DIST_L2, 5) - cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
        ).clip(-16, 16)

    sdf = torch.tensor(np.stack([distance(x) for x in data["reference"]])[:, None], device="cuda")
    metrics = {}
    outputs = {}
    with torch.inference_mode():
        outputs["font-field"] = torch.sigmoid(sdf / 0.35)[:, 0].cpu().numpy()
        for name, m in r["models"].items():
            assert digest(m["checkpoint"]) == m["checkpoint_sha256"]
            net = (
                (
                    FlowInk()
                    if m.get("architecture") == "font-constrained-context-flow/1"
                    else ContourInk(context=m["context"], bound=m["bound"])
                )
                .cuda()
                .eval()
            )
            net.load_state_dict(
                torch.load(m["checkpoint"], map_location="cuda", weights_only=True)["model"]
            )
            start = time.perf_counter()
            parts = []
            for i in range(0, len(sdf), 4):
                parts.append(net(sdf[i : i + 4])[0][:, 0].cpu().numpy())
            outputs[name] = np.concatenate(parts)
            torch.cuda.synchronize()
            trained = np.load(Path(m["run"]) / f"reports/generated-{m.get('steps', 1500)}.npz")[
                "ink"
            ]
            metrics[name] = {
                "seconds": time.perf_counter() - start,
                "replay_exact": bool(np.array_equal(outputs[name], trained)),
                "max_replay_difference": float(np.abs(outputs[name] - trained).max()),
            }
            if m["context"]:
                net.context_enabled = False
                disabled = np.concatenate(
                    [net(sdf[i : i + 4])[0][:, 0].cpu().numpy() for i in range(0, len(sdf), 4)]
                )
                np.savez_compressed(run / "reports/context-disabled.npz", ink=disabled)
                metrics[name]["context_disabled_mean_absolute_change"] = float(
                    np.abs(outputs[name] - disabled).mean()
                )
            del net
        assert digest(r["old_mass"]) == r["old_mass_sha256"]
        net = ContentInk().cuda().eval()
        net.load_state_dict(
            torch.load(r["old_mass"], map_location="cuda", weights_only=True)["model"]
        )
        ref = torch.tensor(legacy[:, None], device="cuda")
        start = time.perf_counter()
        outputs["old-mass"] = np.concatenate(
            [ink_from_sdf(net(ref[i : i + 8]))[:, 0].cpu().numpy() for i in range(0, len(ref), 8)]
        )
        metrics["old-mass"] = {"seconds": time.perf_counter() - start}
    for name, a in outputs.items():
        np.savez_compressed(run / f"reports/generated-{name}.npz", ink=a)
        for i, ink in enumerate(a):
            Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(
                run / f"rendered/pages/{name}-{i:03d}.png"
            )
    write(
        run / "reports/inference.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "models": metrics,
        },
    )


def compare(evaluation: Path) -> Path:
    import shutil

    import numpy as np
    from PIL import Image

    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_comparison import strips
    from research.corpus.context_evaluation_ink import evaluation_ink
    from research.corpus.experiment import create

    verify_run(evaluation)
    rows = json.loads((evaluation / "reports/samples.json").read_text())
    config = json.loads((evaluation / "reports/request.json").read_text())
    run = create(
        "context-contour-comparisons",
        config={
            "evaluation": str(evaluation),
            "evaluation_sha256": digest(evaluation / "run.json"),
            "selection": "all validation, all new English training and first four old training",
        },
    )
    try:
        profile = profile_store().resolve("personal@v1")
        observations = []
        selected = []
        old = 0
        for i, row in enumerate(rows):
            new = "new-english-sol-review" in row["source_package"]
            if row["dataset_split"] == "train" and not new:
                if old >= 4:
                    continue
                old += 1
            selected.append({"row_index": i, **row})
            images, receipt = strips(
                row["normalized_label"], row["sample_id"], profile, {}, methods=("handnote",)
            )
            images["real-reference"] = Image.open(row["crop"]).convert("RGB")
            write(
                run.path / f"reports/source-{i:03d}.json",
                {"sample": row, "handnote_provenance": receipt},
            )
            for method, image in images.items():
                raw = run.path / f"comparisons/{method}-{i:03d}-original.png"
                image.save(raw)
                ink, transform = evaluation_ink(image, (1024, 128))
                dest = run.path / f"rendered/pages/{method}-{i:03d}.png"
                Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(dest)
                observations.append(
                    {
                        "id": f"{method}-{i:03d}",
                        "method": method,
                        "row_index": i,
                        "text": row["normalized_label"],
                        "split": row["dataset_split"],
                        "image": str(dest),
                        "sha256": digest(dest),
                        "raw_image": str(raw),
                        "preprocessing": transform,
                        "sample_id": row["sample_id"],
                    }
                )
            for method in ["font-field", "old-mass", *config["models"]]:
                raw = evaluation / f"rendered/pages/{method}-{i:03d}.png"
                dest = run.path / f"rendered/pages/{method}-{i:03d}.png"
                shutil.copyfile(raw, dest)
                observations.append(
                    {
                        "id": f"{method}-{i:03d}",
                        "method": method,
                        "row_index": i,
                        "text": row["normalized_label"],
                        "split": row["dataset_split"],
                        "image": str(dest),
                        "sha256": digest(dest),
                        "raw_image": str(raw),
                        "preprocessing": None,
                        "sample_id": row["sample_id"],
                    }
                )
        write(run.path / "reports/observations-input.json", observations)
        write(run.path / "reports/selected-regions.json", selected)
        run.finish(
            "prepared",
            result={"regions": len(selected), "images": len(observations), "holdout_opened": False},
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker", "compare"])
    p.add_argument("path", type=Path)
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.path)
    else:
        print(compare(a.path))
