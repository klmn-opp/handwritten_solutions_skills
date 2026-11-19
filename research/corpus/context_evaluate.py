"""Freeze context candidates, then open new regions of a historically exposed document."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.experiment import create
from research.corpus.reference_evaluate import shape_distance


def freeze(training: dict[str, str]) -> Path:
    candidates = {}
    for name, value in training.items():
        path = Path(value)
        verify_run(path)
        result = json.loads((path / "reports/training.json").read_text())
        checkpoint = path / result["checkpoint"]
        candidates[name] = {
            "training_run": str(path.resolve()),
            "run_sha256": sha256(path / "run.json"),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
        }
    run = create(
        "context-candidate-freeze",
        seed=1021,
        config={
            "candidates": candidates,
            "selection": "all fixed-final-step candidates, no winner selected",
            "sample_count": 4,
            "content_gate": "independent transcription must pass before style acceptance",
            "style_gate": "author review pending",
            "fresh_document": False,
            "protocol_sha256": sha256(Path("docs/V010_PROTOCOL.md")),
        },
    )
    atomic_json(run.path / "reports/candidates.json", candidates)
    run.finish(
        "held",
        result={"models": len(candidates), "holdout_opened": False, "production_eligible": False},
    )
    return run.path


def worker(request_path: Path) -> None:
    import time

    import torch

    from research.corpus.context_model import ContextInk
    from research.corpus.reference_generator import ink_from_sdf

    request = json.loads(request_path.read_text())
    run = Path(request["run"])
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA inference required")
    if sha256(run / "reports/inputs.npz") != request["inputs_sha256"]:
        raise ValueError("Evaluation input changed")
    arrays = np.load(run / "reports/inputs.npz", allow_pickle=False)
    torch.set_num_threads(2)
    refs = torch.tensor(arrays["reference"][:, None], device="cuda").repeat_interleave(4, dim=0)
    roles = torch.tensor(arrays["role"], device="cuda").repeat_interleave(4)
    noise = torch.randn(len(refs), 8, generator=torch.Generator().manual_seed(1021)).cuda()
    performance = {}
    for name, row in request["models"].items():
        p = Path(row["checkpoint"])
        if sha256(p) != row["checkpoint_sha256"]:
            raise ValueError("Frozen model changed")
        state = torch.load(p, weights_only=True, map_location="cuda")
        model = ContextInk(
            state["architecture"]["context"], state["architecture"]["global_style"]
        ).cuda()
        model.load_state_dict(state["model"])
        model.eval()
        outputs = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(refs), 8):
                outputs.append(
                    ink_from_sdf(
                        model.generate(refs[i : i + 8], roles[i : i + 8], noise[i : i + 8])
                    )
                )
        torch.cuda.synchronize()
        performance[name] = {
            "seconds": time.perf_counter() - start,
            "samples": len(refs),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        output = torch.cat(outputs)[:, 0].cpu().numpy().reshape(-1, 4, 64, 512)
        np.savez_compressed(
            run / f"reports/generated-{name}.npz", ink=output, noise=noise.cpu().numpy()
        )
        for i, candidates in enumerate(output):
            for j, ink in enumerate(candidates):
                Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(
                    run / f"rendered/pages/{name}-{i:03d}-s{j}.png"
                )
        del model
    atomic_json(run / "reports/inference.json", performance)


def evaluate(frozen: Path, views: Path, holdout: Path) -> Path:
    from research.corpus.context_dataset import validate_rows
    from research.corpus.context_training_data import reference_image, whole_ink

    for path in [frozen, views, holdout]:
        verify_run(path)
    models = json.loads((frozen / "reports/candidates.json").read_text())
    training_rows = json.loads((views / "reports/samples.json").read_text())
    dataset = Path(json.loads((views / "reports/manifest.json").read_text())["dataset"])
    assignments = json.loads((dataset / "reports/manifest.json").read_text())["split_assignments"]
    # This is the first opening of the new region labels, after candidate freeze.
    held = validate_rows(holdout, assignments, allowed={"holdout"})
    profile = profile_store().resolve("personal@v1")
    rows = training_rows + [r for r in held if r["content_type"].startswith("english")]
    run = create(
        "context-independent-evaluation",
        seed=1021,
        config={
            "freeze": str(frozen.resolve()),
            "freeze_sha256": sha256(frozen / "run.json"),
            "views": str(views.resolve()),
            "holdout": str(holdout.resolve()),
            "fresh_document_holdout": False,
            "unseen_regions": True,
            "models": models,
        },
    )
    try:
        targets, refs = [], []
        for row in rows:
            if sha256(Path(row["crop"])) != row["crop_sha256"]:
                raise ValueError("Evaluation crop changed")
            target, transform = whole_ink(Image.open(row["crop"]))
            reference, _ = whole_ink(reference_image(row["normalized_label"], profile.font))
            targets.append(target)
            refs.append(reference)
            row["evaluation_transform"] = transform
            row["role"] = int(len(row["normalized_label"].split()) > 1)
        np.savez_compressed(
            run.path / "reports/inputs.npz",
            target=np.stack(targets),
            reference=np.stack(refs),
            role=np.array([r["role"] for r in rows], dtype=np.int64),
        )
        atomic_json(run.path / "reports/samples.json", rows)
        atomic_json(run.path / "reports/holdout-all-regions.json", held)
        request = {
            "run": str(run.path),
            "models": models,
            "inputs_sha256": sha256(run.path / "reports/inputs.npz"),
        }
        atomic_json(run.path / "reports/request.json", request)
        with (run.path / "logs/inference.log").open("w") as stream:
            subprocess.run(
                [
                    str(Path("workspace/environments/corpus-generator-v09/bin/python").absolute()),
                    "-m",
                    "research.corpus.context_evaluate",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                timeout=900,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        results = {}
        train_images = np.stack(
            [
                target
                for target, row in zip(targets, rows, strict=True)
                if row["dataset_split"] == "train"
            ]
        )
        for name in models:
            generated = np.load(run.path / f"reports/generated-{name}.npz", allow_pickle=False)[
                "ink"
            ]
            per_region = []
            for i, candidates in enumerate(generated):
                masks = candidates > 0.5
                pair = [
                    float(
                        np.logical_xor(masks[a], masks[b]).sum()
                        / max(1, np.logical_or(masks[a], masks[b]).sum())
                    )
                    for a in range(4)
                    for b in range(a + 1, 4)
                ]
                distances = shape_distance(candidates[:, None], train_images[None])
                exact = (masks[:, None] != (train_images[None] > 0.5)).sum((-2, -1)).min(1) == 0
                per_region.append(
                    {
                        "sample_id": rows[i]["sample_id"],
                        "split": rows[i]["dataset_split"],
                        "mean_shape": float(shape_distance(candidates, targets[i]).mean()),
                        "mean_pairwise_ink_xor": float(np.mean(pair)),
                        "nearest_real_train_shape": float(distances.min(1).mean()),
                        "exact_train_binary_copies": int(exact.sum()),
                        "exact_reference_binary_copies": int(
                            ((masks != (refs[i] > 0.5)).sum((-2, -1)) == 0).sum()
                        ),
                        "blank_count": int((~masks.reshape(4, -1).any(1)).sum()),
                        "transcription_accuracy": None,
                        "author_similarity": None,
                    }
                )
            results[name] = {
                "regions": per_region,
                "by_split": {
                    split: {
                        "count": sum(r["split"] == split for r in per_region),
                        "shape_mean": float(
                            np.mean([r["mean_shape"] for r in per_region if r["split"] == split])
                        ),
                    }
                    for split in ["train", "validation", "holdout"]
                },
            }
        atomic_json(run.path / "reports/image-metrics.json", results)
        run.finish(
            "held",
            result={
                "english_regions": len(rows),
                "models": len(models),
                "samples_per_region": 4,
                "holdout_regions": len(held),
                "fresh_document": False,
                "ocr_evaluation": "separate run",
                "author_style": "pending",
                "production_eligible": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "evaluate"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--holdout", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(evaluate(args.path, args.views, args.holdout))
