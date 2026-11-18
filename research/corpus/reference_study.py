"""CPU coordinator for isolated, all-character CUDA experiments; never updates latest."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.experiment import create
from research.corpus.reference_data import export_views, validate_views

DEFAULT_CONFIG = {
    "version": "all-personal-reference/1",
    "steps": 4000,
    "batch_size": 32,
    "real_per_batch": 4,
    "learning_rate": 0.0003,
    "beta": 0.02,
    "warmup": 500,
    "evaluate_every": 250,
    "selection_starts_at": 1000,
    "selection": "real-validation nearest MAE + 0.5*(1-template-match) + blank + border",
    "selection_limit": "template match is not independent content recognition",
    "image_representation": "96x96, signed distance clipped to 8 pixels; fixed 0.35 pixel edge",
    "sampling": "font classes uniform; 4/32 real samples balanced over real classes",
    "holdout_used_for_selection": False,
    "production_eligible": False,
}


def prepare(dataset: Path) -> Path:
    run = create("all-personal-glyph-data", config={"dataset": str(dataset.resolve())})
    try:
        result = export_views(
            profile_store().resolve("personal@v1"), dataset, run.path / "reports/data"
        )
        validate_views(run.path / "reports/data")
        run.finish(
            "prepared",
            result={
                "classes": len(result["labels"]),
                "font_derived": result["font_derived_count"],
                "real_train": result["real_train_count"],
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def train(
    data: Path,
    *,
    seed: int = 1909,
    steps: int = 4000,
    resume: Path | None = None,
    learning_rate: float = 0.0003,
) -> Path:
    if not 1000 <= steps <= 12000 or not 0 < learning_rate <= 0.001:
        raise ValueError("Training bounds violated")
    manifest = validate_views(data)
    config = dict(DEFAULT_CONFIG, steps=steps, learning_rate=learning_rate)
    request = {
        "data": str(data.resolve()),
        "labels": manifest["labels"],
        "config": config,
        "seed": seed,
        "arrays_sha256": sha256(data / "arrays.npz"),
        "manifest_sha256": sha256(data / "manifest.json"),
    }
    if resume:
        verify_run(resume)
        checkpoint = resume / "logs/best-checkpoint.pt"
        request.update(resume=str(checkpoint.resolve()), resume_sha256=sha256(checkpoint))
    run = create("all-personal-reference-training", config=request, seed=seed)
    request["run_path"] = str(run.path)
    atomic_json(run.path / "reports/request.json", request)
    try:
        python = Path("workspace/environments/corpus-generator-v09/bin/python").absolute()
        with (run.path / "logs/training.log").open("w") as stream:
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "research.corpus.reference_worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=7200,
            )
        validate_views(data)
        summary = json.loads((run.path / "reports/training.json").read_text())
        run.finish(
            "held",
            result={
                "real_cuda_training": True,
                "classes": len(manifest["labels"]),
                "steps": steps,
                "best_step": summary["best_step"],
                "production_eligible": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("dataset", type=Path)
    t = sub.add_parser("train")
    t.add_argument("data", type=Path)
    t.add_argument("--seed", type=int, default=1909)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--resume", type=Path)
    t.add_argument("--learning-rate", type=float, default=0.0003)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.dataset))
    else:
        print(
            train(
                args.data,
                seed=args.seed,
                steps=args.steps,
                resume=args.resume,
                learning_rate=args.learning_rate,
            )
        )
