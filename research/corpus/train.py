"""Admission-gated real training, intentionally independent of production and OCR environments."""

import argparse
import json
from functools import cache
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.dataset import readiness
from research.corpus.experiment import create
from research.corpus.label_admission import accepted_label, validate_model_evidence


def validate_dataset(dataset: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((dataset / "manifest.json").read_text())
    records = dataset / "samples.jsonl"
    if sha256(records) != manifest["sample_manifest_sha256"]:
        raise ValueError("Dataset manifest checksum mismatch")
    policy = None
    if manifest.get("label_policy_sha256"):
        file = dataset / "label-policy.json"
        if sha256(file) != manifest["label_policy_sha256"]:
            raise ValueError("Label authorization policy checksum mismatch")
        policy = json.loads(file.read_text())
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    splits = {}
    crop_splits = {}
    document_splits = {}

    @cache
    def checksum(path: str) -> str:
        return sha256(Path(path))

    for row in rows:
        if (
            not accepted_label(row, policy)
            or row["writer_provenance"]["domain"] != "personal-real"
            or row["writer_provenance"]["is_generated"]
            or row.get("quality_status") != "clean"
            or row.get("defects")
            or row["region_role"] != "personal-handwriting"
            or not row["privacy_checked"]
        ):
            raise ValueError("Unverified, damaged or non-personal sample cannot enter training")
        if row["review_status"] == "model-accepted":
            validate_model_evidence(row)
        if (
            checksum(row["image_crop"]) != row["crop_sha256"]
            or checksum(row["source_original"]) != row["source_hash"]
        ):
            raise ValueError("Source/crop integrity mismatch")
        for mapping, key in [
            (splits, row["split_group"]),
            (crop_splits, row["crop_sha256"]),
            (document_splits, row["source_hash"]),
        ]:
            if mapping.setdefault(key, row["dataset_split"]) != row["dataset_split"]:
                raise ValueError("Train/holdout leakage")
    return rows, readiness(rows)


def train(
    dataset: Path,
    *,
    run_id=None,
    seed=902,
    epochs=30,
    python: Path | None = None,
    preprocessing="raw-gray",
):
    from research.corpus.training_data import MODES

    if preprocessing not in MODES:
        raise ValueError("Unknown training preprocessing")
    if not 1 <= epochs <= 500:
        raise ValueError("Epoch count must be between 1 and 500")
    rows, gate = validate_dataset(dataset)
    run = create(
        "personal-glyph-training",
        run_id=run_id,
        seed=seed,
        config={
            "dataset": str(dataset.resolve()),
            "manifest_sha256": sha256(dataset / "manifest.json"),
            "epochs": epochs,
            "architecture": "glyph-cvae/1",
            "preprocessing": preprocessing,
            "gate": gate,
        },
    )
    atomic_json(run.path / "reports/readiness.json", gate)
    if gate["decision"] != "GO-limited-prototype":
        run.finish(
            "held",
            result={
                "training_executed": False,
                "reason": "Insufficient verified independent glyphs",
                "decision": "NO-GO",
            },
        )
        return run.path
    try:
        import subprocess

        interpreter = python or Path("workspace/environments/corpus-generator-v09/bin/python")
        request = {
            "dataset": str(dataset.resolve()),
            "run_path": str(run.path),
            "seed": seed,
            "epochs": epochs,
            "gate": gate,
            "rows": rows,
            "preprocessing": preprocessing,
        }
        atomic_json(run.path / "reports/request.json", request)
        with (run.path / "logs/training.log").open("w") as stream:
            subprocess.run(
                [
                    str(interpreter.absolute()),
                    "-m",
                    "research.corpus.train_worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=14400,
            )
        run.finish(
            "held",
            result={
                "training_executed": True,
                "production_eligible": False,
                "reason": "Independent content, diversity, memorization and style review required",
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--preprocessing", choices=["raw-gray", "dark-ink-v1"], default="raw-gray")
    args = parser.parse_args()
    print(
        train(
            args.dataset,
            run_id=args.run_id,
            epochs=args.epochs,
            python=args.python,
            preprocessing=args.preprocessing,
        )
    )
