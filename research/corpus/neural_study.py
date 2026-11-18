"""Bounded CVAE continuation and personal-font transfer; production stays untouched."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create
from research.corpus.font_auxiliary import validate_auxiliary
from research.corpus.label_admission import label_text
from research.corpus.train import validate_dataset

PROTOCOL = {
    "version": "handnote-glyph-study/1",
    "real_labels": ["1"],
    "preprocessing": "dark-ink-v1",
    "font_steps": 1200,
    "real_steps": 600,
    "batch_size": 32,
    "learning_rate": 0.0003,
    "control_beta": 0.001,
    "corrected_beta": 0.05,
    "font_beta": 0.01,
    "warmup_steps": 100,
    "evaluate_every": 25,
    "selection_starts_at": 150,
    "validation_prior_seed": 2026,
    "validation_prior_count": 64,
    "final_prior_seed": 2902,
    "final_prior_count": 64,
    "optimizer_on_new_stage": "reset AdamW; weights continued with exact checkpoint SHA",
    "selection": "val weighted reconstruction + nearest-prior weighted MAE + blank/border cost",
    "selection_limit": "engineering proxy only; no semantic or author-style acceptance",
    "production_eligible": False,
}
STAGES = {"real-control", "real-prior", "font-pretrain", "font-finetune"}


def admitted_inputs(dataset: Path, auxiliary: Path) -> tuple[list[dict], list[dict], list[str]]:
    rows, gate = validate_dataset(dataset)
    if gate["decision"] != "GO-limited-prototype" or "1" not in gate["eligible_labels"]:
        raise ValueError("Digit 1 needs admitted independent 20/5/5 real samples")
    real = [
        row
        for row in rows
        if label_text(row) == "1" and row["dataset_split"] in {"train", "validation"}
    ]
    fonts, _ = validate_auxiliary(auxiliary)
    labels = sorted({row["label"] for row in fonts}, key=ord)
    if "1" not in labels:
        raise ValueError("Personal font must contain the real target digit")
    return real, fonts, labels


def validate_request(request: dict) -> None:
    if request["protocol"] != PROTOCOL or request["stage"] not in STAGES:
        raise ValueError("Unrecognized frozen training protocol")
    real, fonts, labels = admitted_inputs(Path(request["dataset"]), Path(request["auxiliary"]))
    if (real, fonts, labels) != (request["real_rows"], request["font_rows"], request["labels"]):
        raise ValueError("Training request changed or contains holdout samples")
    source = Path(request["initial_run"])
    verify_run(source)
    checkpoint = source / "logs/best-checkpoint.pt"
    if sha256(checkpoint) != request["initial_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint changed")
    if request["stage"] == "font-finetune":
        origin = json.loads((source / "reports/request.json").read_text())
        if origin.get("stage") != "font-pretrain":
            raise ValueError("Font fine-tuning must start from recorded font pretraining")


def train_study(
    dataset: Path,
    auxiliary: Path,
    initial_run: Path,
    stage: str,
    seed: int = 902,
    run_id: str | None = None,
) -> Path:
    if stage not in STAGES:
        raise ValueError("Unknown study stage")
    real, fonts, labels = admitted_inputs(dataset, auxiliary)
    verify_run(initial_run)
    request = {
        "dataset": str(dataset.resolve()),
        "auxiliary": str(auxiliary.resolve()),
        "initial_run": str(initial_run.resolve()),
        "initial_checkpoint_sha256": sha256(initial_run / "logs/best-checkpoint.pt"),
        "stage": stage,
        "seed": seed,
        "labels": labels,
        "real_rows": real,
        "font_rows": fonts,
        "protocol": PROTOCOL,
    }
    validate_request(request)
    run = create(
        "glyph-" + stage,
        run_id=run_id,
        seed=seed,
        config={
            "stage": stage,
            "protocol": PROTOCOL,
            "dataset_sha256": sha256(dataset / "manifest.json"),
            "auxiliary_sha256": sha256(auxiliary / "manifest.json"),
            "initial_checkpoint_sha256": request["initial_checkpoint_sha256"],
        },
    )
    request["run_path"] = str(run.path)
    atomic_json(run.path / "reports/request.json", request)
    interpreter = Path("workspace/environments/corpus-generator-v09/bin/python").absolute()
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    str(interpreter),
                    "-m",
                    "research.corpus.neural_study_worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        receipt = json.loads((run.path / "reports/training.json").read_text())
        if receipt["parameter_absolute_change"] <= 0 or receipt["optimizer_updates"] <= 0:
            raise ValueError("No actual learning occurred")
        run.finish(
            "held",
            result={
                "training_executed": True,
                "stage": stage,
                "production_eligible": False,
                "quality": "not accepted",
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=902)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(
        train_study(
            args.dataset, args.auxiliary, args.initial_run, args.stage, args.seed, args.run_id
        )
    )
