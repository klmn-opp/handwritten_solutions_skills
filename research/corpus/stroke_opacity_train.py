"""Bounded author-directed continuation from frozen mass; no production dependency."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def train(arm: str) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if arm not in {"control", "core", "stroke"}:
        raise ValueError("Unknown fixed arm")
    source = Path("workspace/runs/20261119T035904738321Z-context-clean-ink-training-5758634f")
    verify_run(source)
    config = json.loads((source / "reports/request.json").read_text())
    config.pop("run")
    verify_run(Path(config["views"]))
    checkpoint = (source / "logs/checkpoint-1500.pt").resolve()
    config.update(
        arm=arm,
        initial_checkpoint=str(checkpoint),
        initial_sha256=digest(checkpoint),
        steps=1000,
        seed=1303,
        lr=0.00005,
        opacity_weight=0.15 if arm != "control" else 0,
        component_weight=0.10 if arm == "stroke" else 0,
        protocol_sha256=digest("docs/V0103_PROTOCOL.md"),
        optimizer="new AdamW; same initial weights and schedule in all three arms",
        target_geometry="real aligned scan unchanged; 1px foreground tolerance, 3px background margin",
        new_training_samples=0,
    )
    run = create("context-stroke-opacity-training", seed=config["seed"], config=config)
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.clean_ink_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        run.finish("held", result=json.loads((run.path / "reports/training.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["control", "core", "stroke"])
    print(train(parser.parse_args().arm))
