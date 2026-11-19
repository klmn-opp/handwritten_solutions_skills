"""Remove hidden font targets and content loss independently; keep data/budget fixed."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def train(arm: str, parent: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if arm not in {"real-content", "real-shape"}:
        raise ValueError("Unknown ablation")
    verify_run(parent)
    config = json.loads((parent / "reports/request.json").read_text())
    config.pop("run")
    if config["stage"] != "full" or config["mode"] != "projected":
        raise ValueError("Expected frozen projected full-stage control")
    config.update(
        arm=arm,
        real_per_batch=4,
        teacher_used=arm == "real-content",
        protocol_sha256=digest("docs/V0104_ABLATION_PROTOCOL.md"),
        comparison_parent=str(parent.resolve()),
        comparison_parent_sha256=digest(parent / "run.json"),
        initial_weights_are_parent_result=False,
        optimizer="new AdamW; original mass start, same seed/data order and 2000 updates",
    )
    run = create("context-levelset-ablation-training", seed=config["seed"], config=config)
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.levelset_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=True,
            )
        run.finish("held", result=json.loads((run.path / "reports/training.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["real-content", "real-shape"])
    parser.add_argument("parent", type=Path)
    a = parser.parse_args()
    print(train(a.arm, a.parent))
