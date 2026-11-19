"""Paired finite data-expansion control, explicitly a font-geometry learner."""

import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create


def launch(data: Path, registration: Path | None = None, checkpoint: Path | None = None) -> Path:
    verify_run(data)
    stage = "train" if checkpoint else "registration"
    if checkpoint:
        if registration is None:
            raise ValueError("Frozen registration is required")
        verify_run(registration)
        reg = json.loads((registration / "reports/request.json").read_text())
        if reg["data_sha256"] != sha256(data / "run.json"):
            raise ValueError("Registration data identity differs")
    cfg = {
        "stage": stage,
        "data": str(data),
        "data_sha256": sha256(data / "run.json"),
        "registration": str(registration) if registration else None,
        "registration_sha256": sha256(registration / "run.json") if registration else None,
        "mode": "stochastic",
        "monotone": True,
        "steps": 3000 if checkpoint else 240,
        "seed": 1909,
        "lr": 0.0002,
        "warmstart": str(checkpoint) if checkpoint else None,
        "warmstart_sha256": sha256(checkpoint) if checkpoint else None,
        "optimizer": "fresh AdamW; identical budget and initial weights for both data arms",
        "C_used": False,
        "output_is_warped_font_ink": True,
        "new_ink_topology": False,
        "protocol_sha256": sha256(Path("docs/V0109_GEOMETRY_CONTROL.md")),
    }
    run = create("diversity-geometry-" + stage, config=cfg, seed=1909)
    atomic_json(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/execution.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.aligned_style_experiment",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
