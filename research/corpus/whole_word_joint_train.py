"""Paired whole-word learning with fixed initial versus fitted training coordinates."""

import json
import subprocess
from pathlib import Path

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def launch(
    data: Path, source: Path, font_teacher: Path, support: Path, geometry: str,
    distance_head=False, partial_support=False,
) -> Path:
    for p in [data, source.parent.parent, font_teacher.parent.parent, support]:
        verify_run(p)
    if geometry not in {"initial", "fitted"}:
        raise ValueError("Only the preregistered coordinate comparison is authorized here")
    origin = json.loads((source.parent.parent / "reports/result.json").read_text())
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "warmup": str(font_teacher.resolve()),
        "warmup_sha256": digest(font_teacher),
        "intact_source": str(source.resolve()),
        "intact_source_sha256": digest(source),
        "initial_model_updates": origin["model_lineage_updates"],
        "intact_geometry": geometry,
        "steps": 4000,
        "seed": 3317,
        "prior_seed": 3417,
        "support": str(support.resolve()),
        "support_sha256": digest(support / "run.json"),
        "support_weight": 0.5,
        "latent": 2,
        "batch": 4,
        "points": 2048,
        "model_lr": 0.0001,
        "code_lr": 0.001,
        "code_l2_weight": 0.01,
        "architecture": "semantic-distance-word-ink/1"
        if distance_head
        else "semantic-word-autodecoder/1",
        "distance_head": bool(distance_head),
        "partial_support": bool(partial_support),
        "optimizer": "reset AdamW at new dataset",
        "teacher": (
            "frozen font-auxiliary neural field; retain entire original training alphabet "
            "without real glyph support"
        ),
        "target_at_generation": False,
        "target_geometry_at_generation": False,
        "training_geometry_is_manual_truth": False,
        "C_used": False,
        "protocol_sha256": digest("docs/V01017_PROTOCOL.md"),
    }
    run = create("intact-word-joint-learning", seed=r["seed"], config=r)
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    name = "partial-support" if partial_support else "distance" if distance_head else geometry
    Path(f"/tmp/context-v01017-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.semantic_autodecoder_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
