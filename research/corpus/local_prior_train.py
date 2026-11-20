"""Versioned, isolated continuation for matching supervision and latent priors."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(source: Path, *, steps: int = 4000) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(source)
    previous = json.loads((source / "reports/request.json").read_text())
    result = json.loads((source / "reports/result.json").read_text())
    if not previous.get("modulated") or not previous.get("density_normalized"):
        raise ValueError("Registered modulated, density-normalized source required")
    if previous.get("distance_regularized") or steps <= 0:
        raise ValueError("Distance experiment is not the registered neighbourhood source")
    if not previous.get("capacity_diagnostic"):
        raise ValueError("Full-class expansion requires a subsequent registered decision")
    if steps != 4000 or previous["steps"] != 8000:
        raise ValueError("First comparison: frozen 8k f/t source, 4000 additional updates")
    checkpoint = source / f"logs/checkpoint-{previous['steps']:05d}.pt"
    cfg = {k: v for k, v in previous.items() if k != "run"}
    cfg.update(
        steps=steps,
        continuation=str(checkpoint),
        continuation_sha256=digest(checkpoint),
        continuation_step=previous["steps"],
        inherited_model_updates=result["model_lineage_updates"],
        matched_neighborhood=True,
        neighborhood_sigma=0.15,
        shared_word_variable_at_generation=False,
        prior_center_is_seen_training_style=True,
        novel_shape_verified=False,
        generation_retrieves_no_bitmap=True,
        protocol_sha256=digest("docs/V01019_PROTOCOL.md"),
    )
    run = create("matched-local-prior-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01019-ft-neighborhood").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.local_real_train",
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
