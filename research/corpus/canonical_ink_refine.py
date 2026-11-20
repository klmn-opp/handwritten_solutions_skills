"""Equal-budget continuation versus verified small-component weighting."""

import json
import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write
from research.corpus.thin_ink_data import component_weights


def launch(source: Path, review: Path, gate: Path, *, balanced: bool) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [source, review]:
        verify_run(p)
    approval = json.loads(gate.read_text())
    if (
        not approval.get("go_paired_component_refinement")
        or approval["source_sha256"] != digest(source / "run.json")
        or approval["review_sha256"] != digest(review / "run.json")
    ):
        raise ValueError("Frozen paired-refinement gate required")
    audit = json.loads((review / "reports/agent-review.json").read_text())
    assert len(audit["items"]) == 35 and all(q["status"] == "accept" for q in audit["items"])
    previous = json.loads((source / "reports/request.json").read_text())
    assert previous["steps"] == 4000
    with np.load(source / "reports/targets.npz") as a:
        ink = a["ink"].copy()
        base = 1 + 3 * ink
        weights = base.copy()
        affected = np.flatnonzero(a["real"] & np.isin(a["labels"], [ord("i"), ord("T")]))
        assert len(affected) == 35
        if balanced:
            for i in affected:
                gain = np.maximum(
                    1, component_weights(ink[i], expected_shape=(96, 96), balance_to_largest=True)
                )
                weights[i] *= gain
                weights[i] *= base[i].sum(dtype=np.float64) / weights[i].sum(dtype=np.float64)
        assert np.allclose(weights.sum((1, 2)), base.sum((1, 2)), rtol=1e-6)
    cp = source / "logs/checkpoint-04000.pt"
    cfg = {
        **previous,
        "source": str(source),
        "source_sha256": digest(source / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "source_checkpoint_step": 4000,
        "gate": str(gate),
        "gate_sha256": digest(gate),
        "review": str(review),
        "review_sha256": digest(review / "run.json"),
        "steps": 1000,
        "seed": 4321,
        "inherited_model_updates": 5000,
        "learning_rate": 0.0003,
        "component_balanced": balanced,
        "preserve_per_image_total_weight": True,
        "raw_target_unchanged": True,
        "only_reweight_verified_i_T_components": True,
        "protocol_sha256": digest("docs/V01021_PROTOCOL.md"),
    }
    cfg.pop("run", None)
    run = create("canonical-ink-component-refinement", seed=cfg["seed"], config=cfg)
    weights_file = run.path / "reports/pixel-weights.npz"
    np.savez_compressed(weights_file, weights=weights)
    cfg.update(pixel_weights=str(weights_file), pixel_weights_sha256=digest(weights_file))
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    name = "component-balanced" if balanced else "component-control"
    Path(f"/tmp/context-v01021-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.canonical_ink_full",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
