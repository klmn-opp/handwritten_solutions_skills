"""A fixed f/t capacity diagnostic, retaining exact verified targets and conditions."""

import json
from pathlib import Path

import numpy as np

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def prepare(parent: Path) -> Path:
    verify_run(parent)
    with np.load(parent / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    ids = np.flatnonzero(np.isin(arrays["labels"], [ord("f"), ord("t")]))
    rows = json.loads((parent / "reports/samples.json").read_text())
    old_groups = sorted(set(int(g) for g in arrays["groups"][ids]))
    arrays = {k: v[ids].copy() for k, v in arrays.items()}
    arrays["groups"] = np.array([old_groups.index(int(g)) for g in arrays["groups"]])
    priors = json.loads((parent / "reports/local-priors.json").read_text())
    run = create(
        "verified-ft-capacity-data",
        config={
            "parent": str(parent),
            "parent_sha256": digest(parent / "run.json"),
            "per_class_coordinates": True,
            "capacity_diagnostic_only": True,
            "targets_and_local_codes_unchanged": True,
            "C_used": False,
            "initial_code_groups": old_groups,
        },
    )
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    write(run.path / "reports/samples.json", [rows[i] for i in ids])
    write(run.path / "reports/local-priors.json", {str(c): priors[str(c)] for c in [102, 116]})
    write(run.path / "reports/original-indices.json", ids.tolist())
    run.finish(
        "held",
        result={
            "real_views": len(ids),
            "f": int((arrays["labels"] == 102).sum()),
            "t": int((arrays["labels"] == 116).sum()),
            "word_groups": len(old_groups),
            "neural_updates": 0,
            "C_used": False,
        },
    )
    return run.path
