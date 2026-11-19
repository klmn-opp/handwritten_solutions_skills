"""Append new document groups without changing existing labels or exposing holdout."""

import json
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.context_dataset import audit, validate_rows
from research.corpus.experiment import create


def check_rows(rows: list[dict], assignments: dict[str, str]) -> None:
    ids, hashes = set(), set()
    for row in rows:
        split = row["dataset_split"]
        if split not in {"train", "validation"} or assignments[row["source_hash"]] != split:
            raise ValueError("Holdout or cross-document split contamination")
        if row["sample_id"] in ids or row["crop_sha256"] in hashes:
            raise ValueError("Duplicate source sample")
        ids.add(row["sample_id"])
        hashes.add(row["crop_sha256"])
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            if (a["source_hash"], a["page_number"]) != (b["source_hash"], b["page_number"]):
                continue
            x, y, r, d = a["bbox"]
            u, v, s, t = b["bbox"]
            intersection = max(0, min(r, s) - max(x, u)) * max(0, min(d, t) - max(y, v))
            if intersection / min((r - x) * (d - y), (s - u) * (t - v)) > 0.5:
                raise ValueError("Overlapping regions are not independent samples")


def extend(parent: Path, packages: list[Path], new_assignments: dict[str, str]) -> Path:
    verify_run(parent)
    manifest = json.loads((parent / "reports/manifest.json").read_text())
    assignments = dict(manifest["split_assignments"])
    for key, split in new_assignments.items():
        if split not in {"train", "validation", "sealed-holdout"}:
            raise ValueError("Unknown split")
        if key in assignments and assignments[key] != split:
            raise ValueError("Cannot reassign an existing source document")
        assignments[key] = split
    rows = [json.loads(s) for s in (parent / "reports/samples.jsonl").read_text().splitlines()]
    added = [
        r for p in packages for r in validate_rows(p, assignments, allowed={"train", "validation"})
    ]
    rows += added
    check_rows(rows, assignments)
    run = create(
        "continuous-corpus-extension",
        config={
            "parent": str(parent),
            "parent_sha256": sha256(parent / "run.json"),
            "new_assignments": new_assignments,
        },
    )
    try:
        data = run.path / "reports/samples.jsonl"
        data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        result = audit(rows)
        result.update(
            added_regions=len(added),
            fresh_document_holdout_available="sealed-holdout" in assignments.values(),
            fresh_holdout_opened=False,
        )
        atomic_json(run.path / "reports/audit.json", result)
        atomic_json(
            run.path / "reports/manifest.json",
            {
                "schema": "personal-continuous-handwriting/1",
                "version": run.path.name,
                "samples_sha256": sha256(data),
                "sample_count": len(rows),
                "split_assignments": assignments,
                "holdout_included": False,
                "parent": str(parent),
                "parent_sha256": sha256(parent / "run.json"),
                "source_packages": {str(p): sha256(p / "run.json") for p in packages},
            },
        )
        run.finish("prepared", result=result)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise
