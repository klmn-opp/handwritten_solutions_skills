"""Versioned exclusion of contaminated old crops; original ink is never repaired."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write


def admission(count, reviews, components, *, legacy_count=89):
    """Require complete review coverage and both semantic and boundary acceptance."""
    if count < legacy_count or len(components) != count:
        raise ValueError("Complete component audit required")
    by_index = {q["index"]: q for q in components}
    reviewed = {q["index"]: q for q in reviews}
    if set(by_index) != set(range(count)) or len(by_index) != len(components):
        raise ValueError("Exactly one component result per target required")
    if set(reviewed) != set(range(legacy_count)) or len(reviewed) != len(reviews):
        raise ValueError("Exactly one semantic review per legacy target required")
    kept, excluded = [], []
    for i in range(count):
        reasons = []
        if not by_index[i]["component_guard_passed"]:
            reasons.append("connected component crosses target boundary")
        if i < legacy_count:
            q = reviewed[i]
            if q["status"] not in {"accept", "reject", "uncertain"}:
                raise ValueError("Unknown review status")
            if q["status"] != "accept":
                reasons.append(f"review {q['status']}: {q['notes']}")
            elif q["foreign_ink"] is not False or q["truncated"] is not False:
                reasons.append("review acceptance did not exclude foreign or missing ink")
        if reasons:
            excluded.append({"index": i, "reasons": reasons})
        else:
            kept.append(i)
    if not kept:
        raise ValueError("No admitted observations")
    return kept, excluded


def prepare(data: Path, review: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    verify_run(review)
    inputs = json.loads((review / "reports/inputs.json").read_text())
    reviews = sum(
        [json.loads((review / f"reports/review-{part}.json").read_text()) for part in "ab"], []
    )
    lookup = {q["id"]: q for q in reviews}
    if set(lookup) != {q["id"] for q in inputs} or len(lookup) != len(reviews):
        raise ValueError("Exact reviewed input identifiers required")
    for q in inputs:
        r = lookup[q["id"]]
        if (
            q["index"] != r["index"]
            or digest(q["image"]) != q["image_sha256"]
            or q["image_sha256"] != r["image_sha256"]
            or digest(q["source_crop"]) != q["source_crop_sha256"]
        ):
            raise ValueError("Review source identity or image hash changed")
    components = json.loads((review / "reports/component-audit.json").read_text())
    with np.load(data / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    if "local_codes" in arrays:
        raise ValueError("Recompute morphology after exclusions; do not reuse polluted PCA")
    rows = json.loads((data / "reports/samples.json").read_text())
    for q in inputs:
        row = rows[q["index"]]
        if (row["source_crop_sha256"], row["position"], row["raw_normalized_bbox"]) != (
            q["source_crop_sha256"],
            q["position"],
            q["raw_normalized_bbox"],
        ):
            raise ValueError("Review applies to a different source crop")
    kept, excluded = admission(len(rows), reviews, components)
    old_groups = sorted(set(int(g) for g in arrays["groups"][kept]))
    selected = {k: v[kept].copy() for k, v in arrays.items()}
    selected["groups"] = np.array([old_groups.index(int(g)) for g in selected["groups"]])
    run = create(
        "reviewed-local-ink-data",
        config={
            "parent": str(data),
            "parent_sha256": digest(data / "run.json"),
            "review": str(review),
            "review_sha256": digest(review / "run.json"),
            "initial_code_groups": old_groups,
            "C_used": False,
            "original_targets_unchanged": True,
            "no_synthetic_repair": True,
            "excluded_originals_preserved": True,
            "new_real_observations": 0,
            "PCA_must_be_recomputed": True,
        },
    )
    selected_rows = []
    for j, i in enumerate(kept):
        row = dict(rows[i])
        original = Path(row["training_image"])
        if digest(original) != row["training_image_sha256"]:
            raise ValueError("Existing target image changed")
        dest = run.path / f"solutions/glyph-{j:04d}.png"
        shutil.copyfile(original, dest)
        row.update(
            original_v18_index=i,
            previous_training_image=str(original),
            training_image=str(dest),
            training_image_sha256=digest(dest),
            boundary_guard_passed=True,
            legacy_review=lookup[f"legacy-{i:03d}"] if i < 89 else None,
        )
        selected_rows.append(row)
    np.savez_compressed(run.path / "reports/arrays.npz", **selected)
    write(run.path / "reports/samples.json", selected_rows)
    write(run.path / "reports/original-indices.json", kept)
    write(
        run.path / "reports/excluded.json",
        [
            {**q, "label": rows[q["index"]]["label"], "word": rows[q["index"]]["word"]}
            for q in excluded
        ],
    )
    coverage = dict(sorted(Counter(q["label"] for q in selected_rows).items()))
    write(run.path / "reports/coverage.json", coverage)
    write(
        run.path / "reports/groups.json",
        {
            q["source_crop_sha256"]: int(g)
            for q, g in zip(selected_rows, selected["groups"], strict=True)
        },
    )
    run.finish(
        "held",
        result={
            "real_views": len(kept),
            "classes": len(coverage),
            "excluded": len(excluded),
            "neural_updates": 0,
            "new_real_observations": 0,
            "C_used": False,
            "no_original_files_changed": True,
        },
    )
    return run.path
