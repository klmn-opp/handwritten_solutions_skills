"""Traceable whole-word/fragment data; no silent labels or split reassignment."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.aligned_style_contract import text_tokens
from research.corpus.context_handnote_reference import render
from research.corpus.experiment import create
from research.corpus.word_style_data import normalize

SEALED_C = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"


def signed_field(ink: np.ndarray) -> np.ndarray:
    if ink.shape != (128, 512) or not np.isfinite(ink).all():
        raise ValueError("Finite complete region required")
    mask = np.uint8(ink > 0.5)
    if not mask.any() or mask.all():
        raise ValueError("Empty or solid region")
    return (
        cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    ).clip(-8, 8) / 8


def admit(row: dict) -> None:
    text_tokens([row["text"]])
    if row["dataset_split"] not in {"train", "validation"} or row["source_hash"] == SEALED_C:
        raise ValueError("Sealed data or unsupported split")
    if row.get("decision") != "accept" or row.get("confidence") != "high":
        raise ValueError("Uncertain crop cannot enter training")
    if sha256(Path(row["source_crop"])) != row["source_crop_sha256"]:
        raise ValueError("Changed parent pixels")
    with Image.open(row["source_crop"]) as image:
        x, y, right, bottom = row["box_in_source_crop"]
        if not (0 <= x < right <= image.width and 0 <= y < bottom <= image.height):
            raise ValueError("Out-of-source crop")


def prepare(parent: Path, additions: Path) -> Path:
    verify_run(parent)
    old = json.loads((parent / "reports/samples.json").read_text())
    proposed = json.loads(additions.read_text())
    if isinstance(proposed, dict):
        proposed = proposed["samples"]
    profile = profile_store().resolve("personal@v1")
    run = create(
        "real-ink-expanded-data",
        config={
            "parent": str(parent),
            "parent_sha256": sha256(parent / "run.json"),
            "additions": str(additions),
            "additions_sha256": sha256(additions),
            "profile": profile.provenance(),
            "C_used": False,
            "neural_updates": 0,
        },
    )
    try:
        previous = np.load(parent / "reports/arrays.npz")
        refs, targets = list(previous["reference"]), list(previous["target"])
        rows = [{**r, "kind": "whole-word", "new_to_word_learner": False} for r in old]
        seen = {r["crop_sha256"]: r for r in rows}
        excluded = []
        for q in proposed:
            if q.get("decision") != "accept":
                excluded.append({"id": q.get("parent_sample_id"), "reason": "review_rejected"})
                continue
            admit(q)
            with Image.open(q["source_crop"]) as image:
                crop = image.convert("RGB").crop(q["box_in_source_crop"])
            dest = run.path / f"solutions/addition-{len(rows):04d}.png"
            crop.save(dest)
            digest = sha256(dest)
            if digest in seen:
                prior = seen[digest]
                if prior["text"] != q["text"] or prior["dataset_split"] != q["dataset_split"]:
                    raise ValueError("Duplicate pixels have conflicting label or split")
                excluded.append({"id": q.get("parent_sample_id"), "reason": "exact_duplicate"})
                continue
            target, tr = normalize(crop)
            raw, _, provenance = render(q["text"], profile)
            reference, rr = normalize(raw)
            row = {
                **q,
                "crop": str(dest),
                "crop_sha256": digest,
                "target_transform": tr,
                "reference_transform": rr,
                "reference_glyphs": provenance,
                "real_personal_pixels": True,
                "new_to_word_learner": True,
                "new_collection": False,
                "kind": q.get("kind", "whole-word"),
            }
            rows.append(row)
            seen[digest] = row
            targets.append(target)
            refs.append(reference)
        # Retain document grouping; no validation pixels can be promoted as new Train.
        original_split = {r["source_hash"]: r["dataset_split"] for r in old}
        for r in rows:
            if (
                r["source_hash"] in original_split
                and r["dataset_split"] != original_split[r["source_hash"]]
            ):
                raise ValueError("Document split changed")
        arrays = {
            "reference": np.stack(refs),
            "target": np.stack(targets),
            "field": np.stack([signed_field(a) for a in targets]),
            "split": np.array([r["dataset_split"] for r in rows]),
        }
        np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
        atomic_json(run.path / "reports/samples.json", rows)
        report = {
            "total": len(rows),
            "previous": len(old),
            "added_views": len(rows) - len(old),
            "by_split": dict(Counter(r["dataset_split"] for r in rows)),
            "by_kind": dict(Counter(r["kind"] for r in rows)),
            "excluded": excluded,
            "new_collection": False,
            "C_used": False,
            "views_not_independent_documents": True,
        }
        atomic_json(run.path / "reports/audit.json", report)
        run.finish("prepared", result=report)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
