"""Conservative partial anchors in intact, already transcribed real Train words."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def validate_partial(review: dict, text: str, ink: np.ndarray) -> list[dict]:
    """Validate declarations and reject rectangles that cut a connected ink component.

    This is an engineering guard, not automatic semantic verification. Accepted
    identities and absence of foreign ink still depend on the recorded reviewer.
    """
    if review["text"] != text or ink.shape != (128, 512):
        raise ValueError("Exact reviewed word and original canvas required")
    occupied = np.zeros_like(ink, bool)
    mask = ink > 0.5
    _, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    totals = np.bincount(components.ravel())
    seen = set()
    accepted = []
    for q in review["certain_glyphs"]:
        j = q["position"]
        if type(j) is not int or not 0 <= j < len(text) or j in seen or q["char"] != text[j]:
            raise ValueError("Duplicate or incorrect character identity/position")
        seen.add(j)
        box = q["bbox"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Integer half-open bounds required")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 128):
            raise ValueError("Box must be inside the unchanged canvas")
        local = mask[y0:y1, x0:x1]
        if not local.any() or (occupied[y0:y1, x0:x1] & local).any():
            raise ValueError("Blank glyph or duplicated claimed ink")
        counts = np.bincount(components[y0:y1, x0:x1].ravel(), minlength=len(totals))
        ids = np.flatnonzero(counts[1:]) + 1
        if np.any(counts[ids] != totals[ids]):
            raise ValueError("Box cuts a connected real ink component")
        occupied[y0:y1, x0:x1] = True
        accepted.append(q)
    unresolved = review["unresolved_positions"]
    if len(set(unresolved)) != len(unresolved) or set(unresolved) != set(range(len(text))) - seen:
        raise ValueError("Every remaining position must remain explicitly unresolved")
    return accepted


def prepare(data: Path, review: Path, old_support: Path) -> Path:
    """Append safe views; never replace old images or reinterpret estimates as truth."""
    for p in [data, review, old_support]:
        verify_run(p)
    with np.load(data / "reports/arrays.npz") as f:
        ink = f["ink"]
    with np.load(old_support / "reports/arrays.npz") as f:
        images, labels = list(f["ink"]), list(f["labels"])
    records = json.loads((old_support / "reports/samples.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    idx = json.loads((data / "reports/indices.json").read_text())
    sources = json.loads((review / "reports/inputs.json").read_text())
    reviews = sum(
        [json.loads((review / f"reports/review-{part}.json").read_text()) for part in "ab"], []
    )
    by_id = {q["id"]: q for q in reviews}
    if len(by_id) != len(reviews) or set(by_id) != {q["id"] for q in sources}:
        raise ValueError("Exactly one review per source required")
    additions, rejected = [], []
    for s in sources:
        q, i = by_id[s["id"]], s["index"]
        if i not in idx["weak"] or q["index"] != i or rows[i]["split"] != "train":
            raise ValueError("Only the preregistered weak Train population is eligible")
        if digest(s["image"]) != s["image_sha256"] or q["image_sha256"] != s["image_sha256"]:
            raise ValueError("Reviewed source changed")
        if not np.array_equal(np.uint8((1 - ink[i]) * 255), np.asarray(Image.open(s["image"]))):
            raise ValueError("Reviewed pixels differ from training source")
        positions = [g["position"] for g in q["certain_glyphs"]]
        unresolved = q["unresolved_positions"]
        if (
            len(set(positions)) != len(positions)
            or len(set(unresolved)) != len(unresolved)
            or set(unresolved) != set(range(len(q["text"]))) - set(positions)
        ):
            raise ValueError("Reviewer must explicitly account for every character once")
        # A single erroneous box must not silently corrupt or discard the other
        # independently reviewed characters. Each rejected claim is retained.
        try:
            validate_partial(q, rows[i]["text"], ink[i])
            safe = q["certain_glyphs"]
        except ValueError:
            safe = []
            for g in q["certain_glyphs"]:
                one = {
                    **q, "certain_glyphs": [g],
                    "unresolved_positions": [
                        j for j in range(len(q["text"])) if j != g["position"]
                    ],
                }
                try:
                    validate_partial(one, rows[i]["text"], ink[i])
                    safe.append(g)
                except ValueError as exc:
                    rejected.append({"id": q["id"], "glyph": g, "reason": str(exc)})
            validate_partial(
                {**q, "certain_glyphs": safe,
                 "unresolved_positions": [j for j in range(len(q["text"]))
                                          if j not in {g["position"] for g in safe}]},
                rows[i]["text"], ink[i],
            )
        for g in safe:
            x0, y0, x1, y1 = g["bbox"]
            crop = ink[i, y0:y1, x0:x1]
            im = np.zeros((96, 96), np.float32)
            im[16:80, 16:80] = np.asarray(
                Image.fromarray(crop).resize((64, 64), Image.Resampling.BILINEAR)
            )
            images.append(im)
            labels.append(ord(g["char"]))
            additions.append({
                "id": f"partial-support-{len(additions):03d}", "label": g["char"],
                "index": i, "position": g["position"], "word": rows[i]["text"],
                "source_crop": rows[i]["crop"], "source_crop_sha256": rows[i]["crop_sha256"],
                "source_hash": rows[i]["source_hash"], "page_number": rows[i]["page_number"],
                "raw_normalized_bbox": g["bbox"], "transform": rows[i]["target_transform"],
                "review_id": q["id"], "review_glyph": g, "original_word_image": s["image"],
                "original_word_image_sha256": s["image_sha256"], "new_real_observation": False,
                "source_role": "reviewed partial English glyph; original word preserved",
            })
    if not additions:
        raise ValueError("No safely admitted additional real glyphs")
    run = create("partial-real-ink-support", config={
        "sources": {str(p): digest(p / "run.json") for p in [data, review, old_support]},
        "C_used": False, "synthetic_ink": False, "estimated_boxes_used": False,
        "normalization": "per-axis reviewed box to central64 in96, TRAIN ONLY",
        "whole_word_images_changed": False, "old_support_preserved": True,
    })
    start = len(records)
    for j, q in enumerate(additions, start):
        path = run.path / f"solutions/support-{j:03d}.png"
        Image.fromarray(np.uint8((1 - images[j]) * 255)).save(path)
        q.update(image=str(path), sha256=digest(path))
    records.extend(additions)
    np.savez_compressed(
        run.path / "reports/arrays.npz", ink=np.array(images), labels=np.array(labels)
    )
    write(run.path / "reports/samples.json", records)
    write(run.path / "reports/rejected-claims.json", rejected)
    coverage = {
        "before": dict(sorted(Counter(q["label"] for q in records[:start]).items())),
        "added": dict(sorted(Counter(q["label"] for q in additions).items())),
        "after": dict(sorted(Counter(q["label"] for q in records).items())),
    }
    write(run.path / "reports/coverage.json", coverage)
    run.finish("held", result={
        "derived_views": len(images), "added_reviewed_views": len(additions),
        "classes": len(set(labels)), "rejected_claims": len(rejected),
        "new_real_observations": 0, "neural_updates": 0, "C_used": False,
    })
    return run.path
