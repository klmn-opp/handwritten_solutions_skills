"""Replace admitted targets with original uniformly scaled ink and actual anchors.

The original whole words remain intact; only reviewed Train records are admitted.
Font references are retained separately for comparison, never as real targets.
"""

import json
from pathlib import Path

import numpy as np

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance
from research.corpus.semantic_anchors import validate


def class_metrics(tokens: np.ndarray, boxes: np.ndarray, indices: list[int]) -> dict:
    """Train-only shape ratios; no evaluation images or target positions."""
    observations = {}
    for i in indices:
        for token, box in zip(tokens[i], boxes[i], strict=True):
            if not token[0]:
                continue
            width, height = box[2:] - box[:2]
            if width <= 0 or height <= 0:
                raise ValueError("Real glyph extent must be positive")
            observations.setdefault(chr(token[0]), []).append(float(width / height))
    return {
        char: {"count": len(values), "aspect_median": float(np.median(values))}
        for char, values in sorted(observations.items())
    }


def apply_metrics(tokens: np.ndarray, boxes: np.ndarray, metrics: dict) -> np.ndarray:
    """Keep baseline/height/centre; change width only from a frozen Train statistic."""
    result = boxes.copy()
    for i in range(len(tokens)):
        for j, token in enumerate(tokens[i]):
            entry = metrics.get(chr(token[0])) if token[0] else None
            if entry is None:
                continue
            center = (boxes[i, j, 0] + boxes[i, j, 2]) / 2
            width = (boxes[i, j, 3] - boxes[i, j, 1]) * entry["aspect_median"]
            result[i, j, 0], result[i, j, 2] = center - width / 2, center + width / 2
    return result


def prepare(parent: Path, raw: Path) -> Path:
    for p in [parent, raw]:
        verify_run(p)
    with np.load(parent / "reports/arrays.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    idx = json.loads((parent / "reports/indices.json").read_text())
    rows = json.loads((parent / "reports/samples.json").read_text())
    records = json.loads((raw / "reports/validated-raw-anchors.json").read_text())
    with np.load(raw / "reports/arrays.npz") as archive:
        images = archive["ink"]
    if len(records) != len(images) or {q["index"] for q in records} != set(idx["train"]):
        raise ValueError("Exact reviewed Train population required")
    for q, ink in zip(records, images, strict=True):
        i = q["index"]
        if rows[i]["split"] != "train" or rows[i]["text"] != q["text"]:
            raise ValueError("Real target cannot change label/split")
        if digest(q["image"]) != q["sha256"] or digest(q["source_crop"]) != q["source_crop_sha256"]:
            raise ValueError("Raw reviewed source changed")
        boxes = validate(q["review"], q["text"], ink)
        arrays["ink"][i] = ink
        arrays["sdf"][i] = signed_distance(ink)
        arrays["anchor_boxes"][i, : len(boxes)] = boxes
        rows[i] = {
            **rows[i],
            "target_preprocessing": "original whole word, uniform scaling ONLY",
            "target_transform": q["transform"],
            "no_inverse_registration": True,
            "review_sha256": digest(raw / "reports/validated-raw-anchors.json"),
            "raw_review_id": q["id"],
        }
    run = create(
        "semantic-raw-word-data",
        config={
            "sources": {str(p.resolve()): digest(p / "run.json") for p in [parent, raw]},
            "C_used": False,
            "real_vs_font_separate": True,
            "no_inverse_registration": True,
            "character_image_cuts": False,
            "unused_real_targets_not_admitted": idx["unused_real"],
        },
    )
    metrics = class_metrics(arrays["tokens"], arrays["anchor_boxes"], idx["train"])
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/indices.json", idx)
    write(run.path / "reports/class-metrics.json", metrics)
    run.finish(
        "held",
        result={
            "raw_train_words": len(idx["train"]),
            "classes": len(metrics),
            "instances": sum(q["count"] for q in metrics.values()),
            "C_used": False,
            "new_neural_updates": 0,
        },
    )
    return run.path
