"""Reviewed character location anchors; source words and connected ink stay intact."""

import json
from pathlib import Path

import numpy as np

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def validate(review: dict, text: str, ink: np.ndarray) -> np.ndarray:
    if review["status"] != "accept" or review["text"] != text or ink.shape != (128, 512):
        raise ValueError("Only accepted, exact-label whole-word anchors are eligible")
    glyphs = review["glyphs"]
    if len(glyphs) != len(text):
        raise ValueError("Every character must have a reliable anchor in this pilot")
    coverage = np.zeros_like(ink, bool)
    boxes = []
    for j, (char, glyph) in enumerate(zip(text, glyphs, strict=True)):
        if glyph["position"] != j or glyph["char"] != char:
            raise ValueError("Anchor character identity/order mismatch")
        box = glyph["bbox"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Integer half-open pixel bounds required")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 128):
            raise ValueError("Anchor crosses the unmodified image")
        if not (ink[y0:y1, x0:x1] > 0.5).any():
            raise ValueError("An anchor cannot describe blank paper")
        coverage[y0:y1, x0:x1] = True
        boxes.append(box)
    mask = ink > 0.5
    if not mask.any() or ((mask & coverage).sum() / mask.sum()) < 0.99:
        raise ValueError("Reviewed anchors omit strong real ink")
    return np.asarray(boxes, np.float32)


def prepare(data: Path, main: Path) -> Path:
    verify_run(data)
    with np.load(data / "reports/arrays.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    reviews = []
    for part in "abcd":
        reviews.extend(json.loads((main / f"reports/real-anchor-review-{part}.json").read_text()))
    sources = json.loads((main / "reports/real-anchor-input.json").read_text()) + json.loads(
        (main / "reports/real-anchor-extra-input.json").read_text()
    )
    source_by_id = {q["id"]: q for q in sources}
    assert len(reviews) == 66 and len({q["index"] for q in reviews}) == 66
    assert {q["index"] for q in reviews} == set(idx["train"])
    selected = []
    anchor_boxes = arrays["boxes"].copy()
    for q in reviews:
        i = q["index"]
        source = source_by_id[q["id"]]
        assert source["index"] == i and source["text"] == rows[i]["text"]
        assert digest(source["image"]) == q["image_sha256"] == source["sha256"]
        if q["status"] != "accept":
            continue
        boxes = validate(q, rows[i]["text"], arrays["ink"][i])
        anchor_boxes[i, : len(boxes)] = boxes
        selected.append(i)
    if len(selected) < 10:
        raise ValueError("Insufficient reliable anchors for the preregistered capacity comparison")
    run = create(
        "semantic-real-anchor-data",
        config={
            "data": str(data.resolve()),
            "data_sha256": digest(data / "run.json"),
            "reviews": {
                str(main / f"reports/real-anchor-review-{p}.json"): digest(
                    main / f"reports/real-anchor-review-{p}.json"
                )
                for p in "abcd"
            },
            "image_changes": False,
            "character_image_cuts": False,
            "C_used": False,
            "target_derived_training_geometry": True,
            "generation_geometry": "unchanged Handnote, NEVER target anchors",
        },
    )
    arrays["anchor_boxes"] = anchor_boxes
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    indices = {
        **idx,
        "train": sorted(selected),
        "unused_real": sorted(set(idx["train"]) - set(selected)),
    }
    write(run.path / "reports/indices.json", indices)
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/anchor-reviews.json", reviews)
    coverage = {}
    for i in selected:
        for char in rows[i]["text"]:
            coverage[char] = coverage.get(char, 0) + 1
    run.finish(
        "held",
        result={
            "training_words": len(selected),
            "anchored_instances": sum(coverage.values()),
            "class_coverage": coverage,
            "unused_words": len(indices["unused_real"]),
            "not_new_real_samples": True,
            "neural_updates": 0,
            "C_used": False,
        },
    )
    return run.path
