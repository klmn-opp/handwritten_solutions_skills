"""Intact raw words: exact text labels and estimated geometry are distinct evidence."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance


def initial_geometry(boxes: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Move layout coordinates to whole-word bounds; never warp the real image."""
    y, x = np.where(ink > 0.5)
    if not len(x) or not len(boxes) or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Nonempty word and legal layout bounds required")
    target_min = np.array([x.min(), y.min()], np.float32)
    target_max = np.array([x.max() + 1, y.max() + 1], np.float32)
    low, high = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    scale = (target_max - target_min) / (high - low)
    return (boxes - np.tile(low, 2)) * np.tile(scale, 2) + np.tile(target_min, 2)


def prepare(parent: Path, reviewed: Path, old_main: Path) -> Path:
    for p in [parent, reviewed, old_main]:
        verify_run(p)
    with np.load(parent / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    rows = json.loads((parent / "reports/samples.json").read_text())
    idx = json.loads((parent / "reports/indices.json").read_text())
    strong = idx["train"]
    inputs = json.loads((reviewed / "reports/inputs.json").read_text())
    internal = {
        q["id"]: q for q in json.loads((reviewed / "reports/source-private.json").read_text())
    }
    reviews = sum(
        [json.loads((old_main / f"reports/whole-word-labels-{s}.json").read_text()) for s in "ab"],
        [],
    )
    review_by_id = {q["id"]: q for q in reviews}
    assert len(reviews) == len(review_by_id) == len(inputs) == 48
    with np.load(reviewed / "reports/arrays.npz") as f:
        raw = f["ink"]
    weak = []
    geometry = arrays["boxes"].copy()
    for i in strong:
        geometry[i] = arrays["anchor_boxes"][i]
    for source, im in zip(inputs, raw, strict=True):
        q = review_by_id[source["id"]]
        if q["status"] != "accept":
            continue
        if not q["expected_label_matches"] or q["observed_text"] != source["text"]:
            raise ValueError("Whole-word text must be reliable before position optimization")
        assert digest(source["image"]) == source["image_sha256"] == q["image_sha256"]
        assert np.array_equal(np.uint8((1 - im) * 255), np.asarray(Image.open(source["image"])))
        i = source["index"]
        assert i in idx["unused_real"] and rows[i]["text"] == q["observed_text"]
        arrays["ink"][i] = im
        arrays["sdf"][i] = signed_distance(im)
        n = int((arrays["tokens"][i, :, 0] != 0).sum())
        geometry[i, :n] = initial_geometry(arrays["boxes"][i, :n], im)
        rows[i] = {
            **rows[i],
            "no_inverse_registration": True,
            "target_transform": internal[source["id"]]["transform"],
            "target_preprocessing": "original whole word, uniform scaling ONLY",
            "word_label_review": q,
            "character_positions_verified": False,
            "character_positions_source": "estimated whole-word frame, NOT manual truth",
        }
        weak.append(i)
    assert len(strong) == 18 and len(weak) == 44
    arrays["train_geometry"] = geometry
    arrays["known_anchors"] = np.array([i in strong for i in range(len(rows))])
    for i in strong:
        rows[i]["character_positions_verified"] = True
    admitted = sorted(strong + weak)
    excluded = sorted(set(idx["unused_real"]) - set(weak))
    indices = {**idx, "train": admitted, "strong": strong, "weak": weak, "unused_real": excluded}
    run = create(
        "raw-intact-word-data",
        config={
            "sources": {
                str(p.resolve()): digest(p / "run.json") for p in [parent, reviewed, old_main]
            },
            "forced_character_cuts": False,
            "synthetic_ink": False,
            "C_used": False,
            "estimated_geometry_is_not_ground_truth": True,
        },
    )
    for i in admitted:
        path = run.path / f"solutions/raw-{i:04d}.png"
        Image.fromarray(np.uint8((1 - arrays["ink"][i]) * 255)).save(path)
        rows[i]["raw_training_image"] = str(path)
        rows[i]["raw_training_image_sha256"] = digest(path)
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/indices.json", indices)
    write(
        run.path / "reports/class-metrics.json",
        json.loads((parent / "reports/class-metrics.json").read_text()),
    )
    run.finish(
        "held",
        result={
            "raw_words": 62,
            "verified_geometry_words": 18,
            "estimated_geometry_words": 44,
            "uncertain_labels_excluded": 4,
            "C_used": False,
            "neural_updates": 0,
        },
    )
    return run.path


def prepare_joint(parent: Path, alignment: Path) -> Path:
    for p in [parent, alignment]:
        verify_run(p)
    with np.load(parent / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    with np.load(alignment / "reports/estimated-geometry.npz") as f:
        estimated = {int(i): b for i, b in zip(f["indices"], f["optimized"], strict=True)}
    idx = json.loads((parent / "reports/indices.json").read_text())
    rows = json.loads((parent / "reports/samples.json").read_text())
    quality = json.loads((alignment / "reports/alignment-quality.json").read_text())
    assert {q["index"] for q in quality} == set(idx["weak"]) == set(estimated)
    weak = [q["index"] for q in quality if q["ordered_centers"] and q["boxes_inside_page"]]
    arrays["fitted_geometry"] = arrays["train_geometry"].copy()
    for i in weak:
        arrays["fitted_geometry"][i] = estimated[i]
        rows[i]["character_positions_source"] = "frozen neural coordinate fit, NOT ground truth"
    assert len(weak) == 40
    excluded = sorted(set(idx["weak"]) - set(weak))
    indices = {
        **idx,
        "train": sorted(idx["strong"] + weak),
        "weak": weak,
        "excluded_geometry": excluded,
    }
    run = create(
        "aligned-intact-word-study-data",
        config={
            "sources": {str(p.resolve()): digest(p / "run.json") for p in [parent, alignment]},
            "geometry_gate": "ordered centers and boxes inside canvas; no loss threshold selection",
            "estimated_coordinates_not_verified_labels": True,
            "C_used": False,
        },
    )
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/indices.json", indices)
    write(
        run.path / "reports/class-metrics.json",
        json.loads((parent / "reports/class-metrics.json").read_text()),
    )
    run.finish(
        "held",
        result={
            "raw_training_words": 58,
            "verified_geometry_words": 18,
            "estimated_geometry_words": 40,
            "geometry_excluded": 4,
            "images_changed": False,
            "new_real_observations": 0,
            "neural_updates": 0,
            "C_used": False,
        },
    )
    return run.path
