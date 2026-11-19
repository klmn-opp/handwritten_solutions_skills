"""All-personal-character views, with separate font-derived and real supervision."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.ink_geometry import signed_distance
from research.corpus.label_admission import label_text
from research.corpus.train import validate_dataset

SIZE = 96


def ink_box(ink: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.where(ink > 0.5)
    if not len(x):
        raise ValueError("No dark ink")
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def align_real(path: Path, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    ink = np.clip((190 - gray) / 140, 0, 1)
    left, top, right, bottom = ink_box(ink)
    a, b, c, d = ink_box(reference)
    scale = min((d - b) / (bottom - top), 84 / (right - left))
    width, height = max(1, round((right - left) * scale)), max(1, round((bottom - top) * scale))
    crop = Image.fromarray(np.uint8(ink[top:bottom, left:right] * 255))
    crop = crop.resize((width, height), Image.Resampling.LANCZOS)
    x, y = (SIZE - width) // 2, round((b + d - height) / 2)
    if y < 3 or y + height > SIZE - 3:
        raise ValueError("Alignment would clip ink")
    canvas = np.zeros((SIZE, SIZE), dtype=np.float32)
    canvas[y : y + height, x : x + width] = np.asarray(crop) / 255
    return canvas, {
        "mode": "dark-ink-v1; uniform whole-ink fit to reference vertical extent",
        "original_size": list(gray.shape[::-1]),
        "ink_bbox": [left, top, right, bottom],
        "scale": scale,
        "destination": [x, y, width, height],
        "reference_bbox": [a, b, c, d],
        "no_stroke_repair": True,
    }


def export_views(profile, dataset: Path, destination: Path) -> dict:
    from handnote.variants import handwritten_outline
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    from hwsol.profiles.metrics import measure_font

    destination.mkdir(parents=True, exist_ok=False)
    images = destination / "images"
    images.mkdir()
    labels = sorted((chr(c) for c in profile.manifest.personal_codepoints), key=ord)
    metrics = measure_font(profile.font, profile.manifest.personal_codepoints)
    records, targets, classes, domains, references = [], [], [], [], {}
    for row in metrics["glyphs"]:
        punctuation_exception = row["character"] in {".", ","}
        if row["excluded"] and not punctuation_exception:
            raise ValueError("Unexpected excluded personal contour requires explicit audit")
        vertices, codes = handwritten_outline(
            profile.font, row["codepoint"], row["slot"], 72 * row["scale"]
        )
        box = MplPath(vertices, codes).get_extents()
        vertices[:, 0] += 48 - (box.x0 + box.x1) / 2
        vertices[:, 1] += 32 - metrics["baseline_em"] * 72 * row["scale"]
        box = MplPath(vertices, codes).get_extents()
        # Only uniform emergency fit; punctuation keeps its small natural extent.
        fit = min(1.0, 86 / max(box.width, box.height))
        if box.x0 < 4 or box.x1 > 92 or box.y0 < 4 or box.y1 > 92:
            vertices = vertices * fit + [
                48 - (box.x0 + box.x1) * fit / 2,
                48 - (box.y0 + box.y1) * fit / 2,
            ]
        figure = Figure(figsize=(1, 1), dpi=384, facecolor="white")
        canvas = FigureCanvasAgg(figure)
        axis = figure.add_axes([0, 0, 1, 1], xlim=(0, SIZE), ylim=(0, SIZE))
        axis.set_axis_off()
        axis.add_patch(PathPatch(MplPath(vertices, codes), facecolor="black", linewidth=0))
        canvas.draw()
        image = Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("L")
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        path = images / f"font-u{row['codepoint']:04x}-v{row['slot']}.png"
        image.save(path)
        ink = 1 - np.asarray(image, dtype=np.float32) / 255
        label = row["character"]
        if row["slot"] == 0:
            references[label] = ink
        records.append(
            {
                "sample_id": f"font-{row['codepoint']}-{row['slot']}",
                "label": label,
                "image": str(path.resolve()),
                "sha256": sha256(path),
                "source_type": "personal-font-derived",
                "is_real_scan": False,
                "dataset_split": "auxiliary-train-only",
                "external_fallback": False,
                "glyph_metric": row,
                "punctuation_exception": punctuation_exception,
                "uniform_fit": fit,
            }
        )
        targets.append(ink)
        classes.append(labels.index(label))
        domains.append(0)
    if set(references) != set(labels):
        raise ValueError("All personal classes require a real personal reference contour")
    rows, gate = validate_dataset(dataset)
    validation, val_classes, val_records = [], [], []
    for row in rows:
        label = label_text(row)
        if label not in labels or row["dataset_split"] not in {"train", "validation"}:
            continue
        if row["content_type"] not in {"glyph", "math-symbol"}:
            continue
        ink, transform = align_real(Path(row["image_crop"]), references[label])
        record = {
            "source_type": "personal-real",
            "label": label,
            "original_record": row,
            "transform": transform,
            "dataset_split": row["dataset_split"],
        }
        if row["dataset_split"] == "train":
            records.append(record)
            targets.append(ink)
            classes.append(labels.index(label))
            domains.append(1)
        else:
            validation.append(ink)
            val_classes.append(labels.index(label))
            val_records.append(record)
    refs = np.stack([references[label] for label in labels])
    np.savez_compressed(
        destination / "arrays.npz",
        reference=refs,
        target=np.stack(targets),
        sdf=np.stack([signed_distance(x) for x in targets]),
        classes=np.array(classes, dtype=np.int64),
        domains=np.array(domains, dtype=np.int64),
        validation=np.stack(validation),
        val_classes=np.array(val_classes, dtype=np.int64),
    )
    atomic_json(destination / "records.json", records)
    atomic_json(destination / "validation.json", val_records)
    atomic_json(destination / "metrics.json", metrics)
    coverage = []
    for index, label in enumerate(labels):
        coverage.append(
            {
                "label": label,
                "codepoint": ord(label),
                "font_derived": sum(
                    c == index and d == 0 for c, d in zip(classes, domains, strict=True)
                ),
                "real_train": sum(
                    c == index and d == 1 for c, d in zip(classes, domains, strict=True)
                ),
                "real_validation": val_classes.count(index),
                "status": "experimental; font support is not independent real validation",
            }
        )
    atomic_json(destination / "coverage.json", coverage)
    manifest = {
        "schema": "reference-glyph-views/1",
        "labels": labels,
        "size": SIZE,
        "profile": f"{profile.manifest.profile}@{profile.manifest.version}",
        "source_files": {
            str(p.resolve()): sha256(p)
            for p in [
                profile.font,
                profile.font.parent / "profile.json",
                profile.font.parent / "sources.json",
                dataset / "manifest.json",
                dataset / "samples.jsonl",
                *[
                    dataset / name
                    for name in ("label-policy.json", "review-snapshot.json", "split-plan.json")
                    if (dataset / name).is_file()
                ],
            ]
        },
        "files": {p.name: sha256(p) for p in destination.iterdir() if p.is_file()},
        "font_derived_count": domains.count(0),
        "real_train_count": domains.count(1),
        "real_validation_count": len(validation),
        "old_readiness": gate,
        "authorization": "all personal characters experimental candidate; no production promotion",
        "real_train_classes": dict(
            Counter(labels[c] for c, d in zip(classes, domains, strict=True) if d)
        ),
        "holdout_in_arrays": False,
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def validate_views(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != "reference-glyph-views/1" or manifest["holdout_in_arrays"]:
        raise ValueError("Invalid reference dataset")
    for name, digest in manifest["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Prepared views changed")
    for path, digest in manifest["source_files"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source asset changed")
    records = json.loads((directory / "records.json").read_text())
    for row in records:
        if row["dataset_split"] not in {"train", "auxiliary-train-only"}:
            raise ValueError("Evaluation sample entered training")
        if row["source_type"] == "personal-font-derived":
            if row["external_fallback"] or row["is_real_scan"]:
                raise ValueError("Invalid font provenance")
            if sha256(Path(row["image"])) != row["sha256"]:
                raise ValueError("Font view changed")
    return manifest
