"""Whole-word supervision recovered from reviewed regions, never character cuts."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_handnote_reference import render
from research.corpus.experiment import create

SIZE = (512, 128)


def source_box(word: dict, alignment: dict, size: tuple[int, int]) -> list[int]:
    box = word["source_box_in_whole_view"]
    if alignment["source_box_coordinate_system"] == "original-crop":
        result = list(box)
    else:
        tr = alignment["original_whole_transform"]
        x, y = tr["ink_bbox"][:2]
        dx, dy = tr["translation"]
        scale = tr["scale"]
        result = [
            math.floor((box[0] - dx) / scale + x),
            math.floor((box[1] - dy) / scale + y),
            math.ceil((box[2] - dx) / scale + x),
            math.ceil((box[3] - dy) / scale + y),
        ]
    # Recover antialiasing outside the original dark-ink boundary; no repair.
    return [
        max(0, result[0] - 2),
        max(0, result[1] - 2),
        min(size[0], result[2] + 2),
        min(size[1], result[3] + 2),
    ]


def normalize(image: Image.Image) -> tuple[np.ndarray, dict]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = np.clip((190 - gray) / 140, 0, 1)
    yy, xx = np.where(ink > 0.1)
    if not len(xx):
        raise ValueError("Empty whole word")
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    x, y, right, bottom = box
    scale = min((SIZE[0] - 32) / (right - x), (SIZE[1] - 32) / (bottom - y))
    w, h = max(1, round((right - x) * scale)), max(1, round((bottom - y) * scale))
    patch = Image.fromarray(np.uint8(ink[y:bottom, x:right] * 255)).resize(
        (w, h), Image.Resampling.LANCZOS
    )
    out = np.zeros(SIZE[::-1], np.float32)
    dx, dy = (SIZE[0] - w) // 2, (SIZE[1] - h) // 2
    out[dy : dy + h, dx : dx + w] = np.asarray(patch) / 255
    return out, {
        "bbox": box,
        "scale": scale,
        "translation": [dx, dy],
        "source_size": list(image.size),
        "target_geometry_used_for_reference": False,
    }


def prepare(views: Path) -> Path:
    verify_run(views)
    rows = json.loads((views / "reports/samples.json").read_text())
    profile = profile_store().resolve("personal@v1")
    run = create(
        "whole-word-style-data",
        config={
            "source": str(views),
            "source_sha256": sha256(views / "run.json"),
            "profile": profile.provenance(),
            "size": SIZE,
            "C_used": False,
            "geometry": "independent uniform whole-word fit, original pixels preserved",
            "new_labels": False,
            "character_segmentation": False,
        },
    )
    metadata, references, targets, excluded, seen = [], [], [], [], {}
    try:
        for ri, row in enumerate(rows):
            alignment = row.get("alignment")
            if not alignment:
                excluded.append({"row": ri, "reason": "prior word boundaries not admitted"})
                continue
            if row["dataset_split"] not in {"train", "validation"}:
                raise ValueError("No Holdout allowed")
            if sha256(Path(row["crop"])) != row["crop_sha256"]:
                raise ValueError("Changed source crop")
            original = Image.open(row["crop"]).convert("RGB")
            tokens = row["normalized_label"].split()
            for wi, (token, word) in enumerate(zip(tokens, alignment["words"], strict=True)):
                if not 2 <= len(token) <= 18:
                    excluded.append({"row": ri, "word": wi, "reason": "finite word pilot length"})
                    continue
                box = source_box(word, alignment, original.size)
                crop = original.crop(box)
                index = len(metadata)
                path = run.path / f"solutions/word-{index:04d}.png"
                crop.save(path)
                digest = sha256(path)
                key = (token, digest)
                if key in seen:
                    path.unlink()  # New unpublished derived file only, never an original.
                    if seen[key] != row["dataset_split"]:
                        raise ValueError("Exact crop crosses split")
                    excluded.append({"row": ri, "word": wi, "reason": "duplicate derived crop"})
                    continue
                seen[key] = row["dataset_split"]
                target, tr = normalize(crop)
                ref_raw, _, glyphs = render(token, profile)
                reference, rr = normalize(ref_raw)
                targets.append(target)
                references.append(reference)
                metadata.append(
                    {
                        "word_id": f"{row['sample_id']}:word:{wi}",
                        "text": token,
                        "parent_sample_id": row["sample_id"],
                        "parent_row": ri,
                        "word_index": wi,
                        "document_id": row["document_id"],
                        "page_number": row["page_number"],
                        "source_hash": row["source_hash"],
                        "source_crop": row["crop"],
                        "source_crop_sha256": row["crop_sha256"],
                        "parent_bbox": row["bbox"],
                        "box_in_source_crop": box,
                        "crop": str(path),
                        "crop_sha256": digest,
                        "dataset_split": row["dataset_split"],
                        "label_source": row["label_source"],
                        "word_boundary_source": alignment["word_boundary_source"],
                        "target_transform": tr,
                        "reference_transform": rr,
                        "reference_glyphs": glyphs,
                        "real_personal_pixels": True,
                        "new_session": "new-english-sol-review" in row["source_package"],
                    }
                )
                for name, ink in [("target", target), ("reference", reference)]:
                    Image.fromarray(np.uint8((1 - ink) * 255)).save(
                        run.path / f"comparisons/{index:04d}-{name}.png"
                    )
        np.savez_compressed(
            run.path / "reports/arrays.npz",
            target=np.stack(targets),
            reference=np.stack(references),
            split=np.array([r["dataset_split"] for r in metadata]),
        )
        counts = Counter(r["text"] for r in metadata if r["dataset_split"] == "train")
        report = {
            "words": len(metadata),
            "by_split": dict(Counter(r["dataset_split"] for r in metadata)),
            "unique_train_labels": len(counts),
            "repeated_train_labels": {k: v for k, v in counts.items() if v > 1},
            "new_real_pixels": False,
            "C_used": False,
            "excluded": excluded,
            "target_normalization_loses_absolute_size_and_spacing": True,
        }
        atomic_json(run.path / "reports/samples.json", metadata)
        atomic_json(run.path / "reports/audit.json", report)
        run.finish("prepared", result={k: v for k, v in report.items() if k != "excluded"})
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def accepted_indices(reviews: list[dict], count: int) -> list[int]:
    decisions = {}
    for row in reviews:
        i = row["index"]
        if type(i) is not int or not 0 <= i < count or i in decisions:
            raise ValueError("Duplicate or invalid word review index")
        if row["decision"] not in {"accept", "reject"}:
            raise ValueError("Unknown word review decision")
        decisions[i] = row["decision"]
    if set(decisions) != set(range(count)):
        raise ValueError("Every derived word requires crop integrity review")
    return [i for i in range(count) if decisions[i] == "accept"]


def filter_reviewed(data: Path, review: Path, decisions: list[dict]) -> Path:
    verify_run(data)
    verify_run(review)
    rows = json.loads((data / "reports/samples.json").read_text())
    ids = accepted_indices(decisions, len(rows))
    run = create(
        "whole-word-reviewed-training-data",
        config={
            "source": str(data),
            "source_sha256": sha256(data / "run.json"),
            "review": str(review),
            "review_sha256": sha256(review / "run.json"),
            "C_used": False,
            "human_review": False,
            "labels_changed": False,
        },
    )
    try:
        arrays = np.load(data / "reports/arrays.npz")
        np.savez_compressed(run.path / "reports/arrays.npz", **{k: arrays[k][ids] for k in arrays})
        selected = [{**rows[i], "parent_word_index": i} for i in ids]
        atomic_json(run.path / "reports/samples.json", selected)
        atomic_json(run.path / "reports/decisions.json", decisions)
        result = {
            "words": len(ids),
            "excluded": len(rows) - len(ids),
            "by_split": dict(Counter(r["dataset_split"] for r in selected)),
            "no_label_repair": True,
            "originals_untouched": True,
        }
        atomic_json(run.path / "reports/audit.json", result)
        run.finish("prepared", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
