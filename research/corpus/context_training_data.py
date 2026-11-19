"""Whole real regions paired to exact text rendered with the unchanged personal font."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_dataset import PILOT_MINIMUM
from research.corpus.experiment import create
from research.corpus.reference_data import signed_distance

SIZE = (512, 64)


def whole_ink(image: Image.Image) -> tuple[np.ndarray, dict]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = np.clip((190 - gray) / 140, 0, 1)
    yy, xx = np.where(ink > 0.5)
    if not len(xx):
        raise ValueError("No dark ink in complete region")
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    x, y, right, bottom = box
    scale = min(480 / (right - x), 44 / (bottom - y))
    w, h = max(1, round((right - x) * scale)), max(1, round((bottom - y) * scale))
    crop = Image.fromarray(np.uint8(255 * ink[y:bottom, x:right])).resize(
        (w, h), Image.Resampling.LANCZOS
    )
    canvas = np.zeros(SIZE[::-1], np.float32)
    dx, dy = (SIZE[0] - w) // 2, (SIZE[1] - h) // 2
    canvas[dy : dy + h, dx : dx + w] = np.asarray(crop) / 255
    return canvas, {
        "recipe": "whole-region-dark-ink-uniform-fit/1",
        "original_size": image.size,
        "ink_bbox": box,
        "scale": scale,
        "translation": [dx, dy],
        "character_segmentation": False,
        "stroke_repair": False,
    }


def reference_image(text: str, font_path: Path) -> Image.Image:
    font = ImageFont.truetype(str(font_path), 64)
    left, top, right, bottom = font.getbbox(text)
    canvas = Image.new("RGB", (right - left + 32, bottom - top + 32), "white")
    ImageDraw.Draw(canvas).text((16 - left, 16 - top), text, font=font, fill="black")
    return canvas


def prepare(dataset: Path) -> Path:
    verify_run(dataset)
    audit = json.loads((dataset / "reports/audit.json").read_text())
    if not audit["short_region_pilot_go"]:
        raise ValueError("Continuous corpus below predeclared limited-pilot threshold")
    rows = [
        json.loads(line) for line in (dataset / "reports/samples.jsonl").read_text().splitlines()
    ]
    rows = [r for r in rows if r["content_type"].startswith("english")]
    profile = profile_store().resolve("personal@v1")
    run = create(
        "context-whole-region-views",
        config={
            "dataset": str(dataset.resolve()),
            "dataset_sha256": sha256(dataset / "run.json"),
            "profile": profile.provenance(),
            "size": SIZE,
            "thresholds": PILOT_MINIMUM,
            "font_reference_is_auxiliary_not_real": True,
        },
    )
    try:
        images, refs, metadata = [], [], []
        for row in rows:
            text = row["normalized_label"]
            if (
                "\n" in text
                or len(text) > 100
                or any(ord(c) not in profile.manifest.coverage for c in text)
            ):
                raise ValueError(
                    "Unrenderable exact text requires explicit exclusion, not truncation"
                )
            raw = Path(row["crop"])
            if sha256(raw) != row["crop_sha256"]:
                raise ValueError("Raw supervised region changed")
            real, transform = whole_ink(Image.open(raw))
            reference, ref_transform = whole_ink(reference_image(text, profile.font))
            images.append(real)
            refs.append(reference)
            metadata.append(
                {
                    **row,
                    "learning_transform": transform,
                    "reference_transform": ref_transform,
                    "reference_type": "personal-font-raster-not-new-handwriting",
                    "role": int(len(text.split()) > 1),
                }
            )
        targets = np.stack(images)
        refs = np.stack(refs)
        np.savez_compressed(
            run.path / "reports/arrays.npz",
            target=targets,
            reference=refs,
            target_sdf=np.stack([signed_distance(x) for x in targets]),
            reference_sdf=np.stack([signed_distance(x) for x in refs]),
            split=np.array([r["dataset_split"] for r in metadata]),
            role=np.array([r["role"] for r in metadata], dtype=np.int64),
        )
        atomic_json(run.path / "reports/samples.json", metadata)
        atomic_json(
            run.path / "reports/manifest.json",
            {
                "architecture_scope": "complete English words and short multiword regions",
                "samples": len(metadata),
                "arrays_sha256": sha256(run.path / "reports/arrays.npz"),
                "samples_sha256": sha256(run.path / "reports/samples.json"),
                "dataset": str(dataset.resolve()),
                "font_sha256": sha256(profile.font),
                "profile_sha256": profile.manifest_sha256,
                "training_math_regions": 0,
                "holdout_included": False,
            },
        )
        run.finish(
            "prepared",
            result={
                "english_regions": len(metadata),
                "true_train": sum(r["dataset_split"] == "train" for r in metadata),
                "validation": sum(r["dataset_split"] == "validation" for r in metadata),
                "character_segmentation": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    print(prepare(parser.parse_args().dataset))
