"""Shared pen geometry followed by fixed mild foreground-tone convergence.

The second stage never changes any alpha .25/.5/.75 mask or dark core. It is a
bounded deterministic correction, not additional neural training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import ink_model as base
from . import ink_shared as shared

SCHEMA = "shared-pen-and-tone/1"
CONFIG = {
    "tone_convergence_fraction": 0.5,
    "maximum_gray_shift": 16,
    "bands": [[64, 127], [128, 191]],
    "minimum_mutable_pixels": 6,
    "bisection_steps": 12,
    "target": "median independent CPU foreground density per explicit size_class",
    "protected": "original dark core 0..63, faint/background192..255 and all .25/.5/.75 masks",
}


def document_target(items) -> dict:
    items = list(items)
    width = shared.document_target(items)
    densities = defaultdict(list)
    for item in items:
        if item.get("teacher_gray") is None:
            continue
        if (
            item.get("role") in shared.CONFIG["excluded_roles"]
            or item.get("kind") in shared.CONFIG["excluded_kinds"]
        ):
            continue
        m = base.measure(item["teacher_gray"])
        if (
            m["area"] >= shared.CONFIG["minimum_area"]
            and m["height"] >= shared.CONFIG["minimum_height"]
        ):
            densities[shared.size_class(item)].append(m["density"])
    return {
        "schema": SCHEMA,
        "width": width,
        "density": {k: float(np.median(v)) for k, v in densities.items()},
        "config": CONFIG,
    }


def tone(image: Image.Image, target: float, eligible: bool) -> tuple[Image.Image, dict]:
    original = np.asarray(image.convert("L"))
    foreground = original < 192
    mutable = (original >= 64) & foreground
    before = base.measure(image)["density"]
    wanted = before + CONFIG["tone_convergence_fraction"] * (target - before)
    if not eligible or mutable.sum() < CONFIG["minimum_mutable_pixels"]:
        return image.copy(), {"eligible": False, "changed_pixels": 0, "target_density": target}

    def shifted(amount):
        output = original.copy()
        for low, high in CONFIG["bands"]:
            mask = (original >= low) & (original <= high)
            output[mask] = np.rint(np.clip(original[mask].astype(float) + amount, low, high))
        observed = float(np.mean(1 - output[foreground].astype(float) / 255))
        return output, observed

    lo, hi = -CONFIG["maximum_gray_shift"], CONFIG["maximum_gray_shift"]
    best, observed = original.copy(), before
    best_error = abs(before - wanted)
    amount_best = 0.0
    for _ in range(CONFIG["bisection_steps"]):
        amount = (lo + hi) / 2
        output, current = shifted(amount)
        if abs(current - wanted) < best_error:
            best, observed, best_error, amount_best = output, current, abs(current - wanted), amount
        if current > wanted:
            lo = amount
        else:
            hi = amount
    assert np.array_equal(best[original < 64], original[original < 64])
    assert np.array_equal(best[original >= 192], original[original >= 192])
    assert all(np.array_equal(best < cut, original < cut) for cut in [64, 128, 192])
    return Image.fromarray(best), {
        "eligible": True,
        "before_density": before,
        "after_density": observed,
        "target_density": target,
        "halfway_density": wanted,
        "applied_gray_shift": amount_best,
        "changed_pixels": int(np.count_nonzero(best != original)),
        "core_background_and_three_masks_exact": True,
    }


class Model:
    def correct_regions(
        self, items: list[dict], mode: str = "combined", target: dict | None = None
    ):
        if not items:
            return [], []
        if mode not in {"combined", "shared", "none"}:
            raise ValueError("Unknown combined ink mode")
        if target is None:
            target = document_target(items)
        if target["schema"] != SCHEMA:
            raise ValueError("Unknown combined target")
        images, logs = shared.Model().correct_regions(
            items, mode="none" if mode == "none" else "shared", target=target["width"]
        )
        results = []
        records = []
        for item, image, log in zip(items, images, logs, strict=True):
            key = shared.size_class(item)
            if mode == "combined":
                image, post = tone(image, target["density"][key], log["eligible"])
            else:
                post = {"changed_pixels": 0, "enabled": False}
            results.append(image)
            records.append(
                {
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "mode": mode,
                    "shape": log,
                    "tone": post,
                    "final_weight": shared.ink_measure(image)["weight"],
                    "final_density": base.measure(image)["density"],
                    "changed_pixels": int(
                        np.count_nonzero(np.asarray(image) != np.asarray(item["gray"].convert("L")))
                    ),
                    "neural_updates": 0,
                }
            )
        return results, records


def load_model(path: Path | str | None = None) -> Model:
    if path is not None:
        bundle = json.loads(Path(path).read_text())
        if bundle["schema"] != SCHEMA or bundle["config"] != CONFIG:
            raise ValueError("Unknown combined ink bundle")
    return Model()
