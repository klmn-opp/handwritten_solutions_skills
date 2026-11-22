"""CPU optical stroke coverage normalization for English and mathematics.

No learned weights. Width and core darkness are separated. The original support,
centerline intensity, and topology at alpha .25/.5 are protected. NumPy/PIL only.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "uniform-optical-ink/1"
CONFIG = {
    "width_ratio_deadband": 0.08,
    "width_absolute_deadband": 0.08,
    "max_edge_fraction": 0.6,
    "edge_steps": 12,
    "mass_ratio_bounds": [0.7, 1.3],
    "minimum_area": 12,
    "minimum_height": 6,
    "minimum_mutable_support_alpha": 0.02,
    "topology_thresholds": [0.25, 0.5],
    "teacher_size_exponent": 0.5,
    "teacher_size_ratio_bounds": [0.75, 1.33],
    "pale_core_threshold": 0.90,
    "maximum_pale_core_gain": 1.2,
    "maximum_new_foreground_ratio": 1.30,
    "minimum_retained_foreground_ratio": 0.70,
}


def alpha(image: Image.Image) -> np.ndarray:
    return 1 - np.asarray(image.convert("L"), dtype=np.float32) / 255


def gray(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(255 * (1 - np.clip(a, 0, 1))).astype(np.uint8))


def neighbors(a: np.ndarray, maximum: bool) -> np.ndarray:
    padded = np.pad(a, 1)
    windows = [padded[y : y + a.shape[0], x : x + a.shape[1]] for y in range(3) for x in range(3)]
    return np.maximum.reduce(windows) if maximum else np.minimum.reduce(windows)


def thin(mask: np.ndarray) -> np.ndarray:
    """Unpruned Zhang-Suen skeleton; protect it, never render it as a glyph."""
    a = np.pad(mask.astype(np.uint8), 1)
    while True:
        removed = 0
        for phase in [0, 1]:
            p = [
                a[:-2, 1:-1],
                a[:-2, 2:],
                a[1:-1, 2:],
                a[2:, 2:],
                a[2:, 1:-1],
                a[2:, :-2],
                a[1:-1, :-2],
                a[:-2, :-2],
            ]
            degree = sum(p)
            transitions = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            condition = (degree >= 2) & (degree <= 6) & (transitions == 1)
            if phase == 0:
                condition &= (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                condition &= (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            center = a[1:-1, 1:-1]
            condition &= center != 0
            removed += int(condition.sum())
            center[condition] = 0
        if not removed:
            return a[1:-1, 1:-1].astype(bool)


def components(mask: np.ndarray, diagonal: bool) -> int:
    rest = set(map(tuple, np.argwhere(mask)))
    number = 0
    offsets = [
        (a, b)
        for a in [-1, 0, 1]
        for b in [-1, 0, 1]
        if (a or b) and (diagonal or abs(a) + abs(b) == 1)
    ]
    while rest:
        queue = deque([rest.pop()])
        number += 1
        while queue:
            y, x = queue.popleft()
            for dy, dx in offsets:
                p = y + dy, x + dx
                if p in rest:
                    rest.remove(p)
                    queue.append(p)
    return number


def topology(mask: np.ndarray) -> tuple[int, int]:
    return components(mask, True), components(np.pad(~mask, 1, constant_values=True), False) - 1


def measure(image: Image.Image) -> dict:
    a = alpha(image)
    fg = a >= 0.25
    y, x = np.where(fg)
    padded = np.pad(a, 1)
    variation = float(np.abs(np.diff(padded, axis=0)).sum() + np.abs(np.diff(padded, axis=1)).sum())
    b = np.pad(fg.astype(float), 1)
    perimeter = float(np.abs(np.diff(b, axis=0)).sum() + np.abs(np.diff(b, axis=1)).sum())
    return {
        "mass": float(a.sum()),
        "soft_width": float(2 * a.sum() / max(variation, 1e-8)),
        "binary_width": float(2 * fg.sum() / max(perimeter, 1e-8)),
        "height": int(y.max() - y.min() + 1) if len(y) else 0,
        "area": int(fg.sum()),
        "density": float(a[fg].mean()) if fg.any() else 0.0,
        "core_q90": float(np.quantile(a[fg], 0.9)) if fg.any() else 0.0,
        "core_q50": float(np.median(a[fg])) if fg.any() else 0.0,
    }


def document_target(items) -> dict:
    refs = [measure(x["teacher_gray"]) for x in items if x.get("teacher_gray") is not None]
    refs = [x for x in refs if x["area"] >= CONFIG["minimum_area"]]
    if not refs:
        raise ValueError("Independent CPU references are required for a document target")
    return {
        "schema": SCHEMA,
        "reference_count": len(refs),
        **{
            k: float(np.median([r[k] for r in refs]))
            for k in ["soft_width", "height", "density", "core_q90"]
        },
        "definition": "median independent CPU reference; never average blank page pixels",
    }


def _guard(
    source: np.ndarray, candidate: np.ndarray, skeleton: np.ndarray, old_topology: dict
) -> list[str]:
    reasons = []
    if np.any(candidate[source == 0] != 0):
        reasons.append("new_background_ink")
    if np.any(candidate[skeleton] < source[skeleton] - 1e-6):
        reasons.append("weakened_centerline")
    ratio = float(candidate.sum() / max(float(source.sum()), 1e-8))
    if not CONFIG["mass_ratio_bounds"][0] <= ratio <= CONFIG["mass_ratio_bounds"][1]:
        reasons.append("mass_bound")
    for t in CONFIG["topology_thresholds"]:
        old, new = source >= t, candidate >= t
        if topology(new) != old_topology[t]:
            reasons.append(f"topology_{t}")
        if old.sum():
            if new.sum() / old.sum() > CONFIG["maximum_new_foreground_ratio"]:
                reasons.append(f"foreground_gain_{t}")
            if (
                np.count_nonzero(new & old) / old.sum()
                < CONFIG["minimum_retained_foreground_ratio"]
            ):
                reasons.append(f"foreground_loss_{t}")
    return reasons


class Model:
    def correct_regions(self, items: list[dict], mode: str = "uniform", target: dict | None = None):
        if mode not in {"uniform", "none", "density_rule"}:
            raise ValueError("Unknown optical ink mode")
        if not items:
            return [], []
        if target is None:
            target = document_target(items)
        if target["schema"] != SCHEMA:
            raise ValueError("Unknown document target")
        images, logs = [], []
        for item in items:
            before = measure(item["gray"])
            teacher = (
                measure(item["teacher_gray"]) if item.get("teacher_gray") is not None else target
            )
            scale = float(
                np.clip(
                    before["height"] / max(teacher["height"], 1),
                    *CONFIG["teacher_size_ratio_bounds"],
                )
            )
            desired = teacher["soft_width"] * scale ** CONFIG["teacher_size_exponent"]
            a = alpha(item["gray"])
            sk = thin(a >= 0.25) | thin(a >= 0.5)
            oldtop = {t: topology(a >= t) for t in CONFIG["topology_thresholds"]}
            result = item["gray"].convert("L").copy()
            attempts = []
            selected = 0.0
            eligible = (
                before["area"] >= CONFIG["minimum_area"]
                and before["height"] >= CONFIG["minimum_height"]
            )
            kind = item.get("kind", "text")
            if kind in {"punctuation", "protected"}:
                eligible = False
            diff = before["soft_width"] - desired
            wanted = abs(diff) > max(
                CONFIG["width_absolute_deadband"], CONFIG["width_ratio_deadband"] * desired
            )
            if mode != "none" and eligible:
                low, high = neighbors(a, False), neighbors(a, True)
                sign = 1.0 if diff > 0 else -1.0
                fractions = np.linspace(0, CONFIG["max_edge_fraction"], CONFIG["edge_steps"] + 1)[
                    1:
                ]
                candidates = []
                if mode == "uniform" and wanted:
                    for fraction in fractions:
                        value = a - fraction * (a - low) if sign > 0 else a + fraction * (high - a)
                        value = np.where(a >= CONFIG["minimum_mutable_support_alpha"], value, a)
                        value[sk] = np.maximum(value[sk], a[sk])
                        candidates.append((sign * float(fraction), value))
                elif mode == "density_rule":
                    ratio = float(
                        np.clip(target["density"] / max(before["density"], 1e-8), 0.8, 1.2)
                    )
                    value = np.clip(a * ratio, 0, 1)
                    value[sk] = np.maximum(value[sk], a[sk])
                    candidates.append((ratio - 1, value))
                # The rare genuinely pale core gets a bounded color correction after shape.
                if before["core_q90"] < CONFIG["pale_core_threshold"]:
                    gain = min(
                        CONFIG["maximum_pale_core_gain"],
                        target["core_q90"] / max(before["core_q90"], 0.1),
                    )
                    candidates = [(f, np.clip(v * gain, 0, 1)) for f, v in candidates] or [
                        (0.0, np.clip(a * gain, 0, 1))
                    ]
                best = abs(diff) if mode == "uniform" else float("inf")
                for fraction, value in candidates:
                    image = gray(value)
                    actual = alpha(image)
                    reasons = _guard(a, actual, sk, oldtop)
                    after = measure(image)
                    score = abs(after["soft_width"] - desired)
                    attempts.append(
                        {"fraction": fraction, "rejected": reasons, "width_error": score}
                    )
                    if not reasons and (score < best or mode == "density_rule"):
                        result, selected, best = image, fraction, score
            after = measure(result)
            output = alpha(result)
            assert not _guard(a, output, sk, oldtop)
            images.append(result)
            logs.append(
                {
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "kind": kind,
                    "mode": mode,
                    "before": before,
                    "after": after,
                    "teacher": teacher,
                    "target_width": desired,
                    "eligible": eligible,
                    "selected_fraction": selected,
                    "attempts": attempts,
                    "changed_pixels": int(
                        np.count_nonzero(
                            np.asarray(result) != np.asarray(item["gray"].convert("L"))
                        )
                    ),
                    "centerline_ink_exact_or_stronger": True,
                    "two_threshold_topology_exact": True,
                    "white_background_exact": True,
                    "learned": False,
                }
            )
        return images, logs


def load_model(path: Path | str | None = None) -> Model:
    if path is not None:
        bundle = json.loads(Path(path).read_text())
        if bundle["schema"] != SCHEMA or bundle["config"] != CONFIG:
            raise ValueError("Unknown frozen optical ink bundle")
        if bundle["source_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("Frozen optical ink source changed")
    return Model()
