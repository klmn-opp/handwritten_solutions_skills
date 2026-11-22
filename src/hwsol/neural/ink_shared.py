"""Shared physical pen coverage target, measured with the frozen M/L definition.

Deterministic optical correction, not neural training. Distinct characters share
one target per explicit size class. References set a document style, not labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import ink_model as base

SCHEMA = "shared-optical-pen/1"
CONFIG = {
    "relative_deadband": 0.06,
    "absolute_deadband": 0.08,
    "maximum_edge_fraction": 0.60,
    "edge_steps": 12,
    "minimum_area": 12,
    "minimum_height": 6,
    "excluded_roles": ["protected structure", "decimal", "superscript"],
    "excluded_kinds": ["punctuation", "protected", "structure"],
    "definition": "M/L at alpha.4; nonredundant 8-neighbor skeleton length plus component caps",
    "target": "median independent CPU M/L per explicit size_class; body shared by English and math",
    "default_size_class": "body; not inferred from individual character ink height",
}


def ink_measure(image: Image.Image) -> dict:
    a = 1 - np.asarray(image.convert("L"), dtype=np.float64) / 255
    mask = a >= 0.4
    s = base.thin(mask)
    horizontal = np.count_nonzero(s[:, :-1] & s[:, 1:])
    vertical = np.count_nonzero(s[:-1] & s[1:])
    diag1 = s[:-1, :-1] & s[1:, 1:] & ~s[:-1, 1:] & ~s[1:, :-1]
    diag2 = s[1:, :-1] & s[:-1, 1:] & ~s[:-1, :-1] & ~s[1:, 1:]
    count = base.components(mask, True)
    length = horizontal + vertical + math.sqrt(2) * (diag1.sum() + diag2.sum()) + count
    return {
        "mass": float(a.sum()),
        "length": float(length),
        "weight": float(a.sum() / max(1.0, length)),
        "components": count,
        "darkness": float(a[a >= 0.25].mean()) if np.any(a >= 0.25) else 0.0,
    }


def size_class(item: dict) -> str:
    if "size_class" in item:
        return str(item["size_class"])
    if "nominal_font_size" in item:
        return f"font-{float(item['nominal_font_size']):.2f}"
    return "body"


def document_target(items) -> dict:
    groups = defaultdict(list)
    references = []
    for item in items:
        if item.get("teacher_gray") is None or item.get("role") in CONFIG["excluded_roles"]:
            continue
        if item.get("kind") in CONFIG["excluded_kinds"]:
            continue
        m = base.measure(item["teacher_gray"])
        if m["area"] < CONFIG["minimum_area"] or m["height"] < CONFIG["minimum_height"]:
            continue
        measured = ink_measure(item["teacher_gray"])
        if not measured["length"]:
            continue
        groups[size_class(item)].append(measured["weight"])
        references.append(
            {"id": item.get("id"), "class": size_class(item), "weight": measured["weight"]}
        )
    if not groups:
        raise ValueError("Independent CPU reference group required")
    return {
        "schema": SCHEMA,
        "groups": {k: {"weight": float(np.median(v)), "n": len(v)} for k, v in groups.items()},
        "references": references,
        "config": CONFIG,
    }


class Model:
    def correct_regions(self, items: list[dict], mode: str = "shared", target: dict | None = None):
        if mode not in {"shared", "none"}:
            raise ValueError("Unknown shared-pen mode")
        if not items:
            return [], []
        if target is None:
            target = document_target(items)
        if target["schema"] != SCHEMA:
            raise ValueError("Unknown shared-pen target")
        outputs, logs = [], []
        for item in items:
            source = item["gray"].convert("L")
            a = base.alpha(source)
            before = ink_measure(source)
            geometric = base.measure(source)
            key = size_class(item)
            if key not in target["groups"]:
                raise ValueError(f"Missing independent target for explicit size class {key}")
            desired = target["groups"][key]["weight"]
            sk = base.thin(a >= 0.25) | base.thin(a >= 0.5)
            oldtop = {t: base.topology(a >= t) for t in [0.25, 0.5]}
            diff = before["weight"] - desired
            eligible = (
                geometric["area"] >= CONFIG["minimum_area"]
                and geometric["height"] >= CONFIG["minimum_height"]
            )
            eligible &= item.get("role") not in CONFIG["excluded_roles"]
            eligible &= item.get("kind") not in CONFIG["excluded_kinds"]
            result = source.copy()
            attempts = []
            selected = 0.0
            if (
                mode == "shared"
                and eligible
                and abs(diff)
                > max(CONFIG["absolute_deadband"], CONFIG["relative_deadband"] * desired)
            ):
                low, high = base.neighbors(a, False), base.neighbors(a, True)
                candidates = []
                for f in np.linspace(0, CONFIG["maximum_edge_fraction"], CONFIG["edge_steps"] + 1)[
                    1:
                ]:
                    value = a - f * (a - low) if diff > 0 else a + f * (high - a)
                    value = np.where(a >= base.CONFIG["minimum_mutable_support_alpha"], value, a)
                    value[sk] = np.maximum(value[sk], a[sk])
                    guess = abs(float(value.sum()) / max(before["length"], 1.0) - desired)
                    candidates.append((guess, float(f), value))
                # Solve a bounded correction objective; this is not generation/best-N selection.
                candidates.sort(key=lambda row: row[0])
                for _, f, value in candidates:
                    image = base.gray(value)
                    actual = base.alpha(image)
                    reasons = base._guard(a, actual, sk, oldtop)
                    after = ink_measure(image)
                    error = abs(after["weight"] - desired)
                    attempts.append(
                        {
                            "edge_fraction": f,
                            "rejected": reasons,
                            "remeasured_weight": after["weight"],
                            "error": error,
                        }
                    )
                    if not reasons and error < abs(diff):
                        result = image
                        selected = f if diff > 0 else -f
                        break
            after = ink_measure(result)
            assert not base._guard(a, base.alpha(result), sk, oldtop)
            outputs.append(result)
            logs.append(
                {
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "kind": item.get("kind"),
                    "role": item.get("role"),
                    "size_class": key,
                    "mode": mode,
                    "before": before,
                    "after": after,
                    "target_weight": desired,
                    "eligible": bool(eligible),
                    "selected_fraction": selected,
                    "attempts": attempts,
                    "changed_pixels": int(
                        np.count_nonzero(np.asarray(source) != np.asarray(result))
                    ),
                    "centerline_not_weakened": True,
                    "topology25_50_exact": True,
                    "white_background_exact": True,
                    "neural_updates": 0,
                }
            )
        return outputs, logs


def load_model(path: Path | str | None = None) -> Model:
    if path is not None:
        bundle = json.loads(Path(path).read_text())
        if bundle["schema"] != SCHEMA or bundle["config"] != CONFIG:
            raise ValueError("Unknown shared-pen protocol")
        for field, source in [
            ("source_sha256", Path(__file__)),
            ("base_source_sha256", Path(base.__file__)),
        ]:
            if hashlib.sha256(source.read_bytes()).hexdigest() != bundle[field]:
                raise ValueError("Frozen shared-pen source changed")
    return Model()
