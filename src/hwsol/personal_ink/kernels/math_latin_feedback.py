"""CPU kernel extracted from research.math_latin_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from hwsol.personal_ink.kernels.author_feedback_revision import identity
from hwsol.personal_ink.kernels.bindings import recipe_value
from hwsol.personal_ink.kernels.cpu_glyph_feedback import CpuGlyphFeedback
from hwsol.personal_ink.kernels.readability import placed_path
from hwsol.profiles.io import sha256

TARGETS = frozenset("dfkr")


def filled_path(item, renderer):
    """Filled outline or conservative round-cap envelope of a structure stroke."""
    if item[0] == "path":
        return placed_path(item, renderer)
    if item[0] != "stroke":
        raise ValueError("Unknown neighbor item kind; replacement cannot be audited")
    a, b, c, d, e, f = item[2]
    points = np.asarray(item[1]) @ np.array([[a, b], [c, d]]) + (e, f)
    radius = max(float(item[4]) / 2, 0.25)
    pieces = [
        MplPath.unit_circle().transformed(Affine2D().scale(radius).translate(*point))
        for point in points
    ]
    for left, right in zip(points[:-1], points[1:], strict=True):
        delta = right - left
        length = np.linalg.norm(delta)
        if not length:
            continue
        normal = np.array([-delta[1], delta[0]]) * radius / length
        pieces.append(
            MplPath(
                np.array(
                    [left + normal, right + normal, right - normal, left - normal, left + normal]
                ),
                closed=True,
            )
        )
    return MplPath.make_compound_path(*pieces)


def contacts(item_index, before_items, page, renderer, records):
    old = filled_path(before_items[item_index], renderer)
    new = filled_path(page.items[item_index], renderer)
    extent = new.get_extents()
    comparisons = []
    for row in records:
        j = row["item_index"]
        if j == item_index:
            continue
        other = filled_path(page.items[j], renderer)
        if not extent.expanded(1.02, 1.02).overlaps(other.get_extents()):
            continue
        old_other = filled_path(before_items[j], renderer)
        prior = bool(old.intersects_path(old_other, filled=True))
        after = bool(new.intersects_path(other, filled=True))
        comparisons.append(
            {
                "other_item": j,
                "semantic": row["semantic"],
                "context": row["context"],
                "before_contact": prior,
                "after_contact": after,
                "new_contact": after and (not prior),
            }
        )
    return comparisons


class MathLatinFeedback:
    """Post-layout d/f/k/r only, rejecting unsupported/contacting replacements."""

    def __init__(self, font):
        if sha256(font) != recipe_value("font_sha256"):
            raise ValueError("Math Latin feedback requires the pinned personal v1 font")
        self.feedback = CpuGlyphFeedback(font)
        self.decisions = []

    def apply(self, page, renderer, provenance):
        before_items = list(page.items)
        before = copy.deepcopy(provenance)
        records = self.feedback.apply(page, renderer, provenance, contexts={"mathematics"})
        self.decisions = []
        for old, row in zip(before, records, strict=True):
            if row["context"] != "mathematics" or row["semantic"] not in TARGETS:
                continue
            i = row["item_index"]
            decision = {
                "item_index": i,
                "semantic": old["semantic"],
                "latex": old["latex"],
                "original_variant": old.get("variant"),
                "original_bounds": old["bounds_px"],
                "candidate_bounds": row["bounds_px"],
                "source_font_sha256": recipe_value("font_sha256"),
                "applied": page.items[i] != before_items[i],
                "status": "unchanged",
                "math_layout_positions_changed": False,
                "original_item_matrix": list(before_items[i][2]),
                "candidate_item_matrix": list(page.items[i][2]),
            }
            if decision["applied"]:
                box = row["bounds_px"]
                errors = []
                if min(box[0], box[1], page.width - box[2], page.height - box[3]) < 3:
                    errors.append("page boundary clearance below 3 px")
                old_box = old["bounds_px"]
                if (
                    row["semantic"] in {"d", "k"}
                    and min(
                        box[0] - old_box[0],
                        box[1] - old_box[1],
                        old_box[2] - box[2],
                        old_box[3] - box[3],
                    )
                    < -0.02
                ):
                    errors.append("d/k exceeds original envelope")
                if row["semantic"] == "k" and np.max(np.abs(np.array(box) - old_box)) > 0.02:
                    errors.append("k no longer preserves the existing fitted envelope")
                nearby = contacts(i, before_items, page, renderer, records)
                if any(r["new_contact"] for r in nearby):
                    errors.append("new neighbor ink contact")
                decision.update(neighbors=nearby, rejection_reasons=errors)
                if errors:
                    page.items[i] = before_items[i]
                    records[i] = copy.deepcopy(old)
                    records[i]["math_latin_feedback"] = {
                        "status": "explicit-original-glyph-fallback",
                        "reason": errors,
                        "attempted_variant": row.get("variant"),
                        "attempted_bounds": box,
                        "original_source_retained": True,
                    }
                    decision.update(applied=False, status="rejected-original-retained")
                else:
                    row["math_latin_feedback"] = {
                        "status": "research-candidate",
                        "source": row["cpu_feedback"],
                        "recorded_variant": row["variant"],
                        "author_acceptance": False,
                        "no_neural_training": True,
                        "neighbor_contact_checked": True,
                        "native_outline_matrix_preserved": row["semantic"] in {"f", "k", "r"},
                    }
                    if row["semantic"] in {"f", "r"}:
                        row["allowed_bounds_px"] = [
                            min(box[0], old_box[0]),
                            min(box[1], old_box[1]),
                            max(box[2], old_box[2]),
                            max(box[3], old_box[3]),
                        ]
                        row["math_latin_feedback"]["primary_slot_bounds_px"] = box
                        row["math_latin_feedback"]["allowed_bounds_reason"] = (
                            "original box union native primary MathText outline at unchanged matrix"
                        )
                    decision.update(
                        status="accepted-for-visual-review",
                        variant=row["variant"],
                        treatment=row["cpu_feedback"],
                    )
            else:
                decision["reason"] = row.get("cpu_feedback_rejection", "already selected d1/f0/r0")
            self.decisions.append(decision)
        if identity(before) != identity(records):
            raise ValueError("Mathematical identity/order changed")
        for old, new, a, b in zip(before_items, page.items, before, records, strict=True):
            if a["context"] != "mathematics" or a["semantic"] not in TARGETS:
                if old != new or a != b:
                    raise ValueError("Unauthorized non-target content or source change")
        for decision in self.decisions:
            if decision["applied"] or decision["semantic"] == "r":
                decision["final_neighbors"] = contacts(
                    decision["item_index"], before_items, page, renderer, records
                )
                if any(r["new_contact"] for r in decision["final_neighbors"]):
                    raise ValueError("Accepted math glyph has a new final contact")
        return records
