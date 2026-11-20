"""CPU kernel extracted from research.f_size_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
import hashlib
import re

import numpy as np
from handnote.markdown import parse
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from hwsol.jobs.layout import plain_emphasis
from hwsol.personal_ink.kernels.author_feedback_revision import identity
from hwsol.personal_ink.kernels.bindings import recipe_value
from hwsol.personal_ink.kernels.math_latin_feedback import filled_path
from hwsol.personal_ink.kernels.math_session_polish import polygons, sections

COMMON_SOURCE = {
    "reference_characters": "hldaceo",
    "method": "least-squares common y affine over reference top and bottom coordinates",
    "source_to_reference_scale": 1.3971949065933285,
    "source_to_reference_y_translation": -43.40055063616691,
    "reference_fit_rmse_font_units": 19.80359711525346,
    "target_f_top_font_units": 719.4678683637904,
    "target_f_bottom_font_units": -314.45636251527264,
}
PRESETS = {"B-intermediate": 1.28, "C-source-ratio": 1.42, "D-common-source": 1.444028255417686}
CROSSBAR_FONT_Y = 436.180928112813
SAFETY_BUDGET = {
    "maximum_uniform_scale": 1.45,
    "maximum_left_or_right_extension_em": 0.075,
    "maximum_top_extension_em": 0.15,
    "maximum_bottom_extension_em": 0.33,
    "minimum_new_neighbor_clearance_px": 0.6,
    "minimum_page_clearance_px": 3.0,
    "audit_neighbor_radius_em": 2.0,
}


def stem_measure(path, *, f=True):
    """Median horizontal section through one narrow stem, excluding f crossbar/hook.

    For f use the lower stem (10..40% up from its bottom). Other tall letters use
    70..90% up, above their bowl. This is outline geometry, not estimated pen pressure.
    """
    box = path.get_extents()
    if f:
        lo, hi = (box.ymin + 0.6 * box.height, box.ymin + 0.9 * box.height)
    else:
        lo, hi = (box.ymin + 0.1 * box.height, box.ymin + 0.3 * box.height)
    values = [
        float(row[0, 1] - row[0, 0])
        for row in sections(polygons(path), np.linspace(lo, hi, 40))
        if len(row) == 1
    ]
    return {
        "method": "single lower-stem sections; crossbar and hook excluded"
        if f
        else "single upper-ascender sections above bowl",
        "sample_count": len(values),
        "median_px": float(np.median(values)) if values else None,
        "p10_px": float(np.quantile(values, 0.1)) if values else None,
        "p90_px": float(np.quantile(values, 0.9)) if values else None,
        "measurement_is_pen_pressure": False,
    }


def contour_distance(a, b):
    """Minimum flattened-contour point/segment distance for near-neighbor gating."""
    if a.intersects_path(b, filled=True):
        return 0.0
    aa, bb = (polygons(a), polygons(b))
    if not aa or not bb:
        return float("inf")

    def directed(points, rings):
        starts = np.concatenate([r[:-1] for r in rings])
        ends = np.concatenate([r[1:] for r in rings])
        delta = ends - starts
        denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
        diff = points[:, None, :] - starts[None, :, :]
        t = np.clip(np.sum(diff * delta, axis=2) / denom, 0, 1)
        return float(np.sqrt(np.min(np.sum((diff - t[:, :, None] * delta) ** 2, axis=2))))

    return min(directed(np.concatenate(aa), bb), directed(np.concatenate(bb), aa))


def consecutive_f_groups(source, title, records):
    """Map real same-word ff runs to recorder items after exact text-order validation."""
    spans = [title]
    for block in parse(plain_emphasis(source)):
        spans.extend((text for kind, text in block.segments if kind == "t"))
    visible = [r for r in records if r["context"] == "text"]
    expected = [ch for span in spans for ch in span if not ch.isspace()]
    if expected != [r["semantic"] for r in visible]:
        raise ValueError("Cannot bind ff groups: source text and visible order differ")
    groups, offset = ([], 0)
    for span_index, span in enumerate(spans):
        mapping = {}
        for pos, ch in enumerate(span):
            if not ch.isspace():
                mapping[pos] = visible[offset]["item_index"]
                offset += 1
        for match in re.finditer("[A-Za-z]+", span):
            for repeat in re.finditer("f{2,}", match.group()):
                start = match.start() + repeat.start()
                end = match.start() + repeat.end()
                groups.append(
                    {
                        "group_id": f"span-{span_index}-word-{match.start()}-ff-{start}",
                        "word": match.group(),
                        "span_index": span_index,
                        "word_span": [match.start(), match.end()],
                        "f_span": [start, end],
                        "item_indices": [mapping[i] for i in range(start, end)],
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "source_word_and_visible_text_order_verified": True,
                        "cross_word_or_math_segment_grouping": False,
                    }
                )
    return groups


class FSizeFeedback:
    """Uniform output geometry only, with explicit per-instance rejection."""

    def __init__(self, preset="D-common-source", *, source=None, title=None):
        if preset not in PRESETS:
            raise ValueError("Unknown preregistered f size preset")
        self.preset, self.scale = (preset, PRESETS[preset])
        self.decisions = []
        self.source, self.title = (source, title)
        self.groups = []

    def apply(self, page, renderer, provenance):
        if self.source is None or self.title is None:
            raise ValueError("Actual source and title required for atomic same-word ff admission")
        self.groups = consecutive_f_groups(self.source, self.title, provenance)
        records = copy.deepcopy(provenance)
        original_items = list(page.items)
        originals = [filled_path(item, renderer) for item in page.items]
        self.decisions = []
        for row in records:
            if row["semantic"] != "f":
                continue
            i = row["item_index"]
            decision = {
                "item_index": i,
                "context": row["context"],
                "preset": self.preset,
                "requested_scale": self.scale,
                "applied": False,
                "rejections": [],
            }
            self.decisions.append(decision)
            if row.get("font_sha256") != recipe_value("font_sha256") or row.get("variant") != 0:
                decision["rejections"].append("Expected approved real f0 source")
                continue
            old = originals[i]
            box = old.get_extents()
            actual_em = box.height * 1000 / 716
            decision["actual_em_px"] = actual_em
            if actual_em < 0.8 * renderer.cfg.body_size:
                decision["rejections"].append("Small script f held for separate scale protocol")
                row["f_size_feedback"] = {"status": "original-retained", **decision}
                continue
            anchor = np.array(
                [(box.xmin + box.xmax) / 2, box.ymin + (715 - CROSSBAR_FONT_Y) / 716 * box.height]
            )
            if self.preset == "D-common-source":
                top_delta = (715 - COMMON_SOURCE["target_f_top_font_units"]) / 1000 * actual_em
                anchor[1] = box.ymin - top_delta / (self.scale - 1)
            path = MplPath(anchor + (old.vertices - anchor) * self.scale, old.codes)
            final_box = path.get_extents().extents.tolist()
            expansion = np.abs(np.asarray(final_box) - box.extents) / actual_em
            decision.update(
                original_bounds=box.extents.tolist(),
                requested_bounds=final_box,
                anchor_px=anchor.tolist(),
                original_stem=stem_measure(old),
                candidate_stem=stem_measure(path),
                preserved_crossbar_height=self.preset != "D-common-source",
                common_source_coordinates=COMMON_SOURCE
                if self.preset == "D-common-source"
                else None,
                uniform_geometry=True,
                added_strokes=False,
                pen_pressure_modified=False,
                source_outline_codes_exact=True,
                safety_budget=SAFETY_BUDGET,
                absolute_bound_displacement_em=expansion.tolist(),
            )
            if (
                self.scale > SAFETY_BUDGET["maximum_uniform_scale"]
                or max(expansion[0], expansion[2])
                > SAFETY_BUDGET["maximum_left_or_right_extension_em"]
                or expansion[1] > SAFETY_BUDGET["maximum_top_extension_em"]
                or (expansion[3] > SAFETY_BUDGET["maximum_bottom_extension_em"])
            ):
                decision["rejections"].append("frozen scale or displacement budget exceeded")
            if (
                min(final_box[:2]) < 3
                or final_box[2] > page.width - 3
                or final_box[3] > page.height - 3
            ):
                decision["rejections"].append("page boundary")
            nearby = []
            for j, other in enumerate(originals):
                ob = other.get_extents()
                dx = max(0, ob.xmin - final_box[2], final_box[0] - ob.xmax)
                dy = max(0, ob.ymin - final_box[3], final_box[1] - ob.ymax)
                if j == i or np.hypot(dx, dy) > actual_em * 2:
                    continue
                prior = contour_distance(old, other)
                after = contour_distance(path, other)
                nearby.append(
                    {
                        "other_item": j,
                        "semantic": records[j]["semantic"],
                        "context": records[j]["context"],
                        "before_clearance_px": prior,
                        "after_clearance_px": after,
                        "relative_line": "above"
                        if ob.ymax < final_box[1]
                        else "below"
                        if ob.ymin > final_box[3]
                        else "overlapping-y-band",
                        "horizontal_box_gap_px": dx,
                    }
                )
                if after < 0.6 and after < prior - 0.05:
                    decision["rejections"].append(f"neighbor {j} clearance {after:.3f}px")
            decision["neighbors"] = nearby
            decision["nearest_neighbor_px"] = min(
                (n["after_clearance_px"] for n in nearby), default=None
            )
            for direction in ("above", "below"):
                decision[f"nearest_{direction}_aligned_ink_px"] = min(
                    (
                        n["after_clearance_px"]
                        for n in nearby
                        if n["relative_line"] == direction and n["horizontal_box_gap_px"] == 0
                    ),
                    default=None,
                )
            if decision["rejections"]:
                row["f_size_feedback"] = {"status": "original-retained", **decision}
                continue
            key = ("f-size-feedback", self.preset, i)
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), original_items[i][3], 0)
            row.update(
                bounds_px=final_box,
                allowed_bounds_px=final_box,
                source_type="personal-outline-geometric-calibration",
                f_size_feedback={
                    "status": "research-candidate",
                    "preset": self.preset,
                    "scale_x": self.scale,
                    "scale_y": self.scale,
                    "anchor_px": anchor.tolist(),
                    "method": (
                        "uniform size and y-position from common h/l/d/a/c/e/o source coordinates"
                    )
                    if self.preset == "D-common-source"
                    else "uniform output enlargement around compiled crossbar height",
                    "common_source_coordinates": COMMON_SOURCE
                    if self.preset == "D-common-source"
                    else None,
                    "original": copy.deepcopy(provenance[i]),
                    "original_contour_codes_and_aspect_ratio_preserved": True,
                    "raw_rgb_labels_font_unchanged": True,
                    "advance_and_other_items_unchanged": True,
                    "output_envelope_scope": (
                        "allowed_bounds_px is the target output envelope, not a "
                        "reserved MathText slot or independent safety proof; fro"
                        "zen displacement budgets, measured neighbor distances a"
                        "nd final non-target checks apply"
                    ),
                    "safety_budget": SAFETY_BUDGET,
                    "source_font_sha256": recipe_value("font_sha256"),
                    "author_acceptance": False,
                },
            )
            decision["applied"] = True
        for decision in self.decisions:
            if not decision["applied"]:
                continue
            i = decision["item_index"]
            path = filled_path(page.items[i], renderer)
            for other_decision in self.decisions:
                j = other_decision["item_index"]
                if i >= j or not other_decision["applied"]:
                    continue
                other = filled_path(page.items[j], renderer)
                if not path.get_extents().expanded(1.1, 1.1).overlaps(other.get_extents()):
                    continue
                before = contour_distance(originals[i], originals[j])
                after = contour_distance(path, other)
                if after < 0.6 and after < before - 0.05:
                    for d in (decision, other_decision):
                        k = d["item_index"]
                        page.items[k] = original_items[k]
                        records[k] = copy.deepcopy(provenance[k])
                        d["applied"] = False
                        d["rejections"].append("simultaneous ff clearance")
                        records[k]["f_size_feedback"] = {"status": "original-retained", **d}
        by_index = {d["item_index"]: d for d in self.decisions}
        for group in self.groups:
            failures = [i for i in group["item_indices"] if not by_index[i]["applied"]]
            group["original_failure_items"] = failures
            group["atomic_action"] = "retain-whole-group" if failures else "accept-whole-group"
            if not failures:
                continue
            causes = [{"item_index": i, "rejections": by_index[i]["rejections"]} for i in failures]
            for i in group["item_indices"]:
                decision = by_index[i]
                independently_accepted = decision["applied"]
                decision.update(
                    applied=False,
                    group_rejection={
                        **group,
                        "causes": causes,
                        "independently_accepted_before_group_policy": independently_accepted,
                    },
                )
                if independently_accepted:
                    decision["rejections"].append("atomic same-word ff group retained")
                page.items[i] = original_items[i]
                records[i] = copy.deepcopy(provenance[i])
                records[i]["f_size_feedback"] = {"status": "original-retained", **decision}
        for decision in self.decisions:
            if not decision["applied"]:
                continue
            i = decision["item_index"]
            current = filled_path(page.items[i], renderer)
            neighbors = []
            for other in self.decisions:
                j = other["item_index"]
                if i == j:
                    continue
                target = filled_path(page.items[j], renderer)
                if not current.get_extents().expanded(1.5, 1.2).overlaps(target.get_extents()):
                    continue
                before = contour_distance(originals[i], originals[j])
                after = contour_distance(current, target)
                neighbors.append({"item_index": j, "before_px": before, "final_px": after})
                if after < 0.6 and after < before - 0.05:
                    raise ValueError("Final f-f clearance failed after group policy")
            decision["final_f_neighbor_distances"] = neighbors
        if identity(records) != identity(provenance):
            raise ValueError("f treatment changed content identity/order")
        for i, (a, b) in enumerate(zip(original_items, page.items, strict=True)):
            if provenance[i]["semantic"] != "f" and (a != b or records[i] != provenance[i]):
                raise ValueError("Non-f item changed")
        return records
