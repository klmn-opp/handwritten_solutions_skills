"""CPU kernel extracted from research.structural_rule_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from hwsol.personal_ink.kernels.bindings import recipe_value
from hwsol.personal_ink.kernels.math_latin_feedback import filled_path
from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.personal_ink.kernels.math_session_polish import polygons, sections
from hwsol.personal_ink.kernels.render import accepted_scans

RULES = {"fraction-bar", "radical-overbar"}


def clean_path(path):
    rings = []
    for ring in polygons(path):
        area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        if area < 0:
            ring = ring[::-1]
        rings.append(MplPath(ring, closed=True))
    return MplPath.make_compound_path(*rings)


def vertical_sections(path, xs):
    return sections([p[:, ::-1] for p in polygons(path)], np.asarray(xs))


def measure(path, count=321):
    low, high = outline_bounds(path.vertices, path.codes)
    eps = min(0.001, (high[0] - low[0]) / 10000)
    xs = np.linspace(low[0] + eps, high[0] - eps, count)
    bands = vertical_sections(path, xs)
    if any(len(b) != 1 for b in bands):
        raise ValueError("Horizontal source contains gaps or multiple vertical ink intervals")
    values = np.asarray([b[0] for b in bands])
    centres = values.mean(axis=1)
    thickness = np.diff(values, axis=1)[:, 0]
    return {
        "x": xs,
        "centre": centres,
        "thickness": thickness,
        "bounds": np.r_[low, high],
        "width": high[0] - low[0],
        "thickness_p75": np.quantile(thickness, 0.75),
        "centre_range": np.ptp(centres),
        "endpoint_delta": centres[-1] - centres[0],
    }


@dataclass
class HorizontalSource:
    source: dict
    path: MplPath
    metric: dict
    clipped_roof: bool


def load_horizontal(directory):
    result = []
    for glyph in accepted_scans(directory):
        if glyph.source["id"] not in recipe_value("horizontal_source_ids"):
            continue
        if glyph.source["source_sha256"] != recipe_value("horizontal_source_sha256"):
            raise ValueError("Unexpected source outside approved Session N")
        path = clean_path(MplPath(glyph.outline.verts, glyph.outline.codes))
        roof = glyph.source["role"] == "radical"
        if roof:
            low, high = outline_bounds(path.vertices, path.codes)
            path = clean_path(
                path.clip_to_bbox(
                    Bbox.from_extents(glyph.source["seam_x"], low[1] - 1, high[0] + 1, high[1] + 1)
                )
            )
        result.append(HorizontalSource(glyph.source, path, measure(path), roof))
    if {x.source["id"] for x in result} != set(recipe_value("horizontal_source_ids")):
        raise ValueError("Missing approved horizontal sources")
    return result


def safe_window(index, record, before_paths, records, em, paired_sign):
    x0, y0, x1, y1 = record["bounds_px"]
    middle = (y0 + y1) / 2
    low, high = (middle - 0.18 * em, middle + 0.18 * em)
    constraints = []
    for j, path in before_paths.items():
        if j in (index, paired_sign):
            continue
        a, b, c, d = path.get_extents().extents
        if c <= x0 or a >= x1:
            continue
        if d <= middle:
            bound = min(y0, d + min(1.0, max(0, y0 - d) / 2))
            if bound > low:
                low = bound
                constraints.append({"item": j, "side": "above", "limit_y": float(bound)})
        elif b >= middle:
            bound = max(y1, b - min(1.0, max(0, b - y1) / 2))
            if bound < high:
                high = bound
                constraints.append({"item": j, "side": "below", "limit_y": float(bound)})
        else:
            low, high = (max(low, y0), min(high, y1))
            constraints.append({"item": j, "side": "straddles; original envelope only"})
    return ([x0, min(low, y0), x1, max(high, y1)], constraints)


def fit_source(sample, old_path, allowed, *, roof=False):
    old = measure(old_path)
    source = sample.metric
    x0, _, x1, _ = old["bounds"]
    median = old["thickness_p75"]
    if x1 - x0 < 2.5 * median:
        return (
            None,
            {"status": "held-short-range", "reason": "too little width for a safe terminal"},
        )
    thickness_scale = median / source["thickness_p75"]
    scale_x = (x1 - x0) / source["width"]
    source_anchor = source["centre"][0] if roof else float(np.median(source["centre"]))
    anchor = old["centre"][0] if roof else (old["bounds"][1] + old["bounds"][3]) / 2
    dense_rings = []
    for ring in polygons(sample.path):
        points = []
        for a, b in zip(ring[:-1], ring[1:], strict=True):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.35)))
            points.append(a + np.arange(n)[:, None] / n * (b - a))
        dense_rings.append(np.vstack([*points, ring[-1:]]))
    vertices = np.vstack(dense_rings)
    centre = np.interp(vertices[:, 0], source["x"], source["centre"])
    residual = vertices[:, 1] - centre
    offset = centre - source_anchor
    for scale in (thickness_scale, thickness_scale * 0.9, thickness_scale * 0.8):
        low_limit, high_limit = (0.0, scale_x)
        zero = anchor + residual * scale
        for z, slope in zip(zero, offset, strict=True):
            if abs(slope) < 1e-08:
                if not allowed[1] - 1e-06 <= z <= allowed[3] + 1e-06:
                    high_limit = -1
                    break
                continue
            bounds = sorted(((allowed[1] - z) / slope, (allowed[3] - z) / slope))
            low_limit, high_limit = (max(low_limit, bounds[0]), min(high_limit, bounds[1]))
        if high_limit >= low_limit:
            thickness_scale = scale
            centre_scale = max(0.0, high_limit)
            break
    else:
        return (
            None,
            {"status": "held-clearance", "reason": "observed nib cannot fit guarded space"},
        )
    mapped, cursor = ([], 0)
    for ring in dense_rings:
        count = len(ring)
        x = x0 + (ring[:, 0] - source["bounds"][0]) * scale_x
        y = anchor + offset[cursor : cursor + count] * centre_scale
        y += residual[cursor : cursor + count] * thickness_scale
        mapped.append(MplPath(np.column_stack((x, y)), closed=True))
        cursor += count
    candidate = clean_path(MplPath.make_compound_path(*mapped))
    seam = None
    if roof:
        seam_width = min(1.0, (x1 - x0) * 0.08)
        original_join = clean_path(
            old_path.clip_to_bbox(
                Bbox.from_extents(
                    x0 - 0.01, old["bounds"][1] - 0.1, x0 + seam_width, old["bounds"][3] + 0.1
                )
            )
        )
        candidate = clean_path(MplPath.make_compound_path(original_join, candidate))
        seam = {
            "method": "additive original left join slice; same winding",
            "preserved_original_x_range": [float(x0), float(x0 + seam_width)],
        }
    info = {
        "status": "candidate",
        "source_id": sample.source["id"],
        "source_character": sample.source["character"],
        "source_role": sample.source["role"],
        "source_sha256": sample.source["source_sha256"],
        "crop_sha256": sample.source["crop_sha256"],
        "source_metadata": sample.source,
        "selected_original_roof_only": sample.clipped_roof,
        "actual_source_label_changed": False,
        "raw_source_pixels_edited": False,
        "role_transfer": "observed horizontal contour to parser-owned structural rule",
        "measured_source": True,
        "procedural_geometry": True,
        "geometry_disclosure": (
            "source centre/nib decomposition, separate coordinate sc"
            "aling, clearance fit, and optional original roof-seam c"
            "omposition"
        ),
        "target_x_span": [float(x0), float(x1)],
        "x_scale": float(scale_x),
        "centre_scale_requested": float(scale_x),
        "centre_scale_applied": float(centre_scale),
        "thickness_scale": float(thickness_scale),
        "target_y_anchor": float(anchor),
        "source_y_anchor": float(source_anchor),
        "source_measurement_count": len(source["x"]),
        "old_thickness_p75": float(median),
        "source_thickness_p75": float(source["thickness_p75"]),
        "target_nominal_thickness_p75": float(source["thickness_p75"] * thickness_scale),
        "source_centre_range": float(source["centre_range"]),
        "target_measured_centre_range": float(source["centre_range"] * centre_scale),
        "source_endpoint_delta": float(source["endpoint_delta"]),
        "target_measured_endpoint_delta": float(source["endpoint_delta"] * centre_scale),
        "random_variation": False,
        "join_preservation": seam,
    }
    return (candidate, info)


class StructuralRuleFeedback:
    def __init__(self, directory):
        self.sources = load_horizontal(Path(directory))
        self.decisions = []

    def apply(self, page, renderer, provenance):
        records = copy.deepcopy(provenance)
        original_items = list(page.items)
        before_paths = {i: filled_path(item, renderer) for i, item in enumerate(page.items)}
        self.decisions = []
        for row in records:
            if row["context"] != "mathematics" or row.get("structure") not in RULES:
                continue
            i = row["item_index"]
            old = clean_path(before_paths[i])
            roof = row["structure"] == "radical-overbar"
            metric = measure(old)
            eligible = [s for s in self.sources if s.clipped_roof == roof]
            ranked = sorted(
                eligible,
                key=lambda s: abs(
                    math.log(
                        metric["width"]
                        / (s.metric["width"] * metric["thickness_p75"] / s.metric["thickness_p75"])
                    )
                ),
            )
            selected = ranked[0]
            paired_signs = [
                j
                for j in row.get("paired_items", [])
                if records[j].get("structure") == "radical-sign"
            ]
            paired_sign = paired_signs[0] if len(paired_signs) == 1 else None
            decision = {
                "item_index": i,
                "structure": row["structure"],
                "latex": row["latex"],
                "source_selection": "closest native width after matching measured nib p75",
                "ranked_source_ids": [s.source["id"] for s in ranked],
                "original_bounds": row["bounds_px"],
                "paired_sign": paired_sign,
            }
            allowed, constraints = safe_window(
                i, row, before_paths, records, renderer.cfg.body_size, paired_sign
            )
            decision.update(allowed_bounds_px=allowed, clearance_constraints=constraints)
            if roof and paired_sign is None:
                candidate, treatment = (None, {"status": "held-unpaired-roof"})
            else:
                candidate, treatment = fit_source(selected, old, allowed, roof=roof)
            decision.update(treatment)
            if candidate is not None:
                bounds = outline_bounds(candidate.vertices, candidate.codes).ravel()
                if np.any(bounds[:2] < np.asarray(allowed[:2]) - 0.002) or np.any(
                    bounds[2:] > np.asarray(allowed[2:]) + 0.002
                ):
                    raise ValueError("Measured candidate left its precomputed clearance envelope")
                contacts = []
                for j, prior_other in before_paths.items():
                    if j == i:
                        continue
                    other = filled_path(page.items[j], renderer)
                    if not candidate.get_extents().overlaps(other.get_extents()):
                        continue
                    before_contact = old.intersects_path(prior_other, filled=True)
                    after_contact = candidate.intersects_path(other, filled=True)
                    if after_contact or before_contact:
                        contacts.append(
                            {
                                "item": j,
                                "before": bool(before_contact),
                                "after": bool(after_contact),
                            }
                        )
                unexpected = [
                    c
                    for c in contacts
                    if c["after"] and (not c["before"]) and (c["item"] != paired_sign)
                ]
                seam_ok = (
                    not roof
                    or not old.intersects_path(before_paths[paired_sign], filled=True)
                    or candidate.intersects_path(before_paths[paired_sign], filled=True)
                )
                decision.update(contacts=contacts, seam_contact_preserved=bool(seam_ok))
                if unexpected or not seam_ok:
                    decision.update(
                        status="held-contact",
                        reason="new neighbour ink contact or lost original join",
                    )
                    candidate = None
            if candidate is not None:
                key = ("structural-rule-feedback", i)
                renderer._outlines[key] = Outline(
                    candidate.vertices, candidate.codes, stroke=False, digits=4
                )
                page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), original_items[i][3], 0)
                row.update(
                    source_type="personal-scan-structural-rule",
                    personal_collected=True,
                    external_fallback=False,
                    procedural_geometry=True,
                    measured_source=True,
                    allowed_bounds_px=allowed,
                    bounds_px=bounds.tolist(),
                    actual_glyph_id=f"structural-rule:{selected.source['id']}",
                    source_sha256=selected.source["source_sha256"],
                    transformation=(
                        "measured source centre/nib decomposition with clearance-limited scale"
                    ),
                    original_source_type=provenance[i]["source_type"],
                    original_glyph_id=provenance[i]["actual_glyph_id"],
                )
                row.pop("fallback_reason", None)
                decision["status"] = "applied"
                decision["candidate_bounds"] = bounds.tolist()
            row["structural_rule_feedback"] = copy.deepcopy(decision)
            self.decisions.append(decision)
        return records
