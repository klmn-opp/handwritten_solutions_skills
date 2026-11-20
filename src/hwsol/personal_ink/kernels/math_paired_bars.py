"""CPU kernel extracted from research.math_paired_bars; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from hwsol.personal_ink.kernels.author_feedback_revision import identity
from hwsol.personal_ink.kernels.bindings import recipe_value
from hwsol.personal_ink.kernels.math_glyph_resolver import placed_outline
from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.personal_ink.kernels.math_session_polish import cap_stem
from hwsol.personal_ink.kernels.readability import placed_path
from hwsol.personal_ink.kernels.render import accepted_scans


def source_bar(directory):
    selected = [
        g
        for g in accepted_scans(directory)
        if g.source["id"] == recipe_value("single_bar_source_id")
    ]
    if len(selected) != 1 or selected[0].source["character"] != "|":
        raise ValueError("Expected the fixed reviewed Session N single-bar source")
    return selected[0]


class PersonalPairedBars:
    def __init__(self, glyph):
        if (
            glyph.source["id"] != recipe_value("single_bar_source_id")
            or glyph.source["character"] != "|"
        ):
            raise ValueError("Paired bar candidate requires the registered single-bar source")
        self.glyph = glyph
        self.decisions = []

    def apply(self, page, renderer, baseline):
        records = copy.deepcopy(baseline)
        before_items = list(page.items)
        self.decisions = []
        for row in records:
            if row["context"] != "mathematics" or row["semantic"] != "‖":
                continue
            i = row["item_index"]
            old = copy.deepcopy(row)
            decision = {
                "item_index": i,
                "applied": False,
                "source_id": recipe_value("single_bar_source_id"),
            }
            self.decisions.append(decision)
            if row["source_type"] not in {"external-font", "inherited-font"}:
                decision["reason"] = "existing non-font source retained"
                continue
            x0, y0, x1, y1 = row["bounds_px"]
            span = np.ptp(
                outline_bounds(self.glyph.outline.verts, self.glyph.outline.codes), axis=0
            )
            width = min((y1 - y0) * span[0] / span[1], (x1 - x0) * 0.32)
            if width <= 0 or (x1 - x0) / 2 - width < 0.05 * renderer.cfg.body_size:
                decision["reason"] = "parser slot cannot separate two source strokes"
                continue
            strokes, treatments = ([], [])
            for fraction in (0.25, 0.75):
                centre = x0 + (x1 - x0) * fraction
                slot = [centre - width / 2, y0, centre + width / 2, y1]
                outline, fit = placed_outline(self.glyph.outline, slot, stretch=True)
                limited = cap_stem(
                    MplPath(outline.verts, outline.codes), 0.055 * renderer.cfg.body_size
                )
                if limited:
                    outline, treatment = limited
                else:
                    treatment = {"method": "v26 stem limit not needed at this scale"}
                strokes.append(MplPath(outline.verts, outline.codes))
                treatments.append({"source_to_page_fit": fit, "stroke_width_policy": treatment})
            left, right = [outline_bounds(p.vertices, p.codes).ravel() for p in strokes]
            gap = float(right[0] - left[2])
            if gap < 0.05 * renderer.cfg.body_size or strokes[0].intersects_path(
                strokes[1], filled=True
            ):
                decision["reason"] = "fitted source strokes overlap or lack clear separation"
                continue
            path = MplPath.make_compound_path(*strokes)
            bounds = outline_bounds(path.vertices, path.codes).ravel().tolist()
            if min(bounds[0] - x0, bounds[1] - y0, x1 - bounds[2], y1 - bounds[3]) < -1e-06:
                decision["reason"] = "composed bars exceed the original math slot"
                continue
            neighbors = []
            for other in records:
                j = other["item_index"]
                if i == j or page.items[j][0] != "path":
                    continue
                if path.intersects_path(placed_path(page.items[j], renderer), filled=True):
                    neighbors.append(j)
            if neighbors:
                decision.update(
                    reason="composed source stroke touches another item", neighbors=neighbors
                )
                continue
            key = ("paired-personal-vertical-bars", i, recipe_value("single_bar_source_id"))
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), page.items[i][3], 0)
            row.pop("fallback_reason", None)
            row.update(
                source_type="personal-component-composition",
                personal_collected=True,
                external_fallback=False,
                procedural_geometry=True,
                font=None,
                font_sha256=None,
                variant=None,
                glyph_name=None,
                actual_glyph_id=f"two-instances:{recipe_value('single_bar_source_id')}",
                session_source=self.glyph.source,
                original_bounds_px=old["bounds_px"],
                allowed_bounds_px=old["bounds_px"],
                bounds_px=bounds,
                paired_bar_feedback={
                    "original": old,
                    "source_label": "|",
                    "source_parent_label": self.glyph.source["parent_latex"],
                    "semantic_mapping": "two vertical bars -> existing norm delimiter slot",
                    "source_instances": 2,
                    "new_collected_double_bar": False,
                    "raw_rgb_and_labels_unchanged": True,
                    "transforms": treatments,
                    "clear_gap_px": gap,
                    "separate_strokes": True,
                    "layout_and_semantic_identity_unchanged": True,
                },
            )
            decision.update(applied=True, clear_gap_px=gap, bounds=bounds, transforms=treatments)
        if identity(records) != identity(baseline):
            raise ValueError("Paired bars changed semantic identity/order")
        allowed = {d["item_index"] for d in self.decisions if d["applied"]}
        for i, (a, b, old, new) in enumerate(
            zip(before_items, page.items, baseline, records, strict=True)
        ):
            if i not in allowed and (a != b or old != new):
                raise ValueError("Paired bars changed non-target ink or provenance")
        return records
