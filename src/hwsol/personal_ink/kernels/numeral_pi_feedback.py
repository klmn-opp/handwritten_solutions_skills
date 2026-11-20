"""CPU kernel extracted from research.numeral_pi_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from handnote.opentype import read_alternates
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from hwsol.personal_ink.kernels.author_feedback_revision import identity
from hwsol.personal_ink.kernels.bindings import recipe_value
from hwsol.personal_ink.kernels.math_latin_feedback import contacts
from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.profiles.io import sha256


class NumeralPiFeedback:
    """Select real lighter 2v0 for heavy 2v2/3; explicitly retain unsupported pi."""

    def __init__(self, font):
        self.font = Path(font)
        if sha256(self.font) != recipe_value("font_sha256"):
            raise ValueError("Requires pinned and traced personal v1 font")
        with TTFont(self.font) as t:
            name = t.getBestCmap()[50]
            self.name, self.gid = (name, t.getGlyphID(name))
            self.names = [name, *read_alternates(t).get(name, [])]
        self.decisions = []

    def apply(self, page, renderer, records):
        before_items, before = (list(page.items), copy.deepcopy(records))
        result = copy.deepcopy(records)
        self.decisions = []
        for old, row in zip(before, result, strict=True):
            if old.get("semantic") not in {"2", "π"}:
                continue
            i = old["item_index"]
            d = {
                "item_index": i,
                "semantic": old["semantic"],
                "context": old["context"],
                "original_variant": old.get("variant"),
                "applied": False,
            }
            if old["semantic"] == "π":
                d.update(
                    status="explicit-no-reliable-lighter-source",
                    reason=(
                        "Only accepted pi U03C0-p55; its real cap/legs are heavy"
                        ". Existing fit has no confirmed scaling defect. No sour"
                        "ce ink thinned or redrawn."
                    ),
                    source_id=old.get("scan_source", {}).get("id"),
                    existing_transform=old.get("transform_scale_translate"),
                )
            elif old.get("variant") in {0, 1}:
                d.update(
                    status="unchanged-real-lighter-variant", reason="Retain existing accepted v0/v1"
                )
            elif old.get("variant") not in {2, 3} or old.get("font_sha256") != recipe_value(
                "font_sha256"
            ):
                d.update(
                    status="explicit-original-source-retained",
                    reason="Unsupported variant or font identity",
                )
            else:
                item = before_items[i]
                if item[0] != "path":
                    raise ValueError("Two requires the recorded outline path")
                v, c = handwritten_outline(self.font, 50, 0, 1000)
                transformed = Affine2D.from_values(*item[2]).transform(v)
                lo, hi = outline_bounds(transformed, c)
                low, high = np.array(old["bounds_px"]).reshape(2, 2)
                scale = float(np.min((high - low) / (hi - lo)))
                shift = (high + low) / 2 - (hi + lo) * scale / 2
                fitted = transformed * scale + shift
                outline = Outline(fitted, c, stroke=False, digits=4)
                key = ("numeral-pi-feedback", "2", i)
                renderer._outlines[key] = outline
                page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), item[3], 0)
                bounds = outline_bounds(fitted, c).ravel().tolist()
                neighbor = contacts(i, before_items, page, renderer, result)
                errors = []
                if any(x["new_contact"] for x in neighbor):
                    errors.append("new neighboring ink contact")
                if (
                    min(
                        bounds[0] - low[0],
                        bounds[1] - low[1],
                        high[0] - bounds[2],
                        high[1] - bounds[3],
                    )
                    < -0.02
                ):
                    errors.append("outside original ink envelope")
                if min(bounds[0], bounds[1], page.width - bounds[2], page.height - bounds[3]) < 3:
                    errors.append("page boundary clearance below 3 px")
                source_contours = int(np.count_nonzero(c == MplPath.MOVETO))
                if scale <= 0 or source_contours != 1 or (not np.isfinite(fitted).all()):
                    errors.append("unsupported source topology or affine transform")
                d.update(
                    candidate_bounds=bounds,
                    original_bounds=old["bounds_px"],
                    neighbors=neighbor,
                    rejection_reasons=errors,
                    source_contours=source_contours,
                    source_topology_preserved=True,
                )
                if errors:
                    page.items[i] = item
                    d.update(status="rejected-original-retained")
                else:
                    treatment = {
                        "method": (
                            "select authentic 2v0; uniformly fit inside previous ink envelope"
                        ),
                        "original_variant": old["variant"],
                        "selected_variant": 0,
                        "source_font_sha256": recipe_value("font_sha256"),
                        "source_sample_id": recipe_value("digit_sample_ids")[0],
                        "source_sample_variant": 0,
                        "original_item_matrix": list(item[2]),
                        "post_matrix_uniform_scale": scale,
                        "post_matrix_translation": shift.tolist(),
                        "source_to_output_affine": (
                            Affine2D.from_values(*item[2]).get_matrix()
                            * np.array([[scale, scale, scale], [scale, scale, scale], [1, 1, 1]])
                            + np.array([[0, 0, shift[0]], [0, 0, shift[1]], [0, 0, 0]])
                        ).tolist(),
                        "ink_width_before_after_px": [
                            float(high[0] - low[0]),
                            bounds[2] - bounds[0],
                        ],
                        "ink_height_before_after_px": [
                            float(high[1] - low[1]),
                            bounds[3] - bounds[1],
                        ],
                        "left_right_clearance_added_px": [bounds[0] - low[0], high[0] - bounds[2]],
                        "original_bounds_px": old["bounds_px"],
                        "source_topology_preserved": True,
                        "source_contours": source_contours,
                        "original_pixels_edited": False,
                        "new_label": False,
                        "neural_updates": 0,
                        "author_acceptance": False,
                        "status": "isolated-research-candidate",
                        "no_new_neighbor_contact": True,
                    }
                    row.update(
                        bounds_px=bounds,
                        allowed_bounds_px=old["bounds_px"],
                        variant=0,
                        actual_glyph_id=self.gid,
                        glyph_name=self.name,
                        numeral_pi_feedback=treatment,
                    )
                    d.update(applied=True, status="accepted-for-visual-review", treatment=treatment)
            self.decisions.append(d)
        if identity(before) != identity(result):
            raise ValueError("Glyph semantics/order changed")
        return result
