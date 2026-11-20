"""CPU kernel extracted from research.cpu_glyph_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from handnote.markdown import Block
from handnote.opentype import read_alternates
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib import rc_context
from matplotlib.ft2font import FT2Font
from matplotlib.path import Path as MplPath
from matplotlib.textpath import text_to_path

from hwsol.adapters.handnote import RENDER_LOCK, upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.handwriting import finish_ink
from hwsol.jobs.layout import MathDiagnostics, SinglePageRenderer, ink_geometry, validate_markdown
from hwsol.jobs.math_strokes import apply_math_strokes
from hwsol.jobs.provenance import classify_rules, item_bounds, visible_provenance
from hwsol.personal_ink.kernels.math_glyph_resolver import placed_outline
from hwsol.personal_ink.kernels.math_session_polish import body_width, polygons
from hwsol.personal_ink.kernels.readability import placed_path
from hwsol.profiles.io import sha256

TARGETS = frozenset("0rdkf")
CONFIG = RenderConfig(dpi=300, font_size_pt=11, seed=1909, ink="#000000", math_strokes="personal")
R_LEFT_BEARING_REDUCTION_EM = 0.021


class FeedbackRenderer(SinglePageRenderer):
    """Reduce r's inherited left bearing and pen advance by exactly the same amount."""

    feedback_r_enabled = True

    def char_metrics(self, ch, size, k=None):
        shift, advance = super().char_metrics(ch, size, k)
        if ch == "r" and self.feedback_r_enabled:
            delta = size * R_LEFT_BEARING_REDUCTION_EM
            return (shift - delta, advance - delta)
        return (shift, advance)

    def draw_matrix(self, *args, **kwargs):
        self.feedback_r_enabled = False
        try:
            return super().draw_matrix(*args, **kwargs)
        finally:
            self.feedback_r_enabled = True


def layout_with_r_metrics(source, identity, profile, cfg=CONFIG):
    """Existing layout contract with the opt-in renderer subclass; no monkeypatching."""
    blocks = validate_markdown(source, profile)
    upstream = upstream_config(profile.font, cfg, bool(profile.manifest.variants))
    upstream = upstream.replace(
        h1_size=upstream.body_size,
        h2_size=upstream.body_size,
        h3_size=upstream.body_size,
        math_scale=1.0,
    )
    with RENDER_LOCK, rc_context():
        logger = logging.getLogger("matplotlib._mathtext")
        handler = MathDiagnostics()
        logger.addHandler(handler)
        try:
            renderer = FeedbackRenderer(upstream)
            renderer.configure("baseline", profile, identity, None, cfg, None)
            pages = renderer.render_vector([Block("h2", (("t", identity),)), *blocks])
        finally:
            logger.removeHandler(handler)
        if len(pages) != 1 or renderer.warnings:
            raise ValueError(f"Unreliable feedback layout: {renderer.warnings}")
        page = pages[0]
        classify_rules(page, renderer)
        strokes = (
            apply_math_strokes(page, renderer, profile) if cfg.math_strokes == "personal" else []
        )
        finish_ink(page, renderer, profile, False, renderer.stroke_seed)
        records = visible_provenance(page, renderer, profile, strokes)
        for record in records:
            if record["context"] == "text" and record["semantic"] == "r":
                record["cpu_feedback_layout"] = {
                    "method": "inherited left bearing 46->25 per 1000 em",
                    "draw_shift_delta_em": -R_LEFT_BEARING_REDUCTION_EM,
                    "advance_delta_em": -R_LEFT_BEARING_REDUCTION_EM,
                    "outline_unchanged": True,
                    "following_text": "moves with reduced advance; wrapping may change",
                    "space_policy": "unchanged",
                }
        bounds, _ = ink_geometry(page, renderer)
        if min(bounds[:2]) < 3 or bounds[2] > page.width - 3 or bounds[3] > page.height - 3:
            raise ValueError("Feedback layout crosses page boundary")
        return (page, renderer, records)


def lower_k_body(vertices, codes):
    """Monotone piecewise output mapping; original outlines are never overwritten."""
    raw = MplPath(vertices, codes)
    lo, hi = (raw.get_extents().ymin, raw.get_extents().ymax)
    knot, target = (470.0, 410.0)
    if not lo < target < knot < hi:
        raise ValueError("Unexpected k geometry; no unbounded fallback")
    paths = []
    for ring in polygons(raw):
        dense = []
        for start, end in zip(ring[:-1], ring[1:], strict=True):
            dense.append(start)
            if (start[1] - knot) * (end[1] - knot) < 0:
                dense.append(start + (end - start) * (knot - start[1]) / (end[1] - start[1]))
        pts = np.vstack([*dense, ring[-1]])
        pts[:, 1] = np.interp(pts[:, 1], [lo, knot, hi], [lo, target, hi])
        paths.append(MplPath(pts, closed=True))
    result = MplPath.make_compound_path(*paths)
    return (result.vertices, result.codes)


class CpuGlyphFeedback:
    """Recorded variants; default prose/title, explicit opt-in math d/f/k/r only."""

    def __init__(self, font: Path):
        self.font = font
        self.font_sha = sha256(font)
        with TTFont(font) as file:
            cmap, alts = (file.getBestCmap(), read_alternates(file))
            self.names = {ch: [cmap[ord(ch)], *alts.get(cmap[ord(ch)], [])] for ch in TARGETS}
            self.ids = {
                name: file.getGlyphID(name) for names in self.names.values() for name in names
            }

    def apply(self, page, renderer, provenance, *, contexts=frozenset({"text"})):
        contexts = frozenset(contexts)
        if not contexts or not contexts <= {"text", "mathematics"}:
            raise ValueError("Explicit supported feedback contexts required")
        records = copy.deepcopy(provenance)
        for row in records:
            ch, i = (row["semantic"], row["item_index"])
            if row["context"] not in contexts or ch not in TARGETS or page.items[i][0] != "path":
                continue
            if row["context"] == "mathematics" and ch not in {"d", "f", "k", "r"}:
                continue
            if not row["personal_collected"] or row.get("font_sha256") != self.font_sha:
                row["cpu_feedback_rejection"] = "Expected pinned personal font provenance"
                continue
            variant = row.get("variant")
            treatment = None
            selected = variant
            old_item = page.items[i]
            before = row["bounds_px"]
            if ch == "0" and variant in (2, 3):
                selected = variant - 2
                v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                new, scale = placed_outline(Outline(v * (1, -1), c), before)
                old_weight = body_width(placed_path(old_item, renderer))
                new_weight = body_width(MplPath(new.verts, new.codes))
                if new_weight >= old_weight * 0.9:
                    row["cpu_feedback_rejection"] = (
                        "Lighter zero did not pass existing v26 width check"
                    )
                    continue
                matrix = (1, 0, 0, 1, 0, 0)
                treatment = {
                    "method": "v26 lighter recorded zero, extended to text",
                    "fit": scale,
                    "stem_p90_px": [old_weight, new_weight],
                }
            elif ch == "d" and variant == 0:
                selected = 1
                v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                new, scale = placed_outline(Outline(v * (1, -1), c), before)
                matrix = (1, 0, 0, 1, 0, 0)
                treatment = {"method": "select recorded d1 with lower bowl", "fit": scale}
            elif ch == "f" and variant == 1:
                selected = 0
                if row["context"] == "mathematics":
                    v, c = renderer._math_glyph(
                        FT2Font(str(self.font)), self.ids[self.names[ch][0]], ord(ch), 0, True
                    )
                else:
                    v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                new, matrix = (Outline(v, c, stroke=False, digits=4), old_item[2])
                treatment = {"method": "select fuller recorded f0 at native em scale; no dilation"}
                if row["context"] == "mathematics":
                    treatment["outline_units_per_em"] = text_to_path.FONT_SCALE
                    treatment["mathtext_primary_slot"] = True
            elif ch == "r" and row["context"] == "mathematics" and (variant == 1):
                selected = 0
                v, c = renderer._math_glyph(
                    FT2Font(str(self.font)), self.ids[self.names[ch][0]], ord(ch), 0, True
                )
                new, matrix = (Outline(v, c, stroke=False, digits=4), old_item[2])
                treatment = {
                    "method": "select recorded r0 in MathText's existing primary slot",
                    "outline_units_per_em": text_to_path.FONT_SCALE,
                    "mathtext_primary_slot": True,
                    "primary_advance_preserved": True,
                    "alternate_not_stretched": True,
                }
            elif ch == "k":
                if row["context"] == "mathematics":
                    existing = renderer._outlines[old_item[1]]
                    units = 1000 / text_to_path.FONT_SCALE
                    v, c = lower_k_body(existing.verts * units, existing.codes)
                    v = v / units
                else:
                    v, c = handwritten_outline(self.font, ord(ch), variant, 1000)
                    v, c = lower_k_body(v, c)
                new, matrix = (Outline(v, c, stroke=False, digits=4), old_item[2])
                treatment = {
                    "method": "output-only monotone lower-body y calibration",
                    "knot_units": [470, 410],
                    "endpoints_fixed": True,
                    "author_style_acceptance": "pending",
                }
                if row["context"] == "mathematics":
                    treatment["existing_mathtext_fitted_outline_preserved"] = True
                    treatment["outline_units_per_em"] = text_to_path.FONT_SCALE
            if treatment is None:
                continue
            key = ("cpu-glyph-feedback", ch, i, selected)
            renderer._outlines[key] = new
            page.items[i] = ("path", key, matrix, old_item[3], 0)
            row.update(
                cpu_feedback={
                    "original_variant": variant,
                    "source_font_sha256": self.font_sha,
                    "original_bounds_px": before,
                    **treatment,
                },
                bounds_px=item_bounds(page.items[i], renderer),
                variant=selected,
                glyph_name=self.names[ch][selected],
                actual_glyph_id=self.ids[self.names[ch][selected]],
                source_type="personal-outline-geometric-calibration"
                if ch == "k"
                else "personal-outline",
            )
        return records
