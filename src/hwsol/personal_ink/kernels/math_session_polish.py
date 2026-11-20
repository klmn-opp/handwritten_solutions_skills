"""CPU kernel extracted from research.corpus.math_session_polish; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from handnote.opentype import read_alternates
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from hwsol.personal_ink.kernels.math_glyph_resolver import placed_outline
from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.personal_ink.kernels.readability import placed_path


def polygons(path: MplPath) -> list[np.ndarray]:
    """Discard close-path placeholders before flattening the outline."""
    if path.codes is None:
        return path.to_polygons()
    pts = path.vertices.copy()
    starts = np.flatnonzero(path.codes == MplPath.MOVETO)
    for start, end in zip(starts, [*starts[1:], len(pts)], strict=True):
        if path.codes[end - 1] == MplPath.CLOSEPOLY:
            pts[end - 1] = pts[start]
    return MplPath(pts, path.codes).to_polygons()


def sections(rings: list[np.ndarray], ys: np.ndarray) -> list[np.ndarray]:
    """Measure horizontal ink intervals, including the two edges of a counter."""
    starts = np.concatenate([p[:-1] for p in rings])
    ends = np.concatenate([p[1:] for p in rings])
    dx, dy = (ends - starts).T
    x, y = starts.T
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dx / dy
    result = []
    for at in ys:
        active = (np.minimum(y, y + dy) <= at) & (at < np.maximum(y, y + dy))
        cross = np.sort(x[active] + (at - y[active]) * slope[active])
        result.append(cross.reshape(-1, 2) if len(cross) % 2 == 0 else np.empty((0, 2)))
    return result


def body_width(path: MplPath) -> float:
    """Use a 90th-percentile ink width through the centre of a glyph."""
    y0, y1 = outline_bounds(path.vertices, path.codes)[:, 1]
    rings = polygons(path)
    values = [
        np.diff(row, axis=1).max()
        for row in sections(rings, np.linspace(y0 + 0.22 * (y1 - y0), y1 - 0.22 * (y1 - y0), 48))
        if len(row)
    ]
    return float(np.quantile(values, 0.9)) if values else 0.0


def cap_stem(path: MplPath, maximum: float) -> tuple[Outline, dict] | None:
    """Limit excess width around the original centres, blending into joins and ends."""
    before = body_width(path)
    if before <= maximum * 1.08:
        return None
    rings = polygons(path)
    y0, y1 = outline_bounds(path.vertices, path.codes)[:, 1]
    height = y1 - y0
    ys = np.linspace(y0, y1, max(50, int(height * 1.5)))
    intervals = sections(rings, ys)
    centres = np.zeros(len(ys))
    scales = np.ones(len(ys))
    relief = np.minimum(ys - y0, y1 - ys)
    blend = np.clip((relief - height * 0.04) / max(1, height * 0.1), 0, 1)
    for i, row in enumerate(intervals):
        if len(row) != 1:
            continue
        left, right = row[0]
        width = right - left
        centres[i] = (left + right) / 2
        if width > maximum:
            scales[i] -= (1 - maximum / width) * blend[i]
    active = scales < 1
    if not active.any():
        return None
    centres = np.interp(ys, ys[active], centres[active])
    contours = []
    for ring in rings:
        dense = []
        for start, end in zip(ring[:-1], ring[1:], strict=True):
            n = max(1, int(np.ceil(np.abs(end - start).max())))
            dense.append(start + np.arange(n)[:, None] / n * (end - start))
        vertices = np.vstack([*dense, ring[-1:]])
        y = vertices[:, 1]
        centre = np.interp(y, ys, centres)
        vertices[:, 0] = centre + (vertices[:, 0] - centre) * np.interp(y, ys, scales)
        contours.append(MplPath(vertices, closed=True))
    result = MplPath.make_compound_path(*contours)
    after = body_width(result)
    if after > before * 0.94:
        return None
    return (
        Outline(result.vertices, result.codes, stroke=False, digits=4),
        {
            "method": "centre-preserving stem width limit",
            "body_width_p90_before_px": before,
            "body_width_p90_after_px": after,
            "stem_width_limit_px": maximum,
        },
    )


def set_outline(page, renderer, record, outline, allowed, treatment, *, pair=None) -> None:
    i = record["item_index"]
    key = ("session-math-polish", i)
    renderer._outlines[key] = outline
    page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), page.items[i][3], 0)
    record.update(
        original_bounds_px=record["bounds_px"],
        allowed_bounds_px=list(allowed),
        bounds_px=outline_bounds(outline.verts, outline.codes).ravel().tolist(),
        polish=treatment,
    )
    if pair is not None:
        record["paired_items"] = pair


def compact_small_roots(page, renderer, records: list[dict]) -> None:
    """Compact a root about its centre while preserving digit ink and roof overhang."""
    for sign in records:
        info = sign.get("radical_join")
        if sign["structure"] != "radical-sign" or not info or "session_source" in sign:
            continue
        si, bi = sign["paired_items"]
        bar = records[bi]
        sx, sy, sr, sb = sign["bounds_px"]
        bx, by, br, bb = bar["bounds_px"]
        local = [
            r
            for r in records
            if r["context"] == "mathematics"
            and r["item_index"] not in (si, bi)
            and (r["latex"] == sign["latex"])
            and (len(r["bounds_px"]) == 4)
            and (bx < (r["bounds_px"][0] + r["bounds_px"][2]) / 2 < br)
            and (r["bounds_px"][1] > bb - 1)
            and (r["bounds_px"][3] < sb + 2.5)
        ]
        if len(local) != 1 or (local[0]["semantic"] or "") not in set("0123456789"):
            continue
        digit = local[0]
        left, top, right, bottom = digit["bounds_px"]
        raw = placed_path(page.items[si], renderer)
        clean = MplPath.make_compound_path(*(MplPath(p, closed=True) for p in polygons(raw)))
        band = clean.clip_to_bbox(Bbox.from_extents(sx - 1, top - 0.5, br, bottom + 0.5))
        if not len(band.vertices):
            continue
        edge = outline_bounds(band.vertices, band.codes)[1, 0]
        gap = left - edge
        reduction = min(
            gap - 0.12 * renderer.cfg.body_size, br - sr - 0.07 * renderer.cfg.body_size
        )
        if reduction < 1:
            continue
        digit_shift = round(reduction / 2)
        stem_shift = reduction - digit_shift
        rule = placed_path(page.items[bi], renderer)
        rule_vertices = rule.vertices.copy()
        rule_vertices[np.isclose(rule_vertices[:, 0], bx), 0] += stem_shift
        rule_vertices[np.isclose(rule_vertices[:, 0], br), 0] -= digit_shift
        allowed = [min(sx, bx), min(sy, by, top), max(sr, br, right), max(sb, bb, bottom)]
        di = digit["item_index"]
        treatment = {
            "method": "close small root side clearance",
            "radicand_item": di,
            "gap_before_px": float(gap),
            "gap_after_px": float(gap - reduction),
            "stem_shift_px": float(stem_shift),
            "digit_shift_px": -digit_shift,
            "roof_endpoint_px": br - digit_shift,
        }
        set_outline(
            page,
            renderer,
            sign,
            Outline(raw.vertices + (stem_shift, 0), raw.codes, stroke=False, digits=4),
            allowed,
            treatment,
            pair=[si, bi, di],
        )
        set_outline(
            page,
            renderer,
            bar,
            Outline(rule_vertices, rule.codes, stroke=False, digits=4),
            allowed,
            treatment,
            pair=[si, bi, di],
        )
        kind, key, matrix, alpha, pen = page.items[di]
        page.items[di] = (kind, key, (*matrix[:4], matrix[4] - digit_shift, matrix[5]), alpha, pen)
        digit.update(
            original_bounds_px=digit["bounds_px"],
            allowed_bounds_px=allowed,
            bounds_px=[left - digit_shift, top, right - digit_shift, bottom],
            polish=treatment,
            paired_items=[si, bi, di],
        )


class SessionInkPolish:
    def __init__(self, calibration, font: Path) -> None:
        self.braces = {char: calibration.variants[char][0] for char in "{}"}
        with TTFont(font) as file:
            base = file.getBestCmap()[ord("0")]
            names = [base, *read_alternates(file).get(base, [])]
            self.zeros = {
                i: (
                    names[i],
                    file.getGlyphID(names[i]),
                    handwritten_outline(font, ord("0"), i, 100),
                )
                for i in (0, 1)
            }

    def apply(self, page, renderer, baseline: list[dict]) -> list[dict]:
        records = [dict(r) for r in baseline]
        compact_small_roots(page, renderer, records)
        for row in records:
            idx = row["item_index"]
            char = row["semantic"]
            if row["context"] != "mathematics" or page.items[idx][0] != "path":
                continue
            bounds = row["bounds_px"]
            if char in self.braces and row.get("fallback_reason"):
                sample = self.braces[char]
                tall = bounds[3] - bounds[1] >= 1.35 * renderer.cfg.body_size
                slot = bounds
                if tall:
                    source_span = np.ptp(
                        outline_bounds(sample.outline.verts, sample.outline.codes), axis=0
                    )
                    x0, y0, x1, y1 = bounds
                    width = min(x1 - x0, (y1 - y0) * source_span[0] / source_span[1])
                    centre = (x0 + x1) / 2
                    slot = [centre - width / 2, y0, centre + width / 2, y1]
                outline, scale = placed_outline(sample.outline, slot, stretch=tall)
                set_outline(
                    page,
                    renderer,
                    row,
                    outline,
                    bounds,
                    {
                        "method": "side-specific handwritten brace",
                        "previous_source": row["source_type"],
                        "scale_translate": scale,
                        "full_height_fit": tall,
                    },
                )
                row.pop("fallback_reason", None)
                row.update(
                    source_type="personal-session-outline",
                    personal_collected=True,
                    external_fallback=False,
                    procedural_geometry=False,
                    actual_glyph_id=sample.source["id"],
                    glyph_name=None,
                    variant=None,
                    font=None,
                    font_sha256=None,
                    session_source=sample.source,
                )
            elif (
                char == "0"
                and row["personal_collected"]
                and (row.get("variant") in (2, 3))
                and ("polish" not in row)
            ):
                name, glyph_id, raw = self.zeros[row["variant"] - 2]
                vertices, codes = raw
                outline, scale = placed_outline(Outline(vertices * (1, -1), codes), bounds)
                before = body_width(placed_path(page.items[idx], renderer))
                after = body_width(MplPath(outline.verts, outline.codes))
                if after >= before * 0.9:
                    continue
                set_outline(
                    page,
                    renderer,
                    row,
                    outline,
                    bounds,
                    {
                        "method": "select the lighter recorded zero variant",
                        "previous_variant": row["variant"],
                        "body_width_p90_before_px": before,
                        "body_width_p90_after_px": after,
                        "scale_translate": scale,
                    },
                )
                row.update(variant=row["variant"] - 2, actual_glyph_id=glyph_id, glyph_name=name)
            elif char in ("|", "∣", "(", ")", "[", "]", "√") and row["personal_collected"]:
                if bounds[3] - bounds[1] < 1.35 * renderer.cfg.body_size or row.get("polish"):
                    continue
                result = cap_stem(
                    placed_path(page.items[idx], renderer), 0.055 * renderer.cfg.body_size
                )
                if result:
                    outline, treatment = result
                    set_outline(page, renderer, row, outline, bounds, treatment)
        return records
