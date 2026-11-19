"""Opt-in research spacing and radical assembly; no model or production mutation."""

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from hwsol.jobs.provenance import item_bounds


def compact_blank_columns(ink: np.ndarray, em: float) -> tuple[np.ndarray, dict]:
    """Remove excess completely empty columns, never rescale/cut a stroke."""
    if ink.ndim != 2 or em <= 0 or not np.isfinite(ink).all():
        raise ValueError("Finite grayscale ink and positive em required")
    occupied = np.any(ink > 0, axis=0)
    where = np.flatnonzero(occupied)
    if not len(where):
        return ink.copy(), {"removed_columns": 0, "gaps": []}
    limit = max(2, round(em * 0.075))
    keep = np.ones(ink.shape[1], bool)
    changes = []
    for left, right in zip(where[:-1], where[1:], strict=True):
        gap = int(right - left - 1)
        remove = min(max(0, gap - limit), gap // 2)
        if remove:
            start = left + 1 + (gap - remove) // 2
            keep[start : start + remove] = False
            changes.append({"left": int(left), "before": gap, "after": gap - remove})
    output = ink[:, keep]
    assert np.array_equal(output[:, np.any(output > 0, axis=0)], ink[:, occupied])
    return output, {"removed_columns": int((~keep).sum()), "gaps": changes}


def word_positions(
    widths, left: float, right: float, em: float, preferred_gaps=None
) -> tuple[list[float], float]:
    """Lay a fixed text line with a visible gap; refuse rather than squash ink."""
    widths = np.asarray(widths, dtype=float)
    if not len(widths) or np.any(widths <= 0) or not np.isfinite(widths).all():
        raise ValueError("Positive word widths required")
    count = len(widths) - 1
    room = right - left - widths.sum()
    if (count and room < count * 0.23 * em) or room < -1e-6:
        raise ValueError("Insufficient line width for readable word separation")
    gaps = np.full(count, 0.34 * em)
    if preferred_gaps is not None:
        if len(preferred_gaps) != count or not np.isfinite(preferred_gaps).all():
            raise ValueError("One finite gap per word boundary required")
        gaps = np.clip(preferred_gaps, 0.30 * em, 0.44 * em)
    if gaps.sum() > room:
        floor = 0.23 * em
        gaps = floor + (gaps - floor) * ((room - count * floor) / (gaps - floor).sum())
    return (left + np.r_[0, np.cumsum(widths[:-1] + gaps)]).tolist(), float(
        gaps.mean()
    ) if count else 0.0


def placed_path(item, renderer) -> MplPath:
    _, key, (a, b, c, d, e, f), *_ = item
    outline = renderer._outlines[key]
    return MplPath(outline.verts @ np.array([[a, b], [c, d]]) + (e, f), outline.codes)


def join_radicals(page, renderer, provenance: list[dict]) -> list[dict]:
    """Keep MathText radicand slots; join the observed stem to one fitted overbar."""
    signs = [r for r in provenance if r["structure"] == "radical-sign"]
    bars = [r for r in provenance if r["structure"] == "radical-overbar"]
    used, changes = set(), []
    em = renderer.cfg.body_size
    for sign in signs:
        si = sign["item_index"]
        sx, sy, sr, sb = item_bounds(page.items[si], renderer)
        candidates = [
            b
            for b in bars
            if b["item_index"] not in used
            and abs(b["bounds_px"][0] - sr) < 0.4 * em
            and abs(b["bounds_px"][1] - sy) < 0.4 * em
        ]
        if not candidates:
            changes.append({"item_index": si, "applied": False, "reason": "no_local_bar"})
            continue
        bar = min(candidates, key=lambda b: abs(b["bounds_px"][0] - sr))
        bi = bar["item_index"]
        bx, by, br, bb = item_bounds(page.items[bi], renderer)
        path = placed_path(page.items[si], renderer)
        # Cross-section before the right terminal; requires exactly one ink interval.
        join = sx + (sr - sx) * 0.82
        ys = np.linspace(sy, sb, max(256, int((sb - sy) * 100)))
        inside = path.contains_points(np.column_stack([np.full_like(ys, join), ys]))
        edges = np.diff(np.r_[False, inside, False].astype(int))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        if len(starts) != 1:
            changes.append({"item_index": si, "applied": False, "reason": "ambiguous_terminal"})
            continue
        low, high = ys[starts[0]], ys[ends[0] - 1]
        center = (low + high) / 2
        contained = [
            r["bounds_px"]
            for r in provenance
            if r["context"] == "mathematics"
            and r["item_index"] not in {si, bi}
            and r["latex"] == sign["latex"]
            and len(r["bounds_px"]) == 4
            and bx < (r["bounds_px"][0] + r["bounds_px"][2]) / 2 < br
            and r["bounds_px"][1] >= bb - 0.02 * em
            and r["bounds_px"][3] <= sb + 0.05 * em
        ]
        if not contained:
            changes.append({"item_index": si, "applied": False, "reason": "no_local_radicand"})
            continue
        top = min(b[1] for b in contained) - 0.12 * em
        bottom = max(b[3] for b in contained) + 0.06 * em
        scale = (bottom - top) / max(1e-8, sb - center)
        if not 0.55 <= scale <= 1.15 or high - low < 0.25:
            changes.append({"item_index": si, "applied": False, "reason": "unsafe_stem_fit"})
            continue
        # Only trim the redundant terminal. The radicand is never transformed.
        # clip_to_bbox otherwise treats glyph CLOSEPOLY dummy vertices as ink,
        # creating a spurious diagonal. Normalize closed contours first.
        closed = MplPath.make_compound_path(
            *(MplPath(polygon, closed=True) for polygon in path.to_polygons())
        )
        head = closed.clip_to_bbox(Bbox.from_extents(sx - 1, sy - 1, join + 0.4, sb + 1))
        vertices = head.vertices.copy()
        vertices[:, 1] = top + (vertices[:, 1] - center) * scale
        thickness = (high - low) * scale
        if thickness < 0.65:
            changes.append({"item_index": si, "applied": False, "reason": "terminal_too_thin"})
            continue
        stem_key, bar_key = ("research-radical-stem", si), ("research-radical-bar", bi)
        renderer._outlines[stem_key] = Outline(vertices, head.codes, stroke=False)
        rect = np.array(
            [
                [join - 0.4, top - thickness / 2],
                [br, top - thickness / 2],
                [br, top + thickness / 2],
                [join - 0.4, top + thickness / 2],
                [join - 0.4, top - thickness / 2],
            ]
        )
        renderer._outlines[bar_key] = Outline(rect, np.array([1, 2, 2, 2, 79]), stroke=False)
        page.items[si] = ("path", stem_key, (1, 0, 0, 1, 0, 0), page.items[si][3], 0)
        page.items[bi] = ("path", bar_key, (1, 0, 0, 1, 0, 0), page.items[bi][3], 0)
        used.add(bi)
        changes.append(
            {
                "item_index": si,
                "bar_item_index": bi,
                "applied": True,
                "old_sign_bounds": [sx, sy, sr, sb],
                "old_bar_bounds": [bx, by, br, bb],
                "new_sign_bounds": item_bounds(page.items[si], renderer),
                "new_bar_bounds": item_bounds(page.items[bi], renderer),
                "terminal_join_x": join,
                "stem_vertical_scale": scale,
                "bar_thickness": thickness,
                "radicand_items_unchanged": True,
                "stem_source": sign["source_type"],
                "bar_source": "procedural-extension-matched-to-stem",
                "new_personal_collection": False,
            }
        )
    return changes
