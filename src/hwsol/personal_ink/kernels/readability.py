"""CPU kernel extracted from research.corpus.readability; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from hwsol.jobs.provenance import item_bounds


def placed_path(item, renderer) -> MplPath:
    _, key, (a, b, c, d, e, f), *_ = item
    outline = renderer._outlines[key]
    return MplPath(outline.verts @ np.array([[a, b], [c, d]]) + (e, f), outline.codes)


def join_radicals(page, renderer, provenance: list[dict], *, snug=False) -> list[dict]:
    """Keep MathText radicand slots; join the observed stem to one fitted overbar."""
    signs = [r for r in provenance if r["structure"] == "radical-sign"]
    bars = [r for r in provenance if r["structure"] == "radical-overbar"]
    used, changes = (set(), [])
    em = renderer.cfg.body_size
    for sign in signs:
        si = sign["item_index"]
        sx, sy, sr, sb = item_bounds(page.items[si], renderer)
        candidates = [
            b
            for b in bars
            if b["item_index"] not in used
            and abs(b["bounds_px"][0] - sr) < 0.4 * em
            and (abs(b["bounds_px"][1] - sy) < 0.4 * em)
        ]
        if not candidates:
            changes.append({"item_index": si, "applied": False, "reason": "no_local_bar"})
            continue
        bar = min(candidates, key=lambda b: abs(b["bounds_px"][0] - sr))
        bi = bar["item_index"]
        bx, by, br, bb = item_bounds(page.items[bi], renderer)
        path = placed_path(page.items[si], renderer)
        join = sx + (sr - sx) * 0.82
        ys = np.linspace(sy, sb, max(256, int((sb - sy) * 100)))
        inside = path.contains_points(np.column_stack([np.full_like(ys, join), ys]))
        edges = np.diff(np.r_[False, inside, False].astype(int))
        starts, ends = (np.where(edges == 1)[0], np.where(edges == -1)[0])
        if len(starts) != 1:
            changes.append({"item_index": si, "applied": False, "reason": "ambiguous_terminal"})
            continue
        low, high = (ys[starts[0]], ys[ends[0] - 1])
        center = (low + high) / 2
        contained = [
            r["bounds_px"]
            for r in provenance
            if r["context"] == "mathematics"
            and r["item_index"] not in {si, bi}
            and (r["latex"] == sign["latex"])
            and (len(r["bounds_px"]) == 4)
            and (bx < (r["bounds_px"][0] + r["bounds_px"][2]) / 2 < br)
            and (r["bounds_px"][1] >= bb - 0.02 * em)
            and (r["bounds_px"][3] <= sb + 0.05 * em)
        ]
        if not contained:
            changes.append({"item_index": si, "applied": False, "reason": "no_local_radicand"})
            continue
        top = min(b[1] for b in contained) - 0.12 * em
        bottom = max(b[3] for b in contained) + 0.06 * em
        scale = (bottom - top) / max(1e-08, sb - center)
        if not 0.55 <= scale <= 1.15 or high - low < 0.25:
            changes.append({"item_index": si, "applied": False, "reason": "unsafe_stem_fit"})
            continue
        closed = MplPath.make_compound_path(
            *(MplPath(polygon, closed=True) for polygon in path.to_polygons())
        )
        head = closed.clip_to_bbox(Bbox.from_extents(sx - 1, sy - 1, join + 0.4, sb + 1))
        vertices = head.vertices.copy()
        vertices[:, 1] = top + (vertices[:, 1] - center) * scale
        clean_head = MplPath.make_compound_path(
            *(MplPath(p, closed=True) for p in MplPath(vertices, head.codes).to_polygons())
        )
        content_left = min(b[0] for b in contained)
        content_top = min(b[1] for b in contained)
        content_bottom = max(b[3] for b in contained)
        band = clean_head.clip_to_bbox(Bbox.from_extents(sx - 1, content_top, br, content_bottom))
        real = band.vertices[band.codes != MplPath.CLOSEPOLY]
        before_gap = float(content_left - real[:, 0].max()) if len(real) else None
        dx = 0.0
        if snug and before_gap is not None:
            dx = max(0.0, min(before_gap - 0.12 * em, br - join - 0.06 * em))
            vertices[:, 0] += dx
            join += dx
        thickness = (high - low) * scale
        if thickness < 0.65:
            changes.append({"item_index": si, "applied": False, "reason": "terminal_too_thin"})
            continue
        stem_key, bar_key = (("research-radical-stem", si), ("research-radical-bar", bi))
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
                "snug": snug,
                "horizontal_stem_shift": dx,
                "stem_to_content_gap_before": before_gap,
                "stem_to_content_gap_after": before_gap - dx if before_gap is not None else None,
            }
        )
    return changes
