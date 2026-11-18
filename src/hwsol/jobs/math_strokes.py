"""Place measured horizontal ink inside the parser's math layout and available clearance."""

from dataclasses import replace

import numpy as np
from fontTools.ttLib import TTFont
from handnote.variants import base_outline, handwritten_outline
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from hwsol.jobs.provenance import item_bounds, structure_kind
from hwsol.profiles.store import LoadedProfile


def personal_delimiters(page, renderer, profile) -> list[dict]:
    """Reuse collected delimiter contours inside MathText/upstream bounding boxes."""
    with TTFont(renderer.font_path) as font:
        cmap = font.getBestCmap()
        ids = {cp: font.getGlyphID(name) for cp, name in cmap.items()}
    matrix_items = {}
    for group in renderer.matrices:
        strokes = [i for i in range(*group["item_range"]) if page.items[i][0] == "stroke"]
        pair = {"pmatrix": "()", "bmatrix": "[]", "vmatrix": "||"}.get(group["environment"], "[]")
        for n, i in enumerate(strokes):
            matrix_items[i] = ord(pair[min(n, 1)])
    changes = []
    renderer.structure_fallbacks = []
    for i, item in enumerate(page.items):
        cp = matrix_items.get(i)
        if item[0] == "path" and item[1][0] == "mglyph":
            key = item[1]
            if key[3] in map(ord, "()[]{}|√") and str(renderer.font_path.resolve()) != str(key[1]):
                cp = key[3]
        if cp not in profile.manifest.personal_codepoints:
            continue
        outline = handwritten_outline(renderer.font_path, cp, 0, 100)
        if outline is None:
            vertices, codes, upem = base_outline(renderer.font_path, cp)
            vertices = vertices * (100 / upem)
        else:
            vertices, codes = outline
        points = vertices[codes != MplPath.CLOSEPOLY]
        lo, hi = points.min(axis=0), points.max(axis=0)
        bounds = item_bounds(item, renderer)
        low, high = np.array(bounds[:2]), np.array(bounds[2:])
        if cp == ord("√"):
            stretch = (high[1] - low[1]) / max(1e-8, hi[1] - lo[1])
            squeeze = (high[0] - low[0]) / max(1e-8, hi[0] - lo[0])
            if not 0.55 <= stretch / max(squeeze, 1e-8) <= 1.8:
                renderer.structure_fallbacks.append(
                    {
                        "item_index": i,
                        "structure": "radical-sign",
                        "reason": "collected contour exceeds aspect distortion limit",
                        "fallback": "parser-font",
                        "personal_collected": False,
                    }
                )
                continue
        # Stroke centerline width is zero for vertical bars; include its actual nib width.
        if high[0] - low[0] < 1:
            low[0] -= item[4] / 2
            high[0] += item[4] / 2
        placed = vertices.copy()
        placed[:, 0] = low[0] + (vertices[:, 0] - lo[0]) / max(1e-8, hi[0] - lo[0]) * (
            high[0] - low[0]
        )
        placed[:, 1] = high[1] - (vertices[:, 1] - lo[1]) / max(1e-8, hi[1] - lo[1]) * (
            high[1] - low[1]
        )
        key = ("personal-structure-glyph", cp, ids[cp], i)
        renderer._outlines[key] = Outline(placed, codes, stroke=False)
        page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), item[3], 0)
        changes.append(
            {
                "item_index": i,
                "codepoint": cp,
                "glyph_id": ids[cp],
                "structure": "matrix-bracket"
                if i in matrix_items
                else "radical-sign"
                if cp == ord("√")
                else "stretchable-delimiter",
            }
        )
    renderer.structure_replacements = changes
    return changes


def apply_math_strokes(page, renderer, profile: LoadedProfile) -> list[dict]:
    personal_delimiters(page, renderer, profile)
    samples = {
        kind: [
            (style.sample_id, stroke)
            for style in profile.styles
            for stroke in style.math_strokes
            if stroke.kind == kind
        ]
        for kind in ("bar", "rule")
    }
    if not any(samples.values()):
        return []
    # Bounds are measured once from the original page, so adjacent replacements cannot
    # progressively expand one another's space. Filled glyph paths already exclude pen edges.
    bounds = {}
    for i, item in enumerate(page.items):
        if item[0] == "path":
            _, key, (a, b, c, d, x, y), *_ = item
            outline = renderer._outlines[key]
            points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
            if len(points):
                points = points @ np.array([[a, b], [c, d]]) + (x, y)
                bounds[i] = (points.min(axis=0), points.max(axis=0))
    records = []
    formulas = {
        i: group["latex"] for group in renderer.formulas for i in range(*group["item_range"])
    }
    renderer.structure_radicals = [
        item_bounds(item, renderer)
        for item in page.items
        if item[0] == "path" and item[1][0] == "mglyph" and item[1][3] == 0x221A
    ]
    for i, (low, high) in bounds.items():
        _, key, matrix, alpha, _ = page.items[i]
        if key[0] == "mglyph" and key[3] == 0x0304:
            kind = "bar"
        elif key[0] == "mrect" and high[0] - low[0] > 2 * (high[1] - low[1]):
            kind = "rule"
        else:
            continue
        if not samples[kind]:
            continue
        # Explicit source selection, stable across runs. Older observations stay in the Profile.
        sample_id, stroke = samples[kind][len(records) % len(samples[kind])]
        width = high[0] - low[0]
        centre = (high[1] + low[1]) / 2
        em = min(renderer.cfg.body_size, (high[1] - low[1]) / 0.0625)
        if kind == "bar":
            a, b, c, d, _, _ = matrix
            em = min(renderer.cfg.body_size, 100 * np.sqrt(abs(a * d - b * c)))
        knots = np.asarray(stroke.envelope)
        xs = low[0] + knots[:, 0] * width
        # Scale the centreline with length, and the actual nib thickness with text size.
        ys = knots[:, 1] * width / stroke.width_em
        thickness = knots[:, 2] * em
        top, bottom = ys - thickness / 2, ys + thickness / 2
        middle = (top.min() + bottom.max()) / 2
        top, bottom = top - middle, bottom - middle
        space = renderer.cfg.body_size * 0.22
        for j, (other_low, other_high) in bounds.items():
            if j == i or other_high[0] <= low[0] or other_low[0] >= high[0]:
                continue
            if other_high[1] <= centre:
                space = min(space, centre - other_high[1] - 0.5)
            elif other_low[1] >= centre:
                space = min(space, other_low[1] - centre - 0.5)
            else:
                space = min(space, (high[1] - low[1]) / 2)
        # The original rectangle remains a safe lower bound for nested accents/radicals.
        space = max((high[1] - low[1]) / 2, space)
        factor = min(1.0, space / max(abs(top.min()), abs(bottom.max())))
        vertices = np.vstack(
            (
                np.column_stack((xs, centre + top * factor)),
                np.column_stack((xs[::-1], centre + bottom[::-1] * factor)),
                [[xs[0], centre + top[0] * factor]],
            )
        )
        codes = np.full(len(vertices), MplPath.LINETO, dtype=MplPath.code_type)
        codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
        new_key = ("personal-math-stroke", kind, sample_id, i)
        renderer._outlines[new_key] = replace(
            renderer._outlines[key], verts=vertices, codes=codes, stroke=False, digits=3
        )
        page.items[i] = ("path", new_key, (1, 0, 0, 1, 0, 0), alpha, 0)
        records.append(
            {
                "kind": kind,
                "item_index": i,
                "structure": structure_kind(
                    key, formulas.get(i, ""), [*low, *high], renderer, page.items[i]
                )
                or "math-rule",
                "sample_id": sample_id,
                "source_sha256": stroke.source_sha256,
                "source_box": stroke.source_box,
                "original_kind": key[0],
                "original_bounds_px": [*low.tolist(), *high.tolist()],
                "placed_bounds_px": [
                    float(xs[0]),
                    float(vertices[:, 1].min()),
                    float(xs[-1]),
                    float(vertices[:, 1].max()),
                ],
                "vertical_fit": float(factor),
            }
        )
    return records
