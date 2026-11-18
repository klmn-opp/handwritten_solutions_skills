"""Place measured horizontal ink inside the parser's math layout and available clearance."""

from dataclasses import replace

import numpy as np
from matplotlib.path import Path as MplPath

from hwsol.profiles.store import LoadedProfile


def apply_math_strokes(page, renderer, profile: LoadedProfile) -> list[dict]:
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
