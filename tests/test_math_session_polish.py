from types import SimpleNamespace

import numpy as np
import pytest
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from research.corpus.math_session_polish import (
    cap_stem,
    compact_small_roots,
    outline_bounds,
    polygons,
    sections,
)


def shape(points):
    vertices = np.asarray(points, dtype=float)
    return MplPath(np.vstack((vertices, vertices[:1])), closed=True)


def test_stretched_pen_width_keeps_the_original_centre_and_height():
    original = shape([(4, 0), (10, 0), (16, 90), (10, 90)])
    outline, metrics = cap_stem(original, 2.5)
    candidate = MplPath(outline.verts, outline.codes)

    before = sections(polygons(original), np.array([45.0]))[0]
    after = sections(polygons(candidate), np.array([45.0]))[0]
    assert after.mean() == pytest.approx(before.mean())
    assert np.diff(after).item() == pytest.approx(2.5)
    assert outline_bounds(candidate.vertices, candidate.codes)[:, 1].tolist() == [0, 90]
    assert metrics["body_width_p90_before_px"] == pytest.approx(6)
    assert metrics["body_width_p90_after_px"] == pytest.approx(2.5)


def test_compact_root_keeps_digit_shape_and_roof_overhang():
    outlines = {
        "stem": shape([(10, 50), (13, 50), (18, 14), (30, 12), (30, 10), (15, 10)]),
        "bar": shape([(29, 10), (75, 10), (75, 12), (29, 12)]),
        "digit": shape([(52, 20), (62, 20), (62, 44), (52, 44)]),
    }
    page = SimpleNamespace(items=[("path", key, (1, 0, 0, 1, 0, 0), 255, 0) for key in outlines])
    renderer = SimpleNamespace(
        cfg=SimpleNamespace(body_size=46),
        _outlines={key: Outline(path.vertices, path.codes) for key, path in outlines.items()},
    )
    records = [
        {
            "context": "mathematics",
            "item_index": index,
            "semantic": ch,
            "structure": role,
            "latex": r"\sqrt{2}",
            "bounds_px": outline_bounds(path.vertices, path.codes).ravel().tolist(),
        }
        for index, (ch, role, path) in enumerate(
            zip(
                ("√", "radical-overbar", "2"),
                ("radical-sign", "radical-overbar", None),
                outlines.values(),
                strict=True,
            )
        )
    ]
    records[0].update(radical_join={"applied": True}, paired_items=[0, 1])
    original_digit = page.items[2]
    compact_small_roots(page, renderer, records)

    treatment = records[0]["polish"]
    assert treatment["gap_after_px"] == pytest.approx(5.52)
    assert 0 < treatment["stem_shift_px"] < treatment["gap_before_px"] - 5.52
    assert page.items[2][1] == original_digit[1]
    assert page.items[2][2][:4] == original_digit[2][:4]
    assert page.items[2][2][5] == original_digit[2][5]
    assert records[1]["bounds_px"][2] - records[2]["bounds_px"][2] == pytest.approx(13)
