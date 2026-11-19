import numpy as np
import pytest

from research.corpus.readability import compact_blank_columns, word_positions


def test_blank_compaction_preserves_every_ink_column_and_vertical_position():
    ink = np.zeros((20, 60), dtype=np.float32)
    ink[3:15, 5:9] = 1
    ink[1:18, 22:27] = 0.35
    ink[2:12, 44:48] = 0.8
    ink[0, 43] = 0.00001  # Even weak nonzero ink is preserved.
    result, report = compact_blank_columns(ink, 40)
    assert report["removed_columns"] > 0
    assert np.array_equal(result[:, np.any(result > 0, axis=0)], ink[:, np.any(ink > 0, axis=0)])
    assert np.count_nonzero(result) == np.count_nonzero(ink)
    assert all(g["after"] >= g["before"] / 2 for g in report["gaps"])


def test_spacing_uses_visible_bounds_and_rejects_overflow():
    positions, gap = word_positions([20, 40, 30], 10, 160, 40)
    assert gap == pytest.approx(13.6)
    assert positions[1] - positions[0] - 20 == pytest.approx(gap)
    assert positions[-1] + 30 <= 160
    with pytest.raises(ValueError, match="Insufficient"):
        word_positions([72, 72], 10, 160, 40)


def test_blank_and_single_word_do_not_gain_artificial_ink():
    image = np.zeros((4, 12))
    actual, report = compact_blank_columns(image, 40)
    assert np.array_equal(actual, image) and report["removed_columns"] == 0
    assert word_positions([30], 5, 50, 40) == ([5.0], 0.0)


def test_radical_join_ignores_closepoly_dummy_and_keeps_radicand():
    from types import SimpleNamespace

    from handnote.vector import Outline
    from matplotlib.path import Path

    from research.corpus.readability import join_radicals

    points = np.array(
        [[0, 20], [0, 3], [10, 3], [10, 5], [2, 5], [2, 20], [0, 20], [99, 99]], float
    )
    codes = np.array([1, 2, 2, 2, 2, 2, 2, 79], dtype=np.uint8)
    bar = np.array([[9, 0], [25, 0], [25, 2], [9, 2], [9, 0]], float)
    digit = np.array([[13, 10], [20, 10], [20, 18], [13, 18], [13, 10]], float)
    outlines = {
        "stem": Outline(points, codes, False),
        "bar": Outline(bar, np.array([1, 2, 2, 2, 79]), False),
        "digit": Outline(digit, np.array([1, 2, 2, 2, 79]), False),
    }
    renderer = SimpleNamespace(cfg=SimpleNamespace(body_size=40), _outlines=outlines)
    page = SimpleNamespace(items=[("path", k, (1, 0, 0, 1, 0, 0), 255, 0) for k in outlines])
    original_digit = page.items[2]
    shared = {"context": "mathematics", "latex": r"\sqrt{2}"}
    records = [
        {
            **shared,
            "item_index": 0,
            "structure": "radical-sign",
            "bounds_px": [0, 3, 10, 20],
            "source_type": "personal-outline-stretched",
        },
        {**shared, "item_index": 1, "structure": "radical-overbar", "bounds_px": [9, 0, 25, 2]},
        {**shared, "item_index": 2, "structure": None, "bounds_px": [13, 10, 20, 18]},
    ]
    result = join_radicals(page, renderer, records)
    assert result[0]["applied"]
    assert page.items[2] == original_digit
    new = renderer._outlines[page.items[0][1]]
    keep = new.codes != Path.CLOSEPOLY
    assert new.verts[keep, 0].min() >= 0
    assert new.verts[keep, 0].max() <= result[0]["terminal_join_x"] + 0.41
    assert not result[0]["new_personal_collection"]


def test_word_spacing_keeps_bounded_existing_rhythm():
    x, _ = word_positions([20, 20, 20], 0, 130, 40, preferred_gaps=[2, 18])
    gaps = np.diff(x) - 20
    assert gaps[0] == pytest.approx(12)
    assert gaps[1] == pytest.approx(17.6)
    assert gaps[1] > gaps[0]
    squeezed, _ = word_positions([20, 20, 20], 0, 80, 40, preferred_gaps=[2, 18])
    assert min(np.diff(squeezed) - 20) >= 9.2
    assert squeezed[-1] + 20 == pytest.approx(80)


def test_disabled_compaction_keeps_original_padded_image_and_origin():
    from research.corpus.readability_study import image_unit

    ink = np.zeros((20, 40), np.float32)
    ink[3:12, 8:14] = 0.8
    image, box, report = image_unit(ink, 100, 200, 40, False)
    assert image.size == (40, 20)
    assert box == [108, 203, 114, 212]
    assert report["image_offset"] == [-8, -3]
    assert np.array_equal(np.asarray(image)[:, :, 3], np.uint8(ink * 255))
    assert [box[0] + report["image_offset"][0], box[1] + report["image_offset"][1]] == [100, 200]
