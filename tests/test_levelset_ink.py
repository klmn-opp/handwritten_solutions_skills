import numpy as np
import pytest

from research.corpus.levelset_ink import contour_distance, mask_scores


def test_projected_contour_preserves_isolated_dots_hairline_and_descender():
    a = np.full((1, 1, 40, 80), -0.02, np.float32)
    a[0, 0, 4, 10] = 0.02
    a[0, 0, 10:32, 10] = 0.02
    a[0, 0, 31, 6:11] = 0.02
    a[0, 0, 28, 50] = 0.02
    before = a.copy()
    p = contour_distance(a)
    assert np.array_equal(a, before)
    assert np.array_equal(p > 0, a > 0)
    assert np.abs(p[0, 0, 0, 70]) > 0.9
    assert mask_scores(p > 0, a > 0)["binary_dice"] == 1


def test_empty_and_solid_projection_are_not_half_gray():
    a = np.stack([np.full((1, 8, 16), -1), np.ones((1, 8, 16))])
    p = contour_distance(a)
    assert np.array_equal(p, a)
    with pytest.raises(ValueError):
        contour_distance(np.full((1, 1, 8, 16), np.nan))
