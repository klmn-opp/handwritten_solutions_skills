import numpy as np
import pytest

from research.corpus.clean_ink import ink_metrics, redistance_batch, sharpen_ink
from research.corpus.ink_geometry import signed_distance


def fixture():
    a = np.zeros((40, 100), np.float32)
    a[5, 12] = 1  # isolated dot
    a[10:25, 12] = 1  # one pixel hairline
    a[10:25, 45:47] = 1
    a[24:34, 46] = 1  # descender
    a[33, 42:46] = 1
    a[24, 70] = 1  # punctuation
    return a


def test_zero_update_contrast_preserves_thin_ink_and_punctuation_contour():
    x = fixture() * 0.26 + 0.4
    out = sharpen_ink(x)
    assert np.array_equal(out > 0.5, x > 0.5)
    assert out[5, 12] > x[5, 12] and out[0, 0] < x[0, 0]
    with pytest.raises(ValueError):
        sharpen_ink(x, 0)


def test_redistance_does_not_invent_or_remove_isolated_ink():
    mask = fixture()
    source = ((mask * 2 - 1) * 0.02)[None, None]
    before = source.copy()
    target, valid = redistance_batch(source)
    assert valid.tolist() == [True]
    assert np.array_equal(target > 0, source > 0)
    assert np.array_equal(source, before)
    assert np.array_equal(target[0, 0], signed_distance(mask))
    assert abs(target[0, 0, 0, 0]) > abs(source[0, 0, 0, 0])


def test_empty_and_solid_self_targets_do_not_create_fake_strokes():
    targets, valid = redistance_batch(np.stack([np.ones((1, 10, 10)), -np.ones((1, 10, 10))]))
    assert valid.tolist() == [False, False] and not targets.any()
    with pytest.raises(ValueError):
        redistance_batch(np.full((1, 1, 5, 5), np.nan))


def test_zero_haze_does_not_count_as_content_success():
    blank = ink_metrics(np.zeros((40, 100), np.float32))
    real = ink_metrics(fixture())
    assert blank["blank"] and blank["core_pixels"] == 0
    assert not real["blank"] and real["components"] == 4
    assert blank["not_content_or_author_style_score"]
