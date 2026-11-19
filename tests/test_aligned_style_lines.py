import numpy as np

from research.corpus.aligned_style_lines import restore_word


def test_inverse_reference_transform_preserves_generated_shift_and_detached_dot():
    transform = {"scale": 2, "bbox": [5, 6, 45, 36], "translation": [20, 20]}
    ink = np.zeros((128, 512), np.float32)
    ink[20:26, 20:26] = 0.8
    ink[70:78, 60:66] = 1
    first = restore_word(ink, transform, 10, (200, 180))
    shifted = restore_word(np.roll(ink, 4, axis=1), transform, 10, (200, 180))
    y, x = np.where(first > 100)
    yy, xx = np.where(shifted > 100)
    assert y.min() == yy.min() and y.max() == yy.max()
    assert xx.min() - x.min() == 2 and xx.max() - x.max() == 2
    assert 77 <= x.min() <= 80 and 68 <= y.min() <= 71
    assert first[68:74, 77:83].max() > 180
    assert first[93:100, 97:104].max() > 180
    assert not first[:64].any() and not first[:, :64].any()
