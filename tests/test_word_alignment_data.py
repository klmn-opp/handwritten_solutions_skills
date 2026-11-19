import numpy as np

from research.corpus.whole_word_alignment_data import initial_geometry


def test_whole_frame_estimate_preserves_ink_and_glyph_order():
    ink = np.zeros((128, 512), np.float32)
    ink[20:80, 100:200] = 1
    before = ink.copy()
    boxes = np.array([[0, 0, 10, 20], [11, 0, 20, 30]], np.float32)
    result = initial_geometry(boxes, ink)
    np.testing.assert_array_equal(ink, before)
    np.testing.assert_allclose(result[:, :2].min(0), [100, 20])
    np.testing.assert_allclose(result[:, 2:].max(0), [200, 80])
    assert result[0, 0] < result[1, 0]
    assert result[0, 3] < result[1, 3]
