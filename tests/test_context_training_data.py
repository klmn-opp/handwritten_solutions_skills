import numpy as np
from PIL import Image

from research.corpus.context_training_data import whole_ink


def test_whole_region_keeps_descender_and_dots_without_character_splitting():
    image = np.full((100, 250, 3), 255, dtype=np.uint8)
    image[20:60, 20:30] = 0
    image[55:80, 30:40] = 0
    image[10:13, 22:25] = 0
    image[20:60, 70:80] = 0
    result, info = whole_ink(Image.fromarray(image))
    assert result.shape == (64, 512)
    assert info["ink_bbox"] == [20, 10, 80, 80]
    assert not info["character_segmentation"] and not info["stroke_repair"]
    assert not (result[:2] > 0.5).any() and not (result[-2:] > 0.5).any()


def test_whole_region_uniformly_fits_long_ink_without_truncation():
    image = np.full((30, 1800, 3), 255, dtype=np.uint8)
    image[8:22, 10:1790] = 0
    result, info = whole_ink(Image.fromarray(image))
    assert info["scale"] == 480 / 1780
    assert (result[:, 16:496] > 0.5).any()
    assert not (result[:, :10] > 0.5).any() and not (result[:, -10:] > 0.5).any()
