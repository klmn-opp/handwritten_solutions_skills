"""Pixel provenance and whole-word geometry, without importing Torch."""

import numpy as np
from PIL import Image

from research.corpus.word_style_data import normalize, source_box


def test_inverse_coordinates_and_margin():
    alignment = {
        "source_box_coordinate_system": "whole-view",
        "original_whole_transform": {
            "ink_bbox": [10, 5, 90, 45],
            "translation": [100, 20],
            "scale": 2,
        },
    }
    assert source_box({"source_box_in_whole_view": [100, 20, 160, 60]}, alignment, (100, 60)) == [
        8,
        3,
        42,
        27,
    ]


def test_original_crop_coordinates_not_scaled_twice():
    assert source_box(
        {"source_box_in_whole_view": [1, 4, 20, 30]},
        {"source_box_coordinate_system": "original-crop"},
        (24, 32),
    ) == [0, 2, 22, 32]


def test_complete_descender_and_dot_survive_uniform_word_fit():
    a = np.full((80, 240), 255, np.uint8)
    a[25:55, 20:210] = 20
    a[50:72, 180:184] = 20
    a[10:13, 70:74] = 20
    out, meta = normalize(Image.fromarray(a))
    assert meta["bbox"] == [20, 10, 210, 72]
    assert not meta["target_geometry_used_for_reference"]
    # Both detached high dot and low descender retain dark pixels.
    dx, dy = meta["translation"]
    s = meta["scale"]
    assert out[dy : dy + max(1, round(4 * s)), dx:].max() > 0.8
    assert out[dy + round(55 * s) :, dx:].max() > 0.8
    assert not out[:8].any() and not out[-8:].any()
