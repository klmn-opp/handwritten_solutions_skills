import numpy as np
import pytest
from PIL import Image

from research.corpus.stroke_opacity import recognition_crop, supervision_masks


def test_one_pixel_marks_and_descenders_are_not_eroded():
    ink = np.zeros((1, 1, 32, 80), np.float32)
    ink[0, 0, 4, 10] = 1
    ink[0, 0, 10:29, 10] = 1
    ink[0, 0, 25, 50] = 1
    before = ink.copy()
    masks = supervision_masks(ink)
    assert np.array_equal(ink, before)
    assert np.array_equal(masks["core"], ink)
    assert np.all(masks["component"][ink > 0] > 0)
    assert masks["component"].max() <= 8
    assert not masks["background"][0, 0, 4, 12]
    assert masks["background"][0, 0, 4, 14]


def test_invalid_and_blank_targets_do_not_invent_foreground():
    masks = supervision_masks(np.zeros((2, 1, 16, 32), np.float32))
    assert not masks["core"].any() and not masks["component"].any()
    assert masks["background"].all()
    for bad in [np.ones((1, 16, 32)), np.full((1, 1, 16, 32), np.nan), np.full((1, 1, 16, 32), 2)]:
        with pytest.raises(ValueError):
            supervision_masks(bad)


def test_recognition_view_copies_all_visible_ink_without_label_or_source_change():
    a = np.full((64, 512), 255, np.uint8)
    a[10:45, 200:250] = 0
    a[7, 255] = 220  # retain faint punctuation at the boundary
    im = Image.fromarray(a)
    view, info = recognition_crop(im)
    assert np.array_equal(np.asarray(im), a)
    assert info["bbox"] == [200, 7, 256, 45]
    assert np.array_equal(np.asarray(view.convert("L"))[8:-8, 8:-8], a[7:45, 200:256])
    blank, receipt = recognition_crop(Image.new("L", (64, 64), 255))
    assert receipt["blank"] and blank.size == (64, 64)
