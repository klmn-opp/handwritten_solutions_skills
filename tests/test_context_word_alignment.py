from pathlib import Path

import cv2
import numpy as np
import pytest
from matplotlib import get_data_path

from research.corpus.context_training_data import reference_image
from research.corpus.context_word_alignment import aligned_view, natural_words


def test_word_alignment_refuses_connected_or_ambiguous_splits():
    ink = np.zeros((64, 512), np.float32)
    ink[15:45, 10:150] = 1
    with pytest.raises(ValueError, match="gaps"):
        natural_words(ink, ["one", "two"])
    ink = np.zeros((64, 512), np.float32)
    for x in [10, 30, 50]:
        ink[15:45, x : x + 10] = 1
    with pytest.raises(ValueError, match="ambiguous"):
        natural_words(ink, ["one", "two"])


def test_word_alignment_keeps_punctuation_and_descenders_without_modifying_source():
    font = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    raw = reference_image("gypq.", font)
    before = np.asarray(raw).copy()
    ref, target, meta = aligned_view("gypq.", raw, font)
    assert np.array_equal(before, np.asarray(raw))
    assert (target > 0.5).sum() > 100 and target.shape == ref.shape == (64, 512)
    assert not meta["character_segmentation"] and not meta["ink_repair"]
    assert len(meta["words"]) == 1
    mask = target > 0.3
    _, _, components, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    components = sorted(components[1:], key=lambda item: item[0])
    assert len(components) == 5
    assert components[-1][3] < min(item[3] for item in components[:-1])
    assert (
        max(item[1] + item[3] for item in components[:-1]) > components[-1][1] + components[-1][3]
    )
    assert meta["derived_view_not_original_sample"]
