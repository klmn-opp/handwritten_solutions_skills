import numpy as np
import pytest

from research.corpus.semantic_anchors import validate


def test_anchor_admission_rejects_wrong_identity_and_missing_descender():
    ink = np.zeros((128, 512), np.float32)
    ink[20:70, 30:40] = 1
    review = {
        "status": "accept",
        "text": "p",
        "glyphs": [{"position": 0, "char": "p", "bbox": [30, 20, 40, 70]}],
    }
    before = ink.copy()
    assert validate(review, "p", ink).tolist() == [[30, 20, 40, 70]]
    assert np.array_equal(ink, before)
    review["glyphs"][0]["bbox"][3] = 50
    with pytest.raises(ValueError, match="omit strong"):
        validate(review, "p", ink)
    review["glyphs"][0]["bbox"][3] = 70
    review["glyphs"][0]["char"] = "q"
    with pytest.raises(ValueError, match="identity"):
        validate(review, "p", ink)


def test_connected_uncertain_words_cannot_silently_become_anchors():
    with pytest.raises(ValueError, match="Only accepted"):
        validate({"status": "uncertain", "text": "on"}, "on", np.ones((128, 512)))
    with pytest.raises(ValueError, match="Every character"):
        validate({"status": "accept", "text": "on", "glyphs": []}, "on", np.ones((128, 512)))
