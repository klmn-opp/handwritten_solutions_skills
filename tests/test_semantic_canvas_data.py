import numpy as np
import pytest

from research.corpus.semantic_canvas_real_data import match_conditions


def test_intact_word_does_not_borrow_a_different_samples_glyph_label_or_condition():
    word = {"text": "ff", "crop_sha256": "actual-word"}
    tokens = np.array([[102, 0, 102], [102, 102, 0], [0, 0, 0]])
    rows = [
        {"source_crop_sha256": "actual-word", "position": 0, "label": "f"},
        {"source_crop_sha256": "another-word", "position": 1, "label": "f"},
    ]
    style, matches = match_conditions(word, tokens, rows, np.ones((2, 8)))
    assert matches == [{"position": 0, "glyph_index": 0}]
    assert np.array_equal(style[0], np.ones(8))
    assert not style[1:].any()


def test_duplicate_provenance_and_disagreeing_semantics_cannot_enter_training():
    word = {"text": "f", "crop_sha256": "actual-word"}
    tokens = np.array([[102, 0, 0]])
    row = {"source_crop_sha256": "actual-word", "position": 0, "label": "f"}
    with pytest.raises(ValueError, match="Unique"):
        match_conditions(word, tokens, [row, row], np.zeros((2, 8)))
    with pytest.raises(ValueError, match="label disagrees"):
        match_conditions(word, tokens, [{**row, "label": "t"}], np.zeros((1, 8)))
    with pytest.raises(ValueError, match="Semantic input"):
        match_conditions(word, np.array([[116, 0, 0]]), [row], np.zeros((1, 8)))
