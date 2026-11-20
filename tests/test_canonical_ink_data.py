import numpy as np
import pytest

from research.corpus.canonical_ink_full import combine_targets


def test_real_classes_never_receive_contradictory_font_targets():
    real = {
        "labels": np.array([102, 102]),
        "ink": np.ones((2, 4, 4)),
        "local_codes": np.ones((2, 8), np.float32),
    }
    font = {"tokens": np.array([[[102, 0, 0]], [[120, 0, 0]]]), "ink": np.zeros((2, 4, 4))}
    out = combine_targets(real, font)
    assert out["labels"].tolist() == [102, 102, 120]
    assert out["real"].tolist() == [True, True, False]
    np.testing.assert_array_equal(out["ink"][:2], real["ink"])
    assert np.count_nonzero(out["codes"][2]) == 0
    assert out["source_indices"].tolist() == [0, 1, 1]
    font["tokens"][1, 0, 0] = 102
    with pytest.raises(ValueError):
        combine_targets(real, font)
