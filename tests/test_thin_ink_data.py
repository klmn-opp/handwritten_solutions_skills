import copy

import numpy as np
import pytest

from research.corpus.thin_ink_data import component_weights, token_positions


def test_positions_use_only_font_coordinates_and_reject_wrong_semantics():
    row = {
        "text": "if",
        "reference_glyphs": {
            "crop_from_pdf": [100, 10, 180, 50],
            "glyph_provenance": [
                {"semantic": "i", "bounds_px": [110, 20, 120, 40]},
                {"semantic": "f", "bounds_px": [130, 20, 150, 40]},
            ],
        },
        "reference_transform": {"scale": 2, "bbox": [10, 0, 50, 40], "translation": [200, 10]},
    }
    actual = token_positions(row)
    assert actual[0, 0] == pytest.approx(210 / 512)
    assert actual[1, 0] == pytest.approx(260 / 512)
    changed = copy.deepcopy(row)
    changed["target_transform"] = {"scale": 99, "translation": [-999, 999]}
    assert np.array_equal(actual, token_positions(changed))
    changed["text"] = "fi"
    with pytest.raises(ValueError, match="semantics"):
        token_positions(changed)


def test_small_components_gain_weight_without_inventing_ink_or_amplifying_single_noise():
    ink = np.zeros((128, 512), np.float32)
    ink[5:8, 20:23] = 1
    ink[30:80, 20:25] = 1
    ink[120, 500] = 1
    original = ink.copy()
    weights = component_weights(ink)
    assert np.array_equal(ink, original)
    assert weights[6, 21] > weights[40, 21]
    assert weights[120, 500] == 1
    assert not weights[ink == 0].any()
    assert weights.max() <= 6
    with pytest.raises(ValueError, match="Blank"):
        component_weights(np.zeros_like(ink))


def test_canonical_component_balance_preserves_background_and_single_pixel_noise():
    ink = np.zeros((96, 96), np.float32)
    ink[10:12, 30:35] = 1
    ink[30:80, 30:35] = 1
    ink[90, 90] = 1
    original = ink.copy()
    weights = component_weights(ink, expected_shape=(96, 96), balance_to_largest=True)
    assert weights[10, 31] == 5
    assert weights[40, 31] == 1
    assert weights[90, 90] == 1
    assert not weights[ink == 0].any()
    np.testing.assert_array_equal(ink, original)
