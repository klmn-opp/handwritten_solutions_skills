import numpy as np
import pytest

from research.corpus.ink_integrity import (
    coordinates,
    effective_scale,
    field_metrics,
    map_components,
    render_word,
)


def test_identity_has_exact_coordinates_and_preserves_dot_hole_descender():
    parts = map_components(np.zeros((2, 5, 17)), guarded=True)
    x, y = np.meshgrid(np.arange(-10, 20), np.arange(-4, 15))
    sx, sy = coordinates(parts, x, y)
    assert np.allclose(sx, x) and np.allclose(sy, y)
    source = np.zeros((128, 512), np.float32)
    source[20:30, 40:48] = 1
    source[45:85, 100:140] = 1
    source[53:77, 108:132] = 0
    source[80:115, 135:140] = 1
    tr = {"bbox": [0, 0, 512, 128], "scale": 1, "translation": [0, 0]}
    out = render_word(source, tr, parts, raster_scale=1, supersampling=1)
    assert np.allclose(out[32:160, 64:576], source)
    assert not out[:32].any() and not out[:, :64].any()


def test_guard_bounds_positive_local_scale_and_shear_without_flattening_variation():
    raw = np.zeros((2, 5, 17))
    nodes = np.linspace(0, 511, 17)
    intervals = np.tile([0.25, 1.75], 8)
    raw[0] = np.r_[0, np.cumsum(intervals * 511 / 16)] - nodes
    raw[1] = (
        np.linspace(-63.5, 63.5, 5)[:, None] * np.linspace(-0.2, 0.2, 17)[None]
        + np.sin(np.arange(17)) * 8
    )
    before = field_metrics(map_components(raw, guarded=False))
    parts = map_components(raw, guarded=True)
    after = field_metrics(parts)
    assert before["horizontal_min"] < 0.8
    assert after["horizontal_min"] >= 0.8 - 1e-10 and after["horizontal_max"] <= 1.25 + 1e-10
    assert after["shear_bound"] <= 0.06 + 63.5 * 0.0015 + 1e-10
    assert after["displacement_rms"] > 0
    assert abs(parts["x"][-1] - 511) < 1e-9


def test_padding_recovers_shifted_ink_without_inventing_missing_structure():
    raw = np.zeros((2, 5, 17))
    raw[1] = 8
    parts = map_components(raw, guarded=False)
    source = np.zeros((128, 512), np.float32)
    source[2:10, 80:90] = 1
    out = render_word(
        source,
        {"bbox": [0, 0, 512, 128], "scale": 1, "translation": [0, 0]},
        parts,
        raster_scale=1,
        supersampling=1,
    )
    assert out[:32].sum() == 60  # six shifted rows that the old 128px canvas discards
    assert out.sum() == 80
    assert not out[:2].any() and not out[-2:].any()


def test_effective_resize_scale_and_reject_folded_or_nontriangular_map():
    assert effective_scale({"bbox": [0, 0, 33, 17], "scale": 1.7}) == (56 / 33, 29 / 17)
    bad = np.zeros((2, 5, 17))
    bad[0, 2, 4] = 1
    with pytest.raises(ValueError, match="column-wise"):
        map_components(bad, guarded=True)
    bad = np.zeros((2, 5, 17))
    bad[0, :, 4] = 100
    with pytest.raises(ValueError, match="Folded"):
        map_components(bad, guarded=True)
