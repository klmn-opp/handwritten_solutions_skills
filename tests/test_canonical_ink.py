import numpy as np
import pytest

from research.corpus.canonical_ink import canonicalize, coordinate_maps


def test_identity_preserves_real_pixels_exactly():
    ink = np.zeros((128, 512), np.float32)
    ink[30:90, 100:104] = 0.9
    ink[35:39, 100:150] = 1
    out, report = canonicalize(ink, np.zeros((2, 5, 17), np.float32))
    assert np.array_equal(ink, out) and report["roundtrip_dice"] == 1 and report["eligible"]


def test_folded_and_nontriangular_maps_are_rejected():
    control = np.zeros((2, 5, 17), np.float32)
    control[0, :, 8] = -100
    with pytest.raises(ValueError, match="Folded"):
        coordinate_maps(control)
    control[:] = 0
    control[0, 2, 2] = 1
    with pytest.raises(ValueError, match="Horizontal"):
        coordinate_maps(control)


def test_cropping_is_never_silently_admitted():
    ink = np.zeros((128, 512), np.float32)
    ink[30:90, 100:104] = 1
    control = np.zeros((2, 5, 17), np.float32)
    control[1] = 128
    _, report = canonicalize(ink, control)
    assert not report["eligible"] and report["source_ink_outside_map"] > 0


def test_guard_reuses_bounded_geometry_before_real_resampling():
    control = np.zeros((2, 5, 17), np.float32)
    control[0] = np.sin(np.linspace(0, np.pi, 17)) * 60
    fx, fy, _, _ = coordinate_maps(control, guarded=True)
    assert np.diff(fx, axis=1).min() >= 0.7999 and np.diff(fx, axis=1).max() <= 1.2501
    assert np.diff(fy, axis=0).min() >= 0.7999
