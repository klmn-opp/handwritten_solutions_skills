"""Training-only ink views must not modify or invent source pixels."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from research.corpus.training_data import ink_view


def test_dark_ink_view_preserves_raw_and_suppresses_light_grid(tmp_path):
    values = np.full((64, 64), 255, dtype=np.uint8)
    values[16, :] = 215  # Light paper rule.
    values[8:56, 30:34] = 35  # Strong handwritten stroke.
    values[8:56, 29] = 120  # Soft real edge remains fractional.
    path = tmp_path / "source.png"
    Image.fromarray(values).save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    raw, ink = ink_view(path), ink_view(path, "dark-ink-v1")
    assert raw[16, 0] > 0 and ink[16, 0] == 0
    assert ink[30, 31] == 1 and 0 < ink[30, 29] < 1
    assert ink[0, 0] == 0 and ink.shape == (64, 64)
    assert np.all(ink[values >= 190] == 0)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with pytest.raises(ValueError, match="preprocessing"):
        ink_view(path, "unversioned-filter")
