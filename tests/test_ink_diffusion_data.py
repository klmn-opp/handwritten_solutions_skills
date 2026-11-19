import numpy as np
import pytest
from PIL import Image

from hwsol.profiles.io import sha256
from research.corpus.ink_diffusion_data import SEALED_C, admit, signed_field


def test_real_distance_field_keeps_dot_hole_and_descender():
    ink = np.zeros((128, 512), np.float32)
    ink[10:13, 20:23] = 1
    ink[30:60, 20:40] = 1
    ink[34:56, 24:36] = 0
    ink[55:110, 38:41] = 1
    field = signed_field(ink)
    assert np.array_equal(field > 0, ink > 0.5)
    assert field[11, 21] > 0 and field[45, 30] < 0 and field[108, 39] > 0


def test_uncertain_labels_sealed_c_and_invalid_boxes_are_rejected(tmp_path):
    p = tmp_path / "source.png"
    Image.new("RGB", (100, 50), "white").save(p)
    row = {
        "text": "ff",
        "source_crop": str(p),
        "source_crop_sha256": sha256(p),
        "source_hash": "train-doc",
        "dataset_split": "train",
        "decision": "accept",
        "confidence": "high",
        "box_in_source_crop": [1, 1, 99, 49],
    }
    admit(row)
    for change in [
        {"confidence": "low"},
        {"source_hash": SEALED_C},
        {"dataset_split": "holdout"},
        {"box_in_source_crop": [0, 0, 101, 20]},
        {"source_crop_sha256": "changed"},
        {"text": "f" * 33},
    ]:
        with pytest.raises(ValueError):
            admit({**row, **change})


def test_empty_region_is_never_a_training_target():
    for ink in [np.zeros((128, 512)), np.ones((128, 512)), np.zeros((64, 256))]:
        with pytest.raises(ValueError):
            signed_field(ink)
