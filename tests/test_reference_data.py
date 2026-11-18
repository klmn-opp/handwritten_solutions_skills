from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.reference_data import align_real, ink_box, signed_distance, validate_views


def test_distance_keeps_disconnected_dot_and_descender():
    ink = np.zeros((96, 96), np.float32)
    ink[12:16, 48:52] = 1
    ink[24:83, 47:51] = 1
    sdf = signed_distance(ink)
    assert np.array_equal(sdf > 0, ink > 0.5)
    assert sdf[82, 48] > 0 and sdf[20, 48] < 0
    with pytest.raises(ValueError, match="Empty"):
        signed_distance(np.zeros_like(ink))


def test_alignment_uniformly_keeps_full_structure(tmp_path):
    raw = np.full((80, 30), 255, np.uint8)
    raw[5:8, 15:18] = 0
    raw[15:75, 13:17] = 0
    path = tmp_path / "j.png"
    Image.fromarray(raw).save(path)
    reference = np.zeros((96, 96), np.float32)
    reference[15:19, 48:52] = 1
    reference[25:84, 46:50] = 1
    ink, transform = align_real(path, reference)
    assert ink_box(ink)[3] >= 82
    assert ink[15:20].max() > 0.5
    assert transform["no_stroke_repair"]
    assert raw.shape[::-1] == tuple(transform["original_size"])


def test_views_fail_closed_on_holdout_or_changed_files(tmp_path):
    atomic_json(tmp_path / "records.json", [{"dataset_split": "holdout"}])
    manifest = {
        "schema": "reference-glyph-views/1",
        "holdout_in_arrays": False,
        "files": {"records.json": sha256(tmp_path / "records.json")},
        "source_files": {},
    }
    atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Evaluation sample"):
        validate_views(tmp_path)
    (tmp_path / "records.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        validate_views(tmp_path)


def test_all_character_training_stays_outside_runtime():
    source = Path("pyproject.toml").read_text()
    dependencies = source.split("[project.optional-dependencies]")[0]
    assert '"torch' not in dependencies
    source = Path("research/corpus/reference_study.py").read_text()
    assert "deliverable=True" not in source
    config = __import__(
        "research.corpus.reference_study", fromlist=["DEFAULT_CONFIG"]
    ).DEFAULT_CONFIG
    assert not config["holdout_used_for_selection"]
    assert not config["production_eligible"]
