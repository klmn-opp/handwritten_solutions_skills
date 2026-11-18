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


def test_metrics_report_copying_and_no_variation_with_serializable_numbers():
    import json

    from research.corpus.reference_evaluate import character_metrics, shape_distance

    ink = np.zeros((96, 96), np.float32)
    ink[20:70, 40:44] = 1
    outputs = np.stack([ink, ink])
    result = character_metrics(outputs, ink[None], np.array([0]), 0)
    assert result["exact_font_mask_copies"] == 2
    assert result["mean_pairwise_ink_xor"] == 0
    assert result["unique_binary_outputs"] == 1
    assert result["blank_count"] == result["border_count"] == 0
    json.dumps(result, allow_nan=False)
    shifted = np.roll(ink, 3, axis=1)
    assert shape_distance(ink, shifted) == shape_distance(shifted, ink)


def test_sample_delivery_pdf_and_provenance(tmp_path):
    import json

    from pypdf import PdfReader

    from research.corpus.reference_evaluate import panels

    for directory in ["rendered/pages", "reports", "comparisons"]:
        (tmp_path / directory).mkdir(parents=True)
    reference = np.zeros((1, 96, 96), np.float32)
    reference[:, 20:80, 45:50] = 1
    panels(tmp_path, ["1"], reference, np.repeat(reference[:, None], 2, axis=1))
    pdf = PdfReader(tmp_path / "rendered/homework_solutions.pdf")
    assert len(pdf.pages) == 1
    assert abs(float(pdf.pages[0].mediabox.width) - 595.276) < 0.2
    provenance = json.loads((tmp_path / "reports/glyph-provenance.json").read_text())
    assert len(provenance) == 2
    assert all(not p["is_real_personal_sample"] for p in provenance)
