import json

import numpy as np
import pytest
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run
from research.corpus import reference_test


def test_holdout_never_opens_without_matching_freeze(tmp_path, monkeypatch):
    sample = Run.create("sample-fixture", workspace=tmp_path)
    atomic_json(sample.path / "reports/request.json", {"checkpoint_sha256": "frozen-model"})
    sample.finish("held")

    def must_not_open(*args):
        raise AssertionError("Holdout opened before freeze verification")

    monkeypatch.setattr(reference_test, "validate_package", must_not_open)
    with pytest.raises(ValueError, match="Freeze candidates"):
        reference_test.evaluate(sample.path, tmp_path, split="holdout")
    frozen = Run.create("freeze-fixture", workspace=tmp_path)
    atomic_json(frozen.path / "reports/freeze.json", {"candidates": []})
    frozen.finish("held")
    with pytest.raises(ValueError, match="not in frozen"):
        reference_test.evaluate(sample.path, tmp_path, split="holdout", frozen=frozen.path)


@pytest.mark.parametrize("dpi", [150, 300])
def test_external_crops_must_match_pdf_and_split(tmp_path, monkeypatch, dpi):
    import pypdfium2 as pdfium

    pdf = tmp_path / "original.pdf"
    ink = np.full((100, 100, 3), 255, np.uint8)
    ink[25:75, 45:50] = 0
    Image.fromarray(ink).save(pdf, resolution=dpi)
    doc = pdfium.PdfDocument(pdf)
    page = doc[0]
    raster = page.render(scale=dpi / 72).to_pil().convert("RGB")
    page.close()
    doc.close()
    original = tmp_path / "page.png"
    crop = tmp_path / "crop.png"
    raster.save(original)
    raster.crop((40, 20, 55, 80)).save(crop)
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps(
            {
                "source_hash": sha256(pdf),
                "dataset_split": "validation",
                "crop_sha256": "old-unrelated",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(reference_test, "verify_run", lambda *args: {})
    monkeypatch.setattr(
        reference_test,
        "validate_views",
        lambda *args: {"source_files": {str(samples): sha256(samples)}},
    )
    (tmp_path / "reports").mkdir()
    row = {
        "split": "validation",
        "source_pdf_sha256": sha256(pdf),
        "decision": "accept",
        "ambiguity_flags": [],
        "quality_flags": [],
        "single_symbol": True,
        "complete_crop": True,
        "writer_personal": True,
        "privacy_safe": True,
        "transcription_confidence": "high",
        "annotation_model_id": "gpt-6.2-sol",
        "annotation_reasoning_effort": "xhigh",
        "human_review": False,
        "candidate_label": "1",
        "render_dpi": dpi,
        "source_pdf_path": str(pdf),
        "page_image_path": str(original),
        "source_image_sha256": sha256(original),
        "crop_path": str(crop),
        "crop_sha256": sha256(crop),
        "source_pdf_page_number": 1,
        "bbox": [40, 20, 55, 80],
    }
    atomic_json(tmp_path / "reports/proposals.json", [row])
    assert len(reference_test.validate_package(tmp_path, tmp_path, "validation")) == 1
    with pytest.raises(ValueError, match="incorrectly split"):
        reference_test.validate_package(tmp_path, tmp_path, "holdout")
    Image.new("RGB", (15, 60), "white").save(crop)
    row["crop_sha256"] = sha256(crop)
    atomic_json(tmp_path / "reports/proposals.json", [row])
    with pytest.raises(ValueError, match="retouched"):
        reference_test.validate_package(tmp_path, tmp_path, "validation")
