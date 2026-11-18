"""Sealed evaluation inputs cannot quietly become inaccurate or duplicated crops."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from hwsol.profiles.io import sha256
from research.corpus.study_evaluation import pairwise_ink_distance, validate_new_test


@pytest.fixture
def sealed_test(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("research.corpus.study_evaluation.verify_run", lambda _: None)
    source = tmp_path / "source_pdf/source_pdf_phy1.pdf"
    source.parent.mkdir()
    source.write_bytes(b"synthetic test source, not a real PDF")
    root = tmp_path / "test-run"
    (root / "reports").mkdir(parents=True)
    image = Image.new("RGB", (80, 80), "white")
    ImageDraw.Draw(image).line((28, 20, 30, 65), width=3, fill="black")
    page, crop = root / "page.png", root / "crop.png"
    image.save(page)
    box = [20, 15, 40, 70]
    image.crop(box).save(crop)
    row = dict(
        candidate_label="1",
        split="holdout",
        decision="accept",
        quality_flags=[],
        ambiguity_flags=[],
        transcription_confidence="high",
        human_review=False,
        annotation_model_id="gpt-6.2-sol",
        annotation_reasoning_effort="xhigh",
        single_symbol=True,
        complete_crop=True,
        writer_personal=True,
        privacy_safe=True,
        crop_sha256=sha256(crop),
        source_image_sha256=sha256(page),
        crop_path=str(crop),
        page_image_path=str(page),
        bbox=box,
        source_pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return root, row


def write_rows(root, rows):
    (root / "reports/proposals.json").write_text(json.dumps(rows))


def test_test_pixels_checked_even_when_new_crop_hash_matches(sealed_test):
    root, row = sealed_test
    write_rows(root, [row])
    assert len(validate_new_test(root)) == 1
    Image.new("RGB", (20, 55), "white").save(row["crop_path"])
    row["crop_sha256"] = sha256(Path(row["crop_path"]))
    write_rows(root, [row])
    with pytest.raises(ValueError, match="original page"):
        validate_new_test(root)


@pytest.mark.parametrize(
    "change",
    [
        {"ambiguity_flags": ["uncertain identity"]},
        {"quality_flags": ["occluded"]},
        {"split": "train"},
    ],
)
def test_uncertain_or_training_samples_cannot_be_final_test(sealed_test, change):
    root, row = sealed_test
    row.update(change)
    write_rows(root, [row])
    with pytest.raises(ValueError, match="Unadmitted"):
        validate_new_test(root)


def test_repeated_same_crop_is_rejected(sealed_test):
    root, row = sealed_test
    write_rows(root, [row, row])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_new_test(root)


def test_shifted_overlapping_crop_cannot_inflate_sample_count(sealed_test):
    root, row = sealed_test
    other = {**row, "bbox": [21, 16, 41, 71], "crop_path": str(root / "shifted.png")}
    with Image.open(row["page_image_path"]) as page:
        page.crop(other["bbox"]).save(other["crop_path"])
    other["crop_sha256"] = sha256(Path(other["crop_path"]))
    write_rows(root, [row, other])
    with pytest.raises(ValueError, match="Overlapping physical"):
        validate_new_test(root)


def test_pairwise_metric_is_symmetric_and_counts_each_pair_once():
    images = np.array([np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 0.5)])
    assert pairwise_ink_distance(images) == pytest.approx(10 / 3)
    assert pairwise_ink_distance(images[[2, 0, 1]]) == pytest.approx(10 / 3)
    assert pairwise_ink_distance(images[:1]) == 0
