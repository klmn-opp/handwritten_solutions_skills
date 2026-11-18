"""Synthetic, temporary fixtures for symbol lineage and authorized model-label admission."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from research.corpus.dataset import export_dataset
from research.corpus.ingest import import_file, png
from research.corpus.label_admission import adopt, authorize, label_text
from research.corpus.store import Corpus
from research.corpus.symbols import isolated_boxes, prepare, publish
from research.corpus.train import validate_dataset


@pytest.fixture
def data(tmp_path):
    im = Image.new("RGB", (120, 100), "white")
    d = ImageDraw.Draw(im)
    d.line((25, 30, 25, 65), fill="black", width=4)
    d.ellipse((65, 30, 82, 65), outline="black", width=3)
    path = tmp_path / "fixture.png"
    im.save(path)
    c = Corpus(tmp_path / "corpus")
    doc = import_file(c, path, course="test-fixture", batch="fixture")
    page = c.rows("page")[0]
    return c, doc, page


def selection(c, page, box=None):
    box = box or [19, 24, 34, 72]
    with Image.open(c.path(page["color_image"])) as im:
        sha = hashlib.sha256(png(im.crop(box))).hexdigest()
    return dict(
        page_id=page["page_id"],
        bbox=box,
        source_image_sha256=Path(page["color_image"]).stem,
        crop_sha256=sha,
        candidate_label="1",
        decision="accept",
        single_symbol=True,
        complete_crop=True,
        writer_personal=True,
        privacy_safe=True,
        transcription_confidence="high",
        ambiguity_flags=[],
        quality_flags=[],
    )


def test_proposals_preserve_pixels_and_do_not_inherit_review(data, tmp_path):
    c, _, page = data
    parent = c.propose(
        page["page_id"],
        [5, 5, 110, 95],
        content_type="math-expression",
        label="10",
        role="personal-handwriting",
    )
    attestation = dict(
        human_review=True, actual_transcription=True, crop_complete=True, writer_checked=True
    )
    parent = c.review(
        parent["sample_id"],
        0,
        dict(review_status="verified", quality_status="clean", privacy_checked=True),
        "fixture-human",
        attestation,
    )
    result = prepare(c, [parent["sample_id"]], tmp_path / "proposals")
    assert len(result["proposals"]) == 2
    proposal = min(result["proposals"], key=lambda x: x["bbox"][0])
    proposal["candidate_label"] = "1"
    result = publish(c, [proposal], batch="fixture", annotator="fixture-model")
    child = result["created"][0]
    assert child["review_status"] == "auto-candidate" and child["verified_text"] is None
    assert child["parents"] == [parent["sample_id"]]
    assert child["derivation"]["parent_revision"] == parent["revision"]
    assert child["crop_sha256"] == proposal["crop_sha256"]
    assert c.get("sample", parent["sample_id"]) == parent
    assert (
        publish(c, [proposal], batch="again", annotator="fixture")["skipped"][0]["reason"]
        == "same-physical-ink"
    )
    c.review(parent["sample_id"], 1, {"review_status": "rejected"}, "fixture-human")
    with pytest.raises(ValueError, match="Parent review changed"):
        publish(c, [proposal], batch="stale", annotator="fixture")


def test_detector_avoids_dotted_stem_and_long_rule():
    im = Image.new("RGB", (100, 100), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((20, 30, 24, 60), fill="black")
    d.rectangle((20, 19, 24, 23), fill="black")
    d.rectangle((50, 80, 90, 82), fill="black")
    boxes, diagnostics = isolated_boxes(im)
    assert not boxes and diagnostics["excluded"]["detached-companion-or-stack"] == 1


def test_modified_pixels_and_parent_escape_rejected_before_publication(data):
    c, _, page = data
    row = selection(c, page)
    with pytest.raises(ValueError, match="unchanged original"):
        publish(c, [row, {**row, "crop_sha256": "bad"}], batch="fixture", annotator="fixture")
    assert c.rows("sample") == []


def test_explicit_model_policy_exports_without_claiming_human_review(data, tmp_path):
    c, doc, page = data
    row = selection(c, page)
    sample = publish(c, [row], batch="fixture", annotator="gpt-6.2-sol/xhigh")["created"][0]
    evidence = tmp_path / "sol.json"
    evidence.write_text(json.dumps([row]))
    with pytest.raises(KeyError):
        adopt(c, [sample["sample_id"]], evidence)
    authorize(
        c,
        instruction="Test fixture: explicitly allow clean model labels",
        documents=[doc["document_id"]],
    )
    accepted = adopt(c, [sample["sample_id"]], evidence)[0]
    assert accepted["review_status"] == "model-accepted"
    assert accepted["review_attestation"] is None and accepted["verified_text"] is None
    assert accepted["model_review"]["human_review"] is False and label_text(accepted) == "1"
    dest = tmp_path / "dataset"
    report = export_dataset(c, dest)
    assert report["model_accepted_samples"] == 1 and report["human_reviewed_samples"] == 0
    assert report["human_verified_ratio"] == 0
    rows, gate = validate_dataset(dest)
    assert len(rows) == 1 and gate["decision"] == "NO-GO"
    evidence.write_text("[]")
    with pytest.raises(ValueError, match="evidence checksum"):
        validate_dataset(dest)


@pytest.mark.parametrize(
    "bad",
    [
        {"quality_flags": ["occluded"]},
        {"ambiguity_flags": ["1-or-l"]},
        {"single_symbol": False},
        {"transcription_confidence": "medium"},
        {"writer_personal": False},
        {"privacy_safe": False},
    ],
)
def test_uncertain_or_damaged_model_decisions_never_admitted(data, tmp_path, bad):
    c, doc, page = data
    row = {**selection(c, page), **bad}
    sample = publish(c, [row], batch="fixture", annotator="fixture")["created"][0]
    authorize(c, instruction="Test authorization", documents=[doc["document_id"]])
    evidence = tmp_path / "bad.json"
    evidence.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="cannot be admitted"):
        adopt(c, [sample["sample_id"]], evidence)
    assert c.get("sample", sample["sample_id"]) == sample
    with pytest.raises(ValueError, match="explicit policy"):
        c.review(sample["sample_id"], 0, {"review_status": "model-accepted"}, "fixture")
