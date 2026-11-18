"""Recognizer metric and holdout contracts; no synthetic output counts as inference."""

import pytest

from research.english.corpus import load
from research.english.metrics import aggregate, score


def test_versioned_english_inventory():
    rows, digest = load("all")
    assert len(rows) == 120 and len(digest) == 64
    assert len(load("holdout")[0]) == 24 and len(load("development")[0]) == 96
    assert len(load("smoke")[0]) == 16
    assert len({r["category"] for r in rows}) == 8
    assert all(not r["allow_training"] for r in rows)
    for purpose in ["diagnostic", "training", "tuning"]:
        with pytest.raises(ValueError):
            load("holdout", purpose=purpose)


def test_recognizer_edits_and_repetition_are_not_truth():
    metric = score("Read the value.", "Read the the value.")
    assert metric["word_errors"]["insertion"] == 1
    assert metric["repeated_inserted_words"] == 1
    assert score("very very clear", "very very clear")["repeated_inserted_words"] == 0
    assert "requires independent" in metric["interpretation"]
    assert score("complete", "comp")["tail_deletions"] == 4
    assert score("i O", "l 0")["character_errors"]["substitution"] == 2
    assert score("A B", "")["cer"] == 1


def test_micro_denominators_and_case_sensitive_exact_match():
    a, b = score("long word", "long word"), score("I", "l")
    assert aggregate([a, b])["micro_cer"] == 0.1
    assert aggregate([a, b])["exact_match_rate"] == 0.5
    assert aggregate([])["micro_cer"] is None
    assert score("A  B", "A B")["exact_match"]
    assert not score("A.", "a")["exact_match"]


def test_lossless_preview_retains_edge_pixels(tmp_path):
    from PIL import Image

    from research.english.artifacts import lossless_pdf, normalize

    image = Image.new("RGB", (123, 64), "white")
    image.putpixel((122, 63), (12, 47, 150))
    path = tmp_path / "edge.png"
    image.save(path)
    assert normalize(path, tmp_path / "normalized.png")["border_ink"]
    result = lossless_pdf([path], tmp_path / "preview.pdf")
    assert result["pages"][0]["pixel_identical"]


def test_review_requires_actual_content_gate():
    from research.english.review import AXES, validate_answers

    packet = {"run_id": "p", "items": [{"blind_id": "B01", "expected": "Good work."}]}
    answers = {
        "packet_run_id": "p",
        "reviewer": "test reviewer",
        "reviewer_type": "human",
        "answers": [
            {
                "blind_id": "B01",
                "transcription": "Good work.",
                "content": "incorrect",
                "style": dict.fromkeys(AXES, 4),
            }
        ],
    }
    with pytest.raises(ValueError, match="Style requires"):
        validate_answers(packet, answers)
    answers["answers"][0]["content"] = "complete"
    assert validate_answers(packet, answers)["remaining"] == 0
    answers["answers"][0]["transcription"] = "Good."
    with pytest.raises(ValueError, match="disagrees"):
        validate_answers(packet, answers)
    answers["reviewer_type"] = "model"
    with pytest.raises(ValueError, match="Named human"):
        validate_answers(packet, answers)


def test_edge_ink_is_suspicion_not_proven_error(tmp_path):
    from PIL import Image

    from research.english.artifacts import normalize

    image = Image.new("L", (20, 20), 255)
    image.putpixel((19, 10), 0)
    path = tmp_path / "raw.png"
    image.save(path)
    result = normalize(path, tmp_path / "normalized.png")
    assert result["border_ink"]
    assert "not proof" in result["border_interpretation"]
    assert result["ink_bbox"] == [19, 10, 20, 11]


def test_sealed_analysis_cannot_be_overwritten(tmp_path):
    import json

    from research.english.summarize import summarize

    (tmp_path / "run.json").write_text(json.dumps({"status": "held"}))
    with pytest.raises(ValueError, match="sealed analysis"):
        summarize(tmp_path / "absent-evaluation", tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({"status": "running"}))
    (tmp_path / "reports").mkdir()
    existing = tmp_path / "reports/summary.json"
    existing.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="already exists"):
        summarize(tmp_path / "absent-evaluation", tmp_path)
    assert existing.read_text() == "existing evidence"
