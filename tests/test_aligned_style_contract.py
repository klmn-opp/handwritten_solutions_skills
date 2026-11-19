import pytest

from research.corpus.aligned_style_contract import text_tokens, training_indices


def test_full_text_encoding_preserves_case_punctuation_and_no_truncation():
    assert text_tokens(["The,"])[0][:4] == [ord(c) + 1 for c in "The,"]
    for bad in ["", "x" * 33, "α"]:
        with pytest.raises(ValueError):
            text_tokens([bad])


def records():
    rows = [
        {"text": str(i % 32), "dataset_split": "train" if i < 64 else "validation"}
        for i in range(80)
    ]
    details = [
        {"index": i, "text": r["text"], "split": r["dataset_split"], "eligible": True}
        for i, r in enumerate(rows)
    ]
    return rows, details


def test_validation_cannot_enter_training_even_when_registration_is_eligible():
    rows, details = records()
    assert training_indices(details, rows) == list(range(64))
    details[-1]["split"] = "train"
    with pytest.raises(ValueError, match="identity"):
        training_indices(details, rows)


def test_registration_coverage_is_not_silently_relaxed():
    rows, details = records()
    details[0]["eligible"] = False
    with pytest.raises(ValueError, match="coverage"):
        training_indices(details, rows)
    with pytest.raises(ValueError, match="exactly once"):
        training_indices(details[:-1] + details[:1], rows)
