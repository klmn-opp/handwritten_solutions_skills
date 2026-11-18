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
