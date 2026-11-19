from research.corpus.context_recognition import summarize_scored
from research.english.metrics import score


def test_custom_exposure_splits_are_counted_without_merging_into_holdout():
    items = [
        {"method": "font", "split": "validation", "score": score("the", "the")},
        {"method": "font", "split": "exposed-train-composition", "score": score("is", "it")},
    ]
    result = summarize_scored(items)["font"]
    assert result["validation"]["observed_count"] == 1
    assert result["exposed-train-composition"]["observed_count"] == 1
    assert result["exposed-train-composition"]["micro_cer"] == 0.5
    assert result["holdout"]["observed_count"] == 0
    assert sum(q["observed_count"] for q in result.values()) == len(items)
