from collections import Counter

import pytest

from research.corpus.context_review import balanced_candidate_orders
from research.corpus.context_review_import import strict_json, summarize, validate_review


def payload(rating):
    return {"schema": "context-human-review/1", "review_id": "fixture", "ratings": {"a": rating}}


@pytest.mark.parametrize("score", [True, 3.0, 0, 6, "4"])
def test_reject_invalid_score_types(score):
    with pytest.raises(ValueError, match="integer"):
        validate_review(
            payload({"content": "correct", "scores": {"naturalness": score}}), "fixture", {"a"}
        )


def test_reject_foreign_export_unknown_ids_and_duplicate_json():
    with pytest.raises(ValueError, match="identity"):
        validate_review(payload({}), "other-run", {"a"})
    with pytest.raises(ValueError, match="Unknown"):
        validate_review(payload({}), "fixture", {"b"})
    with pytest.raises(ValueError, match="Duplicate"):
        strict_json('{"ratings": {}, "ratings": {}}')
    with pytest.raises(ValueError, match="Nonfinite"):
        strict_json('{"unexpected": NaN}')


def test_content_gate_and_unreviewed_scores_never_become_zero():
    with pytest.raises(ValueError, match="confirmed correct"):
        validate_review(
            payload({"content": "uncertain", "scores": {"naturalness": 5}}), "fixture", {"a"}
        )
    clean = validate_review(
        payload({"content": "correct", "scores": {"naturalness": None}}), "fixture", {"a", "b"}
    )
    assert "b" not in clean["ratings"] and clean["ratings"]["a"]["scores"]["naturalness"] is None


def test_pair_generation_is_same_group_gated_and_dimension_specific():
    def row(id, group, method, content, scores):
        return {
            "id": id,
            "group_id": group,
            "method": method,
            "position": id,
            "image_sha256": id * 8,
            "source_split": "holdout",
            "content": content,
            "scores": scores,
        }

    rows = [
        row("a", "g1", "one", "correct", {"naturalness": 4, "similarity": 2}),
        row("b", "g1", "two", "correct", {"naturalness": 2, "similarity": 4}),
        row("c", "g1", "three", "incorrect", {}),
        row("d", "g2", "one", "correct", {"naturalness": 5}),
    ]
    summary, pairs = summarize(rows)
    assert len(pairs) == 2
    assert {p["dimension"]: p["preferred_id"] for p in pairs} == {
        "naturalness": "a",
        "similarity": "b",
    }
    assert all(not p["optimization_eligible"] and p["source_split"] == "holdout" for p in pairs)
    assert summary["methods"]["three"]["scores"]["naturalness"] == {"n": 0, "mean": None}
    assert summary["methods"]["one"]["scores"]["similarity"] == {"n": 1, "mean": 2}


@pytest.mark.parametrize("groups", [0, 4, 5, 13])
def test_future_positions_balanced_and_reproducible(groups):
    methods = ["font", "old", "sharp", "distance", "mass"]
    before = methods.copy()
    orders = balanced_candidate_orders(methods, groups, 1207)
    assert methods == before and orders == balanced_candidate_orders(methods, groups, 1207)
    assert len(orders) == groups and all(set(order) == set(methods) for order in orders)
    for method in methods:
        counts = Counter(order.index(method) for order in orders)
        values = [counts[i] for i in range(len(methods))]
        assert max(values) - min(values) <= 1
        if groups == 4:
            assert len(counts) == 4
