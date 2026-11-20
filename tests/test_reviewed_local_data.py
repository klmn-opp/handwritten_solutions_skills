import pytest

from research.corpus.reviewed_local_data import admission


def fixtures():
    reviews = [
        {"index": i, "status": "accept", "foreign_ink": False, "truncated": False, "notes": "clean"}
        for i in range(2)
    ]
    components = [{"index": i, "component_guard_passed": True} for i in range(3)]
    return reviews, components


def test_semantic_acceptance_cannot_override_cut_component():
    reviews, components = fixtures()
    components[0]["component_guard_passed"] = False
    kept, rejected = admission(3, reviews, components, legacy_count=2)
    assert kept == [1, 2] and rejected[0]["index"] == 0


def test_uncertain_or_unexcluded_foreign_ink_never_enters_training():
    reviews, components = fixtures()
    reviews[0]["status"] = "uncertain"
    reviews[1]["foreign_ink"] = None
    kept, rejected = admission(3, reviews, components, legacy_count=2)
    assert kept == [2] and len(rejected) == 2


def test_missing_or_duplicate_review_and_missing_component_fail_closed():
    reviews, components = fixtures()
    for rows in [reviews[:1], reviews + reviews[:1]]:
        with pytest.raises(ValueError):
            admission(3, rows, components, legacy_count=2)
    with pytest.raises(ValueError):
        admission(3, reviews, components[:2], legacy_count=2)


def test_fully_reviewed_clean_views_preserved_in_source_order():
    reviews, components = fixtures()
    kept, rejected = admission(3, reviews, components, legacy_count=2)
    assert kept == [0, 1, 2] and rejected == []
