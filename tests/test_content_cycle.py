import pytest

from research.corpus.content_cycle import select_admitted


def evidence():
    rows = [dict(text="sample", dataset_split="train"), dict(text="mean", dataset_split="train")]
    packet = [dict(index=i, text=q["text"], sha256=str(i)) for i, q in enumerate(rows)]
    review = [dict(index=1, sha256="1", accept=False), dict(index=0, sha256="0", accept=True)]
    return packet, review, rows


def test_conservative_review_preserves_indices_and_excludes_rejections():
    assert select_admitted(*evidence()) == [0]


def test_missing_or_duplicate_reviews_cannot_admit_data():
    packet, review, rows = evidence()
    for bad in [review[:1], review + [review[0]]]:
        with pytest.raises(ValueError, match="Complete unique"):
            select_admitted(packet, bad, rows)


def test_changed_label_or_validation_cannot_enter_train():
    packet, review, rows = evidence()
    rows[0]["dataset_split"] = "validation"
    with pytest.raises(ValueError, match="Train-only"):
        select_admitted(packet, review, rows)
    rows[0].update(dataset_split="train", text="different")
    with pytest.raises(ValueError, match="Train-only"):
        select_admitted(packet, review, rows)


def test_wrong_pixels_and_nonboolean_acceptance_rejected():
    packet, review, rows = evidence()
    review[0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="Explicit review"):
        select_admitted(packet, review, rows)
    review[0].update(sha256="1", accept="yes")
    with pytest.raises(ValueError, match="Explicit review"):
        select_admitted(packet, review, rows)
