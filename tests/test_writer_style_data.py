import pytest

from research.corpus.writer_style_data import split_indices


def rows():
    return [
        dict(text="abcdefghijklmn", dataset_split="train", source_hash="a"),
        dict(text="opqrstuvwxyz", dataset_split="train", source_hash="a"),
        dict(text="confidence", dataset_split="validation", source_hash="b"),
    ]


def test_novel_prompts_never_enter_train_and_documents_remain_disjoint():
    result = split_indices(rows(), [0])
    assert result == {"train": [0, 1], "validation": [2], "regression": [0]}
    changed = rows()
    changed[2]["source_hash"] = "a"
    with pytest.raises(ValueError, match="Document split"):
        split_indices(changed, [0])


def test_novel_label_contamination_and_sealed_documents_are_rejected():
    changed = rows() + [dict(text="matrix", dataset_split="train", source_hash="a")]
    with pytest.raises(ValueError, match="Novel prompt"):
        split_indices(changed, [0])
    from research.corpus.latent_ink_train import FORBIDDEN

    changed = rows()
    changed[0]["source_hash"] = FORBIDDEN
    with pytest.raises(ValueError, match="Sealed"):
        split_indices(changed, [0])


def test_unknown_regression_id_is_never_silently_substituted():
    with pytest.raises(ValueError, match="Unknown regression"):
        split_indices(rows(), [99])
