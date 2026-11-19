import pytest

from research.corpus.context_dataset_update import check_rows


def row(name="a", **kw):
    return {
        "sample_id": name,
        "crop_sha256": name,
        "source_hash": "doc",
        "page_number": 1,
        "bbox": [0, 0, 10, 10],
        "dataset_split": "train",
        **kw,
    }


def test_document_split_and_sealed_test_cannot_enter_training():
    check_rows([row()], {"doc": "train"})
    with pytest.raises(ValueError, match="contamination"):
        check_rows([row()], {"doc": "sealed-holdout"})
    with pytest.raises(ValueError, match="contamination"):
        check_rows([row(dataset_split="holdout")], {"doc": "holdout"})


def test_duplicate_and_overlapping_ink_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        check_rows([row(), row()], {"doc": "train"})
    with pytest.raises(ValueError, match="Overlapping"):
        check_rows([row(), row("b", bbox=[1, 1, 11, 11])], {"doc": "train"})
    check_rows([row(), row("b", bbox=[11, 0, 21, 10])], {"doc": "train"})
