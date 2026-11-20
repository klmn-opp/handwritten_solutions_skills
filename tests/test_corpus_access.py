"""Synthetic SQLite access controls; no private original or holdout image is read."""

import json
import sqlite3

import pytest

from research import corpus_access
from research.corpus_registry import SEALED


def make_catalogue(tmp_path, documents, samples):
    path = tmp_path / "corpus.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE records(kind TEXT,id TEXT,data TEXT)")

        def add(kind, rid, record):
            db.execute("INSERT INTO records VALUES(?,?,?)", (kind, rid, json.dumps(record)))

        for digest, extra in documents.items():
            add("document", "d-" + digest, {"source_hash": digest, **extra})
            add("page", "p-" + digest, {"page_id": "p-" + digest, "source_hash": digest})
        for sid, record in samples.items():
            label = "safe label" if sid == "safe" else "DO_NOT_FETCH_PROTECTED_LABEL"
            add("sample", sid, {"sample_id": sid, "candidate_text": label, **record})
            add("recognition", "r-" + sid, {"sample_id": sid, "text": label})
    return path


def test_nested_holdout_blocks_whole_source_ancestors_siblings_before_labels(tmp_path, monkeypatch):
    documents = {h: {"source_path": h + ".pdf"} for h in ("safe", "held", "parent", "child")}
    samples = {
        "safe": {"source_hash": "safe", "dataset_split": "train", "parents": []},
        "ancestor": {"source_hash": "parent", "dataset_split": "train", "parents": []},
        "ancestor-sibling": {"source_hash": "parent", "dataset_split": "train", "parents": []},
        "held": {
            "source_hash": "held",
            "dataset_split": None,
            "parents": ["ancestor"],
            "derivation": {"selection": {"split": "holdout"}},
        },
        "same-source-train": {"source_hash": "held", "dataset_split": "train", "parents": []},
        "descendant": {"source_hash": "child", "dataset_split": "train", "parents": ["held"]},
    }
    path = make_catalogue(tmp_path, documents, samples)
    before = path.read_bytes()
    loads = json.loads
    parsed_values = []

    def guarded_loads(raw, *args, **kwargs):
        assert "DO_NOT_FETCH_PROTECTED_LABEL" not in raw
        parsed_values.append(raw)
        return loads(raw, *args, **kwargs)

    monkeypatch.setattr(corpus_access.json, "loads", guarded_loads)
    result = corpus_access.catalogue(path)
    assert [s["sample_id"] for s in result["samples"]] == ["safe"]
    assert [s["source_hash"] for s in result["documents"]] == ["safe"]
    assert [r["text"] for r in result["recognitions"]] == ["safe label"]
    assert {"held", "parent", "child"} <= set(result["filter"]["excluded_source_hashes"])
    assert result["filter"]["blocked_sample_count"] == 5
    assert parsed_values  # Parent ID lists and allowed records were actually decoded.
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["sha", "name", "top-level", "nested-direct"])
def test_existing_sealed_name_and_split_guards_remain_active(tmp_path, mode):
    digest = next(iter(SEALED)) if mode == "sha" else "blocked"
    name = "session-C-original.pdf" if mode == "name" else "ordinary.pdf"
    blocked = {"source_hash": digest, "parents": []}
    if mode == "top-level":
        blocked["dataset_split"] = "test"
    elif mode == "nested-direct":
        blocked["derivation"] = {"split": "sealed"}
    path = make_catalogue(
        tmp_path,
        {"safe": {"source_path": "safe.pdf"}, digest: {"source_path": name}},
        {"safe": {"source_hash": "safe", "parents": []}, "blocked": blocked},
    )
    result = corpus_access.catalogue(path)
    assert [s["sample_id"] for s in result["samples"]] == ["safe"]
    assert [r["sample_id"] for r in result["recognitions"]] == ["safe"]
    assert digest in result["filter"]["excluded_source_hashes"]


def test_both_actual_research_entrypoints_use_guarded_reader():
    from research import missing_math_review, pi_source_feedback

    assert missing_math_review.catalogue is corpus_access.catalogue
    assert pi_source_feedback.catalogue is corpus_access.catalogue
