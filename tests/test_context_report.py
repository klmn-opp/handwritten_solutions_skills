"""Reporting must not overwrite immutable experiment evidence."""

import json

import pytest

from research.corpus.context_report import comparisons, curves, diagnose


def test_reporting_refuses_sealed_destination_without_writing(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"status": "held"}))
    evidence = tmp_path / "reports"
    evidence.mkdir()
    original = evidence / "frozen-diagnosis-summary.json"
    original.write_text("sealed evidence")
    for call in (
        lambda: diagnose(tmp_path / "absent-source", tmp_path),
        lambda: comparisons(tmp_path / "absent-source", tmp_path),
        lambda: curves(tmp_path),
    ):
        with pytest.raises(ValueError, match="immutable"):
            call()
    assert original.read_text() == "sealed evidence"
    assert len(list(evidence.iterdir())) == 1
