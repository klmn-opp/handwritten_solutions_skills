import pytest

from research.corpus.contracts import ink_decision, validate_preference


def test_optional_ink_always_falls_back_without_independent_validation():
    assert ink_decision("x", None)["backend"] == "font-cpu"
    assert (
        ink_decision("x", {"origin": "model-generated", "trajectory_kind": "offline-image"}, True)[
            "backend"
        ]
        == "font-cpu"
    )
    assert ink_decision("x", {"origin": "real-personal"}, True)["backend"] == "font-cpu"


def test_preference_has_content_and_provenance_gates():
    row = {
        "content_gate": "human-verified-correct",
        "reviewer": "unit-fixture",
        "choice": "a",
        "candidates": [
            {"id": i, "artifact_sha256": "a" * 64, "origin": "base-font", "source_run": "unit-test"}
            for i in ("a", "b")
        ],
    }
    assert not validate_preference(row)["automatic_publication"]
    with pytest.raises(ValueError):
        validate_preference({**row, "content_gate": "ocr-passed"})
    row["candidates"][0]["origin"] = "model-generated"
    with pytest.raises(ValueError):
        validate_preference(row)
