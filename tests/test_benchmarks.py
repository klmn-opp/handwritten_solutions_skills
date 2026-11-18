import json
from pathlib import Path

import pytest

from hwsol.benchmarks.metrics import observed_metrics
from hwsol.benchmarks.runner import execute_case, prepare_packet
from hwsol.benchmarks.schema import load_corpus, select
from hwsol.profiles.io import sha256
from hwsol.workspace import Run, verify_run


def test_fixed_inventory_and_holdout_barrier():
    corpus = load_corpus()
    assert len(corpus.cases) == 60
    assert len(select(corpus, "development", "tuning")) == 48
    assert {c.category for c in corpus.cases} == {
        "linear-algebra",
        "calculus",
        "statistics",
        "physics",
        "english",
        "layout",
        "edge",
    }
    for suite in ("holdout", "all"):
        with pytest.raises(ValueError, match="Holdout"):
            select(corpus, suite, "tuning")


def test_agent_packet_has_originals_without_solutions(tmp_path):
    result = prepare_packet(workspace=tmp_path, run_id="packet")
    root = Path(result["run"])
    assert verify_run(root)["run_status"] == "prepared"
    assert not list(root.rglob("*.md"))
    assert list(root.rglob("*.pdf")) and list(root.rglob("*.png")) and list(root.rglob("*.txt"))
    question = json.loads((root / "reports/questions.json").read_text())["cases"][0]
    assert sha256(root / question["source"]) == question["source_sha256"]
    assert '"reference_solution":' not in (root / "run.json").read_text()


def test_independent_transcription_bound_to_output():
    obs = {
        "artifact_sha256": "a" * 64,
        "method": "human-blind",
        "evaluator": "reviewer-1",
        "reference_visible": False,
        "english_transcription": "The median median is a value.",
    }
    scores = observed_metrics("The median is a value.", obs, "a" * 64)
    assert scores["english"]["cer"] > 0 and scores["english"]["word_count_delta"] == 1
    with pytest.raises(ValueError, match="different output"):
        observed_metrics("Expected", obs, "b" * 64)
    obs["reference_visible"] = True
    with pytest.raises(ValueError, match="must not see"):
        observed_metrics("Expected", obs, "a" * 64)


def test_solver_failure_is_not_a_renderer_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "data"))
    parent = Run.create("benchmark", workspace=tmp_path, run_id="evaluation")
    case = select(load_corpus(), "smoke")[0]
    result = execute_case(
        case, parent, "bundled-default@v1", {"seed": 1}, {"agent": "test"}, tmp_path / "packet"
    )
    assert result["outcome"] == "solver_failure"
    assert result["renderer"]["status"] == "not_run"
    parent.finish("held")
    assert not (tmp_path / "latest.json").exists()


def test_expected_rejection_keeps_failed_run(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "data"))
    parent = Run.create("benchmark", workspace=tmp_path, run_id="reject")
    case = next(c for c in load_corpus().cases if c.id == "stress-unsupported-align")
    result = execute_case(case, parent, "bundled-default@v1", {"seed": 1, "dpi": 72})
    assert result["outcome"] == "expected_rejection"
    assert result["expectation_passed"]
    assert verify_run(Path(result["case_run"]))["run_status"] == "failed"
    assert not (tmp_path / "latest.json").exists()
    parent.finish("passed")
