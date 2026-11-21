import json
from pathlib import Path

import pytest

from hwsol.config import data_dir
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, initialize, render_assignment, verify_run


def test_result_payload_cannot_override_run_admission_or_identity(tmp_path):
    run = Run.create("admission-contract", workspace=tmp_path, run_id="actual-candidate")
    response = run.finish(
        "held", result={"status": "passed", "run_id": "wrong-id", "run": "wrong-path"}
    )
    assert response["status"] == run.record()["status"] == "held"
    assert response["result_status"] == run.record()["result"]["status"] == "passed"
    assert response["run_id"] == "actual-candidate"
    assert response["run"] == str(run.path)
    assert not run.record()["deliverable"]
    assert not (tmp_path / "latest.json").exists()
    assert verify_run(run.path)["run_status"] == "held"


def fixture_job(root):
    (root / "question.txt").write_text("Solve 3x+2=11.")
    (root / "solution.md").write_text("Subtract two and divide by three.\n\n$$x=3$$")
    path = root / "job.json"
    atomic_json(
        path,
        {
            "schema_version": 1,
            "generated_by": "test",
            "profile": "bundled-default@v1",
            "render": {"dpi": 72},
            "problems": [
                {
                    "number": "1",
                    "statement": "Solve 3x+2=11.",
                    "sources": [{"path": "question.txt"}],
                    "solution": "solution.md",
                }
            ],
        },
    )
    return path


def test_success_failure_latest_and_immutability(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "data"))
    ws = tmp_path / "workspace"
    result = render_assignment(fixture_job(tmp_path), workspace=ws, run_id="successful")
    run = Path(result["run"])
    assert verify_run(run)["status"] == "passed"
    latest = (ws / "latest.json").read_bytes()
    assert (run / "solutions/01.md").read_text() == (tmp_path / "solution.md").read_text()
    (tmp_path / "solution.md").write_text("$$\\unsupported{x}$$")
    with pytest.raises(ValueError):
        render_assignment(tmp_path / "job.json", workspace=ws, run_id="failed")
    assert (ws / "latest.json").read_bytes() == latest
    assert verify_run(ws / "runs/failed")["run_status"] == "failed"
    with pytest.raises(FileExistsError):
        Run.create("test", workspace=ws, run_id="failed")
    with pytest.raises(ValueError, match="immutable"):
        Run(run).finish("passed")
    (run / "solutions/01.md").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_run(run)


def test_profile_path_compatibility_and_archive(tmp_path, monkeypatch):
    from hwsol.workspace import archive

    old = tmp_path / "legacy"
    old.mkdir()
    atomic_json(old / "settings.json", {"default_profile": "personal@v1"})
    digest = sha256(old / "settings.json")
    ws = tmp_path / "workspace"
    initialize(ws, old)
    monkeypatch.delenv("HWSOL_DATA_DIR", raising=False)
    monkeypatch.setenv("HWSOL_WORKSPACE", str(ws))
    assert data_dir() == old
    index = archive(ws, [old])
    assert (
        json.loads(Path(index["index"]).read_text())["files"][str(old / "settings.json")]["sha256"]
        == digest
    )
    assert sha256(old / "settings.json") == digest
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "explicit"))
    assert data_dir() == tmp_path / "explicit"


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", ".", "a/b"])
def test_invalid_run_id(tmp_path, run_id):
    with pytest.raises(ValueError):
        Run.create("test", workspace=tmp_path, run_id=run_id)


def test_neural_backend_not_admitted(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "data"))
    path = fixture_job(tmp_path)
    job = json.loads(path.read_text())
    job["backend"] = "diffink-experimental"
    atomic_json(path, job)
    with pytest.raises(ValueError, match="not admitted"):
        render_assignment(path, workspace=tmp_path / "workspace")
    assert not (tmp_path / "workspace/latest.json").exists()
