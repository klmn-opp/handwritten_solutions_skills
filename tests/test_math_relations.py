import json

import pytest


@pytest.mark.parametrize(("command", "character"), [("ge", "≥"), ("le", "≤"), ("ne", "≠")])
def test_relation_aliases_stay_auditable(tmp_path, command, character):
    from hwsol.jobs.runtime import render_job
    from hwsol.profiles.io import atomic_json
    from hwsol.profiles.store import ProfileStore

    (tmp_path / "source.txt").write_text("Relation rendering test")
    (tmp_path / "solution.md").write_text("$$x\\" + command + "0$$")
    manifest = tmp_path / "job.json"
    atomic_json(
        manifest,
        {
            "schema_version": 1,
            "generated_by": "test",
            "profile": "bundled-default@v1",
            "problems": [
                {
                    "number": "1",
                    "sources": [{"path": "source.txt"}],
                    "statement": "Relation test",
                    "solution": "solution.md",
                }
            ],
        },
    )
    render_job(manifest, tmp_path / "rendered", ProfileStore(tmp_path / "data"))
    report = json.loads((tmp_path / "rendered/validation_report.json").read_text())
    visible = report["pages"][0]["glyph_provenance"]
    assert character in {g["semantic"] for g in visible}
    assert all(g["source_type"] != "unattributed-raster" for g in visible)
