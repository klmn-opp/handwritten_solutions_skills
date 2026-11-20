"""Public archives retain usable guides without exporting private operational files."""

import json
import re
from pathlib import Path
from zipfile import ZipFile

import pytest

import public_build_backend as backend
from hwsol.jobs.schema import Job
from scripts.package_release import (
    audit_public_files,
    public_files,
    stage_public_source,
    verify_public_source,
    write_skill_archive,
)


def source_fixture(root: Path) -> Path:
    source = root / "source"
    (source / "public_docs").mkdir(parents=True)
    (source / "public_docs/README.md.in").write_text("[Guide](public_docs/GUIDE.md)\n")
    (source / "public_docs/GUIDE.md").write_text("A usable public guide.\n")
    (source / "README.md").write_text("Private local readme stays untouched.\n")
    (source / "private.md").write_text("Private operational notes.\n")
    policy = {
        "schema": "hwsol-public-distribution/1",
        "files": [
            "public_docs/README.md.in",
            "public_docs/GUIDE.md",
            "public_docs/distribution.json",
        ],
        "patterns": [],
        "overlays": {"README.md": "public_docs/README.md.in"},
    }
    (source / "public_docs/distribution.json").write_text(json.dumps(policy))
    return source


def test_public_snapshot_excludes_private_files_and_preserves_originals(tmp_path):
    source = source_fixture(tmp_path)
    original = (source / "README.md").read_bytes()
    staged = tmp_path / "public"
    stage_public_source(source, staged)
    assert (source / "README.md").read_bytes() == original
    assert not (staged / "private.md").exists()
    assert (staged / "README.md").read_bytes() == (source / "public_docs/README.md.in").read_bytes()
    assert audit_public_files(public_files(staged))["relative_links"] == "resolved"
    output = tmp_path / "skill.zip"
    write_skill_archive(staged, output)
    with ZipFile(output) as archive:
        assert "handwritten-solutions/public_docs/GUIDE.md" in archive.namelist()
        assert not any("private.md" in name or ".git/" in name for name in archive.namelist())


def test_public_allowlist_does_not_override_content_privacy(tmp_path):
    source = source_fixture(tmp_path)
    (source / "public_docs/GUIDE.md").write_text("/" + "home" + "/private-author/example\n")
    with pytest.raises(ValueError, match="Private operational references"):
        stage_public_source(source, tmp_path / "public")


def test_required_user_guide_cannot_be_silently_removed(tmp_path):
    source = source_fixture(tmp_path)
    (source / "public_docs/README.md.in").write_text("[Guide](public_docs/absent.md)\n")
    with pytest.raises(ValueError, match="Broken public relative links"):
        stage_public_source(source, tmp_path / "public")


def test_old_ref_cannot_borrow_current_public_layer(tmp_path):
    source = tmp_path / "old-ref"
    source.mkdir()
    with pytest.raises(ValueError, match="Selected ref has no versioned public distribution layer"):
        stage_public_source(source, tmp_path / "public")


def test_unprepared_source_cannot_pass_release_audit():
    with pytest.raises(ValueError, match="Unprepared source"):
        audit_public_files({"README.md": b"Public-looking text without an approved source policy."})


def test_current_public_policy_keeps_skill_links_resolvable(tmp_path):
    source = Path(__file__).resolve().parents[1]
    stage_public_source(source, tmp_path / "public")
    files = public_files(tmp_path / "public")
    for required in (
        "SKILL.md",
        "references/jobs.md",
        "public_docs/PROFILES.md",
        "docs/PHASE0.md",
        "uv.lock",
    ):
        assert required in files
    assert "docs/MASTER_EXECUTION_PLAN.md" not in files
    assert not any(name.startswith("research/") for name in files)
    profile_root = "src/hwsol/assets/presets/patrick-hand"
    profile = json.loads(files[f"{profile_root}/profile.json"])
    for asset in profile["assets"]:
        assert f"{profile_root}/{asset['path']}" in files
    assert "src/hwsol/assets/benchmarks/v1/SHA256SUMS" in files


def test_public_job_example_matches_runtime_schema():
    guide = Path(__file__).resolve().parents[1] / "public_docs/GUIDE.md"
    examples = re.findall(r"```json\n(.*?)\n```", guide.read_text(), flags=re.S)
    assert examples
    for source in examples:
        job = Job.model_validate_json(source)
        assert job.profile == "bundled-default@v1"
        assert job.problems[0].verification.status == "passed"


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_requires_for_build_wheel", ()),
        ("prepare_metadata_for_build_wheel", ("metadata",)),
        ("build_wheel", ("wheels",)),
        ("get_requires_for_build_sdist", ()),
        ("build_sdist", ("sdists",)),
    ],
)
def test_public_pep517_hooks_refuse_before_backend_or_metadata(tmp_path, monkeypatch, name, args):
    monkeypatch.setattr(backend, "__file__", str(tmp_path / "public_build_backend.py"))
    monkeypatch.setattr(
        backend, "_backend", lambda: pytest.fail("Unprepared input reached backend")
    )
    with pytest.raises(RuntimeError, match="Public wheel/sdist build refused"):
        getattr(backend, name)(*args)
    assert not (tmp_path / "metadata").exists()


def test_editable_hooks_remain_local_development_operations(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(backend, "_public", lambda: pytest.fail("Editable hit public guard"))
    monkeypatch.setattr(
        backend,
        "_backend",
        lambda: SimpleNamespace(
            get_requires_for_build_editable=lambda cfg: ["local-development"],
            prepare_metadata_for_build_editable=lambda directory, cfg: directory,
            build_editable=lambda directory, cfg, metadata: directory,
        ),
    )
    assert backend.get_requires_for_build_editable() == ["local-development"]
    assert backend.prepare_metadata_for_build_editable("metadata") == "metadata"
    assert backend.build_editable("wheel") == "wheel"


def test_post_stage_source_change_invalidates_public_receipt(tmp_path):
    source = source_fixture(tmp_path)
    public = tmp_path / "public"
    stage_public_source(source, public)
    (public / "public_docs/GUIDE.md").write_text("Changed after approval.\n")
    with pytest.raises(ValueError, match="differs from its receipt"):
        audit_public_files(public_files(public))


@pytest.mark.parametrize("name", ["build/lib/hwsol/stale.py", "src/hwsol/build/unreviewed.py"])
def test_unreviewed_build_paths_cannot_bypass_source_inventory(tmp_path, name):
    source = source_fixture(tmp_path)
    public = tmp_path / "public"
    stage_public_source(source, public)
    injected = public / name
    injected.parent.mkdir(parents=True)
    injected.write_text("UNREVIEWED = True\n")
    with pytest.raises(ValueError, match="Existing build output|outside the frozen"):
        verify_public_source(public)


def test_public_links_use_archive_paths_independent_of_host(tmp_path, monkeypatch):
    import ntpath
    from types import SimpleNamespace

    import scripts.package_release as release

    source = source_fixture(tmp_path)
    public = tmp_path / "public"
    stage_public_source(source, public)
    monkeypatch.setattr(release, "os", SimpleNamespace(path=ntpath))
    assert audit_public_files(public_files(public))["relative_links"] == "resolved"


def test_generated_sdist_config_accepts_platform_newlines_only(tmp_path):
    source = source_fixture(tmp_path)
    public = tmp_path / "public"
    stage_public_source(source, public)
    files = public_files(public)
    files["setup.cfg"] = b"[egg_info]\r\ntag_build = \r\ntag_date = 0\r\n\r\n"
    assert audit_public_files(files)["status"] == "passed"
    files["setup.cfg"] += b"[options]\r\npackages = dangerous\r\n"
    with pytest.raises(ValueError, match="Unexpected generated sdist"):
        audit_public_files(files)
