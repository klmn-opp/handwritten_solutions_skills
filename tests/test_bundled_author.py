"""Built-in trust boundaries; synthetic receipt is not release or author evidence."""

import json
import shutil

import pytest

from hwsol.identity import runtime_text_capabilities
from hwsol.profiles import presets
from hwsol.profiles.io import atomic_json
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore, copy_snapshot, load_snapshot


def test_clean_default_is_author_and_explicit_compatibility_font_remains(tmp_path):
    store = ProfileStore(tmp_path / "empty-store")
    author = store.resolve()
    assert author.manifest.ref == "author-default@v1"
    assert author.manifest.render.word_gap_scale == 1.1448
    assert author.manifest.ink_bundle
    assert presets.is_installed_author(author)
    assert store.resolve("bundled-default@v1").manifest.source_kind == "test-font"
    assert not store.root.exists(), "Resolving the clean default must not mutate the user store"
    listed = store.list()
    assert listed["default"] == author.manifest.ref
    assert {row["name"] for row in listed["profiles"]} == {"author-default", "bundled-default"}
    assert all(row["read_only"] for row in listed["profiles"])


def test_existing_default_and_explicit_profile_take_precedence(tmp_path):
    store = ProfileStore(tmp_path / "existing-store")
    atomic_json(
        store.root / "settings.json",
        {"default_profile": "bundled-default@v1", "background": {"kind": "dots"}},
    )
    before = (store.root / "settings.json").read_bytes()
    assert store.resolve().manifest.ref == "bundled-default@v1"
    assert store.resolve("author-default").manifest.ref == "author-default@v1"
    assert (store.root / "settings.json").read_bytes() == before


@pytest.mark.parametrize("name", ["author-default", "bundled-default"])
def test_builtin_names_cannot_be_created_or_overwritten(tmp_path, name):
    store = ProfileStore(tmp_path / "store")
    with pytest.raises(ProfileError, match="read-only"):
        store.create(name)
    with pytest.raises(ProfileError, match="Only"):
        store.resolve(f"{name}@v2")
    with pytest.raises(ProfileError, match="Only"):
        store.resolve(f"{name}@v1", candidate=True)


def test_local_shadow_and_import_cannot_replace_the_builtin(tmp_path):
    store = ProfileStore(tmp_path / "store")
    real = store.resolve()
    shadow = store.root / "profiles/author-default/versions/v1"
    copy_snapshot(real, shadow)
    assert store.resolve().directory == presets.author_directory()
    assert not presets.is_installed_author(load_snapshot(shadow))
    with pytest.raises(ProfileError, match="Only the installed"):
        presets.validate_author(load_snapshot(shadow))
    archive = tmp_path / "author.zip"
    store.export(real.manifest.ref, archive)
    with pytest.raises(ProfileError, match="read-only"):
        ProfileStore(tmp_path / "other").import_archive(archive)


@pytest.fixture
def test_release(tmp_path, monkeypatch):
    """Use actual compiled assets with a local synthetic test-admission receipt."""
    from hwsol.profiles import store as store_module

    folder = tmp_path / "installed-preset"
    shutil.copytree(presets.author_directory(), folder)
    receipt = folder / "release.json"
    receipt.unlink(missing_ok=True)
    monkeypatch.setattr(presets, "author_directory", lambda: folder)
    monkeypatch.setattr(store_module, "author_directory", lambda: folder)
    loaded = load_snapshot(folder)
    result = presets.write_release_receipt(
        loaded,
        environments=[runtime_text_capabilities(loaded.font)],
        evidence_sha256=["a" * 64],  # Deliberately synthetic contract fixture.
        output=receipt,
    )
    return loaded, receipt, result


def test_receipt_and_local_activation(test_release, tmp_path):
    loaded, path, _ = test_release
    result = presets.validate_author(loaded)
    assert result["scope"] == "packaged-compiled-author-preset"
    store = ProfileStore(tmp_path / "local")
    assert store.activate(loaded.manifest.ref)["qualification"] == result
    assert store.resolve().manifest.ref == loaded.manifest.ref
    store.activate("bundled-default@v1")
    assert store.resolve().manifest.ref == "bundled-default@v1"
    with pytest.raises(ProfileError, match="already exists"):
        presets.write_release_receipt(loaded, environments=[], evidence_sha256=[], output=path)


@pytest.mark.parametrize("field", ["profile_sha256", "recipe_sha256", "source_sha256"])
def test_changed_profile_recipe_or_runtime_refuses_without_default_mutation(
    test_release, tmp_path, field
):
    loaded, path, receipt = test_release
    receipt[field] = "0" * 64
    atomic_json(path, receipt)
    store = ProfileStore(tmp_path / "local")
    store.activate("bundled-default@v1")
    before = (store.root / "settings.json").read_bytes()
    with pytest.raises(ProfileError):
        store.activate(loaded.manifest.ref)
    assert (store.root / "settings.json").read_bytes() == before


def test_unverified_text_environment_and_missing_receipt_refuse(test_release):
    loaded, path, receipt = test_release
    receipt["environments"][0]["machine"] = "not-the-current-machine"
    atomic_json(path, receipt)
    with pytest.raises(ProfileError, match="text environment"):
        presets.validate_author(loaded)
    path.unlink()
    with pytest.raises(ProfileError, match="no release receipt"):
        presets.validate_author(loaded)


def test_doctor_checks_admission_instead_of_only_font_loading(test_release, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from hwsol.cli import app

    _, receipt, _ = test_release
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "empty-store"))
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["profile_admission"]["scope"] == (
        "packaged-compiled-author-preset"
    )
    receipt.unlink()
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 2
    assert "no release receipt" in json.loads(result.stderr)["error"]


def test_asset_tampering_is_rechecked_after_resolution(test_release):
    loaded, _, _ = test_release
    font = loaded.font
    font.write_bytes(font.read_bytes() + b"tampered")
    with pytest.raises(ProfileError, match="integrity"):
        presets.validate_author(loaded)


def test_ordinary_job_uses_default_author_without_import_or_qualification(test_release, tmp_path):
    from hwsol.jobs.runtime import render_job, verify_job

    loaded, _, _ = test_release
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "question.txt").write_text("Compute one plus one.")
    (inputs / "answer.md").write_text("The sum is $1+1=2$.")
    atomic_json(
        inputs / "job.json",
        {
            "schema_version": 1,
            "generated_by": "Automated CPU integration test; trivial arithmetic checked",
            "problems": [
                {
                    "number": "1",
                    "statement": "Compute one plus one.",
                    "sources": [{"path": "question.txt"}],
                    "solution": "answer.md",
                    "verification": {"status": "passed", "checks": [{"lhs": "1+1", "rhs": "2"}]},
                }
            ],
        },
    )
    store = ProfileStore(tmp_path / "fresh-store")
    output = tmp_path / "rendered"
    result = render_job(inputs / "job.json", output, store=store)
    assert result["status"] == "passed"
    assert verify_job(output, pixels=True)["page_count"] == 1
    report = json.loads((output / "validation_report.json").read_text())
    assert report["profile"]["reference"] == loaded.manifest.ref
    assert report["pages"][0]["candidate"]["local_release_qualification"]["scope"] == (
        "packaged-compiled-author-preset"
    )
    assert not store.root.exists()


def test_legacy_probe_cannot_claim_the_complete_author_recipe(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from hwsol.cli import app

    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "empty-store"))
    source = tmp_path / "answer.md"
    source.write_text("The result is $x=1$.")
    output = tmp_path / "probe"
    runner = CliRunner()
    result = runner.invoke(app, ["render-probe", "--input", str(source), "--output", str(output)])
    assert result.exit_code == 2
    assert "hwsol render --manifest" in result.output
    assert "--profile bundled-default@v1" in result.output
    assert not output.exists()
    result = runner.invoke(
        app,
        [
            "render-probe",
            "--input",
            str(source),
            "--output",
            str(output),
            "--profile",
            "bundled-default@v1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output / "probe.pdf").is_file()


def test_local_only_compiled_export_reports_actual_representation(tmp_path):
    from hwsol.profiles.io import atomic_json

    store = ProfileStore(tmp_path / "store")
    author = store.resolve("author-default@v1")
    folder = store.root / "profiles/compiled-copy/candidates/v1"
    copy_snapshot(author, folder)
    data = json.loads((folder / "profile.json").read_text())
    data.update(profile="compiled-copy", distribution="local-only")
    atomic_json(folder / "profile.json", data)
    result = store.export("compiled-copy@v1", tmp_path / "compiled.zip", candidate=True)
    assert result["contents"] == (
        "compiled runtime outlines and fonts; no raw samples or release authority"
    )
