"""Personal CPU contract guards using public fonts and synthetic local assets.

These fixtures exercise recipe/profile/job boundaries, not handwriting quality.
No private source, research run or mocked successful renderer is required.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hwsol.jobs.runtime import load_job
from hwsol.personal_ink.bundle import load_recipe
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore, copy_snapshot, load_snapshot
from hwsol.training.studio import publish
from hwsol.workspace import render_assignment, verify_run


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def inventory(directory: Path) -> None:
    """Re-sign a synthetic fixture, so tests reach semantic rather than SHA guards."""
    manifest = read(directory / "profile.json")
    manifest["assets"] = [
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": sha256(path),
            "role": "font"
            if path.relative_to(directory).as_posix() == manifest["font"]
            else "personal-ink"
            if path.relative_to(directory).as_posix() == manifest["ink_bundle"]
            else "provenance",
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "profile.json"
    ]
    atomic_json(directory / "profile.json", manifest)


@pytest.fixture
def ink_candidate(tmp_path):
    """A relocatable contract fixture; never claimed to be a render-ready author bank."""
    store = ProfileStore(tmp_path / "source-store")
    base = store.resolve("bundled-default@v1")
    folder = store.root / "profiles/synthetic-ink/candidates/v1"
    folder.mkdir(parents=True)
    shutil.copyfile(base.font, folder / "font.ttf")
    copy_snapshot(base, folder / "nested/base")
    copy_snapshot(base, folder / "nested/strokes")
    banks = {
        name: f"banks/{name}"
        for name in ("legacy", "sessions", "historical", "rules", "f_variants", "session_o")
    }
    for name, relative in banks.items():
        bank = folder / relative
        bank.mkdir(parents=True)
        if name == "historical":
            atomic_json(bank / "manifest.json", {"sources": []})
        elif name == "f_variants":
            atomic_json(bank / "candidates.json", [])
            shutil.copyfile(base.font, bank / "source-sample.ttf")
        else:
            # Real encoded image bytes, labelled synthetic and only used for the loader contract.
            from PIL import Image

            Image.new("RGB", (5, 5), (0, 0, 0)).save(bank / "glyph.png")
            atomic_json(
                bank / "sources.json",
                [
                    {
                        "id": "synthetic-loader-fixture",
                        "character": "X",
                        "review_status": "agent-accepted",
                        "clean_isolated_ink": True,
                        "original_pixels_edited": False,
                        "crop": "glyph.png",
                        "provenance": "synthetic fixture, not author handwriting",
                    }
                ],
            )
    recipe = {
        "schema": "personal-cpu-recipe/1",
        "recipe": "reviewed-outlines/1",
        "base_profile": "nested/base",
        "base_profile_sha256": sha256(folder / "nested/base/profile.json"),
        "stroke_profile": "nested/strokes",
        "stroke_profile_sha256": sha256(folder / "nested/strokes/profile.json"),
        "calibrated_font_sha256": sha256(base.font),
        "banks": banks,
        "bindings": {
            "font_sha256": sha256(base.font),
            "digit_sample_ids": ["synthetic-digit-0", "synthetic-digit-1"],
            "horizontal_source_ids": ["synthetic-bar"],
            "horizontal_source_sha256": "0" * 64,
            "single_bar_source_id": "synthetic-bar",
        },
        "admission": {"fixture_only": True, "production_approval": False},
        "scan_roles": ["X"],
    }
    atomic_json(folder / "ink.json", recipe)
    manifest = base.manifest.model_dump(mode="json")
    manifest.update(
        profile="synthetic-ink",
        version="v1",
        source_kind="derived",
        distribution="local-only",
        font="font.ttf",
        ink_bundle="ink.json",
        description="Synthetic data-only recipe guard fixture; not a handwriting model",
    )
    atomic_json(folder / "profile.json", manifest)
    inventory(folder)
    return store, folder


def job_file(root: Path, **overrides) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "question.txt").write_text("Solve 3x+2=11.")
    (root / "answer.md").write_text("Subtract two, then divide by three.\n\n$$x=3$$")
    data = {
        "schema_version": 1,
        "generated_by": "synthetic-personal-cpu-contract-test",
        "render": {"dpi": 72},
        "problems": [
            {
                "number": "1",
                "sources": [{"path": "question.txt"}],
                "statement": "Solve 3x+2=11.",
                "solution": "answer.md",
            }
        ],
        **overrides,
    }
    target = root / "job.json"
    atomic_json(target, data)
    return target


@pytest.mark.parametrize("profile", [None, "synthetic-ink"])
def test_experimental_job_requires_an_explicit_version(tmp_path, profile):
    path = job_file(tmp_path, backend="personal-cpu-experimental", profile=profile)
    with pytest.raises(ValueError, match="explicit pinned Profile"):
        load_job(path)


def test_candidate_cannot_add_an_unreviewed_realism_policy(tmp_path):
    path = job_file(
        tmp_path,
        backend="personal-cpu-experimental",
        profile="synthetic-ink@v1",
        realism="combined",
    )
    with pytest.raises(ValueError, match="reviewed ink recipe"):
        load_job(path)


@pytest.mark.parametrize("target", ["base_profile", "bank_directory", "source_crop"])
def test_recipe_rejects_paths_outside_its_snapshot(ink_candidate, target):
    _, folder = ink_candidate
    recipe = read(folder / "ink.json")
    if target == "base_profile":
        recipe["base_profile"] = "../outside"
    elif target == "bank_directory":
        recipe["banks"]["session_o"] = "../outside"
    else:
        manifest = read(folder / "banks/session_o/sources.json")
        manifest[0]["crop"] = "../../outside.png"
        atomic_json(folder / "banks/session_o/sources.json", manifest)
    atomic_json(folder / "ink.json", recipe)
    inventory(folder)
    with pytest.raises(ValueError, match="Unsafe asset path"):
        load_recipe(load_snapshot(folder))


def test_existing_but_unregistered_runtime_dependency_is_rejected(ink_candidate):
    _, folder = ink_candidate
    path = "banks/session_o/glyph.png"
    assert (folder / path).is_file()
    manifest = read(folder / "profile.json")
    manifest["assets"] = [a for a in manifest["assets"] if a["path"] != path]
    atomic_json(folder / "profile.json", manifest)
    with pytest.raises(ProfileError, match="absent from the Profile inventory"):
        load_recipe(load_snapshot(folder))


def test_recipe_cannot_request_a_role_without_an_accepted_source(ink_candidate):
    _, folder = ink_candidate
    recipe = read(folder / "ink.json")
    recipe["scan_roles"].append("Σ")
    atomic_json(folder / "ink.json", recipe)
    inventory(folder)
    with pytest.raises(ProfileError, match="no reviewed source"):
        load_recipe(load_snapshot(folder))


def test_rehashed_font_cannot_reuse_another_calibration(ink_candidate):
    _, folder = ink_candidate
    font = folder / "font.ttf"
    # Trailing padding remains a valid SFNT, but its identity no longer matches the recipe.
    font.write_bytes(font.read_bytes() + b"\0\0\0\0")
    inventory(folder)
    with pytest.raises(ProfileError, match="font calibration do not match"):
        load_recipe(load_snapshot(folder))


def test_nested_profile_cannot_hide_unregistered_assets(ink_candidate):
    _, folder = ink_candidate
    nested = read(folder / "nested/base/profile.json")
    nested["description"] = "changed after the recipe was frozen"
    atomic_json(folder / "nested/base/profile.json", nested)
    inventory(folder)
    with pytest.raises(ProfileError, match="Nested Profile identity changed"):
        load_recipe(load_snapshot(folder))


@pytest.mark.parametrize(
    "field,value", [("clean_isolated_ink", False), ("original_pixels_edited", True)]
)
def test_rehashed_source_metadata_cannot_admit_edited_ink(ink_candidate, field, value):
    _, folder = ink_candidate
    source = read(folder / "banks/session_o/sources.json")
    source[0][field] = value
    atomic_json(folder / "banks/session_o/sources.json", source)
    inventory(folder)
    with pytest.raises(ProfileError, match="reviewed unedited isolated"):
        load_recipe(load_snapshot(folder))


def test_candidate_import_stays_private_and_relocates_without_source_store(ink_candidate, tmp_path):
    source_store, folder = ink_candidate
    original = load_snapshot(folder)
    assert load_recipe(original).base.font.is_file()
    archive = tmp_path / "candidate.zip"
    source_store.export(original.manifest.ref, archive, candidate=True)
    destination = ProfileStore(tmp_path / "destination-store")
    destination.activate("bundled-default@v1")
    settings = (destination.root / "settings.json").read_bytes()
    result = destination.import_archive(archive)
    assert result == {"imported": original.manifest.ref, "activated": False, "candidate": True}
    assert destination.version_names("synthetic-ink") == []
    assert destination.version_names("synthetic-ink", "candidates") == ["v1"]
    assert (destination.root / "settings.json").read_bytes() == settings
    with pytest.raises(ProfileError, match="Published Profile not found"):
        destination.resolve(original.manifest.ref)
    source_store.root.rename(tmp_path / "retired-source-store")
    restored = destination.resolve(original.manifest.ref, candidate=True)
    assert restored.manifest_sha256 == original.manifest_sha256
    loaded = load_recipe(restored)
    assert all(p.is_relative_to(destination.root) for p in loaded.directories.values())
    assert loaded.base.font.is_relative_to(destination.root)
    assert loaded.strokes.font.is_relative_to(destination.root)


def test_personal_recipe_cannot_publish_or_activate_even_if_misfiled(ink_candidate):
    store, folder = ink_candidate
    store.activate("bundled-default@v1")
    settings = (store.root / "settings.json").read_bytes()
    with pytest.raises(ProfileError, match="personal ink"):
        publish(store, "synthetic-ink@v1", activate=True)
    assert store.version_names("synthetic-ink") == []
    # Even a manually misplaced experimental snapshot must not become the default.
    copy_snapshot(load_snapshot(folder), store.root / "profiles/synthetic-ink/versions/v1")
    with pytest.raises(ProfileError, match="cannot become the default"):
        store.activate("synthetic-ink@v1")
    assert (store.root / "settings.json").read_bytes() == settings


def test_font_cpu_refuses_a_misfiled_ink_recipe(ink_candidate, tmp_path):
    store, folder = ink_candidate
    copy_snapshot(load_snapshot(folder), store.root / "profiles/synthetic-ink/versions/v1")
    path = job_file(tmp_path / "input", backend="font-cpu", profile="synthetic-ink@v1")
    with pytest.raises(ValueError, match="require personal-cpu-experimental"):
        render_assignment(path, store=store, workspace=tmp_path / "workspace")
    assert not (tmp_path / "workspace/latest.json").exists()


def test_failed_candidate_pin_preserves_real_production_delivery(ink_candidate, tmp_path):
    store, _ = ink_candidate
    workspace = tmp_path / "workspace"
    path = job_file(tmp_path / "stable-input", profile="bundled-default@v1")
    stable = render_assignment(path, store=store, workspace=workspace)
    assert Path(stable["run"], "rendered/homework_solutions.pdf").is_file()
    pointer = (workspace / "latest.json").read_bytes()
    candidate = job_file(
        tmp_path / "candidate-input",
        backend="personal-cpu-experimental",
        profile="synthetic-ink@v1",
        profile_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="Pinned Profile digest does not match"):
        render_assignment(candidate, store=store, workspace=workspace, run_id="bad-candidate")
    assert (workspace / "latest.json").read_bytes() == pointer
    rejected = workspace / "runs/bad-candidate"
    assert verify_run(rejected)["run_status"] == "failed"
    assert not (rejected / "rendered/homework_solutions.pdf").exists()


def test_runtime_import_has_no_research_or_neural_dependency():
    script = """
import importlib.abc
import sys

forbidden = ('research', 'hwsol.training', 'torch', 'tensorflow', 'onnxruntime')
class CpuOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise ImportError('Forbidden runtime dependency: ' + fullname)
sys.meta_path.insert(0, CpuOnly())
from hwsol.personal_ink.runtime import PersonalCpuEngine
assert callable(PersonalCpuEngine.render_one)
assert not any(
    module == name or module.startswith(name + '.')
    for module in sys.modules for name in forbidden
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("license_label", ["local-only", "OFL-1.1 AND MIT"])
def test_export_reports_original_crops_regardless_of_license_label(
    ink_candidate, tmp_path, license_label
):
    import zipfile

    store, folder = ink_candidate
    manifest = read(folder / "profile.json")
    manifest["distribution"] = license_label
    atomic_json(folder / "profile.json", manifest)
    archive = tmp_path / "original-recipe.zip"
    result = store.export("synthetic-ink@v1", archive, candidate=True)
    assert "original crops" in result["contents"]
    assert "no raw samples" not in result["contents"]
    with zipfile.ZipFile(archive) as files:
        assert "banks/legacy/glyph.png" in files.namelist()
