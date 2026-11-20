"""Synthetic local-authority contract tests, NOT native or author-style evidence.

The JSON Job/native witnesses below are explicitly synthetic. verify_job is a unit
boundary stub; actual PDF verification remains covered by Job tests and the separate
real-profile/native acceptance runs. No mock is reported as a real model run.
"""

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest
from test_personal_cpu import ink_candidate  # noqa: F401 -- shared public synthetic fixture

import hwsol.profiles.qualification as q
from hwsol.identity import digest
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore, load_snapshot


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def evidence(request, tmp_path, monkeypatch):
    store, folder = request.getfixturevalue("ink_candidate")
    loaded = load_snapshot(folder)
    recipe_sha = sha256(folder / "ink.json")
    root = tmp_path / "synthetic-evidence"
    root.mkdir()
    source = b"# synthetic code identity; not a built runtime\n"
    current = {
        "commit": "a" * 40,
        "working_tree_dirty": False,
        "lock_sha256": "b" * 64,
        "source": {"sample.py": hashlib.sha256(source).hexdigest()},
        "handnote_source": {"synthetic.py": "c" * 64},
        "packages": {"handwritten-solutions": "1.0.0rc2"},
        "renderer_fingerprint": "d" * 64,
    }
    capabilities = {
        "system": "Linux",
        "machine": "synthetic-machine",
        "python_minor": "3.12",
        "pillow_version": "synthetic",
        "layout_engine": "BASIC",
        "libraries": {
            k: {"enabled": k == "freetype2", "version": "test" if k == "freetype2" else None}
            for k in ("raqm", "harfbuzz", "fribidi", "freetype2")
        },
    }
    monkeypatch.setattr(q, "build_identity", lambda: copy.deepcopy(current))
    monkeypatch.setattr(q, "runtime_text_capabilities", lambda font: copy.deepcopy(capabilities))
    verifications = []

    def synthetic_job_boundary(directory, *, pixels=False):
        assert (directory / "manifest.json").is_file()
        assert read(directory / "validation_report.json")["status"] == "passed"
        verifications.append((directory, pixels))
        return {"status": "passed", "page_count": 1}

    monkeypatch.setattr("hwsol.jobs.runtime.verify_job", synthetic_job_boundary)
    wheel = root / "synthetic.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("hwsol/sample.py", source)
        archive.writestr(
            "hwsol/_build.json",
            json.dumps({"commit": current["commit"], "lock_sha256": current["lock_sha256"]}),
        )
        archive.writestr(
            "synthetic.dist-info/METADATA", "Name: handwritten-solutions\nVersion: 1.0.0rc2\n"
        )

    def file(name):
        return {"path": name, "sha256": sha256(root / name)}

    platforms = []
    for system in ("Linux", "Darwin"):
        env = {**capabilities, "system": system}
        directory = root / system / "job"
        directory.mkdir(parents=True)
        atomic_json(
            directory / "manifest.json",
            {"profile": loaded.manifest.ref, "profile_sha256": loaded.manifest_sha256},
        )
        report = {
            "status": "passed",
            "profile": loaded.provenance(),
            "build": current,
            "pages": [
                {
                    "candidate": {
                        "recipe_sha256": recipe_sha,
                        "semantic_identity_preserved": True,
                        "parser_trace_preserved": True,
                        "whole_recipe_contact_audit": {"unreviewed_contact_count": 0},
                    }
                }
            ],
        }
        atomic_json(directory / "validation_report.json", report)
        report_ref = file(f"{system}/job/validation_report.json")
        receipt = {
            "schema": "cross-platform-candidate-check/1",
            "status": "passed",
            "actual_system": system,
            "source_commit": current["commit"],
            "package_version": "1.0.0rc2",
            "installed_wheel_members_exact": 2,
            "inputs": {"wheel": file("synthetic.whl")},
            "clean_install_completed": True,
            "qualification_environment": env,
            "runtime": {"system": system, "hwsol": "/synthetic/fresh/runtime/hwsol/__init__.py"},
            "outputs": [
                {
                    "report_sha256": report_ref["sha256"],
                    "profile": {"sha256": loaded.manifest_sha256},
                }
            ],
        }
        atomic_json(root / f"{system}/receipt.json", receipt)
        platforms.append(
            {
                "environment": env,
                "renderer_fingerprint": current["renderer_fingerprint"],
                "receipt": file(f"{system}/receipt.json"),
                "job": {"directory": f"{system}/job", "report": report_ref},
            }
        )
    reports = [p["job"]["report"]["sha256"] for p in platforms]
    atomic_json(
        root / "author.json",
        {
            "schema": "personal-cpu-author-review/1",
            "profile_sha256": loaded.manifest_sha256,
            "kind": "direct-pages",
            "approved": True,
            "statement": "SYNTHETIC contract fixture; not a real author approval",
            "job_report_sha256": reports,
        },
    )
    atomic_json(
        root / "math.json",
        {
            "schema": "personal-cpu-math-review/1",
            "independent_math_verified": True,
            "statement": "SYNTHETIC contract fixture; not actual solving evidence",
            "job_report_sha256": reports,
        },
    )
    request = {
        "schema": "personal-cpu-qualification-request/1",
        "profile": loaded.manifest.ref,
        "profile_sha256": loaded.manifest_sha256,
        "recipe_sha256": recipe_sha,
        "build": {
            "commit": current["commit"],
            "version": "1.0.0rc2",
            "lock_sha256": current["lock_sha256"],
            "source_sha256": digest(current["source"]),
            "handnote_source_sha256": digest(current["handnote_source"]),
            "wheel": file("synthetic.whl"),
        },
        "platforms": platforms,
        "author": file("author.json"),
        "math_review": file("math.json"),
        "files": {
            p.relative_to(root).as_posix(): sha256(p) for p in root.rglob("*") if p.is_file()
        },
    }
    path = root / "request.json"
    atomic_json(path, request)
    store.activate("bundled-default@v1")
    return {
        "store": store,
        "folder": folder,
        "loaded": loaded,
        "root": root,
        "request": request,
        "path": path,
        "current": current,
        "capabilities": capabilities,
        "verifications": verifications,
    }


def resign(e, relative, value):
    atomic_json(e["root"] / relative, value)
    digest_value = sha256(e["root"] / relative)
    e["request"]["files"][relative] = digest_value
    return {"path": relative, "sha256": digest_value}


def save(e):
    atomic_json(e["path"], e["request"])


def accepted(e):
    return q.qualify(e["store"], e["loaded"].manifest.ref, e["path"])["qualification_id"]


def test_qualification_publish_activation_are_three_separate_operations(evidence, tmp_path):
    e = evidence
    settings = (e["store"].root / "settings.json").read_bytes()
    identifier = accepted(e)
    assert e["verifications"] and any(pixels for _, pixels in e["verifications"])
    assert e["store"].version_names("synthetic-ink") == []
    assert (e["store"].root / "settings.json").read_bytes() == settings
    q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    assert (e["store"].root / "settings.json").read_bytes() == settings
    assert (
        e["store"].resolve(e["loaded"].manifest.ref, candidate=True).manifest_sha256
        == e["loaded"].manifest_sha256
    )
    e["store"].activate(e["loaded"].manifest.ref)
    assert e["store"].settings()["default_profile"] == e["loaded"].manifest.ref
    e["store"].activate("bundled-default@v1")
    assert e["store"].settings()["default_profile"] == "bundled-default@v1"
    archive = tmp_path / "export.zip"
    e["store"].export(e["loaded"].manifest.ref, archive)
    other = ProfileStore(tmp_path / "fresh-import")
    assert other.import_archive(archive)["candidate"] is True
    assert other.version_names("synthetic-ink") == []
    with ZipFile(archive) as z:
        assert not any("qualification" in n or "publication" in n for n in z.namelist())
    with pytest.raises(ProfileError, match="Published Profile not found"):
        other.activate(e["loaded"].manifest.ref)


@pytest.mark.parametrize(
    "target",
    ["profile", "recipe", "commit", "dirty", "engine", "library", "author", "math", "evidence"],
)
def test_wrong_or_stale_bindings_fail_without_side_effects(evidence, target):
    e = evidence
    settings = (e["store"].root / "settings.json").read_bytes()
    if target == "profile":
        e["request"]["profile_sha256"] = "0" * 64
    elif target == "recipe":
        e["request"]["recipe_sha256"] = "0" * 64
    elif target == "commit":
        e["current"]["commit"] = "0" * 40
    elif target == "dirty":
        e["current"]["working_tree_dirty"] = True
    elif target == "engine":
        e["capabilities"]["layout_engine"] = "RAQM"
    elif target == "library":
        e["capabilities"]["libraries"]["freetype2"]["version"] = "changed"
    elif target in {"author", "math"}:
        key = "author" if target == "author" else "math_review"
        row = read(e["root"] / f"{target}.json")
        row["job_report_sha256"] = ["0" * 64]
        e["request"][key] = resign(e, f"{target}.json", row)
    else:
        (e["root"] / "author.json").write_text("{}")
    save(e)
    with pytest.raises((ProfileError, ValueError)):
        accepted(e)
    assert (e["store"].root / "settings.json").read_bytes() == settings
    assert not e["store"].version_names("synthetic-ink")


def test_missing_native_platform_and_arbitrary_admission_cannot_qualify(evidence):
    e = evidence
    e["request"]["platforms"][1]["environment"]["system"] = "Linux"
    save(e)
    with pytest.raises(ProfileError):
        accepted(e)
    assert not e["store"].version_names("synthetic-ink")


def test_actual_wheel_contents_checked_even_if_all_file_hashes_are_rebound(evidence):
    e = evidence
    path = e["root"] / "synthetic.whl"
    with ZipFile(path, "w") as z:
        z.writestr("hwsol/sample.py", b"# wrong build")
        z.writestr("hwsol/_build.json", json.dumps({"commit": "a" * 40, "lock_sha256": "b" * 64}))
        z.writestr(
            "synthetic.dist-info/METADATA", "Name: handwritten-solutions\nVersion: 1.0.0rc2\n"
        )
    e["request"]["files"]["synthetic.whl"] = sha256(path)
    e["request"]["build"]["wheel"]["sha256"] = sha256(path)
    save(e)
    with pytest.raises(ProfileError, match="Actual wheel"):
        accepted(e)


def test_publish_crash_is_inert_and_retryable(evidence, monkeypatch):
    e = evidence
    identifier = accepted(e)
    original = q.atomic_json

    def crash(path, value):
        if path.parent.name == "publications":
            raise OSError("synthetic publication interruption")
        original(path, value)

    monkeypatch.setattr(q, "atomic_json", crash)
    with pytest.raises(OSError, match="interruption"):
        q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    loaded = e["store"].resolve(e["loaded"].manifest.ref)
    assert loaded.manifest_sha256 == e["loaded"].manifest_sha256
    with pytest.raises(ProfileError, match="without local publication"):
        q.validate_published(e["store"], loaded)
    assert e["store"].settings()["default_profile"] == "bundled-default@v1"
    monkeypatch.setattr(q, "atomic_json", original)
    q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    with pytest.raises(ProfileError, match="immutable"):
        q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)


def test_published_evidence_changed_rejects_activation_and_runtime_guard(evidence):
    e = evidence
    identifier = accepted(e)
    q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    folder = e["store"].profile_dir("synthetic-ink") / "qualifications" / identifier
    (folder / "evidence/author.json").write_text("{}")
    with pytest.raises(ProfileError, match="Evidence absent or changed"):
        e["store"].activate(e["loaded"].manifest.ref)
    with pytest.raises(ProfileError, match="Evidence absent or changed"):
        q.validate_published(e["store"], e["store"].resolve(e["loaded"].manifest.ref))
    assert e["store"].settings()["default_profile"] == "bundled-default@v1"


def test_writer_lock_prevents_partial_qualification(evidence):
    e = evidence
    with e["store"].writer("synthetic-ink"):
        with pytest.raises(ProfileError, match="Another writer"):
            accepted(e)
    assert not e["store"].version_names("synthetic-ink")


def test_waiver_is_not_a_general_style_permission():
    with pytest.raises(ValueError):
        q.GapWaiver.model_validate(
            {
                "schema": "personal-cpu-gap-waiver/1",
                "before_scale": 1.06,
                "after_scale": 1.2,
                "increment": 1.08,
            }
        )


def test_real_text_capability_probe_reports_selected_layout_and_libraries(tmp_path):
    from hwsol.identity import runtime_text_capabilities

    profile = ProfileStore(tmp_path / "store").resolve("bundled-default@v1")
    result = runtime_text_capabilities(profile.font)
    parsed = q.Environment.model_validate(result)
    assert parsed.layout_engine in {"BASIC", "RAQM"}
    assert parsed.libraries.freetype2.enabled is True


@pytest.fixture
def scoped_waiver(evidence):
    """Synthetic policy fixtures with distinct Linux d and native Mac f decisions."""
    from types import SimpleNamespace

    e = evidence
    original = e["loaded"].manifest
    base = original.model_copy(
        update={
            "version": "v3",
            "render": original.render.model_copy(update={"semantic_layout": True}),
        }
    )
    before = base.model_copy(
        update={
            "version": "v4",
            "parent": base.ref,
            "render": base.render.model_copy(update={"word_gap_scale": 1.06}),
        }
    )
    after = before.model_copy(
        update={
            "version": "v5",
            "parent": before.ref,
            "render": before.render.model_copy(update={"word_gap_scale": 1.1448}),
        }
    )
    base_ref = resign(e, "base-profile.json", base.model_dump(mode="json"))
    before_ref = resign(e, "prior-profile.json", before.model_dump(mode="json"))
    after_ref = resign(e, "after-profile.json", after.model_dump(mode="json"))
    loaded = SimpleNamespace(manifest=after, manifest_sha256=after_ref["sha256"])
    font_sha = next(a.sha256 for a in after.assets if a.path == after.font)
    problems = [
        {
            "id": "synthetic-source",
            "number": 1,
            "statement": "Synthetic question",
            "solution": "answer.md",
            "sources": [],
        }
    ]

    def job(directory, profile, rows):
        resign(e, directory + "/manifest.json", {"problems": problems})
        (e["root"] / directory / "answer.md").write_text("Synthetic identical answer\n")
        e["request"]["files"][directory + "/answer.md"] = sha256(
            e["root"] / directory / "answer.md"
        )
        report = {
            "status": "passed",
            "profile": {"manifest_sha256": profile["sha256"]},
            "pages": [{"glyph_provenance": rows, "formula_groups": [], "matrix_groups": []}],
        }
        return {
            "directory": directory,
            "report": resign(e, directory + "/validation_report.json", report),
        }

    base_job = job("base-job", base_ref, [])
    prior_ref = resign(
        e,
        "prior-review.json",
        {
            "schema": "personal-cpu-prior-review/1",
            "basis": "accepted-base-plus-prescribed-spacing",
            "author_accepted_base": True,
            "base_profile": base_ref,
            "base_job": base_job,
            "prescribed_parent_profile_sha256": before_ref["sha256"],
            "statement": "SYNTHETIC accepted base and prescribed revision",
        },
    )
    changes, reports = [], []
    for platform in e["request"]["platforms"]:
        system = platform["environment"]["system"]
        semantic = "d" if system == "Linux" else "f"
        row = {
            "semantic": semantic,
            "context": "text",
            "item_index": 0,
            "source_type": "personal-outline",
            "actual_glyph_id": 1,
            "glyph_name": semantic,
            "variant": 0,
            "font_sha256": font_sha,
            "bounds_px": [0.0, 0.0, 10.0, 20.0],
        }
        revised = {
            **row,
            "source_type": "personal-outline-geometric-calibration",
            "bounds_px": [5.0, 1.0, 15.0, 23.0],
            "variant": 1,
        }
        prior_job = job(system + "/before", before_ref, [row])
        platform["job"] = job(system + "/after", after_ref, [revised])
        reports.append(platform["job"]["report"]["sha256"])
        approval = {
            "schema": "personal-cpu-layout-change-review/1",
            "environment": platform["environment"],
            "profile_sha256": after_ref["sha256"],
            "prior_profile_sha256": before_ref["sha256"],
            "before_report_sha256": prior_job["report"]["sha256"],
            "after_report_sha256": platform["job"]["report"]["sha256"],
            "independent_review_passed": True,
            "statement": "SYNTHETIC environment-specific reviewed selection",
            "accepted_changes": [
                {
                    "page": 1,
                    "item": 0,
                    "semantic": semantic,
                    "before": q._glyph_source(row),
                    "after": q._glyph_source(revised),
                    "before_size": [10.0, 20.0],
                    "after_size": [10.0, 22.0],
                }
            ],
        }
        changes.append(
            {
                "environment": platform["environment"],
                "prior_job": prior_job,
                "review": resign(e, system + "/change.json", approval),
            }
        )
    waiver = {
        "schema": "personal-cpu-gap-waiver/1",
        "scope": "word-gap-only-with-existing-rule-selection",
        "profile_sha256": after_ref["sha256"],
        "prior_profile": before_ref,
        "before_scale": 1.06,
        "after_scale": 1.1448,
        "increment": 1.08,
        "changes": changes,
        "job_report_sha256": reports,
    }
    author = {
        "schema": "personal-cpu-author-review/1",
        "profile_sha256": after_ref["sha256"],
        "kind": "scoped-change-waiver",
        "approved": True,
        "statement": "SYNTHETIC spacing-only waiver",
        "job_report_sha256": reports,
        "prior_review": prior_ref,
        "change_report": resign(e, "waiver.json", waiver),
        "scope": "word-gap-only-with-existing-rule-selection",
    }
    return e, loaded, author, waiver, reports


def check_waiver(fixture):
    e, loaded, author, waiver, reports = fixture
    author["change_report"] = resign(e, "waiver.json", waiver)
    q._validate_waiver(
        loaded,
        q.Request.model_validate(e["request"]),
        e["root"],
        q.AuthorReview.model_validate(author),
        reports,
        pixels=True,
    )


def test_native_waiver_accepts_distinct_reviewed_existing_d_and_f_decisions(scoped_waiver):
    check_waiver(scoped_waiver)


@pytest.mark.parametrize(
    "change",
    [
        "scale",
        "font",
        "source",
        "parent",
        "missing-environment",
        "unreviewed-shape",
        "unreviewed-variant",
        "false-approval",
        "different-answer",
        "wrong-base-review",
        "cross-os-baseline",
    ],
)
def test_scoped_waiver_rejects_unprescribed_changes(scoped_waiver, change):
    e, loaded, author, waiver, reports = scoped_waiver
    if change == "scale":
        loaded.manifest = loaded.manifest.model_copy(
            update={"render": loaded.manifest.render.model_copy(update={"word_gap_scale": 1.2})}
        )
    elif change == "font":
        loaded.manifest = loaded.manifest.model_copy(update={"font": "nested/base/font.ttf"})
    elif change == "parent":
        loaded.manifest = loaded.manifest.model_copy(update={"parent": "synthetic-ink@v3"})
    elif change == "missing-environment":
        waiver["changes"][1] = copy.deepcopy(waiver["changes"][0])
    elif change == "wrong-base-review":
        prior = read(e["root"] / "prior-review.json")
        prior["author_accepted_base"] = False
        author["prior_review"] = resign(e, "prior-review.json", prior)
    elif change == "different-answer":
        (e["root"] / "Darwin/after/answer.md").write_text("A different answer\n")
    elif change == "cross-os-baseline":
        waiver["changes"][1]["prior_job"] = copy.deepcopy(waiver["changes"][0]["prior_job"])
    elif change == "false-approval":
        approval = read(e["root"] / "Darwin/change.json")
        approval["independent_review_passed"] = False
        waiver["changes"][1]["review"] = resign(e, "Darwin/change.json", approval)
    else:
        path = "Darwin/after/validation_report.json"
        row = read(e["root"] / path)
        glyph = row["pages"][0]["glyph_provenance"][0]
        if change == "unreviewed-shape":
            glyph["bounds_px"][3] += 1
        elif change == "unreviewed-variant":
            glyph["variant"] = 8
        else:
            glyph["font_sha256"] = "f" * 64
        e["request"]["platforms"][1]["job"]["report"] = resign(e, path, row)
        reports[1] = e["request"]["platforms"][1]["job"]["report"]["sha256"]
        waiver["job_report_sha256"] = reports
        approval = read(e["root"] / "Darwin/change.json")
        approval["after_report_sha256"] = reports[1]
        waiver["changes"][1]["review"] = resign(e, "Darwin/change.json", approval)
    with pytest.raises((ProfileError, ValueError)):
        check_waiver(scoped_waiver)


@pytest.mark.parametrize("mutation", [None, "wheel", "runtime", "engine", "bool-member-count"])
def test_external_clean_install_is_bound_to_wheel_and_runtime_not_old_profile(evidence, mutation):
    e = evidence
    platform = e["request"]["platforms"][0]
    native = read(e["root"] / "Linux/receipt.json")
    native["clean_install_completed"] = False
    parent = copy.deepcopy(native)
    parent["outputs"][0]["profile"]["sha256"] = "0" * 64  # Installation may precede v5.
    parent_ref = resign(e, "clean-parent.json", parent)
    external = [
        {
            "python": "3.12",
            "status": "passed",
            "clean_install_completed": True,
            "source_commit": e["current"]["commit"],
            "package_version": "1.0.0rc2",
            "wheel_sha256": e["request"]["build"]["wheel"]["sha256"],
            "portable_receipt_sha256": parent_ref["sha256"],
            "fresh_environment": "/synthetic/fresh",
        }
    ]
    if mutation == "wheel":
        external[0]["wheel_sha256"] = "0" * 64
    elif mutation == "runtime":
        native["runtime"]["hwsol"] = "/different/env/hwsol/__init__.py"
    elif mutation == "engine":
        native["qualification_environment"]["layout_engine"] = "RAQM"
    elif mutation == "bool-member-count":
        parent["installed_wheel_members_exact"] = True
        parent_ref = resign(e, "clean-parent.json", parent)
        external[0]["portable_receipt_sha256"] = parent_ref["sha256"]
    platform["receipt"] = resign(e, "Linux/receipt.json", native)
    platform["installation"] = resign(e, "installation.json", external)
    platform["installation_portable"] = parent_ref
    save(e)
    if mutation is None:
        assert accepted(e)
    else:
        with pytest.raises(ProfileError, match="installation proof"):
            accepted(e)


@pytest.mark.parametrize("drift", ["build", "engine"])
def test_published_runtime_drift_blocks_activation_and_normal_runtime_guard(evidence, drift):
    e = evidence
    identifier = accepted(e)
    q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    if drift == "build":
        e["current"]["commit"] = "e" * 40
    else:
        e["capabilities"]["layout_engine"] = "RAQM"
    with pytest.raises(ProfileError):
        e["store"].activate(e["loaded"].manifest.ref)
    with pytest.raises(ProfileError):
        q.validate_published(e["store"], e["store"].resolve(e["loaded"].manifest.ref))
    assert e["store"].settings()["default_profile"] == "bundled-default@v1"


def test_normal_job_dispatch_requires_qualification_and_reviewed_configuration(
    evidence, monkeypatch
):
    from test_personal_cpu import job_file

    from hwsol.jobs.runtime import build
    from hwsol.personal_ink.runtime import PersonalCpuEngine

    e = evidence
    identifier = accepted(e)
    q.publish_ink(e["store"], e["loaded"].manifest.ref, identifier)
    path = job_file(e["root"], backend="font-cpu", profile=e["loaded"].manifest.ref, render={})
    captured = []

    class RoutedToQualifiedEngine(Exception):
        pass

    def stop_at_engine(self, profile, *, qualification=None):
        captured.append((profile.manifest_sha256, qualification))
        raise RoutedToQualifiedEngine

    monkeypatch.setattr(PersonalCpuEngine, "__init__", stop_at_engine)
    with pytest.raises(RoutedToQualifiedEngine):
        build(path, e["root"] / "not-rendered", e["store"], {})
    assert captured[0][0] == e["loaded"].manifest_sha256
    assert captured[0][1]["strategy"] == "personal-cpu-qualified"
    assert e["store"].settings()["default_profile"] == "bundled-default@v1"
    data = read(path)
    data["render"] = {"dpi": 299}
    atomic_json(path, data)
    with pytest.raises(ValueError, match="reviewed Profile render settings"):
        build(path, e["root"] / "not-rendered", e["store"], {})
    assert len(captured) == 1
