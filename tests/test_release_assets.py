"""A release upload cannot silently replace tested bytes or change their source."""

import base64
import csv
import hashlib
import io
import json
import subprocess
import tarfile
from zipfile import ZipFile

import pytest

from scripts.package_release import stage_public_source, write_skill_archive
from scripts.verify_release_assets import archive_files, sha, verify


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def refresh_checksums(assets):
    names = sorted(p.name for p in assets.iterdir() if p.name.endswith((".whl", ".gz", ".zip")))
    names.append("release.json")
    (assets / "SHA256SUMS").write_text(
        "".join(f"{sha((assets / name).read_bytes())}  {name}\n" for name in names)
    )
    receipt = json.loads((assets / "PUBLICATION.json").read_bytes())
    receipt["sha256sums_sha256"] = sha((assets / "SHA256SUMS").read_bytes())
    write_json(assets / "PUBLICATION.json", receipt)


def rewrite_zip(path, files):
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


@pytest.fixture
def release(tmp_path):
    original = tmp_path / "development"
    original.mkdir()
    files = {
        "pyproject.toml": b'[project]\nname="handwritten-solutions"\nversion="1.0.0"\n',
        "uv.lock": b'[[package]]\nname="handwritten-solutions"\nversion="1.0.0"\n',
        "LICENSE": b"MIT license fixture\n",
        "src/hwsol/__init__.py": b'"""Public runtime fixture."""\n',
        "src/hwsol/assets/presets/patrick-hand/OFL.txt": b"OFL license fixture\n",
        "public_docs/distribution.json": b"",
    }
    build = {
        "commit": "a" * 40,
        "version": "1.0.0",
        "tag": "v1.0.0",
        "mode": "stable-tag",
        "requested_ref": "refs/tags/v1.0.0",
        "lock_sha256": sha(files["uv.lock"]),
        "git_history_included": False,
    }
    files["src/hwsol/_build.json"] = json.dumps(build).encode()
    policy = {
        "schema": "hwsol-public-distribution/1",
        "files": list(files),
        "patterns": [],
        "overlays": {},
    }
    files["public_docs/distribution.json"] = json.dumps(policy).encode()
    for name, data in files.items():
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source = tmp_path / "public"
    stage_public_source(original, source)
    frozen = {
        p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "Frozen public fixture",
        ],
        ["tag", "v1.0.0"],
    ):
        subprocess.run(["git", "-C", str(source), *args], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assets = tmp_path / "assets"
    assets.mkdir()
    # The archive must not contain the newly initialized Git checkout.
    staging = tmp_path / "archive-source"
    stage_public_source(original, staging)
    write_skill_archive(staging, assets / "handwritten-solutions-1.0.0-skill.zip")
    package_info = b"Name: handwritten-solutions\nVersion: 1.0.0\n\n"
    with tarfile.open(assets / "handwritten_solutions-1.0.0.tar.gz", "w:gz") as archive:
        for name, data in {**frozen, "PKG-INFO": package_info}.items():
            entry = tarfile.TarInfo(f"handwritten_solutions-1.0.0/{name}")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    wheel = {
        name.removeprefix("src/"): data
        for name, data in frozen.items()
        if name.startswith("src/hwsol/")
    }
    info = "handwritten_solutions-1.0.0.dist-info/"
    wheel.update(
        {
            info + "METADATA": package_info,
            info + "WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
            info + "entry_points.txt": b"",
            info + "top_level.txt": b"hwsol\n",
            info + "licenses/LICENSE": files["LICENSE"],
            info + "licenses/src/hwsol/assets/presets/patrick-hand/OFL.txt": files[
                "src/hwsol/assets/presets/patrick-hand/OFL.txt"
            ],
        }
    )
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in wheel.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, str(len(data))])
    writer.writerow([info + "RECORD", "", ""])
    wheel[info + "RECORD"] = record.getvalue().encode()
    rewrite_zip(assets / "handwritten_solutions-1.0.0-py3-none-any.whl", wheel)
    write_json(assets / "release.json", build)
    write_json(
        assets / "PUBLICATION.json",
        {
            "schema": "hwsol-publication/1",
            "tag": "v1.0.0",
            "version": "1.0.0",
            "public_source_commit": commit,
            "development_source_commit": build["commit"],
            "public_distribution_sha256": sha(frozen["PUBLIC_DISTRIBUTION.json"]),
        },
    )
    (assets / "RELEASE_NOTES.md").write_text("CPU release fixture.\n")
    refresh_checksums(assets)
    return source, assets, commit


def test_prebuilt_release_matches_actual_tagged_source(release):
    source, assets, commit = release
    result = verify(source, assets, "v1.0.0", commit)
    assert result["status"] == "passed"
    assert result["rebuilt"] is False
    assert result["public_source_commit"] != result["development_source_commit"]
    assert len(result["artifact_sha256"]) == 4


def test_changed_prebuilt_bytes_cannot_retain_acceptance(release):
    source, assets, commit = release
    wheel = assets / "handwritten_solutions-1.0.0-py3-none-any.whl"
    wheel.write_bytes(wheel.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(source, assets, "v1.0.0", commit)


@pytest.mark.parametrize("kind", ["wheel", "skill"])
def test_recomputed_hashes_cannot_hide_code_change(release, kind):
    source, assets, commit = release
    name = (
        "handwritten_solutions-1.0.0-py3-none-any.whl"
        if kind == "wheel"
        else "handwritten-solutions-1.0.0-skill.zip"
    )
    path = assets / name
    files = archive_files(path)
    key = "hwsol/__init__.py" if kind == "wheel" else "handwritten-solutions/src/hwsol/__init__.py"
    files[key] = b"CHANGED = True\n"
    rewrite_zip(path, files)
    refresh_checksums(assets)
    with pytest.raises(ValueError, match="differ|receipt"):
        verify(source, assets, "v1.0.0", commit)


def test_publication_cannot_claim_another_source_commit(release):
    source, assets, commit = release
    receipt = json.loads((assets / "PUBLICATION.json").read_bytes())
    receipt["public_source_commit"] = "b" * 40
    write_json(assets / "PUBLICATION.json", receipt)
    with pytest.raises(ValueError, match="does not bind"):
        verify(source, assets, "v1.0.0", commit)


def test_wheel_dependency_changes_cannot_hide_behind_recomputed_hashes(release):
    source, assets, commit = release
    path = assets / "handwritten_solutions-1.0.0-py3-none-any.whl"
    files = archive_files(path)
    key = "handwritten_solutions-1.0.0.dist-info/METADATA"
    files[key] = files[key].replace(b"\n\n", b"\nRequires-Dist: unreviewed-dependency\n\n")
    rewrite_zip(path, files)
    refresh_checksums(assets)
    with pytest.raises(ValueError, match="dependency metadata"):
        verify(source, assets, "v1.0.0", commit)


def test_annotated_tag_binds_peeled_public_commit(release):
    source, assets, commit = release
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "tag",
            "-fa",
            "v1.0.0",
            "-m",
            "Release",
        ],
        check=True,
        capture_output=True,
    )
    assert verify(source, assets, "v1.0.0", commit)["public_source_commit"] == commit


@pytest.mark.parametrize("kind", ["extra", "private-note", "symlink", "changed-source"])
def test_publication_boundary_refuses_extra_private_or_linked_content(release, kind):
    source, assets, commit = release
    if kind == "extra":
        (assets / "unapproved.txt").write_text("Not in the public upload manifest")
    elif kind == "private-note":
        (assets / "RELEASE_NOTES.md").write_text("/" + "home" + "/private-user/work/\n")
    elif kind == "symlink":
        (assets / "RELEASE_NOTES.md").unlink()
        (assets / "RELEASE_NOTES.md").symlink_to(source / "LICENSE")
    else:
        (source / "src/hwsol/__init__.py").write_text("UNREVIEWED = True\n")
    with pytest.raises(ValueError, match="Unexpected|Private|changed"):
        verify(source, assets, "v1.0.0", commit)


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "root/../escape.txt"])
def test_archive_paths_never_escape_before_audit(tmp_path, name):
    path = tmp_path / "unsafe.zip"
    rewrite_zip(path, {name: b"unsafe"})
    with pytest.raises(ValueError, match="Unsafe"):
        archive_files(path)
