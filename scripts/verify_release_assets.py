"""Verify prebuilt release bytes against the tagged public source; never rebuild."""

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import re
import stat
import subprocess
import tarfile
import tomllib
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

try:
    from scripts.package_release import PRIVATE_TEXT, audit_paths, audit_public_files
except ModuleNotFoundError:  # Direct invocation from the public source checkout.
    from package_release import PRIVATE_TEXT, audit_paths, audit_public_files


SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})|"
    rb"AKIA[0-9A-Z]{16}"
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def checked_text(name: str, data: bytes) -> None:
    if SECRET.search(data):
        raise ValueError(f"Potential credential in public file: {name}")
    if name.endswith((".md", ".json", ".txt", ".xml", ".md.in", "METADATA", "PKG-INFO")):
        if PRIVATE_TEXT.search(data.decode("utf-8", errors="replace")):
            raise ValueError(f"Private operational reference: {name}")


def archive_files(path: Path, prefix: str = "") -> dict[str, bytes]:
    """Read archives without extraction; reject duplicate, linked or unsafe entries."""
    files = {}
    names = set()

    def add(name: str, data: bytes | None) -> None:
        member = PurePosixPath(name)
        if (
            member.is_absolute()
            or ".." in member.parts
            or "\\" in name
            or name in names
            or str(member) != name.rstrip("/")
        ):
            raise ValueError(f"Unsafe or duplicate archive entry: {name}")
        names.add(name)
        if prefix:
            if member == PurePosixPath(prefix) and data is None:
                return
            if not name.startswith(prefix + "/"):
                raise ValueError(f"Unexpected archive root: {name}")
            name = name[len(prefix) + 1 :]
        if data is not None:
            files[name] = data

    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            for entry in archive:
                if not entry.isfile() and not entry.isdir():
                    raise ValueError(f"Linked or special tar entry: {entry.name}")
                add(entry.name, archive.extractfile(entry).read() if entry.isfile() else None)
    else:
        with ZipFile(path) as archive:
            for entry in archive.infolist():
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"Linked zip entry: {entry.filename}")
                add(entry.filename, None if entry.is_dir() else archive.read(entry))
    for name, data in files.items():
        checked_text(name, data)
    return files


def verify_wheel(files: dict[str, bytes], source: dict[str, bytes], version: str) -> None:
    audit_paths("src/" + n if n.startswith("hwsol/") else n for n in files)
    expected = {
        name.removeprefix("src/"): data
        for name, data in source.items()
        if name.startswith("src/hwsol/")
        and (
            name.endswith(".py")
            or name == "src/hwsol/_build.json"
            or name.startswith("src/hwsol/assets/presets/patrick-hand/")
            or name.startswith("src/hwsol/assets/benchmarks/v1/")
        )
    }
    actual = {name: data for name, data in files.items() if name.startswith("hwsol/")}
    if actual != expected:
        raise ValueError("Wheel runtime files differ from the frozen public source")
    info = f"handwritten_solutions-{version}.dist-info/"
    metadata = {
        "METADATA",
        "WHEEL",
        "entry_points.txt",
        "top_level.txt",
        "RECORD",
        "licenses/LICENSE",
        "licenses/src/hwsol/assets/presets/patrick-hand/OFL.txt",
    }
    if set(files) - set(expected) != {info + n for n in metadata}:
        raise ValueError("Unexpected or missing wheel metadata")
    message = BytesParser().parsebytes(files[info + "METADATA"])
    if message["Name"] != "handwritten-solutions" or message["Version"] != version:
        raise ValueError("Wheel package name/version mismatch")
    project = tomllib.loads(source["pyproject.toml"].decode())["project"]
    declared = sorted(re.sub(r"\s+", "", d) for d in project.get("dependencies", []))
    installed = sorted(re.sub(r"\s+", "", d) for d in message.get_all("Requires-Dist", []))
    python_declared = sorted(project.get("requires-python", "").replace(" ", "").split(","))
    python_installed = sorted((message["Requires-Python"] or "").replace(" ", "").split(","))
    if declared != installed or python_installed != python_declared:
        raise ValueError("Wheel dependency metadata differs from frozen source")
    if message["License-Expression"] != project.get("license"):
        raise ValueError("Wheel license metadata differs from frozen source")
    entry_points = configparser.ConfigParser(interpolation=None)
    entry_points.optionxform = str
    entry_points.read_string(files[info + "entry_points.txt"].decode())
    expected_entries = {"console_scripts": project["scripts"]} if project.get("scripts") else {}
    actual_entries = {section: dict(entry_points[section]) for section in entry_points.sections()}
    if actual_entries != expected_entries:
        raise ValueError("Wheel command entry points differ from frozen source")
    for target, origin in (
        ("licenses/LICENSE", "LICENSE"),
        (
            "licenses/src/hwsol/assets/presets/patrick-hand/OFL.txt",
            "src/hwsol/assets/presets/patrick-hand/OFL.txt",
        ),
    ):
        if files[info + target] != source[origin]:
            raise ValueError("Wheel license differs from frozen source")
    records = list(csv.reader(io.StringIO(files[info + "RECORD"].decode())))
    if any(len(row) != 3 for row in records) or len({r[0] for r in records}) != len(records):
        raise ValueError("Invalid wheel RECORD")
    if {row[0] for row in records} != set(files):
        raise ValueError("Incomplete wheel RECORD")
    for name, digest, size in records:
        if name == info + "RECORD":
            if digest or size:
                raise ValueError("Invalid wheel RECORD self-entry")
            continue
        expected_hash = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest())
        if digest != "sha256=" + expected_hash.rstrip(b"=").decode() or size != str(
            len(files[name])
        ):
            raise ValueError(f"Wheel RECORD differs from bytes: {name}")


def verify(source: Path, assets: Path, tag: str, expected_commit: str) -> dict:
    if not re.fullmatch(r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", tag):
        raise ValueError("Only canonical stable release tags are accepted")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("Expected public commit must be a full Git SHA")
    commit = git(source, "rev-parse", "HEAD")
    if (
        commit != expected_commit
        or git(source, "rev-parse", f"refs/tags/{tag}^{{commit}}") != commit
    ):
        raise ValueError("Public checkout, tag and expected commit disagree")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked public checkout has changed")
    paths = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode()
    source_files = {}
    for name in paths.split("\0"):
        if not name:
            continue
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Linked or missing public source: {name}")
        source_files[name] = path.read_bytes()
        checked_text(name, source_files[name])
    audit_public_files(source_files)
    version = tag[1:]
    project = tomllib.loads(source_files["pyproject.toml"].decode())["project"]
    if project["name"] != "handwritten-solutions" or project["version"] != version:
        raise ValueError("Tagged project version mismatch")
    locked = tomllib.loads(source_files["uv.lock"].decode())
    own = [p for p in locked["package"] if p["name"] == "handwritten-solutions"]
    if len(own) != 1 or own[0]["version"] != version:
        raise ValueError("Tagged lockfile version mismatch")
    build = json.loads(source_files["src/hwsol/_build.json"])
    if (
        build["version"] != version
        or build["tag"] != tag
        or build["mode"] != "stable-tag"
        or not re.fullmatch(r"[0-9a-f]{40}", build["commit"])
        or build["lock_sha256"] != sha(source_files["uv.lock"])
        or build["git_history_included"] is not False
    ):
        raise ValueError("Embedded development build identity mismatch")
    archive_names = {
        f"handwritten_solutions-{version}-py3-none-any.whl",
        f"handwritten_solutions-{version}.tar.gz",
        f"handwritten-solutions-{version}-skill.zip",
    }
    hashed_names = archive_names | {"release.json"}
    allowed = hashed_names | {"SHA256SUMS", "PUBLICATION.json", "RELEASE_NOTES.md"}
    present = {p.name for p in assets.iterdir() if p.name != ".git"}
    if present != allowed or any(
        (assets / n).is_symlink() or not (assets / n).is_file() for n in allowed
    ):
        raise ValueError("Unexpected, linked or missing publication files")
    publication = json.loads((assets / "PUBLICATION.json").read_bytes())
    expected = {
        "schema": "hwsol-publication/1",
        "tag": tag,
        "version": version,
        "public_source_commit": commit,
        "development_source_commit": build["commit"],
        "public_distribution_sha256": sha(source_files["PUBLIC_DISTRIBUTION.json"]),
        "sha256sums_sha256": sha((assets / "SHA256SUMS").read_bytes()),
    }
    if publication != expected:
        raise ValueError("Publication receipt does not bind this source and artifact manifest")
    sums = {}
    for line in (assets / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match or match[2] in sums:
            raise ValueError("Invalid or duplicate checksum manifest entry")
        sums[match[2]] = match[1]
    if set(sums) != hashed_names:
        raise ValueError(
            "Checksum manifest must contain exactly the three archives and release.json"
        )
    for name in allowed:
        data = (assets / name).read_bytes()
        checked_text(name, data)
        if name in sums and sha(data) != sums[name]:
            raise ValueError(f"Prebuilt asset checksum mismatch: {name}")
    if json.loads((assets / "release.json").read_bytes()) != build:
        raise ValueError("Release receipt differs from embedded build identity")
    skill = archive_files(
        assets / f"handwritten-solutions-{version}-skill.zip", "handwritten-solutions"
    )
    audit_public_files(skill)
    if skill != source_files:
        raise ValueError("Skill archive differs from tagged public source")
    sdist = archive_files(
        assets / f"handwritten_solutions-{version}.tar.gz", f"handwritten_solutions-{version}"
    )
    audit_public_files(sdist)
    if any(sdist.get(name) != data for name, data in source_files.items()):
        raise ValueError("Source archive differs from tagged public source")
    meta = BytesParser().parsebytes(sdist["PKG-INFO"])
    if meta["Name"] != "handwritten-solutions" or meta["Version"] != version:
        raise ValueError("Source archive package metadata mismatch")
    verify_wheel(
        archive_files(assets / f"handwritten_solutions-{version}-py3-none-any.whl"),
        source_files,
        version,
    )
    return {"status": "passed", **expected, "artifact_sha256": sums, "rebuilt": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.assets, args.tag, args.expected_commit), indent=2))
