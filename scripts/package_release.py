"""Build audited public snapshots from a version tag or a local prerelease commit.

Private working documents and Git history are never distribution artifacts.
The public policy and documentation must exist in the selected ref itself.
"""

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
FONT = "src/hwsol/assets/presets/patrick-hand/PatrickHand-Regular.ttf"
AUTHOR_ROOT = "src/hwsol/assets/presets/author-default/"
AUTHOR_FONTS = {
    AUTHOR_ROOT + "ink/base-profile/font.ttf",
    AUTHOR_ROOT + "ink/stroke-profile/font.ttf",
    AUTHOR_ROOT + "ink/banks/f_variants/source-sample.ttf",
}
POLICY = "public_docs/distribution.json"
PUBLIC_SCHEMA = "hwsol-public-distribution/1"
PRIVATE_TEXT = re.compile(
    r"/home/[^/\s`]+/|workspace/runs/20\d{6}T|"
    r"out/(?:next-stage|realism)/|source_pdf/[^\s`<>]+|"
    r"docs/(?:MASTER_EXECUTION_PLAN|EXECUTION_HANDOFF|CURRENT_STATE|V010\d*_)"
)


def audit_paths(names):
    private = {
        ".git",
        ".agents",
        ".aws",
        ".codex",
        ".venv",
        "checkpoints",
        "training-runs",
        "out",
        "dist",
        "workspace",
        "source_pdf",
        "sourcepdf",
        ".vscode",
        ".env",
        "__pycache__",
    }
    for name in names:
        p = PurePosixPath(name)
        if (
            p.is_absolute()
            or (p.parts and p.parts[0] in {"profiles", "samples", "jobs"})
            or ".." in p.parts
            or private.intersection(p.parts)
            or p.name.startswith(".env.")
            or p.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors", ".pkl", ".key", ".pem"}
            or (p.suffix.lower() in {".ttf", ".otf"} and name not in {FONT, *AUTHOR_FONTS})
            or (
                name.startswith(AUTHOR_ROOT)
                and p.suffix.lower() not in {".json", ".npz", ".ttf", ".txt"}
            )
        ):
            raise ValueError(f"Unexpected release file: {name}")


def audit_public_files(files: dict[str, bytes]) -> dict:
    """Audit actual public bytes, including links, rather than trusting an allowlist alone."""
    audit_paths(files)
    if "PUBLIC_DISTRIBUTION.json" not in files:
        raise ValueError("Unprepared source: stage the explicit public distribution before sharing")
    policy = json.loads(files[POLICY])
    if policy.get("schema") != PUBLIC_SCHEMA:
        raise ValueError("Unknown public distribution policy")
    receipt = json.loads(files["PUBLIC_DISTRIBUTION.json"])
    expected = receipt["source_sha256"]
    if (
        receipt.get("schema") != PUBLIC_SCHEMA
        or receipt.get("policy_sha256") != hashlib.sha256(files[POLICY]).hexdigest()
    ):
        raise ValueError("Public source receipt or policy changed")
    for name, digest in expected.items():
        if name not in files or hashlib.sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"Public source differs from its receipt: {name}")
    allowed = set(expected) | set(policy["overlays"]) | {"PUBLIC_DISTRIBUTION.json"}
    metadata_names = {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
        "not-zip-safe",
        "zip-safe",
    }
    for name in set(files) - allowed:
        path = PurePosixPath(name)
        metadata = (
            any(part.endswith(".egg-info") for part in path.parts) and path.name in metadata_names
        )
        if name == "setup.cfg" and files[name].replace(b"\r\n", b"\n") != (
            b"[egg_info]\ntag_build = \ntag_date = 0\n\n"
        ):
            raise ValueError("Unexpected generated sdist configuration")
        if not metadata and name not in {"PKG-INFO", "setup.cfg"}:
            raise ValueError(f"File outside the frozen public source inventory: {name}")
    for target, template in policy["overlays"].items():
        if target not in files or template not in files or files[target] != files[template]:
            raise ValueError(f"Public entry differs from its versioned template: {target}")
    private = []
    for name, data in files.items():
        if PurePosixPath(name).suffix.lower() in {".md", ".txt", ".json", ".xml"} or name.endswith(
            ".md.in"
        ):
            if PRIVATE_TEXT.search(data.decode("utf-8", errors="replace")):
                private.append(name)
    if private:
        raise ValueError(f"Private operational references in public files: {private}")
    if AUTHOR_ROOT + "profile.json" in files:
        profile = json.loads(files[AUTHOR_ROOT + "profile.json"])
        declared = {AUTHOR_ROOT + asset["path"] for asset in profile["assets"]}
        declared |= {AUTHOR_ROOT + "profile.json", AUTHOR_ROOT + "release.json"}
        actual = {name for name in files if name.startswith(AUTHOR_ROOT)}
        if actual != declared:
            raise ValueError("Bundled author assets differ from their explicit runtime inventory")
        for asset in profile["assets"]:
            if hashlib.sha256(files[AUTHOR_ROOT + asset["path"]]).hexdigest() != asset["sha256"]:
                raise ValueError("Bundled author asset integrity failed")
        recipe = json.loads(files[AUTHOR_ROOT + profile["ink_bundle"]])
        if (
            profile["distribution"] != "OFL-1.1 AND MIT"
            or recipe["recipe"] != "compiled-outlines/1"
        ):
            raise ValueError("Only the authorized crop-free author runtime derivative can ship")
    missing = []
    for name, data in files.items():
        if not name.endswith(".md"):
            continue
        text = data.decode("utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            target = match[1].strip().split(" ", 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path = posixpath.normpath(str(PurePosixPath(name).parent / unquote(parsed.path)))
            if path.startswith("../") or path.startswith("/"):
                missing.append({"file": name, "target": target})
            elif path not in files and not any(p.startswith(path.rstrip("/") + "/") for p in files):
                missing.append({"file": name, "target": target})
    if missing:
        raise ValueError(f"Broken public relative links: {missing}")
    return {
        "status": "passed",
        "files": len(files),
        "relative_links": "resolved",
        "private_operational_references": [],
        "git_history_included": False,
    }


def public_files(source: Path) -> dict[str, bytes]:
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def verify_public_source(source: Path) -> dict:
    """Validate fresh build inputs; stale setuptools output is never trusted."""
    if not (source / "PUBLIC_DISTRIBUTION.json").is_file():
        raise ValueError("Unprepared source: no PUBLIC_DISTRIBUTION.json receipt")
    if (source / "build").exists():
        raise ValueError("Existing build output is not audited; prepare a fresh public snapshot")
    files = {}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_symlink():
            raise ValueError(f"Linked public build input: {relative}")
        if path.is_file():
            # Importing this backend creates bytecode. No directory-name exclusion:
            # src/hwsol/build (or dist) can be actual package code, and build/lib
            # can otherwise inject stale files into setuptools' wheel output.
            if relative.parent.name == "__pycache__" and path.suffix == ".pyc":
                continue
            files[relative.as_posix()] = path.read_bytes()
    return audit_public_files(files)


def stage_public_source(source: Path, destination: Path) -> dict:
    """Copy an explicit public view; originals and private instructions stay untouched."""
    policy_path = source / POLICY
    if not policy_path.is_file():
        raise ValueError(
            "Selected ref has no versioned public distribution layer; use its historical "
            "packaging workflow for local inspection or a new reviewed candidate ref. "
            "Current working-tree guides cannot be substituted into an old tag."
        )
    policy = json.loads(policy_path.read_bytes())
    if policy.get("schema") != PUBLIC_SCHEMA:
        raise ValueError("Unsupported public distribution policy")
    selected = set(policy["files"])
    for pattern in policy["patterns"]:
        selected.update(
            p.relative_to(source).as_posix() for p in source.glob(pattern) if p.is_file()
        )
    audit_paths(selected)
    destination.mkdir()
    source_hashes = {}
    for name in sorted(selected):
        original = source / name
        if original.is_symlink() or not original.is_file():
            raise ValueError(f"Missing or linked public source: {name}")
        original.resolve().relative_to(source.resolve())
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        source_hashes[name] = hashlib.sha256(original.read_bytes()).hexdigest()
    for target, template in policy["overlays"].items():
        audit_paths([target, template])
        if template not in selected:
            raise ValueError(f"Public overlay is outside the source allowlist: {template}")
        saved = destination / target
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination / template, saved)
    receipt = {
        "schema": PUBLIC_SCHEMA,
        "policy_sha256": source_hashes[POLICY],
        "source_sha256": source_hashes,
        "overlays": policy["overlays"],
        "git_history_included": False,
    }
    (destination / "PUBLIC_DISTRIBUTION.json").write_text(json.dumps(receipt, indent=2) + "\n")
    audit_public_files(public_files(destination))
    return receipt


def write_skill_archive(source: Path, destination: Path) -> dict:
    files = public_files(source)
    result = audit_public_files(files)
    with ZipFile(destination, "x") as archive:
        for name, data in sorted(files.items()):
            archive.writestr(f"handwritten-solutions/{name}", data)
    return result


def git(*args, cwd=None):
    return subprocess.check_output(["git", *map(str, args)], cwd=cwd or ROOT).decode().strip()


def resolve_package_source(
    *, tag: str | None = None, ref: str | None = None, version: str | None = None
) -> dict:
    """Bind metadata to one Git object, never to mutable checkout files."""
    if tag is not None:
        if ref is not None or version is not None:
            raise ValueError("Choose a stable tag or a local prerelease ref, not both")
        if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            raise ValueError("Use a release tag such as v0.3.0")
        version, ref, mode = tag[1:], f"refs/tags/{tag}", "stable-tag"
    else:
        if not version or not re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)rc[1-9][0-9]*", version
        ):
            raise ValueError("Local prerelease version must be canonical X.Y.ZrcN with N >= 1")
        if not ref or ref.startswith("-") or any(c.isspace() for c in ref):
            raise ValueError("Provide an explicit Git commit or ref for the local prerelease")
        mode = "local-prerelease"
    commit = git("rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    project = tomllib.loads(git("show", f"{commit}:pyproject.toml"))["project"]
    if project["version"] != version:
        raise ValueError("Selected Git source and Python distribution version disagree")
    if mode == "local-prerelease":
        locked = tomllib.loads(git("show", f"{commit}:uv.lock"))
        own = [p for p in locked.get("package", []) if p.get("name") == "handwritten-solutions"]
        if (
            project.get("name") != "handwritten-solutions"
            or len(own) != 1
            or own[0].get("version") != version
        ):
            raise ValueError("Selected prerelease project and lockfile version disagree")
    return {"commit": commit, "version": version, "tag": tag, "requested_ref": ref, "mode": mode}


def package(tag: str, output: Path) -> None:
    """Preserve the stable-tag packaging API."""
    if output.exists():
        raise ValueError("Choose a new release directory")
    _package(resolve_package_source(tag=tag), output)


def package_candidate(ref: str, version: str, output: Path) -> None:
    """Build local RC artifacts without creating a tag or publishing anything."""
    if output.exists():
        raise ValueError("Choose a new release directory")
    _package(resolve_package_source(ref=ref, version=version), output)


def _package(binding: dict, output: Path) -> None:
    commit, version = binding["commit"], binding["version"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        work = Path(temp).resolve()
        stage = work / "artifacts"
        stage.mkdir()
        source_archive = work / "selected-ref.zip"
        git("archive", "--format=zip", "--prefix=selected-ref/", "-o", source_archive, commit)
        with ZipFile(source_archive) as archive:
            archive.extractall(work)
        selected_ref = work / "selected-ref"
        source = work / "handwritten-solutions"
        # The wheel cannot discover a Git checkout. Embed the exact release identity,
        # while the fingerprint itself is computed from installed code/dependencies.
        build_record = {
            **binding,
            "lock_sha256": hashlib.sha256((selected_ref / "uv.lock").read_bytes()).hexdigest(),
            "git_history_included": False,
        }
        (selected_ref / "src/hwsol/_build.json").write_text(
            json.dumps(build_record, indent=2) + "\n"
        )
        stage_public_source(selected_ref, source)
        for name in ("SKILL.md", "uv.lock", "agents/openai.yaml", "LICENSE", FONT):
            if not (source / name).is_file():
                raise ValueError(f"Missing Skill file: {name}")
        zip_path = stage / f"handwritten-solutions-{version}-skill.zip"
        write_skill_archive(source, zip_path)
        epoch = git("show", "-s", "--format=%ct", commit)
        build_dir = work / "build-artifacts"
        subprocess.run(
            ["uv", "build", "--python", sys.executable, "--out-dir", str(build_dir)],
            cwd=source,
            env={**os.environ, "SOURCE_DATE_EPOCH": epoch},
            check=True,
        )
        for binary in (
            f"handwritten_solutions-{version}.tar.gz",
            f"handwritten_solutions-{version}-py3-none-any.whl",
        ):
            shutil.move(build_dir / binary, stage / binary)
        sdist = stage / f"handwritten_solutions-{version}.tar.gz"
        with tarfile.open(sdist) as archive:
            files = {
                str(
                    PurePosixPath(entry.name).relative_to(f"handwritten_solutions-{version}")
                ): archive.extractfile(entry).read()
                for entry in archive
                if entry.isfile()
            }
        audit_public_files(files)
        required = {
            "SKILL.md",
            "uv.lock",
            "agents/openai.yaml",
            FONT,
            "public_docs/LICENSES.md",
            "public_docs/RESEARCH_BOUNDARIES.md",
            "PUBLIC_DISTRIBUTION.json",
            "scripts/smoke_wheel.py",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
        }
        if missing := required - set(files):
            raise ValueError(f"Incomplete sdist: {sorted(missing)}")
        wheel = stage / f"handwritten_solutions-{version}-py3-none-any.whl"
        with ZipFile(wheel) as archive:
            audit_paths(
                "src/" + name if name.startswith("hwsol/") else name for name in archive.namelist()
            )
            for filename in (FONT, FONT.replace("PatrickHand-Regular.ttf", "OFL.txt")):
                if archive.read(filename.removeprefix("src/")) != (source / filename).read_bytes():
                    raise ValueError(f"Wheel asset mismatch: {filename}")
        (stage / "release.json").write_text(json.dumps(build_record, indent=2) + "\n")
        checksums = []
        for path in sorted(stage.iterdir()):
            with path.open("rb") as stream:
                checksums.append(
                    f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.name}"
                )
        (stage / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
        stage.rename(output)
    print(f"{binding['tag'] or version} {commit}  {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ref", help="Frozen Git source for a local prerelease; no tag is created")
    parser.add_argument("--version", help="Local prerelease version, exactly X.Y.ZrcN")
    parser.add_argument("--stage", type=Path, help="Prepare a new public source without releasing")
    parser.add_argument(
        "--source", type=Path, help="Explicit source for --stage; defaults to checkout"
    )
    args = parser.parse_args()
    if args.stage:
        if args.tag or args.output or args.ref or args.version:
            parser.error("--stage cannot be combined with a tag, output, ref or version")
        args.stage.parent.mkdir(parents=True, exist_ok=True)
        stage_public_source((args.source or ROOT).resolve(), args.stage.resolve())
        print(args.stage.resolve())
    elif args.tag and args.output and not any((args.source, args.ref, args.version)):
        package(args.tag, args.output.absolute())
    elif args.ref and args.version and args.output and not any((args.tag, args.source)):
        package_candidate(args.ref, args.version, args.output.absolute())
    else:
        parser.error(
            "Use --stage <new-directory>, <version-tag> --output <new-directory>, "
            "or --ref <commit-or-ref> --version X.Y.ZrcN --output <new-directory>"
        )
