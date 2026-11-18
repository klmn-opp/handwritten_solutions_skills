"""Package a version tag as an installable Skill, wheel, sdist and cloneable Git bundle."""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
FONT = "src/hwsol/assets/presets/patrick-hand/PatrickHand-Regular.ttf"


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
            or (p.suffix.lower() in {".ttf", ".otf"} and name != FONT)
        ):
            raise ValueError(f"Unexpected release file: {name}")


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *map(str, args)], cwd=cwd).decode().strip()


def package(tag, output):
    if output.exists():
        raise ValueError("Choose a new release directory")
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
        raise ValueError("Use a release tag such as v0.3.0")
    version = tag[1:]
    ref = f"refs/tags/{tag}"
    commit = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    project = tomllib.loads(git("show", f"{commit}:pyproject.toml"))["project"]
    if project["version"] != version:
        raise ValueError("Git tag and Python distribution version disagree")
    # Include the release ancestry in the audit, since the bundle includes the same history.
    history = git("-c", "core.quotepath=false", "log", "-z", "--name-only", "--format=", commit)
    audit_paths(name for name in history.split("\0") if name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        work = Path(temp).resolve()
        stage = work / "artifacts"
        stage.mkdir()
        zip_path = stage / f"handwritten-solutions-{version}-skill.zip"
        git("archive", "--format=zip", "--prefix=handwritten-solutions/", "-o", zip_path, commit)
        with ZipFile(zip_path) as archive:
            archive.extractall(work)
        source = work / "handwritten-solutions"
        for name in ("SKILL.md", "uv.lock", "agents/openai.yaml", "LICENSE", FONT):
            if not (source / name).is_file():
                raise ValueError(f"Missing Skill file: {name}")
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
                str(PurePosixPath(entry.name).relative_to(f"handwritten_solutions-{version}"))
                for entry in archive
                if entry.isfile()
            }
        audit_paths(files)
        required = {
            "SKILL.md",
            "uv.lock",
            "agents/openai.yaml",
            FONT,
            "docs/LICENSES.md",
            "docs/PHASE4.md",
            "scripts/smoke_wheel.py",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
        }
        if missing := required - files:
            raise ValueError(f"Incomplete sdist: {sorted(missing)}")
        wheel = stage / f"handwritten_solutions-{version}-py3-none-any.whl"
        with ZipFile(wheel) as archive:
            audit_paths(
                "src/" + name if name.startswith("hwsol/") else name for name in archive.namelist()
            )
            for filename in (FONT, FONT.replace("PatrickHand-Regular.ttf", "OFL.txt")):
                if archive.read(filename.removeprefix("src/")) != (source / filename).read_bytes():
                    raise ValueError(f"Wheel asset mismatch: {filename}")
        # The bundle's main branch always identifies this release, even for a historic tag.
        mirror = work / "mirror.git"
        git("init", "-q", "--bare", "--initial-branch=main", mirror)
        git("fetch", "-q", "--no-tags", ROOT, f"{ref}:{ref}", cwd=mirror)
        git("update-ref", "refs/heads/main", commit, cwd=mirror)
        bundle = stage / f"handwritten-solutions-{tag}.bundle"
        git("bundle", "create", bundle, "HEAD", "refs/heads/main", ref, cwd=mirror)
        git("bundle", "verify", bundle, cwd=mirror)
        checksums = []
        for path in sorted(stage.iterdir()):
            with path.open("rb") as stream:
                checksums.append(
                    f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.name}"
                )
        (stage / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="ascii")
        stage.rename(output)
    print(f"{tag} {commit}  {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.tag, args.output.absolute())
