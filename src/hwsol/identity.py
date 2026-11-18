"""Path-independent code/dependency identity shared by jobs, Studio and experiments."""

import hashlib
import json
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

PACKAGES = (
    "handwritten-solutions",
    "handnote",
    "matplotlib",
    "fonttools",
    "numpy",
    "pillow",
    "opencv-python-headless",
    "pypdf",
    "pypdfium2",
    "sympy",
    "pydantic",
    "typer",
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_identity() -> dict:
    source = Path(__file__).resolve().parent
    root = source.parents[1]
    embedded = source / "_build.json"
    release = json.loads(embedded.read_bytes()) if embedded.is_file() else {}
    files = {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*.py"))
    }
    upstream = distribution("handnote")
    handnote_files = {
        str(p): hashlib.sha256(Path(upstream.locate_file(p)).read_bytes()).hexdigest()
        for p in upstream.files or ()
        if str(p).startswith("handnote/") and str(p).endswith(".py")
    }
    commit = release.get("commit")
    dirty = False
    if (root / ".git").exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
                cwd=root,
                text=True,
            ).strip()
        )
    lock = root / "uv.lock"
    lock_sha = (
        hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.is_file()
        else release.get("lock_sha256")
    )
    core = {
        "source": files,
        "handnote_source": handnote_files,
        "packages": {name: version(name) for name in PACKAGES},
        "lock_sha256": lock_sha,
    }
    return {
        "schema_version": 1,
        "commit": commit,
        "working_tree_dirty": dirty,
        "renderer_fingerprint": digest(core),
        **core,
    }
