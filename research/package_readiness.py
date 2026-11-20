"""Local CPU package preflight; never tag, publish, or change production latest."""

# ruff: noqa: E501 -- Preserve full prose lines in the new editable solution fixture.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run
from scripts.package_release import audit_public_files, stage_public_source, write_skill_archive

ROOT = Path(__file__).resolve().parents[1]
FONT = "hwsol/assets/presets/patrick-hand/PatrickHand-Regular.ttf"
PRIVATE_PARTS = {
    "workspace",
    "source_pdf",
    "sourcepdf",
    "checkpoints",
    "training-runs",
    ".git",
    ".agents",
    ".aws",
    ".codex",
    ".venv",
    ".vscode",
    ".env",
}
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pem", ".key"}


def snapshot_source(destination: Path) -> dict:
    roots = {
        "src",
        "docs",
        "scripts",
        "tests",
        "examples",
        "agents",
        "config",
        "references",
        "public_docs",
    }
    single = {
        "pyproject.toml",
        "public_build_backend.py",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SKILL.md",
        "uv.lock",
        "requirements-compatibility.lock",
        ".github/workflows/ci.yml",
    }
    paths = {ROOT / name for name in single}
    paths.update(p for name in roots for p in (ROOT / name).rglob("*") if p.is_file())
    files = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if not path.is_file() or "__pycache__" in relative.parts:
            continue
        if any(part.endswith(".egg-info") for part in relative.parts) or path.suffix == ".pyc":
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files[str(relative)] = sha256(target)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {
        "commit": commit,
        "lock_sha256": sha256(destination / "uv.lock"),
        "purpose": "unreleased local CPU packaging preflight",
    }
    atomic_json(destination / "src/hwsol/_build.json", identity)
    files["src/hwsol/_build.json"] = sha256(destination / "src/hwsol/_build.json")
    return {"commit": commit, "files": files, "working_tree_snapshot": True}


def audit_package(path: Path) -> dict:
    if path.suffix in {".whl", ".zip"}:
        with ZipFile(path) as archive:
            entries = {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
        if path.suffix == ".zip":
            entries = {
                str(PurePosixPath(k).relative_to("handwritten-solutions")): v
                for k, v in entries.items()
            }
        runtime_font = FONT if path.suffix == ".whl" else "src/" + FONT
    else:
        with tarfile.open(path) as archive:
            entries = {
                str(
                    PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0])
                ): archive.extractfile(member).read()
                for member in archive
                if member.isfile()
            }
        runtime_font = "src/" + FONT
    problems, local_metadata = [], []
    fonts = []
    for name, data in entries.items():
        item = PurePosixPath(name)
        if item.is_absolute() or ".." in item.parts or PRIVATE_PARTS.intersection(item.parts):
            problems.append(name)
        if item.suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(name)
        if item.suffix.lower() in {".ttf", ".otf"}:
            fonts.append(name)
            if name != runtime_font:
                problems.append(name)
        if item.suffix.lower() in {".md", ".txt", ".json", ".xml"}:
            text = data.decode("utf-8", errors="replace")
            count = len(re.findall(r"/home/[^/\s`]+/", text))
            if count:
                local_metadata.append({"path": name, "local_home_references": count})
    expected = (ROOT / "src" / FONT).read_bytes()
    if entries.get(runtime_font) != expected:
        problems.append("bundled font differs from licensed source")
    research = [name for name in entries if "research" in PurePosixPath(name).parts]
    public_audit = audit_public_files(entries) if path.suffix != ".whl" else None
    return {
        "archive": path.name,
        "sha256": sha256(path),
        "entries": len(entries),
        "unsafe_asset_entries": sorted(set(problems)),
        "fonts": fonts,
        "research_modules": research,
        "local_identity_metadata": local_metadata,
        "raw_personal_assets_or_weights_detected": bool(problems),
        "personal_math_candidate_included": False,
        "public_source_audit": public_audit,
        "scope": "archive paths, model/key extensions and exact bundled font bytes; not a legal audit",
    }


def new_text_fixture(folder: Path) -> Path:
    folder.mkdir()
    statement = (
        "A scalar process satisfies x'(t)=-2x(t)+6 for t>=0, with x(0)=1. "
        "(a) Find x(t) and verify the differential equation and initial value. "
        "(b) Evaluate x(log(2)) and the integral of x from 0 to log(2). "
        "(c) Identify the equilibrium and describe its stability."
    )
    solution = r"""(a) The constant forcing sets the equilibrium at three. Solving the homogeneous equation and imposing the initial value gives
$$
x(t)=3-2e^{-2t},\qquad x'(t)=4e^{-2t}=-2x(t)+6.
$$
At zero the solution equals one, so both the differential equation and the prescribed initial condition are satisfied.

(b) Since the exponential factor is one quarter at the requested time,
$$
x(\log 2)=\frac{5}{2},\qquad \int_0^{\log 2}x(t)\,dt=3\log 2-\frac{3}{4}.
$$
An antiderivative is three times t plus the exponential of negative two times t; subtracting its endpoint values gives the stated integral.

(c) The equilibrium is three. Every initial displacement from that value is multiplied by the decaying factor, so the equilibrium is asymptotically stable.
"""
    (folder / "question.txt").write_text(statement + "\n", encoding="utf-8")
    (folder / "solution.md").write_text(solution, encoding="utf-8")
    from sympy import diff, exp, integrate, log, simplify, symbols

    t = symbols("t", real=True)
    x = 3 - 2 * exp(-2 * t)
    assert simplify(diff(x, t) + 2 * x - 6) == 0 and x.subs(t, 0) == 1
    assert simplify(x.subs(t, log(2)) - 5 / 2) == 0
    assert simplify(integrate(x, (t, 0, log(2))) - 3 * log(2) + 3 / 4) == 0
    atomic_json(
        folder / "job.json",
        {
            "schema_version": 1,
            "generated_by": "Codex, new Agent-authored package preflight fixture",
            "profile": "bundled-default@v1",
            "realism": "baseline",
            "language": "en",
            "render": {"dpi": 150, "seed": 20261120, "paper": "plain"},
            "problems": [
                {
                    "id": "cpu-new-ode",
                    "number": "New 1",
                    "statement": statement,
                    "sources": [
                        {"path": "question.txt", "sha256": sha256(folder / "question.txt")}
                    ],
                    "solution": "solution.md",
                    "verification": {
                        "status": "passed",
                        "evidence": "Exact symbolic ODE, initial value, endpoint and integral checks; no rendered-content proof.",
                        "checks": [{"lhs": "3-2/4", "rhs": "5/2"}],
                    },
                }
            ],
        },
    )
    return folder / "job.json"


def launch() -> Path:
    run = Run.create(
        "m2-linux-package-preflight",
        seed=20261120,
        config={
            "platforms": ["Linux"],
            "python_targets": ["3.12", "3.13"],
            "publication": False,
            "profile": "bundled-default@v1",
        },
    )
    print(run.path, flush=True)
    protected = {
        str(ROOT / "workspace/latest.json"): sha256(ROOT / "workspace/latest.json"),
        str(ROOT / "pyproject.toml"): sha256(ROOT / "pyproject.toml"),
    }
    for name in ("README.md", "SKILL.md", "docs/MASTER_EXECUTION_PLAN.md"):
        protected[str(ROOT / name)] = sha256(ROOT / name)
    command_records = []
    env = {
        **os.environ,
        "UV_CACHE_DIR": "/tmp/hwsol-uv-cache",
        "UV_OFFLINE": "1",
        "PYTHONUTF8": "1",
    }
    for key in ("PYTHONPATH", "VIRTUAL_ENV", "HWSOL_DATA_DIR", "HWSOL_WORKSPACE"):
        env.pop(key, None)

    def command(args: list[str], *, cwd: Path, settings: dict | None = None) -> str:
        index = len(command_records) + 1
        start = time.monotonic()
        result = subprocess.run(args, cwd=cwd, env=settings or env, capture_output=True, text=True)
        log = run.path / "logs" / f"command-{index:02d}.txt"
        log.write_text(result.stdout + "\n--- stderr ---\n" + result.stderr, encoding="utf-8")
        command_records.append(
            {
                "args": args,
                "cwd": str(cwd),
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - start, 3),
                "log": str(log.relative_to(run.path)),
            }
        )
        atomic_json(run.path / "logs/commands.json", command_records)
        if result.returncode:
            raise RuntimeError(f"Command {index} exited {result.returncode}: {args[0]}; see {log}")
        return result.stdout

    try:
        raw_source = run.path / "logs/private-input-snapshot"
        source_manifest = snapshot_source(raw_source)
        source = run.path / "logs/source"
        public_receipt = stage_public_source(raw_source, source)
        atomic_json(run.path / "reports/public-source-receipt.json", public_receipt)
        shutil.copyfile(Path(__file__), run.path / "logs/package_readiness.py")
        atomic_json(run.path / "reports/source-snapshot.json", source_manifest)
        (run.path / "logs/task_plan.md").write_text(
            "# M2 package preflight\n\n- [x] Snapshot source.\n- [ ] Build and audit.\n"
            "- [ ] Clean Python 3.12/3.13 installs and 5+1 problems.\n- [ ] Review results.\n"
        )
        dist = run.path / "reports/dist"
        dist.mkdir()
        version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
        write_skill_archive(source, dist / f"handwritten-solutions-{version}-skill.zip")
        command(
            ["uv", "build", "--offline", "--python", sys.executable, "--out-dir", str(dist)],
            cwd=source,
        )
        packages = sorted([*dist.glob("*.whl"), *dist.glob("*.tar.gz"), *dist.glob("*-skill.zip")])
        if len(packages) != 3:
            raise ValueError("Expected one wheel, one sdist and one public Skill archive")
        audits = [audit_package(path) for path in packages]
        atomic_json(run.path / "reports/package-audit.json", audits)
        if any(a["unsafe_asset_entries"] for a in audits):
            raise ValueError("Package contains an unexpected private asset")
        wheel = next(path for path in packages if path.suffix == ".whl")
        fixture = new_text_fixture(run.path / "solutions/new-text")
        reports = []
        interpreters = [
            ("3.12", ROOT / ".venv/bin/python"),
            ("3.13", Path("/tmp/hwsol-v030-py313/bin/python")),
        ]
        for label, interpreter in interpreters:
            entry = {"python_target": label, "interpreter": str(interpreter), "status": "pending"}
            try:
                temporary = Path(tempfile.mkdtemp(prefix=f"hwsol-m2-{label}-"))
                venv = temporary / "runtime"
                settings = {
                    **env,
                    "UV_PROJECT_ENVIRONMENT": str(venv),
                    "HWSOL_DATA_DIR": str(temporary / "private"),
                    "HWSOL_WORKSPACE": str(temporary / "workspace"),
                    "MPLCONFIGDIR": str(temporary / "mpl"),
                }
                entry["environment"] = str(temporary)
                command(
                    [
                        "uv",
                        "sync",
                        "--offline",
                        "--project",
                        str(source),
                        "--python",
                        str(interpreter),
                        "--locked",
                        "--no-dev",
                        "--no-install-project",
                    ],
                    cwd=temporary,
                    settings=settings,
                )
                python, cli = venv / "bin/python", venv / "bin/hwsol"
                command(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--offline",
                        "--python",
                        str(python),
                        "--no-deps",
                        str(wheel),
                    ],
                    cwd=temporary,
                    settings=settings,
                )
                probe = command(
                    [
                        str(python),
                        "-I",
                        "-c",
                        "import json,sys,importlib.metadata as m; from importlib.util import find_spec; "
                        "import hwsol; assert find_spec('torch') is None; assert find_spec('pytest') is None; "
                        "assert find_spec('research') is None; "
                        "print(json.dumps({'python':sys.version,'hwsol':hwsol.__file__,'torch':False,"
                        "'research':False,'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))",
                    ],
                    cwd=temporary,
                    settings=settings,
                )
                entry["isolation"] = json.loads(probe)
                entry["doctor"] = json.loads(
                    command([str(cli), "doctor"], cwd=temporary, settings=settings)
                )
                if entry["doctor"]["status"] != "passed" or entry["doctor"]["training_required"]:
                    raise ValueError("Installed CPU doctor failed")
                inventory = json.loads(
                    command(
                        [str(cli), "benchmark", "list", "--suite", "all"],
                        cwd=temporary,
                        settings=settings,
                    )
                )
                entry["packaged_benchmark_cases"] = len(inventory["cases"])
                entry["jobs"] = []
                for name, manifest, order in (
                    (
                        "five-existing",
                        source / "examples/phase2/mixed/job.json",
                        ["2", "4", "6", "7", "9"],
                    ),
                    ("new-text", fixture, ["New 1"]),
                ):
                    output = run.path / "rendered" / f"python-{label}" / name
                    rendered = json.loads(
                        command(
                            [
                                str(cli),
                                "render",
                                "--manifest",
                                str(manifest),
                                "--output",
                                str(output),
                            ],
                            cwd=temporary,
                            settings=settings,
                        )
                    )
                    verified = json.loads(
                        command(
                            [str(cli), "verify", "--job", str(output), "--pixels"],
                            cwd=temporary,
                            settings=settings,
                        )
                    )
                    if verified["problem_order"] != order:
                        raise ValueError("Installed package lost problem identity")
                    report = json.loads((output / "validation_report.json").read_text())
                    if report["profile"]["reference"] != "bundled-default@v1":
                        raise ValueError("Clean runtime did not use bundled-default")
                    entry["jobs"].append(
                        {
                            "name": name,
                            "render": rendered,
                            "verify": verified,
                            "profile": report["profile"],
                            "output": str(output.relative_to(run.path)),
                        }
                    )
                entry["status"] = "passed"
            except Exception as exc:
                entry.update(status="failed", error=repr(exc))
            reports.append(entry)
            atomic_json(run.path / "reports/installations.json", reports)
        if any(sha256(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("Production latest or package version changed")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        (run.path / "logs/task_plan.md").write_text(
            "# M2 package preflight\n\n- [x] Snapshot original and public source separately.\n"
            "- [x] Build wheel/sdist/Skill; check public assets and links.\n"
            "- [x] Execute isolated Python 3.12/3.13 installs and 5+1 problems.\n"
            "- [x] Preserve results and explicit production-release limitations.\n"
        )
        result = {
            "installations": reports,
            "packages": audits,
            "linux_installation_preflight": "passed"
            if all(r["status"] == "passed" for r in reports)
            else "failed",
            "macos": "not_tested",
            "production_latest_preserved": True,
            "neural_training": False,
            "personal_math_candidate_packaged": False,
            "release": "No-Go: M2 author review, macOS and personal-math integration remain separate gates",
            "public_sdist_metadata_review_required": any(
                a["local_identity_metadata"] for a in audits
            ),
        }
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    launch()
