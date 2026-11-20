"""Build one exact public source ref and test a new Job in fresh Linux CPU installs.

This research coordinator is never imported by an installed runtime. The portable
acceptance script executes outside the checkout and records its own operation evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from zipfile import ZipFile

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.package_readiness import audit_package
from research.personal_cpu_install import compare_pages
from scripts.package_release import stage_public_source, write_skill_archive

ROOT = Path(__file__).resolve().parents[1]


def execute(ref: str, job: Path, profile: Path, rollback: Path, reference: Path | None) -> Path:
    ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    job, profile, rollback = (p.resolve() for p in (job, profile, rollback))
    count = len(json.loads(job.read_text())["problems"])
    run = Run.create(
        "m21-clean-installed-new-job",
        seed=1909,
        config={
            "source_commit": ref,
            "input_job": str(job),
            "input_job_sha256": sha256(job),
            "profile_archive": str(profile),
            "profile_archive_sha256": sha256(profile),
            "rollback_archive": str(rollback),
            "rollback_archive_sha256": sha256(rollback),
            "page_count": count,
            "reference": str(reference) if reference else None,
            "platform": "Linux",
            "new_training": False,
            "production_publication": False,
        },
    )
    print(run.path, flush=True)
    protected = {
        str(p): sha256(p)
        for p in (
            ROOT / "workspace/latest.json",
            ROOT / "uv.lock",
            ROOT / "docs/MASTER_EXECUTION_PLAN.md",
            job,
            profile,
            rollback,
        )
    }
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "VIRTUAL_ENV", "HWSOL_DATA_DIR", "HWSOL_WORKSPACE"}
    }
    env.update(UV_CACHE_DIR="/tmp/hwsol-uv-cache", UV_OFFLINE="1")
    commands = []

    def command(argv, cwd, settings=None):
        start = time.monotonic()
        result = subprocess.run(
            list(map(str, argv)), cwd=cwd, env=settings or env, capture_output=True, text=True
        )
        number = len(commands) + 1
        log = run.path / f"logs/command-{number:02d}.txt"
        log.write_text(result.stdout + "\nSTDERR\n" + result.stderr)
        commands.append(
            {
                "argv": list(map(str, argv)),
                "cwd": str(cwd),
                "returncode": result.returncode,
                "seconds": time.monotonic() - start,
                "log": str(log.relative_to(run.path)),
            }
        )
        atomic_json(run.path / "logs/commands.json", commands)
        if result.returncode:
            raise ValueError(f"Command failed: {log}")
        return result.stdout

    try:
        shutil.copyfile(__file__, run.path / "logs/m21_clean_install.py")
        archive = run.path / "logs/selected-source.zip"
        command(["git", "archive", "--format=zip", "--prefix=selected/", "-o", archive, ref], ROOT)
        with ZipFile(archive) as zipped:
            zipped.extractall(run.path / "logs")
        selected = run.path / "logs/selected"
        for path in ("scripts/package_release.py", "scripts/cross_platform_candidate_check.py"):
            if sha256(selected / path) != sha256(ROOT / path):
                raise ValueError(f"Source helper differs from exact ref: {path}")
        atomic_json(
            selected / "src/hwsol/_build.json",
            {
                "commit": ref,
                "lock_sha256": sha256(selected / "uv.lock"),
                "purpose": "unreleased M2.1 new Job installation acceptance",
                "git_history_included": False,
            },
        )
        public = run.path / "logs/public-source"
        atomic_json(run.path / "reports/public-stage.json", stage_public_source(selected, public))
        env["SOURCE_DATE_EPOCH"] = command(["git", "show", "-s", "--format=%ct", ref], ROOT).strip()
        dist = run.path / "reports/dist"
        dist.mkdir()
        command(["uv", "build", "--offline", "--python", sys.executable, "--out-dir", dist], public)
        version = tomllib.loads((public / "pyproject.toml").read_text())["project"]["version"]
        write_skill_archive(public, dist / f"handwritten-solutions-{version}-skill.zip")
        packages = [*dist.glob("*.whl"), *dist.glob("*.tar.gz"), *dist.glob("*-skill.zip")]
        if len(packages) != 3:
            raise ValueError("Incomplete package inventory")
        audits = [audit_package(p) for p in packages]
        if any(
            a["unsafe_asset_entries"] or a["research_modules"] or a["local_identity_metadata"]
            for a in audits
        ):
            raise ValueError("Public package privacy audit failed")
        atomic_json(run.path / "reports/package-audit.json", audits)
        wheel = next(dist.glob("*.whl"))
        results = []
        for label, interpreter in (
            ("3.12", ROOT / ".venv/bin/python"),
            ("3.13", Path("/tmp/hwsol-v030-py313/bin/python")),
        ):
            temp = Path(tempfile.mkdtemp(prefix=f"hwsol-m21-clean-{label}-"))
            runtime = temp / "runtime"
            settings = {**env, "UV_PROJECT_ENVIRONMENT": str(runtime)}
            command(
                [
                    "uv",
                    "sync",
                    "--offline",
                    "--project",
                    public,
                    "--python",
                    interpreter,
                    "--locked",
                    "--no-dev",
                    "--no-install-project",
                ],
                temp,
                settings,
            )
            python = runtime / "bin/python"
            command(
                ["uv", "pip", "install", "--offline", "--python", python, "--no-deps", wheel],
                temp,
                settings,
            )
            command(
                [
                    python,
                    "-I",
                    selected / "scripts/cross_platform_candidate_check.py",
                    "--wheel",
                    wheel,
                    "--profile",
                    profile,
                    "--rollback-profile",
                    rollback,
                    "--job",
                    job,
                    "--output",
                    temp / "acceptance",
                    "--require-system",
                    "Linux",
                    "--runtime-python",
                    python,
                ],
                temp,
                settings,
            )
            acceptance = temp / "acceptance"
            receipt = json.loads((acceptance / "receipt.json").read_text())
            if receipt["status"] != "passed" or receipt["source_commit"] != ref:
                raise ValueError("Installed acceptance differs from source ref")
            destination = run.path / f"rendered/python-{label}"
            shutil.copytree(
                acceptance,
                destination,
                ignore=lambda path, names: [
                    n
                    for n in names
                    if n in {"workspace", "store", "roundtrip-store", "mpl", "environment"}
                ],
            )
            receipts = run.path / f"reports/cli-receipts/{label}"
            receipts.mkdir(parents=True)
            for child in (acceptance / "workspace/runs").glob("*/run.json"):
                shutil.copyfile(child, receipts / (child.parent.name + ".json"))
            comparison = (
                compare_pages(reference, destination / "candidate", count) if reference else None
            )
            results.append(
                {
                    "python": label,
                    "source_commit": ref,
                    "wheel_sha256": sha256(wheel),
                    "fresh_environment": str(runtime),
                    "clean_install_completed": True,
                    "installer": "offline uv.lock dependency closure + exact wheel",
                    "portable_checker_installation_mode": "audit externally installed runtime",
                    "portable_receipt": str(destination / "receipt.json"),
                    "portable_receipt_sha256": sha256(destination / "receipt.json"),
                    "new_job_sha256": sha256(job),
                    "pages": count,
                    "reference_comparison": comparison,
                    "status": "passed",
                }
            )
            atomic_json(run.path / "reports/installations.json", results)
        cross = compare_pages(
            run.path / "rendered/python-3.12/candidate",
            run.path / "rendered/python-3.13/candidate",
            count,
        )
        atomic_json(run.path / "reports/cross-python.json", cross)
        for p, digest in protected.items():
            if sha256(Path(p)) != digest:
                raise ValueError(f"Protected input changed: {p}")
        atomic_json(run.path / "reports/protected.json", protected)
        run.finish(
            "held",
            result={
                "linux_clean_install": "passed",
                "new_job_pages": count,
                "source_commit": ref,
                "wheel_sha256": sha256(wheel),
                "python_versions": ["3.12", "3.13"],
                "cross_python_exact": True,
                "reference_comparison": reference is not None,
                "macOS": "not executed",
                "production_release": False,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rollback-profile", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    print(execute(args.ref, args.job, args.profile, args.rollback_profile, args.reference))


if __name__ == "__main__":
    main()
