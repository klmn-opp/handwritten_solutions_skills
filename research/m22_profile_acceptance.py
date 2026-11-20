"""Profile-only acceptance in immutable, previously clean-installed CPU runtimes.

No package build or dependency installation occurs here. The parent run remains
separate evidence of clean installation; the frozen harness honestly audits reuse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.personal_cpu_install import compare_pages

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def owner_receipt(path: Path) -> dict:
    for folder in path.parents:
        if (folder / "run.json").is_file():
            verify_run(folder)
            return {"path": str(folder), "run_sha256": sha256(folder / "run.json")}
    raise ValueError(f"Input has no immutable run owner: {path}")


def execute(parent: Path, profile: Path, rollback: Path, reference: Path) -> Path:
    parent, profile, rollback, reference = (
        p.resolve() for p in (parent, profile, rollback, reference)
    )
    verify_run(parent)
    record = read(parent / "run.json")
    accepted = record["result"]
    if accepted.get("linux_clean_install") != "passed" or not accepted.get(
        "canonical_prerelease_build"
    ):
        raise ValueError("A completed canonical clean-install parent is required")
    installations = read(parent / "reports/installations.json")
    if [row["python"] for row in installations] != ["3.12", "3.13"]:
        raise ValueError("Parent must bind both supported Linux Python runtimes")
    job = Path(record["config"]["input_job"]).resolve()
    if sha256(job) != record["config"]["input_job_sha256"]:
        raise ValueError("Parent original Job changed")
    parents = {
        name: owner_receipt(path)
        for name, path in {
            "job": job,
            "profile": profile,
            "rollback": rollback,
            "reference": reference / "validation_report.json",
        }.items()
    }
    wheel = next((parent / "reports/dist").glob("*.whl"))
    if sha256(wheel) != accepted["wheel_sha256"]:
        raise ValueError("Parent wheel changed")
    parent_proof = {"path": str(parent), "run_sha256": sha256(parent / "run.json")}
    run = Run.create(
        "m22-installed-profile-reuse",
        seed=1909,
        config={
            "purpose": "Profile-only acceptance in parent-installed wheel; no build/install",
            "clean_install_parent": parent_proof,
            "parents": parents,
            "source_commit": accepted["source_commit"],
            "package_version": accepted["package_version"],
            "input_job": str(job),
            "input_job_sha256": sha256(job),
            "profile_archive": str(profile),
            "profile_archive_sha256": sha256(profile),
            "rollback_archive": str(rollback),
            "rollback_archive_sha256": sha256(rollback),
            "reference": str(reference),
            "author_word_gap_recheck": "explicitly waived by author",
            "platform": "Linux",
            "new_package_install": False,
            "production_activation": False,
        },
    )
    print(run.path, flush=True)
    protected_latest = sha256(ROOT / "workspace/latest.json")
    commands = []
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "HWSOL_DATA_DIR",
            "HWSOL_WORKSPACE",
        }
    }

    def command(argv: list, cwd: Path) -> str:
        start = time.monotonic()
        proc = subprocess.run(
            list(map(str, argv)), cwd=cwd, env=env, capture_output=True, text=True
        )
        log = run.path / f"logs/command-{len(commands) + 1:02}.txt"
        log.write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
        commands.append(
            {
                "argv": list(map(str, argv)),
                "cwd": str(cwd),
                "returncode": proc.returncode,
                "seconds": time.monotonic() - start,
                "log": str(log.relative_to(run.path)),
            }
        )
        atomic_json(run.path / "logs/commands.json", commands)
        if proc.returncode:
            raise ValueError(f"Command failed: {log}")
        return proc.stdout

    try:
        shutil.copyfile(__file__, run.path / "logs/m22_profile_acceptance.py")
        shutil.copytree(job.parent, run.path / "solutions/input")
        inventory = {
            str(p.relative_to(job.parent)): sha256(p)
            for p in sorted(job.parent.rglob("*"))
            if p.is_file()
        }
        atomic_json(run.path / "reports/input-job-inventory.json", inventory)
        local_job = run.path / "solutions/input" / job.name
        local_profile, local_rollback = (
            run.path / "solutions/candidate.zip",
            run.path / "solutions/rollback.zip",
        )
        shutil.copyfile(profile, local_profile)
        shutil.copyfile(rollback, local_rollback)
        dist = run.path / "reports/dist"
        dist.mkdir()
        local_wheel = dist / wheel.name
        shutil.copyfile(wheel, local_wheel)
        for original, copied in (
            (wheel, local_wheel),
            (profile, local_profile),
            (rollback, local_rollback),
        ):
            if sha256(original) != sha256(copied):
                raise ValueError("Copied acceptance asset differs from immutable input")
        if inventory != {
            str(p.relative_to(local_job.parent)): sha256(p)
            for p in sorted(local_job.parent.rglob("*"))
            if p.is_file()
        }:
            raise ValueError("Copied original Job inventory differs")
        atomic_json(
            run.path / "reports/package-reuse.json",
            {
                "parent": parent_proof,
                "wheel_source": str(wheel),
                "wheel_sha256": sha256(local_wheel),
                "action": "byte-exact copy of parent canonical wheel",
                "package_build_performed": False,
                "dependency_install_performed": False,
            },
        )
        harness = run.path / "logs/selected/scripts/cross_platform_candidate_check.py"
        harness.parent.mkdir(parents=True)
        shutil.copyfile(parent / "logs/selected/scripts/cross_platform_candidate_check.py", harness)
        count = len(read(local_job)["problems"])
        results = []
        for installed in installations:
            label = installed["python"]
            old_receipt_path = Path(installed["portable_receipt"])
            if (
                not installed["clean_install_completed"]
                or installed["status"] != "passed"
                or sha256(old_receipt_path) != installed["portable_receipt_sha256"]
            ):
                raise ValueError("Parent runtime installation evidence changed")
            old_receipt = read(old_receipt_path)
            python = Path(installed["fresh_environment"]) / "bin/python"
            temp = Path(tempfile.mkdtemp(prefix=f"hwsol-m22-profile-{label}-"))
            baseline_build = read(
                parent / f"rendered/python-{label}/candidate/validation_report.json"
            )["build"]
            probe = [
                python,
                "-I",
                "-c",
                "import json,sys,platform,hwsol;from hwsol.identity import build_identity;"
                "from importlib.util import find_spec;"
                "assert find_spec('research') is None;assert find_spec('torch') is None;"
                "print(json.dumps({'version':hwsol.__version__,'python':sys.version,"
                "'system':platform.system(),'hwsol':hwsol.__file__,'build':build_identity()}))",
            ]
            before = json.loads(command(probe, temp))
            if (
                before["system"] != "Linux"
                or before["version"] != accepted["package_version"]
                or before["build"] != baseline_build
                or any(
                    before[k] != old_receipt["runtime"][k] for k in ("python", "system", "hwsol")
                )
            ):
                raise ValueError(
                    "Runtime, dependency or build identity differs from clean-install parent"
                )
            command(
                [
                    python,
                    "-I",
                    harness,
                    "--wheel",
                    local_wheel,
                    "--profile",
                    local_profile,
                    "--rollback-profile",
                    local_rollback,
                    "--job",
                    local_job,
                    "--output",
                    temp / "acceptance",
                    "--require-system",
                    "Linux",
                    "--runtime-python",
                    python,
                ],
                temp,
            )
            receipt = read(temp / "acceptance/receipt.json")
            if (
                receipt["status"] != "passed"
                or receipt["clean_install_completed"]
                or receipt["source_commit"] != accepted["source_commit"]
                or receipt["package_version"] != accepted["package_version"]
            ):
                raise ValueError(
                    "Reuse receipt changed runtime identity or claimed new installation"
                )
            after = json.loads(command(probe, temp))
            if before != after:
                raise ValueError("Runtime changed during Profile-only acceptance")
            destination = run.path / f"rendered/python-{label}"
            shutil.copytree(
                temp / "acceptance",
                destination,
                ignore=lambda path, names: [
                    n
                    for n in names
                    if n in {"workspace", "store", "roundtrip-store", "mpl", "environment"}
                ],
            )
            receipts = run.path / f"reports/cli-receipts/{label}"
            receipts.mkdir(parents=True)
            for child in (temp / "acceptance/workspace/runs").glob("*/run.json"):
                shutil.copyfile(child, receipts / (child.parent.name + ".json"))
            atomic_json(
                run.path / f"reports/runtime-{label}.json",
                {
                    "before": before,
                    "after": after,
                    "parent_render_build_exact": True,
                    "parent_installed_receipt": str(old_receipt_path),
                    "parent_installed_receipt_sha256": sha256(old_receipt_path),
                },
            )
            results.append(
                {
                    "python": label,
                    "status": "passed",
                    "runtime_python": str(python),
                    "source_commit": accepted["source_commit"],
                    "package_version": accepted["package_version"],
                    "wheel_sha256": sha256(local_wheel),
                    "clean_install_completed": False,
                    "clean_install_parent": parent_proof,
                    "portable_receipt": str(destination / "receipt.json"),
                    "portable_receipt_sha256": sha256(destination / "receipt.json"),
                    "reference_comparison": compare_pages(
                        reference, destination / "candidate", count
                    ),
                }
            )
            atomic_json(run.path / "reports/runtime-reuse.json", results)
        atomic_json(
            run.path / "reports/cross-python.json",
            compare_pages(
                run.path / "rendered/python-3.12/candidate",
                run.path / "rendered/python-3.13/candidate",
                count,
            ),
        )
        if inventory != {
            str(p.relative_to(job.parent)): sha256(p)
            for p in sorted(job.parent.rglob("*"))
            if p.is_file()
        }:
            raise ValueError("Original Job inventory changed")
        if sha256(ROOT / "workspace/latest.json") != protected_latest:
            raise ValueError("Production latest changed")
        atomic_json(run.path / "reports/protected.json", {"latest_sha256": protected_latest})
        verify_run(parent)
        for value in parents.values():
            verify_run(Path(value["path"]))
        run.finish(
            "held",
            result={
                "linux_profile_reuse": "passed",
                "clean_install_parent": parent_proof,
                "new_package_install": False,
                "source_commit": accepted["source_commit"],
                "package_version": accepted["package_version"],
                "wheel_sha256": sha256(local_wheel),
                "new_job_pages": count,
                "cross_python_exact": True,
                "reference_comparison": True,
                "macOS": "not executed",
                "production_release": False,
                "author_word_gap_recheck": "explicitly waived by author",
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-install", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rollback-profile", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(execute(args.parent_install, args.profile, args.rollback_profile, args.reference))
