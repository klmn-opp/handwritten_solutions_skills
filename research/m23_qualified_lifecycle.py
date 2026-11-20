"""Exercise a real installed CPU qualification in an isolated store, never production."""

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
from scripts.cross_platform_candidate_check import archive_identity, copy_job

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def execute(linux: Path, bundle: Path) -> Path:
    verify_run(linux)
    request = read(bundle / "request.json")
    source = read(linux / "run.json")
    if request["build"]["commit"] != source["result"]["source_commit"]:
        raise ValueError("Evidence and installed build differ")
    installation = read(linux / "reports/installations.json")[0]
    cli = Path(installation["fresh_environment"]) / "bin/hwsol"
    profile = Path(source["config"]["profile_archive"])
    job = Path(source["config"]["input_job"])
    spec = request["profile"]
    run = Run.create(
        "m23-qualified-cpu-lifecycle",
        seed=1909,
        config={
            "purpose": "Installed qualification, publication and activation in an isolated store",
            "linux_run": str(linux),
            "linux_run_sha256": sha256(linux / "run.json"),
            "request": str(bundle / "request.json"),
            "request_sha256": sha256(bundle / "request.json"),
            "source_commit": request["build"]["commit"],
            "production_activation": False,
            "production_publication": False,
            "new_training": False,
        },
    )
    print(run.path, flush=True)
    shutil.copyfile(__file__, run.path / "logs/m23_qualified_lifecycle.py")
    child_workspace = Path(tempfile.mkdtemp(prefix="hwsol-m23-lifecycle-workspace-"))
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"HWSOL_DATA_DIR", "HWSOL_WORKSPACE", "PYTHONPATH", "VIRTUAL_ENV"}
    }
    env.update(
        HWSOL_DATA_DIR=str(run.path / "logs/isolated-store"),
        HWSOL_WORKSPACE=str(child_workspace),
        MPLCONFIGDIR=str(run.path / "logs/mpl"),
    )
    commands = []
    production_latest = sha256(ROOT / "workspace/latest.json")

    def command(*parts, failure=False, override=None):
        start = time.monotonic()
        proc = subprocess.run(
            [str(cli), *map(str, parts)],
            cwd=run.path,
            env=override or env,
            capture_output=True,
            text=True,
        )
        log = run.path / f"logs/command-{len(commands) + 1:02d}.txt"
        log.write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
        commands.append(
            {
                "argv": [str(cli), *map(str, parts)],
                "returncode": proc.returncode,
                "seconds": time.monotonic() - start,
                "log": str(log.relative_to(run.path)),
            }
        )
        atomic_json(run.path / "logs/commands.json", commands)
        if (proc.returncode != 0) != failure:
            raise ValueError(f"Unexpected CLI result: {log}")
        return json.loads(proc.stdout) if not failure else {"rejected": True}

    try:
        command("profiles", "import", "--source", profile)
        input_job = copy_job(job, run.path / "solutions/input")
        normal = read(input_job)
        normal.update(backend="font-cpu", profile=spec, profile_sha256=request["profile_sha256"])
        atomic_json(input_job, normal)
        command("profiles", "activate", spec, failure=True)
        command(
            "render",
            "--manifest",
            input_job,
            "--output",
            run.path / "rendered/unpublished",
            failure=True,
        )
        qualified = command("profiles", "qualify-ink", spec, "--evidence", bundle / "request.json")
        assert not qualified["published"] and not qualified["activated"]
        command("profiles", "activate", spec, failure=True)
        published = command(
            "profiles", "publish-ink", spec, "--qualification-id", qualified["qualification_id"]
        )
        assert not published["activated"]
        assert command("profiles", "list")["default"] == "bundled-default@v1"
        output = run.path / "rendered/qualified"
        rendered = command("render", "--manifest", input_job, "--output", output)
        assert rendered["status"] == "passed"
        command("verify", "--job", output, "--pixels")
        comparison = compare_pages(
            linux / "rendered/python-3.12/candidate", output, len(normal["problems"])
        )
        report = read(output / "validation_report.json")
        assert all(p["candidate"]["backend"] == "personal-cpu-qualified" for p in report["pages"])
        atomic_json(run.path / "comparisons/qualified-vs-candidate.json", comparison)
        command("profiles", "activate", spec)
        assert command("profiles", "list")["default"] == spec
        default_job = run.path / "solutions/input/default-job.json"
        atomic_json(
            default_job, {k: v for k, v in normal.items() if k not in {"profile", "profile_sha256"}}
        )
        default_output = run.path / "rendered/activated-default"
        command("render", "--manifest", default_job, "--output", default_output)
        command("verify", "--job", default_output, "--pixels")
        compare_pages(output, default_output, len(normal["problems"]))
        command("profiles", "rollback", "bundled-default@v1")
        assert command("profiles", "list")["default"] == "bundled-default@v1"
        command("profiles", "activate", spec)
        command("profiles", "rollback", "bundled-default@v1")
        exported = run.path / "solutions/qualified-profile-export.zip"
        command("profiles", "export", spec, "--output", exported)
        assert archive_identity(exported) == archive_identity(profile)
        second = {**env, "HWSOL_DATA_DIR": str(run.path / "logs/reimport-store")}
        imported = command("profiles", "import", "--source", exported, override=second)
        assert imported["candidate"] and not imported["activated"]
        command("profiles", "activate", spec, failure=True, override=second)
        assert sha256(ROOT / "workspace/latest.json") == production_latest
        for receipt in child_workspace.glob("runs/*/run.json"):
            destination = run.path / "logs/cli-receipts" / f"{receipt.parent.name}-receipt.json"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(receipt, destination)
        atomic_json(
            run.path / "reports/lifecycle.json",
            {
                "qualification": qualified,
                "publication": published,
                "ordinary_font_cpu": "passed",
                "default_and_rollback": "passed",
                "export_reimport_remains_candidate": True,
                "candidate_pixels_exact": True,
                "production_latest_sha256": production_latest,
                "production_untouched": True,
            },
        )
        run.finish(
            "held",
            result={
                "isolated_qualified_lifecycle": "passed",
                "ordinary_cpu_pages": len(normal["problems"]),
                "production_promotion": False,
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
    parser.add_argument("--linux", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()
    print(execute(args.linux.resolve(), args.bundle.resolve()))


if __name__ == "__main__":
    main()
