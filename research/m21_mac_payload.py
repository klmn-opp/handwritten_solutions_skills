"""Prepare the exact locally tested private payload for the author's authorized Mac."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from scripts.cross_platform_candidate_check import copy_job

ROOT = Path(__file__).resolve().parents[1]


def prepare(installation: Path, job: Path, profile: Path, rollback: Path) -> Path:
    verify_run(installation)
    installed = json.loads((installation / "run.json").read_text())
    reused_parent = None
    if installed["result"].get("linux_clean_install") != "passed":
        if installed["result"].get("linux_profile_reuse") != "passed":
            raise ValueError("Require completed Linux installation or linked Profile acceptance")
        reused_parent = installed["result"]["clean_install_parent"]
        parent = Path(reused_parent["path"])
        verify_run(parent)
        if sha256(parent / "run.json") != reused_parent["run_sha256"]:
            raise ValueError("Clean-install parent changed")
        previous = json.loads((parent / "run.json").read_text())["result"]
        if previous.get("linux_clean_install") != "passed" or any(
            installed["result"][field] != previous[field]
            for field in ("source_commit", "package_version", "wheel_sha256")
        ):
            raise ValueError("Profile reuse does not bind the clean-installed package")
    wheel = next((installation / "reports/dist").glob("*.whl"))
    expected = installed["config"]
    for path, digest in (
        (wheel, installed["result"]["wheel_sha256"]),
        (job, expected["input_job_sha256"]),
        (profile, expected["profile_archive_sha256"]),
        (rollback, expected["rollback_archive_sha256"]),
    ):
        if sha256(path) != digest:
            raise ValueError(f"Payload input differs from installed evidence: {path.name}")
    receipt = json.loads((installation / "rendered/python-3.12/receipt.json").read_text())
    binding = {
        "schema": "native-acceptance-input/1",
        "source_commit": receipt["source_commit"],
        "package_version": receipt["package_version"],
        "wheel": wheel.name,
        "profile": "candidate.zip",
        "rollback_profile": "rollback.zip",
        "job": "job/job.json",
        "inputs": {
            name: receipt["inputs"][name]["sha256"]
            for name in ("wheel", "profile", "rollback_profile", "job")
        },
        "profile_reference": receipt["outputs"][0]["profile"]["reference"],
        "rollback_reference": receipt["outputs"][1]["profile"]["reference"],
    }
    run = Run.create(
        "m21-authorized-mac-payload",
        seed=1909,
        config={
            "purpose": "private transferable acceptance input, not rendered homework",
            "installation_run": str(installation),
            "installation_receipt_sha256": sha256(installation / "run.json"),
            "owner_copy_authorization": "author explicitly permits project copy to own Mac",
            "mac_execution_claimed": False,
            "source_commit": binding["source_commit"],
            "package_version": binding["package_version"],
            "clean_install_parent": reused_parent,
        },
    )
    try:
        payload = run.path / "solutions/mac-acceptance"
        payload.mkdir()
        shutil.copyfile(__file__, run.path / "logs/m21_mac_payload.py")
        shutil.copyfile(wheel, payload / wheel.name)
        shutil.copyfile(profile, payload / binding["profile"])
        shutil.copyfile(rollback, payload / binding["rollback_profile"])
        atomic_json(payload / "acceptance-input.json", binding)
        script = ROOT / "scripts/cross_platform_candidate_check.py"
        frozen = installation / "logs/selected/scripts/cross_platform_candidate_check.py"
        if sha256(script) != sha256(frozen):
            raise ValueError("Platform harness differs from the tested frozen source")
        shutil.copyfile(script, payload / script.name)
        copy_job(job, payload / "job")
        copied = {
            str(p.relative_to(payload / "job")): sha256(p)
            for p in (payload / "job").rglob("*")
            if p.is_file()
        }
        if copied != receipt["job_input_inventory"]:
            raise ValueError("Payload source/solution/background differs from installed Job")
        (payload / "run-macos.sh").write_text(
            '#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\n'
            '[ "$(uname -s)" = Darwin ] || { echo "Requires actual macOS"; exit 2; }\n'
            "shasum -a 256 -c SHA256SUMS\n"
            "runtime=''\nfor candidate in python3.12 python3.13; do\n"
            '  if command -v "$candidate" >/dev/null 2>&1; then runtime="$candidate"; break; fi\n'
            "done\n"
            '[ -n "$runtime" ] || { echo "Python 3.12 or 3.13 is required"; exit 2; }\n'
            'output=${1:-"$PWD/results-$(date -u +%Y%m%dT%H%M%SZ)"}\n'
            'exec "$runtime" -I cross_platform_candidate_check.py --require-system Darwin '
            f'--wheel "{wheel.name}" --profile candidate.zip '
            '--rollback-profile rollback.zip --job job/job.json --output "$output"\n',
            encoding="utf-8",
        )
        (payload / "README.txt").write_text(
            "Private authorized Mac acceptance input. Do not publish this folder.\n"
            "Requires real macOS, Python 3.12 or 3.13, Git and public dependency downloads.\n"
            "Run: sh run-macos.sh\n"
            "Uses a new isolated environment/store/workspace, never activates production.\n"
            "This is an unactivated test build; consult its exact acceptance receipts.\n"
            "Return receipt.json and logs from the new results directory for review.\n",
            encoding="utf-8",
        )
        files = {
            str(p.relative_to(payload)): sha256(p)
            for p in sorted(payload.rglob("*"))
            if p.is_file()
        }
        (payload / "SHA256SUMS").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in files.items()), encoding="utf-8"
        )
        archive = run.path / "solutions/mac-acceptance.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            output.add(payload, arcname="mac-acceptance")
        atomic_json(
            run.path / "reports/payload.json",
            {"files": files, "archive_sha256": sha256(archive), "binding": binding},
        )
        run.finish(
            "held",
            result={
                "payload_prepared": True,
                "archive_sha256": sha256(archive),
                "transferred": False,
                "mac_executed": False,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("installation", "job", "profile", "rollback"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(
        prepare(
            args.installation.resolve(),
            args.job.resolve(),
            args.profile.resolve(),
            args.rollback.resolve(),
        )
    )


if __name__ == "__main__":
    main()
