"""Install an exact public commit and exercise private CPU candidates offline.

The public archives never contain the private Profile. All installed commands run
outside the checkout in fresh temporary stores; production pointers stay intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageChops

from hwsol.jobs.runtime import load_job, verify_job
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.package_readiness import audit_package, new_text_fixture
from scripts.package_release import stage_public_source, write_skill_archive

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def digest_or_missing(path: Path) -> str | None:
    return sha256(path) if path.is_file() else None


def copy_job_input(manifest: Path, target: Path) -> Path:
    """Copy only the explicit, safely relative inputs of the fixed job contract."""
    job = load_job(manifest)
    root = manifest.resolve().parent
    target.mkdir(parents=True)
    names = {problem.solution for problem in job.problems}
    names.update(source.path for problem in job.problems for source in problem.sources)
    names.update(
        background.path
        for background in [job.background, *[problem.background for problem in job.problems]]
        if background and background.path
    )
    for name in sorted(names):
        source, destination = inside(root, name), inside(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise ValueError("A fixed job input changed while copying")
    shutil.copyfile(manifest, target / "job.json")
    return target / "job.json"


def compare_pages(expected: Path, actual: Path, count: int) -> list[dict]:
    expected_report = read(expected / "validation_report.json")
    actual_report = read(actual / "validation_report.json")
    if len(expected_report["pages"]) != count or len(actual_report["pages"]) != count:
        raise ValueError("Comparison page inventories differ")
    fields = (
        "semantic",
        "context",
        "codepoint",
        "latex",
        "matrix_environment",
        "structure",
        "bounds_px",
        "source_type",
        "actual_glyph_id",
        "variant",
        "font_sha256",
    )
    checks = []
    for number in range(1, count + 1):
        left, right = expected / f"pages/{number:02d}.png", actual / f"pages/{number:02d}.png"
        with Image.open(left) as a, Image.open(right) as b:
            equal = (
                a.size == b.size and a.mode == b.mode and not ImageChops.difference(a, b).getbbox()
            )
        if not equal:
            raise ValueError(f"Installed output differs from the reviewed page {number}")
        old_page, new_page = (
            expected_report["pages"][number - 1],
            actual_report["pages"][number - 1],
        )
        old_records = [
            tuple(row.get(key) for key in fields) for row in old_page["glyph_provenance"]
        ]
        new_records = [
            tuple(row.get(key) for key in fields) for row in new_page["glyph_provenance"]
        ]
        if old_records != new_records or old_page["formula_groups"] != new_page["formula_groups"]:
            raise ValueError(
                f"Installed page {number} changed identities, source classes or geometry"
            )
        checks.append(
            {
                "page": number,
                "pixels_exact": True,
                "png_bytes_exact": sha256(left) == sha256(right),
                "expected_png_sha256": sha256(left),
                "actual_png_sha256": sha256(right),
                "semantic_source_geometry_and_formula_groups_exact": True,
                "fallback_count": new_page.get("candidate", {}).get("fallback_count"),
            }
        )
    return checks


def zip_inventory(path: Path) -> dict[str, str]:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate portable Profile entries")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}


def launch(
    ref: str,
    asset_run: Path,
    reference_job: Path,
    negative_manifest: Path,
    *,
    workspace: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[a-f0-9]{40}", ref):
        raise ValueError("Use the approved full source commit, not a moving branch")
    actual_ref = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if actual_ref != ref:
        raise ValueError("Unexpected source ref identity")
    asset_run = asset_run.resolve()
    reference_job = reference_job.resolve()
    negative_manifest = negative_manifest.resolve()
    verify_run(asset_run)
    verify_job(reference_job, pixels=True)
    source_job = load_job(reference_job / "manifest.json")
    if source_job.backend != "personal-cpu-experimental" or len(source_job.problems) != 6:
        raise ValueError("Expected the fixed six-page experimental CPU reference job")
    negative = load_job(negative_manifest)
    if negative.backend != "personal-cpu-experimental" or len(negative.problems) != 1:
        raise ValueError("Expected one explicit experimental rejection input")
    assets = read(asset_run / "reports/result.json")
    private_zip = Path(assets["archive"])
    if sha256(private_zip) != assets["archive_sha256"]:
        raise ValueError("Private candidate archive changed")
    profile_ref = source_job.profile
    if not profile_ref:
        raise ValueError("The reference must explicitly pin its candidate Profile")
    run = Run.create(
        "personal-cpu-installed-verification",
        workspace=workspace_root(workspace),
        seed=source_job.render.seed,
        config={
            "source_commit": ref,
            "asset_run": str(asset_run),
            "asset_run_sha256": sha256(asset_run / "run.json"),
            "reference_job": str(reference_job),
            "reference_report_sha256": sha256(reference_job / "validation_report.json"),
            "reference_manifest_sha256": sha256(reference_job / "manifest.json"),
            "negative_manifest": str(negative_manifest),
            "negative_manifest_sha256": sha256(negative_manifest),
            "python_targets": ["3.12", "3.13"],
            "platform": "Linux",
            "publication": False,
            "neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    protected = {
        str(path): sha256(path)
        for path in (
            ROOT / "workspace/latest.json",
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
            ROOT / "docs/MASTER_EXECUTION_PLAN.md",
        )
    }
    commands: list[dict] = []
    environment = {
        **os.environ,
        "UV_CACHE_DIR": "/tmp/hwsol-uv-cache",
        "UV_OFFLINE": "1",
        "PYTHONUTF8": "1",
    }
    for name in ("PYTHONPATH", "VIRTUAL_ENV", "HWSOL_DATA_DIR", "HWSOL_WORKSPACE"):
        environment.pop(name, None)

    def command(
        args: list[str],
        *,
        cwd: Path,
        settings: dict | None = None,
        expected_failure: bool = False,
    ) -> subprocess.CompletedProcess:
        number = len(commands) + 1
        started = time.monotonic()
        completed = subprocess.run(
            args, cwd=cwd, env=settings or environment, capture_output=True, text=True
        )
        log = run.path / "logs" / f"command-{number:02d}.txt"
        log.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
        commands.append(
            {
                "args": args,
                "cwd": str(cwd),
                "returncode": completed.returncode,
                "expected_failure": expected_failure,
                "seconds": round(time.monotonic() - started, 3),
                "log": log.relative_to(run.path).as_posix(),
            }
        )
        atomic_json(run.path / "logs/commands.json", commands)
        if (completed.returncode != 0) != expected_failure:
            raise RuntimeError(f"Unexpected command {number} status; see {log}")
        return completed

    def result(args: list[str], **kwargs) -> dict:
        return json.loads(command(args, **kwargs).stdout)

    def archive_job(output: Path, label: str, name: str) -> Path:
        destination = run.path / "rendered" / f"python-{label}" / name
        shutil.copytree(output, destination)
        verify_job(destination, pixels=True)
        return destination

    def save_cli_receipt(receipt: dict, label: str, name: str) -> None:
        source = Path(receipt["run"]) / "run.json"
        target = run.path / "reports/cli-receipts" / f"python-{label}-{name}.json"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)

    try:
        shutil.copyfile(__file__, run.path / "logs/personal_cpu_install.py")
        source_archive = run.path / "logs/selected-source.zip"
        command(
            [
                "git",
                "archive",
                "--format=zip",
                "--prefix=selected-ref/",
                "-o",
                str(source_archive),
                ref,
            ],
            cwd=ROOT,
        )
        extracted = run.path / "logs/selected-ref"
        with ZipFile(source_archive) as archive:
            archive.extractall(run.path / "logs")
        # Use exactly the selected ref's existing policy/stager. Refuse a mixed
        # implementation if this harness is being run from a different helper.
        if sha256(extracted / "scripts/package_release.py") != sha256(
            ROOT / "scripts/package_release.py"
        ):
            raise ValueError("Checkout packaging helper differs from the approved source commit")
        atomic_json(
            extracted / "src/hwsol/_build.json",
            {
                "commit": ref,
                "lock_sha256": sha256(extracted / "uv.lock"),
                "purpose": "unreleased personal CPU installation verification",
                "git_history_included": False,
            },
        )
        source = run.path / "logs/public-source"
        receipt = stage_public_source(extracted, source)
        atomic_json(run.path / "reports/public-source-receipt.json", receipt)
        epoch = command(["git", "show", "-s", "--format=%ct", ref], cwd=ROOT).stdout.strip()
        environment["SOURCE_DATE_EPOCH"] = epoch
        distribution = run.path / "reports/dist"
        distribution.mkdir()
        version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
        write_skill_archive(source, distribution / f"handwritten-solutions-{version}-skill.zip")
        command(
            [
                "uv",
                "build",
                "--offline",
                "--python",
                sys.executable,
                "--out-dir",
                str(distribution),
            ],
            cwd=source,
        )
        packages = sorted(
            [
                *distribution.glob("*.whl"),
                *distribution.glob("*.tar.gz"),
                *distribution.glob("*-skill.zip"),
            ]
        )
        if len(packages) != 3:
            raise ValueError("Expected exactly wheel, sdist and Skill archives")
        audits = [audit_package(path) for path in packages]
        if any(
            audit["unsafe_asset_entries"]
            or audit["research_modules"]
            or audit["local_identity_metadata"]
            for audit in audits
        ):
            raise ValueError("A public archive contains private or research assets")
        wheel = next(path for path in packages if path.suffix == ".whl")
        with ZipFile(wheel) as archive:
            names = archive.namelist()
            kernel_names = [name for name in names if name.startswith("hwsol/personal_ink/")]
            if not any(name.endswith("/runtime.py") for name in kernel_names):
                raise ValueError("Installed wheel omits the new CPU runtime")
            private_strings = []
            for name in names:
                if name.endswith((".py", ".json", ".md", ".txt")):
                    data = archive.read(name)
                    if b"/home/klmn/" in data or b"source_pdf/session-" in data:
                        private_strings.append(name)
            if private_strings:
                raise ValueError(f"Private literal paths in public runtime: {private_strings}")
        atomic_json(
            run.path / "reports/package-audit.json",
            {
                "archives": audits,
                "personal_cpu_runtime_members": kernel_names,
                "private_payload_in_public_archives": False,
            },
        )
        control = new_text_fixture(run.path / "solutions/default-control")
        fixed_reference = copy_job_input(
            reference_job / "manifest.json", run.path / "solutions/reference-input"
        )
        fixed_negative = copy_job_input(negative_manifest, run.path / "solutions/unsupported-input")
        results = []
        interpreters = [
            ("3.12", ROOT / ".venv/bin/python"),
            ("3.13", Path("/tmp/hwsol-v030-py313/bin/python")),
        ]
        for label, interpreter in interpreters:
            temporary = Path(tempfile.mkdtemp(prefix=f"hwsol-personal-cpu-{label}-"))
            venv = temporary / "runtime"
            settings = {
                **environment,
                "UV_PROJECT_ENVIRONMENT": str(venv),
                "HWSOL_DATA_DIR": str(temporary / "private"),
                "HWSOL_WORKSPACE": str(temporary / "workspace"),
                "MPLCONFIGDIR": str(temporary / "mpl"),
            }
            entry = {"python_target": label, "environment": str(temporary), "status": "running"}
            results.append(entry)
            atomic_json(run.path / "reports/installations.json", results)
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
            entry["isolation"] = result(
                [
                    str(python),
                    "-I",
                    "-c",
                    "import json,sys,importlib.metadata as m;from importlib.util import find_spec;"
                    "import hwsol,hwsol.personal_ink.runtime;"
                    "assert find_spec('torch') is None;assert find_spec('research') is None;"
                    "assert find_spec('pytest') is None;"
                    "print(json.dumps({'python':sys.version,'hwsol':hwsol.__file__,"
                    "'torch':False,'research':False,'packages':"
                    "{d.metadata['Name']:d.version for d in m.distributions()}}))",
                ],
                cwd=temporary,
                settings=settings,
            )
            if not Path(entry["isolation"]["hwsol"]).resolve().is_relative_to(venv):
                raise ValueError(
                    "The installed process imported hwsol outside its clean environment"
                )
            entry["doctor_before"] = result([str(cli), "doctor"], cwd=temporary, settings=settings)
            if entry["doctor_before"]["status"] != "passed":
                raise ValueError("Installed CPU doctor failed")
            control_input = copy_job_input(control, temporary / "inputs/control")
            control_output = temporary / "outputs/control"
            control_render = result(
                [
                    str(cli),
                    "render",
                    "--manifest",
                    str(control_input),
                    "--output",
                    str(control_output),
                ],
                cwd=temporary,
                settings=settings,
            )
            entry["control_verify"] = result(
                [str(cli), "verify", "--job", str(control_output), "--pixels"],
                cwd=temporary,
                settings=settings,
            )
            control_report = read(control_output / "validation_report.json")
            if control_report["build"]["commit"] != ref:
                raise ValueError("Installed control does not identify the approved source commit")
            if control_report["profile"]["reference"] != "bundled-default@v1":
                raise ValueError("Clean default control selected private handwriting")
            archive_job(control_output, label, "control")
            save_cli_receipt(control_render, label, "control")
            latest = Path(settings["HWSOL_WORKSPACE"]) / "latest.json"
            settings_file = Path(settings["HWSOL_DATA_DIR"]) / "settings.json"
            before = {"latest": sha256(latest), "settings": digest_or_missing(settings_file)}
            entry["candidate_import"] = result(
                [str(cli), "profiles", "import", "--source", str(private_zip)],
                cwd=temporary,
                settings=settings,
            )
            if (
                not entry["candidate_import"].get("candidate")
                or entry["candidate_import"]["activated"]
            ):
                raise ValueError("Private Profile was not kept as an inactive candidate")
            candidate_input = copy_job_input(fixed_reference, temporary / "inputs/candidate")
            candidate_output = temporary / "outputs/candidate"
            candidate_render = result(
                [
                    str(cli),
                    "render",
                    "--manifest",
                    str(candidate_input),
                    "--output",
                    str(candidate_output),
                ],
                cwd=temporary,
                settings=settings,
            )
            if candidate_render["status"] != "held":
                raise ValueError("Candidate rendering incorrectly became a production delivery")
            entry["candidate_verify"] = result(
                [str(cli), "verify", "--job", str(candidate_output), "--pixels"],
                cwd=temporary,
                settings=settings,
            )
            if read(candidate_output / "validation_report.json")["build"]["commit"] != ref:
                raise ValueError("Installed candidate does not identify the approved source commit")
            entry["reference_comparison"] = compare_pages(reference_job, candidate_output, 6)
            candidate_saved = archive_job(candidate_output, label, "candidate")
            save_cli_receipt(candidate_render, label, "candidate")
            exported = temporary / "candidate-export.zip"
            entry["candidate_export"] = result(
                [
                    str(cli),
                    "profiles",
                    "export",
                    profile_ref,
                    "--output",
                    str(exported),
                    "--candidate",
                ],
                cwd=temporary,
                settings=settings,
            )
            if zip_inventory(exported) != zip_inventory(private_zip):
                raise ValueError("Candidate CLI export changed a portable asset")
            exports = run.path / "solutions/exports"
            exports.mkdir(exist_ok=True)
            shutil.copyfile(exported, exports / f"python-{label}.zip")
            roundtrip_settings = {
                **settings,
                "HWSOL_DATA_DIR": str(temporary / "roundtrip-private"),
            }
            entry["roundtrip_import"] = result(
                [str(cli), "profiles", "import", "--source", str(exported)],
                cwd=temporary,
                settings=roundtrip_settings,
            )
            roundtrip_profile = (
                Path(roundtrip_settings["HWSOL_DATA_DIR"])
                / "profiles"
                / profile_ref.split("@")[0]
                / "candidates"
                / profile_ref.split("@")[1]
                / "profile.json"
            )
            if sha256(roundtrip_profile) != assets["profile_sha256"]:
                raise ValueError("Round-trip import changed Profile identity")
            if (Path(roundtrip_settings["HWSOL_DATA_DIR"]) / "settings.json").exists():
                raise ValueError("Round-trip import activated a candidate")
            entry["roundtrip_assets_exact"] = True
            negative_input = copy_job_input(fixed_negative, temporary / "inputs/unsupported")
            rejected_output = temporary / "outputs/unsupported"
            rejected = command(
                [
                    str(cli),
                    "render",
                    "--manifest",
                    str(negative_input),
                    "--output",
                    str(rejected_output),
                ],
                cwd=temporary,
                settings=settings,
                expected_failure=True,
            )
            rejection = read(rejected_output / "validation_report.json")
            if (
                rejection.get("status") != "failed"
                or (rejected_output / "homework_solutions.pdf").exists()
            ):
                raise ValueError("Unsupported formula produced an accepted partial output")
            entry["unsupported_rejection"] = {
                "returncode": rejected.returncode,
                "report": rejection,
            }
            negative_copy = run.path / "reports" / f"python-{label}-unsupported.json"
            shutil.copyfile(rejected_output / "validation_report.json", negative_copy)
            command(
                [str(cli), "verify", "--job", str(rejected_output), "--pixels"],
                cwd=temporary,
                settings=settings,
                expected_failure=True,
            )
            entry["doctor_after"] = result([str(cli), "doctor"], cwd=temporary, settings=settings)
            entry["profiles_after"] = result(
                [str(cli), "profiles", "list"], cwd=temporary, settings=settings
            )
            if entry["profiles_after"]["default"] != "bundled-default@v1":
                raise ValueError("Candidate use changed the default Profile")
            after = {"latest": sha256(latest), "settings": digest_or_missing(settings_file)}
            if before != after:
                raise ValueError("Candidate actions changed the baseline latest or settings")
            entry.update(
                status="passed",
                baseline_pointers_before=before,
                baseline_pointers_after=after,
                output=str(candidate_saved),
            )
            atomic_json(run.path / "reports/installations.json", results)
        cross = {
            "candidate": compare_pages(
                run.path / "rendered/python-3.12/candidate",
                run.path / "rendered/python-3.13/candidate",
                6,
            ),
            "control": compare_pages(
                run.path / "rendered/python-3.12/control",
                run.path / "rendered/python-3.13/control",
                1,
            ),
        }
        atomic_json(run.path / "reports/cross-python-pixels.json", cross)
        if any(sha256(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("A protected production source changed")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        summary = {
            "source_commit": ref,
            "linux_python_3_12_3_13": "passed",
            "candidate_pages_per_environment": 6,
            "all_reference_and_cross_python_pixels_exact": True,
            "inactive_private_candidate_roundtrip": "passed",
            "unsupported_input_explicitly_rejected": True,
            "default_profile_and_baseline_latest_preserved": True,
            "public_archives_exclude_private_data_and_research": True,
            "no_torch_required": True,
            "macos": "not tested on this Linux host",
            "production_release": False,
            "new_style_acceptance_claim": False,
            "neural_updates": 0,
        }
        run.finish("held", result=summary)
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--asset-run", required=True, type=Path)
    parser.add_argument("--reference-job", required=True, type=Path)
    parser.add_argument("--negative-manifest", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    print(
        launch(
            args.ref,
            args.asset_run,
            args.reference_job,
            args.negative_manifest,
            workspace=args.workspace,
        )
    )


if __name__ == "__main__":
    main()
