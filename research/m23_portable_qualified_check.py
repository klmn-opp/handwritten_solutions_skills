"""Standalone installed-CLI qualification check; no package install or production writes.

All mutable stores/workspaces are fresh external temporary directories. This is an
operation witness, not a claim of clean installation or independent author approval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def safe(root: Path, name: str) -> Path:
    value = PurePosixPath(name)
    if (
        not name
        or value.is_absolute()
        or "\\" in name
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"Unsafe relative input: {name}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Input escapes its directory: {name}")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink input is not accepted: {name}")
    return path


def members(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}


def copy_job(original: Path, destination: Path) -> tuple[Path, dict]:
    data = read(original)
    paths = set()
    if (data.get("background") or {}).get("path"):
        paths.add(data["background"]["path"])
    for problem in data["problems"]:
        paths.add(problem["solution"])
        paths.update(source["path"] for source in problem["sources"])
        if (problem.get("background") or {}).get("path"):
            paths.add(problem["background"]["path"])
    destination.mkdir(parents=True)
    inventory = {}
    for name in sorted(paths):
        source, target = safe(original.parent, name), safe(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory[name] = digest(source)
        if digest(target) != inventory[name]:
            raise ValueError("Copied Job asset changed")
    target = destination / "job.json"
    if target.exists():
        raise ValueError("Job input asset collides with generated manifest")
    save(target, data)
    return target, inventory


PROBE = """import hashlib,json,platform,sys,hwsol
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from zipfile import ZipFile
from hwsol.identity import build_identity,runtime_text_capabilities
from hwsol.profiles.store import ProfileStore
assert find_spec('research') is None and find_spec('torch') is None
root=Path(hwsol.__file__).parent.parent
with ZipFile(sys.argv[1]) as z:
 names=[n for n in z.namelist() if n.startswith('hwsol/') and not n.endswith('/')]
 assert names and len(names)==len(set(names))
 for n in names:
  assert hashlib.sha256((root/n).read_bytes()).digest()==hashlib.sha256(z.read(n)).digest()
profile=ProfileStore().resolve(sys.argv[2],candidate=True)
print(json.dumps({'actual_system':platform.system(),'machine':platform.machine(),
 'python':sys.version,'runtime_python':sys.executable,'module_path':hwsol.__file__,
 'package_version':version('handwritten-solutions'),'module_version':hwsol.__version__,
 'installed_wheel_members_exact':len(names),'build':build_identity(),
 'qualification_environment':runtime_text_capabilities(profile.font)}))
"""


def execute(args: argparse.Namespace) -> dict:
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    state = Path(tempfile.mkdtemp(prefix="hwsol-qualified-lifecycle-"))
    python = args.runtime_python.absolute()  # Retain the venv launcher symlink.
    cli = python.parent / ("hwsol.exe" if os.name == "nt" else "hwsol")
    evidence = args.evidence.resolve()
    request_path = evidence / "request.json" if evidence.is_dir() else evidence
    job, profile = args.job.resolve(), args.profile.resolve()
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "HWSOL_DATA_DIR",
            "HWSOL_WORKSPACE",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }
    }
    env.update(
        HWSOL_DATA_DIR=str(state / "store"),
        HWSOL_WORKSPACE=str(state / "workspace"),
        MPLCONFIGDIR=str(state / "mpl"),
    )
    receipt = {
        "schema": "portable-qualified-cpu-check/1",
        "status": "running",
        "actual_system": platform.system(),
        "machine": platform.machine(),
        "runtime_python": str(python),
        "isolated_state": str(state),
        "clean_install_completed": False,
        "package_install_performed": False,
        "production_activation": False,
        "production_publication": False,
        "author_approval_inferred": False,
        "commands": [],
        "operations": {},
    }
    shutil.copyfile(__file__, output / "m23_portable_qualified_check.py")
    save(output / "receipt.json", receipt)

    def command(argv: list, *, failure: str | None = None, settings: dict | None = None) -> dict:
        start = time.monotonic()
        proc = subprocess.run(
            list(map(str, argv)), cwd=output, env=settings or env, capture_output=True, text=True
        )
        log = output / f"logs/{len(receipt['commands']) + 1:02}.txt"
        log.parent.mkdir(exist_ok=True)
        log.write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
        receipt["commands"].append(
            {
                "argv": list(map(str, argv)),
                "returncode": proc.returncode,
                "seconds": time.monotonic() - start,
                "expected_failure": failure,
                "log": str(log.relative_to(output)),
            }
        )
        save(output / "receipt.json", receipt)
        if failure is not None:
            if proc.returncode == 0 or failure not in proc.stderr + proc.stdout:
                raise ValueError(f"Expected explicit rejection was not observed: {log}")
            return {"rejected": True, "reason": failure}
        if proc.returncode:
            raise ValueError(f"Installed command failed: {log}")
        return json.loads(proc.stdout)

    def call(*parts, **kwargs):
        return command([cli, *parts], **kwargs)

    try:
        request = read(request_path)
        with ZipFile(profile) as archive:
            manifest_bytes = archive.read("profile.json")
            manifest = json.loads(manifest_bytes)
        spec = f"{manifest['profile']}@{manifest['version']}"
        profile_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if spec != request["profile"] or profile_sha != request["profile_sha256"]:
            raise ValueError("Profile archive and qualification request disagree")
        copied = output / "evidence"
        copied.mkdir()
        for name, expected in request["files"].items():
            source, target = safe(request_path.parent, name), safe(copied, name)
            if target == copied / "request.json" or digest(source) != expected:
                raise ValueError("Qualification evidence changed or collides with request")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if digest(target) != expected:
                raise ValueError("Copied qualification evidence differs")
        shutil.copyfile(request_path, copied / "request.json")
        shutil.copyfile(profile, output / "profile.zip")
        shutil.copyfile(job, output / "original-job.json")
        receipt["inputs"] = {
            "request": {"path": str(request_path), "sha256": digest(request_path)},
            "profile": {
                "path": str(profile),
                "sha256": digest(profile),
                "reference": spec,
                "manifest_sha256": profile_sha,
            },
            "job": {"path": str(job), "sha256": digest(job)},
        }
        wheel = safe(copied, request["build"]["wheel"]["path"])
        if digest(wheel) != request["build"]["wheel"]["sha256"]:
            raise ValueError("Wheel evidence mismatch")
        receipt["wheel_sha256"] = digest(wheel)
        result = call("profiles", "import", "--source", output / "profile.zip")
        if not result.get("candidate") or result.get("activated") or result.get("imported") != spec:
            raise ValueError("Imported ink gained authority or changed identity")
        before = command([python, "-I", "-c", PROBE, wheel, spec])
        if (
            before["actual_system"] != platform.system()
            or before["package_version"] != request["build"]["version"]
            or before["module_version"] != before["package_version"]
            or before["build"]["commit"] != request["build"]["commit"]
        ):
            raise ValueError("Actual installed runtime differs from qualification build")
        receipt["runtime"] = before
        receipt["source_commit"] = before["build"]["commit"]
        receipt["package_version"] = before["package_version"]
        if call("profiles", "list")["default"] != "bundled-default@v1":
            raise ValueError("Isolated store did not begin with bundled default")
        call("profiles", "activate", spec, failure="Published Profile not found")
        receipt["operations"]["unpublished_activation_rejected"] = True
        qualified = call("profiles", "qualify-ink", spec, "--evidence", copied / "request.json")
        if qualified.get("published") or qualified.get("activated"):
            raise ValueError("Qualification unexpectedly published or activated")
        call("profiles", "activate", spec, failure="Published Profile not found")
        if call("profiles", "list")["default"] != "bundled-default@v1":
            raise ValueError("Qualification changed the default")
        published = call(
            "profiles", "publish-ink", spec, "--qualification-id", qualified["qualification_id"]
        )
        if (
            published.get("activated")
            or call("profiles", "list")["default"] != "bundled-default@v1"
        ):
            raise ValueError("Publication unexpectedly activated")
        receipt["qualification"], receipt["publication"] = qualified, published
        normal_path, inventory = copy_job(job, output / "inputs")
        normal = read(normal_path)
        normal.update(backend="font-cpu", profile=spec, profile_sha256=profile_sha)
        save(normal_path, normal)
        receipt["job_input_inventory"] = inventory
        rendered = call("render", "--manifest", normal_path, "--output", output / "qualified")
        if rendered["status"] != "passed":
            raise ValueError("Qualified ordinary font-cpu did not pass")
        call("verify", "--job", output / "qualified", "--pixels")
        report = read(output / "qualified/validation_report.json")
        if not all(
            p["candidate"]["backend"] == "personal-cpu-qualified"
            and p["candidate"]["production_qualified"] is True
            for p in report["pages"]
        ):
            raise ValueError("Normal Job did not use the qualified personal engine")
        receipt["operations"]["ordinary_font_cpu"] = True
        call("profiles", "activate", spec)
        if call("profiles", "list")["default"] != spec:
            raise ValueError("Explicit isolated activation did not take effect")
        default_path = output / "inputs/default-job.json"
        save(
            default_path,
            {k: v for k, v in normal.items() if k not in {"profile", "profile_sha256"}},
        )
        default = call(
            "render", "--manifest", default_path, "--output", output / "activated-default"
        )
        if default["status"] != "passed":
            raise ValueError("Activated default did not render through the ordinary backend")
        call("verify", "--job", output / "activated-default", "--pixels")
        first = sorted((output / "qualified/pages").glob("*.png"))
        second = sorted((output / "activated-default/pages").glob("*.png"))
        if len(first) != len(normal["problems"]) or [digest(p) for p in first] != [
            digest(p) for p in second
        ]:
            raise ValueError("Explicit and activated-default page bytes differ")
        call("profiles", "rollback", "bundled-default@v1")
        if call("profiles", "list")["default"] != "bundled-default@v1":
            raise ValueError("Rollback did not restore bundled default")
        receipt["operations"]["activation_and_rollback"] = True
        exported = output / "profile-export.zip"
        call("profiles", "export", spec, "--output", exported)
        if members(exported) != members(output / "profile.zip"):
            raise ValueError("Export changed Profile assets or carried extra authority")
        other = {
            **env,
            "HWSOL_DATA_DIR": str(state / "reimport-store"),
            "HWSOL_WORKSPACE": str(state / "reimport-workspace"),
        }
        imported = call("profiles", "import", "--source", exported, settings=other)
        if not imported.get("candidate") or imported.get("activated"):
            raise ValueError("Reimport granted authority")
        call("profiles", "activate", spec, failure="Published Profile not found", settings=other)
        if call("profiles", "list", settings=other)["default"] != "bundled-default@v1":
            raise ValueError("Reimport changed the second store default")
        after = command([python, "-I", "-c", PROBE, wheel, spec])
        if before != after:
            raise ValueError("Runtime or text capabilities changed during lifecycle")
        receipt.update(
            status="passed",
            runtime_unchanged=True,
            page_count=len(first),
            png_sha256=[digest(p) for p in first],
            report_sha256=digest(output / "qualified/validation_report.json"),
        )
        receipt["operations"].update(
            qualification=True,
            publication_without_activation=True,
            export_reimport_without_authority=True,
        )
    except Exception as exc:
        receipt.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        receipts = output / "cli-receipts"
        receipts.mkdir(exist_ok=True)
        child_runs = [
            p
            for workspace in (state / "workspace", state / "reimport-workspace")
            for p in (workspace / "runs").glob("*/run.json")
        ]
        for source in sorted(child_runs):
            target = receipts / f"{source.parent.name}-receipt.json"
            shutil.copyfile(source, target)
        # Preserve raw child receipts in the external state; archive the final local
        # store without exposing nested run.json files to a caller's Run inventory.
        if (state / "store").is_dir():
            with ZipFile(
                output / "isolated-store-snapshot.zip", "w", compression=ZIP_DEFLATED
            ) as archive:
                for source in sorted((state / "store").rglob("*")):
                    if source.is_file():
                        archive.write(source, source.relative_to(state / "store").as_posix())
        receipt["child_receipts"] = {p.name: digest(p) for p in sorted(receipts.glob("*.json"))}
        save(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--profileZIP", "--profile", dest="profile", type=Path, required=True)
    parser.add_argument(
        "--evidence", type=Path, required=True, help="Evidence directory or request.json"
    )
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args)
    print(
        json.dumps(
            {"status": result["status"], "receipt": str(args.output.absolute() / "receipt.json")}
        )
    )


if __name__ == "__main__":
    main()
