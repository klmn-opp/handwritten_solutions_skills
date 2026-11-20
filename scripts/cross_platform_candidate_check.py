"""Portable local acceptance using only stdlib and the installed hwsol CLI.

Run on the actual target OS; this never simulates macOS or uploads private files.
Every invocation requires a new output directory. Failed evidence is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path
from zipfile import ZipFile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text("utf-8"))


def inside(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if Path(name).is_absolute() or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe input path: {name}")
    return target


def archive_identity(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        return {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names}


def copy_job(source: Path, destination: Path, profile: dict | None = None) -> Path:
    data = read(source)
    names = {p["solution"] for p in data["problems"]}
    names.update(s["path"] for p in data["problems"] for s in p["sources"])
    for bg in [data.get("background"), *[p.get("background") for p in data["problems"]]]:
        if bg and bg.get("path"):
            names.add(bg["path"])
    destination.mkdir(parents=True)
    for name in names:
        target = inside(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(source.parent, name), target)
    if profile:
        data.update(
            backend="personal-cpu-experimental",
            realism="baseline",
            profile=profile["reference"],
            profile_sha256=profile["sha256"],
        )
    save(destination / "job.json", data)
    return destination / "job.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rollback-profile", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-system", choices=("Darwin", "Linux"))
    parser.add_argument(
        "--runtime-python",
        type=Path,
        help="Check an already installed runtime; DOES NOT count as clean install",
    )
    args = parser.parse_args()
    paths = {k: getattr(args, k).resolve() for k in ("wheel", "profile", "rollback_profile", "job")}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"PYTHONPATH", "VIRTUAL_ENV", "HWSOL_DATA_DIR", "HWSOL_WORKSPACE"}
    }
    env.update(
        HWSOL_DATA_DIR=str(output / "store"),
        HWSOL_WORKSPACE=str(output / "workspace"),
        MPLCONFIGDIR=str(output / "mpl"),
    )
    receipt = {
        "schema": "cross-platform-candidate-check/1",
        "status": "running",
        "actual_system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "launcher_python": sys.version,
        "require_system": args.require_system,
        "clean_install_requested": not args.runtime_python,
        "inputs": {k: {"path": str(v), "sha256": digest(v)} for k, v in paths.items()},
        "commands": [],
        "private_upload": False,
        "production_activation": False,
    }
    save(output / "receipt.json", receipt)
    shutil.copyfile(__file__, output / "cross_platform_candidate_check.py")

    def call(argv: list[str], *, fail=False, settings: dict | None = None):
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=output, env=settings or env, capture_output=True, text=True)
        number = len(receipt["commands"]) + 1
        log = output / f"logs/{number:02d}.txt"
        log.parent.mkdir(exist_ok=True)
        log.write_text(proc.stdout + "\nSTDERR\n" + proc.stderr)
        receipt["commands"].append(
            {
                "argv": argv,
                "returncode": proc.returncode,
                "seconds": time.monotonic() - started,
                "log": str(log),
            }
        )
        save(output / "receipt.json", receipt)
        if (proc.returncode != 0) != fail:
            raise ValueError(f"Unexpected command status: {log}")
        return proc.stdout

    try:
        if args.require_system and platform.system() != args.require_system:
            raise ValueError(
                "Actual OS does not match --require-system; no emulation claim allowed"
            )
        if args.runtime_python:
            python = args.runtime_python.absolute()
        else:
            venv.EnvBuilder(with_pip=True).create(output / "environment")
            python = (
                output / "environment" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            )
            install = [str(python), "-m", "pip", "install"]
            call(install + [str(paths["wheel"])])
        probe = call(
            [
                str(python),
                "-I",
                "-c",
                "import json,sys,platform,hwsol;from importlib.util import find_spec;"
                "assert find_spec('research') is None;assert find_spec('torch') is None;"
                "print(json.dumps({'python':sys.version,'system':platform.system(),"
                "'hwsol':hwsol.__file__}))",
            ]
        )
        receipt["runtime"] = json.loads(probe)
        if receipt["runtime"]["system"] != platform.system():
            raise ValueError("Runtime OS differs from launcher OS")
        # Installed package bytes must correspond to the supplied wheel, not an editable checkout.
        with ZipFile(paths["wheel"]) as archive:
            members = {
                n: hashlib.sha256(archive.read(n)).hexdigest()
                for n in archive.namelist()
                if n.startswith("hwsol/") and not n.endswith("/")
            }
            build = json.loads(archive.read("hwsol/_build.json"))
        root = Path(receipt["runtime"]["hwsol"]).parent.parent
        if any(not (root / n).is_file() or digest(root / n) != h for n, h in members.items()):
            raise ValueError("Installed package bytes differ from supplied wheel")
        receipt["installed_wheel_members_exact"] = len(members)
        receipt["source_commit"] = build["commit"]
        with ZipFile(paths["wheel"]) as archive:
            meta = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            receipt["package_version"] = next(
                line[9:]
                for line in archive.read(meta).decode().splitlines()
                if line.startswith("Version: ")
            )
        cli = python.parent / ("hwsol.exe" if os.name == "nt" else "hwsol")

        def command(*parts):
            return json.loads(call([str(cli), *map(str, parts)]))

        command("doctor")
        control = output / "control-input"
        control.mkdir()
        (control / "question.txt").write_text("Solve 5x-7=8.")
        (control / "answer.md").write_text("Add seven and divide by five.\n\n$$x=3$$\n")
        save(
            control / "job.json",
            {
                "schema_version": 1,
                "generated_by": "cross-platform-candidate-check/1",
                "problems": [
                    {
                        "number": "1",
                        "statement": "Solve 5x-7=8.",
                        "sources": [{"path": "question.txt"}],
                        "solution": "answer.md",
                    }
                ],
            },
        )
        command("render", "--manifest", control / "job.json", "--output", output / "control")
        command("verify", "--job", output / "control", "--pixels")
        latest = output / "workspace/latest.json"
        settings = output / "store/settings.json"
        before = {
            "latest": digest(latest),
            "settings": digest(settings) if settings.exists() else None,
        }
        profiles = {}
        for name in ("profile", "rollback_profile"):
            with ZipFile(paths[name]) as archive:
                raw = archive.read("profile.json")
                data = json.loads(raw)
            profiles[name] = {
                "reference": f"{data['profile']}@{data['version']}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            imported = command("profiles", "import", "--source", paths[name])
            if not imported.get("candidate") or imported["activated"]:
                raise ValueError("Import escaped inactive candidate area")
        if profiles["profile"]["reference"] == profiles["rollback_profile"]["reference"]:
            raise ValueError("Rollback requires a distinct version")
        frozen_job = copy_job(paths["job"], output / "inputs/frozen")
        receipt["job_input_inventory"] = {
            str(p.relative_to(frozen_job.parent)): digest(p)
            for p in sorted(frozen_job.parent.rglob("*"))
            if p.is_file()
        }
        outputs = []
        for label, name in (
            ("candidate", "profile"),
            ("rollback", "rollback_profile"),
            ("candidate-replay", "profile"),
        ):
            job = copy_job(frozen_job, output / f"inputs/{label}", profiles[name])
            rendered = output / label
            result = command("render", "--manifest", job, "--output", rendered)
            if result["status"] != "held" or result.get("result_status") != "passed":
                raise ValueError("Candidate status does not distinguish engineering/admission")
            command("verify", "--job", rendered, "--pixels")
            report = read(rendered / "validation_report.json")
            if report["build"]["commit"] != build["commit"]:
                raise ValueError("Render build identity differs from installed wheel")
            if not (rendered / "homework_solutions.pdf").is_file():
                raise ValueError("No complete PDF")
            pngs = sorted((rendered / "pages").glob("*.png"))
            if len(pngs) != len(read(job)["problems"]):
                raise ValueError("Page inventory mismatch")
            outputs.append(
                {
                    "label": label,
                    "profile": profiles[name],
                    "pngs": [digest(p) for p in pngs],
                    "report_sha256": digest(rendered / "validation_report.json"),
                }
            )
        if outputs[0]["pngs"] != outputs[2]["pngs"]:
            raise ValueError("Candidate changed after rollback/reselection")
        exported = output / "candidate-export.zip"
        command(
            "profiles",
            "export",
            profiles["profile"]["reference"],
            "--candidate",
            "--output",
            exported,
        )
        if archive_identity(exported) != archive_identity(paths["profile"]):
            raise ValueError("Profile export changed a packaged asset")
        roundtrip_env = {**env, "HWSOL_DATA_DIR": str(output / "roundtrip-store")}
        roundtrip = json.loads(
            call(
                [str(cli), "profiles", "import", "--source", str(exported)], settings=roundtrip_env
            )
        )
        if (
            roundtrip.get("imported") != profiles["profile"]["reference"]
            or not roundtrip.get("candidate")
            or roundtrip.get("activated")
        ):
            raise ValueError("Exported candidate reimport changed identity or admission")
        second_export = output / "roundtrip-export.zip"
        call(
            [
                str(cli),
                "profiles",
                "export",
                profiles["profile"]["reference"],
                "--candidate",
                "--output",
                str(second_export),
            ],
            settings=roundtrip_env,
        )
        if archive_identity(second_export) != archive_identity(paths["profile"]):
            raise ValueError("Export/import round trip changed candidate assets")
        if (output / "roundtrip-store/settings.json").exists():
            raise ValueError("Reimport unexpectedly initialized default settings")
        receipt["roundtrip_import"] = roundtrip
        receipt["roundtrip_archive_members_exact"] = True
        after = {
            "latest": digest(latest),
            "settings": digest(settings) if settings.exists() else None,
        }
        if before != after:
            raise ValueError("Candidate use changed defaults/latest")
        listing = command("profiles", "list")
        if listing["default"] != "bundled-default@v1":
            raise ValueError("Private candidate became default")
        receipt.update(
            status="passed",
            outputs=outputs,
            defaults_before=before,
            defaults_after=after,
            clean_install_completed=not args.runtime_python,
            operations={
                k: "passed"
                for k in (
                    "installed_identity",
                    "default_control",
                    "profile_import",
                    "formula_pdf_png",
                    "profile_export",
                    "profile_reimport",
                    "rollback_and_reselection",
                    "default_latest_protection",
                )
            },
            interpretation="Actual recorded OS only; no author style or cross-OS inference",
        )
    except Exception as exc:
        receipt.update(status="failed", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        save(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
