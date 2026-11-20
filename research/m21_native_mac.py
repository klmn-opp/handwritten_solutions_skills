"""Run unchanged CPU candidate acceptance on an explicitly authorized native Mac."""

from __future__ import annotations

# ruff: noqa: E501 -- embedded remote shell/Python scripts preserve exact commands
import argparse
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.personal_cpu_install import compare_pages


def execute(
    host: str,
    socket: str,
    payload: Path,
    linux: Path,
    python312: str | None = None,
    installed_native: Path | None = None,
) -> Path:
    verify_run(payload)
    verify_run(linux)
    archive = payload / "solutions/mac-acceptance.tar.gz"
    expected_linux = json.loads((linux / "rendered/python-3.13/receipt.json").read_text())
    if expected_linux["status"] != "passed":
        raise ValueError("Require passed exact Linux installed evidence")
    files = list((payload / "solutions/mac-acceptance").iterdir())
    bound_names = {}
    for role in ("wheel", "profile", "rollback_profile"):
        matches = [
            p
            for p in files
            if p.is_file() and sha256(p) == expected_linux["inputs"][role]["sha256"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Payload lacks one exact bound {role}")
        bound_names[role] = matches[0].name
    binding = json.loads((payload / "solutions/mac-acceptance/acceptance-input.json").read_text())
    if binding.get("schema") != "native-acceptance-input/1":
        raise ValueError("Unknown native payload schema")
    for key in ("source_commit", "package_version"):
        if binding[key] != expected_linux[key]:
            raise ValueError(f"Payload {key} differs from installed candidate")
    for role, name in bound_names.items():
        if (
            binding[role] != name
            or binding["inputs"][role] != expected_linux["inputs"][role]["sha256"]
        ):
            raise ValueError(f"Payload binding mismatch: {role}")
    if (
        binding["job"] != "job/job.json"
        or binding["inputs"]["job"] != expected_linux["inputs"]["job"]["sha256"]
    ):
        raise ValueError("Job binding mismatch")
    parent_binding = None
    reuse = {"3.12": "", "3.13": ""}
    if installed_native:
        verify_run(installed_native)
        parent_receipts = {}
        for version in reuse:
            path = installed_native / f"rendered/native/python-{version}/receipt.json"
            prior = json.loads(path.read_text())
            if (
                prior["status"] != "passed"
                or prior["actual_system"] != "Darwin"
                or not prior["clean_install_completed"]
            ):
                raise ValueError("Require genuinely clean, completed native installation parent")
            if (
                prior["source_commit"] != expected_linux["source_commit"]
                or prior["package_version"] != expected_linux["package_version"]
                or prior["inputs"]["wheel"]["sha256"] != expected_linux["inputs"]["wheel"]["sha256"]
            ):
                raise ValueError("Reused native environment differs from selected package")
            remote_root = Path(prior["inputs"]["wheel"]["path"]).parents[1]
            runtime = str(remote_root / f"python-{version}/environment/bin/python")
            reuse[version] = "--runtime-python " + shlex.quote(runtime)
            if version == "3.12":
                python312 = runtime
            parent_receipts[version] = {
                "path": str(path),
                "sha256": sha256(path),
                "runtime_python": runtime,
            }
        parent_binding = {
            "run": str(installed_native),
            "run_sha256": sha256(installed_native / "run.json"),
            "receipts": parent_receipts,
        }
    run = Run.create(
        "m21-native-macos",
        seed=1909,
        config={
            "purpose": "actual owner-authorized native Mac clean installs and new four Job",
            "payload_run": str(payload),
            "payload_archive_sha256": sha256(archive),
            "linux_run": str(linux),
            "linux_receipt_sha256": sha256(linux / "run.json"),
            "source_commit": expected_linux["source_commit"],
            "production_activation": False,
            "clean_install_parent": parent_binding,
        },
    )
    shutil.copyfile(__file__, run.path / "logs/m21_native_mac.py")
    print(run.path, flush=True)
    ssh = ["ssh", "-S", socket, "-o", "BatchMode=yes", host]
    scp = ["scp", "-o", f"ControlPath={socket}", "-o", "BatchMode=yes"]
    commands = []

    def command(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
        number = len(commands) + 1
        log = run.path / f"logs/{number:02d}.txt"
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        commands.append(
            {
                "argv": argv,
                "returncode": process.returncode,
                "seconds": time.monotonic() - started,
                "log": str(log),
            }
        )
        atomic_json(run.path / "logs/commands.json", commands)
        print(f"command {number}: {process.returncode}", flush=True)
        if check and process.returncode:
            raise RuntimeError(f"Command {number} failed; see {log}")
        return process

    try:
        remote = "hwsol-acceptance/" + run.path.name
        qremote = shlex.quote(remote)
        command(ssh + [f"mkdir -p hwsol-acceptance && mkdir {qremote} && uname -sm && sw_vers"])
        command(scp + [str(archive), f"{host}:{remote}/payload.tar.gz"])
        command(
            ssh
            + [
                f"cd {qremote} && test \"$(shasum -a 256 payload.tar.gz | cut -d ' ' -f 1)\" = {sha256(archive)} && tar -xzf payload.tar.gz && cd mac-acceptance && shasum -a 256 -c SHA256SUMS"
            ]
        )
        # This script only modifies the fresh acceptance directory. Public dependencies
        # are downloaded; the original personal assets remain private on the owned Mac.
        remote_script = run.path / "logs/run-native.sh"
        remote_script.write_text(
            """#!/bin/sh
set -u
cd "$(dirname "$0")"
ROOT="$PWD"
[ "$(uname -s)" = Darwin ] || exit 2
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export UV_CACHE_DIR="$ROOT/uv-cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/toolchains"
export UV_PYTHON_BIN_DIR="$ROOT/toolchain-bin"
cd mac-acceptance
/usr/local/bin/python3.13 -I cross_platform_candidate_check.py --require-system Darwin --wheel @WHEEL@ --profile @PROFILE@ --rollback-profile @ROLLBACK@ --job job/job.json --output "$ROOT/python-3.13" @REUSE313@ > "$ROOT/python-3.13.log" 2>&1
printf '%s\\n' "$?" > "$ROOT/python-3.13.exit"
cd "$ROOT"
runtime=@PYTHON312@
bootstrap_status=0
if [ -n "$runtime" ]; then
    test -x "$runtime" || bootstrap_status=2
    "$runtime" --version > bootstrap.log 2>&1
else
    /usr/local/bin/python3.13 -m venv bootstrap && bootstrap/bin/python -m pip install uv > bootstrap.log 2>&1 && bootstrap/bin/uv python install 3.12 >> bootstrap.log 2>&1
    bootstrap_status=$?
    if [ "$bootstrap_status" = 0 ]; then runtime=$(bootstrap/bin/uv python find 3.12); fi
fi
if [ "$bootstrap_status" = 0 ]; then
    cd mac-acceptance
    "$runtime" -I cross_platform_candidate_check.py --require-system Darwin --wheel @WHEEL@ --profile @PROFILE@ --rollback-profile @ROLLBACK@ --job job/job.json --output "$ROOT/python-3.12" @REUSE312@ > "$ROOT/python-3.12.log" 2>&1
    printf '%s\\n' "$?" > "$ROOT/python-3.12.exit"
else
    printf '%s\\n' "$bootstrap_status" > "$ROOT/python-3.12.exit"
fi
cd "$ROOT"
/usr/local/bin/python3.13 - <<'COLLECT'
import json, os, pathlib, shutil, subprocess, zipfile
root=pathlib.Path.cwd()
(root/'hardware.json').write_text(json.dumps({name:subprocess.check_output(cmd,text=True).strip() for name,cmd in {'uname':['uname','-a'],'macOS':['sw_vers'],'hardware':['sysctl','-n','hw.model'],'architecture':['uname','-m']}.items()},indent=2))
with zipfile.ZipFile('evidence.zip','w',zipfile.ZIP_DEFLATED) as out:
    for folder in ['python-3.12','python-3.13']:
        base=root/folder
        if not base.exists(): continue
        runtime=base/'environment/bin/python'
        if not runtime.exists() and (base/'receipt.json').exists():
            receipt=json.loads((base/'receipt.json').read_text())
            installed=pathlib.Path(receipt.get('runtime',{}).get('hwsol',''))
            for parent in installed.parents:
                if parent.name=='environment': runtime=parent/'bin/python'; break
        if runtime.exists():
            probe="import json,platform; from PIL import features; import matplotlib.ft2font as ft; print(json.dumps({'system':platform.system(),'python':platform.python_version(),'Pillow':{n:{'available':features.check(n),'version':features.version(n)} for n in ['freetype2','raqm','harfbuzz','fribidi']},'matplotlib_freetype':ft.__freetype_version__,'matplotlib_freetype_build':ft.__freetype_build_type__},indent=2))"
            result=subprocess.run([str(runtime),'-c',probe],text=True,capture_output=True,env={**os.environ,'MPLCONFIGDIR':str(base/'mpl')})
            (base/'runtime-features.json').write_text(result.stdout if result.returncode==0 else json.dumps({'probe_failed':True,'returncode':result.returncode,'stderr':result.stderr}))
        for file in base.rglob('*'):
            if not file.is_file(): continue
            relative=file.relative_to(base)
            if relative.parts[0] in {'environment','store','roundtrip-store','mpl'}: continue
            if relative.parts[0]=='workspace':
                if file.name=='run.json': out.write(file,f'{folder}/cli-receipts/{file.parent.name}.json')
                continue
            out.write(file,file.relative_to(root))
    for name in ['hardware.json','run-native.sh','python-3.12.log','python-3.13.log','python-3.12.exit','python-3.13.exit','bootstrap.log']:
        if (root/name).is_file(): out.write(root/name,name)
COLLECT
""".replace("@WHEEL@", shlex.quote(bound_names["wheel"]))
            .replace("@PROFILE@", shlex.quote(bound_names["profile"]))
            .replace("@ROLLBACK@", shlex.quote(bound_names["rollback_profile"]))
            .replace("@PYTHON312@", shlex.quote(python312 or ""))
            .replace("@REUSE312@", reuse["3.12"])
            .replace("@REUSE313@", reuse["3.13"]),
            encoding="utf-8",
        )
        command(scp + [str(remote_script), f"{host}:{remote}/run-native.sh"])
        command(ssh + [f"sh {qremote}/run-native.sh"], check=False)
        destination = run.path / "reports/native-evidence.zip"
        command(scp + [f"{host}:{remote}/evidence.zip", str(destination)])
        shutil.unpack_archive(destination, run.path / "rendered/native")
        results = []
        for version in ("3.12", "3.13"):
            base = run.path / f"rendered/native/python-{version}"
            receipt_file = base / "receipt.json"
            if not receipt_file.exists():
                results.append({"python": version, "status": "failed", "reason": "no receipt"})
                continue
            receipt = json.loads(receipt_file.read_text())
            item = {
                "python": version,
                "status": receipt["status"],
                "receipt_sha256": sha256(receipt_file),
                "actual_system": receipt["actual_system"],
                "clean_install_completed": receipt.get("clean_install_completed"),
            }
            if receipt["status"] == "passed":
                assert receipt["actual_system"] == "Darwin"
                assert receipt["source_commit"] == expected_linux["source_commit"]
                assert receipt["package_version"] == expected_linux["package_version"]
                assert (
                    receipt["inputs"]["wheel"]["sha256"]
                    == expected_linux["inputs"]["wheel"]["sha256"]
                )
                assert receipt["job_input_inventory"] == expected_linux["job_input_inventory"]
                for name in ("profile", "rollback_profile"):
                    assert (
                        receipt["inputs"][name]["sha256"]
                        == expected_linux["inputs"][name]["sha256"]
                    )
                try:
                    item["linux_comparison"] = compare_pages(
                        linux / "rendered/python-3.13/candidate", base / "candidate", 4
                    )
                    item["linux_exact"] = True
                except ValueError as exc:
                    item["linux_exact"] = False
                    item["linux_difference"] = str(exc)
            results.append(item)
        atomic_json(run.path / "reports/native-results.json", results)
        run.finish(
            "held",
            result={
                "mac_native_execution": True,
                "installations": results,
                "production_release": False,
                "author_style_approval": "separate gate",
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
    parser.add_argument("--host", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--linux", type=Path, required=True)
    parser.add_argument(
        "--installed-native",
        type=Path,
        help="Completed clean native parent for samewheel profile-only acceptance",
    )
    parser.add_argument(
        "--python312",
        help="Optional existing isolated native3.12 launcher; fresh venv still created",
    )
    args = parser.parse_args()
    print(
        execute(
            args.host,
            args.socket,
            args.payload.resolve(),
            args.linux.resolve(),
            args.python312,
            args.installed_native,
        )
    )


if __name__ == "__main__":
    main()
