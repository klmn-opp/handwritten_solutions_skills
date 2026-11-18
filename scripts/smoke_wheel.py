"""Install the built wheel into a clean, locked runtime, then exercise its CLI outside the tree."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = "hwsol/assets/presets/patrick-hand/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("handwritten-solutions")
    wheel = (args.dist / f"handwritten_solutions-{version}-py3-none-any.whl").resolve()
    with ZipFile(wheel) as archive:
        for asset in ("PatrickHand-Regular.ttf", "OFL.txt", "profile.json", "provenance.json"):
            if archive.read(ASSETS + asset) != (ROOT / "src" / ASSETS / asset).read_bytes():
                raise ValueError(f"Bundled asset differs: {asset}")
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        venv = work / "runtime"
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": str(venv),
            "PYTHONUTF8": "1",
            "HWSOL_DATA_DIR": str(work / "private"),
            "MPLCONFIGDIR": str(work / "mpl"),
        }
        for key in ("PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(key, None)

        def run(*cmd):
            return subprocess.run(cmd, cwd=work, env=env, check=True, stdout=subprocess.PIPE).stdout

        run(
            "uv",
            "sync",
            "--project",
            str(ROOT),
            "--python",
            sys.executable,
            "--locked",
            "--no-dev",
            "--no-install-project",
        )
        binary = venv / ("Scripts" if os.name == "nt" else "bin")
        python = binary / ("python.exe" if os.name == "nt" else "python")
        run("uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel))
        run(
            str(python),
            "-I",
            "-c",
            "from importlib.util import find_spec; "
            "assert find_spec('pytest') is None and find_spec('torch') is None",
        )
        cli = str(binary / ("hwsol.exe" if os.name == "nt" else "hwsol"))
        doctor = json.loads(run(cli, "doctor"))
        if doctor["status"] != "passed" or doctor["training_required"]:
            raise ValueError("Unexpected Runtime environment")
        run(
            cli,
            "render",
            "--manifest",
            str(ROOT / "examples/phase2/mixed/job.json"),
            "--output",
            "五题",
        )
        result = json.loads(run(cli, "verify", "--job", "五题", "--pixels"))
        if result["problem_order"] != ["2", "4", "6", "7", "9"]:
            raise ValueError("Installed wheel lost problem identity")
        print(
            json.dumps(
                {
                    **result,
                    "wheel": wheel.name,
                    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                }
            )
        )


if __name__ == "__main__":
    main()
