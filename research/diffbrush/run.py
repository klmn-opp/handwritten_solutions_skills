"""Create an immutable experiment and launch genuine Handnote DiffBrush inference."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import handnote
from handnote.linegen.weights import FILES

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, profile_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    source = Path(handnote.__file__).parent
    worker = Path(__file__).with_name("reproduce.py")
    run = Run.create(
        "handnote-diffbrush",
        run_id=args.run_id,
        seed=args.seed,
        config={
            "text": args.text,
            "steps": args.steps,
            "backend": "handnote-diffbrush-cuda",
            "handnote_commit": "114919f183817643c795e47e851ea25b96e3c393",
            "upstream_commit": "da9addc1140bdfec463b2d41a974a0a392f80798",
            "worker_sha256": sha256(worker),
            "profile": "personal@v3 author, static scan reference",
        },
    )
    try:
        if not args.text or any(c in args.text for c in "$\\\n="):
            raise ValueError("Prose only; no mathematical structures")
        reference = run.path / "solutions/reference.png"
        shutil.copyfile(args.reference, reference)
        checkpoints = {
            name: {
                "path": str(args.cache / "diffbrush" / name),
                "bytes": spec[1],
                "sha256": spec[2],
            }
            for name, spec in FILES.items()
        }
        record = run.record()
        record["checkpoints"] = checkpoints
        record["profile"] = profile_store().resolve("personal@v3").provenance()
        atomic_json(run.path / "run.json", record)
        request = {
            "run": str(run.path),
            "cache": str(args.cache.resolve()),
            "reference": str(reference),
            "reference_sha256": sha256(reference),
            "text": args.text,
            "steps": args.steps,
            "seed": args.seed,
            "content_sha256": sha256(args.cache / "diffbrush/content.npz"),
            "handnote_files": {
                str(f.relative_to(source)): sha256(f) for f in (source / "linegen").rglob("*.py")
            },
        }
        atomic_json(run.path / "reports/request.json", request)
        shutil.copyfile(
            Path(__file__).with_name("requirements-inference.lock"),
            run.path / "reports/dependencies.lock",
        )
        with (
            (run.path / "logs/stdout.txt").open("w") as out,
            (run.path / "logs/stderr.txt").open("w") as err,
        ):
            child = subprocess.run(
                [
                    str(args.python.absolute()),
                    str(worker.resolve()),
                    "--request",
                    str(run.path / "reports/request.json"),
                ],
                stdout=out,
                stderr=err,
                timeout=600,
                check=False,
            )
        report = json.loads((run.path / "reports/inference.json").read_text())
        if child.returncode or not report["model_executed"]:
            raise ValueError(report.get("error", "No real model output"))
        record = run.record()
        record["hardware"]["cuda"] = {key: report[key] for key in ("device", "torch", "cuda")}
        atomic_json(run.path / "run.json", record)
        print(
            run.finish(
                "held",
                result={
                    "actual_model_inference": "passed",
                    "content_and_style": "unvalidated",
                    "inference_seconds": report["timings"]["total"],
                },
            )
        )
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
