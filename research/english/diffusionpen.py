"""Limited English model comparison; never runs training or changes CPU runtime."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256

from .artifacts import create, finish_generated
from .corpus import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()
    cases, corpus_sha = load("smoke")
    if args.limit:
        cases = cases[: args.limit]
    checkout = Path("workspace/sources/v08/DiffusionPen").resolve()
    base = Path("workspace/checkpoints/diffusionpen-v08").resolve()
    refs = json.loads(
        Path("workspace/runs/v08-personal-references/reports/five-word-references.json").read_text()
    )
    locations = {
        "checkpoint": base / "diffusionpen_iam_model_path/models/ema_ckpt.pt",
        "style_checkpoint": base / "style_models/iam_style_diffusionpen.pth",
        "vae": Path("workspace/checkpoints/handnote-diffbrush/diffbrush/vae").resolve(),
        "scheduler": Path("workspace/checkpoints/sd15-config-v08/scheduler").resolve(),
        "canine": Path("workspace/checkpoints/canine-config-v08").resolve(),
    }
    hashes = {
        str(f): sha256(f)
        for path in locations.values()
        for f in (path.rglob("*") if path.is_dir() else [path])
        if f.is_file() and ".cache" not in f.parts
    }
    hashes.update({str(p): sha256(p) for p in checkout.glob("*.py")})
    hashes.update({r["image"]: r["sha256"] for r in refs})
    run = create(
        "english-diffusionpen",
        run_id=args.run_id,
        seed=args.seeds[0],
        config={
            "corpus_sha256": corpus_sha,
            "split": "smoke",
            "steps": 50,
            "seeds": args.seeds,
            "upstream_commit": "059e2f2bcb7372231038719e955597917c1a8783",
            "model_revision": "5636a68365001435ec76553114313c22c6f1ca07",
            "references": "five real personal scanned words; no online strokes",
            "training": False,
        },
    )
    try:
        req = {
            "run": str(run.path),
            "cases": cases,
            "seeds": args.seeds,
            "checkout": str(checkout),
            "references": refs,
            "file_sha256": hashes,
            **{k: str(v) for k, v in locations.items()},
            "upstream_commit": "059e2f2bcb7372231038719e955597917c1a8783",
        }
        atomic_json(run.path / "reports/request.json", req)
        rec = run.record()
        rec["checkpoints"] = hashes
        atomic_json(run.path / "run.json", rec)
        with (run.path / "logs/inference.txt").open("w") as log:
            child = subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    str(Path(__file__).with_name("diffusionpen_worker.py")),
                    "--request",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=log,
                timeout=7200,
                check=False,
            )
        report = json.loads((run.path / "reports/lines.json").read_text())
        if child.returncode or not report["model_executed"]:
            raise ValueError(report.get("error", "No genuine model execution"))
        rec = run.record()
        rec["hardware"]["cuda"] = report["hardware"]
        atomic_json(run.path / "run.json", rec)
        print(json.dumps(finish_generated(run), indent=2))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
