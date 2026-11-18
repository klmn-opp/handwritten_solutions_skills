"""Isolated v0.8 benchmark CLI: python -m research.english.cli --help."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import handnote
from handnote.linegen.weights import FILES

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store

from .artifacts import create, finish_generated
from .corpus import load


def diffbrush(args):
    cases, digest = load(args.split, purpose="diagnostic" if args.official_pairs else "evaluation")
    if args.limit:
        if args.split in {"all", "holdout"}:
            raise ValueError("Do not select holdout cases for diagnosis")
        cases = cases[: args.limit]
    run = create(
        "english-diffbrush",
        run_id=args.run_id,
        seed=args.seeds[0],
        profile=profile_store().resolve("personal@v3").provenance(),
        config={
            "corpus_sha256": digest,
            "split": args.split,
            "seeds": args.seeds,
            "steps": args.steps,
            "official_pairs": args.official_pairs,
            "style_mode": args.style_mode,
        },
    )
    try:
        reference = run.path / "solutions/reference.png"
        shutil.copyfile(args.reference, reference)
        checkout = args.checkout.resolve()
        asset_root = args.cache.resolve() / "diffbrush"
        assets = {str(asset_root / name): value[2] for name, value in FILES.items()}
        assets[str(asset_root / "content.npz")] = sha256(asset_root / "content.npz")
        sources = {str(p.resolve()): sha256(p) for p in checkout.rglob("*.py")}
        hn = Path(handnote.__file__).parent
        sources.update({str(p.resolve()): sha256(p) for p in (hn / "linegen").rglob("*.py")})
        req = {
            "run": str(run.path),
            "cases": cases,
            "seeds": args.seeds,
            "steps": args.steps,
            "cache": str(args.cache.resolve()),
            "checkout": str(checkout),
            "reference": str(reference),
            "reference_origin": "scanned-personal-line",
            "reference_sha256": sha256(reference),
            "style_mode": args.style_mode,
            "official_pairs": args.official_pairs,
            "file_sha256": {**sources, **assets},
        }
        atomic_json(run.path / "reports/request.json", req)
        rec = run.record()
        rec["checkpoints"] = assets
        rec["config"].update(
            handnote_commit="114919f183817643c795e47e851ea25b96e3c393",
            upstream_commit="da9addc1140bdfec463b2d41a974a0a392f80798",
        )
        atomic_json(run.path / "run.json", rec)
        with (
            (run.path / "logs/stdout.txt").open("w") as out,
            (run.path / "logs/stderr.txt").open("w") as err,
        ):
            child = subprocess.run(
                [
                    str(args.python.absolute()),
                    str(Path(__file__).with_name("diffbrush_worker.py").resolve()),
                    "--request",
                    str(run.path / "reports/request.json"),
                ],
                stdout=out,
                stderr=err,
                timeout=3600,
                check=False,
            )
        result = json.loads((run.path / "reports/lines.json").read_text())
        if child.returncode or not result["model_executed"]:
            raise ValueError(result.get("error", "No genuine inference"))
        rec = run.record()
        rec["hardware"]["cuda"] = result["hardware"]
        atomic_json(run.path / "run.json", rec)
        return finish_generated(run)
    except BaseException as exc:
        run.fail(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    font = sub.add_parser("font")
    font.add_argument("--profile", required=True)
    font.add_argument("--split", default="development")
    font.add_argument("--run-id", required=True)
    db = sub.add_parser("diffbrush")
    db.add_argument("--python", type=Path, required=True)
    db.add_argument("--cache", type=Path, default=Path("workspace/checkpoints/handnote-diffbrush"))
    db.add_argument("--checkout", type=Path, default=Path("workspace/sources/v08/DiffBrush"))
    db.add_argument(
        "--reference",
        type=Path,
        default=Path("out/next-stage-2026-11-18/line-experiment-final/reference-line.png"),
    )
    db.add_argument("--split", default="development")
    db.add_argument("--run-id", required=True)
    db.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    db.add_argument("--steps", type=int, default=50)
    db.add_argument("--style-mode", choices=["raw", "fit"], default="raw")
    db.add_argument("--official-pairs", action="store_true")
    db.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.command == "font":
        from .cpu import render

        result = render(args.profile, args.split, args.run_id)
    else:
        result = diffbrush(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
