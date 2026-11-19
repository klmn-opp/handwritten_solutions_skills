"""Local independent OCR observations; discrepancy is not verified text error."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create
from research.english.metrics import aggregate, score


def evaluate(comparison: Path) -> Path:
    verify_run(comparison)
    rows = json.loads((comparison / "reports/observations-input.json").read_text())
    checkpoint = Path("workspace/checkpoints/trocr-base-handwritten-v08-xet").resolve()
    run = create(
        "context-independent-recognition",
        config={
            "comparison": str(comparison.resolve()),
            "comparison_sha256": sha256(comparison / "run.json"),
            "interpretation": "recognizer discrepancies, not human-verified errors",
            "checkpoint": {
                str(p.relative_to(checkpoint)): sha256(p)
                for p in checkpoint.rglob("*")
                if p.is_file()
            },
        },
    )
    print(run.path, flush=True)
    try:
        request = {
            "checkpoint": str(checkpoint),
            "images": [],
            "output": str(run.path / "reports/htr-raw.json"),
        }
        for row in rows:
            if sha256(Path(row["image"])) != row["sha256"]:
                raise ValueError("Changed input")
            request["images"].append({k: row[k] for k in ("id", "image", "sha256")})
        atomic_json(run.path / "reports/htr-request.json", request)

        def tesseract(row):
            path = run.path / f"comparisons/{row['id']}-ocr.png"
            with Image.open(row["image"]) as image:
                image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS).save(
                    path
                )
            p = subprocess.run(
                ["tesseract", str(path), "stdout", "-l", "eng", "--psm", "7"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (run.path / f"logs/{row['id']}.stderr").write_text(p.stderr)
            (run.path / f"logs/{row['id']}.txt").write_text(p.stdout)
            if p.returncode:
                raise RuntimeError(p.stderr)
            return {"id": row["id"], "text": p.stdout.strip()}

        with ThreadPoolExecutor(max_workers=4) as pool:
            ocr = list(pool.map(tesseract, rows))
        atomic_json(run.path / "reports/tesseract-raw.json", ocr)
        (run.path / "logs/tesseract-version.txt").write_text(
            subprocess.check_output(["tesseract", "--version"], text=True)
        )
        python = Path("workspace/environments/english-htr-v08/bin/python")
        with (run.path / "logs/htr.log").open("w") as log:
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "research.english.htr_worker",
                    "--request",
                    str(run.path / "reports/htr-request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        htr = json.loads((run.path / "reports/htr-raw.json").read_text())["rows"]
        summaries, observations = {}, []
        for recognizer, raw in [("tesseract", ocr), ("trocr", htr)]:
            by_id = {r["id"]: r for r in raw}
            scored = [
                {
                    **r,
                    "recognizer": recognizer,
                    "observed": by_id[r["id"]]["text"],
                    "score": score(r["text"], by_id[r["id"]]["text"]),
                }
                for r in rows
            ]
            observations.extend(scored)
            summaries[recognizer] = {
                method: {
                    split: aggregate(
                        [
                            r["score"]
                            for r in scored
                            if r["method"] == method and r["split"] == split
                        ]
                    )
                    for split in ["train", "validation", "holdout"]
                }
                for method in sorted({r["method"] for r in rows})
            }
        atomic_json(run.path / "reports/observations.json", observations)
        atomic_json(run.path / "reports/summary.json", summaries)
        run.finish(
            "held",
            result={
                "images": len(rows),
                "observations": len(observations),
                "human_adjudicated": False,
                "training": False,
                "production_eligible": False,
            },
        )
    except Exception as exc:
        run.finish("failed", result={"error": repr(exc)})
        raise
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    print(evaluate(parser.parse_args().comparison))
