"""Real optional Tesseract evaluation, only on English-only cases; no prompt fed to OCR."""

import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from hwsol.benchmarks.metrics import observed_metrics, prose
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run


def evaluate_ocr(target: Path, *, workspace=None, run_id=None, executable="tesseract"):
    verify_run(target)
    binary = shutil.which(executable)
    if not binary:
        raise ValueError("Tesseract unavailable; OCR must remain unmeasured")
    version = subprocess.check_output([binary, "--version"], text=True).splitlines()[0]
    rows = json.loads((target / "reports/benchmark.json").read_text())["cases"]
    run = Run.create(
        "benchmark-ocr",
        workspace=workspace,
        run_id=run_id,
        config={
            "target": str(target.resolve()),
            "target_sha256": sha256(target / "run.json"),
            "ocr_executable": binary,
            "ocr_sha256": sha256(Path(binary)),
            "version": version,
            "psm": 6,
            "language": "eng",
            "scale": 3,
            "normalization": "collapse whitespace only; case/punctuation preserved",
            "crop": "English body bounding box from provenance, excluding case header",
        },
    )
    results = []
    try:
        for row in rows:
            if row["category"] != "english" or row.get("outcome") != "rendered":
                continue
            root = Path(row["case_run"]) / "rendered"
            verify_run(Path(row["case_run"]))
            report = json.loads((root / "validation_report.json").read_text())["pages"][0]
            items = report["glyph_provenance"]
            label = row["case_id"]
            if "".join(g["semantic"] or "" for g in items[: len(label)]) != label:
                raise ValueError("Cannot identify English header safely")
            boxes = [g["bounds_px"] for g in items[len(label) :] if len(g["bounds_px"]) == 4]
            bounds = (
                math.floor(min(b[0] for b in boxes)) - 6,
                math.floor(min(b[1] for b in boxes)) - 6,
                math.ceil(max(b[2] for b in boxes)) + 6,
                math.ceil(max(b[3] for b in boxes)) + 6,
            )
            crop = run.path / "comparisons" / f"{label}.png"
            with Image.open(root / "pages/01.png") as image:
                body = image.crop(bounds)
                body.resize((body.width * 3, body.height * 3), Image.Resampling.LANCZOS).save(crop)
            child = subprocess.run(
                [binary, str(crop), "stdout", "-l", "eng", "--psm", "6"],
                capture_output=True,
                text=True,
                timeout=90,
                check=True,
            )
            (run.path / "logs" / f"{label}.txt").write_text(child.stdout, encoding="utf-8")
            (run.path / "logs" / f"{label}.stderr.txt").write_text(child.stderr, encoding="utf-8")
            actual = " ".join(child.stdout.split())
            expected = prose((root / "solutions/01.md").read_text("utf-8"))
            obs = {
                "artifact_sha256": sha256(root / "homework_solutions.pdf"),
                "method": "ocr",
                "evaluator": version,
                "reference_visible": False,
                "english_transcription": actual,
            }
            metrics = observed_metrics(expected, obs, obs["artifact_sha256"])
            results.append(
                {
                    "case_id": label,
                    "crop_bounds_px": bounds,
                    "observation": obs,
                    "expected_characters": len(expected),
                    "expected_words": len(expected.split()),
                    "scores": metrics["english"],
                }
            )
        chars = sum(r["expected_characters"] for r in results)
        words = sum(r["expected_words"] for r in results)
        summary = {
            "engine": version,
            "case_count": len(results),
            "micro_cer": sum(r["scores"]["cer"] * r["expected_characters"] for r in results) / chars
            if chars
            else None,
            "micro_wer": sum(r["scores"]["wer"] * r["expected_words"] for r in results) / words
            if words
            else None,
            "interpretation": (
                "OCR recoverability on English-only crops; "
                "not human naturalness or formula accuracy"
            ),
            "cases": results,
        }
        atomic_json(run.path / "reports/ocr.json", summary)
        return run.finish("passed", result={k: v for k, v in summary.items() if k != "cases"})
    except BaseException as exc:
        run.fail(exc)
        raise
