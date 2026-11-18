"""Same English target, common white-paper display and actual independent OCR."""

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from hwsol.lab.text_lines import transcription_scores
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-run", type=Path, required=True)
    parser.add_argument("--model-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    binary = Path(shutil.which("tesseract"))
    run = Run.create(
        "neural-english-comparison",
        run_id=args.run_id,
        seed=42,
        config={
            "normalization": "white background, ink height 64px, uniform scaling, 20px border",
            "ocr": "Tesseract eng psm7, normalized image scaled 4x, no expected text supplied",
            "ocr_binary_sha256": sha256(binary),
            "ocr_version": subprocess.check_output(
                [str(binary), "--version"], text=True
            ).splitlines()[0],
            "ocr_english_model_sha256": sha256(
                Path("/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata")
            ),
            "author": "personal; font uses scanned glyphs, model uses natural scanned line",
            "style_equivalence": "same author, different reference modalities; style not graded",
        },
    )
    try:
        rows, displays = [], []
        text = (args.font_run / "rendered/solutions/01.md").read_text().strip()
        for root in [args.font_run, *args.model_runs]:
            verify_run(root)
            native = root / "rendered/pages/01.png"
            image = Image.open(native).convert("L")
            if root == args.font_run:
                report = json.loads((root / "rendered/validation_report.json").read_text())
                glyphs = report["pages"][0]["glyph_provenance"]
                boxes = [g["bounds_px"] for g in glyphs[1:] if len(g["bounds_px"]) == 4]
                box = (
                    math.floor(min(b[0] for b in boxes)) - 2,
                    math.floor(min(b[1] for b in boxes)) - 2,
                    math.ceil(max(b[2] for b in boxes)) + 2,
                    math.ceil(max(b[3] for b in boxes)) + 2,
                )
                image = image.crop(box)
            else:
                target = json.loads((root / "reports/request.json").read_text())["text"]
                if target != text:
                    raise ValueError("Cannot compare different target text")
            ink = ImageChops.invert(image).point(lambda x: 255 if x >= 50 else 0)
            box = ink.getbbox()
            if box is None:
                raise ValueError("Blank output is not a valid comparison")
            image = image.crop(box)
            image = image.resize(
                (round(image.width * 64 / image.height), 64), Image.Resampling.LANCZOS
            )
            normalized = Image.new("L", (image.width + 40, 104), 255)
            normalized.paste(image, (20, 20))
            path = run.path / "comparisons" / f"{root.name}.png"
            normalized.save(path)
            ocr_path = path.with_name(root.name + "-ocr.png")
            normalized.resize((normalized.width * 4, 416), Image.Resampling.LANCZOS).save(ocr_path)
            child = subprocess.run(
                [str(binary), str(ocr_path), "stdout", "-l", "eng", "--psm", "7"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            (run.path / "logs" / (root.name + ".txt")).write_text(child.stdout)
            (run.path / "logs" / (root.name + ".stderr.txt")).write_text(child.stderr)
            actual = " ".join(child.stdout.split())
            rows.append(
                {
                    "run": str(root),
                    "run_sha256": sha256(root / "run.json"),
                    "native_png_sha256": sha256(native),
                    "normalized_png_sha256": sha256(path),
                    "transcription": actual,
                    "scores": transcription_scores(text, actual),
                    "style_similarity": None,
                    "human_naturalness": None,
                }
            )
            displays.append((root.name, normalized))
        sheet = Image.new("L", (max(im.width for _, im in displays) + 40, len(displays) * 150), 255)
        draw = ImageDraw.Draw(sheet)
        for index, (name, image) in enumerate(displays):
            draw.text((20, index * 150 + 5), name, fill=0)
            sheet.paste(image, (20, index * 150 + 30))
        sheet.save(run.path / "rendered/pages/01.png")
        sheet.convert("RGB").save(run.path / "rendered/homework_solutions.pdf")
        (run.path / "solutions/target.txt").write_text(text + "\n")
        atomic_json(
            run.path / "reports/comparison.json",
            {
                "text": text,
                "rows": rows,
                "interpretation": "One smoke sentence; not benchmark or human style acceptance",
            },
        )
        print(run.finish("held", result={"comparison_executed": True, "accepted": False}))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
