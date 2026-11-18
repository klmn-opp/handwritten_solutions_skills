"""Independent Chinese OCR smoke check of real DiffInk runs, never a style score."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from hwsol.lab.diffink import full_stroke_preview
from hwsol.lab.text_lines import transcription_scores
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    binary = Path(shutil.which("tesseract"))
    run = Run.create(
        "diffink-ocr",
        run_id=args.run_id,
        config={
            "tesseract_sha256": sha256(binary),
            "tessdata": json.loads((args.tessdata / "source.json").read_text()),
            "psm": 7,
            "normalization": "remove whitespace only, preserve punctuation; Chinese WER undefined",
            "target_reference_fed_to_ocr": False,
        },
    )
    try:
        scores = []
        for root in args.runs:
            verify_run(root)
            target = json.loads((root / "solutions/target.json").read_text())["text"]
            ref = json.loads((root / "solutions/reference.json").read_text())
            folder = run.path / "comparisons" / root.name
            full_stroke_preview(ref["points"], folder / "reference")
            for kind, image, expected in [
                ("reference", folder / "reference/pages/01.png", ref["text"]),
                ("generated", root / "rendered/pages/01.png", target),
            ]:
                child = subprocess.run(
                    [
                        str(binary),
                        str(image),
                        "stdout",
                        "--tessdata-dir",
                        str(args.tessdata),
                        "-l",
                        "chi_sim",
                        "--psm",
                        "7",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                (run.path / "logs" / f"{root.name}-{kind}.txt").write_text(child.stdout)
                (run.path / "logs" / f"{root.name}-{kind}.stderr.txt").write_text(child.stderr)
                actual = "".join(child.stdout.split())
                metric = transcription_scores("".join(expected.split()), actual)
                scores.append(
                    {
                        "run": str(root),
                        "run_sha256": sha256(root / "run.json"),
                        "image_sha256": sha256(image),
                        "kind": kind,
                        "expected": expected,
                        "ocr": actual,
                        "cer": metric["cer"],
                        "length_delta": metric["length_delta"],
                        "wer": None,
                        "style_similarity": None,
                        "human_naturalness": None,
                    }
                )
        atomic_json(
            run.path / "reports/ocr.json",
            {
                "cases": scores,
                "scope": "Two short targets with 30% conditioned prefix; not benchmark accuracy",
                "ocr_limitations": "Printed-font OCR also errs on real ink; require human reading",
            },
        )
        print(run.finish("held", result={"ocr_executed": True, "independent_acceptance": False}))
    except BaseException as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
