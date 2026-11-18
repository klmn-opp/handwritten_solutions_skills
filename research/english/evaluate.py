"""Independent recognizer observations with explicit pending human adjudication."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256

from .artifacts import create, lossless_pdf, normalize
from .metrics import aggregate, score


def evaluate(args):
    parents = [p.resolve() for p in args.runs]
    for parent in parents:
        state = json.loads((parent / "run.json").read_text())
        if state["status"] not in {"passed", "held"}:
            raise ValueError("Evaluation requires a sealed source run: " + str(parent))
    run = create(
        "english-independent-evaluation",
        run_id=args.run_id,
        seed=42,
        config={
            "parents": {str(p): sha256(p / "run.json") for p in parents},
            "recognition_only": True,
            "human_ground_truth": "pending",
        },
    )
    try:
        rows, requests = [], []
        for parent in parents:
            data = json.loads((parent / "reports/lines.json").read_text())
            for source in data["cases"]:
                if source["status"] != "generated":
                    continue
                sid = parent.name + "--" + source.get("sample_id", source["id"])
                original = parent / source["image"]
                image = run.path / "rendered/pages" / f"{sid}.png"
                info = normalize(original, image)
                row = {
                    **source,
                    "evaluation_id": sid,
                    "parent_run": parent.name,
                    "original_image": str(original),
                    "normalization": info,
                    "human_adjudication": None,
                    "true_content_error": None,
                }
                ocr_image = run.path / "comparisons" / f"{sid}-tesseract.png"
                from PIL import Image

                with Image.open(image) as im:
                    im.resize((im.width * 3, im.height * 3), Image.Resampling.LANCZOS).save(
                        ocr_image
                    )
                result = subprocess.run(
                    ["tesseract", str(ocr_image), "stdout", "-l", "eng", "--psm", "7"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
                (run.path / "logs" / f"{sid}-tesseract.txt").write_text(result.stderr)
                row["tesseract"] = {"text": result.stdout.strip(), "returncode": result.returncode}
                if result.returncode == 0:
                    row["tesseract"]["score"] = score(source["text"], result.stdout)
                rows.append(row)
                requests.append({"id": sid, "image": str(image), "sha256": sha256(image)})
        model = args.checkpoint.resolve()
        model_hashes = {str(p): sha256(p) for p in model.iterdir() if p.is_file()}
        rec = run.record()
        rec["checkpoints"] = model_hashes
        atomic_json(run.path / "run.json", rec)
        req = {
            "checkpoint": str(model),
            "images": requests,
            "output": str(run.path / "reports/htr.json"),
        }
        atomic_json(run.path / "reports/recognizer-request.json", req)
        with (run.path / "logs/htr.txt").open("w") as log:
            proc = subprocess.run(
                [
                    str(args.python.absolute()),
                    str(Path(__file__).with_name("htr_worker.py").resolve()),
                    "--request",
                    str(run.path / "reports/recognizer-request.json"),
                ],
                stdout=log,
                stderr=log,
                timeout=7200,
                check=False,
            )
        htr = (
            json.loads((run.path / "reports/htr.json").read_text())
            if (run.path / "reports/htr.json").exists()
            else {}
        )
        by_id = {r["id"]: r for r in htr.get("rows", [])}
        for row in rows:
            got = by_id.get(row["evaluation_id"])
            row["htr"] = got
            if got is not None:
                got["score"] = score(row["text"], got["text"])
            row["recognizers_agree_with_expected"] = bool(
                got
                and got["score"]["exact_match"]
                and row["tesseract"].get("score", {}).get("exact_match")
            )
            row["style_gate"] = "human-content-review-required"
        groups = {}
        for parent in parents:
            subset = [r for r in rows if r["parent_run"] == parent.name]
            groups[parent.name] = {
                name: aggregate(
                    [r[name]["score"] for r in subset if r.get(name) and "score" in r[name]]
                )
                for name in ("tesseract", "htr")
            }
            groups[parent.name]["source_case_count"] = len(
                json.loads((parent / "reports/lines.json").read_text())["cases"]
            )
            groups[parent.name]["recognizer_consensus_count"] = sum(
                r["recognizers_agree_with_expected"] for r in subset
            )
            groups[parent.name]["border_ink_count"] = sum(
                r["normalization"]["border_ink"] for r in subset
            )
            groups[parent.name]["by_category"] = {
                cat: {
                    name: aggregate(
                        [
                            r[name]["score"]
                            for r in subset
                            if r["category"] == cat and r.get(name) and "score" in r[name]
                        ]
                    )
                    for name in ("tesseract", "htr")
                }
                for cat in sorted({r["category"] for r in subset})
            }
        for parent in parents:
            subset = [r for r in rows if r["parent_run"] == parent.name]
            for field in ("split", "seed"):
                groups[parent.name]["by_" + field] = {
                    str(value): {
                        name: aggregate(
                            [
                                r[name]["score"]
                                for r in subset
                                if r.get(field) == value and r.get(name) and "score" in r[name]
                            ]
                        )
                        for name in ("tesseract", "htr")
                    }
                    for value in sorted({str(r.get(field)) for r in subset})
                }
                # Match serialized labels while preserving numeric seeds in observations.
                for value in groups[parent.name]["by_" + field]:
                    groups[parent.name]["by_" + field][value] = {
                        name: aggregate(
                            [
                                r[name]["score"]
                                for r in subset
                                if str(r.get(field)) == value and r.get(name) and "score" in r[name]
                            ]
                        )
                        for name in ("tesseract", "htr")
                    }
        atomic_json(
            run.path / "reports/observations.json",
            {
                "rows": rows,
                "groups": groups,
                "interpretation": "Recognizer discrepancies; human ground truth pending.",
                "htr_exit_code": proc.returncode,
            },
        )
        atomic_json(
            run.path / "reports/recognizer-provenance.json",
            {
                "tesseract_version": subprocess.check_output(["tesseract", "--version"], text=True),
                "eng_traineddata": {
                    str(p): sha256(p)
                    for p in Path("/usr/share/tesseract-ocr").rglob("eng.traineddata")
                },
                "htr_revision": "eaacaf452b06415df8f10bb6fad3a4c11e609406",
                "checkpoints": model_hashes,
            },
        )
        # One recognizer input per page, lossless and unaltered after normalization.
        atomic_json(
            run.path / "reports/pdf-integrity.json",
            lossless_pdf(
                [Path(r["image"]) for r in requests], run.path / "rendered/homework_solutions.pdf"
            ),
        )
        return run.finish(
            "held",
            result={
                "observations": len(rows),
                "htr_completed": len(by_id),
                "htr_model_executed": htr.get("model_executed", False),
                "human_content_verified": False,
            },
        )
    except BaseException as exc:
        run.fail(exc)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--python", type=Path, default=Path("workspace/environments/english-htr-v08/bin/python")
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("workspace/checkpoints/trocr-base-handwritten-v08-xet"),
    )
    print(json.dumps(evaluate(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
