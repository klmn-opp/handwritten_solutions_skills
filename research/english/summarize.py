"""Generate comparative evidence without converting recognizer errors to ground truth."""

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json

from .artifacts import lossless_pdf
from .metrics import aggregate


def summarize(evaluation, root):
    data = json.loads((evaluation / "reports/observations.json").read_text())
    rows = data["rows"]
    summary = {"human_scores": None, "model_training_executed": False, "groups": data["groups"]}
    summary["paired_smoke"] = {}
    names = [
        "v08-font-v1-frozen",
        "v08-font-v3-frozen",
        "v08-diffbrush-official-smoke",
        "v08-diffusionpen-smoke",
        "v08-diffbrush-fit-smoke",
    ]
    for name in names:
        sample = [r for r in rows if r["parent_run"] == name and r.get("smoke")]
        summary["paired_smoke"][name] = {
            k: aggregate([r[k]["score"] for r in sample if r.get(k) and "score" in r[k]])
            for k in ["tesseract", "htr"]
        }
    generated = [
        r for r in rows if r["parent_run"] in {"v08-diffbrush-development", "v08-diffbrush-holdout"}
    ]
    summary["diffbrush_full"] = {
        k: aggregate([r[k]["score"] for r in generated if r.get(k) and "score" in r[k]])
        for k in ["tesseract", "htr"]
    }
    summary["diffbrush_full"]["requests"] = 240
    summary["diffbrush_full"]["by_input_length"] = {
        label: {
            k: aggregate(
                [
                    r[k]["score"]
                    for r in generated
                    if condition(len(r["text"])) and r.get(k) and "score" in r[k]
                ]
            )
            for k in ["tesseract", "htr"]
        }
        for label, condition in [
            ("below35", lambda n: n < 35),
            ("35to61", lambda n: 35 <= n <= 61),
            ("above61", lambda n: n > 61),
        ]
    }
    runtimes = {}
    for name in data["groups"]:
        parent = evaluation.parent / name
        content = json.loads((parent / "reports/lines.json").read_text())
        measured = [
            r
            for r in content["cases"]
            if r["status"] == "generated" and ("elapsed_seconds" in r or "render_seconds" in r)
        ]
        if not measured:
            continue
        times = [r.get("elapsed_seconds", r.get("render_seconds")) for r in measured]
        runtimes[name] = {
            "lines": len(measured),
            "median_seconds": statistics.median(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "sum_seconds": sum(times),
            "peak_cuda_bytes": max(r.get("peak_cuda_allocated_bytes", 0) for r in measured),
            "peak_host_rss_bytes": content.get(
                "peak_host_rss_bytes",
                json.loads((parent / "run.json").read_text())
                .get("result", {})
                .get("peak_host_rss_bytes"),
            ),
            "scope": "CPU includes verification; paired run excluded from speed ranking",
        }
    summary["runtime"] = runtimes
    # Export every suspect row, not only attractive examples. These remain review candidates.
    failures = [
        {
            "evaluation_id": r["evaluation_id"],
            "text": r["text"],
            "image": r["original_image"],
            "tesseract": r["tesseract"]["text"],
            "htr": r["htr"]["text"] if r.get("htr") else None,
            "border_ink": r["normalization"]["border_ink"],
            "verified_error": None,
        }
        for r in rows
        if not r["recognizers_agree_with_expected"]
    ]
    atomic_json(root / "reports/failure-review-queue.json", failures)
    atomic_json(root / "reports/summary.json", summary)
    pages = []
    page_times = {}
    for name in names[:4]:
        sample = [
            r
            for r in rows
            if r["parent_run"] == name
            and r.get("smoke")
            and r.get("seed") == 42
            and r["id"].endswith("-01")
        ]
        if len(sample) != 8:
            raise ValueError("Expected same eight categories for page comparison")
        sample.sort(key=lambda r: r["id"])
        start = time.perf_counter()
        page = Image.new("RGB", (1600, 2263), "white")
        draw = ImageDraw.Draw(page)
        draw.text((80, 30), name + " | 8 fixed development lines | seed42", fill="black")
        for i, row in enumerate(sample):
            with Image.open(row["original_image"]) as raw:
                im = raw.convert("RGB")
                ratio = min(1440 / im.width, 110 / im.height)
                im = im.resize(
                    (round(im.width * ratio), round(im.height * ratio)), Image.Resampling.LANCZOS
                )
                y = 100 + i * 260
                draw.text((80, y), row["id"] + ": " + row["text"], fill="black")
                page.paste(im, (80, y + 40))
        path = root / "rendered/pages" / (name + ".png")
        page.save(path)
        pages.append(path)
        page_times[name] = {
            "composition_seconds": time.perf_counter() - start,
            "sum_previously_measured_line_seconds": sum(
                r.get("elapsed_seconds", r.get("render_seconds", 0)) for r in sample
            ),
            "not_measured_end_to_end_page_inference": True,
            "line_count": len(sample),
            "native_ink_cropped": False,
        }
    atomic_json(root / "reports/page-cost.json", page_times)
    atomic_json(
        root / "reports/pdf-integrity.json",
        lossless_pdf(pages, root / "rendered/homework_solutions.pdf"),
    )
    # Failure sheet with equal maximum canvas per model; preserves each raw image separately.
    examples = ["ehb1-short-01", "ehb1-long-01", "ehb1-mixed-01", "ehb1-longwords-01"]
    selected = [
        r
        for key in examples
        for name in names[:4]
        for r in rows
        if r["id"] == key and r["parent_run"] == name and r.get("seed") == 42
    ]
    sheet = Image.new("RGB", (1600, 180 * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        y = i * 180
        draw.text((10, y + 5), row["parent_run"] + " | " + row["text"], fill="black")
        with Image.open(row["original_image"]) as raw:
            im = raw.convert("RGB")
            ratio = min(1560 / im.width, 100 / im.height)
            im = im.resize((round(im.width * ratio), round(im.height * ratio)))
            sheet.paste(im, (10, y + 30))
        draw.text((10, y + 140), "Tesseract: " + row["tesseract"]["text"], fill="black")
        if row.get("htr"):
            draw.text((10, y + 156), "TrOCR: " + row["htr"]["text"], fill="black")
    sheet.save(root / "comparisons/failure-examples.png")
    print(json.dumps({"paired_smoke": summary["paired_smoke"], "runtime": runtimes}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output-run", type=Path, required=True)
    a = p.parse_args()
    summarize(a.evaluation, a.output_run)
