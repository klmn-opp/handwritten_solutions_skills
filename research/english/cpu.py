"""Evaluate existing CPU profiles without modifying production rendering or latest."""

import json
import math
import resource
import shutil
import time

from PIL import Image
from pypdf import PdfWriter

from hwsol.jobs.runtime import render_job, verify_job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store

from .artifacts import create
from .corpus import load


def render(profile: str, split: str, run_id: str) -> dict:
    cases, digest = load(split)
    store = profile_store()
    selected = store.resolve(profile)
    run = create(
        "english-font-benchmark",
        run_id=run_id,
        seed=42,
        profile=selected.provenance(),
        config={
            "corpus_sha256": digest,
            "split": split,
            "backend": "font-cpu",
            "rendering": "existing production renderer, 8pt, white 300x100mm",
        },
    )
    rows, writer = [], PdfWriter()
    started = time.perf_counter()
    try:
        for case in cases:
            folder = run.path / "solutions" / case["id"]
            folder.mkdir()
            (folder / "01.md").write_text(case["text"] + "\n")
            (folder / "question.txt").write_text(case["text"] + "\n")
            job = {
                "schema_version": 1,
                "generated_by": "Fixed English render-only benchmark",
                "profile": profile,
                "realism": "stroke-enhanced",
                "render": {
                    "seed": 42,
                    "dpi": 150,
                    "font_size_pt": 8,
                    "page_width_mm": 300,
                    "page_height_mm": 100,
                    "margin_mm": 8,
                    "ink": "#000000",
                    "glyph_metrics": "contextual",
                    "semantic_layout": False,
                },
                "problems": [
                    {
                        "number": "1",
                        "statement": "Typeset exact supplied English.",
                        "sources": [{"path": "question.txt"}],
                        "solution": "01.md",
                    }
                ],
            }
            atomic_json(folder / "job.json", job)
            row = {**case, "backend": "font-cpu", "profile": profile, "seed": 42}
            target = run.path / "comparisons/jobs" / case["id"]
            target.parent.mkdir(exist_ok=True)
            start = time.perf_counter()
            try:
                render_job(folder / "job.json", target, store)
                verify_job(target, pixels=True)
                report = json.loads((target / "validation_report.json").read_text())
                glyphs = report["pages"][0]["glyph_provenance"][1:]
                boxes = [g["bounds_px"] for g in glyphs if len(g["bounds_px"]) == 4]
                bounds = (
                    math.floor(min(b[0] for b in boxes)) - 4,
                    math.floor(min(b[1] for b in boxes)) - 4,
                    math.ceil(max(b[2] for b in boxes)) + 4,
                    math.ceil(max(b[3] for b in boxes)) + 4,
                )
                line = run.path / "comparisons/lines" / (case["id"] + ".png")
                line.parent.mkdir(exist_ok=True)
                with Image.open(target / "pages/01.png") as image:
                    image.crop(bounds).save(line)
                page = run.path / "rendered/pages" / f"{len(writer.pages) + 1:03d}.png"
                shutil.copyfile(target / "pages/01.png", page)
                writer.append(target / "homework_solutions.pdf")
                row.update(
                    status="generated",
                    image=str(line.relative_to(run.path)),
                    image_sha256=sha256(line),
                    source_type="personal-font-composed",
                    profile_sha256=selected.provenance()["manifest_sha256"],
                    render_seconds=time.perf_counter() - start,
                    crop_bounds=bounds,
                    pdf_pixel_verification="passed",
                    glyph_source_counts={
                        kind: sum(g["source_type"] == kind for g in glyphs)
                        for kind in sorted({g["source_type"] for g in glyphs})
                    },
                )
            except (ValueError, OSError) as exc:
                row.update(status="failed", error=str(exc))
            rows.append(row)
            print(case["id"], row["status"], flush=True)
        if writer.pages:
            writer.write(run.path / "rendered/homework_solutions.pdf")
        atomic_json(run.path / "reports/lines.json", {"cases": rows})
        return run.finish(
            "held" if any(r["status"] == "failed" for r in rows) else "passed",
            result={
                "cases": len(rows),
                "generated": sum(r["status"] == "generated" for r in rows),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "human_content_verified": False,
            },
        )
    except BaseException as exc:
        run.fail(exc)
        raise
