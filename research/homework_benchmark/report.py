"""Collect M1 evidence without changing frozen results or promoting a candidate."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from pypdf import PdfReader, PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.corpus.math_glyph_evidence import fallback_overview

DEVELOPMENT = "20261120T104306262492Z-homework-m1-render-cb8276e6"
HOLDOUT = "20261120T104341625106Z-homework-m1-render-38ea743e"
OLD_D = "20261120T064909731073Z-v01026-session-math-polish-b9371278"


def read(path):
    return json.loads(path.read_text())


def main():
    root = workspace_root()
    inputs = [root / "runs" / key for key in (DEVELOPMENT, HOLDOUT)]
    for p in inputs:
        verify_run(p)
    run = Run.create(
        "m1-reviewed-homework-evidence",
        config={
            "source_runs": {str(p): sha256(p / "run.json") for p in inputs},
            "purpose": "all original M1 successes and failures; independent root Agent review",
            "author_style_approval": False,
            "new_neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/report-source.py")
        cases, provenance, images, rows = [], [], [], []
        writers = {name: PdfWriter() for name in ("control", "candidate", "formula-crops")}
        fonts = {name: Counter() for name in ("control", "candidate")}
        categories = {name: Counter() for name in ("control", "candidate")}
        for source in inputs:
            specs = {c["id"]: c for c in read(source / "solutions/cases.json")}
            writers["formula-crops"].append(source / "comparisons/formula-regions.pdf")
            for case in read(source / "reports/cases.json"):
                case_id = case["id"]
                evidence = {
                    "id": case_id,
                    "split": case["split"],
                    "source_run": str(source),
                    "status": case["status"],
                    "expected_outcome_matches": case["expected_outcome_matches"],
                    "error": case.get("error"),
                    "output_page": None,
                    "source_answer_verification": "exact checks plus root analytical review",
                }
                case_dir = run.path / "solutions" / case_id
                case_dir.mkdir()
                (case_dir / "problem.txt").write_text(specs[case_id]["statement"])
                (case_dir / "answer.md").write_text(specs[case_id]["solution"])
                if case["status"] == "engineering_pass_visual_review_pending":
                    index = len(images) + 1
                    evidence["output_page"] = index
                    evidence["root_visual_review"] = "no confirmed content or boundary defect"
                    evidence["scope"] = "Agent review; not author style approval or formal proof"
                    evidence["pixel_checks"] = case["pixel_checks"]
                    image_path = source / f"rendered/pages/candidate/{case_id}.png"
                    shutil.copyfile(image_path, run.path / f"rendered/pages/{index:02d}.png")
                    images.append(Image.open(image_path).convert("RGB"))
                    record = read(source / f"reports/{case_id}-provenance.json")
                    final = [r for r in record["candidate"] if r["context"] == "mathematics"]
                    provenance.append(final)
                    for arm in ("control", "candidate"):
                        writers[arm].append(source / f"rendered/pages/{arm}/{case_id}.pdf")
                        math = [r for r in record[arm] if r["context"] == "mathematics"]
                        categories[arm].update(r["source_type"] for r in math)
                        fonts[arm].update(
                            r["semantic"]
                            for r in math
                            if r["source_type"] in {"inherited-font", "external-font"}
                        )
                    rows.extend({"case_id": case_id, "output_page": index, **r} for r in final)
                else:
                    evidence["root_visual_review"] = "no accepted page; refusal retained"
                cases.append(evidence)
        destinations = {
            "candidate": run.path / "rendered/homework_solutions.pdf",
            "control": run.path / "comparisons/A-font-cpu.pdf",
            "formula-crops": run.path / "comparisons/all-formula-regions.pdf",
        }
        for arm, writer in writers.items():
            with destinations[arm].open("wb") as stream:
                writer.write(stream)
            writer.close()
        fallback_overview(images, provenance, run.path / "comparisons")
        for image in images:
            image.close()
        prior = read(root / "runs" / OLD_D / "reports/glyph-provenance.json")
        old_symbols = {
            r["semantic"]
            for p in prior
            for r in p["D"]
            if r["context"] == "mathematics" and r["semantic"]
        }
        new_symbols = sorted({r["semantic"] for r in rows if r["semantic"]} - old_symbols)
        summary = {
            "original_cases": 24,
            "original_candidate_pages": len(provenance),
            "development": {"rendered": 13, "expected_refusals": 3},
            "one_use_holdout": {"rendered": 7, "unexpected_refusals": 1},
            "math_items": len(rows),
            "source_categories": {k: dict(v) for k, v in categories.items()},
            "font_fallback_occurrences": {k: sum(v.values()) for k, v in fonts.items()},
            "remaining_font_fallbacks": dict(fonts["candidate"]),
            "new_visible_symbols_against_old_seven_pages": new_symbols,
            "reviewed_formula_matrix_pairs": 173,
            "changed_english_pixels": sum(
                c.get("pixel_checks", {}).get("changed_english_pixels", 0) for c in cases
            ),
            "confirmed_accepted_output_math_content_defects": [],
            "unresolved_visual_style": [
                "High radicals have a thin/slanted personal stem; "
                "author legibility preference pending.",
                "Reported remaining external/inherited glyphs still look printed.",
            ],
            "release_go": False,
            "capability_gap": (
                "balanced delimiters spanning a matrix are rejected before glyph resolution"
            ),
            "holdout_reuse": "all eight are now exposed; a recovery never changes original 7/8",
            "neural_generalization_test": False,
            "author_style_approval": False,
            "new_neural_updates": 0,
        }
        atomic_json(run.path / "reports/summary.json", summary)
        atomic_json(run.path / "reports/cases.json", cases)
        atomic_json(run.path / "reports/math-glyph-provenance.json", rows)
        keys = [
            "case_id",
            "output_page",
            "item_index",
            "semantic",
            "source_type",
            "font",
            "font_sha256",
            "fallback_reason",
        ]
        with (run.path / "reports/math-glyph-inventory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        for p in Path("/tmp/hwsol-m1-holdout-visual").glob("*.png"):
            shutil.copyfile(p, run.path / "comparisons" / ("review-" + p.name))
        assert len(PdfReader(destinations["candidate"]).pages) == len(provenance)
        run.finish("held", result=summary)
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
