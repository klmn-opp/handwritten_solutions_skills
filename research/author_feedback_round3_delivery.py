"""Assemble root-reviewed authentic f variation; retain unresolved original pi.

This is exposed author feedback regression, never production activation or a new
holdout. Parent runs and actual treatment sources must be approved by exact SHA.
"""

import argparse
import copy
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from handnote.markdown import parse
from PIL import Image
from pypdf import PdfWriter

from hwsol.jobs.layout import plain_emphasis
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import crop_pair, json_value, snapshot_sources
from research.author_feedback_revision import group_crops, trace
from research.author_feedback_round2_checks import page_geometry
from research.author_feedback_round3 import load_baseline, replay_page, validate_increment
from research.corpus.math_glyph_completion import local_comparison
from research.corpus.math_glyph_evidence import fallback_overview
from research.homework_benchmark.render import fallback_summary, read, save_pdf
from research.math_feedback_candidate import paired_panel

ROOT = Path(__file__).resolve().parents[1]


def f_words(source, title, records):
    """Bind complete original words to visible text, independent of f treatment."""
    spans = [title]
    for block in parse(plain_emphasis(source)):
        spans.extend(text for kind, text in block.segments if kind == "t")
    rows = [r for r in records if r["context"] == "text"]
    if [ch for span in spans for ch in span if not ch.isspace()] != [r["semantic"] for r in rows]:
        raise ValueError("Full-word comparison text binding failed")
    result, offset = [], 0
    for span_index, span in enumerate(spans):
        mapping = {}
        for j, ch in enumerate(span):
            if not ch.isspace():
                mapping[j] = rows[offset]["item_index"]
                offset += 1
        for match in re.finditer(r"[A-Za-z]+", span):
            if "f" in match.group():
                result.append(
                    {
                        "word": match.group(),
                        "span": span_index,
                        "range": list(match.span()),
                        "items": [mapping[j] for j in range(*match.span())],
                    }
                )
    return result


def group_box(rows, indices):
    boxes = [rows[i]["bounds_px"] for i in indices]
    return {
        "bounds_px": [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    approval = read(args.approval)
    if approval["scope"] != "isolated-root-reviewed-f-variation" or approval["author_acceptance"]:
        raise ValueError("Scoped source/root acceptance without invented author approval required")
    required_sources = {
        "research/f_variation_feedback.py",
        "research/author_feedback_round3.py",
        "research/author_feedback_round3_delivery.py",
    }
    if not required_sources <= approval["source_sha256"].keys() or "f" not in approval["parents"]:
        raise ValueError("Approval is missing a required f implementation or parent identity")
    sources = {}
    for key, record in approval["parents"].items():
        path = Path(record["path"])
        verify_run(path)
        if sha256(path / "run.json") != record["sha256"]:
            raise ValueError(f"Parent changed: {key}")
        sources[key] = path
    for name, digest in approval["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Reviewed source changed: {name}")
    baseline = load_baseline()
    if baseline["run_sha256"] != approval["baseline_sha256"]:
        raise ValueError("Author-reviewed shape baseline changed")
    # Import only after the actual source is present and explicitly approved.
    from research.f_variation_feedback import FVariationFeedback

    sources["baseline"] = baseline["path"]
    run = Run.create(
        "author-feedback-round3-authentic-f-complete-20",
        config={
            "argv": sys.argv,
            "approval": approval,
            "render": baseline["cfg"].model_dump(mode="json"),
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "production_activation": False,
            "pi": "retained; lighter historical 2pi is connected and not a safe isolated glyph",
        },
        seed=baseline["cfg"].seed,
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, sources)
        atomic_json(run.path / "reports/approval-receipt.json", approval)
        protected = {str(p / "run.json"): sha256(p / "run.json") for p in sources.values()}
        latest = workspace_root() / "latest.json"
        protected[str(latest)] = sha256(latest)
        atomic_json(run.path / "reports/protected-assets.json", protected)
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline")
        f_source = sources["f"] / approval["f_source_directory"]
        shutil.copytree(f_source, run.path / "solutions/f-sources")
        for key, parent in sources.items():
            if key != "baseline":
                shutil.copytree(parent / "reports", run.path / f"reports/inherited/{key}")
                for rel in approval.get("evidence_directories", {}).get(key, []):
                    shutil.copytree(parent / rel, run.path / "solutions/evidence" / key / rel)
        shutil.copyfile(
            baseline["path"] / "rendered/homework_solutions.pdf",
            run.path / "comparisons/previous-reviewed20.pdf",
        )
        cases, rows, decisions, images, maths = [], [], [], [], []
        words, targets, formulas, word_index = [], [], [], []
        for entry in baseline["cases"]:
            n, case_id = entry["page"], entry["case_id"]
            print(f"page {n:02d}: {case_id}", flush=True)
            replay = replay_page(baseline, n)
            page, renderer, original = replay["page"], replay["renderer"], replay["records"]
            before_paths = page_geometry(page, renderer)
            treatment = FVariationFeedback(
                run.path / "solutions/f-sources",
                source=replay["source"],
                title=case_id,
                seed=baseline["cfg"].seed,
            )
            records = json_value(treatment.apply(page, renderer, original))
            decision = json_value(copy.deepcopy(treatment.decisions))
            if json_value(trace(renderer)) != replay["parser"]:
                raise ValueError("Mathematical parser structure changed")
            if any(a != b for a, b in zip(original, records, strict=True) if a["semantic"] == "π"):
                raise ValueError("Unapproved pi record changed")
            image = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"rendered/pages/{n:02d}.pdf"
            )
            accepted_image = Image.open(
                sources["f"] / approval["f_rendered_directory"] / f"{n:02d}.png"
            ).convert("RGB")
            if image.size != accepted_image.size or image.tobytes() != accepted_image.tobytes():
                raise ValueError("Final delivery differs from the approved f candidate pixels")
            checks = validate_increment(
                before_paths, page, renderer, original, records, replay["image"], image
            )
            if any(original[i]["semantic"] != "f" for i in checks["changed_items"]):
                raise ValueError("Only f geometry is admitted; pi paths must remain exact")
            checks["pi_placed_geometry_exact"] = True
            checks["approved_f_parent_pixels_exact"] = True
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {"baseline": original, "candidate": records, "parser_trace": replay["parser"]},
            )
            atomic_json(run.path / f"reports/{case_id}-checks.json", checks)
            atomic_json(run.path / f"reports/{case_id}-decisions.json", decision)
            cases.append(
                {
                    "page": n,
                    "case_id": case_id,
                    "checks": checks,
                    "fallback": fallback_summary(records),
                }
            )
            decisions.append({"page": n, "f": decision})
            paired_panel(
                replay["image"],
                image,
                run.path / f"comparisons/pages/{n:02d}.png",
                f"Page {n}",
                ("Author-reviewed f shape", "Authentic source variation"),
            )
            formulas.extend(
                group_crops(
                    original,
                    replay["parser"],
                    replay["image"],
                    records,
                    replay["parser"],
                    image,
                    run.path / f"comparisons/all-formulas/{n:02d}",
                    n,
                    ("Author-reviewed f shape", "Authentic source variation"),
                )
            )
            for row in original:
                if row["semantic"] in {"f", "π"}:
                    i = row["item_index"]
                    a, b = crop_pair(row, records[i], replay["image"], image, (70, 35))
                    targets.append((n, i, a, b))
            for wi, word in enumerate(f_words(replay["source"], case_id, original)):
                a, b = crop_pair(
                    group_box(original, word["items"]),
                    group_box(records, word["items"]),
                    replay["image"],
                    image,
                    (25, 25),
                )
                words.append((n, wi, a, b))
                word_index.append({"page": n, "region": wi, **word})
            rows.extend({"page": n, "case_id": case_id, **r} for r in records)
            images.append(image)
            maths.append([r for r in records if r["context"] == "mathematics"])
        with PdfWriter() as writer:
            for c in cases:
                writer.append(run.path / f"rendered/pages/{c['page']:02d}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        for key, values in (
            ("all-f-words", words),
            ("all-f-pi-targets", targets),
            ("all-formulas", formulas),
        ):
            folder = run.path / f"comparisons/{key}"
            folder.mkdir(exist_ok=True)
            local_comparison(
                folder, values, labels=("Author-reviewed f shape", "Authentic source variation")
            )
            atomic_json(
                folder / "index.json",
                word_index
                if key == "all-f-words"
                else [{"page": v[0], "region": v[1]} for v in values],
            )
        overlay = run.path / "comparisons/fallback-overlay"
        overlay.mkdir()
        fallback_overview(images, maths, overlay)
        math_rows = [r for r in rows if r["context"] == "mathematics"]
        fallback = [r for r in math_rows if r["source_type"] in {"inherited-font", "external-font"}]
        if (
            len(cases) != 20
            or len(math_rows) != 2265
            or len(targets) != 165
            or len(formulas) != 173
        ):
            raise ValueError("Original benchmark inventory changed")
        summary = {
            "pages": 20,
            "math_visible_items": 2265,
            "all_f_pi_pairs": len(targets),
            "whole_f_word_pairs": len(words),
            "all_formula_pairs": len(formulas),
            "changed_items": sum(len(c["checks"]["changed_items"]) for c in cases),
            "fallbacks": len(fallback),
            "fallback_symbols": dict(Counter(r["semantic"] for r in fallback)),
            "math_source_types": dict(Counter(r["source_type"] for r in math_rows)),
            "pi_unchanged": 20,
            "pixels_changed_outside_target_boxes": 0,
            "new_unrelated_contacts": 0,
            "all_non_target_provenance_fields_exact": True,
            "all_approved_parent_page_pixels_exact": True,
            "new_neural_updates": 0,
            "author_acceptance": False,
            "production_go": False,
            "root_final_visual_review": "pending after this immutable run",
        }
        for key, value in (
            ("cases", cases),
            ("decisions", decisions),
            ("glyph-provenance", rows),
            ("fallback-reasons", fallback),
            ("glyph-summary", summary),
        ):
            atomic_json(run.path / f"reports/{key}.json", value)
        if any(sha256(Path(p)) != digest for p, digest in protected.items()):
            raise ValueError("Protected baseline/source/latest receipt changed")
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
