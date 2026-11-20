"""Integrate only explicitly reviewed second-round author corrections.

An external private approval receipt pins the individual experiments and sources.
Nothing here updates a production Profile, evaluation ledger, or latest pointer.
"""

import argparse
import copy
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfWriter

from hwsol.jobs.layout import ink_geometry
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import crop_pair, json_value, snapshot_sources
from research.author_feedback_revision import group_crops, trace
from research.author_feedback_round2 import load_baseline, replay_page
from research.author_feedback_round2_checks import page_geometry, validate_increment
from research.corpus.math_glyph_completion import local_comparison
from research.corpus.math_glyph_evidence import fallback_overview
from research.f_size_feedback import FSizeFeedback
from research.homework_benchmark.render import fallback_summary, read, save_pdf
from research.math_feedback_candidate import paired_panel
from research.numeral_pi_feedback import NumeralPiFeedback
from research.structural_rule_feedback import StructuralRuleFeedback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    approval = read(args.approval)
    if approval["scope"] != "isolated-root-reviewed-candidate" or approval["author_acceptance"]:
        raise ValueError(
            "A scoped root review receipt, without invented author approval, is required"
        )
    sources = {}
    for key, record in approval["parents"].items():
        source = Path(record["path"])
        verify_run(source)
        if sha256(source / "run.json") != record["sha256"]:
            raise ValueError(f"Approved parent changed: {key}")
        sources[key] = source
    for path, digest in approval["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Reviewed source changed: {path}")
    baseline = load_baseline()
    if baseline["run_sha256"] != approval["baseline_sha256"]:
        raise ValueError("Reviewed baseline changed")
    sources["baseline"] = baseline["path"]
    cfg = baseline["cfg"]
    run = Run.create(
        "author-feedback-round2-complete-20",
        config={
            "argv": sys.argv,
            "render": cfg.model_dump(mode="json"),
            "approval": approval,
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "production_activation": False,
            "author_style_acceptance": False,
        },
        seed=cfg.seed,
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
        for key in ("digits", "f", "rules"):
            shutil.copytree(sources[key] / "reports", run.path / f"reports/inherited/{key}")
            for relative in approval.get("evidence_directories", {}).get(key, []):
                shutil.copytree(
                    sources[key] / relative, run.path / "solutions/evidence" / key / relative
                )
        shutil.copyfile(
            baseline["path"] / "rendered/homework_solutions.pdf",
            run.path / "comparisons/previous-reviewed20.pdf",
        )
        digit = NumeralPiFeedback(baseline["profile"].font)
        rules_source = sources["rules"] / approval["rules_source_directory"]
        shutil.copytree(rules_source, run.path / "solutions/structural-sources")
        rules = StructuralRuleFeedback(run.path / "solutions/structural-sources")
        cases, all_rows, decisions, images, maths, panels, focused = [], [], [], [], [], [], []
        chosen = Counter()
        for entry in baseline["cases"]:
            number, case_id = entry["page"], entry["case_id"]
            print(f"page {number:02d}: {case_id}", flush=True)
            replay = replay_page(baseline, number)
            page, renderer = replay["page"], replay["renderer"]
            original = replay["records"]
            before_paths = page_geometry(page, renderer)
            records = digit.apply(page, renderer, original)
            digits_decision = copy.deepcopy(digit.decisions)
            f_size = FSizeFeedback(approval["f_preset"], source=replay["source"], title=case_id)
            records = f_size.apply(page, renderer, records)
            f_decision = copy.deepcopy(f_size.decisions)
            records = rules.apply(page, renderer, records)
            records = json_value(records)
            decision = json_value(
                {"digits_pi": digits_decision, "f": f_decision, "rules": rules.decisions}
            )
            if json_value(trace(renderer)) != replay["parser"]:
                raise ValueError("Original mathematical parser/layout structure changed")
            destination = run.path / f"rendered/pages/{number:02d}.pdf"
            image = save_pdf(renderer.to_vector_pdf([page]), destination)
            checks = validate_increment(
                before_paths, page, renderer, original, records, replay["image"], image
            )
            bounds, _ = ink_geometry(page, renderer)
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {"baseline": original, "candidate": records, "parser_trace": replay["parser"]},
            )
            atomic_json(run.path / f"reports/{case_id}-checks.json", checks)
            atomic_json(run.path / f"reports/{case_id}-decisions.json", decision)
            cases.append(
                {
                    "page": number,
                    "case_id": case_id,
                    "checks": checks,
                    "ink_bounds_px": bounds,
                    "fallback": fallback_summary(records),
                }
            )
            decisions.append({"page": number, **decision})
            paired_panel(
                replay["image"],
                image,
                run.path / f"comparisons/pages/{number:02d}.png",
                f"Page {number}",
                ("Previous author-reviewed candidate", "Second correction"),
            )
            groups = group_crops(
                original,
                replay["parser"],
                replay["image"],
                records,
                replay["parser"],
                image,
                run.path / f"comparisons/all-formulas/{number:02d}",
                number,
                ("Previous author-reviewed candidate", "Second correction"),
            )
            panels.extend(groups)
            for i in checks["changed_items"]:
                old, new = original[i], records[i]
                key = old.get("structure") or f"{old['context']}-{old['semantic']}"
                # A small index into the complete comparisons: first occurrences,
                # plus all actual changed roots and mathematical f, without seed selection.
                if chosen[key] < 2 or key == "radical-overbar" or key == "mathematics-f":
                    focused.append(
                        (number, i, *crop_pair(old, new, replay["image"], image, (90, 45)))
                    )
                    chosen[key] += 1
                a, b = crop_pair(old, new, replay["image"], image, (65, 30))
                paired_panel(
                    a,
                    b,
                    run.path / f"comparisons/all-targets/{number:02d}/{i:04d}.png",
                    f"Page {number} / item {i} / {key}",
                    ("Previous", "Second correction"),
                )
            all_rows.extend({"page": number, "case_id": case_id, **r} for r in records)
            images.append(image)
            maths.append([r for r in records if r["context"] == "mathematics"])
        with PdfWriter() as writer:
            for entry in cases:
                writer.append(run.path / f"rendered/pages/{entry['page']:02d}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        for name, values in (("focused", focused), ("all-formulas", panels)):
            folder = run.path / f"comparisons/{name}"
            folder.mkdir(exist_ok=True)
            local_comparison(
                folder, values, labels=("Previous author-reviewed candidate", "Second correction")
            )
            atomic_json(folder / "index.json", [{"page": x[0], "region": x[1]} for x in values])
        overlay = run.path / "comparisons/fallback-overlay"
        overlay.mkdir()
        fallback_overview(images, maths, overlay)
        math_rows = [r for r in all_rows if r["context"] == "mathematics"]
        fallbacks = [
            r for r in math_rows if r["source_type"] in {"inherited-font", "external-font"}
        ]
        if len(cases) != 20 or len(math_rows) != 2265 or len(panels) != 173:
            raise ValueError("Complete exposed benchmark inventory changed")
        summary = {
            "pages": len(cases),
            "math_visible_items": len(math_rows),
            "math_source_types": dict(Counter(r["source_type"] for r in math_rows)),
            "fallbacks": len(fallbacks),
            "fallback_symbols": dict(Counter(r["semantic"] for r in fallbacks)),
            "all_formula_pairs": len(panels),
            "focused_pairs": len(focused),
            "changed_items": sum(len(c["checks"]["changed_items"]) for c in cases),
            "pixels_changed_outside_target_boxes": 0,
            "new_unrelated_contacts": 0,
            "semantics_and_non_target_geometry_preserved": True,
            "pi": "original heavy real source retained; no reliable lighter original yet",
            "new_neural_updates": 0,
            "author_acceptance": False,
            "production_go": False,
            "root_full_page_visual_review": "pending after this immutable run",
            "reconstruction_or_new_holdout_claim": False,
        }
        for name, value in (
            ("cases", cases),
            ("decisions", decisions),
            ("glyph-provenance", all_rows),
            ("fallback-reasons", fallbacks),
            ("glyph-summary", summary),
        ):
            atomic_json(run.path / f"reports/{name}.json", value)
        with (run.path / "reports/math-glyph-coverage.csv").open("w", newline="") as stream:
            cols = [
                "page",
                "item_index",
                "semantic",
                "source_type",
                "structure",
                "variant",
                "fallback_reason",
            ]
            writer = csv.DictWriter(stream, cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(math_rows)
        if any(sha256(Path(p)) != h for p, h in protected.items()):
            raise ValueError("Protected receipt or latest changed")
        (run.path / "reports/REVIEW.md").write_text(
            "# Second author correction\n\n"
            "Complete20, all173 formula pairs, focused comparisons, per-item provenance and "
            "explicit treatment decisions are retained. Every page first replays the reviewed "
            "baseline exactly. The composition then checks actual final contours, new contacts, "
            "item identities, parser structures and pixels outside target boxes.\n\n"
            "This is exposed regression work with zero neural updates. pi remains unchanged "
            "because its only verified original is itself heavy. Font coverage remains explicit. "
            "Rejected glyph/structure instances retain their prior content and reasons.\n\n"
            "Full-page root visual review follows this immutable run, and author style approval "
            "is still needed. No production activation is implied.\n"
        )
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
