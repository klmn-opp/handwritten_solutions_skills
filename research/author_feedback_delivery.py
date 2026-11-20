"""Assemble approved, exposed author-feedback candidates into a fresh private run.

This copies 17 immutable page PDF/PNG pairs and regenerates only pages 2/6/18,
replaying their approved MathLatin pixels before the approved symbol increments.
It consumes no new holdout and activates no production Profile or model.
"""

import copy
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_revision import (
    MATH_ONLY,
    group_crops,
    identity,
    personal_layout,
    trace,
)
from research.corpus.math_glyph_completion import SELECTION, local_comparison, raster
from research.corpus.math_glyph_evidence import fallback_overview
from research.corpus.math_glyph_resolver import MathGlyphResolver
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.historical_math_candidate import HistoricalMathFeedback, load_sources
from research.homework_benchmark.render import (
    accepted_scans,
    fallback_summary,
    pixel_boundary_checks,
    read,
    save_pdf,
    source_paths,
)
from research.math_feedback_candidate import AGGREGATE, paired_panel
from research.math_latin_feedback import MathLatinFeedback
from research.math_paired_bars import BASE, PersonalPairedBars, source_bar

LATIN = "20261120T114049432720Z-math-latin-dfkr-feedback-19af9217"
HISTORICAL = "20261120T113900120096Z-historical-membership-double-integral-candidate-9e6e74a8"
BARS = "20261120T114158318267Z-paired-personal-bars-candidate-1369e4a0"
LATIN_SHA = "2b5320a5da3af23a14200e422985e013e0ffe2cf0c1067612d83c537fc9f5e73"
REPLACED = {2, 6, 18}
ROOT = Path(__file__).resolve().parents[1]


def copy_file(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def json_value(value):
    """Use archival JSON types when comparing runtime tuples to saved lists."""
    return json.loads(json.dumps(value))


def original_trace(combined, number):
    """Recover the archived original parser groups, before fraction reflow."""
    entries = read(combined / f"comparisons/formulas/original/{number:02d}/index.json")
    result = {"formulas": [], "matrices": []}
    for item in entries:
        kind = item["panel"].split("-", 1)[1].split(".")[0]
        result[kind].append(item["original_parser"])
    return result


def crop_pair(old, new, a, b, padding=(45, 20)):
    boxes = np.asarray([old["bounds_px"], new["bounds_px"]])
    x0, y0 = np.floor(boxes[:, :2].min(0) - padding).astype(int)
    x1, y1 = np.ceil(boxes[:, 2:].max(0) + padding).astype(int)
    box = max(0, x0), max(0, y0), min(a.width, x1), min(a.height, y1)
    return a.crop(box), b.crop(box)


def transformation(row):
    """Additional disclosure; never collapse transformed sources into untouched originals."""
    value = []
    if row.get("cpu_feedback_layout"):
        value.append("text-r-bearing-and-advance-calibration")
    if row.get("cpu_feedback"):
        ch = row["semantic"]
        value.append(
            {
                "k": "monotone-output-y-geometry-calibration",
                "d": "real-d1-variant-fitted-to-existing-slot",
                "f": "real-f0-native-variant-selection",
                "r": "real-r0-native-MathText-primary-slot-selection",
                "0": "real-lighter-zero-variant-selection",
            }.get(ch, "recorded-cpu-output-calibration")
        )
    if row.get("historical_math_feedback"):
        value.append(
            "two-original-integral-components-mapped-to-iint"
            if row["semantic"] == "∬"
            else "fixed-reviewed-left-membership-source-fitted-to-slot"
        )
    if row.get("paired_bar_feedback"):
        value.append("two-single-bar-instances-procedurally-composed-with-width-cap")
    if row.get("feedback_history"):
        value.append("radical-stem-roof-and-radicand-clearance-layout")
    if row.get("scan_source") or row.get("session_source"):
        value.append("source-contour-to-parser-slot-transform-recorded")
    if row.get("procedural_geometry"):
        value.append("procedural-geometry-disclosed")
    return value


def snapshot_sources(run, sources):
    hashes = {}
    paths = set(source_paths()) | set(ROOT.glob("research/**/*.py")) | {Path(__file__)}
    for source in sorted(paths):
        relative = source.resolve().relative_to(ROOT)
        hashes[str(relative)] = sha256(source)
        copy_file(source, run.path / "logs/source" / relative)
    for name, parent in sources.items():
        copy_file(parent / "run.json", run.path / f"reports/parents/{name}-run.json")
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True
    )
    atomic_json(
        run.path / "reports/source-identity.json",
        {
            "parent_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_status_porcelain": status,
            "research_and_runtime_source_sha256": hashes,
            "source_location": "logs/source/<repository-relative-path>",
            "scope_note": (
                "Run.build tracks the core runtime; its dirty flag is not an all-repository "
                "cleanliness claim. This explicit research inventory and Git status identify "
                "the experiment, including uncommitted sources. "
                "Earlier run receipts remain unchanged."
            ),
        },
    )


def approval_receipt(sources):
    return {
        "reviewer": "root Codex Agent",
        "review_scope": "isolated research candidate; no production or author approval",
        "author_acceptance": False,
        "production_activation": False,
        "new_holdout": False,
        "approved": {
            name: {
                "run": str(sources[name]),
                "run_sha256": sha256(sources[name] / "run.json"),
                "review": review,
            }
            for name, review in {
                "math_latin": "72 targets and native f_avg/rad/s reviewed; 37 changes accepted",
                "historical": "Fixed left membership and two original integral components reviewed",
                "paired_bars": "All six native norm delimiters and complete page 2 reviewed",
            }.items()
        },
        "assembly_authorization": (
            "Root explicitly approved final twenty-page assembly after all three reviews; "
            "replace only pages 2/6/18 over the sealed MathLatin twenty-page baseline."
        ),
    }


def main():
    root = workspace_root()
    sources = {
        name: root / "runs" / run_id
        for name, run_id in {
            "original_B": AGGREGATE,
            "combined": BASE,
            "math_only": MATH_ONLY,
            "math_latin": LATIN,
            "historical": HISTORICAL,
            "paired_bars": BARS,
        }.items()
    }
    for source in sources.values():
        verify_run(source)
    latin, combined, original = (sources[k] for k in ("math_latin", "combined", "original_B"))
    if sha256(latin / "run.json") != LATIN_SHA:
        raise ValueError("MathLatin reviewed receipt changed")
    critical = {
        "math_latin": [
            "author_feedback_revision",
            "cpu_glyph_feedback",
            "math_feedback_candidate",
            "math_latin_feedback",
        ],
        "historical": ["historical_math_candidate"],
        "paired_bars": ["math_paired_bars"],
    }
    for name, modules in critical.items():
        for module in modules:
            rel = Path(f"research/{module}.py")
            if sha256(ROOT / rel) != sha256(sources[name] / "logs/source" / rel):
                raise ValueError(f"Unreviewed source mutation: {rel}")
    cfg = RenderConfig.model_validate(read(latin / "run.json")["config"]["render"])
    run = Run.create(
        "author-feedback-final-20-pages",
        config={
            "argv": sys.argv,
            "render": cfg.model_dump(mode="json"),
            "parent_runs": {k: str(p) for k, p in sources.items()},
            "regenerated_pages": sorted(REPLACED),
            "byte_copied_pages": 17,
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "author_style_acceptance": False,
            "production_activation": False,
        },
        seed=cfg.seed,
    )
    print(run.path, flush=True)
    try:
        protected = {str(p / "run.json"): sha256(p / "run.json") for p in sources.values()}
        protected[str(root / "latest.json")] = sha256(root / "latest.json")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        atomic_json(run.path / "reports/approval-receipt.json", approval_receipt(sources))
        snapshot_sources(run, sources)
        for name in ("profile-v1", "profile-v3", "inputs", "historical-glyphs", "session-glyphs"):
            shutil.copytree(latin / "solutions" / name, run.path / "solutions" / name)
        shutil.copytree(
            sources["historical"] / "solutions/new-glyphs",
            run.path / "solutions/new-historical-glyphs",
        )
        for name in ("combined", "math_latin", "historical", "paired_bars"):
            shutil.copytree(sources[name] / "reports", run.path / f"reports/inherited/{name}")
        for source, name in (
            (original, "original-B20"),
            (combined, "previous-combined20"),
            (latin, "approved-MathLatin20"),
        ):
            src = source / "rendered/homework_solutions.pdf"
            if not src.exists():
                src = source / "rendered/B-homework.pdf"
            copy_file(src, run.path / f"comparisons/{name}.pdf")
        profile, strokes = (load_snapshot(run.path / f"solutions/profile-v{v}") for v in (1, 3))
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        hist = HistoricalMathFeedback(
            load_sources(run.path / "solutions/new-historical-glyphs"), strokes
        )
        bars = PersonalPairedBars(source_bar(run.path / "solutions/session-glyphs"))
        cases, all_rows, images, math_pages = [], [], [], []
        focused_math, focused_letters = [], []
        selected_text = Counter()
        review_selection = {12: {6, 7}, 14: {9}, 18: {2, 3, 7, 8}, 17: {1}}
        all_formula_count = 0
        copies = []
        for entry in read(latin / "reports/cases.json"):
            number, case_id = entry["original_page"], entry["case_id"]
            print(f"page {number:02d}: {case_id}", flush=True)
            archived = read(latin / f"reports/{case_id}-provenance.json")
            records = copy.deepcopy(archived["candidate"])
            parser = archived["parser_trace"]
            old = read(combined / f"reports/{case_id}-provenance.json")
            a = Image.open(latin / f"rendered/pages/{number:02d}.png").convert("RGB")
            base_image = Image.open(combined / f"rendered/pages/{number:02d}.png").convert("RGB")
            orig_image = Image.open(original / f"rendered/pages/{number:02d}.png").convert("RGB")
            dest = run.path / f"rendered/pages/{number:02d}.pdf"
            extra_decisions, changed = [], set()
            if number in REPLACED:
                inputs = run.path / f"solutions/inputs/{case_id}"
                job = Job.model_validate_json((inputs / "job.json").read_text())
                bg = job.background
                asset = inputs / bg.path if bg.path else None
                effective = cfg.model_copy(update={"paper": "plain" if asset else bg.kind})
                page, renderer, replay, _, _ = personal_layout(
                    (inputs / "solution.md").read_text(),
                    case_id,
                    profile,
                    effective,
                    resolver,
                    sessions,
                    polish,
                    bg,
                    asset,
                    cpu=True,
                    clearance=True,
                )
                replay = json_value(MathLatinFeedback(profile.font).apply(page, renderer, replay))
                if replay != records or json_value(trace(renderer)) != parser:
                    raise ValueError("MathLatin replay provenance or parser trace differs")
                if raster(renderer.to_vector_pdf([page])).tobytes() != a.tobytes():
                    raise ValueError("Approved MathLatin pixels did not replay exactly")
                if number == 2:
                    records = bars.apply(page, renderer, replay)
                    extra_decisions = bars.decisions
                else:
                    records = hist.apply(page, renderer, replay, profile)
                    extra_decisions = [
                        {"item_index": r["item_index"], **r["historical_math_feedback"]}
                        for r in records
                        if r.get("historical_math_feedback")
                    ]
                records = json_value(records)
                changed = {r["item_index"] for r, s in zip(records, replay, strict=True) if r != s}
                expected = 6 if number == 2 else 1
                if len(changed) != expected:
                    raise ValueError("Unexpected incremental symbol change count")
                if json_value(trace(renderer)) != parser or identity(replay) != identity(records):
                    raise ValueError(
                        "Symbol composition changed parser structure or identity/order"
                    )
                b = save_pdf(renderer.to_vector_pdf([page]), dest)
                bounds, _ = ink_geometry(page, renderer)
                checks = pixel_boundary_checks(a, b, replay, records, changed)
            else:
                for suffix in ("pdf", "png"):
                    src = latin / f"rendered/pages/{number:02d}.{suffix}"
                    target = dest.with_suffix(f".{suffix}")
                    copy_file(src, target)
                    copies.append(
                        {
                            "page": number,
                            "kind": suffix,
                            "parent_sha256": sha256(src),
                            "final_sha256": sha256(target),
                        }
                    )
                    if sha256(src) != sha256(target):
                        raise ValueError("Unmodified page byte identity failed")
                b = a.copy()
                bounds = entry["ink_bounds_px"]
                checks = pixel_boundary_checks(a, b, records, records, set())
            if [r for r in records if r["context"] == "text"] != [
                r for r in old["combined"] if r["context"] == "text"
            ]:
                raise ValueError("Final text provenance differs from approved combined")
            if [r["semantic"] for r in records if r["context"] == "text"] != [
                r["semantic"] for r in old["original_B"] if r["context"] == "text"
            ]:
                raise ValueError("Final English character sequence differs from original B20")
            # Combined -> final is math-only; exact record/pixel protection applies to text.
            changed_since_combined = {
                r["item_index"] for r, s in zip(records, old["combined"], strict=True) if r != s
            }
            cumulative = pixel_boundary_checks(
                base_image, b, old["combined"], records, changed_since_combined
            )
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {
                    "candidate": records,
                    "math_latin_before": archived["candidate"],
                    "parser_trace": parser,
                    "incremental_decisions": extra_decisions,
                    "source_revision_parent": f"reports/inherited/combined/{case_id}-checks.json",
                },
            )
            report = {
                "page": number,
                "case_id": case_id,
                "regenerated": number in REPLACED,
                "incremental_checks": checks,
                "combined_to_final_checks": cumulative,
                "original_english_semantic_sequence_preserved": True,
                "ink_bounds_px": bounds,
                "fallbacks": fallback_summary(records),
                "provenance": f"reports/{case_id}-provenance.json",
            }
            cases.append(report)
            for folder, before_image, label in (
                ("original-to-final", orig_image, "Original user B"),
                ("combined-to-final", base_image, "Previous combined"),
            ):
                paired_panel(
                    before_image,
                    b,
                    run.path / f"comparisons/{folder}/{number:02d}.png",
                    f"Original page {number}",
                    (label, "Final isolated candidate"),
                )
            old_parser = original_trace(combined, number)
            panels = group_crops(
                old["original_B"],
                old_parser,
                orig_image,
                records,
                parser,
                b,
                run.path / f"comparisons/all-formulas/{number:02d}",
                number,
                ("Original user B", "Final isolated candidate"),
            )
            all_formula_count += len(panels)
            focused_math.extend(p for p in panels if p[1] in review_selection.get(number, set()))
            for before_row, row in zip(old["combined"], records, strict=True):
                if row.get("historical_math_feedback") or row.get("paired_bar_feedback"):
                    if row["semantic"] == "‖" and any(p[0] == 2 for p in focused_math):
                        continue
                    if row["semantic"] == "‖":
                        # The entire six-norm expression, not an isolated bar.
                        match = next(
                            p
                            for p, idx in zip(
                                panels,
                                read(
                                    run.path / f"comparisons/all-formulas/{number:02d}/index.json"
                                ),
                                strict=True,
                            )
                            if idx["candidate_parser"]["item_range"][0]
                            <= row["item_index"]
                            < idx["candidate_parser"]["item_range"][1]
                        )
                        focused_math.append(match)
                    else:
                        focused_math.append(
                            (
                                number,
                                row["item_index"],
                                *crop_pair(before_row, row, base_image, b, (85, 45)),
                            )
                        )
                if (
                    row.get("math_latin_feedback")
                    and row.get("cpu_feedback")
                    and (
                        (number == 6 and row["semantic"] in {"d", "f"})
                        or (number == 12 and row["semantic"] == "r")
                        or (number == 20 and row["semantic"] == "k")
                    )
                ):
                    token = "math-" + row["semantic"]
                    if selected_text[token] < 2:
                        focused_letters.append(
                            (
                                number,
                                row["item_index"],
                                *crop_pair(before_row, row, base_image, b, (65, 30)),
                            )
                        )
                        selected_text[token] += 1
            # First two real occurrence contexts per target; no cherry-picked seeds.
            text_index = read(combined / f"comparisons/text/{number:02d}/index.json")
            # Archived spans use complete-page item_index, not text-only ordinal.
            old_text = old["original_B"]
            new_text = records
            for item in text_index:
                token = "text-" + item["target"]
                if selected_text[token] >= 2:
                    continue
                crops = []
                for rows, indices, image in (
                    (old_text, item["items"][0], orig_image),
                    (new_text, item["items"][1], b),
                ):
                    boxes = np.asarray([rows[i]["bounds_px"] for i in indices])
                    box = (
                        *np.floor(boxes[:, :2].min(0) - 14).astype(int),
                        *np.ceil(boxes[:, 2:].max(0) + 14).astype(int),
                    )
                    crops.append(image.crop(box))
                focused_letters.append((number, f"text {item['target']}", *crops))
                selected_text[token] += 1
            for row in records:
                all_rows.append(
                    {
                        "page": number,
                        "case_id": case_id,
                        **row,
                        "transformation_disclosure": transformation(row),
                    }
                )
            images.append(b)
            math_pages.append([r for r in records if r["context"] == "mathematics"])
        with PdfWriter() as writer:
            for case in cases:
                writer.append(run.path / f"rendered/pages/{case['page']:02d}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        for name, panels in (
            ("roots-fractions-symbols", focused_math),
            ("english-and-math-letters", focused_letters),
        ):
            folder = run.path / "comparisons" / name
            folder.mkdir()
            local_comparison(
                folder, panels, labels=("Earlier reviewed baseline", "Final candidate")
            )
            atomic_json(folder / "index.json", [{"page": p[0], "region": p[1]} for p in panels])
        overlay = run.path / "comparisons/fallback-overlay"
        overlay.mkdir()
        fallback_overview(images, math_pages, overlay)
        math_rows = [r for r in all_rows if r["context"] == "mathematics"]
        fallback_rows = [
            r for r in math_rows if r["source_type"] in {"external-font", "inherited-font"}
        ]
        summary = {
            "pages": 20,
            "regenerated_pages": sorted(REPLACED),
            "copied_pages": 17,
            "math_visible_items": len(math_rows),
            "math_source_types": dict(Counter(r["source_type"] for r in math_rows)),
            "remaining_font_fallbacks": len(fallback_rows),
            "fallback_counts_by_symbol": dict(Counter(r["semantic"] for r in fallback_rows)),
            "math_transformations": dict(
                Counter(t for r in math_rows for t in r["transformation_disclosure"])
            ),
            "changes_inherited": {
                "stacked_fraction_revisions": 37,
                "compound_root_clearance_adjustments": 7,
                "math_latin": {"d": 18, "f": 2, "k": 7, "r": 10},
                "historical_membership": 1,
                "double_integral_components": 1,
                "norm_delimiters_composed_from_single_bars": 6,
            },
            "all_formula_crops": all_formula_count,
            "final_symbol_increment_english_pixels_changed": 0,
            "combined_to_final_english_pixels_changed": 0,
            "original_B_to_final_scope": (
                "English character sequence and non-target source/contours preserved; "
                "37 fraction changes and r metric corrections permit recorded reflow. "
                "No claim of original-B pixel invariance. Full checks inherited from combined."
            ),
            "math_semantics_scope": (
                "37 explicit source revisions independently checked; later steps preserve "
                "item identity/order, parser trace and non-target outlines. These engineering "
                "checks support but do not replace analytical/visual review."
            ),
            "go_no_go": {
                "isolated_author_review_candidate": "GO",
                "production_release_or_default_profile": "NO-GO; author approval pending",
                "new_independent_holdout_claim": "NO-GO; all content previously exposed",
            },
            "new_neural_updates": 0,
            "author_acceptance": False,
        }
        atomic_json(run.path / "reports/cases.json", cases)
        atomic_json(run.path / "reports/page-byte-identity.json", copies)
        atomic_json(run.path / "reports/glyph-provenance.json", all_rows)
        atomic_json(run.path / "reports/fallback-reasons.json", fallback_rows)
        atomic_json(run.path / "reports/glyph-summary.json", summary)
        with (run.path / "reports/math-glyph-coverage.csv").open("w", newline="") as stream:
            cols = [
                "page",
                "case_id",
                "item_index",
                "semantic",
                "source_type",
                "font_sha256",
                "variant",
                "fallback_reason",
                "transformation_disclosure",
            ]
            writer = csv.DictWriter(stream, cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(math_rows)
        text = (
            "# Final isolated author-feedback candidate\n\n"
            "Twenty original successful pages, in their original order and backgrounds. "
            "This is an exposed-content recovery candidate, not new independent evaluation.\n\n"
            "Open `rendered/homework_solutions.pdf` for the complete work. "
            "Focused review: `comparisons/roots-fractions-symbols/formula-regions.pdf` and "
            "`comparisons/english-and-math-letters/formula-regions.pdf`. "
            "All formula crops and full-page original-to-final comparisons are also retained.\n\n"
            "37 slash fractions are now stacked; 7 compound roots have corrected clearance. "
            "Text 0/r/d/k/f and 37 math d/f/k/r corrections retain real source lineage. "
            "One membership sign uses the fixed reviewed left historical sample; a double integral "
            "uses two original integral components. Six norm delimiters use two instances of a "
            "reviewed Session N single vertical stroke. These compositions are not new collected "
            "single-character truth. Original labels and RGB remain unchanged.\n\n"
            f"Remaining font fallbacks: {len(fallback_rows)}. See `reports/fallback-reasons.json` "
            "and the 20-page annotated overlay. Full conventional university math/physics "
            "collection materials are supplied separately per the author's latest request.\n\n"
            "Pages 2/6/18 were regenerated over a byte-exact replay of approved MathLatin. "
            "The other 17 page PDFs and PNGs are byte-identical copies. English pixels are "
            "unchanged since the approved combined revision; relative to the original B20, "
            "recorded layout reflow and target glyph changes exist and are disclosed.\n\n"
            "GO for author review only; NO-GO for production activation without author approval. "
            "No network training, new holdout consumption, latest update or Profile replacement. "
            "Research source hashes and full Git status are recorded independently of the "
            "core-runtime build fingerprint.\n"
        )
        (run.path / "reports/FINAL_REVIEW.md").write_text(text)
        if any(sha256(Path(p)) != h for p, h in protected.items()):
            raise ValueError("Protected parent receipt or latest changed")
        run.finish("held", result=summary)
        verify_run(run.path)
        print(summary, flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
