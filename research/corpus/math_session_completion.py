# ruff: noqa: E501
"""Compare the author-preferred v0.10.24 B with a calibrated Session M/N candidate."""

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import verify_run, workspace_root
from research.corpus.context_composer import layout
from research.corpus.experiment import create
from research.corpus.math_glyph_completion import (
    SELECTION,
    frozen_english,
    local_comparison,
    overlap_verification,
    raster,
)
from research.corpus.math_glyph_evidence import extract_sources, read, source_counts
from research.corpus.math_glyph_resolver import MathGlyphResolver, ScanGlyph
from research.corpus.math_glyph_semantics import check_reference_math
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.corpus.mixed_ink_pages import label_pdf

PREVIOUS = "20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5"

OPERATORS = r"""Basic operations and their results are listed below.

$$3+2=5,\quad 5-2=3,\quad 2\times3=6,\quad 6\div3=2$$

The first line orders three real numbers. Equality is included in the bounds.

$$2<3,\quad 3>2,\quad 2\leq 3\leq 4,\quad 4\geq 3\geq 2$$

The signs and limits are tested at different sizes. The inner roots stay aligned with their arguments.

$$x=\pm\sqrt{3},\quad \sqrt{\frac{7}{15}},\quad \sqrt{1+\sqrt{3}}$$

$$\sqrt{1+x+x^2+x^3+x^4},\quad \sqrt{\frac{1+x^2}{1+\sqrt{2}}},\quad x\to 0$$

Tall delimiters enclose a fraction. The accent examples keep their own marks.

$$\left[\frac{1}{2}+x\right],\quad \left|\frac{1}{2}\right|,\quad \{x\vert x>0\}$$

$$\vec F,\quad \hat y,\quad \bar x,\quad \overline{AB},\quad \ddot x,\quad \nabla f$$

The last two lines show Greek forms from the author's worksheet.

$$\mu,\quad\theta,\quad\varphi,\quad\chi,\quad\Delta,\quad\omega_d$$

$$\gamma,\quad\phi,\quad\epsilon,\quad\Omega,\quad\nu$$
"""


def load_scans(folder: Path, rows: list[dict]) -> list[ScanGlyph]:
    return [ScanGlyph.load(folder, row) for row in rows if row["review_status"] == "agent-accepted"]


def collect_panels(records, renderer, before, after, page_number, *, change_key="session_source"):
    panels = []
    changes = {r["item_index"] for r in records if r.get(change_key)}
    groups = sorted(
        [*renderer.formulas, *renderer.matrices],
        key=lambda group: group["item_range"],
    )
    for group, formula in enumerate(groups, 1):
        start, stop = formula["item_range"]
        if not changes.intersection(range(start, stop)):
            continue
        bounds = np.asarray(
            [
                box
                for row in records[start:stop]
                for box in (row["bounds_px"], row.get("original_bounds_px", row["bounds_px"]))
            ]
        )
        lo, hi = bounds[:, :2].min(axis=0), bounds[:, 2:].max(axis=0)
        # Keep stems and accent neighbours in the paired panel.
        box = (*np.floor(lo - 14).astype(int), *np.ceil(hi + 14).astype(int))
        panels.append((page_number, group, before.crop(box), after.crop(box)))
        changes.difference_update(range(start, stop))
    if changes:
        raise ValueError(f"Updated slots omitted from the paired crops on page {page_number}")
    return panels


def session_experiment(
    selection: Path, previous: Path, *, polish_baseline: Path | None = None
) -> Path:
    verify_run(previous)
    cfg_record = read(previous / "run.json")["config"]
    cfg = RenderConfig.model_validate(cfg_record["render_config"])
    root = workspace_root()
    corpus = root / "datasets/personal-homework/corpus.sqlite"
    previous_records = read(previous / "reports/glyph-provenance.json")
    previous_sources = previous / "solutions/historical-glyphs"
    specs = read(selection)
    protected = read(previous / "reports/protected-assets.json")
    protected.update({str(Path(r["pdf"])): sha256(Path(r["pdf"])) for r in specs})
    protected[str(previous / "run.json")] = sha256(previous / "run.json")
    old_c = []
    if polish_baseline is not None:
        verify_run(polish_baseline)
        old_c = read(polish_baseline / "reports/glyph-provenance.json")
        protected[str(polish_baseline / "run.json")] = sha256(polish_baseline / "run.json")
    before_arm, after_arm = "CD" if old_c else "BC"
    change_key = "polish" if old_c else "session_source"
    run = create(
        "v01026-session-math-polish" if old_c else "v01025-session-mn-math-calibration",
        seed=cfg.seed,
        config={
            "author_preferred_baseline": str(previous),
            "author_preferred_baseline_sha256": sha256(previous / "run.json"),
            "polish_baseline": str(polish_baseline) if old_c else None,
            "render_config": cfg.model_dump(mode="json"),
            "layout_profile": cfg_record["layout_profile"],
            "measured_stroke_profile": cfg_record["measured_stroke_profile"],
            "english_seed": cfg_record["english_seed"],
            "selection_sha256": sha256(selection),
            "session_documents": sorted({r["pdf"] for r in specs}),
            "calibration_method": "reviewed original contours and measured radical stem and roof",
            "personal_neural_updates": 0,
            "session_C_used": False,
        },
    )
    output = run.path
    print(output, flush=True)
    try:
        atomic_json(
            output / "reports/author-feedback.json",
            {
                "comparison": "v0.10.24 A/B",
                "preferred_arm": "B",
                "verbatim_feedback": "明显B的更好",
                "scope": "whole comparison; no symbol-level or new-candidate ratings",
                "source": "author follow-up accompanying Sessions M and N",
            },
        )
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(previous / f"solutions/profile-v{version}"),
                output / f"solutions/profile-v{version}",
            )
        profile = load_snapshot(output / "solutions/profile-v1")
        stroke_profile = load_snapshot(output / "solutions/profile-v3")
        shutil.copyfile(selection, output / "solutions/session-selection.json")
        (output / "solutions/session-pdfs").mkdir()
        for path in {Path(r["pdf"]) for r in specs}:
            shutil.copyfile(path, output / "solutions/session-pdfs" / path.name)
        raw_session = output / "solutions/session-glyphs"
        session = extract_sources(specs, raw_session, corpus)
        new_scans = load_scans(raw_session, session)
        calibration = SessionGlyphCalibration(new_scans)
        polish = SessionInkPolish(calibration, profile.font) if old_c else None
        atomic_json(
            output / "reports/radical-calibration.json",
            [sample.metrics() for sample in calibration.radicals],
        )
        legacy_scans = load_scans(previous_sources, read(previous_sources / "sources.json"))
        legacy = MathGlyphResolver(
            {glyph.source["character"]: glyph for glyph in legacy_scans},
            stroke_profile,
            selected=SELECTION,
        )
        inputs = [
            (r["identity"], (previous / f"solutions/problem-{r['page']:02d}.md").read_text(), r)
            for r in previous_records
        ] + [("Session math operators and notation", OPERATORS, None)]
        reports, panels = [], []
        for arm in (before_arm, after_arm):
            (output / f"rendered/pages/{arm}").mkdir()
        for number, (identity, source, old) in enumerate(inputs, 1):
            source_md = output / f"solutions/problem-{number:02d}.md"
            source_md.write_text(source, encoding="utf-8")
            page, renderer, anchor = layout(source, identity, profile, cfg)
            trace = {"formulas": renderer.formulas, "matrices": renderer.matrices}
            baseline = legacy.resolve(page, renderer, anchor, profile)
            before_math = [r for r in baseline if r["context"] == "mathematics"]
            frozen, rhythm = (old["word_provenance"], old["word_rhythm"]) if old else ([], [])
            b_items, words = frozen_english(page.items, frozen)
            b_pdf = renderer.to_vector_pdf([replace(page, items=b_items)])
            method = (
                "Frozen neural English | Measured word spacing"
                if number <= 4
                else "Handnote English | Extended mathematics exercise"
            )
            b_marked_pdf = label_pdf(b_pdf, method, math_label="personal glyph candidate")
            replayed = raster(b_marked_pdf)
            if old:
                if (
                    old["source_sha256"] != sha256(source_md)
                    or json.loads(json.dumps(before_math)) != old["B"]
                    or json.loads(json.dumps(trace)) != old["parser_structure_trace"]
                ):
                    raise ValueError(f"Archived B trace changed on page {number}")
                archived = previous / f"rendered/pages/B/page-{number:02d}.pdf"
                if raster(archived).tobytes() != replayed.tobytes():
                    raise ValueError(f"Archived B does not replay on page {number}")
                b_marked_pdf = archived.read_bytes()
            revised = calibration.apply(page, renderer, baseline, operator_page=old is None)
            if polish is not None:
                c_items, _ = frozen_english(page.items, frozen)
                b_pdf = renderer.to_vector_pdf([replace(page, items=c_items)])
                before_math = [r for r in revised if r["context"] == "mathematics"]
                reference = old_c[number - 1]
                if (
                    reference["source_sha256"] != sha256(source_md)
                    or json.loads(json.dumps(before_math)) != reference["C"]
                    or json.loads(json.dumps(trace)) != reference["parser_structure_trace"]
                ):
                    raise ValueError(f"Archived C trace changed on page {number}")
                archived_c = polish_baseline / f"rendered/pages/C/page-{number:02d}.pdf"
                b_marked_pdf = archived_c.read_bytes()
                if (
                    raster(
                        label_pdf(b_pdf, method, math_label="Session M/N glyph calibration")
                    ).tobytes()
                    != raster(b_marked_pdf).tobytes()
                ):
                    raise ValueError(f"Archived C pixels changed on page {number}")
                baseline = revised
                original_items = list(page.items)
                revised = polish.apply(page, renderer, baseline)
                if any(
                    a is not b and "polish" not in row
                    for a, b, row in zip(original_items, page.items, revised, strict=True)
                ):
                    raise ValueError("Math polish changed an unassigned item")
            c_items, new_words = frozen_english(page.items, frozen)
            if words != new_words:
                raise ValueError("Frozen English shifted in the session candidate")
            candidate = replace(page, items=c_items)
            bounds, _ = ink_geometry(candidate, renderer)
            if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
                raise ValueError("Session ink crosses the page boundary")
            c_pdf = renderer.to_vector_pdf([candidate])
            if any(
                (a["item_index"], a["semantic"], a["latex"], a["context"])
                != (b["item_index"], b["semantic"], b["latex"], b["context"])
                for a, b in zip(baseline, revised, strict=True)
            ):
                raise ValueError("Session calibration changed math identity or order")
            new_math = [r for r in revised if r["context"] == "mathematics"]
            changes = [r for r in new_math if change_key in r]
            neural_items = {i for w in frozen if not w["fallback_reason"] for i in w["items"]}
            field = [w["box"] for w in words] + [
                r["bounds_px"]
                for r in anchor
                if r["context"] == "text" and r["item_index"] not in neural_items
            ]
            before, after = raster(b_pdf), raster(c_pdf)
            pixel_proof = overlap_verification(before, after, changes, field)
            radicands = {
                row["polish"]["radicand_item"]: row["polish"]["digit_shift_px"]
                for row in changes
                if "polish" in row and "radicand_item" in row["polish"]
            }
            antialias_delta = 0
            for index, dx in radicands.items():
                x0, y0, x1, y1 = baseline[index]["bounds_px"]
                crop = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
                moved = (crop[0] + dx, crop[1], crop[2] + dx, crop[3])
                original = np.asarray(before.crop(crop), dtype=np.int16)
                shifted = np.asarray(after.crop(moved), dtype=np.int16)
                delta = int(np.abs(original - shifted).max())
                # PDFium can round edge coverage differently after an integer
                # translation. Any mismatch beyond that edge rounding fails.
                if delta > 4:
                    raise ValueError("Compact root changed the digit under the roof")
                antialias_delta = max(antialias_delta, delta)
            if polish is not None:
                pixel_proof["preserved_radicand_shapes"] = len(radicands)
                pixel_proof["radicand_max_antialias_delta"] = antialias_delta
            panels.extend(
                collect_panels(revised, renderer, before, after, number, change_key=change_key)
            )
            c_marked_pdf = label_pdf(
                c_pdf,
                method,
                math_label="Session math spacing and ink polish"
                if old_c
                else "Session M/N glyph calibration",
            )
            for arm, pdf in ((before_arm, b_marked_pdf), (after_arm, c_marked_pdf)):
                dest = output / f"rendered/pages/{arm}/page-{number:02d}.pdf"
                dest.write_bytes(pdf)
                raster(pdf).save(dest.with_suffix(".png"), dpi=(300, 300))
            reports.append(
                {
                    "page": number,
                    "identity": identity,
                    "B_matches_v01024_pixels": old is not None,
                    before_arm: before_math,
                    after_arm: new_math,
                    "word_provenance": frozen,
                    "word_rhythm": rhythm,
                    "neural_words": len(words),
                    "source_sha256": sha256(source_md),
                    "parser_structure_trace": trace,
                    "parser_item_order_and_content_equal": True,
                    "candidate_ink_bounds_px": bounds,
                    **pixel_proof,
                }
            )
        for arm in (before_arm, after_arm):
            with PdfWriter() as writer:
                for pdf in sorted((output / f"rendered/pages/{arm}").glob("*.pdf")):
                    writer.append(pdf)
                writer.write(output / f"rendered/{arm}-homework.pdf")
        with PdfWriter() as pair:
            for number in range(1, len(reports) + 1):
                for arm in (before_arm, after_arm):
                    pair.append(output / f"rendered/pages/{arm}/page-{number:02d}.pdf")
            pair.write(output / f"comparisons/{before_arm}-{after_arm}-homework.pdf")
        local_comparison(
            output / "comparisons",
            panels,
            labels=("C   v0.10.25 before polish", "D   Spacing and ink corrected")
            if old_c
            else ("B   v0.10.24 author preference", "C   Session M/N calibration"),
        )
        atomic_json(output / "reports/glyph-provenance.json", reports)
        atomic_json(
            output / "reports/semantic-verification.json",
            {
                "original_calculations": check_reference_math(),
                "added_operators_and_inequalities": all(
                    [
                        3 + 2 == 5,
                        5 - 2 == 3,
                        2 * 3 == 6,
                        6 / 3 == 2,
                        2 < 3,
                        3 > 2,
                        2 <= 3 <= 4,
                        4 >= 3 >= 2,
                    ]
                ),
                "layout_syntax_and_item_order_preserved": True,
                "independent_candidate_author_review": None,
            },
        )
        if any(sha256(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("Previously archived inputs, Profiles or latest changed")
        atomic_json(output / "reports/protected-assets.json", protected)
        first_four = [r for p in reports[:4] for r in p[after_arm]]
        used = [r for p in reports for r in p[after_arm] if r.get("session_source")]
        result = {
            "pages_per_arm": len(reports),
            "exact_v01024_B_pages": len(previous_records),
            "neural_words_identical": sum(p["neural_words"] for p in reports),
            "retained_word_gap_observations": sum(len(p["word_rhythm"]) for p in reports),
            "baseline_four_page_source_types": source_counts(
                [r for page in reports[:4] for r in page[before_arm]]
            ),
            "candidate_four_page_source_types": source_counts(first_four),
            "candidate_four_page_fallbacks": dict(
                Counter(r["semantic"] for r in first_four if r.get("fallback_reason"))
            ),
            "reviewed_session_crops": len(new_scans),
            "held_session_crops": len(session) - len(new_scans),
            "reviewed_session_codepoints": len({q.source["character"] for q in new_scans}),
            "radical_sizes_measured": len(calibration.radicals),
            "session_used_instances_by_symbol": dict(Counter(r["semantic"] for r in used)),
            "used_session_samples": sorted({r["session_source"]["id"] for r in used}),
            "formula_region_pairs": len(panels),
            "protected_assets_exact": len(protected),
            "torch_loaded_in_render_process": "torch" in sys.modules,
            "new_neural_updates": 0,
            "release_decision": "held for new candidate author review and wider validation",
        }
        if polish is not None:
            changes = [row for p in reports for row in p[after_arm] if "polish" in row]
            result.update(
                exact_v01025_C_pages=len(old_c),
                polished_items_by_symbol=dict(Counter(r["semantic"] for r in changes)),
                preserved_radicand_shapes=sum(p["preserved_radicand_shapes"] for p in reports),
                radicand_max_antialias_delta=max(
                    p["radicand_max_antialias_delta"] for p in reports
                ),
                final_fallbacks=dict(
                    Counter(
                        r["semantic"]
                        for p in reports
                        for r in p[after_arm]
                        if r.get("fallback_reason")
                    )
                ),
            )
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return output


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--selection", required=True, type=Path)
    cli.add_argument("--previous", type=Path, default=workspace_root() / "runs" / PREVIOUS)
    cli.add_argument("--polish-baseline", type=Path)
    args = cli.parse_args()
    session_experiment(args.selection, args.previous, polish_baseline=args.polish_baseline)
