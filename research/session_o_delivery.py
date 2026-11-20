"""Complete exposed twenty-page Session O comparison from a reviewed source bank.

No missing bank is simulated. Every source file and previous candidate is pinned,
and only pi or an explicit font fallback in PRIORITY may be replaced. Publication,
default Profile changes and neural training are outside this isolated command.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import crop_pair, json_value, snapshot_sources
from research.author_feedback_revision import group_crops, trace
from research.author_feedback_round2_checks import page_geometry, same_path
from research.corpus.math_glyph_completion import local_comparison
from research.corpus.math_glyph_evidence import fallback_overview
from research.corpus.math_glyph_resolver import ScanGlyph
from research.homework_benchmark.render import fallback_summary, read, save_pdf
from research.math_feedback_candidate import paired_panel
from research.session_o_baseline import load_baseline, replay_page, validate_increment
from research.session_o_resolver import PRIORITY, SessionOResolver

FALLBACK_TYPES = frozenset(("external-font", "inherited-font"))


def snapshot_bank_review(bank, run, protected):
    """Copy explicitly pinned review/body evidence; do not traverse new corpus data."""
    bank_run = bank.parent.parent
    if not (bank_run / "run.json").is_file():
        atomic_json(run.path / "reports/bank-review-evidence.json", {"bank_parent_run": None})
        return
    verify_run(bank_run)
    receipt = read(bank_run / "run.json")
    protected[str(bank_run / "run.json")] = sha256(bank_run / "run.json")
    shutil.copyfile(bank_run / "run.json", run.path / "reports/bank-run.json")
    shutil.copytree(bank_run / "reports", run.path / "reports/bank-admission")
    approval = receipt["config"].get("approval", {})
    evidence = []
    for index, (name, digest) in enumerate(approval.get("review_evidence_sha256", {}).items()):
        original = Path(name)
        if sha256(original) != digest:
            raise ValueError("Pinned bank review evidence changed")
        destination = run.path / f"reports/bank-review-evidence/{index:02d}-{original.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        protected[str(original)] = digest
        record = {
            "original": str(original),
            "sha256": digest,
            "archived": str(destination.relative_to(run.path)),
        }
        evidence_run = original.parent.parent
        if (evidence_run / "run.json").is_file():
            verify_run(evidence_run)
            folder = run.path / f"solutions/bank-evidence-{index:02d}"
            folder.mkdir()
            # A nested receipt is evidence, not this run's active manifest.
            # Run.finish excludes the reserved basename run.json everywhere.
            shutil.copyfile(evidence_run / "run.json", folder / "source-run-receipt.json")
            for part in ("solutions", "comparisons", "reports"):
                if (evidence_run / part).is_dir():
                    shutil.copytree(evidence_run / part, folder / part)
            protected[str(evidence_run / "run.json")] = sha256(evidence_run / "run.json")
            record["evidence_run_copy"] = str(folder.relative_to(run.path))
        evidence.append(record)
    atomic_json(run.path / "reports/bank-review-evidence.json", evidence)


def bank_inventory(directory):
    directory = directory.resolve()
    if not directory.is_dir() or not (directory / "sources.json").is_file():
        raise ValueError("A real reviewed bank containing sources.json is required")
    rows = read(directory / "sources.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The reviewed source bank is empty or invalid")
    accepted = [row for row in rows if row.get("review_status") == "agent-accepted"]
    if not any(row.get("character") in PRIORITY for row in accepted):
        raise ValueError("No reviewed priority source is available")
    if len({row["id"] for row in accepted}) != len(accepted):
        raise ValueError("Reviewed source IDs must be unique")
    # Real loader verifies contours/raw files before allocating a rendering run.
    SessionOResolver(directory, seed=1909)
    return rows, {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def target_decisions(before, after, decisions):
    """Cover absent/refused sources as well as successful resolver attempts."""
    attempted = {}
    for decision in decisions:
        i = decision["item_index"]
        if i in attempted:
            raise ValueError("Resolver produced duplicate decisions for one visible item")
        attempted[i] = decision
    result = []
    for old, new in zip(before, after, strict=True):
        target = old["context"] == "mathematics" and old["semantic"] in PRIORITY
        eligible = target and (old["semantic"] == "π" or old["source_type"] in FALLBACK_TYPES)
        if not eligible:
            if old != new:
                raise ValueError("A glyph outside the authorized replacement eligibility changed")
            if not target:
                continue
        i = old["item_index"]
        decision = copy.deepcopy(
            attempted.pop(
                i,
                {
                    "item_index": i,
                    "semantic": old["semantic"],
                    "accepted": False,
                    "source_id": None,
                    "reason": "No reviewed source available"
                    if eligible
                    else "Existing personal source retained",
                },
            )
        )
        decision.update(
            old_source_type=old["source_type"],
            final_source_type=new["source_type"],
            old_bounds_px=old["bounds_px"],
            final_bounds_px=new["bounds_px"],
            authorized_for_replacement=eligible,
        )
        if decision["accepted"]:
            if new["source_type"] not in {"personal-session-outline", "personal-stretched-outline"}:
                raise ValueError("Accepted Session O item has an incorrect source classification")
            if new.get("external_fallback") or not new.get("session_o_source"):
                raise ValueError("Accepted Session O item has incomplete source provenance")
        elif old != new:
            raise ValueError("A refused source changed the old glyph or source record")
        result.append(decision)
    if attempted:
        raise ValueError("Resolver attempted an item outside the explicit target set")
    return result


def verify_source_placements(bank, before, after, before_paths, page, renderer):
    """Independently recreate accepted placed paths from immutable RGB ink."""
    catalog = {row["id"]: row for row in read(bank / "sources.json")}
    placed = page_geometry(page, renderer)
    evidence, english_count = [], 0
    for i, row in enumerate(after):
        if before[i]["context"] == "text":
            if before[i] != row or not same_path(before_paths[i], placed[i]):
                raise ValueError(f"English record or outline changed at {i}")
            english_count += 1
        source = row.get("session_o_source")
        if source is None:
            continue
        if source.get("id") not in catalog or source != catalog[source["id"]]:
            raise ValueError(f"Actual source record differs from the archived bank at {i}")
        if row.get("font") is not None or row.get("font_sha256") is not None:
            raise ValueError("Personal scan outline must not claim a source font file")
        crop = bank / source["crop"]
        if sha256(crop) != source["crop_sha256"]:
            raise ValueError("The actual glyph RGB crop changed")
        sample = ScanGlyph.load(bank, source)
        transform = row["transformation"]
        scales = np.asarray(transform["scale_xy"], dtype=float)
        translation = np.asarray(transform["translation_xy"], dtype=float)
        if not np.isfinite(scales).all() or not np.isfinite(translation).all() or min(scales) <= 0:
            raise ValueError("Invalid source geometry transform")
        anisotropic = bool(abs(scales[0] - scales[1]) > 1e-10)
        if anisotropic and (row["semantic"] not in "⟨⟩" or max(scales) / min(scales) > 4):
            raise ValueError("Unapproved nonuniform source transformation")
        if transform.get("stroke_redrawing") or transform.get("ink_erosion_or_dilation"):
            raise ValueError("Unapproved modification of original ink")
        expected = sample.outline.verts * scales + translation
        actual = placed[i]
        if not np.array_equal(expected, actual.vertices) or not np.array_equal(
            sample.outline.codes, actual.codes
        ):
            raise ValueError(f"Actual placed outline does not reproduce from its source at {i}")
        evidence.append(
            {
                "item_index": i,
                "semantic": row["semantic"],
                "source_id": source["id"],
                "crop": str(crop),
                "crop_sha256": sha256(crop),
                "raw_crop_sha_exact": True,
                "source_contour_boundary_check_passed": True,
                "placed_vertices_and_codes_exact": True,
                "font_and_font_sha256_are_none": True,
                "transformation": transform,
                "anisotropic": anisotropic,
            }
        )
    return {
        "source_placements": evidence,
        "english_items": english_count,
        "all_english_records_and_placed_outlines_exact": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--source-run", type=Path)
    args = parser.parse_args()
    bank = args.bank.resolve()
    source_rows, inventory = bank_inventory(bank)
    baseline = load_baseline()
    if baseline["cfg"].seed != 1909:
        raise ValueError("The frozen source selection seed changed")
    parents = {"third-candidate": baseline["path"]}
    if args.source_run:
        source_run = args.source_run.resolve()
        verify_run(source_run)
        parents["session-o-import"] = source_run
    run = Run.create(
        "session-o-reviewed-math-complete-20",
        seed=1909,
        config={
            "argv": sys.argv,
            "baseline": str(baseline["path"]),
            "baseline_sha256": baseline["run_sha256"],
            "bank": str(bank),
            "bank_manifest_sha256": sha256(bank / "sources.json"),
            "targets": sorted(PRIORITY),
            "render": baseline["cfg"].model_dump(mode="json"),
            "pages": list(range(1, 21)),
            "expected_target_items": 50,
            "expected_formulas": 173,
            "replacement_eligibility": (
                "pi or external/inherited fallback among PRIORITY math items"
            ),
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "production_activation": False,
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, parents)
        protected = {str(path / "run.json"): sha256(path / "run.json") for path in parents.values()}
        protected[str(workspace_root() / "latest.json")] = sha256(workspace_root() / "latest.json")
        snapshot_bank_review(bank, run, protected)
        atomic_json(run.path / "reports/protected-assets.json", protected)
        atomic_json(run.path / "reports/original-bank-files.json", inventory)
        shutil.copytree(bank, run.path / "solutions/session-o-bank")
        copied_bank = run.path / "solutions/session-o-bank"
        for relative, digest in inventory.items():
            if sha256(copied_bank / relative) != digest:
                raise ValueError("Reviewed source copy differs from the original bank")
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline-assets")
        if args.source_run:
            shutil.copytree(source_run / "solutions", run.path / "solutions/source-run")
            shutil.copytree(source_run / "reports", run.path / "reports/source-run")
        shutil.copyfile(
            baseline["path"] / "rendered/homework_solutions.pdf",
            run.path / "comparisons/A-third-candidate.pdf",
        )
        cases, rows, all_decisions, targets, formulas = [], [], [], [], []
        images, mathematical_rows, old_images, old_maths = [], [], [], []
        for entry in baseline["cases"]:
            number, case_id = entry["page"], entry["case_id"]
            print(f"page {number:02d}: {case_id}", flush=True)
            replay = replay_page(baseline, number)
            page, renderer = replay["page"], replay["renderer"]
            before = copy.deepcopy(replay["records"])
            resolver = SessionOResolver(copied_bank, seed=1909)
            candidate = json_value(resolver.apply(page, renderer, before, case_id=case_id))
            decisions = target_decisions(before, candidate, json_value(resolver.decisions))
            image = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"rendered/pages/{number:02d}.pdf"
            )
            checks = validate_increment(
                replay["geometry"],
                page,
                renderer,
                before,
                candidate,
                replay["image"],
                image,
                targets=PRIORITY,
                parser_before=replay["parser"],
            )
            parser_after = json_value(trace(renderer))
            source_checks = verify_source_placements(
                copied_bank, before, candidate, replay["geometry"], page, renderer
            )
            case = {
                "page": number,
                "case_id": case_id,
                "checks": checks,
                "source_checks": source_checks,
                "targets": len(decisions),
                "accepted": sum(d["accepted"] for d in decisions),
                "fallback": fallback_summary(candidate),
            }
            cases.append(case)
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {
                    "baseline": before,
                    "candidate": candidate,
                    "parser_trace": parser_after,
                    "baseline_replay_checks": replay["replay_checks"],
                },
            )
            atomic_json(run.path / f"reports/{case_id}-checks.json", checks)
            atomic_json(run.path / f"reports/{case_id}-source-placement-checks.json", source_checks)
            atomic_json(run.path / f"reports/{case_id}-decisions.json", decisions)
            all_decisions.extend({"page": number, "case_id": case_id, **d} for d in decisions)
            rows.extend({"page": number, "case_id": case_id, **r} for r in candidate)
            paired_panel(
                replay["image"],
                image,
                run.path / f"comparisons/pages/{number:02d}.png",
                f"Page {number}",
                ("A: frozen third candidate", "B: reviewed Session O sources"),
            )
            for decision in decisions:
                i = decision["item_index"]
                old, new = crop_pair(before[i], candidate[i], replay["image"], image, (70, 35))
                paired_panel(
                    old,
                    new,
                    run.path / f"comparisons/targets/{number:02d}-{i:04d}.png",
                    f"Page {number} / item {i} / U+{ord(decision['semantic']):04X}",
                    ("A: previous source", "B: accepted source or explicit retention"),
                )
                targets.append((number, i, old, new))
            formulas.extend(
                group_crops(
                    before,
                    replay["parser"],
                    replay["image"],
                    candidate,
                    parser_after,
                    image,
                    run.path / f"comparisons/formulas/{number:02d}",
                    number,
                    ("A: frozen third candidate", "B: reviewed Session O sources"),
                )
            )
            images.append(image)
            mathematical_rows.append([r for r in candidate if r["context"] == "mathematics"])
            old_images.append(replay["image"])
            old_maths.append([r for r in before if r["context"] == "mathematics"])
        if len(cases) != 20 or len(targets) != 50 or len(formulas) != 173 or len(rows) != 10774:
            raise ValueError("Frozen page, target, formula or visible-item inventory changed")
        with PdfWriter() as writer:
            for n in range(1, 21):
                writer.append(run.path / f"rendered/pages/{n:02d}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        for name, panels in (("all-targets", targets), ("all-formulas", formulas)):
            folder = run.path / f"comparisons/{name}"
            folder.mkdir()
            local_comparison(folder, panels, labels=("A: third candidate", "B: Session O"))
            atomic_json(
                folder / "index.json", [{"page": p[0], "item_or_formula": p[1]} for p in panels]
            )
        for name, ims, maths in (
            ("fallback-overlay-A", old_images, old_maths),
            ("fallback-overlay-B", images, mathematical_rows),
        ):
            folder = run.path / "comparisons" / name
            folder.mkdir()
            fallback_overview(ims, maths, folder)
        fallback = [
            r for r in rows if r["context"] == "mathematics" and r["source_type"] in FALLBACK_TYPES
        ]
        refused = [d for d in all_decisions if not d["accepted"]]
        summary = {
            "pages": len(cases),
            "visible_items": len(rows),
            "math_items": sum(map(len, mathematical_rows)),
            "target_regions": len(targets),
            "formula_regions": len(formulas),
            "accepted_replacements": sum(d["accepted"] for d in all_decisions),
            "replacements_by_symbol": dict(
                Counter(d["semantic"] for d in all_decisions if d["accepted"])
            ),
            "refused_or_retained_targets": len(refused),
            "fallback_count": len(fallback),
            "fallback_symbols": dict(Counter(r["semantic"] for r in fallback)),
            "math_source_types": dict(
                Counter(r["source_type"] for page in mathematical_rows for r in page)
            ),
            "bank_accepted_source_count": sum(
                r.get("review_status") == "agent-accepted" for r in source_rows
            ),
            "bank_unaccepted_source_count": sum(
                r.get("review_status") != "agent-accepted" for r in source_rows
            ),
            "used_real_source_counts": dict(
                Counter(d["source_id"] for d in all_decisions if d["accepted"])
            ),
            "all_non_target_records_geometry_exact": True,
            "english_items": sum(c["source_checks"]["english_items"] for c in cases),
            "all_english_records_and_placed_outlines_exact": True,
            "all_session_o_raw_crop_sha_and_placed_contours_exact": True,
            "pixels_changed_outside_target_regions": 0,
            "all_parser_traces_exact": True,
            "new_contacts_or_subpixel_crowding": 0,
            "new_neural_updates": 0,
            "author_acceptance": False,
            "production_go": False,
            "root_visual_review": "pending after immutable delivery",
            "semantic_scope": (
                "Frozen answers and formula structures preserved; "
                "source/ink review remains separate"
            ),
        }
        for name, content in (
            ("cases", cases),
            ("glyph-provenance", rows),
            ("decisions", all_decisions),
            ("fallback-reasons", fallback),
            ("refused-targets", refused),
            (
                "bank-unaccepted-sources",
                [r for r in source_rows if r.get("review_status") != "agent-accepted"],
            ),
            ("source-summary", summary),
        ):
            atomic_json(run.path / f"reports/{name}.json", content)
        (run.path / "reports/SESSION_O_DELIVERY.md").write_text(
            "# Session O isolated complete homework comparison\n\n"
            "A is the exact third author-feedback candidate. B uses only agent-reviewed O-FULL "
            "sources for explicitly selected mathematical items. English, all other glyphs, "
            "formula parsing, item identity/order and page dimensions are frozen.\n\n"
            f"{summary['accepted_replacements']} / 50 eligible targets changed; "
            f"{len(refused)} retain their prior source with a recorded reason. "
            f"{len(fallback)} mathematical font fallbacks remain.\n\n"
            "Each transformation is recorded separately from the original RGB sample. "
            "Geometric scaling is not original untransformed ink or neural training. "
            "The unchanged baseline structure does not replace independent symbol recognition "
            "and visual review. This exposed candidate awaits root and author visual acceptance. "
            "No production Profile, latest pointer or historical source was changed.\n",
        )
        for relative, digest in inventory.items():
            if sha256(bank / relative) != digest or sha256(copied_bank / relative) != digest:
                raise ValueError("Original or archived source bank changed during rendering")
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Protected asset changed: {path}")
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        # A post-seal verification failure must not write logs into a sealed run.
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    main()
