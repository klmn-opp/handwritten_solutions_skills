"""Exercise the public candidate entrypoint on archived and new complete answers.

All inputs are immutable per run. Failures and differences remain evidence rather
than triggering source rewrites, page-specific transforms or seed selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfWriter

from hwsol.jobs.runtime import verify_job
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.store import ProfileStore
from hwsol.workspace import Run, render_assignment, verify_run, workspace_root
from research.author_feedback_delivery import snapshot_sources
from research.author_feedback_revision import group_crops
from research.corpus.math_glyph_completion import local_comparison
from research.cpu_candidate_cases import solution_checks
from research.math_feedback_candidate import paired_panel

ROOT = Path(__file__).resolve().parents[1]
ASSETS = "20261120T144402240289Z-portable-personal-cpu-assets-597a21c6"
ACCEPTED20 = "20261120T140811736339Z-session-o-reviewed-math-complete-20-de331ba4"
INPUT20 = "20261120T115012607677Z-author-feedback-final-20-pages-50910a3a"
ACCEPTED4 = "20261120T140350095212Z-session-o-fresh-four-homework-pages-ad9aab55"
FROZEN_CASES = ROOT / "workspace/analysis/20261120T144256699338Z-cpu-candidate-new-cases-frozen"


def read(path: Path):
    return json.loads(path.read_text("utf-8"))


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(source) != sha256(target):
        raise ValueError(f"Copied input changed: {source}")


def frozen_inputs(mode: str) -> list[dict]:
    receipt = read(FROZEN_CASES / "FREEZE.json")
    if sha256(ROOT / "research/cpu_candidate_cases.py") != receipt["code_sha256"]:
        raise ValueError("New-case source changed after its initial freeze")
    for name, digest in receipt["files_sha256"].items():
        if sha256(FROZEN_CASES / name) != digest:
            raise ValueError(f"Frozen case evidence changed: {name}")
    rows = read(FROZEN_CASES / ("cases.json" if mode == "new" else "stress-cases.json"))
    for row in rows:
        row.setdefault("statement", "Engineering diagnostic: " + row["title"])
        row.setdefault("expected", "render")
        row["background"] = {"kind": "plain"}
        row["render_override"] = {}
    if mode == "stress":
        # These two controls are specified by the original pre-render protocol.
        for row in list(rows):
            if row["category"] == "supported-geometry":
                control = copy.deepcopy(row)
                control["id"] += "-grid-seed1943"
                control["background"] = {"kind": "grid"}
                control["render_override"] = {"seed": 1943}
                rows.append(control)
    return rows


def regression_inputs(runs: Path) -> list[dict]:
    rows = []
    accepted, inputs = runs / ACCEPTED20, runs / INPUT20
    for case in read(accepted / "reports/cases.json"):
        case_id, page = case["case_id"], case["page"]
        folder = inputs / "solutions/inputs" / case_id
        original_job = read(folder / "job.json")
        problem = original_job["problems"][0]
        source = inside(folder, problem["solution"])
        rows.append(
            {
                "id": case_id,
                "title": problem["number"],
                "statement": problem["statement"],
                "source_path": source,
                "markdown": source.read_text("utf-8"),
                "original_job_path": folder / "job.json",
                "original_job": original_job,
                "expected": "render",
                "reference_run": accepted,
                "reference_page": page,
                "reference_records": accepted / f"reports/{case_id}-provenance.json",
            }
        )
    accepted = runs / ACCEPTED4
    for case in read(accepted / "reports/cases.json"):
        page = case["page"]
        source = accepted / f"solutions/{page:02d}.md"
        rows.append(
            {
                "id": f"session-o-fresh-{page:02d}",
                "title": case["title"],
                "statement": "Replay the complete already reviewed solution: " + case["title"],
                "source_path": source,
                "markdown": source.read_text("utf-8"),
                "expected": "render",
                "background": {"kind": "plain"},
                "render_override": {},
                "reference_run": accepted,
                "reference_page": page,
                "reference_records": accepted / f"reports/{page:02d}-provenance.json",
            }
        )
    return rows


def materialize(run: Run, number: int, case: dict, profile, mode: str) -> Path:
    folder = run.path / f"solutions/inputs/{number:02d}-{case['id']}"
    folder.mkdir(parents=True)
    source = folder / "solution.md"
    if case.get("source_path"):
        copy_file(case["source_path"], source)
    else:
        source.write_text(case["markdown"], encoding="utf-8")
    if case.get("original_job"):
        original = case["original_job_path"].parent
        data = copy.deepcopy(case["original_job"])
        bindings = []
        copy_file(case["original_job_path"], folder / "original-job.not-executed.json")
        for problem in data["problems"]:
            for src in problem["sources"]:
                copy_file(inside(original, src["path"]), inside(folder, src["path"]))
                actual = sha256(inside(folder, src["path"]))
                if src.get("sha256") != actual:
                    bindings.append(
                        {
                            "path": src["path"],
                            "historical_authoring_digest": src.get("sha256"),
                            "actual_unchanged_source_digest": actual,
                            "reason": "Bind execution to archived bytes; old Job preserved",
                            "source_text_edited": False,
                        }
                    )
                src["sha256"] = actual
            problem["solution"] = "solution.md"
        atomic_json(folder / "source-digest-rebinding.json", bindings)
        for bg in [data.get("background"), *[p.get("background") for p in data["problems"]]]:
            if bg and bg.get("path"):
                copy_file(inside(original, bg["path"]), inside(folder, bg["path"]))
    else:
        (folder / "problem.txt").write_text(case["statement"], encoding="utf-8")
        data = {
            "schema_version": 1,
            "background": case["background"],
            "problems": [
                {
                    "id": case["id"],
                    "number": case["title"],
                    "statement": case["statement"],
                    "sources": [{"path": "problem.txt"}],
                    "solution": "solution.md",
                    "verification": {
                        "status": "passed" if mode != "stress" else "not_checked",
                        "evidence": (
                            "Exact analytic results independently checked in frozen new cases; "
                            "rendering and style require separate review."
                            if mode == "new"
                            else "Already reviewed original solution replay; no new answer claim."
                            if mode == "regression"
                            else "Engineering input only; not a solved homework claim."
                        ),
                    },
                }
            ],
        }
    data.update(
        backend="personal-cpu-experimental",
        realism="baseline",
        profile=profile.manifest.ref,
        profile_sha256=profile.manifest_sha256,
        generated_by="personal-cpu-generic-entrypoint-evidence/1",
        render=case.get("render_override", {}),
    )
    target = folder / "job.json"
    atomic_json(target, data)
    return target


def semantic_identity(rows: list[dict]) -> list:
    return [[r["context"], r["semantic"], r.get("latex")] for r in rows]


def glyph_identity(row: dict) -> dict:
    result = {
        key: row.get(key)
        for key in (
            "semantic",
            "source_type",
            "actual_glyph_id",
            "glyph_name",
            "variant",
            "font_sha256",
        )
    }
    for key in ("scan_source", "session_source", "session_o_source"):
        if isinstance(row.get(key), dict):
            result[key] = {
                k: row[key].get(k) for k in ("id", "source_id", "crop_sha256", "character")
            }
    return result


def compare_reference(run: Run, number: int, case: dict, page: dict, image: Image.Image) -> tuple:
    parent, parent_page = case["reference_run"], case["reference_page"]
    original = read(case["reference_records"])
    old, new = original["candidate"], page["glyph_provenance"]
    old_trace = original.get("parser_trace", original.get("parser"))
    new_trace = {"formulas": page["formula_groups"], "matrices": page["matrix_groups"]}
    before_path = parent / f"rendered/pages/{parent_page:02d}.png"
    before = Image.open(before_path).convert("RGB")
    archived_png_size, public_png_size = list(before.size), list(image.size)
    original_png_equal = before.size == image.size and before.tobytes() == image.tobytes()

    # Historical PNGs retain PDFium's native ceil width2481. Public Runtime has
    # always normalized to config.mm(A4)=2480. Compare vectors at identical scale,
    # while recording the unchanged original/canonical PNG contract difference.
    def native_pdf(path: Path) -> Image.Image:
        with pdfium.PdfDocument(path) as document:
            with closing(document[0]) as sheet, closing(sheet.render(scale=300 / 72)) as bitmap:
                return bitmap.to_pil().convert("RGB")

    before = native_pdf(parent / f"rendered/pages/{parent_page:02d}.pdf")
    image = native_pdf(run.path / f"rendered/pages/{number:02d}.pdf")
    a, b = np.asarray(before), np.asarray(image)
    same_size = a.shape == b.shape
    diff = np.any(a != b, axis=2) if same_size else None
    english = np.zeros(a.shape[:2], dtype=bool)
    for row in old:
        if row["context"] != "text":
            continue
        x0, y0, x1, y1 = row["bounds_px"]
        english[
            max(0, math.floor(y0)) : min(a.shape[0], math.ceil(y1)),
            max(0, math.floor(x0)) : min(a.shape[1], math.ceil(x1)),
        ] = True
    differences = [
        {"index": i, "before": glyph_identity(left), "after": glyph_identity(right)}
        for i, (left, right) in enumerate(zip(old, new, strict=False))
        if glyph_identity(left) != glyph_identity(right)
    ]
    text_bounds = lambda rows: [  # noqa: E731 -- compact projection for this paired comparison
        [r["item_index"], r["semantic"], r["bounds_px"]] for r in rows if r["context"] == "text"
    ]
    result = {
        "reference_run": str(parent),
        "reference_page": parent_page,
        "reference_record_sha256": sha256(case["reference_records"]),
        "reference_png_sha256": sha256(before_path),
        "archived_png_dimensions": archived_png_size,
        "public_png_dimensions": public_png_size,
        "original_png_pixels_equal": original_png_equal,
        "comparison_rasterization": "Both actual PDFs rendered by PDFium at300/72, no resize",
        "same_dimensions": same_size,
        "pixels_equal": bool(same_size and not diff.any()),
        "changed_pixels": int(diff.sum()) if same_size else None,
        "changed_pixels_in_original_english_boxes": int((diff & english).sum())
        if same_size
        else None,
        "item_count_before": len(old),
        "item_count_after": len(new),
        "semantic_identity_equal": semantic_identity(old) == semantic_identity(new),
        "parser_trace_equal": old_trace == new_trace,
        "english_bounds_equal": text_bounds(old) == text_bounds(new),
        "source_glyph_identity_equal": len(old) == len(new) and not differences,
        "source_glyph_differences": differences,
        "scope": "Mechanical comparison; differences retained for independent review",
    }
    paired_panel(
        before,
        image,
        run.path / f"comparisons/pages/{number:02d}.png",
        case["title"],
        ("Author-accepted archived candidate", "Generic public candidate entrypoint"),
    )
    panels = []
    try:
        panels = group_crops(
            old,
            old_trace,
            before,
            new,
            new_trace,
            image,
            run.path / f"comparisons/formulas/{number:02d}",
            number,
            ("Accepted archived candidate", "Generic public entrypoint"),
        )
    except (ValueError, IndexError, TypeError) as exc:
        result["formula_comparison_unavailable"] = str(exc)
    return result, panels


def combine_new_assignment(evidence: Path) -> dict:
    """Exercise one real six-problem public transaction after all single cases pass."""
    verify_run(evidence)
    cases = read(evidence / "reports/cases.json")
    if len(cases) != 6 or any(c["status"] != "rendered" for c in cases):
        raise ValueError("Combined assignment requires all six unmodified new cases")
    if read(evidence / "reports/summary.json")["mode"] != "new":
        raise ValueError("Only the frozen new-case evidence may be combined here")
    workspace = workspace_root()
    latest = workspace / "latest.json"
    latest_sha = sha256(latest)
    run = Run.create(
        "personal-cpu-six-problem-assignment",
        seed=1909,
        config={
            "entry": "research.personal_cpu_evidence.combine_new_assignment",
            "parent": str(evidence),
            "parent_sha256": sha256(evidence / "run.json"),
            "purpose": "public multi-problem integration and portable-install reference",
            "latest_argument": True,
            "source_rewrites": False,
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, {"six-single-cases": evidence})
        archived_name = read(evidence / "run.json")["config"].get(
            "copied_archive", "personal-cpu-candidate-v1.zip"
        )
        archive = inside(evidence, f"solutions/{archived_name}")
        store = ProfileStore(run.path / "solutions/isolated-profile-store")
        imported = store.import_archive(archive)
        profile = store.resolve(imported["imported"], candidate=True)
        folder = run.path / "solutions/input"
        folder.mkdir()
        combined = None
        for number, case in enumerate(cases, 1):
            original = inside(evidence, case["input_manifest"])
            data = read(original)
            problem = data["problems"][0]
            prefix = f"items/{number:02d}"
            answer = inside(original.parent, problem["solution"])
            if sha256(answer) != case["solution_sha256"]:
                raise ValueError("Previously accepted new answer changed")
            problem["solution"] = f"{prefix}/solution.md"
            copy_file(answer, inside(folder, problem["solution"]))
            for src in problem["sources"]:
                source = inside(original.parent, src["path"])
                src["path"] = f"{prefix}/{src['path']}"
                copy_file(source, inside(folder, src["path"]))
            if combined is None:
                combined = data
                combined["problems"] = []
            combined["problems"].append(problem)
        combined["profile"] = profile.manifest.ref
        combined["profile_sha256"] = profile.manifest_sha256
        manifest = folder / "job.json"
        atomic_json(manifest, combined)
        child_id = f"{run.path.name}-public"
        result = render_assignment(manifest, store=store, workspace=workspace, run_id=child_id)
        child = Path(result["run"])
        record = read(child / "run.json")
        if record["status"] != "held" or record["deliverable"] or sha256(latest) != latest_sha:
            raise ValueError("Combined candidate changed production admission")
        verification = verify_job(child / "rendered", pixels=True)
        if verification["page_count"] != 6:
            raise ValueError("Combined assignment lost a problem")
        checks = []
        for number, case in enumerate(cases, 1):
            first = evidence / f"rendered/pages/{number:02d}.png"
            second = child / f"rendered/pages/{number:02d}.png"
            a = Image.open(first).convert("RGB")
            b = Image.open(second).convert("RGB")
            if a.size != b.size or a.tobytes() != b.tobytes():
                raise ValueError(f"Single/combined page pixels differ: {case['id']}")
            checks.append({"id": case["id"], "page": number, "pixels_equal": True})
        shutil.copytree(child / "rendered", run.path / "rendered", dirs_exist_ok=True)
        copy_file(child / "run.json", run.path / "reports/public-child-receipt.json")
        atomic_json(run.path / "reports/page-checks.json", checks)
        summary = {
            "pages": 6,
            "all_single_case_pixels_equal": True,
            "public_job": str(child / "rendered"),
            "canonical_manifest": str(child / "rendered/manifest.json"),
            "latest_unchanged": True,
            "latest_sha256": latest_sha,
            "author_style_approval": False,
            "production_go": False,
        }
        run.finish("held", result=summary)
        verify_run(run.path)
        return {"run": str(run.path), **summary}
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("regression", "new", "stress"))
    parser.add_argument("--asset-run", default=ASSETS)
    args = parser.parse_args()
    workspace = workspace_root()
    runs = workspace / "runs"
    parents = {"assets": inside(runs, args.asset_run)}
    if args.mode == "regression":
        parents.update(
            accepted20=runs / ACCEPTED20, input20=runs / INPUT20, accepted4=runs / ACCEPTED4
        )
    for parent in parents.values():
        verify_run(parent)
    asset_result = read(parents["assets"] / "reports/result.json")
    archived_relative = Path(asset_result["archive"]).resolve().relative_to(parents["assets"])
    archive = inside(parents["assets"], archived_relative.as_posix())
    if sha256(archive) != asset_result["archive_sha256"]:
        raise ValueError("Exported candidate archive changed")
    latest = workspace / "latest.json"
    if not latest.is_file():
        raise ValueError("This evidence requires the real existing production latest pointer")
    protected = {
        str(p): sha256(p)
        for p in (
            latest,
            archive,
            ROOT / "uv.lock",
            ROOT / "source_pdf/session-O_full_source.pdf",
            ROOT / "docs/MASTER_EXECUTION_PLAN.md",
            *[p / "run.json" for p in parents.values()],
        )
    }
    refs = {
        name: subprocess.check_output(["git", "rev-parse", name], cwd=ROOT, text=True).strip()
        for name in ("main", "v0.6.0^{}")
    }
    cases = regression_inputs(runs) if args.mode == "regression" else frozen_inputs(args.mode)
    run = Run.create(
        f"personal-cpu-{args.mode}-evidence",
        config={
            "mode": args.mode,
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "copied_archive": archive.name,
            "latest_argument": True,
            "cases": [case["id"] for case in cases],
            "source_rewrites": False,
            "neural_updates": 0,
            "fresh_holdout": False,
            "author_style_scope": "Previous24 broadly approved; new cases need separate review",
        },
        seed=1909,
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, parents)
        atomic_json(run.path / "reports/protected-before.json", {"files": protected, "refs": refs})
        copy_file(archive, run.path / "solutions" / archive.name)
        store = ProfileStore(run.path / "solutions/isolated-profile-store")
        imported = store.import_archive(run.path / "solutions" / archive.name)
        if not imported["candidate"] or imported["activated"]:
            raise ValueError("Portable candidate did not stay isolated")
        profile = store.resolve(imported["imported"], candidate=True)
        atomic_json(run.path / "reports/import.json", {**imported, "profile": profile.provenance()})
        if args.mode == "regression":
            expected_cfg = read(runs / ACCEPTED20 / "run.json")["config"]["render"]
            if profile.manifest.render.model_dump(mode="json") != expected_cfg:
                raise ValueError("Candidate defaults differ from accepted regression configuration")
        else:
            for name in ("FREEZE.json", "cases.json", "stress-cases.json", "protocol.json"):
                copy_file(FROZEN_CASES / name, run.path / "reports/frozen-inputs" / name)
            if args.mode == "new":
                atomic_json(run.path / "reports/analytic-checks.json", solution_checks())
        all_rows, reports, accepted_pdfs, panels = [], [], [], []
        for number, case in enumerate(cases, 1):
            print(number, case["id"], flush=True)
            manifest = materialize(run, number, case, profile, args.mode)
            source_sha = sha256(manifest.parent / "solution.md")
            child_id = f"{run.path.name}-case{number:02d}"
            child = runs / child_id
            record = {
                "number": number,
                "id": case["id"],
                "title": case["title"],
                "expected": case["expected"],
                "child_run": str(child),
                "input_manifest": str(manifest.relative_to(run.path)),
                "input_sha256": sha256(manifest),
                "solution_sha256": source_sha,
                "source_digest_rebindings": (
                    read(manifest.parent / "source-digest-rebinding.json")
                    if (manifest.parent / "source-digest-rebinding.json").exists()
                    else []
                ),
            }
            try:
                # Deliberately exercise the public default latest=True admission boundary.
                render_assignment(manifest, store=store, workspace=workspace, run_id=child_id)
            except Exception as exc:
                record.update(status="rejected", error=str(exc), error_type=type(exc).__name__)
                if not (child / "run.json").exists():
                    raise
                if verify_run(child)["run_status"] != "failed":
                    raise ValueError(
                        "A rejected child did not retain a valid failure receipt"
                    ) from exc
                failure = child / "payload/validation_report.json"
                if failure.exists():
                    copy_file(failure, run.path / f"reports/cases/{number:02d}-failure.json")
                if (child / "rendered/homework_solutions.pdf").exists():
                    raise ValueError("Rejected candidate exposed an accepted partial PDF") from exc
            else:
                metadata = read(child / "run.json")
                if metadata["status"] != "held" or metadata["deliverable"]:
                    raise ValueError("Experimental success escaped candidate admission")
                verify_run(child)
                record["job_verification"] = verify_job(child / "rendered", pixels=True)
                validation = read(child / "rendered/validation_report.json")
                page = validation["pages"][0]
                if len(validation["pages"]) != 1:
                    raise ValueError("One problem produced multiple accepted pages")
                copy_file(
                    child / "rendered/validation_report.json",
                    run.path / f"reports/cases/{number:02d}-validation.json",
                )
                dest = run.path / f"rendered/pages/{number:02d}.pdf"
                copy_file(child / "rendered/homework_solutions.pdf", dest)
                copy_file(child / "rendered/pages/01.png", dest.with_suffix(".png"))
                if sha256(child / "rendered/solutions/01.md") != source_sha:
                    raise ValueError("Runtime rewrote an accepted solution")
                image = Image.open(dest.with_suffix(".png")).convert("RGB")
                record.update(
                    status="rendered",
                    fallback_count=len(page["math_fallback_glyphs"]),
                    math_fallbacks=page["math_fallback_glyphs"],
                    math_source_types=page["candidate"]["math_source_types"],
                    visible_math_items=page["candidate"]["visible_math_items"],
                )
                if case.get("reference_run"):
                    comparison, local_panels = compare_reference(run, number, case, page, image)
                    record["regression"] = comparison
                    panels.extend(local_panels)
                all_rows.extend(
                    {**r, "page": number, "case_id": case["id"]} for r in page["glyph_provenance"]
                )
                accepted_pdfs.append((number, case["id"], dest))
            if sha256(latest) != protected[str(latest)]:
                raise ValueError("Production latest changed during candidate execution")
            if sha256(manifest.parent / "solution.md") != source_sha:
                raise ValueError("Evidence input changed during execution")
            copy_file(
                child / "run.json", run.path / f"reports/cases/{number:02d}-child-receipt.json"
            )
            record["child_run_sha256"] = sha256(child / "run.json")
            record["production_latest_unchanged"] = True
            expected_rejection = case["expected"].startswith("explicit-")
            record["expectation_met"] = (record["status"] == "rejected") == expected_rejection
            atomic_json(run.path / f"reports/cases/{number:02d}-result.json", record)
            reports.append(record)
        if accepted_pdfs:
            with PdfWriter() as writer:
                for _, _, pdf in accepted_pdfs:
                    writer.append(pdf)
                writer.write(run.path / "rendered/homework_solutions.pdf")
        if panels:
            local_comparison(
                run.path / "comparisons/formulas",
                panels,
                labels=("Accepted archived candidate", "Generic public entrypoint"),
            )
        if any(sha256(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("Protected source or production asset changed")
        for name, digest in refs.items():
            if (
                subprocess.check_output(["git", "rev-parse", name], cwd=ROOT, text=True).strip()
                != digest
            ):
                raise ValueError("Production Git reference changed")
        math_rows = [row for row in all_rows if row["context"] == "mathematics"]
        summary = {
            "mode": args.mode,
            "cases": len(cases),
            "rendered": len(accepted_pdfs),
            "rejected": len(cases) - len(accepted_pdfs),
            "expectations_met": sum(r["expectation_met"] for r in reports),
            "unexpected_outcomes": [r["id"] for r in reports if not r["expectation_met"]],
            "math_visible_items": len(math_rows),
            "font_fallbacks": dict(
                Counter(
                    r["semantic"]
                    for r in math_rows
                    if r["source_type"] in {"inherited-font", "external-font"}
                )
            ),
            "regression_pixel_equal": sum(
                r.get("regression", {}).get("pixels_equal", False) for r in reports
            ),
            "regression_identity_equal": sum(
                r.get("regression", {}).get("semantic_identity_equal", False) for r in reports
            ),
            "all_child_latest_checks_passed": True,
            "protected_assets_unchanged": True,
            "source_rewrites": 0,
            "historical_source_digest_rebindings": sum(
                len(r["source_digest_rebindings"]) for r in reports
            ),
            "neural_updates": 0,
            "visual_review": "pending independent inspection of actual pages",
            "author_style_approval": False,
            "production_go": False,
            "merged_pdf_contains": [case_id for _, case_id, _ in accepted_pdfs],
            "no_placeholder_pdf": True,
        }
        atomic_json(run.path / "reports/cases.json", reports)
        atomic_json(run.path / "reports/glyph-provenance.json", all_rows)
        atomic_json(run.path / "reports/summary.json", summary)
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    main()
