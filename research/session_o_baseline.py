"""Exact frozen third-candidate replay before reviewed Session O substitution.

This module neither imports new ink nor alters previous checkpoints, sources,
Profiles, evaluation ledgers or production pointers. A new symbol treatment must
start from an exact page replay and use an explicit mathematical target scope.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import json_value, snapshot_sources
from research.author_feedback_revision import identity, trace
from research.author_feedback_round2_checks import page_geometry, same_path
from research.author_feedback_round3 import load_baseline as load_previous
from research.author_feedback_round3 import replay_page as replay_previous
from research.corpus.math_glyph_completion import raster
from research.f_size_feedback import contour_distance
from research.f_variation_feedback import FVariationFeedback
from research.homework_benchmark.render import read, save_pdf

ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = "20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e"
BASE_SHA = "9419cb9713df8d5118603bd166f5f6c4ec31e2148afcea5b2058a11e4789248f"
PDF_SHA = "c10f557a97ef0485b583d7111f0ecf418d9d6aec1b4cfe1f4f571cc410732933"
SEMANTIC_FIELDS = (
    "item_index",
    "context",
    "semantic",
    "latex",
    "matrix_environment",
    "structure",
    "codepoint",
)


def load_baseline():
    """Load verified 4af assets and the source-pinned prior replay chain."""
    base = workspace_root() / "runs" / BASE_RUN
    verify_run(base)
    if sha256(base / "run.json") != BASE_SHA:
        raise ValueError("Frozen third-candidate receipt changed")
    if sha256(base / "rendered/homework_solutions.pdf") != PDF_SHA:
        raise ValueError("Frozen third-candidate PDF changed")
    approval = read(base / "reports/approval-receipt.json")
    for relative, digest in approval["source_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Frozen third-candidate source changed: {relative}")
    previous = load_previous()
    if previous["run_sha256"] != approval["baseline_sha256"]:
        raise ValueError("Frozen prior candidate ancestry changed")
    cases = read(base / "reports/cases.json")
    if [row["page"] for row in cases] != list(range(1, 21)):
        raise ValueError("Frozen twenty-page order changed")
    return {
        **previous,
        "path": base,
        "run_sha256": BASE_SHA,
        "pdf_sha256": PDF_SHA,
        "previous": previous,
        "cases": cases,
        "f_sources": base / "solutions/f-sources",
    }


def geometry_inventory(paths, records):
    """Store exact placed path identities separately from reported ink bounds."""
    result = []
    for i, (path, row) in enumerate(zip(paths, records, strict=True)):
        vertices = np.asarray(path.vertices, dtype="<f8")
        codes = np.asarray(path.codes, dtype="u1")
        result.append(
            {
                "item_index": i,
                "semantic": row["semantic"],
                "context": row["context"],
                "vertices_shape": list(vertices.shape),
                "vertices_f64le_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
                "codes_uint8_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
                "bounds_px": path.get_extents().extents.tolist(),
            }
        )
    return result


def replay_page(baseline, number):
    """Return page/renderer/records/parser/image/entry/inputs/source plus proof.

    Uses the final 4af f-source path, so provenance path strings replay exactly.
    A PDF byte match verifies emitted drawing commands/resources as well as the
    independent pixel, item-source, semantic-order and mathematical parse checks.
    """
    if number not in range(1, 21):
        raise ValueError("Only the frozen original twenty pages are supported")
    replay = replay_previous(baseline["previous"], number)
    page, renderer = replay["page"], replay["renderer"]
    case_id = replay["entry"]["case_id"]
    treatment = FVariationFeedback(
        baseline["f_sources"],
        source=replay["source"],
        title=case_id,
        seed=baseline["cfg"].seed,
    )
    records = json_value(treatment.apply(page, renderer, replay["records"]))
    parser = json_value(trace(renderer))
    base = baseline["path"]
    original = read(base / f"reports/{case_id}-provenance.json")
    if records != original["candidate"]:
        raise ValueError(f"Page {number}: frozen item provenance does not replay exactly")
    if parser != original["parser_trace"]:
        raise ValueError(f"Page {number}: frozen mathematical parser trace changed")
    decisions = json_value(treatment.decisions)
    if decisions != read(base / f"reports/{case_id}-decisions.json"):
        raise ValueError(f"Page {number}: original f admission decisions changed")
    blob = renderer.to_vector_pdf([page])
    prior = base / f"rendered/pages/{number:02d}.pdf"
    if blob != prior.read_bytes():
        raise ValueError(f"Page {number}: emitted PDF drawing commands/resources differ")
    image = raster(blob)
    previous_image = Image.open(base / f"rendered/pages/{number:02d}.png").convert("RGB")
    if image.size != previous_image.size or image.tobytes() != previous_image.tobytes():
        raise ValueError(f"Page {number}: pixels differ from delivered candidate")
    geometry = page_geometry(page, renderer)
    checks = {
        "page": number,
        "case_id": case_id,
        "items": len(records),
        "math_items": sum(r["context"] == "mathematics" for r in records),
        "full_provenance_exact": True,
        "f_admission_decisions_exact": True,
        "parser_trace_exact": True,
        "all_pdf_bytes_exact": True,
        "all_emitted_vector_geometry_and_resources_exact": True,
        "all_rgb_pixels_exact": True,
        "pdf_sha256": sha256(prior),
        "semantic_claim_scope": (
            "Exact frozen rendering replay; not new independent answer verification"
        ),
    }
    return {
        **replay,
        "page": page,
        "renderer": renderer,
        "records": copy.deepcopy(records),
        "parser": parser,
        "image": image,
        "entry": copy.deepcopy(next(r for r in baseline["cases"] if r["page"] == number)),
        "pdf_bytes": blob,
        "geometry": geometry,
        "f_decisions": decisions,
        "replay_checks": checks,
    }


def validate_increment(
    before_paths,
    page,
    renderer,
    before,
    after,
    old_image,
    image,
    *,
    targets,
    parser_before,
):
    """Gate explicitly named math symbols without weakening earlier sealed gates.

    This is a separate extension for new symbols: no old check is monkey-patched.
    The caller supplies independent pre-mutation geometry and parser snapshots.
    Target envelopes are output declarations, not independent safety evidence;
    path bounds, page edges, neighbors, non-target items and pixels are checked.
    """
    target_set = frozenset(targets)
    if not target_set or any(not isinstance(ch, str) or len(ch) != 1 for ch in target_set):
        raise ValueError("A nonempty explicit set of individual mathematical symbols is required")
    if len(before) != len(after) or len(before_paths) != len(page.items):
        raise ValueError("Visible item count changed")
    if identity(before) != identity(after):
        raise ValueError("Mathematical/text identity or order changed")
    if json_value(trace(renderer)) != json_value(parser_before):
        raise ValueError("Mathematical parser/layout trace changed")
    after_paths = page_geometry(page, renderer)
    changed = []
    for i, (old, new, a, b) in enumerate(
        zip(before, after, before_paths, after_paths, strict=True)
    ):
        if old["item_index"] != i or new["item_index"] != i:
            raise ValueError("Provenance drawing order mismatch")
        if any(old.get(field) != new.get(field) for field in SEMANTIC_FIELDS):
            raise ValueError(f"Semantic/structural field changed at {i}")
        permitted = old["context"] == "mathematics" and old["semantic"] in target_set
        same = same_path(a, b)
        if not permitted and (old != new or not same):
            raise ValueError(f"Non-target source or geometry changed at {i}: {old['semantic']}")
        if not same:
            changed.append(i)
    previous_pixels, current_pixels = np.asarray(old_image), np.asarray(image)
    if previous_pixels.shape != current_pixels.shape:
        raise ValueError("Page dimensions changed")
    difference = np.any(previous_pixels != current_pixels, axis=2)
    mask = np.zeros(difference.shape, dtype=bool)
    bounds = np.array([path.get_extents().extents for path in after_paths])
    neighbors = []
    for i in changed:
        row, path = after[i], after_paths[i]
        box = path.get_extents()
        declared = row.get("allowed_bounds_px")
        if declared is None or not np.allclose(box.extents, row["bounds_px"], atol=0.03, rtol=0):
            raise ValueError(f"Missing output envelope or inaccurate bounds for {i}")
        if (
            min(
                box.xmin - declared[0],
                box.ymin - declared[1],
                declared[2] - box.xmax,
                declared[3] - box.ymax,
            )
            < -0.03
        ):
            raise ValueError(f"Target {i} exceeds its declared output envelope")
        if min(box.xmin, box.ymin, image.width - box.xmax, image.height - box.ymax) < 3:
            raise ValueError(f"Target {i} reaches page boundary")
        for bb in (before[i]["bounds_px"], row["bounds_px"]):
            x0, y0, x1, y1 = bb
            mask[
                max(0, math.floor(y0) - 2) : min(image.height, math.ceil(y1) + 2),
                max(0, math.floor(x0) - 2) : min(image.width, math.ceil(x1) + 2),
            ] = True
        close = np.flatnonzero(
            (bounds[:, 2] >= box.xmin - 0.60)
            & (bounds[:, 0] <= box.xmax + 0.60)
            & (bounds[:, 3] >= box.ymin - 0.60)
            & (bounds[:, 1] <= box.ymax + 0.60)
        )
        for j in close:
            if i == j:
                continue
            was = bool(before_paths[i].intersects_path(before_paths[j], filled=True))
            now = bool(path.intersects_path(after_paths[j], filled=True))
            if now and not was:
                raise ValueError(f"New ink contact between target {i} and item {j}")
            previous_gap = contour_distance(before_paths[i], before_paths[j])
            current_gap = contour_distance(path, after_paths[j])
            if current_gap < 0.60 and current_gap < previous_gap - 0.05:
                raise ValueError(f"New subpixel crowding between target {i} and item {j}")
            neighbors.append(
                {
                    "item": i,
                    "other": int(j),
                    "original_contact": was,
                    "final_contact": now,
                    "previous_gap_px": previous_gap,
                    "final_gap_px": current_gap,
                }
            )
    outside = int(np.count_nonzero(difference & ~mask))
    if outside:
        raise ValueError(f"Unexpected {outside} changed pixels outside explicit target regions")
    return {
        "targets": sorted(target_set),
        "scope": "mathematics only",
        "changed_items": changed,
        "changed_pixels": int(difference.sum()),
        "pixels_changed_outside_target_boxes": outside,
        "non_target_records_exact": True,
        "non_target_placed_geometry_exact": True,
        "semantic_identity_order_count_exact": True,
        "parser_trace_exact": True,
        "new_ink_contacts": 0,
        "new_subpixel_crowding": 0,
        "neighbors": neighbors,
        "semantic_claim_scope": (
            "Frozen formula structure preserved; independent ink identity review required"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    baseline = load_baseline()
    run = Run.create(
        "session-o-exact-baseline-replay",
        seed=baseline["cfg"].seed,
        config={
            "baseline": BASE_RUN,
            "baseline_sha256": BASE_SHA,
            "pages": list(range(1, 21)),
            "render": baseline["cfg"].model_dump(mode="json"),
            "new_ink_imported": False,
            "neural_updates": 0,
            "production_activation": False,
            "fresh_holdout": False,
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, {"third-candidate": baseline["path"]})
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline-assets")
        protected = {
            str(baseline["path"] / "run.json"): BASE_SHA,
            str(workspace_root() / "latest.json"): sha256(workspace_root() / "latest.json"),
        }
        atomic_json(run.path / "reports/protected-assets.json", protected)
        checks, records = [], []
        for entry in baseline["cases"]:
            n, case_id = entry["page"], entry["case_id"]
            print(f"page {n:02d} {case_id}", flush=True)
            result = replay_page(baseline, n)
            output = run.path / f"rendered/pages/{n:02d}.pdf"
            save_pdf(result["pdf_bytes"], output)
            png = output.with_suffix(".png")
            if sha256(png) != sha256(baseline["path"] / f"rendered/pages/{n:02d}.png"):
                raise ValueError("PNG file byte identity failed despite exact raster replay")
            check = {
                **result["replay_checks"],
                "png_file_bytes_exact": True,
                "png_sha256": sha256(png),
            }
            checks.append(check)
            atomic_json(
                run.path / f"reports/{case_id}-replay.json",
                {
                    "checks": check,
                    "records": result["records"],
                    "parser": result["parser"],
                    "geometry": geometry_inventory(result["geometry"], result["records"]),
                    "f_decisions": result["f_decisions"],
                },
            )
            records.extend(result["records"])
        with PdfWriter() as writer:
            for n in range(1, 21):
                writer.append(run.path / f"rendered/pages/{n:02d}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        if sha256(run.path / "rendered/homework_solutions.pdf") != PDF_SHA:
            raise ValueError("Merged PDF does not match the final third-candidate bytes")
        for path, digest in protected.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Protected asset changed: {path}")
        maths = [row for row in records if row["context"] == "mathematics"]
        result = {
            "status": "exact-frozen-baseline-reproduced",
            "pages": len(checks),
            "visible_items": len(records),
            "math_items": len(maths),
            "f_count": sum(row["semantic"] == "f" for row in records),
            "pi_count": sum(row["semantic"] == "π" for row in maths),
            "math_source_types": dict(Counter(row["source_type"] for row in maths)),
            "all_page_pdf_png_bytes_exact": True,
            "merged_pdf_bytes_exact": True,
            "all_provenance_and_parser_exact": True,
            "protected_assets_unchanged": True,
            "new_ink_imported": False,
            "neural_updates": 0,
        }
        atomic_json(run.path / "reports/pages.json", checks)
        (run.path / "reports/BASELINE_REPLAY.md").write_text(
            "# Session O substitution baseline\n\n"
            f"Frozen input: `{BASE_RUN}`. All 20 page PDFs and PNGs, the merged PDF, "
            "every item source record, original f decisions and every mathematical parser trace "
            "replay exactly. Placed outline hashes are recorded per item.\n\n"
            "This is an exposed baseline reproduction, not new answer or style validation. "
            "No new symbol has been imported or admitted and no training occurred. "
            "Original assets and production latest remain unchanged.\n\n"
            "API: `load_baseline()` then `replay_page(b, n)`. For subsequent isolated symbols, "
            "supply pre-mutation `geometry`, records, image and parser to `validate_increment`, "
            "with an explicit target set. Independent source/ink review remains required.\n",
        )
        run.finish("held", result=result)
        verify_run(run.path)
        print(result, flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
