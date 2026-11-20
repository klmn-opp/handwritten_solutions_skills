"""Isolated real-source d/f/k/r corrections in the mathematical rendering path.

Replay the held combined candidate before postprocessing. The narrow r alternate
is replaced with the real primary r in MathText's existing primary slot. No source
expression edit, layout change, neural inference or training is performed.
"""

from __future__ import annotations

import copy
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from PIL import Image
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_revision import identity, personal_layout, trace
from research.corpus.math_glyph_completion import SELECTION, local_comparison, raster
from research.corpus.math_glyph_resolver import MathGlyphResolver
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.corpus.readability import placed_path
from research.cpu_glyph_feedback import (
    CpuGlyphFeedback,
    compiled_lineage,
    font_metrics,
    source_evidence,
)
from research.homework_benchmark.render import (
    accepted_scans,
    fallback_summary,
    pixel_boundary_checks,
    read,
    save_pdf,
    source_paths,
)
from research.math_feedback_candidate import paired_panel

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1"
FONT_SHA = "60466bbde4f39280838ac15f1121fc975cf841f42b581f74618eff50c8b2bcd9"
TARGETS = frozenset("dfkr")


def filled_path(item, renderer):
    """Filled outline or conservative round-cap envelope of a structure stroke."""
    if item[0] == "path":
        return placed_path(item, renderer)
    if item[0] != "stroke":
        raise ValueError("Unknown neighbor item kind; replacement cannot be audited")
    a, b, c, d, e, f = item[2]
    points = np.asarray(item[1]) @ np.array([[a, b], [c, d]]) + (e, f)
    radius = max(float(item[4]) / 2, 0.25)
    pieces = [
        MplPath.unit_circle().transformed(Affine2D().scale(radius).translate(*point))
        for point in points
    ]
    for left, right in zip(points[:-1], points[1:], strict=True):
        delta = right - left
        length = np.linalg.norm(delta)
        if not length:
            continue
        normal = np.array([-delta[1], delta[0]]) * radius / length
        pieces.append(
            MplPath(
                np.array(
                    [left + normal, right + normal, right - normal, left - normal, left + normal]
                ),
                closed=True,
            )
        )
    return MplPath.make_compound_path(*pieces)


def contacts(item_index, before_items, page, renderer, records):
    old = filled_path(before_items[item_index], renderer)
    new = filled_path(page.items[item_index], renderer)
    extent = new.get_extents()
    comparisons = []
    for row in records:
        j = row["item_index"]
        if j == item_index:
            continue
        other = filled_path(page.items[j], renderer)
        if not extent.expanded(1.02, 1.02).overlaps(other.get_extents()):
            continue
        old_other = filled_path(before_items[j], renderer)
        prior = bool(old.intersects_path(old_other, filled=True))
        after = bool(new.intersects_path(other, filled=True))
        comparisons.append(
            {
                "other_item": j,
                "semantic": row["semantic"],
                "context": row["context"],
                "before_contact": prior,
                "after_contact": after,
                "new_contact": after and not prior,
            }
        )
    return comparisons


class MathLatinFeedback:
    """Post-layout d/f/k/r only, rejecting unsupported/contacting replacements."""

    def __init__(self, font):
        if sha256(font) != FONT_SHA:
            raise ValueError("Math Latin feedback requires the pinned personal v1 font")
        self.feedback = CpuGlyphFeedback(font)
        self.decisions = []

    def apply(self, page, renderer, provenance):
        before_items = list(page.items)
        before = copy.deepcopy(provenance)
        records = self.feedback.apply(page, renderer, provenance, contexts={"mathematics"})
        self.decisions = []
        for old, row in zip(before, records, strict=True):
            if row["context"] != "mathematics" or row["semantic"] not in TARGETS:
                continue
            i = row["item_index"]
            decision = {
                "item_index": i,
                "semantic": old["semantic"],
                "latex": old["latex"],
                "original_variant": old.get("variant"),
                "original_bounds": old["bounds_px"],
                "candidate_bounds": row["bounds_px"],
                "source_font_sha256": FONT_SHA,
                "applied": page.items[i] != before_items[i],
                "status": "unchanged",
                "math_layout_positions_changed": False,
                "original_item_matrix": list(before_items[i][2]),
                "candidate_item_matrix": list(page.items[i][2]),
            }
            if decision["applied"]:
                box = row["bounds_px"]
                errors = []
                if min(box[0], box[1], page.width - box[2], page.height - box[3]) < 3:
                    errors.append("page boundary clearance below 3 px")
                old_box = old["bounds_px"]
                if (
                    row["semantic"] in {"d", "k"}
                    and min(
                        box[0] - old_box[0],
                        box[1] - old_box[1],
                        old_box[2] - box[2],
                        old_box[3] - box[3],
                    )
                    < -0.02
                ):
                    errors.append("d/k exceeds original envelope")
                if row["semantic"] == "k" and np.max(np.abs(np.array(box) - old_box)) > 0.02:
                    errors.append("k no longer preserves the existing fitted envelope")
                nearby = contacts(i, before_items, page, renderer, records)
                if any(r["new_contact"] for r in nearby):
                    errors.append("new neighbor ink contact")
                decision.update(neighbors=nearby, rejection_reasons=errors)
                if errors:
                    page.items[i] = before_items[i]
                    records[i] = copy.deepcopy(old)
                    records[i]["math_latin_feedback"] = {
                        "status": "explicit-original-glyph-fallback",
                        "reason": errors,
                        "attempted_variant": row.get("variant"),
                        "attempted_bounds": box,
                        "original_source_retained": True,
                    }
                    decision.update(applied=False, status="rejected-original-retained")
                else:
                    row["math_latin_feedback"] = {
                        "status": "research-candidate",
                        "source": row["cpu_feedback"],
                        "recorded_variant": row["variant"],
                        "author_acceptance": False,
                        "no_neural_training": True,
                        "neighbor_contact_checked": True,
                        "native_outline_matrix_preserved": row["semantic"] in {"f", "k", "r"},
                    }
                    if row["semantic"] in {"f", "r"}:
                        # Primary glyph uses the already reserved MathText primary slot.
                        row["allowed_bounds_px"] = [
                            min(box[0], old_box[0]),
                            min(box[1], old_box[1]),
                            max(box[2], old_box[2]),
                            max(box[3], old_box[3]),
                        ]
                        row["math_latin_feedback"]["primary_slot_bounds_px"] = box
                        row["math_latin_feedback"]["allowed_bounds_reason"] = (
                            "original box union native primary MathText outline at unchanged matrix"
                        )
                    decision.update(
                        status="accepted-for-visual-review",
                        variant=row["variant"],
                        treatment=row["cpu_feedback"],
                    )
            else:
                decision["reason"] = row.get("cpu_feedback_rejection", "already selected d1/f0/r0")
            self.decisions.append(decision)
        if identity(before) != identity(records):
            raise ValueError("Mathematical identity/order changed")
        for old, new, a, b in zip(before_items, page.items, before, records, strict=True):
            if a["context"] != "mathematics" or a["semantic"] not in TARGETS:
                if old != new or a != b:
                    raise ValueError("Unauthorized non-target content or source change")
        for decision in self.decisions:
            if decision["applied"] or decision["semantic"] == "r":
                decision["final_neighbors"] = contacts(
                    decision["item_index"], before_items, page, renderer, records
                )
                if any(r["new_contact"] for r in decision["final_neighbors"]):
                    raise ValueError("Accepted math glyph has a new final contact")
        return records


def target_crops(before, after, before_image, after_image, path, page_number):
    panels, report = [], []
    for old, new in zip(before, after, strict=True):
        if old["context"] != "mathematics" or old["semantic"] not in TARGETS:
            continue
        center = (np.asarray(old["bounds_px"][:2]) + old["bounds_px"][2:]) / 2
        # Native, same-coordinate region including neighbors; no target-specific warp.
        bounds = np.array([old["bounds_px"], new["bounds_px"]])
        x0, y0 = np.floor(bounds[:, :2].min(0) - [65, 28]).astype(int)
        x1, y1 = np.ceil(bounds[:, 2:].max(0) + [90, 28]).astype(int)
        box = [max(0, x0), max(0, y0), min(before_image.width, x1), min(before_image.height, y1)]
        images = before_image.crop(box), after_image.crop(box)
        name = f"{old['item_index']:04d}-{old['semantic']}.png"
        paired_panel(
            *images,
            path / name,
            f"Page {page_number} / math item {old['item_index']} / {old['semantic']}",
            ("Previous combined", "Math d/f/k/r source candidate"),
        )
        report.append(
            {
                "item_index": old["item_index"],
                "semantic": old["semantic"],
                "center_px": center.tolist(),
                "crop_box": list(map(int, box)),
                "old": old,
                "candidate": new,
                "panel": name,
            }
        )
        panels.append((page_number, old["item_index"], *images))
    atomic_json(path / "index.json", report)
    return panels


def launch():
    verify_run(COMBINED)
    root = workspace_root()
    cfg = RenderConfig.model_validate(read(COMBINED / "run.json")["config"]["render"])
    cases = read(COMBINED / "reports/cases.json")
    protected = {
        str(COMBINED / "run.json"): sha256(COMBINED / "run.json"),
        str(root / "latest.json"): sha256(root / "latest.json"),
    }
    run = Run.create(
        "math-latin-dfkr-feedback",
        config={
            "combined": str(COMBINED),
            "render": cfg.model_dump(),
            "contexts": ["mathematics"],
            "targets": sorted(TARGETS),
            "math_r_policy": "real r0 fills existing primary slot; r1 never stretched",
            "new_neural_updates": 0,
            "author_acceptance": False,
            "production_activation": False,
            "original_pages": [c["original_page"] for c in cases],
            "purpose": "Fix math d/f/k plus narrow r1 in fixed MathText primary slots",
        },
        seed=cfg.seed,
    )
    print(run.path, flush=True)
    try:
        paths = source_paths() + [
            Path(__file__),
            Path("research/cpu_glyph_feedback.py"),
            Path("research/author_feedback_revision.py"),
            Path("research/math_feedback_candidate.py"),
        ]
        for source in paths:
            dest = run.path / "logs/source" / source.resolve().relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        for version in [1, 3]:
            copy_snapshot(
                load_snapshot(COMBINED / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for name in ["historical-glyphs", "session-glyphs"]:
            shutil.copytree(COMBINED / f"solutions/{name}", run.path / f"solutions/{name}")
        profile = load_snapshot(run.path / "solutions/profile-v1")
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        atomic_json(run.path / "reports/source-evidence.json", source_evidence())
        atomic_json(run.path / "reports/font-lineage.json", compiled_lineage(profile.font))
        atomic_json(run.path / "reports/font-metrics.json", font_metrics(profile.font))
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        reports, outputs, panels = [], [], []
        for case in cases:
            number, case_id = case["original_page"], case["case_id"]
            print(f"page {number:02d}: {case_id}", flush=True)
            inputs = run.path / f"solutions/inputs/{case_id}"
            shutil.copytree(COMBINED / f"solutions/inputs/{case_id}", inputs)
            source = (inputs / "solution.md").read_text()
            job = Job.model_validate_json((inputs / "job.json").read_text())
            bg = job.background
            asset = inputs / bg.path if bg.path else None
            effective = cfg.model_copy(update={"paper": "plain" if asset else bg.kind})
            page, renderer, records, _, _ = personal_layout(
                source,
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
            before_image = raster(renderer.to_vector_pdf([page]))
            archive = Image.open(COMBINED / f"rendered/pages/{number:02d}.png").convert("RGB")
            if before_image.tobytes() != archive.tobytes():
                raise ValueError("Default text-only replay differs from sealed combined pixels")
            old_records = copy.deepcopy(records)
            old_trace = trace(renderer)
            feedback = MathLatinFeedback(profile.font)
            records = feedback.apply(page, renderer, records)
            if old_trace != trace(renderer):
                raise ValueError("Formula parser/layout trace changed")
            dest = run.path / f"rendered/pages/{number:02d}.pdf"
            image = save_pdf(renderer.to_vector_pdf([page]), dest)
            changed = {d["item_index"] for d in feedback.decisions if d["applied"]}
            checks = pixel_boundary_checks(before_image, image, old_records, records, changed)
            bounds, _ = ink_geometry(page, renderer)
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {
                    "before": old_records,
                    "candidate": records,
                    "parser_trace": old_trace,
                    "decisions": feedback.decisions,
                },
            )
            panels.extend(
                target_crops(
                    old_records,
                    records,
                    before_image,
                    image,
                    run.path / f"comparisons/targets/{number:02d}",
                    number,
                )
            )
            paired_panel(
                before_image,
                image,
                run.path / f"comparisons/pages/{number:02d}.png",
                f"Original page {number}",
                ("Previous combined", "Math Latin candidate"),
            )
            checks.update(
                original_page=number,
                case_id=case_id,
                replay_pixels_exact=True,
                parser_trace_exact=True,
                math_r_primary_slot_only=True,
                decisions=feedback.decisions,
                ink_bounds_px=bounds,
                fallbacks=fallback_summary(records),
            )
            reports.append(checks)
            atomic_json(run.path / f"reports/{case_id}-checks.json", checks)
            outputs.append(dest)
        with PdfWriter() as writer:
            for path in outputs:
                writer.append(path)
            writer.write(run.path / "rendered/homework_solutions.pdf")
        folder = run.path / "comparisons/all-targets"
        folder.mkdir()
        local_comparison(folder, panels, labels=("Previous combined", "Math d/f/k/r candidate"))
        atomic_json(run.path / "reports/cases.json", reports)
        atomic_json(run.path / "reports/protected.json", protected)
        summary = {
            "pages": len(reports),
            "original_pages": [r["original_page"] for r in reports],
            "targets": len(panels),
            "accepted": dict(
                Counter(d["semantic"] for r in reports for d in r["decisions"] if d["applied"])
            ),
            "rejected": [
                {"page": r["original_page"], **d}
                for r in reports
                for d in r["decisions"]
                if d["status"] == "rejected-original-retained"
            ],
            "default_text_replay_pixels_exact": True,
            "english_pixels_changed": 0,
            "math_semantics_and_parser_trace_unchanged": True,
            "math_r_audit": {
                "total": sum(d["semantic"] == "r" for r in reports for d in r["decisions"]),
                "primary_r0_unchanged": sum(
                    d["semantic"] == "r" and d["original_variant"] == 0 and not d["applied"]
                    for r in reports
                    for d in r["decisions"]
                ),
                "r1_to_primary_r0": sum(
                    d["semantic"] == "r" and d["applied"] for r in reports for d in r["decisions"]
                ),
                "meaning_or_unit_spaces_changed": False,
            },
            "new_neural_updates": 0,
            "author_acceptance": False,
            "status": "engineering checked; independent visual review pending",
        }
        atomic_json(run.path / "reports/summary.json", summary)
        if any(sha256(Path(p)) != h for p, h in protected.items()):
            raise ValueError("Protected asset changed")
        # Root must review native/full-page artifacts before this candidate is sealed.
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    launch()
