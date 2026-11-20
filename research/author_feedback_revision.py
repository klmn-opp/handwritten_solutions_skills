"""Integrate the exposed 20-page author revision without changing frozen results.

Replay original and math-only archives before accepting comparisons. The accepted
PDF is still a held research candidate; no production pointers or data are edited.
"""

import copy
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.paper import paper_image
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.corpus.context_composer import layout
from research.corpus.math_glyph_completion import SELECTION, local_comparison, raster
from research.corpus.math_glyph_resolver import MathGlyphResolver
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.corpus.readability import placed_path
from research.cpu_glyph_feedback import (
    TARGETS,
    CpuGlyphFeedback,
    compiled_lineage,
    font_metrics,
    layout_with_r_metrics,
    source_evidence,
)
from research.homework_benchmark.render import (
    BASELINE,
    LEGACY,
    accepted_scans,
    fallback_summary,
    read,
    save_pdf,
    source_paths,
)
from research.math_feedback_candidate import (
    AGGREGATE,
    MATH,
    RadicalClearancePolish,
    paired_panel,
    rewrite_fractions,
)

MATH_ONLY = "20261120T111953064576Z-math-author-feedback-clearance-fractions-6af6df03"


def identity(records):
    return [(r["context"], r["semantic"], r.get("latex")) for r in records]


def shape(item, renderer):
    """Placed geometry with only absolute translation removed, including stroke style."""
    if item[0] == "path":
        path = placed_path(item, renderer)
        vertices, codes = path.vertices, path.codes
    elif item[0] == "stroke":
        a, b, c, d, e, f = item[2]
        vertices = np.asarray(item[1]) @ np.array([[a, b], [c, d]]) + (e, f)
        codes = np.full(len(vertices), MplPath.LINETO)
        codes[0] = MplPath.MOVETO
    else:
        raise ValueError(f"Unsupported shape type: {item[0]}")
    origin = vertices[np.flatnonzero(codes != MplPath.CLOSEPOLY)[0]]
    return vertices - origin, codes, item[3:]


def shape_check(a_item, a_renderer, b_item, b_renderer):
    a, ac, ast = shape(a_item, a_renderer)
    b, bc, bst = shape(b_item, b_renderer)
    if a.shape != b.shape or not np.array_equal(ac, bc) or ast != bst:
        raise ValueError("Unapproved contour topology or stroke-style change")
    delta = float(np.abs(a - b).max())
    if delta > 0.0001:
        raise ValueError(f"Unapproved shape change: {delta:.8f}px")
    return delta


def provenance_signature(row):
    return {
        key: row.get(key)
        for key in (
            "semantic",
            "variant",
            "glyph_name",
            "actual_glyph_id",
            "source_type",
            "font_sha256",
            "personal_collected",
            "external_fallback",
        )
    }


def compare_text(original, original_renderer, old_records, final, final_renderer, records):
    before = [r for r in old_records if r["context"] == "text"]
    after = [r for r in records if r["context"] == "text"]
    if [r["semantic"] for r in before] != [r["semantic"] for r in after]:
        raise ValueError("English semantic sequence changed relative to original user-visible B")
    changed, positions, maximum = [], [], 0.0
    for ordinal, (a, b) in enumerate(zip(before, after, strict=True)):
        ai, bi = a["item_index"], b["item_index"]
        if a["semantic"] not in TARGETS:
            if provenance_signature(a) != provenance_signature(b):
                raise ValueError(
                    f"Non-target English source changed at ordinal {ordinal}: {a['semantic']}"
                )
            maximum = max(
                maximum,
                shape_check(original.items[ai], original_renderer, final.items[bi], final_renderer),
            )
        elif provenance_signature(a) != provenance_signature(b) or b.get("cpu_feedback"):
            changed.append(
                {
                    "text_ordinal": ordinal,
                    "original": provenance_signature(a),
                    "candidate": provenance_signature(b),
                    "treatment": b.get("cpu_feedback"),
                }
            )
        if a["bounds_px"] != b["bounds_px"]:
            positions.append(
                {
                    "text_ordinal": ordinal,
                    "semantic": b["semantic"],
                    "original_bounds": a["bounds_px"],
                    "candidate_bounds": b["bounds_px"],
                }
            )
    return {
        "english_identity_matches_original": True,
        "visible_text_items": len(before),
        "non_target_shapes_match_original": True,
        "maximum_shape_delta_px": maximum,
        "target_changes": changed,
        "position_changes": positions,
        "position_policy": "fraction line height and r bearing/advance may reflow unchanged prose",
    }


def compare_math_incremental(base_page, base_renderer, base_records, page, renderer, records):
    before = [r for r in base_records if r["context"] == "mathematics"]
    after = [r for r in records if r["context"] == "mathematics"]
    if identity(before) != identity(after):
        raise ValueError("CPU integration changed math semantics or item ordering")
    maximum, moves = 0.0, []
    for ordinal, (a, b) in enumerate(zip(before, after, strict=True)):
        if provenance_signature(a) != provenance_signature(b):
            raise ValueError(f"CPU integration changed mathematical source at ordinal {ordinal}")
        maximum = max(
            maximum,
            shape_check(
                base_page.items[a["item_index"]],
                base_renderer,
                page.items[b["item_index"]],
                renderer,
            ),
        )
        if a["bounds_px"] != b["bounds_px"]:
            moves.append(
                {
                    "math_ordinal": ordinal,
                    "semantic": b["semantic"],
                    "before": a["bounds_px"],
                    "after": b["bounds_px"],
                }
            )
    return {
        "math_identity_matches_math_only": True,
        "math_source_matches_math_only": True,
        "math_shapes_match_math_only": True,
        "items": len(before),
        "maximum_shape_delta_px": maximum,
        "position_changes": moves,
    }


def f_collision_check(page, renderer, before_items, records):
    """Reject newly touching filled outlines caused by a widened f."""
    result = []
    for row in records:
        if row["semantic"] != "f" or not row.get("cpu_feedback"):
            continue
        i = row["item_index"]
        new = placed_path(page.items[i], renderer)
        old = placed_path(before_items[i], renderer)
        comparisons, new_contacts = [], []
        x0, y0, x1, y1 = row["bounds_px"]
        for other in records:
            j = other["item_index"]
            if j == i or page.items[j][0] != "path":
                continue
            a, b, c, d = other["bounds_px"]
            if c < x0 - 1 or a > x1 + 1 or d < y0 - 1 or b > y1 + 1:
                continue
            path = placed_path(page.items[j], renderer)
            touched = bool(new.intersects_path(path, filled=True))
            prior = bool(old.intersects_path(placed_path(before_items[j], renderer), filled=True))
            comparisons.append(
                {
                    "other_item": j,
                    "semantic": other["semantic"],
                    "before_contact": prior,
                    "after_contact": touched,
                }
            )
            if touched and not prior:
                new_contacts.append(j)
        result.append(
            {
                "item_index": i,
                "before_bounds": row["cpu_feedback"]["original_bounds_px"],
                "after_bounds": row["bounds_px"],
                "nearby": comparisons,
                "new_contacts": new_contacts,
            }
        )
    return result


def personal_layout(
    source,
    title,
    profile,
    cfg,
    resolver,
    sessions,
    polish,
    background,
    asset,
    *,
    cpu=False,
    clearance=False,
):
    page, renderer, records = (layout_with_r_metrics if cpu else layout)(
        source, title, profile, cfg
    )
    if asset:
        page.background = paper_image(background, asset, (page.width, page.height))
        page.paper = None
    records = resolver.resolve(page, renderer, records, profile)
    records = sessions.apply(page, renderer, records, operator_page=False)
    records = polish.apply(page, renderer, records)
    collision = []
    if cpu:
        before_items = list(page.items)
        before_records = copy.deepcopy(records)
        records = CpuGlyphFeedback(profile.font).apply(page, renderer, records)
        if identity(before_records) != identity(records):
            raise ValueError("CPU glyph replacement changed recorder identity")
        for old, new, a, b in zip(before_items, page.items, before_records, records, strict=True):
            if a["context"] != "text" or a["semantic"] not in TARGETS:
                if old != new or a != b:
                    raise ValueError("CPU post-layout replacement altered unauthorized item")
        collision = f_collision_check(page, renderer, before_items, records)
    update = RadicalClearancePolish()
    if clearance:
        records = update.apply(page, renderer, records)
    return page, renderer, records, update.decisions, collision


def group_crops(
    before_records,
    before_trace,
    before_image,
    after_records,
    after_trace,
    after_image,
    output,
    number,
    labels,
):
    groups = []
    for trace in (before_trace, after_trace):
        groups.append(
            sorted(
                [(k, row) for k in ("formulas", "matrices") for row in trace[k]],
                key=lambda r: r[1]["item_range"],
            )
        )
    if len(groups[0]) != len(groups[1]):
        raise ValueError("Formula group cardinality changed")
    panels, index = [], []
    for n, ((kind, old), (other_kind, new)) in enumerate(zip(*groups, strict=True), 1):
        if kind != other_kind:
            raise ValueError("Formula/matrix group order changed")
        crops, boxes = [], []
        for group, rows, image in (
            (old, before_records, before_image),
            (new, after_records, after_image),
        ):
            start, stop = group["item_range"]
            bounds = np.asarray([r["bounds_px"] for r in rows[start:stop]])
            box = (
                *np.floor(bounds[:, :2].min(axis=0) - 12).astype(int),
                *np.ceil(bounds[:, 2:].max(axis=0) + 12).astype(int),
            )
            boxes.append(list(map(int, box)))
            crops.append(image.crop(box))
        path = output / f"{n:03d}-{kind}.png"
        paired_panel(*crops, path, f"Original page {number} / region {n}", labels)
        panels.append((number, n, *crops))
        index.append(
            {
                "original_parser": old,
                "candidate_parser": new,
                "crop_boxes": boxes,
                "panel": path.name,
            }
        )
    atomic_json(output / "index.json", index)
    return panels


def text_crops(old_rows, old_image, rows, image, output, number, em):
    old = [r for r in old_rows if r["context"] == "text"]
    new = [r for r in rows if r["context"] == "text"]
    selected = []
    # Every changed f receives neighbour context; also one fragment per other
    # target per page, selected by first occurrence rather than visual quality.
    seen = set()
    for i, row in enumerate(new):
        if row["semantic"] not in TARGETS:
            continue
        target = row["semantic"]
        if not (target == "f" and row.get("cpu_feedback")) and target in seen:
            continue
        seen.add(target)
        crops, spans = [], []
        for records, source in ((old, old_image), (new, image)):
            anchor = records[i]["bounds_px"]
            near = [
                r
                for r in records[max(0, i - 4) : i + 5]
                if abs((r["bounds_px"][1] + r["bounds_px"][3]) - (anchor[1] + anchor[3])) < em
            ]
            bounds = np.asarray([r["bounds_px"] for r in near])
            box = (
                *np.floor(bounds[:, :2].min(axis=0) - 10).astype(int),
                *np.ceil(bounds[:, 2:].max(axis=0) + 10).astype(int),
            )
            crops.append(source.crop(box))
            spans.append([r["item_index"] for r in near])
        path = output / f"{i:04d}-{target}.png"
        paired_panel(
            *crops,
            path,
            f"Page {number} / text ordinal {i} / {target}",
            ("Original user-visible B", "Combined source-traceable revision"),
        )
        selected.append({"text_ordinal": i, "target": target, "items": spans, "panel": path.name})
    atomic_json(output / "index.json", selected)
    return len(selected)


def trace(renderer):
    return copy.deepcopy({"formulas": renderer.formulas, "matrices": renderer.matrices})


def main():
    root = workspace_root()
    aggregate, math_only, frozen, legacy = (
        root / "runs" / key for key in (AGGREGATE, MATH_ONLY, BASELINE, LEGACY)
    )
    for path in (aggregate, math_only, frozen, legacy):
        verify_run(path)
    cfg = RenderConfig.model_validate(read(frozen / "run.json")["config"]["render_config"])
    protected = {
        str(root / "latest.json"): sha256(root / "latest.json"),
        **{
            str(p / "run.json"): sha256(p / "run.json")
            for p in (aggregate, math_only, frozen, legacy)
        },
    }
    run = Run.create(
        "author-feedback-combined-20-pages",
        seed=cfg.seed,
        config={
            "argv": sys.argv,
            "original_user_visible_B": str(aggregate),
            "math_only_comparison": str(math_only),
            "render": cfg.model_dump(mode="json"),
            "targets": sorted(TARGETS),
            "new_neural_updates": 0,
            "fresh_holdout": False,
            "goal": (
                "37 equivalent stacked fractions, root clearance, recorded 0/r/d/k/f corrections"
            ),
            "production_activation": False,
            "author_style_acceptance": False,
        },
    )
    print(run.path, flush=True)
    try:
        atomic_json(run.path / "reports/protected-assets.json", protected)
        paths = source_paths() + [
            Path(__file__),
            Path("research/cpu_glyph_feedback.py"),
            Path("research/math_feedback_candidate.py"),
        ]
        for path in paths:
            target = run.path / "logs/source" / path.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(frozen / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for name, source in (("session-glyphs", frozen), ("historical-glyphs", legacy)):
            shutil.copytree(source / f"solutions/{name}", run.path / f"solutions/{name}")
        profile = load_snapshot(run.path / "solutions/profile-v1")
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        atomic_json(run.path / "reports/cpu-font-lineage.json", compiled_lineage(profile.font))
        atomic_json(run.path / "reports/cpu-source-evidence.json", source_evidence())
        atomic_json(run.path / "reports/cpu-font-metrics.json", font_metrics(profile.font))
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        all_reports, transformations, old_panels, cpu_panels = [], [], [], []
        outputs, math_outputs = [], []
        for case in read(aggregate / "reports/cases.json"):
            number = case.get("output_page")
            if number is None:
                continue
            case_id, previous = case["id"], Path(case["source_run"])
            print(f"page {number:02d}: {case_id}", flush=True)
            original = (aggregate / f"solutions/{case_id}/answer.md").read_text()
            revised, edits = rewrite_fractions(case_id, original)
            if revised != (math_only / f"solutions/inputs/{case_id}/solution.md").read_text():
                raise ValueError("Revised source differs from recorded math-only comparison")
            unit_slashes = []
            for expression in MATH.finditer(revised):
                stripped = re.sub(r"\\mathrm\{[^{}]*\}", "", expression[0])
                if "/" in stripped:
                    raise ValueError(f"Non-unit mathematical slash remains: {expression[0]}")
                if "/" in expression[0]:
                    unit_slashes.append(expression[0])
            inputs = run.path / f"solutions/inputs/{case_id}"
            shutil.copytree(math_only / f"solutions/inputs/{case_id}", inputs)
            job = Job.model_validate_json((inputs / "job.json").read_text())
            job = job.model_copy(
                update={
                    "generated_by": "author-feedback-combined/1; research postprocessor required"
                }
            )
            atomic_json(inputs / "job.json", job.model_dump(mode="json"))
            bg = job.background
            asset = inputs / bg.path if bg.path else None
            effective = cfg.model_copy(update={"paper": "plain" if asset else bg.kind})
            arguments = (case_id, profile, effective, resolver, sessions, polish, bg, asset)
            old_page, old_renderer, old_records, _, _ = personal_layout(original, *arguments)
            old_image = raster(old_renderer.to_vector_pdf([old_page]))
            archive_old = Image.open(previous / f"rendered/pages/candidate/{case_id}.png").convert(
                "RGB"
            )
            if old_image.tobytes() != archive_old.tobytes():
                raise ValueError("Original B replay differs from the user-visible archive")
            m_page, m_renderer, m_records, _, _ = personal_layout(
                revised, *arguments, clearance=True
            )
            m_image = raster(m_renderer.to_vector_pdf([m_page]))
            m_file = next((math_only / "rendered/pages/B").glob(f"{number:02d}-*.pdf"))
            archive_m = Image.open(m_file.with_suffix(".png")).convert("RGB")
            if m_image.tobytes() != archive_m.tobytes():
                raise ValueError("Math-only replay differs from its immutable archive")
            page, renderer, records, root_decisions, contacts = personal_layout(
                revised, *arguments, cpu=True, clearance=True
            )
            dest = run.path / f"rendered/pages/{number:02d}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            image = save_pdf(renderer.to_vector_pdf([page]), dest)
            # Preserve output/provenance before any rejection, for diagnosis.
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {
                    "original_B": old_records,
                    "math_only": m_records,
                    "combined": records,
                    "parser_structure_trace": trace(renderer),
                    "root_decisions": root_decisions,
                    "f_neighbour_contact": contacts,
                },
            )
            if any(row["new_contacts"] for row in contacts):
                raise ValueError(f"Widened f touches a new neighbour on page {number}")
            text_check = compare_text(old_page, old_renderer, old_records, page, renderer, records)
            math_check = compare_math_incremental(
                m_page, m_renderer, m_records, page, renderer, records
            )
            bounds, _ = ink_geometry(page, renderer)
            if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
                raise ValueError("Combined page crosses its boundary")
            non_target_cpu = [
                r for r in records if r.get("cpu_feedback") and r["semantic"] not in TARGETS
            ]
            if non_target_cpu:
                raise ValueError("Unexplained CPU target change")
            old_panels.extend(
                group_crops(
                    old_records,
                    trace(old_renderer),
                    old_image,
                    records,
                    trace(renderer),
                    image,
                    run.path / f"comparisons/formulas/original/{number:02d}",
                    number,
                    ("Original user-visible B", "Combined revision"),
                )
            )
            cpu_panels.extend(
                group_crops(
                    m_records,
                    trace(m_renderer),
                    m_image,
                    records,
                    trace(renderer),
                    image,
                    run.path / f"comparisons/formulas/cpu-increment/{number:02d}",
                    number,
                    ("Math-only revision", "Combined with CPU corrections"),
                )
            )
            fragments = text_crops(
                old_records,
                old_image,
                records,
                image,
                run.path / f"comparisons/text/{number:02d}",
                number,
                renderer.cfg.body_size,
            )
            for name, before, labels in (
                (
                    "original-to-combined",
                    old_image,
                    ("Original user-visible B", "Combined revision"),
                ),
                ("math-only-to-combined", m_image, ("Math-only revision", "Combined CPU + math")),
            ):
                paired_panel(
                    before,
                    image,
                    run.path / f"comparisons/{name}/{number:02d}.png",
                    f"Original page {number}",
                    labels,
                )
            report = {
                "original_page": number,
                "case_id": case_id,
                "source_revisions": len(edits),
                "original_B_replay_pixels_exact": True,
                "math_only_replay_pixels_exact": True,
                "english": text_check,
                "math_cpu_increment": math_check,
                "f_neighbour_contact": contacts,
                "root_decisions": root_decisions,
                "ink_bounds_px": bounds,
                "remaining_unit_slashes": unit_slashes,
                "text_crops": fragments,
                "fallbacks": fallback_summary(records),
                "changed_pixels_from_original": int(
                    np.any(np.asarray(old_image) != np.asarray(image), axis=2).sum()
                ),
                "changed_pixels_from_math_only": int(
                    np.any(np.asarray(m_image) != np.asarray(image), axis=2).sum()
                ),
                "status": "engineering_pass_independent_visual_review_pending",
            }
            atomic_json(run.path / f"reports/{case_id}-checks.json", report)
            all_reports.append(report)
            transformations.append(
                {
                    "case_id": case_id,
                    "original_page": number,
                    "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "revised_sha256": hashlib.sha256(revised.encode()).hexdigest(),
                    "revisions": edits,
                    "prose_exact": True,
                    "unit_slashes_preserved": unit_slashes,
                }
            )
            outputs.append(dest)
            math_outputs.append(m_file)
        for files, target in (
            (outputs, run.path / "rendered/homework_solutions.pdf"),
            (math_outputs, run.path / "comparisons/math-only-20.pdf"),
        ):
            with PdfWriter() as writer:
                for file in files:
                    writer.append(file)
                writer.write(target)
        shutil.copyfile(
            aggregate / "rendered/homework_solutions.pdf", run.path / "comparisons/original-B20.pdf"
        )
        for name, panels, labels in (
            ("all-original-formulas", old_panels, ("Original B", "Combined revision")),
            ("all-cpu-increment-formulas", cpu_panels, ("Math-only", "Combined CPU + math")),
        ):
            folder = run.path / "comparisons" / name
            folder.mkdir()
            local_comparison(folder, panels, labels=labels)
        atomic_json(run.path / "reports/cases.json", all_reports)
        atomic_json(run.path / "reports/source-transformations.json", transformations)
        assert len(all_reports) == 20 and all(sha256(Path(p)) == h for p, h in protected.items())
        result = {
            "pages": 20,
            "source_fraction_revisions": sum(len(r["revisions"]) for r in transformations),
            "cpu_changes": dict(
                Counter(
                    r["candidate"]["semantic"]
                    for c in all_reports
                    for r in c["english"]["target_changes"]
                )
            ),
            "all_original_english_sequences_preserved": True,
            "all_non_target_english_shapes_preserved": True,
            "math_cpu_increment_shapes_preserved": True,
            "new_f_contacts": 0,
            "font_fallbacks": sum(len(c["fallbacks"]["font_fallbacks"]) for c in all_reports),
            "neural_updates": 0,
            "independent_holdout": False,
            "author_style_acceptance": False,
            "warning": (
                "positions and pixels change from authorized reflow; no zero-pixel regression claim"
            ),
        }
        run.finish("held", result=result)
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
