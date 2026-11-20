"""Held, source-preserving historical ∈ / two-integral contour experiment.

Original RGB is never cleaned or rewritten. The two-integral source retains its
actual label ``\\int\\int``; U+222C is an explicit structural-equivalence mapping.
No source truth, production Profile, renderer, or frozen run is changed.
"""

import copy
import hashlib
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_revision import identity, personal_layout, trace
from research.corpus.math_glyph_completion import SELECTION
from research.corpus.math_glyph_resolver import MathGlyphResolver, ScanGlyph
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.corpus.readability import placed_path
from research.homework_benchmark.render import (
    accepted_scans,
    fallback_summary,
    read,
    save_pdf,
    source_paths,
)
from research.math_feedback_candidate import paired_panel

BASE = "20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1"
REVIEW = "20261120T112651209228Z-missing-math-historical-review-88800d28"
TARGETS = frozenset(("∈", "∬"))
LEVEL = 180
PAGES = {6: "m1-ca-02-ellipse-integral", 18: "m1-st-06-exponential-scale"}
EXPECTED_INTEGRALS = [(15, 17, 50, 131, 705), (65, 18, 42, 148, 722)]


def connected_ink(mask):
    """Eight-connected binary components; no morphology or pixel changes."""
    labels = np.zeros(mask.shape, dtype=np.uint16)
    components = []
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if labels[y, x]:
            continue
        number = len(components) + 1
        pending = deque([(int(y), int(x))])
        labels[y, x] = number
        pixels = []
        while pending:
            cy, cx = pending.popleft()
            pixels.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = number
                        pending.append((ny, nx))
        points = np.asarray(pixels)
        y0, x0 = points.min(axis=0)
        y1, x1 = points.max(axis=0) + 1
        components.append(
            {
                "component": number,
                "bounds_xywh": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "area_px": len(pixels),
            }
        )
    return labels, components


def component_outline(grey, selected_ids):
    """Select existing full luminance contours, without erasing adjacent raw ink."""
    labels, components = connected_ink(grey < LEVEL)
    vertices, codes = contour_generator(z=-grey, fill_type="OuterCode").filled(-LEVEL, 1)
    yy, xx = np.indices(grey.shape)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    retained_v, retained_c, audit = [], [], []
    selected_contour_mask = np.zeros(grey.size, dtype=bool)
    for number, (v, c) in enumerate(zip(vertices, codes, strict=True), 1):
        interior = MplPath(v, c).contains_points(xy)
        ids = sorted(map(int, set(labels.ravel()[interior]) - {0}))
        if len(ids) != 1:
            raise ValueError(f"Ambiguous original contour/component relation: {ids}")
        selected = ids[0] in selected_ids
        points = v[c != MplPath.CLOSEPOLY]
        audit.append(
            {
                "contour": number,
                "component": ids[0],
                "selected": selected,
                "vertex_count": len(v),
                "bounds": [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()],
                "vertices_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
                "codes_sha256": hashlib.sha256(c.tobytes()).hexdigest(),
            }
        )
        if selected:
            if (
                points[:, 0].min() <= 0
                or points[:, 1].min() <= 0
                or points[:, 0].max() >= grey.shape[1] - 1
                or points[:, 1].max() >= grey.shape[0] - 1
            ):
                raise ValueError("Selected original integral contour touches crop boundary")
            retained_v.append(v)
            retained_c.append(c)
            selected_contour_mask |= interior
    selected_pixels = np.isin(labels, list(selected_ids))
    if not np.array_equal(
        selected_contour_mask.reshape(grey.shape) & (grey < LEVEL), selected_pixels
    ):
        raise ValueError("Contour selection removes original integral pixels or includes A")
    evidence = {
        "level": LEVEL,
        "components": components,
        "contours": audit,
        "selected_components": sorted(selected_ids),
        "selected_component_ink_pixels": int(selected_pixels.sum()),
        "all_selected_component_dark_pixels_preserved": True,
        "no_excluded_component_dark_pixels_in_selected_contours": True,
        "geometry": "original contourpy luminance-isocontours; no smoothing/morphology/filling",
    }
    return (
        Outline(np.vstack(retained_v), np.concatenate(retained_c), stroke=False),
        labels,
        evidence,
    )


def source_panel(raw, labels, evidence, outline, destination):
    """Diagnostic illustrations are separate from all original RGB source files."""
    scale, pad, header = 3, 18, 65
    selected = set(evidence["selected_components"])
    diagnostic = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for cid in range(1, int(labels.max()) + 1):
        diagnostic[labels == cid] = (25, 110, 35) if cid in selected else (205, 35, 35)
    component_image = Image.fromarray(diagnostic).resize(
        (raw.width * scale, raw.height * scale), Image.Resampling.NEAREST
    )
    contour_image = Image.new("RGB", component_image.size, "white")
    draw = ImageDraw.Draw(contour_image)
    start = None
    previous = None
    for point, code in zip(outline.verts * scale, outline.codes, strict=True):
        point = tuple(point)
        if code == MplPath.MOVETO:
            start = previous = point
        elif code == MplPath.CLOSEPOLY:
            draw.line((previous, start), fill="black", width=1)
            previous = start
        else:
            draw.line((previous, point), fill="black", width=1)
            previous = point
    column = max(255, component_image.width)
    panel = Image.new(
        "RGB", (3 * (column + pad) + pad, header + component_image.height + pad), "white"
    )
    draw = ImageDraw.Draw(panel)
    for i, (title, picture) in enumerate(
        [
            (
                "Original RGB / 3x nearest\nNo raw edits",
                raw.resize(component_image.size, Image.Resampling.NEAREST),
            ),
            ("Original components / 3x\nGreen selected; red excluded", component_image),
            ("Selected original contours / 3x\nOutline diagnostic only", contour_image),
        ]
    ):
        x = pad + i * (column + pad)
        draw.multiline_text((x, 12), title, fill="black", spacing=5)
        panel.paste(picture, (x, header))
    panel.save(destination)


def prepare_sources(run, review):
    destination = run / "solutions/new-glyphs"
    destination.mkdir(parents=True)
    originals = run / "solutions/original-evidence"
    originals.mkdir()
    extracted = read(review / "reports/reviewed-extractions.json")
    atomic_json(originals / "prior-reviewed-extractions.json", extracted)
    entries, held = [], []
    for prior in extracted:
        raw_path = review / prior["crop"]
        if sha256(raw_path) != prior["sha256"]:
            raise ValueError("Original crop changed")
        source_pdf = review / prior["source_pdf"]
        pdf_copy = originals / source_pdf.name
        if not pdf_copy.exists():
            shutil.copyfile(source_pdf, pdf_copy)
        if sha256(pdf_copy) != prior["source_pdf_sha256"]:
            raise ValueError("Original PDF copy is not exact")
        parent_path = review / f"solutions/crops/{prior['parent_region']}.png"
        parent_copy = originals / parent_path.name
        if not parent_copy.exists():
            shutil.copyfile(parent_path, parent_copy)
        if sha256(parent_copy) != prior["parent_crop_sha256"]:
            raise ValueError("Original page changed")
        crop_path = destination / f"{prior['id']}.png"
        shutil.copyfile(raw_path, crop_path)
        raw = Image.open(crop_path).convert("RGB")
        if (
            raw.tobytes()
            != Image.open(parent_copy).crop(prior["box_in_parent_300dpi"]).convert("RGB").tobytes()
        ):
            raise ValueError("Source crop differs from original page pixels")
        grey = np.asarray(raw.convert("L"), dtype=float)
        double = prior["id"] == "double-integral-context"
        edge = grey < LEVEL
        if not double and any(a.any() for a in (edge[0], edge[-1], edge[:, 0], edge[:, -1])):
            held.append(
                {
                    "source": prior,
                    "crop": crop_path.name,
                    "crop_sha256": sha256(crop_path),
                    "status": "held-boundary-ink",
                    "reason": "Original RGB has dark edge pixels; no clean outline accepted",
                    "dark_edge_pixels_top_bottom_left_right": [
                        int(a.sum()) for a in (edge[0], edge[-1], edge[:, 0], edge[:, -1])
                    ],
                }
            )
            continue
        selected_ids = {1, 2} if double else {1}
        outline, labels, evidence = component_outline(grey, selected_ids)
        if double:
            observed = [
                tuple(c["bounds_xywh"] + [c["area_px"]]) for c in evidence["components"][:2]
            ]
            if observed != EXPECTED_INTEGRALS or len(evidence["components"]) != 3:
                raise ValueError("Reviewed double-integral component identity changed")
        elif len(evidence["components"]) != 1:
            raise ValueError("Membership crop is not one complete original component")
        stability = {}
        for level in (130, 160, 180, 200):
            _, components = connected_ink(grey < level)
            stability[str(level)] = components
            if len(components) != (3 if double else 1):
                raise ValueError("Component separation is unstable across luminance levels")
        evidence["threshold_diagnostic_components"] = stability
        evidence["original_rgb_sha256"] = sha256(crop_path)
        evidence["source_label"] = prior["actual_visible_label"]
        evidence["excluded_component_reason"] = (
            "component 3 is neighbouring lower-limit A; raw RGB remains intact" if double else None
        )
        np.savez_compressed(destination / f"{prior['id']}-components.npz", labels=labels)
        contour_path = destination / f"{prior['id']}-contours.npz"
        np.savez_compressed(contour_path, vertices=outline.verts, codes=outline.codes)
        evidence_path = destination / f"{prior['id']}-selection.json"
        atomic_json(evidence_path, evidence)
        source = {
            **prior,
            "id": f"historical-{prior['id']}",
            "character": "∬" if double else "∈",
            "crop": crop_path.name,
            "crop_sha256": sha256(crop_path),
            "contour_luminance": LEVEL,
            "review_status": "agent-accepted",
            "clean_isolated_ink": not double,
            "original_context_contains_neighbour": double,
            "reviewed_component_selection": evidence_path.name,
            "reviewed_component_selection_sha256": sha256(evidence_path),
            "contour_archive": contour_path.name,
            "contour_archive_sha256": sha256(contour_path),
            "source_pdf": str(pdf_copy.relative_to(run)),
            "original_page": str(parent_copy.relative_to(run)),
            "independent_review": (
                "parent Agent accepted raw labels/bounds; approved isolated experiment"
            ),
            "source_label_policy": (
                "actual_visible_label retained verbatim; prior review record preserved"
            ),
            "mapping": (
                "two original adjacent integral signs mapped to equivalent double-integral "
                "operator U+222C; not isolated-character truth"
                if double
                else "original membership sign mapped to U+2208"
            ),
            "applied_to_pages": [6] if double else [18],
        }
        if not double:
            conventional = ScanGlyph.load(destination, source)
            if not np.array_equal(conventional.outline.verts, outline.verts) or not np.array_equal(
                conventional.outline.codes, outline.codes
            ):
                raise ValueError("Membership extraction differs from existing ScanGlyph loader")
        source_panel(
            raw, labels, evidence, outline, run / f"comparisons/{prior['id']}-source-to-contour.png"
        )
        entries.append(source)
    manifest = {
        "schema": "historical-math-selected-contours/1",
        "sources": entries,
        "held_sources": held,
        "selected": {"∈": "historical-membership-left", "∬": "historical-double-integral-context"},
        "selection_reason": (
            "left membership has complete enclosed ink; right original retained but held "
            "because of boundary ink; no seed selection"
        ),
        "review_run": str(review),
        "review_run_sha256": sha256(review / "run.json"),
        "production": False,
        "neural_training": False,
    }
    atomic_json(destination / "manifest.json", manifest)
    return load_sources(destination)


def load_sources(directory):
    """Load immutable reviewed outlines, including explicit component provenance."""
    manifest = read(directory / "manifest.json")
    glyphs = {}
    for source in manifest["sources"]:
        for path_key, hash_key in (
            ("crop", "crop_sha256"),
            ("contour_archive", "contour_archive_sha256"),
            ("reviewed_component_selection", "reviewed_component_selection_sha256"),
        ):
            if sha256(directory / source[path_key]) != source[hash_key]:
                raise ValueError("Reviewed source/contour evidence changed")
        with np.load(directory / source["contour_archive"], allow_pickle=False) as data:
            outline = Outline(data["vertices"], data["codes"], stroke=False)
        if manifest["selected"].get(source["character"]) == source["id"]:
            glyphs[source["character"]] = ScanGlyph(source, outline)
    if set(glyphs) != TARGETS:
        raise ValueError("Incomplete accepted historical target map")
    return glyphs


class HistoricalMathFeedback:
    def __init__(self, glyphs, strokes):
        self.resolver = MathGlyphResolver(glyphs, strokes, selected=set(TARGETS))

    def apply(self, page, renderer, records, profile):
        targets = [
            r
            for r in records
            if r["context"] == "mathematics"
            and r["semantic"] in TARGETS
            and r["source_type"] in {"external-font", "inherited-font"}
        ]
        resolved = self.resolver.resolve(page, renderer, targets, profile)
        replacements = {}
        for old, row in zip(targets, resolved, strict=True):
            if row["source_type"] != "personal-scan-outline":
                raise ValueError("Historical target unexpectedly failed its source fit")
            row.pop("fallback_reason", None)
            row["historical_math_feedback"] = {
                "version": 1,
                "original": copy.deepcopy(old),
                "selected_contours": row["scan_source"]["reviewed_component_selection"],
                "original_source_label": row["scan_source"]["actual_visible_label"],
                "mapped_operator": row["semantic"],
                "structure_equivalent_mapping": row["semantic"] == "∬",
                "no_other_items_or_parser_structure_changed": True,
            }
            replacements[row["item_index"]] = row
        return [replacements.get(r["item_index"], r) for r in records]


def unchanged_checks(page, renderer, before_items, before_records, records, old_image, image):
    if identity(before_records) != identity(records) or len(before_items) != len(page.items):
        raise ValueError("Mathematical/source identity or item cardinality changed")
    indices = {r["item_index"] for r in records if r.get("historical_math_feedback")}
    allowed = np.zeros((image.height, image.width), dtype=bool)
    for i, (a, b) in enumerate(zip(before_records, records, strict=True)):
        if i not in indices:
            if a != b or before_items[i] != page.items[i]:
                raise ValueError(f"Non-target item/provenance changed: {i}")
            continue
        old_box = np.asarray(a["bounds_px"])
        box = np.asarray(b["bounds_px"])
        if np.any(box[:2] < old_box[:2] - 1e-3) or np.any(box[2:] > old_box[2:] + 1e-3):
            raise ValueError("New source ink exceeds old parser slot")
        x0, y0 = np.floor(old_box[:2] - 3).astype(int)
        x1, y1 = np.ceil(old_box[2:] + 3).astype(int)
        allowed[max(0, y0) : min(image.height, y1), max(0, x0) : min(image.width, x1)] = True
    diff = np.any(np.asarray(old_image) != np.asarray(image), axis=2)
    if (diff & ~allowed).any():
        raise ValueError("Pixel change outside original target slots plus 3px antialias margin")
    english_pixels = np.zeros_like(allowed)
    for r in records:
        if r["context"] == "text":
            x0, y0, x1, y1 = np.asarray(r["bounds_px"])
            english_pixels[
                max(0, int(y0) - 2) : min(image.height, int(np.ceil(y1)) + 2),
                max(0, int(x0) - 2) : min(image.width, int(np.ceil(x1)) + 2),
            ] = True
    if (english_pixels & diff).any():
        raise ValueError("English raster pixels changed")
    return {
        "target_items": sorted(indices),
        "unchanged_non_target_items": len(records) - len(indices),
        "identity_trace_exact": True,
        "identity_trace_is_not_independent_answer_verification": True,
        "non_target_items_and_provenance_exact": True,
        "english_pixels_exact": True,
        "changed_pixels": int(diff.sum()),
        "changed_pixels_outside_target_slots": 0,
        "target_slots_margin_px": 3,
    }


def launch():
    root = workspace_root()
    base, review = root / "runs" / BASE, root / "runs" / REVIEW
    verify_run(base)
    verify_run(review)
    cfg = RenderConfig.model_validate(read(base / "run.json")["config"]["render"])
    run = Run.create(
        "historical-membership-double-integral-candidate",
        seed=cfg.seed,
        config={
            "argv": sys.argv,
            "base_run": str(base),
            "source_review_run": str(review),
            "render": cfg.model_dump(mode="json"),
            "original_pages": list(PAGES),
            "hypothesis": (
                "reviewed real membership / adjacent integral contours replace two external-font "
                "roles without touching other ink"
            ),
            "contour_luminance": LEVEL,
            "neural_updates": 0,
            "production_activation": False,
            "holdout": False,
            "status": "isolated candidate; final math-Latin chain not yet included",
        },
    )
    print(run.path, flush=True)
    try:
        protected = {
            str(p): sha256(p)
            for p in (root / "latest.json", base / "run.json", review / "run.json")
        }
        atomic_json(run.path / "reports/protected-assets.json", protected)
        paths = source_paths() + [
            Path(__file__),
            Path("research/author_feedback_revision.py"),
            Path("research/cpu_glyph_feedback.py"),
            Path("research/math_feedback_candidate.py"),
        ]
        for path in paths:
            destination = run.path / "logs/source" / path.resolve().relative_to(Path.cwd())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(base / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for kind in ("historical-glyphs", "session-glyphs"):
            shutil.copytree(base / "solutions" / kind, run.path / "solutions" / kind)
        glyphs = prepare_sources(run.path, review)
        profile = load_snapshot(run.path / "solutions/profile-v1")
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        historical = HistoricalMathFeedback(glyphs, strokes)
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        outputs, checks, delivery = [], [], []
        for number, case_id in PAGES.items():
            print(f"page {number}: {case_id}", flush=True)
            inputs = run.path / f"solutions/inputs/{case_id}"
            shutil.copytree(base / f"solutions/inputs/{case_id}", inputs)
            source = (inputs / "solution.md").read_text()
            job = Job.model_validate_json((inputs / "job.json").read_text())
            bg = job.background
            asset = inputs / bg.path if bg.path else None
            effective = cfg.model_copy(update={"paper": "plain" if asset else bg.kind})
            page, renderer, records, root_decisions, contacts = personal_layout(
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
            before_items, before_records = list(page.items), copy.deepcopy(records)
            parser_before = trace(renderer)
            # Retain every non-target resolved path, including outlines under mutable renderer keys.
            before_paths = {
                i: (
                    placed_path(item, renderer).vertices.copy(),
                    placed_path(item, renderer).codes.copy(),
                )
                for i, item in enumerate(page.items)
                if item[0] == "path"
            }
            baseline_pdf = run.path / f"comparisons/baseline-{number:02d}.pdf"
            old_image = save_pdf(renderer.to_vector_pdf([page]), baseline_pdf)
            archived = Image.open(base / f"rendered/pages/{number:02d}.png").convert("RGB")
            if old_image.tobytes() != archived.tobytes():
                raise ValueError("Combined baseline replay is not pixel exact")
            records = historical.apply(page, renderer, records, profile)
            destination = run.path / f"rendered/pages/{number:02d}.pdf"
            image = save_pdf(renderer.to_vector_pdf([page]), destination)
            check = unchanged_checks(
                page, renderer, before_items, before_records, records, old_image, image
            )
            if trace(renderer) != parser_before:
                raise ValueError("Formula parser trace changed")
            for i, (vertices, codes) in before_paths.items():
                if i not in check["target_items"]:
                    after = placed_path(page.items[i], renderer)
                    if not np.array_equal(vertices, after.vertices) or not np.array_equal(
                        codes, after.codes
                    ):
                        raise ValueError(
                            "Non-target outline changed through renderer key collision"
                        )
            bounds, _ = ink_geometry(page, renderer)
            if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
                raise ValueError("New contour exceeds page boundary")
            if len(check["target_items"]) != 1:
                raise ValueError("Expected exactly one reviewed target on each selected page")
            for i in check["target_items"]:
                x0, y0, x1, y1 = before_records[i]["bounds_px"]
                box = (int(x0) - 50, int(y0) - 40, int(np.ceil(x1)) + 180, int(np.ceil(y1)) + 65)
                paired_panel(
                    old_image.crop(box),
                    image.crop(box),
                    run.path / f"comparisons/{number:02d}-target-{i}.png",
                    f"Page {number} / item {i} / original source label retained",
                    ("Combined baseline", "Historical contour candidate"),
                )
            paired_panel(
                old_image,
                image,
                run.path / f"comparisons/{number:02d}-full-page.png",
                f"Original page {number}; same source, layout, and background",
                ("Combined baseline", "Historical contour candidate"),
            )
            provenance_path = run.path / f"reports/{case_id}-provenance.json"
            atomic_json(
                provenance_path,
                {
                    "baseline": before_records,
                    "candidate": records,
                    "parser_structure_trace": parser_before,
                    "root_decisions": root_decisions,
                    "f_neighbour_contact": contacts,
                },
            )
            check.update(
                original_page=number,
                case_id=case_id,
                baseline_replay_pixels_exact=True,
                ink_bounds_px=bounds,
                before_fallbacks=fallback_summary(before_records),
                after_fallbacks=fallback_summary(records),
                source_markdown_sha256=sha256(inputs / "solution.md"),
                source_job_sha256=sha256(inputs / "job.json"),
                status="engineering_pass_independent_visual_review_pending",
            )
            atomic_json(run.path / f"reports/{case_id}-checks.json", check)
            checks.append(check)
            outputs.append(destination)
            delivery.append(
                {
                    "page": number,
                    "case_id": case_id,
                    "pdf": str(destination.relative_to(run.path)),
                    "png": str(destination.with_suffix(".png").relative_to(run.path)),
                    "provenance_report": str(provenance_path.relative_to(run.path)),
                    "provenance_key": "candidate",
                }
            )
        with PdfWriter() as writer:
            for output in outputs:
                writer.append(output)
            writer.write(run.path / "rendered/homework_solutions.pdf")
        atomic_json(
            run.path / "reports/delivery-pages.json",
            {
                "base_run": str(base),
                "base_run_sha256": sha256(base / "run.json"),
                "pages": delivery,
                "final_integrated_delivery": False,
                "warning": "Requires approved math-Latin rebase before final twenty-page assembly",
            },
        )
        if any(sha256(Path(p)) != h for p, h in protected.items()):
            raise ValueError("Protected source or production pointer changed")
        run.finish(
            "held",
            result={
                "pages": 2,
                "targets_replaced": 2,
                "raw_source_pixels_edited": False,
                "prior_labels_changed": False,
                "non_target_pixels_exact": True,
                "neural_updates": 0,
                "production_activation": False,
                "visual_review": "pending parent",
                "checks": checks,
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    launch()
