"""Bounded real-envelope fraction/roof experiment; no random ink or source edits.

Reviewed Session N overline, minus and radical-roof contours supply genuine edge
and terminal observations. The semantic role transfer is explicit, never a new
handwritten label. MathText geometry and every non-rule item remain fixed.
"""

import argparse
import copy
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter

from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.schema import Background
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import json_value
from research.author_feedback_revision import identity, personal_layout, trace
from research.author_feedback_round2 import load_baseline, replay_page
from research.corpus.math_glyph_completion import local_comparison, raster
from research.corpus.math_session_glyphs import outline_bounds
from research.corpus.math_session_polish import polygons, sections
from research.homework_benchmark.render import accepted_scans, read, save_pdf, source_paths
from research.math_feedback_candidate import paired_panel
from research.math_latin_feedback import MathLatinFeedback, filled_path

SOURCE_IDS = (
    "N-031-U0305-p1",
    "N-042-U2212-p1",
    "N-043-U2212-p1",
    "N-063-U221A-p1",
    "N-064-U221A-p1",
    "N-065-U221A-p1",
)
RULES = {"fraction-bar", "radical-overbar"}
SOURCE_SHA = "d455f36021e9a0e4c7d9f1c319dd091c3b1f9da04edd4cbe28412e1a52f8f6f8"
STRESS = r"""Inspect the original mathematical expressions; all variables remain fixed.

$$a=\frac{1}{2},\quad b=\frac{x+1}{y-1},\quad c=\frac{a+b+c+d+e+f}{u+v+w}$$

$$r_1=\sqrt{2},\quad r_2=\sqrt{x^2+y^2+z^2},\quad r_3=\sqrt{\frac{a^2+b^2}{c^2+d^2}}$$

$$r_4=\sqrt{1+\sqrt{1+x^2}},\quad r_5=\sqrt{\frac{1+\sqrt{3}}{2}}$$

$$q=\frac{\frac{a+b}{c+d}}{\frac{e+f}{g+h}},\quad t=\frac{1}{1+\frac{1}{1+x}}$$
"""


def clean_path(path):
    rings = []
    for ring in polygons(path):
        # Same winding for additive pieces; avoid cancellation in a seam overlap.
        area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
        if area < 0:
            ring = ring[::-1]
        rings.append(MplPath(ring, closed=True))
    return MplPath.make_compound_path(*rings)


def vertical_sections(path, xs):
    return sections([p[:, ::-1] for p in polygons(path)], np.asarray(xs))


def measure(path, count=321):
    low, high = outline_bounds(path.vertices, path.codes)
    eps = min(0.001, (high[0] - low[0]) / 10000)
    xs = np.linspace(low[0] + eps, high[0] - eps, count)
    bands = vertical_sections(path, xs)
    if any(len(b) != 1 for b in bands):
        raise ValueError("Horizontal source contains gaps or multiple vertical ink intervals")
    values = np.asarray([b[0] for b in bands])
    centres = values.mean(axis=1)
    thickness = np.diff(values, axis=1)[:, 0]
    return {
        "x": xs,
        "centre": centres,
        "thickness": thickness,
        "bounds": np.r_[low, high],
        "width": high[0] - low[0],
        "thickness_p75": np.quantile(thickness, 0.75),
        "centre_range": np.ptp(centres),
        "endpoint_delta": centres[-1] - centres[0],
    }


@dataclass
class HorizontalSource:
    source: dict
    path: MplPath
    metric: dict
    clipped_roof: bool


def load_horizontal(directory):
    result = []
    for glyph in accepted_scans(directory):
        if glyph.source["id"] not in SOURCE_IDS:
            continue
        if glyph.source["source_sha256"] != SOURCE_SHA:
            raise ValueError("Unexpected source outside approved Session N")
        path = clean_path(MplPath(glyph.outline.verts, glyph.outline.codes))
        roof = glyph.source["role"] == "radical"
        if roof:
            low, high = outline_bounds(path.vertices, path.codes)
            path = clean_path(
                path.clip_to_bbox(
                    Bbox.from_extents(
                        glyph.source["seam_x"],
                        low[1] - 1,
                        high[0] + 1,
                        high[1] + 1,
                    )
                )
            )
        result.append(HorizontalSource(glyph.source, path, measure(path), roof))
    if {x.source["id"] for x in result} != set(SOURCE_IDS):
        raise ValueError("Missing approved horizontal sources")
    return result


def prepare_sources(run, baseline):
    old = baseline["path"] / "solutions/session-glyphs"
    directory = run / "solutions/structural-sources"
    directory.mkdir(parents=True)
    rows = [r for r in read(old / "sources.json") if r["id"] in SOURCE_IDS]
    for row in rows:
        for key in ("crop", "parent_300dpi_crop"):
            target = directory / row[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old / row[key], target)
        pdf = Path(row["pdf"])
        if sha256(pdf) != SOURCE_SHA:
            raise ValueError("Original Session N PDF changed")
    shutil.copyfile(Path(rows[0]["pdf"]), directory / "session-N-original.pdf")
    atomic_json(directory / "sources.json", rows)
    samples = load_horizontal(directory)
    audit = []
    for sample in samples:
        metrics = {
            key: value.tolist() if isinstance(value, np.ndarray) else float(value)
            for key, value in sample.metric.items()
        }
        archive = directory / f"{sample.source['id']}-selected-contour.npz"
        np.savez_compressed(archive, vertices=sample.path.vertices, codes=sample.path.codes)
        entry = {
            "source": sample.source,
            "measurement": metrics,
            "selected_roof_only": sample.clipped_roof,
            "selection": {"x_greater_equal_source_seam": sample.source.get("seam_x")}
            if sample.clipped_roof
            else {"complete_isolated_contour": True},
            "contour_archive": archive.name,
            "contour_sha256": sha256(archive),
            "actual_source_label_unchanged": True,
            "raw_pixels_edited": False,
            "role_transfer": "measured horizontal ink used for a parser-owned structural rule",
        }
        audit.append(entry)
        fig = Figure(figsize=(9, 3))
        left, right = fig.subplots(1, 2)
        left.imshow(Image.open(directory / sample.source["crop"]))
        left.set_title("Unmodified 300 DPI RGB")
        left.axis("off")
        for ring in polygons(sample.path):
            right.plot(ring[:, 0], ring[:, 1], linewidth=0.6, color="black")
        right.plot(sample.metric["x"], sample.metric["centre"], color="red", linewidth=0.5)
        right.invert_yaxis()
        right.set_aspect("equal")
        right.set_title("Selected contour + measured centre")
        fig.suptitle(sample.source["id"] + " / label and source role retained")
        fig.tight_layout()
        fig.savefig(run / f"comparisons/{sample.source['id']}-source.png", dpi=150)
        fig.clear()
    atomic_json(directory / "measurement-manifest.json", audit)
    return directory


def safe_window(index, record, before_paths, records, em, paired_sign):
    x0, y0, x1, y1 = record["bounds_px"]
    middle = (y0 + y1) / 2
    low, high = middle - 0.18 * em, middle + 0.18 * em
    constraints = []
    for j, path in before_paths.items():
        if j in (index, paired_sign):
            continue
        a, b, c, d = path.get_extents().extents
        if c <= x0 or a >= x1:
            continue
        if d <= middle:
            bound = min(y0, d + min(1.0, max(0, y0 - d) / 2))
            if bound > low:
                low = bound
                constraints.append({"item": j, "side": "above", "limit_y": float(bound)})
        elif b >= middle:
            bound = max(y1, b - min(1.0, max(0, b - y1) / 2))
            if bound < high:
                high = bound
                constraints.append({"item": j, "side": "below", "limit_y": float(bound)})
        else:
            low, high = max(low, y0), min(high, y1)
            constraints.append({"item": j, "side": "straddles; original envelope only"})
    return [x0, min(low, y0), x1, max(high, y1)], constraints


def fit_source(sample, old_path, allowed, *, roof=False):
    old = measure(old_path)
    source = sample.metric
    x0, _, x1, _ = old["bounds"]
    median = old["thickness_p75"]
    if x1 - x0 < 2.5 * median:
        return None, {
            "status": "held-short-range",
            "reason": "too little width for a safe terminal",
        }
    thickness_scale = median / source["thickness_p75"]
    scale_x = (x1 - x0) / source["width"]
    source_anchor = source["centre"][0] if roof else float(np.median(source["centre"]))
    anchor = old["centre"][0] if roof else (old["bounds"][1] + old["bounds"][3]) / 2
    dense_rings = []
    for ring in polygons(sample.path):
        points = []
        for a, b in zip(ring[:-1], ring[1:], strict=True):
            n = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.35)))
            points.append(a + np.arange(n)[:, None] / n * (b - a))
        dense_rings.append(np.vstack([*points, ring[-1:]]))
    vertices = np.vstack(dense_rings)
    centre = np.interp(vertices[:, 0], source["x"], source["centre"])
    residual = vertices[:, 1] - centre
    offset = centre - source_anchor
    # Keep nib variation; constrain only its source-derived centre movement first.
    for scale in (thickness_scale, thickness_scale * 0.9, thickness_scale * 0.8):
        low_limit, high_limit = 0.0, scale_x
        zero = anchor + residual * scale
        for z, slope in zip(zero, offset, strict=True):
            if abs(slope) < 1e-8:
                if not allowed[1] - 1e-6 <= z <= allowed[3] + 1e-6:
                    high_limit = -1
                    break
                continue
            bounds = sorted(((allowed[1] - z) / slope, (allowed[3] - z) / slope))
            low_limit, high_limit = max(low_limit, bounds[0]), min(high_limit, bounds[1])
        if high_limit >= low_limit:
            thickness_scale = scale
            centre_scale = max(0.0, high_limit)
            break
    else:
        return None, {"status": "held-clearance", "reason": "observed nib cannot fit guarded space"}
    mapped, cursor = [], 0
    for ring in dense_rings:
        count = len(ring)
        x = x0 + (ring[:, 0] - source["bounds"][0]) * scale_x
        y = anchor + offset[cursor : cursor + count] * centre_scale
        y += residual[cursor : cursor + count] * thickness_scale
        mapped.append(MplPath(np.column_stack((x, y)), closed=True))
        cursor += count
    candidate = clean_path(MplPath.make_compound_path(*mapped))
    seam = None
    if roof:
        # This is the original join geometry, not a fabricated raw-source terminal.
        seam_width = min(1.0, (x1 - x0) * 0.08)
        original_join = clean_path(
            old_path.clip_to_bbox(
                Bbox.from_extents(
                    x0 - 0.01,
                    old["bounds"][1] - 0.1,
                    x0 + seam_width,
                    old["bounds"][3] + 0.1,
                )
            )
        )
        candidate = clean_path(MplPath.make_compound_path(original_join, candidate))
        seam = {
            "method": "additive original left join slice; same winding",
            "preserved_original_x_range": [float(x0), float(x0 + seam_width)],
        }
    info = {
        "status": "candidate",
        "source_id": sample.source["id"],
        "source_character": sample.source["character"],
        "source_role": sample.source["role"],
        "source_sha256": sample.source["source_sha256"],
        "crop_sha256": sample.source["crop_sha256"],
        "source_metadata": sample.source,
        "selected_original_roof_only": sample.clipped_roof,
        "actual_source_label_changed": False,
        "raw_source_pixels_edited": False,
        "role_transfer": "observed horizontal contour to parser-owned structural rule",
        "measured_source": True,
        "procedural_geometry": True,
        "geometry_disclosure": (
            "source centre/nib decomposition, separate coordinate scaling, "
            "clearance fit, and optional original roof-seam composition"
        ),
        "target_x_span": [float(x0), float(x1)],
        "x_scale": float(scale_x),
        "centre_scale_requested": float(scale_x),
        "centre_scale_applied": float(centre_scale),
        "thickness_scale": float(thickness_scale),
        "target_y_anchor": float(anchor),
        "source_y_anchor": float(source_anchor),
        "source_measurement_count": len(source["x"]),
        "old_thickness_p75": float(median),
        "source_thickness_p75": float(source["thickness_p75"]),
        "target_nominal_thickness_p75": float(source["thickness_p75"] * thickness_scale),
        "source_centre_range": float(source["centre_range"]),
        "target_measured_centre_range": float(source["centre_range"] * centre_scale),
        "source_endpoint_delta": float(source["endpoint_delta"]),
        "target_measured_endpoint_delta": float(source["endpoint_delta"] * centre_scale),
        "random_variation": False,
        "join_preservation": seam,
    }
    return candidate, info


class StructuralRuleFeedback:
    def __init__(self, directory):
        self.sources = load_horizontal(Path(directory))
        self.decisions = []

    def apply(self, page, renderer, provenance):
        records = copy.deepcopy(provenance)
        original_items = list(page.items)
        before_paths = {i: filled_path(item, renderer) for i, item in enumerate(page.items)}
        self.decisions = []
        for row in records:
            if row["context"] != "mathematics" or row.get("structure") not in RULES:
                continue
            i = row["item_index"]
            old = clean_path(before_paths[i])
            roof = row["structure"] == "radical-overbar"
            metric = measure(old)
            # A radical seam is not a free pen terminal. Only intact horizontal
            # samples can supply both endpoints of a fraction rule.
            eligible = [s for s in self.sources if s.clipped_roof == roof]
            # Deterministic length/nib compatibility, never a random fake handwriting variant.
            ranked = sorted(
                eligible,
                key=lambda s: abs(
                    math.log(
                        metric["width"]
                        / (s.metric["width"] * metric["thickness_p75"] / s.metric["thickness_p75"])
                    )
                ),
            )
            selected = ranked[0]
            paired_signs = [
                j
                for j in row.get("paired_items", [])
                if records[j].get("structure") == "radical-sign"
            ]
            paired_sign = paired_signs[0] if len(paired_signs) == 1 else None
            decision = {
                "item_index": i,
                "structure": row["structure"],
                "latex": row["latex"],
                "source_selection": "closest native width after matching measured nib p75",
                "ranked_source_ids": [s.source["id"] for s in ranked],
                "original_bounds": row["bounds_px"],
                "paired_sign": paired_sign,
            }
            allowed, constraints = safe_window(
                i, row, before_paths, records, renderer.cfg.body_size, paired_sign
            )
            decision.update(allowed_bounds_px=allowed, clearance_constraints=constraints)
            if roof and paired_sign is None:
                candidate, treatment = None, {"status": "held-unpaired-roof"}
            else:
                candidate, treatment = fit_source(selected, old, allowed, roof=roof)
            decision.update(treatment)
            if candidate is not None:
                bounds = outline_bounds(candidate.vertices, candidate.codes).ravel()
                if np.any(bounds[:2] < np.asarray(allowed[:2]) - 0.002) or np.any(
                    bounds[2:] > np.asarray(allowed[2:]) + 0.002
                ):
                    raise ValueError("Measured candidate left its precomputed clearance envelope")
                contacts = []
                for j, prior_other in before_paths.items():
                    if j == i:
                        continue
                    other = filled_path(page.items[j], renderer)
                    if not candidate.get_extents().overlaps(other.get_extents()):
                        continue
                    before_contact = old.intersects_path(prior_other, filled=True)
                    after_contact = candidate.intersects_path(other, filled=True)
                    if after_contact or before_contact:
                        contacts.append(
                            {
                                "item": j,
                                "before": bool(before_contact),
                                "after": bool(after_contact),
                            }
                        )
                unexpected = [
                    c
                    for c in contacts
                    if c["after"] and not c["before"] and c["item"] != paired_sign
                ]
                seam_ok = (
                    not roof
                    or not old.intersects_path(before_paths[paired_sign], filled=True)
                    or candidate.intersects_path(before_paths[paired_sign], filled=True)
                )
                decision.update(contacts=contacts, seam_contact_preserved=bool(seam_ok))
                if unexpected or not seam_ok:
                    decision.update(
                        status="held-contact",
                        reason="new neighbour ink contact or lost original join",
                    )
                    candidate = None
            if candidate is not None:
                key = ("structural-rule-feedback", i)
                renderer._outlines[key] = Outline(
                    candidate.vertices, candidate.codes, stroke=False, digits=4
                )
                page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), original_items[i][3], 0)
                row.update(
                    source_type="personal-scan-structural-rule",
                    personal_collected=True,
                    external_fallback=False,
                    procedural_geometry=True,
                    measured_source=True,
                    allowed_bounds_px=allowed,
                    bounds_px=bounds.tolist(),
                    actual_glyph_id=f"structural-rule:{selected.source['id']}",
                    source_sha256=selected.source["source_sha256"],
                    transformation=(
                        "measured source centre/nib decomposition with clearance-limited scale"
                    ),
                    original_source_type=provenance[i]["source_type"],
                    original_glyph_id=provenance[i]["actual_glyph_id"],
                )
                row.pop("fallback_reason", None)
                decision["status"] = "applied"
                decision["candidate_bounds"] = bounds.tolist()
            row["structural_rule_feedback"] = copy.deepcopy(decision)
            self.decisions.append(decision)
        return records


def validate_page(old_items, before_paths, before_records, state, records, image):
    page, renderer = state["page"], state["renderer"]
    if identity(before_records) != identity(records) or json_value(trace(renderer)) != json_value(
        state["parser"]
    ):
        raise ValueError("Rule replacement changed semantic identity / parser trace")
    allowed = np.zeros((image.height, image.width), bool)
    changed_ids = []
    for i, (old, new) in enumerate(zip(before_records, records, strict=True)):
        change = new.get("structural_rule_feedback")
        if not change:
            if old != new or old_items[i] != page.items[i]:
                raise ValueError("Non-target item/record changed")
            actual = filled_path(page.items[i], renderer)
            if not np.array_equal(actual.vertices, before_paths[i].vertices) or not np.array_equal(
                actual.codes, before_paths[i].codes
            ):
                raise ValueError("Non-target outline changed through shared renderer key")
            continue
        if change["status"] != "applied":
            if old_items[i] != page.items[i]:
                raise ValueError("Held target changed pixels")
            continue
        changed_ids.append(i)
        x0, y0, x1, y1 = change["allowed_bounds_px"]
        allowed[
            max(0, int(np.floor(y0)) - 3) : min(image.height, int(np.ceil(y1)) + 3),
            max(0, int(np.floor(x0)) - 3) : min(image.width, int(np.ceil(x1)) + 3),
        ] = True
    difference = np.any(np.asarray(image) != np.asarray(state["image"]), axis=2)
    if (difference & ~allowed).any():
        raise ValueError("Pixels changed outside measured rule clearance envelopes")
    for row in before_records:
        if row["context"] != "text":
            continue
        x0, y0, x1, y1 = row["bounds_px"]
        if difference[
            max(0, int(y0) - 2) : int(np.ceil(y1)) + 2, max(0, int(x0) - 2) : int(np.ceil(x1)) + 2
        ].any():
            raise ValueError("English pixels changed")
    bounds, _ = ink_geometry(page, renderer)
    if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
        raise ValueError("Page boundary clipping")
    return {
        "applied_items": changed_ids,
        "non_target_items_and_outlines_exact": True,
        "english_pixels_exact": True,
        "parser_and_identity_trace_exact": True,
        "independent_semantic_correctness_claimed": False,
        "changed_pixels": int(difference.sum()),
        "outside_allowed_pixels": 0,
        "ink_bounds": bounds,
    }


def stress_page(baseline):
    page, renderer, records, _, _ = personal_layout(
        STRESS,
        "exposed-structural-rule-stress",
        baseline["profile"],
        baseline["cfg"].model_copy(update={"paper": "plain"}),
        baseline["resolver"],
        baseline["sessions"],
        baseline["polish"],
        Background(kind="plain"),
        None,
        cpu=True,
        clearance=True,
    )
    records = MathLatinFeedback(baseline["profile"].font).apply(page, renderer, records)
    return {
        "page": page,
        "renderer": renderer,
        "records": records,
        "parser": trace(renderer),
        "image": raster(renderer.to_vector_pdf([page])),
        "source": STRESS,
        "entry": {"page": 21, "case_id": "exposed-structural-rule-stress"},
    }


def compact_contacts(output, panels):
    """Thirty complete A/B rules per sheet, same scale within each pair."""
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    sheets = []
    for start in range(0, len(panels), 30):
        sheet = Image.new("RGB", (2480, 3508), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (45, 35),
            "All structural horizontal strokes / A frozen | B measured",
            fill="black",
            font=font,
        )
        for slot, (page, item, a, b) in enumerate(panels[start : start + 30]):
            x, y = 35 + (slot % 3) * 820, 105 + (slot // 3) * 334
            draw.text((x, y), f"Page {page} / item {item}", fill="black", font=small)
            factor = min(1.65, 385 / max(a.width, b.width), 240 / max(a.height, b.height))
            for offset, image in ((0, a), (410, b)):
                size = tuple(max(1, round(d * factor)) for d in image.size)
                sheet.paste(image.resize(size, Image.Resampling.LANCZOS), (x + offset, y + 35))
            draw.line((x, y + 315, x + 790, y + 315), fill="#dddddd")
        sheet.save(output / f"contact-{start // 30 + 1:02d}.png", dpi=(300, 300))
        sheets.append(sheet)
    sheets[0].save(
        output / "all-rule-contacts.pdf", save_all=True, append_images=sheets[1:], resolution=300
    )


def launch():
    baseline = load_baseline()
    run = Run.create(
        "structural-rule-source-envelope-candidate",
        seed=baseline["cfg"].seed,
        config={
            "hypothesis": "measured source edge/terminal variation survives guarded rule placement",
            "baseline": str(baseline["path"]),
            "baseline_sha256": baseline["run_sha256"],
            "render": baseline["cfg"].model_dump(mode="json"),
            "sources": SOURCE_IDS,
            "new_neural_updates": 0,
            "source_images_edited": False,
            "new_labels": False,
            "max_half_envelope_em": 0.18,
            "candidate_limit": "one deterministic geometry-matched treatment",
            "scope": "all 150 baseline horizontal rules plus explicitly exposed stress page",
        },
    )
    print(run.path, flush=True)
    try:
        protected = {
            str(p): sha256(p)
            for p in (workspace_root() / "latest.json", baseline["path"] / "run.json")
        }
        atomic_json(run.path / "reports/protected-assets.json", protected)
        paths = source_paths() + [
            Path(__file__),
            Path("research/author_feedback_round2.py"),
            Path("research/author_feedback_delivery.py"),
            Path("research/author_feedback_revision.py"),
            Path("research/cpu_glyph_feedback.py"),
            Path("research/math_feedback_candidate.py"),
            Path("research/math_latin_feedback.py"),
            Path("research/math_paired_bars.py"),
            Path("research/historical_math_candidate.py"),
        ]
        for source in paths:
            target = run.path / "logs/source" / source.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for version, key in ((1, "profile"), (3, "strokes")):
            copy_snapshot(baseline[key], run.path / f"solutions/profile-v{version}")
        directory = prepare_sources(run.path, baseline)
        treatment = StructuralRuleFeedback(directory)
        panels, all_decisions, checks, outputs = [], [], [], []
        for number in range(1, 22):
            print(f"page {number:02d}", flush=True)
            state = replay_page(baseline, number) if number <= 20 else stress_page(baseline)
            case_id = state["entry"]["case_id"]
            if number <= 20:
                shutil.copytree(state["inputs"], run.path / f"solutions/inputs/{case_id}")
            else:
                (run.path / "solutions/exposed-stress.md").write_text(STRESS)
            page, renderer = state["page"], state["renderer"]
            old_items, before_records = list(page.items), copy.deepcopy(state["records"])
            before_paths = {i: filled_path(item, renderer) for i, item in enumerate(page.items)}
            records = treatment.apply(page, renderer, state["records"])
            path = run.path / f"rendered/pages/{number:02d}.pdf"
            image = save_pdf(renderer.to_vector_pdf([page]), path)
            check = validate_page(old_items, before_paths, before_records, state, records, image)
            decisions = copy.deepcopy(treatment.decisions)
            for d in decisions:
                d.update(page=number, case_id=case_id)
                i = d["item_index"]
                box = np.asarray([before_records[i]["bounds_px"], records[i]["bounds_px"]])
                low = np.floor(box[:, :2].min(axis=0) - [10, 25]).astype(int)
                high = np.ceil(box[:, 2:].max(axis=0) + [10, 25]).astype(int)
                crop = (*low.tolist(), *high.tolist())
                old_crop, new_crop = state["image"].crop(crop), image.crop(crop)
                dest = run.path / f"comparisons/rules/{number:02d}-{i:04d}.png"
                paired_panel(
                    old_crop,
                    new_crop,
                    dest,
                    f"Page {number} / item {i} / {d['status']}",
                    ("A frozen candidate", "B measured horizontal ink"),
                )
                panels.append((number, i, old_crop, new_crop))
                d["comparison"] = str(dest.relative_to(run.path))
                if d["structure"] == "radical-overbar" and d["paired_sign"] is not None:
                    si = d["paired_sign"]
                    root_boxes = np.asarray(
                        [
                            before_records[si]["bounds_px"],
                            records[si]["bounds_px"],
                            before_records[i]["bounds_px"],
                            records[i]["bounds_px"],
                        ]
                    )
                    root_low = np.floor(root_boxes[:, :2].min(axis=0) - 15).astype(int)
                    root_high = np.ceil(root_boxes[:, 2:].max(axis=0) + 15).astype(int)
                    root_crop = (*root_low.tolist(), *root_high.tolist())
                    root_dest = run.path / f"comparisons/full-roots/{number:02d}-{i:04d}.png"
                    paired_panel(
                        state["image"].crop(root_crop),
                        image.crop(root_crop),
                        root_dest,
                        f"Complete root / page {number} / item {i} / {d['status']}",
                        ("A frozen candidate", "B measured horizontal ink"),
                    )
                    d["root_comparison"] = str(root_dest.relative_to(run.path))
            atomic_json(
                run.path / f"reports/{case_id}-provenance.json",
                {
                    "baseline": before_records,
                    "candidate": records,
                    "parser_trace": state["parser"],
                    "decisions": decisions,
                    "checks": check,
                },
            )
            check.update(
                page=number,
                case_id=case_id,
                rules=len(decisions),
                statuses=dict(Counter(d["status"] for d in decisions)),
            )
            checks.append(check)
            all_decisions.extend(decisions)
            outputs.append(path)
        for paths, target in (
            (outputs[:20], run.path / "rendered/homework_solutions.pdf"),
            (outputs[20:], run.path / "rendered/exposed-stress.pdf"),
        ):
            with PdfWriter() as writer:
                for path in paths:
                    writer.append(path)
                writer.write(target)
        local_comparison(
            run.path / "comparisons",
            panels,
            labels=("Frozen structural ink", "Measured source envelope"),
        )
        compact_contacts(
            run.path / "comparisons/compact-baseline", [p for p in panels if p[0] <= 20]
        )
        compact_contacts(run.path / "comparisons/compact-stress", [p for p in panels if p[0] == 21])
        atomic_json(run.path / "reports/rule-decisions.json", all_decisions)
        atomic_json(run.path / "reports/page-checks.json", checks)
        assert sum(c["rules"] for c in checks[:20]) == 150
        assert all(sha256(Path(p)) == digest for p, digest in protected.items())
        run.finish(
            "held",
            result={
                "baseline_pages": 20,
                "baseline_rules": 150,
                "stress_pages": 1,
                "statuses": dict(Counter(d["status"] for d in all_decisions)),
                "selected_sources": dict(Counter(d.get("source_id") for d in all_decisions)),
                "english_pixels_exact": True,
                "non_target_shapes_positions_exact": True,
                "new_neural_updates": 0,
                "production_activation": False,
                "visual_review": "pending root",
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


def revise_metadata(parent):
    """Correct procedural-source disclosure in a fresh receipt, with identical ink."""
    parent = Path(parent).resolve()
    verify_run(parent)
    prior = read(parent / "run.json")
    if prior["kind"] != "structural-rule-source-envelope-candidate":
        raise ValueError("Metadata migration accepts only this isolated structural experiment")
    run = Run.create(
        "structural-rule-provenance-correction",
        seed=prior["seed"],
        config={
            **prior["config"],
            "parent_run": str(parent),
            "parent_sha256": sha256(parent / "run.json"),
            "purpose": (
                "metadata/source-code correction only; "
                "all pixel and PDF artifacts inherited exactly"
            ),
        },
    )
    print(run.path, flush=True)
    try:
        for old in parent.rglob("*"):
            if (
                not old.is_file()
                or old.name == "run.json"
                or old.relative_to(parent).as_posix() == "reports/result.json"
            ):
                continue
            target = run.path / old.relative_to(parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, target)
        original_code = run.path / "logs/source/research/structural_rule_feedback.py"
        shutil.copyfile(original_code, run.path / "logs/source-before-provenance-correction.py")
        shutil.copyfile(__file__, original_code)
        atomic_json(run.path / "reports/prior-result.json", prior["result"])
        patched = []

        def correction(decision):
            if decision["status"] == "applied":
                decision.update(
                    measured_source=True,
                    procedural_geometry=True,
                    geometry_disclosure=(
                        "source centre/nib decomposition, separate coordinate scaling, "
                        "clearance fit, and optional original roof-seam composition"
                    ),
                )

        def geometry(value):
            if isinstance(value, list):
                return [geometry(v) for v in value]
            if isinstance(value, dict):
                return {
                    k: geometry(v)
                    for k, v in value.items()
                    if k not in {"measured_source", "procedural_geometry", "geometry_disclosure"}
                }
            return value

        for path in sorted((run.path / "reports").glob("*-provenance.json")):
            before = read(path)
            after = copy.deepcopy(before)
            for row in after["candidate"]:
                if row["source_type"] == "personal-scan-structural-rule":
                    row.update(procedural_geometry=True, measured_source=True)
                    correction(row["structural_rule_feedback"])
            for decision in after["decisions"]:
                correction(decision)
            if before["baseline"] != after["baseline"] or geometry(before) != geometry(after):
                raise ValueError("Metadata migration modified a baseline record or geometry")
            atomic_json(path, after)
            patched.append(str(path.relative_to(run.path)))
        decisions_path = run.path / "reports/rule-decisions.json"
        before = read(decisions_path)
        after = copy.deepcopy(before)
        for decision in after:
            correction(decision)
        if geometry(before) != geometry(after):
            raise ValueError("Metadata migration modified a placement decision")
        atomic_json(decisions_path, after)
        same_pixels = {}
        for path in parent.rglob("*"):
            if path.suffix.lower() not in {".pdf", ".png", ".npz", ".ttf"}:
                continue
            rel = path.relative_to(parent)
            if sha256(path) != sha256(run.path / rel):
                raise ValueError("A visual/source/font artifact changed during metadata correction")
            same_pixels[str(rel)] = sha256(path)
        atomic_json(
            run.path / "reports/metadata-correction.json",
            {
                "parent_run": str(parent),
                "parent_sha256": sha256(parent / "run.json"),
                "patched_provenance": patched,
                "all_baseline_records_exact": True,
                "all_placement_and_source_mapping_values_exact": True,
                "byte_exact_visual_and_font_artifacts": same_pixels,
                "meaning": (
                    "personal_collected identifies real source; "
                    "procedural_geometry identifies generated structure"
                ),
            },
        )
        run.finish(
            "held",
            result={
                **prior["result"],
                "parent_run": str(parent),
                "provenance_corrected": True,
                "visual_artifacts_byte_exact": True,
                "measured_source_and_procedural_geometry_both_true": True,
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-correction", type=Path)
    args = parser.parse_args()
    if args.metadata_correction:
        revise_metadata(args.metadata_correction)
    else:
        launch()
