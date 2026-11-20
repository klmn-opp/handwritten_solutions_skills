"""Bounded real-f size recovery around its crossbar; no redrawing or dilation.

Historical merge_fonts/_fit_glyph fitted the author's descending f into an
ordinary ascender box. This experiment restores part of the lost relative scale
without changing any original font, raster, label, layout advance or other item.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from handnote import symbols
from handnote.markdown import parse
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image
from pypdf import PdfWriter

from hwsol.jobs.layout import plain_emphasis
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_revision import identity, trace
from research.author_feedback_round2 import load_baseline, replay_page
from research.corpus.math_glyph_completion import local_comparison
from research.corpus.math_session_polish import polygons, sections
from research.corpus.readability import placed_path
from research.cpu_glyph_feedback import compiled_lineage
from research.homework_benchmark.render import save_pdf
from research.math_feedback_candidate import paired_panel
from research.math_latin_feedback import FONT_SHA, filled_path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FONT = ROOT / (
    "profiles/realism-studio/training-runs/personal/"
    "r-e9ef6e0126b84037ad2633745370d424/sample-0000.ttf"
)
SOURCE_SAMPLE = ROOT / (
    "profiles/realism-studio/samples/personal/s-c222860c59414473bf606d42eade46b9"
)
# A=1.15 was too subtle in the first sealed probe and is retired, not deleted.
# D is the one additional source-coordinate control requested after anchor review.
COMMON_SOURCE = {
    "reference_characters": "hldaceo",
    "method": "least-squares common y affine over reference top and bottom coordinates",
    "source_to_reference_scale": 1.3971949065933285,
    "source_to_reference_y_translation": -43.40055063616691,
    "reference_fit_rmse_font_units": 19.80359711525346,
    "target_f_top_font_units": 719.4678683637904,
    "target_f_bottom_font_units": -314.45636251527264,
}
PRESETS = {"B-intermediate": 1.28, "C-source-ratio": 1.42, "D-common-source": 1.444028255417686}
CROSSBAR_FONT_Y = 436.180928112813
PROBES = (1, 6, 15, 20)
SAFETY_BUDGET = {
    "maximum_uniform_scale": 1.45,
    "maximum_left_or_right_extension_em": 0.075,
    "maximum_top_extension_em": 0.15,
    "maximum_bottom_extension_em": 0.33,
    "minimum_new_neighbor_clearance_px": 0.60,
    "minimum_page_clearance_px": 3.0,
    "audit_neighbor_radius_em": 2.0,
}


def json_value(value):
    return json.loads(json.dumps(value))


def stem_measure(path, *, f=True):
    """Median horizontal section through one narrow stem, excluding f crossbar/hook.

    For f use the lower stem (10..40% up from its bottom). Other tall letters use
    70..90% up, above their bowl. This is outline geometry, not estimated pen pressure.
    """
    box = path.get_extents()
    if f:
        lo, hi = box.ymin + 0.60 * box.height, box.ymin + 0.90 * box.height
    else:
        lo, hi = box.ymin + 0.10 * box.height, box.ymin + 0.30 * box.height
    values = [
        float(row[0, 1] - row[0, 0])
        for row in sections(polygons(path), np.linspace(lo, hi, 40))
        if len(row) == 1
    ]
    return {
        "method": "single lower-stem sections; crossbar and hook excluded"
        if f
        else "single upper-ascender sections above bowl",
        "sample_count": len(values),
        "median_px": float(np.median(values)) if values else None,
        "p10_px": float(np.quantile(values, 0.1)) if values else None,
        "p90_px": float(np.quantile(values, 0.9)) if values else None,
        "measurement_is_pen_pressure": False,
    }


def contour_distance(a, b):
    """Minimum flattened-contour point/segment distance for near-neighbor gating."""
    if a.intersects_path(b, filled=True):
        return 0.0
    aa, bb = polygons(a), polygons(b)
    if not aa or not bb:
        return float("inf")

    def directed(points, rings):
        starts = np.concatenate([r[:-1] for r in rings])
        ends = np.concatenate([r[1:] for r in rings])
        delta = ends - starts
        denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
        diff = points[:, None, :] - starts[None, :, :]
        t = np.clip(np.sum(diff * delta, axis=2) / denom, 0, 1)
        return float(np.sqrt(np.min(np.sum((diff - t[:, :, None] * delta) ** 2, axis=2))))

    return min(directed(np.concatenate(aa), bb), directed(np.concatenate(bb), aa))


def consecutive_f_groups(source, title, records):
    """Map real same-word ff runs to recorder items after exact text-order validation."""
    spans = [title]
    for block in parse(plain_emphasis(source)):
        spans.extend(text for kind, text in block.segments if kind == "t")
    visible = [r for r in records if r["context"] == "text"]
    expected = [ch for span in spans for ch in span if not ch.isspace()]
    if expected != [r["semantic"] for r in visible]:
        raise ValueError("Cannot bind ff groups: source text and visible order differ")
    groups, offset = [], 0
    for span_index, span in enumerate(spans):
        mapping = {}
        for pos, ch in enumerate(span):
            if not ch.isspace():
                mapping[pos] = visible[offset]["item_index"]
                offset += 1
        for match in re.finditer(r"[A-Za-z]+", span):
            for repeat in re.finditer(r"f{2,}", match.group()):
                start = match.start() + repeat.start()
                end = match.start() + repeat.end()
                groups.append(
                    {
                        "group_id": f"span-{span_index}-word-{match.start()}-ff-{start}",
                        "word": match.group(),
                        "span_index": span_index,
                        "word_span": [match.start(), match.end()],
                        "f_span": [start, end],
                        "item_indices": [mapping[i] for i in range(start, end)],
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                        "source_word_and_visible_text_order_verified": True,
                        "cross_word_or_math_segment_grouping": False,
                    }
                )
    return groups


class FSizeFeedback:
    """Uniform output geometry only, with explicit per-instance rejection."""

    def __init__(self, preset="D-common-source", *, source=None, title=None):
        if preset not in PRESETS:
            raise ValueError("Unknown preregistered f size preset")
        self.preset, self.scale = preset, PRESETS[preset]
        self.decisions = []
        self.source, self.title = source, title
        self.groups = []

    def apply(self, page, renderer, provenance):
        if self.source is None or self.title is None:
            raise ValueError("Actual source and title required for atomic same-word ff admission")
        self.groups = consecutive_f_groups(self.source, self.title, provenance)
        records = copy.deepcopy(provenance)
        original_items = list(page.items)
        originals = [filled_path(item, renderer) for item in page.items]
        self.decisions = []
        for row in records:
            if row["semantic"] != "f":
                continue
            i = row["item_index"]
            decision = {
                "item_index": i,
                "context": row["context"],
                "preset": self.preset,
                "requested_scale": self.scale,
                "applied": False,
                "rejections": [],
            }
            self.decisions.append(decision)
            if row.get("font_sha256") != FONT_SHA or row.get("variant") != 0:
                decision["rejections"].append("Expected approved real f0 source")
                continue
            old = originals[i]
            box = old.get_extents()
            actual_em = box.height * 1000 / 716
            decision["actual_em_px"] = actual_em
            if actual_em < 0.80 * renderer.cfg.body_size:
                decision["rejections"].append("Small script f held for separate scale protocol")
                row["f_size_feedback"] = {"status": "original-retained", **decision}
                continue
            anchor = np.array(
                [(box.xmin + box.xmax) / 2, box.ymin + (715 - CROSSBAR_FONT_Y) / 716 * box.height]
            )
            if self.preset == "D-common-source":
                top_delta = (715 - COMMON_SOURCE["target_f_top_font_units"]) / 1000 * actual_em
                anchor[1] = box.ymin - top_delta / (self.scale - 1)
            path = MplPath(anchor + (old.vertices - anchor) * self.scale, old.codes)
            final_box = path.get_extents().extents.tolist()
            expansion = np.abs(np.asarray(final_box) - box.extents) / actual_em
            decision.update(
                original_bounds=box.extents.tolist(),
                requested_bounds=final_box,
                anchor_px=anchor.tolist(),
                original_stem=stem_measure(old),
                candidate_stem=stem_measure(path),
                preserved_crossbar_height=self.preset != "D-common-source",
                common_source_coordinates=COMMON_SOURCE
                if self.preset == "D-common-source"
                else None,
                uniform_geometry=True,
                added_strokes=False,
                pen_pressure_modified=False,
                source_outline_codes_exact=True,
                safety_budget=SAFETY_BUDGET,
                absolute_bound_displacement_em=expansion.tolist(),
            )
            if (
                self.scale > SAFETY_BUDGET["maximum_uniform_scale"]
                or max(expansion[0], expansion[2])
                > SAFETY_BUDGET["maximum_left_or_right_extension_em"]
                or expansion[1] > SAFETY_BUDGET["maximum_top_extension_em"]
                or expansion[3] > SAFETY_BUDGET["maximum_bottom_extension_em"]
            ):
                decision["rejections"].append("frozen scale or displacement budget exceeded")
            if (
                min(final_box[:2]) < 3
                or final_box[2] > page.width - 3
                or final_box[3] > page.height - 3
            ):
                decision["rejections"].append("page boundary")
            nearby = []
            for j, other in enumerate(originals):
                ob = other.get_extents()
                dx = max(0, ob.xmin - final_box[2], final_box[0] - ob.xmax)
                dy = max(0, ob.ymin - final_box[3], final_box[1] - ob.ymax)
                if j == i or np.hypot(dx, dy) > actual_em * 2:
                    continue
                prior = contour_distance(old, other)
                after = contour_distance(path, other)
                nearby.append(
                    {
                        "other_item": j,
                        "semantic": records[j]["semantic"],
                        "context": records[j]["context"],
                        "before_clearance_px": prior,
                        "after_clearance_px": after,
                        "relative_line": (
                            "above"
                            if ob.ymax < final_box[1]
                            else "below"
                            if ob.ymin > final_box[3]
                            else "overlapping-y-band"
                        ),
                        "horizontal_box_gap_px": dx,
                    }
                )
                if after < 0.60 and after < prior - 0.05:
                    decision["rejections"].append(f"neighbor {j} clearance {after:.3f}px")
            decision["neighbors"] = nearby
            decision["nearest_neighbor_px"] = min(
                (n["after_clearance_px"] for n in nearby), default=None
            )
            for direction in ("above", "below"):
                decision[f"nearest_{direction}_aligned_ink_px"] = min(
                    (
                        n["after_clearance_px"]
                        for n in nearby
                        if n["relative_line"] == direction and n["horizontal_box_gap_px"] == 0
                    ),
                    default=None,
                )
            if decision["rejections"]:
                row["f_size_feedback"] = {"status": "original-retained", **decision}
                continue
            key = ("f-size-feedback", self.preset, i)
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), original_items[i][3], 0)
            row.update(
                bounds_px=final_box,
                allowed_bounds_px=final_box,
                source_type="personal-outline-geometric-calibration",
                f_size_feedback={
                    "status": "research-candidate",
                    "preset": self.preset,
                    "scale_x": self.scale,
                    "scale_y": self.scale,
                    "anchor_px": anchor.tolist(),
                    "method": (
                        "uniform size and y-position from common h/l/d/a/c/e/o source coordinates"
                        if self.preset == "D-common-source"
                        else "uniform output enlargement around compiled crossbar height"
                    ),
                    "common_source_coordinates": (
                        COMMON_SOURCE if self.preset == "D-common-source" else None
                    ),
                    "original": copy.deepcopy(provenance[i]),
                    "original_contour_codes_and_aspect_ratio_preserved": True,
                    "raw_rgb_labels_font_unchanged": True,
                    "advance_and_other_items_unchanged": True,
                    "output_envelope_scope": (
                        "allowed_bounds_px is the target output envelope, not a reserved "
                        "MathText slot or independent safety proof; frozen displacement "
                        "budgets, measured neighbor distances and final non-target checks apply"
                    ),
                    "safety_budget": SAFETY_BUDGET,
                    "source_font_sha256": FONT_SHA,
                    "author_acceptance": False,
                },
            )
            decision["applied"] = True
        # Final simultaneous neighbor check catches adjacent ff interactions.
        for decision in self.decisions:
            if not decision["applied"]:
                continue
            i = decision["item_index"]
            path = filled_path(page.items[i], renderer)
            for other_decision in self.decisions:
                j = other_decision["item_index"]
                if i >= j or not other_decision["applied"]:
                    continue
                other = filled_path(page.items[j], renderer)
                if not path.get_extents().expanded(1.1, 1.1).overlaps(other.get_extents()):
                    continue
                before = contour_distance(originals[i], originals[j])
                after = contour_distance(path, other)
                if after < 0.60 and after < before - 0.05:
                    # Symmetric refusal; do not privilege whichever f was traversed first.
                    for d in (decision, other_decision):
                        k = d["item_index"]
                        page.items[k] = original_items[k]
                        records[k] = copy.deepcopy(provenance[k])
                        d["applied"] = False
                        d["rejections"].append("simultaneous ff clearance")
                        records[k]["f_size_feedback"] = {"status": "original-retained", **d}
        by_index = {d["item_index"]: d for d in self.decisions}
        for group in self.groups:
            failures = [i for i in group["item_indices"] if not by_index[i]["applied"]]
            group["original_failure_items"] = failures
            group["atomic_action"] = "retain-whole-group" if failures else "accept-whole-group"
            if not failures:
                continue
            causes = [{"item_index": i, "rejections": by_index[i]["rejections"]} for i in failures]
            for i in group["item_indices"]:
                decision = by_index[i]
                independently_accepted = decision["applied"]
                decision.update(
                    applied=False,
                    group_rejection={
                        **group,
                        "causes": causes,
                        "independently_accepted_before_group_policy": independently_accepted,
                    },
                )
                if independently_accepted:
                    decision["rejections"].append("atomic same-word ff group retained")
                page.items[i] = original_items[i]
                records[i] = copy.deepcopy(provenance[i])
                records[i]["f_size_feedback"] = {"status": "original-retained", **decision}
        # Audit final simultaneous f-f distance after atomic restoration as well.
        for decision in self.decisions:
            if not decision["applied"]:
                continue
            i = decision["item_index"]
            current = filled_path(page.items[i], renderer)
            neighbors = []
            for other in self.decisions:
                j = other["item_index"]
                if i == j:
                    continue
                target = filled_path(page.items[j], renderer)
                if not current.get_extents().expanded(1.5, 1.2).overlaps(target.get_extents()):
                    continue
                before = contour_distance(originals[i], originals[j])
                after = contour_distance(current, target)
                neighbors.append({"item_index": j, "before_px": before, "final_px": after})
                if after < 0.60 and after < before - 0.05:
                    raise ValueError("Final f-f clearance failed after group policy")
            decision["final_f_neighbor_distances"] = neighbors
        if identity(records) != identity(provenance):
            raise ValueError("f treatment changed content identity/order")
        for i, (a, b) in enumerate(zip(original_items, page.items, strict=True)):
            if provenance[i]["semantic"] != "f" and (a != b or records[i] != provenance[i]):
                raise ValueError("Non-f item changed")
        return records


def audit_page(replay, number):
    renderer, page, rows = replay["renderer"], replay["page"], replay["records"]
    result = []
    for row in rows:
        if row["semantic"] != "f":
            continue
        i = row["item_index"]
        path = placed_path(page.items[i], renderer)
        box = path.get_extents()
        em = box.height * 1000 / 716
        neighbours = []
        for other in rows[max(0, i - 40) : i + 41]:
            if other["semantic"] not in "aceoxhldt" or other["context"] != row["context"]:
                continue
            b = np.array(other["bounds_px"])
            if abs(b[3] - box.ymax) > 0.20 * em:
                continue
            if abs((b[0] + b[2]) / 2 - (box.xmin + box.xmax) / 2) > 4 * em:
                continue
            entry = {
                "item_index": other["item_index"],
                "semantic": other["semantic"],
                "height_px": float(b[3] - b[1]),
                "height_over_f_em": float((b[3] - b[1]) / em),
            }
            if other["semantic"] in "hld":
                entry["stem"] = stem_measure(
                    placed_path(page.items[other["item_index"]], renderer), f=False
                )
            neighbours.append(entry)
        result.append(
            {
                "page": number,
                **row,
                "actual_em_px": em,
                "ink_height_em": box.height / em,
                "ink_width_em": box.width / em,
                "lower_stem": stem_measure(path),
                "same_row_comparators": neighbours,
                "small_script": bool(em < 0.8 * renderer.cfg.body_size),
            }
        )
    return result


def pixel_checks(before, after, original, records, changed):
    diff = np.any(np.asarray(before) != np.asarray(after), axis=2)
    permitted = np.zeros(diff.shape, dtype=bool)
    for i in changed:
        for row in (original[i], records[i]):
            x0, y0, x1, y1 = row["bounds_px"]
            permitted[
                max(0, int(np.floor(y0)) - 2) : min(after.height, int(np.ceil(y1)) + 2),
                max(0, int(np.floor(x0)) - 2) : min(after.width, int(np.ceil(x1)) + 2),
            ] = True
    outside = int(np.count_nonzero(diff & ~permitted))
    if outside:
        raise ValueError(f"Unexpected {outside} changed pixels outside f regions")
    return {
        "changed_pixels": int(diff.sum()),
        "outside_target_regions": outside,
        "changed_f_items": len(changed),
        "non_f_vector_items_and_provenance_exact": True,
    }


def panels_for_f(replay, records, image, destination, number):
    panels = []
    for old, new in zip(replay["records"], records, strict=True):
        if old["semantic"] != "f":
            continue
        boxes = np.array([old["bounds_px"], new["bounds_px"]])
        x0, y0 = np.floor(boxes[:, :2].min(0) - [75, 30]).astype(int)
        x1, y1 = np.ceil(boxes[:, 2:].max(0) + [90, 30]).astype(int)
        box = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
        a, b = replay["image"].crop(box), image.crop(box)
        paired_panel(
            a,
            b,
            destination / f"{old['item_index']:04d}.png",
            f"Page {number} / {old['context']} f item {old['item_index']}",
            ("Author-reviewed previous f0", "Bounded size candidate"),
        )
        panels.append((number, old["item_index"], a, b))
    return panels


def source_audit(font, destination):
    destination.mkdir(parents=True)
    for f in (
        SOURCE_FONT,
        SOURCE_SAMPLE / "template.json",
        SOURCE_SAMPLE / "sample.json",
        SOURCE_SAMPLE / "original-000.jpg",
        SOURCE_SAMPLE / "original-001.jpg",
    ):
        shutil.copyfile(f, destination / f.name)
    evidence = []
    source_y, reference_y, references = [], [], []
    for ch in COMMON_SOURCE["reference_characters"]:
        boxes = []
        for file in (SOURCE_FONT, font):
            v, c = handwritten_outline(file, ord(ch), 0, 1000)
            boxes.append(MplPath(v, c).get_extents())
        source_y.extend([boxes[0].ymin, boxes[0].ymax])
        reference_y.extend([boxes[1].ymin, boxes[1].ymax])
        references.append(
            {
                "character": ch,
                "source_box": boxes[0].extents.tolist(),
                "compiled_box": boxes[1].extents.tolist(),
            }
        )
    fitted = np.linalg.lstsq(
        np.column_stack([source_y, np.ones(len(source_y))]), reference_y, rcond=None
    )[0]
    if not np.allclose(
        fitted,
        [
            COMMON_SOURCE["source_to_reference_scale"],
            COMMON_SOURCE["source_to_reference_y_translation"],
        ],
        atol=1e-10,
    ):
        raise ValueError("Frozen common-source fit changed")
    atomic_json(
        destination / "common-source-fit.json",
        {
            "references": references,
            "measured_scale_and_translation": fitted.tolist(),
            "frozen_control": COMMON_SOURCE,
            "fit_is_shared_geometry_estimate_not_new_handwriting_truth": True,
        },
    )
    fig = Figure(figsize=(12, 7))
    FigureCanvasAgg(fig)
    for ri, file in enumerate((SOURCE_FONT, font)):
        for j, ch in enumerate("fhtdoa"):
            v, c = handwritten_outline(file, ord(ch), 0, 1000)
            p = MplPath(v, c)
            b = p.get_extents()
            evidence.append(
                {
                    "stage": "historical-traced-sample" if ri == 0 else "merged-profile",
                    "character": ch,
                    "bounds": b.extents.tolist(),
                    "height": b.height,
                    "width": b.width,
                    "font_sha256": sha256(file),
                }
            )
            ax = fig.add_subplot(2, 6, ri * 6 + j + 1)
            ax.add_patch(PathPatch(p, fc="black", lw=0))
            ax.set(
                xlim=(0, 600),
                ylim=(-250, 1000),
                aspect="equal",
                title=("Source " if ri == 0 else "Compiled ") + ch,
            )
            ax.axhline(0, c="blue", lw=0.5)
            ax.axhline(470, c="green", lw=0.5)
    fig.tight_layout()
    fig.savefig(destination / "source-versus-compiled.png", dpi=150)
    merged = []
    for ch in "fhtdoa":
        old = next(
            x for x in evidence if x["stage"] == "historical-traced-sample" and x["character"] == ch
        )
        new = next(x for x in evidence if x["stage"] == "merged-profile" and x["character"] == ch)
        scale = new["height"] / old["height"]
        merged.append(
            {
                "character": ch,
                "scale_y": scale,
                "scale_x": new["width"] / old["width"],
                "y_translation": new["bounds"][1] - scale * old["bounds"][1],
            }
        )
    (destination / "historical-merge-code.py").write_text(inspect.getsource(symbols._fit_glyph))
    atomic_json(
        destination / "source-metrics.json",
        {
            "metrics": evidence,
            "compiled_transforms": merged,
            "font_lineage": compiled_lineage(font),
            "notes": [
                "Historical sample TTF is an existing registered/traced asset, not raw pixels.",
                "The old tracing used threshold/morphology; no new tracing or raster edit occurs.",
                "Uniform enlargement preserves existing f contour aspect but is output geometry.",
                "Stem sections exclude the broad crossbar; not pen-pressure estimation.",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", choices=PRESETS)
    parser.add_argument("--probe-only", choices=PRESETS)
    parser.add_argument("--atomic-parent", type=Path)
    args = parser.parse_args()
    baseline = load_baseline()
    parent, parent_pages, source_groups, redraw = args.atomic_parent, {}, {}, set()
    if parent:
        verify_run(parent)
        receipt = json.loads((parent / "run.json").read_text())
        if (
            args.full != "D-common-source"
            or receipt["config"]["full_preset"] != args.full
            or receipt["config"]["baseline_sha"] != baseline["run_sha256"]
        ):
            raise ValueError("Atomic parent must be the same source-pinned full D experiment")
        for entry in baseline["cases"]:
            n = entry["page"]
            report = json.loads((parent / f"reports/{args.full}/page-{n:02d}.json").read_text())
            parent_pages[n] = report
            source = (
                baseline["path"] / f"solutions/inputs/{entry['case_id']}/solution.md"
            ).read_text()
            groups = consecutive_f_groups(source, entry["case_id"], report["before"])
            source_groups[n] = groups
            decisions = {d["item_index"]: d for d in report["decisions"]}
            for group in groups:
                flags = [decisions[i]["applied"] for i in group["item_indices"]]
                if any(flags) and not all(flags):
                    redraw.add(n)
                elif not all(flags):
                    raise ValueError(
                        "Previously fully rejected ff group needs explicit metadata replay"
                    )
    run = Run.create(
        "f-size-source-recovery",
        seed=baseline["cfg"].seed,
        config={
            "argv": sys.argv,
            "baseline": str(baseline["path"]),
            "baseline_sha": baseline["run_sha256"],
            "hypothesis": "per-glyph merge compressed descending f relative to adjacent letters",
            "presets": PRESETS,
            "crossbar_font_y": CROSSBAR_FONT_Y,
            "minimum_clearance_px": 0.60,
            "independent_safety_budget": SAFETY_BUDGET,
            "probe_pages": PROBES,
            "full_preset": args.full,
            "new_neural_updates": 0,
            "author_approval": False,
            "fresh_holdout": False,
            "atomic_parent": str(parent) if parent else None,
            "atomic_parent_sha256": sha256(parent / "run.json") if parent else None,
            "atomic_group_regenerated_pages": sorted(redraw),
        },
    )
    print(run.path, flush=True)
    try:
        protected = {
            str(workspace_root() / "latest.json"): sha256(workspace_root() / "latest.json"),
            str(baseline["path"] / "run.json"): sha256(baseline["path"] / "run.json"),
        }
        if parent:
            protected[str(parent / "run.json")] = sha256(parent / "run.json")
        atomic_json(run.path / "reports/protected.json", protected)
        for path in ROOT.glob("research/**/*.py"):
            dest = run.path / "logs/source" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        atomic_json(
            run.path / "reports/research-source-identity.json",
            {
                "parent_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True
                ),
                "source_sha256": {
                    str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("research/**/*.py")
                },
                "scope": "explicit full research snapshot; Run.build is core-runtime scope",
            },
        )
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline")
        source_audit(baseline["profile"].font, run.path / "solutions/f-source-evidence")
        audits = []
        results = []
        allpanels = {k: [] for k in PRESETS}
        outputs = {k: [] for k in PRESETS}
        copied = []
        order = list(PROBES) + [n for n in range(1, 21) if n not in PROBES]
        for n in order:
            print(f"page {n}", flush=True)
            if parent and n not in redraw:
                previous = parent_pages[n]
                out = run.path / f"rendered/{args.full}/pages/{n:02d}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ("pdf", "png"):
                    src = parent / f"rendered/{args.full}/pages/{n:02d}.{suffix}"
                    shutil.copyfile(src, out.with_suffix(f".{suffix}"))
                    hashes = [sha256(src), sha256(out.with_suffix(f".{suffix}"))]
                    if hashes[0] != hashes[1]:
                        raise ValueError("Unaffected page byte identity failed")
                    copied.append(
                        {
                            "page": n,
                            "kind": suffix,
                            "parent_sha": hashes[0],
                            "copied_sha": hashes[1],
                        }
                    )
                audit = json.loads((parent / f"reports/audit/page-{n:02d}.json").read_text())
                audits.extend(audit)
                atomic_json(run.path / f"reports/audit/page-{n:02d}.json", audit)
                groups = [
                    {**g, "atomic_action": "accept-whole-group", "original_failure_items": []}
                    for g in source_groups[n]
                ]
                report = {**previous, "atomic_groups": groups, "exact_parent_pdf_png_copy": True}
                atomic_json(run.path / f"reports/{args.full}/page-{n:02d}.json", report)
                results.append(
                    {k: v for k, v in report.items() if k not in {"before", "candidate", "parser"}}
                )
                outputs[args.full].append(out)
                before_image = Image.open(baseline["path"] / f"rendered/pages/{n:02d}.png").convert(
                    "RGB"
                )
                after_image = Image.open(out.with_suffix(".png")).convert("RGB")
                allpanels[args.full].extend(
                    panels_for_f(
                        {"records": previous["before"], "image": before_image},
                        previous["candidate"],
                        after_image,
                        run.path / f"comparisons/{args.full}/pages/{n:02d}",
                        n,
                    )
                )
                continue
            replay = replay_page(baseline, n)
            audit = audit_page(replay, n)
            audits.extend(audit)
            atomic_json(run.path / f"reports/audit/page-{n:02d}.json", audit)
            if not audit and args.full:
                out = run.path / f"rendered/{args.full}/pages/{n:02d}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ("pdf", "png"):
                    src = baseline["path"] / f"rendered/pages/{n:02d}.{suffix}"
                    shutil.copyfile(src, out.with_suffix(f".{suffix}"))
                    if sha256(src) != sha256(out.with_suffix(f".{suffix}")):
                        raise ValueError("No-f page must remain byte-identical")
                report = {
                    "page": n,
                    "preset": args.full,
                    "decisions": [],
                    "checks": {
                        "changed_pixels": 0,
                        "outside_target_regions": 0,
                        "changed_f_items": 0,
                        "exact_pdf_png_copy": True,
                    },
                }
                results.append(report)
                atomic_json(
                    run.path / f"reports/{args.full}/page-{n:02d}.json",
                    {
                        **report,
                        "before": replay["records"],
                        "candidate": replay["records"],
                        "parser": replay["parser"],
                    },
                )
                outputs[args.full].append(out)
            if not audit:
                continue
            presets = (
                [args.full]
                if args.full
                else [args.probe_only]
                if args.probe_only and n in PROBES
                else list(PRESETS)
                if not args.probe_only and n in PROBES
                else []
            )
            for preset in presets:
                page = copy.deepcopy(replay["page"])
                renderer = replay["renderer"]
                treatment = FSizeFeedback(
                    preset, source=replay["source"], title=replay["entry"]["case_id"]
                )
                rows = treatment.apply(page, renderer, replay["records"])
                if json_value(trace(renderer)) != replay["parser"]:
                    raise ValueError("Formula parser changed")
                path = run.path / f"rendered/{preset}/pages/{n:02d}.pdf"
                image = save_pdf(renderer.to_vector_pdf([page]), path)
                changed = {d["item_index"] for d in treatment.decisions if d["applied"]}
                checks = pixel_checks(replay["image"], image, replay["records"], rows, changed)
                report = {
                    "page": n,
                    "preset": preset,
                    "decisions": treatment.decisions,
                    "checks": checks,
                    "atomic_groups": treatment.groups,
                }
                results.append(report)
                atomic_json(
                    run.path / f"reports/{preset}/page-{n:02d}.json",
                    {
                        **report,
                        "before": replay["records"],
                        "candidate": rows,
                        "parser": replay["parser"],
                    },
                )
                allpanels[preset].extend(
                    panels_for_f(
                        replay, rows, image, run.path / f"comparisons/{preset}/pages/{n:02d}", n
                    )
                )
                outputs[preset].append(path)
                print(
                    preset, n, "accepted", len(changed), "/", len(treatment.decisions), flush=True
                )
        for preset in PRESETS:
            if not outputs[preset]:
                continue
            with PdfWriter() as writer:
                for path in sorted(outputs[preset]):
                    writer.append(path)
                writer.write(run.path / f"rendered/{preset}/homework.pdf")
            dest = run.path / f"comparisons/{preset}/all-f"
            dest.mkdir(parents=True)
            local_comparison(dest, allpanels[preset], labels=("Previous f0", preset))
        atomic_json(run.path / "reports/all-145-f-audit.json", audits)
        atomic_json(run.path / "reports/cases.json", results)
        atomic_json(run.path / "reports/copied-page-byte-identity.json", copied)
        summary = {
            "all_f_count": len(audits),
            "contexts": dict(Counter(r["context"] for r in audits)),
            "small_f_observed": sum(r["small_script"] for r in audits),
            "evaluations": {
                p: {
                    "accepted": sum(
                        d["applied"] for r in results if r["preset"] == p for d in r["decisions"]
                    ),
                    "rejected": sum(
                        not d["applied"]
                        for r in results
                        if r["preset"] == p
                        for d in r["decisions"]
                    ),
                }
                for p in PRESETS
            },
            "author_acceptance": False,
            "new_neural_updates": 0,
            "status": "isolated-size-candidates-awaiting-independent-visual-review",
            "full_pages": len(outputs[args.full]) if args.full else 0,
            "atomic_groups": [
                {"page": r["page"], **g} for r in results for g in r.get("atomic_groups", [])
            ],
            "retained_due_to_group_only": sum(
                d.get("group_rejection", {}).get(
                    "independently_accepted_before_group_policy", False
                )
                for r in results
                for d in r["decisions"]
            ),
            "rejected_occurrences": [
                {"page": r["page"], **d}
                for r in results
                for d in r["decisions"]
                if not d["applied"]
            ],
        }
        (run.path / "reports/F_SIZE_REVIEW.md").write_text(
            "# Personal f size audit and bounded recovery\n\n"
            "The earlier candidate already used real f0 in all 145 occurrences (140 text, "
            "5 math). Its height was 0.716 em, approximately the same as h/l/d; the author's "
            "historical traced f was 740 units high versus h 517, with a true descender. "
            "Handnote _fit_glyph scaled each incoming glyph to the replaced glyph's own "
            "height and vertical centre. This compressed f relative to the neighboring "
            "personal letters and moved its crossbar up.\n\n"
            "Measured lower f stem excludes the crossbar and hook: median about 1.499 px "
            "versus same-row h/l/d about 2.009 px. These are geometric ink widths, not "
            "measurements of pen pressure. Raw sources, historical trace, merge code, "
            "template and source-to-compiled contour comparison are retained.\n\n"
            "The first three controlled variants enlarged around the compiled crossbar. "
            "Root review identified the inherited crossbar anchor itself as incorrect. "
            "D uses one shared y transform fitted from h/l/d/a/c/e/o source coordinates: "
            "f top 719.468, bottom -314.456 units. This keeps the top near h and restores "
            "the lower crossbar and descender; the uniform output factor is 1.444028. "
            "It is disclosed output geometry, not an untouched scan or neural training.\n\n"
            "D was selected for full-page isolated review. Unsafe occurrences retain the "
            "original f explicitly; no B fallback is mixed in. Same-word ff groups are "
            "bound to actual source spans and exact visible-text order; if either f is "
            "rejected, the entire group stays original. Cross-word f pairs are excluded. "
            "Small-script f is held "
            "for separate evaluation and none occurs among these 145 samples. Frozen "
            "scale/displacement budgets, all nearby actual contour distances, above/below "
            "line clearance and simultaneous ff checks provide separate safety evidence. "
            "allowed_bounds_px denotes a target output envelope, not a pre-existing safe "
            "MathText slot. Non-f items, advances and semantic ordering are unchanged.\n\n"
            "GO for isolated review where accepted; NO-GO for source/font replacement or "
            "production activation. Author style acceptance remains pending. Previous "
            "failed probes and original scans are unchanged.\n"
        )
        if any(sha256(Path(p)) != h for p, h in protected.items()):
            raise ValueError("Protected asset changed")
        run.finish("held", result=summary)
        verify_run(run.path)
        print(
            {
                k: v
                for k, v in summary.items()
                if k not in {"rejected_occurrences", "atomic_groups"}
            },
            flush=True,
        )
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
