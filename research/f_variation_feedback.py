"""Real f source diversity audit; immutable original ink, no synthetic variants.

The initial audit does not admit any source to page rendering. Font slots are
counted by their independent GSUB glyphs, never by modulo-wrapped indices.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from handnote import symbols
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.author_feedback_delivery import crop_pair
from research.author_feedback_revision import trace
from research.author_feedback_round2_checks import page_geometry
from research.author_feedback_round3 import (
    BASE_RUN,
    BASE_SHA,
    load_baseline,
    replay_page,
    validate_increment,
)
from research.author_feedback_round3_delivery import f_words, group_box
from research.corpus.math_glyph_completion import local_comparison
from research.f_size_feedback import (
    COMMON_SOURCE,
    SOURCE_FONT,
    SOURCE_SAMPLE,
    consecutive_f_groups,
    contour_distance,
    stem_measure,
)
from research.homework_benchmark.render import save_pdf
from research.math_feedback_candidate import paired_panel

ROOT = Path(__file__).resolve().parents[1]
GLYPHS = (
    ROOT
    / "workspace/runs/20261120T013257031389Z-semantic-canvas-real-data-a1d6bf10"
    / "reports/glyph-samples.json"
)
WORDS = (
    ROOT
    / "workspace/runs/20261119T094439834081Z-whole-word-style-data-81a8c810/reports/samples.json"
)
POLICY = {
    "minimum_new_clearance_px": 0.60,
    "maximum_side_extension_em": 0.16,
    "maximum_top_extension_em": 0.18,
    "maximum_bottom_extension_em": 0.10,
    "minimum_page_clearance_px": 3.0,
    "minimum_actual_em_fraction": 0.80,
    "maximum_height_over_approved_D": 1.20,
    "original_30_held_instances": (
        "retain exact approved baseline; no attempt to bypass previous safety rejection"
    ),
    "same_word_ff": "one source family per group; any failure retains whole group baseline",
}
SELECTION = {
    "schema": "real-f-source-choice/1",
    "seed": 1909,
    "choices": ["baseline-f0", "sheet-f1"],
    "nominal_probability_each": 0.5,
    "method": (
        "SHA256(seed/title/actual word span or mathematical LaTeX context), first 64 bits mod 2"
    ),
    "same_word_all_f_share_source": True,
    "same_word_ff_atomic_admission": True,
    "seed_or_distribution_search": False,
    "historical_final": (
        "No-Go: genuine large/wide style incompatible with this layout and ink weight"
    ),
    "red_historical_sources": "Held: no density-enhanced derivatives admitted",
    "source_and_local_review_run": "20261120T123831414373Z-f-real-source-variation-probe-af65bccb",
}


def read(path):
    return json.loads(Path(path).read_text())


def registered_source(original, spec):
    """Repeat the recorded homography, retaining the actual RGB originals."""
    rgb = np.asarray(Image.open(original).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = min(1.0, 1800.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(symbols.MARKER_DICT), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(small)
    known = {m["id"]: m for m in spec["markers"]}
    by_sheet = {}
    for quad, identifier in zip(corners, ids.ravel(), strict=True):
        m = known.get(int(identifier))
        if m:
            by_sheet.setdefault(m["sheet"], []).append((quad.reshape(4, 2) / scale, m["points"]))
    sheet, matches = max(by_sheet.items(), key=lambda entry: len(entry[1]))
    matrix, _ = cv2.findHomography(
        np.concatenate([p for p, _ in matches]).astype(np.float32),
        np.array([p for _, ps in matches for p in ps], np.float32),
        cv2.RANSAC,
        4.0,
    )
    registered = cv2.warpPerspective(
        rgb,
        matrix,
        tuple(spec["page_size_px"]),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    return rgb, registered, matrix, sheet


def source_audit(destination):
    destination.mkdir(parents=True)
    spec = read(SOURCE_SAMPLE / "template.json")
    sample = read(SOURCE_SAMPLE / "sample.json")
    if sample["state"] != "accepted":
        raise ValueError("Source sample is not accepted")
    for entry in sample["files"]:
        if sha256(SOURCE_SAMPLE / entry["path"]) != entry["sha256"]:
            raise ValueError("Original source changed")
    for name in ("sample.json", "template.json"):
        shutil.copyfile(SOURCE_SAMPLE / name, destination / name)
    shutil.copyfile(SOURCE_FONT, destination / "source-sample.ttf")
    cell = next(c for c in spec["cells"] if c["char"] == "f")
    sources, panels = [], []
    for slot in (0, 1):
        original = SOURCE_SAMPLE / f"original-{slot:03d}.jpg"
        raw, registered, matrix, sheet = registered_source(original, spec)
        if sheet != slot:
            raise ValueError("Source sheet order changed")
        box = cell["bbox"]
        corners = np.array(
            [[[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]], np.float32
        )
        polygon = cv2.perspectiveTransform(corners, np.linalg.inv(matrix))[0]
        raw_box = [
            int(np.floor(polygon[:, 0].min())),
            int(np.floor(polygon[:, 1].min())),
            int(np.ceil(polygon[:, 0].max())),
            int(np.ceil(polygon[:, 1].max())),
        ]
        out = destination / f"sheet-{slot}"
        out.mkdir()
        shutil.copyfile(original, out / original.name)
        Image.fromarray(raw).crop(raw_box).save(out / "raw-rgb-cell.png")
        rgb_cell = Image.fromarray(registered).crop(box)
        rgb_cell.save(out / "registered-rgb-cell.png", dpi=(300, 300))
        vertices, codes = handwritten_outline(SOURCE_FONT, ord("f"), slot, 1000)
        path = MplPath(vertices, codes)
        target = MplPath(
            vertices
            * (
                COMMON_SOURCE["source_to_reference_scale"] * 0.046,
                -COMMON_SOURCE["source_to_reference_scale"] * 0.046,
            ),
            codes,
        )
        row = {
            "id": f"sheet-f{slot}",
            "kind": "accepted-collection-sheet",
            "original_label": "f",
            "original": str(original),
            "original_sha256": sha256(original),
            "sample_state": sample["state"],
            "sheet": sheet,
            "template_sha256": sha256(SOURCE_SAMPLE / "template.json"),
            "canonical_cell_bbox_300dpi": box,
            "raw_rgb_bbox": raw_box,
            "raw_rgb_polygon": polygon.tolist(),
            "raw_to_canonical_homography": matrix.tolist(),
            "raw_image_size": [raw.shape[1], raw.shape[0]],
            "canonical_rgb_derivative": (
                "INTER_LINEAR homography; original RGB crop separately unchanged"
            ),
            "font_source_path": str(SOURCE_FONT),
            "font_source_sha256": sha256(SOURCE_FONT),
            "source_glyph": "uni0066" if slot == 0 else "uni0066.s1",
            "font_variant": slot,
            "source_font_bounds": path.get_extents().extents.tolist(),
            "source_tracing": (
                "Existing Handnote threshold/morphology/guide alignment, not raw RGB exact contour"
            ),
            "common_source_46px_em_stem": stem_measure(target),
            "render_admission": "baseline-control" if slot == 0 else "pending-independent-review",
        }
        atomic_json(out / "source.json", row)
        np.savez(out / "source-outline.npz", vertices=vertices, codes=codes)
        sources.append(row)
        panels.append((f"Sheet f{slot}: original registered RGB", rgb_cell, path))
    words = {r["crop"]: r for r in read(WORDS)}
    historical = [r for r in read(GLYPHS) if r["label"] == "f"]
    if [r["id"] for r in historical] != ["support-079", "rare-05-0", "rare-06-2"]:
        raise ValueError("Frozen accepted historical f inventory changed")
    for row in historical:
        parent = words[row["source_crop"]]
        if parent["dataset_split"] != "train":
            raise ValueError("Only exposed training sources are authorized")
        original = Path(row["source_crop"])
        if sha256(original) != row["source_crop_sha256"]:
            raise ValueError("Historical RGB source changed")
        out = destination / row["id"]
        out.mkdir()
        shutil.copyfile(original, out / "word-original-rgb.png")
        shutil.copyfile(parent["source_crop"], out / "parent-original-rgb.png")
        trans = row["transform"]
        bb = np.asarray(row["raw_normalized_bbox"], float).reshape(2, 2)
        source_box_float = (bb - trans["translation"]) / trans["scale"] + np.asarray(
            trans["bbox"][:2]
        )
        source_box = [
            *np.floor(source_box_float[0]).astype(int),
            *np.ceil(source_box_float[1]).astype(int),
        ]
        source_box = [int(v) for v in source_box]
        image = Image.open(original).convert("RGB")
        image.crop(source_box).save(out / "f-original-rgb.png", dpi=(300, 300))
        raw = np.asarray(image.crop(source_box)).astype(float)
        if row["id"] == "support-079":
            mask = raw.min(axis=2) < 110
            threshold = "min(R,G,B)<110; no dilation/morphology"
        else:
            mask = (raw[:, :, 0] - np.minimum(raw[:, :, 1], raw[:, :, 2]) > 25) & (
                raw[:, :, 1] < 180
            )
            threshold = "R-min(G,B)>25 and G<180; no dilation/morphology"
        Image.fromarray(np.uint8(mask) * 255).save(out / "ink-segmentation-preview.png")
        evidence = {
            **row,
            "kind": "accepted-historical-word-source",
            "word_source_record": parent,
            "raw_rgb_f_box_in_word": source_box,
            "derived_segmentation": threshold,
            "before_density_source_only": True,
            "training_image_not_used": True,
            "render_admission": "pending-source-review-not-font-admitted",
            "source_character_label_unchanged": True,
        }
        atomic_json(out / "source.json", evidence)
        sources.append(evidence)
        panels.append((f"{row['id']}: {row['word']} (raw RGB)", image, None))
    fig = Figure(figsize=(15, 13))
    FigureCanvasAgg(fig)
    for i, (title, rgb, path) in enumerate(panels):
        ax = fig.add_subplot(5, 2, 2 * i + 1)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
        ax = fig.add_subplot(5, 2, 2 * i + 2)
        if path is not None:
            ax.add_patch(PathPatch(path, facecolor="black", linewidth=0))
            ax.set(
                xlim=(0, 500),
                ylim=(-250, 650),
                aspect="equal",
                title="Existing source-font contour, common coordinates",
            )
            ax.axhline(0, color="blue", lw=0.5)
            ax.axhline(354, color="green", lw=0.5)
        else:
            out = destination / historical[i - 2]["id"]
            ax.imshow(Image.open(out / "f-original-rgb.png"))
            ax.set_title("Unmodified raw f crop (no density gain)")
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination / "all-source-contact.png", dpi=160)
    fig.savefig(destination / "all-source-contact.pdf")
    atomic_json(destination / "inventory.json", sources)
    return sources


def historical_page_chain(row):
    parent = row["word_source_record"]
    path = Path(parent["source_crop"]).parents[1] / "source-pages.json"
    manifest = read(path)
    if row["id"] == "support-079":
        page = manifest["pages"]["cal3-186"]
        document = page
    else:
        page = manifest["pages"][f"ics-{row['page_number']}"]
        document = manifest["documents"]["ics"]
    if document["split"] != "train" or sha256(Path(document["source_pdf"])) != row["source_hash"]:
        raise ValueError("Historical source document/split mismatch")
    if sha256(Path(page["page_image"])) != page["page_sha256"]:
        raise ValueError("Historical source page changed")
    page_rgb = Image.open(page["page_image"]).convert("RGB")
    parent_rgb = Image.open(parent["source_crop"]).convert("RGB")
    if page_rgb.crop(parent["parent_bbox"]).tobytes() != parent_rgb.tobytes():
        raise ValueError("Parent RGB does not reproduce from recorded page coordinates")
    origin = np.asarray(parent["parent_bbox"][:2]) + np.asarray(parent["box_in_source_crop"][:2])
    fbox = np.asarray(row["raw_rgb_f_box_in_word"]).reshape(2, 2) + origin
    return {
        "source_manifest": str(path),
        "source_manifest_sha256": sha256(path),
        "document": document,
        "page": page,
        "parent_pixel_identity_verified": True,
        "f_box_on_300dpi_page": fbox.reshape(-1).tolist(),
        "f_box_pdf_points": (fbox.reshape(-1) * 72 / 300).tolist(),
        "coordinate_origin": "top-left; PDF point coordinates in rendered orientation",
    }


def prepare_candidates(destination, font):
    sources = source_audit(destination)
    for row in sources[2:]:
        row["pdf_coordinate_chain"] = historical_page_chain(row)
        atomic_json(destination / row["id"] / "source.json", row)
    # A second sheet uses its own common author coordinates, not f0's crossbar.
    src, dst, refs = [], [], []
    for ch in COMMON_SOURCE["reference_characters"]:
        v, c = handwritten_outline(SOURCE_FONT, ord(ch), 1, 1000)
        sb = MplPath(v, c).get_extents()
        v, c = handwritten_outline(font, ord(ch), 0, 1000)
        db = MplPath(v, c).get_extents()
        src.extend([sb.ymin, sb.ymax])
        dst.extend([db.ymin, db.ymax])
        refs.append(
            {"character": ch, "source": sb.extents.tolist(), "reference": db.extents.tolist()}
        )
    scale, shift = np.linalg.lstsq(np.column_stack([src, np.ones(len(src))]), dst, rcond=None)[0]
    v, c = handwritten_outline(SOURCE_FONT, ord("f"), 1, 1000)
    v = v * scale
    v[:, 1] += shift
    specifications = [
        {
            "id": "sheet-f1",
            "source": sources[1],
            "vertices": v,
            "codes": c,
            "geometry": {
                "method": "same-sheet common h/l/d/a/c/e/o coordinate fit",
                "scale": float(scale),
                "y_translation": float(shift),
                "references": refs,
                "fit_rmse_font_units": float(
                    np.sqrt(np.mean((np.asarray(src) * scale + shift - dst) ** 2))
                ),
                "uniform_scale_no_stroke_dilation": True,
            },
        }
    ]
    row = sources[2]
    rgb = np.asarray(Image.open(destination / row["id"] / "word-original-rgb.png"))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((rgb.min(2) < 110).astype(np.uint8))
    # Named source components, frozen by exact original coordinates. No generated ink.
    expected = {
        "f": [2, 3, 100, 176, 2742],
        "l": [322, 20, 13, 113, 1164],
        "n": [168, 69, 47, 59, 1154],
        "a": [237, 69, 52, 55, 1375],
    }
    component = {}
    for ch, match in expected.items():
        found = [i for i in range(1, count) if stats[i].tolist() == match]
        if len(found) != 1:
            raise ValueError(f"Frozen final source component changed: {ch}")
        component[ch] = found[0]
    src, dst, refs = [], [], []
    for ch in "lna":
        _, y, _, h, _ = expected[ch]
        v, c = handwritten_outline(font, ord(ch), 0, 1000)
        box = MplPath(v, c).get_extents()
        src.extend([-(y + h - 1), -y])
        dst.extend([box.ymin, box.ymax])
        refs.append(
            {
                "character": ch,
                "original_component": expected[ch],
                "reference_bounds": box.extents.tolist(),
            }
        )
    scale, shift = np.linalg.lstsq(np.column_stack([src, np.ones(len(src))]), dst, rcond=None)[0]
    contours, _ = cv2.findContours(
        np.uint8(labels == component["f"]), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    vs, cs = [], []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        vs.extend(points.tolist() + [points[0].tolist()])
        cs.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1) + [MplPath.CLOSEPOLY])
    v, c = np.asarray(vs), np.asarray(cs, np.uint8)
    v *= [scale, -scale]
    v[:, 1] += shift
    specifications.append(
        {
            "id": "support-079",
            "source": row,
            "vertices": v,
            "codes": c,
            "geometry": {
                "method": (
                    "whole-word l/n/a common coordinate fit; f is untouched segmented component"
                ),
                "source_rgb_threshold": "min(R,G,B)<110",
                "component_stats": expected["f"],
                "scale": float(scale),
                "y_translation": float(shift),
                "references": refs,
                "fit_rmse_font_units": float(
                    np.sqrt(np.mean((np.asarray(src) * scale + shift - dst) ** 2))
                ),
                "uniform_scale_no_stroke_dilation": True,
                "new_vectorization_is_not_original_pixels": True,
            },
        }
    )
    candidates = []
    for spec in specifications:
        v, c = spec.pop("vertices"), spec.pop("codes")
        path = MplPath(v, c)
        out = (
            destination
            / ("sheet-1" if spec["id"] == "sheet-f1" else spec["id"])
            / "candidate-outline.npz"
        )
        np.savez(out, vertices=v, codes=c)
        spec.update(
            outline=str(out.relative_to(destination)),
            outline_sha256=sha256(out),
            bounds_font_units=path.get_extents().extents.tolist(),
            lower_stem_at_46px_em=stem_measure(MplPath(v * (0.046, -0.046), c)),
        )
        candidates.append(spec)
    atomic_json(destination / "candidates.json", candidates)
    atomic_json(destination / "inventory.json", sources)
    return candidates


class FVariationFeedback:
    """Only reviewed real sources; the approved f0 and held groups are fallback."""

    def __init__(self, source_dir, *, source, title, seed=1909, candidate="balanced"):
        self.source_dir = Path(source_dir)
        self.source, self.title, self.seed = source, title, seed
        self.candidate = candidate
        self.candidates = {r["id"]: r for r in read(self.source_dir / "candidates.json")}
        self.outlines = {}
        for key, row in self.candidates.items():
            path = self.source_dir / row["outline"]
            if sha256(path) != row["outline_sha256"]:
                raise ValueError("Reviewed candidate outline changed")
            arrays = np.load(path)
            self.outlines[key] = MplPath(arrays["vertices"], arrays["codes"])
        if (
            sha256(self.source_dir / "source-sample.ttf")
            != self.candidates["sheet-f1"]["source"]["font_source_sha256"]
        ):
            raise ValueError("Real source font changed")
        self.decisions, self.groups = [], []

    def apply(self, page, renderer, records):
        self.groups = consecutive_f_groups(self.source, self.title, records)
        group_for = {i: g for g in self.groups for i in g["item_indices"]}
        words = f_words(self.source, self.title, records)
        word_for = {i: w for w in words for i in w["items"]}
        result = copy.deepcopy(records)
        items = list(page.items)
        paths = page_geometry(page, renderer)
        proposals = {}
        self.decisions = []
        for row in records:
            if row["semantic"] != "f":
                continue
            i = row["item_index"]
            group = group_for.get(i)
            word = word_for.get(i)
            choice_key = (
                f"word/{word['span']}/{word['range']}/{word['word']}"
                if word
                else f"math/{row.get('latex')}/{row.get('matrix_environment')}"
            )
            digest = hashlib.sha256(f"{self.seed}/{self.title}/{choice_key}".encode()).hexdigest()
            selected = self.candidate
            if selected == "balanced":
                selected = ("baseline-f0", "sheet-f1")[int(digest[:16], 16) % 2]
            decision = {
                "item_index": i,
                "context": row["context"],
                "selected_source": selected,
                "selection_sha256": digest,
                "selection_context": choice_key,
                "group_id": group["group_id"] if group else None,
                "applied": False,
                "rejections": [],
                "safety_policy": POLICY,
            }
            self.decisions.append(decision)
            previous = row.get("f_size_feedback", {})
            if previous.get("status") != "research-candidate":
                decision["rejections"].append("Previously held f is exact baseline fallback")
                continue
            if selected == "baseline-f0":
                decision["retained_by_fixed_source_selection"] = True
                continue
            if selected not in self.candidates:
                raise ValueError("Unknown source candidate")
            old = paths[i]
            box = old.get_extents()
            prebox = previous["original"]["bounds_px"]
            em = (prebox[3] - prebox[1]) * 1000 / 716
            if em < 0.8 * renderer.cfg.body_size:
                decision["rejections"].append("Small-script variant requires separate protocol")
                continue
            source_path = self.outlines[selected]
            sb = source_path.get_extents()
            center = (box.xmin + box.xmax) / 2
            baseline_y = prebox[1] + 715 / 1000 * em
            v = source_path.vertices * (em / 1000, -em / 1000)
            v += (center - (sb.xmin + sb.xmax) / 2 * em / 1000, baseline_y)
            proposed = MplPath(v, source_path.codes)
            bb = proposed.get_extents()
            extension = (
                np.maximum(
                    [
                        box.xmin - bb.xmin,
                        box.ymin - bb.ymin,
                        bb.xmax - box.xmax,
                        bb.ymax - box.ymax,
                    ],
                    0,
                )
                / em
            )
            decision.update(
                actual_em_px=em,
                original_bounds=box.extents.tolist(),
                candidate_bounds=bb.extents.tolist(),
                extension_em=extension.tolist(),
                original_lower_stem=stem_measure(old),
                candidate_lower_stem=stem_measure(proposed),
            )
            if (
                max(extension[0], extension[2]) > 0.16
                or extension[1] > 0.18
                or extension[3] > 0.10
                or bb.height / box.height > 1.20
            ):
                decision["rejections"].append("Frozen source-size displacement budget exceeded")
            if min(bb.xmin, bb.ymin, page.width - bb.xmax, page.height - bb.ymax) < 3:
                decision["rejections"].append("Page boundary")
            neighbors = []
            for j, other in enumerate(paths):
                ob = other.get_extents()
                dx = max(0, ob.xmin - bb.xmax, bb.xmin - ob.xmax)
                dy = max(0, ob.ymin - bb.ymax, bb.ymin - ob.ymax)
                if i == j or np.hypot(dx, dy) > 2 * em:
                    continue
                before, after = contour_distance(old, other), contour_distance(proposed, other)
                neighbors.append(
                    {
                        "item_index": j,
                        "semantic": records[j]["semantic"],
                        "before_px": before,
                        "after_px": after,
                    }
                )
                if after < 0.60 and after < before - 0.05:
                    decision["rejections"].append(f"Neighbor {j} clearance {after:.3f}px")
            decision["neighbors"] = neighbors
            if not decision["rejections"]:
                proposals[i] = proposed
        by_index = {r["item_index"]: r for r in self.decisions}
        # Simultaneous new sources must retain all minimum distances too.
        for i, a in list(proposals.items()):
            for j, b in list(proposals.items()):
                aa, bb = a.get_extents(), b.get_extents()
                dx = max(0, aa.xmin - bb.xmax, bb.xmin - aa.xmax)
                dy = max(0, aa.ymin - bb.ymax, bb.ymin - aa.ymax)
                if i >= j or np.hypot(dx, dy) >= 0.60:
                    continue
                distance = contour_distance(a, b)
                if distance < 0.60 and distance < contour_distance(paths[i], paths[j]) - 0.05:
                    for k in (i, j):
                        by_index[k]["rejections"].append("Simultaneous candidate f clearance")
        for group in self.groups:
            ids = group["item_indices"]
            failed = [i for i in ids if by_index[i]["rejections"]]
            if failed:
                for i in ids:
                    if not by_index[i]["rejections"]:
                        by_index[i]["rejections"].append("Atomic same-word ff group fallback")
                    by_index[i]["group_rejection"] = {
                        "original_failure_items": failed,
                        "word": group["word"],
                    }
                group["action"] = "retain-entire-baseline-group"
            else:
                group["action"] = "apply-one-source-family-or-retain-f0-selection"
        for i, path in proposals.items():
            decision = by_index[i]
            if decision["rejections"]:
                continue
            source_id = decision["selected_source"]
            spec = self.candidates[source_id]
            key = ("f-real-variation", source_id, i)
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), items[i][3], 0)
            bb = path.get_extents().extents.tolist()
            result[i].update(
                bounds_px=bb,
                allowed_bounds_px=bb,
                source_type="personal-outline-geometric-calibration",
                variant=1 if source_id == "sheet-f1" else None,
                actual_glyph_id=None,
                glyph_name=spec["source"].get("source_glyph", "historical-word-final-f"),
                font=str(self.source_dir / "source-sample.ttf")
                if source_id == "sheet-f1"
                else None,
                font_sha256=spec["source"].get("font_source_sha256"),
                f_variation_feedback={
                    "status": "isolated-source-candidate",
                    "source_id": source_id,
                    "source": spec["source"],
                    "geometry": spec["geometry"],
                    "outline_sha256": spec["outline_sha256"],
                    "baseline": copy.deepcopy(records[i]),
                    "output_envelope_scope": (
                        "target geometry only; measured clearance and fixed budgets "
                        "independently gate"
                    ),
                    "neural_updates": 0,
                    "synthetic_noise": False,
                    "ink_dilation": False,
                },
            )
            decision["applied"] = True
        for decision in self.decisions:
            decision["actual_source"] = (
                decision["selected_source"] if decision["applied"] else "baseline-f0"
            )
        return result


def probe_run(full=False):
    baseline = load_baseline()
    run = Run.create(
        "f-real-source-variation-probe",
        seed=1909,
        config={
            "baseline": BASE_RUN,
            "baseline_sha256": BASE_SHA,
            "admitted_sources": (
                ["baseline-f0", "sheet-f1"] if full else ["sheet-f1", "support-079"]
            ),
            "audited_sources": ["sheet-f0", "sheet-f1", "support-079", "rare-05-0", "rare-06-2"],
            "policy": POLICY,
            "pages": list(range(1, 21)) if full else [1, 6, 15, 20],
            "mode": "full-balanced" if full else "two-fixed-source-probes",
            "neural_updates": 0,
            "render": baseline["cfg"].model_dump(mode="json"),
            "selection_protocol": SELECTION,
        },
    )
    print(run.path, flush=True)
    try:
        for path in ROOT.glob("research/**/*.py"):
            dest = run.path / "logs/source" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        atomic_json(
            run.path / "reports/source-identity.json",
            {
                "parent_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "git_status": subprocess.check_output(
                    ["git", "status", "--short", "--untracked-files=all"], text=True
                ),
                "research_source_sha256": {
                    str(p.relative_to(ROOT)): sha256(p) for p in ROOT.glob("research/**/*.py")
                },
            },
        )
        prepare_candidates(run.path / "solutions/sources", baseline["profile"].font)
        atomic_json(run.path / "reports/selection-protocol.json", SELECTION)
        shutil.copytree(baseline["path"] / "solutions", run.path / "solutions/baseline")
        cases = []
        panels = {}
        pdfs = {}
        word_panels = {}
        modes = ["balanced"] if full else ["sheet-f1", "support-079"]
        for n in range(1, 21) if full else (1, 6, 15, 20):
            replay = replay_page(baseline, n)
            print("page", n, flush=True)
            for mode in modes:
                page = copy.deepcopy(replay["page"])
                renderer = replay["renderer"]
                before_paths = page_geometry(page, renderer)
                treatment = FVariationFeedback(
                    run.path / "solutions/sources",
                    source=replay["source"],
                    title=replay["entry"]["case_id"],
                    candidate=mode,
                )
                rows = treatment.apply(page, renderer, replay["records"])
                output = run.path / f"rendered/{mode}/pages/{n:02d}.pdf"
                image = save_pdf(renderer.to_vector_pdf([page]), output)
                checks = validate_increment(
                    before_paths, page, renderer, replay["records"], rows, replay["image"], image
                )
                if json.loads(json.dumps(trace(renderer))) != replay["parser"]:
                    raise ValueError("Parser trace changed")
                case = {
                    "page": n,
                    "mode": mode,
                    "decisions": treatment.decisions,
                    "groups": treatment.groups,
                    "checks": checks,
                    "case_id": replay["entry"]["case_id"],
                }
                cases.append(case)
                atomic_json(
                    run.path / f"reports/{mode}/page-{n:02d}.json",
                    {
                        **case,
                        "before": replay["records"],
                        "candidate": rows,
                        "parser": replay["parser"],
                    },
                )
                dest = run.path / f"solutions/inputs/{replay['entry']['case_id']}"
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "solution.md").write_text(replay["source"])
                panels.setdefault(mode, [])
                pdfs.setdefault(mode, []).append(output)
                for decision in treatment.decisions:
                    i = decision["item_index"]
                    boxes = np.array([replay["records"][i]["bounds_px"], rows[i]["bounds_px"]])
                    a, b = np.floor(boxes[:, :2].min(0) - [125, 35]).astype(int)
                    c, d = np.ceil(boxes[:, 2:].max(0) + [160, 35]).astype(int)
                    box = (max(0, a), max(0, b), min(image.width, c), min(image.height, d))
                    old, new = replay["image"].crop(box), image.crop(box)
                    paired_panel(
                        old,
                        new,
                        run.path / f"comparisons/{mode}/page-{n:02d}/{i:04d}.png",
                        f"P{n} item {i}: {decision['selected_source']} / "
                        f"{'APPLIED' if decision['applied'] else 'BASELINE'}",
                        ("Approved f0", "Real-source variation"),
                    )
                    panels[mode].append((n, i, old, new))
                words = f_words(replay["source"], replay["entry"]["case_id"], replay["records"])
                word_panels.setdefault(mode, [])
                for wi, word in enumerate(words):
                    old, new = crop_pair(
                        group_box(replay["records"], word["items"]),
                        group_box(rows, word["items"]),
                        replay["image"],
                        image,
                        padding=(18, 22),
                    )
                    paired_panel(
                        old,
                        new,
                        run.path / f"comparisons/{mode}/words/{n:02d}-{wi:03d}.png",
                        f"P{n} {word['word']} / items {word['items'][0]}..{word['items'][-1]}",
                        ("Approved f0", "Real-source variation"),
                    )
                    word_panels[mode].append((n, wi, old, new))
                atomic_json(run.path / f"reports/{mode}/words-{n:02d}.json", words)
                print(
                    mode,
                    sum(d["applied"] for d in treatment.decisions),
                    "/",
                    len(treatment.decisions),
                    flush=True,
                )
        for mode, paths in pdfs.items():
            with PdfWriter() as writer:
                for path in paths:
                    writer.append(path)
                writer.write(run.path / f"rendered/{mode}/homework.pdf")
            dest = run.path / f"comparisons/{mode}/all-f"
            dest.mkdir(parents=True)
            local_comparison(dest, panels[mode], labels=("Approved f0", mode))
            dest = run.path / f"comparisons/{mode}/all-words"
            dest.mkdir(parents=True)
            local_comparison(dest, word_panels[mode], labels=("Approved f0", mode))
        atomic_json(run.path / "reports/cases.json", cases)
        result = {
            "status": "root-review-pending",
            "source_counts": dict(
                Counter(
                    d["selected_source"]
                    for case in cases
                    for d in case["decisions"]
                    if d["applied"]
                )
            ),
            "all_actual_source_counts": dict(
                Counter(d["actual_source"] for case in cases for d in case["decisions"])
            ),
            "all_rejections": dict(
                Counter(
                    reason
                    for case in cases
                    for d in case["decisions"]
                    for reason in d["rejections"]
                )
            ),
            "all_non_f_pixels_and_geometry_protected": True,
            "author_acceptance": False,
            "pages": len(cases),
            "source_code_sha256": sha256(Path(__file__)),
        }
        run.finish("held", result=result)
        verify_run(run.path)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.probe or args.full:
        probe_run(args.full)
        return
    if not args.audit_only:
        parser.error("Only source auditing is authorized before independent source review")
    run = Run.create(
        "f-real-variation-source-audit",
        seed=1909,
        config={
            "baseline": BASE_RUN,
            "baseline_sha256": BASE_SHA,
            "mode": "source-audit-only",
            "maximum_nominated_alternates": 3,
            "neural_updates": 0,
        },
    )
    try:
        shutil.copyfile(__file__, run.path / "logs/f_variation_feedback.py")
        sources = source_audit(run.path / "solutions/sources")
        result = {
            "status": "source-review-pending",
            "independent_collection_f": 2,
            "independent_historical_f": 3,
            "sources": [r["id"] for r in sources],
            "variants_admitted_to_pages": 0,
            "source_code_sha256": sha256(Path(__file__)),
        }
        run.finish("held", result=result)
        verify_run(run.path)
        print(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
