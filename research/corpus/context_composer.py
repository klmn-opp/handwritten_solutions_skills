"""Experimental frozen-glyph composition in Handnote's unmodified layout recorder.

This is NOT a learned word/line renderer. CPU-only composition consumes previously
generated ink; no Torch imports, model training, default Profile or latest writes.
"""

import copy
import hashlib
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
from handnote.markdown import Block
from handnote.vector import Outline
from matplotlib import rc_context
from matplotlib.path import Path as MplPath

from hwsol.adapters.handnote import RENDER_LOCK, upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.handwriting import finish_ink
from hwsol.jobs.layout import MathDiagnostics, SinglePageRenderer, ink_geometry, validate_markdown
from hwsol.jobs.math_strokes import apply_math_strokes
from hwsol.jobs.provenance import classify_rules, item_bounds, visible_provenance
from hwsol.profiles.io import sha256
from hwsol.workspace import verify_run
from research.corpus.reference_evaluate import character_metrics, shape_distance


def mask_outline(mask: np.ndarray) -> Outline:
    """Exact foreground row-run polygons, including tiny dots and all holes.

    Binary 96px image vectorization is NOT recovery of online pen trajectories.
    Pixel-edge geometry avoids contour simplification deleting punctuation.
    """
    if mask.ndim != 2 or not mask.any():
        raise ValueError("Empty/invalid neural ink mask")
    vertices, codes = [], []
    for y, row in enumerate(mask):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        for left, right in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
            vertices.extend([(left, y), (right, y), (right, y + 1), (left, y + 1), (left, y)])
            codes.extend([1, 2, 2, 2, 79])
    return Outline(np.array(vertices, dtype=float), np.array(codes, dtype=np.uint8), False, 4)


def foreground_bounds(ink: np.ndarray) -> np.ndarray:
    y, x = np.where(ink > 0.5)
    if not len(x):
        raise ValueError("Empty glyph")
    return np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], dtype=float)


class FrozenBank:
    def __init__(self, sample_run: Path):
        verify_run(sample_run)
        self.path = sample_run.resolve()
        self.request = json.loads((sample_run / "reports/request.json").read_text())
        checkpoint = Path(self.request["checkpoint"])
        if sha256(checkpoint) != self.request["checkpoint_sha256"]:
            raise ValueError("Model checkpoint changed")
        self.ink = np.load(sample_run / "reports/generated.npz", allow_pickle=False)["ink"]
        data = Path(self.request["data"])
        manifest = json.loads((data / "manifest.json").read_text())
        self.labels = manifest["labels"]
        arrays = np.load(data / "arrays.npz", allow_pickle=False)
        self.reference = arrays["reference"]
        self.fonts = arrays["target"][arrays["domains"] == 0]
        self.font_labels = arrays["classes"][arrays["domains"] == 0]
        self.gates = {}

    def select(self, ch: str, identity: str, item: int) -> tuple | None:
        if ch not in self.labels:
            return None
        index = self.labels.index(ch)
        # Same schedule across methods; not best-of-N quality selection.
        seed = hashlib.sha256(f"1010|{identity}|{item}|{ch}".encode()).digest()
        variant = int.from_bytes(seed[:4], "big") % self.ink.shape[1]
        key = (index, variant)
        if key not in self.gates:
            ink = self.ink[index, variant]
            metrics = character_metrics(ink[None], self.fonts, self.font_labels, index)
            identity_ok = self.font_labels[shape_distance(ink[None], self.fonts).argmin()] == index
            reasons = [
                k for k in ["blank_count", "border_count", "fragment_proxy_count"] if metrics[k]
            ]
            if not identity_ok:
                reasons.append("template_identity_mismatch")
            # Frozen v0.9 blind evidence: ambiguous isolated identities are not promoted.
            if ch in {"5", "o", "√"}:
                reasons.append("historical_blind_identity_unresolved")
            self.gates[key] = reasons
        return index, variant, self.gates[key]


def layout(source: str, identity: str, profile, cfg: RenderConfig):
    blocks = validate_markdown(source, profile)
    upstream = upstream_config(profile.font, cfg, bool(profile.manifest.variants))
    upstream = upstream.replace(
        h1_size=upstream.body_size,
        h2_size=upstream.body_size,
        h3_size=upstream.body_size,
        math_scale=1.0,
    )
    with RENDER_LOCK, rc_context():
        logger = logging.getLogger("matplotlib._mathtext")
        handler = MathDiagnostics()
        logger.addHandler(handler)
        try:
            renderer = SinglePageRenderer(upstream)
            renderer.configure("baseline", profile, identity, None, cfg, None)
            pages = renderer.render_vector([Block("h2", (("t", identity),)), *blocks])
        finally:
            logger.removeHandler(handler)
        if len(pages) != 1 or renderer.warnings:
            raise ValueError(f"Unreliable single page: {renderer.warnings}")
        page = pages[0]
        classify_rules(page, renderer)
        strokes = (
            apply_math_strokes(page, renderer, profile) if cfg.math_strokes == "personal" else []
        )
        finish_ink(page, renderer, profile, False, renderer.stroke_seed)
        provenance = visible_provenance(page, renderer, profile, strokes)
        bounds, _ = ink_geometry(page, renderer)
        if min(bounds[:2]) < 3 or bounds[2] > page.width - 3 or bounds[3] > page.height - 3:
            raise ValueError("Baseline content crosses page boundary")
        return page, renderer, provenance


def replace_ink(page, renderer, provenance: list[dict], bank: FrozenBank, identity: str) -> tuple:
    result = copy.deepcopy(page)
    report = copy.deepcopy(provenance)
    base_outlines = dict(renderer._outlines)
    unchanged, changed = 0, 0
    for i, (item, record) in enumerate(zip(page.items, report, strict=True)):
        reason = None
        selected = None
        if item[0] != "path" or not record["personal_collected"]:
            reason = "original_nonpersonal_or_geometry"
        elif record["structure"]:
            reason = "parser_structure_kept"
        else:
            selected = bank.select(record["semantic"], identity, i)
            if selected is None:
                reason = "model_class_missing"
            elif selected[2]:
                reason = ",".join(selected[2])
        if reason:
            record["neural_fallback_reason"] = reason
            unchanged += 1
            continue
        label, variant, _ = selected
        old = base_outlines[item[1]]
        old_box = MplPath(old.verts, old.codes).get_extents().extents
        reference = foreground_bounds(bank.reference[label])
        neural = mask_outline(bank.ink[label, variant] > 0.5)
        # Map the unchanged reference scale; reject overflow instead of squashing ink.
        scale = min((old_box[2:] - old_box[:2]) / (reference[2:] - reference[:2]))
        center = (old_box[:2] + old_box[2:]) / 2
        ref_center = (reference[:2] + reference[2:]) / 2
        vertices = neural.verts.copy()
        # Recorder glyph outlines are y-up, model images are y-down.
        vertices = (vertices - ref_center) * [scale, -scale] + center
        new = replace(neural, verts=vertices)
        box = MplPath(new.verts, new.codes).get_extents().extents
        tolerance = max(old_box[2:] - old_box[:2]) * 0.025
        if np.any(box[:2] < old_box[:2] - tolerance) or np.any(box[2:] > old_box[2:] + tolerance):
            record["neural_fallback_reason"] = "generated_ink_exceeds_fixed_slot"
            unchanged += 1
            continue
        key = ("context-neural", identity, i, record["semantic"], variant)
        renderer._outlines[key] = new
        result.items[i] = (item[0], key, item[2], item[3], 0)
        record.update(
            source_type="neural-offline-generated",
            personal_collected=False,
            actual_glyph_id=f"{bank.path.name}:u{ord(record['semantic']):04x}-s{variant:02d}",
            source_sample_run=str(bank.path),
            checkpoint_sha256=bank.request["checkpoint_sha256"],
            generated_image_sha256=sha256(
                bank.path / f"rendered/glyphs/u{ord(record['semantic']):04x}-s{variant:02d}.png"
            ),
            vectorization="threshold0.5 exact row-run polygons; not online strokes",
            original_glyph=provenance[i],
            neural_fallback_reason=None,
        )
        record["bounds_px"] = item_bounds(result.items[i], renderer)
        changed += 1
    # Hard structure contract: same item order/count, source labels and every placement.
    assert len(result.items) == len(page.items)
    assert all(a[2] == b[2] for a, b in zip(result.items, page.items, strict=True))
    for i, original in enumerate(provenance):
        if original["structure"]:
            assert result.items[i] == page.items[i]
    bounds, _ = ink_geometry(result, renderer)
    if min(bounds[:2]) < 2 or bounds[2] > page.width - 2 or bounds[3] > page.height - 2:
        raise ValueError("Experimental ink crosses page boundary")
    return (
        result,
        report,
        {
            "neural_items": changed,
            "original_items": unchanged,
            "bounds": bounds,
            "structure_and_placement_preserved": True,
            "end_to_end_line_generation": False,
            "visual_content_verified": False,
            "production_eligible": False,
        },
    )
