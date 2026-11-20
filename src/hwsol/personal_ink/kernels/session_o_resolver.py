"""CPU kernel extracted from research.session_o_resolver; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from handnote.vector import Outline

from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph
from hwsol.personal_ink.kernels.math_latin_feedback import contacts
from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.profiles.io import sha256

PRIORITY = frozenset("πκΦξη⟨⟩∏≈ζψ∮")


class SessionOResolver:
    """Opt-in personal source bank; no parsing, layout or English changes."""

    def __init__(self, directory: Path, *, seed: int = 1909, targets: frozenset[str] = PRIORITY):
        self.directory = directory
        self.seed = seed
        self.targets = targets
        self.manifest_sha256 = sha256(directory / "sources.json")
        self.variants: dict[str, list[ScanGlyph]] = defaultdict(list)
        for row in json.loads((directory / "sources.json").read_text()):
            if row["review_status"] != "agent-accepted":
                continue
            if row.get("capture_session") != "O-FULL" or row.get("original_pixels_edited"):
                raise ValueError("Only reviewed, unchanged original O-FULL ink is allowed")
            self.variants[row["character"]].append(ScanGlyph.load(directory, row))
        for variants in self.variants.values():
            variants.sort(key=lambda sample: sample.source["id"])
        self.decisions: list[dict] = []

    @staticmethod
    def fit(sample: ScanGlyph, bounds: list[float], *, stretch: bool) -> tuple[Outline, dict]:
        source_box = outline_bounds(sample.outline.verts, sample.outline.codes)
        target = np.asarray(bounds, dtype=float).reshape(2, 2)
        scale = np.ptp(target, axis=0) / np.ptp(source_box, axis=0)
        if not np.isfinite(scale).all() or min(scale) <= 0:
            raise ValueError("Empty or invalid source/target span")
        if stretch:
            if max(scale) / min(scale) > 4:
                raise ValueError("Delimiter stretch exceeds reviewed bound")
        else:
            scale[:] = min(scale)
        translation = target.mean(axis=0) - source_box.mean(axis=0) * scale
        reference = sample.source.get("body_reference")
        if sample.source["character"] == "Φ" and reference:
            if reference["role"] != "capital-phi-bowl":
                raise ValueError("Unrecognized capital Phi body reference")
            body = np.asarray(reference["bounds_crop_px"], dtype=float).reshape(2, 2)
            scale[:] = np.ptp(target, axis=0)[1] / np.ptp(body, axis=0)[1]
            translation = target.mean(axis=0) - body.mean(axis=0) * scale
        if sample.source["character"] == "∮":
            translation[0] = target[0, 0] - source_box[0, 0] * scale[0]
        result = Outline(
            sample.outline.verts * scale + translation,
            sample.outline.codes.copy(),
            stroke=False,
            digits=4,
        )
        return (
            result,
            {
                "rule": "source contour scale and translation only",
                "scale_xy": scale.tolist(),
                "translation_xy": translation.tolist(),
                "source_bounds_px": source_box.ravel().tolist(),
                "anisotropic": bool(abs(scale[0] - scale[1]) > 1e-10),
                "horizontal_anchor": "left" if sample.source["character"] == "∮" else "center",
                "body_reference": reference,
                "stroke_redrawing": False,
                "ink_erosion_or_dilation": False,
            },
        )

    def apply(self, page, renderer, provenance: list[dict], *, case_id: str) -> list[dict]:
        rows = copy.deepcopy(provenance)
        before_items = list(page.items)
        for row in rows:
            char = row["semantic"]
            if row["context"] != "mathematics" or char not in self.targets:
                continue
            if char not in self.variants:
                self.decisions.append(
                    {
                        "item_index": row["item_index"],
                        "semantic": char,
                        "accepted": False,
                        "reason": "No reviewed Session O source; prior item retained",
                    }
                )
                continue
            if char != "π" and row["source_type"] not in {"external-font", "inherited-font"}:
                continue
            index = row["item_index"]
            variants = self.variants[char]
            key = json.dumps(
                [self.seed, case_id, row.get("latex"), index, char], ensure_ascii=False
            )
            selected = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(variants)
            sample = variants[selected]
            decision = {
                "item_index": index,
                "semantic": char,
                "source_id": sample.source["id"],
                "selection_key": key,
                "available_real_samples": len(variants),
                "accepted": False,
            }
            try:
                outline, transform = self.fit(sample, row["bounds_px"], stretch=char in "⟨⟩")
                allowed = list(row["bounds_px"])
                if transform["body_reference"]:
                    width, height = (allowed[2] - allowed[0], allowed[3] - allowed[1])
                    allowed = (
                        np.asarray(allowed)
                        + np.asarray([-0.2 * width, -0.4 * height, 0.2 * width, 0.6 * height])
                    ).tolist()
                box = outline_bounds(outline.verts, outline.codes).ravel().tolist()
                if (
                    min(
                        box[0] - allowed[0],
                        box[1] - allowed[1],
                        allowed[2] - box[2],
                        allowed[3] - box[3],
                    )
                    < -0.03
                ):
                    raise ValueError("Real glyph exceeds the fixed placement envelope")
                outline_key = ("session-o-reviewed-original", index)
                renderer._outlines[outline_key] = outline
                page.items[index] = (
                    "path",
                    outline_key,
                    (1, 0, 0, 1, 0, 0),
                    page.items[index][3],
                    0,
                )
                collision = contacts(index, before_items, page, renderer, rows)
                if any(item["new_contact"] for item in collision):
                    raise ValueError("New contact with a neighboring item")
                prior = copy.deepcopy(row)
                for old_key in (
                    "fallback_reason",
                    "scan_source",
                    "session_source",
                    "transform_scale_translate",
                    "original_source_type",
                    "original_glyph_id",
                    "prior_source_type",
                    "prior_glyph_id",
                ):
                    row.pop(old_key, None)
                row.update(
                    session_o_previous=prior,
                    session_o_source=sample.source,
                    session_o_bank_sha256=self.manifest_sha256,
                    actual_glyph_id=sample.source["id"],
                    variant=selected,
                    source_type="personal-stretched-outline"
                    if transform["anisotropic"]
                    else "personal-session-outline",
                    personal_collected=True,
                    external_fallback=False,
                    procedural_geometry=False,
                    font=None,
                    font_sha256=None,
                    glyph_name=None,
                    original_bounds_px=prior["bounds_px"],
                    allowed_bounds_px=allowed,
                    bounds_px=box,
                    transformation=transform,
                )
                decision.update(accepted=True, transformation=transform, contacts=collision)
            except ValueError as exc:
                page.items[index] = before_items[index]
                decision["reason"] = str(exc)
            self.decisions.append(decision)
        return rows
