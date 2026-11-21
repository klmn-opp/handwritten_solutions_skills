"""CPU kernel extracted from research.historical_math_candidate; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy

import numpy as np
from handnote.vector import Outline

from hwsol.personal_ink.kernels.math_glyph_resolver import MathGlyphResolver, ScanGlyph
from hwsol.personal_ink.kernels.render import read
from hwsol.profiles.io import sha256

TARGETS = frozenset(("∈", "∬"))


def load_sources(directory):
    """Load immutable reviewed outlines, including explicit component provenance."""
    manifest = read(directory / "manifest.json")
    glyphs = {}
    for source in manifest["sources"]:
        checks = (
            ("contour_archive", "contour_archive_sha256"),
            ("reviewed_component_selection", "reviewed_component_selection_sha256"),
        )
        if source.get("representation") == "compiled-contour/1":
            from hwsol.personal_ink.compiled import load_compiled_outline

            outline = load_compiled_outline(directory, source)
        else:
            if source.get("representation") is not None:
                raise ValueError("Unknown historical source representation")
            checks = (("crop", "crop_sha256"), *checks)
            outline = None
        for path_key, hash_key in checks:
            if sha256(directory / source[path_key]) != source[hash_key]:
                raise ValueError("Reviewed source/contour evidence changed")
        if outline is None:
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
            and (r["source_type"] in {"external-font", "inherited-font"})
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
