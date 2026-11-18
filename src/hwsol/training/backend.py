"""Thin CPU build adapter: Handnote traces/merges, FontTools normalizes font assets."""

import json
from pathlib import Path
from typing import Protocol

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ttLib.scaleUpem import scale_upem
from handnote.opentype import add_alternates, read_alternates

from hwsol.profiles.schema import ProfileError
from hwsol.training.samples import Sample


class TrainingBackend(Protocol):
    """Future neural adapters must also write isolated candidate assets."""

    def build(self, sample: Sample, folder: Path, output: Path) -> dict: ...


class HandnoteCPU:
    def build(self, sample: Sample, folder: Path, output: Path) -> dict:
        from handnote.capture import build_capture_font
        from handnote.symbols import Template, build_font

        if sample.kind == "font":
            import shutil

            shutil.copyfile(folder / sample.originals[0], output)
            return {"backend": "font-import", "sample": sample.id}
        if not sample.template:
            raise ProfileError("Missing template for structured sample")
        spec = Template.from_dict(json.loads((folder / sample.template).read_text("utf-8")))
        if sample.kind == "capture":
            data = json.loads((folder / sample.originals[0]).read_text("utf-8"))
            result = build_capture_font(data.get("capture", data), spec, output)
        elif sample.kind == "scan":
            result = build_font([folder / p for p in sample.originals], spec, output)
        else:
            raise ProfileError("Unstructured archived material cannot be trained")
        if not result.get("glyphs"):
            raise ProfileError(f"No usable ink in sample {sample.id}")
        return {**result, "backend": "handnote-cpu", "sample": sample.id}


def merge_variants(base: Path, overlay: Path, output: Path, personal: set[int]) -> set[int]:
    """Keep the previous preferred glyph and all real alternates on repeated samples.

    Handnote does the geometry. Its replace=True keeps old glyph IDs; use that to
    restore the preferred cmap and attach incoming shapes via public OpenType helpers.
    Original fallback glyphs are not mislabeled as personal variants.
    """
    from handnote.symbols import merge_fonts

    with TTFont(base) as old, TTFont(overlay) as incoming:
        if "glyf" not in old or "glyf" not in incoming:
            raise ProfileError("Cumulative merging requires glyf TTF/OTF; CFF is standalone only")
        old_cmap = old.getBestCmap() or {}
        previous_alts = read_alternates(old)
        ids = {
            cp: [old.getGlyphID(g), *[old.getGlyphID(a) for a in previous_alts.get(g, [])]]
            for cp, g in old_cmap.items()
            if cp in personal
        }
        scale_upem(incoming, old["head"].unitsPerEm)
        # Handnote copies new composites verbatim. Decompose with FontTools so they
        # cannot accidentally reference differently named glyphs in the base font.
        glyphs = incoming.getGlyphSet()
        decomposed = {}
        for glyph in incoming.getGlyphOrder():
            if incoming["glyf"][glyph].isComposite():
                pen = DecomposingRecordingPen(glyphs)
                glyphs[glyph].draw(pen)
                target = TTGlyphPen(None)
                pen.replay(target)
                decomposed[glyph] = target.glyph()
        incoming["glyf"].glyphs.update(decomposed)
        normalized = output.with_suffix(".normalized.ttf")
        incoming.save(normalized)
        new_codes = set(incoming.getBestCmap() or {})
    try:
        merge_fonts(base, normalized, output, replace=True, family_suffix=" Personal")
    finally:
        normalized.unlink(missing_ok=True)
    with TTFont(output) as merged:
        cmap = merged.getBestCmap() or {}
        alts = read_alternates(merged)
        additions = {}
        for cp in sorted(personal & new_codes):
            old_names = [merged.getGlyphName(gid) for gid in ids[cp]]
            incoming_name = cmap[cp]
            all_alts = list(
                dict.fromkeys([*old_names[1:], incoming_name, *alts.get(incoming_name, [])])
            )
            preferred = old_names[0]
            for table in merged["cmap"].tables:
                if table.isUnicode() and cp in table.cmap:
                    table.cmap[cp] = preferred
            updated = False
            # Extend an existing rand lookup in place; read_alternates returns its first match.
            if "GSUB" in merged:
                gsub = merged["GSUB"].table
                if gsub.FeatureList and gsub.LookupList:
                    for feature in gsub.FeatureList.FeatureRecord:
                        if feature.FeatureTag != "rand":
                            continue
                        for index in feature.Feature.LookupListIndex:
                            for sub in gsub.LookupList.Lookup[index].SubTable:
                                if getattr(sub, "LookupType", None) == 7:
                                    sub = sub.ExtSubTable
                                mapping = getattr(sub, "alternates", {})
                                if preferred in mapping:
                                    mapping[preferred] = [preferred, *all_alts]
                                    updated = True
            if not updated:
                additions[preferred] = all_alts
        add_alternates(merged, additions)
        merged.save(output)
    return personal | new_codes
