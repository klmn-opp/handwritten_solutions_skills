"""Font inspection through FontTools; no tracing or training imports."""

import hashlib
import json
from pathlib import Path

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont, TTLibError
from handnote.opentype import read_alternates

from hwsol.profiles.schema import ProfileError


def inspect_font(path: Path) -> dict:
    try:
        return _inspect_font(path)
    except (TTLibError, KeyError) as exc:
        raise ProfileError(f"Invalid font asset: {path.name}: {exc}") from exc


def _inspect_font(path: Path) -> dict:
    with TTFont(path) as font:
        cmap = font.getBestCmap() or {}
        if not cmap or "hmtx" not in font or "fvar" in font:
            raise ProfileError("Font needs Unicode cmap and metrics; variable fonts unsupported")
        alts = read_alternates(font)
        glyphs = font.getGlyphSet()
        empty = []
        for cp, glyph in cmap.items():
            if not chr(cp).isprintable() or chr(cp).isspace():
                continue
            for shape in [glyph, *alts.get(glyph, [])]:
                pen = DecomposingRecordingPen(glyphs)
                glyphs[shape].draw(pen)
                if not pen.value or shape == ".notdef":
                    empty.append(cp)
                    break
        return {
            "coverage": sorted(cmap),
            "variants": {
                str(cp): 1 + len(alts[glyph]) for cp, glyph in cmap.items() if glyph in alts
            },
            "empty_codepoints": sorted(empty),
            "glyf": "glyf" in font,
            "family": font["name"].getDebugName(1),
            "upem": font["head"].unitsPerEm,
        }


def outline_fingerprints(path: Path, codepoint: int) -> list[str]:
    """Include decomposed paths and metrics; glyph names may change after saving."""
    with TTFont(path) as font:
        glyph = (font.getBestCmap() or {}).get(codepoint)
        if glyph is None:
            return []
        alts = read_alternates(font)
        glyphs = font.getGlyphSet()
        result = []
        for item in [glyph, *alts.get(glyph, [])]:
            pen = DecomposingRecordingPen(glyphs)
            glyphs[item].draw(pen)
            data = [pen.value, font["hmtx"][item], font["head"].unitsPerEm]
            result.append(hashlib.sha256(json.dumps(data).encode()).hexdigest())
        return result
