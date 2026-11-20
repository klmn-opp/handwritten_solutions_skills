"""Measure real outlines and normalize a derived font before either layout engine sees it.

All coordinates are font ems, positive y upwards. Uniform scaling around the measured
baseline preserves descenders and contour topology; original assets are never changed.
"""

import hashlib
import math
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import median

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.momentsPen import MomentsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from handnote.opentype import read_alternates

from hwsol.profiles.io import sha256

METRIC_VERSION = "glyph-metrics-v1"


def category(ch: str) -> str:
    if ch in "ij":
        return "dotted-descender" if ch == "j" else "dotted"
    if ch in "bdfhklt":
        return "ascender"
    if ch in "gpqy":
        return "descender"
    if ch.isascii() and ch.islower():
        return "lowercase"
    if ch.isascii() and ch.isupper():
        return "capital"
    if ch.isascii() and ch.isdigit():
        return "digit"
    return "symbol"


def measure_font(path: Path, personal: tuple[int, ...]) -> dict:
    records = []
    with TTFont(path) as font:
        upem = font["head"].unitsPerEm
        glyphs, cmap, alts = font.getGlyphSet(), font.getBestCmap(), read_alternates(font)
        for cp in sorted(personal):
            primary = cmap.get(cp)
            if primary is None:
                continue
            for slot, name in enumerate([primary, *alts.get(primary, [])]):
                pen = BoundsPen(glyphs)
                glyphs[name].draw(pen)
                if pen.bounds is None:
                    continue
                x0, y0, x1, y1 = [v / upem for v in pen.bounds]
                moments = MomentsPen(glyphs)
                glyphs[name].draw(moments)
                center = (
                    [moments.momentX / moments.area / upem, moments.momentY / moments.area / upem]
                    if abs(moments.area) > 1e-6
                    else [(x0 + x1) / 2, (y0 + y1) / 2]
                )
                records.append(
                    {
                        "codepoint": cp,
                        "character": chr(cp),
                        "slot": slot,
                        "glyph_name": name,
                        "glyph_id": font.getGlyphID(name),
                        "category": category(chr(cp)),
                        "ink_bounds_em": [x0, y0, x1, y1],
                        "advance_em": font["hmtx"][name][0] / upem,
                        "visual_center_em": center,
                        "baseline_offset_em": y0,
                        "personal_collected": True,
                    }
                )
    # Descenders and dotted letters must never be used as whole-height x-height samples.
    baseline_values = [
        r["ink_bounds_em"][1]
        for r in records
        if r["category"] in {"lowercase", "ascender", "capital", "digit"}
    ]
    baseline = median(baseline_values) if baseline_values else 0.0
    heights = defaultdict(list)
    for r in records:
        heights[r["category"]].append(r["ink_bounds_em"][3] - baseline)
    references = {k: median(v) for k, v in heights.items()}
    # A body-top measurement for p/g/q/y excludes the tail below the baseline.
    if "lowercase" in references:
        references["descender"] = references["lowercase"]
    descents = [
        baseline - r["ink_bounds_em"][1]
        for r in records
        if r["category"] in {"descender", "dotted-descender"}
    ]
    by_code = defaultdict(list)
    for r in records:
        by_code[r["codepoint"]].append(r)
    for r in records:
        x0, y0, x1, y1 = r["ink_bounds_em"]
        peers = by_code[r["codepoint"]]
        own_top = median(p["ink_bounds_em"][3] - baseline for p in peers)
        class_top = references[r["category"]]
        target = 0.75 * class_top + 0.25 * own_top
        observed = max(1e-6, y1 - baseline)
        widths = [p["ink_bounds_em"][2] - p["ink_bounds_em"][0] for p in peers]
        ratio = observed / max(target, 1e-6)
        width_ratio = (x1 - x0) / max(median(widths), 1e-6)
        outlier = ratio < 0.75 or ratio > 1.25 or width_ratio < 0.55 or width_ratio > 1.8
        extreme = ratio < 0.4 or ratio > 2.5 or x1 <= x0 or y1 <= y0
        scale = max(0.75, min(1.3, (target / observed) ** 0.85))
        if r["category"] == "symbol":
            scale = 1.0  # Different mathematical semantics have different sizes.
        r.update(
            baseline_offset_em=y0 - baseline,
            reference_top_em=target,
            scale=scale,
            translate_y_em=baseline * (1 - scale),
            outlier=outlier,
            excluded=extreme,
            selection_weight=0.0 if extreme else 0.15 if outlier else 1.0,
        )
    return {
        "schema_version": 1,
        "algorithm": METRIC_VERSION,
        "font_sha256": sha256(path),
        "measurement": "FontTools ink bounds and signed-area centroid; estimated font baseline",
        "units_per_em": upem,
        "baseline_em": baseline,
        "references_em": {
            "x_height": references.get("lowercase"),
            "cap_height": references.get("capital"),
            "ascender": references.get("ascender"),
            "digit_height": references.get("digit"),
            "descender": median(descents) if descents else None,
        },
        "glyphs": records,
    }


def normalize_font(source: Path, destination: Path, metrics: dict) -> None:
    if sha256(source) != metrics["font_sha256"]:
        raise ValueError("Glyph metrics belong to a different font")
    if destination.exists():
        raise FileExistsError(destination)
    with TTFont(source, recalcTimestamp=False) as font:
        if "glyf" not in font:
            raise ValueError(
                "Metric normalization currently requires glyf outlines; use raw for CFF"
            )
        glyphs = font.getGlyphSet()
        upem = font["head"].unitsPerEm
        transformed = {}
        for row in metrics["glyphs"]:
            name, scale = row["glyph_name"], row["scale"]
            if row["excluded"] or name in transformed:
                continue
            drawing = DecomposingRecordingPen(glyphs)
            glyphs[name].draw(drawing)
            target = TTGlyphPen(None)
            drawing.replay(
                TransformPen(target, (scale, 0, 0, scale, 0, row["translate_y_em"] * upem))
            )
            transformed[name] = target.glyph()
            advance, bearing = font["hmtx"][name]
            font["hmtx"][name] = (math.ceil(advance * scale), round(bearing * scale))
        font["glyf"].glyphs.update(transformed)
        font.save(destination)


# Process-local temporary assets avoid modifying any published snapshot. Retaining the
# directory until process exit also keeps Windows FreeType handles valid.
_CACHE = tempfile.TemporaryDirectory(prefix="hwsol-metrics-")


@lru_cache(maxsize=64)
def normalized_asset(source: str, font_sha: str, personal: tuple[int, ...]) -> tuple[Path, dict]:
    path = Path(source)
    if sha256(path) != font_sha:
        raise ValueError("Font changed before normalization")
    metrics = measure_font(path, personal)
    scope = hashlib.sha256(repr(personal).encode()).hexdigest()[:16]
    dest = Path(_CACHE.name) / f"{font_sha}-{scope}-{METRIC_VERSION}.ttf"
    if not dest.exists():
        normalize_font(path, dest, metrics)
    return dest, metrics
