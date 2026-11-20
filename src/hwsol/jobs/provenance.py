"""One provenance record per visible recorder item, after final outline selection."""

from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from handnote.opentype import read_alternates
from matplotlib.path import Path as MplPath

from hwsol.profiles.io import sha256


def structure_kind(key: tuple, latex: str, bounds: list, renderer=None, item=None) -> str | None:
    if renderer is not None and key in getattr(renderer, "rule_semantics", {}):
        return renderer.rule_semantics[key]
    if key[0] == "mglyph":
        return {
            0x304: "mean-accent",
            0x305: "overline-accent",
            0x221A: "radical-sign",
            0x20D7: "vector-arrow",
            0x2192: "vector-arrow",
        }.get(key[3]) or ("stretchable-delimiter" if chr(key[3]) in "()[]{}|" else None)
    if key[0] == "personal-math-stroke":
        return "mean-accent" if key[1] == "bar" else None
    if key[0] != "mrect":
        return None
    # The recorder preserves all nested glyph positions. A radical's bar begins at
    # the right of the radical and lies near its top; fraction bars lie between rows.
    if renderer is not None and item is not None:
        radicals = getattr(renderer, "structure_radicals", ())
        for x0, y0, x1, _y1 in radicals:
            if (
                x0 < bounds[0] < x1 + renderer.cfg.body_size * 0.5
                and abs(bounds[1] - y0) < renderer.cfg.body_size * 0.3
            ):
                return "radical-overbar"
    if r"\overline" in latex and r"\frac" not in latex and r"\dfrac" not in latex:
        return "overline-accent"
    return "fraction-bar" if r"\frac" in latex or r"\dfrac" in latex else "math-rule"


def item_bounds(item, renderer) -> list[float]:
    kind, key, (a, b, c, d, e, f) = item[:3]
    if kind == "path":
        outline = renderer._outlines[key]
        points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
    elif kind == "stroke":
        points = np.asarray(key)
    else:
        return []
    if not len(points):
        return []
    placed = points @ np.array([[a, b], [c, d]]) + (e, f)
    return [*placed.min(axis=0).tolist(), *placed.max(axis=0).tolist()]


def visible_provenance(page, renderer, profile, replacements: list[dict]) -> list[dict]:
    font_path = renderer.font_path.resolve()
    with TTFont(font_path) as font:
        cmap, alts = font.getBestCmap(), read_alternates(font)
        names = {cp: [name, *alts.get(name, [])] for cp, name in cmap.items()}
        ids = {name: font.getGlyphID(name) for seq in names.values() for name in seq}
    replacement = {row["item_index"]: row for row in replacements}
    delimiters = {row["item_index"]: row for row in getattr(renderer, "structure_replacements", ())}
    formulas = {
        i: group["latex"] for group in renderer.formulas for i in range(*group["item_range"])
    }
    matrices = {
        i: group["environment"] for group in renderer.matrices for i in range(*group["item_range"])
    }
    bounds = [item_bounds(item, renderer) for item in page.items]
    renderer.structure_radicals = [
        bounds[i]
        for i, item in enumerate(page.items)
        if item[0] == "path" and item[1][0] == "mglyph" and item[1][3] == 0x221A
    ]
    font_digest = sha256(font_path)
    records = []
    for i, item in enumerate(page.items):
        kind, key = item[:2]
        record = {
            "item_index": i,
            "profile": profile.manifest.ref,
            "profile_sha256": profile.manifest_sha256,
            "semantic": None,
            "actual_glyph_id": None,
            "glyph_name": None,
            "source_type": "procedural-geometry",
            "personal_collected": False,
            "external_fallback": False,
            "procedural_geometry": True,
            "bounds_px": bounds[i],
            "latex": formulas.get(i),
            "context": "mathematics" if i in formulas or i in matrices else "text",
            "matrix_environment": matrices.get(i),
            "structure": None,
        }
        if kind == "path" and key[0] in {"glyph", "mglyph"}:
            code = key[2] if key[0] == "glyph" else key[3]
            slot = key[1] if key[0] == "glyph" else key[4]
            if "pen" in key:
                slot = key[key.index("pen") + 1]
            slot = max(0, slot)
            if str(code) not in profile.manifest.variants:
                slot = 0
            fallback = key[0] == "mglyph" and Path(key[1]).resolve() != font_path
            personal = code in profile.manifest.personal_codepoints and not fallback
            name = names.get(code, [None])[min(slot, len(names.get(code, [None])) - 1)]
            record.update(
                semantic=chr(code),
                codepoint=f"U+{code:04X}",
                variant=slot,
                actual_glyph_id=key[2] if fallback else ids.get(name),
                glyph_name=None if fallback else name,
                source_type="external-font"
                if fallback
                else "personal-outline"
                if personal
                else "inherited-font",
                personal_collected=personal,
                external_fallback=fallback,
                procedural_geometry=False,
                font=Path(key[1]).name if fallback else profile.font.name,
                font_sha256=sha256(Path(key[1])) if fallback else font_digest,
                structure=structure_kind(key, formulas.get(i, ""), bounds[i], renderer, item),
            )
        elif i in delimiters:
            row = delimiters[i]
            record.update(
                semantic=chr(row["codepoint"]),
                codepoint=f"U+{row['codepoint']:04X}",
                structure=row["structure"],
                actual_glyph_id=row["glyph_id"],
                glyph_name=names[row["codepoint"]][0],
                variant=0,
                source_type="personal-outline-stretched",
                personal_collected=True,
                procedural_geometry=False,
                font_sha256=font_digest,
                transformation="fit collected outline to parser delimiter bounds",
            )
        elif i in replacement:
            row = replacement[i]
            record.update(
                semantic=row["structure"],
                structure=row["structure"],
                actual_glyph_id=f"stroke:{row['sample_id']}:{row['kind']}",
                source_type="personal-scan-envelope",
                personal_collected=True,
                procedural_geometry=False,
                source_sha256=row["source_sha256"],
                source_box=row["source_box"],
            )
        else:
            structure = (
                "matrix-bracket"
                if i in matrices and kind == "stroke"
                else (
                    structure_kind(key, formulas.get(i, ""), bounds[i], renderer, item)
                    if kind == "path"
                    else "geometry"
                    if kind == "stroke"
                    else "raster"
                )
            )
            record.update(
                semantic=structure,
                structure=structure,
                source_type="unattributed-raster" if kind == "image" else "procedural-geometry",
            )
        records.append(record)
    return records


def classify_rules(page, renderer) -> None:
    """Associate parser rectangles with their local formula, before any replacement."""
    renderer.rule_semantics = {}
    for group in renderer.formulas:
        indices = range(*group["item_range"])
        glyphs = [
            (page.items[i][1], item_bounds(page.items[i], renderer))
            for i in indices
            if page.items[i][0] == "path" and page.items[i][1][0] == "mglyph"
        ]
        radicals = [box for key, box in glyphs if key[3] == 0x221A]
        for i in indices:
            item = page.items[i]
            if item[0] != "path" or item[1][0] != "mrect":
                continue
            x0, y0, x1, y1 = item_bounds(item, renderer)
            if any(
                a < x0 < c + renderer.cfg.body_size * 0.5
                and abs(y0 - b) < renderer.cfg.body_size * 0.3
                for a, b, c, _ in radicals
            ):
                semantic = "radical-overbar"
            else:
                above = any(
                    box[2] > x0 and box[0] < x1 and box[3] < (y0 + y1) / 2 for _, box in glyphs
                )
                semantic = (
                    "overline-accent"
                    if r"\overline" in group["latex"] and not above
                    else ("fraction-bar" if r"\frac" in group["latex"] else "math-rule")
                )
            renderer.rule_semantics[item[1]] = semantic
