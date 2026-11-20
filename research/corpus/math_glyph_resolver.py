"""Opt-in CPU placement of reviewed personal math ink in fixed Handnote geometry."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from fontTools.ttLib import TTFont
from handnote.markdown import MATRIX_RE
from handnote.opentype import read_alternates
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from PIL import Image

from hwsol.profiles.io import sha256
from research.corpus.readability import join_radicals


@dataclass(frozen=True)
class ScanGlyph:
    source: dict
    outline: Outline

    @classmethod
    def load(cls, directory: Path, source: dict) -> "ScanGlyph":
        if source["review_status"] != "agent-accepted" or not source["clean_isolated_ink"]:
            raise ValueError("Math glyph requires a reviewed, isolated personal sample")
        path = directory / source["crop"]
        if sha256(path) != source["crop_sha256"]:
            raise ValueError(f"Math glyph crop changed: {path.name}")
        with Image.open(path) as image:
            grey = np.asarray(image.convert("L"), dtype=float)
        level = source["contour_luminance"]
        ink = grey < level
        if not ink.any() or any(edge.any() for edge in (ink[0], ink[-1], ink[:, 0], ink[:, -1])):
            raise ValueError(f"Empty or clipped math glyph: {path.name}")
        vertices, codes = contour_generator(z=-grey, fill_type="OuterCode").filled(-level, 1)
        return cls(source, Outline(np.vstack(vertices), np.concatenate(codes), stroke=False))


def matrix_strokes(page, renderer) -> dict[int, tuple[str, str]]:
    """Decode known Handnote bracket stroke groups without guessing from their shape."""
    layouts = [(group["item_range"], [group["environment"]]) for group in renderer.matrices]
    outer = {i for span, _ in layouts for i in range(*span)}
    for group in renderer.formulas:
        span = group["item_range"]
        if not set(range(*span)).intersection(outer):
            layouts.append((span, [m[1] for m in MATRIX_RE.finditer(group["latex"])]))

    result = {}
    for span, environments in layouts:
        explicit = {"pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "vmatrix": ("|", "|")}
        expected = [
            (character, env)
            for env in environments
            for character in explicit.get(env, ())
            for _ in range(3 if env == "bmatrix" else 1)
        ]
        indices = [i for i in range(*span) if page.items[i][0] == "stroke"]
        if len(expected) == len(indices):
            result.update(zip(indices, expected, strict=True))
    return result


def math_role(record: dict) -> str | None:
    return {"→": "relation-arrow", "̂": "hat-accent"}.get(record["semantic"], record["structure"])


def placed_outline(outline: Outline, bounds, *, stretch: bool = False) -> tuple[Outline, list]:
    low, high = np.asarray(bounds, dtype=float).reshape(2, 2)
    points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = (high - low) / (hi - lo)
    if not stretch:
        # Keep the scan's aspect ratio within a factor of 1.35 while fitting in the slot.
        scale = np.minimum(scale, scale.min() * 1.35)
    translation = (low + high) / 2 - (lo + hi) * scale / 2
    verts = outline.verts * scale + translation
    return Outline(verts, outline.codes, stroke=False, digits=4), [
        *scale.tolist(),
        *translation.tolist(),
    ]


def rule_outline(stroke, bounds) -> Outline:
    """Fit an existing measured pen envelope inside the fraction's reserved box."""
    x0, y0, x1, y1 = bounds
    x, centre, thickness = np.asarray(stroke.envelope).T
    # Keep the observed thickness variation and slope in separate budgets. Scaling
    # them together makes long sloped samples too faint when fitted to a thin line.
    height = y1 - y0
    centre = (
        (centre - (centre.min() + centre.max()) / 2) * height * 0.16 / max(1e-8, np.ptp(centre))
    )
    thickness = height * (0.56 + 0.24 * thickness / thickness.max())
    top, bottom = centre - thickness / 2, centre + thickness / 2
    y = np.concatenate([top, bottom[::-1], top[:1]])
    x = np.concatenate([x, x[::-1], x[:1]])
    verts = np.column_stack((x0 + x * (x1 - x0), (y0 + y1) / 2 + y))
    codes = np.full(len(verts), MplPath.LINETO, dtype=MplPath.code_type)
    codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
    return Outline(verts, codes, stroke=False, digits=4)


def resolve_radicals(page, renderer, records: list[dict]) -> None:
    """Run the established join on a scratch page and accept only contained pairs."""
    trial = replace(page, items=list(page.items))
    eligible = [r for r in records if r["structure"] != "radical-sign" or r["personal_collected"]]
    for join in join_radicals(trial, renderer, eligible):
        if not join["applied"]:
            continue
        si, bi = join["item_index"], join["bar_item_index"]
        sign, bar = records[si], records[bi]
        if sign["latex"] != bar["latex"]:
            continue
        old = np.asarray([sign["bounds_px"], bar["bounds_px"]])
        low, high = old[:, :2].min(axis=0), old[:, 2:].max(axis=0)
        new = np.asarray([join["new_sign_bounds"], join["new_bar_bounds"]])
        if np.any(new[:, :2] < low - 1e-6) or np.any(new[:, 2:] > high + 1e-6):
            continue
        for i, row, box, action in (
            (si, sign, new[0], "trim personal stem terminal to join overbar"),
            (bi, bar, new[1], "extend overbar using measured personal stem thickness"),
        ):
            row.pop("fallback_reason", None)
            row.update(
                original_source_type=row["source_type"],
                original_glyph_id=row["actual_glyph_id"],
                original_bounds_px=row["bounds_px"],
                allowed_bounds_px=[*low.tolist(), *high.tolist()],
                bounds_px=box.tolist(),
                transformation=action,
                radical_join=join,
                paired_items=[si, bi],
            )
            page.items[i] = trial.items[i]


class MathGlyphResolver:
    """Use parser slots as limits and keep a source decision for every fallback."""

    def __init__(self, glyphs: dict[str, ScanGlyph], stroke_profile, *, selected: set[str]):
        self.glyphs = glyphs
        self.stroke_profile = stroke_profile
        self.selected = selected
        self.rules = [
            (style.sample_id, stroke)
            for style in stroke_profile.styles
            for stroke in style.math_strokes
            if stroke.kind == "rule"
        ]

    def resolve(self, page, renderer, provenance: list[dict], profile) -> list[dict]:
        brackets = matrix_strokes(page, renderer)
        parser_fallbacks = {
            r["item_index"]: r["reason"] for r in getattr(renderer, "structure_fallbacks", ())
        }
        with TTFont(renderer.font_path) as font:
            cmap, alts = font.getBestCmap(), read_alternates(font)
            variants = {}
            for char in "()":
                name = cmap.get(ord(char))
                if name and ord(char) in profile.manifest.personal_codepoints:
                    names = [name, *alts.get(name, [])]
                    slot = min(1, len(names) - 1)
                    variants[char] = (slot, names[slot], font.getGlyphID(names[slot]))
        output = []
        for old in provenance:
            if old["context"] != "mathematics":
                output.append(old)
                continue
            index = old["item_index"]
            item = page.items[index]
            char = old["semantic"]
            bounds = list(old["bounds_px"])
            matrix = brackets.get(index)
            record = {**old, "structure": math_role(old)}
            if matrix:
                char, env = matrix
                record.update(
                    semantic=char,
                    codepoint=f"U+{ord(char):04X}",
                    structure="matrix-bracket",
                    matrix_environment=env,
                )
                pen = item[4] / 2
                bounds = [
                    b + delta for b, delta in zip(bounds, [-pen, -pen, pen, pen], strict=True)
                ]
            outline = None
            if matrix and matrix[1] == "pmatrix" and char in self.selected and char in variants:
                slot, name, glyph_id = variants[char]
                raw = handwritten_outline(renderer.font_path, ord(char), slot, 100)
                if raw is None:
                    raise ValueError(f"Missing registered Profile contour for {char}")
                # Preserve the parser's full bracket height and side. Enforce the original pen box.
                outline, transform = placed_outline(
                    Outline(raw[0] * (1, -1), raw[1], stroke=False), bounds, stretch=True
                )
                record.update(
                    source_type="personal-outline-stretched",
                    actual_glyph_id=glyph_id,
                    glyph_name=name,
                    variant=slot,
                    font=profile.font.name,
                    font_sha256=profile.provenance()["font_sha256"],
                    transformation="fit Profile delimiter to Handnote matrix pen box",
                    transform_scale_translate=transform,
                )
            elif (
                char in self.glyphs
                and char in self.selected
                and old["source_type"] in {"external-font", "inherited-font"}
            ):
                glyph = self.glyphs[char]
                # Tall delimiters need a reserved vertical envelope. Keep their original
                # geometry when the bounded fit would make them too short to group the content.
                fit = placed_outline(glyph.outline, bounds)
                aspect = (bounds[3] - bounds[1]) / max(1e-8, bounds[2] - bounds[0])
                if char not in "[]" or aspect <= 6:
                    outline, transform = fit
                    record.update(
                        source_type="personal-scan-outline",
                        actual_glyph_id=glyph.source["id"],
                        glyph_name=None,
                        font=None,
                        font_sha256=None,
                        scan_source=glyph.source,
                        transformation="luminance contour inside parser slot",
                        transform_scale_translate=transform,
                    )
                else:
                    record["fallback_reason"] = "isolated bracket cannot span the parser height"
            elif (
                char == "fraction-bar"
                and char in self.selected
                and self.rules
                and old["procedural_geometry"]
            ):
                sample_id, stroke = self.rules[0]
                outline = rule_outline(stroke, bounds)
                record.update(
                    source_type="personal-scan-envelope",
                    actual_glyph_id=f"stroke:{sample_id}:rule",
                    source_sha256=stroke.source_sha256,
                    source_box=stroke.source_box,
                    stroke_profile=self.stroke_profile.provenance(),
                    transformation="measured envelope bounded by the parser fraction rectangle",
                )
            if outline is None:
                if not record["personal_collected"]:
                    reason = parser_fallbacks.get(index)
                    if reason is None:
                        if char in self.glyphs:
                            reason = "reviewed sample retained for visual or geometry validation"
                        elif matrix:
                            reason = "Handnote matrix stroke geometry retained"
                        elif char == "radical-overbar":
                            reason = "preserve sign-to-bar join and radicand clearance"
                        elif record["procedural_geometry"]:
                            reason = "parser geometry retained"
                        else:
                            reason = "no reviewed and compatible personal contour for this role"
                    record.setdefault("fallback_reason", reason)
            else:
                key = ("math-glyph-resolver", index)
                renderer._outlines[key] = outline
                page.items[index] = ("path", key, (1, 0, 0, 1, 0, 0), item[3], 0)
                points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
                record.update(
                    personal_collected=True,
                    external_fallback=False,
                    procedural_geometry=False,
                    bounds_px=[*points.min(axis=0).tolist(), *points.max(axis=0).tolist()],
                    original_bounds_px=old["bounds_px"],
                    allowed_bounds_px=bounds,
                    original_source_type=old["source_type"],
                    original_glyph_id=old["actual_glyph_id"],
                )
            output.append(record)
        if "radical-overbar" in self.selected:
            resolve_radicals(page, renderer, output)
        return output
