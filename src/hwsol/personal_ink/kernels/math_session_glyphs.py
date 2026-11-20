"""CPU kernel extracted from research.corpus.math_session_glyphs; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox

from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph, placed_outline

ALTERNATES = set("+−=≥→ϵ")
ROLE_ALIASES = {"overline-accent": "̅", "∣": "|"}


def outline_bounds(verts: np.ndarray, codes: np.ndarray) -> np.ndarray:
    points = verts[codes != MplPath.CLOSEPOLY]
    return np.array([points.min(axis=0), points.max(axis=0)])


@dataclass(frozen=True)
class RadicalSample:
    glyph: ScanGlyph
    source_bounds: np.ndarray
    seam_x: float
    roof: np.ndarray
    components: int

    @classmethod
    def measure(cls, glyph: ScanGlyph) -> RadicalSample:
        raw = glyph.outline
        usable = raw.codes != MplPath.CLOSEPOLY
        coords = raw.verts[usable]
        seam = glyph.source["seam_x"]
        bounds = outline_bounds(raw.verts, raw.codes)
        if not bounds[0, 0] < seam < bounds[1, 0]:
            raise ValueError(f"Radical seam outside original crop: {glyph.source['id']}")
        roof = coords[coords[:, 0] >= seam, 1]
        if len(roof) < 2:
            raise ValueError(f"No roof after radical seam: {glyph.source['id']}")
        return cls(
            glyph,
            bounds,
            seam,
            np.array([roof.min(), roof.max()]),
            int(np.count_nonzero(raw.codes == MplPath.MOVETO)),
        )

    def metrics(self) -> dict:
        span = self.source_bounds[1] - self.source_bounds[0]
        return {
            "id": self.glyph.source["id"],
            "source_width_px": float(span[0]),
            "source_height_px": float(span[1]),
            "roof_reference_y_px": self.roof.tolist(),
            "seam_x_in_crop_px": self.seam_x,
            "ink_contours": self.components,
            "continuous_candidate": self.components == 1,
        }

    def fit(self, sign: dict, bar: dict) -> tuple[list[Outline], list] | None:
        """Map the roof length and stem height separately with a shared seam."""
        sx, sy, _, sb = sign["bounds_px"]
        bx, by, br, bb = bar["bounds_px"]
        lo, hi = self.source_bounds
        ry, rb = self.roof
        if self.components != 1 or min(rb - ry, bb - by, sb - bb, hi[1] - rb) <= 0:
            return None
        stem_scale = (sb - bb) / (hi[1] - rb)
        head_left = max(sx, bx - (self.seam_x - lo[0]) * stem_scale * 1.2)
        if bx - head_left < 3 or br - bx < 2:
            return None
        vertices = self.glyph.outline.verts
        path = MplPath(vertices, self.glyph.outline.codes)
        mapped = []
        for polygon in path.to_polygons(closed_only=True):
            pieces = []
            for start, end in zip(polygon[:-1], polygon[1:], strict=True):
                steps = max(1, int(np.ceil(np.abs(end - start).max())))
                pieces.append(start + np.arange(steps)[:, None] / steps * (end - start))
            dense = np.vstack([*pieces, polygon[-1:]])
            x, y = dense.T
            tx = np.interp(x, [lo[0], self.seam_x, hi[0]], [head_left, bx, br])
            ty = (
                np.interp(y, [lo[1], rb, hi[1]], [by, bb, sb])
                if lo[1] == ry
                else np.interp(y, [lo[1], ry, rb, hi[1]], [min(sy, by), by, bb, sb])
            )
            mapped.append(MplPath(np.column_stack((tx, ty)), closed=True))
        if not mapped:
            return None
        candidate = MplPath.make_compound_path(*mapped)
        top = min(sy, by)
        bands = [
            Bbox.from_extents(head_left - 0.1, top - 0.1, bx + 0.35, sb + 0.1),
            Bbox.from_extents(bx - 0.35, top - 0.1, br + 0.1, sb + 0.1),
        ]
        paths = [candidate.clip_to_bbox(box) for box in bands]
        if any(len(p.vertices) == 0 for p in paths):
            return None
        return (
            [Outline(p.vertices, p.codes, stroke=False, digits=4) for p in paths],
            [sx, top, br, max(sb, bb)],
        )


class SessionGlyphCalibration:
    """Choose original variants from M/N by their measured aspect and role."""

    def __init__(self, samples: list[ScanGlyph]):
        self.variants: dict[str, list[ScanGlyph]] = defaultdict(list)
        self.radicals = []
        for glyph in samples:
            if glyph.source["role"] == "radical":
                self.radicals.append(RadicalSample.measure(glyph))
            else:
                self.variants[glyph.source["character"]].append(glyph)

    @staticmethod
    def _draw(page, renderer, row, outline, allowed, glyph, *, transform, paired=None):
        index = row["item_index"]
        coords = outline_bounds(outline.verts, outline.codes)
        key = ("session-math-contour", index)
        renderer._outlines[key] = outline
        page.items[index] = ("path", key, (1, 0, 0, 1, 0, 0), page.items[index][3], 0)
        row.pop("fallback_reason", None)
        row.update(
            prior_source_type=row["source_type"],
            prior_glyph_id=row["actual_glyph_id"],
            original_bounds_px=row["bounds_px"],
            allowed_bounds_px=list(allowed),
            bounds_px=coords.ravel().tolist(),
            source_type="personal-session-outline",
            session_source=glyph.source,
            actual_glyph_id=f"{glyph.source['id']}:{row['structure'] or 'glyph'}",
            glyph_name=None,
            personal_collected=True,
            external_fallback=False,
            procedural_geometry=False,
            font=None,
            font_sha256=None,
            transformation=transform,
        )
        if paired:
            row["paired_items"] = paired

    def _glyph(self, row, *, operator_page: bool) -> tuple | None:
        char = ROLE_ALIASES.get(row["semantic"], row["semantic"])
        tex = row.get("latex") or ""
        if char == "ε" and "\\epsilon" in tex and ("\\varepsilon" not in tex):
            char = "ϵ"
        bounds = row["bounds_px"]
        if row["structure"] == "matrix-bracket" or len(bounds) != 4:
            return None
        if row["personal_collected"] and (not (operator_page and char in ALTERNATES)):
            return None
        if char not in self.variants:
            return None
        if char in "{}":
            row["fallback_reason"] = "reviewed brace sides remain visually ambiguous at page size"
            return None
        fitted = self._fit_character(char, bounds)
        if not fitted:
            row["session_fallback_reason"] = "reviewed outline does not fit the required span"
        return fitted

    def _fit_character(self, char: str, bounds: list) -> tuple | None:
        target = np.ptp(np.asarray(bounds).reshape(2, 2), axis=0)
        if min(target) <= 0:
            return None
        ranked = []
        for sample in self.variants[char]:
            span = np.ptp(outline_bounds(sample.outline.verts, sample.outline.codes), axis=0)
            if min(span) <= 0:
                continue
            stretch = sample.source["role"] in {"tall-delimiter", "overline"}
            if char == "|":
                stretch = True
            scale = target / span
            distortion = max(scale) / min(scale)
            if stretch and char != "|" and (distortion > 4.0):
                continue
            rank = span[0] / span[1] if char == "|" else distortion
            ranked.append((rank, sample.source["id"], sample, stretch, span))
        if not ranked:
            return None
        _, _, sample, stretch, span = min(ranked)
        if char == "|":
            x0, y0, x1, y1 = bounds
            width = min(target[0], target[1] * span[0] / span[1])
            bounds = [(x0 + x1 - width) / 2, y0, (x0 + x1 + width) / 2, y1]
        outline, scale = placed_outline(sample.outline, bounds, stretch=stretch)
        return (outline, sample, scale)

    def _square_matrix(self, page, renderer, records):
        """Keep the recorder's three pieces for each confirmed square bracket side."""
        for char, items in groupby(
            records,
            key=lambda row: (
                row["semantic"]
                if row["structure"] == "matrix-bracket" and row["matrix_environment"] == "bmatrix"
                else None
            ),
        ):
            if char not in ("[", "]") or char not in self.variants:
                continue
            side = list(items)
            if len(side) != 3:
                continue
            indices = [row["item_index"] for row in side]
            strokes = [page.items[index] for index in indices]
            if any(item[0] != "stroke" for item in strokes):
                continue
            padded = np.array(
                [
                    np.array(row["bounds_px"]) + item[4] / 2 * np.array([-1, -1, 1, 1])
                    for row, item in zip(side, strokes, strict=True)
                ]
            )
            x0, y0 = padded[:, :2].min(axis=0)
            x1, y1 = padded[:, 2:].max(axis=0)
            allowed = [x0, y0, x1, y1]
            fit = self._fit_character(char, allowed)
            if not fit:
                continue
            outline, sample, scale = fit
            path = MplPath(outline.verts, outline.codes)
            inset = (y1 - y0) * 0.15
            top, bottom = (y0 + inset, y1 - inset)
            bands = [(top - 0.3, bottom + 0.3), (y0 - 0.1, top + 0.3), (bottom - 0.3, y1 + 0.1)]
            paths = [
                path.clip_to_bbox(Bbox.from_extents(x0 - 0.1, a, x1 + 0.1, b)) for a, b in bands
            ]
            if any(len(p.vertices) == 0 for p in paths):
                continue
            for row, p, previous in zip(side, paths, padded, strict=True):
                self._draw(
                    page,
                    renderer,
                    row,
                    Outline(p.vertices, p.codes, stroke=False, digits=4),
                    allowed,
                    sample,
                    transform={
                        "rule": "fit whole scan then partition matrix bracket",
                        "scale_translate": scale,
                    },
                    paired=indices,
                )
                row["original_bounds_px"] = previous.tolist()

    def _radical_pairs(self, records: list[dict], formulas: list[dict]):
        for formula in formulas:
            start, stop = formula["item_range"]
            group = records[start:stop]
            bars = [
                r
                for r in group
                if r["structure"] == "radical-overbar"
                and r.get("fallback_reason")
                and (not r.get("paired_items"))
            ]
            for sign in group:
                if sign["structure"] != "radical-sign" or sign.get("paired_items"):
                    continue
                x, y, right, bottom = sign["bounds_px"]
                near = []
                for bar in bars:
                    bx, by, br, bb = bar["bounds_px"]
                    sx_error = abs(bx - right)
                    y_error = abs(by - y)
                    if x < bx < br and by < bb < bottom and (sx_error < 9) and (y_error < 10):
                        near.append((sx_error + 2 * y_error, bar["item_index"], bar))
                if near:
                    paired = min(near)[2]
                    bars.remove(paired)
                    yield (sign, paired)

    def apply(self, page, renderer, baseline: list[dict], *, operator_page: bool = False):
        records = [dict(row) for row in baseline]
        original = list(page.items)
        for row in records:
            if row["context"] != "mathematics":
                continue
            fit = self._glyph(row, operator_page=operator_page)
            if not fit:
                continue
            outline, sample, scale = fit
            self._draw(
                page,
                renderer,
                row,
                outline,
                row["bounds_px"],
                sample,
                transform={"rule": "fit verified contour to glyph slot", "scale_translate": scale},
            )
        for sign, bar in self._radical_pairs(records, renderer.formulas):
            bounds = sign["bounds_px"]
            target_aspect = (bar["bounds_px"][2] - bounds[0]) / (bounds[3] - bounds[1])
            connected = [sample for sample in self.radicals if sample.components == 1]
            ranked = sorted(
                connected,
                key=lambda sample: abs(
                    np.log(
                        np.diff(sample.source_bounds, axis=0)[0, 0]
                        / np.diff(sample.source_bounds, axis=0)[0, 1]
                        / target_aspect
                    )
                ),
            )
            for sample in ranked:
                fit = sample.fit(sign, bar)
                if not fit:
                    continue
                outlines, allowed = fit
                paired = [sign["item_index"], bar["item_index"]]
                for row, outline in zip((sign, bar), outlines, strict=True):
                    self._draw(
                        page,
                        renderer,
                        row,
                        outline,
                        allowed,
                        sample.glyph,
                        paired=paired,
                        transform={
                            "rule": "shared seam with independently fitted stem and roof",
                            "sample_measurements": sample.metrics(),
                        },
                    )
                break
        self._square_matrix(page, renderer, records)
        if any(
            (
                before is not after and "session_source" not in row
                for before, after, row in zip(original, page.items, records, strict=True)
            )
        ):
            raise ValueError("Session calibration changed an unassigned item")
        return records
