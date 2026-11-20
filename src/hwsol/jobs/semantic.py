"""Conservative content roles and formula-group constraints in page coordinates."""

import re
from statistics import median
from typing import Literal

import numpy as np

ContentRole = Literal[
    "ordinary-explanation",
    "initial-equation",
    "equation-continuation",
    "final-result",
    "inline-mathematics",
    "multi-line-derivation",
]


def content_role(text: str, kind: str, previous_math: bool = False) -> str:
    if kind == "text":
        return "ordinary-explanation"
    if text.lstrip().startswith(("=", r"\approx", r"\equiv")):
        return "equation-continuation"
    return "multi-line-derivation" if previous_math else "initial-equation"


def top_equality(text: str) -> int | None:
    depth = 0
    for i, ch in enumerate(text):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        elif ch == "=" and depth == 0:
            return i
    return None


class SemanticLayout:
    def __init__(self, renderer, styles, rng, left: float, right: float):
        self.r, self.rng, self.left, self.right = renderer, rng, left, right
        self.observations = [
            (style.sample_id, row) for style in styles for row in style.lines if row.kind == "math"
        ]

    def anchor(self, role: str) -> tuple[float, str]:
        matching = [row for _, row in self.observations if row.role == role]
        basis = "role-distribution"
        if not matching:
            matching = [row for _, row in self.observations]
            basis = "unlabelled-inset-distribution" if matching else "margin-prior"
        if not matching:
            return self.left + 0.35 * self.r.cfg.body_size, basis
        # Transfer distance from column start, never a fraction of remaining formula width.
        insets = [
            max(0, (row.inset - row.column[0]) / (row.column[1] - row.column[0]))
            for row in matching
        ]
        centre = float(np.quantile(insets, 0.25 if basis.startswith("unlabelled") else 0.5))
        mad = median(abs(v - centre) for v in insets)
        offset = self.rng.uniform(-1, 1) * min(0.008, mad * 0.25)
        fraction = min(0.28, max(0, centre + offset))
        return max(self.left, fraction * self.r.cfg.page_width), basis

    def equality_offset(self, line: dict) -> float | None:
        tex = line.get("tex", "")
        cut = top_equality(tex)
        if cut is None:
            return None
        # Use MathText's placed equality glyph, including its actual math spacing.
        _, _, items = self.r._math_item(tex, self.r.cfg.display_size)
        positions = []
        for item in items:
            if item[0] == "path" and item[1][0] == "mglyph" and item[1][3] == ord("="):
                positions.append(item[2][4])
        if not positions:
            return None
        # Internal equalities in fractions/scripts must not become the group anchor.
        ordinal = tex[:cut].count("=")
        return positions[min(ordinal, len(positions) - 1)]

    def place(self, lines: list[dict]) -> None:
        group = []
        group_id = 0
        final_hint = False

        def flush():
            nonlocal group_id, group
            if not group:
                return
            group_id += 1
            start, basis = self.anchor(group[0]["semantic"]["role"])
            offsets = [
                self.equality_offset(row) if row["kind"] == "display" else None for row in group
            ]
            widths = [
                row["img"].width
                if row["kind"] == "display"
                else self.r.matrix_width(row["prefix"], row["rows"])
                for row in group
            ]
            eq_offsets = [v for v in offsets if v is not None]
            equality = start + max(eq_offsets, default=0)
            measured_equalities = [
                (row.equality_inset - row.column[0]) / (row.column[1] - row.column[0])
                for _, row in self.observations
                if row.equality_inset is not None and row.role == group[0]["semantic"]["role"]
            ]
            if measured_equalities and eq_offsets:
                equality = max(equality, median(measured_equalities) * self.r.cfg.page_width)
            positions = [equality - v if v is not None else start for v in offsets]
            overflow = max(
                [0, *[x + w - self.right for x, w in zip(positions, widths, strict=True)]]
            )
            # Shift the whole group together; if equality alignment is infeasible, keep
            # each complete expression in its column and explicitly report the fallback.
            shift = min(overflow, max(0, min(positions) - self.left))
            positions = [x - shift for x in positions]
            feasible = all(
                x >= self.left and x + w <= self.right
                for x, w in zip(positions, widths, strict=True)
            )
            if not feasible:
                positions = [max(self.left, min(start, self.right - w)) for w in widths]
            for row, x, width in zip(group, positions, widths, strict=True):
                row["x" if row["kind"] == "display" else "cx"] = (
                    x if row["kind"] == "display" else x + width / 2
                )
                row["semantic"].update(
                    group_id=group_id,
                    start_anchor_px=start - shift,
                    equality_anchor_px=equality - shift if eq_offsets and feasible else None,
                    equality_anchor_source="role-distribution"
                    if measured_equalities
                    else "formula-geometry",
                    anchor_source=basis,
                    constraint="aligned" if feasible else "column-fit-fallback",
                )
            group = []

        for row in lines:
            if row["kind"] in {"blank", "gap"}:
                continue
            if row["kind"] not in {"display", "matrix"} or row.get("explicit_position"):
                flush()
                words = " ".join(run[1] for run in row.get("runs", ()))
                final_hint = bool(
                    re.search(r"\b(answer|result|therefore|hence|thus)\b", words, re.I)
                )
                row["semantic"] = {"role": "ordinary-explanation"}
                continue
            role = content_role(row.get("tex", row.get("prefix", "")), "math", bool(group))
            if final_hint and not group:
                role = "final-result"
            row["semantic"] = {"role": role}
            group.append(row)
        flush()
