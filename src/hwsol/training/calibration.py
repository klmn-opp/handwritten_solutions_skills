"""Calibrate from annotated pages in their native pixel or tablet coordinate space."""

from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from hwsol.profiles.schema import Digest, FrozenModel, ProfileError
from hwsol.profiles.style import LineStyle, MathStroke, PageStyle

Positive = Annotated[float, Field(gt=0)]
Point = tuple[float, float]


class PenPoint(FrozenModel):
    x: float
    y: float
    time: float | None = None
    pressure: float | None = Field(None, ge=0, le=1)


class Trace(FrozenModel):
    kind: Literal["glyph", "word", "line", "math"]
    text: str = Field(min_length=1)
    # Each new stroke starts after a pen-up. Omitted time/pressure means unknown.
    strokes: tuple[Annotated[tuple[PenPoint, ...], Field(min_length=1)], ...] = Field(min_length=1)


class AnnotatedLine(FrozenModel):
    kind: Literal["text", "math"]
    text: str = Field(min_length=1)
    baseline: tuple[Point, ...] = Field(min_length=2)
    # Ink spans in reading order, in the same coordinates as the page.
    words: tuple[Point, ...] = Field(default=(), max_length=257)


class AnnotatedPage(FrozenModel):
    schema_version: Literal[1]
    width: Positive
    height: Positive
    em: Positive
    lines: tuple[AnnotatedLine, ...] = Field(default=(), max_length=256)
    traces: tuple[Trace, ...] = ()
    math_strokes: tuple[MathStroke, ...] = Field(default=(), max_length=64)
    source_sha256: Digest | None = None

    @model_validator(mode="after")
    def geometry(self):
        if not self.lines and not self.math_strokes:
            raise ProfileError("Supply annotated lines or measured math strokes")
        for stroke in self.math_strokes:
            if stroke.source_sha256 != self.source_sha256:
                raise ProfileError("Math stroke must identify the annotated source")
            if stroke.source_box[2] > self.width or stroke.source_box[3] > self.height:
                raise ProfileError("Math stroke crop outside page")
        previous = -1.0
        for line in self.lines:
            if any(not (0 <= x <= self.width and 0 <= y <= self.height) for x, y in line.baseline):
                raise ProfileError("Baseline outside page")
            if any(a[0] >= b[0] for a, b in pairwise(line.baseline)):
                raise ProfileError("Mark baseline points from left to right")
            if line.baseline[0][1] <= previous:
                raise ProfileError("List lines in top-to-bottom reading order")
            previous = line.baseline[0][1]
            end = line.baseline[0][0]
            for left, right in line.words:
                if not end <= left < right <= line.baseline[-1][0]:
                    raise ProfileError("Word spans must be ordered and inside their line")
                end = right
        for trace in self.traces:
            for stroke in trace.strokes:
                times = [point.time for point in stroke if point.time is not None]
                if any(b < a for a, b in pairwise(times)):
                    raise ProfileError("Stroke timestamps must be monotonic")
                if any(not (0 <= p.x <= self.width and 0 <= p.y <= self.height) for p in stroke):
                    raise ProfileError("Trace outside page")
        return self


def measure(path: Path, sample_id: str) -> PageStyle:
    page = AnnotatedPage.model_validate_json(path.read_bytes())
    lines = []
    for index, line in enumerate(page.lines):
        xs, ys = np.asarray(line.baseline).T
        knots = (np.interp(np.linspace(xs[0], xs[-1], 5), xs, ys) - ys[0]) / page.em
        next_y = page.lines[index + 1].baseline[0][1] if index + 1 < len(page.lines) else None
        gaps = [(b[0] - a[1]) / page.em for a, b in pairwise(line.words)]
        lines.append(
            LineStyle(
                kind=line.kind,
                anchor=float(xs[0] / max(1, page.width - (xs[-1] - xs[0]))),
                inset=float(xs[0] / page.width),
                pitch_em=None
                if next_y is None
                else float(np.clip((next_y - ys[0]) / page.em, 1.2, 3)),
                baseline_em=tuple(np.clip(knots, -0.16, 0.16)),
                word_gaps_em=tuple(np.clip(gaps, 0.12, 1)),
            )
        )
    pressures = [
        p.pressure
        for trace in page.traces
        for stroke in trace.strokes
        for p in stroke
        if p.pressure is not None
    ]
    # Slant proxy: horizontal lean of near-vertical, downward pen travel.
    leans = [
        -(b.x - a.x) / (b.y - a.y)
        for trace in page.traces
        if trace.kind in {"word", "line", "glyph"}
        for stroke in trace.strokes
        for a, b in pairwise(stroke)
        if b.y - a.y > page.em * 0.005 and abs(b.x - a.x) < (b.y - a.y) * 0.65
    ]
    return PageStyle(
        sample_id=sample_id,
        top=max(0, page.lines[0].baseline[0][1] - 1.28 * page.em) / page.height
        if page.lines
        else 0,
        right=1 - max(line.baseline[-1][0] for line in page.lines) / page.width
        if page.lines
        else 0,
        lines=tuple(lines),
        pen_lean=median(leans) if leans else None,
        median_pressure=median(pressures) if pressures else None,
        math_strokes=page.math_strokes,
    )
