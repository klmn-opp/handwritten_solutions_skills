"""Portable, dimensionless observations. Raw tablet coordinates stay in Studio."""

from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, model_validator

from hwsol.jobs.semantic import ContentRole
from hwsol.profiles.schema import FrozenModel

Fraction = Annotated[float, Field(ge=0, le=1)]
Offset = Annotated[float, Field(ge=-0.16, le=0.16)]
Gap = Annotated[float, Field(ge=0.12, le=1.0)]


class MathStroke(FrozenModel):
    """Measured horizontal ink envelope, in em units with left-to-right samples."""

    kind: Literal["bar", "rule"]
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_box: tuple[int, int, int, int]
    width_em: float = Field(gt=0, le=100)
    # x fraction, centre y (positive down), thickness. These are ink measurements.
    envelope: tuple[tuple[Fraction, float, Annotated[float, Field(gt=0, le=1)]], ...] = Field(
        min_length=3, max_length=256
    )

    @model_validator(mode="after")
    def geometry(self):
        x0, y0, x1, y1 = self.source_box
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise ValueError("Invalid source crop")
        xs = [point[0] for point in self.envelope]
        if xs[0] != 0 or xs[-1] != 1 or any(b <= a for a, b in pairwise(xs)):
            raise ValueError("Ink envelope must run from x=0 to x=1")
        if any(abs(y) > 1 for _, y, _ in self.envelope):
            raise ValueError("Horizontal ink envelope exceeds one em")
        return self


class LineStyle(FrozenModel):
    kind: Literal["text", "math"]
    role: ContentRole | None = None
    column: tuple[Fraction, Fraction] = (0, 1)
    equality_inset: Fraction | None = None
    # Position in the spare horizontal space. 0 = flush left, 0.5 = centred.
    anchor: Fraction
    inset: Fraction
    pitch_em: float | None = Field(None, ge=1.2, le=3)
    baseline_em: tuple[Offset, Offset, Offset, Offset, Offset]
    word_gaps_em: tuple[Gap, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def column_bounds(self):
        if self.column[0] >= self.column[1]:
            raise ValueError("Column must have positive width")
        return self


class PageStyle(FrozenModel):
    sample_id: str = Field(pattern=r"^s-[a-f0-9]{32}$")
    top: Fraction
    right: Fraction
    lines: tuple[LineStyle, ...] = Field(default=(), max_length=256)
    pen_lean: float | None = Field(None, ge=-0.65, le=0.65)
    median_pressure: Fraction | None = None
    math_strokes: tuple[MathStroke, ...] = Field(default=(), max_length=64)
    trace_inventory: dict[Literal["recorded-pen", "estimated-trajectory", "scan-contour"], int] = (
        Field(default_factory=dict)
    )

    @model_validator(mode="after")
    def observations(self):
        if not self.lines and not self.math_strokes and not any(self.trace_inventory.values()):
            raise ValueError("A style needs line observations or measured math strokes")
        return self
