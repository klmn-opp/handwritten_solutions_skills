"""Measure word gaps and baseline envelopes inside transcribed, single-line scan regions."""

from itertools import pairwise
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import Field

from hwsol.profiles.io import sha256
from hwsol.profiles.schema import FrozenModel, ProfileError
from hwsol.training.calibration import AnnotatedLine, AnnotatedPage, Positive


class Region(FrozenModel):
    kind: Literal["text", "math"]
    text: str = Field(min_length=1)
    box: tuple[int, int, int, int]


class Regions(FrozenModel):
    em: Positive
    lines: tuple[Region, ...] = Field(min_length=1, max_length=256)


def measure_scan(source: Path, annotations: Path, output: Path) -> dict:
    regions = Regions.model_validate_json(annotations.read_bytes())
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("L")
    lines = []
    for row in regions.lines:
        x0, y0, x1, y1 = row.box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ProfileError("Line box outside the oriented image")
        tile = np.asarray(image.crop(row.box))
        _, mask = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        xs = np.flatnonzero(mask.any(axis=0))
        if xs.size < 5 or mask.mean() > 128:
            raise ProfileError("Empty or mostly dark line region; inspect the crop")
        xs = xs.astype(int)
        bottom = mask.shape[0] - 1 - np.argmax(mask[::-1, xs], axis=0)
        knots = [
            (int(np.median(col)), float(np.median(base)) + y0)
            for col, base in zip(np.array_split(xs, 5), np.array_split(bottom, 5), strict=True)
        ]
        knots[0] = (int(xs[0]), knots[0][1])
        knots[-1] = (int(xs[-1]) + 1, knots[-1][1])
        spans = []
        if row.kind == "text":
            count = len(row.text.split())
            gaps = sorted(((b - a - 1, a, b) for a, b in pairwise(xs) if b - a > 1), reverse=True)
            if not count or count > len(gaps) + 1:
                raise ProfileError("Cannot separate the transcribed words; annotate spans manually")
            cuts = sorted((a, b) for _, a, b in gaps[: count - 1])
            starts = [xs[0], *(b for _, b in cuts)]
            ends = [*(a + 1 for a, _ in cuts), xs[-1] + 1]
            spans = [(int(a) + x0, int(b) + x0) for a, b in zip(starts, ends, strict=True)]
        lines.append(
            AnnotatedLine(
                kind=row.kind,
                text=row.text,
                words=tuple(spans),
                baseline=tuple((x + x0, y) for x, y in knots),
            )
        )
    page = AnnotatedPage(
        schema_version=1,
        width=image.width,
        height=image.height,
        em=regions.em,
        lines=tuple(lines),
        source_sha256=sha256(source),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page.model_dump_json(indent=2, exclude_none=True) + "\n")
    return {
        "output": str(output),
        "lines": len(lines),
        "source_sha256": page.source_sha256,
        "method": "annotated scan, Otsu ink projection; inspect word spans before import",
    }
