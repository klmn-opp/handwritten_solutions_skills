"""CPU kernel extracted from research.homework_benchmark.render; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import json
from pathlib import Path

from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_scans(folder: Path) -> list[ScanGlyph]:
    return [
        ScanGlyph.load(folder, row)
        for row in read(folder / "sources.json")
        if row["review_status"] == "agent-accepted"
    ]
