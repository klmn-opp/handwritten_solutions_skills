"""Conservative triage, not a learned erasure detector or automatic ink repair."""

from pathlib import Path

import numpy as np
from PIL import Image

DEFECTS = {
    "crossed-out",
    "occluded",
    "overwritten",
    "erasure",
    "faint",
    "blurred",
    "clipped",
    "mixed-writers",
    "printed-overlap",
    "background-interference",
    "unknown",
}


def inspect_image(path: Path) -> dict:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    gray = image.mean(axis=2)
    darkness = gray < 180
    saturation = image.max(axis=2) - image.min(axis=2)
    colored = (saturation > 65) & (gray < 225)
    row_coverage = darkness.mean(axis=1)
    col_coverage = darkness.mean(axis=0)
    edge = np.concatenate([darkness[0], darkness[-1], darkness[:, 0], darkness[:, -1]])
    stats = {
        "dark_pixel_fraction": float(darkness.mean()),
        "colored_pixel_fraction": float(colored.mean()),
        "edge_ink_fraction": float(edge.mean()),
        "long_horizontal_rows": int((row_coverage > 0.70).sum()),
        "long_vertical_columns": int((col_coverage > 0.70).sum()),
    }
    flags = []
    if stats["colored_pixel_fraction"] > 0.005:
        flags.append(
            "colored-ink-or-annotation; author must be checked, color alone is not authorship"
        )
    if stats["edge_ink_fraction"] > 0.08:
        flags.append("possible-clipping-or-grid-at-edge")
    if stats["long_horizontal_rows"] or stats["long_vertical_columns"]:
        flags.append("long-lines; may be grid, fraction, vector or strike-through")
    if stats["dark_pixel_fraction"] < 0.001:
        flags.append("blank-or-faint; do not discard automatically")
    if stats["dark_pixel_fraction"] > 0.30:
        flags.append("dense-ink-or-photo; possible occlusion")
    return {
        "method": "conservative-pixel-heuristics/1",
        "metrics": stats,
        "flags": flags,
        "review_required": True,
        "defects_verified": False,
        "limitations": (
            "Cannot reliably distinguish math strokes from crossings; "
            "no inpainting, semantic guessing or automatic acceptance."
        ),
    }
