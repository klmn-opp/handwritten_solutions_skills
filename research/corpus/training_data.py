"""Versioned offline input views; raw personal crops are never overwritten."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MODES = {"raw-gray", "dark-ink-v1"}


def ink_view(path: Path, mode: str = "raw-gray") -> np.ndarray:
    if mode not in MODES:
        raise ValueError("Unknown training image preprocessing")
    with Image.open(path) as im:
        image = ImageOps.pad(im.convert("L"), (64, 64), color=255)
        gray = np.asarray(image, dtype=np.float32)
    if mode == "dark-ink-v1":
        # Fixed before training: suppress light paper/grid, keep soft dark-stroke edges.
        # Not suitable for faint or colored ink; Sol's accepted set excludes these.
        return np.clip((190 - gray) / 140, 0, 1)
    return 1 - gray / 255
