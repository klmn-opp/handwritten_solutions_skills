"""Reference-only token positions and real-pixel component weights, CPU utilities."""

import cv2
import numpy as np


def token_positions(row: dict) -> np.ndarray:
    text = row["text"]
    provenance = row["reference_glyphs"]
    glyphs = provenance["glyph_provenance"]
    if "".join(q["semantic"] for q in glyphs) != text or len(text) > 32:
        raise ValueError("Exact single-word reference semantics required")
    transform = row["reference_transform"]
    crop = provenance["crop_from_pdf"]
    scale = transform["scale"]
    offset = transform["translation"][0] - transform["bbox"][0] * scale
    result = np.zeros((32, 2), np.float32)
    for i, glyph in enumerate(glyphs):
        left, _, right, _ = glyph["bounds_px"]
        center = ((left + right) / 2 - crop[0]) * scale + offset
        width = (right - left) * scale
        if not 0 <= center <= 512 or width <= 0:
            raise ValueError("Reference glyph out of canvas")
        result[i] = [center / 512, width / 512]
    return result


def component_weights(ink: np.ndarray) -> np.ndarray:
    if ink.shape != (128, 512) or not np.isfinite(ink).all():
        raise ValueError("Complete finite real ink required")
    mask = np.uint8(ink > 0.5)
    if not mask.any():
        raise ValueError("Blank target")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    weights = np.zeros_like(ink, dtype=np.float32)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        # Isolated single pixels must not dominate; no pixels/labels are fabricated.
        value = min(6.0, max(1.0, (128 / area) ** 0.5)) if area >= 3 else 1.0
        weights[labels == label] = value
    return weights
