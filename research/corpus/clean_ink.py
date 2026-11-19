"""Ink calibration diagnostics; derived masks are not new real training samples."""

import cv2
import numpy as np

from research.corpus.ink_geometry import signed_distance


def sharpen_ink(ink: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """Fixed zero-update logit contrast, preserving the .5 contour and all locations."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("Positive finite gain required")
    a = np.clip(ink, 1e-7, 1 - 1e-7)
    logit = np.log(a / (1 - a)) * gain
    return (1 / (1 + np.exp(np.clip(-logit, -80, 80)))).astype(np.float32)


def redistance_batch(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stop-gradient SDF self-consistency target; invalid masks receive zero weight."""
    if prediction.ndim != 4 or prediction.shape[1] != 1 or not np.isfinite(prediction).all():
        raise ValueError("Expected Bx1xHxW")
    targets, valid = [], []
    for item in prediction[:, 0]:
        mask = item > 0
        usable = bool(mask.any() and not mask.all())
        targets.append(signed_distance(mask.astype(np.float32)) if usable else np.zeros_like(item))
        valid.append(usable)
    return np.stack(targets)[:, None], np.asarray(valid)


def ink_metrics(ink: np.ndarray) -> dict:
    if ink.ndim != 2 or not np.isfinite(ink).all():
        raise ValueError("Expected finite single-channel image")
    visible, mask = ink > 0.1, ink > 0.5
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return {
        "ink_mass": float(ink.sum()),
        "visible_pixels": int(visible.sum()),
        "core_pixels": int(mask.sum()),
        "soft_fraction": float(((ink > 0.1) & (ink < 0.9)).sum() / max(visible.sum(), 1)),
        "blank": not bool(mask.any()),
        "components": int(count - 1),
        "small_components": int(np.sum(stats[1:, cv2.CC_STAT_AREA] <= 3)),
        "border_ink": bool(
            mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        ),
        "not_content_or_author_style_score": True,
    }
