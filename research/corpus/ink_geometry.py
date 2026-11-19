"""Dependency-light distance representation shared by CPU data and isolated CUDA workers."""

import cv2
import numpy as np


def signed_distance(ink: np.ndarray) -> np.ndarray:
    mask = (ink > 0.5).astype(np.uint8)
    if not mask.any() or mask.all():
        raise ValueError("Empty or solid training image")
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
    # Subpixel boundary lies halfway between opposite pixel centers.
    result = np.where(mask, inside - 0.5, -(outside - 0.5))
    return np.clip(result / 8, -1, 1).astype(np.float32)
