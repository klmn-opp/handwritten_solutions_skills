"""Evaluation display fallback for light ink; never changes training preprocessing."""

import numpy as np
from PIL import Image

from research.corpus.context_training_data import whole_ink


def evaluation_ink(image: Image.Image):
    try:
        return whole_ink(image)
    except ValueError as exc:
        if str(exc) != "No dark ink in complete region":
            raise
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low, high = float(gray.min()), float(np.percentile(gray, 95))
    if high - low < 20:
        raise ValueError("Insufficient contrast for evaluation; keep original for review")
    adjusted = np.uint8(np.clip((gray - low) * 255 / (high - low), 0, 255))
    ink, transform = whole_ink(Image.fromarray(adjusted))
    return ink, {
        **transform,
        "evaluation_only_contrast_fallback": {"black": low, "white": high},
        "training_preprocessing_changed": False,
        "original_unchanged": True,
    }
