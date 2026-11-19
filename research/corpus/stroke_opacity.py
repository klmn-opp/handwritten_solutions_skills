"""Real-target foreground/background supervision, with explicit uncertainty margins.

Masks are derived views of existing scans, not repaired ink or new samples.
Torch is imported only by the experimental training loss.
"""

import cv2
import numpy as np


def supervision_masks(targets: np.ndarray) -> dict[str, np.ndarray]:
    if targets.ndim != 4 or targets.shape[1] != 1 or not np.isfinite(targets).all():
        raise ValueError("Expected finite Bx1xHxW ink")
    if targets.min() < 0 or targets.max() > 1:
        raise ValueError("Ink must lie in [0, 1]")
    cores, backgrounds, weights = [], [], []
    for ink in targets[:, 0]:
        core = ink > 0.5  # no erosion: keep one-pixel dots and descenders
        far = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5) > 3
        count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8))
        weight = np.zeros_like(ink, dtype=np.float32)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            # Balance components while limiting amplification of tiny scan residue.
            weight[labels == label] = min(8.0, 32.0 / np.sqrt(area))
        cores.append(core)
        backgrounds.append(far)
        weights.append(weight)
    return {
        name: np.asarray(value, np.float32)[:, None]
        for name, value in [("core", cores), ("background", backgrounds), ("component", weights)]
    }


def opacity_losses(predicted_sdf, masks):
    """Balanced region loss plus component recall with one-pixel location tolerance."""
    import torch.nn.functional as F

    logits = predicted_sdf * (8 / 0.35)
    nearby = F.max_pool2d(logits, 3, stride=1, padding=1)

    def average(value, weight):
        return ((value * weight).sum((1, 2, 3)) / weight.sum((1, 2, 3)).clamp_min(1)).mean()

    foreground = average(F.softplus(-nearby), masks["core"])
    background = average(F.softplus(logits), masks["background"])
    components = average(F.softplus(-nearby), masks["component"])
    return (foreground + background) / 2, components


def recognition_crop(image):
    """Fixed, label-free whitespace view; original image always stays archived."""
    from PIL import Image

    gray = np.asarray(image.convert("L"))
    yy, xx = np.where(gray < 230)
    if not len(xx):
        return image.copy(), {"blank": True, "bbox": [0, 0, image.width, image.height]}
    box = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
    cropped = image.crop(box).convert("RGB")
    result = Image.new("RGB", (cropped.width + 16, cropped.height + 16), "white")
    result.paste(cropped, (8, 8))
    return result, {"blank": False, "bbox": box, "padding": 8, "threshold": 230}
