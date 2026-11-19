"""Explicit narrow-band rasterization of learned contours; no font pixels are copied.

Projection is deterministic and has zero trainable parameters. Any improvement from
using it on old weights is a display/representation baseline, not neural training.
"""

import numpy as np

from research.corpus.ink_geometry import signed_distance


def contour_distance(fields: np.ndarray) -> np.ndarray:
    if fields.ndim != 4 or fields.shape[1] != 1 or not np.isfinite(fields).all():
        raise ValueError("Expected finite Bx1xHxW fields")
    result = []
    for field in fields[:, 0]:
        mask = field > 0
        if not mask.any():
            distance = np.full(field.shape, -1, np.float32)
        elif mask.all():
            distance = np.ones(field.shape, np.float32)
        else:
            distance = signed_distance(mask.astype(np.float32))
        result.append(distance)
    return np.stack(result)[:, None]


def render_field(field, mode: str):
    """Exact projection forward, soft surrogate backward; biased gradient is explicit."""
    import torch

    from research.corpus.reference_generator import ink_from_sdf

    soft = ink_from_sdf(field)
    if mode == "soft":
        return soft
    if mode != "projected":
        raise ValueError("Unknown contour renderer")
    distance = torch.as_tensor(
        contour_distance(field.detach().cpu().numpy()), device=field.device, dtype=field.dtype
    )
    projected = ink_from_sdf(distance)
    if torch.is_grad_enabled() and field.requires_grad:
        return projected.detach() + (soft - soft.detach())
    return projected


def mask_scores(ink: np.ndarray, target: np.ndarray) -> dict[str, float]:
    a, b = ink > 0.5, target > 0.5
    intersection = (a & b).sum()
    return {
        "binary_iou": float(intersection / max((a | b).sum(), 1)),
        "binary_dice": float(2 * intersection / max(a.sum() + b.sum(), 1)),
        "target_core_recall": float(intersection / max(b.sum(), 1)),
        "generated_core_precision": float(intersection / max(a.sum(), 1)),
        "not_transcription_accuracy": True,
    }
