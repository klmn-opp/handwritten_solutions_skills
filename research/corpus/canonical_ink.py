"""Reversible canonical coordinates for real ink; never creates or repairs strokes."""

import cv2
import numpy as np


def coordinate_maps(
    control: np.ndarray, *, guarded: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if control.shape != (2, 5, 17) or not np.isfinite(control).all():
        raise ValueError("Finite triangular registration required")
    if not np.allclose(control[0], control[0, 0], atol=1e-4):
        raise ValueError("Horizontal map must not depend on row")
    if not np.allclose(control[1], np.linspace(control[1, 0], control[1, -1], 5), atol=1e-4):
        raise ValueError("Vertical map must be affine per column")
    x = np.arange(512, dtype=np.float32)
    y = np.arange(128, dtype=np.float32)
    nodes = np.linspace(0, 511, 17)
    dx = np.interp(x, nodes, control[0, 0])
    b = np.interp(x, nodes, control[1, 0])
    a = 1 + (np.interp(x, nodes, control[1, -1]) - b) / 127
    source_x = x + dx
    if np.min(np.diff(source_x)) <= 0 or np.min(a) <= 0:
        raise ValueError("Folded map rejected")
    if guarded:
        from research.corpus.ink_integrity import map_components

        parts = map_components(control, guarded=True)
        source_x = parts["x"]
        a = parts["scale"]
        b = 63.5 * (1 - a) + parts["shift"]
    inverse_x = np.interp(x, source_x, x)
    inverse_a = np.interp(inverse_x, x, a)
    inverse_b = np.interp(inverse_x, x, b)
    return (
        np.broadcast_to(source_x, (128, 512)).astype(np.float32),
        (a[None] * y[:, None] + b[None]).astype(np.float32),
        np.broadcast_to(inverse_x, (128, 512)).astype(np.float32),
        ((y[:, None] - inverse_b[None]) / inverse_a[None]).astype(np.float32),
    )


def canonicalize(
    target: np.ndarray, control: np.ndarray, *, guarded: bool = False
) -> tuple[np.ndarray, dict]:
    if target.shape != (128, 512) or not np.isfinite(target).all():
        raise ValueError("Whole real ink image required")
    fx, fy, ix, iy = coordinate_maps(control, guarded=guarded)
    canonical = cv2.remap(
        target, ix, iy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    recovered = cv2.remap(
        canonical, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    original = target > 0.5
    restored = recovered > 0.5
    new = canonical > 0.5
    outside = ((fx < 0) | (fx > 511) | (fy < 0) | (fy > 127)) & original
    dice = float(2 * (original & restored).sum() / max(1, original.sum() + restored.sum()))
    border = bool(new[:2].any() or new[-2:].any() or new[:, :2].any() or new[:, -2:].any())
    report = {
        "roundtrip_dice": dice,
        "source_ink_outside_map": int(outside.sum()),
        "canonical_border_ink": border,
        "blank": not bool(new.any()),
        "resampled_real_pixels": True,
        "geometry_guard": guarded,
        "synthetic_stroke_repair": False,
    }
    report["eligible"] = dice >= 0.90 and not outside.any() and not border and bool(new.any())
    return canonical, report
