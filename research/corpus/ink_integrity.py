"""Frozen geometric rendering diagnostics. No neural parameter updates or ink repair."""

import cv2
import numpy as np

HEIGHT, WIDTH = 128, 512
PAD_X, PAD_Y = 64, 32


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        a[None].astype(np.float64), (0, 0), sigmaX=sigma, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )[0]


def _lipschitz(a: np.ndarray, limit: float) -> np.ndarray:
    result = a.copy()
    for i in range(1, len(a)):
        result[i] = np.clip(result[i], result[i - 1] - limit, result[i - 1] + limit)
    for i in range(len(a) - 2, -1, -1):
        result[i] = np.clip(result[i], result[i + 1] - limit, result[i + 1] + limit)
    return result


def map_components(controls: np.ndarray, *, guarded: bool) -> dict[str, np.ndarray]:
    """Triangular map x=f(x), y=63.5+a(x)*(y-63.5)+b(x)."""
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape != (2, 5, 17) or not np.isfinite(controls).all():
        raise ValueError("Finite 2x5x17 controls required")
    if np.max(np.abs(controls[0] - controls[0, :1])) > 1e-4:
        raise ValueError("Only column-wise monotone word maps are supported")
    nodes = np.linspace(0, WIDTH - 1, 17)
    x = np.arange(WIDTH, dtype=np.float64)
    fx = x + np.interp(x, nodes, controls[0, 0])
    lo = np.interp(x, nodes, controls[1, 0])
    hi = np.interp(x, nodes, controls[1, -1])
    scale = 1 + (hi - lo) / (HEIGHT - 1)
    shift = (hi + lo) / 2
    expected = (
        controls[1, 0][None]
        + np.linspace(0, 1, 5)[:, None] * (controls[1, -1] - controls[1, 0])[None]
    )
    if (
        np.max(np.abs(expected - controls[1])) > 1e-4
        or (np.diff(fx) <= 0).any()
        or (scale <= 0).any()
    ):
        raise ValueError("Folded or non-affine vertical map")
    if guarded:
        derivative = _blur(np.diff(fx), 12)
        # Euclidean projection onto bounded slopes with fixed integral; not a
        # global interpolation of the image back toward the font.
        low, high = -10.0, 10.0
        for _ in range(60):
            mid = (low + high) / 2
            if np.clip(derivative + mid, 0.8, 1.25).mean() > 1:
                high = mid
            else:
                low = mid
        derivative = np.clip(derivative + (low + high) / 2, 0.8, 1.25)
        fx = np.r_[0, np.cumsum(derivative)]
        scale = np.clip(_lipschitz(_blur(scale, 12), 0.0015), 0.8, 1.2)
        shift = np.clip(_lipschitz(_blur(shift, 12), 0.06), -8, 8)
    return {"x": fx, "scale": scale, "shift": shift}


def coordinates(parts: dict, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nodes = np.arange(WIDTH)
    # Identity outside horizontal endpoints. Source glyphs are inside the word canvas.
    sx = np.interp(x, nodes, parts["x"])
    sx = np.where(x < 0, x, np.where(x > WIDTH - 1, x, sx))
    scale = np.interp(x, nodes, parts["scale"])
    shift = np.interp(x, nodes, parts["shift"])
    sy = 63.5 + scale * (y - 63.5) + shift
    return sx, sy


def field_metrics(parts: dict) -> dict:
    horizontal = np.diff(parts["x"])
    scale = parts["scale"]
    shear = np.abs(np.diff(parts["shift"])) + 63.5 * np.abs(np.diff(scale))
    dx = parts["x"] - np.arange(WIDTH)
    vertical = np.stack(
        [-0.5 * 127 * (scale - 1) + parts["shift"], 0.5 * 127 * (scale - 1) + parts["shift"]]
    )
    return {
        "horizontal_min": float(horizontal.min()),
        "horizontal_max": float(horizontal.max()),
        "vertical_min": float(scale.min()),
        "vertical_max": float(scale.max()),
        "shear_bound": float(shear.max()),
        "displacement_rms": float(np.sqrt((np.mean(dx**2) + np.mean(vertical**2)) / 2)),
    }


def ink_from_reference(image) -> np.ndarray:
    return np.clip((190 - np.asarray(image.convert("L"), dtype=np.float32)) / 140, 0, 1)


def sample_reference(source: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    if source.ndim != 2 or sx.shape != sy.shape:
        raise ValueError("A single source plane and matching coordinate arrays are required")
    return cv2.remap(
        source.astype(np.float32),
        sx.astype(np.float32),
        sy.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def effective_scale(transform: dict) -> tuple[float, float]:
    x, y, right, bottom = transform["bbox"]
    nominal = transform["scale"]
    return (
        round((right - x) * nominal) / (right - x),
        round((bottom - y) * nominal) / (bottom - y),
    )


def render_word(
    source: np.ndarray,
    transform: dict,
    parts: dict,
    *,
    raster_scale: int = 4,
    supersampling: int = 4,
) -> np.ndarray:
    """Sample original high resolution font once; retain old normalized coordinates."""
    if raster_scale < 1 or supersampling < 1:
        raise ValueError("Positive raster scales required")
    w, h = WIDTH + 2 * PAD_X, HEIGHT + 2 * PAD_Y
    xx = (np.arange(w * supersampling) + 0.5) / supersampling - 0.5 - PAD_X
    yy = (np.arange(h * supersampling) + 0.5) / supersampling - 0.5 - PAD_Y
    sx, sy = coordinates(parts, xx[None], yy[:, None])
    sx = np.broadcast_to(sx, sy.shape)
    # PIL normalization resizes pixel areas; account for pixel-center alignment.
    scalex, scaley = effective_scale(transform)
    dx, dy = transform["translation"]
    x, y = transform["bbox"][:2]
    sx = ((sx - dx + 0.5) / scalex + x) * raster_scale - 0.5
    sy = ((sy - dy + 0.5) / scaley + y) * raster_scale - 0.5
    patch = sample_reference(source, sx, sy)
    return cv2.resize(patch, (w, h), interpolation=cv2.INTER_AREA)


def untrained_control(parts: dict, seed: int) -> tuple[dict, dict]:
    """Matched-amplitude random geometry; no labels, real scans or learned weights."""
    rng = np.random.default_rng(seed)
    nodes = np.linspace(0, WIDTH - 1, 17)
    raw = np.zeros((2, 5, 17), np.float64)
    intervals = np.exp(rng.normal(0, 0.18, 16))
    x = np.r_[0, np.cumsum(intervals / intervals.sum() * 511)]
    raw[0] = np.broadcast_to(x - nodes, (5, 17))
    s = rng.normal(0, 0.04, 17)
    b = rng.normal(0, 3, 17)
    raw[1] = np.linspace(-63.5, 63.5, 5)[:, None] * s[None] + b[None]
    random = map_components(raw, guarded=True)
    target = field_metrics(parts)["displacement_rms"]
    actual = field_metrics(random)["displacement_rms"]
    factor = target / max(actual, 1e-12)
    dx = random["x"] - np.arange(WIDTH)
    a = random["scale"] - 1
    b = random["shift"]
    random = {"x": np.arange(WIDTH) + dx * factor, "scale": 1 + a * factor, "shift": b * factor}
    # If matching amplitude would violate safety, retain the constrained result
    # and report imperfect matching rather than quietly claim equal amplitude.
    raw[0] = np.interp(nodes, np.arange(WIDTH), random["x"]) - nodes
    aa = np.interp(nodes, np.arange(WIDTH), random["scale"])
    bb = np.interp(nodes, np.arange(WIDTH), random["shift"])
    raw[1] = np.linspace(-63.5, 63.5, 5)[:, None] * (aa - 1)[None] + bb[None]
    if (np.diff(nodes + raw[0, 0]) <= 0).any() or (aa <= 0).any():
        return map_components(np.zeros((2, 5, 17)), guarded=False), {
            "valid_control": False,
            "reason": "amplitude matching exceeded valid map",
            "source": "not usable; do not call this random output",
        }
    random = map_components(raw, guarded=True)
    final = field_metrics(random)["displacement_rms"]
    return random, {
        "valid_control": True,
        "target_rms": target,
        "actual_rms": final,
        "relative_error": abs(final - target) / max(target, 1e-12),
        "neural_updates": 0,
    }
