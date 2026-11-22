"""Small float32 neural operators used by the frozen next-generation model.

The published weights were trained with PyTorch but the CPU product must not
depend on Torch. These operators mirror the exact modules used at training and
inference time: Linear, SiLU, Tanh, average/max pooling, bilinear interpolation
with ``align_corners=True`` and zero-padded bilinear grid sampling.
"""

from __future__ import annotations

import numpy as np


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def avg_pool2d(x: np.ndarray, kernel: int) -> np.ndarray:
    """Non-overlapping square average pool, as ``F.avg_pool2d(x, kernel)``."""
    n, c, h, w = x.shape
    if h % kernel or w % kernel:
        raise ValueError("Average pool expects an exact multiple")
    blocks = x.reshape(n, c, h // kernel, kernel, w // kernel, kernel)
    return blocks.mean(axis=(3, 5), dtype=np.float32)


def max_pool2d(x: np.ndarray, kernel: int) -> np.ndarray:
    n, c, h, w = x.shape
    if h % kernel or w % kernel:
        raise ValueError("Max pool expects an exact multiple")
    blocks = x.reshape(n, c, h // kernel, kernel, w // kernel, kernel)
    return blocks.max(axis=(3, 5))


def adaptive_avg_pool2d(x: np.ndarray, size: int) -> np.ndarray:
    """PyTorch's adaptive average pool with equal output bins."""
    n, c, h, w = x.shape
    ys = [(i * h) // size for i in range(size + 1)]
    xs = [(i * w) // size for i in range(size + 1)]
    out = np.empty((n, c, size, size), np.float32)
    for j in range(size):
        for i in range(size):
            out[:, :, j, i] = x[:, :, ys[j] : ys[j + 1], xs[i] : xs[i + 1]].mean(axis=(2, 3))
    return out


def bilinear_resize(x: np.ndarray, height: int, width: int) -> np.ndarray:
    """``F.interpolate(mode='bilinear', align_corners=True)`` for NCHW input."""
    n, c, h, w = x.shape
    if h == height and w == width:
        return x
    ys = np.linspace(0, h - 1, height, dtype=np.float32)
    xs = np.linspace(0, w - 1, width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[None, None, :, None]
    wx = (xs - x0)[None, None, None, :]
    a = x[:, :, y0][:, :, :, x0]
    b = x[:, :, y0][:, :, :, x1]
    d = x[:, :, y1][:, :, :, x0]
    e = x[:, :, y1][:, :, :, x1]
    return ((a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy).astype(np.float32)


def grid_sample(anchor: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``F.grid_sample(align_corners=True, padding_mode='zeros')`` for NCHW."""
    n, c, h, w = anchor.shape
    if grid.shape != (n, h, w, 2):
        raise ValueError(f"Unexpected grid shape {grid.shape}")
    result = np.zeros((n, c, h, w), np.float32)
    for i in range(n):
        gx = grid[i, :, :, 0].astype(np.float32)
        gy = grid[i, :, :, 1].astype(np.float32)
        ix = (gx + 1.0) * 0.5 * (w - 1)
        iy = (gy + 1.0) * 0.5 * (h - 1)
        x0 = np.floor(ix).astype(np.int64)
        y0 = np.floor(iy).astype(np.int64)
        wx = (ix - x0).astype(np.float32)
        wy = (iy - y0).astype(np.float32)
        for yy, xx, wyi, wxi in (
            (y0, x0, 1 - wy, 1 - wx),
            (y0, x0 + 1, 1 - wy, wx),
            (y0 + 1, x0, wy, 1 - wx),
            (y0 + 1, x0 + 1, wy, wx),
        ):
            inside = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            ys = np.clip(yy, 0, h - 1)
            xs = np.clip(xx, 0, w - 1)
            weight = (wyi * wxi * inside).astype(np.float32)
            result[i] += anchor[i][:, ys, xs] * weight[None]
    return result


def one_hot(index: int | np.ndarray, classes: int) -> np.ndarray:
    values = np.asarray(index, dtype=np.int64).reshape(-1)
    out = np.zeros((values.size, classes), np.float32)
    out[np.arange(values.size), values] = 1.0
    return out


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, padding: int) -> np.ndarray:
    """Square 3x3/5x5 convolution with zero padding, stride one."""
    n, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    if kh != kw:
        raise ValueError("Only square kernels are supported")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw), axis=(2, 3))
    # windows: N, C, H, W, kh, kw
    out = np.einsum("nchwuv,ocuv->nohw", windows, weight, optimize=True)
    return (out + bias[None, :, None, None]).astype(np.float32)
