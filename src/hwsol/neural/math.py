"""Torch-free inference for the frozen mathematical glyph-variation branch.

Mirrors the accepted 757afef8 endpoint: a condition-and-support gated 9x9
displacement field, one bounded bilinear sample of the real base glyph, a
topology guard and the frozen single-symbol recognition check.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .bundle import ModelBundle
from .ops import (
    adaptive_avg_pool2d,
    avg_pool2d,
    bilinear_resize,
    conv2d,
    grid_sample,
    linear,
    max_pool2d,
    one_hot,
    silu,
)

SIZE = 80
MAX_DISPLACEMENT = 4.0
MAX_GRADIENT = 0.45
THRESHOLD = 180


def canonical(image: Image.Image) -> tuple[np.ndarray, dict]:
    """Keep aspect, grayscale and component relationships on an 80x80 canvas."""
    gray = np.asarray(image.convert("L"))
    ys, xs = np.where(gray < THRESHOLD)
    if not len(xs):
        raise ValueError("Empty real source")
    bbox = [
        max(0, int(xs.min()) - 2),
        max(0, int(ys.min()) - 2),
        min(gray.shape[1], int(xs.max()) + 3),
        min(gray.shape[0], int(ys.max()) + 3),
    ]
    crop = Image.fromarray(gray).crop(bbox)
    scale = 68 / max(crop.size)
    dims = tuple(max(1, round(v * scale)) for v in crop.size)
    crop = crop.resize(dims, Image.Resampling.LANCZOS)
    out = Image.new("L", (SIZE, SIZE), 255)
    pos = ((SIZE - dims[0]) // 2, (SIZE - dims[1]) // 2)
    out.paste(crop, pos)
    return np.asarray(out), dict(
        bbox=bbox, scale=scale, size=list(dims), offset=list(pos), threshold=THRESHOLD
    )


def topology(gray: np.ndarray) -> list[int]:
    import cv2

    ink = (np.asarray(gray) < 160).astype("uint8")
    components = cv2.connectedComponents(ink, connectivity=8)[0] - 1
    holes = (
        cv2.connectedComponents(
            np.pad(~ink.astype(bool), 1, constant_values=True).astype("uint8"), connectivity=4
        )[0]
        - 2
    )
    return [int(components), int(holes)]


class MathGenerator:
    def __init__(self, bundle: ModelBundle):
        self.a = bundle.math
        self.classes: list[str] = bundle.model["math"]["classes"]
        self.recognition_labels: list[str] = bundle.model["math"]["recognition_labels"]
        self.targets = set(bundle.model["math"]["targets"])
        self.support: dict = bundle.model["math"]["support"]

    def index(self, label: str, role: str = "ordinary") -> int:
        key = label + "\u241f" + role
        if key not in self.classes:
            raise ValueError(f"Unsupported semantic/role {key!r}")
        return self.classes.index(key)

    @staticmethod
    def _features(ink: np.ndarray) -> np.ndarray:
        return avg_pool2d(ink, 8).reshape(len(ink), -1)

    def _field(self, canonical_gray: np.ndarray, label: str, seed: int, role: str, z):
        index = self.index(label, role)
        key = self.classes[index]
        support = self.support[key]
        if not support["stochastic_pairs"]:
            z = np.zeros((1, 4), np.float32)
        model = "learned" if key in self.targets else "parent"
        a = (1.0 - canonical_gray.astype(np.float32) / 255.0)[None, None]
        x = np.concatenate([self._features(a), one_hot(index, len(self.classes)), z], axis=1)
        raw = linear(x, self.a[f"{model}/decoder.0.weight"], self.a[f"{model}/decoder.0.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a[f"{model}/decoder.2.weight"], self.a[f"{model}/decoder.2.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a[f"{model}/decoder.4.weight"], self.a[f"{model}/decoder.4.bias"])
        field = np.tanh(raw.reshape(-1, 2, 81))
        field = field - (field @ self.a[f"{model}/pinv"].T) @ self.a[f"{model}/basis"].T
        field = field / np.maximum(np.abs(field).max(axis=(1, 2)), 1.0)[:, None, None]
        field = (field.reshape(-1, 2, 9, 9) * MAX_DISPLACEMENT).astype(np.float32)
        field = bilinear_resize(field, SIZE, SIZE)
        field = field * self.a[f"{model}/amplitude"][index]
        gradient = self._gradient(field)
        field = (
            field * np.minimum(MAX_GRADIENT / np.maximum(gradient, 1e-6), 1.0)[:, None, None, None]
        )
        grid = self.a[f"{model}/identity"] + np.transpose(field, (0, 2, 3, 1)) * (2.0 / (SIZE - 1))
        return grid_sample(a, grid), field

    @staticmethod
    def _gradient(field: np.ndarray) -> np.ndarray:
        dx = np.abs(np.diff(field, axis=3)).max(axis=(2, 3))
        dy = np.abs(np.diff(field, axis=2)).max(axis=(2, 3))
        return np.sqrt((dx * dx + dy * dy).sum(axis=1))

    def field(self, canonical_gray: np.ndarray, label: str, seed: int, role: str = "ordinary"):
        if canonical_gray.dtype != np.uint8 or canonical_gray.shape != (SIZE, SIZE):
            raise ValueError("Need canonical 80x80 uint8")
        z = np.random.default_rng(int(seed)).standard_normal((1, 4)).astype(np.float32)
        out, field = self._field(canonical_gray, label, seed, role, z)
        return field[0].numpy() if hasattr(field, "numpy") else field[0], out

    def warp_native(
        self, gray: np.ndarray, label: str, seed: int, role: str = "ordinary", guard: bool = True
    ) -> tuple[np.ndarray, dict]:
        import cv2

        gray = np.asarray(gray)
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("Expected native grayscale uint8")
        can, transform = canonical(Image.fromarray(gray))
        field, _ = self.field(can, label, seed, role)
        x0, y0, x1, y1 = transform["bbox"]
        w, h = transform["size"]
        ox, oy = transform["offset"]
        sx, sy = w / (x1 - x0), h / (y1 - y0)
        y, x = np.indices(gray.shape, dtype="float32")
        u = (x - x0 + 0.5) * sx - 0.5 + ox
        v = (y - y0 + 0.5) * sy - 0.5 + oy
        dx = cv2.remap(field[0], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / sx
        dy = cv2.remap(field[1], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) / sy
        warped = cv2.remap(
            gray, x + dx, y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        before, after = topology(gray), topology(warped)
        rejected = guard and before != after
        diag = dict(
            transform=transform,
            topology_before=before,
            topology_after=after,
            identity_fallback=bool(rejected),
            guard=guard,
            max_native_displacement=float(max(abs(dx).max(), abs(dy).max())),
        )
        return (gray.copy() if rejected else warped), diag

    def predict(self, gray: np.ndarray) -> dict:
        can, _ = canonical(Image.fromarray(np.asarray(gray, dtype=np.uint8)))
        x = (1.0 - can.astype(np.float32) / 255.0)[None, None]
        x = silu(
            conv2d(
                x, self.a["recognizer/features.0.weight"], self.a["recognizer/features.0.bias"], 1
            )
        )
        x = max_pool2d(x, 2)
        x = silu(
            conv2d(
                x, self.a["recognizer/features.3.weight"], self.a["recognizer/features.3.bias"], 1
            )
        )
        x = max_pool2d(x, 2)
        x = silu(
            conv2d(
                x, self.a["recognizer/features.6.weight"], self.a["recognizer/features.6.bias"], 1
            )
        )
        x = adaptive_avg_pool2d(x, 5).reshape(1, -1)
        x = silu(
            linear(
                x, self.a["recognizer/embedding.0.weight"], self.a["recognizer/embedding.0.bias"]
            )
        )
        logits = linear(x, self.a["recognizer/head.weight"], self.a["recognizer/head.bias"])[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        return dict(label=self.recognition_labels[index], probability=float(probabilities[index]))
