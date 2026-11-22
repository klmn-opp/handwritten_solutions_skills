"""Torch-free inference for the frozen English glyph-variation branch.

The forward graph mirrors the accepted 5cfb0165 endpoint: a bounded coarse warp,
a conditional latent prior, a f/t/h/r local residual decoder and one bilinear
resample of the fixed real base glyph. No parameter is fitted here.
"""

from __future__ import annotations

import numpy as np

from .bundle import ModelBundle
from .ops import avg_pool2d, bilinear_resize, grid_sample, linear, one_hot, silu

SIZE = 80
MAX_GRADIENT = 0.72
LOCAL_SEED_OFFSET = 314159


class EnglishGenerator:
    def __init__(self, bundle: ModelBundle):
        self.a = bundle.english
        self.letters = bundle.model["english"]["letters"]
        self.local_labels = bundle.model["english"]["local_labels"]
        self.identity = self.a["base/identity"]

    # -- graph pieces -----------------------------------------------------
    def _features(self, ink: np.ndarray) -> np.ndarray:
        return avg_pool2d(ink, 5).reshape(len(ink), -1)

    def _prior_features(self, ink: np.ndarray, label: int) -> np.ndarray:
        return np.concatenate(
            [one_hot(label, 52), avg_pool2d(ink, 10).reshape(len(ink), -1)], axis=1
        )

    def _local_features(self, ink: np.ndarray, label: int, global_z: np.ndarray) -> np.ndarray:
        code = one_hot(self.local_labels.index(self.letters[label]), 4)
        pooled = avg_pool2d(ink, 10).reshape(len(ink), -1)
        return np.concatenate([code, pooled, global_z, np.ones((1, 1), np.float32)], axis=1)

    @staticmethod
    def _gradient(field: np.ndarray) -> np.ndarray:
        dx = np.abs(np.diff(field, axis=3)).max(axis=(2, 3))
        dy = np.abs(np.diff(field, axis=2)).max(axis=(2, 3))
        return np.sqrt((dx * dx + dy * dy).sum(axis=1))

    def _prior(
        self, ink: np.ndarray, label: int, noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self._prior_features(ink, label)
        mean = np.concatenate([x, np.ones((1, 1), np.float32)], axis=1) @ self.a["prior/coef"]
        delta = linear(x, self.a["prior/residual.0.weight"], self.a["prior/residual.0.bias"])
        delta = silu(delta)
        delta = linear(delta, self.a["prior/residual.2.weight"], self.a["prior/residual.2.bias"])
        mean = mean + np.tanh(delta[:, :6]) * 1.8
        scale = np.tanh(delta[:, 6:])
        return mean + np.exp(scale) * (noise @ self.a["prior/chol"].T)

    def _local_z(self, ink: np.ndarray, label: int, global_z: np.ndarray, noise: np.ndarray):
        x = self._local_features(ink, label, global_z)
        return x @ self.a["local/coef"] + noise @ self.a["local/chol"].T

    def _base(self, ink: np.ndarray, label: int, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.concatenate(
            [self._features(ink), np.repeat(self.a["base/label.weight"][[label]], 1, 0), z], axis=1
        )
        raw = linear(x, self.a["base/decoder.0.weight"], self.a["base/decoder.0.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a["base/decoder.2.weight"], self.a["base/decoder.2.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a["base/decoder.4.weight"], self.a["base/decoder.4.bias"])
        coarse = 14.0 * np.tanh(raw.reshape(-1, 2, 5, 5))
        lipschitz = self._gradient(coarse) * (4.0 / (SIZE - 1))
        factor = np.minimum(MAX_GRADIENT / np.maximum(lipschitz, 1e-5), 1.0)
        coarse = coarse * factor[:, None, None, None]
        field = bilinear_resize(coarse, SIZE, SIZE)
        grid = self.identity + np.transpose(field, (0, 2, 3, 1)) * (2.0 / (SIZE - 1))
        return grid_sample(ink, grid), coarse

    def _local_residual(self, ink, label, global_z, local_z):
        x = np.concatenate(
            [
                self._features(ink),
                one_hot(self.local_labels.index(self.letters[label]), 4),
                global_z,
                local_z,
            ],
            axis=1,
        )
        raw = linear(x, self.a["decoder/0.weight"], self.a["decoder/0.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a["decoder/2.weight"], self.a["decoder/2.bias"])
        raw = silu(raw)
        raw = linear(raw, self.a["decoder/4.weight"], self.a["decoder/4.bias"])
        field = np.tanh(raw.reshape(-1, 2, 81))
        field = field - (field @ self.a["local/affine_pinv"].T) @ self.a["local/affine_basis"].T
        field = field / np.maximum(np.abs(field).max(axis=(1, 2)), 1.0)[:, None, None]
        return (field.reshape(-1, 2, 9, 9) * 5.0).astype(np.float32)

    # -- public -----------------------------------------------------------
    def draw(
        self,
        gray: np.ndarray,
        label: str,
        seed: int,
        *,
        global_z: np.ndarray | None = None,
        local_z: np.ndarray | None = None,
    ) -> np.ndarray:
        if gray.dtype != np.uint8 or gray.shape != (SIZE, SIZE) or label not in self.letters:
            raise ValueError("Expected a canonical 80x80 grayscale glyph and a Latin letter")
        index = self.letters.index(label)
        ink = (1.0 - gray.astype(np.float32) / 255.0)[None, None]
        rng = np.random.default_rng(int(seed))
        if global_z is None:
            noise = rng.standard_normal((1, 6)).astype(np.float32)
            global_z = self._prior(ink, index, noise)
        if label not in self.local_labels:
            out, _ = self._base(ink, index, global_z)
            return as_gray(out[0, 0])
        if local_z is None:
            noise = rng.standard_normal((1, 4)).astype(np.float32)
            local_z = self._local_z(ink, index, global_z, noise)
        _, coarse = self._base(ink, index, global_z)
        field = bilinear_resize(coarse, SIZE, SIZE)
        fine = bilinear_resize(self._local_residual(ink, index, global_z, local_z), SIZE, SIZE)
        remaining = np.maximum(0.0, MAX_GRADIENT - self._gradient(field))
        scale = np.minimum(remaining / np.maximum(self._gradient(fine), 1e-6), 1.0)
        total = field + fine * scale[:, None, None, None]
        grid = self.identity + np.transpose(total, (0, 2, 3, 1)) * (2.0 / (SIZE - 1))
        out = grid_sample(ink, grid)
        return as_gray(out[0, 0])


def as_gray(ink: np.ndarray) -> np.ndarray:
    return np.uint8(np.rint((1.0 - np.clip(ink, 0.0, 1.0)) * 255.0))
