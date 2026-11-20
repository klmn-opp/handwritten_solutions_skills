"""Bounded real-glyph selection, with measured ink and correlated text slant."""

import math
import random
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from handnote.variants import base_outline, handwritten_outline
from matplotlib.path import Path as MplPath

from hwsol.profiles.store import LoadedProfile
from hwsol.render_policy import RenderPolicy


def fitted_variant(outline, shape: tuple, font: Path, code: int, slot: int, *, normalized=False):
    points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
    low, high = points.min(axis=0), points.max(axis=0)
    if shape[0] == "glyph":
        primary, primary_codes, upem = base_outline(font, code)
        base = primary[primary_codes != MplPath.CLOSEPOLY]
        vertices, codes = handwritten_outline(font, code, slot, upem)
        ink = vertices[codes != MplPath.CLOSEPOLY]
        lo, hi = ink.min(axis=0), ink.max(axis=0)
        base_low, base_high = base.min(axis=0), base.max(axis=0)
        base_height, height = base_high[1] - base_low[1], hi[1] - lo[1]
        if height > 0 and base_height > 0 and not normalized:
            # Keep ordinary sample variation and real ascenders/descenders. Bound
            # outliers against this character's primary, not the randomly drawn slot.
            target = np.clip(height, 0.95 * base_height, 1.05 * base_height)
            if target != height:
                factor = target / height
                vertices = vertices * factor
                vertices[:, 1] += (base_low[1] + base_high[1]) / 2 - (lo[1] + hi[1]) / 2 * factor
        ink = vertices[codes != MplPath.CLOSEPOLY]
        lo, hi = ink.min(axis=0), ink.max(axis=0)
        # Only fit width into the allocated advance; never enlarge a narrow shape
        # to fill a wide slot, or shrink its height to fit a narrow one.
        horizontal = min(1.0, (high[0] - low[0]) / max(hi[0] - lo[0], 1e-8))
        vertices[:, 0] = (vertices[:, 0] - (lo[0] + hi[0]) / 2) * horizontal + (
            low[0] + high[0]
        ) / 2
        return replace(outline, verts=vertices, codes=codes)
    vertices, codes = handwritten_outline(font, code, slot, 100)
    ink = vertices[codes != MplPath.CLOSEPOLY]
    lo, hi = ink.min(axis=0), ink.max(axis=0)
    ratios = (high - low) / np.maximum(hi - lo, 1e-8)
    scale = np.array([min(1.0, ratios[0]), 1.0 if normalized else ratios[1]])
    fitted = (vertices - (lo + hi) / 2) * scale + (low + high) / 2
    if normalized:
        fitted[:, 1] = vertices[:, 1]
    vertices = fitted
    return replace(outline, verts=vertices, codes=codes)


@lru_cache(maxsize=4096)
def _ink_width(vertices: bytes, codes: bytes, em: float) -> float:
    verts = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 2)
    path = MplPath(verts * (192 / em), np.frombuffer(codes, dtype=np.uint8))
    polygons = path.to_polygons()
    if not polygons:
        return 0.0
    points = np.concatenate(polygons)
    low = np.floor(points.min(axis=0)) - 3
    size = np.ceil(points.max(axis=0) - low).astype(int) + 4
    if min(size) < 1 or max(size) > 4096:
        raise ValueError("Glyph too large for ink measurement")
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    contours = [np.rint(p - low).astype(np.int32) for p in polygons]
    # A single drawContours call applies the even-odd fill rule, preserving counters.
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (distance > 0) & (distance >= cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    return float(2 * np.median(distance[ridge]) / 192) if ridge.any() else 0.0


def ink_width(vertices: np.ndarray, codes: np.ndarray, em: float) -> float:
    """Approximate median ridge diameter/em; not a pressure or naturalness score."""
    return _ink_width(
        np.asarray(vertices, dtype=np.float64).tobytes(),
        np.asarray(codes, dtype=np.uint8).tobytes(),
        float(em),
    )


@lru_cache(maxsize=32)
def font_ink(font: str, digest: str, personal: tuple[int, ...]) -> tuple[float, int]:
    # The immutable font digest deliberately participates in the cache identity.
    with TTFont(font) as source:
        upem = source["head"].unitsPerEm
    widths = []
    for code in personal:
        if chr(code).isalnum():
            vertices, codes, _ = base_outline(Path(font), code)
            widths.append(ink_width(vertices, codes, upem))
    if not widths:
        raise ValueError("Ink policy needs personal letter or digit samples")
    return float(np.median(widths)), upem


class PolicyInk:
    def __init__(self, policy: RenderPolicy, profile: LoadedProfile, seed: str):
        if policy.profile_sha256 != profile.manifest_sha256:
            raise ValueError("Renderer policy belongs to a different Profile snapshot")
        self.policy = policy
        self.rng = random.Random(seed + "|policy")
        self.action = self.rng.choices(policy.actions, policy.probabilities)[0]
        font_asset = next(a for a in profile.manifest.assets if a.role == "font")
        self.target, self.upem = font_ink(
            str(profile.font.resolve()),
            font_asset.sha256,
            tuple(profile.manifest.personal_codepoints),
        )
        self.last = {}
        self.previous_width = self.target
        self.widths = []
        self.variants = defaultdict(list)
        self.choices = {}
        self.drift = 0.0
        self.cache = {}

    def width(self, outline, shape: tuple) -> float:
        return ink_width(outline.verts, outline.codes, 100 if shape[0] == "mglyph" else self.upem)

    def choose(self, shape, outline, font, code, choices) -> int:
        if choices == 1:
            return 0
        widths = []
        for slot in range(choices):
            key = (shape, slot)
            if key not in self.cache:
                self.cache[key] = self.width(
                    fitted_variant(outline, shape, font, code, slot), shape
                )
            widths.append(max(1e-5, self.cache[key]))
        action = self.action
        scores = [
            -action.ink_balance * math.log(width / self.target) ** 2
            - action.continuity * math.log(width / self.previous_width) ** 2
            - action.repeat_penalty * (self.last.get(code) == slot)
            for slot, width in enumerate(widths)
        ]
        weights = np.exp(np.asarray(scores) - max(scores))
        # Retain a small chance for every real sample, even with a strong ink preference.
        probabilities = 0.94 * weights / weights.sum() + 0.06 / choices
        return self.rng.choices(range(choices), probabilities)[0]

    def lean(self) -> float:
        self.drift = 0.85 * self.drift + 0.15 * self.rng.uniform(-1, 1)
        return self.action.lean + self.action.lean_variation * self.drift

    def observe(self, outline, shape, code, slot, choices):
        self.last[code] = slot
        self.choices[code] = choices
        self.variants[code].append(slot)
        if chr(code).isalnum():
            width = max(1e-5, self.width(outline, shape))
            self.widths.append(width)
            self.previous_width = width

    def report(self) -> dict:
        widths = np.asarray(self.widths)
        if not len(widths):
            raise ValueError("No measurable personal ink in this policy job")
        logs = np.log(widths / self.target)
        transitions = [
            int(a != b)
            for code, slots in self.variants.items()
            if self.choices[code] > 1
            for a, b in zip(slots, slots[1:], strict=False)
        ]
        entropy = []
        for code, slots in self.variants.items():
            if len(slots) > 1 and self.choices[code] > 1:
                counts = np.asarray(list(Counter(slots).values()), dtype=float) / len(slots)
                entropy.append(
                    float(
                        -sum(counts * np.log(counts))
                        / math.log(min(len(slots), self.choices[code]))
                    )
                )
        return {
            "action": self.action.name,
            "reward_source": self.policy.reward_source,
            "glyph_count": len(widths),
            "target_ink_width_em": self.target,
            "median_ink_width_em": float(np.median(widths)),
            "ink_log_rmse": float(np.sqrt(np.mean(logs**2))),
            "ink_step": float(np.mean(np.abs(np.diff(np.log(widths))))) if len(widths) > 1 else 0,
            "heavy_fraction": float(np.mean(widths > 1.8 * self.target)),
            "variant_entropy": float(np.mean(entropy)) if entropy else 0,
            "variant_switch_rate": float(np.mean(transitions)) if transitions else 0,
            "measurement": "outline ridge diameter/em; no recorded pressure",
        }
