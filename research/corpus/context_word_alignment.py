"""Conservative natural-space word alignment; original crops remain untouched."""

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

from research.corpus.context_training_data import reference_image, whole_ink


def bounds(ink: np.ndarray) -> tuple[int, int, int, int]:
    y, x = np.where(ink > 0.3)
    if not len(x):
        raise ValueError("empty word ink")
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def natural_words(ink: np.ndarray, tokens: list[str]) -> list[tuple[int, int, int, int]]:
    """Only choose clearly separated empty-column gaps, never cut connecting strokes."""
    x0, y0, x1, y1 = bounds(ink)
    if len(tokens) == 1:
        return [(x0, y0, x1, y1)]
    occupied = (ink > 0.3).any(0)
    gaps = []
    start = None
    for x in range(x0, x1):
        if not occupied[x] and start is None:
            start = x
        if occupied[x] and start is not None:
            gaps.append((x - start, start, x))
            start = None
    gaps.sort(reverse=True)
    count = len(tokens) - 1
    if len(gaps) < count:
        raise ValueError("no complete natural word gaps")
    chosen = gaps[:count]
    if min(g[0] for g in chosen) < max(2, (y1 - y0) * 0.12):
        raise ValueError("word gaps too narrow")
    if len(gaps) > count and chosen[-1][0] < 1.25 * gaps[count][0]:
        raise ValueError("ambiguous word versus letter gaps")
    edges = [x0] + [round((a + b) / 2) for _, a, b in sorted(chosen, key=lambda g: g[1])] + [x1]
    result = []
    for a, b in zip(edges, edges[1:], strict=False):
        left, top, right, bottom = bounds(ink[:, a:b])
        result.append((a + left, top, a + right, bottom))
    ratios = np.array(
        [
            (b[2] - b[0]) / max(1, len(re.sub(r"\W", "", word)))
            for b, word in zip(result, tokens, strict=True)
        ]
    )
    if ratios.max() / max(0.1, ratios.min()) > 3.0:
        raise ValueError("word width/label count inconsistent")
    return result


def aligned_view(
    text: str,
    raw: Image.Image,
    font_path: Path,
    size: tuple[int, int] = (512, 64),
    *,
    reviewed_word_boxes: list[list[int]] | None = None,
    reference_raw: Image.Image | None = None,
    reference_word_boundaries: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    tokens = text.split()
    if not tokens or len(tokens) > 6:
        raise ValueError("limited word-group pilot supports 1–6 words")
    real, transform = whole_ink(raw, size)
    reference, ref_transform = whole_ink(
        reference_image(text, font_path, size[1]) if reference_raw is None else reference_raw, size
    )
    source_boxes = natural_words(real, tokens) if reviewed_word_boxes is None else []
    if reviewed_word_boxes is not None:
        if len(reviewed_word_boxes) != len(tokens):
            raise ValueError("Reviewed word count does not match actual transcription")
        real = np.clip((190 - np.asarray(raw.convert("L"), dtype=np.float32)) / 140, 0, 1)
        covered = np.zeros_like(real, dtype=bool)
        last_right = 0
        for box in reviewed_word_boxes:
            x0, y0, x1, y1 = box
            if not all(type(v) is int for v in box) or not (
                last_right <= x0 < x1 <= raw.width and 0 <= y0 < y1 <= raw.height
            ):
                raise ValueError("Reviewed word boxes must be disjoint, ordered and inside crop")
            covered[y0:y1, x0:x1] = True
            a, b, c, d = bounds(real[y0:y1, x0:x1])
            source_boxes.append((x0 + a, y0 + b, x0 + c, y0 + d))
            last_right = x1
        if ((real > 0.3) & ~covered).sum() / max(1, (real > 0.3).sum()) > 0.005:
            raise ValueError("Reviewed word boxes omit visible ink")
    font = ImageFont.truetype(str(font_path), size[1])
    left = font.getbbox(text)[0]
    prefix = ""
    cuts = []
    for word in tokens[:-1]:
        prefix += word
        # Space midpoint in the exact Pillow reference coordinates.
        pixel = 16 - left + font.getlength(prefix) + font.getlength(" ") / 2
        cuts.append(
            round(
                (pixel - ref_transform["ink_bbox"][0]) * ref_transform["scale"]
                + ref_transform["translation"][0]
            )
        )
        prefix += " "
    if reference_raw is not None:
        if reference_word_boundaries is None or len(reference_word_boundaries) != len(tokens) - 1:
            raise ValueError("Handnote semantic word bounds required")
        cuts = [
            round(
                (x - ref_transform["ink_bbox"][0]) * ref_transform["scale"]
                + ref_transform["translation"][0]
            )
            for x in reference_word_boundaries
        ]
    edges = [0] + cuts + [size[0]]
    slots = []
    for a, b in zip(edges, edges[1:], strict=False):
        left, t, r, d = bounds(reference[:, a:b])
        slots.append((a + left, t, a + r, d))
    target = np.zeros_like(reference)
    records = []
    for source, slot in zip(source_boxes, slots, strict=True):
        a, b, c, d = source
        x, y, r, t = slot
        scale = min((r - x) / (c - a), (t - y) / (d - b))
        w = max(1, round((c - a) * scale))
        h = max(1, round((d - b) * scale))
        dx = x + (r - x - w) // 2
        dy = y + (t - y - h) // 2
        patch = Image.fromarray(np.uint8(real[b:d, a:c] * 255)).resize(
            (w, h), Image.Resampling.LANCZOS
        )
        target[dy : dy + h, dx : dx + w] = np.asarray(patch) / 255
        records.append(
            {
                "source_box_in_whole_view": source,
                "reference_slot": slot,
                "scale": scale,
                "destination": [dx, dy, w, h],
            }
        )
    return (
        reference,
        target,
        {
            "recipe": "natural-space-word-uniform-alignment/1",
            "word_boundary_source": (
                "authorized-model-reviewed-original-crop"
                if reviewed_word_boxes is not None
                else "natural-empty-columns"
            ),
            "source_box_coordinate_system": (
                "original-crop" if reviewed_word_boxes is not None else "whole-view"
            ),
            "original_whole_transform": transform,
            "reference_transform": ref_transform,
            "words": records,
            "character_segmentation": False,
            "ink_repair": False,
            "derived_view_not_original_sample": True,
            "word_spacing_canonicalized_not_learned": True,
            "reference_backend": "handnote" if reference_raw is not None else "pillow",
        },
    )
