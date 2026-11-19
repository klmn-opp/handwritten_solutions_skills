"""Text/layout conditions with no outline or scan pixels in the inference contract."""

import numpy as np


def conditions(text: str, provenance: dict, transform: dict) -> tuple[np.ndarray, np.ndarray]:
    glyphs = provenance["glyph_provenance"]
    chars = "".join(text.split())
    if not chars or "".join(g["semantic"] for g in glyphs) != chars:
        raise ValueError("Exact semantic glyph trace required")
    if any(not 33 <= ord(c) <= 126 for c in chars):
        raise ValueError("This English pilot supports printable ASCII only")
    if provenance.get("raster_scale", 1) != 1:
        raise ValueError("Pilot geometry requires the original 300 DPI coordinates")
    scale = transform["scale"]
    crop = np.array(provenance["crop_from_pdf"][:2])
    offset = np.array(transform["translation"]) - np.array(transform["bbox"][:2]) * scale
    boxes = np.array([g["bounds_px"] for g in glyphs], dtype=np.float32)
    boxes = (boxes - np.tile(crop, 2)) * scale + np.tile(offset, 2)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Finite nonempty glyph bounds required")
    # Whitespace breaks context rather than joining two different words.
    tokens = []
    for word in text.split():
        for j, char in enumerate(word):
            tokens.append(
                [
                    ord(char),
                    ord(word[j - 1]) if j else 0,
                    ord(word[j + 1]) if j + 1 < len(word) else 0,
                ]
            )
    return np.array(tokens, dtype=np.int64), boxes.astype(np.float32)


def pad_conditions(items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    length = max(2, max(len(t) for t, _ in items))
    tokens = np.zeros((len(items), length, 3), np.int64)
    boxes = np.zeros((len(items), length, 4), np.float32)
    for i, (t, b) in enumerate(items):
        tokens[i, : len(t)] = t
        boxes[i, : len(t)] = b
    return tokens, boxes
