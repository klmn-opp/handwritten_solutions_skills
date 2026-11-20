"""Group only existing verified local observations, never invent missing cuts."""

import numpy as np


def group_conditions(rows: list[dict], codes: np.ndarray) -> tuple[dict, list[dict]]:
    groups = {}
    for i, q in enumerate(rows):
        text, pos = q["word"], q["position"]
        if q["source_hash"] == "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d":
            raise ValueError("Sealed source cannot enter training")
        if text[pos] != q["label"]:
            raise ValueError("Position identity mismatch")
        key = q["source_crop_sha256"]
        if key not in groups:
            groups[key] = {
                "text": text,
                "source_crop_sha256": key,
                "source_crop": q["source_crop"],
                "source_hash": q["source_hash"],
                "positions": {},
            }
        group = groups[key]
        if group["text"] != text or pos in group["positions"]:
            raise ValueError("Conflicting or duplicate verified source position")
        group["positions"][pos] = i
    meta = [groups[k] for k in sorted(groups)]
    length = max(len(q["text"]) for q in meta)
    tokens = np.zeros((len(meta), length, 3), np.int64)
    values = np.zeros((len(meta), length, 8), np.float32)
    ids = np.full((len(meta), length), -1, np.int64)
    train = np.zeros(len(meta), bool)
    for i, q in enumerate(meta):
        train[i] = int(q["source_crop_sha256"][:8], 16) % 5 != 0
        q["style_prior_split"] = "train" if train[i] else "optimization-validation-upstream-exposed"
        for j, char in enumerate(q["text"]):
            tokens[i, j] = [
                ord(char),
                ord(q["text"][j - 1]) if j else 0,
                ord(q["text"][j + 1]) if j + 1 < len(q["text"]) else 0,
            ]
        for pos, index in q["positions"].items():
            values[i, pos], ids[i, pos] = codes[index], index
    if not train.any() or train.all():
        raise ValueError("Both optimization partitions required")
    return {
        "tokens": tokens,
        "codes": values,
        "glyph_indices": ids,
        "known": ids >= 0,
        "train": train,
    }, meta
