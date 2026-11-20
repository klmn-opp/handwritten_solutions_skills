"""Optional Train-derived placement correction, explicitly not neural training."""

import numpy as np

BODY = frozenset("acemnorsuvwxz")


def adjust_boxes(
    tokens: np.ndarray, boxes: np.ndarray, metrics: dict, *, preserve_word_gaps: bool = False
) -> np.ndarray:
    if tokens.shape != (len(boxes), 3) or boxes.shape[1:] != (4,):
        raise ValueError("Semantic tokens and matching physical boxes required")
    result = boxes.copy()
    heights = [
        b[3] - b[1] for t, b in zip(tokens, boxes, strict=True) if t[0] and chr(t[0]) in BODY
    ]
    if not heights:
        return result
    xheight = float(np.median(heights))
    if xheight <= 0:
        raise ValueError("Positive body height required")
    for j, token in enumerate(tokens):
        if token[0] != ord("t"):
            continue
        height = result[j, 3] - result[j, 1]
        if height > metrics["t_upper"] * xheight:
            corrected = (height + metrics["t_median"] * xheight) / 2
            factor = corrected / height
            center = (result[j, 0] + result[j, 2]) / 2
            width = (result[j, 2] - result[j, 0]) * factor
            result[j, [0, 2]] = [center - width / 2, center + width / 2]
            result[j, 1] = result[j, 3] - corrected
    # Close a targeted gap only inside its own word.
    for j in range(1, len(tokens)):
        if not tokens[j, 0] or not tokens[j, 1]:
            continue
        bounds = []
        if tokens[j, 0] == ord("l"):
            bounds.append(metrics["l_left_upper"])
        if tokens[j - 1, 0] == ord("r"):
            bounds.append(metrics["r_right_upper"])
        if not bounds:
            continue
        gap = result[j, 0] - result[j - 1, 2]
        allowed = max(bounds) * xheight
        if gap <= allowed:
            continue
        end = j + 1
        while end < len(tokens) and tokens[end, 0] and tokens[end, 1]:
            end += 1
        result[j:end, 0] -= gap - allowed
        result[j:end, 2] -= gap - allowed
    if preserve_word_gaps:
        valid = np.flatnonzero(tokens[:, 0])
        starts = [valid[0], *[i for i in valid[1:] if not tokens[i, 1]]]
        shift = 0.0
        for start, end in zip(starts, starts[1:] + [valid[-1] + 1], strict=True):
            source = boxes[start:end, :]
            corrected = result[start:end, :]
            left_delta = float(corrected[:, 0].min() - source[:, 0].min())
            right_delta = float(corrected[:, 2].max() - source[:, 2].max())
            corrected[:, [0, 2]] += shift - left_delta
            shift += right_delta - left_delta
    return result
