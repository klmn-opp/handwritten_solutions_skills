"""Deterministic integer word spacing; copy final pixels without changing glyphs.

Public interface: plan_spacing(page, owners, words), then
apply_spacing(page, owners, plan) -> (uint8_page, receipt).
An owner is {id, box: [x, y, X, Y], mask: local bool array}. All non-owner
ink is protected. Additional ink treatments must supply their actual masks.
The returned receipt contains the exact accepted translation rows.
"""

from __future__ import annotations

import json

import numpy as np

FRACTION = 0.10


def region(array, box):
    x, y, X, Y = box
    return array[y:Y, x:X]


def shifted(box, dx):
    x, y, X, Y = box
    return [x + dx, y, X + dx, Y]


def _owners(page, owners):
    page = np.asarray(page)
    if page.ndim != 2 or page.dtype != np.uint8:
        raise ValueError("The final page must be native uint8 grayscale")
    index = np.zeros(page.shape, np.int32)
    byid = {}
    for n, row in enumerate(owners, 1):
        key = row["id"]
        if key in byid:
            raise ValueError("Duplicate owner id")
        box = row["box"]
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Ownership requires integer half-open boxes")
        x, y, X, Y = box
        if not (0 <= x < X <= page.shape[1] and 0 <= y < Y <= page.shape[0]):
            raise ValueError("Ownership box outside page")
        mask = np.asarray(row["mask"])
        if mask.dtype != bool or mask.shape != (Y - y, X - x):
            raise ValueError("Ownership mask shape or dtype mismatch")
        v = region(index, box)
        if np.any(v[mask]):
            raise ValueError("Ambiguous overlapping ownership")
        v[mask] = n
        byid[key] = row
    return page, index, byid


def _ink_box(page, owner):
    yy, xx = np.where((region(page, owner["box"]) <= 191) & owner["mask"])
    if not len(xx):
        return None
    x, y = owner["box"][:2]
    return [int(x + xx.min()), int(y + yy.min()), int(x + xx.max() + 1), int(y + yy.max() + 1)]


def plan_spacing(page, owners, words):
    """Freeze 10% cumulative positive-gap contraction, within original words.

    At least one visible pixel remains at positive projected boundaries. Pure
    letter segments centre the reclaimed space; a segment beside punctuation
    anchors to that punctuation. Segments enclosed by punctuation are held.
    Planning needs actual final ink; previously accepted plans can also be reused
    with new ink and explicit new ownership, subject to apply_spacing's guards.
    """
    page, _, byid = _owners(page, owners)
    result = []
    seen = set()
    groups = []
    for word in words:
        ids = word["glyphs"]
        word_key = word["index"]
        runs = []
        current = []
        for pos, key in enumerate(ids):
            if key in byid:
                if key in seen:
                    raise ValueError("An owner belongs to more than one word")
                seen.add(key)
                current.append((pos, key))
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        for segment in runs:
            keys = [key for _, key in segment]
            boxes = [_ink_box(page, byid[key]) for key in keys]
            left_static = segment[0][0] > 0
            right_static = segment[-1][0] < len(ids) - 1
            hold = "punctuation on both sides" if left_static and right_static else None
            if any(v is None for v in boxes):
                hold = "no visible foreground"
            gaps = [] if hold else [b[0] - a[2] for a, b in zip(boxes, boxes[1:], strict=False)]
            reductions = []
            debt = 0.0
            used = 0
            for gap in gaps:
                debt += max(0, gap) * FRACTION
                take = max(0, min(max(0, gap - 1), int(np.floor(debt + 0.5)) - used))
                reductions.append(take)
                used += take
            total = sum(reductions)
            anchor = total if right_static else 0 if left_static else (total + 1) // 2
            running = 0
            for n, key in enumerate(keys):
                if n and not hold:
                    running += reductions[n - 1]
                dx = 0 if hold else anchor - running
                result.append(
                    {
                        "id": key,
                        "word_index": word_key,
                        "word": word["word"],
                        "dx": dx,
                        "dy": 0,
                        "box": list(byid[key]["box"]),
                        "word_box": list(word["box"]),
                        "planning_hold": hold,
                    }
                )
            groups.append(
                {
                    "word_index": word_key,
                    "word": word["word"],
                    "ids": keys,
                    "visible_gaps_before": gaps,
                    "requested_reductions": reductions,
                    "anchor": "right punctuation"
                    if right_static
                    else "left punctuation"
                    if left_static
                    else "word centre",
                    "planning_hold": hold,
                }
            )
    if set(byid) != seen:
        raise ValueError("Every owner must belong to one word")
    return {
        "schema": "word-gap-integer-plan/1",
        "fraction": FRACTION,
        "scope": "integer x only; no training, shape, y, tone, line wrapping changes",
        "rows": result,
        "segments": groups,
    }


def apply_spacing(page, owners, plan):
    """Move final page samples losslessly; hold whole words on any collision.

    The input may have new per-letter ink, but ownership must cover that ink and
    exclude all other letters/formulas/punctuation. Plans never grant ownership.
    Unknown ink, stationary owners and occupied destinations block a move.
    """
    page, source_index, byid = _owners(page, owners)
    rows = {r["id"]: dict(r) for r in plan["rows"]}
    if set(rows) != set(byid) or len(rows) != len(plan["rows"]):
        raise ValueError("Plan and ownership ids differ")
    for key, r in rows.items():
        if type(r["dx"]) is not int or r.get("dy", 0) != 0:
            raise ValueError("Only integer horizontal translations are permitted")
        if list(r["box"]) != list(byid[key]["box"]):
            raise ValueError("Plan refers to a different original box")
    protected = (page < 255) & (source_index == 0)
    held = {}
    while True:
        rejected = {}
        target_index = np.zeros(page.shape, np.int32)
        numbered = list(rows)
        for n, key in enumerate(numbered, 1):
            r, owner = rows[key], byid[key]
            dx = 0 if r["word_index"] in held else r["dx"]
            target = shifted(owner["box"], dx)
            x, y, X, Y = target
            group = r["word_index"]
            if not (0 <= x < X <= page.shape[1] and 0 <= y < Y <= page.shape[0]):
                rejected[group] = "page boundary"
                continue
            if dx:
                # Original boxes sometimes include white padding outside the word
                # frame; constrain actual owned pixels rather than that padding.
                yy, xx = np.where(owner["mask"])
                if len(xx):
                    wx, wy, wX, wY = r["word_box"]
                    if x + xx.min() < wx or x + xx.max() >= wX:
                        rejected[group] = "word boundary"
                if np.any(region(protected, target)[owner["mask"]]):
                    rejected[group] = "protected non-owner ink"
            other = region(target_index, target)[owner["mask"]]
            for value in np.unique(other[other > 0]):
                other_row = rows[numbered[int(value) - 1]]
                if dx:
                    rejected[group] = "another owner destination"
                if other_row["dx"] and other_row["word_index"] not in held:
                    rejected[other_row["word_index"]] = "another owner destination"
            region(target_index, target)[owner["mask"]] = n
        new = {k: v for k, v in rejected.items() if k not in held}
        if not new:
            break
        held.update(new)
    output = page.copy()
    for key, row in rows.items():
        dx = 0 if row["word_index"] in held else row["dx"]
        row["requested_dx"] = row["dx"]
        row["dx"] = dx
        row["hold_reason"] = held.get(row["word_index"])
        row["new_box"] = shifted(byid[key]["box"], dx)
        if dx:
            region(output, byid[key]["box"])[byid[key]["mask"]] = 255
    for key, row in rows.items():
        if row["dx"]:
            owner = byid[key]
            region(output, row["new_box"])[owner["mask"]] = region(page, owner["box"])[
                owner["mask"]
            ]
    for key, row in rows.items():
        owner = byid[key]
        if not np.array_equal(
            region(output, row["new_box"])[owner["mask"]], region(page, owner["box"])[owner["mask"]]
        ):
            raise AssertionError("A glyph's original gray samples changed")
    if not np.array_equal(output[protected], page[protected]):
        raise AssertionError("Protected ink changed")
    if not np.array_equal(
        np.bincount(output.ravel(), minlength=256), np.bincount(page.ravel(), minlength=256)
    ):
        raise AssertionError("Full-page grayscale histogram changed")
    # Stronger than total histogram: no pixel value changes its vertical row.
    for y in np.where(np.any(page != output, axis=1))[0]:
        if not np.array_equal(
            np.bincount(page[y], minlength=256), np.bincount(output[y], minlength=256)
        ):
            raise AssertionError("A pixel moved vertically or changed intensity")
    return output, {
        "schema": "word-gap-integer-applied/1",
        "rows": list(rows.values()),
        "held_words": [{"word_index": k, "reason": v} for k, v in held.items()],
        "moved_letters": sum(r["dx"] != 0 for r in rows.values()),
        "changed_pixels": int(np.sum(page != output)),
        "protected_ink_exact": True,
        "per_glyph_samples_exact": True,
        "per_scanline_grayscale_histogram_exact": True,
        "full_grayscale_histogram_exact": True,
        "dy_all_zero": True,
    }


def self_check():
    return {"schema": "word-gap-integer/1", "fraction": FRACTION}


if __name__ == "__main__":
    import json

    print(json.dumps(self_check()))
