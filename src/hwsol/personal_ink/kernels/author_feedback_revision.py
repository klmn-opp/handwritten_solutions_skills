"""CPU kernel extracted from research.author_feedback_revision; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy

from hwsol.personal_ink.kernels.readability import placed_path


def identity(records):
    return [(r["context"], r["semantic"], r.get("latex")) for r in records]


def f_collision_check(page, renderer, before_items, records):
    """Reject newly touching filled outlines caused by a widened f."""
    result = []
    for row in records:
        if row["semantic"] != "f" or not row.get("cpu_feedback"):
            continue
        i = row["item_index"]
        new = placed_path(page.items[i], renderer)
        old = placed_path(before_items[i], renderer)
        comparisons, new_contacts = ([], [])
        x0, y0, x1, y1 = row["bounds_px"]
        for other in records:
            j = other["item_index"]
            if j == i or page.items[j][0] != "path":
                continue
            a, b, c, d = other["bounds_px"]
            if c < x0 - 1 or a > x1 + 1 or d < y0 - 1 or (b > y1 + 1):
                continue
            path = placed_path(page.items[j], renderer)
            touched = bool(new.intersects_path(path, filled=True))
            prior = bool(old.intersects_path(placed_path(before_items[j], renderer), filled=True))
            comparisons.append(
                {
                    "other_item": j,
                    "semantic": other["semantic"],
                    "before_contact": prior,
                    "after_contact": touched,
                }
            )
            if touched and (not prior):
                new_contacts.append(j)
        result.append(
            {
                "item_index": i,
                "before_bounds": row["cpu_feedback"]["original_bounds_px"],
                "after_bounds": row["bounds_px"],
                "nearby": comparisons,
                "new_contacts": new_contacts,
            }
        )
    return result


def trace(renderer):
    return copy.deepcopy({"formulas": renderer.formulas, "matrices": renderer.matrices})
