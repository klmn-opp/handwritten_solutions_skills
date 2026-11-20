"""Geometry, pixel, and semantic checks for explicitly scoped author corrections."""

import math

import numpy as np

from research.author_feedback_revision import identity
from research.math_latin_feedback import filled_path


def page_geometry(page, renderer):
    return [filled_path(item, renderer) for item in page.items]


def same_path(a, b):
    return np.array_equal(a.vertices, b.vertices) and np.array_equal(a.codes, b.codes)


def validate_increment(before_paths, page, renderer, before, after, old_image, image):
    """Permit identified f/2/pi/rules only; reject new contact or stray pixels.

    Existing incidental contacts are disclosed. The radical's intended own stem
    contact is allowed, but never a new contact to its radicand or another rule.
    Individual treatments must also record and justify their safe envelopes.
    """
    if identity(before) != identity(after) or len(page.items) != len(before_paths):
        raise ValueError("Item semantics, order, or count changed")
    after_paths = page_geometry(page, renderer)
    changed = {
        i
        for i, (a, b) in enumerate(zip(before_paths, after_paths, strict=True))
        if not same_path(a, b)
    }
    old_pixels, new_pixels = np.asarray(old_image), np.asarray(image)
    if old_pixels.shape != new_pixels.shape:
        raise ValueError("Page dimensions changed")
    diff = np.any(old_pixels != new_pixels, axis=2)
    allowed = np.zeros(diff.shape, dtype=bool)
    bounds = np.array([p.get_extents().extents for p in after_paths])
    contacts = []
    root_joins = []
    for i, (old, new) in enumerate(zip(before, after, strict=True)):
        if i != old["item_index"] or i != new["item_index"]:
            raise ValueError("Provenance index does not match drawing order")
        if i not in changed:
            continue
        target = old["semantic"] in {"2", "π", "f"} or old.get("structure") in {
            "fraction-bar",
            "radical-overbar",
        }
        if not target:
            raise ValueError(f"Unauthorized changed item {i}: {old['semantic']}")
        safe = new.get("allowed_bounds_px")
        if safe is None:
            raise ValueError(f"Changed item {i} has no recorded safe envelope")
        x0, y0, x1, y1 = new["bounds_px"]
        if min(x0 - safe[0], y0 - safe[1], safe[2] - x1, safe[3] - y1) < -0.03:
            raise ValueError(f"Changed item {i} exceeds recorded safe envelope")
        if min(x0, y0, image.width - x1, image.height - y1) < 3:
            raise ValueError(f"Changed item {i} touches page boundary")
        if not np.allclose(bounds[i], new["bounds_px"], atol=0.03, rtol=0):
            raise ValueError(f"Changed item {i} reports inaccurate ink bounds")
        if old.get("structure") == "radical-overbar":
            signs = [
                j
                for j in old.get("paired_items", ())
                if before[j].get("structure") == "radical-sign"
            ]
            if len(signs) != 1:
                raise ValueError(f"Changed roof {i} has no unique original paired stem")
            j = signs[0]
            was = bool(before_paths[i].intersects_path(before_paths[j], filled=True))
            now = bool(after_paths[i].intersects_path(after_paths[j], filled=True))
            if was and not now:
                raise ValueError(f"Changed roof {i} lost its original stem join {j}")
            root_joins.append({"roof": i, "stem": j, "before": was, "after": now})
        for box in (old["bounds_px"], new["bounds_px"]):
            a, b, c, d = box
            allowed[
                max(0, math.floor(b) - 2) : min(image.height, math.ceil(d) + 2),
                max(0, math.floor(a) - 2) : min(image.width, math.ceil(c) + 2),
            ] = True
        nearby = np.flatnonzero(
            (bounds[:, 2] >= x0 - 0.5)
            & (bounds[:, 0] <= x1 + 0.5)
            & (bounds[:, 3] >= y0 - 0.5)
            & (bounds[:, 1] <= y1 + 0.5)
        )
        for j in nearby:
            if j == i:
                continue
            was = bool(before_paths[i].intersects_path(before_paths[j], filled=True))
            now = bool(after_paths[i].intersects_path(after_paths[j], filled=True))
            joint = (
                bool(new.get("structural_rule_feedback"))
                and j in new.get("paired_items", ())
                and after[j].get("structure") == "radical-sign"
            )
            if now and not was and not joint:
                raise ValueError(f"New contact between changed item {i} and {j}")
            if now or was:
                contacts.append(
                    {
                        "item": i,
                        "other": int(j),
                        "before": was,
                        "after": now,
                        "own_root_join": joint,
                    }
                )
    outside = int(np.count_nonzero(diff & ~allowed))
    if outside:
        raise ValueError(f"{outside} changed pixels outside exact target boxes")
    return {
        "identity_order_count_preserved": True,
        "non_target_placed_geometry_exact": True,
        "changed_items": sorted(changed),
        "changed_pixels": int(diff.sum()),
        "pixels_changed_outside_target_boxes": outside,
        "contacts": contacts,
        "root_joins": root_joins,
        "lost_original_root_joins": 0,
        "new_unrelated_contacts": 0,
    }
