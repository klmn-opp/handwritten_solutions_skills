"""Independent placed-ink checks for the explicitly selected CPU candidate.

These checks inspect the actual vector paths after transformation. Recorded
envelopes and parser identities alone are not geometric safety evidence.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from hwsol.personal_ink.kernels.author_feedback_revision import trace
from hwsol.personal_ink.kernels.f_size_feedback import contour_distance
from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph
from hwsol.personal_ink.kernels.math_latin_feedback import filled_path
from hwsol.personal_ink.kernels.session_o_resolver import PRIORITY
from hwsol.profiles.io import inside, sha256

SEMANTIC_FIELDS = (
    "item_index",
    "context",
    "semantic",
    "latex",
    "matrix_environment",
    "structure",
    "codepoint",
)
TARGETS = PRIORITY | frozenset("*")


@dataclass(frozen=True)
class GeometryState:
    records: tuple[dict, ...]
    paths: tuple[MplPath, ...]
    paint: tuple[tuple, ...]
    parser: dict
    dimensions: tuple[int, int]


def _paths(page, renderer) -> tuple[MplPath, ...]:
    result = []
    for item in page.items:
        path = filled_path(item, renderer)
        if not len(path.vertices) or not np.isfinite(path.vertices).all():
            raise ValueError("Empty or nonfinite recorded ink")
        result.append(
            MplPath(
                path.vertices.copy(),
                path.codes.copy() if path.codes is not None else None,
            )
        )
    return tuple(result)


def capture(page, renderer, records) -> GeometryState:
    """Freeze actual paths and records before any next-stage mutation."""
    if len(page.items) != len(records):
        raise ValueError("Cannot capture inconsistent visible-item inventory")
    return GeometryState(
        tuple(copy.deepcopy(records)),
        _paths(page, renderer),
        tuple((item[0], *item[3:]) for item in page.items),
        trace(renderer),
        (page.width, page.height),
    )


def _same_path(a: MplPath, b: MplPath) -> bool:
    return np.array_equal(a.vertices, b.vertices) and np.array_equal(a.codes, b.codes)


def _nearby(bounds: np.ndarray, box: np.ndarray, margin: float) -> np.ndarray:
    return np.flatnonzero(
        (bounds[:, 2] >= box[0] - margin)
        & (bounds[:, 0] <= box[2] + margin)
        & (bounds[:, 3] >= box[1] - margin)
        & (bounds[:, 1] <= box[3] + margin)
    )


def validate_session_increment(
    state: GeometryState, page, renderer, records, bank_dir: Path
) -> dict:
    """Preserve the sealed Session O geometry, source and clearance criteria.

    Every non-target path and record stays exact. New source contours are recreated
    independently from unchanged RGB crops. A fresh contact or a worsening gap below
    0.60 px rejects the page; an unsupported source remains an explicit old glyph.
    """
    if (page.width, page.height) != state.dimensions:
        raise ValueError("Session glyph replacement changed page dimensions")
    if len(page.items) != len(records) or len(records) != len(state.records):
        raise ValueError("Session glyph replacement changed visible-item count")
    if trace(renderer) != state.parser:
        raise ValueError("Session glyph replacement changed formula structure")
    paths = _paths(page, renderer)
    bounds = np.array([path.get_extents().extents for path in paths])
    catalog_rows = json.loads(inside(bank_dir, "sources.json").read_text("utf-8"))
    catalog = {row["id"]: row for row in catalog_rows}
    if len(catalog) != len(catalog_rows):
        raise ValueError("Session bank has duplicate source identities")
    manifest_sha = sha256(inside(bank_dir, "sources.json"))
    changed, placements, neighbors = [], [], []
    for i, (old, new, a, b) in enumerate(
        zip(state.records, records, state.paths, paths, strict=True)
    ):
        if old["item_index"] != i or new["item_index"] != i:
            raise ValueError("Session glyph provenance drawing order changed")
        if any(old.get(key) != new.get(key) for key in SEMANTIC_FIELDS):
            raise ValueError(f"Session glyph semantic field changed at {i}")
        eligible = (
            old["context"] == "mathematics"
            and old["semantic"] in TARGETS
            and (
                old["semantic"] == "π" or old["source_type"] in {"external-font", "inherited-font"}
            )
        )
        same = _same_path(a, b)
        if not eligible and (old != new or not same):
            raise ValueError(f"Session replacement changed a non-target item {i}")
        if state.paint[i] != (page.items[i][0], *page.items[i][3:]):
            raise ValueError(f"Session replacement changed item paint at {i}")
        if not same:
            changed.append(i)
        source = new.get("session_o_source")
        if source is None:
            if old != new or not same:
                raise ValueError(f"Changed Session glyph has no actual source at {i}")
            continue
        if source.get("id") not in catalog or source != catalog[source["id"]]:
            raise ValueError(f"Actual Session source differs from the reviewed bank at {i}")
        if new.get("session_o_previous") != old:
            raise ValueError(f"Session source lost its previous item record at {i}")
        if new.get("session_o_bank_sha256") != manifest_sha:
            raise ValueError("Session bank digest does not match the actual bank")
        if new.get("font") is not None or new.get("font_sha256") is not None:
            raise ValueError("Personal scan outline must not claim a source font")
        if new.get("external_fallback") or new.get("source_type") not in {
            "personal-session-outline",
            "personal-stretched-outline",
        }:
            raise ValueError("Session glyph source classification is inconsistent")
        crop = inside(bank_dir, source["crop"])
        if sha256(crop) != source["crop_sha256"]:
            raise ValueError("Session original crop changed")
        sample = ScanGlyph.load(bank_dir, source)
        transform = new["transformation"]
        scales = np.asarray(transform["scale_xy"], dtype=float)
        shift = np.asarray(transform["translation_xy"], dtype=float)
        if (
            scales.shape != (2,)
            or shift.shape != (2,)
            or not np.isfinite(scales).all()
            or not np.isfinite(shift).all()
            or min(scales) <= 0
        ):
            raise ValueError("Invalid Session source placement transform")
        anisotropic = bool(abs(scales[0] - scales[1]) > 1e-10)
        if anisotropic and (new["semantic"] not in "⟨⟩" or max(scales) / min(scales) > 4):
            raise ValueError("Unapproved nonuniform Session source transform")
        if transform.get("stroke_redrawing") or transform.get("ink_erosion_or_dilation"):
            raise ValueError("Unapproved Session source ink modification")
        if not np.array_equal(
            sample.outline.verts * scales + shift, b.vertices
        ) or not np.array_equal(sample.outline.codes, b.codes):
            raise ValueError(
                f"Placed Session contour cannot be reconstructed from its source at {i}"
            )
        placements.append(
            {
                "item_index": i,
                "source_id": source["id"],
                "crop_sha256": source["crop_sha256"],
                "placed_contour_exact": True,
            }
        )
    for i in changed:
        path, row, box = paths[i], records[i], bounds[i]
        safe = np.asarray(row.get("allowed_bounds_px"), dtype=float)
        reported = np.asarray(row.get("bounds_px"), dtype=float)
        if (
            safe.shape != (4,)
            or reported.shape != (4,)
            or not np.isfinite(safe).all()
            or not np.isfinite(reported).all()
            or not np.allclose(box, reported, atol=0.03, rtol=0)
        ):
            raise ValueError(f"Session glyph has inaccurate or missing bounds at {i}")
        if min(box[0] - safe[0], box[1] - safe[1], safe[2] - box[2], safe[3] - box[3]) < -0.03:
            raise ValueError(f"Session glyph exceeds its declared envelope at {i}")
        if min(box[0], box[1], page.width - box[2], page.height - box[3]) < 3:
            raise ValueError(f"Session glyph reaches the page boundary at {i}")
        for j in _nearby(bounds, box, 0.60):
            j = int(j)
            if i == j:
                continue
            before = bool(state.paths[i].intersects_path(state.paths[j], filled=True))
            after = bool(path.intersects_path(paths[j], filled=True))
            if after and not before:
                raise ValueError(f"New Session glyph contact between {i} and {j}")
            old_gap = contour_distance(state.paths[i], state.paths[j])
            new_gap = contour_distance(path, paths[j])
            if new_gap < 0.60 and new_gap < old_gap - 0.05:
                raise ValueError(f"New Session glyph subpixel crowding between {i} and {j}")
            neighbors.append(
                {
                    "item": i,
                    "other": j,
                    "before_contact": before,
                    "after_contact": after,
                    "before_gap_px": old_gap,
                    "after_gap_px": new_gap,
                }
            )
    return {
        "changed_items": changed,
        "source_placements": placements,
        "non_target_records_and_paths_exact": True,
        "source_contours_reconstructed_exactly": True,
        "semantic_fields_and_parser_exact": True,
        "new_contact_or_subpixel_crowding": 0,
        "neighbors": neighbors,
    }


def audit_new_contacts(state: GeometryState, page, renderer, records) -> dict:
    """Report whole-chain new contacts; only explicit own-root joins are classified.

    This diagnostic does not reject or blanket-exempt a category. The caller must
    review/reject every unexpected contact before declaring an accepted output.
    """
    if len(page.items) != len(state.paths) or len(records) != len(state.records):
        raise ValueError("Cannot audit contact changes across differing item inventories")
    paths = _paths(page, renderer)
    bounds = np.array([path.get_extents().extents for path in paths])
    changed = {
        i for i, (a, b) in enumerate(zip(state.paths, paths, strict=True)) if not _same_path(a, b)
    }
    new_contacts, seen = [], set()
    for i in sorted(changed):
        for j in _nearby(bounds, bounds[i], 0):
            j = int(j)
            pair = tuple(sorted((i, j)))
            if i == j or pair in seen:
                continue
            seen.add(pair)
            if not paths[i].intersects_path(paths[j], filled=True) or state.paths[
                i
            ].intersects_path(state.paths[j], filled=True):
                continue
            a, b = records[i], records[j]
            root_join = (
                {a.get("structure"), b.get("structure")} == {"radical-sign", "radical-overbar"}
                and a.get("context") == b.get("context") == "mathematics"
                and bool(a.get("latex"))
                and a.get("latex") == b.get("latex")
                and j in a.get("paired_items", ())
                and i in b.get("paired_items", ())
            )
            new_contacts.append(
                {
                    "items": list(pair),
                    "semantics": [records[k]["semantic"] for k in pair],
                    "structures": [records[k].get("structure") for k in pair],
                    "classification": "explicit-own-radical-join" if root_join else "unexpected",
                }
            )
    unexpected = [row for row in new_contacts if row["classification"] == "unexpected"]
    return {
        "scope": "actual whole-chain new path intersections relative to captured raw layout",
        "changed_items": sorted(changed),
        "new_contacts": new_contacts,
        "unexpected_contacts": unexpected,
        "unreviewed_contact_count": len(unexpected),
    }
