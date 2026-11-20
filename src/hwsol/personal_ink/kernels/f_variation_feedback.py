"""CPU kernel extracted from research.f_variation_feedback; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from hwsol.personal_ink.kernels.author_feedback_round2_checks import page_geometry
from hwsol.personal_ink.kernels.author_feedback_round3_delivery import f_words
from hwsol.personal_ink.kernels.f_size_feedback import (
    consecutive_f_groups,
    contour_distance,
    stem_measure,
)
from hwsol.profiles.io import sha256

POLICY = {
    "minimum_new_clearance_px": 0.6,
    "maximum_side_extension_em": 0.16,
    "maximum_top_extension_em": 0.18,
    "maximum_bottom_extension_em": 0.1,
    "minimum_page_clearance_px": 3.0,
    "minimum_actual_em_fraction": 0.8,
    "maximum_height_over_approved_D": 1.2,
    "original_30_held_instances": (
        "retain exact approved baseline; no attempt to bypass previous safety rejection"
    ),
    "same_word_ff": "one source family per group; any failure retains whole group baseline",
}


def read(path):
    return json.loads(Path(path).read_text())


class FVariationFeedback:
    """Only reviewed real sources; the approved f0 and held groups are fallback."""

    def __init__(self, source_dir, *, source, title, seed=1909, candidate="balanced"):
        self.source_dir = Path(source_dir)
        self.source, self.title, self.seed = (source, title, seed)
        self.candidate = candidate
        self.candidates = {r["id"]: r for r in read(self.source_dir / "candidates.json")}
        self.outlines = {}
        for key, row in self.candidates.items():
            path = self.source_dir / row["outline"]
            if sha256(path) != row["outline_sha256"]:
                raise ValueError("Reviewed candidate outline changed")
            arrays = np.load(path)
            self.outlines[key] = MplPath(arrays["vertices"], arrays["codes"])
        if (
            sha256(self.source_dir / "source-sample.ttf")
            != self.candidates["sheet-f1"]["source"]["font_source_sha256"]
        ):
            raise ValueError("Real source font changed")
        self.decisions, self.groups = ([], [])

    def apply(self, page, renderer, records):
        self.groups = consecutive_f_groups(self.source, self.title, records)
        group_for = {i: g for g in self.groups for i in g["item_indices"]}
        words = f_words(self.source, self.title, records)
        word_for = {i: w for w in words for i in w["items"]}
        result = copy.deepcopy(records)
        items = list(page.items)
        paths = page_geometry(page, renderer)
        proposals = {}
        self.decisions = []
        for row in records:
            if row["semantic"] != "f":
                continue
            i = row["item_index"]
            group = group_for.get(i)
            word = word_for.get(i)
            choice_key = (
                f"word/{word['span']}/{word['range']}/{word['word']}"
                if word
                else f"math/{row.get('latex')}/{row.get('matrix_environment')}"
            )
            digest = hashlib.sha256(f"{self.seed}/{self.title}/{choice_key}".encode()).hexdigest()
            selected = self.candidate
            if selected == "balanced":
                selected = ("baseline-f0", "sheet-f1")[int(digest[:16], 16) % 2]
            decision = {
                "item_index": i,
                "context": row["context"],
                "selected_source": selected,
                "selection_sha256": digest,
                "selection_context": choice_key,
                "group_id": group["group_id"] if group else None,
                "applied": False,
                "rejections": [],
                "safety_policy": POLICY,
            }
            self.decisions.append(decision)
            previous = row.get("f_size_feedback", {})
            if previous.get("status") != "research-candidate":
                decision["rejections"].append("Previously held f is exact baseline fallback")
                continue
            if selected == "baseline-f0":
                decision["retained_by_fixed_source_selection"] = True
                continue
            if selected not in self.candidates:
                raise ValueError("Unknown source candidate")
            old = paths[i]
            box = old.get_extents()
            prebox = previous["original"]["bounds_px"]
            em = (prebox[3] - prebox[1]) * 1000 / 716
            if em < 0.8 * renderer.cfg.body_size:
                decision["rejections"].append("Small-script variant requires separate protocol")
                continue
            source_path = self.outlines[selected]
            sb = source_path.get_extents()
            center = (box.xmin + box.xmax) / 2
            baseline_y = prebox[1] + 715 / 1000 * em
            v = source_path.vertices * (em / 1000, -em / 1000)
            v += (center - (sb.xmin + sb.xmax) / 2 * em / 1000, baseline_y)
            proposed = MplPath(v, source_path.codes)
            bb = proposed.get_extents()
            extension = (
                np.maximum(
                    [
                        box.xmin - bb.xmin,
                        box.ymin - bb.ymin,
                        bb.xmax - box.xmax,
                        bb.ymax - box.ymax,
                    ],
                    0,
                )
                / em
            )
            decision.update(
                actual_em_px=em,
                original_bounds=box.extents.tolist(),
                candidate_bounds=bb.extents.tolist(),
                extension_em=extension.tolist(),
                original_lower_stem=stem_measure(old),
                candidate_lower_stem=stem_measure(proposed),
            )
            if (
                max(extension[0], extension[2]) > 0.16
                or extension[1] > 0.18
                or extension[3] > 0.1
                or (bb.height / box.height > 1.2)
            ):
                decision["rejections"].append("Frozen source-size displacement budget exceeded")
            if min(bb.xmin, bb.ymin, page.width - bb.xmax, page.height - bb.ymax) < 3:
                decision["rejections"].append("Page boundary")
            neighbors = []
            for j, other in enumerate(paths):
                ob = other.get_extents()
                dx = max(0, ob.xmin - bb.xmax, bb.xmin - ob.xmax)
                dy = max(0, ob.ymin - bb.ymax, bb.ymin - ob.ymax)
                if i == j or np.hypot(dx, dy) > 2 * em:
                    continue
                before, after = (contour_distance(old, other), contour_distance(proposed, other))
                neighbors.append(
                    {
                        "item_index": j,
                        "semantic": records[j]["semantic"],
                        "before_px": before,
                        "after_px": after,
                    }
                )
                if after < 0.6 and after < before - 0.05:
                    decision["rejections"].append(f"Neighbor {j} clearance {after:.3f}px")
            decision["neighbors"] = neighbors
            if not decision["rejections"]:
                proposals[i] = proposed
        by_index = {r["item_index"]: r for r in self.decisions}
        for i, a in list(proposals.items()):
            for j, b in list(proposals.items()):
                aa, bb = (a.get_extents(), b.get_extents())
                dx = max(0, aa.xmin - bb.xmax, bb.xmin - aa.xmax)
                dy = max(0, aa.ymin - bb.ymax, bb.ymin - aa.ymax)
                if i >= j or np.hypot(dx, dy) >= 0.6:
                    continue
                distance = contour_distance(a, b)
                if distance < 0.6 and distance < contour_distance(paths[i], paths[j]) - 0.05:
                    for k in (i, j):
                        by_index[k]["rejections"].append("Simultaneous candidate f clearance")
        for group in self.groups:
            ids = group["item_indices"]
            failed = [i for i in ids if by_index[i]["rejections"]]
            if failed:
                for i in ids:
                    if not by_index[i]["rejections"]:
                        by_index[i]["rejections"].append("Atomic same-word ff group fallback")
                    by_index[i]["group_rejection"] = {
                        "original_failure_items": failed,
                        "word": group["word"],
                    }
                group["action"] = "retain-entire-baseline-group"
            else:
                group["action"] = "apply-one-source-family-or-retain-f0-selection"
        for i, path in proposals.items():
            decision = by_index[i]
            if decision["rejections"]:
                continue
            source_id = decision["selected_source"]
            spec = self.candidates[source_id]
            key = ("f-real-variation", source_id, i)
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), items[i][3], 0)
            bb = path.get_extents().extents.tolist()
            result[i].update(
                bounds_px=bb,
                allowed_bounds_px=bb,
                source_type="personal-outline-geometric-calibration",
                variant=1 if source_id == "sheet-f1" else None,
                actual_glyph_id=None,
                glyph_name=spec["source"].get("source_glyph", "historical-word-final-f"),
                font=str(self.source_dir / "source-sample.ttf")
                if source_id == "sheet-f1"
                else None,
                font_sha256=spec["source"].get("font_source_sha256"),
                f_variation_feedback={
                    "status": "isolated-source-candidate",
                    "source_id": source_id,
                    "source": spec["source"],
                    "geometry": spec["geometry"],
                    "outline_sha256": spec["outline_sha256"],
                    "baseline": copy.deepcopy(records[i]),
                    "output_envelope_scope": (
                        "target geometry only; measured clearance and fixed budg"
                        "ets independently gate"
                    ),
                    "neural_updates": 0,
                    "synthetic_noise": False,
                    "ink_dilation": False,
                },
            )
            decision["applied"] = True
        for decision in self.decisions:
            decision["actual_source"] = (
                decision["selected_source"] if decision["applied"] else "baseline-f0"
            )
        return result
