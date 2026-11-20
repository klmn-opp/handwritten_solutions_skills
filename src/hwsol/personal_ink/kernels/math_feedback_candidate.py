"""CPU kernel extracted from research.math_feedback_candidate; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

import copy

import numpy as np
from handnote.vector import Outline

from hwsol.personal_ink.kernels.math_session_glyphs import outline_bounds
from hwsol.personal_ink.kernels.math_session_polish import polygons, sections
from hwsol.personal_ink.kernels.readability import placed_path


class RadicalClearancePolish:
    """Compact only excessive visible clearance, after the unchanged v26 polish.

    All radicand items translate together. Their contours, relative positions,
    semantic IDs and order are unchanged. The stem translates and the matched
    roof shortens; the whole result remains inside the old root envelope.
    """

    def __init__(self, clearance_em: float = 0.12):
        self.clearance_em = clearance_em
        self.decisions: list[dict] = []

    def apply(self, page, renderer, baseline: list[dict]) -> list[dict]:
        records = copy.deepcopy(baseline)
        self.decisions = []
        signs = sorted(
            (r for r in records if r["structure"] == "radical-sign"),
            key=lambda r: -(r["bounds_px"][3] - r["bounds_px"][1]),
        )
        for sign in signs:
            si = sign["item_index"]
            decision = {"sign_item": si, "latex": sign["latex"], "applied": False}
            self.decisions.append(decision)
            pair = sign.get("paired_items", [])
            if len(pair) != 2:
                decision["reason"] = "existing single-digit compact root or no verified pair"
                continue
            bi = next((i for i in pair if records[i]["structure"] == "radical-overbar"), None)
            if bi is None:
                decision["reason"] = "no verified overbar"
                continue
            bar = records[bi]
            bx, by, br, bb = bar["bounds_px"]
            sx, sy, sr, sb = sign["bounds_px"]
            local = [
                r
                for r in records
                if r["context"] == "mathematics"
                and r["item_index"] not in (si, bi)
                and (r["latex"] == sign["latex"])
                and (len(r["bounds_px"]) == 4)
                and (bx < (r["bounds_px"][0] + r["bounds_px"][2]) / 2 < br)
                and (r["bounds_px"][1] > bb - 1)
                and (r["bounds_px"][3] < sb + 2.5)
            ]
            if not local or any(page.items[r["item_index"]][0] != "path" for r in local):
                decision["reason"] = "unresolved radicand membership or non-path ink"
                continue
            ids = [r["item_index"] for r in local]
            stem = placed_path(page.items[si], renderer)
            ys = np.arange(
                min(r["bounds_px"][1] for r in local), max(r["bounds_px"][3] for r in local), 0.25
            )
            stem_rows = sections(polygons(stem), ys)
            contents = [sections(polygons(placed_path(page.items[i], renderer)), ys) for i in ids]
            gaps = [
                min(row[j][:, 0].min() for row in contents if len(row[j])) - edge[:, 1].max()
                for j, edge in enumerate(stem_rows)
                if len(edge) and any(len(row[j]) for row in contents)
            ]
            if not gaps:
                decision["reason"] = "no shared vertical ink samples"
                continue
            gap = float(min(gaps))
            target = self.clearance_em * renderer.cfg.body_size
            reduction = min(gap - target, br - bx - 2)
            decision.update(
                gap_before_px=gap,
                target_gap_px=target,
                radicand_items=ids,
                bar_item=bi,
                method="minimum shared-row visible ink clearance at 0.25px spacing",
            )
            if reduction < 1:
                decision["reason"] = "visible clearance already within target"
                continue
            content_shift = round(reduction / 2)
            stem_shift = reduction - content_shift
            allowed = [min(sx, bx), min(sy, by), max(sr, br), max(sb, bb)]
            old_items = {i: page.items[i] for i in ids}
            history = {
                "method": "compact multi-item radical around its old centre",
                "sign_item": si,
                "bar_item": bi,
                "radicand_items": ids,
                "gap_before_px": gap,
                "gap_after_px": gap - reduction,
                "stem_shift_px": stem_shift,
                "radicand_shift_px": -content_shift,
                "all_radicand_geometry_preserved": True,
            }

            def remember(row, dx, *, history=history, allowed=allowed):
                before = row["bounds_px"]
                row.setdefault("feedback_original_bounds_px", before)
                row.setdefault("feedback_history", []).append({**history, "item_shift_px": dx})
                row["allowed_bounds_px"] = allowed
                row["bounds_px"] = [before[0] + dx, before[1], before[2] + dx, before[3]]

            for r in local:
                i = r["item_index"]
                kind, key, mat, alpha, pen = page.items[i]
                page.items[i] = (kind, key, (*mat[:4], mat[4] - content_shift, mat[5]), alpha, pen)
                remember(r, -content_shift)
            for row, path, vertices in (
                (sign, stem, stem.vertices + (stem_shift, 0)),
                (bar, placed_path(page.items[bi], renderer), None),
            ):
                if vertices is None:
                    vertices = path.vertices.copy()
                    vertices[:, 0] = (
                        bx + stem_shift + (vertices[:, 0] - bx) * (br - bx - reduction) / (br - bx)
                    )
                key = ("feedback-radical-clearance", row["item_index"])
                renderer._outlines[key] = Outline(vertices, path.codes, stroke=False, digits=4)
                i = row["item_index"]
                page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), page.items[i][3], 0)
                remember(row, 0)
                row["bounds_px"] = outline_bounds(vertices, path.codes).ravel().tolist()
            for i in ids:
                before = placed_path(old_items[i], renderer)
                after = placed_path(page.items[i], renderer)
                if not np.array_equal(before.codes, after.codes) or not np.allclose(
                    after.vertices, before.vertices + (-content_shift, 0), atol=1e-08
                ):
                    raise ValueError("Radicand shape changed during root compaction")
            decision.update(
                applied=True,
                gap_after_px=gap - reduction,
                stem_shift_px=stem_shift,
                radicand_shift_px=-content_shift,
            )
        return records
