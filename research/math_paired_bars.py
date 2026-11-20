"""Held double-bar construction from two instances of a reviewed Session N stroke.

This is structural composition of a real single bar, not a newly collected norm
symbol. Original RGB, labels, source font, old pages and production stay unchanged.
"""

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath
from PIL import Image

from hwsol.config import RenderConfig
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_revision import identity, personal_layout, trace
from research.corpus.math_glyph_completion import SELECTION, raster
from research.corpus.math_glyph_resolver import MathGlyphResolver, placed_outline
from research.corpus.math_session_glyphs import SessionGlyphCalibration, outline_bounds
from research.corpus.math_session_polish import SessionInkPolish, cap_stem
from research.corpus.readability import placed_path
from research.homework_benchmark.render import (
    accepted_scans,
    fallback_summary,
    pixel_boundary_checks,
    read,
    save_pdf,
    source_paths,
)
from research.math_feedback_candidate import paired_panel

BASE = "20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1"
SAMPLE = "N-034-U007C-p1"
CASE = "m1-la-02-orthogonal-projection"


def source_bar(directory):
    selected = [g for g in accepted_scans(directory) if g.source["id"] == SAMPLE]
    if len(selected) != 1 or selected[0].source["character"] != "|":
        raise ValueError("Expected the fixed reviewed Session N single-bar source")
    return selected[0]


class PersonalPairedBars:
    def __init__(self, glyph):
        if glyph.source["id"] != SAMPLE or glyph.source["character"] != "|":
            raise ValueError("Paired bar candidate requires the registered single-bar source")
        self.glyph = glyph
        self.decisions = []

    def apply(self, page, renderer, baseline):
        records = copy.deepcopy(baseline)
        before_items = list(page.items)
        self.decisions = []
        for row in records:
            if row["context"] != "mathematics" or row["semantic"] != "‖":
                continue
            i = row["item_index"]
            old = copy.deepcopy(row)
            decision = {"item_index": i, "applied": False, "source_id": SAMPLE}
            self.decisions.append(decision)
            if row["source_type"] not in {"external-font", "inherited-font"}:
                decision["reason"] = "existing non-font source retained"
                continue
            x0, y0, x1, y1 = row["bounds_px"]
            span = np.ptp(
                outline_bounds(self.glyph.outline.verts, self.glyph.outline.codes), axis=0
            )
            width = min((y1 - y0) * span[0] / span[1], (x1 - x0) * 0.32)
            if width <= 0 or (x1 - x0) / 2 - width < 0.05 * renderer.cfg.body_size:
                decision["reason"] = "parser slot cannot separate two source strokes"
                continue
            strokes, treatments = [], []
            for fraction in (0.25, 0.75):
                centre = x0 + (x1 - x0) * fraction
                slot = [centre - width / 2, y0, centre + width / 2, y1]
                outline, fit = placed_outline(self.glyph.outline, slot, stretch=True)
                limited = cap_stem(
                    MplPath(outline.verts, outline.codes), 0.055 * renderer.cfg.body_size
                )
                if limited:
                    outline, treatment = limited
                else:
                    treatment = {"method": "v26 stem limit not needed at this scale"}
                strokes.append(MplPath(outline.verts, outline.codes))
                treatments.append({"source_to_page_fit": fit, "stroke_width_policy": treatment})
            left, right = [outline_bounds(p.vertices, p.codes).ravel() for p in strokes]
            gap = float(right[0] - left[2])
            if gap < 0.05 * renderer.cfg.body_size or strokes[0].intersects_path(
                strokes[1], filled=True
            ):
                decision["reason"] = "fitted source strokes overlap or lack clear separation"
                continue
            path = MplPath.make_compound_path(*strokes)
            bounds = outline_bounds(path.vertices, path.codes).ravel().tolist()
            if min(bounds[0] - x0, bounds[1] - y0, x1 - bounds[2], y1 - bounds[3]) < -1e-6:
                decision["reason"] = "composed bars exceed the original math slot"
                continue
            neighbors = []
            for other in records:
                j = other["item_index"]
                if i == j or page.items[j][0] != "path":
                    continue
                if path.intersects_path(placed_path(page.items[j], renderer), filled=True):
                    neighbors.append(j)
            if neighbors:
                decision.update(
                    reason="composed source stroke touches another item", neighbors=neighbors
                )
                continue
            key = ("paired-personal-vertical-bars", i, SAMPLE)
            renderer._outlines[key] = Outline(path.vertices, path.codes, stroke=False, digits=4)
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), page.items[i][3], 0)
            row.pop("fallback_reason", None)
            row.update(
                source_type="personal-component-composition",
                personal_collected=True,
                external_fallback=False,
                procedural_geometry=True,
                font=None,
                font_sha256=None,
                variant=None,
                glyph_name=None,
                actual_glyph_id=f"two-instances:{SAMPLE}",
                session_source=self.glyph.source,
                original_bounds_px=old["bounds_px"],
                allowed_bounds_px=old["bounds_px"],
                bounds_px=bounds,
                paired_bar_feedback={
                    "original": old,
                    "source_label": "|",
                    "source_parent_label": self.glyph.source["parent_latex"],
                    "semantic_mapping": "two vertical bars -> existing norm delimiter slot",
                    "source_instances": 2,
                    "new_collected_double_bar": False,
                    "raw_rgb_and_labels_unchanged": True,
                    "transforms": treatments,
                    "clear_gap_px": gap,
                    "separate_strokes": True,
                    "layout_and_semantic_identity_unchanged": True,
                },
            )
            decision.update(applied=True, clear_gap_px=gap, bounds=bounds, transforms=treatments)
        if identity(records) != identity(baseline):
            raise ValueError("Paired bars changed semantic identity/order")
        allowed = {d["item_index"] for d in self.decisions if d["applied"]}
        for i, (a, b, old, new) in enumerate(
            zip(before_items, page.items, baseline, records, strict=True)
        ):
            if i not in allowed and (a != b or old != new):
                raise ValueError("Paired bars changed non-target ink or provenance")
        return records


def main():
    root = workspace_root()
    base = root / "runs" / BASE
    verify_run(base)
    cfg = RenderConfig.model_validate(read(base / "run.json")["config"]["render"])
    run = Run.create(
        "paired-personal-bars-candidate",
        seed=cfg.seed,
        config={
            "argv": sys.argv,
            "base_run": str(base),
            "base_run_sha256": sha256(base / "run.json"),
            "render": cfg.model_dump(mode="json"),
            "source_id": SAMPLE,
            "page": 2,
            "new_neural_updates": 0,
            "new_collected_double_bar": False,
            "production": False,
            "hypothesis": (
                "two reviewed single-bar instances can fill each existing norm delimiter slot"
            ),
        },
    )
    print(run.path, flush=True)
    try:
        protected = {
            str(root / "latest.json"): sha256(root / "latest.json"),
            str(base / "run.json"): sha256(base / "run.json"),
        }
        atomic_json(run.path / "reports/protected-assets.json", protected)
        for path in [
            *source_paths(),
            Path(__file__),
            Path("research/author_feedback_revision.py"),
            Path("research/cpu_glyph_feedback.py"),
            Path("research/math_feedback_candidate.py"),
        ]:
            target = run.path / "logs/source" / path.resolve().relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(base / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for name in ("historical-glyphs", "session-glyphs"):
            shutil.copytree(base / f"solutions/{name}", run.path / f"solutions/{name}")
        inputs = run.path / f"solutions/inputs/{CASE}"
        shutil.copytree(base / f"solutions/inputs/{CASE}", inputs)
        profile = load_snapshot(run.path / "solutions/profile-v1")
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        job = Job.model_validate_json((inputs / "job.json").read_text())
        bg = job.background
        effective = cfg.model_copy(update={"paper": "plain" if bg.path else bg.kind})
        page, renderer, before, _, _ = personal_layout(
            (inputs / "solution.md").read_text(),
            CASE,
            profile,
            effective,
            resolver,
            sessions,
            polish,
            bg,
            inputs / bg.path if bg.path else None,
            cpu=True,
            clearance=True,
        )
        a_image = raster(renderer.to_vector_pdf([page]))
        if a_image.tobytes() != Image.open(base / "rendered/pages/02.png").convert("RGB").tobytes():
            raise ValueError("Combined baseline did not replay exactly")
        before_trace = trace(renderer)
        candidate = PersonalPairedBars(source_bar(run.path / "solutions/session-glyphs"))
        records = candidate.apply(page, renderer, before)
        output = run.path / "rendered/pages/02.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)
        image = save_pdf(renderer.to_vector_pdf([page]), output)
        changes = {d["item_index"] for d in candidate.decisions if d["applied"]}
        atomic_json(
            run.path / f"reports/{CASE}-provenance.json",
            {
                "before": before,
                "candidate": records,
                "decisions": candidate.decisions,
                "parser_structure_trace": trace(renderer),
            },
        )
        if len(changes) != 6 or trace(renderer) != before_trace:
            raise ValueError("Not all six original norm slots accepted without structural change")
        checks = pixel_boundary_checks(a_image, image, before, records, changes)
        rows = [r for r in before if r["latex"] == before[min(changes)]["latex"]]
        boxes = np.asarray([r["bounds_px"] for r in rows])
        box = (
            *np.floor(boxes[:, :2].min(0) - 14).astype(int),
            *np.ceil(boxes[:, 2:].max(0) + 14).astype(int),
        )
        paired_panel(
            a_image.crop(box),
            image.crop(box),
            run.path / "comparisons/all-six-norms.png",
            "Page 2 / six norm delimiters",
            ("A: existing external font", "B: two real single-bar instances"),
        )
        paired_panel(
            a_image,
            image,
            run.path / "comparisons/page-02.png",
            "Original page 2",
            ("Combined baseline", "Personal paired-bar candidate"),
        )
        atomic_json(run.path / "reports/checks.json", checks)
        atomic_json(
            run.path / "reports/delivery-pages.json",
            {
                "base_run": str(base),
                "pages": [
                    {
                        "page": 2,
                        "case_id": CASE,
                        "pdf": "rendered/pages/02.pdf",
                        "png": "rendered/pages/02.png",
                        "provenance_report": f"reports/{CASE}-provenance.json",
                        "provenance_key": "candidate",
                    }
                ],
            },
        )
        assert all(sha256(Path(p)) == h for p, h in protected.items())
        run.finish(
            "held",
            result={
                "page": 2,
                "composed_norm_delimiters": 6,
                "new_collected_double_bar": False,
                "pixel_checks": checks,
                "fallbacks": fallback_summary(records),
                "author_acceptance": False,
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
