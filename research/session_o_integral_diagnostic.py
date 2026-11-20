"""One fixed integral-left alignment proof; rejected centered placement retained."""

import copy
import shutil
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import crop_pair, json_value, snapshot_sources
from research.author_feedback_round2_checks import page_geometry
from research.corpus.math_session_glyphs import outline_bounds
from research.f_size_feedback import contour_distance
from research.homework_benchmark.render import read, save_pdf
from research.math_feedback_candidate import paired_panel
from research.session_o_baseline import load_baseline, replay_page, validate_increment
from research.session_o_delivery import verify_source_placements
from research.session_o_resolver import SessionOResolver

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = "20261120T134632687261Z-session-o-reviewed-math-complete-20-adf8f1c8"
DELIVERY_SHA = "99b456d9e0de2dfb01f1ef4144082ae12eb55f60023fcc70a737c1c112f9b79c"


def main():
    parent = workspace_root() / "runs" / DELIVERY
    verify_run(parent)
    if sha256(parent / "run.json") != DELIVERY_SHA:
        raise ValueError("Frozen first Session O delivery changed")
    resolver_file = ROOT / "research/session_o_resolver.py"
    if sha256(resolver_file) != sha256(parent / "logs/source/research/session_o_resolver.py"):
        raise ValueError("Diagnostic requires the exact originally rejected resolver")
    run = Run.create(
        "session-o-integral-left-alignment-proof",
        seed=1909,
        config={
            "parent": DELIVERY,
            "parent_sha256": DELIVERY_SHA,
            "page": 13,
            "item_index": 200,
            "rule": (
                "Uniformly fit integral as before, then anchor ink left edge "
                "to original slot left edge"
            ),
            "parameter_search": False,
            "font_geometry_or_advance_changes": False,
            "neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, {"first-session-o-delivery": parent})
        bank = parent / "solutions/session-o-bank"
        shutil.copytree(bank, run.path / "solutions/source-bank")
        b = load_baseline()
        r = replay_page(b, 13)
        page, renderer = r["page"], r["renderer"]
        case_id = r["entry"]["case_id"]
        i = 200
        resolver = SessionOResolver(bank, seed=1909)
        before = json_value(resolver.apply(page, renderer, r["records"], case_id=case_id))
        archived = read(parent / f"reports/{case_id}-provenance.json")
        if before != archived["candidate"]:
            raise ValueError("Diagnostic baseline provenance does not reproduce adf")
        a_pdf = renderer.to_vector_pdf([page])
        if a_pdf != (parent / "rendered/pages/13.pdf").read_bytes():
            raise ValueError("Diagnostic baseline PDF is not exact adf P13")
        a = save_pdf(a_pdf, run.path / "rendered/A-retained-font.pdf")
        paths = page_geometry(page, renderer)
        original_items = list(page.items)
        sample = resolver.variants["∮"][0]
        if len(resolver.variants["∮"]) != 1:
            raise ValueError("This proof does not choose among new source variants")
        centered, transform = SessionOResolver.fit(sample, before[i]["bounds_px"], stretch=False)
        centered_box = outline_bounds(centered.verts, centered.codes).ravel()
        dx = before[i]["bounds_px"][0] - centered_box[0]
        moved_transform = {
            **transform,
            "translation_xy": [transform["translation_xy"][0] + dx, transform["translation_xy"][1]],
            "horizontal_anchor": "original integral slot left edge",
            "x_translation_from_rejected_center_px": dx,
        }
        shifted = Outline(
            sample.outline.verts * np.asarray(moved_transform["scale_xy"])
            + np.asarray(moved_transform["translation_xy"]),
            sample.outline.codes.copy(),
            stroke=False,
            digits=4,
        )
        bb = outline_bounds(shifted.verts, shifted.codes).ravel().tolist()
        modes = {"B-rejected-centered": centered, "C-left-aligned": shifted}
        images = {}
        neighbors = []
        for mode, outline in modes.items():
            key = ("integral-diagnostic", mode)
            renderer._outlines[key] = outline
            page.items[i] = ("path", key, (1, 0, 0, 1, 0, 0), original_items[i][3], 0)
            images[mode] = save_pdf(
                renderer.to_vector_pdf([page]), run.path / f"rendered/{mode}.pdf"
            )
            np.savez(
                run.path / f"solutions/{mode}-contour.npz",
                vertices=outline.verts,
                codes=outline.codes,
            )
        proposed = MplPath(shifted.verts, shifted.codes)
        centered_path = MplPath(centered.verts, centered.codes)
        for j, other in enumerate(paths):
            if i == j:
                continue
            ob = other.get_extents()
            pb = proposed.get_extents()
            gap = np.hypot(
                max(0, ob.xmin - pb.xmax, pb.xmin - ob.xmax),
                max(0, ob.ymin - pb.ymax, pb.ymin - ob.ymax),
            )
            if gap > 100:
                continue
            old_gap = contour_distance(paths[i], other)
            rejected_gap = contour_distance(centered_path, other)
            new_gap = contour_distance(proposed, other)
            neighbors.append(
                {
                    "item_index": j,
                    "semantic": before[j]["semantic"],
                    "old_font_clearance_px": old_gap,
                    "rejected_center_clearance_px": rejected_gap,
                    "left_aligned_clearance_px": new_gap,
                    "original_bounds_px": before[j]["bounds_px"],
                }
            )
            if new_gap < 0.60:
                raise ValueError(f"Fixed left alignment does not meet 0.60px clearance: item {j}")
        candidate = copy.deepcopy(before)
        candidate[i].update(
            session_o_previous=copy.deepcopy(before[i]),
            session_o_source=sample.source,
            session_o_bank_sha256=resolver.manifest_sha256,
            actual_glyph_id=sample.source["id"],
            source_type="personal-session-outline",
            personal_collected=True,
            external_fallback=False,
            procedural_geometry=False,
            font=None,
            font_sha256=None,
            glyph_name=None,
            variant=0,
            original_bounds_px=before[i]["bounds_px"],
            allowed_bounds_px=before[i]["bounds_px"],
            bounds_px=bb,
            transformation=moved_transform,
        )
        candidate[i].pop("fallback_reason", None)
        checks = validate_increment(
            paths,
            page,
            renderer,
            before,
            candidate,
            a,
            images["C-left-aligned"],
            targets={"∮"},
            parser_before=r["parser"],
        )
        sources = verify_source_placements(bank, before, candidate, paths, page, renderer)
        atomic_json(
            run.path / "reports/provenance.json",
            {"before": before, "candidate": candidate, "parser": r["parser"]},
        )
        atomic_json(run.path / "reports/checks.json", checks)
        atomic_json(run.path / "reports/source-placement-checks.json", sources)
        atomic_json(run.path / "reports/neighbors.json", neighbors)
        atomic_json(run.path / "solutions/source.json", sample.source)
        shutil.copyfile(bank / sample.source["crop"], run.path / "solutions/original-rgb.png")
        pair_old = {
            "bounds_px": [
                before[i]["bounds_px"][0] - 8,
                before[i]["bounds_px"][1] - 12,
                before[202]["bounds_px"][2] + 12,
                before[i]["bounds_px"][3] + 12,
            ]
        }
        for mode, image in images.items():
            aa, cc = crop_pair(pair_old, pair_old, a, image, padding=(15, 12))
            paired_panel(
                aa,
                cc,
                run.path / f"comparisons/{mode}.png",
                "P13 closed integral, arrow and E",
                ("Original retained font", mode),
            )
        fig = Figure(figsize=(12, 5))
        FigureCanvasAgg(fig)
        for k, (label, target) in enumerate(
            (
                ("Original font", paths[i]),
                ("Rejected centered original ink", centered_path),
                ("Fixed left-edge original ink", proposed),
            ),
            1,
        ):
            ax = fig.add_subplot(1, 3, k)
            ax.add_patch(PathPatch(target, fc="#237d42", ec="none"))
            for j in (201, 202):
                ax.add_patch(PathPatch(paths[j], fc="#222222", ec="none"))
            x0, y0, x1, y1 = before[i]["bounds_px"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c="#ca3b3b", lw=0.7)
            ax.set(
                xlim=(x0 - 5, before[202]["bounds_px"][2] + 8),
                ylim=(y1 + 5, y0 - 15),
                aspect="equal",
                title=label,
            )
        fig.tight_layout()
        fig.savefig(run.path / "comparisons/actual-geometry.png", dpi=160)
        fig.savefig(run.path / "comparisons/actual-geometry.pdf")
        result = {
            "status": "fixed-integral-left-alignment-passes-engineering-awaiting-visual",
            "source_id": sample.source["id"],
            "dx_px": dx,
            "source_scale_unchanged": transform["scale_xy"],
            "vertical_position_unchanged": True,
            "bounds_inside_original_slot": True,
            "minimum_neighbor_clearance_px": min(n["left_aligned_clearance_px"] for n in neighbors),
            "E_clearance_px": next(
                n["left_aligned_clearance_px"] for n in neighbors if n["item_index"] == 202
            ),
            "no_neighbor_or_layout_changes": True,
            "no_ink_redrawing_or_rescaling": True,
            "generic_rule": (
                "Integral ink left edge aligns to original slot left edge "
                "after existing uniform fit"
            ),
            "parameter_search": False,
            "author_acceptance": False,
            "production_activation": False,
        }
        if sha256(parent / "run.json") != DELIVERY_SHA:
            raise ValueError("Parent modified")
        run.finish("held", result=result)
        verify_run(run.path)
        print(result, flush=True)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
