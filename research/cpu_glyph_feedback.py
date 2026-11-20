"""Isolated, source-traceable CPU 0/r/d/k/f feedback candidates; no model training.

The source font, original scans, labels and production renderer remain untouched.
Only r needs a layout-stage metric change; the other four are post-layout ink.
"""

import copy
import hashlib
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from handnote.markdown import Block
from handnote.opentype import read_alternates
from handnote.variants import handwritten_outline
from handnote.vector import Outline
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import text_to_path
from PIL import Image, ImageDraw

from hwsol.adapters.handnote import RENDER_LOCK, upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.handwriting import finish_ink
from hwsol.jobs.layout import MathDiagnostics, SinglePageRenderer, ink_geometry, validate_markdown
from hwsol.jobs.math_strokes import apply_math_strokes
from hwsol.jobs.provenance import classify_rules, item_bounds, visible_provenance
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run
from research.corpus.context_composer import layout
from research.corpus.math_glyph_resolver import placed_outline
from research.corpus.math_session_polish import body_width, polygons
from research.corpus.readability import placed_path
from research.homework_benchmark.render import save_pdf

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "workspace/runs/20261120T064909731073Z-v01026-session-math-polish-b9371278"
TARGETS = frozenset("0rdkf")
CONFIG = RenderConfig(dpi=300, font_size_pt=11, seed=1909, ink="#000000", math_strokes="personal")
R_LEFT_BEARING_REDUCTION_EM = 0.021


class FeedbackRenderer(SinglePageRenderer):
    """Reduce r's inherited left bearing and pen advance by exactly the same amount."""

    feedback_r_enabled = True

    def char_metrics(self, ch, size, k=None):
        shift, advance = super().char_metrics(ch, size, k)
        if ch == "r" and self.feedback_r_enabled:
            delta = size * R_LEFT_BEARING_REDUCTION_EM
            return shift - delta, advance - delta
        return shift, advance

    def draw_matrix(self, *args, **kwargs):
        # Single-letter matrix cells must retain the mathematical layout policy.
        self.feedback_r_enabled = False
        try:
            return super().draw_matrix(*args, **kwargs)
        finally:
            self.feedback_r_enabled = True


def layout_with_r_metrics(source, identity, profile, cfg=CONFIG):
    """Existing layout contract with the opt-in renderer subclass; no monkeypatching."""
    blocks = validate_markdown(source, profile)
    upstream = upstream_config(profile.font, cfg, bool(profile.manifest.variants))
    upstream = upstream.replace(
        h1_size=upstream.body_size,
        h2_size=upstream.body_size,
        h3_size=upstream.body_size,
        math_scale=1.0,
    )
    with RENDER_LOCK, rc_context():
        logger = logging.getLogger("matplotlib._mathtext")
        handler = MathDiagnostics()
        logger.addHandler(handler)
        try:
            renderer = FeedbackRenderer(upstream)
            renderer.configure("baseline", profile, identity, None, cfg, None)
            pages = renderer.render_vector([Block("h2", (("t", identity),)), *blocks])
        finally:
            logger.removeHandler(handler)
        if len(pages) != 1 or renderer.warnings:
            raise ValueError(f"Unreliable feedback layout: {renderer.warnings}")
        page = pages[0]
        classify_rules(page, renderer)
        strokes = (
            apply_math_strokes(page, renderer, profile) if cfg.math_strokes == "personal" else []
        )
        finish_ink(page, renderer, profile, False, renderer.stroke_seed)
        records = visible_provenance(page, renderer, profile, strokes)
        for record in records:
            if record["context"] == "text" and record["semantic"] == "r":
                record["cpu_feedback_layout"] = {
                    "method": "inherited left bearing 46->25 per 1000 em",
                    "draw_shift_delta_em": -R_LEFT_BEARING_REDUCTION_EM,
                    "advance_delta_em": -R_LEFT_BEARING_REDUCTION_EM,
                    "outline_unchanged": True,
                    "following_text": "moves with reduced advance; wrapping may change",
                    "space_policy": "unchanged",
                }
        bounds, _ = ink_geometry(page, renderer)
        if min(bounds[:2]) < 3 or bounds[2] > page.width - 3 or bounds[3] > page.height - 3:
            raise ValueError("Feedback layout crosses page boundary")
        return page, renderer, records


def lower_k_body(vertices, codes):
    """Monotone piecewise output mapping; original outlines are never overwritten."""
    raw = MplPath(vertices, codes)
    lo, hi = raw.get_extents().ymin, raw.get_extents().ymax
    # x-height is measured from v1 a/c/e/o, approximately 470/1000 em.
    knot, target = 470.0, 410.0
    if not lo < target < knot < hi:
        raise ValueError("Unexpected k geometry; no unbounded fallback")
    paths = []
    for ring in polygons(raw):
        dense = []
        for start, end in zip(ring[:-1], ring[1:], strict=True):
            dense.append(start)
            if (start[1] - knot) * (end[1] - knot) < 0:
                dense.append(start + (end - start) * (knot - start[1]) / (end[1] - start[1]))
        pts = np.vstack([*dense, ring[-1]])
        pts[:, 1] = np.interp(pts[:, 1], [lo, knot, hi], [lo, target, hi])
        paths.append(MplPath(pts, closed=True))
    result = MplPath.make_compound_path(*paths)
    return result.vertices, result.codes


class CpuGlyphFeedback:
    """Recorded variants; default prose/title, explicit opt-in math d/f/k/r only."""

    def __init__(self, font: Path):
        self.font = font
        self.font_sha = sha256(font)
        with TTFont(font) as file:
            cmap, alts = file.getBestCmap(), read_alternates(file)
            self.names = {ch: [cmap[ord(ch)], *alts.get(cmap[ord(ch)], [])] for ch in TARGETS}
            self.ids = {
                name: file.getGlyphID(name) for names in self.names.values() for name in names
            }

    def apply(self, page, renderer, provenance, *, contexts=frozenset({"text"})):
        contexts = frozenset(contexts)
        if not contexts or not contexts <= {"text", "mathematics"}:
            raise ValueError("Explicit supported feedback contexts required")
        records = copy.deepcopy(provenance)
        for row in records:
            ch, i = row["semantic"], row["item_index"]
            if row["context"] not in contexts or ch not in TARGETS or page.items[i][0] != "path":
                continue
            if row["context"] == "mathematics" and ch not in {"d", "f", "k", "r"}:
                continue
            if not row["personal_collected"] or row.get("font_sha256") != self.font_sha:
                row["cpu_feedback_rejection"] = "Expected pinned personal font provenance"
                continue
            variant = row.get("variant")
            treatment = None
            selected = variant
            old_item = page.items[i]
            before = row["bounds_px"]
            if ch == "0" and variant in (2, 3):
                selected = variant - 2
                v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                new, scale = placed_outline(Outline(v * (1, -1), c), before)
                old_weight = body_width(placed_path(old_item, renderer))
                new_weight = body_width(MplPath(new.verts, new.codes))
                if new_weight >= old_weight * 0.90:
                    row["cpu_feedback_rejection"] = (
                        "Lighter zero did not pass existing v26 width check"
                    )
                    continue
                matrix = (1, 0, 0, 1, 0, 0)
                treatment = {
                    "method": "v26 lighter recorded zero, extended to text",
                    "fit": scale,
                    "stem_p90_px": [old_weight, new_weight],
                }
            elif ch == "d" and variant == 0:
                selected = 1
                v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                # Fit the alternate in the old body box: no neighbour collision or new d metrics.
                new, scale = placed_outline(Outline(v * (1, -1), c), before)
                matrix = (1, 0, 0, 1, 0, 0)
                treatment = {"method": "select recorded d1 with lower bowl", "fit": scale}
            elif ch == "f" and variant == 1:
                selected = 0
                if row["context"] == "mathematics":
                    # MathText outlines use FONT_SCALE (100), unlike prose's 1000 em.
                    v, c = renderer._math_glyph(
                        FT2Font(str(self.font)), self.ids[self.names[ch][0]], ord(ch), 0, True
                    )
                else:
                    v, c = handwritten_outline(self.font, ord(ch), selected, 1000)
                new, matrix = Outline(v, c, stroke=False, digits=4), old_item[2]
                treatment = {"method": "select fuller recorded f0 at native em scale; no dilation"}
                if row["context"] == "mathematics":
                    treatment["outline_units_per_em"] = text_to_path.FONT_SCALE
                    treatment["mathtext_primary_slot"] = True
            elif ch == "r" and row["context"] == "mathematics" and variant == 1:
                selected = 0
                v, c = renderer._math_glyph(
                    FT2Font(str(self.font)), self.ids[self.names[ch][0]], ord(ch), 0, True
                )
                new, matrix = Outline(v, c, stroke=False, digits=4), old_item[2]
                treatment = {
                    "method": "select recorded r0 in MathText's existing primary slot",
                    "outline_units_per_em": text_to_path.FONT_SCALE,
                    "mathtext_primary_slot": True,
                    "primary_advance_preserved": True,
                    "alternate_not_stretched": True,
                }
            elif ch == "k":
                if row["context"] == "mathematics":
                    existing = renderer._outlines[old_item[1]]
                    units = 1000 / text_to_path.FONT_SCALE
                    v, c = lower_k_body(existing.verts * units, existing.codes)
                    v = v / units
                else:
                    v, c = handwritten_outline(self.font, ord(ch), variant, 1000)
                    v, c = lower_k_body(v, c)
                new, matrix = Outline(v, c, stroke=False, digits=4), old_item[2]
                treatment = {
                    "method": "output-only monotone lower-body y calibration",
                    "knot_units": [470, 410],
                    "endpoints_fixed": True,
                    "author_style_acceptance": "pending",
                }
                if row["context"] == "mathematics":
                    treatment["existing_mathtext_fitted_outline_preserved"] = True
                    treatment["outline_units_per_em"] = text_to_path.FONT_SCALE
            if treatment is None:
                continue
            key = ("cpu-glyph-feedback", ch, i, selected)
            renderer._outlines[key] = new
            page.items[i] = ("path", key, matrix, old_item[3], 0)
            row.update(
                cpu_feedback={
                    "original_variant": variant,
                    "source_font_sha256": self.font_sha,
                    "original_bounds_px": before,
                    **treatment,
                },
                bounds_px=item_bounds(page.items[i], renderer),
                variant=selected,
                glyph_name=self.names[ch][selected],
                actual_glyph_id=self.ids[self.names[ch][selected]],
                source_type="personal-outline-geometric-calibration"
                if ch == "k"
                else "personal-outline",
            )
        return records


def font_metrics(font):
    result = []
    with TTFont(font) as file:
        cmap, alts = file.getBestCmap(), read_alternates(file)
        for ch in "0rdkfaceo":
            names = [cmap[ord(ch)], *alts.get(cmap[ord(ch)], [])]
            for variant, name in enumerate(names):
                v, c = handwritten_outline(font, ord(ch), variant, 1000)
                path = MplPath(v, c)
                box = path.get_extents()
                advance, lsb = file["hmtx"][name]
                points = v[c != MplPath.CLOSEPOLY]
                region = points[points[:, 0] < box.xmin + box.width * 0.55] if ch == "d" else points
                result.append(
                    {
                        "char": ch,
                        "variant": variant,
                        "glyph": name,
                        "advance": advance,
                        "lsb": lsb,
                        "rsb": advance - box.xmax,
                        "bounds": box.extents.tolist(),
                        "ink_width": box.width,
                        "stem_width_p90": body_width(path),
                        "left_body_min_y": float(region[:, 1].min()),
                    }
                )
    return result


def variant_panel(font, destination):
    fig = Figure(figsize=(12, 10), dpi=160)
    FigureCanvasAgg(fig)
    for row, ch in enumerate("0rdkf"):
        for variant in range(4 if ch == "0" else 2):
            ax = fig.add_subplot(5, 4, row * 4 + variant + 1)
            v, c = handwritten_outline(font, ord(ch), variant, 1000)
            ax.add_patch(PathPatch(MplPath(v, c), fc="#202838", lw=0))
            ax.axhline(0, lw=0.5, color="blue")
            ax.axhline(470, lw=0.5, color="green")
            ax.set(
                xlim=(-50, 850),
                ylim=(-90, 850),
                aspect="equal",
                title=f"{ch}: real variant {variant}",
            )
            ax.axis("off")
        if ch == "k":
            for variant in (0, 1):
                ax = fig.add_subplot(5, 4, row * 4 + variant + 3)
                v, c = handwritten_outline(font, ord(ch), variant, 1000)
                v, c = lower_k_body(v, c)
                ax.add_patch(PathPatch(MplPath(v, c), fc="#202838", lw=0))
                ax.axhline(0, lw=0.5, color="blue")
                ax.axhline(470, lw=0.5, color="green")
                ax.set(
                    xlim=(-50, 850),
                    ylim=(-90, 850),
                    aspect="equal",
                    title=f"k{variant}: lower-body candidate",
                )
                ax.axis("off")
    fig.tight_layout()
    fig.savefig(destination)


def source_evidence():
    rows = []
    samples = ROOT / "profiles/realism-studio/samples/personal"
    for sid in ("s-c222860c59414473bf606d42eade46b9", "s-6f23c7cfdf304bafa153c2b86c002a3c"):
        folder = samples / sid
        manifest = json.loads((folder / "sample.json").read_text())
        for asset in manifest["files"]:
            assert sha256(folder / asset["path"]) == asset["sha256"]
        rows.append(
            {
                "sample": manifest,
                "folder": str(folder),
                "manifest_sha256": sha256(folder / "sample.json"),
                "originals_preserved": True,
                "new_crop_or_relabel": False,
            }
        )
    return rows


def compiled_lineage(font):
    """Bind selected existing contours to retained sample font, labels and scans."""
    sid = "s-c222860c59414473bf606d42eade46b9"
    folder = ROOT / "profiles/realism-studio/samples/personal" / sid
    source = ROOT / (
        "profiles/realism-studio/training-runs/personal/"
        "r-e9ef6e0126b84037ad2633745370d424/sample-0000.ttf"
    )
    template = json.loads((folder / "template.json").read_text())
    cells = {cell["char"]: cell for cell in template["cells"]}
    rows = []
    for ch in sorted(TARGETS):
        for variant in (0, 1):
            final, codes = handwritten_outline(font, ord(ch), variant, 1000)
            original, source_codes = handwritten_outline(source, ord(ch), variant, 1000)
            if not np.array_equal(codes, source_codes) or final.shape != original.shape:
                raise ValueError(f"Unverified compiled glyph lineage: {ch}/{variant}")
            a, b = final[codes != MplPath.CLOSEPOLY], original[codes != MplPath.CLOSEPOLY]
            a = (a - a.min(axis=0)) / np.ptp(a, axis=0)
            b = (b - b.min(axis=0)) / np.ptp(b, axis=0)
            deviation = float(np.abs(a - b).max())
            if deviation > 0.004:
                raise ValueError(f"Compiled contour mismatch: {ch}/{variant}")
            original_scan = folder / f"original-{variant:03d}.jpg"
            rows.append(
                {
                    "character": ch,
                    "variant": variant,
                    "sample_id": sid,
                    "sample_state": "accepted",
                    "template_cell": cells[ch],
                    "original_scan": str(original_scan),
                    "original_sha256": sha256(original_scan),
                    "compiled_sample_font": str(source),
                    "sample_font_sha256": sha256(source),
                    "final_font_sha256": sha256(font),
                    "contour_codes_exact": True,
                    "normalized_vertex_max_delta": deviation,
                    "interpretation": (
                        "same traced contour, with existing merge fit and integer rounding"
                    ),
                    "coordinate_scope": (
                        "template bbox is in registered sheet coordinates, not raw JPG pixels"
                    ),
                    "original_pixels_or_labels_edited": False,
                }
            )
    return rows


def launch():
    run = Run.create(
        "cpu-glyph-feedback-diagnostic",
        seed=1909,
        config={
            "targets": sorted(TARGETS),
            "render": CONFIG.model_dump(),
            "neural_training": False,
            "purpose": "exposed-development author feedback, not fresh holdout",
            "activation": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        profile = load_snapshot(BASELINE / "solutions/profile-v1")
        copy_snapshot(profile, run.path / "solutions/profile-v1")
        atomic_json(run.path / "reports/sources.json", source_evidence())
        atomic_json(run.path / "reports/compiled-font-lineage.json", compiled_lineage(profile.font))
        atomic_json(run.path / "reports/font-metrics.json", font_metrics(profile.font))
        variant_panel(profile.font, run.path / "comparisons/real-variants.png")
        text = (
            "0 00 10 20 30 40 50 60 70 80 90 100.\n\n"
            "Prior error terms remain near zero. Rare roots are proper and regular.\n\n"
            "Derived data and added details determine the independent directions.\n\n"
            "Known kernels keep the rank and make the linked blocks work.\n\n"
            "From the first coefficient, find a finite difference for each factor.\n\n"
            "Different effective offsets follow from careful verification of the final formula."
        )
        title = "Problem 010: recorded CPU glyph feedback"
        (run.path / "solutions/probe.md").write_text(text + "\n")
        a, ar, ap = layout(text, title, profile, CONFIG)
        b, br, bp = layout_with_r_metrics(text, title, profile, CONFIG)
        assert [(r["semantic"], r["context"]) for r in ap] == [
            (r["semantic"], r["context"]) for r in bp
        ]
        feedback = CpuGlyphFeedback(profile.font)
        final = feedback.apply(b, br, bp)
        for old, new, record in zip(a.items, b.items, ap, strict=True):
            if record["semantic"] not in TARGETS:
                assert old[1] == new[1]
                assert ar._outlines[old[1]].verts.tobytes() == br._outlines[new[1]].verts.tobytes()
                assert ar._outlines[old[1]].codes.tobytes() == br._outlines[new[1]].codes.tobytes()
        ai = save_pdf(ar.to_vector_pdf([a]), run.path / "comparisons/A-original.pdf")
        bi = save_pdf(br.to_vector_pdf([b]), run.path / "comparisons/B-candidate.pdf")
        all_boxes = np.asarray([r["bounds_px"] for r in [*ap, *final]])
        box = [
            int(all_boxes[:, 0].min()) - 15,
            int(all_boxes[:, 1].min()) - 20,
            int(all_boxes[:, 2].max()) + 15,
            int(all_boxes[:, 3].max()) + 20,
        ]
        crops = [im.crop(box).convert("RGB") for im in (ai, bi)]
        canvas = Image.new(
            "RGB",
            (sum(im.width for im in crops) + 32, max(im.height for im in crops) + 80),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 16), "A: existing v1 CPU", fill="black")
        draw.text((crops[0].width + 32, 16), "B: explicit 0/r/d/k/f candidate", fill="black")
        canvas.paste(crops[0], (8, 60))
        canvas.paste(crops[1], (crops[0].width + 24, 60))
        canvas.save(run.path / "comparisons/word-and-title-comparison.png", dpi=(300, 300))
        changes = [r for r in final if "cpu_feedback" in r]
        atomic_json(run.path / "reports/provenance.json", {"A": ap, "B": final})
        atomic_json(
            run.path / "reports/diagnosis.json",
            {
                "zero": (
                    "v26 lighter variant policy had mathematics-only guard; "
                    "title text retained coarse variants 2/3"
                ),
                "r": (
                    "Inherited LSB46, RSB15; no large advance-width mismatch. LSB46->25 "
                    "subtracts equal amount from draw shift and advance; no glyph stretch."
                ),
                "d": "Choose recorded d1 with a lower bowl; fit within previous ink box",
                "f": (
                    "f1 is narrower/thinner, with normal height. Choose actual f0 at native em; "
                    "no synthesis/dilation/alpha trick"
                ),
                "k": (
                    "Both variants have a high lower arm; output-only monotone y calibration "
                    "470->410/1000 em, endpoints fixed; author review pending"
                ),
                "r_delta_at_11pt_300dpi_px": CONFIG.pt(11) * R_LEFT_BEARING_REDUCTION_EM,
                "non_target_outlines": "byte-exact; positions may follow r advance reduction",
                "word_space_policy": "unchanged char_metrics for spaces and non-r characters",
                "changes": dict(Counter(r["semantic"] for r in changes)),
                "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "not_proven": "Author style and whole-20-page integration pending review",
            },
        )
        run.finish(
            "held",
            result={
                "diagnostics": "rendered A/B title and word probes plus all actual variants",
                "source_font_sha256": sha256(profile.font),
                "changed_glyphs": len(changes),
                "neural_updates": 0,
                "production": "unchanged",
                "decision": "pending root visual review before page integration",
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    launch()
