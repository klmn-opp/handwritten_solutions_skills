"""Exposed M1 feedback: explicit fraction revisions and conservative root clearance.

This module is isolated from the frozen M1/v26 implementation. It changes no
source handwriting and trains no model. A source revision is not a new holdout.
"""

import argparse
import copy
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import sympy as sp
from handnote.vector import Outline
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.paper import paper_image
from hwsol.jobs.schema import Background, Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.corpus.context_composer import layout
from research.corpus.math_glyph_completion import SELECTION, local_comparison
from research.corpus.math_glyph_resolver import MathGlyphResolver
from research.corpus.math_session_glyphs import SessionGlyphCalibration, outline_bounds
from research.corpus.math_session_polish import SessionInkPolish, polygons, sections
from research.corpus.readability import placed_path
from research.homework_benchmark.render import (
    BASELINE,
    LEGACY,
    accepted_scans,
    fallback_summary,
    formula_crops,
    pixel_boundary_checks,
    read,
    save_pdf,
    source_paths,
)

AGGREGATE = "20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7"
MATH = re.compile(r"\$\$.*?\$\$|\$[^$]+\$", re.S)
SOURCE_SHA256 = {
    "m1-la-01-rref-null": "8739ab77e500a9a0eeade8280cee87e955e0ed60577ab98f656c02408a7cb637",
    "m1-la-02-orthogonal-projection": (
        "57d9643876ab3b241ddedf3fa99bb8164458b26e529cffa119b8616e18fa6358"
    ),
    "m1-la-03-two-projectors": "b03235e7e3d0e0530d15b690a4eb146a49e782f8482322238ad1714101d9159a",
    "m1-la-04-jordan-recurrence": (
        "7b47ca17a24d40955e9e36b5eb730123af1f13c22319392fd06dcfeac61f09c1"
    ),
    "m1-ca-01-two-substitutions": (
        "8dd3d41787a1d842b4f2178b36107e57240c60c17773561c754073df23fd82d8"
    ),
    "m1-ca-02-ellipse-integral": "c792e0d311a1e8313dc038f19ed42f5bc5986298620a2f18423ad5e5600dada6",
    "m1-ca-04-taylor-series": "17978c4145af76c73dcce2b9f56de0bdfda9773afa7f343129cf047e57f3db40",
    "m1-st-01-normal-likelihood": (
        "6012ede39303d9a902024a180c521e2241246a86b24a62d1147eb5372c67602c"
    ),
    "m1-st-02-weighted-fit": "8a0903c00e621eec6bd146ebc3e4064f628f119795b710b1a7fecffa288de13b",
    "m1-st-03-contingency": "eccf7947462bd88a83322e7cccd4847c8a566cf187577f770b67836a8ed605bc",
    "m1-ph-01-potential-field": "58dd022e9120701b6cd4e359f32f9555a97ea7b340f2452a492d167c1c6c77b6",
    "m1-ph-02-rlc-phase": "46fd159c56a8630992ae9b78a971fd1ad76a42b177194a90c00f3811948b50cb",
    "m1-ph-03-faraday-loop": "1b68534b7f2c62f9207751950fe9f5f90ce8adcd55e0ed56eafdd82c2bae9876",
    "m1-la-05-hermitian": "9acdf244dda5f59eed2fd8ca8b2d883b504dcf2cf25541f00450e7f3a7798ee4",
    "m1-ca-05-critical-points": "faa09aabe46d0658f751785e007c6cba24591369cbf82babeebfec6926a06fe9",
    "m1-ca-06-improper-log": "83ef388f3c2f2899308405b3a8dc991a5c03df31f68e78b4b56fa6be4f79a493",
    "m1-st-05-paired-inference": "41f67a350ef2a390e4136895f27af2b2f4a709178be529b87c0fc24790df200c",
    "m1-st-06-exponential-scale": (
        "93ff15a6ad4662f6bb26d427257c9f43a30c2eea69194132bf4728b60323c5c5"
    ),
    "m1-ph-05-two-lenses": "0e7ceb8f5b1ba834356340de6b40b51b0e1be0130c2f6a8d7dcfef0af2b68dba",
    "m1-ph-06-travelling-wave": "509557a8afc1b241bbb974184d51c414fbb1a655248bb833e97f8bd7215a8b92",
}
# Exact, reviewed source fragments; this is deliberately not a slash rewriter.
# Fields: literal source, numerator LaTeX, denominator LaTeX, independent
# symbolic source expression, numerator expression, denominator expression.
FRACTIONS = {
    "m1-la-02-orthogonal-projection": [
        (f"{n}/3", str(n), "3", f"{n}/3", str(n), "3") for n in (8, 1, 7, 2)
    ],
    "m1-la-03-two-projectors": [("1/2", "1", "2", "1/2", "1", "2")],
    "m1-ca-01-two-substitutions": [
        ("1/[2(x^2+4)]", "1", "2(x^2+4)", "1/(2*(x**2+4))", "1", "2*(x**2+4)")
    ],
    "m1-ca-04-taylor-series": [
        ("x^2/2", "x^2", "2", "x**2/2", "x**2", "2"),
        ("1/2", "1", "2", "1/2", "1", "2"),
        ("R_4(x)/x^3", "R_4(x)", "x^3", "R4/x**3", "R4", "x**3"),
    ],
    "m1-st-01-normal-likelihood": [
        (r"3/\sqrt{4}", "3", r"\sqrt{4}", "3/sqrt(4)", "3", "sqrt(4)"),
        ("3/2", "3", "2", "3/2", "3", "2"),
    ],
    "m1-st-02-weighted-fit": [
        ("(8,-6,4)/19", "(8,-6,4)", "19", "Matrix([8,-6,4])/19", "Matrix([8,-6,4])", "19"),
        (r"2\sigma/\sqrt{19}", r"2\sigma", r"\sqrt{19}", "2*sigma/sqrt(19)", "2*sigma", "sqrt(19)"),
    ],
    "m1-st-03-contingency": [
        ("9/15", "9", "15", "9/15", "9", "15"),
        ("2.4/2", "2.4", "2", "Rational('2.4')/2", "Rational('2.4')", "2"),
        ("6/5", "6", "5", "6/5", "6", "5"),
    ],
    "m1-ph-02-rlc-phase": [(r"\pi/4", r"\pi", "4", "pi/4", "pi", "4")],
    "m1-la-05-hermitian": [("(9-1)/8", "9-1", "8", "(9-1)/8", "9-1", "8")],
    "m1-ca-05-critical-points": [
        ("1/2", "1", "2", "1/2", "1", "2"),
        ("1/6", "1", "6", "1/6", "1", "6"),
        ("1/12", "1", "12", "1/12", "1", "12"),
    ],
    "m1-ca-06-improper-log": [
        ("1/n", "1", "n", "1/n", "1", "n"),
        ("1/2", "1", "2", "1/2", "1", "2"),
    ],
    "m1-st-05-paired-inference": [(r"1/\sqrt{3}", "1", r"\sqrt{3}", "1/sqrt(3)", "1", "sqrt(3)")],
    "m1-st-06-exponential-scale": [
        (r"\theta^2/5", r"\theta^2", "5", "theta**2/5", "theta**2", "5"),
        ("9/5", "9", "5", "9/5", "9", "5"),
        ("1/3", "1", "3", "1/3", "1", "3"),
        (r"\lambda/4", r"\lambda", "4", "lam/4", "lam", "4"),
        (r"2\sum_iX_i/\theta", r"2\sum_iX_i", r"\theta", "2*sum_X/theta", "2*sum_X", "theta"),
    ],
}


def rewrite_fractions(case_id: str, source: str) -> tuple[str, list[dict]]:
    """Revise only registered math fragments; prose and unit notation stay exact."""
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256.get(case_id):
        raise ValueError("Fraction revision requires the exact frozen exposed-case source")
    rules = {row[0]: row for row in FRACTIONS.get(case_id, [])}
    if not rules:
        return source, []
    pattern = re.compile("|".join(re.escape(key) for key in sorted(rules, key=len, reverse=True)))
    seen, records = set(), []

    def math_region(match):
        def revise(fragment):
            old, num, den, independent, symbolic_num, symbolic_den = rules[fragment[0]]
            original = sp.sympify(independent)
            revised = sp.sympify(symbolic_num) / sp.sympify(symbolic_den)
            difference = sp.simplify(original - revised)
            if difference != (
                sp.zeros(*difference.shape) if isinstance(difference, sp.MatrixBase) else 0
            ):
                raise ValueError(f"Fraction semantics failed: {old}")
            new = rf"\frac{{{num}}}{{{den}}}"
            start = match.start() + fragment.start()
            records.append(
                {
                    "original_span": [start, start + len(old)],
                    "original": old,
                    "revised": new,
                    "operation": "explicit division to stacked numerator/denominator",
                    "independent_symbolic_source": independent,
                    "symbolic_numerator": symbolic_num,
                    "symbolic_denominator": symbolic_den,
                    "sympy_difference": str(difference),
                    "domain": "unchanged; denominator must be nonzero where required",
                }
            )
            seen.add(old)
            return new

        return pattern.sub(revise, match[0])

    revised = MATH.sub(math_region, source)
    if set(rules) != seen:
        raise ValueError(f"Reviewed source fragment missing for {case_id}: {set(rules) - seen}")
    if MATH.sub("<MATH>", source) != MATH.sub("<MATH>", revised):
        raise ValueError("Fraction revision changed prose")
    # Reconstruct from independently recorded disjoint spans; no unreported edit.
    replay = source
    for row in reversed(records):
        a, b = row["original_span"]
        assert replay[a:b] == row["original"]
        replay = replay[:a] + row["revised"] + replay[b:]
    assert replay == revised
    return revised, records


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
        # Outer roots first. Subsequent roots use the already-translated records.
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
                and r["latex"] == sign["latex"]
                and len(r["bounds_px"]) == 4
                and bx < (r["bounds_px"][0] + r["bounds_px"][2]) / 2 < br
                and r["bounds_px"][1] > bb - 1
                and r["bounds_px"][3] < sb + 2.5
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
            reduction = min(gap - target, (br - bx) - 2)
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
                        bx
                        + stem_shift
                        + (vertices[:, 0] - bx) * ((br - bx) - reduction) / (br - bx)
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
                    after.vertices, before.vertices + (-content_shift, 0), atol=1e-8
                ):
                    raise ValueError("Radicand shape changed during root compaction")
            decision.update(
                applied=True,
                gap_after_px=gap - reduction,
                stem_shift_px=stem_shift,
                radicand_shift_px=-content_shift,
            )
        return records


def render_pair(
    source,
    case_id,
    number,
    destination,
    cfg,
    profile,
    resolver,
    sessions,
    polish,
    background,
    background_asset,
):
    """Same-source personal-v26 versus clearance candidate, with explicit evidence."""
    page, renderer, initial = layout(source, case_id, profile, cfg)
    if background.kind == "image":
        page.background = paper_image(background, background_asset, (page.width, page.height))
        page.paper = None
    baseline = polish.apply(
        page,
        renderer,
        sessions.apply(
            page, renderer, resolver.resolve(page, renderer, initial, profile), operator_page=False
        ),
    )
    trace = copy.deepcopy({"formulas": renderer.formulas, "matrices": renderer.matrices})
    old_items = list(page.items)
    before_pdf = renderer.to_vector_pdf([page])
    a_path = destination / f"rendered/pages/A/{number:02d}-{case_id}.pdf"
    a_path.parent.mkdir(parents=True, exist_ok=True)
    before = save_pdf(before_pdf, a_path)
    update = RadicalClearancePolish()
    candidate = update.apply(page, renderer, baseline)
    actual = [(r["item_index"], r["semantic"], r["context"], r.get("latex")) for r in candidate]
    expected = [(r["item_index"], r["semantic"], r["context"], r.get("latex")) for r in baseline]
    if actual != expected or trace != {
        "formulas": renderer.formulas,
        "matrices": renderer.matrices,
    }:
        raise ValueError("Root candidate changed mathematical identity or parse trace")
    b_path = destination / f"rendered/pages/B/{number:02d}-{case_id}.pdf"
    b_path.parent.mkdir(parents=True, exist_ok=True)
    after = save_pdf(renderer.to_vector_pdf([page]), b_path)
    changed = {i for i, (a, b) in enumerate(zip(old_items, page.items, strict=True)) if a is not b}
    pixels = pixel_boundary_checks(before, after, baseline, candidate, changed)
    bounds, _ = ink_geometry(page, renderer)
    if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
        raise ValueError("Feedback candidate crosses page boundary")
    crops, panels = formula_crops(
        case_id,
        trace,
        baseline,
        candidate,
        before,
        after,
        destination / f"comparisons/formulas/{case_id}",
        number,
    )
    # The frozen helper's default headings describe the older font-vs-personal
    # comparison. These new panels compare personal v26 with personal v26 plus
    # spacing, so render accurate headings from the untouched native crops.
    for crop, (_, index, a_image, b_image) in zip(crops, panels, strict=True):
        paired_panel(
            a_image,
            b_image,
            destination / f"comparisons/formulas/{case_id}" / crop["comparison"],
            f"Page {number} / formula {index}",
            ("A: personal v26, revised fractions", "B: same source, root clearance"),
        )
    report = {
        "case_id": case_id,
        "output_page": number,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "status": "engineering checks passed; independent visual review pending",
        "root_decisions": update.decisions,
        "pixels": pixels,
        "ink_bounds_px": bounds,
        "A_provenance_summary": fallback_summary(baseline),
        "B_provenance_summary": fallback_summary(candidate),
        "formula_crops": len(crops),
        "semantic_scope": (
            "source revisions independently checked; trace equality is not a semantic proof"
        ),
        "english_scope": (
            "same-source A/B English pixels exact; fraction revision can reflow original prose"
        ),
    }
    atomic_json(
        destination / f"reports/{case_id}-provenance.json",
        {
            "A": baseline,
            "B": candidate,
            "parser_structure_trace": trace,
            "root_decisions": update.decisions,
        },
    )
    return report, panels, a_path, b_path


def paired_panel(a, b, destination, title, labels):
    width = max(a.width, b.width, 300) + 24
    panel = Image.new("RGB", (width * 2, max(a.height, b.height) + 64), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), title, fill="black")
    for index, (picture, label) in enumerate(zip((a, b), labels, strict=True)):
        draw.text((width * index + 8, 27), label, fill="black")
        panel.paste(picture, (width * index + 8, 56))
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel.save(destination, dpi=(300, 300))


def source_revision_crops(previous, case_id, number, destination, final_image):
    """Match the original/revised parser groups in order; crop their own bounds."""
    old = read(previous / f"reports/{case_id}-provenance.json")
    new = read(destination / f"reports/{case_id}-provenance.json")
    groups = []
    for data in (old, new):
        trace = data["parser_structure_trace"]
        groups.append(
            sorted(
                [("formula", r) for r in trace["formulas"]]
                + [("matrix", r) for r in trace["matrices"]],
                key=lambda r: r[1]["item_range"],
            )
        )
    if len(groups[0]) != len(groups[1]):
        raise ValueError("Source revision changed parser group cardinality")
    old_image = Image.open(previous / f"rendered/pages/candidate/{case_id}.png").convert("RGB")
    new_image = Image.open(final_image).convert("RGB")
    inventory = []
    for index, ((kind, a), (new_kind, b)) in enumerate(zip(*groups, strict=True), 1):
        if kind != new_kind:
            raise ValueError("Source revision reordered parser group kinds")
        crops, boxes = [], []
        for group, records, image in ((a, old["candidate"], old_image), (b, new["B"], new_image)):
            start, stop = group["item_range"]
            bounds = np.asarray([r["bounds_px"] for r in records[start:stop]])
            box = (
                *np.floor(bounds[:, :2].min(axis=0) - 12).astype(int),
                *np.ceil(bounds[:, 2:].max(axis=0) + 12).astype(int),
            )
            boxes.append(list(map(int, box)))
            crops.append(image.crop(box))
        output = destination / f"comparisons/original-to-revised/{case_id}/{index:03d}-{kind}.png"
        paired_panel(
            *crops,
            output,
            f"Original page {number} / {kind} {index}",
            ("Original exposed v26 page", "Revised fractions and root clearance"),
        )
        inventory.append(
            {
                "kind": kind,
                "index": index,
                "original_parser": a,
                "revised_parser": b,
                "crop_boxes": boxes,
                "output": str(output.relative_to(destination)),
            }
        )
    atomic_json(destination / f"comparisons/original-to-revised/{case_id}/index.json", inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages",
        default="12,14,17,18",
        help="Original aggregate page numbers; exposed regression only",
    )
    parser.add_argument("--stress", action="store_true", help="Also exercise tall and nested roots")
    args = parser.parse_args()
    selected = {int(n) for n in args.pages.split(",")}
    root = workspace_root()
    aggregate, frozen, legacy = (root / "runs" / key for key in (AGGREGATE, BASELINE, LEGACY))
    for source in (aggregate, frozen, legacy):
        verify_run(source)
    cfg = RenderConfig.model_validate(read(frozen / "run.json")["config"]["render_config"])
    protected = {
        str(root / "latest.json"): sha256(root / "latest.json"),
        **{str(p / "run.json"): sha256(p / "run.json") for p in (aggregate, frozen, legacy)},
    }
    run = Run.create(
        "math-author-feedback-clearance-fractions",
        config={
            "baseline_aggregate": str(aggregate),
            "baseline_math": BASELINE,
            "render_config": cfg.model_dump(mode="json"),
            "selected_pages": sorted(selected),
            "fraction_revision": "exposed-m1-stacked-fractions/1",
            "new_neural_updates": 0,
            "hypothesis": (
                "visible-ink clearance removes unused advance and preserves radicand geometry"
            ),
            "author_feedback": "long/high radical left gap; all mathematical divisions stacked",
            "independent_holdout": False,
            "publication": "held; no default/latest changes",
        },
    )
    print(run.path, flush=True)
    try:
        atomic_json(run.path / "reports/protected-assets.json", protected)
        for path in [*source_paths(), Path(__file__)]:
            path = path.resolve()
            target = run.path / "logs/source" / path.relative_to(Path.cwd())
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(frozen / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for name, source in (("session-glyphs", frozen), ("historical-glyphs", legacy)):
            shutil.copytree(source / f"solutions/{name}", run.path / f"solutions/{name}")
        profile = load_snapshot(run.path / "solutions/profile-v1")
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        scans = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        summaries, all_panels, transformations, paths = [], [], [], {"A": [], "B": []}
        for row in read(aggregate / "reports/cases.json"):
            number = row.get("output_page")
            if number is None:
                continue
            case_id = row["id"]
            previous = Path(row["source_run"])
            original = (aggregate / f"solutions/{case_id}/answer.md").read_text()
            revised, edits = rewrite_fractions(case_id, original)
            dest = run.path / f"solutions/inputs/{case_id}"
            dest.mkdir(parents=True)
            (dest / "original.md").write_text(original)
            (dest / "solution.md").write_text(revised)
            shutil.copyfile(aggregate / f"solutions/{case_id}/problem.txt", dest / "problem.txt")
            template = Job.model_validate_json(
                (previous / f"solutions/inputs/{case_id}/job.json").read_text()
            )
            actual = template.model_copy(
                update={
                    "realism": "baseline",
                    "render": cfg,
                    "profile_sha256": profile.manifest_sha256,
                    "generated_by": (
                        "exposed-m1-stacked-fractions/1; recorded equivalent math-only revisions"
                    ),
                }
            )
            atomic_json(dest / "job.json", actual.model_dump(mode="json"))
            bg = actual.background
            if bg.path:
                shutil.copyfile(previous / f"solutions/inputs/{case_id}" / bg.path, dest / bg.path)
            transformations.append(
                {
                    "case_id": case_id,
                    "original_page": number,
                    "original_source_run": str(previous),
                    "original_sha256": sha256(dest / "original.md"),
                    "revised_sha256": sha256(dest / "solution.md"),
                    "revisions": edits,
                    "prose_unchanged": True,
                    "remaining_unit_slashes": [m[0] for m in MATH.finditer(revised) if "/" in m[0]],
                }
            )
            if number not in selected:
                continue
            effective = cfg.model_copy(update={"paper": "plain" if bg.path else bg.kind})
            report, panels, a, b = render_pair(
                revised,
                case_id,
                number,
                run.path,
                effective,
                profile,
                resolver,
                sessions,
                polish,
                bg,
                dest / bg.path if bg.path else None,
            )
            shutil.copyfile(
                previous / f"rendered/pages/candidate/{case_id}.pdf", dest / "original-v26.pdf"
            )
            source_revision_crops(previous, case_id, number, run.path, b.with_suffix(".png"))
            summaries.append(report)
            all_panels.extend(panels)
            paths["A"].append(a)
            paths["B"].append(b)
        if args.stress:
            source = r"""The same expressions test short, long, high and nested roots.

$$\sqrt{3},\quad\sqrt{LC},\quad\sqrt{1+x+x^2+x^3+x^4}$$

$$\sqrt{\frac{7}{15}},\quad\sqrt{1+\sqrt{3}},\quad\sqrt{\frac{1+x^2}{1+\sqrt{2}}}$$

$$\sqrt{\omega_0^2-\alpha^2}=25\sqrt{7},\quad\sqrt{\frac{1}{\sqrt{3}}}$$
"""
            (run.path / "solutions/radical-stress.md").write_text(source)
            report, panels, a, b = render_pair(
                source,
                "exposed-radical-stress",
                21,
                run.path,
                cfg,
                profile,
                resolver,
                sessions,
                polish,
                Background(kind="plain"),
                None,
            )
            summaries.append(report)
            all_panels.extend(panels)
            paths["A"].append(a)
            paths["B"].append(b)
        for arm, files in paths.items():
            with PdfWriter() as writer:
                for file in files:
                    writer.append(file)
                writer.write(run.path / f"rendered/{arm}-homework.pdf")
        if all_panels:
            local_comparison(
                run.path / "comparisons",
                all_panels,
                labels=("A v26, revised fractions", "B root clearance candidate"),
            )
        atomic_json(run.path / "reports/source-transformations.json", transformations)
        atomic_json(run.path / "reports/cases.json", summaries)
        assert all(sha256(Path(p)) == digest for p, digest in protected.items())
        run.finish(
            "held",
            result={
                "pages": len(summaries),
                "fraction_revisions": sum(len(r["revisions"]) for r in transformations),
                "root_compactions": sum(
                    sum(d["applied"] for d in r["root_decisions"]) for r in summaries
                ),
                "changed_english_pixels_same_source": sum(
                    r["pixels"]["changed_english_pixels"] for r in summaries
                ),
                "neural_updates": 0,
                "independent_holdout": False,
                "author_style_acceptance": False,
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
