# ruff: noqa: E501
"""Reproduce the v0.10.24 math-only A/B and privately archive its evidence."""

import argparse
import math
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.layout import ink_geometry
from hwsol.jobs.paper import pdf_image
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import profile_store, verify_run, workspace_root
from research.corpus.context_composer import layout
from research.corpus.experiment import create
from research.corpus.math_glyph_evidence import (
    benchmark_inventory,
    collection_sheet,
    extract_sources,
    fallback_overview,
    read,
    source_counts,
)
from research.corpus.math_glyph_resolver import (
    MathGlyphResolver,
    ScanGlyph,
    math_role,
    matrix_strokes,
)
from research.corpus.math_glyph_semantics import check_reference_math
from research.corpus.mixed_ink_pages import label_pdf

BASELINE_ID = "20261120T042107038172Z-measured-word-rhythm-mixed-homework-3be93094"
BENCHMARK_ID = "v06-benchmark-v1-final"
SELECTION = set("λ∂βαρ∑∞[]∩≥→′πτεδ:()") | {"fraction-bar", "radical-overbar"}

ADDITIONAL = {
    "Matrix and weighted mean": r"""Find the determinant and characteristic roots of the matrix.

$$A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$$

$$\mathrm{det}(A-\lambda I)=(2-\lambda)^2-1$$

The positive square root of one is one, so the roots are one and three.

$$(2-\lambda)^2=1,\qquad \lambda_1=1,\quad \lambda_2=3$$

Now let the two observations be two and four, with weights one and three.

$$x_1=2,\quad x_2=4,\qquad w_1=1,\quad w_2=3$$

The weighted mean is the ratio of the two finite sums.

$$\hat\mu=\frac{\sum_{i=1}^{2}w_i x_i}{\sum_{i=1}^{2}w_i}=\frac{14}{4}=\frac{7}{2}$$

Both signs solve the next equation. Evaluate the radicals as written.

$$z^2=3,\qquad z=\pm\sqrt{3},\qquad r=\sqrt{\frac{7}{15}},\qquad u=\sqrt{1+\sqrt{3}}$$
""",
    "Derivatives and Greek parameters": r"""Consider a quadratic with independent parameters.

$$F(\alpha,\beta)=\alpha^2+\rho\alpha\beta+\beta^2,\qquad \frac{\partial F}{\partial \alpha}=2\alpha+\rho\beta$$

An independent time variable gives the following derivative.

$$g(t)=\sin(\omega t+\varphi),\qquad g'(t)=\omega\cos(\omega t+\varphi)$$

With positive time scale, this exponential decays and its integral equals the scale.

$$\tau>0,\qquad \lim_{t\to\infty}e^{-t/\tau}=0,\qquad \int_0^{\infty} e^{-t/\tau}\,dt=\tau$$

The trigonometric limit approaches one. The two intervals share one point.

$$\lim_{\theta\to 0}\frac{\sin\theta}{\theta}=1,\qquad [0,1]\cap[1,2]=\{1\}$$

Define the vector and the perturbed coefficient. The last two relations hold for real parameters.

$$\vec{v}=\begin{pmatrix}1\\2\end{pmatrix},\qquad \hat\beta_1=\beta_1+\varepsilon_1,\qquad \delta_1^2\geq 0,\qquad \sin(\pi)=0$$
""",
}

PRIORITY = [
    (r"\omega\quad \omega_d\quad \sin(\omega t)", "Angular frequency"),
    (r"\vec{v}\quad \vec{x}\quad \vec{F}", "Vector arrows above letters"),
    (r"\hat{\beta}_1\quad \hat{y}\quad \hat{\theta}", "Hats above letters"),
    (r"x=\pm\sqrt{3}\qquad 3\pm t", "Plus-minus"),
    (r"\sqrt{\frac{7}{15}}\qquad\sqrt{1+\sqrt{3}}", "Tall and nested radicals"),
    (
        r"\left[\frac{1}{2}+x\right]\qquad \left[\frac{1}{2}\right]_0^1",
        "Tall brackets with a fraction",
    ),
]
OPTIONAL = [
    (r"\mu\quad\theta\quad\varphi\quad\chi\quad\Delta", "Greek forms seen in the comparisons"),
    (r"\gamma\quad\phi\quad\epsilon\quad\Omega\quad\nu", "Additional Greek forms if you use them"),
    (r"\nabla f\quad\nabla\cdot\vec{F}\quad\sin\theta", "Nabla and vector context"),
    (r"|x|\quad\left|\frac{1}{2}\right|\quad 2\times3\quad x<1", "Absolute value and operators"),
    (r"\bar{x}\quad\overline{AB}\quad\ddot{x}", "Accents and lengths"),
    (r"\left[\;\;\;\right]\qquad(\;\;\;)\qquad|\;\;\;|", "Tall pairs as you write a matrix"),
    (r"\{x\mid x>0\}\qquad x\geq 0", "Set braces and inequality"),
]


def raster(blob: bytes | Path) -> Image.Image:
    with pdfium.PdfDocument(blob) as document:
        if len(document) != 1:
            raise ValueError("Expected a single page for proof rasterization")
        return pdf_image(document, 0, 300 / 72)


def frozen_english(items, words: list[dict]) -> tuple[list, list[dict]]:
    replacements, removed, boxes = {}, set(), []
    for word in words:
        if word["fallback_reason"]:
            continue
        alpha = Path(word["alpha"])
        if sha256(alpha) != word["sha256"]:
            raise ValueError("Frozen English ink asset changed")
        with Image.open(alpha) as image:
            rgba = Image.new("RGBA", image.size)
            rgba.putalpha(image.convert("L"))
        ids = word["items"]
        x, y = word["origin"]
        x += word["word_shift_x_px"]
        replacements[ids[0]] = ("image", rgba, (1, 0, 0, 1, x, y))
        removed.update(ids)
        a, b, c, d = rgba.getbbox()
        boxes.append({"word_id": word["word_id"], "box": [a + x, b + y, c + x, d + y]})
    return [
        replacements.get(i, item)
        for i, item in enumerate(items)
        if i not in removed or i in replacements
    ], boxes


def summarize_gaps(rows: list[dict], covered: set[str]) -> list[dict]:
    priorities = {
        # (visual salience 1..3, replacement risk 1..3)
        "λ": (3, 1),
        "ω": (3, 2),
        "{": (2, 3),
        "∑": (3, 2),
        "∞": (2, 1),
        "⃗": (3, 3),
        "̂": (3, 3),
        "′": (2, 2),
        "β": (3, 2),
        "∂": (3, 1),
        "→": (2, 2),
        "√": (3, 3),
        "(": (2, 2),
        ")": (2, 2),
        "radical-overbar": (2, 3),
        "fraction-bar": (2, 2),
        "geometry": (2, 3),
        "[": (2, 3),
        "]": (2, 3),
        "±": (2, 1),
        ":": (1, 1),
    }
    groups = Counter(
        (r["semantic"], r["source_type"], r["structure"], r["audit_reason"])
        for r in rows
        if not r["personal_collected"]
    )
    ranking = []
    for (char, source, structure, reason), count in groups.items():
        salience, risk = priorities.get(char, (2, 2))
        ranking.append(
            {
                "semantic": char,
                "baseline_source": source,
                "structure": structure,
                "count": count,
                "visual_salience": salience,
                "replacement_risk": risk,
                "priority_score": count * salience / risk,
                "audit_reason": reason,
                "historical_source_reviewed": char in covered,
            }
        )
    return sorted(ranking, key=lambda row: (-row["priority_score"], str(row["semantic"])))


def baseline_audit(page, renderer, glyphs, collected: set[str], candidate: set[str]) -> list[dict]:
    mapping = matrix_strokes(page, renderer)
    blocked = {a["item_index"]: a["reason"] for a in getattr(renderer, "structure_fallbacks", ())}
    result = []
    for row in glyphs:
        if row["context"] != "mathematics":
            continue
        index = row["item_index"]
        char = row["semantic"]
        route = mapping.get(index)
        if route:
            char = route[0]
        if row["personal_collected"]:
            reason = "personal contour used"
        elif index in blocked:
            reason = "Profile contour available; " + blocked[index]
        elif route and char in collected:
            reason = "Profile contour available; inline matrix uses recorded program strokes"
        elif char == "fraction-bar":
            reason = "measured rule exists in personal@v3; v0.10.23 uses personal@v1"
        elif char == "radical-overbar":
            reason = "dynamic radical join uses parser geometry; there is no validated personal fit"
        elif char in candidate:
            reason = "absent from Profile; reviewed original handwriting located"
        elif char in collected:
            reason = "Profile contour present but selected MathText route did not use it"
        elif row["procedural_geometry"]:
            reason = "parser structure; no validated personal fit"
        else:
            reason = "absent from Profile; no clean matching original found in the reviewed sources"
        result.append(
            {
                **row,
                "audit_reason": reason,
                "audit_semantic": char,
                "audit_structure": "matrix-bracket" if route else math_role(row),
                "audit_matrix_environment": route[1] if route else row["matrix_environment"],
            }
        )
    return result


def overlap_verification(
    before: Image.Image, after: Image.Image, changes: list[dict], protected: list[list[float]]
) -> dict:
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape:
        raise ValueError("Page pixel dimensions changed")
    diff = np.any(a != b, axis=-1)
    allowed = np.zeros(diff.shape, dtype=bool)

    def rectangle(box, padding):
        x, y, right, bottom = box
        return (
            slice(max(0, math.floor(y) - padding), math.ceil(bottom) + padding),
            slice(max(0, math.floor(x) - padding), math.ceil(right) + padding),
        )

    for change in changes:
        box = change["allowed_bounds_px"]
        x, y, right, bottom = change["bounds_px"]
        if (
            min(x - box[0], y - box[1], box[2] - right, box[3] - bottom) < -1e-6
            or min(x, y, a.shape[1] - right, a.shape[0] - bottom) < 3
        ):
            raise ValueError(f"Math ink exceeds original safe slot: {change['item_index']}")
        slots = (
            (change["original_bounds_px"], change["bounds_px"])
            if "paired_items" in change
            else (box,)
        )
        if not any(diff[rectangle(slot, 2)].any() for slot in slots):
            raise ValueError(f"Math replacement did not affect visible ink: {change['item_index']}")
        for slot in slots:
            allowed[rectangle(slot, 2)] = True
    if np.any(diff & ~allowed):
        raise ValueError("Candidate modified a pixel outside the allowed math slots")
    if any(diff[rectangle(box, 1)].any() for box in protected):
        raise ValueError("Candidate overlaps English or its bounding box")
    return {
        "changed_pixels": int(diff.sum()),
        "pixels_changed_outside_math_slots": 0,
        "english_boxes_checked": len(protected),
        "changed_english_pixels": 0,
    }


def local_comparison(
    output: Path,
    panels: list[tuple],
    labels: tuple[str, str] = ("A   v0.10.23", "B   Personal math candidate"),
) -> None:
    """Print all formula regions with replacements at the same scale for A and B."""
    with PdfPages(output / "formula-regions.pdf") as pdf:
        for start in range(0, len(panels), 6):
            part = panels[start : start + 6]
            sheet = Image.new("RGB", (2490, 3508), "white")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.truetype("DejaVuSans.ttf", 31)
            small = ImageFont.truetype("DejaVuSans.ttf", 22)
            draw.text((92, 82), "Formula comparison", fill="#182530", font=font)
            for x, label in zip((92, 1290), labels, strict=True):
                draw.text((x, 148), label, fill="#49596b", font=font)
            for j, (page, group, a, b) in enumerate(part):
                y = 239 + j * 515
                draw.line((92, y - 14, 2398, y - 14), fill="#d6dee6", width=2)
                draw.text(
                    (92, y), f"Page {page} / formula group {group}", fill="#444e5b", font=small
                )
                for x, image in [(92, a), (1290, b)]:
                    factor = min(1.52, 1103 / image.width, 407 / image.height)
                    size = tuple(round(z * factor) for z in image.size)
                    sheet.paste(image.resize(size, Image.Resampling.LANCZOS), (x, y + 51))
            sheet.save(output / f"formula-regions-{start // 6 + 1:02d}.png", dpi=(300, 300))
            fig = Figure(figsize=(8.30, 11.69))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(sheet, interpolation="none")
            ax.set_axis_off()
            pdf.savefig(fig, dpi=300)


def run_experiment(selection: Path, *, baseline: Path, compare_only: bool = False) -> Path:
    root = workspace_root()
    historical_benchmark = root / "runs" / BENCHMARK_ID
    verify_run(baseline)
    verify_run(historical_benchmark)
    request = read(baseline / "reports/request.json")
    input_run = Path(request["source"])
    verify_run(input_run)
    mixed = Path(request["mixed"])
    verify_run(mixed)
    model = Path(read(mixed / "reports/request.json")["model"])
    model_receipt = read(model / "run.json")
    cfg = RenderConfig.model_validate(read(input_run / "run.json")["config"]["render_config"])
    original_pdf = baseline / "rendered/homework_solutions.pdf"
    original = read(baseline / "reports/page-provenance.json")
    corpus = root / "datasets/personal-homework/corpus.sqlite"
    snapshots = [profile_store().resolve(f"personal@v{v}", candidate=v > 3) for v in range(1, 6)]
    protected = [
        root / "latest.json",
        corpus,
        original_pdf,
        baseline / "run.json",
        input_run / "run.json",
        mixed / "run.json",
        model / "run.json",
        profile_store().root / "settings.json",
    ]
    for loaded in snapshots:
        protected.extend(
            loaded.directory / p
            for p in ["profile.json", *(asset.path for asset in loaded.manifest.assets)]
        )
    protected.extend(model / asset for asset in model_receipt["artifacts"] if asset.endswith(".pt"))
    specs = read(selection)
    protected.extend({Path(row["pdf"]) for row in specs})
    protection = {str(p): sha256(p) for p in protected}
    run = create(
        "v01024-personal-math-glyph-audit"
        if compare_only
        else "v01024-personal-math-glyph-completion",
        seed=cfg.seed,
        config={
            "baseline": str(baseline),
            "baseline_sha256": sha256(baseline / "run.json"),
            "source_layout": str(input_run),
            "source_layout_sha256": sha256(input_run / "run.json"),
            "render_config": cfg.model_dump(mode="json"),
            "frozen_english_reference": request["mixed"],
            "frozen_english_reference_sha256": request["mixed_sha256"],
            "selection_sha256": sha256(selection),
            "selected_symbols_and_roles": sorted(SELECTION),
            "layout_profile": snapshots[0].provenance(),
            "measured_stroke_profile": snapshots[2].provenance(),
            "english_checkpoint_run": str(model),
            "english_seed": read(baseline / "run.json")["seed"],
            "C_opened": False,
            "torch_required": False,
        },
    )
    output = run.path
    print(output, flush=True)
    try:
        for v in (0, 2):
            copy_snapshot(snapshots[v], output / "solutions" / f"profile-v{v + 1}")
        profile = load_snapshot(output / "solutions/profile-v1")
        stroke_profile = load_snapshot(output / "solutions/profile-v3")
        shutil.copyfile(selection, output / "solutions/source-selection.json")
        bank = extract_sources(specs, output / "solutions/historical-glyphs", corpus)
        scan_glyphs = {
            entry["character"]: ScanGlyph.load(output / "solutions/historical-glyphs", entry)
            for entry in bank
            if entry["review_status"] == "agent-accepted"
        }
        resolver = MathGlyphResolver(scan_glyphs, stroke_profile, selected=SELECTION)
        input_pages = read(input_run / "reports/pages.json")
        pages = [(x["identity"], x["source"]) for x in input_pages]
        if not compare_only:
            pages.extend(ADDITIONAL.items())
        old_inventory, old_images, records, panels = [], [], [], []
        writers = {arm: PdfWriter() for arm in "AB"}
        (output / "rendered/pages/A").mkdir()
        (output / "rendered/pages/B").mkdir()
        all_chars = {chr(code) for s in snapshots for code in s.manifest.personal_codepoints}
        for number, (identity, source) in enumerate(pages, 1):
            (output / f"solutions/problem-{number:02d}.md").write_text(source, encoding="utf-8")
            page, renderer, glyphs = layout(source, identity, profile, cfg)
            audited = baseline_audit(page, renderer, glyphs, all_chars, set(scan_glyphs))
            if number <= 4:
                entry = original[number - 1]
                if entry["math_glyph_provenance"] != [
                    {k: v for k, v in r.items() if not k.startswith("audit_")} for r in audited
                ]:
                    raise ValueError(
                        "Current CPU baseline does not reproduce archived math provenance"
                    )
                old_inventory.append(audited)
                with Image.open(baseline / f"rendered/pages/page-{number:02d}.png") as saved:
                    old_images.append(saved.convert("RGB"))
                frozen = entry["word_provenance"]
                rhythm = entry["word_rhythm"]
            else:
                frozen, rhythm = [], []
            if compare_only:
                continue
            baseline_items, words = frozen_english(page.items, frozen)
            raw_a = renderer.to_vector_pdf([replace(page, items=baseline_items)])
            a_method = "Frozen neural English | Measured word spacing"
            if number <= 4:
                a_blob = (baseline / f"comparisons/neural-{number}.pdf").read_bytes()
                a_replay = raster(label_pdf(raw_a, a_method))
                if a_replay.tobytes() != old_images[-1].tobytes():
                    raise ValueError(f"Archived v0.10.23 page {number} did not replay exactly")
            else:
                a_method = "Handnote English | Extended mathematics exercise"
                a_blob = label_pdf(raw_a, a_method)
            structure_trace = {
                "formulas": renderer.formulas,
                "matrices": renderer.matrices,
            }
            original_items = list(page.items)
            new = resolver.resolve(page, renderer, glyphs, profile)
            candidate_items, new_words = frozen_english(page.items, frozen)
            if words != new_words:
                raise ValueError("English ink placement changed")
            mixed_page = replace(page, items=candidate_items)
            bounds, _ = ink_geometry(mixed_page, renderer)
            if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
                raise ValueError("Candidate crosses a page boundary")
            raw_b = renderer.to_vector_pdf([mixed_page])
            a, b = raster(raw_a), raster(raw_b)
            new_math = [r for r in new if r["context"] == "mathematics"]
            changes = [r for r in new_math if "original_source_type" in r]
            replaced = {r["item_index"] for r in changes}
            if any(
                index not in replaced and old is not current
                for index, (old, current) in enumerate(zip(original_items, page.items, strict=True))
            ):
                raise ValueError("Candidate altered an item outside the resolver decisions")
            if len(glyphs) != len(new) or any(
                old["latex"] != revised["latex"]
                or old["item_index"] != revised["item_index"]
                or old["context"] != revised["context"]
                or (
                    old["semantic"] != revised["semantic"]
                    and revised["structure"] != "matrix-bracket"
                )
                for old, revised in zip(glyphs, new, strict=True)
            ):
                raise ValueError("The visible semantic trace changed")
            neural_item_ids = {i for w in frozen if not w["fallback_reason"] for i in w["items"]}
            protected = [w["box"] for w in words] + [
                r["bounds_px"]
                for r in glyphs
                if r["context"] == "text" and r["item_index"] not in neural_item_ids
            ]
            pixel_proof = overlap_verification(a, b, changes, protected)
            for group, formula in enumerate(renderer.formulas, 1):
                start, end = formula["item_range"]
                if not replaced.intersection(range(start, end)):
                    continue
                local = np.asarray([r["bounds_px"] for r in glyphs[start:end]])
                low, high = local[:, :2].min(axis=0), local[:, 2:].max(axis=0)
                box = (*np.floor(low - 12).astype(int), *np.ceil(high + 12).astype(int))
                panels.append((number, group, a.crop(box), b.crop(box)))
            b_blob = label_pdf(raw_b, a_method, math_label="personal glyph candidate")
            for arm, blob in [("A", a_blob), ("B", b_blob)]:
                dest = output / f"rendered/pages/{arm}/page-{number:02d}.pdf"
                dest.write_bytes(blob)
                writers[arm].append(dest)
                raster(blob).save(dest.with_suffix(".png"), dpi=(300, 300))
            records.append(
                {
                    "page": number,
                    "identity": identity,
                    "source_sha256": sha256(output / f"solutions/problem-{number:02d}.md"),
                    "A": audited,
                    "B": new_math,
                    "parser_structure_trace": structure_trace,
                    "parser_item_order_and_content_equal": True,
                    "neural_words": len(words),
                    "word_provenance": frozen,
                    "word_rhythm": rhythm,
                    "english_placements_equal": True,
                    "A_matches_v01023_pixels": True if number <= 4 else None,
                    "candidate_vector_bounds_px": bounds,
                    **pixel_proof,
                }
            )
        all_audit = [row for page in old_inventory for row in page]
        ranked_rows = [
            {**r, "semantic": r["audit_semantic"], "structure": r["audit_structure"]}
            for r in all_audit
        ]
        benchmark = benchmark_inventory(historical_benchmark)
        atomic_json(output / "reports/benchmark-baseline-inventory.json", benchmark)
        gaps = summarize_gaps(ranked_rows, set(scan_glyphs))
        atomic_json(
            output / "reports/math-glyph-coverage.json",
            {
                "mixed_four_pages": old_inventory,
                "mixed_four_page_source_types": source_counts(all_audit),
                "ranked_gaps": gaps,
                "historical_benchmark_case_count": len(benchmark),
            },
        )
        fallback_overview(old_images, old_inventory, output / "comparisons")
        prompts = collection_sheet(output / "collection", "M", PRIORITY)
        prompts += collection_sheet(output / "collection", "N", OPTIONAL)
        atomic_json(output / "collection/prompts.json", prompts)
        if compare_only:
            run.finish(
                "held",
                result={"audited_math_items": len(all_audit), "reviewed_sources": len(scan_glyphs)},
            )
            return output
        for arm, writer in writers.items():
            writer.write(output / f"rendered/{arm}-homework.pdf")
            writer.close()
        with PdfWriter() as side_by_side:
            for number in range(1, len(records) + 1):
                for arm in "AB":
                    side_by_side.append(output / f"rendered/pages/{arm}/page-{number:02d}.pdf")
            side_by_side.write(output / "comparisons/A-B-homework.pdf")
        local_comparison(output / "comparisons", panels)
        atomic_json(output / "reports/glyph-provenance.json", records)
        atomic_json(
            output / "reports/semantic-verification.json",
            {
                "checks": check_reference_math(),
                "parser_glyph_order_preserved": True,
                "source": "SymPy checks of fixed input calculations plus per-item parser provenance",
                "independent_handwriting_transcription": None,
            },
        )
        if any(sha256(Path(p)) != digest for p, digest in protection.items()):
            raise ValueError("An upstream protected asset changed during the run")
        atomic_json(output / "reports/protected-assets.json", protection)
        result = {
            "sheets_per_arm": len(records),
            "exact_v01023_pages": 4,
            "neural_words_identical": sum(r["neural_words"] for r in records),
            "before_four_pages": source_counts([x for r in records[:4] for x in r["A"]]),
            "after_four_pages": source_counts([x for r in records[:4] for x in r["B"]]),
            "changed_items_four_pages": dict(
                Counter(
                    x["semantic"]
                    for r in records[:4]
                    for x in r["B"]
                    if "original_source_type" in x
                )
            ),
            "open_gaps_four_pages": dict(
                Counter(
                    str(x["semantic"])
                    for r in records[:4]
                    for x in r["B"]
                    if x.get("fallback_reason")
                )
            ),
            "after_extension": source_counts([x for r in records[4:] for x in r["B"]]),
            "reviewed_personal_scan_classes": len(scan_glyphs),
            "formula_region_pairs": len(panels),
            "torch_loaded_in_render_process": "torch" in sys.modules,
            "new_neural_updates": 0,
            "author_review": "pending",
            "release_decision": "no-go for release; held math-only candidate",
        }
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=workspace_root() / "runs" / BASELINE_ID)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    run_experiment(args.selection, baseline=args.baseline, compare_only=args.audit_only)
