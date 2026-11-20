"""Source-only Phi body measurements; immutable O-FULL ink is never edited.

The three existing accepted crops stay authoritative. Manually inspected bowl
landmarks are distinct from the long central stem and from the printed prompt.
No new glyph, layout candidate, model or approval is produced by this audit.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus.math_glyph_resolver import ScanGlyph
from research.corpus.math_session_glyphs import outline_bounds

ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT / "workspace/runs/20261120T134632687261Z-session-o-reviewed-math-complete-20-adf8f1c8"
)
SOURCE_RUN = (
    ROOT / "workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738"
)
ROW_RUN = ROOT / "workspace/runs/20261120T134345155374Z-session-o-full-row-source-index-6886ffcf"
SOURCE_SHA = "23a84ff4cf5dc4882b5fb76d5eb4cc3a32c59e9799ea7a351f545598401f4d43"
# Independent manual observation of the real RGB bowl boundary, in each accepted
# crop's pixel coordinates. Half-open pixel envelopes; uncertainty +/-1 px.
BODY_BOXES = [[5, 29, 77, 94], [5, 30, 69, 103], [5, 29, 83, 107]]
OMEGA_PAGE_BOXES = [
    [704, 2621, 791, 2696],
    [1576, 2604, 1684, 2688],
    [2207, 2603, 2320, 2694],
]


def bounds(mask: np.ndarray) -> list[int]:
    y, x = np.where(mask)
    if not len(x):
        raise ValueError("Empty source ink")
    return [int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)]


def run_audit() -> Path:
    bank = BASELINE / "solutions/session-o-bank"
    all_sources = json.loads((bank / "sources.json").read_text())
    sources = sorted((r for r in all_sources if r["character"] == "Φ"), key=lambda r: r["id"])
    if len(sources) != 3:
        raise ValueError("Expected precisely the three frozen P02R06 Phi sources")
    source_sha = sources[0]["source_sha256"]
    original = SOURCE_RUN / "solutions/session-O_full_source.pdf"
    if sha256(original) != source_sha or source_sha != SOURCE_SHA:
        raise ValueError("Original O-FULL PDF changed")
    page_path = SOURCE_RUN / "solutions/pages/page-02.png"
    if sha256(page_path) != sources[0]["source_page"]["rgb_page_sha256"]:
        raise ValueError("Original page changed")
    run = Run.create(
        "session-o-phi-body-source-audit",
        config={
            "purpose": "Measure existing authentic Phi body vs long stem; no rendering integration",
            "baseline": str(BASELINE),
            "baseline_run_sha256": sha256(BASELINE / "run.json"),
            "original_source_sha256": source_sha,
            "accepted_source_manifest_sha256": sha256(bank / "sources.json"),
            "scope": "Only new O-FULL; no corpus, historical documents or holdout access",
            "body_measurement": (
                "Manual RGB inspection, recorded half-open pixel envelope "
                "at existing L160 contour level"
            ),
            "production_or_source_modifications": False,
            "neural_updates": 0,
        },
        seed=0,
    )
    path = run.path
    print(path, flush=True)
    for directory in ("logs/source", "solutions/crops", "solutions/parents"):
        (path / directory).mkdir()
    shutil.copyfile(__file__, path / "logs/source/session_o_phi_audit.py")
    shutil.copyfile(bank / "sources.json", path / "solutions/accepted-bank-original-manifest.json")
    shutil.copyfile(original, path / "solutions/session-O_full_source.pdf")
    shutil.copyfile(page_path, path / "solutions/page-02-rgb.png")
    parent = ROW_RUN / "solutions/rows/O-FULL-P02-R06/context-rgb.png"
    shutil.copyfile(parent, path / "solutions/parents/P02R06-context-rgb.png")
    page = Image.open(page_path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    panel = Image.new("RGB", (1460, 1160), "white")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (20, 12),
        "Authentic Phi / adjacent Omega: original RGB at 2x, separate annotated duplicate",
        font=font,
        fill="black",
    )
    draw.text(
        (20, 40),
        "Blue = observed bowl body; all stem ink retained. "
        "Annotation does not modify source crops.",
        font=font,
        fill="black",
    )
    records = []
    for index, (source, body, omega_box) in enumerate(
        zip(sources, BODY_BOXES, OMEGA_PAGE_BOXES, strict=True)
    ):
        raw_path = bank / source["crop"]
        if sha256(raw_path) != source["crop_sha256"]:
            raise ValueError("Accepted Phi crop changed")
        raw = Image.open(raw_path).convert("RGB")
        if raw.tobytes() != page.crop(source["box"]).tobytes():
            raise ValueError("Accepted Phi crop no longer equals original page region")
        output = path / "solutions/crops" / raw_path.name
        shutil.copyfile(raw_path, output)
        omega = page.crop(omega_box)
        omega_out = path / f"solutions/crops/adjacent-omega-V{index + 1:02d}.png"
        omega.save(omega_out, dpi=(300, 300))
        phi_mask = np.asarray(raw.convert("L")) < source["contour_luminance"]
        omega_mask = np.asarray(omega.convert("L")) < source["contour_luminance"]
        if any(
            edge.any()
            for edge in (omega_mask[0], omega_mask[-1], omega_mask[:, 0], omega_mask[:, -1])
        ):
            raise ValueError("Adjacent Omega evidence has ink touching its crop edge")
        if source["row_id"] != "O-FULL-P02-R06" or source["contour_luminance"] != 160:
            raise ValueError("Body measurements do not describe this source or threshold")
        full = bounds(phi_mask)
        omega_ink = bounds(omega_mask)
        bowl_h, bowl_w = body[3] - body[1], body[2] - body[0]
        full_h = full[3] - full[1]
        omega_h = omega_ink[3] - omega_ink[1]
        glyph = ScanGlyph.load(bank, source)
        contour_box = outline_bounds(glyph.outline.verts, glyph.outline.codes).ravel().tolist()
        body_page = (np.array(body) + np.array(source["box"][:2] * 2)).tolist()
        matrix = np.asarray(source["raster_to_pdf"])
        crop_to_pdf = matrix @ np.array([[1, 0, omega_box[0]], [0, 1, omega_box[1]], [0, 0, 1]])
        first, last = (
            matrix @ [body_page[0], body_page[1], 1],
            matrix @ [body_page[2], body_page[3], 1],
        )
        widths = []
        for y in range(body[1] + bowl_h // 3, body[1] + 2 * bowl_h // 3):
            columns = np.flatnonzero(phi_mask[y])
            groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
            if len(groups) >= 3:
                widths.append(len(groups[0]))
        record = {
            "source_id": source["id"],
            "source_record": source,
            "source_crop": str(output.relative_to(path)),
            "source_crop_sha256": sha256(output),
            "full_ink_bbox_crop_px_L160": full,
            "full_contour_bbox_crop_px_L160": contour_box,
            "body_bounds_crop_px": body,
            "body_bounds_page_px": body_page,
            "body_bounds_pdf_points": [
                float(first[0]),
                float(last[1]),
                float(last[0]),
                float(first[1]),
            ],
            "body_width_px": bowl_w,
            "body_height_px": bowl_h,
            "full_ink_height_px": full_h,
            "full_to_body_height_ratio": full_h / bowl_h,
            "measurement_status": "Agent observed source landmarks; placement approval separate",
            "body_boundary_uncertainty_px": 1,
            "body_envelope_is_not_a_crop_or_ink_removal": True,
            "same_row_omega": {
                "role": (
                    "Actual handwritten adjacent capital Omega in same repetition; "
                    "geometry comparison only"
                ),
                "crop": str(omega_out.relative_to(path)),
                "sha256": sha256(omega_out),
                "box_page_px": omega_box,
                "crop_to_pdf": crop_to_pdf.tolist(),
                "ink_bbox_crop_px_L160": omega_ink,
                "ink_height_px": omega_h,
                "phi_body_to_omega_height_ratio": bowl_h / omega_h,
                "training_or_rendering_approved": False,
            },
            "left_bowl_horizontal_scanline_widths": {
                "method": (
                    "First disconnected ink run in middle third body rows having "
                    ">=3 separated runs; slant-sensitive, not physical pen diameter"
                ),
                "rows": len(widths),
                "median_px": float(np.median(widths)) if widths else None,
                "projected_median_at_full_height_30": float(np.median(widths)) * 30 / full_h
                if widths
                else None,
                "projected_median_at_body_height_30": float(np.median(widths)) * 30 / bowl_h
                if widths
                else None,
            },
            "fixed_body_height_30_diagnostic": {
                "uniform_scale": 30 / bowl_h,
                "full_ink_height_px": full_h * 30 / bowl_h,
                "body_width_px": bowl_w * 30 / bowl_h,
                "stem_above_body_px": (body[1] - full[1]) * 30 / bowl_h,
                "stem_below_body_px": (full[3] - body[3]) * 30 / bowl_h,
                "not_an_output_acceptance": True,
            },
        }
        records.append(record)
        y = 90 + index * 350
        draw.text(
            (20, y),
            f"V{index + 1:02d}: Phi full {full_h}, bowl {bowl_h} x {bowl_w}; "
            f"same-row Omega height {omega_h} px",
            font=font,
            fill="black",
        )
        raw_2x = raw.resize((raw.width * 2, raw.height * 2), Image.Resampling.NEAREST)
        panel.paste(raw_2x, (30, y + 35))
        annotated = raw_2x.copy()
        mark = ImageDraw.Draw(annotated)
        mark.rectangle(tuple(v * 2 for v in body), outline=(20, 100, 240), width=2)
        panel.paste(annotated, (240, y + 35))
        panel.paste(
            omega.resize((omega.width * 2, omega.height * 2), Image.Resampling.NEAREST),
            (470, y + 35),
        )
        draw.text((760, y + 35), f"body crop bbox {body}; +/- 1 px", font=font, fill="black")
        draw.text(
            (760, y + 65),
            f"full/body = {full_h / bowl_h:.3f}; body/Omega = {bowl_h / omega_h:.3f}",
            font=font,
            fill="black",
        )
        draw.text(
            (760, y + 95),
            f"body -> 30 px: full height {full_h * 30 / bowl_h:.2f} px",
            font=font,
            fill="black",
        )
        draw.text(
            (760, y + 125),
            "No clipping, thinning, dilation or stroke redraw.",
            font=font,
            fill="black",
        )
    panel.save(path / "comparisons/phi-body-and-omega.png")
    panel.save(path / "comparisons/phi-body-and-omega.pdf", "PDF", resolution=150)
    atomic_json(path / "reports/body-measurements.json", records)
    rule = {
        "purpose": (
            "Candidate generic placement contract, not integrated or accepted in this source audit"
        ),
        "applies_to": "Only reviewed capital Phi from these three original variants",
        "scale": "target MathText cap slot height / observed source bowl body height",
        "translation": "target cap slot center - source bowl body center * uniform scale",
        "geometry": (
            "Apply same scalar and translation to EVERY original contour vertex, including stem"
        ),
        "gates_required": [
            "Complete glyph and same topology",
            "Page and neighbor contacts",
            "Stem and width protrusions into actual reserved free space",
            "Non-target pixels and provenance unchanged",
        ],
        "failure": (
            "Retain prior item and record explicit rejection; never clip stem to old font bbox"
        ),
        "source_variant_selection": (
            "Preserve frozen deterministic selection, no favorable-seed selection"
        ),
        "no_stroke_redraw_or_morphology": True,
    }
    atomic_json(path / "reports/proposed-placement-rule.json", rule)
    result = {
        "existing_accepted_phi_sources_measured": 3,
        "same_row_actual_omega_comparators": 3,
        "new_glyphs_accepted": 0,
        "source_or_bank_changed": False,
        "rendering_executed": False,
        "diagnosis": (
            "Entire stem envelope is about 1.82x bowl height. Bowl matches adjacent natural "
            "Omega height within 2.7%; full-envelope-to-cap fitting compresses the body "
            "and all stroke widths together."
        ),
        "next_step": (
            "Root may independently inspect source landmarks and test body-fit placement "
            "with complete contour and contact gates"
        ),
    }
    atomic_json(path / "reports/result.json", result)
    (path / "reports/README.md").write_text(
        "# O-FULL Phi body source audit\n\n"
        "Three already accepted P02R06 samples are measured without replacing any source. "
        "Blue annotations exist only on duplicate review images. Raw crops are exact copies.\n\n"
        "The observed bowl heights are 65 / 73 / 78 px; adjacent real Omega heights "
        "are 64 / 72 / 76 px. Full Phi ink heights are 118 / 133 / 142 px. "
        "These are source facts, not output-tuned values. "
        "Body landmarks are half-open pixel envelopes with +/-1 px inspection uncertainty.\n\n"
        "P02R07 and P03R02 contexts were also visually inspected as alternative-source "
        "locations before this audit; no smaller alternative source was admitted. "
        "The task was narrowed to the three existing sources after the body/stem mismatch "
        "was identified. No historical PDF, corpus or sealed source was accessed.\n\n"
        "The proposed rule scales the complete source uniformly and centers the bowl "
        "in the existing cap slot. Its free stem and lateral extent must pass actual "
        "page/neighbor checks. This run produces no candidate homework "
        "and gives no visual admission for new placement.\n"
    )
    run.finish("prepared", result=result)
    verify_run(path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return path


if __name__ == "__main__":
    run_audit()
