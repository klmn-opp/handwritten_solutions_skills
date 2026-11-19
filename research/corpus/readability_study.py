"""Frozen v0109 output replay with opt-in spacing and radical repairs, CPU only."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.config import RenderConfig
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_composer import layout
from research.corpus.experiment import create
from research.corpus.ink_integrity import ink_from_reference, map_components
from research.corpus.ink_integrity_lines import render_line
from research.corpus.mixed_ink_pages import label_pdf, text_groups
from research.corpus.readability import compact_blank_columns, join_radicals, word_positions


def image_unit(ink, x, y, em, compact):
    yy, xx = np.where(ink > 0)
    if not len(xx):
        raise ValueError("Empty source word")
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    crop = ink[y0:y1, x0:x1]
    trimmed, report = compact_blank_columns(crop, em) if compact else (crop, {})
    report.update(
        original_width=int(crop.shape[1]),
        compacted_width=int(trimmed.shape[1]),
        nonzero_pixels_before=int(np.count_nonzero(crop)),
        nonzero_pixels_after=int(np.count_nonzero(trimmed)),
        image_offset=[0, 0] if compact else [-int(x0), -int(y0)],
    )
    # Preserve the original full padded image when the switch is off; trimming
    # changes PDF image interpolation at subpixel page coordinates.
    drawn = trimmed if compact else ink
    rgba = np.zeros((*drawn.shape, 4), np.uint8)
    rgba[:, :, 3] = np.uint8(drawn * 255)
    return (
        Image.fromarray(rgba),
        [float(x + x0), float(y + y0), float(x + x0 + trimmed.shape[1]), float(y + y1)],
        report,
    )


def repair_page(
    info, profile, cfg, words, controls, old_records, *, spacing=True, radicals=True, snug=False
):
    from hwsol.jobs.provenance import item_bounds

    page, renderer, provenance = layout(info["source"], info["identity"], profile, cfg)
    if provenance != info["provenance"]:
        raise ValueError("Frozen source no longer replays")
    original_items = list(page.items)
    radical_report = join_radicals(page, renderer, provenance, snug=snug) if radicals else []
    groups = text_groups(info["source"], info["identity"], provenance, profile)
    lookup = {tuple(q["items"]): q for q in words}
    old = {q["word_id"]: q for q in old_records}
    units = []
    for group in groups:
        indices = group["items"]
        bounds = np.array([provenance[i]["bounds_px"] for i in indices])
        box = [*bounds[:, :2].min(0), *bounds[:, 2:].max(0)]
        unit = {
            **group,
            "box": list(map(float, box)),
            "original_box": list(map(float, box)),
            "compact": None,
        }
        q = lookup.get(tuple(indices))
        if q and old[q["id"]]["fallback_reason"] is None:
            if sha256(Path(q["image"])) != q["sha256"]:
                raise ValueError("Changed source word")
            parts = map_components(controls[q["id"]], guarded=True)
            ink = render_line(
                ink_from_reference(Image.open(q["image"])),
                [{"transform": q["transform"], "left": 0, "right": q["size"][0]}],
                [parts],
                (q["size"][0] + 128, q["size"][1] + 128),
            )
            image, box, compact = image_unit(
                ink, q["box"][0] - 64, q["box"][1] - 64, renderer.cfg.body_size, spacing
            )
            unit.update(
                image=image, box=box, compact=compact, word_id=q["id"], original_model=old[q["id"]]
            )
        units.append(unit)
        unit["uncompacted_box"] = list(unit["box"])
        if unit["compact"]:
            unit["uncompacted_box"][2] = unit["box"][0] + unit["compact"]["original_width"]
    lines = []
    for unit in units:
        if unit["heading"]:
            continue
        box = unit["original_box"]
        cy = (box[1] + box[3]) / 2
        match = next(
            (
                line
                for line in reversed(lines)
                if abs(cy - line["cy"]) < renderer.cfg.body_size * 0.5
            ),
            None,
        )
        if match is None:
            match = {"cy": cy, "units": []}
            lines.append(match)
        match["units"].append(unit)
    line_reports = []
    for line in lines:
        uu = sorted(line["units"], key=lambda u: u["original_box"][0])
        before = [
            b["uncompacted_box"][0] - a["uncompacted_box"][2]
            for a, b in zip(uu[:-1], uu[1:], strict=True)
        ]
        reason = None
        top, bottom = min(u["box"][1] for u in uu), max(u["box"][3] for u in uu)
        math = [
            p
            for p in provenance
            if p["context"] == "mathematics"
            and len(p["bounds_px"]) == 4
            and p["bounds_px"][1] < bottom
            and p["bounds_px"][3] > top
        ]
        if math:
            reason = "inline_math_anchor_preserved"
        if spacing and not reason:
            try:
                xx, gap = word_positions(
                    [u["box"][2] - u["box"][0] for u in uu],
                    uu[0]["box"][0],
                    page.width - renderer.cfg.margin_x,
                    renderer.cfg.body_size,
                    preferred_gaps=before,
                )
                for u, x in zip(uu, xx, strict=True):
                    width = u["box"][2] - u["box"][0]
                    u["box"][0], u["box"][2] = x, x + width
            except ValueError:
                reason = "line_capacity_fallback"
        after = [b["box"][0] - a["box"][2] for a, b in zip(uu[:-1], uu[1:], strict=True)]
        line_reports.append(
            {
                "texts": [u["text"] for u in uu],
                "before_gaps": before,
                "after_gaps": after,
                "fallback_reason": reason,
            }
        )
    replacements, removed, word_report = {}, set(), []
    for u in units:
        indices = u["items"]
        if "image" in u:
            ox, oy = u["compact"]["image_offset"]
            replacements[indices[0]] = [
                ("image", u["image"], (1, 0, 0, 1, u["box"][0] + ox, u["box"][1] + oy))
            ]
        else:
            dx = u["box"][0] - u["original_box"][0]
            items = []
            for i in indices:
                item = page.items[i]
                a, b, c, d, x, y = item[2]
                items.append((item[0], item[1], (a, b, c, d, x + dx, y), *item[3:]))
            replacements[indices[0]] = items
        removed.update(indices)
        word_report.append({k: v for k, v in u.items() if k != "image"})
    math_report = []
    changed = {q["item_index"] for q in radical_report if q["applied"]} | {
        q["bar_item_index"] for q in radical_report if q["applied"]
    }
    for p in provenance:
        if p["context"] == "mathematics":
            i = p["item_index"]
            if i not in changed:
                assert page.items[i] == original_items[i]
            math_report.append(
                {
                    **p,
                    "original_source": p["source_type"],
                    "bounds_px": item_bounds(page.items[i], renderer),
                    "radical_assembly_changed": i in changed,
                    "new_personal_sample": False,
                    "source_type": "derived-radical-assembly" if i in changed else p["source_type"],
                    "personal_collected": False if i in changed else p["personal_collected"],
                    "derived_from_personal": p["personal_collected"] if i in changed else False,
                }
            )
    final_items = []
    for i, item in enumerate(page.items):
        if i in replacements:
            final_items.extend(replacements[i])
        elif i not in removed:
            final_items.append(item)
    page.items = final_items
    return (
        page,
        renderer,
        {
            "words": word_report,
            "lines": line_reports,
            "radicals": radical_report,
            "mathematics": math_report,
            "nonradical_math_items_unchanged": True,
            "spacing": spacing,
            "radicals_enabled": radicals,
        },
    )


def build(source: Path, model_output: Path, *, spacing=True, radicals=True, snug=False) -> Path:
    import pypdfium2 as pdfium
    from pypdf import PdfReader, PdfWriter

    from hwsol.jobs.paper import pdf_image

    verify_run(source)
    verify_run(model_output)
    profile = profile_store().resolve("personal@v1")
    cfg = RenderConfig(**json.loads((source / "run.json").read_text())["config"]["render_config"])
    run = create(
        "readability-math-comparison",
        config={
            "source": str(source.resolve()),
            "source_sha256": sha256(source / "run.json"),
            "model_output": str(model_output.resolve()),
            "model_output_sha256": sha256(model_output / "run.json"),
            "checkpoint": json.loads((model_output / "reports/request.json").read_text())[
                "checkpoints"
            ]["expanded-data"],
            "training_updates": 0,
            "C_used": False,
            "profile": profile.provenance(),
            "math_personal_coverage_increased": False,
            "spacing": spacing,
            "radicals": radicals,
            "snug": snug,
        },
    )
    try:
        infos = json.loads((source / "reports/pages.json").read_text())
        words = json.loads((source / "reports/words.json").read_text())
        old = json.loads((model_output / "reports/page-provenance.json").read_text())
        fields = np.load(model_output / "reports/controls.npz")["expanded-data"]
        combined, repaired = PdfWriter(), PdfWriter()
        reports = []
        for info in infos:
            ci = info["page"]
            records = next(
                q["words"] for q in old if q["page"] == ci and q["method"] == "expanded-data"
            )
            page, renderer, report = repair_page(
                info,
                profile,
                cfg,
                [q for q in words if q["page"] == ci],
                fields,
                records,
                spacing=spacing,
                radicals=radicals,
                snug=snug,
            )
            report["page"] = ci
            reports.append(report)
            pdf = run.path / f"comparisons/repaired-page-{ci}.pdf"
            pdf.write_bytes(
                label_pdf(
                    renderer.to_vector_pdf([page]),
                    f"spacing={spacing} radicals={radicals} snug={snug}",
                    "Handnote + joined radicals" if radicals else "unchanged Handnote",
                )
            )
            baseline = run.path / f"comparisons/before-page-{ci}.pdf"
            baseline.write_bytes(
                label_pdf(
                    (model_output / f"comparisons/expanded-data-page-{ci}.pdf").read_bytes(),
                    "v0109 frozen before",
                )
            )
            combined.append(baseline)
            combined.append(pdf)
            repaired.append(pdf)
            for name, p in [("before", baseline), ("after", pdf)]:
                with pdfium.PdfDocument(p) as doc:
                    if len(doc) != 1:
                        raise ValueError("One problem per page required")
                    pdf_image(doc, 0, 200 / 72).save(run.path / f"rendered/pages/{name}-{ci}.png")
            (run.path / f"solutions/problem-{ci}.md").write_text(info["source"])
        repaired.write(run.path / "rendered/homework_solutions.pdf")
        combined.write(run.path / "comparisons/before-after.pdf")
        repaired.close()
        combined.close()
        atomic_json(run.path / "reports/repairs.json", reports)
        assert len(PdfReader(run.path / "rendered/homework_solutions.pdf").pages) == 4
        run.finish(
            "held",
            result={
                "repaired_pages": 4,
                "comparison_pages": 8,
                "training_updates": 0,
                "C_used": False,
                "production_eligible": False,
            },
        )
    except BaseException as exc:
        run.finish("failed", result={"error": repr(exc)})
        raise
    return run.path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("model_output", type=Path)
    p.add_argument("--no-spacing", action="store_true")
    p.add_argument("--no-radicals", action="store_true")
    p.add_argument("--snug-radicals", action="store_true")
    args = p.parse_args()
    print(
        build(
            args.source,
            args.model_output,
            spacing=not args.no_spacing,
            radicals=not args.no_radicals,
            snug=args.snug_radicals,
        )
    )
