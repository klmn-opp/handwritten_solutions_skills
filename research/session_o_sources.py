"""Extract or admit reviewed source rectangles without modifying original ink."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus.math_glyph_resolver import ScanGlyph


def read(path: Path):
    return json.loads(path.read_text())


def extract(source_run: Path, selection: Path) -> Path:
    verify_run(source_run)
    pages = {row["page"]: row for row in read(source_run / "reports/pages.json")}
    selected = read(selection)
    run = Run.create(
        "session-o-priority-original-crops",
        seed=1909,
        config={
            "source_run": str(source_run.resolve()),
            "source_run_sha256": sha256(source_run / "run.json"),
            "selection_sha256": sha256(selection),
            "contour_luminance": 160,
            "new_neural_updates": 0,
            "labels": "actual-ink review pending final crop approval",
        },
    )
    print(run.path, flush=True)
    try:
        bank = run.path / "solutions/glyphs"
        (bank / "crops").mkdir(parents=True)
        (bank / "parents").mkdir()
        shutil.copyfile(selection, run.path / "solutions/selection.json")
        shutil.copyfile(source_run / "solutions/session-O_full_source.pdf", bank / "source.pdf")
        (run.path / "logs/source").mkdir(parents=True)
        shutil.copyfile(__file__, run.path / "logs/source/session_o_sources.py")
        results = []
        for spec in selected:
            metadata = pages[spec["page"]]
            page_path = source_run / metadata["rgb_page"]
            if sha256(page_path) != metadata["rgb_page_sha256"]:
                raise ValueError("Original rendered page has changed")
            with Image.open(page_path) as source:
                page = source.convert("RGB")
            x0, y0, x1, y1 = spec["box"]
            if not (0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height):
                raise ValueError("Source rectangle outside original page")
            crop = page.crop(spec["box"])
            ink = np.asarray(crop.convert("L")) < 160
            if not ink.any() or any(e.any() for e in (ink[0], ink[-1], ink[:, 0], ink[:, -1])):
                raise ValueError(f"Source empty or touches crop border: {spec['id']}")
            name = spec["id"] + ".png"
            crop.save(bank / "crops" / name, dpi=(300, 300))
            parent_box = [
                max(0, x0 - 100),
                max(0, y0 - 140),
                min(page.width, x1 + 100),
                min(page.height, y1 + 50),
            ]
            page.crop(parent_box).save(bank / "parents" / name, dpi=(300, 300))
            transform = np.asarray(metadata["raster_to_pdf"])
            crop_to_pdf = transform @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
            corners = np.array([transform @ [x, y, 1] for x, y in [(x0, y0), (x1, y1)]])
            results.append(
                {
                    **spec,
                    "capture_session": "O-FULL",
                    "writer": "personal",
                    "source_sha256": metadata["source_sha256"],
                    "pdf": str((source_run / "solutions/session-O_full_source.pdf").resolve()),
                    "source_page": metadata,
                    "render_dpi": 300,
                    "raster_to_pdf": transform.tolist(),
                    "pdf_to_raster": metadata["pdf_to_raster"],
                    "crop_to_pdf": crop_to_pdf.tolist(),
                    "box_pdf_points": [
                        *corners[:, :2].min(0).tolist(),
                        *corners[:, :2].max(0).tolist(),
                    ],
                    "crop": "crops/" + name,
                    "crop_sha256": sha256(bank / "crops" / name),
                    "crop_decoded_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                    "parent_box": parent_box,
                    "parent_300dpi_crop": "parents/" + name,
                    "parent_300dpi_sha256": sha256(bank / "parents" / name),
                    "contour_luminance": 160,
                    "original_pixels_edited": False,
                    "clean_isolated_ink": False,
                    "derived_contour_only": True,
                    "human_review": False,
                    "source_group": metadata["source_sha256"],
                    "split": "author-feedback-development",
                    "prompt_is_ground_truth": False,
                }
            )
        atomic_json(bank / "sources.json", results)
        sheet = Image.new("RGB", (1100, 200 * ((len(results) + 4) // 5)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(results):
            x, y = (i % 5) * 220, (i // 5) * 200
            draw.text((x + 5, y + 5), row["id"].replace("O-FULL-", ""), fill="black")
            crop = Image.open(bank / row["crop"])
            sheet.paste(crop, (x + int((220 - crop.width) / 2), y + 35))
        sheet.save(run.path / "comparisons/original-crops-native.png")
        run.finish(
            "prepared",
            result={
                "glyph_crops": len(results),
                "symbols": dict(Counter(r["character"] for r in results)),
                "accepted": 0,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def admit(parent: Path, approval: Path) -> Path:
    verify_run(parent)
    review = read(approval)
    if review["parent_run_sha256"] != sha256(parent / "run.json"):
        raise ValueError("Crop approval does not bind this source run")
    rows = read(parent / "solutions/glyphs/sources.json")
    accepted = set(review["accepted_source_ids"])
    if not accepted or not accepted <= {row["id"] for row in rows}:
        raise ValueError("Invalid approved source IDs")
    for path, digest in review["review_evidence_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Independent review evidence changed")
    run = Run.create(
        "session-o-reviewed-glyph-bank",
        seed=1909,
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": sha256(parent / "run.json"),
            "approval": review,
            "new_neural_updates": 0,
            "production_activation": False,
        },
    )
    print(run.path, flush=True)
    try:
        bank = run.path / "solutions/glyphs"
        shutil.copytree(parent / "solutions/glyphs", bank)
        shutil.copyfile(approval, run.path / "reports/source-approval.json")
        (run.path / "logs/source").mkdir(parents=True)
        shutil.copyfile(__file__, run.path / "logs/source/session_o_sources.py")
        for row in rows:
            row["review_status"] = "agent-accepted" if row["id"] in accepted else "held"
            row["clean_isolated_ink"] = row["id"] in accepted
            row["review_approval_sha256"] = sha256(approval)
            if reference := review.get("body_references", {}).get(row["id"]):
                if row["character"] != "Φ" or reference["role"] != "capital-phi-bowl":
                    raise ValueError("Body reference is not approved for this source role")
                with Image.open(bank / row["crop"]) as image:
                    x0, y0, x1, y1 = reference["bounds_crop_px"]
                    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                        raise ValueError("Body reference lies outside the original glyph crop")
                row["body_reference"] = reference
            if row["clean_isolated_ink"]:
                ScanGlyph.load(bank, row)
        atomic_json(bank / "sources.json", rows)
        run.finish(
            "held",
            result={
                "accepted_original_glyphs": len(accepted),
                "symbols": dict(Counter(r["character"] for r in rows if r["id"] in accepted)),
                "author_style_acceptance": False,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("extract")
    first.add_argument("--source-run", type=Path, required=True)
    first.add_argument("--selection", type=Path, required=True)
    second = sub.add_parser("admit")
    second.add_argument("--parent", type=Path, required=True)
    second.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.source_run, args.selection)
    else:
        admit(args.parent, args.approval)


if __name__ == "__main__":
    main()
