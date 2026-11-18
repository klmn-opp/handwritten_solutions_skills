"""Produce four CPU jobs from the same frozen source, then verify the exported PDF pages."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.jobs.runtime import PDF, REPORT, load_job, render_job, verify_job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import ProfileStore

MODES = ("baseline", "layout-enhanced", "stroke-enhanced", "combined")


def math_structure(page):
    return {
        "formulas": [(f["latex"], f["geometry_sha256"]) for f in page["formula_groups"]],
        "matrices": page["matrix_groups"],
    }


def compare(manifest: Path, output: Path, profile: str | None, page: int) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    store = ProfileStore()
    job = load_job(manifest)
    if not 1 <= page <= len(job.problems):
        raise ValueError("Preview page outside the assignment")
    if profile is not None and store.resolve(job.profile).manifest.ref != profile:
        raise ValueError("Set the requested Profile in the input manifest or activate it first")
    summary = {"assignment": sha256(manifest), "preview_page": page, "modes": {}}
    reference = []
    bounds = []
    for mode in MODES:
        folder = output / mode
        # Baseline snapshots and hashes sources, custom paper and all resolved settings.
        frozen = manifest if mode == "baseline" else output / "baseline/manifest.json"
        render_job(frozen, folder, store, realism=mode)
        verified = verify_job(folder, pixels=True)
        report = json.loads((folder / REPORT).read_bytes())
        pages = report["pages"]
        bounds.append(pages[page - 1]["attempts"][-1]["ink_bounds_px"])
        if mode == "baseline":
            reference = pages
            summary["profile"] = report["profile"]
        differences = []
        for i in range(1, len(pages) + 1):
            filename = f"pages/{i:02d}.png"
            with (
                Image.open(output / "baseline" / filename) as orig,
                Image.open(folder / filename) as new,
            ):
                differences.append(int(np.any(np.asarray(orig) != np.asarray(new), axis=2).sum()))
        summary["modes"][mode] = {
            "pdf_sha256": sha256(folder / PDF),
            "pixel_verification": verified,
            "changed_pixels_by_page": differences,
            "math_geometry_matches_baseline": [
                math_structure(p) == math_structure(b)
                for p, b in zip(pages, reference, strict=True)
            ],
            "verified_equalities": sum(len(p["verification"].get("equalities", [])) for p in pages),
        }
    # All panels show exactly the same page region and scale.
    x0, y0 = np.floor(np.min(np.array(bounds)[:, :2], axis=0) - 15).astype(int)
    x1, y1 = np.ceil(np.max(np.array(bounds)[:, 2:], axis=0) + 15).astype(int)
    box = (max(0, x0), max(0, y0), x1, y1)
    width, height = x1 - box[0], y1 - box[1]
    scale = min(1.5, 740 / width, 820 / height)
    w, h = round(width * scale), round(height * scale)
    sheet = Image.new("RGB", (2 * (w + 24), 2 * (h + 54)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, mode in enumerate(MODES):
        x, y = (i % 2) * (w + 24) + 12, (i // 2) * (h + 54) + 8
        draw.text((x, y), mode.capitalize(), font=ImageFont.load_default(size=18), fill="#202838")
        with Image.open(output / mode / f"pages/{page:02d}.png") as im:
            sheet.paste(im.crop(box).resize((w, h), Image.Resampling.LANCZOS), (x, y + 28))
    sheet.save(output / "comparison.png")
    atomic_json(output / "comparison.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect-profile", help="Require this pinned profile@version")
    parser.add_argument("--page", type=int, default=1, help="1-based page for the preview")
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.output, args.expect_profile, args.page), indent=2))
