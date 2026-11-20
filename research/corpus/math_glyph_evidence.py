"""Private math glyph crops, frozen coverage evidence, and author collection sheets."""

import hashlib
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract_sources(selection: list[dict], dest: Path, corpus: Path) -> list[dict]:
    """Keep original RGB crops, plus the contextual label and coordinate lineage."""
    manifest = []
    pages = {}
    digests = {}
    with sqlite3.connect(f"{corpus.resolve().as_uri()}?mode=ro", uri=True) as db:

        def rows(kind):
            return {
                key: json.loads(value)
                for key, value in db.execute("select id, data from records where kind=?", (kind,))
            }

        documents = rows("document")
        indexed_pages = rows("page")
        indexed_samples = rows("sample")
        recognitions = rows("recognition")
    (dest / "crops").mkdir(parents=True)
    (dest / "parents").mkdir()
    for item in selection:
        source, number = Path(item["pdf"]), item["page"]
        key = (source, number)
        if source.name.startswith("session-C"):
            raise ValueError("Reserved session cannot be used for math selection")
        if source not in digests:
            digests[source] = sha256(source)
        digest = digests[source]
        if key not in pages:
            with pdfium.PdfDocument(source) as document, closing(document[number - 1]) as page:
                with closing(page.render(scale=300 / 72)) as bitmap:
                    im = bitmap.to_pil().convert("RGB")
                    conv = bitmap.get_posconv(page)
                    origin = np.asarray(conv.to_page(0, 0))
                    dx = (np.asarray(conv.to_page(im.width, 0)) - origin) / im.width
                    dy = (np.asarray(conv.to_page(0, im.height)) - origin) / im.height
                    transform = np.array([np.r_[dx, 0], np.r_[dy, 0], np.r_[origin, 1]]).T
                    pages[key] = (im, transform, page.get_rotation(), list(page.get_cropbox()))
        im, transform, rotation, pdf_box = pages[key]
        x, y, right, bottom = item["box"]
        X, Y, Right, Bottom = item["parent_box"]
        if not (
            0 <= X <= x < right <= Right <= im.width and 0 <= Y <= y < bottom <= Bottom <= im.height
        ):
            raise ValueError(f"Glyph outside source/parent box: {item['character']}")
        crop, parent = im.crop(item["box"]), im.crop(item["parent_box"])
        if item.get("parent_crop"):
            old = Path(item["parent_crop"])
            with Image.open(old) as saved:
                if sha256(old) != item["parent_crop_sha256"] or (
                    parent.tobytes() != saved.convert("RGB").tobytes() or parent.size != saved.size
                ):
                    raise ValueError(f"Reviewed parent does not match original PDF: {old.name}")
        name = f"U{ord(item['character']):04X}-p{number}"
        if sample_key := item.get("sample_key"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_key):
                raise ValueError("Invalid math source sample key")
            name = f"{sample_key}-{name}"
        crop_path = dest / "crops" / f"{name}.png"
        parent_path = dest / "parents" / f"{name}.png"
        if crop_path.exists():
            raise ValueError(f"Duplicate math source sample: {name}")
        crop.save(crop_path, dpi=(300, 300))
        parent.save(parent_path, dpi=(300, 300))
        parent_page = next(
            (
                p
                for p in indexed_pages.values()
                if p["source_hash"] == digest and p["page_number"] == number
            ),
            None,
        )
        relevant = []
        if parent_page:
            px_to_original = np.array(parent_page["pdf_to_raster"]) @ transform
            near, far = [px_to_original @ [a, b, 1] for a, b in [(X, Y), (Right, Bottom)]]
            for r_id, recognition in recognitions.items():
                if recognition.get("engine") != "unimernet":
                    continue
                sample = indexed_samples.get(recognition["raw"]["id"])
                if not sample or sample["page_id"] != parent_page["page_id"]:
                    continue
                a, b, c, d = sample["bounding_box"]
                if max(a, near[0]) < min(c, far[0]) and max(b, near[1]) < min(d, far[1]):
                    relevant.append(
                        {
                            "id": r_id,
                            "text": recognition["text"],
                            "sample_id": sample["sample_id"],
                            "input_sha256": recognition["input_sha256"],
                            "run": recognition.get("run"),
                        }
                    )
        crop_to_pdf = transform @ np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
        corners = [transform @ [a, b, 1] for a, b in [(x, y), (right, bottom)]]
        glyph = {
            **item,
            "id": name,
            "pdf": str(source.resolve()),
            "source_sha256": digest,
            "source_document_id": next(
                (i for i, d in documents.items() if d["source_hash"] == digest), None
            ),
            "source_page_id": parent_page["page_id"] if parent_page else None,
            "pdf_rotation": rotation,
            "pdf_crop_box": pdf_box,
            "box_pdf_points": [
                float(min(corners[0][0], corners[1][0])),
                float(min(corners[0][1], corners[1][1])),
                float(max(corners[0][0], corners[1][0])),
                float(max(corners[0][1], corners[1][1])),
            ],
            "render_dpi": 300,
            "raster_size": list(im.size),
            "raster_to_pdf": transform.tolist(),
            "source_page_rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
            "pdf_to_raster": np.linalg.inv(transform).tolist(),
            "crop_to_pdf": crop_to_pdf.tolist(),
            "crop": str(crop_path.relative_to(dest)),
            "crop_sha256": sha256(crop_path),
            "parent_300dpi_crop": str(parent_path.relative_to(dest)),
            "parent_300dpi_sha256": sha256(parent_path),
            "derived_contour_only": True,
            "original_pixels_edited": False,
            "writer": "personal",
            "label_source": (
                "v0.10.25 Codex visual review of author session"
                if item.get("capture_session")
                else "v0.10.24 Codex visual review of original ink"
            ),
            "review_status": "agent-accepted" if item["selected"] else "held",
            "clean_isolated_ink": item["selected"],
            "human_review": False,
            "prior_unimernet_candidates": relevant,
        }
        manifest.append(glyph)
    atomic_json(dest / "sources.json", manifest)
    return manifest


def source_counts(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(row["source_type"] for row in rows))


def benchmark_inventory(baseline: Path) -> list[dict]:
    summary = read(baseline / "reports/benchmark.json")
    entries = []
    for case in summary["cases"]:
        case_id = case["case_id"]
        report = (
            baseline.with_name(f"{baseline.name}-{case_id}") / "rendered/validation_report.json"
        )
        records = []
        if report.is_file():
            for number, page in enumerate(read(report)["pages"], 1):
                records.extend(
                    {"page": number, **r}
                    for r in page["glyph_provenance"]
                    if r["context"] == "mathematics"
                )
        entries.append(
            {
                "id": case_id,
                "suite": case["suite"],
                "renderer": case["renderer"]["status"],
                "source_types": source_counts(records),
                "items": records,
                "report_sha256": sha256(report) if report.is_file() else None,
            }
        )
    return entries


HIGHLIGHTS = {
    "external-font": ("External font", "#d76726"),
    "inherited-font": ("Inherited font", "#7653a8"),
    "procedural-geometry": ("Program geometry", "#28777f"),
}


def fallback_overview(images: list[Image.Image], pages: list[list[dict]], output: Path):
    """Mark item boxes over the exact original exported pages."""
    font = ImageFont.truetype("DejaVuSans.ttf", 23)
    with PdfPages(output / "math-fallback-overlay.pdf") as pdf:
        for no, (im, rows) in enumerate(zip(images, pages, strict=True), 1):
            overlay = Image.new("RGBA", im.size)
            draw = ImageDraw.Draw(overlay)
            for row in rows:
                kind = row["source_type"]
                if kind not in HIGHLIGHTS:
                    continue
                _, color = HIGHLIGHTS[kind]
                rgb = tuple(bytes.fromhex(color[1:]))
                x, y, right, bottom = row["bounds_px"]
                draw.rectangle(
                    (x - 3, y - 3, right + 3, bottom + 3),
                    fill=(*rgb, 43),
                    outline=(*rgb, 235),
                    width=2,
                )
            x, y = 165, im.height - 188
            for kind, (label, color) in HIGHLIGHTS.items():
                count = sum(row["source_type"] == kind for row in rows)
                draw.rectangle((x, y, x + 26, y + 19), outline=color, width=3)
                draw.text((x + 36, y - 4), f"{label} ({count})", fill=color, font=font)
                x += 345
            result = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
            result.save(output / f"fallback-page-{no:02d}.png", dpi=(300, 300))
            fig = Figure(figsize=(8.27, 11.69))
            fig.add_axes((0, 0, 1, 1)).imshow(result, interpolation="none")
            fig.axes[0].set_axis_off()
            pdf.savefig(fig, dpi=300)


def collection_sheet(output: Path, session: str, prompts: list[tuple[str, str]]) -> list[dict]:
    """Match the imported Session A/B format with a prompt and one ruled answer line."""
    fig = Figure(figsize=(595.44 / 72, 841.68 / 72))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 595.44), ylim=(0, 841.68))
    ax.axis("off")
    ax.text(42, 804, f"Personal Math Writing / Session {session}", fontsize=15)
    role = "PRIORITY GLYPH SAMPLES" if session == "M" else "OPTIONAL GLYPH SAMPLES"
    ax.text(42, 783, role, fontsize=10)
    ax.text(
        42, 765, "Write each prompt naturally on the line below it. Use your usual pen.", fontsize=9
    )
    ax.text(42, 750, "Keep the original colour and save this page as a separate PDF.", fontsize=8)
    records = []
    for index, (latex, note) in enumerate(prompts, 1):
        y = 714 - (index - 1) * 64
        prompt = f"{session}{index:02d}"
        ax.text(42, y, f"{prompt}.", fontsize=9)
        ax.text(80, y, f"${latex}$", fontsize=12)
        ax.text(313, y, note, fontsize=8)
        ax.plot([42, 553], [y - 40, y - 40], linewidth=0.5, color="#cccccc")
        records.append(
            {
                "id": prompt,
                "intended_latex": latex,
                "review_status": "not-collected",
                "verified_latex": None,
                "answer_bbox_pdf_points": [42, y - 48, 553, y - 6],
            }
        )
    ax.text(42, 45, "Printed prompts are references for this collection session.", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"session-{session}.pdf")
    fig.savefig(output / f"session-{session}.png", dpi=150)
    return records
