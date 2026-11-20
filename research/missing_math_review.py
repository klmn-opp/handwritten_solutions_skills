"""Bounded local historical-symbol search, with sealed-source filtering before labels.

Recognition is a locator only. RGB evidence is rendered from unchanged originals;
no cloud call, new OCR, glyph construction, label mutation or integration occurs.
"""

import argparse
import json
import math
import re
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run
from research.corpus_registry import SEALED

ROOT = Path(__file__).resolve().parents[1]
CATALOGS = [
    ROOT / "workspace/datasets/personal-homework/corpus.sqlite",
    ROOT
    / (
        "workspace/runs/20261119T081423714624Z-new-continuous-corpus-v0105-970c87ac/"
        "logs/imported_corpus/corpus.sqlite"
    ),
]
TARGETS = {
    "Φ": r"\\Phi(?![A-Za-z])|Φ",
    "κ": r"\\kappa(?![A-Za-z])|κ",
    "ξ": r"\\xi(?![A-Za-z])|ξ",
    "η": r"\\eta(?![A-Za-z])|η",
    "ζ": r"\\zeta(?![A-Za-z])|ζ",
    "ψ": r"\\psi(?![A-Za-z])|ψ",
    "∏": r"\\prod(?![A-Za-z])|∏",
    "∮": r"\\oint(?![A-Za-z])|∮",
    "≈": r"\\approx(?![A-Za-z])|≈",
    "∈": r"\\in(?![A-Za-z])|∈",
    "⟨": r"\\langle(?![A-Za-z])|⟨",
    "⟩": r"\\rangle(?![A-Za-z])|⟩",
    "‖": r"\\(?:Vert|lVert|rVert)(?![A-Za-z])|‖",
    "∬": r"\\iint(?![A-Za-z])|∬",
    "ℝ": r"\\mathbb\s*\{\s*R\s*\}|ℝ",
}
FORBIDDEN_NAME = re.compile(r"session[ _-]*c(?:[ _.-]|$)|heldout", re.I)
FORMULA_SHEET = "c6fa4ab4f530863d0d3c207e4ef782bdc24930452f54f73998a145246c575de9"


def hits(text):
    return [symbol for symbol, pattern in TARGETS.items() if re.search(pattern, text or "")]


def catalogue(path):
    """Resolve blocked ancestry using IDs/SHA only, before retrieving any candidate text."""
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        doc_meta = db.execute(
            "SELECT id,json_extract(data,'$.source_hash'),json_extract(data,'$.source_path') "
            "FROM records WHERE kind='document'"
        ).fetchall()
        blocked_sources = set(SEALED) | {
            digest for _, digest, name in doc_meta if FORBIDDEN_NAME.search(name or "")
        }
        sample_meta = db.execute(
            "SELECT id,json_extract(data,'$.source_hash'),json_extract(data,'$.parents'),"
            "json_extract(data,'$.dataset_split') FROM records WHERE kind='sample'"
        ).fetchall()
        blocked_ids = {
            sid
            for sid, digest, _, split in sample_meta
            if not digest or digest in blocked_sources or split in {"holdout", "test", "sealed"}
        }
        while True:
            extra = {
                sid
                for sid, _, parents, _ in sample_meta
                if any(parent in blocked_ids for parent in json.loads(parents or "[]"))
            }
            if extra <= blocked_ids:
                break
            blocked_ids |= extra
        allowed_ids = [sid for sid, *_ in sample_meta if sid not in blocked_ids]
        allowed_hashes = [digest for _, digest, _ in doc_meta if digest not in blocked_sources]

        def safe_rows(kind, field, values):
            if not values:
                return []
            placeholders = ",".join("?" for _ in values)
            return [
                json.loads(raw)
                for (raw,) in db.execute(
                    f"SELECT data FROM records WHERE kind=? AND {field} IN ({placeholders})",
                    [kind, *values],
                )
            ]

        documents = safe_rows("document", "json_extract(data,'$.source_hash')", allowed_hashes)
        pages = safe_rows("page", "json_extract(data,'$.source_hash')", allowed_hashes)
        samples = safe_rows("sample", "id", allowed_ids)
        recognitions = safe_rows("recognition", "json_extract(data,'$.sample_id')", allowed_ids)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "documents": documents,
        "pages": pages,
        "samples": samples,
        "recognitions": recognitions,
        "filter": {
            "excluded_source_hashes": sorted(blocked_sources),
            "blocked_sample_count": len(blocked_ids),
            "method": "source SHA and ancestry IDs before retrieving sample/recognition text",
        },
    }


def candidates(catalogues):
    rows, source_lookup, counts = {}, {}, []
    for cat in catalogues:
        docs = {d["source_hash"]: d for d in cat["documents"]}
        pages = {p["page_id"]: p for p in cat["pages"]}
        samples = {s["sample_id"]: s for s in cat["samples"]}
        recognition_hits = 0
        for digest, doc in docs.items():
            source_lookup[digest] = {**doc, "catalog_path": cat["path"]}

        def add(sample, label, locator, *, pages=pages, docs=docs, catalog_path=cat["path"]):
            matched = hits(label)
            if not matched:
                return
            digest = sample["source_hash"]
            assert digest not in SEALED
            page = pages[sample["page_id"]]
            key = (digest, page["page_number"], *sample["bounding_box"])
            if key not in rows:
                rows[key] = {
                    "source_hash": digest,
                    "document": docs[digest],
                    "page": page,
                    "sample": sample,
                    "catalog_path": catalog_path,
                    "locators": [],
                    "targets": [],
                    "review_status": "pending",
                    "kind": "catalog-region",
                }
            rows[key]["locators"].append({"original_candidate_label": label, **locator})
            rows[key]["targets"] = sorted(set(rows[key]["targets"]) | set(matched))

        for rec in cat["recognitions"]:
            if hits(rec.get("text")):
                recognition_hits += 1
                add(
                    samples[rec["sample_id"]],
                    rec["text"],
                    {
                        "recognition_id": rec["id"],
                        "engine": rec["engine"],
                        "is_ground_truth": False,
                        "existing_run": rec.get("run"),
                    },
                )
        for sample in samples.values():
            for field in ("verified_latex", "verified_text", "candidate_text"):
                if hits(sample.get(field)):
                    add(
                        sample,
                        sample[field],
                        {
                            "sample_field": field,
                            "existing_review_status": sample["review_status"],
                            "is_ground_truth": False,
                        },
                    )
        for page in cat["pages"]:
            if page["source_hash"] != FORMULA_SHEET:
                continue
            key = (FORMULA_SHEET, page["page_number"], 0, 0, page["width"], page["height"])
            rows[key] = {
                "source_hash": FORMULA_SHEET,
                "document": docs[FORMULA_SHEET],
                "page": page,
                "sample": {"bounding_box": list(key[2:]), "sample_id": None},
                "catalog_path": cat["path"],
                "locators": [{"method": "math formula-sheet context"}],
                "targets": [],
                "review_status": "pending",
                "kind": "catalog-page-context",
            }
        counts.append(
            {
                "catalog": cat["path"],
                "documents": len(docs),
                "pages": len(pages),
                "samples": len(samples),
                "recognitions": len(cat["recognitions"]),
                "recognition_target_hits": recognition_hits,
                "filter": cat["filter"],
            }
        )
    result = sorted(
        rows.values(),
        key=lambda r: (
            r["kind"] != "catalog-region",
            r["document"]["original_name"],
            r["page"]["page_number"],
            r["sample"]["bounding_box"],
        ),
    )
    if len(result) > 30:
        raise ValueError("Bounded review would exceed 30 unique original regions")
    for index, row in enumerate(result, 1):
        row["review_id"] = f"math-review-{index:02d}"
    return result, counts


def source_path(row):
    digest = row["source_hash"]
    if digest in SEALED:
        raise ValueError("Sealed source cannot be accessed")
    original = Path(row["document"]["source_path"])
    if not original.is_file():
        original = Path(row["catalog_path"]).parent / row["document"]["original"]
    if FORBIDDEN_NAME.search(str(original)):
        raise ValueError("Sealed or heldout path cannot be accessed")
    if sha256(original) != digest:
        raise ValueError("Original source differs from catalog SHA")
    return original


def registered_math_search():
    registry_path = ROOT / (
        "workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533/"
        "reports/registry.json"
    )
    registry = json.loads(registry_path.read_text())
    allowed = {
        key
        for key, doc in registry["documents"].items()
        if doc["source_sha256"] not in SEALED
        and not any(FORBIDDEN_NAME.search(p) for p in doc.get("original_assets", []))
    }
    # Filter document lineage before inspecting observation labels or view assets.
    observations = [
        r
        for r in registry["observations"].values()
        if r["document"] in allowed and r["kind"] == "math-symbol"
    ]
    matched = [{**r, "target_hits": hits(r["label"])} for r in observations if hits(r["label"])]
    return {
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "math_observations": len(observations),
        "matching_existing_labels": matched,
        "no_image_or_training_view_opened": True,
    }


def extract(row, run):
    original = source_path(row)
    dest = run / "solutions/originals" / (row["source_hash"] + original.suffix.lower())
    dest.parent.mkdir(exist_ok=True)
    if not dest.exists():
        shutil.copyfile(original, dest)
    pixel_box = np.asarray(row["sample"]["bounding_box"], dtype=float)
    page_record = row["page"]
    if original.suffix.lower() == ".pdf":
        with (
            pdfium.PdfDocument(original) as document,
            closing(document[page_record["page_number"] - 1]) as page,
        ):
            with closing(page.render(scale=300 / 72)) as bitmap:
                image = bitmap.to_pil().convert("RGB")
                conv = bitmap.get_posconv(page)
                origin = np.asarray(conv.to_page(0, 0))
                dx = (np.asarray(conv.to_page(image.width, 0)) - origin) / image.width
                dy = (np.asarray(conv.to_page(0, image.height)) - origin) / image.height
                new_to_pdf = np.array([np.r_[dx, 0], np.r_[dy, 0], np.r_[origin, 1]]).T
                old_to_pdf = np.asarray(page_record["raster_to_pdf"])
                old_to_new = np.linalg.inv(new_to_pdf) @ old_to_pdf
                points = np.array([[x, y, 1] for x in pixel_box[[0, 2]] for y in pixel_box[[1, 3]]])
                mapped = (old_to_new @ points.T).T[:, :2]
                box = [
                    max(0, math.floor(mapped[:, 0].min())),
                    max(0, math.floor(mapped[:, 1].min())),
                    min(image.width, math.ceil(mapped[:, 0].max())),
                    min(image.height, math.ceil(mapped[:, 1].max())),
                ]
                lineage = {
                    "render_dpi": 300,
                    "raster_to_pdf": new_to_pdf.tolist(),
                    "pdf_to_raster": np.linalg.inv(new_to_pdf).tolist(),
                    "catalog_raster_to_new_raster": old_to_new.tolist(),
                    "pdf_rotation": page.get_rotation(),
                    "pdf_crop_box": list(page.get_cropbox()),
                    "crop_to_pdf": (
                        new_to_pdf @ np.array([[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]])
                    ).tolist(),
                }
                dpi = (300, 300)
    else:
        with Image.open(original) as photo:
            image = photo.convert("RGB")
        if image.size != (page_record["width"], page_record["height"]):
            raise ValueError("Native photo coordinates need an explicit transform")
        box = pixel_box.astype(int).tolist()
        dpi = tuple(page_record.get("scan_dpi") or (0, 0))
        lineage = {
            "render_dpi": None,
            "native_photo_dpi_metadata": list(dpi),
            "no_pdf_source": True,
            "upsampling_performed": False,
            "crop_to_original_pixels": [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]],
        }
    crop = image.crop(box)
    output = run / "solutions/crops" / (row["review_id"] + ".png")
    output.parent.mkdir(exist_ok=True)
    crop.save(output, dpi=dpi if all(dpi) else (72, 72))
    return {
        **row,
        "original_copy": str(dest.relative_to(run)),
        "source_path": str(original),
        "crop": str(output.relative_to(run)),
        "crop_sha256": sha256(output),
        "crop_rgb_size": list(crop.size),
        "box_in_new_raster": box,
        "original_pixels_edited": False,
        "lineage": lineage,
        "candidate_label_is_truth": False,
    }


def contact_sheets(rows, run):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 23)
    panels = []
    for offset in range(0, len(rows), 4):
        canvas = Image.new("RGB", (1800, 1500), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(rows[offset : offset + 4]):
            x, y = (index % 2) * 900, (index // 2) * 750
            label = (
                f"{row['review_id']} | {row['document']['original_name']} "
                f"p{row['page']['page_number']}"
            )
            draw.text((x + 15, y + 12), label, fill="black", font=font)
            draw.text(
                (x + 15, y + 45),
                "Locator targets: " + " ".join(row["targets"]),
                fill="black",
                font=font,
            )
            with Image.open(run / row["crop"]) as im:
                thumb = im.copy()
            thumb.thumbnail((870, 650))
            canvas.paste(thumb, (x + 15, y + 85))
        canvas.save(run / f"comparisons/contact-{offset // 4 + 1:02d}.png")
        panels.append(canvas)
    if panels:
        panels[0].save(
            run / "comparisons/candidate-contact-sheets.pdf",
            save_all=True,
            append_images=panels[1:],
            resolution=150,
        )


def prepare():
    run = Run.create(
        "missing-math-historical-review",
        seed=20261120,
        config={
            "targets": list(TARGETS),
            "maximum_regions": 30,
            "sealed_hashes": sorted(SEALED),
            "new_ocr_or_external_upload": False,
            "automatic_integration": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        catalogues = [catalogue(p) for p in CATALOGS]
        rows, counts = candidates(catalogues)
        atomic_json(run.path / "reports/search-scope.json", counts)
        atomic_json(run.path / "reports/registry-search.json", registered_math_search())
        rows = [extract(row, run.path) for row in rows]
        atomic_json(run.path / "reports/candidates.json", rows)
        atomic_json(
            run.path / "reports/catalog-source-sha.json",
            {cat["path"]: cat["sha256"] for cat in catalogues},
        )
        contact_sheets(rows, run.path)
        atomic_json(
            run.path / "reports/review-template.json",
            [
                {
                    "review_id": r["review_id"],
                    "status": "pending",
                    "actual_visible_symbols": [],
                    "reason": "Original RGB visual inspection required; OCR is not truth",
                }
                for r in rows
            ],
        )
        print(
            json.dumps(
                {
                    "regions": len(rows),
                    "locator_target_counts": dict(
                        Counter(ch for row in rows for ch in row["targets"])
                    ),
                    "review": "pending local visual inspection",
                }
            )
        )
        # Keep running until the bounded visual decisions have been recorded.
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def finish(path, decisions):
    run = Run(path)
    rows = json.loads((path / "reports/candidates.json").read_text())
    choices = json.loads(decisions.read_text())
    if {r["review_id"] for r in rows} != {r["review_id"] for r in choices}:
        raise ValueError("Every bounded candidate requires exactly one decision")
    if any(
        r["status"] not in {"accepted-region", "rejected", "needs-author-review"} for r in choices
    ):
        raise ValueError("Unresolved review status")
    for name, digest in json.loads((path / "reports/catalog-source-sha.json").read_text()).items():
        if sha256(Path(name)) != digest:
            raise ValueError("Read-only catalog changed during review")
    atomic_json(path / "reports/visual-review.json", choices)
    atomic_json(path / "reports/registry-search.json", registered_math_search())
    shutil.copyfile(__file__, path / "logs/final-review-source.py")
    by_id = {r["review_id"]: r for r in rows}
    selected = []
    output = path / "solutions/reviewed-extractions"
    output.mkdir()
    selections = [
        ("membership-left", "math-review-16", [1060, 970, 1130, 1055], "∈", "agent-accepted"),
        ("membership-right", "math-review-16", [2117, 963, 2170, 1043], "∈", "agent-accepted"),
        (
            "double-integral-context",
            "math-review-19",
            [1655, 1045, 1775, 1220],
            r"\int\int",
            "held-neighbour-contamination",
        ),
    ]
    for name, parent_id, box, label, state in selections:
        parent = by_id[parent_id]
        if parent["source_hash"] in SEALED or parent["lineage"].get("render_dpi") != 300:
            raise ValueError("Only reviewed allowed-source 300 DPI PDF crops are selected")
        expected_page = 5 if name.startswith("double") else 2
        if parent["source_hash"] != FORMULA_SHEET or parent["page"]["page_number"] != expected_page:
            raise ValueError("Reviewed coordinates do not belong to this frozen original page")
        if next(r for r in choices if r["review_id"] == parent_id)["status"] != "accepted-region":
            raise ValueError("Extraction parent has not passed region-level visual review")
        with Image.open(path / parent["crop"]) as image:
            crop = image.convert("RGB").crop(box)
        dest = output / (name + ".png")
        crop.save(dest, dpi=(300, 300))
        transform = np.asarray(parent["lineage"]["crop_to_pdf"])
        crop_to_pdf = transform @ np.array([[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]])
        selected.append(
            {
                "id": name,
                "actual_visible_label": label,
                "review_status": state,
                "target_codepoint_request": "∬" if name.startswith("double") else "∈",
                "writer": "personal; inherited source attribution, Agent visual inspection",
                "source_pdf_sha256": parent["source_hash"],
                "source_pdf": parent["original_copy"],
                "page_number": parent["page"]["page_number"],
                "parent_region": parent_id,
                "parent_crop_sha256": parent["crop_sha256"],
                "box_in_parent_300dpi": box,
                "crop_to_pdf": crop_to_pdf.tolist(),
                "render_dpi": 300,
                "color_mode": "RGB",
                "crop": str(dest.relative_to(path)),
                "sha256": sha256(dest),
                "original_pixels_edited": False,
                "original_label_changed": False,
                "independent_review": "pending parent Agent; no automatic integration",
                "issue": (
                    "A lower-limit stroke enters the rectangle; no pixels removed or filled"
                    if name.startswith("double")
                    else None
                ),
            }
        )
    atomic_json(path / "reports/reviewed-extractions.json", selected)
    run.finish(
        "held",
        result={
            "regions_reviewed": len(rows),
            "decisions": dict(Counter(r["status"] for r in choices)),
            "new_glyphs_integrated": 0,
            "new_ocr_or_training": False,
            "original_labels_changed": False,
            "accepted_original_membership_crops": 2,
            "held_double_integral_context": 1,
            "decision": "Source review only; isolated extraction and independent review required",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finish", type=Path)
    parser.add_argument("--decisions", type=Path)
    args = parser.parse_args()
    if args.finish:
        finish(args.finish, args.decisions)
    else:
        prepare()
