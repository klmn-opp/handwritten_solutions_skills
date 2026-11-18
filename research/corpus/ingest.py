"""Incremental local PDF/image ingestion; source bytes and page coordinates are retained."""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from hwsol.profiles.io import sha256
from research.corpus.store import DOMAINS, SCHEMA, Corpus, digest, now

RECIPE = "pdfium-rgb-exif-median-ink/1"
EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def import_file(
    corpus: Corpus,
    source: Path,
    *,
    course: str,
    batch: str,
    domain: str = "personal-real",
    writer: str = "personal",
    dpi: int = 150,
    page_indices: list[int] | None = None,
) -> dict:
    if source.suffix.lower() not in EXTENSIONS or domain not in DOMAINS or not batch.strip():
        raise ValueError("Supported file, provenance domain and collection batch required")
    if not 72 <= dpi <= 600:
        raise ValueError("PDF rendering DPI must be between 72 and 600")
    data = source.read_bytes()
    import hashlib

    sha = hashlib.sha256(data).hexdigest()
    document_id = "d-" + sha
    settings = {"recipe": RECIPE, "pdf_render_dpi": dpi}
    provenance = {
        "domain": domain,
        "writer_id": writer,
        "trajectory_kind": "offline-scan",
        "is_generated": domain in {"synthetic", "model-generated", "base-font"},
    }
    with corpus.db() as db:
        try:
            doc = corpus._get(db, "document", document_id)
            if doc["settings"] != settings or doc["writer_provenance"] != provenance:
                raise ValueError(
                    "Existing original has different processing/provenance; do not overwrite"
                )
            corpus.event(
                db,
                "duplicate-import" if doc["status"] == "complete" else "resume-import",
                document_id,
                {"source": str(source.resolve()), "batch": batch},
            )
            if doc["status"] == "complete":
                return {**doc, "deduplicated": True}
        except KeyError:
            original = corpus.asset(data, source.suffix.lower())
            doc = {
                "schema": SCHEMA,
                "document_id": document_id,
                "source_hash": sha,
                "original": original,
                "original_name": source.name,
                "source_path": str(source.resolve()),
                "course": course,
                "batch": batch,
                "split_group": "session:" + batch,
                "writer_provenance": provenance,
                "settings": settings,
                "status": "importing",
                "created_at": now(),
                "privacy_status": "not-reviewed",
            }
            corpus._put(db, "document", document_id, doc)
            corpus.event(db, "import-start", document_id, doc)
    original = corpus.path(doc["original"])
    pdf = None
    try:
        if source.suffix.lower() == ".pdf":
            import pypdfium2 as pdfium
            from pypdf import PdfReader

            pdf = pdfium.PdfDocument(original)
            reader = PdfReader(original)
            count = len(pdf)
        else:
            reader = None
            with Image.open(original) as im:
                count = getattr(im, "n_frames", 1)
        indices = list(range(count)) if page_indices is None else sorted(set(page_indices))
        if any(not isinstance(i, int) or not 0 <= i < count for i in indices):
            raise ValueError("Requested page outside document")
        for index in indices:
            page_id = "p-" + digest([sha, index])[:32]
            try:
                corpus.get("page", page_id)
                continue
            except KeyError:
                pass
            if pdf is not None:
                page = pdf[index]
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil().convert("RGB")
                    conv = bitmap.get_posconv(page)
                    p0 = np.array(conv.to_page(0, 0))
                    px = (np.array(conv.to_page(image.width, 0)) - p0) / image.width
                    py = (np.array(conv.to_page(0, image.height)) - p0) / image.height
                    matrix = np.array([[px[0], py[0], p0[0]], [px[1], py[1], p0[1]], [0, 0, 1]])
                    text = reader.pages[index].extract_text() or ""
                    meta = {
                        "source_kind": "pdf-text-layer"
                        if text.strip()
                        else "pdf-image-or-vector-without-text",
                        "text_layer": text,
                        "text_layer_is_verified": False,
                        "render_dpi": dpi,
                        "scan_dpi": None,
                        "raster_to_pdf": matrix.tolist(),
                        "pdf_to_raster": np.linalg.inv(matrix).tolist(),
                        "pdf_rotation": page.get_rotation(),
                        "pdf_crop_box": list(page.get_cropbox()),
                        "pdf_media_box": list(page.get_mediabox()),
                    }
                finally:
                    bitmap.close()
                    page.close()
            else:
                with Image.open(original) as im:
                    im.seek(index)
                    orientation = int(im.getexif().get(274, 1))
                    meta = {
                        "source_kind": "image",
                        "scan_dpi": [float(v) for v in im.info["dpi"]]
                        if im.info.get("dpi")
                        else None,
                        "render_dpi": None,
                        "original_size": list(im.size),
                        "exif_orientation": orientation,
                        "orientation_operation": "Pillow.ImageOps.exif_transpose",
                        "text_layer": "",
                    }
                    image = ImageOps.exif_transpose(im).convert("RGB")
            import cv2

            gray = np.array(image.convert("L"))
            # Estimate paper only for inspection. Raw color is always used for transcription.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            ink = cv2.divide(gray, np.maximum(background, 1), scale=255)
            thumb = image.convert("L").resize((17, 16))
            arr = np.asarray(thumb)
            dhash = np.packbits(arr[:, 1:] > arr[:, :-1]).tobytes().hex()
            record = {
                "schema": SCHEMA,
                "page_id": page_id,
                "document_id": document_id,
                "source_hash": sha,
                "page_index": index,
                "page_number": index + 1,
                "width": image.width,
                "height": image.height,
                "pixel_hash": digest([image.size, hashlib.sha256(image.tobytes()).hexdigest()]),
                "color_image": corpus.asset(png(image), ".png"),
                "ink_image": corpus.asset(png(Image.fromarray(ink)), ".png"),
                "background_image": corpus.asset(png(Image.fromarray(background)), ".png"),
                "dhash": dhash,
                "writer_provenance": provenance,
                "transforms": [
                    {
                        "operation": "estimated-paper-removal",
                        "kernel": 31,
                        "coordinate_matrix": np.eye(3).tolist(),
                        "warning": "derived view only; marks may be lost",
                    }
                ],
                **meta,
            }
            with corpus.db() as db:
                corpus._put(db, "page", page_id, record)
                corpus.event(
                    db,
                    "page-import",
                    page_id,
                    {"document_id": document_id, "pixel_hash": record["pixel_hash"]},
                )
        with corpus.db() as db:
            rendered = sum(
                json.loads(row[0])["document_id"] == document_id
                for row in db.execute("SELECT data FROM records WHERE kind='page'")
            )
            doc.update(
                status="complete" if rendered == count else "partial",
                page_count=count,
                rendered_page_count=rendered,
            )
            corpus._put(db, "document", document_id, doc)
            corpus.event(db, "import-complete", document_id, {"page_count": count})
        if sha256(source) != sha:
            raise ValueError("Source changed during import; archived original is retained")
        return {**doc, "deduplicated": False}
    except Exception as exc:
        with corpus.db() as db:
            doc.update(status="interrupted", error=str(exc))
            corpus._put(db, "document", document_id, doc)
            corpus.event(db, "import-error", document_id, {"error": str(exc)})
        raise
    finally:
        if pdf is not None:
            pdf.close()


def propose_lines(corpus: Corpus, page_id: str) -> list[dict]:
    """Conservative page-wide line groups. No forced character/cursive segmentation."""
    page = corpus.get("page", page_id)
    with Image.open(corpus.path(page["ink_image"])) as image:
        ink = np.asarray(image) < 150
    counts = ink.sum(axis=1)
    active = np.flatnonzero(
        (counts > max(3, page["width"] * 0.002)) & (counts < page["width"] * 0.7)
    )
    if not active.size:
        return []
    groups = np.split(active, np.where(np.diff(active) > max(8, page["height"] // 250))[0] + 1)
    proposals = []
    for rows in groups:
        if len(rows) < 3:
            continue
        y0, y1 = max(0, int(rows[0]) - 5), min(page["height"], int(rows[-1]) + 6)
        cols = np.flatnonzero(ink[y0:y1].sum(axis=0) > 1)
        if not cols.size:
            continue
        box = [max(0, int(cols[0]) - 5), y0, min(page["width"], int(cols[-1]) + 6), y1]
        proposals.append(
            corpus.propose(
                page_id,
                box,
                notes=(
                    "Projection proposal may join printed, handwritten or mathematical regions. "
                    "Human classification required."
                ),
            )
        )
    return proposals
