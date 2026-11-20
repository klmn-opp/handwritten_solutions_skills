"""Prepare the background and rasterize the actual delivered PDF using local PDFium."""

from contextlib import closing
from pathlib import Path
from threading import RLock

import pypdfium2 as pdfium
from PIL import Image, ImageChops, ImageOps

from hwsol.jobs.schema import Background

# PDFium requires serial access across documents in a single Python process.
PDF_LOCK = RLock()


def read_page(path: Path, page: int, output: Path) -> dict:
    """Expose the original PDF page to the Agent for reading scanned or drawn mathematics."""
    if output.suffix.lower() != ".png":
        raise ValueError("Choose a new .png output for the source page")
    with PDF_LOCK, closing(pdfium.PdfDocument(path)) as doc:
        if not 1 <= page <= len(doc):
            raise ValueError("Source PDF page out of range")
        w, h = doc.get_page_size(page - 1)
        image = pdf_image(doc, page - 1, min(2.5, 3200 / max(w, h)))
        total = len(doc)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        image.save(stream, format="PNG")
    return {"status": "passed", "page": page, "page_count": total, "image": str(output)}


def pdf_image(doc, index: int, scale: float) -> Image.Image:
    page = doc[index]
    try:
        bitmap = page.render(scale=scale)
        try:
            return bitmap.to_pil().convert("RGB")
        finally:
            bitmap.close()
    finally:
        page.close()


def paper_image(bg: Background, path: Path, size: tuple[int, int]) -> Image.Image:
    if bg.kind == "pdf":
        with PDF_LOCK, closing(pdfium.PdfDocument(path)) as doc:
            if bg.page > len(doc):
                raise ValueError("Background PDF page does not exist")
            w, h = doc.get_page_size(bg.page - 1)
            scale = min(max(size) / max(w, h), 8)
            image = pdf_image(doc, bg.page - 1, scale)
    else:
        with Image.open(path) as raw:
            if raw.format not in {"PNG", "JPEG"}:
                raise ValueError("Image paper must be PNG or JPEG")
            rgba = ImageOps.exif_transpose(raw).convert("RGBA")
            image = Image.new("RGB", rgba.size, "white")
            image.paste(rgba, mask=rgba.getchannel("A"))
    resampling = Image.Resampling.LANCZOS
    if bg.fit == "stretch":
        return image.resize(size, resampling)
    if bg.fit == "cover":
        return ImageOps.fit(image, size, method=resampling)
    return ImageOps.pad(image, size, method=resampling, color="white")


def visible_ink(page: bytes, paper: bytes) -> bool:
    """Catch invisible output even on textured paper by comparing the same background layer."""
    with (
        PDF_LOCK,
        closing(pdfium.PdfDocument(page)) as ink_doc,
        closing(pdfium.PdfDocument(paper)) as paper_doc,
    ):
        difference = ImageChops.difference(pdf_image(ink_doc, 0, 1), pdf_image(paper_doc, 0, 1))
        return max(high for _, high in difference.getextrema()) > 8
