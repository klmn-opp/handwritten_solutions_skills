"""Experiment receipts, lossless preview PDFs and recognizer-neutral image preparation."""

import json
import subprocess
import zlib
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, EncodedStreamObject, NameObject, NumberObject

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run


def create(kind: str, *, run_id: str, config: dict, **kwargs) -> Run:
    source = Path(__file__).parent
    files = {str(p): sha256(p) for p in source.iterdir() if p.suffix in {".py", ".html"}}
    run = Run.create(
        kind,
        run_id=run_id,
        config={
            **config,
            "research_source": files,
            "research_git_status": subprocess.check_output(
                ["git", "status", "--porcelain", "--", "research/english"], text=True
            ),
        },
        **kwargs,
    )
    folder = run.path / "logs/source"
    folder.mkdir()
    for path in files:
        p = Path(path)
        (folder / p.name).write_bytes(p.read_bytes())
    return run


def lossless_pdf(images: list[Path], destination: Path) -> dict:
    writer = PdfWriter()
    for path in images:
        with Image.open(path) as raw:
            image = raw.convert("RGB")
            w, h = image.size
            stream = EncodedStreamObject()
            stream.update(
                {
                    NameObject("/Type"): NameObject("/XObject"),
                    NameObject("/Subtype"): NameObject("/Image"),
                    NameObject("/Width"): NumberObject(w),
                    NameObject("/Height"): NumberObject(h),
                    NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
                    NameObject("/BitsPerComponent"): NumberObject(8),
                    NameObject("/Filter"): NameObject("/FlateDecode"),
                }
            )
            stream._data = zlib.compress(image.tobytes())
            page = writer.add_blank_page(w, h)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/XObject"): DictionaryObject(
                        {NameObject("/Im"): writer._add_object(stream)}
                    )
                }
            )
            content = EncodedStreamObject()
            content.update({NameObject("/Filter"): NameObject("/FlateDecode")})
            content._data = zlib.compress(f"q {w} 0 0 {h} 0 0 cm /Im Do Q".encode())
            page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(destination)
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(destination)
    checks = []
    for i, path in enumerate(images):
        page = document[i]
        bitmap = page.render(scale=1)
        with Image.open(path) as original:
            expected = original.convert("RGB")
            actual = bitmap.to_pil().convert("RGB")
            same = actual.size == expected.size and actual.tobytes() == expected.tobytes()
            checks.append({"page": i + 1, "pixel_identical": same, "png_sha256": sha256(path)})
        bitmap.close()
        page.close()
    document.close()
    if not all(r["pixel_identical"] for r in checks):
        raise ValueError("PDF rasterization differs from the raw generated image")
    return {"page_count": len(images), "pdf_sha256": sha256(destination), "pages": checks}


def normalize(image_path: Path, output: Path) -> dict:
    with Image.open(image_path) as raw:
        image = raw.convert("L")
    bbox = ImageChops.invert(image).point(lambda x: 255 if x >= 55 else 0).getbbox()
    if bbox is None:
        raise ValueError("Blank image")
    border = bbox[0] == 0 or bbox[1] == 0 or bbox[2] == image.width or bbox[3] == image.height
    cropped = image.crop(bbox)
    width = max(1, round(cropped.width * 64 / cropped.height))
    scaled = cropped.resize((width, 64), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width + 40, 104), 255)
    canvas.paste(scaled, (20, 20))
    canvas.save(output)
    return {
        "ink_bbox": list(bbox),
        "raw_size": list(image.size),
        "border_ink": border,
        "border_interpretation": "possible clipping; not proof of true truncation",
        "normalization": "ink bbox, uniform height64, white margin20; no stroke edits",
        "normalized_sha256": sha256(output),
    }


def finish_generated(run: Run) -> dict:
    data = json.loads((run.path / "reports/lines.json").read_text())
    rows = data["cases"]
    images = [run.path / r["image"] for r in rows if r["status"] == "generated"]
    if images:
        atomic_json(
            run.path / "reports/pdf-integrity.json",
            lossless_pdf(images, run.path / "rendered/homework_solutions.pdf"),
        )
    return run.finish(
        "held",
        result={
            "cases": len(rows),
            "generated": len(images),
            "failed": len(rows) - len(images),
            "human_content_verified": False,
            "model_executed": data.get("model_executed", False),
        },
    )
