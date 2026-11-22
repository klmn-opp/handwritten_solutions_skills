"""Compose the frozen neural glyph layer onto a stable personal-CPU page.

Isolated per-item layers give exact ink ownership. A replacement is admitted only
when it introduces no new dark contact and keeps the item's connected components
and holes. The deterministic shared-pen/tone correction and the frozen integer
word spacing then run on the accepted pixels, and the page is embedded losslessly
as native grayscale in an A4 PDF.
"""

from __future__ import annotations

import math
import re
import string
from contextlib import closing
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from hwsol.jobs.paper import PDF_LOCK
from hwsol.jobs.runtime import export_raster
from hwsol.neural import ink_shared

MARGIN_PAD = 7


def gray(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return np.ascontiguousarray(image, dtype=np.uint8)
    return np.asarray(image.convert("L")).copy()


def raster(pdf: bytes, config) -> np.ndarray:
    import pypdfium2 as pdfium

    with PDF_LOCK, closing(pdfium.PdfDocument(pdf)) as document:
        return np.asarray(export_raster(document, 0, config).convert("L")).copy()


def ibox(record: dict) -> list[int]:
    x, y, X, Y = record["bounds_px"]
    return [math.floor(x), math.floor(y), math.ceil(X), math.ceil(Y)]


def signature(image: np.ndarray) -> tuple[int, int]:
    fg = image < 160
    return (
        int(cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)[0]),
        int(
            cv2.connectedComponents(
                np.pad(~fg, 1, constant_values=True).astype(np.uint8), connectivity=4
            )[0]
        ),
    )


def fit(gray_image: np.ndarray, width: int, height: int) -> np.ndarray:
    ys, xs = np.where(gray_image < 208)
    if not len(xs):
        raise ValueError("Empty glyph")
    crop = gray_image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return np.asarray(
        Image.fromarray(crop).resize((width, height), Image.Resampling.LANCZOS)
    ).astype(np.uint8)


def fit_english(generated: np.ndarray, stable: np.ndarray) -> np.ndarray | None:
    """Scale by the limiting axis and paste centred; reject implausible pen weight."""
    ys, xs = np.where(generated < 208)
    if not len(xs):
        return None
    h, w = stable.shape
    crop = generated[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min(h / crop.shape[0], w / crop.shape[1])
    height, width = max(1, round(crop.shape[0] * scale)), max(1, round(crop.shape[1] * scale))
    if height < h * 0.80:
        return None
    out = np.full_like(stable, 255)
    x = (w - width) // 2
    out[h - height :, x : x + width] = np.asarray(
        Image.fromarray(crop).resize((width, height), Image.Resampling.LANCZOS)
    )
    wanted = ink_shared.ink_measure(Image.fromarray(stable))["weight"]
    actual = ink_shared.ink_measure(Image.fromarray(out))["weight"]
    return out if 0.64 * wanted <= actual <= 1.56 * wanted else None


def separated_layers(
    records: list[dict], size: tuple[int, int]
) -> list[list[tuple[int, list[int]]]]:
    layers: list[list[tuple[int, list[int]]]] = []
    for index, record in enumerate(records):
        x, y, X, Y = ibox(record)
        box = [
            max(0, x - MARGIN_PAD),
            max(0, y - MARGIN_PAD),
            min(size[0], X + MARGIN_PAD),
            min(size[1], Y + MARGIN_PAD),
        ]
        for layer in layers:
            if all(
                box[2] <= other[0] or box[0] >= other[2] or box[3] <= other[1] or box[1] >= other[3]
                for _, other in layer
            ):
                break
        else:
            layer = []
            layers.append(layer)
        layer.append((index, box))
    return layers


def item_tiles(page, renderer, records, config) -> list[dict]:
    full = raster(renderer.to_vector_pdf([page]), config)
    items = []
    for layer in separated_layers(records, full.shape[::-1]):
        subset = replace(page, items=[page.items[i] for i, _ in layer])
        layer_image = raster(renderer.to_vector_pdf([subset]), config)
        for index, box in layer:
            cut = Image.fromarray(layer_image).crop(box)
            tight = ImageChops.invert(cut).getbbox()
            if not tight:
                continue
            native = np.asarray(cut.crop(tight).convert("L")).copy()
            bounds = [box[0] + tight[0], box[1] + tight[1], box[0] + tight[2], box[1] + tight[3]]
            record = records[index]
            items.append(
                dict(
                    id=index,
                    box=bounds,
                    image=native,
                    stable=native,
                    context=record["context"],
                    semantic=record["semantic"],
                    structure=record.get("structure"),
                    matrix_environment=record.get("matrix_environment"),
                    source_type=record["source_type"],
                    latex=record.get("latex"),
                    role="protected structure" if record.get("structure") else "ordinary",
                    kind=(
                        "math"
                        if record["context"] == "mathematics"
                        else "text"
                        if record["semantic"] in string.ascii_letters
                        and len(record["semantic"]) == 1
                        else "punctuation"
                    ),
                    record=record,
                )
            )
    return full, items


def compose_page(shape, items, replacements, previous=None):
    """Paste accepted replacements over isolated original tiles with contact guards."""
    isolated = np.full(shape, 255, np.uint8)
    counts = np.zeros(shape, np.int16)
    for item in items:
        x, y, X, Y = item["box"]
        view = isolated[y:Y, x:X]
        np.minimum(view, item["stable"], out=view)
        counts[y:Y, x:X] += item["stable"] < 160
    page = np.full(shape, 255, np.uint8)
    diagnostics = []
    for item in items:
        x, y, X, Y = item["box"]
        old = item["stable"]
        strong = counts[y:Y, x:X].copy()
        strong -= old < 160
        target = replacements.get(item["id"])
        hazard = bool(target is not None and np.any((target < 160) & (strong > 0) & ~(old < 160)))
        admitted = target is not None and not hazard
        placed = target if admitted else old
        item["placed"] = placed
        strong += placed < 160
        view = page[y:Y, x:X]
        np.minimum(view, placed, out=view)
        diagnostics.append(
            dict(
                id=item["id"],
                label=item["semantic"],
                admitted=admitted,
                collision=hazard,
                route=item["route"] if admitted else "layout",
            )
        )
    unchanged = isolated == page
    if previous is not None:
        page[unchanged] = previous[unchanged]
    return page, diagnostics


def areas_from_page(page, items):
    """Per-item ownership from each placed tile, reconciled with neighbours.

    A page crop contains neighbouring glyphs whenever boxes overlap (for example a
    radical sign and the radicand). Ownership must come from the item's own placed
    tile; otherwise the neighbour claims every pixel and the glyph becomes blank.
    """
    areas = []
    for item in items:
        x, y, X, Y = item["box"]
        local = page[y:Y, x:X]
        mask = np.asarray(item.get("placed", item["stable"])) < 255
        areas.append(
            dict(
                id=item["id"],
                kind=item["kind"],
                role=item["role"],
                label=item["semantic"],
                box=item["box"],
                mask=mask,
                group=(
                    ("word", item.get("word"))
                    if item["kind"] == "text"
                    else ("formula", item.get("latex"))
                    if item["kind"] == "math"
                    else ("mark", item["id"])
                ),
                gray=Image.fromarray(np.where(mask, local, 255).astype(np.uint8)),
                teacher_gray=Image.fromarray(item["stable"]),
            )
        )
    ownership = np.zeros(page.shape, np.uint16)
    for area in areas:
        x, y, X, Y = area["box"]
        ownership[y:Y, x:X] += area["mask"]
    for area in areas:
        x, y, X, Y = area["box"]
        local = page[y:Y, x:X]
        area["mask"] &= (ownership[y:Y, x:X] == 1) & (local < 255)
        area["gray"] = Image.fromarray(np.where(area["mask"], local, 255).astype(np.uint8))
    return areas


def paste_tone(page: np.ndarray, areas: list[dict], outputs: list[np.ndarray]) -> np.ndarray:
    """Paste corrected grayscale only where it preserves local and page topology."""
    result = page.copy()
    owners = np.zeros(page.shape, np.uint16)
    moves = []
    for area, new in zip(areas, outputs, strict=True):
        before = np.asarray(area["gray"])
        new = np.asarray(new)
        if before.shape != new.shape:
            raise ValueError("Ink stage resized a glyph")
        changed = new != before
        added = changed & ~area["mask"]
        x, y, X, Y = area["box"]
        if np.any(added & (page[y:Y, x:X] < 255)):
            changed[:] = False
        owners[y:Y, x:X] += changed
        moves.append(changed)
    for area, new, move in zip(areas, outputs, moves, strict=True):
        x, y, X, Y = area["box"]
        if np.any(move & (owners[y:Y, x:X] > 1)):
            move[:] = False
        local = result[y:Y, x:X]
        if move.any():
            left, top = max(0, x - 2), max(0, y - 2)
            around = result[top : min(page.shape[0], Y + 2), left : min(page.shape[1], X + 2)]
            previous = np.asarray(area["gray"])
            dims = (slice(y - top, Y - top), slice(x - left, X - left))
            thresholds = (128, 160, 192)
            before = [
                (ink_shared.base.topology(around < t), ink_shared.base.topology(previous < t))
                for t in thresholds
            ]
            delta = np.asarray(new, dtype=float) - previous
            for strength in (1.0, 0.5, 0.25):
                proposed = np.uint8(np.rint(previous + strength * delta))
                test = around.copy()
                test[dims][move] = proposed[move]
                if all(
                    ink_shared.base.topology(test < t) == outer
                    and ink_shared.base.topology(proposed < t) == inner
                    for t, (outer, inner) in zip(thresholds, before, strict=True)
                ):
                    local[move] = proposed[move]
                    break
        area["mask"] = (area["mask"] | move) & (local < 255)
        area["after"] = Image.fromarray(np.where(area["mask"], local, 255).astype(np.uint8))
    return result


def check_page_contours(before: np.ndarray, after: np.ndarray) -> dict:
    yy, xx = np.where((before < 255) | (after < 255))
    patch = np.s_[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    tests = {
        t: (ink_shared.base.topology(before[patch] < t), ink_shared.base.topology(after[patch] < t))
        for t in (128, 160, 192)
    }
    if any(a != b for a, b in tests.values()):
        raise ValueError(f"Composite page changed connected marks or holes: {tests}")
    return {str(k): list(v[1]) for k, v in tests.items()}


def write_a4(pages: list[np.ndarray], destination) -> dict:
    """Embed native grayscale pixels losslessly on ISO A4 and verify every page."""
    writer = PdfWriter()
    width, height = 210 * 72 / 25.4, 297 * 72 / 25.4
    for array in pages:
        image = Image.fromarray(array).convert("L")
        page = writer.add_blank_page(width, height)
        stream = DecodedStreamObject()
        stream.set_data(image.tobytes())
        img = stream.flate_encode()
        for key, value in {
            "Type": "XObject",
            "Subtype": "Image",
            "ColorSpace": "DeviceGray",
        }.items():
            img[NameObject("/" + key)] = NameObject("/" + value)
        for key, value in {
            "Height": image.height,
            "Width": image.width,
            "BitsPerComponent": 8,
        }.items():
            img[NameObject("/" + key)] = NumberObject(value)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/XObject"): DictionaryObject({NameObject("/Im"): writer._add_object(img)})}
        )
        content = DecodedStreamObject()
        content.set_data(f"q {width:.9f} 0 0 {height:.9f} 0 0 cm /Im Do Q".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(destination)
    reader = PdfReader(destination, strict=True)
    if len(reader.pages) != len(pages):
        raise ValueError("Incorrect page count")
    for pdf_page, array in zip(reader.pages, pages, strict=True):
        image = Image.fromarray(array).convert("L")
        img = pdf_page["/Resources"]["/XObject"]["/Im"]
        if (
            abs(float(pdf_page.mediabox.width) - width) > 1e-4
            or abs(float(pdf_page.mediabox.height) - height) > 1e-4
            or (img["/Width"], img["/Height"], img.get_data())
            != (image.width, image.height, image.tobytes())
        ):
            raise ValueError("A4 dimensions or native PDF samples changed")
    return dict(pages=len(pages), size_mm=[210, 297], embedded_native_grayscale_exact=True)


def text_word_groups(records: list[dict], config) -> list[dict]:
    """Group visible text glyphs into words from layout geometry, not prompt labels."""
    font_px = float(config.font_size_pt) * float(config.dpi) / 72.0
    threshold = 0.30 * font_px
    words = []
    current = []

    def flush():
        if not current:
            return
        bounds = np.array([ibox(r) for r in current])
        words.append(
            {
                "index": len(words) + 1,
                "word": "".join(r["semantic"] for r in current),
                "glyphs": [r["item_index"] for r in current],
                "box": [
                    int(bounds[:, 0].min()) - 2,
                    int(bounds[:, 1].min()) - 2,
                    int(bounds[:, 2].max()) + 2,
                    int(bounds[:, 3].max()) + 2,
                ],
            }
        )

    for record in records:
        if record["context"] != "text":
            continue
        box = record["bounds_px"]
        if not current:
            current.append(record)
            continue
        previous = current[-1]
        gap = box[0] - previous["bounds_px"][2]
        if gap > threshold or box[0] < previous["bounds_px"][0]:
            flush()
            current = [record]
        else:
            current.append(record)
    flush()
    return words


_COMMAND = re.compile(r"\\([a-zA-Z]+|.)")


def prepared_source(source: str) -> str:
    """Normalize display-math line breaks and the ``\\det`` relation only inside math.

    Newlines are whitespace in TeX mathematics; the pinned MathText parser otherwise
    splits a display block into unbalanced dollar fragments. Prose is untouched.
    """
    from hwsol.jobs.layout import _LITERAL

    def replace(match):
        value = match[0]
        if not value.startswith("$"):
            return value
        display = value.startswith("$$")
        size = 2 if display else 1
        body = value[size:-size]
        body = _COMMAND.sub(lambda m: r"\mathrm{det}" if m[1] == "det" else m[0], body)
        # MathText treats these named delimiters as unauditable fallbacks; their
        # canonical literal forms are semantically identical and reviewable.
        body = (
            body.replace(r"\lvert", "|")
            .replace(r"\rvert", "|")
            .replace(r"\vert", "|")
            .replace(r"\Vert", r"\|")
        )
        if display:
            content = re.search(r"\S(?:[\s\S]*\S)?", body)
            if content:
                start, end = content.span()
                inner = body[start:end].replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
                body = body[:start] + inner + body[end:]
        return "$" * size + body + "$" * size

    return _LITERAL.sub(replace, source)


def substitute_punctuation(page, renderer, records, bundle):
    """Reuse reviewed personal colon/prime contours in existing prose slots."""
    import copy

    from matplotlib.path import Path as MplPath

    from hwsol.personal_ink.kernels.author_feedback_revision import identity, trace
    from hwsol.personal_ink.kernels.math_glyph_completion import SELECTION
    from hwsol.personal_ink.kernels.math_glyph_resolver import placed_outline
    from hwsol.personal_ink.kernels.render import accepted_scans
    from hwsol.personal_ink.validation import audit_new_contacts, capture

    if not any(
        row["context"] == "text"
        and row["semantic"] in (":", "'")
        and row["source_type"] in {"external-font", "inherited-font"}
        for row in records
    ):
        return records
    original_identity = identity(records)
    original_trace = copy.deepcopy(trace(renderer))
    before = capture(page, renderer, records)
    glyphs = {g.source["character"]: g for g in accepted_scans(bundle.directories["legacy"])}
    result = []
    for old in records:
        mark = {":": ":", "'": "′"}.get(old["semantic"])
        if (
            old["context"] != "text"
            or mark not in SELECTION
            or mark not in glyphs
            or old["source_type"] not in {"external-font", "inherited-font"}
        ):
            result.append(old)
            continue
        glyph = glyphs[mark]
        index = old["item_index"]
        item = page.items[index]
        bounds = list(old["bounds_px"])
        outline, transform = placed_outline(glyph.outline, bounds)
        key = ("neural-personal-punctuation", index)
        renderer._outlines[key] = outline
        page.items[index] = ("path", key, (1, 0, 0, 1, 0, 0), item[3], 0)
        points = outline.verts[outline.codes != MplPath.CLOSEPOLY]
        result.append(
            {
                **old,
                "source_type": "personal-scan-outline",
                "actual_glyph_id": glyph.source["id"],
                "glyph_name": None,
                "font": None,
                "font_sha256": None,
                "scan_source": glyph.source,
                "source_character": mark,
                "transformation": "reviewed personal punctuation in existing text slot",
                "personal_collected": True,
                "external_fallback": False,
                "procedural_geometry": False,
                "bounds_px": [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()],
                "original_bounds_px": old["bounds_px"],
                "allowed_bounds_px": bounds,
                "original_source_type": old["source_type"],
                "original_glyph_id": old["actual_glyph_id"],
            }
        )
    contacts = audit_new_contacts(before, page, renderer, result)
    if contacts["unexpected_contacts"]:
        raise ValueError("Personal punctuation introduced unreviewed contacts")
    if identity(result) != original_identity or trace(renderer) != original_trace:
        raise ValueError("Personal punctuation changed identity or formula structure")
    return result
