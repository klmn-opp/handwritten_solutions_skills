"""Reference ink from the real Handnote path, including its selected glyph IDs."""

import math

import numpy as np
import pypdfium2 as pdfium

from hwsol.config import RenderConfig
from hwsol.jobs.paper import pdf_image
from research.corpus.context_composer import layout


def render(text: str, profile, *, raster_scale: int = 1, canonical_glyphs: bool = False):
    if type(raster_scale) is not int or not 1 <= raster_scale <= 8:
        raise ValueError("Reference raster scale must be integer 1 through 8")
    config = RenderConfig(seed=1010, font_size_pt=12, ink="#000000", semantic_layout=True, dpi=300)
    page, renderer, provenance = layout(
        text, "Reference", profile, config, canonical_glyphs=canonical_glyphs
    )
    body = provenance[len("Reference") :]
    expected = "".join(text.split())
    if "".join(r["semantic"] for r in body) != expected or not body:
        raise ValueError("Handnote reference semantic trace differs from exact input")
    boxes = np.array([r["bounds_px"] for r in body])
    crop = [
        math.floor(boxes[:, 0].min()) - 8,
        math.floor(boxes[:, 1].min()) - 8,
        math.ceil(boxes[:, 2].max()) + 8,
        math.ceil(boxes[:, 3].max()) + 8,
    ]
    cursor = 0
    word_boxes = []
    for word in text.split():
        part = boxes[cursor : cursor + len(word)]
        cursor += len(word)
        word_boxes.append(
            [
                float(part[:, 0].min()),
                float(part[:, 1].min()),
                float(part[:, 2].max()),
                float(part[:, 3].max()),
            ]
        )
    cuts = [
        (left[2] + right[0]) / 2 - crop[0]
        for left, right in zip(word_boxes, word_boxes[1:], strict=False)
    ]
    if any(b[0] >= c[0] or b[2] >= c[0] for b, c in zip(word_boxes, word_boxes[1:], strict=False)):
        raise ValueError("Reference wraps or word bounds overlap")
    blob = renderer.to_vector_pdf([page])
    with pdfium.PdfDocument(blob) as doc:
        image = pdf_image(doc, 0, config.dpi / 72 * raster_scale).crop(
            [v * raster_scale for v in crop]
        )
    return (
        image,
        [v * raster_scale for v in cuts],
        {
            "backend": "handnote-vector-reference/1",
            "seed": 1010,
            "dpi": 300,
            "crop_from_pdf": crop,
            "glyph_provenance": body,
            "word_bounds_page": word_boxes,
            "personal_profile_unchanged": True,
            "zero_neural_updates": True,
            **({"canonical_glyphs": True} if canonical_glyphs else {}),
            **(
                {
                    "raster_scale": raster_scale,
                    "raster_dpi": config.dpi * raster_scale,
                    "bounds_coordinate_dpi": config.dpi,
                }
                if raster_scale != 1
                else {}
            ),
        },
    )
