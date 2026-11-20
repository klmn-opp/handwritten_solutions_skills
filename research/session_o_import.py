"""Immutable, direct-PDF import of O-FULL; prompt text never becomes ink truth.

Only the explicitly pinned new source and its blank collection template are
opened. No historical corpus catalogue, OCR service or training dependency is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_pdf/session-O_full_source.pdf"
SOURCE_SHA = "23a84ff4cf5dc4882b5fb76d5eb4cc3a32c59e9799ea7a351f545598401f4d43"
TEMPLATE = ROOT / (
    "workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3"
)
PURE_LIST_ROWS = {
    1: [1, 5],
    2: [1, 2, 3, 4, 5],
    3: [1, 3, 4, 6],
    7: [1, 2, 3, 4, 5, 7],
    8: [1, 3, 4],
    9: [1, 2, 3, 4, 6],
    10: [1, 2, 3, 4, 5, 6, 7],
}


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def render_source() -> Path:
    """Copy the original once and render every page without registering or cleaning ink."""
    if sha256(SOURCE) != SOURCE_SHA:
        raise ValueError("Uploaded O-FULL PDF does not match the explicitly approved source SHA")
    verify_run(TEMPLATE)
    run = Run.create(
        "session-o-full-source-rgb-import",
        config={
            "purpose": "Direct immutable RGB page import; no accepted labels or training",
            "source": str(SOURCE),
            "source_sha256": SOURCE_SHA,
            "template_run": str(TEMPLATE),
            "template_run_sha256": sha256(TEMPLATE / "run.json"),
            "dpi": 300,
            "expected_pages": 19,
            "expected_prompt_rows": 123,
            "no_corpus_database_access": True,
            "new_neural_updates": 0,
        },
        seed=0,
    )
    path = run.path
    print(path, flush=True)
    try:
        (path / "logs/source").mkdir()
        shutil.copyfile(Path(__file__), path / "logs/source/session_o_import.py")
        original = path / "solutions/session-O_full_source.pdf"
        shutil.copyfile(SOURCE, original)
        shutil.copyfile(
            TEMPLATE / "solutions/collection/catalog.json", path / "solutions/prompt-catalog.json"
        )
        shutil.copyfile(
            TEMPLATE / "solutions/collection/session-O_full.pdf",
            path / "solutions/blank-template.pdf",
        )
        (path / "solutions/pages").mkdir()
        (path / "comparisons/pages").mkdir()
        pages = []
        with pdfium.PdfDocument(original) as document:
            if len(document) != 19:
                raise ValueError("Unexpected O-FULL page count")
            for index in range(len(document)):
                number = index + 1
                with (
                    closing(document[index]) as page,
                    closing(page.render(scale=300 / 72)) as bitmap,
                ):
                    rgb = bitmap.to_pil().convert("RGB")
                    conv = bitmap.get_posconv(page)
                    origin = np.asarray(conv.to_page(0, 0))
                    dx = (np.asarray(conv.to_page(rgb.width, 0)) - origin) / rgb.width
                    dy = (np.asarray(conv.to_page(0, rgb.height)) - origin) / rgb.height
                    transform = np.array([np.r_[dx, 0], np.r_[dy, 0], np.r_[origin, 1]]).T
                    out = path / f"solutions/pages/page-{number:02d}.png"
                    rgb.save(out, dpi=(300, 300))
                    preview = rgb.copy()
                    preview.thumbnail((1240, 1754))
                    preview.save(path / f"comparisons/pages/page-{number:02d}.png")
                    pages.append(
                        {
                            "page": number,
                            "source_sha256": SOURCE_SHA,
                            "source_pdf": "solutions/session-O_full_source.pdf",
                            "rgb_page": str(out.relative_to(path)),
                            "rgb_page_sha256": sha256(out),
                            "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                            "rgb_size": list(rgb.size),
                            "render_dpi": 300,
                            "pdf_page_size_pt": list(page.get_size()),
                            "pdf_rotation": page.get_rotation(),
                            "pdf_crop_box": list(page.get_cropbox()),
                            "raster_to_pdf": transform.tolist(),
                            "pdf_to_raster": np.linalg.inv(transform).tolist(),
                            "pixels_modified": False,
                            "label_review_status": "pending",
                        }
                    )
                atomic_json(path / "reports/pages.json", pages)
                print(f"RGB page {number:02d}/19", flush=True)
        overview(path, pages)
        result = {
            "pages": 19,
            "original_sha256": sha256(original),
            "all_rgb_300dpi": True,
            "raw_page_authoritative": True,
            "verified_handwriting_labels": 0,
            "scope": "Source-only; row indexing and actual handwriting review follow separately",
            "no_database_or_ocr_access": True,
            "no_neural_updates": True,
        }
        atomic_json(path / "reports/result.json", result)
        if sha256(SOURCE) != SOURCE_SHA or sha256(original) != SOURCE_SHA:
            raise ValueError("Original source changed during import")
        run.finish("prepared", result=result)
        verify_run(path)
        return path
    except Exception as exc:
        run.fail(exc)
        raise


def overview(path: Path, pages: list[dict]) -> None:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for offset in range(0, len(pages), 4):
        panel = Image.new("RGB", (1280, 1880), "#eeeeee")
        draw = ImageDraw.Draw(panel)
        for slot, row in enumerate(pages[offset : offset + 4]):
            x, y = (slot % 2) * 640, (slot // 2) * 940
            im = Image.open(path / row["rgb_page"]).convert("RGB")
            im.thumbnail((620, 880))
            draw.text(
                (x + 12, y + 8), f"O-FULL original page {row['page']:02d}", font=font, fill="black"
            )
            panel.paste(im, (x + 10, y + 48))
        panel.save(path / f"comparisons/overview-{offset // 4 + 1:02d}.png")


def pixel_box(pdf_box: list[float], transform: np.ndarray, size: tuple[int, int]) -> list[int]:
    corners = np.asarray(
        [transform @ [x, y, 1] for x in (pdf_box[0], pdf_box[2]) for y in (pdf_box[1], pdf_box[3])]
    )[:, :2]
    left, top = np.floor(corners.min(axis=0)).astype(int)
    right, bottom = np.ceil(corners.max(axis=0)).astype(int)
    return [
        max(0, int(left)),
        max(0, int(top)),
        min(size[0], int(right)),
        min(size[1], int(bottom)),
    ]


def row_registration(
    scan: Image.Image, blank: Image.Image, page: dict, catalog: dict
) -> list[dict]:
    """Check printed row IDs only; no handwriting pixels are warped or subtracted."""
    original = np.asarray(scan.convert("L"))
    template = np.asarray(blank.convert("L"))
    matrix = np.asarray(page["pdf_to_raster"])
    results = []
    for row in catalog["rows"]:
        b = row["label_bounds_pdf_pt"]
        box = pixel_box([b[0] - 1, b[1] - 1, min(143, b[2]), b[3] + 1], matrix, scan.size)
        x, y, right, bottom = box
        padding = 8
        patch = template[y:bottom, x:right]
        search = original[y - padding : bottom + padding, x - padding : right + padding]
        response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        offset = [int(location[0] - padding), int(location[1] - padding)]
        results.append(
            {
                "row_id": row["id"],
                "printed_id_box_px": box,
                "offset_px": offset,
                "normalized_correlation": score,
                "mapping_checked": score >= 0.95 and max(map(abs, offset)) <= 3,
                "purpose": (
                    "Printed locator geometry only; no label or handwriting quality inference"
                ),
            }
        )
    return results


def save_crop(
    path: Path, image: Image.Image, box: list[int], destination: Path, matrix: np.ndarray
) -> dict:
    crop = image.crop(box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    crop.save(destination, dpi=(300, 300))
    transform = matrix @ np.array([[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]])
    return {
        "path": str(destination.relative_to(path)),
        "sha256": sha256(destination),
        "decoded_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
        "box_page_px": box,
        "size_px": list(crop.size),
        "crop_to_pdf": transform.tolist(),
        "render_dpi": 300,
        "pixels_equal_original_page_rectangle": True,
        "pixels_modified": False,
    }


def row_contacts(path: Path, page: int, rows: list[dict]) -> Image.Image:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    cell_width, pad, heading = 1240, 18, 58
    thumbnails = []
    for row in rows:
        im = Image.open(path / row["context_crop"]["path"]).convert("RGB")
        im.thumbnail((cell_width - 2 * pad, 800))
        thumbnails.append(im)
    heights = [
        max(im.height for im in thumbnails[k : k + 2]) + heading + pad
        for k in range(0, len(rows), 2)
    ]
    panel = Image.new("RGB", (2 * cell_width, sum(heights) + 10), "#eeeeee")
    draw = ImageDraw.Draw(panel)
    y = 0
    for k in range(0, len(rows), 2):
        for j, (row, im) in enumerate(zip(rows[k : k + 2], thumbnails[k : k + 2], strict=True)):
            x = j * cell_width
            draw.text(
                (x + pad, y + 12),
                row["id"] + " | RAW context; label pending",
                font=font,
                fill="black",
            )
            panel.paste(im, (x + pad, y + heading))
        y += heights[k // 2]
    output = path / f"comparisons/row-contexts/page-{page:02d}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    return panel


def index_rows(parent: Path) -> Path:
    """Index 123 intended rows; wide overlapping RGB contexts keep overspill intact."""
    parent = parent.resolve()
    verify_run(parent)
    if sha256(parent / "solutions/session-O_full_source.pdf") != SOURCE_SHA:
        raise ValueError("Unexpected source import")
    catalog = read(parent / "solutions/prompt-catalog.json")
    pages = read(parent / "reports/pages.json")
    if len(pages) != 19 or sum(len(p["rows"]) for p in catalog["pages"]) != 123:
        raise ValueError("Unexpected source/template inventory")
    run = Run.create(
        "session-o-full-row-source-index",
        config={
            "purpose": "123 prompt-located raw row candidates; actual labels not inferred",
            "parent": str(parent),
            "parent_sha256": sha256(parent / "run.json"),
            "source_sha256": SOURCE_SHA,
            "dpi": 300,
            "whole_page_is_authoritative": True,
            "new_neural_updates": 0,
            "no_corpus_database_access": True,
        },
        seed=0,
    )
    path = run.path
    print(path, flush=True)
    try:
        shutil.copytree(parent / "solutions", path / "solutions", dirs_exist_ok=True)
        (path / "logs/source").mkdir()
        shutil.copyfile(Path(__file__), path / "logs/source/session_o_import.py")
        atomic_json(path / "reports/pages.json", pages)
        atomic_json(
            path / "reports/parent.json",
            {"path": str(parent), "sha256": sha256(parent / "run.json")},
        )
        all_rows, registrations, contacts = [], [], []
        with pdfium.PdfDocument(path / "solutions/blank-template.pdf") as template:
            for meta, spec in zip(pages, catalog["pages"], strict=True):
                number = meta["page"]
                image = Image.open(path / meta["rgb_page"]).convert("RGB")
                if sha256(path / meta["rgb_page"]) != meta["rgb_page_sha256"]:
                    raise ValueError("Original RGB page hash mismatch")
                with (
                    closing(template[number - 1]) as blank_page,
                    closing(blank_page.render(scale=300 / 72)) as bitmap,
                ):
                    blank = bitmap.to_pil().convert("RGB")
                    if (
                        list(blank_page.get_size()) != meta["pdf_page_size_pt"]
                        or blank.size != image.size
                    ):
                        raise ValueError(
                            "Template/source page geometry requires explicit new registration"
                        )
                checked = row_registration(image, blank, meta, spec)
                registrations.extend({"page": number, **r} for r in checked)
                if not all(r["mapping_checked"] for r in checked):
                    raise ValueError(f"Printed row locator registration uncertain on page {number}")
                delta = np.median([r["offset_px"] for r in checked], axis=0)
                if any(np.max(np.abs(np.asarray(r["offset_px"]) - delta)) > 1 for r in checked):
                    raise ValueError(
                        "Nonuniform row displacement; do not silently assume affine alignment"
                    )
                pdf_to_scan = np.array(
                    [[1, 0, delta[0]], [0, 1, delta[1]], [0, 0, 1]]
                ) @ np.asarray(meta["pdf_to_raster"])
                page_rows = []
                for i, row in enumerate(spec["rows"]):
                    following = spec["rows"][i + 1] if i + 1 < len(spec["rows"]) else None
                    width = meta["pdf_page_size_pt"][0]
                    # Context includes the current prompt and the following prompt's
                    # side space. Real extra repetitions were observed there; the
                    # overlap is explicit and is not an automatic row truth assignment.
                    context_pdf = [
                        0,
                        max(0, following["printed_prompt_bounds_pdf_pt"][1] - 14)
                        if following
                        else 0,
                        width,
                        row["label_bounds_pdf_pt"][3] + 5,
                    ]
                    band_pdf = [
                        0,
                        following["label_bounds_pdf_pt"][3] + 2 if following else 48,
                        width,
                        row["printed_prompt_bounds_pdf_pt"][1] - 3,
                    ]
                    context_box = pixel_box(context_pdf, pdf_to_scan, image.size)
                    band_box = pixel_box(band_pdf, pdf_to_scan, image.size)
                    # The lower overlap is context only: adjacent printed labels
                    # and source handwriting coexist and must be reviewed visually.
                    overflow_box = [context_box[0], band_box[3], context_box[2], context_box[3]]
                    destination = path / "solutions/rows" / row["id"]
                    entry = {
                        "id": row["id"],
                        "page": number,
                        "row_number": i + 1,
                        "source_sha256": SOURCE_SHA,
                        "source_pdf": "solutions/session-O_full_source.pdf",
                        "source_page": meta,
                        "prompt_catalog_record": row,
                        "intended_prompt_is_actual_label": False,
                        "actual_transcription": None,
                        "review_status": "pending-actual-ink-review",
                        "training_ready": False,
                        "source_group": "sha256:" + SOURCE_SHA,
                        "dataset_split": "unassigned",
                        "expected_repetition_count": None,
                        "glyph_segmentation_performed": False,
                        "printed_locator_registration": checked[i],
                        "template_pdf_to_source_raster": pdf_to_scan.tolist(),
                        "context_crop": save_crop(
                            path,
                            image,
                            context_box,
                            destination / "context-rgb.png",
                            np.asarray(meta["raster_to_pdf"]),
                        ),
                        "nominal_writing_band": save_crop(
                            path,
                            image,
                            band_box,
                            destination / "writing-band-rgb.png",
                            np.asarray(meta["raster_to_pdf"]),
                        ),
                        "overflow_context": save_crop(
                            path,
                            image,
                            overflow_box,
                            destination / "lower-overlap-rgb.png",
                            np.asarray(meta["raster_to_pdf"]),
                        ),
                        "context_contains_printed_material": True,
                        "overlap_with_next_row_is_intentional": bool(following),
                        "ink_outside_nominal_band_must_not_be_discarded": True,
                        "row_association_requires_review": True,
                    }
                    page_rows.append(entry)
                all_rows.extend(page_rows)
                contacts.append(row_contacts(path, number, page_rows))
                atomic_json(path / "reports/rows.json", all_rows)
                print(f"Indexed page {number:02d}; rows {len(all_rows)}/123", flush=True)
        atomic_json(path / "reports/printed-locator-registration.json", registrations)
        contacts[0].save(
            path / "comparisons/all-row-contexts.pdf",
            save_all=True,
            append_images=contacts[1:],
            resolution=180,
        )
        result = {
            "source_pages": 19,
            "prompt_located_rows": len(all_rows),
            "verified_handwriting_labels": 0,
            "raw_crop_files": 3 * len(all_rows),
            "all_raw_page_and_crop_rgb_300dpi": True,
            "printed_locator_min_correlation": min(
                r["normalized_correlation"] for r in registrations
            ),
            "maximum_locator_offset_px": max(max(map(abs, r["offset_px"])) for r in registrations),
            "whole_page_authoritative": True,
            "overlapping_contexts_preserve_extra_writing": True,
            "glyphs_accepted_for_rendering": 0,
            "neural_updates": 0,
        }
        atomic_json(path / "reports/result.json", result)
        (path / "reports/README.md").write_text(
            "# O-FULL raw row index\n\n"
            "The immutable original PDF and nineteen RGB pages are authoritative. All 123 catalog "
            "rows have original-color contextual crops, a nominal writing band and a lower-overlap "
            "context crop, with exact pixel/PDF transforms and hashes. The context intentionally "
            "includes printed prompts and may overlap its neighbor. Extra handwritten repetitions "
            "were visibly written below guide lines and beside following prompts; do not discard "
            "them or treat the nominal band as a complete glyph extraction.\n\n"
            "Printed row IDs were image-matched to the blank template to check geometry. No PDF "
            "text extraction/OCR, background removal, handwriting warping or corpus database "
            "access occurred. Printed mathematical prompts are intended content only: all actual "
            "transcriptions remain null and all labels remain pending independent ink review. "
            "Repetition counts are not assumed, and continuous expressions are not split.\n"
        )
        if (
            sha256(SOURCE) != SOURCE_SHA
            or sha256(parent / "run.json") != run.record()["config"]["parent_sha256"]
        ):
            raise ValueError("Protected source or parent changed")
        run.finish("prepared", result=result)
        verify_run(path)
        return path
    except Exception as exc:
        run.fail(exc)
        raise


def column_spans(band: Image.Image) -> list[tuple[int, int]]:
    """Unlabelled coarse spans separated by visible whitespace; never cut a dark path.

    The threshold and gap are fixed locator settings, not stroke editing. Full
    RGB rows remain authoritative, including faint marks and extra repetitions.
    """
    occupied = np.flatnonzero((np.asarray(band.convert("L")) < 180).any(axis=0))
    if not len(occupied):
        return []
    split = np.flatnonzero(np.diff(occupied) > 15) + 1
    groups = np.split(occupied, split)
    return [(max(0, int(g[0]) - 4), min(band.width, int(g[-1]) + 5)) for g in groups]


def native_column_contact(path: Path, row: dict, candidates: list[dict]) -> None:
    """Paste raw row and raw candidate crops at native resolution, without resampling."""
    band = Image.open(path / row["nominal_writing_band"]["path"]).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    columns, cell_width = 8, 315
    height = 70 + band.height + ((len(candidates) + columns - 1) // columns) * (band.height + 55)
    panel = Image.new("RGB", (max(band.width + 20, columns * cell_width), height + 12), "#eeeeee")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (12, 10),
        f"{row['id']} | {len(candidates)} unlabelled column regions | native pixels",
        font=font,
        fill="black",
    )
    panel.paste(band, (10, 54))
    for index, candidate in enumerate(candidates):
        x = (index % columns) * cell_width
        y = 70 + band.height + (index // columns) * (band.height + 55)
        crop = Image.open(path / candidate["crop"]["path"]).convert("RGB")
        if crop.width > cell_width - 16:
            # Such a connected span is preserved intact in its file and row view;
            # it is not squeezed or falsely presented as a complete isolated glyph.
            draw.text(
                (x + 8, y + 5), f"C{index + 1:02d}: wide span; see raw", font=font, fill="black"
            )
        else:
            draw.text(
                (x + 8, y + 5),
                f"C{index + 1:02d} x={candidate['crop']['box_page_px'][0]}",
                font=font,
                fill="black",
            )
            panel.paste(crop, (x + 8, y + 38))
    output = path / f"comparisons/native-columns/{row['id']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output, dpi=(300, 300))


def propose_columns(parent: Path) -> Path:
    """Make unlabelled geometry proposals; mixed/priority rows remain whole-row sources."""
    parent = parent.resolve()
    verify_run(parent)
    if sha256(parent / "solutions/session-O_full_source.pdf") != SOURCE_SHA:
        raise ValueError("Wrong O-FULL source")
    rows = read(parent / "reports/rows.json")
    selected = [r for r in rows if r["row_number"] in PURE_LIST_ROWS.get(r["page"], [])]
    selected_ids = {r["id"] for r in selected}
    run = Run.create(
        "session-o-unlabelled-symbol-column-proposals",
        config={
            "purpose": "Raw unlabelled column regions for later actual-ink review",
            "parent": str(parent),
            "parent_sha256": sha256(parent / "run.json"),
            "source_sha256": SOURCE_SHA,
            "selected_rows": sorted(selected_ids),
            "locator_luminance_threshold": 180,
            "minimum_blank_column_gap_px": 15,
            "priority_bank_recropped": False,
            "actual_labels_inferred": False,
            "no_neural_updates": True,
            "no_database_access": True,
        },
        seed=0,
    )
    path = run.path
    print(path, flush=True)
    try:
        shutil.copytree(parent / "solutions", path / "solutions", dirs_exist_ok=True)
        (path / "logs/source").mkdir()
        shutil.copyfile(Path(__file__), path / "logs/source/session_o_import.py")
        atomic_json(path / "reports/source-rows.json", rows)
        atomic_json(
            path / "reports/parent.json",
            {"path": str(parent), "sha256": sha256(parent / "run.json")},
        )
        candidates, groups = [], []
        for row in selected:
            band = Image.open(path / row["nominal_writing_band"]["path"]).convert("RGB")
            page = Image.open(path / row["source_page"]["rgb_page"]).convert("RGB")
            X, Y, _, B = row["nominal_writing_band"]["box_page_px"]
            proposals = []
            for index, (left, right) in enumerate(column_spans(band), 1):
                box = [X + left, Y, X + right, B]
                crop = save_crop(
                    path,
                    page,
                    box,
                    path / f"solutions/column-candidates/{row['id']}-C{index:02d}.png",
                    np.asarray(row["source_page"]["raster_to_pdf"]),
                )
                raw = np.asarray(page.crop(box).convert("L"))
                near_edge = bool((raw[:3] < 180).any() or (raw[-3:] < 180).any())
                proposal = {
                    "id": f"{row['id']}-C{index:02d}",
                    "row_id": row["id"],
                    "page": row["page"],
                    "candidate_ordinal": index,
                    "actual_label": None,
                    "actual_label_review": "pending-independent-ink-review",
                    "expected_prompt_context_only": row["prompt_catalog_record"]["intended_latex"],
                    "expected_slot": None,
                    "expected_repetition": None,
                    "source_group": row["source_group"],
                    "source_sha256": SOURCE_SHA,
                    "source_pdf": "solutions/session-O_full_source.pdf",
                    "source_page": row["source_page"],
                    "parent_context": row["context_crop"],
                    "parent_writing_band": row["nominal_writing_band"],
                    "crop": crop,
                    "proposal_type": "unlabelled-column-region-not-accepted-glyph",
                    "may_contain_multiple_symbols_or_components": True,
                    "near_nominal_vertical_boundary": near_edge,
                    "not_guaranteed_complete": True,
                    "training_ready": False,
                    "rendering_approved": False,
                    "boundary_in_empty_columns_at_locator_threshold": True,
                    "locator_warning": (
                        "Faint marks and beyond-band repetitions require the complete RGB row; "
                        "no content is discarded from the source layer"
                    ),
                    "guide_line_preserved_in_raw_crop": True,
                }
                proposals.append(proposal)
            candidates.extend(proposals)
            native_column_contact(path, row, proposals)
            groups.append(
                {
                    "row_id": row["id"],
                    "page": row["page"],
                    "candidate_count": len(proposals),
                    "candidate_ids": [r["id"] for r in proposals],
                    "native_contact": f"comparisons/native-columns/{row['id']}.png",
                    "parent_context": row["context_crop"],
                    "actual_labels": None,
                }
            )
            print(f"{row['id']}: {len(proposals)} unlabelled regions", flush=True)
        deferred = []
        for row in rows:
            if row["id"] not in selected_ids:
                deferred.append(
                    {
                        "id": row["id"],
                        "page": row["page"],
                        "parent_context": row["context_crop"],
                        "reason": (
                            "Priority-bank overlap, mixed formula/continuous structure, "
                            "or outside this bounded column-proposal scope"
                        ),
                        "status": "whole-row source preserved; not atomized; actual label pending",
                        "not_missing_from_corpus": True,
                    }
                )
        atomic_json(path / "reports/candidates.json", candidates)
        atomic_json(path / "reports/column-groups.json", groups)
        atomic_json(path / "reports/unatomized-rows.json", deferred)
        result = {
            "source_rows": 123,
            "column_proposal_rows": len(groups),
            "unlabelled_column_regions": len(candidates),
            "whole_unatomized_rows": len(deferred),
            "accepted_atomic_glyphs": 0,
            "accepted_actual_labels": 0,
            "all_crop_pixels_original_rgb": True,
            "new_neural_updates": 0,
            "not_an_independent_sample_count": True,
        }
        atomic_json(path / "reports/result.json", result)
        (path / "reports/README.md").write_text(
            "# O-FULL unlabelled symbol-column proposals\n\n"
            "These raw RGB crops are geometry proposals for independent review, not an accepted "
            "glyph bank. Candidate ordinals and the original intended prompt are separate fields; "
            "no expected symbol order, repetition count or Unicode identity becomes ground truth. "
            "Native contacts preserve pixel scale. All actual labels and approvals are empty.\n\n"
            "Only selected non-priority list rows were partitioned at broad empty columns. A "
            "continuous stroke is not cut into guessed characters. Multiple components can belong "
            "to one glyph, and a column span can still contain multiple symbols; review the whole "
            "row and original page. Light guide lines remain in every raw crop. Threshold 180 and "
            "the fixed blank-gap rule locate regions only; they do not clean or repaint pixels.\n\n"
            "All 123 original row contexts and 19 pages remain in this run. Mixed formulas, "
            "priority-symbol rows and other unatomized material are explicitly listed in "
            "unatomized-rows.json, including Greek letters sharing rows with the priority bank. "
            "No source was lost or implicitly approved. Pages 11–12 remain whole-row material "
            "because their symbol-plus-formula rows require semantic span review.\n"
        )
        if sha256(SOURCE) != SOURCE_SHA:
            raise ValueError("Protected original PDF changed")
        run.finish("prepared", result=result)
        verify_run(path)
        return path
    except Exception as exc:
        run.fail(exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "rows", "columns"])
    parser.add_argument("--parent", type=Path)
    args = parser.parse_args()
    if args.action == "render":
        render_source()
    else:
        if not args.parent:
            parser.error("rows requires --parent pointing to the immutable RGB import")
        if args.action == "rows":
            index_rows(args.parent)
        else:
            propose_columns(args.parent)


if __name__ == "__main__":
    main()
