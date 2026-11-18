"""Conservative raw-ink proposals and explicitly unverified symbol publication.

Detection masks locate connected ink only. Saved crops always use unchanged source pixels.
No character identity is inferred from a parent transcription or from component order.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from research.corpus.ingest import png
from research.corpus.store import Corpus, digest, validate_bbox

METHOD = "isolated-ink-components/1"


def isolated_boxes(image: Image.Image) -> tuple[list[list[int]], dict]:
    """Deliberately miss uncertain joins/dotted glyphs rather than cut them apart."""
    gray = np.asarray(image.convert("L"))
    mask = (gray < 130).astype(np.uint8)
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result, excluded = [], Counter()
    height, width = mask.shape
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area < 12 or h < 10 or w < 3:
            excluded["tiny-or-short-stroke"] += 1
            continue
        if w / h > 1.6 or h / w > 12 or h > 90:
            excluded["joined-or-structure-stroke"] += 1
            continue
        pad = 3
        if x < pad or y < pad or x + w + pad > width or y + h + pad > height:
            excluded["touches-parent-boundary"] += 1
            continue
        box = [x - pad, y - pad, x + w + pad, y + h + pad]
        local = components[box[1] : box[3], box[0] : box[2]]
        if np.any((local != 0) & (local != i)):
            excluded["neighbor-ink-in-padding"] += 1
            continue
        # Detached dot/mark above or below an overlapping stem: do not emit half an i/j.
        companion = False
        for j in range(1, n):
            if j == i:
                continue
            u, v, a, b, mass = map(int, stats[j])
            gap = max(y - (v + b), v - (y + h), 0)
            if mass >= 3 and min(x + w, u + a) > max(x, u) and gap < max(5, h * 0.6):
                companion = True
                break
        if companion:
            excluded["detached-companion-or-stack"] += 1
            continue
        result.append(box)
    return sorted(result, key=lambda b: (b[1], b[0])), {
        "method": METHOD,
        "threshold": 130,
        "padding": 3,
        "excluded": dict(excluded),
        "limitation": "A component may still be several joined characters; visual review required",
    }


def verified_parent(corpus: Corpus, sample_id: str, revision: int | None = None) -> dict:
    parent = corpus.get("sample", sample_id)
    if revision is not None and parent["revision"] != revision:
        raise ValueError("Parent review changed; inspect the proposal again")
    if (
        parent["review_status"] != "verified"
        or not (parent.get("review_attestation") or {}).get("human_review")
        or parent["quality_status"] != "clean"
        or parent["defects"]
        or parent["region_role"] != "personal-handwriting"
        or not parent["privacy_checked"]
        or parent["writer_provenance"]["domain"] != "personal-real"
        or parent["writer_provenance"]["is_generated"]
    ):
        raise ValueError("A verified clean personal parent is required")
    return parent


def prepare(corpus: Corpus, parent_ids: list[str], destination: Path) -> dict:
    """Dry proposal run: no new sample or label enters the corpus yet."""
    destination.mkdir(parents=True, exist_ok=False)
    proposals, reports = [], []
    for sid in dict.fromkeys(parent_ids):
        parent = verified_parent(corpus, sid)
        page = corpus.get("page", parent["page_id"])
        with Image.open(corpus.path(parent["image_crop"])) as im:
            boxes, diagnostics = isolated_boxes(im)
        reports.append({"parent_id": sid, **diagnostics})
        with Image.open(corpus.path(page["color_image"])) as page_image:
            for local in boxes:
                x, y = parent["bounding_box"][:2]
                box = [local[0] + x, local[1] + y, local[2] + x, local[3] + y]
                key = "symbol-" + digest([page["page_id"], box, METHOD])[:24]
                data = png(page_image.crop(box))
                (destination / f"{key}.png").write_bytes(data)
                proposals.append(
                    {
                        "proposal_id": key,
                        "page_id": page["page_id"],
                        "bbox": box,
                        "source_image_sha256": Path(page["color_image"]).stem,
                        "crop_sha256": hashlib.sha256(data).hexdigest(),
                        "image": str((destination / f"{key}.png").resolve()),
                        "parent_id": sid,
                        "parent_revision": parent["revision"],
                        "candidate_label": None,
                        "method": METHOD,
                        "is_verified": False,
                    }
                )
    report = {"proposals": proposals, "parent_diagnostics": reports, "labels_inferred": False}
    (destination / "proposals.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def publish(corpus: Corpus, selections: list[dict], *, batch: str, annotator: str) -> dict:
    """Publish visually selected raw crops as candidates, with no inherited attestation."""
    if not batch.strip() or not annotator.strip() or len(selections) > 200:
        raise ValueError("Named batch/annotator and at most 200 selections required")
    prepared = []
    for selection in selections:
        label = selection["candidate_label"]
        if not isinstance(label, str) or len(label) != 1 or label.isspace():
            raise ValueError("Select one visible character; not a word or LaTeX command")
        page = corpus.get("page", selection["page_id"])
        if (
            page["writer_provenance"]["domain"] != "personal-real"
            or page["writer_provenance"]["is_generated"]
        ):
            raise ValueError("Only original personal source pages are accepted")
        box = validate_bbox(selection["bbox"], page["width"], page["height"])
        if selection["source_image_sha256"] != Path(page["color_image"]).stem:
            raise ValueError("Source image digest mismatch")
        parent = None
        if selection.get("parent_id"):
            parent = verified_parent(corpus, selection["parent_id"], selection["parent_revision"])
            a, b, c, d = parent["bounding_box"]
            x, y, X, Y = box
            if parent["page_id"] != page["page_id"] or not (a <= x < X <= c and b <= y < Y <= d):
                raise ValueError("Child crop must stay inside its reviewed parent")
        with Image.open(corpus.path(page["color_image"])) as image:
            data = png(image.crop(box))
        if hashlib.sha256(data).hexdigest() != selection["crop_sha256"]:
            raise ValueError("Crop differs from unchanged original pixels")
        prepared.append((selection, page, box, parent))
    created, skipped = [], []
    with corpus.db() as db:
        # One transaction also prevents another review racing the parent check and creation.
        existing = [
            json.loads(r[0]) for r in db.execute("SELECT data FROM records WHERE kind='sample'")
        ]
        for selection, page, box, parent in prepared:
            if parent and corpus._get(db, "sample", parent["sample_id"]) != parent:
                raise ValueError("Parent changed during publication")
            duplicate = None
            for old in existing:
                if old["page_id"] != page["page_id"] or old["content_type"] not in {
                    "glyph",
                    "math-symbol",
                }:
                    continue
                a, b, c, d = old["bounding_box"]
                x, y, X, Y = box
                intersection = max(0, min(c, X) - max(a, x)) * max(0, min(d, Y) - max(b, y))
                if intersection / min((c - a) * (d - b), (X - x) * (Y - y)) >= 0.65:
                    duplicate = old["sample_id"]
                    break
            if duplicate:
                skipped.append(
                    {"selection": selection, "reason": "same-physical-ink", "existing": duplicate}
                )
                continue
            sample = corpus._propose(
                db,
                page["page_id"],
                box,
                content_type="glyph",
                label=selection["candidate_label"],
                label_source="symbol-preparation/1",
                role="unknown",
                parents=[parent["sample_id"]] if parent else [],
                notes="助手提出的单字符候选；请核对完整边界、字符身份及书写者。不是人工确认。",
                derivation={
                    "batch": batch,
                    "annotator": annotator,
                    "method": selection.get("method", "assistant-visual-proposal"),
                    "source_image_sha256": selection["source_image_sha256"],
                    "parent_revision": parent["revision"] if parent else None,
                    "selection": selection,
                    "label_status": "unverified-assistant-suggestion",
                },
            )
            existing.append(sample)
            created.append(sample)
    return {"created": created, "skipped": skipped, "automatic_verification": False}
