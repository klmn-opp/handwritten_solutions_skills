"""Bounded authentic historical pi-source audit, with unchanged RGB and labels.

Recognition supplies locators only. No contour is thinned, repainted or relabeled
in the source corpus. A source review precedes any isolated rendering candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus_access import catalogue
from research.missing_math_review import CATALOGS, contact_sheets, extract

ROOT = Path(__file__).resolve().parents[1]
BASE = "20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6"
PI = re.compile(r"\\pi(?![A-Za-z])|π")


def located_regions(catalogues):
    """Keep raw locator records, original statuses and source-group identities."""
    regions = {}
    counts = []
    for cat in catalogues:
        documents = {x["source_hash"]: x for x in cat["documents"]}
        pages = {x["page_id"]: x for x in cat["pages"]}
        samples = {x["sample_id"]: x for x in cat["samples"]}
        hits = Counter()

        def add(sample, label, locator, *, pages=pages, documents=documents, cat=cat):
            if not PI.search(label or ""):
                return
            page = pages[sample["page_id"]]
            key = (sample["source_hash"], page["page_number"], *sample["bounding_box"])
            if key not in regions:
                regions[key] = {
                    "source_hash": sample["source_hash"],
                    "document": documents[sample["source_hash"]],
                    "page": page,
                    "sample": sample,
                    "catalog_path": cat["path"],
                    "targets": ["π"],
                    "kind": "catalog-region",
                    "locators": [],
                    "review_status": "pending-agent-visual-review",
                    "candidate_label_is_truth": False,
                }
            regions[key]["locators"].append({"original_candidate_label": label, **locator})

        for rec in cat["recognitions"]:
            if PI.search(rec.get("text") or ""):
                hits["recognition"] += 1
                add(samples[rec["sample_id"]], rec["text"], {"raw_recognition": rec})
        for sample in cat["samples"]:
            for field in ("verified_latex", "verified_text", "candidate_text"):
                if PI.search(sample.get(field) or ""):
                    hits[field] += 1
                    add(sample, sample[field], {"sample_field": field})
        counts.append(
            {
                "path": cat["path"],
                "sha256": cat["sha256"],
                "allowed_documents": len(documents),
                "allowed_samples": len(samples),
                "allowed_recognitions": len(cat["recognitions"]),
                "matching_labels": dict(hits),
                "filter": cat["filter"],
            }
        )
    rows = sorted(
        regions.values(),
        key=lambda r: (
            "phy" not in r["document"]["original_name"],
            r["document"]["original_name"],
            r["page"]["page_number"],
            r["sample"]["bounding_box"],
        ),
    )
    return rows, counts


def source_audit(context=False):
    catalogues = [catalogue(path) for path in CATALOGS]
    rows, counts = located_regions(catalogues)
    if context:
        selected = []
        for cat in catalogues:
            samples = {s["sample_id"]: s for s in cat["samples"]}
            pages = {p["page_id"]: p for p in cat["pages"]}
            docs = {d["source_hash"]: d for d in cat["documents"]}
            sample = samples.get("s-4cef13386bf7a8f78603876d")
            if sample:
                row = {
                    "source_hash": sample["source_hash"],
                    "document": docs[sample["source_hash"]],
                    "page": pages[sample["page_id"]],
                    "sample": sample,
                    "catalog_path": cat["path"],
                    "targets": ["π"],
                    "kind": "existing-period-formula-locator",
                    "locators": [
                        {"raw_recognition": r}
                        for r in cat["recognitions"]
                        if r["sample_id"] == sample["sample_id"]
                    ],
                    "review_status": "pending-agent-visual-review",
                    "candidate_label_is_truth": False,
                }
                selected.append(row)
                page_context = copy.deepcopy(row)
                page_context["kind"] = "same-unsealed-page-context"
                page_context["sample"]["bounding_box"] = [
                    0,
                    0,
                    row["page"]["width"],
                    row["page"]["height"],
                ]
                page_context["original_locator_sample"] = copy.deepcopy(sample)
                selected.append(page_context)
        if len(selected) != 2:
            raise ValueError(
                "Expected one allowed existing period locator and its same-page context"
            )
        rows = selected
    if len(rows) > 12:
        raise ValueError("Source audit exceeds preregistered twelve-region budget")
    run = Run.create(
        "pi-historical-context-audit" if context else "pi-historical-source-audit",
        config={
            "purpose": "Original RGB source review; no inference or training",
            "rendering_baseline": BASE,
            "region_budget": 12,
            "accepted_candidate_budget": 4,
            "recognition_is_ground_truth": False,
            "new_neural_updates": 0,
            "context_followup": context,
        },
        seed=0,
    )
    path = run.path
    (path / "logs/source").mkdir()
    for source in (Path(__file__), ROOT / "research/missing_math_review.py"):
        shutil.copyfile(source, path / "logs/source" / source.name)
    atomic_json(path / "reports/search-scope.json", counts)
    atomic_json(path / "reports/raw-locator-regions.json", rows)
    extracted = []
    for index, original in enumerate(rows, 1):
        row = copy.deepcopy(original)
        row["review_id"] = f"pi-region-{index:02d}"
        row["original_sample_record"] = copy.deepcopy(row["sample"])
        # Context padding is recorded separately and never written to the corpus.
        x, y, right, bottom = row["sample"]["bounding_box"]
        row["sample"]["bounding_box"] = [
            max(0, x - 18),
            max(0, y - 18),
            min(row["page"]["width"], right + 18),
            min(row["page"]["height"], bottom + 18),
        ]
        row["crop_region_context_padding_catalog_pixels"] = 18
        extracted.append(extract(row, path))
    atomic_json(path / "reports/source-regions.json", extracted)
    contact_sheets(extracted, path)
    result = {
        "unique_contextual_regions": len(rows),
        "source_documents": len({r["source_hash"] for r in rows}),
        "by_document": dict(Counter(r["document"]["original_name"] for r in rows)),
        "accepted_glyphs": 0,
        "review": "pending; these OCR/previous-label locations are not new ground truth",
        "all_source_crops_rgb_300dpi": all(
            r["lineage"].get("render_dpi") == 300 for r in extracted
        ),
        "source_pixels_edited": False,
        "production_changed": False,
    }
    atomic_json(path / "reports/result.json", result)
    run.finish("prepared", result=result)
    print(path)


def source_review():
    """Archive genuine connected 2pi ink, refusing to invent an isolated pi."""
    first = ROOT / "workspace/runs/20261120T123003181086Z-pi-historical-source-audit-a8e07a8b"
    context = ROOT / "workspace/runs/20261120T123149898019Z-pi-historical-context-audit-1f44e860"
    old = ROOT / "workspace/runs/20261120T120009782106Z-numeral-pi-source-audit-f0f658a2"
    for source in (first, context, old):
        verify_run(source)
    raw = json.loads((context / "reports/source-regions.json").read_text())
    parent = raw[0]
    source_pdf = context / parent["original_copy"]
    if sha256(source_pdf) != parent["source_hash"]:
        raise ValueError("Original PDF hash mismatch")
    # Recheck the nested split filters before reading any new image derivatives.
    cats = [catalogue(p) for p in CATALOGS]
    safe, counts = located_regions(cats)
    previous = json.loads((first / "reports/raw-locator-regions.json").read_text())
    def key(r):
        return (r["source_hash"], r["page"]["page_number"], *r["sample"]["bounding_box"])
    if {key(r) for r in safe} != {key(r) for r in previous}:
        raise ValueError("Initial image review includes a newly forbidden source")
    run = Run.create(
        "pi-connected-source-review",
        config={
            "purpose": "Source-only No-Go: connected authentic 2pi is not an isolated pi",
            "baseline": BASE,
            "source_regions_reviewed": 10,
            "maximum_new_glyph_candidates": 4,
            "new_neural_updates": 0,
            "no_rendered_homework_expected": True,
        },
        seed=0,
    )
    path = run.path
    (path / "logs/source").mkdir()
    shutil.copyfile(Path(__file__), path / "logs/source/pi_source_feedback.py")
    evidence = path / "solutions/source-evidence"
    evidence.mkdir()
    shutil.copyfile(source_pdf, evidence / "source-original.pdf")
    shutil.copyfile(context / parent["crop"], evidence / "period-formula-original-rgb.png")
    shutil.copyfile(context / raw[1]["crop"], evidence / "page-55-original-rgb.png")
    for filename in ("glyph.png", "parent.png", "source.json"):
        shutil.copyfile(old / "solutions/pi-source" / filename, evidence / ("current-" + filename))
    image = Image.open(context / parent["crop"]).convert("RGB")
    box = [165, 100, 240, 170]
    token = image.crop(box)
    token.save(evidence / "connected-2pi-original-rgb.png", dpi=(300, 300))
    token.resize((600, 560), Image.Resampling.NEAREST).save(
        path / "comparisons/connected-2pi-8x-nearest.png"
    )
    gray = np.asarray(token.convert("L"))
    components = []
    for level in (120, 160, 180, 200):
        number, labels, stats, _ = cv2.connectedComponentsWithStats(
            (gray < level).astype(np.uint8), 8
        )
        if number != 3:
            raise ValueError("Expected two full token components, with grid and neighbor absent")
        components.append(
            {"luminance_threshold": level, "components_xywh_area": stats[1:].tolist()}
        )
        if level == 180:
            diagram = np.full((*labels.shape, 3), 255, dtype=np.uint8)
            diagram[labels == 1] = (180, 50, 45)
            diagram[labels == 2] = (25, 100, 185)
            Image.fromarray(diagram).resize((600, 560), Image.Resampling.NEAREST).save(
                path / "comparisons/connected-components-180-diagnostic.png"
            )
    x, y, _, _ = box
    X, Y, _, _ = parent["box_in_new_raster"]
    matrix = np.asarray(parent["lineage"]["crop_to_pdf"]) @ np.array(
        [[1, 0, x], [0, 1, y], [0, 0, 1]]
    )
    corners = [matrix @ [u, v, 1] for u, v in ((0, 0), (token.width, token.height))]
    record = {
        "actual_agent_transcription": "2π",
        "label_basis": "visible unchanged original ink",
        "original_recognition_labels_unchanged": parent["locators"],
        "original_sample_record_unchanged": parent["original_sample_record"],
        "source_sha256": parent["source_hash"],
        "page_number": 55,
        "source_pdf": "solutions/source-evidence/source-original.pdf",
        "rgb_crop": "solutions/source-evidence/connected-2pi-original-rgb.png",
        "rgb_crop_sha256": sha256(evidence / "connected-2pi-original-rgb.png"),
        "render_dpi": 300,
        "original_pixels_edited": False,
        "box_in_parent_300dpi": box,
        "box_in_full_page_300dpi": [X + x, Y + y, X + box[2], Y + box[3]],
        "crop_to_pdf": matrix.tolist(),
        "box_pdf_points": [
            float(min(corners[0][0], corners[1][0])),
            float(min(corners[0][1], corners[1][1])),
            float(max(corners[0][0], corners[1][0])),
            float(max(corners[0][1], corners[1][1])),
        ],
        "parent_source_record": parent,
        "pixel_components": components,
        "component_review": (
            "Component 1 contains entire numeral 2 plus pi left leg and top; "
            "component 2 is pi right leg. Connection remains at all four diagnostic thresholds."
        ),
        "grid_handling": (
            "No raw pixel edits. Existing grid is lighter than the four diagnostic thresholds; "
            "no removal or morphology applied."
        ),
        "isolated_pi_extract_decision": "no-go: would cut shared authentic ink",
        "whole_2pi_readability": "high-confidence Agent transcription; not approved for rendering",
        "root_scope_decision": "Do not expand to multi-character composition in this round",
        "author_review": False,
    }
    atomic_json(path / "reports/connected-source.json", record)
    atomic_json(path / "reports/hardened-search-scope.json", counts)
    atomic_json(
        path / "reports/parent-runs.json",
        [{"path": str(p), "run_sha256": sha256(p / "run.json")} for p in (first, context, old)],
    )
    review = [
        {
            "region": "first-01",
            "source": "phy2 p55 upper period formula",
            "status": "existing true heavy pi; not new",
        },
        {"region": "first-02", "source": "as p68", "status": "p parameter; false OCR pi locator"},
        {
            "region": "first-03",
            "source": "cal3 p123",
            "status": "integral upper x; false OCR pi locator",
        },
        {
            "region": "first-04",
            "source": "cal3 p124 upper region",
            "status": "Chinese prose; false OCR pi locator",
        },
        {
            "region": "first-05",
            "source": "cal3 p124 lower region",
            "status": "integral upper x; false OCR pi locator",
        },
        {
            "region": "first-06/07/08",
            "source": "ics p17",
            "status": "Condition/True/False prose; false OCR pi locators",
        },
        {
            "region": "context-01",
            "source": "phy2 p55 lower period formula",
            "status": "authentic lighter pi in connected 2pi; isolated extraction no-go",
        },
        {
            "region": "context-02",
            "source": "phy2 p55 full page",
            "status": (
                "same page context confirms both period formulas; "
                "not another independent observation"
            ),
        },
    ]
    atomic_json(path / "reports/visual-review.json", review)
    result = {
        "decision": "No-Go isolated pi replacement; retain current original",
        "reliable_new_isolated_pi": 0,
        "genuine_connected_2pi_observations": 1,
        "initial_ocr_hits": 11,
        "initial_unique_regions": 8,
        "additional_context_views": 2,
        "distinct_source_documents_viewed": 4,
        "not_exhaustive_historical_pdf_search": True,
        "no_source_label_or_image_changed": True,
        "no_training_or_rendering": True,
        "next_input": (
            "Await natural pi samples in already delivered O-FULL collection; "
            "do not demand altered handwriting"
        ),
    }
    atomic_json(path / "reports/result.json", result)
    (path / "reports/REVIEW.md").write_text(
        "# Third-round pi source review\n\n"
        "No-Go for a new isolated π replacement. The existing source is heavy. A second, "
        "visibly lighter π was found in the lower period formula on the same authentic physics "
        "page 55, despite OCR omitting it. Its left leg is physically joined to the preceding 2. "
        "Selecting an isolated π would cut authentic ink; no such contour was created.\n\n"
        "The unchanged full 2π token is preserved with original RGB, PDF SHA, 300 DPI crop "
        "coordinates, PDF transforms, raw OCR, original review status and separate new Agent "
        "transcription. Four diagnostic thresholds (120/160/180/200) all keep the 2 and π-left/top "
        "in one connected component; π-right is a separate second component. Diagnostics do not "
        "modify the source.\n\n"
        "The initial 11 π OCR hits collapse to eight regions across four documents. Seven are "
        "false π locators (x, p, Chinese prose, Condition/True/False); the eighth is the current "
        "heavy π. Two additional views cover the other pre-existing period-formula location "
        "and its same-page context. This is a bounded local indexed-source review, not exhaustive "
        "inspection of all historical PDF pages.\n\n"
        "Nested legacy holdout membership was discovered under derivation.selection.split for "
        "physics1 pages 15/25. No physics1 images were opened. The hardened query now excludes "
        "that entire source before labels, alongside sealed Session C and blocked ancestry. "
        "Its remaining π region keys exactly match the eight previously viewed sources.\n\n"
        "Root explicitly kept this round out of multi-character 2π composition because it needs "
        "a new provenance/layout contract, would change an already approved numeral 2, and "
        "would not cover every π context. No candidate rendering, fake placeholder PDF, pixel "
        "cleanup or neural update was performed. Current π remains honestly unresolved pending "
        "new natural handwriting in the already delivered O-FULL booklet.\n"
    )
    run.finish("held", result=result)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "context", "review"])
    args = parser.parse_args()
    if args.action == "review":
        source_review()
    else:
        source_audit(args.action == "context")


if __name__ == "__main__":
    main()
