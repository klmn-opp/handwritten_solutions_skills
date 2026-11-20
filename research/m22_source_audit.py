"""Bounded original-source audit for the remaining universal/integer-set roles."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus_access import catalogue
from research.missing_math_review import CATALOGS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738"
ROWS = ROOT / "workspace/runs/20261120T134345155374Z-session-o-full-row-source-index-6886ffcf"
PATTERN = re.compile(r"∀|ℤ|\\forall\b|\\mathbb\s*\{\s*Z\s*\}")


def leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from leaves(child)


def main():
    verify_run(SOURCE)
    verify_run(ROWS)
    run = Run.create(
        "m22-math-source-gap-audit",
        seed=1909,
        config={
            "source_sha256": sha256(SOURCE / "run.json"),
            "row_index_sha256": sha256(ROWS / "run.json"),
            "scope": "existing approved O import and metadata-filtered historical catalogues only",
            "neural_updates": 0,
            "production_activation": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/m22_source_audit.py")
        historical = []
        for path in CATALOGS:
            cat = catalogue(path)
            hits = []
            for kind in ("samples", "recognitions"):
                for row in cat[kind]:
                    texts = [text for text in leaves(row) if PATTERN.search(text)]
                    if texts:
                        hits.append(
                            {
                                "kind": kind,
                                "id": row.get("sample_id", row.get("id")),
                                "source_hash": row.get("source_hash"),
                                "matched_metadata": texts,
                            }
                        )
            historical.append(
                {
                    "catalogue": str(path),
                    "sha256": cat["sha256"],
                    "filter": cat["filter"],
                    "allowed_documents": len(cat["documents"]),
                    "allowed_samples": len(cat["samples"]),
                    "hits": hits,
                    "pixels_opened": False,
                    "labels_are_truth": False,
                }
            )
            if sha256(path) != cat["sha256"]:
                raise ValueError("Historical catalogue changed during read-only audit")
        atomic_json(run.path / "reports/historical-search.json", historical)
        rows = json.loads((ROWS / "reports/rows.json").read_text())
        evidence = []
        for name in ("O-FULL-P10-R01", "O-FULL-P09-R06"):
            row = next(row for row in rows if row["id"] == name)
            original = ROWS / row["context_crop"]["path"]
            target = run.path / "comparisons" / f"{name}-context.png"
            shutil.copyfile(original, target)
            evidence.append(
                {
                    "row": row,
                    "context_sha256": sha256(target),
                    "decision": "One isolated universal quantifier candidate; crop review pending"
                    if name.endswith("P10-R01")
                    else "Hold double-struck role: actual ink is ordinary single-line N Z Q R C",
                }
            )
        atomic_json(run.path / "reports/source-context.json", evidence)
        page_meta = next(
            row
            for row in json.loads((SOURCE / "reports/pages.json").read_text())
            if row["page"] == 10
        )
        box = [252, 707, 311, 785]
        page = Image.open(SOURCE / page_meta["rgb_page"]).convert("RGB")
        crop = page.crop(box)
        target = run.path / "solutions/universal-original-300dpi.png"
        crop.save(target, dpi=(300, 300))
        mask = np.asarray(crop.convert("L")) < 160
        ys, xs = np.where(mask)
        clearance = [
            int(xs.min()),
            int(ys.min()),
            int(crop.width - 1 - xs.max()),
            int(crop.height - 1 - ys.max()),
        ]
        if min(clearance) < 8:
            raise ValueError("Original source not isolated with sufficient margin")
        transform = np.asarray(page_meta["raster_to_pdf"]) @ np.array(
            [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]]
        )
        selection = [
            {
                "id": "O-FULL-P10-R01-U2200-V01",
                "character": "∀",
                "page": 10,
                "row_id": "O-FULL-P10-R01",
                "box": box,
                "role": "glyph",
                "actual_label_basis": (
                    "Agent inspected actual inverted-A bowl and crossbar; prompt cross-check only"
                ),
                "review_status": "pending-independent-crop-review",
            }
        ]
        atomic_json(run.path / "solutions/selection.json", selection)
        atomic_json(
            run.path / "reports/universal-source.json",
            {
                **selection[0],
                "source_page": page_meta,
                "crop_sha256": sha256(target),
                "crop_to_pdf": transform.tolist(),
                "ink_clearance_px": clearance,
                "render_dpi": 300,
                "original_pixels_edited": False,
                "training_ready": False,
                "source_approval": "root independent review and render placement still required",
            },
        )
        run.finish(
            "prepared",
            result={
                "universal_candidates": 1,
                "double_struck_Z_admitted": 0,
                "historical_locator_hits": sum(len(c["hits"]) for c in historical),
                "historical_match_is_not_exhaustive_source_absence": True,
                "profiles_or_renderers_modified": False,
                "neural_updates": 0,
            },
        )
        verify_run(run.path)
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    main()
