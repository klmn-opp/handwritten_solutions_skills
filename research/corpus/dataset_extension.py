"""Append sealed Sol-reviewed real samples without replacing Corpus pages or labels.

Higher-resolution page views are independent assets. They retain the original PDF
and document split; the legacy Studio's 150-DPI page records are never overwritten.
"""

import argparse
import json
import shutil
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.dataset import readiness
from research.corpus.experiment import create
from research.corpus.label_admission import CHECKS, accepted_label, label_text
from research.corpus.reference_data import export_views, validate_views
from research.corpus.reference_test import validate_package
from research.corpus.store import SCHEMA, digest, now
from research.corpus.train import validate_dataset


@cache
def page_geometry(original: Path, number: int, dpi: int) -> dict:
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(original) as document:
        page = document[number - 1]
        bitmap = page.render(scale=dpi / 72)
        try:
            convert = bitmap.get_posconv(page)
            origin = np.array(convert.to_page(0, 0))
            x = (np.array(convert.to_page(bitmap.width, 0)) - origin) / bitmap.width
            y = (np.array(convert.to_page(0, bitmap.height)) - origin) / bitmap.height
            return {
                "raster_to_pdf": [[x[0], y[0], origin[0]], [x[1], y[1], origin[1]], [0, 0, 1]],
                "pdf_crop_box": list(page.get_cropbox()),
                "pdf_rotation": page.get_rotation(),
            }
        finally:
            bitmap.close()
            page.close()


def source_dataset(data: Path) -> Path:
    verify_run(data.parent.parent)
    manifest = validate_views(data)
    return next(Path(p).parent for p in manifest["source_files"] if Path(p).name == "samples.jsonl")


def new_sample(row: dict, policy: dict, doc: dict, original: Path, evidence: Path) -> dict:
    if row["split"] not in {"train", "validation"}:
        raise ValueError("Holdout cannot enter an expanded training/validation version")
    receipt = {
        **{
            k: row[k]
            for k in (
                *CHECKS,
                "decision",
                "transcription_confidence",
                "ambiguity_flags",
                "quality_flags",
            )
        },
        "model": "gpt-6.2-sol",
        "reasoning_effort": "xhigh",
        "human_review": False,
        "label": row["candidate_label"],
        "crop_sha256": row["crop_sha256"],
        "policy_digest": digest(policy),
        "evidence_file": str(evidence.resolve()),
        "evidence_sha256": sha256(evidence),
        "evidence_record_digest": digest(row),
        "time": now(),
    }
    view_id = (
        "view-"
        + digest(
            [doc["source_hash"], row["page_number"], row["render_dpi"], row["source_image_sha256"]]
        )[:24]
    )
    sample = {
        "schema": SCHEMA,
        "sample_id": "s-" + digest([view_id, row["bbox"]])[:24],
        "document_id": doc["document_id"],
        "page_id": view_id,
        "source_hash": doc["source_hash"],
        "source_original": str(original.resolve()),
        "source_page_number": row["page_number"],
        "source_page_image": row["page_image_path"],
        "source_page_sha256": row["source_image_sha256"],
        "source_page_size": row["page_size"],
        "render_dpi": row["render_dpi"],
        "image_crop": row["crop_path"],
        "crop_sha256": row["crop_sha256"],
        "bounding_box": row["bbox"],
        "content_type": "glyph",
        "category": "Verified Glyphs",
        "candidate_text": row["candidate_label"],
        "verified_text": None,
        "verified_latex": None,
        "model_text": row["candidate_label"],
        "label_source": "model-review:gpt-6.2-sol",
        "recognizer_name": "Sol visual review",
        "recognizer_version": "gpt-6.2-sol/xhigh",
        "confidence": None,
        "review_status": "model-accepted",
        "model_review": receipt,
        "review_attestation": None,
        "writer_provenance": doc["writer_provenance"],
        "region_role": "personal-handwriting",
        "quality_status": "clean",
        "defects": [],
        "privacy_checked": True,
        "privacy_review_source": "model-single-symbol; not human review",
        "dataset_split": row["split"],
        "split_group": doc["split_group"],
        "preprocessing_transforms": [
            {
                "operation": "pdfium-rgb-page-view",
                "render_dpi": row["render_dpi"],
                "page_number": row["page_number"],
                "page_size": row["page_size"],
                "source_pdf": str(original.resolve()),
                **page_geometry(original, row["page_number"], row["render_dpi"]),
            },
            {"operation": "raw-pixel-crop", "box": row["bbox"]},
        ],
        "notes": row.get("quality_notes", ""),
        "parents": [],
        "revision": 1,
        "created_at": now(),
        "derivation": {
            "method": "sealed-sol-cleaning-batch/1",
            "selection": row,
            "source_image_sha256": row["source_image_sha256"],
        },
    }
    if not accepted_label(sample, policy):
        raise ValueError("New sample does not satisfy the user's existing model-label policy")
    return sample


def physical_box(row: dict, pages: dict) -> tuple[str, int, np.ndarray] | None:
    if "source_page_number" in row:
        number, size = row["source_page_number"], row["source_page_size"]
    elif row.get("page_id") in pages:
        page = pages[row["page_id"]]
        number, size = page["page_number"], [page["width"], page["height"]]
    else:
        return None
    box = np.asarray(row["bounding_box"], np.float64) / np.array([*size, *size])
    return row["source_hash"], number, box


def overlaps_same_ink(a: dict, b: dict, pages: dict) -> bool:
    left, right = physical_box(a, pages), physical_box(b, pages)
    if left is None or right is None or left[:2] != right[:2]:
        return False
    x, y, X, Y = left[2]
    u, v, U, V = right[2]
    intersection = max(0, min(X, U) - max(x, u)) * max(0, min(Y, V) - max(y, v))
    return intersection / min((X - x) * (Y - y), (U - u) * (V - v)) >= 0.65


def append_dataset(parent_data: Path, packages: list[tuple[Path, str]], destination: Path) -> dict:
    base = source_dataset(parent_data)
    rows, _ = validate_dataset(base)
    original_count = len(rows)
    manifest = json.loads((base / "manifest.json").read_text())
    snapshot = json.loads((base / "review-snapshot.json").read_text())
    docs = {d["document_id"]: d for d in snapshot["document"]}
    pages = {p["page_id"]: p for p in snapshot["page"]}
    policy = json.loads((base / "label-policy.json").read_text())
    seen = {r["crop_sha256"] for r in rows}
    batches, skipped = [], []
    for package, split in packages:
        if split not in {"train", "validation"}:
            raise ValueError("No holdout admission")
        proposals = validate_package(package, parent_data, split)
        count = 0
        for proposal in proposals:
            doc = docs[proposal["document_id"]]
            original = Path(manifest["corpus_root"]) / doc["original"]
            if sha256(original) != proposal["source_pdf_sha256"]:
                raise ValueError("Archived PDF differs from cleaning evidence")
            sample = new_sample(proposal, policy, doc, original, package / "reports/proposals.json")
            duplicate = next(
                (
                    r["sample_id"]
                    for r in rows
                    if r["content_type"] in {"glyph", "math-symbol"}
                    and overlaps_same_ink(sample, r, pages)
                ),
                None,
            )
            if sample["crop_sha256"] in seen or duplicate:
                skipped.append(
                    {
                        "sample_id": sample["sample_id"],
                        "existing": duplicate,
                        "reason": "same raw crop or physical ink across resolutions",
                    }
                )
                continue
            rows.append(sample)
            seen.add(sample["crop_sha256"])
            count += 1
        batches.append(
            {
                "run": str(package.resolve()),
                "run_sha256": sha256(package / "run.json"),
                "split": split,
                "admitted": count,
            }
        )
    if len(rows) == original_count:
        raise ValueError("An extension must add new real samples")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["review-snapshot.json", "split-plan.json", "label-policy.json"]:
        shutil.copy2(base / name, destination / name)
    file = destination / "samples.jsonl"
    file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    report = {
        **manifest,
        "created_at": now(),
        "sample_manifest_sha256": sha256(file),
        "eligible_samples": len(rows),
        "model_accepted_samples": sum(r["review_status"] == "model-accepted" for r in rows),
        "human_reviewed_samples": sum(r["review_status"] == "verified" for r in rows),
        "label_sources": dict(Counter(r["label_source"] for r in rows)),
        "splits": dict(Counter(r["dataset_split"] for r in rows)),
        "categories": dict(Counter(r["category"] for r in rows)),
        "character_coverage": dict(Counter("".join(label_text(r) for r in rows))),
        "parent_dataset": str(base.resolve()),
        "parent_manifest_sha256": sha256(base / "manifest.json"),
        "external_batches": [*manifest.get("external_batches", []), *batches],
        "legacy_snapshot_scope": "Studio unchanged; new raw views listed in external_batches",
        "external_samples": manifest.get("external_samples", 0) + len(rows) - original_count,
        "view_page_count": len({r["page_id"] for r in rows}),
        "pages": len(pages),
        "pages_scope": "legacy Corpus page records; see view_page_count",
    }
    atomic_json(destination / "manifest.json", report)
    atomic_json(
        destination / "admission.json",
        {
            "batches": batches,
            "skipped": skipped,
            "old_records_unchanged": True,
            "old_count": original_count,
        },
    )
    atomic_json(destination / "readiness.json", readiness(rows))
    validate_dataset(destination)
    return report


def certify_extension(parent: Path, child: Path, *, write: bool = True) -> dict:
    old, new = validate_views(parent), validate_views(child)
    if old["labels"] != new["labels"] or old["profile"] != new["profile"]:
        raise ValueError("Reference identities must remain fixed")
    a, b = np.load(parent / "arrays.npz"), np.load(child / "arrays.npz")
    if not np.array_equal(a["reference"], b["reference"]):
        raise ValueError("Personal reference pixels changed")
    for filename, array_name in [("records.json", "target"), ("validation.json", "validation")]:
        previous = json.loads((parent / filename).read_text())
        current = json.loads((child / filename).read_text())

        def key(r):
            return (
                r["sample_id"]
                if r["source_type"] == "personal-font-derived"
                else r["original_record"]["sample_id"]
            )

        indices = {key(r): i for i, r in enumerate(current)}
        if len(indices) != len(current):
            raise ValueError("Duplicate view identity")
        for i, row in enumerate(previous):
            j = indices.get(key(row))
            if (
                j is None
                or row["label"] != current[j]["label"]
                or not np.array_equal(a[array_name][i], b[array_name][j])
            ):
                raise ValueError("Old data removed, relabeled or reprocessed")
            if (
                row["source_type"] == "personal-real"
                and row["original_record"] != current[j]["original_record"]
            ):
                raise ValueError("Old real provenance changed")
    certificate = {
        "schema": "append-only-reference-data/1",
        "parent_data": str(parent.resolve()),
        "parent_manifest_sha256": sha256(parent / "manifest.json"),
        "parent_arrays_sha256": sha256(parent / "arrays.npz"),
        "new_arrays_sha256": sha256(child / "arrays.npz"),
        "old_samples_retained": True,
        "reference_unchanged": True,
    }
    if write:
        atomic_json(child / "extension.json", certificate)
        new["files"]["extension.json"] = sha256(child / "extension.json")
        new["extension"] = certificate
        atomic_json(child / "manifest.json", new)
    elif certificate != new.get("extension"):
        raise ValueError("Extension certificate does not match actual preserved views")
    return certificate


def prepare_extension(
    parent_data: Path, train_packages: list[Path], validation_packages: list[Path]
) -> Path:
    run = create(
        "expanded-real-glyph-data",
        config={
            "parent_data": str(parent_data.resolve()),
            "train_packages": [str(p.resolve()) for p in train_packages],
            "validation_packages": [str(p.resolve()) for p in validation_packages],
        },
    )
    try:
        dataset = run.path / "reports/dataset"
        append_dataset(
            parent_data,
            [(p, "train") for p in train_packages]
            + [(p, "validation") for p in validation_packages],
            dataset,
        )
        manifest = export_views(
            profile_store().resolve("personal@v1"), dataset, run.path / "reports/data"
        )
        certify_extension(parent_data, run.path / "reports/data")
        run.finish(
            "prepared",
            result={
                "real_train": manifest["real_train_count"],
                "real_validation": manifest["real_validation_count"],
                "real_train_classes": manifest["real_train_classes"],
                "production_eligible": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent_data", type=Path)
    parser.add_argument("--train-package", type=Path, action="append", required=True)
    parser.add_argument("--validation-package", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(prepare_extension(args.parent_data, args.train_package, args.validation_package))
