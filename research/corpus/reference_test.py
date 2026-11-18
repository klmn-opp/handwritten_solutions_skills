"""Independent real-crop comparison, with sealed holdout opened only after model freeze."""

import argparse
import json
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create
from research.corpus.reference_data import align_real, validate_views
from research.corpus.reference_evaluate import shape_distance
from research.corpus.reference_worker import contact_sheet


def validate_package(
    package: Path, data: Path, split: str, *, allow_existing_validation: bool = False
) -> list[dict]:
    """Mechanical provenance validation only; do not relabel Sol's accepted transcriptions."""
    import pypdfium2 as pdfium

    verify_run(package)
    manifest = validate_views(data)
    source_manifest = next(
        Path(p) for p in manifest["source_files"] if Path(p).name == "samples.jsonl"
    )
    source_rows = [json.loads(line) for line in source_manifest.read_text().splitlines()]
    doc_splits = {r["source_hash"]: r["dataset_split"] for r in source_rows}
    snapshot_path = source_manifest.parent / "review-snapshot.json"
    plan_path = source_manifest.parent / "split-plan.json"
    if snapshot_path.exists() and plan_path.exists():
        plan = json.loads(plan_path.read_text())["assignments"]
        for doc in json.loads(snapshot_path.read_text()).get("document", []):
            assigned = plan[doc["split_group"]]
            if doc_splits.setdefault(doc["source_hash"], assigned) != assigned:
                raise ValueError("Frozen document split changed")
    existing_hashes = {r["crop_sha256"]: r for r in source_rows}
    new_hashes = set()
    rows = json.loads((package / "reports/proposals.json").read_text())

    @cache
    def checksum(path):
        return sha256(Path(path))

    @cache
    def page(path, number, dpi):
        with pdfium.PdfDocument(path) as doc:
            pdf_page = doc[number - 1]
            try:
                bitmap = pdf_page.render(scale=dpi / 72)
                try:
                    return np.asarray(bitmap.to_pil().convert("RGB")).copy()
                finally:
                    bitmap.close()
            finally:
                pdf_page.close()

    previous = []
    for row in rows:
        if (
            row["split"] != split
            or doc_splits.get(row["source_pdf_sha256"]) != split
            or row["decision"] != "accept"
            or row["ambiguity_flags"]
            or row["quality_flags"]
            or not row["single_symbol"]
            or not row["complete_crop"]
            or not row["writer_personal"]
            or not row["privacy_safe"]
            or row["transcription_confidence"] != "high"
            or row["annotation_model_id"] != "gpt-6.2-sol"
            or row["annotation_reasoning_effort"] != "xhigh"
            or row["human_review"]
            or len(row["candidate_label"]) != 1
            or row["render_dpi"] not in {150, 300}
        ):
            raise ValueError("Uncertain, damaged, untraceable or incorrectly split external sample")
        for path, digest in [
            (row["source_pdf_path"], row["source_pdf_sha256"]),
            (row["page_image_path"], row["source_image_sha256"]),
            (row["crop_path"], row["crop_sha256"]),
        ]:
            if checksum(path) != digest:
                raise ValueError("External source checksum changed")
        old = existing_hashes.get(row["crop_sha256"])
        if old:
            from research.corpus.label_admission import label_text

            if (
                not allow_existing_validation
                or split != "validation"
                or old["dataset_split"] != "validation"
                or old["source_hash"] != row["source_pdf_sha256"]
                or label_text(old) != row["candidate_label"]
            ):
                raise ValueError("Duplicate real crop in new independent package")
        if row["crop_sha256"] in new_hashes:
            raise ValueError("Duplicate crop within external package")
        new_hashes.add(row["crop_sha256"])
        original = page(row["source_pdf_path"], row["source_pdf_page_number"], row["render_dpi"])
        saved_page = np.asarray(Image.open(row["page_image_path"]).convert("RGB"))
        if not np.array_equal(original, saved_page):
            raise ValueError("Saved page differs from original PDF rendering")
        if row.get("page_number", row["source_pdf_page_number"]) != row[
            "source_pdf_page_number"
        ] or row.get("page_size", list(original.shape[1::-1])) != list(original.shape[1::-1]):
            raise ValueError("Page coordinates/number do not match original rendering")
        x0, y0, x1, y1 = row["bbox"]
        if not all(type(v) is int for v in row["bbox"]) or not (
            0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0]
        ):
            raise ValueError("Invalid crop boundary")
        crop = np.asarray(Image.open(row["crop_path"]).convert("RGB"))
        if not np.array_equal(original[y0:y1, x0:x1], crop):
            raise ValueError("Raw crop changed or retouched")
        for old in previous:
            if old["source_image_sha256"] != row["source_image_sha256"]:
                continue
            a, b, c, d = old["bbox"]
            overlap = max(0, min(c, x1) - max(a, x0)) * max(0, min(d, y1) - max(b, y0))
            if overlap / min((x1 - x0) * (y1 - y0), (c - a) * (d - b)) > 0.65:
                raise ValueError("Overlapping independent crops")
        previous.append(row)
    return rows


def freeze(models: list[Path], reason: str) -> Path:
    records = []
    for model in models:
        verify_run(model)
        request = json.loads((model / "reports/request.json").read_text())
        records.append(
            {
                "sample_run": str(model.resolve()),
                "sample_run_sha256": sha256(model / "run.json"),
                "checkpoint": request["checkpoint"],
                "checkpoint_sha256": request["checkpoint_sha256"],
            }
        )
    run = create("all-character-candidate-freeze", config={"reason": reason, "candidates": records})
    atomic_json(
        run.path / "reports/freeze.json",
        {
            "candidates": records,
            "reason": reason,
            "holdout_used": False,
            "production_eligible": False,
        },
    )
    run.finish("held", result={"frozen_candidates": len(records), "production_eligible": False})
    return run.path


def evaluate(sample: Path, package: Path, *, split: str, frozen: Path | None = None) -> Path:
    if split not in {"validation", "holdout"}:
        raise ValueError("Only independent validation/holdout packages accepted")
    verify_run(sample)
    request = json.loads((sample / "reports/request.json").read_text())
    if split == "holdout":
        if frozen is None:
            raise ValueError("Freeze candidates before opening holdout")
        verify_run(frozen)
        candidates = json.loads((frozen / "reports/freeze.json").read_text())["candidates"]
        if not any(
            r["sample_run_sha256"] == sha256(sample / "run.json")
            and r["checkpoint_sha256"] == request["checkpoint_sha256"]
            for r in candidates
        ):
            raise ValueError("Model/sample set not in frozen candidates")
    data = Path(request["data"])
    rows = validate_package(package, data, split, allow_existing_validation=split == "validation")
    labels = json.loads((data / "manifest.json").read_text())["labels"]
    arrays = np.load(data / "arrays.npz", allow_pickle=False)
    generated = np.load(sample / "reports/generated.npz", allow_pickle=False)["ink"]
    run = create(
        "all-character-" + split + "-evaluation",
        config={
            "sample": str(sample),
            "package": str(package),
            "package_sha256": sha256(package / "run.json"),
            "freeze": str(frozen) if frozen else None,
        },
    )
    results, images, captions, unsupported = [], [], [], []
    rng = np.random.default_rng(2909)
    try:
        for row in rows:
            label = row["candidate_label"]
            if label not in labels:
                unsupported.append({"proposal_id": row["proposal_id"], "label": label})
                continue
            index = labels.index(label)
            real, transform = align_real(Path(row["crop_path"]), arrays["reference"][index])
            fonts = arrays["target"][(arrays["classes"] == index) & (arrays["domains"] == 0)]
            # Equal candidate counts; the geometric baseline is never called neural generation.
            geometric = []
            for _ in range(generated.shape[1]):
                font = fonts[rng.integers(len(fonts))]
                image = Image.fromarray(np.uint8(font * 255))
                image = image.rotate(
                    float(rng.uniform(-3, 3)),
                    Image.Resampling.BICUBIC,
                    translate=tuple(rng.uniform(-1, 1, 2)),
                    fillcolor=0,
                )
                geometric.append(np.asarray(image, dtype=np.float32) / 255)
            distances = shape_distance(generated[index], real)
            results.append(
                {
                    "proposal_id": row["proposal_id"],
                    "label": label,
                    "neural_best_of_8_shape_mae": float(distances.min()),
                    "font_best_available_shape_mae": float(shape_distance(fonts, real).min()),
                    "geometric_best_of_8_shape_mae": float(
                        shape_distance(np.stack(geometric), real).min()
                    ),
                    "transform": transform,
                    "label_source": "user-authorized Sol xhigh; not human verified",
                    "content_correctness": "not measured by image distance",
                }
            )
            images.extend([real, arrays["reference"][index], generated[index, 0]])
            captions.extend([f"{label} real", f"{label} font", f"{label} neural"])
        atomic_json(run.path / "reports/per-sample.json", results)
        summary = {
            "split": split,
            "source_count": len(rows),
            "supported_count": len(results),
            "supported_classes": len({r["label"] for r in results}),
            "unsupported": unsupported,
            "means": {
                key: float(np.mean([r[key] for r in results]))
                for key in (
                    "neural_best_of_8_shape_mae",
                    "font_best_available_shape_mae",
                    "geometric_best_of_8_shape_mae",
                )
            },
            "limitation": "sparse aligned shape proxy; not OCR, author style or semantic accuracy",
            "production_eligible": False,
        }
        atomic_json(run.path / "reports/summary.json", summary)
        contact_sheet(
            np.stack(images), captions, run.path / "comparisons/real-font-neural.png", columns=9
        )
        run.finish("held", result=summary)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("models", type=Path, nargs="+")
    f.add_argument("--reason", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("sample", type=Path)
    e.add_argument("package", type=Path)
    e.add_argument("--split", choices=["validation", "holdout"], required=True)
    e.add_argument("--frozen", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.models, args.reason))
    else:
        print(evaluate(args.sample, args.package, split=args.split, frozen=args.frozen))
