"""Private continuous-region dataset, preserving original pixels and document splits."""

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.dataset_extension import page_geometry
from research.corpus.experiment import create

CONTENT_TYPES = {"english-word", "english-line", "math-expression", "multi-line-derivation"}
# Fixed before optimization; these permit a small short-region research pilot only.
PILOT_MINIMUM = {
    "english_train": 60,
    "multiword_train": 20,
    "english_validation": 10,
    "train_documents": 3,
}


def latex_structure(text: str) -> dict:
    braces = 0
    balanced = True
    for i, ch in enumerate(text):
        if i and text[i - 1] == "\\":
            continue
        if ch == "{":
            braces += 1
        elif ch == "}":
            braces -= 1
            balanced &= braces >= 0
    environments = []
    for action, env in re.findall(r"\\(begin|end)\{([^{}]+)\}", text):
        if action == "begin":
            environments.append(env)
        elif not environments or environments.pop() != env:
            balanced = False
    return {
        "balanced_braces_environments": bool(balanced and not braces and not environments),
        "structures": sorted(set(re.findall(r"\\(frac|sqrt|sum|int|begin|overline|vec)\b", text))),
        "semantic_equivalence_verified": False,
        "normalization": "outer whitespace only; actual written meaning preserved",
    }


def completeness_proof(package: Path, attestation: Path | None) -> dict:
    if attestation is None:
        return {}
    verify_run(attestation)
    proof = json.loads((attestation / "reports/amendment.json").read_text())
    if (proof.get("model"), proof.get("reasoning_effort"), proof.get("human_review")) != (
        "gpt-6.2-sol",
        "xhigh",
        False,
    ):
        raise ValueError("Invalid completeness attestation authority")
    matches = [p for p in proof["parents"] if Path(p["run"]).resolve() == package.resolve()]
    if not matches:
        return {}
    if len(matches) != 1 or matches[0]["sha256"] != sha256(package / "run.json"):
        raise ValueError("Completeness attestation parent changed")
    return {
        "complete": set(matches[0]["complete_samples"]),
        "rejected": set(matches[0]["rejected_samples"]),
        "run": str(attestation.resolve()),
        "sha256": sha256(attestation / "run.json"),
    }


def validate_rows(
    package: Path, assignments: dict, *, allowed: set[str], attestation: Path | None = None
) -> list[dict]:
    verify_run(package)
    rows = json.loads((package / "reports/proposals.json").read_text())
    proof = completeness_proof(package, attestation)

    @lru_cache(maxsize=2)
    def source_page(path, number):
        with pdfium.PdfDocument(path) as doc:
            page = doc[number - 1]
            bitmap = page.render(scale=300 / 72)
            try:
                return np.asarray(bitmap.to_pil().convert("RGB")).copy()
            finally:
                bitmap.close()
                page.close()

    hashes = {}
    result = []
    for row in rows:
        complete = row.get("complete_crop")
        if proof:
            if row["sample_id"] in proof["rejected"]:
                raise ValueError("Labeler rejected completeness")
            if complete is None and row["sample_id"] in proof["complete"]:
                complete = True
        if (
            row["split"] not in allowed
            or assignments.get(row["source_hash"]) != row["split"]
            or row["content_type"] not in CONTENT_TYPES
            or row["dpi"] != 300
            or row["review_status"] != "model-accepted"
            or row.get("human_review") is not False
            or row.get("model", row.get("annotation_model_id")) != "gpt-6.2-sol"
            or row.get("reasoning_effort", row.get("annotation_reasoning_effort")) != "xhigh"
            or row.get("confidence", row.get("transcription_confidence")) != "high"
            or row.get("quality_flags")
            or row.get("ambiguity_flags")
            or not row.get("writer_personal")
            or complete is not True
            or row.get("writer_provenance", {}).get("is_generated") is not False
        ):
            raise ValueError("Continuous sample violates frozen admission policy/split")
        for path, digest in [
            (row["source_pdf"], row["source_hash"]),
            (row["page_image"], row["page_sha256"]),
            (row["crop"], row["crop_sha256"]),
        ]:
            if path not in hashes:
                hashes[path] = sha256(Path(path))
            if hashes[path] != digest:
                raise ValueError("Source asset hash changed")
        original = source_page(row["source_pdf"], row["page_number"])
        if not np.array_equal(original, np.asarray(Image.open(row["page_image"]).convert("RGB"))):
            raise ValueError("Page differs from original 300 DPI raster")
        x0, y0, x1, y1 = row["bbox"]
        if not all(type(v) is int for v in row["bbox"]) or not (
            0 <= x0 < x1 <= original.shape[1] and 0 <= y0 < y1 <= original.shape[0]
        ):
            raise ValueError("Invalid crop geometry")
        crop = np.asarray(Image.open(row["crop"]).convert("RGB"))
        if not np.array_equal(crop, original[y0:y1, x0:x1]):
            raise ValueError("Crop is not untouched source pixels")
        label_key = "text" if row["content_type"].startswith("english") else "latex"
        text = row.get(label_key, "").strip()
        if not text:
            raise ValueError("Empty continuous transcription")
        item = dict(row)
        if row.get("complete_crop") is None:
            item["completeness_attestation"] = {
                k: v for k, v in proof.items() if k not in {"complete", "rejected"}
            }
            item["complete_crop"] = complete
        item.update(
            source_package=str(package.resolve()),
            source_package_sha256=sha256(package / "run.json"),
            raw_label=row[label_key],
            normalized_label=text,
            human_verified=False,
            preprocessing_transforms=[],
            dataset_split=row["split"],
            page_geometry=page_geometry(Path(row["source_pdf"]), row["page_number"], 300),
        )
        if label_key == "latex":
            item["structure_validation"] = latex_structure(text)
            if not item["structure_validation"]["balanced_braces_environments"]:
                raise ValueError("Unbalanced candidate LaTeX must be fixed by labeler")
        result.append(item)
    return result


def audit(rows: list[dict]) -> dict:
    english = [r for r in rows if r["content_type"].startswith("english")]
    train = [r for r in english if r["dataset_split"] == "train"]
    val = [r for r in english if r["dataset_split"] == "validation"]
    actual = {
        "english_train": len(train),
        "multiword_train": sum(len(r["normalized_label"].split()) >= 2 for r in train),
        "english_validation": len(val),
        "train_documents": len({r["source_hash"] for r in train}),
    }
    missing = {
        key: {"actual": actual[key], "required": value}
        for key, value in PILOT_MINIMUM.items()
        if actual[key] < value
    }
    repeated_labels = Counter(r["normalized_label"] for r in english)
    characters = Counter("".join(r["normalized_label"] for r in train))
    val_missing = sorted(set("".join(r["normalized_label"] for r in val)) - characters.keys())
    return {
        "counts": dict(Counter((r["dataset_split"] + ":" + r["content_type"]) for r in rows)),
        "pages": len({(r["source_hash"], r["page_number"]) for r in rows}),
        "documents": len({r["source_hash"] for r in rows}),
        "pilot_thresholds": PILOT_MINIMUM,
        "pilot_counts": actual,
        "missing": missing,
        "short_region_pilot_go": not missing,
        "train_characters": dict(sorted(characters.items())),
        "validation_unseen_characters": val_missing,
        "english_unique_transcriptions": len(repeated_labels),
        "english_repeated_transcription_count": sum(v - 1 for v in repeated_labels.values()),
        "lines_at_least_five_words": dict(
            Counter(r["dataset_split"] for r in english if len(r["normalized_label"].split()) >= 5)
        ),
        "label_accuracy": None,
        "label_source": "user-authorized Sol xhigh; not human truth",
        "fresh_document_holdout_available": False,
        "full_sentence_production_go": False,
        "rl_go": False,
    }


def build(packages: list[Path], attestation: Path | None = None) -> Path:
    parent = Path(
        "workspace/runs/20261118T180020607941Z-expanded-real-glyph-data-fe7d5df4/reports/dataset"
    )
    snapshot = json.loads((parent / "review-snapshot.json").read_text())
    split = json.loads((parent / "split-plan.json").read_text())["assignments"]
    assignments = {d["source_hash"]: split[d["split_group"]] for d in snapshot["document"]}
    run = create(
        "continuous-context-dataset",
        config={
            "packages": [str(p) for p in packages],
            "label_policy": "authorized gpt-6.2-sol xhigh, no root re-recognition",
            "thresholds": PILOT_MINIMUM,
        },
    )
    try:
        rows = [
            r
            for p in packages
            for r in validate_rows(
                p, assignments, allowed={"train", "validation"}, attestation=attestation
            )
        ]
        seen, ids = set(), set()
        for row in rows:
            if row["crop_sha256"] in seen or row["sample_id"] in ids:
                raise ValueError("Duplicate continuous sample")
            seen.add(row["crop_sha256"])
            ids.add(row["sample_id"])
        # Overlap is recorded, never counted as independent examples of the same source ink.
        overlaps = []
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                if (a["source_hash"], a["page_number"]) != (b["source_hash"], b["page_number"]):
                    continue
                x, y, xx, yy = a["bbox"]
                u, v, uu, vv = b["bbox"]
                area = max(0, min(xx, uu) - max(x, u)) * max(0, min(yy, vv) - max(y, v))
                if area / min((xx - x) * (yy - y), (uu - u) * (vv - v)) > 0.5:
                    overlaps.append([a["sample_id"], b["sample_id"]])
        if overlaps:
            raise ValueError(f"Overlapping context crop count: {len(overlaps)}")
        (run.path / "reports/samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        result = audit(rows)
        atomic_json(run.path / "reports/audit.json", result)
        atomic_json(
            run.path / "reports/manifest.json",
            {
                "schema": "personal-continuous-handwriting/1",
                "version": run.path.name,
                "samples_sha256": sha256(run.path / "reports/samples.jsonl"),
                "sample_count": len(rows),
                "split_assignments": assignments,
                "holdout_included": False,
                "source_packages": {str(p.resolve()): sha256(p / "run.json") for p in packages},
            },
        )
        run.finish("prepared", result=result)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("packages", nargs="+", type=Path)
    parser.add_argument("--attestation", type=Path)
    args = parser.parse_args()
    print(build(args.packages, args.attestation))
