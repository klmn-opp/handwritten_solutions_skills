"""Immutable verified exports, grouped splits, duplicate quarantine and readiness gates."""

import json
import re
from collections import Counter
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.store import Corpus, digest, now

CATEGORIES = {
    "english-line": "English Words & Lines",
    "english-word": "English Words & Lines",
    "math-symbol": "Mathematical Symbols",
    "math-expression": "Mathematical Expressions",
    "page": "Page Layout",
    "glyph": "Verified Glyphs",
}


def split_for(group: str, seed: int = 902) -> str:
    bucket = int(digest([seed, group])[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "holdout"


def export_dataset(corpus: Corpus, destination: Path, *, seed: int = 902) -> dict:
    if destination.exists():
        raise FileExistsError("Dataset versions are immutable; use a new version")
    documents = {d["document_id"]: d for d in corpus.rows("document")}
    pages = corpus.rows("page")
    samples = corpus.rows("sample")
    # Near-identical pages from different original documents cannot straddle splits.
    duplicate_documents = set()
    duplicate_pairs = []
    for i, a in enumerate(pages):
        for b in pages[i + 1 :]:
            if a["document_id"] == b["document_id"]:
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            ratio = (a["width"] / a["height"]) / (b["width"] / b["height"])
            if a["pixel_hash"] == b["pixel_hash"] or (distance <= 8 and 0.97 < ratio < 1.03):
                duplicate_pairs.append(
                    {
                        "a": a["page_id"],
                        "b": b["page_id"],
                        "dhash_distance": distance,
                        "status": "needs-review; conservative quarantine",
                    }
                )
                duplicate_documents.update([a["document_id"], b["document_id"]])
    eligible, pending = [], []
    seen_crops = {}
    for sample in samples:
        reasons = []
        if sample["review_status"] != "verified" or not (
            sample.get("review_attestation") or {}
        ).get("human_review"):
            reasons.append("unverified-transcription")
        if (
            sample["writer_provenance"]["domain"] != "personal-real"
            or sample["writer_provenance"]["is_generated"]
        ):
            reasons.append("not-real-personal-source")
        if sample["region_role"] != "personal-handwriting":
            reasons.append("not-exclusive-personal-handwriting")
        if not sample["privacy_checked"]:
            reasons.append("privacy-not-reviewed")
        if sample.get("quality_status") != "clean" or sample.get("defects"):
            reasons.append("ink-quality-not-clean; readable corrections remain recognition-only")
        if sample["content_type"] not in CATEGORIES:
            reasons.append("unknown-content")
        if sample["document_id"] in duplicate_documents:
            reasons.append("near-duplicate-document")
        if sha256(corpus.path(sample["image_crop"])) != sample["crop_sha256"]:
            raise ValueError("Crop checksum mismatch")
        group = documents[sample["document_id"]]["split_group"]
        previous = seen_crops.get(sample["crop_sha256"])
        if previous:
            reasons.append("exact-duplicate-crop")
        if reasons:
            pending.append(
                {
                    "sample_id": sample["sample_id"],
                    "reasons": reasons,
                    "review_status": sample["review_status"],
                }
            )
            continue
        seen_crops[sample["crop_sha256"]] = group
        eligible.append(
            {
                **sample,
                "dataset_split": split_for(group, seed),
                "split_group": group,
                "category": CATEGORIES[sample["content_type"]],
                "image_crop": str(corpus.path(sample["image_crop"])),
                "source_original": str(corpus.path(documents[sample["document_id"]]["original"])),
            }
        )
    destination.mkdir(parents=True)
    records = destination / "samples.jsonl"
    records.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in eligible))
    atomic_json(destination / "pending.json", pending)
    atomic_json(destination / "near-duplicates.json", duplicate_pairs)
    labels = [s.get("verified_text") or s.get("verified_latex") or "" for s in eligible]
    report = {
        "schema": "personal-homework-dataset/1",
        "created_at": now(),
        "seed": seed,
        "split_policy": "source-session SHA256 buckets 70/15/15; immutable seed; no reroll",
        "sample_manifest_sha256": sha256(records),
        "documents": len(documents),
        "pages": len(pages),
        "verified_pages": len({s["page_id"] for s in eligible}),
        "eligible_samples": len(eligible),
        "pending_samples": len(pending),
        "statuses": dict(Counter(s["review_status"] for s in samples)),
        "categories": dict(Counter(s["category"] for s in eligible)),
        "splits": dict(Counter(s["dataset_split"] for s in eligible)),
        "split_groups": {
            d["split_group"]: split_for(d["split_group"], seed) for d in documents.values()
        },
        "verified_english_lines": sum(s["content_type"] == "english-line" for s in eligible),
        "english_vocabulary": sorted(set(re.findall(r"[A-Za-z]+", " ".join(labels)))),
        "character_coverage": dict(Counter("".join(labels))),
        "math_structures": dict(
            Counter(
                t for s in eligible for t in s.get("latex_validation", {}).get("structures", [])
            )
        ),
        "near_duplicate_pairs": len(duplicate_pairs),
        "near_duplicate_document_ratio": len(duplicate_documents) / max(1, len(documents)),
        "human_verified_ratio": len(eligible) / max(1, len(samples)),
        "unlabelled_or_ineligible_ratio": len(pending) / max(1, len(samples)),
        "automatic_recognition_results": len(corpus.rows("recognition")),
        "corpus_root": str(corpus.root),
    }
    atomic_json(destination / "manifest.json", report)
    atomic_json(destination / "readiness.json", readiness(eligible))
    return report


def readiness(samples: list[dict]) -> dict:
    # Minimum research gate, not a claim that these counts guarantee quality.
    counts = {}
    groups = set()
    for sample in samples:
        label = sample.get("verified_text") or sample.get("verified_latex") or ""
        if sample["content_type"] not in {"glyph", "math-symbol"} or len(label) != 1:
            continue
        groups.add(sample["split_group"])
        counts.setdefault(label, Counter())[sample["dataset_split"]] += 1
    ready = [
        k
        for k, c in counts.items()
        if c["train"] >= 20 and c["validation"] >= 5 and c["holdout"] >= 5
    ]
    return {
        "decision": "GO-limited-prototype" if ready and len(groups) >= 3 else "NO-GO",
        "task": "offline 64x64 conditional glyph generation; not continuous text",
        "thresholds": {
            "per_label_train": 20,
            "per_label_validation": 5,
            "per_label_holdout": 5,
            "independent_sessions": 3,
        },
        "eligible_labels": ready,
        "counts_by_label": {k: dict(c) for k, c in counts.items()},
        "independent_sessions": len(groups),
        "missing": (
            "Need human-verified glyphs across independent sessions with fixed validation/holdout. "
            "Lines are not automatically split into glyphs."
        ),
        "sft_rl_ready": False,
        "style_acceptance": "not-assessed",
    }
