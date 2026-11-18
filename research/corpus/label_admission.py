"""Explicitly authorized model labels remain distinct from human-reviewed labels."""

import json
from pathlib import Path

from hwsol.profiles.io import sha256
from research.corpus.store import Corpus, digest, now

POLICY_ID = "sol-clean-symbols/1"
CHECKS = ("single_symbol", "complete_crop", "writer_personal", "privacy_safe")


def label_text(sample: dict) -> str:
    if sample.get("review_status") == "model-accepted":
        return sample.get("model_text") or ""
    return sample.get("verified_text") or sample.get("verified_latex") or ""


def accepted_label(sample: dict, policy: dict | None = None) -> bool:
    if sample.get("review_status") == "verified":
        return (
            sample.get("label_source") == "human-review"
            and (sample.get("review_attestation") or {}).get("human_review") is True
        )
    if not policy or sample.get("review_status") != "model-accepted":
        return False
    receipt = sample.get("model_review") or {}
    return bool(
        policy.get("policy_id") == POLICY_ID
        and policy.get("enabled") is True
        and policy.get("authorization_source") == "explicit-user-instruction"
        and sample["document_id"] in policy.get("documents", [])
        and sample.get("content_type") in {"glyph", "math-symbol"}
        and len(sample.get("model_text") or "") == 1
        and not sample.get("verified_text")
        and not sample.get("verified_latex")
        and sample.get("review_attestation") is None
        and sample.get("label_source") == "model-review:gpt-6.2-sol"
        and receipt.get("policy_digest") == digest(policy)
        and receipt.get("model") == "gpt-6.2-sol"
        and receipt.get("reasoning_effort") == "xhigh"
        and receipt.get("human_review") is False
        and receipt.get("label") == sample.get("model_text")
        and receipt.get("crop_sha256") == sample.get("crop_sha256")
        and receipt.get("decision") == "accept"
        and receipt.get("transcription_confidence") == "high"
        and receipt.get("ambiguity_flags") == []
        and receipt.get("quality_flags") == []
        and all(receipt.get(k) is True for k in CHECKS)
        and receipt.get("evidence_sha256")
    )


def validate_model_evidence(sample: dict) -> None:
    receipt = sample["model_review"]
    source = Path(receipt["evidence_file"])
    if sha256(source) != receipt["evidence_sha256"]:
        raise ValueError("Model label evidence checksum mismatch")
    records = json.loads(source.read_text())
    if not isinstance(records, list) or not any(
        digest(r) == receipt["evidence_record_digest"]
        and r.get("candidate_label") == sample["model_text"]
        and r.get("crop_sha256") == sample["crop_sha256"]
        for r in records
    ):
        raise ValueError("Model label is not present in its recorded evidence")


def authorize(corpus: Corpus, *, instruction: str, documents: list[str]) -> dict:
    """Call only after an explicit user instruction, never from a default Studio route."""
    if not instruction.strip() or not documents:
        raise ValueError("Explicit authorization text and source documents required")
    for document in documents:
        corpus.get("document", document)
    policy = {
        "policy_id": POLICY_ID,
        "enabled": True,
        "authorization_source": "explicit-user-instruction",
        "instruction": instruction,
        "documents": sorted(documents),
        "model": "gpt-6.2-sol",
        "reasoning_effort": "xhigh",
        "scope": "clean isolated symbols only; no mixed, corrected, occluded or ambiguous ink",
        "created_at": now(),
        "human_review_implied": False,
    }
    with corpus.db() as db:
        try:
            old = corpus._get(db, "policy", POLICY_ID)
            if old["instruction"] == instruction and old["documents"] == policy["documents"]:
                return old
            raise ValueError("Use a separate policy version; do not overwrite authorization")
        except KeyError:
            pass
        corpus._put(db, "policy", POLICY_ID, policy)
        corpus.event(db, "authorize-model-labels", POLICY_ID, policy)
    return policy


def adopt(corpus: Corpus, sample_ids: list[str], evidence: Path) -> list[dict]:
    """Adopt exactly matching Sol records, never manufacture a human attestation."""
    policy = corpus.get("policy", POLICY_ID)
    records = json.loads(evidence.read_text())
    if not isinstance(records, list):
        raise ValueError("Expected an array of model decisions")
    evidence_hash = sha256(evidence)
    updated = []
    with corpus.db() as db:
        for sid in sample_ids:
            old = corpus._get(db, "sample", sid)
            if old["review_status"] != "auto-candidate" or old["revision"] != 0:
                raise ValueError("Model adoption cannot overwrite an existing review")
            matches = [
                r
                for r in records
                if r.get("candidate_label") == old["candidate_text"]
                and r.get("bbox") == old["bounding_box"]
                and r.get("crop_sha256") == old["crop_sha256"]
                and r.get("source_image_sha256")
                == (old.get("derivation") or {}).get("source_image_sha256")
            ]
            if len(matches) != 1:
                raise ValueError("Expected one exact model decision for this raw crop")
            record = matches[0]
            receipt = {
                **{
                    k: record.get(k)
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
                "label": record["candidate_label"],
                "crop_sha256": record["crop_sha256"],
                "policy_digest": digest(policy),
                "evidence_file": str(evidence.resolve()),
                "evidence_sha256": evidence_hash,
                "evidence_record_digest": digest(record),
                "time": now(),
            }
            new = {
                **old,
                "review_status": "model-accepted",
                "revision": old["revision"] + 1,
                "model_text": record["candidate_label"],
                "model_review": receipt,
                "label_source": "model-review:gpt-6.2-sol",
                "region_role": "personal-handwriting",
                "quality_status": "clean",
                "defects": [],
                "privacy_checked": True,
                "privacy_review_source": "model-single-symbol; not human review",
            }
            if not accepted_label(new, policy):
                raise ValueError("Uncertain, mixed or damaged model annotation cannot be admitted")
            corpus._put(db, "sample", sid, new)
            corpus.event(db, "adopt-model-label", sid, {"before": old, "after": new})
            updated.append(new)
    return updated
