"""Future rendering and preference admission. No production backend or automatic publication."""

from pathlib import Path

from hwsol.profiles.io import sha256


def ink_decision(character: str, candidate: dict | None, enabled: bool = False) -> dict:
    fallback = {"backend": "font-cpu", "character": character, "generated_ink_used": False}
    if not enabled:
        return {**fallback, "reason": "experimental generator disabled"}
    if not candidate:
        return {**fallback, "reason": "no trained candidate"}
    if (
        candidate.get("origin") != "model-generated"
        or candidate.get("trajectory_kind") != "offline-image"
    ):
        return {**fallback, "reason": "invalid generated-ink provenance"}
    gates = candidate.get("validation", {})
    if not all(
        gates.get(k) is True
        for k in (
            "identity",
            "readability",
            "style",
            "independent_holdout",
            "human_visual_acceptance",
        )
    ):
        return {**fallback, "reason": "independent validation incomplete"}
    if character not in candidate.get("validated_characters", []) or not candidate.get(
        "checkpoint_sha256"
    ):
        return {**fallback, "reason": "character or checkpoint not validated"}
    image = Path(candidate["image"])
    if sha256(image) != candidate.get("image_sha256"):
        raise ValueError("Generated ink digest mismatch")
    return {
        "backend": "neural-candidate",
        "character": character,
        "generated_ink_used": False,
        "eligible_for_research_integration": True,
        "candidate": candidate,
        "placement_owner": "Handnote/MathText",
        "production_published": False,
        "reason": "Admission only; raster/outline placement still requires integration evaluation",
    }


def validate_preference(record: dict) -> dict:
    if record.get("content_gate") != "human-verified-correct":
        raise ValueError("Content correctness is a hard preference gate")
    candidates = record.get("candidates", [])
    if len(candidates) < 2 or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("At least two distinct candidates required")
    for c in candidates:
        if not all(c.get(k) for k in ("id", "artifact_sha256", "origin", "source_run")):
            raise ValueError("Missing candidate provenance")
        if c["origin"] == "model-generated" and not c.get("checkpoint_sha256"):
            raise ValueError("Generated candidate needs checkpoint provenance")
    if record.get("choice") not in [c["id"] for c in candidates] + [
        "tie",
        "neither",
    ] or not record.get("reviewer"):
        raise ValueError("Explicit reviewer and preference required")
    return {
        **record,
        "schema": "personal-ink-preference/1",
        "automatic_publication": False,
        "may_be_relabelled_real_personal": False,
        "rsi_next_step": "find real samples; verify labels; train candidate; evaluate holdout",
    }
