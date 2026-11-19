"""Fixed capacity admission; model observations are never author style ratings."""

import math
import unicodedata


def _normalize(value):
    return unicodedata.normalize("NFKC", value).strip()


def capacity_gate(key, observations, *, mean_dice, blank, severe_defects):
    """Require all 24 blind observations and exact transcriptions; never infer missing labels."""
    if len(key) != 24 or len(observations) != 24:
        raise ValueError("All fixed 24 observations required")
    lookup = {q["anonymous_id"].removesuffix(".png"): q for q in key}
    seen = [q["id"].removesuffix(".png") for q in observations]
    if len(lookup) != 24 or len(set(seen)) != 24 or set(seen) != set(lookup):
        raise ValueError("Duplicate or mismatched anonymous IDs")
    if not math.isfinite(mean_dice) or not 0 <= mean_dice <= 1 or blank < 0:
        raise ValueError("Invalid numerical evidence")
    if type(severe_defects) is not bool:
        raise ValueError("Explicit severe-defect assessment required")
    rows = []
    for q in observations:
        expected = lookup[q["id"].removesuffix(".png")]
        if type(q["complete_legible"]) is not bool or not isinstance(q["transcription"], str):
            raise ValueError("Explicit transcription and completeness required")
        if q["image_sha256"] != expected["sha256"]:
            raise ValueError("Observed image differs from frozen output")
        exact = _normalize(q["transcription"]) == _normalize(expected["text"])
        rows.append(
            {
                "id": q["id"],
                "exact_transcription": exact,
                "complete_and_exact": exact and q["complete_legible"],
            }
        )
    count = sum(q["complete_and_exact"] for q in rows)
    eligible = mean_dice >= 0.45 and blank == 0 and not severe_defects
    decision = "stop"
    if eligible and count >= 20:
        decision = "expand"
    elif eligible and 12 <= count <= 19:
        decision = "one-capacity-extension-only"
    return {
        "decision": decision,
        "complete_and_exact": count,
        "total": 24,
        "mean_dice": mean_dice,
        "blank": blank,
        "severe_defects": severe_defects,
        "observations": rows,
        "human_style_approval": False,
        "unseen_text_generalization_established": False,
    }
