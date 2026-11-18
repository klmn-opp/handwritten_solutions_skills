"""Renderer instrumentation is kept separate from independent transcription and human ratings."""

import re
from collections import Counter

from hwsol.lab.text_lines import transcription_scores


def prose(source: str) -> str:
    source = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", source, flags=re.S)
    return " ".join(source.split())


def measurements(report: dict, case) -> dict:
    pages = report["pages"]
    glyphs = [g for p in pages for g in p["glyph_provenance"]]
    maths = [g for g in glyphs if g["context"] == "mathematics"]
    structures = Counter(g["structure"] for g in maths if g["structure"])
    source_types = Counter(g["source_type"] for g in glyphs)
    missing_structures = [name for name in case.expected.structures if not structures[name]]
    # Bounding-box overlaps are candidates only: crossing strokes and accents can be intentional.
    overlaps = []
    for p in pages:
        groups = {}
        for kind, key in [("formula", "formula_groups"), ("matrix", "matrix_groups")]:
            for index, group in enumerate(p[key]):
                for i in range(*group["item_range"]):
                    groups[i] = (kind, index)
        items = p["glyph_provenance"]
        for index, a in enumerate(items):
            if len(a["bounds_px"]) != 4:
                continue
            x0, y0, x1, y1 = a["bounds_px"]
            for b in items[index + 1 :]:
                if len(b["bounds_px"]) != 4:
                    continue
                if groups.get(a["item_index"]) is not None and groups.get(
                    a["item_index"]
                ) == groups.get(b["item_index"]):
                    continue
                u0, v0, u1, v1 = b["bounds_px"]
                area = max(0, min(x1, u1) - max(x0, u0)) * max(0, min(y1, v1) - max(y0, v0))
                small = min((x1 - x0) * (y1 - y0), (u1 - u0) * (v1 - v0))
                if small > 0 and area / small > 0.25:
                    overlaps.append([a["item_index"], b["item_index"]])
    return {
        "page_count": {
            "actual": len(pages),
            "expected": case.expected.page_count,
            "passed": len(pages) == case.expected.page_count,
        },
        "layout": {
            "clipping": "passed-runtime-ink-boundary-check",
            "bbox_overlap_candidates": len(overlaps),
            "candidate_pairs": overlaps[:100],
            "collision_accuracy": None,
            "method": "bbox proxy excludes same math group; manual confirmation required",
        },
        "glyph_coverage": {
            "visible_items": len(glyphs),
            "source_types": dict(source_types),
            "personal_items": sum(g["personal_collected"] for g in glyphs),
            "external_fallback_items": sum(g["external_fallback"] for g in glyphs),
            "missing": [c for p in pages for c in p["coverage"]["missing"]],
        },
        "mathematics": {
            "visible_items": len(maths),
            "structures": dict(structures),
            "expected_structures_missing": missing_structures,
            "symbol_accuracy": None,
            "structure_accuracy": None,
            "evaluation": "instrumentation only; independent mathematical transcription pending",
        },
        "english": {"cer": None, "wer": None, "status": "not_measured"},
        "style": {
            "similarity": None,
            "naturalness": None,
            "legibility": None,
            "status": "not_measured",
        },
    }


def observed_metrics(expected: str, observation: dict, artifact_sha: str) -> dict:
    if observation.get("artifact_sha256") != artifact_sha:
        raise ValueError("Observation belongs to a different output")
    if observation.get("method") not in {"human-blind", "ocr"} or not observation.get("evaluator"):
        raise ValueError("Provide an independent evaluator and transcription method")
    if observation.get("reference_visible", True):
        raise ValueError("Independent observers must not see the expected transcription")
    result = {"method": observation["method"], "evaluator": observation["evaluator"]}
    actual = observation.get("english_transcription")
    if actual is not None and expected:
        result["english"] = transcription_scores(expected, actual)
    for key in (
        "symbol_accuracy",
        "structure_accuracy",
        "style_similarity",
        "naturalness",
        "legibility",
    ):
        value = observation.get(key)
        if value is not None:
            if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                raise ValueError("Observation scores use [0,1]")
            result[key] = value
    return result
