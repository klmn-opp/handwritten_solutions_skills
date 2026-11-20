"""Exact replay of the author-reviewed source-relative-f / personal-rule pages.

The third visual review accepts the f shape but reports collapsed source variety
and the retained heavy pi. Replacements must start from this actual candidate.
"""

import copy
from pathlib import Path

from PIL import Image

from hwsol.profiles.io import sha256
from hwsol.workspace import verify_run, workspace_root
from research.author_feedback_delivery import json_value
from research.author_feedback_revision import trace
from research.author_feedback_round2 import load_baseline as load_previous
from research.author_feedback_round2 import replay_page as replay_previous
from research.author_feedback_round2_checks import validate_increment as check_round2
from research.corpus.math_glyph_completion import raster
from research.f_size_feedback import FSizeFeedback
from research.homework_benchmark.render import read
from research.numeral_pi_feedback import NumeralPiFeedback
from research.structural_rule_feedback import StructuralRuleFeedback

BASE_RUN = "20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6"
BASE_SHA = "c6a2fbc94e384335271262828de7c7e1e1a329932527b42a2cf5b0e963e3323f"


def validate_increment(before_paths, page, renderer, before, after, old_image, image):
    """Third-round scope: only f/pi; all other provenance fields remain exact."""
    for old, new in zip(before, after, strict=True):
        if old["semantic"] not in {"f", "π"} and old != new:
            raise ValueError(f"Non-target provenance changed: item {old['item_index']}")
    result = check_round2(before_paths, page, renderer, before, after, old_image, image)
    if any(before[i]["semantic"] not in {"f", "π"} for i in result["changed_items"]):
        raise ValueError("Third-round geometry changed outside f/pi")
    result["all_non_target_provenance_fields_exact"] = True
    result["third_round_f_pi_scope_only"] = True
    return result


def load_baseline():
    base = workspace_root() / "runs" / BASE_RUN
    verify_run(base)
    if sha256(base / "run.json") != BASE_SHA:
        raise ValueError("Author-reviewed round2 receipt changed")
    previous = load_previous()
    approval = read(base / "reports/approval-receipt.json")
    if previous["run_sha256"] != approval["baseline_sha256"]:
        raise ValueError("Round2's original baseline changed")
    for path, digest in approval["source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[1] / path) != digest:
            raise ValueError(f"Approved round2 source changed: {path}")
    return {
        **previous,
        "path": base,
        "run_sha256": BASE_SHA,
        "previous": previous,
        "cases": read(base / "reports/cases.json"),
        "round2_digits": NumeralPiFeedback(previous["profile"].font),
        "round2_rules": StructuralRuleFeedback(base / "solutions/structural-sources"),
    }


def replay_page(baseline, number):
    result = replay_previous(baseline["previous"], number)
    page, renderer = result["page"], result["renderer"]
    case_id = result["entry"]["case_id"]
    records = baseline["round2_digits"].apply(page, renderer, result["records"])
    f_size = FSizeFeedback("D-common-source", source=result["source"], title=case_id)
    records = f_size.apply(page, renderer, records)
    records = json_value(baseline["round2_rules"].apply(page, renderer, records))
    parser = json_value(trace(renderer))
    base = baseline["path"]
    archived = read(base / f"reports/{case_id}-provenance.json")
    if records != archived["candidate"] or parser != archived["parser_trace"]:
        raise ValueError(f"Page {number}: round2 records/parser do not replay")
    image = Image.open(base / f"rendered/pages/{number:02d}.png").convert("RGB")
    if raster(renderer.to_vector_pdf([page])).tobytes() != image.tobytes():
        raise ValueError(f"Page {number}: round2 pixels do not replay")
    return {
        **result,
        "records": copy.deepcopy(records),
        "parser": parser,
        "image": image,
        "entry": copy.deepcopy(next(c for c in baseline["cases"] if c["page"] == number)),
    }
