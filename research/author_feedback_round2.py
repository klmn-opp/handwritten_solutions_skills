"""Exact replay of the exposed 20-page candidate for the second author review.

No production state, original sample, or consumed evaluation is modified here.
Independent bounded treatments start only after pixels and provenance replay.
"""

import copy

from PIL import Image

from hwsol.config import RenderConfig
from hwsol.jobs.schema import Job
from hwsol.profiles.io import sha256
from hwsol.profiles.store import load_snapshot
from hwsol.workspace import verify_run, workspace_root
from research.author_feedback_delivery import json_value
from research.author_feedback_revision import personal_layout, trace
from research.corpus.math_glyph_completion import SELECTION, raster
from research.corpus.math_glyph_resolver import MathGlyphResolver
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish
from research.historical_math_candidate import HistoricalMathFeedback, load_sources
from research.homework_benchmark.render import accepted_scans, read
from research.math_latin_feedback import MathLatinFeedback
from research.math_paired_bars import PersonalPairedBars, source_bar

BASE_RUN = "20261120T115012607677Z-author-feedback-final-20-pages-50910a3a"


def load_baseline():
    """Read the sealed candidate and its own Profile/configuration snapshots."""
    base = workspace_root() / "runs" / BASE_RUN
    verify_run(base)
    profile = load_snapshot(base / "solutions/profile-v1")
    strokes = load_snapshot(base / "solutions/profile-v3")
    scans = accepted_scans(base / "solutions/historical-glyphs")
    sessions = SessionGlyphCalibration(accepted_scans(base / "solutions/session-glyphs"))
    return {
        "path": base,
        "run_sha256": sha256(base / "run.json"),
        "cfg": RenderConfig.model_validate(read(base / "run.json")["config"]["render"]),
        "profile": profile,
        "strokes": strokes,
        "resolver": MathGlyphResolver(
            {g.source["character"]: g for g in scans}, strokes, selected=SELECTION
        ),
        "sessions": sessions,
        "polish": SessionInkPolish(sessions, profile.font),
        "historical": HistoricalMathFeedback(
            load_sources(base / "solutions/new-historical-glyphs"), strokes
        ),
        "bars": PersonalPairedBars(source_bar(base / "solutions/session-glyphs")),
        "cases": read(base / "reports/cases.json"),
    }


def replay_page(baseline, number):
    """Return an independently recreated page only if all archived evidence agrees.

    Return keys: page, renderer, records, parser, image, entry, inputs, source.
    Callers must snapshot any sources/configs used and keep new runs immutable.
    """
    base = baseline["path"]
    entry = next(x for x in baseline["cases"] if x["page"] == number)
    case_id = entry["case_id"]
    inputs = base / f"solutions/inputs/{case_id}"
    source = (inputs / "solution.md").read_text()
    job = Job.model_validate_json((inputs / "job.json").read_text())
    bg = job.background
    asset = inputs / bg.path if bg.path else None
    cfg = baseline["cfg"].model_copy(update={"paper": "plain" if asset else bg.kind})
    page, renderer, records, _, _ = personal_layout(
        source,
        case_id,
        baseline["profile"],
        cfg,
        baseline["resolver"],
        baseline["sessions"],
        baseline["polish"],
        bg,
        asset,
        cpu=True,
        clearance=True,
    )
    records = MathLatinFeedback(baseline["profile"].font).apply(page, renderer, records)
    if number == 2:
        records = baseline["bars"].apply(page, renderer, records)
    elif number in (6, 18):
        records = baseline["historical"].apply(page, renderer, records, baseline["profile"])
    records, parser = json_value(records), json_value(trace(renderer))
    archived = read(base / f"reports/{case_id}-provenance.json")
    if records != archived["candidate"] or parser != archived["parser_trace"]:
        raise ValueError(f"Page {number}: original provenance or parser does not replay")
    image = Image.open(base / f"rendered/pages/{number:02d}.png").convert("RGB")
    if raster(renderer.to_vector_pdf([page])).tobytes() != image.tobytes():
        raise ValueError(f"Page {number}: original pixels do not replay")
    return {
        "page": page,
        "renderer": renderer,
        "records": copy.deepcopy(records),
        "parser": parser,
        "image": image,
        "entry": copy.deepcopy(entry),
        "inputs": inputs,
        "source": source,
    }
