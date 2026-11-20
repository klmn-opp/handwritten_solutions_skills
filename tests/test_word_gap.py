"""Visible inter-word target scaling preserves ink and mathematical token metrics."""

import random

import pytest
from handnote.markdown import Block
from pydantic import ValidationError

from hwsol.adapters.handnote import upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.handwriting import HandwritingLayoutPlanner
from hwsol.jobs.layout import SinglePageRenderer
from hwsol.profiles.store import ProfileStore


def planned(tmp_path, scale):
    profile = ProfileStore(tmp_path / "profiles").resolve("bundled-default@v1")
    cfg = RenderConfig(semantic_layout=True, word_gap_scale=scale, seed=1909)
    renderer = SinglePageRenderer(upstream_config(profile.font, cfg, False))
    renderer.configure("baseline", profile, "word-gap-test", settings=cfg)
    planner = HandwritingLayoutPlanner(renderer, None, random.Random(19), ())
    block = Block("p", (("t", "The field is "), ("m", r"x^2+1"), ("t", " and positive.")))
    lines = planner.text_lines(block, 0)
    assert len(lines) == 1
    return planner, lines[0]


def test_default_gap_matches_explicit_identity(tmp_path):
    assert RenderConfig().word_gap_scale == 1.0
    _, a = planned(tmp_path, RenderConfig(semantic_layout=True).word_gap_scale)
    _, b = planned(tmp_path, 1.0)
    assert a == b


def test_six_percent_changes_visible_gap_not_side_bearings_or_math(tmp_path):
    p, a = planned(tmp_path, 1.0)
    _, b = planned(tmp_path, 1.06)
    assert [(r[0], r[1]) for r in a["runs"]] == [(r[0], r[1]) for r in b["runs"]]
    checked = 0
    for index, (old, new) in enumerate(zip(a["runs"], b["runs"], strict=True)):
        if old[1].isspace():
            left = a["runs"][index - 1] if index else None
            right = a["runs"][index + 1] if index + 1 < len(a["runs"]) else None
            bearing = 100.0 - p.space_advance(100.0, left, right, a["size"])
            assert new[3] + bearing == pytest.approx((old[3] + bearing) * 1.06)
            checked += 1
        else:
            assert old == new  # Actual font character widths, kerning and formula width.
    assert checked >= 2
    assert a["curve"] == b["curve"] and a["size"] == b["size"]


@pytest.mark.parametrize("scale", [0.99, 1.251, float("inf"), float("nan")])
def test_gap_rejects_unbounded_values(scale):
    with pytest.raises(ValidationError):
        RenderConfig(semantic_layout=True, word_gap_scale=scale)


def test_nondefault_gap_cannot_silently_ignore_semantic_layout():
    with pytest.raises(ValidationError, match="requires semantic_layout"):
        RenderConfig(word_gap_scale=1.06)
