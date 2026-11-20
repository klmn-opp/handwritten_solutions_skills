"""Synthetic engineering checks. Preference labels here are never author feedback."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from handnote.opentype import add_alternates
from handnote.render import Outline
from handnote.variants import base_outline, handwritten_outline
from matplotlib.path import Path as MplPath
from typer.testing import CliRunner

from hwsol.cli import app
from hwsol.config import RenderConfig
from hwsol.config import test_font as fixture_font
from hwsol.jobs.ink_policy import fitted_variant, ink_width
from hwsol.jobs.layout import render_one
from hwsol.jobs.runtime import render_job, verify_job
from hwsol.jobs.schema import Background
from hwsol.lab.rl import ACTIONS, gradient_bandit, prepare, train
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import ProfileStore
from hwsol.render_policy import RenderPolicy
from hwsol.training.samples import import_sample
from hwsol.training.studio import evaluate, publish
from hwsol.training.studio import train as train_profile


@pytest.fixture
def writer(tmp_path):
    path = tmp_path / "fixture.ttf"
    with TTFont(fixture_font()) as font:
        base = font.getBestCmap()[ord("2")]
        glyphs = font.getGlyphSet()
        pen = TTGlyphPen(glyphs)
        glyphs[base].draw(TransformPen(pen, (1.25, 0, 0, 1, 0, 0)))
        font["glyf"]["test-two-wide"] = pen.glyph()
        advance, bearing = font["hmtx"][base]
        font["hmtx"]["test-two-wide"] = (round(advance * 1.25), bearing)
        add_alternates(font, {base: ["test-two-wide"]})
        font.save(path)
    store = ProfileStore(tmp_path / "studio")
    import_sample(store, "fixture", [path], kind="font")
    ref = train_profile(store, "fixture")["candidate"]
    assert evaluate(store, ref)["status"] == "passed"
    publish(store, ref)
    return store, ref


def test_gradient_bandit_learns_from_rewards_and_is_replayable():
    rewards = [[0.05, 0.1], [0.85, 0.9]]
    fit = gradient_bandit(rewards, [0.5, 0.5], seed=3)
    assert fit["probabilities"][1] > 0.8
    assert fit["expected_reward_after"] > fit["expected_reward_before"]
    assert fit == gradient_bandit(rewards, [0.5, 0.5], seed=3)
    with pytest.raises(ValueError, match="finite"):
        gradient_bandit([[float("nan")], [1]], [0.5, 0.5], seed=3)


def test_ink_measurement_distinguishes_actual_widths():
    codes = np.asarray([MplPath.MOVETO, *([MplPath.LINETO] * 3), MplPath.CLOSEPOLY])
    widths = []
    for thickness in (4, 12):
        vertices = np.asarray([[0, 0], [80, 0], [80, thickness], [0, thickness], [0, 0]])
        widths.append(ink_width(vertices, codes, 100))
    assert widths[1] > 2 * widths[0] > 0


def test_text_variant_size_uses_character_metrics_not_random_slot(tmp_path):
    path = tmp_path / "size-variants.ttf"
    with TTFont(fixture_font()) as font:
        cmap, glyphs = font.getBestCmap(), font.getGlyphSet()
        mapping = {}
        for ch, (sx, sy, dy) in {
            "l": (0.5, 1, 0),
            "p": (1.8, 1, 0),
            "y": (0.6, 1, 0),
            "0": (1.3, 1.3, -40),
        }.items():
            name = f"size-test-{ord(ch)}"
            pen = TTGlyphPen(glyphs)
            glyphs[cmap[ord(ch)]].draw(TransformPen(pen, (sx, 0, 0, sy, 0, dy)))
            font["glyf"][name] = pen.glyph()
            advance, bearing = font["hmtx"][cmap[ord(ch)]]
            font["hmtx"][name] = (round(advance * sx), bearing)
            mapping[cmap[ord(ch)]] = [name]
        add_alternates(font, mapping)
        font.save(path)
    for ch in "lpy0":
        primary, pcodes, upem = base_outline(path, ord(ch))
        base = primary[pcodes != MplPath.CLOSEPOLY]
        target_height = np.ptp(base[:, 1])
        for original in (0, 1):
            vertices, codes = handwritten_outline(path, ord(ch), original, upem)
            before = vertices[codes != MplPath.CLOSEPOLY]
            for selected in (0, 1):
                actual = fitted_variant(
                    Outline(vertices, codes), ("glyph", original, ord(ch)), path, ord(ch), selected
                )
                ink = actual.verts[actual.codes != MplPath.CLOSEPOLY]
                assert (
                    0.95 * target_height - 1e-6
                    <= np.ptp(ink[:, 1])
                    <= (1.05 * target_height + 1e-6)
                )
                assert ink[:, 0].min() >= before[:, 0].min() - 1e-6
                assert ink[:, 0].max() <= before[:, 0].max() + 1e-6
                if ch in "py":
                    assert ink[:, 1].min() == pytest.approx(base[:, 1].min())
                    assert ink[:, 1].min() < 0
                if ch == "l":
                    assert np.ptp(ink[:, 1]) == pytest.approx(target_height)


def test_policy_preserves_legacy_output_math_and_profile_binding(writer):
    store, ref = writer
    profile = store.resolve(ref)
    content = r"Two 2s and $2+2+\frac{2}{2}=5$."
    cfg = RenderConfig(seed=73)
    plain, original = render_one(content, "1", profile, cfg, Background(), None, "stroke-enhanced")
    for action in (0, 1, 2):
        policy = RenderPolicy(
            profile_sha256=profile.manifest_sha256,
            actions=ACTIONS,
            probabilities=tuple(float(i == action) for i in range(6)),
        )
        settings = cfg.model_copy(update={"policy": policy})
        pdf, report = render_one(
            content, "1", profile, settings, Background(), None, "stroke-enhanced"
        )
        if action == 0:
            assert pdf == plain
        else:
            assert pdf != plain
        assert report["formula_groups"] == original["formula_groups"]
        assert report["math_fallback_glyphs"] == original["math_fallback_glyphs"]
        assert report["handwriting"]["glyph_ink"] == "fill-only"
        assert report["handwriting"]["policy"]["action"] == ACTIONS[action].name
        assert (
            render_one(content, "1", profile, settings, Background(), None, "stroke-enhanced")[0]
            == pdf
        )
    with pytest.raises(ValueError, match="requires"):
        render_one(content, "1", profile, settings, Background(), None, "baseline")
    invalid = policy.model_copy(update={"profile_sha256": "a" * 64})
    with pytest.raises(ValueError, match="different Profile"):
        render_one(
            content,
            "1",
            profile,
            cfg.model_copy(update={"policy": invalid}),
            Background(),
            None,
            "stroke-enhanced",
        )


def test_rollout_training_preferences_migration_and_tamper_rejection(writer, tmp_path, monkeypatch):
    store, ref = writer
    fixture = Path(__file__).resolve().parents[1] / "examples/phase2/mixed"
    source = tmp_path / "input"
    shutil.copytree(fixture, source)
    raw = json.loads((source / "job.json").read_bytes())
    raw.update(profile=ref, realism="stroke-enhanced")
    raw["problems"] = raw["problems"][:3]
    atomic_json(source / "job.json", raw)
    run = tmp_path / "trial"
    prepared = prepare(source / "job.json", run, store)
    assert prepared["rollout_pages"] == 36 and prepared["pairs"] == 12
    result = train(run, tmp_path / "warmup", store, steps=40)
    assert result["human_pairs"] == 0 and result["reward_source"] == "proxy"
    ratings = run / "review/ratings.csv"
    with pytest.raises(ValueError, match="six rated"):
        train(run, tmp_path / "unlabeled", store, ratings=ratings, steps=40)
    assert not (tmp_path / "unlabeled").exists()
    with ratings.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for i, row in enumerate(rows):
        row.update(
            winner="A" if i % 2 else "B",
            math_error_a="no",
            math_error_b="no",
            notes="SYNTHETIC TEST LABEL; not human feedback",
        )

    def write_ratings():
        with ratings.open("w", newline="", encoding="utf-8") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            csv_writer.writeheader()
            csv_writer.writerows(rows)

    rows[0]["math_error_a"] = "yes"
    write_ratings()
    monkeypatch.setenv("HWSOL_DATA_DIR", str(store.root))
    rejected = CliRunner().invoke(
        app,
        [
            "lab",
            "rl",
            "train",
            "--run",
            str(run),
            "--output",
            str(tmp_path / "rejected"),
            "--ratings",
            str(ratings),
            "--steps",
            "40",
        ],
    )
    assert rejected.exit_code == 2 and "Math error" in rejected.output
    assert not (tmp_path / "rejected").exists()
    rows[0]["math_error_a"] = "no"
    write_ratings()
    human = train(
        run,
        tmp_path / "human",
        store,
        ratings=ratings,
        steps=40,
        parent=tmp_path / "warmup/policy.json",
    )
    assert human["human_pairs"] == 12 and human["reward_source"] == "human-preference"
    assert human["parent_sha256"] == sha256(tmp_path / "warmup/policy.json")
    archive = tmp_path / "profile.zip"
    store.export(ref, archive)
    migrated = ProfileStore(tmp_path / "migrated")
    migrated.import_archive(archive)
    candidate = tmp_path / "human/candidate"
    render_job(candidate / "manifest.json", tmp_path / "replay", migrated)
    assert verify_job(tmp_path / "replay", pixels=True)["page_count"] == 3
    assert sha256(candidate / "homework_solutions.pdf") == sha256(
        tmp_path / "replay/homework_solutions.pdf"
    )
    from hwsol.lab import rl

    with monkeypatch.context() as changed:
        changed.setattr(rl, "_renderer_stamp", lambda: {"changed": "renderer"})
        with pytest.raises(ValueError, match="Renderer changed"):
            train(run, tmp_path / "stale", store, ratings=ratings, steps=40)
    (run / "pair-map.json").write_text("[]")
    with pytest.raises(ValueError, match="evidence changed"):
        train(run, tmp_path / "tampered", store, steps=40)
