"""Engineering fixtures: these synthetic fonts are never personal-quality evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fontTools.ttLib import TTFont
from handnote.markdown import parse

from hwsol.adapters.handnote import upstream_config
from hwsol.config import RenderConfig
from hwsol.jobs.layout import SinglePageRenderer, render_one
from hwsol.jobs.schema import Background
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.metrics import measure_font, normalize_font
from hwsol.profiles.schema import Asset
from hwsol.profiles.store import ProfileStore
from hwsol.profiles.style import LineStyle, PageStyle
from hwsol.training.calibration import AnnotatedPage, measure


@pytest.fixture
def collected(tmp_path):
    base = ProfileStore(tmp_path / "store").resolve()
    return replace(
        base,
        manifest=base.manifest.model_copy(
            update={
                "personal_codepoints": tuple(
                    ord(ch)
                    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()√"
                )
            }
        ),
    )


def test_measured_normalization_keeps_descenders_and_raw_asset(collected, tmp_path):
    raw = sha256(collected.font)
    metrics = measure_font(collected.font, collected.manifest.personal_codepoints)
    path = tmp_path / "normalized.ttf"
    normalize_font(collected.font, path, metrics)
    after = measure_font(path, collected.manifest.personal_codepoints)
    assert sha256(collected.font) == raw
    assert metrics["references_em"]["ascender"] > metrics["references_em"]["x_height"]
    for ch in "pygq":
        before = next(r for r in metrics["glyphs"] if r["character"] == ch)
        result = next(r for r in after["glyphs"] if r["character"] == ch)
        baseline = metrics["baseline_em"]
        low, high = before["ink_bounds_em"][1::2]
        new_low, new_high = result["ink_bounds_em"][1::2]
        assert new_low < baseline < new_high
        assert (new_low - baseline) / (new_high - baseline) == pytest.approx(
            (low - baseline) / (high - baseline), abs=0.006
        )
        assert result["glyph_id"] == before["glyph_id"]
    with pytest.raises(FileExistsError):
        normalize_font(collected.font, path, metrics)


def test_outlier_is_excluded_without_deleting_its_original(collected, tmp_path):
    path = tmp_path / "bad.ttf"
    with TTFont(collected.font) as font:
        name = font.getBestCmap()[ord("x")]
        coords, ends, flags = font["glyf"][name].getCoordinates(font["glyf"])
        for i, (x, y) in enumerate(coords):
            coords[i] = (x, y * 9)
        font["glyf"][name].coordinates = coords
        font.save(path)
    metrics = measure_font(path, collected.manifest.personal_codepoints)
    row = next(r for r in metrics["glyphs"] if r["character"] == "x")
    assert row["outlier"] and row["excluded"] and row["selection_weight"] == 0


@pytest.mark.parametrize("metrics", ["raw", "normalized", "contextual"])
def test_actual_paths_have_complete_provenance_and_replay(collected, metrics):
    cfg = RenderConfig(glyph_metrics=metrics, semantic_layout=True)
    text = (
        "Happy glyphs.\n\n"
        + r"$$\sqrt{\frac{1}{2}}+\vec{x}+\overline{x+1}$$"
        + "\n\n"
        + r"$$A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$$"
    )
    blob, report = render_one(text, "1", collected, cfg, Background(), None, "combined")
    repeat, again = render_one(text, "1", collected, cfg, Background(), None, "combined")
    assert blob == repeat and report == again
    rows = report["glyph_provenance"]
    assert [r["item_index"] for r in rows] == list(range(len(rows)))
    assert all(r["profile_sha256"] == collected.manifest_sha256 for r in rows)
    assert any(r["structure"] == "vector-arrow" and not r["personal_collected"] for r in rows)
    assert any(r["structure"] == "matrix-bracket" and r["personal_collected"] for r in rows)
    assert all(r["actual_glyph_id"] is not None for r in rows if r["personal_collected"])
    assert not any(r["procedural_geometry"] and r["personal_collected"] for r in rows)
    structures = {r["structure"] for r in rows if r["structure"]}
    assert {"radical-overbar", "fraction-bar", "overline-accent"} <= structures


def style(inset=0.12, role=None, column=(0, 1)):
    return PageStyle(
        sample_id="s-" + "1" * 32,
        top=0.06,
        right=0.08,
        lines=(
            LineStyle(
                kind="math",
                role=role,
                column=column,
                anchor=0.9,
                inset=inset,
                baseline_em=(0, 0, 0, 0, 0),
            ),
        ),
    )


def planned(profile, text, semantic=True):
    cfg = RenderConfig(semantic_layout=semantic)
    renderer = SinglePageRenderer(upstream_config(profile.font, cfg, False))
    renderer.configure("combined", profile, "test", settings=cfg)
    return renderer, [r for r in renderer.layout(parse(text)) if r["kind"] == "display"]


def test_short_and_long_equations_share_page_start_not_spare_width(collected):
    profile = replace(collected, styles=(style(),))
    _, short = planned(profile, "$$x=2$$")
    _, long = planned(profile, "$$a+b+c+d+e+f+g+h=123456$$")
    assert short[0]["x"] == pytest.approx(long[0]["x"])
    _, old_short = planned(profile, "$$x=2$$", False)
    assert old_short[0]["x"] > short[0]["x"] + 300


def test_equation_group_aligns_actual_equalities(collected):
    profile = replace(collected, styles=(style(),))
    r, rows = planned(profile, "$$x+2=5$$\n\n$$=3+2$$\n\n$$=5$$")
    anchors = [row["x"] + r.planner.semantic.equality_offset(row) for row in rows]
    assert max(anchors) - min(anchors) < 1e-8
    assert [row["semantic"]["role"] for row in rows] == [
        "initial-equation",
        "equation-continuation",
        "equation-continuation",
    ]


def test_column_relative_training_coordinates_transfer(collected):
    left = replace(collected, styles=(style(0.12, "initial-equation"),))
    right = replace(collected, styles=(style(0.56, "initial-equation", (0.5, 1)),))
    _, a = planned(left, "$$x=2$$")
    _, b = planned(right, "$$x=2$$")
    assert a[0]["x"] == pytest.approx(b[0]["x"])


def test_geometry_switch_discloses_bracket_fallback(collected):
    source = r"$$A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$$"
    reports = [
        render_one(source, "1", collected, RenderConfig(math_strokes=mode), Background(), None)[1]
        for mode in ("off", "personal")
    ]
    for report, personal in zip(reports, (False, True), strict=True):
        brackets = [r for r in report["glyph_provenance"] if r["structure"] == "matrix-bracket"]
        assert len(brackets) == 2 and all(r["personal_collected"] == personal for r in brackets)


@pytest.mark.parametrize("origin", ["recorded-pen", "estimated-trajectory", "scan-contour"])
def test_trace_only_import_preserves_origin(tmp_path, origin):
    from hwsol.training.samples import import_sample
    from hwsol.training.studio import train

    source = tmp_path / "trace.json"
    atomic_json(
        source,
        {
            "schema_version": 1,
            "width": 100,
            "height": 100,
            "em": 20,
            "traces": [
                {
                    "kind": "word",
                    "origin": origin,
                    "text": "Hi",
                    "strokes": [[{"x": 1, "y": 1}, {"x": 2, "y": 2}]],
                }
            ],
        },
    )
    assert AnnotatedPage.model_validate_json(source.read_bytes()).traces[0].origin == origin
    assert measure(source, "s-" + "1" * 32).trace_inventory[origin] == 1
    store = ProfileStore(tmp_path / "studio")
    sample = import_sample(store, "writer", [source], kind="calibration")
    assert sample.state == "accepted" and sample.has_recorded_strokes == (origin == "recorded-pen")
    ref = train(store, "writer")["candidate"]
    p = store.resolve(ref, candidate=True)
    assert any(a.role == "glyph-metrics" for a in p.manifest.assets)
    assert store.settings() == {}


def test_metric_asset_mismatch_is_rejected(collected, tmp_path):
    from hwsol.profiles.store import load_snapshot

    path = tmp_path / "bad.json"
    atomic_json(path, {"font_sha256": "0" * 64})
    font = tmp_path / collected.font.name
    font.write_bytes(collected.font.read_bytes())
    manifest = collected.manifest.model_copy(
        update={
            "assets": (
                Asset(path=font.name, sha256=sha256(font), role="font"),
                Asset(path=path.name, sha256=sha256(path), role="glyph-metrics"),
            )
        }
    )
    atomic_json(tmp_path / "profile.json", manifest.model_dump(mode="json"))
    with pytest.raises(ValueError, match="Glyph metrics"):
        load_snapshot(tmp_path)


def test_collection_queue_deduplicates_and_uses_handnote(collected, tmp_path):
    from hwsol.training.coverage import collection_plan

    source = tmp_path / "sample.md"
    source.write_text("Newton's result $x=2$.")
    plan = collection_plan(collected, [source, source], tmp_path / "queue")
    apostrophe = next(row for row in plan["glyphs"] if row["character"] == "'")
    assert apostrophe["personal_variants"] == 0 and apostrophe["frequency"] == 1
    assert len(plan["sources"]) == 1 and not plan["runtime_blocking"]
    assert Path(plan["templates"][0]["pdf"]).is_file()
    assert (
        json.loads((tmp_path / "queue/collection-plan.json").read_bytes())["source_failures"] == []
    )


def test_line_backend_rejects_math_and_detects_omission_or_repetition(tmp_path):
    from hwsol.lab.text_lines import TextLineRequest, transcription_scores

    with pytest.raises(ValueError, match="never mathematics"):
        TextLineRequest("x=2", tmp_path / "reference.png", tmp_path / "line.png").validate()
    missing = transcription_scores("The median is four", "The median four")
    repeated = transcription_scores("The median is four", "The median is is four")
    assert missing["wer"] == 0.25 and missing["word_count_delta"] == -1
    assert repeated["wer"] == 0.25 and repeated["word_count_delta"] == 1


def test_word_backend_missing_observations_returns_cpu_fallback(tmp_path):
    from PIL import Image

    from hwsol.lab.text_lines import RecordedWordRenderer, TextLineRequest

    reference = tmp_path / "reference.png"
    Image.new("L", (10, 10), 255).save(reference)
    request = TextLineRequest("Hello world", reference, tmp_path / "result.png")
    result = RecordedWordRenderer((), "a" * 64).render(request)
    assert result["status"] == "unavailable" and result["fallback"] == "personal-font-runtime"
    assert not request.output.exists()


def test_new_factors_keep_the_one_page_overflow_contract(collected):
    from hwsol.jobs.layout import LayoutError

    with pytest.raises(LayoutError):
        render_one(
            "\n\n".join(["A complete line of explanation."] * 100),
            "1",
            collected,
            RenderConfig(glyph_metrics="contextual", semantic_layout=True),
            Background(),
            None,
        )
