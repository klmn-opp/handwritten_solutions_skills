import json
import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader

from hwsol.config import RenderConfig
from hwsol.jobs.layout import render_one
from hwsol.jobs.runtime import render_job, verify_job
from hwsol.jobs.schema import Background
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore
from hwsol.training.samples import import_sample
from hwsol.training.scan_layout import measure_scan
from hwsol.training.studio import evaluate, publish, train

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MODES = ("baseline", "layout-enhanced", "stroke-enhanced", "combined")


@pytest.mark.parametrize("mode", MODES)
def test_result_uses_the_same_ink_as_the_derivation(tmp_path, mode):
    profile = ProfileStore(tmp_path / "data").resolve()
    cfg = RenderConfig()
    source = (
        "The result is $x_{1}=4$.\n\n## Answer: $x_{1}=4$\n\n"
        r"$$\frac{8}{2}=4$$"
        "\n\n"
        r"$$A=\begin{bmatrix}4&0\\0&4\end{bmatrix}$$"
    )
    marked = source.replace("The result", "__The result__").replace(
        "Answer: $x_{1}=4$", "**Answer: $x_{1}=4$**"
    )
    pdf, report = render_one(marked, "1", profile, cfg, Background(), None, mode)
    plain, _ = render_one(source, "1", profile, cfg, Background(), None, mode)
    assert pdf == plain
    assert report["handwriting"]["glyph_ink"] == "fill-only"
    assert {item["size_px"] for item in report["formula_groups"]} == {cfg.pt(cfg.font_size_pt)}
    # Inspect the emitted PDF. Setting the line width to zero alone still strokes a hairline.
    resources = PdfReader(BytesIO(pdf)).pages[0]["/Resources"]["/XObject"]
    forms = [
        ref.get_object() for ref in resources.values() if ref.get_object()["/Subtype"] == "/Form"
    ]
    assert forms and all(form.get_data().split()[-1] == b"f" for form in forms)


def test_uniform_ink_preserves_literal_content_and_mathematical_distinctions(tmp_path):
    from hwsol.jobs.layout import plain_emphasis, validate_markdown

    profile = ProfileStore(tmp_path / "data").resolve()
    literal = "field__name__suffix\n\n" + "```\npow = a**b**c\n```"
    assert plain_emphasis(literal) == literal
    assert plain_emphasis(r"__Result $x_{i_j}$__") == r"Result $x_{i_j}$"
    for expr in (r"\mathbf{v}", r"\boldsymbol{\alpha}", "𝐯"):
        with pytest.raises(ValueError, match="Uniform answer weight"):
            validate_markdown("$$" + expr + "$$", profile)
    assert validate_markdown(r"$$\vec{v}\in\mathbb{R}^2$$", profile)


def test_controlled_ablation_and_formula_groups(tmp_path):
    problem = tmp_path / "input"
    shutil.copytree(EXAMPLES / "phase2/mixed", problem)
    job = json.loads((problem / "job.json").read_bytes())
    job["problems"] = job["problems"][2:3]
    job["render"] = {"seed": 91, "dpi": 96, "page_width_mm": 180}
    paper = Image.new("RGB", (200, 300), "#fff5ed")
    paper.save(problem / "sheet.png")
    job["background"] = {"kind": "image", "path": "sheet.png", "fit": "stretch"}
    expr = r"$$\sqrt{\frac{1}{2}} + \sum_{i=1}^3 x_i^2 = \int_0^1 x^2\,dx$$"
    matrix = r"$$A=\begin{pmatrix}2&-1\\3&4\end{pmatrix}$$"
    (problem / "03.md").write_text(f"A word appears again.\n\n{expr}\n\n{matrix}")
    # Verification belongs to the input answer; this geometry fixture supplies none.
    job["problems"][0]["verification"] = {}
    atomic_json(problem / "job.json", job)
    store = ProfileStore(tmp_path / "data")
    modes = {}
    hashes = []
    for mode in MODES:
        out = tmp_path / mode
        render_job(problem / "job.json", out, store, realism=mode)
        assert verify_job(out, pixels=True)["page_count"] == 1
        modes[mode] = json.loads((out / "validation_report.json").read_bytes())["pages"][0]
        hashes.append(sha256(out / "pages/01.png"))
    assert len(set(hashes)) == 4
    base = modes["baseline"]
    assert base["matrix_groups"][0]["rows"] == [["2", "-1"], ["3", "4"]]
    for page in modes.values():
        assert [f["geometry_sha256"] for f in page["formula_groups"]] == [
            f["geometry_sha256"] for f in base["formula_groups"]
        ]
        assert page["matrix_groups"] == base["matrix_groups"]
    assert modes["stroke-enhanced"]["formula_groups"] == base["formula_groups"]
    assert modes["layout-enhanced"]["formula_groups"] == modes["combined"]["formula_groups"]
    assert (
        modes["combined"]["formula_groups"][0]["position_px"]
        != base["formula_groups"][0]["position_px"]
    )
    assert (
        modes["combined"]["handwriting"]["planned_lines"]
        == modes["layout-enhanced"]["handwriting"]["planned_lines"]
    )
    assert base["glyph_sources"] == modes["layout-enhanced"]["glyph_sources"]
    replay = tmp_path / "replay"
    render_job(tmp_path / "combined/manifest.json", replay, store)
    assert sha256(replay / "homework_solutions.pdf") == sha256(
        tmp_path / "combined/homework_solutions.pdf"
    )


def test_calibration_history_trace_portability_and_math_cache(tmp_path):
    store = ProfileStore(tmp_path / "studio")
    source = EXAMPLES / "realism/calibration.json"
    imported = import_sample(store, "pen-writer", [source], kind="calibration")
    assert imported.has_recorded_strokes and imported.state == "accepted"
    first = train(store, "pen-writer")["candidate"]
    candidate = store.resolve(first, candidate=True)
    assert len(candidate.styles) == 1 and candidate.styles[0].median_pressure == 0.5
    assert not any(a.role == "trajectory" for a in candidate.manifest.assets)
    assert evaluate(store, first)["status"] == "passed"
    publish(store, first, activate=True)

    def page(profile):
        return render_one(
            r"Two measured words with $x=2+3$.",
            "1",
            profile,
            RenderConfig(),
            Background(),
            None,
            "combined",
        )

    page(store.resolve("bundled-default@v1"))
    blob, report = page(store.resolve())
    assert report["math_fallback_glyphs"] == []
    assert report["handwriting"]["calibration_sample"] == imported.id
    assert report["handwriting"]["planned_lines"][0]["baseline_knots_em"][1] != 0
    # Colours and layout remain reproducible after publishing another round and rolling back.
    import_sample(store, "pen-writer", [source], kind="calibration")
    second = train(store, "pen-writer", include_traces=True)["candidate"]
    newer = store.resolve(second, candidate=True)
    assert len(newer.styles) == 2 and newer.manifest.parent == first
    traces = [a for a in newer.manifest.assets if a.role == "trajectory"]
    assert len(traces) == 2 and all(a.sha256 == sha256(source) for a in traces)
    assert evaluate(store, second)["status"] == "passed"
    publish(store, second, activate=True)
    archive = tmp_path / "portable.zip"
    store.export(second, archive)
    remote = ProfileStore(tmp_path / "runtime-only")
    remote.import_archive(archive)
    assert page(remote.resolve(second))[0] == page(store.resolve())[0]
    store.activate(first)
    assert page(store.resolve())[0] == blob

    invalid = tmp_path / "bad.json"
    raw = json.loads(source.read_bytes())
    raw["traces"][0]["strokes"][0][1]["time"] = -1
    atomic_json(invalid, raw)
    assert import_sample(store, "pen-writer", [invalid], kind="calibration").state == "quarantined"


def test_separate_text_and_math_calibrations_are_used_on_one_page(tmp_path):
    from dataclasses import replace

    from hwsol.training.calibration import measure

    fixture = EXAMPLES / "realism/calibration.json"
    full = measure(fixture, "s-" + "1" * 32)
    styles = tuple(
        full.model_copy(
            update={
                "sample_id": "s-" + digit * 32,
                "lines": tuple(line for line in full.lines if line.kind == kind),
            }
        )
        for digit, kind in (("1", "text"), ("2", "math"))
    )
    profile = replace(ProfileStore(tmp_path / "store").resolve(), styles=styles)
    for identity in ("2", "6"):
        _, report = render_one(
            "Two measured words.\n\n$$x=2$$",
            identity,
            profile,
            RenderConfig(),
            Background(),
            None,
            "combined",
        )
        assert report["handwriting"]["line_calibration_samples"] == {
            "text": styles[0].sample_id,
            "math": styles[1].sample_id,
        }
        rows = report["handwriting"]["planned_lines"]
        assert rows[0]["baseline_knots_em"] == pytest.approx(
            styles[0].lines[0].baseline_em, abs=1e-5
        )


def test_unsegmented_line_uses_spacing_from_other_observed_lines(tmp_path):
    from dataclasses import replace

    from hwsol.training.calibration import measure

    style = measure(EXAMPLES / "realism/calibration.json", "s-" + "1" * 32)
    loaded = ProfileStore(tmp_path / "store").resolve()
    blobs = []
    for spacing in (0.2, 0.8):
        # The numeric heading uses the first row; only the unsegmented body row has spaces.
        first = style.lines[0].model_copy(update={"word_gaps_em": (spacing,)})
        second = style.lines[1].model_copy(update={"word_gaps_em": ()})
        sample = style.model_copy(update={"lines": (first, second)})
        blob, _ = render_one(
            "Two ordinary words.",
            "1",
            replace(loaded, styles=(sample,)),
            RenderConfig(),
            Background(),
            None,
            "layout-enhanced",
        )
        blobs.append(blob)
    assert blobs[0] != blobs[1]


def test_scan_estimates_preserve_coordinates_and_absent_pressure(tmp_path):
    source = tmp_path / "scan.png"
    img = Image.new("L", (160, 90), 255)
    draw = ImageDraw.Draw(img)
    for box in [(20, 15, 44, 32), (56, 16, 82, 31), (26, 51, 60, 66)]:
        draw.rectangle(box, outline=0, width=2)
    img.save(source)
    labels = tmp_path / "regions.json"
    atomic_json(
        labels,
        {
            "em": 20,
            "lines": [
                {"kind": "text", "text": "Two words", "box": [10, 10, 100, 40]},
                {"kind": "math", "text": "a=2", "box": [20, 45, 75, 76]},
            ],
        },
    )
    out = tmp_path / "annotated.json"
    measure_scan(source, labels, out)
    data = json.loads(out.read_bytes())
    assert data["source_sha256"] == sha256(source) and not data["traces"]
    assert data["lines"][0]["words"] == [[20, 45], [56, 83]]
    assert data["lines"][1]["baseline"][0] == [26, 66]
    store = ProfileStore(tmp_path / "data")
    scan = import_sample(store, "pen-writer", [out], kind="calibration")
    assert scan.state == "accepted" and not scan.has_recorded_strokes
    style = store.resolve(train(store, "pen-writer")["candidate"], candidate=True).styles[0]
    assert style.median_pressure is None and style.lines[0].word_gaps_em == (11 / 20,)
    with pytest.raises(FileExistsError):
        measure_scan(source, labels, out)
    rows = json.loads(labels.read_bytes())
    rows["lines"][0]["box"] = [0, 0, 5, 5]
    atomic_json(labels, rows)
    with pytest.raises(ProfileError, match="Empty"):
        measure_scan(source, labels, tmp_path / "empty.json")


def test_formula_folding_respects_structural_tokens():
    from handnote.markdown import Block

    from hwsol.jobs.handwriting import split_simple_chain

    chain = Block("display", (("m", "a=b+1=c-2=d"),))
    result = split_simple_chain(chain)
    assert [b.segments[0][1] for b in result] == ["a=b+1", "=c-2", "=d"]
    for tex in (r"a=\frac{x=y}{2}=3", "a=(x=y)=3", "x_{i=0}=y=z", r"\left|x=y\right|=z=2"):
        block = Block("display", (("m", tex),))
        assert split_simple_chain(block) == [block]


def test_wide_math_alternate_leaves_room_for_next_operator(tmp_path):
    from fontTools.ttLib import TTFont
    from handnote.opentype import add_alternates
    from handnote.variants import handwritten_outline
    from matplotlib import rc_context
    from matplotlib.path import Path as MplPath

    from hwsol.adapters.handnote import upstream_config
    from hwsol.adapters.profile_renderer import ProfileRenderer
    from hwsol.config import test_font

    # Use a deliberately oversized stand-in to exercise the real GSUB/MathText path.
    path = tmp_path / "wide.ttf"
    with TTFont(test_font()) as font:
        cmap = font.getBestCmap()
        add_alternates(font, {cmap[ord("2")]: [cmap[ord("∞")]]})
        font.save(path)
    with rc_context():
        renderer = ProfileRenderer(upstream_config(path, RenderConfig(), True))
        parts = []
        renderer._math_path("$2+2+2+2+2+2$", 100, "wide-alternate", parts)
    raw, codes = handwritten_outline(path, ord("2"), 1, 100)
    raw_right = raw[codes != MplPath.CLOSEPOLY, 0].max()
    exercised = False
    for first, following in zip(parts, parts[1:], strict=False):
        key, scale, x, _ = first
        other, other_scale, other_x, _ = following
        if key[3:5] == (ord("2"), 1) and other[3] == ord("+"):
            digit = renderer._outlines[key]
            plus = renderer._outlines[other]
            digit_right = x + scale * digit.verts[digit.codes != MplPath.CLOSEPOLY, 0].max()
            plus_left = other_x + other_scale * plus.verts[plus.codes != MplPath.CLOSEPOLY, 0].min()
            assert digit_right < plus_left < x + scale * raw_right
            primary, primary_codes = handwritten_outline(path, ord("2"), 0, 100)
            # Width adaptation must not also shrink a tall digit.
            digit_y = digit.verts[digit.codes != MplPath.CLOSEPOLY, 1]
            height = digit_y.max() - digit_y.min()
            primary_y = primary[primary_codes != MplPath.CLOSEPOLY, 1]
            assert height == pytest.approx(primary_y.max() - primary_y.min(), abs=0.05)
            exercised = True
    assert exercised


@pytest.mark.parametrize("dpi", [150, 300])
@pytest.mark.parametrize("mode", MODES)
def test_body_sized_math_fraction_and_matrix_glyphs(tmp_path, monkeypatch, dpi, mode):
    import numpy as np
    from fontTools.ttLib import TTFont
    from handnote.opentype import add_alternates
    from handnote.variants import base_outline
    from matplotlib.path import Path as MplPath

    from hwsol.config import test_font
    from hwsol.jobs.layout import SinglePageRenderer

    # A very wide variant exposes accidental vertical shrinking in both selection passes.
    path = tmp_path / "wide.ttf"
    with TTFont(test_font()) as font:
        cmap = font.getBestCmap()
        add_alternates(font, {cmap[ord("2")]: [cmap[ord("∞")]]})
        font.save(path)
    store = ProfileStore(tmp_path / "studio")
    import_sample(store, "sized", [path], kind="font")
    ref = train(store, "sized")["candidate"]
    profile = store.resolve(ref, candidate=True)
    observed = []
    original = SinglePageRenderer.to_vector_pdf

    def capture(renderer, pages):
        for page in pages:
            for kind, key, matrix, *_ in page.items:
                if kind != "path" or key[0] not in {"glyph", "mglyph"}:
                    continue
                codepoint = key[2] if key[0] == "glyph" else key[3]
                if codepoint not in {ord("2"), ord("8")}:
                    continue
                outline = renderer._outlines[key]
                verts = outline.verts[outline.codes != MplPath.CLOSEPOLY]
                a, b, c, d, x, y = matrix
                placed = verts @ np.array([[a, b], [c, d]]) + (x, y)
                observed.append((key, np.ptp(placed[:, 1])))
        return original(renderer, pages)

    monkeypatch.setattr(SinglePageRenderer, "to_vector_pdf", capture)
    content = (
        r"8 $8+\frac{8}{8}+8^8+\frac{\frac{8}{8}}{8}$."
        "\n\n"
        r"Wide $2+2+2+2+2+2$."
        "\n\n"
        r"Matrix $\begin{pmatrix}8&88\\8&8\end{pmatrix}$."
        "\n\n"
        r"$$A=\begin{pmatrix}8&\frac{8}{8}\\88&8\end{pmatrix}$$"
    )
    cfg = RenderConfig(dpi=dpi)
    pdf, report = render_one(content, "1", profile, cfg, Background(), None, mode)
    assert pdf.startswith(b"%PDF") and report["attempts"][-1]["font_size_pt"] == cfg.font_size_pt
    expected = {}
    for ch in "28":
        verts, codes, upem = base_outline(profile.font, ord(ch))
        expected[ord(ch)] = (
            np.ptp(verts[codes != MplPath.CLOSEPOLY, 1]) * cfg.pt(cfg.font_size_pt) / upem
        )
    ratios = []
    variants = set()
    for key, height in observed:
        if key[0] == "mglyph":
            ratio = height / expected[key[3]]
            ratios.append(ratio)
            if key[3] == ord("2"):
                variants.add(key[-1] if "pen" in key else key[4])
                assert ratio == pytest.approx(1, abs=0.025)
    # Only the exponent is smaller; numerator/denominator and both matrix paths use body em.
    assert sum(0.67 < ratio < 0.73 for ratio in ratios) == 1
    assert all(abs(ratio - 1) < 0.025 or abs(ratio - 0.7) < 0.03 for ratio in ratios)
    assert variants == {0, 1}
    assert {formula["size_px"] for formula in report["formula_groups"]} == {
        cfg.pt(cfg.font_size_pt)
    }


def test_measured_math_strokes_are_portable_and_preserve_formula_layout(tmp_path):
    from dataclasses import replace

    from pydantic import ValidationError

    from hwsol.training.calibration import AnnotatedPage

    page = {
        "schema_version": 1,
        "width": 400,
        "height": 300,
        "em": 30,
        "source_sha256": "a" * 64,
        "math_strokes": [
            {
                "kind": kind,
                "source_sha256": "a" * 64,
                "source_box": [10, 10, 100, 30],
                "width_em": 3,
                "envelope": [[0, 0.06, 0.02], [0.5, 0, 0.055], [1, -0.06, 0.08]],
            }
            for kind in ("bar", "rule")
        ],
    }
    source = tmp_path / "strokes.json"
    atomic_json(source, page)
    store = ProfileStore(tmp_path / "studio")
    sample = import_sample(store, "writer", [source], kind="calibration")
    assert sample.state == "accepted" and not sample.has_recorded_strokes
    ref = train(store, "writer")["candidate"]
    candidate = store.resolve(ref, candidate=True)
    assert len(candidate.styles) == 1 and not candidate.styles[0].lines
    assert evaluate(store, ref)["status"] == "passed"
    publish(store, ref)
    profile = store.resolve(ref)
    plain = replace(profile, styles=())
    # Nested rules, an accent inside a radical, and an inline matrix exercise both math routes.
    content = (
        r"Mean $\bar{x}$ and $\begin{pmatrix}\frac{1}{2}&3\\4&5\end{pmatrix}$."
        "\n\n"
        r"$$\frac{\sqrt{\bar{x}+\frac{1}{2}}}{3}+\overline{x+1}$$"
    )
    for dpi in (150, 300):
        cfg = RenderConfig(dpi=dpi)
        for mode in MODES:
            args = (cfg, Background(), None, mode)
            blob, report = render_one(content, "6", profile, *args)
            _, original = render_one(content, "6", plain, *args)
            assert report["formula_groups"] == original["formula_groups"]
            assert report["matrix_groups"] == original["matrix_groups"]
            assert report["math_fallback_glyphs"] == [
                glyph
                for glyph in original["math_fallback_glyphs"]
                if glyph["codepoint"] != "U+0304"
            ]
            rules = report["handwriting"]["math_strokes"]
            assert {r["kind"] for r in rules} == {"bar", "rule"}
            assert all(r["source_sha256"] == "a" * 64 for r in rules)
            for rule in rules:
                before, after = rule["original_bounds_px"], rule["placed_bounds_px"]
                assert after[0::2] == before[0::2]
                assert (after[1] + after[3]) / 2 == pytest.approx((before[1] + before[3]) / 2)
                assert 0 < rule["vertical_fit"] <= 1
            assert report["handwriting"]["calibration_sample"] is None
    archive = tmp_path / "writer.zip"
    store.export(ref, archive)
    migrated = ProfileStore(tmp_path / "clean")
    migrated.import_archive(archive)
    again, _ = render_one(content, "6", migrated.resolve(ref), *args)
    assert again == blob
    page["math_strokes"][0]["source_box"] = [10, 10, 500, 30]
    with pytest.raises(ValidationError, match="outside page"):
        AnnotatedPage.model_validate(page)
