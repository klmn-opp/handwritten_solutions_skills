"""Phase 2 assignment acceptance through the CLI, including atomic failure paths."""

import json
import shutil
import socket
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from typer.testing import CliRunner

from hwsol.cli import app
from hwsol.profiles.io import atomic_json, sha256

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/phase2"
PDF = "homework_solutions.pdf"


def invoke(*args, ok=True):
    result = CliRunner().invoke(app, list(map(str, args)))
    if ok:
        assert result.exit_code == 0, (result.output, result.exception)
        return json.loads(result.stdout)
    assert result.exit_code == 2, (result.output, result.exception)
    return json.loads(result.stderr)


@pytest.fixture
def assignment(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "private"))
    folder = tmp_path / "问题"
    shutil.copytree(EXAMPLES / "mixed", folder)
    path = folder / "job.json"
    return path, json.loads(path.read_text()), tmp_path / "result"


def run(assignment, ok=True):
    path, job, output = assignment
    atomic_json(path, job)
    response = invoke("render", "--manifest", path, "--output", output, ok=ok)
    if not ok:
        assert response["status"] == "failed"
        assert {p.name for p in output.iterdir()} == {"validation_report.json"}
    return json.loads((output / "validation_report.json").read_text())


def test_a_and_i_real_cpu_job(assignment, monkeypatch):
    def offline(*args):
        raise AssertionError("Attempted to connect while rendering")

    monkeypatch.setattr(socket.socket, "connect", offline)
    path, job, out = assignment
    job["problems"] = job["problems"][:1]
    job["render"] = {"dpi": 300}
    report = run(assignment)
    assert report["selected_problem_count"] == len(PdfReader(out / PDF).pages) == 1
    assert report["pages"][0]["verification"]["equalities"][0]["value"] == "15/2"
    assert report["profile"]["reference"] == "bundled-default@v1"
    assert Image.open(out / "pages/01.png").size == (2480, 3508)
    assert (out / "solutions/01.md").read_bytes() == (path.parent / "01.md").read_bytes()
    assert invoke("verify", "--job", out)["page_count"] == 1


def test_b_order_replay_and_corruption(assignment, tmp_path):
    _, _, out = assignment
    run(assignment)
    assert invoke("verify", "--job", out)["problem_order"] == ["2", "4", "6", "7", "9"]
    replay = tmp_path / "replay"
    invoke("render", "--manifest", out / "manifest.json", "--output", replay)
    assert sha256(out / PDF) == sha256(replay / PDF)
    for i in range(1, 6):
        path = f"pages/{i:02d}.png"
        assert sha256(out / path) == sha256(replay / path)
    assert "(b)" in (out / "solutions/03.md").read_text()
    png = out / "pages/01.png"
    original = png.read_bytes()
    png.write_bytes((out / "pages/02.png").read_bytes())
    assert "digest" in invoke("verify", "--job", out, ok=False)["error"]
    png.write_bytes(original)
    # Swap actual PDF pages, even refreshing the whole-file digest cannot conceal it.
    writer = PdfWriter()
    reader = PdfReader(out / PDF)
    for i in (1, 0, 2, 3, 4):
        writer.add_page(reader.pages[i])
    writer.write(out / PDF)
    record = json.loads((out / "validation_report.json").read_text())
    record["artifacts"][PDF] = sha256(out / PDF)
    atomic_json(out / "validation_report.json", record)
    assert "order" in invoke("verify", "--job", out, ok=False)["error"]


def test_c_paper_layers_and_input_references(assignment, tmp_path):
    path, job, out = assignment
    paper = Image.new("RGB", (120, 80), (205, 240, 210))
    paper.save(path.parent / "sample.jpg")
    other = Image.new("RGB", (120, 80), (230, 210, 245))
    paper.save(path.parent / "sample.pdf", save_all=True, append_images=[other])
    # Agent transcriptions can reference several local materials, including page/frame locations.
    question_image = Image.new("RGB", (400, 60), "white")
    ImageDraw.Draw(question_image).text((5, 5), "Find the speed: 15 m in 2 s.", fill="black")
    question_image.save(path.parent / "statement.png")
    question_image.save(path.parent / "statement.pdf")
    job["problems"][0]["sources"].extend(
        [{"path": "statement.pdf", "page": 1}, {"path": "statement.png"}]
    )
    atomic_json(tmp_path / "private/settings.json", {"background": {"kind": "dots"}})
    job["background"] = {"kind": "lined"}
    for p, bg in zip(
        job["problems"][1:],
        [
            {"kind": "grid"},
            {"kind": "plain"},
            {"kind": "image", "path": "sample.jpg", "fit": "stretch"},
            {"kind": "pdf", "path": "sample.pdf", "page": 2, "fit": "cover"},
        ],
        strict=True,
    ):
        p["background"] = bg
    run(assignment)
    resolved = json.loads((out / "manifest.json").read_text())
    assert [p["background"]["kind"] for p in resolved["problems"]] == [
        "lined",
        "grid",
        "plain",
        "image",
        "pdf",
    ]
    for i, color in [(4, (205, 240, 210)), (5, (230, 210, 245))]:
        with Image.open(out / f"pages/{i:02d}.png") as img:
            assert max(abs(a - b) for a, b in zip(img.getpixel((10, 10)), color, strict=True)) < 6
            assert img.convert("L").getextrema()[0] < 80
    # Lower half contains paper only. Ruling, grid and white have different pixel patterns.
    patterns = []
    for i in (1, 2, 3):
        with Image.open(out / f"pages/{i:02d}.png") as img:
            patterns.append(img.crop((100, 1000, 400, 1300)).tobytes())
    assert len(set(patterns)) == 3
    job.pop("background")
    job["problems"] = job["problems"][:1]
    run((path, job, tmp_path / "default-paper"))
    stored = json.loads((tmp_path / "default-paper/manifest.json").read_text())
    assert stored["problems"][0]["background"]["kind"] == "dots"


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (r"$$\unlisted{x}$$", "Unsupported LaTeX"),
        ("Unsupported glyph 🦄", "Missing text"),
        ("Unfinished $x+1", "Unclosed inline"),
        (r"$$\begin{bmatrix}🦄\end{bmatrix}$$", "matrix glyph"),
        ("\n\n---\n\n", "page break"),
    ],
)
def test_f_failed_last_problem_publishes_only_report(assignment, content, error):
    path, job, out = assignment
    job["problems"] = job["problems"][:2]
    (path.parent / "02.md").write_text(content)
    report = run(assignment, ok=False)
    assert report["problem"] == "4" and error in report["error"]


@pytest.mark.parametrize(
    "content",
    [
        "Keep every step here.\n\n" * 150,
        "$$" + "x+" * 75 + "x=0$$",
    ],
)
def test_g_refuse_vertical_and_horizontal_overflow(assignment, content):
    path, job, out = assignment
    (path.parent / "01.md").write_text(content)
    job["problems"] = job["problems"][:1]
    report = run(assignment, ok=False)
    assert report["type"] == "LayoutError" and len(report["attempts"]) <= 6
    assert (path.parent / "01.md").read_text() == content


def test_retry_keeps_original_and_records_layout_changes(assignment):
    path, job, out = assignment
    job["problems"] = job["problems"][:1]
    body = "Given the measurements.\n\n" + "<br>\n\n" * 60 + "$$v=15/2=7.5$$\n"
    (path.parent / "01.md").write_text(body)
    report = run(assignment)
    assert len(report["pages"][0]["attempts"]) >= 2
    assert (out / "solutions/01.md").read_text() == body


@pytest.mark.parametrize(
    "bad", ["empty", "statement", "incomplete", "source", "check", "traversal"]
)
def test_h_input_and_verification_failure(assignment, bad):
    path, job, out = assignment
    problem = job["problems"][0]
    if bad == "empty":
        job["problems"] = []
    elif bad == "statement":
        problem["statement"] = " "
    elif bad == "incomplete":
        problem["complete"] = False
    elif bad == "source":
        (path.parent / "questions.md").write_text(" ")
    elif bad == "check":
        problem["verification"]["checks"][0]["rhs"] = "100"
    else:
        problem["solution"] = "../outside.md"
    run(assignment, ok=False)


def test_repeated_original_numbers_have_distinct_source_ids(assignment):
    _, job, out = assignment
    job["problems"] = job["problems"][:2]
    for i, p in enumerate(job["problems"]):
        p.update(number="1", id=f"sheet-{i}/1")
    run(assignment)
    assert invoke("verify", "--job", out)["problem_order"] == ["1", "1"]


def test_invisible_ink_cannot_pass_as_a_solution(assignment):
    _, job, _ = assignment
    job["problems"] = job["problems"][:1]
    job["render"] = {"ink": "#ffffff"}
    assert "not visible" in run(assignment, ok=False)["error"]


def test_read_scanned_pdf_original(tmp_path):
    source = tmp_path / "question.pdf"
    first = Image.new("RGB", (200, 200), "white")
    second = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(second).text((10, 10), "Find x if 2x = 4.", fill="black")
    first.save(source, save_all=True, append_images=[second])
    preview = tmp_path / "read.png"
    assert (
        invoke("read-page", "--source", source, "--page", "2", "--output", preview)["page_count"]
        == 2
    )
    with Image.open(preview) as image:
        assert image.convert("L").getextrema()[0] < 100
    assert (
        "range"
        in invoke("read-page", "--source", source, "--page", "3", "--output", preview, ok=False)[
            "error"
        ]
    )


@pytest.mark.parametrize("width,height", [(215.9, 279.4), (297, 210), (180, 180)])
def test_complex_notation_multiple_paper_sizes(assignment, width, height):
    path, job, out = assignment
    job["problems"] = job["problems"][:1]
    job["render"] = {"page_width_mm": width, "page_height_mm": height, "dpi": 96, "seed": 217}
    source = r"""The limit equals one.

$$\lim_{x\to0}\frac{\sin{x}}{x}=1$$

$$r=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2}$$

$$A=\begin{pmatrix}1&-2\\3&4\end{pmatrix}$$

The vector is $\vec{v}=\frac{d\vec{r}}{dt}$ in $\mathrm{m/s}$.
"""
    (path.parent / "01.md").write_text(source, encoding="utf-8")
    report = run(assignment)
    assert (out / "solutions/01.md").read_text("utf-8") == source
    assert report["pages"][0]["math_fallback_glyphs"]
    assert invoke("verify", "--job", out, "--pixels")["page_count"] == 1


def test_independent_pixel_verification(assignment):
    _, job, out = assignment
    job["problems"] = job["problems"][:2]
    report = run(assignment)
    first = "pages/01.png"
    shutil.copyfile(out / "pages/02.png", out / first)
    report["artifacts"][first] = sha256(out / first)
    atomic_json(out / "validation_report.json", report)
    assert "PDF page 1" in invoke("verify", "--job", out, "--pixels", ok=False)["error"]


@pytest.mark.parametrize("path", ["NUL.png", "paper/COM1.jpg", "a./b", "a /b", "a|b", "a?b"])
def test_portable_asset_paths(path):
    from hwsol.profiles.schema import ProfileError, relative

    with pytest.raises(ProfileError, match="asset path"):
        relative(path)
