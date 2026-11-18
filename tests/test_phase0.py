import json
import socket
import sys
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from pypdf import PdfReader
from typer.testing import CliRunner

from hwsol.adapters.base import RenderError
from hwsol.adapters.handnote import HandnoteAdapter
from hwsol.cli import app
from hwsol.config import RenderConfig, data_dir
from hwsol.config import test_font as bundled_test_font
from hwsol.probe import run_probe

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def render(tmp_path):
    def run(source: str, name="result", config=None, font=None):
        path = tmp_path / f"{name}.md"
        path.write_text(source, encoding="utf-8")
        return run_probe(
            path,
            tmp_path / name,
            font or bundled_test_font(),
            config or RenderConfig(),
            HandnoteAdapter(),
        )

    return run


def test_common_math_cpu_offline(render, monkeypatch, tmp_path):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Rendering attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    source = (ROOT / "examples/phase0/common.md").read_text("utf-8")
    report = render(source)
    assert report["status"] == "passed"
    assert report["page_count"] == len(PdfReader(tmp_path / "result/probe.pdf").pages) == 1
    with Image.open(tmp_path / "result/pages/01.png") as image:
        assert image.size == (1240, 1754)
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert "torch" not in sys.modules
    assert report["semantic_validation"] == "not_performed"


def test_explicit_multiple_pages(render, tmp_path):
    report = render("# Page one\n\n$x=1$\n\n---\n\n# Page two\n\n$x=2$")
    assert report["page_count"] == 2
    assert len(list((tmp_path / "result/pages").glob("*.png"))) == 2


def test_font_selection_changes_pixels(render, tmp_path):
    from matplotlib import font_manager

    second_font = Path(font_manager.findfont("DejaVu Sans"))
    a = render("Abc 123 $x^2=4$", "handwriting")
    b = render("Abc 123 $x^2=4$", "comparison", font=second_font)
    assert a["font_sha256"] != b["font_sha256"]
    assert a["artifacts"]["pages/01.png"] != b["artifacts"]["pages/01.png"]


def test_fixed_seed_repeats_pixels_and_vector_pdf(render):
    a = render("Reproducible $x=\\frac{1}{2}$", "first")
    b = render("Reproducible $x=\\frac{1}{2}$", "second")
    assert a["artifacts"] == b["artifacts"]


@pytest.mark.parametrize("source", ["", "   ", "Missing glyph: 汉", r"$$\notARealCommand{x}$$"])
def test_invalid_content_has_failure_report_and_no_pdf(render, tmp_path, source):
    with pytest.raises(RenderError):
        render(source)
    output = tmp_path / "result"
    assert json.loads((output / "validation_report.json").read_text())["status"] == "failed"
    assert not (output / "probe.pdf").exists()
    assert not (output / "pages").exists()


def test_stale_output_is_not_overwritten(render, tmp_path):
    render("Keep this first result.")
    pdf = (tmp_path / "result/probe.pdf").read_bytes()
    with pytest.raises(RenderError, match="new or empty"):
        render("Do not overwrite.")
    assert (tmp_path / "result/probe.pdf").read_bytes() == pdf


def test_config_validation_and_user_data_override(monkeypatch, tmp_path):
    for bad in (
        {"dpi": 10},
        {"font_size_pt": float("nan")},
        {"typo": 1},
        {"page_width_mm": 100, "margin_mm": 50},
    ):
        with pytest.raises(ValidationError):
            RenderConfig.model_validate(bad)
    monkeypatch.setenv("HWSOL_DATA_DIR", str(tmp_path / "personal"))
    assert data_dir() == tmp_path / "personal"
    assert not data_dir().exists()


def test_300dpi_keeps_a4_physical_size(render, tmp_path):
    report = render("A4 at 300 DPI", config=RenderConfig(dpi=300))
    assert report["png_dimensions"] == [[2480, 3508]]
    page = PdfReader(tmp_path / "result/probe.pdf").pages[0]
    assert abs(float(page.mediabox.height) - 841.89) < 1


def test_cli_diagnostics_and_exit_codes(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["training_required"] is False
    result = runner.invoke(
        app,
        [
            "render-probe",
            "--input",
            str(tmp_path / "missing.md"),
            "--output",
            str(tmp_path / "missing"),
        ],
    )
    assert result.exit_code == 2
    assert not (tmp_path / "missing/probe.pdf").exists()
