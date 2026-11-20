"""Reproducible engineering probes; not homework manifest execution."""

import hashlib
import json
import platform
import shutil
import tempfile
import time
from importlib.metadata import distributions
from pathlib import Path

from PIL import Image, ImageChops
from pypdf import PdfReader

from hwsol.adapters.base import RendererAdapter, RenderError
from hwsol.config import RenderConfig


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    installed = list(distributions())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in installed},
        "dependency_sources": {
            d.metadata["Name"]: json.loads(source)
            for d in installed
            if (source := d.read_text("direct_url.json"))
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_probe(
    source_path: Path,
    output: Path,
    font: Path,
    config: RenderConfig,
    adapter: RendererAdapter,
    *,
    profile: dict | None = None,
) -> dict:
    """Publish outputs only after structural checks; preserve a failure report."""
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RenderError(f"Output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema_version": 1,
        "phase": 1 if profile else 0,
        "status": "failed",
        "backend": adapter.name,
        "source_path": str(source_path.resolve()),
        "font_path": str(font.resolve()),
        "config": config.model_dump(),
        "environment": environment(),
        "validation_scope": "PDF/PNG structure and upstream diagnostics; NOT solution correctness",
        "semantic_validation": "not_performed",
        "one_problem_per_page": "not_implemented",
        "profile": profile,
    }
    try:
        source = source_path.read_text(encoding="utf-8")
        if not source.strip():
            raise RenderError("No Markdown input; refusing to generate a blank answer PDF")
        if not font.is_file():
            raise RenderError(f"Font file does not exist: {font}")
        report.update(source_sha256=digest(source_path), font_sha256=digest(font))
        (output / "source.md").write_text(source, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix=".probe-", dir=output) as tmp:
            stage = Path(tmp)
            result = adapter.render(source, font.resolve(), config, stage)
            pdf = stage / "probe.pdf"
            reader = PdfReader(pdf, strict=True)
            pngs = sorted((stage / "pages").glob("*.png"))
            count = len(reader.pages)
            if count < 1 or count != result.page_count or count != len(pngs):
                raise RenderError("PDF and PNG page counts disagree or are empty")
            dimensions = []
            for page, png in zip(reader.pages, pngs, strict=True):
                expected = (config.mm(config.page_width_mm), config.mm(config.page_height_mm))
                with Image.open(png) as image:
                    image.load()
                    if image.size != expected:
                        raise RenderError(
                            f"Wrong PNG dimensions: {image.size}, expected {expected}"
                        )
                    if not ImageChops.difference(
                        image.convert("RGB"), Image.new("RGB", image.size, "white")
                    ).getbbox():
                        raise RenderError("Renderer produced a completely blank page")
                    dimensions.append(list(image.size))
                # Handnote rounds page dimensions to whole pixels before writing PDF.
                expected_pt = (config.page_width_mm * 72 / 25.4, config.page_height_mm * 72 / 25.4)
                if any(
                    abs(actual - wanted) > 1
                    for actual, wanted in zip(
                        (float(page.mediabox.width), float(page.mediabox.height)),
                        expected_pt,
                        strict=True,
                    )
                ):
                    raise RenderError("PDF physical page dimensions do not match configuration")
            shutil.move(str(pdf), output / "probe.pdf")
            shutil.move(str(stage / "pages"), output / "pages")
            report.update(
                status="passed",
                page_count=count,
                png_dimensions=dimensions,
                warnings=result.warnings,
                coverage=result.coverage,
                backend_details=result.backend_details,
            )
        report["artifacts"] = {
            path.relative_to(output).as_posix(): digest(path)
            for path in [output / "probe.pdf", *sorted((output / "pages").glob("*.png"))]
        }
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise RenderError(str(exc)) from exc
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(output / "validation_report.json", report)
    return report
