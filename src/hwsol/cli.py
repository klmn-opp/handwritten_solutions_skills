import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated

import typer
from fontTools.ttLib import TTFont

from hwsol.adapters.base import RenderError
from hwsol.adapters.handnote import HandnoteAdapter
from hwsol.adapters.markdown_to_handwrite import MarkdownToHandwriteAdapter
from hwsol.config import RenderConfig, data_dir
from hwsol.lab.cli import lab
from hwsol.probe import digest, run_probe
from hwsol.profile_cli import profiles, studio
from hwsol.profiles.store import ProfileStore

app = typer.Typer(
    no_args_is_help=True, help="Handwritten Solutions — local Profiles, CPU Studio and rendering."
)
app.add_typer(profiles, name="profiles")
app.add_typer(lab, name="lab")
app.add_typer(studio, name="profile")


def fail(exc: Exception) -> None:
    typer.echo(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), err=True)
    raise typer.Exit(2) from exc


@app.command()
def doctor() -> None:
    """Check the primary CPU backend and licensed test font; never download assets."""
    try:
        selected = ProfileStore().resolve()
        with TTFont(selected.font) as font:
            points = len(font.getBestCmap() or {})
        packages = {
            name: version(name)
            for name in (
                "handnote",
                "pillow",
                "fonttools",
                "matplotlib",
                "pydantic",
                "pypdf",
                "pypdfium2",
                "sympy",
                "typer",
            )
        }
        typer.echo(
            json.dumps(
                {
                    "status": "passed",
                    "phase": 3,
                    "profile": selected.provenance(),
                    "packages": packages,
                    "font": {
                        "path": str(selected.font),
                        "role": selected.manifest.source_kind,
                        "codepoints": points,
                        "sha256": digest(selected.font),
                    },
                    "data_dir": str(data_dir()),
                    "training_required": False,
                    "compatibility_backend": "separate environment required; see docs/PHASE0.md",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (OSError, ValueError, PackageNotFoundError) as exc:
        fail(exc)


@app.command("render-probe")
def render_probe(
    input_path: Annotated[Path, typer.Option("--input", "-i", help="Existing Markdown fixture.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="New/empty output directory.")],
    font: Annotated[
        Path | None, typer.Option(help="Explicit TTF/OTF instead of Profile selection.")
    ] = None,
    config: Annotated[
        Path | None, typer.Option(help="Strict JSON rendering configuration.")
    ] = None,
    backend: Annotated[str, typer.Option(help="handnote or markdown-to-handwrite.")] = "handnote",
    backend_python: Annotated[
        Path | None,
        typer.Option(help="Python executable in the isolated MarkdownToHandwrite environment."),
    ] = None,
    profile: Annotated[str | None, typer.Option(help="Published profile or profile@vN.")] = None,
) -> None:
    """Render supplied text for technical evaluation, without solving or splitting problems."""
    try:
        if font and profile:
            raise RenderError("Use either --font or --profile, not both")
        selected = None if font else ProfileStore().resolve(profile)
        values = selected.manifest.render.model_dump() if selected else {}
        if config:
            values.update(json.loads(config.read_text("utf-8")))
        options = RenderConfig.model_validate(values)
        if backend == "handnote":
            adapter = HandnoteAdapter(real_variants=bool(selected and selected.manifest.variants))
        elif backend == "markdown-to-handwrite" and backend_python:
            if selected and selected.manifest.variants:
                raise RenderError(
                    "Compatibility backend cannot guarantee Profile variant selection"
                )
            adapter = MarkdownToHandwriteAdapter(backend_python)
        else:
            raise RenderError(
                "Unknown backend or missing --backend-python for compatibility backend"
            )
        report = run_probe(
            input_path,
            output,
            font or selected.font,
            options,
            adapter,
            profile=selected.provenance() if selected else None,
        )
        typer.echo(
            json.dumps(
                {
                    key: report[key]
                    for key in ("status", "backend", "page_count", "elapsed_seconds", "warnings")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except (OSError, ValueError) as exc:
        fail(exc)


@app.command()
def render(manifest: Annotated[Path, typer.Option()], output: Annotated[Path, typer.Option()]):
    """Render a local assignment and publish all verified pages together."""
    from hwsol.jobs.runtime import render_job

    try:
        typer.echo(json.dumps(render_job(manifest, output), ensure_ascii=False))
    except (ValueError, OSError) as exc:
        fail(exc)


@app.command()
def verify(
    job: Annotated[Path, typer.Option()],
    pixels: Annotated[
        bool, typer.Option(help="Compare PNG pixels with newly rasterized PDF pages.")
    ] = False,
):
    """Verify a delivered assignment without needing its original Profile or Studio."""
    from hwsol.jobs.runtime import verify_job

    try:
        typer.echo(json.dumps(verify_job(job, pixels=pixels), ensure_ascii=False))
    except Exception as exc:
        fail(exc)


@app.command("read-page")
def read_page_command(
    source: Annotated[Path, typer.Option()], output: Annotated[Path, typer.Option()], page: int = 1
):
    """Make one local PDF page visible to the Agent and report the source page count."""
    from hwsol.jobs.paper import read_page

    try:
        typer.echo(json.dumps(read_page(source, page, output), ensure_ascii=False))
    except Exception as exc:
        fail(exc)
