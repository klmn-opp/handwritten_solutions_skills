"""Offline lab commands. The Runtime never selects a research backend."""

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

lab = typer.Typer(no_args_is_help=True)
OptionPath = Annotated[Path, typer.Option()]


def call(function, *args):
    try:
        return function(*args)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        typer.echo(
            json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), err=True
        )
        raise typer.Exit(2) from exc


def emit(record):
    typer.echo(json.dumps(record, indent=2, ensure_ascii=False))


@lab.command()
def models():
    """List studied revisions and supported output granularity."""
    from hwsol.lab.protocol import catalog

    emit(catalog())


@lab.command()
def prepare(model: Annotated[str, typer.Option()], output: OptionPath):
    """Create a new local trial directory and fixed prompt list."""
    from hwsol.lab.protocol import prepare as make_trial

    emit(call(make_trial, model, output))


@lab.command()
def sample(
    run: OptionPath,
    checkout: OptionPath,
    workdir: OptionPath,
    torchrun: OptionPath,
    checkpoint: OptionPath,
    vae: OptionPath,
    devices: int = 1,
    dry_run: bool = False,
):
    """Run a local pinned One-DM or DiffBrush checkout with operator-provided assets."""
    from hwsol.lab.sampling import sample as run_sampler

    emit(call(run_sampler, run, checkout, workdir, torchrun, checkpoint, vae, devices, dry_run))


@lab.command()
def collect(run: OptionPath, source: OptionPath, writer: str | None = None, numbered: bool = False):
    """Import a single writer's native PNGs or numbered 01.png, 02.png, ... for blind reading."""
    from hwsol.lab.protocol import collect as collect_images

    emit(call(collect_images, run, source, writer, numbered))


@lab.command()
def evaluate(run: OptionPath):
    """Score every blinded reading. A held trial exits with code 2 and a JSON report."""
    from hwsol.lab.protocol import evaluate as score

    record = call(score, run)
    emit(record)
    if record["status"] == "held":
        raise typer.Exit(2)
