"""Offline lab commands. The Runtime never selects a research backend."""

import json
import subprocess
from pathlib import Path
from typing import Annotated

import typer

lab = typer.Typer(no_args_is_help=True)
rl = typer.Typer(no_args_is_help=True)
lab.add_typer(rl, name="rl")
OptionPath = Annotated[Path, typer.Option()]


def call(function, *args, **kwargs):
    try:
        from hwsol.workspace import tracked_call

        return tracked_call("lab-" + function.__name__, function, *args, **kwargs)
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


@lab.command("diffink")
def diffink(
    text: Annotated[str, typer.Option()],
    reference: OptionPath,
    python: OptionPath,
    checkout: OptionPath,
    assets_root: OptionPath,
    research: OptionPath,
    workspace: Annotated[Path | None, typer.Option()] = None,
    run_id: str | None = None,
    seed: int = 4206,
    steps: int = 20,
    cfg_scale: float = 1.0,
):
    """Run real isolated CUDA continuation; held output never updates latest.json."""
    from hwsol.lab.diffink import DiffInkAdapter

    try:
        result = DiffInkAdapter(python, checkout, assets_root, research).render(
            text,
            reference,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            workspace=workspace,
            run_id=run_id,
        )
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        typer.echo(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        raise typer.Exit(2) from exc
    emit(result)
    if result["status"] == "held":
        raise typer.Exit(2)


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


@rl.command("prepare")
def rl_prepare(manifest: OptionPath, output: OptionPath, seed: int = 73):
    """Render bounded actions, measure proxy rewards, and make a blind preference ballot."""
    from hwsol.lab.rl import prepare as prepare_rl
    from hwsol.profiles.store import ProfileStore

    emit(call(prepare_rl, manifest, output, ProfileStore(), seed=seed))


@rl.command("train")
def rl_train(
    run: OptionPath,
    output: OptionPath,
    ratings: Annotated[Path | None, typer.Option()] = None,
    parent: Annotated[Path | None, typer.Option()] = None,
    seed: int = 903,
    steps: int = 800,
):
    """Run CPU gradient-bandit training; ratings omitted means proxy-only warm-up."""
    from hwsol.lab.rl import train as train_rl
    from hwsol.profiles.store import ProfileStore

    report = call(
        train_rl,
        run,
        output,
        ProfileStore(),
        ratings=ratings,
        parent=parent,
        seed=seed,
        steps=steps,
    )
    emit({k: v for k, v in report.items() if k not in {"trace", "evaluation"}})
