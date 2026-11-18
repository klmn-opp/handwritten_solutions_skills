"""Workspace lifecycle commands; immutable receipts are distinct from mutable authoring inputs."""

import json
from pathlib import Path
from typing import Annotated

import typer

from hwsol.workspace import archive, initialize, verify_run, workspace_root

workspace = typer.Typer(no_args_is_help=True)


@workspace.command("init")
def init(
    path: Annotated[Path, typer.Option()] = Path("workspace"),
    profile_store: Annotated[Path | None, typer.Option()] = None,
):
    emit(lambda: initialize(path, profile_store))


@workspace.command("archive")
def archive_command(
    source: Annotated[list[Path], typer.Option()],
    path: Annotated[Path | None, typer.Option()] = None,
):
    emit(lambda: archive(workspace_root(path), source))


@workspace.command("verify")
def verify(run: Annotated[Path, typer.Option()]):
    emit(lambda: verify_run(run))


@workspace.command("latest")
def latest(path: Annotated[Path | None, typer.Option()] = None):
    def read():
        root = workspace_root(path)
        value = json.loads((root / "latest.json").read_text())
        from hwsol.profiles.io import inside, sha256

        run = inside(root, value["run"])
        if sha256(run / "run.json") != value["run_sha256"]:
            raise ValueError("Latest run identity changed")
        verify_run(run)
        return value

    emit(read)


def emit(function):
    try:
        typer.echo(json.dumps(function(), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(json.dumps({"status": "failed", "error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
