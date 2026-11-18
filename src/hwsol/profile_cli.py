"""CLI definitions import Studio only inside explicit authoring commands."""

import json
import zipfile
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer

from hwsol.profiles.schema import ProfileError
from hwsol.profiles.store import ProfileStore

profiles = typer.Typer(no_args_is_help=True, help="Published Profiles and portable runtime assets.")
studio = typer.Typer(no_args_is_help=True, help="Local CPU Profile Studio; no neural models.")


def command(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
            typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
            if isinstance(result, dict) and result.get("status") == "failed":
                raise typer.Exit(2)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            typer.echo(
                json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), err=True
            )
            raise typer.Exit(2) from exc

    return wrapped


@profiles.command("list")
@command
def list_profiles() -> dict:
    return ProfileStore().list()


@profiles.command("activate")
@command
def activate(reference: str) -> dict:
    return ProfileStore().activate(reference)


@profiles.command("rollback")
@command
def rollback(reference: str) -> dict:
    """Activate an existing pinned historical version without deleting newer versions."""
    if "@" not in reference:
        raise ProfileError("Rollback requires profile@version")
    return ProfileStore().activate(reference)


@profiles.command("compare")
@command
def compare(older: str, newer: str) -> dict:
    return ProfileStore().compare(older, newer)


@profiles.command("export")
@command
def export_profile(reference: str, output: Annotated[Path, typer.Option()]) -> dict:
    return ProfileStore().export(reference, output)


@profiles.command("import")
@command
def import_profile(source: Annotated[Path, typer.Option()]) -> dict:
    return ProfileStore().import_archive(source)


@studio.command("create")
@command
def create(profile: Annotated[str, typer.Option()], base: str = "bundled-default@v1") -> dict:
    """Create a personal Profile derived from a published snapshot."""
    return ProfileStore().create(profile, base)


@studio.command("template")
@command
def make_template(
    output: Annotated[Path, typer.Option()], chars: str | None = None, sets: int = 1
) -> dict:
    from hwsol.training.studio import template

    return template(output, chars, sets)


@studio.command("measure-page")
@command
def measure_page(
    source: Annotated[Path, typer.Option()],
    regions: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> dict:
    """Measure a scanned page from transcribed line boxes in oriented image pixels."""
    from hwsol.training.scan_layout import measure_scan

    return measure_scan(source, regions, output)


@studio.command("import")
@command
def import_samples(
    profile: Annotated[str, typer.Option()],
    source: Annotated[list[Path], typer.Option()],
    kind: str | None = None,
    template: Path | None = None,
    quality: float = 1,
    weight: float = 1,
    license_label: str = "LicenseRef-Unspecified-Local-Only",
    license_file: Path | None = None,
) -> dict:
    from hwsol.training.samples import import_sample

    sample = import_sample(
        ProfileStore(),
        profile,
        source,
        kind=kind,
        template=template,
        quality=quality,
        weight=weight,
        license_label=license_label,
        license_file=license_file,
    )
    return sample.model_dump(mode="json")


@studio.command("samples")
@command
def samples(profile: Annotated[str, typer.Option()]) -> dict:
    from hwsol.training.samples import list_samples

    return {
        "samples": [s.model_dump(mode="json") for _, s in list_samples(ProfileStore(), profile)]
    }


@studio.command("train")
@command
def train(
    profile: Annotated[str, typer.Option()], append: bool = True, include_traces: bool = False
) -> dict:
    from hwsol.training.studio import train as build

    if not append:
        raise ProfileError("Only cumulative --append builds are supported; history is retained")
    return build(ProfileStore(), profile, include_traces=include_traces)


@studio.command("evaluate")
@command
def evaluate(
    profile: Annotated[str, typer.Option()], version: Annotated[str, typer.Option()]
) -> dict:
    from hwsol.training.studio import evaluate as assess

    return assess(ProfileStore(), f"{profile}@{version}")


@studio.command("publish")
@command
def publish(
    profile: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    activate: bool = False,
) -> dict:
    from hwsol.training.studio import publish as release

    return release(ProfileStore(), f"{profile}@{version}", activate=activate)
