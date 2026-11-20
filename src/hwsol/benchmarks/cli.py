from pathlib import Path
from typing import Annotated

import typer

from hwsol.benchmarks.schema import load_corpus, select
from hwsol.workspace_cli import emit

benchmark = typer.Typer(no_args_is_help=True)


@benchmark.command("list")
def listing(suite: str = "all"):
    emit(
        lambda: {
            "version": load_corpus().benchmark_version,
            "cases": [
                {
                    "id": c.id,
                    "category": c.category,
                    "suite": c.suite,
                    "complexity": c.complexity,
                    "expected": c.expected.model_dump(),
                }
                for c in select(load_corpus(), suite)
            ],
        }
    )


@benchmark.command("run")
def run(
    suite: str = "smoke",
    profile: str = "bundled-default@v1",
    purpose: str = "evaluation",
    workspace: Annotated[Path | None, typer.Option()] = None,
    run_id: str | None = None,
    glyph_metrics: str = "contextual",
    packet: Annotated[Path | None, typer.Option()] = None,
    submissions: Annotated[Path | None, typer.Option()] = None,
):
    from hwsol.benchmarks.runner import run_benchmark

    emit(
        lambda: run_benchmark(
            suite=suite,
            profile=profile,
            workspace=workspace,
            run_id=run_id,
            purpose=purpose,
            packet=packet,
            submissions=submissions,
            glyph_metrics=glyph_metrics,
        )
    )


@benchmark.command("prepare-agent")
def prepare(
    suite: str = "smoke",
    workspace: Annotated[Path | None, typer.Option()] = None,
    run_id: str | None = None,
):
    from hwsol.benchmarks.runner import prepare_packet

    emit(lambda: prepare_packet(suite=suite, workspace=workspace, run_id=run_id))


@benchmark.command("observe")
def observe(
    run: Annotated[Path, typer.Option()],
    observations: Annotated[Path, typer.Option()],
    workspace: Annotated[Path | None, typer.Option()] = None,
    run_id: str | None = None,
):
    from hwsol.benchmarks.runner import evaluate_observations

    emit(lambda: evaluate_observations(run, observations, workspace=workspace, run_id=run_id))


app = benchmark


@benchmark.command("ocr")
def ocr(
    run: Annotated[Path, typer.Option()],
    workspace: Annotated[Path | None, typer.Option()] = None,
    run_id: str | None = None,
):
    from hwsol.benchmarks.ocr import evaluate_ocr

    emit(lambda: evaluate_ocr(run, workspace=workspace, run_id=run_id))
