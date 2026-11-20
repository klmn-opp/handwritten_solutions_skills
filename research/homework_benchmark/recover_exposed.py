"""Explicit source revision for one consumed M1 case; never a fresh holdout pass."""

import argparse
import copy
import difflib
import hashlib
import json
import shutil
from pathlib import Path

import sympy as sp

from hwsol.jobs.runtime import render_job, verify_job
from hwsol.jobs.schema import Job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import ProfileStore, copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.homework_benchmark.render import EVALUATION_LEDGER, launch

SOURCE_RUN = "20261120T104341625106Z-homework-m1-render-38ea743e"
CASE = "m1-la-06-rank-compatibility"
SOURCE_SHA = "f3e9e72dd0c3cabdc12c929f9f7d638714ea3943eb885c7ce0f4fa9d927011e8"
REVISION = "homework-m1-v1-exposed-recovery-1"
ORIGINAL = r"\mathrm{Null}(A)=\left\{t\begin{pmatrix}-2\\1\\1\end{pmatrix}:t\in\mathbb{R}\right\}"
REPLACEMENT = (
    r"v=\begin{pmatrix}-2\\1\\1\end{pmatrix}"
    "$$\n\n$$"
    r"\mathrm{Null}(A)=\left\{tv:t\in\mathbb{R}\right\}"
)


def exact_checks() -> dict:
    """Independent exact linear algebra from the original statement, not a render trace."""
    a = sp.Matrix([[1, 0, 2], [0, 1, -1], [2, 1, 3]])
    b = sp.Matrix([[1, 0], [0, 1], [2, 1]])
    c = sp.Matrix([[1, 0, 2], [0, 1, -1]])
    v = sp.Matrix([-2, 1, 1])
    left = sp.Matrix([[-2, -1, 1]])
    t, b1, b2 = sp.symbols("t b1 b2", real=True)
    checks = {
        "factorization_A_equals_BC": a == b * c,
        "independent_B_columns": b.rank() == 2,
        "rank_A": a.rank() == 2,
        "left_null_compatibility": left * a == sp.zeros(1, 3),
        "compatibility_is_sufficient": a * sp.Matrix([b1, b2, 0])
        == sp.Matrix([b1, b2, 2 * b1 + b2]),
        "parameterized_particular_solution": a * sp.Matrix([3 - 2 * t, -2 + t, t])
        == sp.Matrix([3, -2, 4]),
        "named_vector_is_original_column": list(v) == [-2, 1, 1],
        "named_vector_generates_entire_nullspace": a.nullspace() == [v],
        "every_real_multiple_is_null": a * (t * v) == sp.zeros(3, 1),
        "rank_nullity": a.rank() + len(a.nullspace()) == a.cols == 3,
    }
    if not all(checks.values()):
        raise ValueError("Independent recovery algebra failed")
    return {
        "method": "Exact SymPy matrix algebra; manual transcription of the original statement",
        "sympy": sp.__version__,
        "A": [[int(x) for x in a.row(i)] for i in range(a.rows)],
        "nullspace_basis": [-2, 1, 1],
        "checks": checks,
        "semantic_equivalence": "v is defined as the original column, with the same real scalar t",
        "limitations": "These checks do not certify visible glyph identity or author style",
    }


def recover(source_run: Path | None = None) -> Path:
    source_run = source_run or workspace_root() / "runs" / SOURCE_RUN
    verify_run(source_run)
    folder = source_run / "solutions/inputs" / CASE
    original = (folder / "source.md").read_text(encoding="utf-8")
    if sha256(folder / "source.md") != SOURCE_SHA or original.count(ORIGINAL) != 1:
        raise ValueError("The recovery applies only to the exact archived, once-failed source")
    failure = json.loads((source_run / f"reports/{CASE}.json").read_text())
    if failure["status"] != "rejected" or failure["failure_stage"] != "layout":
        raise ValueError("Original archived layout rejection is required")
    source_record = json.loads((source_run / "run.json").read_text())
    protected = {
        str(p.resolve()): sha256(p)
        for p in (
            source_run / "run.json",
            workspace_root() / "latest.json",
            EVALUATION_LEDGER / "homework-m1-v1-holdout.json",
            Path("research/homework_benchmark/cases.py"),
            Path("docs/M1_BENCHMARK_PROTOCOL.md"),
        )
    }
    revised = original.replace(ORIGINAL, REPLACEMENT, 1)
    start = original.index(ORIGINAL)
    suffix = original[start + len(ORIGINAL) :]
    if revised[:start] != original[:start] or revised[start + len(REPLACEMENT) :] != suffix:
        raise ValueError("Recovery changed content outside its declared formula span")
    checks = exact_checks()
    run = Run.create(
        "m1-exposed-recovery",
        seed=source_record["seed"],
        config={
            "revision": REVISION,
            "source_run": str(source_run.resolve()),
            "source_case": CASE,
            "source_sha256": SOURCE_SHA,
            "exposure": "already consumed holdout; recovery only",
            "original_holdout_outcome": {"rendered": 7, "rejected": 1, "total": 8},
            "neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/recover_exposed.py")
        inputs = run.path / "solutions/input"
        inputs.mkdir()
        for filename in ("problem.txt", "job.json"):
            shutil.copyfile(folder / filename, inputs / f"original-{filename}")
        (inputs / "original.md").write_text(original, encoding="utf-8")
        (inputs / "solution.md").write_text(revised, encoding="utf-8")
        shutil.copyfile(folder / "problem.txt", inputs / "problem.txt")
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                revised.splitlines(keepends=True),
                fromfile="original-holdout.md",
                tofile=f"{REVISION}.md",
            )
        )
        (run.path / "reports/one-formula-change.diff").write_text(diff, encoding="utf-8")
        atomic_json(run.path / "reports/semantic-checks.json", checks)
        atomic_json(run.path / "reports/original-rejection.json", failure)
        atomic_json(
            run.path / "reports/source-revision.json",
            {
                "revision": REVISION,
                "original_sha256": SOURCE_SHA,
                "revised_sha256": hashlib.sha256(revised.encode()).hexdigest(),
                "original_span": [start, start + len(ORIGINAL)],
                "original_formula": ORIGINAL,
                "replacement": REPLACEMENT,
                "other_solution_characters_unchanged": True,
                "independent_holdout": False,
                "new_renderer_capability": False,
            },
        )
        profile = load_snapshot(source_run / "solutions/profile-v1")
        store = ProfileStore(run.path / "solutions/runtime-profiles")
        copy_snapshot(profile, store.root / "profiles/personal/versions/v1")
        job = copy.deepcopy(json.loads((folder / "job.json").read_text()))
        job.update(
            realism="baseline",
            profile_sha256=profile.manifest_sha256,
            generated_by=f"OpenAI Codex; explicit exposed-source recovery {REVISION}",
            render=source_record["config"]["render_config"],
        )
        job["problems"][0].update(
            id=f"{CASE}-exposed-recovery-1",
            solution="solution.md",
            verification={
                "status": "passed",
                "evidence": "Independent exact matrix checks are saved in semantic-checks.json. "
                "Rendered glyphs and author style require separate review.",
                "checks": job["problems"][0]["verification"]["checks"],
            },
        )
        manifest = inputs / "job.json"
        atomic_json(manifest, Job.model_validate(job).model_dump(mode="json"))
        child = launch(manifest=manifest)
        verify_run(child)
        atomic_json(
            run.path / "reports/comparison-run.json",
            {
                "path": str(child),
                "manifest_sha256": sha256(child / "run.json"),
                "result": json.loads((child / "reports/result.json").read_text()),
            },
        )
        formal = run.path / "rendered/production-cpu"
        render_job(manifest, formal, store=store)
        formal_result = verify_job(formal, pixels=True)
        atomic_json(run.path / "reports/formal-cpu-check.json", formal_result)
        verify_run(source_run)
        if any(sha256(Path(p)) != digest for p, digest in protected.items()):
            raise ValueError("Protected original cases, protocol, run, ledger or latest changed")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        run.finish(
            "held",
            result={
                "revision": REVISION,
                "exposed_recovery": True,
                "original_holdout_result_unchanged": "7 rendered, 1 rejected, 8 total",
                "comparison_run": str(child),
                "formal_cpu": formal_result,
                "exact_semantic_checks_passed": all(checks["checks"].values()),
                "visual_review": "pending; see docs/M1_LIMITATIONS.md after review",
                "production_latest_preserved": True,
                "neural_updates": 0,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path)
    recover(parser.parse_args().source_run)
