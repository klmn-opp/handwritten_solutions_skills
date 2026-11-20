"""Reference rendering and question-only Agent packets use distinct input paths."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter

from hwsol.benchmarks.metrics import measurements, observed_metrics, prose
from hwsol.benchmarks.schema import corpus_path, load_corpus, select
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.workspace import Run, profile_store, verify_run


def question_asset(case, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    text = case.question
    if case.input_format == "text":
        path = folder / "question.txt"
        path.write_text(text + "\n", encoding="utf-8")
    else:
        lines = textwrap.wrap(text, 80)
        image = Image.new("RGB", (1400, max(240, 80 + 36 * len(lines))), "white")
        draw = ImageDraw.Draw(image)
        # Use our installed dependency's font, not an OS-specific font search.
        font_path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
        font = ImageFont.truetype(str(font_path), 26)
        draw.multiline_text((30, 30), "\n".join(lines), fill="black", font=font, spacing=9)
        path = folder / ("question.png" if case.input_format == "image" else "question.pdf")
        image.save(path)
    return path


def background(case, folder):
    bg = {"kind": case.paper}
    if case.paper in {"image", "pdf"}:
        image = Image.new("RGB", (800, 1100), (247, 243, 228))
        draw = ImageDraw.Draw(image)
        for y in range(70, 1100, 40):
            draw.line((30, y, 770, y), fill=(200, 210, 220))
        name = "paper.png" if case.paper == "image" else "paper.pdf"
        image.save(folder / name)
        bg.update(path=name, fit="stretch")
    return bg


def prepare_packet(*, suite="smoke", workspace=None, run_id=None):
    corpus = load_corpus()
    cases = select(corpus, suite)
    run = Run.create(
        "agent-packet",
        workspace=workspace,
        run_id=run_id,
        config={
            "benchmark_version": corpus.benchmark_version,
            "corpus_sha256": sha256(corpus_path()),
            "suite": suite,
            "references_exposed": False,
        },
    )
    entries = []
    for case in cases:
        source = question_asset(case, run.path / "solutions" / case.id)
        entries.append(
            {
                "id": case.id,
                "source": str(source.relative_to(run.path)),
                "source_sha256": sha256(source),
                "category": case.category,
                "instruction": (
                    "Read the original source. Solve independently, preserve conditions and units. "
                    "Submit Markdown, verification and source digest. "
                    "Do not read benchmark references."
                ),
            }
        )
    atomic_json(run.path / "reports/questions.json", {"cases": entries, "schema_version": 1})
    return run.finish(
        "prepared", result={"case_count": len(entries), "reference_solutions_in_packet": False}
    )


def execute_case(case, parent: Run, profile: str, config: dict, submission=None, packet=None):
    store = profile_store(parent.path.parent.parent)
    run = Run.create(
        "benchmark-case",
        workspace=parent.path.parent.parent,
        run_id=parent.path.name + "-" + case.id,
        config={
            "case_id": case.id,
            "parent": parent.path.name,
            "case": case.model_dump(mode="json"),
            "render": config,
            "path": "end-to-end" if packet else "render-only",
        },
        seed=config["seed"],
        profile=store.resolve(profile).provenance(),
    )
    inputs = run.path / "solutions"
    solver = {"status": "not_applicable"}
    if packet:
        if not submission:
            result = {
                "case_id": case.id,
                "outcome": "not_run",
                "solver": {"status": "not_submitted"},
                "renderer": {"status": "not_run"},
            }
            return {**run.finish("held", result=result), "case_run": str(run.path)}
        try:
            source = inside(packet, submission["source"])
            expected_source = inside(
                packet,
                f"solutions/{case.id}/question"
                + (
                    ".txt"
                    if case.input_format == "text"
                    else ".png"
                    if case.input_format == "image"
                    else ".pdf"
                ),
            )
            if source != expected_source or sha256(source) != submission["source_sha256"]:
                raise ValueError("Agent source identity does not match the question packet")
            if not submission.get("agent") or not submission.get("verification", {}).get(
                "evidence"
            ):
                raise ValueError("Agent identity and actual verification evidence required")
            if submission["verification"].get("status") == "failed":
                raise ValueError("Agent reports failed solution checks")
            solution = submission["solution_markdown"]
            if not solution.strip():
                raise ValueError("Agent solution is empty")
            shutil.copyfile(source, inputs / source.name)
            source = inputs / source.name
            solver = {
                "status": "submitted",
                "agent": submission["agent"],
                "correctness": "not_independently_graded",
                "verification": submission["verification"],
            }
        except (ValueError, KeyError, OSError) as exc:
            result = {
                "case_id": case.id,
                "outcome": "solver_failure",
                "solver": {"status": "failed", "error": str(exc)},
                "renderer": {"status": "not_run"},
            }
            return {**run.finish("failed", result=result), "case_run": str(run.path)}
    else:
        source = question_asset(case, inputs)
        solution = case.reference_solution
    (inputs / "solution.md").write_text(solution, encoding="utf-8")
    verification = (
        submission["verification"]
        if packet
        else {
            "status": "passed" if case.reference_verification.checks else "inconclusive",
            "evidence": case.reference_verification.method,
            "checks": [v.model_dump() for v in case.reference_verification.checks],
        }
    )
    manifest = {
        "schema_version": 1,
        "generated_by": "Benchmark reference renderer" if not packet else submission["agent"],
        "backend": "font-cpu",
        "profile": profile,
        "realism": "combined",
        "render": config,
        "background": background(case, inputs),
        "problems": [
            {
                "id": case.id,
                "number": case.id,
                "statement": case.question,
                "sources": [{"path": source.name}],
                "solution": "solution.md",
                "verification": verification,
            }
        ],
    }
    atomic_json(inputs / "job.json", manifest)
    argv = [
        sys.executable,
        "-m",
        "hwsol.benchmarks.worker",
        str(inputs / "job.json"),
        str(run.path / "payload"),
        str(run.path / "reports/worker.json"),
    ]
    started = time.perf_counter()
    try:
        with (
            (run.path / "logs/stdout.txt").open("w") as out,
            (run.path / "logs/stderr.txt").open("w") as err,
        ):
            child = subprocess.run(
                argv,
                stdout=out,
                stderr=err,
                timeout=120,
                check=False,
                env={**os.environ, "HWSOL_DATA_DIR": str(store.root)},
            )
        result_path = run.path / "reports/worker.json"
        worker = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {"status": "failed", "error": f"Worker exit {child.returncode}"}
        )
    except subprocess.TimeoutExpired:
        worker = {"status": "failed", "error": "worker-timeout", "error_type": "TimeoutExpired"}
    record = {
        "case_id": case.id,
        "suite": case.suite,
        "category": case.category,
        "solver": solver,
        "renderer": worker,
        "wall_seconds": time.perf_counter() - started,
    }
    if worker["status"] == "passed":
        (run.path / "rendered/pages").rmdir()
        (run.path / "rendered").rmdir()
        (run.path / "payload").rename(run.path / "rendered")
        report = json.loads((run.path / "rendered/validation_report.json").read_text())
        record["metrics"] = measurements(report, case)
        matches = (
            case.expected.outcome == "render"
            and not record["metrics"]["mathematics"]["expected_structures_missing"]
        )
        record["outcome"] = "rendered" if matches else "unexpected_success_or_missing_structure"
    else:
        matches = case.expected.outcome == "reject" and case.expected.error_contains in worker.get(
            "error", ""
        )
        record["outcome"] = "expected_rejection" if matches else "renderer_failure"
    record["expectation_passed"] = matches
    # An expected rejection passes the benchmark, but remains a failed renderer run.
    run.finish("passed" if worker["status"] == "passed" else "failed", result=record)
    return {**record, "case_run": str(run.path)}


def run_benchmark(
    *,
    suite="smoke",
    profile="bundled-default@v1",
    workspace=None,
    run_id=None,
    purpose="evaluation",
    packet: Path | None = None,
    submissions: Path | None = None,
    glyph_metrics="contextual",
):
    corpus = load_corpus()
    cases = select(corpus, suite, purpose)
    entries = {}
    if bool(packet) != bool(submissions):
        raise ValueError("End-to-end requires both packet and submissions")
    if packet:
        verify_run(packet)
        packet_record = json.loads((packet / "run.json").read_text())
        if packet_record["config"].get("corpus_sha256") != sha256(corpus_path()):
            raise ValueError("Agent packet belongs to a different benchmark")
        items = json.loads(submissions.read_text())["submissions"]
        entries = {r["case_id"]: r for r in items}
        if len(entries) != len(items) or set(entries) - {c.id for c in cases}:
            raise ValueError("Duplicate or unexpected Agent submission ID")
    config = {
        "seed": 4206,
        "dpi": 96,
        "glyph_metrics": glyph_metrics,
        "semantic_layout": True,
        "math_strokes": "personal",
    }
    run = Run.create(
        "benchmark",
        workspace=workspace,
        run_id=run_id,
        seed=config["seed"],
        config={
            "benchmark_version": corpus.benchmark_version,
            "corpus_sha256": sha256(corpus_path()),
            "suite": suite,
            "purpose": purpose,
            "path": "end-to-end" if packet else "render-only",
            "render": config,
            "holdout_evaluation_only": True,
        },
        profile=profile_store(workspace).resolve(profile).provenance(),
    )
    if submissions:
        shutil.copyfile(submissions, run.path / "logs/agent-submissions.json")
    records = []
    try:
        for case in cases:
            records.append(execute_case(case, run, profile, config, entries.get(case.id), packet))
        # A comparison anthology is not a successful homework delivery and cannot advance latest.
        writer = PdfWriter()
        page_number = 0
        for record in records:
            child = Path(record["case_run"])
            pdf = child / "rendered/homework_solutions.pdf"
            if pdf.is_file():
                for page in PdfReader(pdf).pages:
                    writer.add_page(page)
                page_number += 1
                shutil.copyfile(
                    child / "rendered/pages/01.png",
                    run.path / f"rendered/pages/{page_number:02d}.png",
                )
            (run.path / "comparisons" / f"{record['case_id']}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
        if page_number:
            writer.write(run.path / "rendered/homework_solutions.pdf")
        writer.close()
        summary = {
            "benchmark_version": corpus.benchmark_version,
            "path": "end-to-end" if packet else "render-only",
            "case_count": len(cases),
            "expectations_passed": sum(r.get("expectation_passed", False) for r in records),
            "rendered_pages": page_number,
            "expected_rejections": sum(r.get("outcome") == "expected_rejection" for r in records),
            "solver_failures": sum(r.get("outcome") == "solver_failure" for r in records),
            "renderer_failures": sum(r.get("outcome") == "renderer_failure" for r in records),
            "not_run": sum(r.get("outcome") == "not_run" for r in records),
            "independent_content_style_evaluation": "pending",
            "cases": records,
        }
        atomic_json(run.path / "reports/benchmark.json", summary)
        status = "passed" if summary["expectations_passed"] == len(cases) else "held"
        return run.finish(status, result={k: v for k, v in summary.items() if k != "cases"})
    except BaseException as exc:
        run.fail(exc)
        raise


def evaluate_observations(run_path: Path, observations: Path, *, workspace=None, run_id=None):
    verify_run(run_path)
    rows = json.loads((run_path / "reports/benchmark.json").read_text())["cases"]
    incoming = json.loads(observations.read_text())["observations"]
    indexed = {r["case_id"]: r for r in rows}
    if len({o["case_id"] for o in incoming}) != len(incoming):
        raise ValueError("Duplicate observation ID")
    run = Run.create(
        "benchmark-observations",
        workspace=workspace,
        run_id=run_id,
        config={
            "target_run": str(run_path),
            "target_sha256": sha256(run_path / "run.json"),
            "observations_sha256": sha256(observations),
        },
    )
    try:
        results = {}
        for item in incoming:
            case_id = item["case_id"]
            row = indexed[case_id]
            pdf = Path(row["case_run"]) / "rendered/homework_solutions.pdf"
            if not pdf.exists():
                raise ValueError("Cannot assess a missing rendered output")
            expected = (pdf.parent / "solutions/01.md").read_text("utf-8")
            results[case_id] = observed_metrics(prose(expected), item, sha256(pdf))
        shutil.copyfile(observations, run.path / "logs/observations.json")
        atomic_json(run.path / "reports/scores.json", results)
        return run.finish(
            "passed",
            result={
                "observed_cases": len(results),
                "total_cases": len(rows),
                "complete": len(results) == len(rows),
            },
        )
    except BaseException as exc:
        run.fail(exc)
        raise
