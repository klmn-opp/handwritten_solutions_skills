"""Frozen-model context and full-homework replay, in a private experiment run."""

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter

from hwsol.config import RenderConfig
from hwsol.jobs.checks import check_answer
from hwsol.jobs.paper import pdf_image
from hwsol.jobs.schema import Verification
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store
from research.corpus.context_benchmark import manifest
from research.corpus.context_composer import FrozenBank, layout, replace_ink
from research.corpus.context_homework import complete_homework
from research.corpus.experiment import create


def render() -> Path:
    previous = Path("workspace/runs/20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2")
    frozen = json.loads((previous / "reports/candidates.json").read_text())["candidates"]
    banks = {name: FrozenBank(Path(frozen[name]["sample_run"])) for name in ["round2", "repair"]}
    profile = profile_store().resolve("personal@v1")
    benchmark = manifest()
    cfg = RenderConfig(seed=1010, font_size_pt=12, semantic_layout=True, ink="#000000")
    run = create(
        "context-composition",
        seed=1010,
        config={
            "profile": profile.provenance(),
            "models": {k: frozen[k] for k in banks},
            "benchmark_sha256": benchmark["sha256"],
            "config": cfg.model_dump(),
            "technique": "frozen glyph composition by Handnote; no context learning",
        },
    )
    atomic_json(run.path / "reports/benchmark.json", benchmark)
    all_reports = []
    try:
        homework = json.loads(Path("src/hwsol/assets/benchmarks/v1/cases.json").read_text())[
            "cases"
        ]
        selected = [
            next(c for c in homework if c["id"] == name)
            for name in [
                "core-elimination",
                "core-regression",
                "core-improper-integral",
                "core-rl-transient",
            ]
        ]
        groups = []
        for start in range(0, 32, 4):
            rows = benchmark["cases"][start : start + 4]
            source = "\n\n".join(f"### {r['id']}\n\n{r['source']}" for r in rows)
            groups.append(("benchmark", f"Context {start // 4 + 1}", source, rows))
        for i, case in enumerate(selected, 1):
            verify = Verification.model_validate(
                {
                    "status": "passed",
                    "evidence": "Fixed verified reference replay, no new solving",
                    "checks": case["reference_verification"]["checks"],
                }
            )
            checked = check_answer(verify)
            atomic_json(run.path / f"reports/homework-{i}-answer-check.json", checked)
            (run.path / f"solutions/homework-{i}-question.txt").write_text(case["question"])
            groups.append(
                ("homework", f"Problem {i}", complete_homework()[case["id"]], [case["id"]])
            )
        writers = {kind: PdfWriter() for kind in ["benchmark", "homework"]}
        for kind, identity, source, ids in groups:
            slug = identity.lower().replace(" ", "-")
            (run.path / f"solutions/{slug}.md").write_text(source)
            page, renderer, provenance = layout(source, identity, profile, cfg)
            originals = dict(renderer._outlines)
            for method in ["handnote", "round2", "repair"]:
                renderer._outlines = dict(originals)
                if method == "handnote":
                    active, records = page, provenance
                    result = {
                        "neural_items": 0,
                        "original_items": len(page.items),
                        "structure_and_placement_preserved": True,
                    }
                else:
                    active, records, result = replace_ink(
                        page, renderer, provenance, banks[method], identity
                    )
                pdf = renderer.to_vector_pdf([active])
                target = run.path / f"comparisons/{slug}-{method}.pdf"
                target.write_bytes(pdf)
                reader = PdfReader(io.BytesIO(pdf))
                if len(reader.pages) != 1:
                    raise ValueError("One source group must produce one A4 page")
                writers[kind].add_page(reader.pages[0])
                png = run.path / f"rendered/pages/{slug}-{method}.png"
                with pdfium.PdfDocument(pdf) as doc:
                    pdf_image(doc, 0, cfg.dpi / 72).save(png)
                record = {
                    "kind": kind,
                    "id": slug,
                    "method": method,
                    "cases": ids,
                    "source_sha256": sha256(run.path / f"solutions/{slug}.md"),
                    "pdf": str(target.relative_to(run.path)),
                    "png": str(png.relative_to(run.path)),
                    "pdf_sha256": sha256(target),
                    "png_sha256": sha256(png),
                    "result": result,
                    "provenance": records,
                    "formulas": renderer.formulas,
                    "matrices": renderer.matrices,
                    "layout": renderer.rows,
                    "fallback_reasons": dict(
                        Counter(r.get("neural_fallback_reason") for r in records)
                    ),
                }
                atomic_json(run.path / f"reports/{slug}-{method}.json", record)
                all_reports.append(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in {"provenance", "formulas", "matrices", "layout"}
                    }
                )
        writers["homework"].write(run.path / "rendered/homework_solutions.pdf")
        writers["benchmark"].write(run.path / "comparisons/context-benchmark.pdf")
        atomic_json(run.path / "reports/pages.json", all_reports)
        run.finish(
            "held",
            result={
                "benchmark_cases": 32,
                "benchmark_pages": 24,
                "homework_subjects": 4,
                "homework_pages": 12,
                "methods": ["handnote", "round2", "repair"],
                "structure_preserved": True,
                "end_to_end_line_generation": False,
                "visual_content_verified": False,
                "new_neural_training": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(render())
