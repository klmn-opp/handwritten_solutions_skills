"""Four subject CPU regression pages: explicit baseline, never a claimed neural output."""

import json
from pathlib import Path

from hwsol.jobs.runtime import render_job, verify_job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store
from research.corpus.contracts import ink_decision
from research.corpus.experiment import create


def render(run_id: str):
    source = Path("src/hwsol/assets/benchmarks/v1/cases.json")
    cases = json.loads(source.read_text())["cases"]
    selected = [
        next(c for c in cases if c["id"] == name)
        for name in [
            "core-elimination",
            "core-regression",
            "core-improper-integral",
            "core-rl-transient",
        ]
    ]
    run = create(
        "corpus-cpu-compatibility-delivery",
        run_id=run_id,
        seed=902,
        config={
            "backend": "font-cpu",
            "profile": "personal@v1",
            "reference_source_sha256": sha256(source),
            "neural_generated_glyphs": 0,
        },
    )
    try:
        problems = []
        for i, case in enumerate(selected, 1):
            (run.path / "solutions" / f"{i}.md").write_text(case["reference_solution"])
            (run.path / "solutions" / f"question-{i}.txt").write_text(case["question"])
            problems.append(
                {
                    "id": case["id"],
                    "number": str(i),
                    "statement": case["question"],
                    "sources": [{"path": f"question-{i}.txt"}],
                    "solution": f"{i}.md",
                    "verification": {
                        "status": "passed"
                        if case["reference_verification"]["checks"]
                        else "inconclusive",
                        "evidence": "Fixed reference replay with exact checks; no new solving",
                        "checks": case["reference_verification"]["checks"],
                    },
                }
            )
        manifest = {
            "schema_version": 1,
            "backend": "font-cpu",
            "generated_by": "v0.9 CPU compatibility reference replay; no neural handwriting",
            "profile": "personal@v1",
            "realism": "combined",
            "background": {"kind": "plain"},
            "render": {
                "seed": 902,
                "dpi": 150,
                "paper": "plain",
                "glyph_metrics": "raw",
                "semantic_layout": True,
                "math_strokes": "personal",
            },
            "problems": problems,
        }
        path = run.path / "solutions/manifest.json"
        atomic_json(path, manifest)
        # Run contract creates this empty placeholder; Runtime itself creates page outputs.
        (run.path / "rendered/pages").rmdir()
        render_job(path, run.path / "rendered", store=profile_store())
        verified = verify_job(run.path / "rendered", pixels=True)
        atomic_json(run.path / "reports/verification.json", verified)
        atomic_json(
            run.path / "reports/neural-fallback.json",
            {
                "decision": ink_decision("x", None, True),
                "neural_glyphs_used": 0,
                "unchanged_structure_owner": "Handnote/MathText",
                "model_training": False,
            },
        )
        return run.finish(
            "passed",
            result={"pages": 4, "backend": "font-cpu", "neural_glyphs_used": 0},
            deliverable=False,
        )
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    import sys

    print(render(sys.argv[1]))
