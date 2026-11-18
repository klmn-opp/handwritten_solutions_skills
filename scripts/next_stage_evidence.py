"""Reproduce CPU comparisons without publishing or activating a personal candidate."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from pypdf import PdfWriter

from hwsol.identity import build_identity
from hwsol.jobs.runtime import render_job, verify_job
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.metrics import measure_font, normalized_asset
from hwsol.profiles.store import ProfileStore, compare
from hwsol.training.coverage import collection_plan

ROOT = Path(__file__).resolve().parents[1]


class CandidateEvidenceStore(ProfileStore):
    """An explicit Studio-only candidate reader; does not change Runtime resolution."""

    def __init__(self, root: Path, candidate: str):
        super().__init__(root)
        self.candidate = candidate

    def resolve(self, spec=None, *, candidate=False):
        return super().resolve(spec, candidate=candidate or spec == self.candidate)


def prepare_input(source: Path, target: Path) -> dict:
    job = json.loads(source.read_bytes())
    target.mkdir(parents=True)
    paths = {p["solution"] for p in job["problems"]}
    paths.update(s["path"] for p in job["problems"] for s in p["sources"])
    for path in paths:
        dest = target / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parent / path, dest)
    return job


def coverage(report: dict) -> dict:
    rows = [r for page in report["pages"] for r in page["glyph_provenance"]]
    math = [r for r in rows if r.get("latex") or r.get("structure")]
    return {
        "visible_items": len(rows),
        "math_items": len(math),
        "personal_math_items": sum(r["personal_collected"] for r in math),
        "personal_math_fraction": sum(r["personal_collected"] for r in math) / max(1, len(math)),
        "source_types": dict(Counter(r["source_type"] for r in rows)),
        "structures": {
            key: dict(Counter(r["source_type"] for r in rows if r["structure"] == key))
            for key in sorted({r["structure"] for r in rows if r["structure"]})
        },
        "uncollected": sorted(
            {r["codepoint"] for r in rows if r.get("codepoint") and not r["personal_collected"]}
        ),
    }


def run(output: Path, candidate: str):
    output.mkdir(parents=True, exist_ok=False)
    store = CandidateEvidenceStore(ROOT / "profiles/realism-studio", candidate)
    old, new = store.resolve("personal@v3"), store.resolve(candidate)
    summary = {
        "build": build_identity(),
        "old_profile": old.provenance(),
        "candidate": new.provenance(),
        "comparison": compare(old, new),
        "jobs": {},
    }

    def render(label, job, folder):
        path = folder / f"{label}.json"
        atomic_json(path, job)
        target = output / label
        render_job(path, target, store)
        verified = verify_job(target, pixels=True)
        report = json.loads((target / "validation_report.json").read_bytes())
        summary["jobs"][label] = {
            **verified,
            "pdf_sha256": sha256(target / "homework_solutions.pdf"),
            "coverage": coverage(report),
        }
        return report

    archived = ROOT / "out/realism/rl-size-feedback/candidate/manifest.json"
    inputs = output / "five-input"
    original = prepare_input(archived, inputs)
    original["render"].pop("policy", None)
    original["realism"] = "combined"
    for mode in ("raw", "normalized", "contextual"):
        job = json.loads(json.dumps(original))
        job["render"].update(glyph_metrics=mode, semantic_layout=True)
        render("five-" + mode, job, inputs)
    job["profile"], job["profile_sha256"] = candidate, new.manifest_sha256
    render("five-candidate", job, inputs)

    # Identical input, Profile, paper, seed and explicit factor settings on all three pages.
    sample = json.loads(json.dumps(original))
    sample["problems"] = [sample["problems"][0]]
    sample["problems"][0].update(
        number="1",
        verification={"status": "not_checked"},
        solution="words.md",
        statement="Glyph scale comparison",
    )
    (inputs / "words.md").write_text(
        "Happy people quickly apply typography.\n\n"
        "minimum momentum parallel gradient sample velocity.\n\n"
        "Alphabet BODY digits 0123456789.\n\n"
        "p y g q j b d f h k l t i.\n\n"
        "The median is the middle ordered value.\n\n"
        "Newton's equations explain uniform motion.\n\n"
        r"$$p+y+g+q=2,\quad x_1^2+\frac{1}{2}=\frac{3}{2}$$",
        encoding="utf-8",
    )
    combined = PdfWriter()
    for mode in ("raw", "normalized", "contextual"):
        sample["render"].update(glyph_metrics=mode, semantic_layout=True)
        render("glyph-" + mode, sample, inputs)
        combined.append(output / f"glyph-{mode}/homework_solutions.pdf")
    combined.write(output / "glyph-normalization-comparison.pdf")
    combined.close()

    metrics = measure_font(old.font, old.manifest.personal_codepoints)
    normalized, _ = normalized_asset(
        str(old.font.resolve()), sha256(old.font), old.manifest.personal_codepoints
    )
    measured = measure_font(normalized, old.manifest.personal_codepoints)
    atomic_json(output / "raw-glyph-metrics.json", metrics)
    atomic_json(output / "normalized-glyph-metrics.json", measured)
    summary["scale_measurements"] = {}
    for label, data in (("raw", metrics), ("normalized", measured)):
        summary["scale_measurements"][label] = {
            group: float(np.std(values) / np.mean(values))
            for group in ("lowercase", "ascender", "descender", "capital", "digit")
            if (
                values := [
                    r["ink_bounds_em"][3] - data["baseline_em"]
                    for r in data["glyphs"]
                    if r["category"] == group
                ]
            )
        }

    structures = json.loads(json.dumps(sample))
    structures["problems"][0]["solution"] = "structures.md"
    (inputs / "structures.md").write_text(
        "Mean, fraction, radical, vector and matrix structures.\n\n"
        r"$$\bar{x}=\frac{2+4+6}{3}=4$$"
        + "\n\n"
        + r"$$\sqrt{\frac{1}{2}}+\overline{x+1}+\vec{v}$$"
        + "\n\n"
        + r"$$A=\begin{pmatrix}1&2\\3&4\end{pmatrix}$$"
        + "\n\n"
        + r"$$B=\begin{bmatrix}1&2\\3&4\end{bmatrix}$$",
        encoding="utf-8",
    )
    for mode in ("off", "personal"):
        structures["render"].update(glyph_metrics="normalized", math_strokes=mode)
        render("math-" + mode, structures, inputs)

    history = ROOT / "out/realism/personal-v2/holdout/combined/manifest.json"
    regress_inputs = output / "regression-input"
    regression = prepare_input(history, regress_inputs)
    for semantic in (False, True):
        regression["render"]["semantic_layout"] = semantic
        report = render(
            "formula-fixed" if semantic else "formula-legacy", regression, regress_inputs
        )
        formulas = report["pages"][0]["formula_groups"]
        summary["jobs"]["formula-fixed" if semantic else "formula-legacy"]["formula_positions"] = [
            {"latex": f["latex"], "position_px": f["position_px"], "role": f["role"]}
            for f in formulas
        ]
    historical = json.loads((history.parent.parent / "measurements.json").read_bytes())
    summary["regression_context"] = {
        "previous_indent_error_pct": historical["combined"]["indent_error"],
        "usage": "known counterexample; no longer independent holdout",
    }

    sources = sorted((ROOT / "out/realism").glob("**/solutions/*.md"))
    queue = collection_plan(old, sources, output / "collection-queue")
    summary["collection"] = {
        "unique_sources": len(queue["sources"]),
        "source_failures": queue["source_failures"],
        "missing_personal": [r for r in queue["glyphs"] if not r["personal_variants"]],
    }
    summary["quality"] = {
        "content_correctness": "declared checks passed; independent reading pending",
        "word_scale_coordination": "outline-class variation measured; author review pending",
        "math_coverage_and_readability": "coverage measured; independent readability pending",
        "author_style_similarity": "not-evaluated",
        "page_naturalness": "known counterexample tested; independent assessment pending",
        "engineering_consistency": "PDF/PNG verified; clean-wheel replay reported separately",
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "jobs": len(summary["jobs"]), "candidate": candidate}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    run(args.output.absolute(), args.candidate)
