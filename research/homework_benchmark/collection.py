"""Optional small Session O sheet for measured, still-uncovered math symbols."""

import shutil

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.corpus.math_glyph_evidence import collection_sheet, read


def main():
    root = workspace_root()
    sources = [
        root
        / "runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5"
        / "solutions/historical-glyphs/sources.json",
        root
        / "runs/20261120T064909731073Z-v01026-session-math-polish-b9371278"
        / "solutions/session-glyphs/sources.json",
    ]
    approved = {
        row["character"]
        for source in sources
        for row in read(source)
        if row["review_status"] == "agent-accepted"
    }
    missing = set("Φκζξηψ∏∮≈∈⟨⟩‖∬ℝ")
    if approved & missing:
        raise ValueError("A requested symbol now has an accepted source; revise the minimum sheet")
    run = Run.create(
        "m1-optional-math-session-o",
        config={
            "purpose": "author collection template; not original handwriting or model training",
            "source_manifests": {str(p): sha256(p) for p in sources},
            "missing_verified_symbols": sorted(missing),
            "historical_search_scope": (
                "accepted historical and M/N symbol manifests, current Profiles, reviewed corpus "
                "labels; no claim of exhaustive review of all unlabelled PDF regions"
            ),
            "mandatory_for_cpu_use": False,
            "template_revision": 2,
            "revision_reason": "add real-number glyph exposed by semantic-preserving recovery",
            "new_neural_updates": 0,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/collection-source.py")
        prompts = [
            (r"\Phi\quad\kappa\quad\zeta", "Capital Phi; kappa; zeta"),
            (r"\xi\quad\eta\quad\psi", "Xi; eta; psi"),
            (r"\prod\quad\prod\quad\prod", "Product: small, normal, tall"),
            (r"\oint\quad\oint\quad\oint", "Closed integral: usual sizes"),
            (r"\approx\quad\in\quad\mathbb{R}", "Approximate; belongs to; real numbers"),
            (r"\langle x,y\rangle\quad\langle u,v\rangle", "Your usual angle brackets"),
            (r"\Vert x\Vert\quad\iint_D f\,dx\,dy", "Optional paired bars / integrals"),
        ]
        records = collection_sheet(run.path / "solutions/collection", "O", prompts)
        atomic_json(run.path / "reports/prompts.json", records)
        (run.path / "reports/README.md").write_text(
            "# Optional Session O\n\n"
            "This follows the earlier M/N prompt-plus-answer-line format. It targets only "
            "symbols still using reported fonts on the new homework pages. No repeated root, "
            "digit or already accepted M/N alphabet collection is requested.\n\n"
            "The first six lines cover thirteen currently missing source characters. The seventh "
            "is optional: double bars and double integrals could alternatively be evaluated as "
            "explicit compositions of existing personal strokes, not original paired glyphs.\n\n"
            "Write naturally, with spaces between separate samples, using the usual pen. Keep "
            "the original colour. A completed scan can be placed in "
            "source_pdf/session-O_source.pdf. "
            "Keep this printed blank template outside the original-source corpus.\n\n"
            "Prompts are intended text, never verified labels. Import must preserve the actual "
            "written content, original PDF, 300-DPI RGB crop and coordinate transforms. "
            "Ambiguous writing remains held. Collection is not neural-network training.\n\n"
            "No reliable approved source was found in the searched registries. Unreviewed "
            "historical PDF regions may still contain these symbols; this is an optional shortcut, "
            "not a claim that every historical page has been exhausted.\n",
            encoding="utf-8",
        )
        run.finish(
            "held",
            result={
                "template_pages": 1,
                "priority_source_characters": 13,
                "optional_source_characters": 2,
                "actual_new_handwriting_samples": 0,
                "label_status": "not-collected",
                "pdf": "solutions/collection/session-O.pdf",
            },
        )
        verify_run(run.path)
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    main()
