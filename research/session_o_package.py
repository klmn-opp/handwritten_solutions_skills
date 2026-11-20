"""Package approved isolated pages and evidence; never activate a Profile."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfWriter

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run, workspace_root
from research.author_feedback_delivery import snapshot_sources


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    approval = read(args.approval)
    if approval["scope"] != "isolated-session-o-24-page-delivery" or not approval["root_visual_go"]:
        raise ValueError("A completed, scoped root visual review is required")
    parents = {key: Path(record["path"]) for key, record in approval["parents"].items()}
    for key, parent in parents.items():
        verify_run(parent)
        if sha256(parent / "run.json") != approval["parents"][key]["sha256"]:
            raise ValueError("Reviewed parent changed")
    for name, digest in approval["independent_evidence_sha256"].items():
        if sha256(Path(name)) != digest:
            raise ValueError("Independent review changed")
    run = Run.create(
        "session-o-reviewed-24-page-delivery",
        seed=1909,
        config={"approval": approval, "production_activation": False, "new_neural_updates": 0},
    )
    print(run.path, flush=True)
    try:
        snapshot_sources(run, parents)
        atomic_json(run.path / "reports/approval.json", approval)
        main_run, extra_run, bank_run = parents["main"], parents["fresh"], parents["bank"]
        bank = run.path / "solutions/candidate-profile/math-bank"
        shutil.copytree(bank_run / "solutions/glyphs", bank)
        for key, parent in parents.items():
            shutil.copytree(parent / "reports", run.path / f"reports/inherited/{key}")
        for number, (name, digest) in enumerate(approval["independent_evidence_sha256"].items()):
            out = run.path / f"reports/review/{number:02}-{Path(name).name}"
            out.parent.mkdir(exist_ok=True)
            shutil.copyfile(name, out)
            if sha256(out) != digest:
                raise ValueError("Review copy changed")
        atomic_json(
            run.path / "solutions/candidate-profile/profile.json",
            {
                "schema": "isolated-personal-math-candidate/1",
                "profile_name": "personal-session-o-candidate",
                "source_bank": "math-bank/sources.json",
                "source_bank_sha256": sha256(bank / "sources.json"),
                "accepted_unicode": sorted({r["character"] for r in read(bank / "sources.json")}),
                "baseline_profile_and_layout_assets": str(main_run / "solutions"),
                "baseline_receipt_sha256": sha256(main_run / "run.json"),
                "renderer": "research.session_o_resolver.SessionOResolver",
                "new_content_entry": "research.session_o_fresh_pages",
                "source_selection_seed": 1909,
                "neural_weight_updates": 0,
                "public_runtime_activation": False,
                "author_style_approval": False,
                "rollback": approval["frozen_baseline"],
                "coverage": (
                    "13 reviewed Unicode roles; tested pages only, not arbitrary mathematics"
                ),
            },
        )
        rows, page_map = [], []
        for offset, parent, count in [(0, main_run, 20), (20, extra_run, 4)]:
            for n in range(1, count + 1):
                for suffix in ("pdf", "png"):
                    source = parent / f"rendered/pages/{n:02}.{suffix}"
                    dest = run.path / f"rendered/pages/{n + offset:02}.{suffix}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
                    if sha256(source) != sha256(dest):
                        raise ValueError("An approved page changed during packaging")
                page_map.append({"page": n + offset, "parent": str(parent), "parent_page": n})
            rows.extend(
                {**r, "page": r["page"] + offset}
                for r in read(parent / "reports/glyph-provenance.json")
            )
        with PdfWriter() as writer:
            for n in range(1, 25):
                writer.append(run.path / f"rendered/pages/{n:02}.pdf")
            writer.write(run.path / "rendered/homework_solutions.pdf")
        baseline = Path(approval["frozen_baseline"]["path"])
        with PdfWriter() as writer:
            writer.append(baseline / "rendered/homework_solutions.pdf")
            writer.append(extra_run / "comparisons/baseline.pdf")
            writer.write(run.path / "comparisons/baseline-24-pages.pdf")
        checks = []
        with pdfium.PdfDocument(run.path / "rendered/homework_solutions.pdf") as document:
            if len(document) != 24:
                raise ValueError("Wrong merged page count")
            for n in range(24):
                with closing(document[n]) as page, closing(page.render(scale=300 / 72)) as bitmap:
                    rgb = bitmap.to_pil().convert("RGB")
                    expected = Image.open(run.path / f"rendered/pages/{n + 1:02}.png").convert(
                        "RGB"
                    )
                    if rgb.size != expected.size or rgb.tobytes() != expected.tobytes():
                        raise ValueError(f"Merged PDF differs from approved page {n + 1}")
                checks.append({"page": n + 1, "merged_pdf_pixels_exact": True})
        with zipfile.ZipFile(
            run.path / "rendered/pages-300dpi.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for n in range(1, 25):
                archive.write(run.path / f"rendered/pages/{n:02}.png", f"pages/{n:02}.png")
        maths = [r for r in rows if r["context"] == "mathematics"]
        fallback = [r for r in maths if r["source_type"] in {"external-font", "inherited-font"}]
        summary = {
            "pages": 24,
            "math_visible_items": len(maths),
            "font_fallbacks": len(fallback),
            "new_source_placements": sum("session_o_source" in r for r in maths),
            "source_types": dict(Counter(r["source_type"] for r in maths)),
            "all_merged_page_pixels_exact": True,
            "root_and_independent_agent_visual_go": True,
            "author_style_acceptance": False,
            "production_release_go": False,
            "neural_updates": 0,
            "known_limits": approval["known_limits"],
        }
        if len(maths) != 2932 or fallback:
            raise ValueError("Expected accepted tested-page coverage differs")
        atomic_json(run.path / "reports/glyph-provenance.json", rows)
        atomic_json(run.path / "reports/page-map.json", page_map)
        atomic_json(run.path / "reports/package-checks.json", checks)
        atomic_json(run.path / "reports/summary.json", summary)
        protected = approval["protected_sha256"]
        if any(sha256(Path(p)) != digest for p, digest in protected.items()):
            raise ValueError("Protected production/source asset changed")
        run.finish("held", result=summary)
        verify_run(run.path)
        registry = workspace_root() / "registry/increments" / f"{run.path.name}.json"
        atomic_json(
            registry,
            {
                "schema": "research-candidate-increment/1",
                "run": str(run.path),
                "run_sha256": sha256(run.path / "run.json"),
                "production_activation": False,
                "status": "author-review-pending",
            },
        )
    except Exception as exc:
        # Never mutate a run whose terminal receipt was already written.
        if read(run.path / "run.json")["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    main()
