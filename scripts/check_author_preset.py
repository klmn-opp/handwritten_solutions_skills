"""Actual installed CPU preset preflight, without granting release qualification.

Run on each real target platform. This reads the packaged compiled outlines and
checked public inputs, renders actual PDF/PNG pages, and preserves failure output.
It never creates a Profile store or changes a default or delivery pointer.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter

import hwsol
from hwsol.identity import build_identity, digest, runtime_text_capabilities
from hwsol.jobs.checks import check_answer
from hwsol.jobs.paper import PDF_LOCK
from hwsol.jobs.runtime import export_raster, load_job, validate_source
from hwsol.jobs.schema import Background
from hwsol.personal_ink.bundle import load_recipe
from hwsol.personal_ink.runtime import PersonalCpuEngine
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.presets import AUTHOR_REF, author_directory
from hwsol.profiles.store import load_snapshot


def check(output: Path, manifest: Path, profile_directory: Path | None = None) -> dict:
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "hwsol-author-preset-check/1",
        "status": "running",
        "experimental_verification_only": True,
        "qualification_granted": False,
        "production_defaults_changed": False,
        "pages": [],
    }
    try:
        profile = load_snapshot(profile_directory or author_directory())
        recipe = load_recipe(profile)
        if profile.manifest.ref != AUTHOR_REF or recipe.description.recipe != "compiled-outlines/1":
            raise ValueError("Preflight requires the compiled author-default@v1 preset")
        job = load_job(manifest)
        if job.profile not in (None, AUTHOR_REF):
            raise ValueError("Use a public Job selecting author-default or leaving Profile unset")
        if job.realism != "baseline" or job.render.model_fields_set:
            raise ValueError("Use the author's frozen baseline settings without Job overrides")
        if job.background and job.background.kind != "plain":
            raise ValueError("This release preflight covers plain paper only")
        engine = PersonalCpuEngine(profile)
        build = build_identity()
        report.update(
            profile=profile.provenance(),
            recipe_sha256=recipe.sha256,
            environment=runtime_text_capabilities(profile.font),
            build=build,
            source_sha256=digest(build["source"]),
            handnote_source_sha256=digest(build["handnote_source"]),
            runtime_module=str(Path(hwsol.__file__).resolve()),
            executable=sys.executable,
            input_manifest_sha256=sha256(manifest),
            render_settings=profile.manifest.render.model_dump(),
        )
        writer = PdfWriter()
        for index, problem in enumerate(job.problems, 1):
            if not problem.complete or (problem.background and problem.background.kind != "plain"):
                raise ValueError("Preflight requires complete problems on plain paper")
            checked = check_answer(problem.verification)
            if checked["agent_status"] != "passed":
                raise ValueError("Public preflight answers must carry passing verification")
            for source in problem.sources:
                path = inside(manifest.parent, source.path)
                validate_source(path, source)
                if source.sha256 and sha256(path) != source.sha256:
                    raise ValueError("Public input digest differs from its Job")
            answer = inside(manifest.parent, problem.solution)
            source_text = answer.read_text("utf-8")
            pdf, diagnostics = engine.render_one(
                source_text,
                problem.number,
                profile,
                profile.manifest.render,
                Background(kind="plain"),
                None,
                identity=problem.id or problem.number,
            )
            pdf_path = output / f"page-{index:02d}.pdf"
            png_path = output / f"page-{index:02d}.png"
            pdf_path.write_bytes(pdf)
            with PDF_LOCK, closing(pdfium.PdfDocument(pdf)) as document:
                if len(document) != 1:
                    raise ValueError("One top-level problem must occupy one page")
                raster = export_raster(document, 0, profile.manifest.render)
                raster.save(png_path)
            reader = PdfReader(pdf_path, strict=True)
            writer.add_page(reader.pages[0])
            candidate = diagnostics["candidate"]
            if not (
                candidate["semantic_identity_preserved"]
                and candidate["parser_trace_preserved"]
                and candidate["whole_recipe_contact_audit"]["unreviewed_contact_count"] == 0
            ):
                raise ValueError("Compiled preset failed content, parser or contact checks")
            report["pages"].append(
                {
                    "number": problem.number,
                    "id": problem.id or problem.number,
                    "answer_sha256": sha256(answer),
                    "verification": checked,
                    "pdf": pdf_path.name,
                    "pdf_sha256": sha256(pdf_path),
                    "png": png_path.name,
                    "png_sha256": sha256(png_path),
                    "diagnostics": diagnostics,
                }
            )
        combined = output / "homework_solutions.pdf"
        writer.write(combined)
        with PDF_LOCK, closing(pdfium.PdfDocument(combined)) as document:
            if len(document) != len(job.problems):
                raise ValueError("Combined PDF has the wrong page inventory")
            from PIL import Image, ImageChops

            for index, record in enumerate(report["pages"]):
                with Image.open(output / record["png"]) as png:
                    raster = export_raster(document, index, profile.manifest.render)
                    if ImageChops.difference(raster, png).getbbox():
                        raise ValueError("Merged PDF differs from the inspected page pixels")
        report.update(
            status="passed",
            page_count=len(job.problems),
            combined_pdf_sha256=sha256(combined),
            pixels_verified=True,
        )
    except Exception as error:
        report.update(status="failed", error=str(error), error_type=type(error).__name__)
        atomic_json(output / "report.json", report)
        raise
    atomic_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/release-demo/personal-job.json",
    )
    parser.add_argument(
        "--profile-directory", type=Path, help="Explicit local derivative preflight"
    )
    args = parser.parse_args()
    result = check(args.output, args.manifest, args.profile_directory)
    print(json.dumps({"status": result["status"], "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
