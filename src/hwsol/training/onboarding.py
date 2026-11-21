"""One local, review-gated scan-to-preview workflow using the existing CPU Studio."""

import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.schema import ProfileError, name
from hwsol.profiles.store import ProfileStore
from hwsol.training.samples import import_sample
from hwsol.training.studio import evaluate, train
from hwsol.workspace import Run


def build_from_scans(
    store: ProfileStore,
    profile: str,
    sources: list[Path],
    template: Path,
    *,
    labels_reviewed: bool = False,
) -> dict:
    """Retain inputs, build a cumulative candidate, and return its actual preview.

    The explicit attestation concerns the observed ink, not the printed prompts.
    It is recorded with input hashes; it is not an automated handwriting labeler.
    Publication and activation remain separate, explicit existing commands.
    """
    name(profile)
    if not labels_reviewed:
        raise ProfileError(
            "Review the actual ink against every template label first, then pass "
            "--labels-reviewed. Printed prompts alone are not verified labels."
        )
    if not sources or not template.is_file() or any(not p.is_file() for p in sources):
        raise ProfileError("Provide the original template JSON and existing scan files")
    if any(p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".pdf"} for p in sources):
        raise ProfileError("Scans must be PNG, JPEG, or a scanned PDF of the matching template")
    if sum(p.stat().st_size for p in [template, *sources]) > 128 * 1024**2:
        raise ProfileError("A collection round is limited to 128 MiB; split larger collections")
    run = Run.create("profile-build", config={"profile": profile, "labels_reviewed": True})
    try:
        inputs = run.path / "inputs"
        inputs.mkdir()
        retained_template = inputs / "template.json"
        shutil.copyfile(template, retained_template)
        retained = []
        for index, source in enumerate(sources):
            target = inputs / f"scan-{index + 1:03d}{source.suffix.lower()}"
            shutil.copyfile(source, target)
            retained.append(target)
        spec = json.loads(retained_template.read_text("utf-8"))
        pages, sets, dpi = spec.get("pages"), spec.get("sets", 1), spec.get("dpi")
        if not (
            type(pages) is int
            and 1 <= pages <= 128
            and type(sets) is int
            and 1 <= sets <= 8
            and type(dpi) is int
            and 72 <= dpi <= 600
        ):
            raise ProfileError("Invalid template page count, set count, or DPI")
        review = {
            "schema_version": 1,
            "review": "caller attests actual ink matches template labels; not OCR approval",
            "labels_reviewed": True,
            "template_sha256": sha256(retained_template),
            "originals": [{"path": p.name, "sha256": sha256(p)} for p in retained],
        }
        atomic_json(run.path / "reports/label-review.json", review)
        images = _scan_pages(retained, run.path / "scan-pages", dpi, pages * sets)
        if len(images) != pages * sets:
            raise ProfileError(
                f"Expected {pages * sets} scanned pages ({pages} pages × {sets} sets), "
                f"received {len(images)}. Include every page in printed order."
            )
        sample = import_sample(store, profile, images, template=retained_template)
        if sample.state != "accepted":
            raise ProfileError(f"Scan retained but not accepted: {sample.reason}")
        build = train(store, profile)
        assessment = evaluate(store, build["candidate"])
        atomic_json(run.path / "reports/evaluation.json", assessment)
        if assessment["status"] != "passed":
            raise ProfileError(
                f"Candidate evaluation failed; retained at {assessment['run']}: "
                f"{assessment.get('error')}"
            )
        candidate = store.resolve(build["candidate"], candidate=True)
        requested = {ord(cell["char"]) for cell in spec["cells"]}
        personal = set(candidate.manifest.personal_codepoints)
        preview = run.path / "rendered"
        shutil.copytree(Path(assessment["run"]) / "preview", preview, dirs_exist_ok=True)
        result = {
            "candidate": build["candidate"],
            "candidate_sha256": candidate.manifest_sha256,
            "sample": sample.id,
            "build_run": build["run"],
            "evaluation_run": assessment["run"],
            "preview_pdf": str(preview / "probe.pdf"),
            "preview_pages": str(preview / "pages"),
            "operation": "CPU glyph extraction and font construction; no neural training",
            "engineering_evaluation": "passed",
            "manual_style_review": "pending",
            "personal_characters": "".join(chr(cp) for cp in sorted(personal)),
            "requested_without_personal_ink": "".join(
                chr(cp) for cp in sorted(requested - personal)
            ),
            "published": False,
            "activated": False,
            "next_step": (
                "Inspect the preview and missing personal coverage. If acceptable, run "
                f"hwsol profile publish --profile {profile} "
                f"--version {candidate.manifest.version} --activate"
            ),
        }
        return run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise ProfileError(
            f"Profile build failed; inputs and failure retained at {run.path}: {exc}"
        ) from exc


def _scan_pages(sources: list[Path], output: Path, dpi: int, limit: int) -> list[Path]:
    """Make ordered lossless image derivatives; preserve every original separately."""
    output.mkdir()
    result = []

    def save(image: Image.Image) -> None:
        if len(result) >= limit:
            raise ProfileError(f"More scans than the template's {limit} printed pages")
        target = output / f"page-{len(result) + 1:04d}.png"
        image.convert("RGB").save(target)
        result.append(target)

    for source in sources:
        if source.suffix == ".pdf":
            import pypdfium2 as pdfium

            from hwsol.jobs.paper import PDF_LOCK

            with PDF_LOCK, pdfium.PdfDocument(source) as document:
                if len(result) + len(document) > limit:
                    raise ProfileError(f"More scans than the template's {limit} printed pages")
                for page in document:
                    try:
                        width, height = page.get_size()
                        if max(width, height) * dpi / 72 > 10000:
                            raise ProfileError("Scanned PDF page exceeds 10000 raster pixels")
                        bitmap = page.render(scale=dpi / 72)
                        try:
                            save(bitmap.to_pil())
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
        else:
            with Image.open(source) as image:
                save(ImageOps.exif_transpose(image))
    return result
