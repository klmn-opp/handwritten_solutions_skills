"""Render into a sibling staging directory and publish only a fully verified assignment."""

import io
import json
import shutil
import tempfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

from hwsol.config import RenderConfig
from hwsol.jobs.checks import check_answer
from hwsol.jobs.paper import PDF_LOCK, pdf_image
from hwsol.jobs.schema import Background, Job, Source
from hwsol.profiles.io import atomic_json, inside, lock, sha256
from hwsol.profiles.store import ProfileStore

PDF = "homework_solutions.pdf"
REPORT = "validation_report.json"


def load_job(path: Path) -> Job:
    return Job.model_validate_json(path.read_text("utf-8"))


def validate_source(path: Path, ref: Source):
    ext = path.suffix.lower()
    if ext == ".pdf":
        count = len(PdfReader(path).pages)
        if not count or (ref.page or 1) > count:
            raise ValueError(f"Source PDF page out of range: {ref.path}")
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        with Image.open(path) as image:
            if (ref.page or 1) > getattr(image, "n_frames", 1):
                raise ValueError("Source image frame out of range")
            image.verify()
    elif ext in {".txt", ".md"}:
        if not path.read_text("utf-8").strip():
            raise ValueError(f"Empty problem source: {ref.path}")
        if ref.page not in (1, None):
            raise ValueError("Page reference on an unpaginated source")
    else:
        raise ValueError(f"Unsupported problem source type: {ext}")


def snapshot(path: Path, stage: Path, area: str, cache: dict) -> str:
    key = (area, path.resolve())
    if key not in cache:
        target = f"{area}/{len(cache) + 1:02d}{path.suffix.lower()}"
        dest = inside(stage, target)
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
        cache[key] = target
    return cache[key]


def verify_job(root: Path) -> dict:
    """Check files, order, page/source identities and dimensions independently of Profile state."""
    report = json.loads(inside(root, REPORT).read_text("utf-8"))
    if report.get("status") != "passed":
        raise ValueError("Job does not have a successful validation report")
    job = load_job(inside(root, "manifest.json"))
    reader = PdfReader(inside(root, PDF), strict=True)
    n = len(job.problems)
    records = report["pages"]
    if not (n == len(reader.pages) == report["selected_problem_count"] == len(records)):
        raise ValueError("Selected problem count and PDF/page inventory disagree")
    expected = {PDF, "manifest.json"}
    for i, (p, page, record) in enumerate(zip(job.problems, reader.pages, records, strict=True), 1):
        png = f"pages/{i:02d}.png"
        if p.solution != f"solutions/{i:02d}.md":
            raise ValueError("Solution paths must follow page order")
        expected.update([png, p.solution, *(src.path for src in p.sources)])
        if p.background and p.background.path:
            expected.add(p.background.path)
        if record["number"] != p.number or page.get("/HwsolNumber") != p.number:
            raise ValueError("Problem identity or PDF page order mismatch")
        if page.get("/HwsolID") != (p.id or p.number) or record.get("id") != (p.id or p.number):
            raise ValueError("Problem ID mismatch")
        if record["verification"] != check_answer(p.verification):
            raise ValueError("Verification evidence mismatch")
        if page.get("/HwsolSolutionSHA256") != sha256(inside(root, p.solution)):
            raise ValueError("Solution source no longer matches PDF page")
        for src in p.sources:
            if src.sha256 != sha256(inside(root, src.path)):
                raise ValueError("Problem source digest mismatch")
        with Image.open(inside(root, png)) as image:
            size = (
                job.render.mm(job.render.page_width_mm),
                job.render.mm(job.render.page_height_mm),
            )
            if image.format != "PNG" or image.size != size:
                raise ValueError("Wrong PNG format or size")
            image.verify()
        if any(
            abs(float(actual) - px * 72 / job.render.dpi) > 0.1
            for actual, px in zip((page.mediabox.width, page.mediabox.height), size, strict=True)
        ):
            raise ValueError("Wrong PDF page dimensions")
    inventory = report["artifacts"]
    if set(inventory) != expected:
        raise ValueError("Incomplete artifact inventory")
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_files != expected | {REPORT}:
        raise ValueError("Unexpected or missing job files")
    for path, digest in inventory.items():
        if sha256(inside(root, path)) != digest:
            raise ValueError(f"Artifact digest mismatch: {path}")
    return {"status": "passed", "page_count": n, "problem_order": [p.number for p in job.problems]}


def build(manifest: Path, stage: Path, store: ProfileStore, progress: dict):
    from hwsol.jobs.layout import render_one

    job = load_job(manifest)
    root = manifest.resolve().parent
    selected = store.resolve(job.profile)
    if job.profile_sha256 and job.profile_sha256 != selected.manifest_sha256:
        raise ValueError("Pinned Profile digest does not match")
    job.profile, job.profile_sha256 = selected.manifest.ref, selected.manifest_sha256
    settings = selected.manifest.render.model_dump()
    settings.update(job.render.model_dump(exclude_unset=True))
    job.render = RenderConfig.model_validate(settings)
    default_paper = store.settings().get("background")
    cache = {}
    validated = set()
    pages = []
    writer = PdfWriter()
    (stage / "solutions").mkdir()
    (stage / "pages").mkdir()
    for i, problem in enumerate(job.problems, 1):
        progress["problem"] = problem.number
        if not problem.complete:
            raise ValueError("Incomplete problem statement. Supply the missing information.")
        for src in problem.sources:
            src.path = snapshot(inside(root, src.path), stage, "inputs", cache)
            path = stage / src.path
            digest = sha256(path)
            if src.sha256 and src.sha256 != digest:
                raise ValueError(f"Input digest mismatch for {src.path}")
            src.sha256 = digest
            if (path, src.page) not in validated:
                validate_source(path, src)
                validated.add((path, src.page))
        answer = check_answer(problem.verification)
        original = inside(root, problem.solution)
        problem.solution = f"solutions/{i:02d}.md"
        target = stage / problem.solution
        shutil.copyfile(original, target)
        bg = problem.background or job.background
        asset = None
        if bg is None and default_paper is not None:
            bg = Background.model_validate(default_paper)
            if bg.path:
                asset = Path(bg.path).expanduser()
                if not asset.is_absolute():
                    asset = inside(store.root, bg.path)
        bg = (bg or Background(kind=job.render.paper)).model_copy()
        if bg.path:
            asset = asset or inside(root, bg.path)
            bg.path = snapshot(asset, stage, "paper", cache)
            asset = stage / bg.path
        problem.background = bg
        blob, diagnostics = render_one(
            target.read_text("utf-8"), problem.number, selected, job.render, bg, asset
        )
        page = writer.add_page(PdfReader(io.BytesIO(blob), strict=True).pages[0])
        page[NameObject("/HwsolID")] = TextStringObject(problem.id or problem.number)
        page[NameObject("/HwsolNumber")] = TextStringObject(problem.number)
        page[NameObject("/HwsolSolutionSHA256")] = TextStringObject(sha256(target))
        pages.append(
            {
                "id": problem.id or problem.number,
                "number": problem.number,
                "verification": answer,
                **diagnostics,
            }
        )
    job.background = None  # Resolved per problem, including packaged custom paper.
    writer.add_metadata({"/Creator": job.generated_by, "/Title": "Homework solutions"})
    writer.write(stage / PDF)
    writer.close()
    # The exported PNG is a raster of the actual merged PDF page.
    with PDF_LOCK, closing(pdfium.PdfDocument(stage / PDF)) as doc:
        if len(doc) != len(pages):
            raise ValueError("Merged PDF lost a problem")
        for index in range(len(doc)):
            image = pdf_image(doc, index, job.render.dpi / 72)
            size = (
                job.render.mm(job.render.page_width_mm),
                job.render.mm(job.render.page_height_mm),
            )
            if image.size != size:
                if any(abs(a - b) > 1 for a, b in zip(image.size, size, strict=True)):
                    raise ValueError("Unexpected raster dimensions")
                image = image.resize(size, Image.Resampling.LANCZOS)
            image.save(stage / f"pages/{index + 1:02d}.png", dpi=(job.render.dpi, job.render.dpi))
    atomic_json(stage / "manifest.json", job.model_dump(mode="json"))
    artifacts = {
        p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()
    }
    atomic_json(
        stage / REPORT,
        {
            "status": "passed",
            "schema_version": 1,
            "profile": selected.provenance(),
            "selected_problem_count": len(pages),
            "pages": pages,
            "packages": {
                k: version(k)
                for k in (
                    "handwritten-solutions",
                    "handnote",
                    "pypdf",
                    "pypdfium2",
                    "pillow",
                    "fonttools",
                    "sympy",
                    "numpy",
                    "matplotlib",
                )
            },
            "artifacts": artifacts,
        },
    )
    return verify_job(stage)


def render_job(manifest: Path, output: Path, store: ProfileStore | None = None) -> dict:
    from hwsol.jobs.layout import LayoutError

    output = output.absolute()
    if output.is_symlink():
        raise ValueError("Output must be a local directory")
    progress = {}
    with lock(output.parent / f".{output.name}.hwsol.lock"):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output must be new or empty")
        with tempfile.TemporaryDirectory(prefix=".hwsol-job-", dir=output.parent) as tmp:
            stage = Path(tmp) / "job"
            stage.mkdir()
            try:
                result = build(manifest, stage, store or ProfileStore(), progress)
                if output.exists():
                    output.rmdir()
                stage.rename(output)
                return result
            except Exception as exc:
                failure = {
                    "status": "failed",
                    **progress,
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
                if isinstance(exc, LayoutError):
                    failure["attempts"] = exc.attempts
                atomic_json(output / REPORT, failure)
                raise ValueError(f"{failure.get('problem', 'Job')}: {exc}") from exc
