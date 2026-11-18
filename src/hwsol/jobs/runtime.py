"""Render into a sibling staging directory and publish only a fully verified assignment."""

import io
import json
import shutil
import tempfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, NameObject, TextStringObject

from hwsol.config import RenderConfig
from hwsol.identity import build_identity
from hwsol.identity import digest as identity_digest
from hwsol.jobs.checks import check_answer
from hwsol.jobs.paper import PDF_LOCK, pdf_image
from hwsol.jobs.schema import Background, Job, Realism, Source
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


def export_raster(doc, index: int, config: RenderConfig) -> Image.Image:
    image = pdf_image(doc, index, config.dpi / 72)
    size = (config.mm(config.page_width_mm), config.mm(config.page_height_mm))
    if image.size != size:
        if any(abs(a - b) > 1 for a, b in zip(image.size, size, strict=True)):
            raise ValueError("Unexpected raster dimensions")
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def verify_job(root: Path, *, pixels: bool = False) -> dict:
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
        ink = record.get("handwriting", {}).get("glyph_ink")
        if page.get("/HwsolGlyphInk") != ink:
            raise ValueError("PDF and report disagree on glyph ink")
        if ink is not None:
            if ink != "fill-only":
                raise ValueError("Unknown glyph ink policy")
            # Handnote stores glyphs and filled math rules as Forms. Paper and open
            # strokes such as matrix brackets are drawn separately on the page.
            for ref in page["/Resources"].get("/XObject", {}).values():
                form = ref.get_object()
                if form["/Subtype"] == "/Form":
                    ops = [op for _, op in ContentStream(form, reader).operations]
                    if (
                        not ops
                        or ops[-1] not in {b"f", b"f*"}
                        or any(
                            op not in {b"m", b"l", b"c", b"v", b"y", b"h", b"re"} for op in ops[:-1]
                        )
                    ):
                        raise ValueError(f"Expected a single un-stroked fill on page {i}")
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
    if pixels:
        with PDF_LOCK, closing(pdfium.PdfDocument(root / PDF)) as doc:
            for i in range(n):
                with Image.open(root / f"pages/{i + 1:02d}.png") as png:
                    actual = export_raster(doc, i, job.render)
                    if png.mode != "RGB" or ImageChops.difference(actual, png).getbbox():
                        raise ValueError(f"PNG pixels differ from PDF page {i + 1}")
    return {"status": "passed", "page_count": n, "problem_order": [p.number for p in job.problems]}


def build(
    manifest: Path, stage: Path, store: ProfileStore, progress: dict, realism: Realism | None = None
):
    from hwsol.jobs.layout import render_one

    job = load_job(manifest)
    if job.backend != "font-cpu":
        raise ValueError("diffink-experimental is not admitted to production; use the isolated lab")
    if realism is not None:
        job = Job.model_validate({**job.model_dump(exclude_unset=True), "realism": realism})
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
    input_digests = {}
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
            if path not in input_digests:
                input_digests[path] = sha256(path)
            digest = input_digests[path]
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
            target.read_text("utf-8"),
            problem.number,
            selected,
            job.render,
            bg,
            asset,
            job.realism,
            problem.id or problem.number,
        )
        page = writer.add_page(PdfReader(io.BytesIO(blob), strict=True).pages[0])
        page[NameObject("/HwsolID")] = TextStringObject(problem.id or problem.number)
        page[NameObject("/HwsolNumber")] = TextStringObject(problem.number)
        page[NameObject("/HwsolSolutionSHA256")] = TextStringObject(sha256(target))
        page[NameObject("/HwsolGlyphInk")] = TextStringObject(
            diagnostics["handwriting"]["glyph_ink"]
        )
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
            problem = job.problems[index]
            progress["problem"] = problem.number
            image = export_raster(doc, index, job.render)
            if (
                problem.background.kind == "plain"
                and min(lo for lo, _ in image.getextrema()) >= 247
            ):
                raise ValueError("Ink is not visible against the selected paper")
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
            "build": build_identity(),
            "render_identity": {
                "seed": job.render.seed,
                "mode": job.realism,
                "config_sha256": identity_digest(job.render.model_dump(mode="json")),
                "policy_sha256": identity_digest(job.render.policy.model_dump(mode="json"))
                if job.render.policy
                else None,
                "policy_compatibility": "legacy-profile-only-unvalidated-migration"
                if job.render.policy
                else "disabled",
                "font_sha256": sha256(selected.font),
                "profile_assets": [a.model_dump(mode="json") for a in selected.manifest.assets],
            },
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


def render_job(
    manifest: Path,
    output: Path,
    store: ProfileStore | None = None,
    *,
    realism: Realism | None = None,
) -> dict:
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
                result = build(manifest, stage, store or ProfileStore(), progress, realism)
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
