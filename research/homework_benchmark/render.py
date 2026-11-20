"""New-content CPU homework comparison with explicit mathematical provenance.

English uses the font in both arms. No historical neural word images are read.
Passing recorder/pixel checks is not a semantic proof or an author style review.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfWriter

from hwsol.config import RenderConfig
from hwsol.identity import build_identity
from hwsol.jobs.checks import check_answer
from hwsol.jobs.layout import MultiplePages, ink_geometry
from hwsol.jobs.paper import paper_image, visible_ink
from hwsol.jobs.runtime import load_job, validate_source
from hwsol.jobs.schema import Background, Verification
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.store import copy_snapshot, load_snapshot
from hwsol.workspace import Run, verify_run, workspace_root
from research.corpus.context_composer import layout
from research.corpus.math_glyph_completion import SELECTION, local_comparison, raster
from research.corpus.math_glyph_resolver import MathGlyphResolver, ScanGlyph, matrix_strokes
from research.corpus.math_session_glyphs import SessionGlyphCalibration
from research.corpus.math_session_polish import SessionInkPolish

BASELINE = "20261120T064909731073Z-v01026-session-math-polish-b9371278"
LEGACY = "20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5"
PROTOCOL = Path("docs/M1_BENCHMARK_PROTOCOL.md")
REGISTRY = Path("workspace/registry/research_best.json")
FREEZE_SCHEMA = "homework-candidate-freeze/2"
CORPUS_ID = "homework-m1-v1"
REPOSITORY = Path(__file__).resolve().parents[2]
EVALUATION_LEDGER = REPOSITORY / "workspace/registry/evaluation_uses"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def source_paths() -> list[Path]:
    """Capture all local implementation inputs without private training assets."""
    roots = [Path("src/hwsol"), Path("research/corpus"), Path("research/homework_benchmark")]
    return sorted({p for root in roots for p in root.rglob("*.py")})


def renderer_environment() -> dict:
    """Bind upstream code, dependency versions/lock and resolved MathText fonts."""
    from matplotlib import _mathtext, font_manager

    families = {
        value
        for kind in ("StixFonts", "BakomaFonts", "DejaVuSansFonts", "DejaVuSerifFonts")
        for value in getattr(_mathtext, kind)._fontmap.values()
    }
    fonts = {}
    for family in sorted(families):
        path = Path(font_manager.findfont(family, fallback_to_default=False)).resolve()
        fonts[family] = {"path": str(path), "sha256": sha256(path)}
    return {"build": build_identity(), "python": sys.version, "mathtext_fonts": fonts}


def freeze_candidate(
    dest: Path,
    *,
    protocol: Path = PROTOCOL,
    registry: Path = REGISTRY,
    baseline: Path | None = None,
    legacy: Path | None = None,
) -> Path:
    """Freeze actual sources/assets/protocol before a one-use holdout evaluation."""
    baseline = baseline or workspace_root() / "runs" / BASELINE
    legacy = legacy or workspace_root() / "runs" / LEGACY
    for root in (baseline, legacy):
        verify_run(root)
    required = [protocol, registry, baseline / "run.json", legacy / "run.json"]
    files = {str(p.resolve()): sha256(p) for p in [*source_paths(), *required]}
    payload = {
        "schema": FREEZE_SCHEMA,
        "corpus_id": CORPUS_ID,
        "files": files,
        "baseline": str(baseline.resolve()),
        "legacy": str(legacy.resolve()),
        "protocol": str(protocol.resolve()),
        "registry": str(registry.resolve()),
        "arms": ["font-cpu", "personal-math-v01026"],
        "english": "font-cpu in both arms; zero historical word-image cache inputs",
        "holdout_uses": 1,
        "renderer_environment": renderer_environment(),
        "semantic_review": "required separately; recorder identity is not semantic correctness",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return dest


def validate_freeze(receipt: Path, baseline: Path, legacy: Path) -> dict:
    record = read(receipt)
    if (
        record.get("schema") != FREEZE_SCHEMA
        or record.get("holdout_uses") != 1
        or record.get("corpus_id") != CORPUS_ID
    ):
        raise ValueError("Unsupported or nonexclusive holdout freeze receipt")
    if record.get("baseline") != str(baseline.resolve()) or record.get("legacy") != str(
        legacy.resolve()
    ):
        raise ValueError("Holdout assets differ from the frozen candidate")
    required = {str(p.resolve()) for p in source_paths()}
    required.update(record.get(k, "") for k in ("protocol", "registry"))
    required.update(str(p.resolve() / "run.json") for p in (baseline, legacy))
    if required != set(record["files"]):
        raise ValueError("Frozen source inventory differs from the current implementation")
    for name, digest in record["files"].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f"Frozen evaluation input changed: {name}")
    if record.get("renderer_environment") != renderer_environment():
        raise ValueError("Frozen renderer environment, upstream code or MathText fonts changed")
    return record


def claim_holdout(
    receipt: Path,
    run_path: Path,
    *,
    cases: list[dict] | None = None,
) -> Path:
    """Consume the corpus, independently of candidate identity or successful completion.

    A new freeze cannot rebrand already evaluated content as a new holdout.
    Saved outputs may still be mechanically replayed as exposed regression evidence.
    """
    # Output directories and HWSOL_WORKSPACE must never create a fresh evaluation ledger.
    dest = EVALUATION_LEDGER / f"{CORPUS_ID}-holdout.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "corpus_id": CORPUS_ID,
                "consumed_at": datetime.now(UTC).isoformat(),
                "freeze": str(receipt.resolve()),
                "sha256": sha256(receipt),
                "run": str(run_path),
                "cases": [
                    {
                        "id": c["id"],
                        "statement_sha256": hashlib.sha256(c["statement"].encode()).hexdigest(),
                        "solution_sha256": hashlib.sha256(c["solution"].encode()).hexdigest(),
                    }
                    for c in (cases or [])
                ],
                "future_use": "exposed regression only; new candidates require new holdout content",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    return dest


def normalized_identities(page, renderer, records: list[dict]) -> list[tuple]:
    """Name parser matrix stroke roles without inventing new visual content."""
    brackets = matrix_strokes(page, renderer)
    return [
        (
            r["item_index"],
            brackets[r["item_index"]][0] if r["item_index"] in brackets else r["semantic"],
            r["context"],
            r.get("latex"),
        )
        for r in records
    ]


def pixel_boundary_checks(
    before: Image.Image,
    after: Image.Image,
    original: list[dict],
    candidate: list[dict],
    changed: set[int],
) -> dict:
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Candidate changed page raster dimensions")
    diff = np.any(a != b, axis=2)
    allowed = np.zeros(diff.shape, dtype=bool)
    english = np.zeros(diff.shape, dtype=bool)

    def region(box: list[float], padding: int) -> tuple[slice, slice]:
        x0, y0, x1, y1 = box
        return (
            slice(max(0, math.floor(y0) - padding), min(a.shape[0], math.ceil(y1) + padding)),
            slice(max(0, math.floor(x0) - padding), min(a.shape[1], math.ceil(x1) + padding)),
        )

    for old, new in zip(original, candidate, strict=True):
        if old["context"] == "text":
            if old != new:
                raise ValueError("Candidate changed an English provenance record")
            if len(old["bounds_px"]) == 4:
                english[region(old["bounds_px"], 1)] = True
        i = old["item_index"]
        if i not in changed:
            continue
        if old["context"] != "mathematics":
            raise ValueError("Candidate modified a nonmathematical drawing item")
        bounds = new["bounds_px"]
        safe = new.get("allowed_bounds_px", old["bounds_px"])
        x0, y0, x1, y1 = bounds
        if min(x0 - safe[0], y0 - safe[1], safe[2] - x1, safe[3] - y1) < -0.02:
            raise ValueError(f"Candidate item {i} exceeds its recorded safe slot")
        if min(x0, y0, a.shape[1] - x1, a.shape[0] - y1) < 3:
            raise ValueError(f"Candidate item {i} touches page boundary")
        # Cover old and final glyphs, including original stroked cap widths.
        for box in (old["bounds_px"], bounds, safe):
            allowed[region(box, 2)] = True
    external = int(np.count_nonzero(diff & ~allowed))
    in_english = int(np.count_nonzero(diff & english))
    if external:
        raise ValueError(f"Candidate changes {external} pixels outside recorded mathematical slots")
    if in_english:
        raise ValueError(f"Candidate changes {in_english} pixels in protected English boxes")
    return {
        "changed_pixels": int(diff.sum()),
        "changed_english_pixels": in_english,
        "pixels_changed_outside_math_slots": external,
        "text_items_checked": sum(r["context"] == "text" for r in original),
        "changed_math_items": len(changed),
    }


def fallback_summary(records: list[dict]) -> dict:
    math_rows = [r for r in records if r["context"] == "mathematics"]
    fonts = [r for r in math_rows if r["source_type"] in {"external-font", "inherited-font"}]
    return {
        "visible_math_items": len(math_rows),
        "source_types": dict(Counter(r["source_type"] for r in math_rows)),
        "font_fallbacks": [
            {
                "item_index": r["item_index"],
                "semantic": r["semantic"],
                "font": r.get("font"),
                "source_type": r["source_type"],
                "reason": r.get("fallback_reason", "no personal contour selected by font-cpu"),
            }
            for r in fonts
        ],
        "procedural_structures": [
            {
                "item_index": r["item_index"],
                "structure": r["structure"],
                "reason": r.get("fallback_reason", "parser structural geometry"),
            }
            for r in math_rows
            if r["procedural_geometry"]
        ],
        "other_explicit_fallbacks": [
            {
                "item_index": r["item_index"],
                "semantic": r["semantic"],
                "reason": r.get("fallback_reason") or r.get("session_fallback_reason"),
            }
            for r in math_rows
            if r.get("fallback_reason") or r.get("session_fallback_reason")
        ],
    }


def accepted_scans(folder: Path) -> list[ScanGlyph]:
    return [
        ScanGlyph.load(folder, row)
        for row in read(folder / "sources.json")
        if row["review_status"] == "agent-accepted"
    ]


def save_pdf(pdf: bytes, dest: Path) -> Image.Image:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf)
    image = raster(pdf)
    image.save(dest.with_suffix(".png"), dpi=(300, 300))
    return image


def require_visible_ink(
    pdf: bytes, image: Image.Image, bg: Background, paper: bytes | None = None
) -> None:
    """Reuse the runtime's blank-white and background-relative visibility gates."""
    if bg.kind == "plain":
        if min(lo for lo, _ in image.convert("RGB").getextrema()) >= 247:
            raise ValueError("Ink is not visible against the selected paper")
    elif paper is None or not visible_ink(pdf, paper):
        raise ValueError("Ink is not visible against the selected paper")


def compare_page(a: Image.Image, b: Image.Image, dest: Path) -> None:
    """A fixed, labelled overview; native resolution pages remain alongside it."""
    scale = 0.4
    size = (round(a.width * scale), round(a.height * scale))
    sheet = Image.new("RGB", (size[0] * 2, size[1] + 45), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), "A: font-cpu | B: personal math v0.10.26 | English: font-cpu", fill="black")
    sheet.paste(a.resize(size, Image.Resampling.LANCZOS), (0, 45))
    sheet.paste(b.resize(size, Image.Resampling.LANCZOS), (size[0], 45))
    sheet.save(dest)


def formula_crops(
    case_id: str,
    trace: dict,
    original: list[dict],
    records: list[dict],
    before: Image.Image,
    after: Image.Image,
    output: Path,
    page_number: int,
) -> tuple[list[dict], list[tuple]]:
    """Include every parsed formula and matrix, including unchanged and fallback ink."""
    output.mkdir(parents=True, exist_ok=True)
    inventory, panels = [], []
    groups = [("formula", q) for q in trace["formulas"]]
    groups.extend(("matrix", q) for q in trace["matrices"])
    groups.sort(key=lambda q: (q[1]["item_range"], q[0]))
    for index, (kind, group) in enumerate(groups, 1):
        start, stop = group["item_range"]
        boxes = [
            r["bounds_px"]
            for rows in (original, records)
            for r in rows[start:stop]
            if len(r["bounds_px"]) == 4
        ]
        if not boxes:
            raise ValueError(f"Parsed {kind} {index} has no visible ink bounds")
        array = np.array(boxes)
        lo, hi = array[:, :2].min(0), array[:, 2:].max(0)
        box = [
            max(0, math.floor(lo[0]) - 10),
            max(0, math.floor(lo[1]) - 10),
            min(before.width, math.ceil(hi[0]) + 10),
            min(before.height, math.ceil(hi[1]) + 10),
        ]
        a, b = before.crop(box), after.crop(box)
        column = max(a.width, b.width, 180)
        sheet = Image.new("RGB", (column * 2 + 16, max(a.height, b.height) + 48), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), f"Page {page_number} / {kind} {index}", fill="black")
        draw.text((4, 24), "A: font-cpu", fill="black")
        draw.text((column + 20, 24), "B: personal math", fill="black")
        sheet.paste(a, (0, 48))
        sheet.paste(b, (column + 16, 48))
        dest = output / f"{index:03d}-{kind}.png"
        sheet.save(dest, dpi=(300, 300))
        inventory.append(
            {
                "index": index,
                "kind": kind,
                "parser": group,
                "crop_box_px": box,
                "comparison": str(dest.name),
                "includes_unchanged_items": True,
            }
        )
        panels.append((page_number, index, a, b))
    atomic_json(output / "index.json", inventory)
    return inventory, panels


def render_case(
    case: dict,
    run: Path,
    cfg: RenderConfig,
    profile,
    resolver,
    sessions,
    polish,
    *,
    panels: list | None = None,
    page_number: int = 1,
) -> dict:
    case_id = case["id"]
    if not case_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in case_id
    ):
        raise ValueError("Unsafe benchmark case identifier")
    inputs = run / "solutions/inputs" / case_id
    source = case["solution"]
    (inputs / "source.md").write_text(source, encoding="utf-8")
    report = {
        "id": case_id,
        "domain": case["domain"],
        "split": case["split"],
        "expected": case["expected"],
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_kind": case["source"],
        "original_id": case.get("original_id", case_id),
        "english_backend": "font-cpu",
        "cached_neural_words_used": 0,
        "semantic_review": "pending independent solution and visual review",
        "author_style_review": "not_evaluated",
    }
    stage = "source_validation"
    try:
        if case.get("input_verification"):
            check_answer(Verification.model_validate(case["input_verification"]))
        report["numeric_checks"] = check_answer(
            Verification(
                status="inconclusive",
                evidence="Listed arithmetic only. Derivation requires separate review.",
                checks=case.get("numeric_checks", []),
            )
        )
        bg = Background.model_validate(case.get("background", {"kind": "plain"}))
        asset = inputs / bg.path if bg.path else None
        effective = cfg.model_copy(update={"paper": "plain" if asset else bg.kind})
        stage = "layout"
        title = case.get("display_number", case_id)
        if any(ord(c) not in profile.manifest.coverage for c in title if not c.isspace()):
            raise ValueError(f"Missing glyph in problem number {title}")
        page, renderer, original = layout(source, title, profile, effective)
        if asset:
            page.background = paper_image(bg, asset, (page.width, page.height))
            page.paper = None
        trace = copy.deepcopy({"formulas": renderer.formulas, "matrices": renderer.matrices})
        identities = normalized_identities(page, renderer, original)
        old_items = list(page.items)
        control_pdf = renderer.to_vector_pdf([page])
        before = save_pdf(control_pdf, run / f"rendered/pages/control/{case_id}.pdf")
        paper = renderer.to_vector_pdf([replace(page, items=[])]) if bg.kind != "plain" else None
        require_visible_ink(control_pdf, before, bg, paper)
        report["control"] = {"status": "rendered", **fallback_summary(original)}
        atomic_json(
            run / f"reports/{case_id}-provenance.json",
            {
                "control": original,
                "candidate": None,
                "parser_structure_trace": trace,
                "semantic_proof": "not provided by the recorder comparison",
            },
        )
        stage = "candidate_resolver"
        records = resolver.resolve(page, renderer, original, profile)
        records = sessions.apply(page, renderer, records, operator_page=False)
        records = polish.apply(page, renderer, records)
        candidate_pdf = renderer.to_vector_pdf([page])
        after = save_pdf(candidate_pdf, run / f"comparisons/raw-candidate/{case_id}.pdf")
        report["candidate"] = {"status": "rendered_pending_checks", **fallback_summary(records)}
        inventory, new_panels = formula_crops(
            case_id,
            trace,
            original,
            records,
            before,
            after,
            run / f"comparisons/formulas/{case_id}",
            page_number,
        )
        report["formula_and_matrix_crops"] = len(inventory)
        if panels is not None:
            panels.extend(new_panels)
        provenance = {"control": original, "candidate": records, "parser_structure_trace": trace}
        atomic_json(run / f"reports/{case_id}-provenance.json", provenance)
        stage = "candidate_integrity"
        actual = [(r["item_index"], r["semantic"], r["context"], r.get("latex")) for r in records]
        if identities != actual:
            raise ValueError("Mathematical identity, item order, or context changed")
        if trace != {"formulas": renderer.formulas, "matrices": renderer.matrices}:
            raise ValueError("Parser structural trace changed")
        changed = {
            i for i, (a, b) in enumerate(zip(old_items, page.items, strict=True)) if a is not b
        }
        report["pixel_checks"] = pixel_boundary_checks(before, after, original, records, changed)
        bounds, _ = ink_geometry(page, renderer)
        if min(*bounds[:2], page.width - bounds[2], page.height - bounds[3]) < 3:
            raise ValueError("Candidate ink crosses page boundary")
        report["candidate_ink_bounds_px"] = bounds
        report["item_identity_and_order_preserved"] = True
        report["parser_trace_preserved"] = True
        report["one_top_level_problem_one_page"] = True
        report["background"] = bg.model_dump()
        require_visible_ink(candidate_pdf, after, bg, paper)
        dest = run / f"rendered/pages/candidate/{case_id}.pdf"
        save_pdf(candidate_pdf, dest)
        compare_page(before, after, run / f"comparisons/{case_id}.png")
        report.update(status="engineering_pass_visual_review_pending")
        report["candidate"]["status"] = "engineering_pass_visual_review_pending"
        report["expected_outcome_matches"] = case["expected"]["outcome"] == "render"
    except Exception as exc:
        # Preserve every failure and continue independent inputs. Never repair the source here.
        rejection = isinstance(exc, (ValueError, MultiplePages))
        report.update(
            status="rejected" if rejection else "failed",
            failure_stage=stage,
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        (run / f"logs/{case_id}-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        expected = case["expected"]
        report["expected_outcome_matches"] = (
            rejection
            and expected["outcome"] == "reject"
            and (not expected.get("error_contains") or expected["error_contains"] in str(exc))
        )
    atomic_json(run / f"reports/{case_id}.json", report)
    return report


def manifest_cases(manifest: Path, profile) -> tuple[list[dict], dict, dict]:
    """Load new solutions and original inputs through the existing editable Job contract."""
    job = load_job(manifest)
    if job.backend != "font-cpu" or job.profile not in (None, "personal@v1"):
        raise ValueError("This frozen math comparison requires font-cpu and personal@v1")
    if job.realism != "baseline":
        raise ValueError("Frozen context_composer supports only baseline realism")
    if job.profile_sha256 is not None and job.profile_sha256 != profile.manifest_sha256:
        raise ValueError("Requested Profile SHA-256 differs from the frozen snapshot")
    root = manifest.resolve().parent
    protected = {str(manifest.resolve()): sha256(manifest)}
    result = []
    for index, problem in enumerate(job.problems, 1):
        if not problem.complete:
            raise ValueError(f"Incomplete original problem: {problem.id or problem.number}")
        solution = inside(root, problem.solution)
        protected[str(solution)] = sha256(solution)
        sources = []
        for source in problem.sources:
            path = inside(root, source.path)
            validate_source(path, source)
            digest = sha256(path)
            if source.sha256 is not None and digest != source.sha256:
                raise ValueError(f"Original problem source changed: {source.path}")
            protected[str(path)] = digest
            sources.append({**source.model_dump(), "original_path": str(path), "sha256": digest})
        bg = (
            problem.background or job.background or Background(kind=job.render.paper)
        ).model_dump()
        asset = None
        if bg.get("path"):
            asset = inside(root, bg["path"])
            protected[str(asset)] = sha256(asset)
            bg["path"] = "background" + asset.suffix.lower()
        result.append(
            {
                "id": f"job-{index:03d}",
                "display_number": problem.number,
                "original_id": problem.id or problem.number,
                "split": "external",
                "domain": "external-job",
                "title": problem.number,
                "statement": problem.statement,
                "solution": solution.read_text(encoding="utf-8"),
                "source": {
                    "kind": "external-job",
                    "generated_by": job.generated_by,
                    "original_sources": sources,
                },
                "expected": {"outcome": "render", "page_count": 1},
                "background": bg,
                "background_original_path": str(asset) if asset else None,
                "numeric_checks": [c.model_dump() for c in problem.verification.checks],
                "input_verification": problem.verification.model_dump(),
            }
        )
    return result, protected, job.render.model_dump(mode="json", exclude_unset=True)


def write_manifest_inputs(manifest: Path, selected: list[dict], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest, dest / "original-job.json")
    for case in selected:
        folder = dest / case["id"]
        folder.mkdir()
        (folder / "statement.md").write_text(case["statement"], encoding="utf-8")
        (folder / "solution.md").write_text(case["solution"], encoding="utf-8")
        if case["background_original_path"]:
            shutil.copyfile(case["background_original_path"], folder / case["background"]["path"])
        sources = []
        for i, source in enumerate(case["source"]["original_sources"], 1):
            original = Path(source["original_path"])
            saved = folder / f"source-{i:02d}{original.suffix.lower()}"
            shutil.copyfile(original, saved)
            sources.append({**source, "snapshot": str(saved.relative_to(dest))})
        atomic_json(folder / "source-manifest.json", sources)


def bind_execution_jobs(manifests: list[Path], cfg: RenderConfig, profile) -> list[dict]:
    """Keep authored templates, then export the exact settings used by this run."""
    receipts = []
    for manifest in manifests:
        template = manifest.with_name("job-template.not-executed.json")
        if template.exists():
            raise FileExistsError("An input template cannot be replaced or rebound")
        job = load_job(manifest)
        shutil.copyfile(manifest, template)
        actual = job.model_copy(
            update={
                "realism": "baseline",
                "profile": profile.manifest.ref,
                "profile_sha256": profile.manifest_sha256,
                "render": cfg.model_copy(deep=True),
            }
        )
        atomic_json(manifest, actual.model_dump(mode="json"))
        _, _, loaded_cfg = manifest_cases(manifest, profile)
        if loaded_cfg != cfg.model_dump(mode="json"):
            raise ValueError("Exported Job does not preserve the executed render settings")
        receipts.append(
            {
                "case": manifest.parent.name,
                "execution_job": str(manifest),
                "execution_sha256": sha256(manifest),
                "template": str(template),
                "template_sha256": sha256(template),
                "template_status": "authoring template only; not the executed configuration",
                "realism": "baseline",
                "profile_sha256": profile.manifest_sha256,
                "render_config": cfg.model_dump(mode="json"),
            }
        )
    return receipts


def launch(
    split: str = "development",
    *,
    freeze_receipt: Path | None = None,
    baseline: Path | None = None,
    legacy: Path | None = None,
    workspace: Path | None = None,
    manifest: Path | None = None,
) -> Path:
    if manifest is not None and split != "development":
        raise ValueError("External manifests cannot be evaluated as the benchmark holdout")
    if split not in {"development", "holdout"}:
        raise ValueError("Evaluate development or holdout separately")
    root = workspace_root(workspace)
    baseline = baseline or workspace_root() / "runs" / BASELINE
    legacy = legacy or workspace_root() / "runs" / LEGACY
    frozen = None
    if split == "holdout":
        if freeze_receipt is None:
            raise ValueError("Holdout requires an explicit immutable --freeze-receipt")
        frozen = validate_freeze(freeze_receipt, baseline, legacy)
    for archive in (baseline, legacy):
        verify_run(archive)
    cfg = RenderConfig.model_validate(read(baseline / "run.json")["config"]["render_config"])
    input_protected = {}
    if manifest is None:
        from research.homework_benchmark.cases import cases, verify_references, write_inputs

        selected = cases(split)
    else:
        selected, input_protected, overrides = manifest_cases(
            manifest, load_snapshot(baseline / "solutions/profile-v1")
        )
        cfg = RenderConfig.model_validate({**cfg.model_dump(), **overrides})
        if cfg.dpi != 300 or cfg.page_width_mm != 210 or cfg.page_height_mm != 297:
            raise ValueError("Frozen math benchmark currently requires A4 and 300 DPI")
        if cfg.glyph_metrics != "raw" or cfg.policy is not None:
            raise ValueError("Frozen comparison supports raw metrics without a rendering policy")
    protected = {
        str(p.resolve()): sha256(p)
        for p in (baseline / "run.json", legacy / "run.json", workspace_root() / "latest.json")
    }
    protected.update(read(baseline / "reports/protected-assets.json"))
    protected.update(input_protected)
    run = Run.create(
        "homework-m1-render",
        seed=cfg.seed,
        workspace=root,
        config={
            "split": "external" if manifest else split,
            "input_manifest": str(manifest.resolve()) if manifest else None,
            "baseline": str(baseline),
            "legacy": str(legacy),
            "freeze_sha256": sha256(freeze_receipt) if freeze_receipt else None,
            "render_config": cfg.model_dump(mode="json"),
            "english_backend": "font-cpu",
            "neural_updates": 0,
            "cases_sha256": hashlib.sha256(
                json.dumps(selected, sort_keys=True).encode()
            ).hexdigest(),
        },
    )
    print(run.path, flush=True)
    try:
        if split == "holdout":
            marker = claim_holdout(freeze_receipt, run.path, cases=selected)
            atomic_json(run.path / "reports/holdout-use.json", read(marker))
            atomic_json(run.path / "reports/frozen-candidate.json", frozen)
        for p in source_paths():
            target = run.path / "logs/source" / p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        protocol = Path(frozen["protocol"]) if frozen else PROTOCOL
        if protocol.is_file():
            shutil.copyfile(protocol, run.path / "reports/protocol.md")
        if frozen:
            shutil.copyfile(frozen["registry"], run.path / "reports/frozen-registry.json")
        execution_manifests = []
        if manifest is None:
            execution_manifests = write_inputs(run.path / "solutions/inputs", split=split)
            references = verify_references()
        else:
            write_manifest_inputs(manifest, selected, run.path / "solutions/inputs")
            references = {
                "status": "external verification recorded, not automatically trusted",
                "cases": [{"id": c["id"], "evidence": c["input_verification"]} for c in selected],
            }
        atomic_json(run.path / "solutions/cases.json", selected)
        atomic_json(run.path / "reports/reference-calculations.json", references)
        for version in (1, 3):
            copy_snapshot(
                load_snapshot(baseline / f"solutions/profile-v{version}"),
                run.path / f"solutions/profile-v{version}",
            )
        for name, old in (("historical-glyphs", legacy), ("session-glyphs", baseline)):
            shutil.copytree(old / f"solutions/{name}", run.path / f"solutions/{name}")
        profile = load_snapshot(run.path / "solutions/profile-v1")
        if execution_manifests:
            atomic_json(
                run.path / "reports/execution-jobs.json",
                bind_execution_jobs(execution_manifests, cfg, profile),
            )
        strokes = load_snapshot(run.path / "solutions/profile-v3")
        scanned = accepted_scans(run.path / "solutions/historical-glyphs")
        resolver = MathGlyphResolver(
            {g.source["character"]: g for g in scanned}, strokes, selected=SELECTION
        )
        sessions = SessionGlyphCalibration(accepted_scans(run.path / "solutions/session-glyphs"))
        polish = SessionInkPolish(sessions, profile.font)
        for folder in ("control", "candidate"):
            (run.path / f"rendered/pages/{folder}").mkdir(parents=True)
        (run.path / "comparisons/raw-candidate").mkdir()
        panels = []
        reports = [
            render_case(
                c,
                run.path,
                cfg,
                profile,
                resolver,
                sessions,
                polish,
                panels=panels,
                page_number=index,
            )
            for index, c in enumerate(selected, 1)
        ]
        if panels:
            local_comparison(
                run.path / "comparisons", panels, labels=("A font-cpu", "B personal math v0.10.26")
            )
        combined = {}
        control_valid = {
            r["id"] for r in reports if r.get("control", {}).get("status") == "rendered"
        }
        for arm in ("control", "candidate"):
            paths = [
                run.path / f"rendered/pages/{arm}/{c['id']}.pdf"
                for c in selected
                if arm != "control" or c["id"] in control_valid
            ]
            paths = [p for p in paths if p.exists()]
            combined[arm] = len(paths)
            if paths:
                with PdfWriter() as writer:
                    for p in paths:
                        writer.append(p)
                    dest = run.path / (
                        "rendered/homework_solutions.pdf"
                        if arm == "candidate"
                        else "comparisons/font-cpu-homework.pdf"
                    )
                    writer.write(dest)
        with PdfWriter() as writer:
            for c in selected:
                for arm in ("control", "candidate"):
                    if arm == "control" and c["id"] not in control_valid:
                        continue
                    p = run.path / f"rendered/pages/{arm}/{c['id']}.pdf"
                    if p.exists():
                        writer.append(p)
            if writer.pages:
                writer.write(run.path / "comparisons/A-B-homework.pdf")
        if any(sha256(Path(p)) != digest for p, digest in protected.items()):
            raise ValueError("Protected baseline archive or production latest changed")
        atomic_json(run.path / "reports/protected-assets.json", protected)
        atomic_json(run.path / "reports/cases.json", reports)
        result = {
            "cases": len(reports),
            "split": "external" if manifest else split,
            "pages": combined,
            "outcomes": dict(Counter(r["status"] for r in reports)),
            "expected_outcomes_matching": sum(r["expected_outcome_matches"] for r in reports),
            "font_fallbacks": {
                arm: sum(len(r.get(arm, {}).get("font_fallbacks", [])) for r in reports)
                for arm in ("control", "candidate")
            },
            "all_sources_frozen": True,
            "production_latest_preserved": True,
            "torch_loaded": "torch" in sys.modules,
            "neural_updates": 0,
            "cached_neural_word_images_used": 0,
            "all_parsed_formula_and_matrix_crops": len(panels),
            "content_and_author_style": "pending independent review; engineering checks only",
            "release_decision": "held",
        }
        if result["torch_loaded"]:
            raise ValueError("CPU renderer imported Torch")
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--split", choices=("development", "holdout"), default="development")
    cli.add_argument(
        "--freeze", type=Path, help="Write a new frozen candidate receipt without rendering"
    )
    cli.add_argument("--freeze-receipt", type=Path)
    cli.add_argument("--protocol", type=Path, default=PROTOCOL)
    cli.add_argument("--registry", type=Path, default=REGISTRY)
    cli.add_argument("--baseline", type=Path)
    cli.add_argument("--legacy", type=Path)
    cli.add_argument("--workspace", type=Path)
    cli.add_argument("--manifest", type=Path, help="Render a new existing Job manifest")
    args = cli.parse_args()
    if args.freeze:
        print(
            freeze_candidate(
                args.freeze,
                protocol=args.protocol,
                registry=args.registry,
                baseline=args.baseline,
                legacy=args.legacy,
            )
        )
    else:
        launch(
            args.split,
            freeze_receipt=args.freeze_receipt,
            baseline=args.baseline,
            legacy=args.legacy,
            workspace=args.workspace,
            manifest=args.manifest,
        )


if __name__ == "__main__":
    main()
