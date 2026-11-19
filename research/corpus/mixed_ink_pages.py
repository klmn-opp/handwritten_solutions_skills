"""Research word composition inside unchanged Handnote mathematical layouts."""

import argparse
import copy
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from research.corpus.context_content_train import digest as sha256
from research.corpus.context_content_train import write as atomic_json


def label_pdf(blob: bytes, method: str, math_label: str = "unchanged Handnote") -> bytes:
    """Print provenance in page content; annotations disappear in some viewers."""
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(blob))
    if len(reader.pages) != 1:
        raise ValueError("One problem per page required")
    page = reader.pages[0]
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    figure = Figure(figsize=(width / 72, height / 72))
    figure.patch.set_alpha(0)
    figure.text(
        28 / width,
        14 / height,
        f"{method} | mathematics: {math_label} | experimental word composition",
        fontsize=7,
    )
    overlay = io.BytesIO()
    FigureCanvasPdf(figure).print_pdf(overlay)
    writer = PdfWriter()
    writer.add_page(page)
    writer.remove_annotations(subtypes="/FreeText")
    writer.pages[0].merge_page(PdfReader(overlay).pages[0])
    output = io.BytesIO()
    writer.write(output)
    writer.close()
    return output.getvalue()


def collision_reason(box, occupied, ignored, width, height):
    if min(box[:2]) < 3 or box[2] > width - 3 or box[3] > height - 3:
        return "page_boundary"
    overlap = (np.minimum(occupied[:, 2:], box[2:]) - np.maximum(occupied[:, :2], box[:2]) > 0).all(
        1
    )
    overlap[ignored] = False
    return "would_overlap_neighbour" if overlap.any() else None


def text_groups(source, identity, records, profile):
    from hwsol.jobs.layout import validate_markdown

    segments = [identity]
    for block in validate_markdown(source, profile):
        segments.extend(text for kind, text in block.segments if kind == "t")
    visible = [r for r in records if r["context"] == "text" and r.get("semantic")]
    expected = "".join("".join(s.split()) for s in segments)
    if "".join(r["semantic"] for r in visible) != expected:
        raise ValueError("Full text provenance differs; cannot silently drop characters")
    groups, pos = [], 0
    for si, segment in enumerate(segments):
        for token in segment.split():
            part = visible[pos : pos + len(token)]
            pos += len(token)
            groups.append(
                {"text": token, "heading": si == 0, "items": [r["item_index"] for r in part]}
            )
    return groups


def prepare() -> Path:
    import pypdfium2 as pdfium
    from handnote.vector import shift_items

    from hwsol.config import RenderConfig
    from hwsol.jobs.paper import pdf_image
    from hwsol.workspace import profile_store
    from research.corpus.context_composer import layout
    from research.corpus.experiment import create
    from research.corpus.hard_homework import cases
    from research.corpus.word_style_data import normalize

    profile = profile_store().resolve("personal@v1")
    cfg = RenderConfig(seed=1909, font_size_pt=11, ink="#000000", dpi=300, semantic_layout=True)
    run = create(
        "hard-mixed-page-input",
        config={
            "profile": profile.provenance(),
            "render_config": cfg.model_dump(mode="json"),
            "C_used": False,
            "training_allowed": False,
            "mathematics_backend": "unchanged Handnote/MathText",
        },
    )
    rows, arrays, page_records = [], [], []
    try:
        for ci, (identity, source) in enumerate(cases().items()):
            page, renderer, provenance = layout(source, identity, profile, cfg)
            groups = text_groups(source, identity, provenance, profile)
            (run.path / f"solutions/problem-{ci}.md").write_text(source)
            blob = renderer.to_vector_pdf([page])
            (run.path / f"comparisons/handnote-{ci}.pdf").write_bytes(blob)
            with pdfium.PdfDocument(blob) as doc:
                pdf_image(doc, 0, 300 / 72).save(run.path / f"rendered/pages/handnote-{ci}.png")
            for g in groups:
                if g["heading"] or len(g["text"]) < 2 or len(g["text"]) > 32:
                    continue
                boxes = np.array([provenance[i]["bounds_px"] for i in g["items"]])
                box = [
                    math.floor(boxes[:, 0].min()) - 8,
                    math.floor(boxes[:, 1].min()) - 8,
                    math.ceil(boxes[:, 2].max()) + 8,
                    math.ceil(boxes[:, 3].max()) + 8,
                ]
                x, y, right, bottom = box
                if bottom - y > 160:
                    raise ValueError("Word spans multiple lines")
                sub = copy.copy(page)
                sub.width, sub.height = right - x, bottom - y
                sub.items = shift_items([page.items[i] for i in g["items"]], -x, -y)
                sub.rules, sub.background = [], None
                subblob = renderer.to_vector_pdf([sub])
                wid = len(rows)
                with pdfium.PdfDocument(subblob) as doc:
                    low = pdf_image(doc, 0, 300 / 72).crop((0, 0, sub.width, sub.height))
                    high = pdf_image(doc, 0, 1200 / 72).crop((0, 0, sub.width * 4, sub.height * 4))
                reference, tr = normalize(low)
                p = run.path / f"comparisons/word-{wid:04d}-4x.png"
                high.save(p)
                arrays.append(reference)
                rows.append(
                    {
                        **g,
                        "id": wid,
                        "page": ci,
                        "box": box,
                        "size": [sub.width, sub.height],
                        "transform": tr,
                        "image": str(p),
                        "sha256": sha256(p),
                        "source_type": "personal-font-reference",
                        "training_allowed": False,
                    }
                )
            page_records.append(
                {
                    "page": ci,
                    "identity": identity,
                    "source": source,
                    "provenance": provenance,
                    "size": [page.width, page.height],
                    "text_groups": groups,
                    "baseline_sha256": sha256(run.path / f"comparisons/handnote-{ci}.pdf"),
                }
            )
        np.savez_compressed(run.path / "reports/references.npz", reference=np.stack(arrays))
        atomic_json(run.path / "reports/words.json", rows)
        atomic_json(run.path / "reports/pages.json", page_records)
        run.finish(
            "prepared",
            result={"pages": 4, "words": len(rows), "C_used": False, "training_allowed": False},
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def infer(request: Path) -> None:
    import torch

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.aligned_style_model import AlignedStyle

    r = json.loads(request.read_text())
    setup(1909)
    source = Path(r["source"])
    run = Path(r["run"])
    if sha256(source / "run.json") != r["source_sha256"]:
        raise ValueError("Changed page input")
    rows = json.loads((source / "reports/words.json").read_text())
    refs = torch.tensor(
        np.load(source / "reports/references.npz")["reference"][:, None], device="cuda"
    )
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    output = {}
    for name, cp in r["checkpoints"].items():
        if sha256(Path(cp["path"])) != cp["sha256"]:
            raise ValueError("Changed model")
        model = AlignedStyle(stochastic=True, monotone=True).cuda().eval()
        model.load_state_dict(
            torch.load(cp["path"], map_location="cuda", weights_only=False)["model"]
        )
        generator = torch.Generator(device="cuda").manual_seed(r["seed"])
        chunks = []
        with torch.inference_mode():
            for k in range(0, len(rows), 8):
                eps = torch.randn((len(rows[k : k + 8]), 4), device="cuda", generator=generator)
                _, controls = model.generate(refs[k : k + 8], tokens[k : k + 8], eps)
                chunks.append(controls.cpu().numpy())
        output[name] = np.concatenate(chunks)
    np.savez_compressed(run / "reports/controls.npz", **output)
    atomic_json(
        run / "reports/inference.json",
        {
            "words": len(rows),
            "models": list(output),
            "training_updates": 0,
            "real_target_read": False,
            "seed": r["seed"],
        },
    )


def compose(source: Path, models: dict[str, Path]) -> Path:
    import subprocess

    import pypdfium2 as pdfium
    from pypdf import PdfWriter

    from hwsol.config import RenderConfig
    from hwsol.jobs.paper import pdf_image
    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_composer import layout
    from research.corpus.experiment import create
    from research.corpus.ink_integrity import ink_from_reference, map_components
    from research.corpus.ink_integrity_lines import render_line

    verify_run(source)
    profile = profile_store().resolve("personal@v1")
    source_cfg = json.loads((source / "run.json").read_text())["config"]
    cfg = RenderConfig(**source_cfg["render_config"])
    run = create(
        "hard-mixed-page-comparison",
        config={
            "source": str(source),
            "source_sha256": sha256(source / "run.json"),
            "C_used": False,
            "new_ink_model_failed_and_excluded": True,
            "word_composition_not_line_generation": True,
        },
    )
    req = {
        "run": str(run.path),
        "source": str(source),
        "source_sha256": sha256(source / "run.json"),
        "checkpoints": {k: {"path": str(p), "sha256": sha256(p)} for k, p in models.items()},
        "seed": 1910,
    }
    atomic_json(run.path / "reports/request.json", req)
    try:
        with (run.path / "logs/inference.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.mixed_ink_pages",
                    "infer",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        controls = np.load(run.path / "reports/controls.npz")
        pages = json.loads((source / "reports/pages.json").read_text())
        words = json.loads((source / "reports/words.json").read_text())
        all_pdf = PdfWriter()
        methods = {k: PdfWriter() for k in ["handnote", *models]}
        reports = []
        for info in pages:
            ci = info["page"]
            base, renderer, provenance = layout(info["source"], info["identity"], profile, cfg)
            assert provenance == info["provenance"]
            all_boxes = np.array([q["bounds_px"] for q in provenance])
            for name in methods:
                page = copy.copy(base)
                page.items = list(base.items)
                occupied = all_boxes.copy()
                replacements = {}
                removed = set()
                records = []
                for q in [q for q in words if q["page"] == ci and name != "handnote"]:
                    if sha256(Path(q["image"])) != q["sha256"]:
                        raise ValueError("Changed word reference")
                    parts = map_components(controls[name][q["id"]], guarded=True)
                    ww = {"transform": q["transform"], "left": 0, "right": q["size"][0]}
                    ink = render_line(
                        ink_from_reference(Image.open(q["image"])),
                        [ww],
                        [parts],
                        (q["size"][0] + 128, q["size"][1] + 128),
                    )
                    x, y = q["box"][:2]
                    x -= 64
                    y -= 64
                    yy, xx = np.where(ink > 0.05)
                    reason = None
                    if not len(xx):
                        reason = "blank"
                    else:
                        bb = np.array(
                            [xx.min() + x, yy.min() + y, xx.max() + x + 1, yy.max() + y + 1]
                        )
                        reason = collision_reason(bb, occupied, q["items"], base.width, base.height)
                    raw = run.path / f"comparisons/{name}-word-{q['id']:04d}.png"
                    Image.fromarray(np.uint8((1 - ink) * 255)).save(raw)
                    record = {
                        "word_id": q["id"],
                        "text": q["text"],
                        "page": ci,
                        "method": name,
                        "image": str(raw),
                        "sha256": sha256(raw),
                        "fallback_reason": reason,
                        "checkpoint_sha256": req["checkpoints"][name]["sha256"],
                        "source_type": "learned-geometry-of-personal-font",
                        "personal_collected": False,
                        "original_glyph_provenance": [provenance[i] for i in q["items"]],
                        "free_ink_generation": False,
                        "seed": 1910,
                        "generated_bounds_px": bb.tolist() if len(xx) else None,
                    }
                    records.append(record)
                    if reason:
                        continue
                    # Check later candidates against accepted placements as well as
                    # untouched originals; two shifts must not meet inside an old gap.
                    occupied[q["items"]] = bb
                    rgba = np.zeros((*ink.shape, 4), np.uint8)
                    rgba[:, :, 3] = np.uint8(ink * 255)
                    replacements[q["items"][0]] = (
                        "image",
                        Image.fromarray(rgba),
                        (1, 0, 0, 1, x, y),
                    )
                    removed.update(q["items"])
                page.items = [
                    replacements[i] if i in replacements else item
                    for i, item in enumerate(base.items)
                    if i not in removed or i in replacements
                ]
                math_ids = [i for i, q in enumerate(provenance) if q["context"] == "mathematics"]
                assert not removed.intersection(math_ids)
                assert all(i not in replacements for i in math_ids)
                blob = renderer.to_vector_pdf([page])
                dest = run.path / f"comparisons/{name}-page-{ci}.pdf"
                dest.write_bytes(label_pdf(blob, name))
                with pdfium.PdfDocument(dest) as doc:
                    assert len(doc) == 1
                    pdf_image(doc, 0, 200 / 72).save(
                        run.path / f"rendered/pages/{name}-page-{ci}.png"
                    )
                methods[name].append(dest)
                all_pdf.append(dest)
                reports.append(
                    {
                        "page": ci,
                        "method": name,
                        "words": records,
                        "math_items_unchanged": len(math_ids),
                        "semantic_trace": provenance,
                        "model_words": sum(q["fallback_reason"] is None for q in records),
                        "fallback_words": sum(q["fallback_reason"] is not None for q in records),
                        "page_count": 1,
                        "content_visually_verified": False,
                    }
                )
        for name, writer in methods.items():
            writer.write(run.path / f"comparisons/{name}-homework.pdf")
            writer.close()
        all_pdf.write(run.path / "rendered/homework_solutions.pdf")
        all_pdf.close()
        atomic_json(run.path / "reports/page-provenance.json", reports)
        run.finish(
            "held",
            result={
                "pages": len(reports),
                "problems": 4,
                "methods": list(methods),
                "mathematical_items_unchanged": True,
                "C_used": False,
                "automatic_publication": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["infer"])
    p.add_argument("request", type=Path)
    infer(p.parse_args().request)
