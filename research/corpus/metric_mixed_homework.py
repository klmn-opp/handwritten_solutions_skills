"""A4 homework experiment with sampled neural English and Handnote formulas."""

import copy
import shutil
import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write
from research.corpus.context_metric_evaluate import read


def launch(source: Path, model: Path, metric: Path, class_metrics_file: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for path in [source, model, metric]:
        verify_run(path)
    cfg = {
        "source": str(source),
        "source_sha256": digest(source / "run.json"),
        "model": str(model),
        "model_sha256": digest(model / "run.json"),
        "metric": str(metric),
        "metric_sha256": digest(metric / "run.json"),
        "class_metrics_sha256": digest(class_metrics_file),
        "class_metrics": read(class_metrics_file),
        "seed": 2012,
        "C_used": False,
        "new_neural_updates": 0,
        "style_sampling": "matched local PCA neighborhood",
    }
    run = create("metric-neural-english-math-homework", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-mixed-homework").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        prepare(run.path, source, metric, cfg["class_metrics"])
        for mode in ["generate", "replay"]:
            with (run.path / f"logs/{mode}.log").open("w") as log:
                subprocess.run(
                    [
                        "workspace/environments/english-htr-v08/bin/python",
                        "-m",
                        __name__,
                        str(run.path / "reports/request.json"),
                        mode,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=220,
                )
        result = assemble(run.path, source)
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def prepare(run: Path, source: Path, metric: Path, class_metrics: dict) -> None:
    from research.corpus.context_metric_layout import adjust_boxes
    from research.corpus.semantic_ink_data import conditions
    from research.corpus.semantic_raw_data import apply_metrics

    summary = read(metric / "reports/summary.json")
    anchored = summary["measurements"]["anchored"]
    assert (
        anchored["empty_glyphs"]
        == anchored["edge_cases"]
        == anchored["added_adjacent_overlap_pixels"]
        == 0
    )
    thresholds = read(metric / "reports/request.json")["thresholds"]
    words = []
    for page in read(source / "reports/pages.json"):
        index = page["page"]
        shutil.copyfile(
            source / f"solutions/problem-{index}.md", run / f"solutions/problem-{index}.md"
        )
        for group in page["text_groups"]:
            word_id = len(words)
            record = {**group, "page": index, "word_id": word_id}
            if group["heading"]:
                words.append({**record, "fallback_reason": "retain_heading"})
                continue
            glyphs = [page["provenance"][i] for i in group["items"]]
            if not group["text"].isascii():
                words.append({**record, "fallback_reason": "non_ascii"})
                continue
            tokens, boxes = conditions(
                group["text"],
                {"glyph_provenance": glyphs, "crop_from_pdf": [0, 0]},
                {"scale": 1, "translation": [0, 0], "bbox": [0, 0]},
            )
            if boxes[:, 3].max() - boxes[:, 1].min() > 120:
                words.append({**record, "fallback_reason": "spans_lines"})
                continue
            mapped = apply_metrics(tokens[None], boxes[None], class_metrics)[0]
            corrected = adjust_boxes(tokens, mapped, thresholds)
            origin = (
                np.floor(np.minimum(mapped[:, :2].min(0), corrected[:, :2].min(0))).astype(int) - 28
            )
            end = (
                np.ceil(np.maximum(mapped[:, 2:].max(0), corrected[:, 2:].max(0))).astype(int) + 28
            )
            np.savez_compressed(
                run / f"solutions/word-{word_id:04d}.npz",
                tokens=tokens,
                original=mapped - np.tile(origin, 2).astype(np.float32),
                corrected=corrected - np.tile(origin, 2).astype(np.float32),
            )
            words.append(
                {
                    **record,
                    "origin": origin.tolist(),
                    "size": (end - origin).tolist(),
                    "fallback_reason": None,
                }
            )
    write(run / "reports/words.json", words)


def worker(request: Path, mode: str) -> None:
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.context_metric_evaluate import ink_metrics

    cfg = read(request)
    run, model = Path(cfg["run"]), Path(cfg["model"])
    assert digest(model / "run.json") == cfg["model_sha256"]
    old = read(model / "reports/request.json")
    cp = model / f"logs/checkpoint-{old['steps']:05d}.pt"
    checkpoint_sha256 = digest(cp)
    assert checkpoint_sha256 == read(model / "reports/result.json")["checkpoint_sha256"]
    setup(cfg["seed"])
    decoder = CanonicalInk().cuda().eval()
    decoder.load_state_dict(torch.load(cp, map_location="cuda", weights_only=False)["model"])
    with np.load(model / "reports/targets.npz") as a:
        codes = torch.tensor(a["codes"], device="cuda")
        by = {int(c): np.flatnonzero(a["labels"] == c) for c in np.unique(a["labels"])}
        real = {c for c, indices in by.items() if a["real"][indices[0]]}
    std = {c: codes[ids].std(0, unbiased=False) for c, ids in by.items()}
    generator = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 200000)
    output = []
    original = read(run / "reports/neural-words.json") if mode == "replay" else None
    for row in read(run / "reports/words.json"):
        if row["fallback_reason"]:
            output.append(row)
            continue
        word_id = row["word_id"]
        with np.load(run / f"solutions/word-{word_id:04d}.npz") as a:
            tokens, boxes, uncorrected = [a[k] for k in ["tokens", "corrected", "original"]]
        chars = tokens[:, 0].tolist()
        if any(c not in by for c in chars):
            output.append({**row, "fallback_reason": "unsupported_neural_character"})
            continue
        style = torch.zeros((len(chars), 8), device="cuda")
        centers = []
        for pos, c in enumerate(chars):
            if c in real:
                ids = by[c]
                k = int(torch.randint(len(ids), (1,), generator=generator, device="cuda"))
                center = int(ids[k])
                style[pos] = codes[center] + 0.15 * std[c] * torch.randn(
                    8, device="cuda", generator=generator
                )
                centers.append(center)
            else:
                centers.append(None)
        width, height = row["size"]
        semantic = torch.tensor(tokens, device="cuda")
        layers = compose(
            decoder,
            semantic,
            torch.tensor(boxes, device="cuda"),
            style,
            height,
            width,
            return_layers=True,
        )
        metrics = ink_metrics(layers, tokens)
        reason = None
        if metrics["empty_glyphs"] or metrics["ink_at_edge"]:
            reason = "empty_glyph_or_patch_boundary"
        elif sum(metrics["overlap"]) > 0:
            prior = compose(
                decoder,
                semantic,
                torch.tensor(uncorrected, device="cuda"),
                style,
                height,
                width,
                return_layers=True,
            )
            original_overlap = ink_metrics(prior, tokens)["overlap"]
            if any(x > y for x, y in zip(metrics["overlap"], original_overlap, strict=True)):
                reason = "increased_adjacent_overlap"
        ink = layers.amax(0).cpu().numpy()
        dest = run / f"comparisons/word-{word_id:04d}.png"
        alpha = np.uint8(ink * 255)
        bbox_y, bbox_x = np.where(alpha > 0)
        extent = (
            [int(bbox_x.min()), int(bbox_y.min()), int(bbox_x.max()) + 1, int(bbox_y.max()) + 1]
            if len(bbox_x)
            else None
        )
        receipt = run / f"reports/word-{word_id:04d}.npz"
        if mode == "generate":
            Image.fromarray(alpha).save(dest)
            np.savez_compressed(receipt, ink=ink, style=style.cpu().numpy())
        else:
            with np.load(receipt) as a:
                assert np.array_equal(ink, a["ink"]) and np.array_equal(
                    style.cpu().numpy(), a["style"]
                )
            assert np.array_equal(alpha, np.asarray(Image.open(dest)))
            assert original[len(output)]["sha256"] == digest(dest)
        output.append(
            {
                **row,
                "fallback_reason": reason,
                "alpha": str(dest),
                "sha256": digest(dest),
                "ink_extent": extent,
                "centers": centers,
                "checkpoint_sha256": checkpoint_sha256,
                "glyphs": [
                    {
                        "semantic": chr(c),
                        "actual_glyph_id": f"{checkpoint_sha256[:16]}:{word_id}:{position}",
                        "source_type": "learned-offline-ink",
                        "real_scan_training_coverage": c in real,
                        "font_teacher_only_training": c not in real,
                        "personal_collected": False,
                    }
                    for position, c in enumerate(chars)
                ],
            }
        )
    if mode == "generate":
        write(run / "reports/neural-words.json", output)
    else:
        assert output == original
        write(
            run / "reports/replay.json",
            {
                "all_word_conditions_float_ink_png_exact": True,
                "generated_words": sum("alpha" in q for q in output),
            },
        )
    print(mode, len(output), flush=True)


def assemble(run: Path, source: Path) -> dict:
    import pypdfium2 as pdfium
    from PIL import Image
    from pypdf import PdfWriter

    from hwsol.config import RenderConfig
    from hwsol.jobs.paper import pdf_image
    from hwsol.workspace import profile_store
    from research.corpus.context_composer import layout
    from research.corpus.mixed_ink_pages import collision_reason, label_pdf

    cfg = RenderConfig.model_validate(read(source / "run.json")["config"]["render_config"])
    profile = profile_store().resolve("personal@v1")
    all_words = read(run / "reports/neural-words.json")
    reports = []
    neural_pdf, baseline_pdf = PdfWriter(), PdfWriter()

    def raster(blob):
        with pdfium.PdfDocument(blob) as doc:
            assert len(doc) == 1
            return pdf_image(doc, 0, cfg.dpi / 72)

    for case in read(source / "reports/pages.json"):
        index = case["page"]
        base, renderer, glyphs = layout(case["source"], case["identity"], profile, cfg)
        assert glyphs == case["provenance"]
        occupied = np.asarray([g["bounds_px"] for g in glyphs])
        replacements, removed, words = {}, set(), []
        for row in [q for q in all_words if q["page"] == index]:
            reason = row["fallback_reason"]
            ids = row["items"]
            if not reason:
                offset = np.tile(row["origin"], 2)
                bounds = np.array(row["ink_extent"]) + offset
                reason = collision_reason(bounds, occupied, ids, base.width, base.height)
                if not reason:
                    with Image.open(row["alpha"]) as im:
                        rgba = np.zeros((*im.size[::-1], 4), np.uint8)
                        rgba[..., 3] = np.asarray(im)
                    replacements[ids[0]] = (
                        "image",
                        Image.fromarray(rgba),
                        (1, 0, 0, 1, *row["origin"]),
                    )
                    removed.update(ids)
                    occupied[ids] = bounds
            words.append({**row, "fallback_reason": reason})
        page = copy.copy(base)
        page.items = [
            replacements.get(i, x)
            for i, x in enumerate(base.items)
            if i not in removed or i in replacements
        ]
        math_ids = [i for i, g in enumerate(glyphs) if g["context"] == "mathematics"]
        assert not removed.intersection(math_ids)
        math_page = copy.copy(base)
        math_page.items = [base.items[i] for i in math_ids]
        math_pixels = np.asarray(raster(renderer.to_vector_pdf([math_page])).convert("L")) < 255
        baseline = renderer.to_vector_pdf([base])
        result = renderer.to_vector_pdf([page])
        cpu_pixels, neural_pixels = [
            np.asarray(raster(blob).convert("RGB")) for blob in [baseline, result]
        ]
        assert np.array_equal(cpu_pixels[math_pixels], neural_pixels[math_pixels])
        comparison = run / f"comparisons/handnote-{index + 1}.pdf"
        comparison.write_bytes(baseline)
        deliverable = run / f"comparisons/neural-{index + 1}.pdf"
        label = label_pdf(result, "Frozen neural English with measured placement")
        deliverable.write_bytes(label)
        raster(label).save(run / f"rendered/pages/page-{index + 1:02d}.png")
        baseline_pdf.append(comparison)
        neural_pdf.append(deliverable)
        reports.append(
            {
                "page": index,
                "identity": case["identity"],
                "word_provenance": words,
                "math_glyph_provenance": [glyphs[i] for i in math_ids],
                "math_pixels_equal": True,
                "math_items": len(math_ids),
                "neural_words": sum(not q["fallback_reason"] for q in words),
                "fallback_words": sum(bool(q["fallback_reason"]) for q in words),
                "handnote_word_provenance": [
                    glyphs[i] for q in words if q["fallback_reason"] for i in q["items"]
                ],
            }
        )
    assert len(reports) == 4
    baseline_pdf.write(run / "comparisons/handnote-homework.pdf")
    baseline_pdf.close()
    neural_pdf.write(run / "rendered/homework_solutions.pdf")
    neural_pdf.close()
    write(run / "reports/page-provenance.json", reports)
    return {
        "a4_pages": len(reports),
        "neural_words": sum(q["neural_words"] for q in reports),
        "fallback_words": sum(q["fallback_words"] for q in reports),
        "math_items_equal": sum(q["math_items"] for q in reports),
        "math_pixels_equal": True,
        "new_neural_updates": 0,
        "C_used": False,
    }


if __name__ == "__main__":
    import sys

    worker(Path(sys.argv[1]), sys.argv[2])
