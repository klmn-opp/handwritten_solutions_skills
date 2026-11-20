"""Sixteen intact source-span annotations and zero-update canonical inference.

CPU preparation/analysis does not import Torch. The explicit worker uses the
isolated research environment; B0 tests an untrained physical-state contract.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, profile_store, verify_run

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "workspace/runs/20261120T110917401810Z-context-supervision-audit-cacf3188"
MODEL = ROOT / "workspace/runs/20261120T022439649806Z-canonical-ink-component-refinement-fbea19c9"
MIXED = ROOT / "workspace/runs/20261120T040707343032Z-metric-neural-english-math-homework-88d48c44"
CP_SHA = "779a2ab51902a17a7068df2c32bf7f5ceddef938ee6982f116f39e0dfb3ad15d"
FORBIDDEN = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
SIZE = (512, 128)

# Explicit Agent visual annotations on unchanged RGB originals. Coordinates are
# half-open source-word pixels, not inferred individual-character boundaries.
ANNOTATIONS = [
    (
        "w017",
        "ff",
        (3, 5),
        (91, 0, 156, 93),
        "medium",
        False,
        "Complete shared ff bar/descenders; rectangle also contains preceding e tail.",
    ),
    (
        "w070",
        "ff",
        (2, 4),
        (57, 0, 120, 84),
        "high",
        True,
        "Both f stems, joined crossbar and descenders retained; no per-f split.",
    ),
    (
        "w012",
        "fo",
        (0, 2),
        (0, 0, 75, 97),
        "high",
        True,
        "Complete f descender and crossbar contacting o; following l separated.",
    ),
    (
        "w063",
        "fo",
        (0, 2),
        (0, 0, 64, 85),
        "high",
        True,
        "Full fo crossbar contact; following l outside ROI.",
    ),
    (
        "w168",
        "fo",
        (0, 2),
        (0, 0, 52, 81),
        "high",
        True,
        "Full black fo with grid retained; following r separate.",
    ),
    (
        "w170",
        "fo",
        (0, 2),
        (0, 0, 65, 86),
        "high",
        True,
        "Full red fo with grid retained; source color differs from decoder ink.",
    ),
    (
        "w020",
        "th",
        (0, 2),
        (0, 7, 74, 88),
        "high",
        True,
        "Full t return rising into h; following r separated.",
    ),
    (
        "w060",
        "th",
        (0, 2),
        (0, 0, 82, 61),
        "high",
        True,
        "Full t return and h; following e separated.",
    ),
    (
        "w006",
        "th",
        (0, 2),
        (20, 42, 108, 100),
        "medium",
        False,
        "Full th, but h-to-e ink meets rectangle boundary; clean local mask unavailable.",
    ),
    (
        "w093",
        "th",
        (0, 2),
        (0, 0, 90, 53),
        "high",
        True,
        "Full th return and h; following a separated.",
    ),
    (
        "w101",
        "th",
        (0, 2),
        (0, 0, 78, 50),
        "medium",
        False,
        "Full th with connected following e overlapping ROI; whole-word supervision only.",
    ),
    (
        "w118",
        "th",
        (0, 2),
        (0, 0, 70, 41),
        "high",
        True,
        "Full th, adjacent e retained in full original outside ROI.",
    ),
    (
        "w113",
        "fo",
        (0, 2),
        (0, 0, 67, 90),
        "high",
        True,
        "Natural unconnected red f/o, same ICS page as joined w170; grid retained.",
    ),
    (
        "w013",
        "th",
        (0, 2),
        (0, 0, 70, 66),
        "high",
        True,
        "Natural unconnected t/h control; e outside ROI.",
    ),
    (
        "w045",
        "th",
        (2, 4),
        (59, 0, 137, 76),
        "high",
        True,
        "Natural unconnected t/h, complete right-hand span; preceding i dot outside.",
    ),
    (
        "w076",
        "th",
        (2, 4),
        (60, 11, 128, 71),
        "high",
        True,
        "Natural unconnected t/h; complete bars and h tail at original crop edge.",
    ),
]


def read(path):
    return json.loads(Path(path).read_text())


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def fit_ink(ink):
    """Independent, uniform whole-word fit; no per-character target warp."""
    yy, xx = np.where(ink > 0.1)
    if not len(xx):
        raise ValueError("Empty input ink")
    x, y, right, bottom = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
    scale = min(480 / (right - x), 96 / (bottom - y))
    width, height = max(1, round((right - x) * scale)), max(1, round((bottom - y) * scale))
    dx, dy = (SIZE[0] - width) // 2, (SIZE[1] - height) // 2
    # Same uint8/Lanczos convention as the existing word_style_data.normalize.
    patch = Image.fromarray(np.uint8(ink[y:bottom, x:right] * 255)).resize(
        (width, height), Image.Resampling.LANCZOS
    )
    out = np.zeros(SIZE[::-1], np.float32)
    out[dy : dy + height, dx : dx + width] = np.asarray(patch) / 255
    sx, sy = width / (right - x), height / (bottom - y)
    matrix = [[sx, 0, dx - sx * x], [0, sy, dy - sy * y], [0, 0, 1]]
    return out, {
        "bbox": [x, y, right, bottom],
        "ideal_uniform_scale": scale,
        "resized_size": [width, height],
        "translation": [dx, dy],
        "matrix": matrix,
        "inverse": np.linalg.inv(matrix).tolist(),
        "integer_resize_rounding_preserved": True,
    }


def map_box(box, matrix):
    x, y, right, bottom = box
    points = np.array([[x, y, 1], [right, bottom, 1]]) @ np.asarray(matrix).T
    return [float(points[0, 0]), float(points[0, 1]), float(points[1, 0]), float(points[1, 1])]


def roi_sheets(path, rows):
    pages = []
    for start in range(0, len(rows), 4):
        page = Image.new("RGB", (1800, 1480), "white")
        draw = ImageDraw.Draw(page)
        for j, row in enumerate(rows[start : start + 4]):
            x, y = 55 + (j % 2) * 850, 45 + (j // 2) * 700
            annotation = row["annotation"]
            draw.text(
                (x + 12, y + 8),
                f"{row['id']} {row['label']} | {annotation['span']}",
                font=font(25),
                fill="black",
            )
            draw.text(
                (x + 12, y + 43),
                f"{row['source_sha256'][:8]} p{row['page']} | {annotation['confidence']}",
                font=font(18),
                fill="black",
            )
            raw = Image.open(path / f"solutions/{row['id']}-original.png").convert("RGB")
            scale = min(750 / raw.width, 350 / raw.height, 5)
            image = raw.resize(
                (round(raw.width * scale), round(raw.height * scale)), Image.Resampling.NEAREST
            )
            page.paste(image, (x + 15, y + 85))
            box = annotation["roi_word_px"]
            marked = [
                x + 15 + box[0] * scale,
                y + 85 + box[1] * scale,
                x + 15 + box[2] * scale - 1,
                y + 85 + box[3] * scale - 1,
            ]
            draw.rectangle(
                marked, outline="blue" if annotation["local_mask_available"] else "orange", width=3
            )
            draw.text(
                (x + 12, y + 455),
                f"word ROI {box} | {annotation['role']}",
                font=font(17),
                fill="black",
            )
            draw.text(
                (x + 12, y + 486),
                f"page ROI {annotation['roi_page_raster_px']}",
                font=font(16),
                fill="black",
            )
            note = annotation["note"]
            words, lines, line = note.split(), [], ""
            for word in words:
                if len(line + word) > 67:
                    lines.append(line)
                    line = ""
                line += word + " "
            lines.append(line)
            for k, line in enumerate(lines):
                draw.text((x + 12, y + 522 + k * 24), line, font=font(17), fill="black")
        page.save(path / f"comparisons/roi-{start // 4 + 1:02d}.png")
        pages.append(page)
    pages[0].save(
        path / "comparisons/roi-review.pdf", save_all=True, append_images=pages[1:], resolution=150
    )


def prepare():
    from hwsol.config import RenderConfig
    from hwsol.profiles.store import copy_snapshot
    from research.corpus.context_composer import layout
    from research.corpus.context_metric_layout import adjust_boxes
    from research.corpus.semantic_ink_data import conditions
    from research.corpus.semantic_raw_data import apply_metrics

    verify_run(AUDIT)
    cp = MODEL / "logs/checkpoint-01000.pt"
    if sha256(cp) != CP_SHA:
        raise ValueError("Frozen v21 checkpoint mismatch")
    profile = profile_store().resolve("personal@v1")
    if profile.provenance()["font_sha256"] != (
        "60466bbde4f39280838ac15f1121fc975cf841f42b581f74618eff50c8b2bcd9"
    ):
        raise ValueError("Frozen personal v1 font changed")
    old = read(MIXED / "reports/request.json")
    metric = Path(old["metric"])
    thresholds = read(metric / "reports/request.json")["thresholds"]
    config = RenderConfig(
        dpi=300,
        font_size_pt=11,
        line_spacing=1.6,
        seed=1909,
        semantic_layout=True,
        glyph_metrics="raw",
        math_strokes="personal",
    )
    cfg = {
        "audit": str(AUDIT),
        "checkpoint": str(cp),
        "checkpoint_sha256": CP_SHA,
        "model": str(MODEL),
        "class_metrics": old["class_metrics"],
        "thresholds": thresholds,
        "seed": 2012,
        "layout": config.model_dump(),
        "new_neural_updates": 0,
        "C_implemented": False,
        "sealed_C_used": False,
        "B0": "untrained physical word-state zero contract, independent decoder call",
        "sampling": "per-word reset seed 202012; same-class PCA center + 0.15 std noise",
        "geometry_sensitivity": {"scale": [0.97, 1.0, 1.03], "dx": [-2, 0, 2], "dy": [-2, 0, 2]},
        "independent_holdout": False,
    }
    run = Run.create(
        "context-alignment-pilot",
        config=cfg,
        seed=2012,
        profile=profile.provenance(),
        checkpoints={"v21": CP_SHA},
    )
    try:
        copy_snapshot(profile, run.path / "solutions/profile")
        shutil.copyfile(__file__, run.path / "logs/context_alignment_pilot_prepare.py")
        protected = {
            str(cp): CP_SHA,
            str(ROOT / "workspace/latest.json"): sha256(ROOT / "workspace/latest.json"),
        }
        for source in [
            AUDIT / "run.json",
            AUDIT / "reports/inventory.json",
            MODEL / "reports/targets.npz",
            MIXED / "reports/request.json",
            metric / "reports/request.json",
        ]:
            protected[str(source)] = sha256(source)
        inventory = {r["id"]: r for r in read(AUDIT / "reports/inventory.json")["words"]}
        rows = []
        for index, (identity, span, positions, box, confidence, mask, note) in enumerate(
            ANNOTATIONS
        ):
            source = inventory[identity]
            assert source["source_sha256"] != FORBIDDEN
            assert source["label"][positions[0] : positions[1]] == span
            assert sha256(Path(source["source_path"])) == source["crop_sha256"]
            protected[source["source_path"]] = source["crop_sha256"]
            copied = run.path / f"solutions/{identity}-original.png"
            shutil.copyfile(source["source_path"], copied)
            image = Image.open(copied).convert("RGB")
            assert 0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height
            image.crop(box).save(run.path / f"solutions/{identity}-span-rgb.png")
            parent = source["geometry"]["parent_bbox"]
            child = source["geometry"]["box_in_source_crop"]
            offset = [parent[0] + child[0], parent[1] + child[1]]
            word_page = [offset[0], offset[1], offset[0] + image.width, offset[1] + image.height]
            annotation = {
                "span": span,
                "string_positions": list(positions),
                "roi_word_px": list(box),
                "roi_page_raster_px": [
                    box[0] + offset[0],
                    box[1] + offset[1],
                    box[2] + offset[0],
                    box[3] + offset[1],
                ],
                "word_page_raster_px": word_page,
                "coordinate_origin": "top-left",
                "confidence": confidence,
                "local_mask_available": mask,
                "mask_scope": "intact span rectangle; background retained"
                if mask
                else "whole-word only",
                "role": "natural-unconnected-control" if index >= 12 else "cross-letter-ink",
                "note": note,
                "reviewer": "Codex Agent visual annotation",
                "author_confirmation": False,
                "parent_review": "pending",
                "original_label_or_review_modified": False,
                "individual_character_cuts": False,
                "word_to_page_raster": [[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]],
            }
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            target, transform = fit_ink(np.clip((190 - gray) / 140, 0, 1))
            # Fresh deterministic layout supplies boxes only; no font bitmap is a neural input.
            _, _, provenance = layout(source["label"], "Pilot", profile, config)
            glyphs = provenance[len("Pilot") :]
            tokens, boxes = conditions(
                source["label"],
                {"glyph_provenance": glyphs, "crop_from_pdf": [0, 0]},
                {"scale": 1, "translation": [0, 0], "bbox": [0, 0]},
            )
            mapped = apply_metrics(tokens[None], boxes[None], old["class_metrics"])[0]
            corrected = adjust_boxes(tokens, mapped, thresholds)
            origin = (
                np.floor(np.minimum(mapped[:, :2].min(0), corrected[:, :2].min(0))).astype(int) - 28
            )
            end = (
                np.ceil(np.maximum(mapped[:, 2:].max(0), corrected[:, 2:].max(0))).astype(int) + 28
            )
            np.savez_compressed(
                run.path / f"solutions/{identity}-inputs.npz",
                tokens=tokens,
                original=mapped - np.tile(origin, 2).astype(np.float32),
                corrected=corrected - np.tile(origin, 2).astype(np.float32),
                target=target,
            )
            row = {
                **source,
                "annotation": annotation,
                "target_transform": transform,
                "target_normalized_roi": map_box(box, transform["matrix"]),
                "inference_size": (end - origin).tolist(),
                "layout_origin": origin.tolist(),
                "fresh_layout_provenance": glyphs,
                "target_pixels_in_inference_input": False,
            }
            rows.append(row)
        atomic_json(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
        atomic_json(run.path / "reports/rows.json", rows)
        atomic_json(run.path / "reports/protected.json", protected)
        roi_sheets(run.path, rows)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def pdf_geometry(path):
    """Recover rotation-aware PDFium mapping only after exact source-crop replay."""
    import pypdfium2 as pdfium

    rows = read(path / "reports/rows.json")
    result = []
    cache = {}
    for row in rows:
        source = Path(row["document"]["original_assets"][0])
        assert sha256(source) == row["source_sha256"] != FORBIDDEN
        key = (str(source), row["page"])
        if key not in cache:
            document = pdfium.PdfDocument(source)
            page = document[row["page"] - 1]
            bitmap = page.render(scale=300 / 72)
            image = bitmap.to_pil().convert("RGB")
            conv = bitmap.get_posconv(page)
            origin, x, y = [np.array(conv.to_page(*point)) for point in [(0, 0), (1, 0), (0, 1)]]
            matrix = np.array(
                [
                    [x[0] - origin[0], y[0] - origin[0], origin[0]],
                    [x[1] - origin[1], y[1] - origin[1], origin[1]],
                    [0, 0, 1],
                ]
            )
            cache[key] = (image, matrix, page.get_rotation(), list(page.get_bbox()))
            bitmap.close()
            page.close()
            document.close()
        image, matrix, rotation, bbox = cache[key]
        crop = image.crop(row["annotation"]["word_page_raster_px"])
        original = Image.open(path / f"solutions/{row['id']}-original.png").convert("RGB")
        equal = np.array_equal(np.asarray(crop), np.asarray(original))
        offset = np.array(row["annotation"]["word_to_page_raster"])
        result.append(
            {
                "id": row["id"],
                "source_pdf": str(source),
                "source_sha256": row["source_sha256"],
                "page": row["page"],
                "dpi": 300,
                "raster_size": list(image.size),
                "pdf_rotation": rotation,
                "pdf_bbox": bbox,
                "original_word_rgb_exactly_reproduced": equal,
                "mapping_validated": equal,
                "page_raster_to_pdf": matrix.tolist() if equal else None,
                "word_to_pdf": (matrix @ offset).tolist() if equal else None,
                "method": "PDFium render and DeviceToPage; mapping withheld if crop differs",
            }
        )
    atomic_json(path / "reports/pdf-transforms.json", result)


def zero_word_state(boxes, state):
    """Physical shared-word state (log scale x/y, shear, baseline); zero is identity.

    Caller chooses the tensor library. This function owns no trained parameters.
    At nonzero state corner extrema handle shear; no cross-class PCA axes mixed.
    """
    import torch

    center = (boxes[:, :2].amin(0) + boxes[:, 2:].amax(0)) / 2
    corners = torch.stack(
        [boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [0, 3]], boxes[:, [2, 3]]], 1
    )
    delta = corners - center
    dx = delta[..., 0] * torch.expm1(state[0]) + delta[..., 1] * state[2]
    dy = delta[..., 1] * torch.expm1(state[1]) + state[3]
    transformed = corners + torch.stack([dx, dy], -1)
    return torch.cat([transformed.amin(1), transformed.amax(1)], -1)


def worker(path):
    import torch

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk

    assert Run(path).record()["status"] == "running"
    request = read(path / "reports/request.json")
    cp = Path(request["checkpoint"])
    assert sha256(cp) == request["checkpoint_sha256"] == CP_SHA
    setup(request["seed"])
    decoder = CanonicalInk().cuda().eval()
    decoder.load_state_dict(torch.load(cp, map_location="cuda", weights_only=False)["model"])
    initial = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    with np.load(Path(request["model"]) / "reports/targets.npz") as data:
        codes = torch.tensor(data["codes"], device="cuda")
        by = {int(c): np.flatnonzero(data["labels"] == c) for c in np.unique(data["labels"])}
        real = {c for c, ids in by.items() if data["real"][ids].all()}
    std = {c: codes[ids].std(0, unbiased=False) for c, ids in by.items()}
    calls = []
    current = {}

    def called(_module, inputs, _output):
        calls.append(
            {**current, "character_count": len(inputs[0]), "grad_enabled": torch.is_grad_enabled()}
        )

    hook = decoder.register_forward_hook(called)
    result = []
    started = time.monotonic()
    for row in read(path / "reports/rows.json"):
        with np.load(path / f"solutions/{row['id']}-inputs.npz") as data:
            tokens = torch.tensor(data["tokens"], device="cuda")
            boxes = torch.tensor(data["corrected"], device="cuda")
        chars = tokens[:, 0].tolist()
        if any(c not in by for c in chars):
            result.append({"id": row["id"], "status": "unsupported-condition", "outputs": []})
            continue
        generator = torch.Generator(device="cuda").manual_seed(request["seed"] + 200000)
        style = torch.zeros((len(chars), 8), device="cuda")
        centers = []
        for pos, char in enumerate(chars):
            if char in real:
                ids = by[char]
                selected = int(torch.randint(len(ids), (1,), generator=generator, device="cuda"))
                center = int(ids[selected])
                style[pos] = codes[center] + 0.15 * std[char] * torch.randn(
                    8, device="cuda", generator=generator
                )
                centers.append(center)
            else:
                centers.append(None)
        state = torch.zeros(4, device="cuda")
        bboxes = zero_word_state(boxes, state)
        assert torch.equal(boxes, bboxes)
        width, height = row["inference_size"]
        outputs = []
        for arm, actual_boxes in [("A", boxes), ("B0", bboxes)]:
            current.update(id=row["id"], arm=arm, independent_forward=True)
            torch.cuda.synchronize()
            clock = time.monotonic()
            with torch.inference_mode():
                alpha = compose(decoder, tokens, actual_boxes, style, height, width)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - clock
            image = alpha.detach().cpu().numpy()
            outputs.append(image)
            Image.fromarray(np.uint8((1 - image.clip(0, 1)) * 255)).save(
                path / f"rendered/pages/{row['id']}-{arm}.png"
            )
            calls[-1]["elapsed_seconds"] = elapsed
        np.savez_compressed(
            path / f"rendered/{row['id']}-inference.npz",
            A=outputs[0],
            B0=outputs[1],
            local_conditions=style.detach().cpu().numpy(),
            tokens=tokens.cpu().numpy(),
            boxes_A=boxes.cpu().numpy(),
            boxes_B0=bboxes.cpu().numpy(),
            word_state=state.cpu().numpy(),
        )
        result.append(
            {
                "id": row["id"],
                "status": "executed",
                "text": row["label"],
                "pca_center_indices": centers,
                "font_only_characters": [chr(c) for c in chars if c not in real],
                "local_conditions_source": "independent same-class sampled PCA neighborhood",
                "target_pixels_used": False,
                "A_B0_bit_equal": np.array_equal(*outputs),
                "A_B0_max_abs_difference": float(np.abs(outputs[0] - outputs[1]).max()),
            }
        )
    hook.remove()
    assert all(torch.equal(v, initial[k]) for k, v in decoder.state_dict().items())
    assert sha256(cp) == CP_SHA
    shutil.copyfile(__file__, path / "logs/context_alignment_pilot_worker.py")
    atomic_json(
        path / "reports/inference.json",
        {
            "rows": result,
            "actual_forward_calls": calls,
            "elapsed_seconds": time.monotonic() - started,
            "architecture": decoder.architecture,
            "parameters": sum(p.numel() for p in decoder.parameters()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "weights_unchanged": True,
            "checkpoint_sha256": CP_SHA,
            "new_neural_updates": 0,
            "optimizer_created": False,
            "B0_is_trained_context_model": False,
            "reconstruction": False,
            "target_pixels_or_true_target_codes_used": False,
        },
    )


def revise(parent):
    """Keep first-round ROI/inference immutable; child is an explicit reanalysis."""
    original = Run(parent)
    inference = read(parent / "reports/inference.json")
    if original.record()["status"] == "running":
        original.finish(
            "prepared",
            result={
                "purpose": "first-round RGB ROI and real frozen-model inference evidence",
                "actual_forward_calls": len(inference["actual_forward_calls"]),
                "A_B0_all_bit_equal": all(r["A_B0_bit_equal"] for r in inference["rows"]),
                "new_neural_updates": 0,
                "needs": "parent-reviewed ROI margins and sensitivity analysis",
            },
        )
    verify_run(parent)
    run = Run.create(
        "context-alignment-review",
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": sha256(parent / "run.json"),
            "purpose": "ROI review and analysis of this parent's 32 actual new forwards",
            "new_neural_inference_in_child": 0,
            "new_neural_updates": 0,
        },
        seed=2012,
        profile=original.record()["profile"],
        checkpoints={"v21": CP_SHA},
    )
    for folder in ["solutions", "rendered", "logs", "reports"]:
        shutil.copytree(parent / folder, run.path / folder, dirs_exist_ok=True)
    (run.path / "reports/result.json").unlink()
    shutil.copyfile(parent / "run.json", run.path / "reports/parent-run.json")
    shutil.copyfile(__file__, run.path / "logs/context_alignment_pilot_review.py")
    rows = read(run.path / "reports/rows.json")
    revised = {r[0]: r for r in ANNOTATIONS}
    for row in rows:
        annotation = row["annotation"]
        old_box = annotation["roi_word_px"]
        new_box = list(revised[row["id"]][3])
        annotation["first_round_roi_word_px"] = old_box
        annotation["roi_word_px"] = new_box
        offset = annotation["word_page_raster_px"][:2]
        annotation["roi_page_raster_px"] = [
            new_box[0] + offset[0],
            new_box[1] + offset[1],
            new_box[2] + offset[0],
            new_box[3] + offset[1],
        ]
        annotation["parent_review"] = "root visually accepted categories for feasibility only"
        annotation["recheck"] = (
            "child Agent checked native pixels and enlarged margins where needed"
        )
        annotation["annotation_promoted_to_training_truth"] = False
        row["target_normalized_roi"] = map_box(new_box, row["target_transform"]["matrix"])
        image = Image.open(run.path / f"solutions/{row['id']}-original.png").convert("RGB")
        image.crop(new_box).save(run.path / f"solutions/{row['id']}-span-rgb.png")
    atomic_json(run.path / "reports/rows.json", rows)
    atomic_json(
        run.path / "reports/lineage.json",
        {
            "inference_origin_run": str(parent.resolve()),
            "sha256": sha256(parent / "run.json"),
            "inference_manifest_sha256": sha256(parent / "reports/inference.json"),
            "actual_forwards_in_origin": 32,
            "actual_forwards_in_child": 0,
            "copied_float_arrays_not_regenerated": True,
            "annotation_changes": {
                r["id"]: {
                    "old": r["annotation"]["first_round_roi_word_px"],
                    "new": r["annotation"]["roi_word_px"],
                }
                for r in rows
                if r["annotation"]["first_round_roi_word_px"] != r["annotation"]["roi_word_px"]
            },
            "review": (
                "Root inspected four original sheets; no author confirmation or training admission"
            ),
        },
    )
    roi_sheets(run.path, rows)
    return run.path


def transform_word(ink, scale, dx, dy):
    # Explicit global center-based transform only; no per-character fitting.
    cx, cy = SIZE[0] / 2, SIZE[1] / 2
    inverse = (1 / scale, 0, cx - (cx + dx) / scale, 0, 1 / scale, cy - (cy + dy) / scale)
    return np.asarray(
        Image.fromarray(ink.astype(np.float32), mode="F").transform(
            SIZE, Image.Transform.AFFINE, inverse, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def iou(first, second):
    x, y = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - x) * max(0, bottom - y)

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    return overlap / max(area(first) + area(second) - overlap, 1e-12)


def analyze(path):
    rows = read(path / "reports/rows.json")
    inference = read(path / "reports/inference.json")
    results, panels, normalized = [], [], {}
    for row in rows:
        with np.load(path / f"solutions/{row['id']}-inputs.npz") as data:
            target = data["target"]
        with np.load(path / f"rendered/{row['id']}-inference.npz") as data:
            assert np.array_equal(data["A"], data["B0"])
            a, transform = fit_ink(data["A"])
            boxes = data["boxes_A"]
        normalized[row["id"]] = (a, target)
        roi = row["target_normalized_roi"]
        ix = [
            max(0, int(np.floor(roi[0]))),
            max(0, int(np.floor(roi[1]))),
            min(512, int(np.ceil(roi[2]))),
            min(128, int(np.ceil(roi[3]))),
        ]
        roi_slice = np.s_[ix[1] : ix[3], ix[0] : ix[2]]
        start, end = row["annotation"]["string_positions"]
        span = boxes[start:end]
        predicted = map_box([*span[:, :2].min(0), *span[:, 2:].max(0)], transform["matrix"])
        union = (a > 0.1) | (target > 0.1)

        def metric(image, truth=target, union=union, roi_slice=roi_slice):
            return {
                "word_l1": float(np.abs(image - truth).mean()),
                "fixed_foreground_union_l1": float(np.abs(image - truth)[union].mean()),
                "span_roi_l1": float(np.abs(image - truth)[roi_slice].mean()),
            }

        grid = []
        for scale in [0.97, 1.0, 1.03]:
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    image = transform_word(a, scale, dx, dy)
                    grid.append({"scale": scale, "dx": dx, "dy": dy, **metric(image)})
        best = min(grid, key=lambda r: r["word_l1"])
        corrected = transform_word(a, best["scale"], best["dx"], best["dy"])
        base = metric(a)
        p90 = float(np.quantile(target[target > 0.1], 0.9))
        tone = np.minimum(target / max(p90, 0.1), 1)
        tone_metric = metric(a, tone)
        item = {
            "id": row["id"],
            "text": row["label"],
            "group": row["group_id"],
            "role": row["annotation"]["role"],
            "local_mask_available": row["annotation"]["local_mask_available"],
            "baseline": base,
            "word_transform_grid": grid,
            "oracle_best_whole_word_transform": best,
            "word_l1_relative_change": (base["word_l1"] - best["word_l1"]) / base["word_l1"],
            "span_l1_relative_change": (base["span_roi_l1"] - best["span_roi_l1"])
            / base["span_roi_l1"],
            "target_ink_p90": p90,
            "target_p90_gain_sensitivity_only": tone_metric,
            "generated_span_box_normalized": predicted,
            "target_span_box_normalized": roi,
            "span_box_iou": iou(predicted, roi),
            "generated_normalization": transform,
            "condition_sampling_independent_of_target": True,
            "local_residual_identified_as_context_only": False,
            "interpretation": (
                "Unpaired class-sampled shapes and original geometry/pen differences "
                "remain confounded."
            ),
        }
        results.append(item)
        np.savez_compressed(
            path / f"comparisons/{row['id']}-diagnostics.npz",
            A_normalized=a,
            target_normalized=target,
            A_global_oracle=corrected,
            target_gain_diagnostic=tone,
        )
        panel = Image.new("RGB", (1650, 440), "white")
        draw = ImageDraw.Draw(panel)
        draw.text(
            (40, 14),
            f"{row['id']} {row['label']} | {row['annotation']['span']} | "
            f"{row['annotation']['role']}",
            font=font(24),
            fill="black",
        )
        for column, (title, array) in enumerate(
            [
                ("Original ink diagnostic (RGB preserved separately)", target),
                ("Fresh A / B0 (equal); independent word fit", a),
                ("A global transform sensitivity (oracle, not generator)", corrected),
            ]
        ):
            x = 40 + column * 535
            draw.text((x, 61), title, font=font(14), fill="black")
            panel.paste(
                Image.fromarray(np.uint8((1 - array.clip(0, 1)) * 255)).convert("RGB"), (x, 94)
            )
            rectangle = roi if column == 0 else predicted
            if column == 2:
                scale = best["scale"]
                rectangle = map_box(
                    predicted,
                    [
                        [scale, 0, 256 * (1 - scale) + best["dx"]],
                        [0, scale, 64 * (1 - scale) + best["dy"]],
                        [0, 0, 1],
                    ],
                )
            draw.rectangle(
                [x + rectangle[0], 94 + rectangle[1], x + rectangle[2], 94 + rectangle[3]],
                outline="blue" if row["annotation"]["local_mask_available"] else "orange",
                width=2,
            )
        draw.text(
            (40, 258),
            f"Box IoU {item['span_box_iou']:.3f} | whole-word L1 change "
            f"{item['word_l1_relative_change']:+.2%} | span L1 change "
            f"{item['span_l1_relative_change']:+.2%}",
            font=font(21),
            fill="black",
        )
        draw.text(
            (40, 300),
            f"Fixed one seed; target pixels/codes never supplied to decoder. "
            f"P90 target tone {p90:.3f}. No context-only residual claim.",
            font=font(19),
            fill="black",
        )
        draw.text(
            (40, 341),
            "Blue boxes are semantic span regions, not per-letter cuts or foreground masks. "
            "Orange is whole-word-only.",
            font=font(18),
            fill="black",
        )
        panel.save(path / f"comparisons/{row['id']}-comparison.png")
        panels.append(panel)
    pages = []
    for i in range(0, len(panels), 4):
        page = Image.new("RGB", (1650, 1760), "white")
        for j, panel in enumerate(panels[i : i + 4]):
            page.paste(panel, (0, j * 440))
        page.save(path / f"comparisons/inference-{i // 4 + 1:02d}.png")
        pages.append(page)
    pages[0].save(
        path / "comparisons/inference-review.pdf",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )
    repeated = []
    for text in sorted({r["label"] for r in rows}):
        matches = [r for r in rows if r["label"] == text]
        for i, first in enumerate(matches):
            for second in matches[i + 1 :]:
                a1, t1 = normalized[first["id"]]
                a2, t2 = normalized[second["id"]]
                repeated.append(
                    {
                        "text": text,
                        "ids": [first["id"], second["id"]],
                        "same_group": first["group_id"] == second["group_id"],
                        "A_normalized_bit_equal": np.array_equal(a1, a2),
                        "original_targets_normalized_l1": float(np.abs(t1 - t2).mean()),
                    }
                )
    protected = read(path / "reports/protected.json")
    for source, digest in protected.items():
        if sha256(Path(source)) != digest:
            raise ValueError(f"Protected source changed: {source}")
    lineage = read(path / "reports/lineage.json")
    verify_run(Path(lineage["inference_origin_run"]))
    summary = {
        "words": len(rows),
        "source_groups": len({r["group_id"] for r in rows}),
        "locally_bounded_spans": sum(r["annotation"]["local_mask_available"] for r in rows),
        "whole_word_only_spans": [
            r["id"] for r in rows if not r["annotation"]["local_mask_available"]
        ],
        "actual_forward_calls_in_parent": len(inference["actual_forward_calls"]),
        "actual_forward_calls_in_child": 0,
        "new_neural_updates": 0,
        "A_B0_all_bit_equal": all(r["A_B0_bit_equal"] for r in inference["rows"]),
        "weights_unchanged": inference["weights_unchanged"],
        "pdf_transforms_exact_replay": sum(
            r["mapping_validated"] for r in read(path / "reports/pdf-transforms.json")
        ),
        "median_span_box_iou": float(np.median([r["span_box_iou"] for r in results])),
        "median_whole_word_geometry_change": float(
            np.median([r["word_l1_relative_change"] for r in results])
        ),
        "median_span_geometry_change": float(
            np.median([r["span_l1_relative_change"] for r in results])
        ),
        "B0_is_trained_context_model": False,
        "C_implemented": False,
        "training_gate": "no-go for this unpaired local A/B/C protocol",
        "continuous_source_data_available": True,
        "author_action_required_for_engineering": False,
        "reason": (
            "Semantic ROI does not pair sampled glyph morphology with original context; "
            "global geometry, pen tone and source shapes cannot be assigned to context alone."
        ),
        "independent_holdout_claim": False,
        "sealed_C_used": False,
    }
    atomic_json(path / "reports/measurements.json", results)
    atomic_json(path / "reports/repeated-content-controls.json", repeated)
    atomic_json(path / "reports/summary.json", summary)
    shutil.copyfile(__file__, path / "logs/context_alignment_pilot_analyze.py")


def seal(path):
    if Run(path).record()["status"] != "running":
        raise ValueError("Run already immutable")
    review = read(path / "reports/visual-review.json")
    rows = read(path / "reports/rows.json")
    if set(review["inspected_word_ids"]) != {r["id"] for r in rows}:
        raise ValueError("All sixteen outputs and source spans need explicit visual review")
    if review["training_admission"] is not False or review["production_admission"] is not False:
        raise ValueError("This zero-update pilot cannot grant model admission")
    for source, digest in read(path / "reports/protected.json").items():
        if sha256(Path(source)) != digest:
            raise ValueError("Protected source changed")
    lineage = read(path / "reports/lineage.json")
    parent = Path(lineage["inference_origin_run"])
    verify_run(parent)
    if sha256(parent / "run.json") != lineage["sha256"]:
        raise ValueError("Parent identity changed")
    for row in rows:
        source = Path(row["document"]["original_assets"][0])
        if sha256(source) != row["source_sha256"]:
            raise ValueError("Original PDF changed")
        inference = f"rendered/{row['id']}-inference.npz"
        if sha256(path / inference) != sha256(parent / inference):
            raise ValueError("Original fresh inference was modified in reanalysis")
        if sha256(path / f"solutions/{row['id']}-original.png") != row["crop_sha256"]:
            raise ValueError("Original RGB changed")
    shutil.copyfile(__file__, path / "logs/context_alignment_pilot_seal.py")
    Run(path).finish("held", result=read(path / "reports/summary.json"))
    verify_run(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["prepare", "pdf-geometry", "worker", "revise", "analyze", "seal"]
    )
    parser.add_argument("run", type=Path, nargs="?")
    args = parser.parse_args()
    if args.phase == "prepare":
        print(prepare(), flush=True)
    elif args.phase == "pdf-geometry":
        pdf_geometry(args.run)
    elif args.phase == "worker":
        worker(args.run)
    elif args.phase == "revise":
        print(revise(args.run), flush=True)
    elif args.phase == "analyze":
        analyze(args.run)
    else:
        seal(args.run)


if __name__ == "__main__":
    main()
