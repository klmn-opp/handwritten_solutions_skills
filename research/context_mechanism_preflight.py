"""Zero-update endpoint/geometry diagnosis using already exposed original words.

No model inference or training: frozen float arrays and original RGB crops are
reused. All target-informed oracles are diagnostic upper bounds, not generation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/runs"
AUDIT = RUNS / "20261120T023639068683Z-canonical-intact-word-audit-627f2eae"
MODEL = RUNS / "20261120T022439649806Z-canonical-ink-component-refinement-fbea19c9"
CONTEXT = RUNS / "20261120T030504700173Z-bounded-real-word-context-training-9bae57b9"
SEALED = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"


def read(path):
    return json.loads(Path(path).read_text())


def translate(array, dx, dy):
    result = np.zeros_like(array)
    height, width = array.shape
    source = (
        slice(max(0, -dy), min(height, height - dy)),
        slice(max(0, -dx), min(width, width - dx)),
    )
    destination = (
        slice(max(0, dy), min(height, height + dy)),
        slice(max(0, dx), min(width, width + dx)),
    )
    result[destination] = array[source]
    return result


def dilate(mask, radius):
    result = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                result |= translate(mask, dx, dy)
    return result


def components(mask):
    labels = np.zeros(mask.shape, np.int32)
    height, width = mask.shape
    count = 0
    for y, x in zip(*np.nonzero(mask), strict=True):
        if labels[y, x]:
            continue
        count += 1
        labels[y, x] = count
        queue = deque([(int(y), int(x))])
        while queue:
            cy, cx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = count
                        queue.append((ny, nx))
    return labels


def regions(boxes, shape):
    y, x = np.indices(shape)
    interior = np.zeros(shape, bool)
    per_character = []
    for left, top, right, bottom in boxes:
        area = (x >= left) & (x <= right) & (y >= top) & (y <= bottom)
        per_character.append(area)
        interior |= area
    gaps = np.zeros(shape, bool)
    for previous, following in zip(boxes[:-1], boxes[1:], strict=True):
        gaps |= (
            (x > previous[2])
            & (x < following[0])
            & (y >= boxes[:, 1].min())
            & (y <= boxes[:, 3].max())
        )
    gaps &= ~interior
    return interior, gaps, per_character


def odds_bounds(base):
    # Same attainable interval as base + base*(1-base)*(exp(delta)-1)/(1+base*(exp(delta)-1)).
    return (base / (base + (1 - base) * np.exp(3)), base / (base + (1 - base) * np.exp(-3)))


def measure(base, target, boxes):
    foreground, prediction = target > 0.5, base > 0.5
    fn, fp = foreground & ~prediction, prediction & ~foreground
    inside, gaps, masks = regions(boxes, base.shape)
    lo, hi = odds_bounds(base)
    oracle = np.clip(target, lo, hi)
    endpoint = (base == 0) | (base == 1)
    distance2, distance4 = dilate(prediction, 2), dilate(prediction, 4)
    cc = components(foreground)
    overlaps, spans = [], []
    for position in range(len(boxes) - 1):
        left = Counter(cc[foreground & masks[position]].tolist())
        right = Counter(cc[foreground & masks[position + 1]].tolist())
        shared = [
            int(k) for k in left.keys() & right.keys() if k and left[k] >= 5 and right[k] >= 5
        ]
        if shared:
            overlaps.append(
                {
                    "left_position": position,
                    "right_position": position + 1,
                    "components": shared,
                    "interpretation": "overlapping boxes can double-count the same stroke",
                }
            )
        # Two overlapping rectangles alone are not evidence of a ligature.
        left = Counter(cc[foreground & masks[position] & ~masks[position + 1]].tolist())
        right = Counter(cc[foreground & masks[position + 1] & ~masks[position]].tolist())
        connected = [
            int(k) for k in left.keys() & right.keys() if k and left[k] >= 5 and right[k] >= 5
        ]
        if connected:
            spans.append(
                {"left_position": position, "right_position": position + 1, "components": connected}
            )
    total_error = np.abs(base - target)
    result = {
        "l1_sum": float(total_error.sum()),
        "dice": float(
            2 * (foreground & prediction).sum() / max(1, foreground.sum() + prediction.sum())
        ),
        "target_ink_mass": float(target.sum()),
        "target_foreground_pixels": int(foreground.sum()),
        "exact_zero_pixels": int((base == 0).sum()),
        "exact_one_pixels": int((base == 1).sum()),
        "endpoint_locked_l1": float(total_error[endpoint].sum()),
        "endpoint_locked_foreground_pixels": int((foreground & (base == 0)).sum()),
        "endpoint_locked_wrong_background_pixels": int((~foreground & (base == 1)).sum()),
        "odds3_oracle_l1_floor": float(np.abs(oracle - target).sum()),
        "odds3_unreachable_foreground_pixels": int((foreground & (hi <= 0.5)).sum()),
        "odds3_unremovable_background_pixels": int((~foreground & (lo > 0.5)).sum()),
        "false_negative_pixels": int(fn.sum()),
        "false_positive_pixels": int(fp.sum()),
        "fn_inside_character_boxes": int((fn & inside).sum()),
        "fn_in_between_box_gaps": int((fn & gaps).sum()),
        "fn_other": int((fn & ~inside & ~gaps).sum()),
        "target_foreground_in_between_box_gaps": int((foreground & gaps).sum()),
        "fn_within_two_pixels_of_base_stroke": int((fn & distance2).sum()),
        "fn_beyond_two_pixels": int((fn & ~distance2).sum()),
        "fn_beyond_four_pixels": int((fn & ~distance4).sum()),
        "connected_pair_candidates": spans,
        "overlapping_box_warnings": overlaps,
    }
    return result, oracle


def best_translation(base, target, radius=2):
    best, shift, error = base, [0, 0], float(np.abs(base - target).sum())
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            candidate = translate(base, dx, dy)
            value = float(np.abs(candidate - target).sum())
            if value < error:
                best, shift, error = candidate, [dx, dy], value
    return best, {"shift": shift, "l1_sum": error, "target_informed_not_generation": True}


def layer(ink, box, shape=(128, 512)):
    y, x = np.indices(shape)
    positions = np.stack((x, y), -1).astype(np.float32)
    local = 16 + 64 * (positions - box[:2]) / (box[2:] - box[:2])
    lx, ly = local[..., 0], local[..., 1]
    xx, yy = np.floor(lx).astype(int), np.floor(ly).astype(int)
    fx, fy = lx - xx, ly - yy
    result = np.zeros(shape, np.float64)
    for dy, dx, weight in [
        (0, 0, (1 - fx) * (1 - fy)),
        (0, 1, fx * (1 - fy)),
        (1, 0, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        ix, iy = xx + dx, yy + dy
        valid = (ix >= 0) & (ix < 96) & (iy >= 0) & (iy < 96)
        result[valid] += ink[iy[valid], ix[valid]] * weight[valid]
    return result


def geometry_oracle(layers, target):
    layers = layers.copy()
    original = np.max(layers, axis=0)
    shifts = []
    for i in range(len(layers)):
        others = (
            np.max(np.delete(layers, i, axis=0), axis=0)
            if len(layers) > 1
            else np.zeros_like(target)
        )
        best = layers[i]
        best_error = np.abs(np.maximum(others, best) - target).sum()
        shift = [0, 0]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                candidate = translate(layers[i], dx, dy)
                error = np.abs(np.maximum(others, candidate) - target).sum()
                if error < best_error:
                    best, shift, best_error = candidate, [dx, dy], error
        layers[i] = best
        shifts.append(shift)
    result = np.max(layers, axis=0)
    return result, {
        "baseline_l1_sum": float(np.abs(original - target).sum()),
        "oracle_l1_sum": float(np.abs(result - target).sum()),
        "character_shifts": shifts,
        "target_informed_not_generation": True,
        "search": "one left-to-right coordinate pass, integer x/y in [-2,2]",
    }


def image(ink):
    return Image.fromarray(np.uint8(np.clip(1 - ink, 0, 1) * 255)).convert("RGB")


def error_image(base, target):
    rgb = np.full((*base.shape, 3), 255, np.uint8)
    foreground, prediction = target > 0.5, base > 0.5
    rgb[foreground & prediction] = [45, 45, 45]
    rgb[foreground & ~prediction] = [210, 45, 45]
    rgb[~foreground & prediction] = [45, 95, 210]
    return Image.fromarray(rgb)


def panel(index, row, raw, target, base, corrected, oracle, aligned, report):
    canvas = Image.new("RGB", (1100, 790), "white")
    draw = ImageDraw.Draw(canvas)
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(fontpath, 16)
    title = ImageFont.truetype(fontpath, 22)
    draw.text(
        (15, 10),
        f"{index:02d} {row['text']}  |  "
        f"{'complete' if report['complete_conditions'] else 'INCOMPLETE'} original conditions",
        fill="black",
        font=title,
    )
    draw.text(
        (15, 43),
        "Exposed true-condition reconstruction. Zero updates. No independent generation.",
        fill="black",
        font=font,
    )
    source = raw.copy()
    source.thumbnail((1024, 92))
    canvas.paste(source, (20, 78))
    draw.text(
        (20, 174),
        "Original RGB word crop (scaled preview; untouched original saved separately)",
        fill="black",
        font=font,
    )
    pairs = [
        ("True intact-word target (existing normalization)", image(target)),
        ("Frozen v21 true-condition reconstruction", image(base)),
        ("Old v22 odds adapter (historical trained output)", image(corrected)),
        ("Error: red missing / blue extra / black shared", error_image(base, target)),
        ("Target-informed +/-3 odds oracle: NOT generation", image(oracle)),
        ("Target-informed geometry oracle: NOT generation", image(aligned)),
    ]
    for j, (text, bitmap) in enumerate(pairs):
        left, top = 20 + (j % 2) * 550, 210 + (j // 2) * 175
        draw.text((left, top), text, fill="black", font=font)
        canvas.paste(bitmap, (left, top + 26))
    draw.text(
        (20, 746),
        f"L1 {report['l1_sum']:.2f}; endpoint floor {report['endpoint_locked_l1']:.5f}; "
        f"odds3 oracle floor {report['odds3_oracle_l1_floor']:.2f}",
        fill="black",
        font=font,
    )
    return canvas


def summarize(rows):
    additive = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in {"index", "dice"}
    ]
    result = {key: sum(row[key] for row in rows) for key in additive}
    result.update(
        words=len(rows),
        mean_dice=float(np.mean([row["dice"] for row in rows])),
        endpoint_fraction_of_l1=result["endpoint_locked_l1"] / result["l1_sum"],
        odds3_floor_fraction_of_l1=result["odds3_oracle_l1_floor"] / result["l1_sum"],
    )
    result["global_translation_l1"] = sum(row["global_translation"]["l1_sum"] for row in rows)
    result["global_translation_improvement_fraction"] = (
        1 - result["global_translation_l1"] / result["l1_sum"]
    )
    geo = [row["per_character_geometry"] for row in rows if row["per_character_geometry"]]
    if geo:
        result["per_character_geometry_words"] = len(geo)
        result["per_character_geometry_baseline_l1"] = sum(r["baseline_l1_sum"] for r in geo)
        result["per_character_geometry_oracle_l1"] = sum(r["oracle_l1_sum"] for r in geo)
        result["per_character_geometry_improvement_fraction"] = (
            1
            - result["per_character_geometry_oracle_l1"]
            / result["per_character_geometry_baseline_l1"]
        )
    return result


def run_preflight(audit=AUDIT, model=MODEL, context=CONTEXT):
    for path in [audit, model, context]:
        verify_run(path)
    request = read(audit / "reports/request.json")
    assert Path(request["model"]).resolve() == model.resolve()
    assert request["checkpoint_sha256"] == sha256(Path(request["checkpoint"]))
    sources = [
        audit / "run.json",
        model / "run.json",
        context / "run.json",
        audit / "solutions/original-words.npz",
        audit / "reports/reconstructions.npz",
        audit / "reports/samples.json",
        model / "reports/targets.npz",
        model / "reports/outputs.npz",
        context / "reports/reconstructions.npz",
        ROOT / "workspace/latest.json",
        ROOT / "research/corpus/word_context_model.py",
        ROOT / "research/corpus/canonical_ink_composer.py",
        Path(request["checkpoint"]),
    ]
    protected = {str(path.resolve()): sha256(path) for path in sources}
    run = Run.create(
        "context-mechanism-preflight",
        config={
            "purpose": "zero-update mechanism diagnosis of exposed reconstruction",
            "source_hashes": protected,
            "binary_threshold": 0.5,
            "near_stroke_radii": [2, 4],
            "translation_search_radius": 2,
            "odds_delta_bound": 3,
            "new_neural_updates": 0,
            "new_neural_inference": 0,
            "sealed_C_used": False,
        },
    )
    shutil.copyfile(__file__, run.path / "logs/context_mechanism_preflight.py")
    for name in ["word_context_model.py", "canonical_ink_composer.py"]:
        shutil.copyfile(ROOT / "research/corpus" / name, run.path / "logs" / name)
    try:
        with np.load(audit / "solutions/original-words.npz") as data:
            arrays = {key: data[key] for key in data.files}
        base = np.load(audit / "reports/reconstructions.npz")["ink"].astype(np.float64)
        previous = np.load(context / "reports/reconstructions.npz")["ink"]
        local = np.load(model / "reports/outputs.npz")["ink"]
        with np.load(model / "reports/targets.npz") as data:
            targets = {key: data[key] for key in data.files}
        lookup = {
            int(value): key
            for key, value in enumerate(targets["source_indices"])
            if targets["real"][key]
        }
        rows = read(audit / "reports/samples.json")
        assert len(rows) == len(base) == 18
        reports, pages, previews, source_records = [], [], [], []
        for i, row in enumerate(rows):
            assert row["source_hash"] != SEALED and row["split"] == "train"
            crop = Path(row["crop"])
            assert sha256(crop) == row["crop_sha256"]
            protected[str(crop)] = row["crop_sha256"]
            raw = Image.open(crop).convert("RGB")
            shutil.copyfile(crop, run.path / f"solutions/original-rgb-{i:02d}.png")
            target = arrays["ink"][i].astype(np.float64)
            boxes = arrays["boxes"][i, : len(row["text"])].astype(np.float32)
            report, oracle = measure(base[i], target, boxes)
            complete = len(row["text"]) == len(row["local_conditions_from_exact_source"])
            known_positions = {
                item["position"] for item in row["local_conditions_from_exact_source"]
            }
            missing_positions = sorted(set(range(len(row["text"]))) - known_positions)
            _, _, character_masks = regions(boxes, base[i].shape)
            missing_mask = np.zeros(base[i].shape, bool)
            for position in missing_positions:
                missing_mask |= character_masks[position]
            report["missing_conditions"] = [
                {"position": position, "character": row["text"][position]}
                for position in missing_positions
            ]
            report["missing_condition_box_l1_sum"] = float(
                np.abs(base[i] - target)[missing_mask].sum()
            )
            report["missing_condition_box_fn_pixels"] = int(
                ((target > 0.5) & (base[i] <= 0.5) & missing_mask).sum()
            )
            if missing_positions:
                assert np.all(arrays["style"][i, missing_positions] == 0)
            shifted, global_result = best_translation(base[i], target)
            geo = None
            if complete:
                layers = []
                for item in row["local_conditions_from_exact_source"]:
                    position, key = item["position"], lookup[item["glyph_index"]]
                    assert np.allclose(arrays["style"][i, position], targets["codes"][key])
                    layers.append(layer(local[key], boxes[position]))
                recomposed = np.max(layers, axis=0)
                aligned, geo = geometry_oracle(np.asarray(layers), target)
                geo["recomposition_mae_vs_saved_audit"] = float(np.abs(recomposed - base[i]).mean())
                geo["recomposition_max_abs_vs_saved_audit"] = float(
                    np.abs(recomposed - base[i]).max()
                )
                geo["recomposition_binary_disagreements"] = int(
                    ((recomposed > 0.5) != (base[i] > 0.5)).sum()
                )
                assert geo["recomposition_mae_vs_saved_audit"] < 0.0001
            else:
                aligned = shifted
            assert np.array_equal(previous[i][base[i] == 0], base[i][base[i] == 0])
            report.update(
                index=i,
                text=row["text"],
                complete_conditions=complete,
                source_hash=row["source_hash"],
                global_translation=global_result,
                per_character_geometry=geo,
                historical_adapter_l1_sum=float(np.abs(previous[i] - target).sum()),
            )
            threshold_pairs = {}
            for threshold in [0.25, 0.75]:
                foreground = target > threshold
                cc = components(foreground)
                _, _, masks = regions(boxes, base[i].shape)
                found = []
                for j in range(len(masks) - 1):
                    left = Counter(cc[foreground & masks[j] & ~masks[j + 1]].tolist())
                    right = Counter(cc[foreground & masks[j + 1] & ~masks[j]].tolist())
                    if any(
                        k and left[k] >= 5 and right[k] >= 5 for k in left.keys() & right.keys()
                    ):
                        found.append(j)
                threshold_pairs[str(threshold)] = found
            report["connectivity_threshold_sensitivity"] = threshold_pairs
            for pair in report["connected_pair_candidates"]:
                pair["text"] = row["text"][pair["left_position"] : pair["right_position"] + 1]
            reports.append(report)
            source_records.append(
                {
                    "index": i,
                    "text": row["text"],
                    "original": row,
                    "known_local_positions": len(row["local_conditions_from_exact_source"]),
                    "no_new_label_review": True,
                }
            )
            page = panel(i, row, raw, target, base[i], previous[i], oracle, aligned, report)
            destination = run.path / f"comparisons/word-{i:02d}.png"
            page.save(destination)
            pages.append(page)
            thumb = page.crop((0, 0, 1100, 549))
            thumb.thumbnail((550, 275))
            previews.append(thumb)
            np.savez_compressed(
                run.path / f"reports/diagnostic-{i:02d}.npz",
                target=target,
                base=base[i],
                odds_oracle=oracle,
                geometry_oracle=aligned,
            )
        montage = Image.new("RGB", (1650, 6 * 275), "white")
        for i, thumb in enumerate(previews):
            montage.paste(thumb, ((i % 3) * 550, (i // 3) * 275))
        montage.save(run.path / "comparisons/all-18-overview.png")
        pages[0].save(
            run.path / "comparisons/mechanism-diagnosis.pdf",
            save_all=True,
            append_images=pages[1:],
            resolution=150,
        )
        atomic_json(run.path / "reports/words.json", reports)
        atomic_json(run.path / "reports/original-sources.json", source_records)
        atomic_json(run.path / "reports/protected.json", protected)
        summary = {
            "all18": summarize(reports),
            "complete13": summarize([row for row in reports if row["complete_conditions"]]),
            "incomplete5": summarize([row for row in reports if not row["complete_conditions"]]),
            "new_neural_updates": 0,
            "new_neural_inference": 0,
            "independent_generation_evaluated": False,
            "sealed_C_used": False,
            "oracles_use_actual_target_and_are_not_models": True,
            "training_admission": "held pending mechanism and visual review",
        }
        for path, expected in protected.items():
            assert sha256(Path(path)) == expected
        run.finish("held", result=summary)
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--context", type=Path, default=CONTEXT)
    args = parser.parse_args()
    print(run_preflight(args.audit, args.model, args.context))


if __name__ == "__main__":
    main()
