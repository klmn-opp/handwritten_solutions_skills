"""One author-requested +6% visible word-gap candidate, with frozen-content evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw

from hwsol.jobs.handwriting import HandwritingLayoutPlanner
from hwsol.jobs.runtime import render_job, verify_job
from hwsol.personal_ink.bundle import load_recipe
from hwsol.personal_ink.runtime import PersonalCpuEngine
from hwsol.personal_ink.validation import SEMANTIC_FIELDS, capture
from hwsol.profiles.io import atomic_json, sha256
from hwsol.profiles.store import ProfileStore, load_snapshot
from hwsol.workspace import Run, verify_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "workspace/runs"
PARENT = RUNS / "20261120T154816694269Z-m22-universal-private-v3-018681e8"
INPUT = RUNS / "20261120T154856740364Z-m21-explicit-source-recovery-ab6496e5"
REFERENCE = RUNS / "20261120T155718248776Z-m21-clean-installed-new-job-43bb9000"
FEEDBACK = "每个单词的间隔应稍微大一点点，其他整体满意"
SCALE = 1.06


def read(path):
    return json.loads(path.read_text())


def prepare():
    verify_run(PARENT)
    run = Run.create(
        "m22-word-gap-v4-assets",
        seed=1909,
        config={
            "parent": str(PARENT),
            "parent_sha256": sha256(PARENT / "run.json"),
            "author_feedback": FEEDBACK,
            "word_gap_scale": SCALE,
            "single_preselected_amount": True,
            "neural_updates": 0,
            "author_approved_prior_output_except_spacing": True,
            "revised_spacing_author_approval": False,
        },
    )
    print(run.path, flush=True)
    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        folder = run.path / "solutions/profile"
        shutil.copytree(PARENT / "solutions/profile", folder)
        manifest = read(folder / "profile.json")
        original_assets = copy.deepcopy(manifest["assets"])
        manifest.update(
            version="v4",
            parent="personal-cpu-candidate@v3",
            created_at=datetime.now(UTC).isoformat(),
            description="Private +6% visible English word-gap candidate; same v3 ink assets",
        )
        manifest["render"]["word_gap_scale"] = SCALE
        manifest["build"]["word_gap_revision"] = run.record()["config"]
        atomic_json(folder / "profile.json", manifest)
        assert manifest["assets"] == original_assets
        for item in original_assets:
            assert sha256(folder / item["path"]) == item["sha256"]
        loaded = load_snapshot(folder)
        load_recipe(loaded)
        archive = run.path / "solutions/personal-cpu-candidate-v4.zip"
        with ZipFile(archive, "x", ZIP_DEFLATED) as target:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    target.write(path, path.relative_to(folder).as_posix())
        run.finish(
            "prepared",
            result={
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "profile": loaded.manifest.ref,
                "profile_sha256": loaded.manifest_sha256,
                "all_assets_exact_to_v3": True,
                "assets": len(original_assets),
                "source_glyphs_added": 0,
                "production_activation": False,
            },
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def source_identity(row):
    result = {
        key: row.get(key)
        for key in ("source_type", "actual_glyph_id", "glyph_name", "variant", "font_sha256")
    }
    for name in ("scan_source", "session_source", "session_o_source"):
        if isinstance(row.get(name), dict):
            result[name] = {
                key: row[name].get(key) for key in ("id", "source_id", "crop_sha256", "character")
            }
    return result


def word_boxes(lines, records):
    result = []
    for line_number, line in enumerate(lines):
        text = [r for r in records[line["start"] : line["end"]] if r["context"] == "text"]
        offset = 0
        previous = None
        separated = False
        for kind, value, _, width in line["runs"]:
            if kind != "t":
                previous = None
                separated = False
                continue
            if value.isspace():
                separated = True
                continue
            rows = text[offset : offset + len(value)]
            assert "".join(r["semantic"] for r in rows) == value
            offset += len(value)
            boxes = np.asarray([r["bounds_px"] for r in rows])
            box = [
                float(boxes[:, 0].min()),
                float(boxes[:, 1].min()),
                float(boxes[:, 2].max()),
                float(boxes[:, 3].max()),
            ]
            word = {
                "text": value,
                "first": rows[0]["item_index"],
                "last": rows[-1]["item_index"],
                "line": line_number,
                "bounds": box,
                "advance_px": width,
            }
            if previous is not None and not separated:
                # Handnote separates punctuation tokens; they belong to the same
                # word envelope, not a newly measured inter-word gap.
                previous["text"] += value
                previous["last"] = word["last"]
                previous["bounds"] = [
                    min(previous["bounds"][0], box[0]),
                    min(previous["bounds"][1], box[1]),
                    max(previous["bounds"][2], box[2]),
                    max(previous["bounds"][3], box[3]),
                ]
                previous["advance_px"] += width
                continue
            if previous is not None:
                result.append(
                    {
                        "key": f"{previous['last']}:{word['first']}",
                        "left": previous,
                        "right": word,
                        "visible_gap_px": box[0] - previous["bounds"][2],
                    }
                )
            previous = word
            separated = False
        assert offset == len(text)
    return result


def compare(assets):
    for parent in (assets, PARENT, INPUT, REFERENCE):
        verify_run(parent)
    run = Run.create(
        "m22-word-gap-full-four",
        seed=1909,
        config={
            "assets": str(assets),
            "asset_run_sha256": sha256(assets / "run.json"),
            "input": str(INPUT),
            "input_sha256": sha256(INPUT / "run.json"),
            "reference": str(REFERENCE),
            "author_feedback": FEEDBACK,
            "word_gap_scale": SCALE,
            "neural_updates": 0,
            "method": "normal render_job plus observation-only line/path capture",
            "installed_wheel_claim": False,
            "author_style_approval": False,
        },
    )
    print(run.path, flush=True)
    old_compose, old_draw = PersonalCpuEngine.compose, HandwritingLayoutPlanner.draw_line
    observations, lines = [], []

    def draw(planner, line, top, state):
        start = len(planner.renderer._rec)
        result = old_draw(planner, line, top, state)
        lines.append(
            {"start": start, "end": len(planner.renderer._rec), "runs": copy.deepcopy(line["runs"])}
        )
        return result

    def compose(engine, source, title, cfg):
        lines.clear()
        result = old_compose(engine, source, title, cfg)
        page, renderer, records, _ = result
        observations.append(
            {
                "state": capture(page, renderer, records),
                "word_gaps": word_boxes(lines, records),
                "text_lines": len(lines),
            }
        )
        return result

    try:
        shutil.copyfile(__file__, run.path / "logs/source.py")
        for name in ("config.py", "jobs/handwriting.py"):
            target = run.path / "logs/runtime" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / "src/hwsol" / name, target)
        store = ProfileStore(run.path / "solutions/store")
        banks = {}
        for mode, origin in [("v3", PARENT), ("v4", assets)]:
            imported = store.import_archive(Path(read(origin / "reports/result.json")["archive"]))
            profile = store.resolve(imported["imported"], candidate=True)
            folder = run.path / f"solutions/{mode}"
            shutil.copytree(INPUT / "solutions", folder)
            job = read(folder / "job.json")
            job.update(profile=profile.manifest.ref, profile_sha256=profile.manifest_sha256)
            atomic_json(folder / "job.json", job)
            for problem in job["problems"]:
                assert sha256(folder / problem["solution"]) == sha256(
                    INPUT / "solutions" / problem["solution"]
                )
            observations.clear()
            PersonalCpuEngine.compose = compose
            HandwritingLayoutPlanner.draw_line = draw
            render_job(folder / "job.json", run.path / f"rendered/{mode}", store=store)
            PersonalCpuEngine.compose, HandwritingLayoutPlanner.draw_line = old_compose, old_draw
            verification = verify_job(run.path / f"rendered/{mode}", pixels=True)
            atomic_json(run.path / f"reports/{mode}-verification.json", verification)
            banks[mode] = list(observations)
            atomic_json(
                run.path / f"reports/{mode}-gaps.json",
                [
                    {"text_lines": o["text_lines"], "word_gaps": o["word_gaps"]}
                    for o in observations
                ],
            )
        reports = []
        comparison_images = []
        validations = {
            mode: read(run.path / f"rendered/{mode}/validation_report.json")["pages"]
            for mode in ("v3", "v4")
        }
        for index, (before, after) in enumerate(zip(banks["v3"], banks["v4"], strict=True), 1):
            a, b = before["state"], after["state"]
            assert len(a.records) == len(b.records)
            assert [[r.get(k) for k in SEMANTIC_FIELDS] for r in a.records] == [
                [r.get(k) for k in SEMANTIC_FIELDS] for r in b.records
            ]
            source_changes = [
                i
                for i, (x, y) in enumerate(zip(a.records, b.records, strict=True))
                if source_identity(x) != source_identity(y)
            ]
            shape_changes = []
            for i, (x, y) in enumerate(zip(a.paths, b.paths, strict=True)):
                if x.vertices.shape != y.vertices.shape or not np.array_equal(x.codes, y.codes):
                    shape_changes.append({"item": i, "reason": "path topology differs"})
                    continue
                delta = y.vertices - x.vertices
                deviation = float(np.max(np.abs(delta - np.median(delta, axis=0))))
                if deviation > 0.002:
                    shape_changes.append({"item": i, "max_nontranslation_px": deviation})
            assert not source_changes and not shape_changes
            before_page, after_page = (validations[mode][index - 1] for mode in ("v3", "v4"))
            for field, keys in (
                ("formula_groups", ("latex", "geometry_sha256", "items")),
                (
                    "matrix_groups",
                    ("environment", "prefix", "rows", "items", "cell_and_matrix_size_px"),
                ),
            ):
                assert [[g[k] for k in keys] for g in before_page[field]] == [
                    [g[k] for k in keys] for g in after_page[field]
                ]
            old_gaps = {g["key"]: g for g in before["word_gaps"]}
            new_gaps = {g["key"]: g for g in after["word_gaps"]}
            shared = sorted(old_gaps.keys() & new_gaps.keys())
            gap_rows = [
                {
                    "key": key,
                    "before": old_gaps[key]["visible_gap_px"],
                    "after": new_gaps[key]["visible_gap_px"],
                    "delta": new_gaps[key]["visible_gap_px"] - old_gaps[key]["visible_gap_px"],
                }
                for key in shared
            ]
            left = Image.open(run.path / f"rendered/v3/pages/{index:02d}.png").convert("RGB")
            original = Image.open(
                REFERENCE / f"rendered/python-3.12/candidate/pages/{index:02d}.png"
            ).convert("RGB")
            assert left.size == original.size and left.tobytes() == original.tobytes()
            right = Image.open(run.path / f"rendered/v4/pages/{index:02d}.png").convert("RGB")
            # Same first two complete prose lines on each page; includes title separately.
            selected = sorted(shared, key=lambda k: old_gaps[k]["left"]["line"])
            line_numbers = sorted(set(old_gaps[k]["left"]["line"] for k in selected))[:3]
            for line_number in line_numbers:
                entries = [
                    old_gaps[k] for k in selected if old_gaps[k]["left"]["line"] == line_number
                ]
                if not entries:
                    continue
                ids = [g["left"]["first"] for g in entries] + [g["right"]["last"] for g in entries]
                box_old = np.array(
                    [a.records[i]["bounds_px"] for i in range(min(ids), max(ids) + 1)]
                )
                box_new = np.array(
                    [b.records[i]["bounds_px"] for i in range(min(ids), max(ids) + 1)]
                )
                crop = [
                    max(0, int(min(box_old[:, 0].min(), box_new[:, 0].min())) - 12),
                    max(0, int(min(box_old[:, 1].min(), box_new[:, 1].min())) - 10),
                    min(left.width, int(max(box_old[:, 2].max(), box_new[:, 2].max())) + 12),
                    min(left.height, int(max(box_old[:, 3].max(), box_new[:, 3].max())) + 10),
                ]
                ca, cb = left.crop(crop), right.crop(crop)
                panel = Image.new(
                    "RGB", (max(ca.width, cb.width), ca.height + cb.height + 44), "white"
                )
                pen = ImageDraw.Draw(panel)
                pen.text(
                    (2, 2), f"Page {index}, line {line_number}: v3 accepted baseline", fill="black"
                )
                panel.paste(ca, (0, 20))
                pen.text((2, ca.height + 23), "v4: +6% visible inter-word target", fill="black")
                panel.paste(cb, (0, ca.height + 44))
                panel.save(run.path / f"comparisons/page-{index:02d}-line-{line_number:02d}.png")
                comparison_images.append(panel)
            reports.append(
                {
                    "page": index,
                    "visible_items": len(a.records),
                    "semantic_identity_exact": True,
                    "formula_parser_local_geometry_exact": True,
                    "matrix_structure_local_sizes_exact": True,
                    "source_changes": source_changes,
                    "shape_changes": shape_changes,
                    "text_lines_before_after": [before["text_lines"], after["text_lines"]],
                    "same_line_gap_pairs": len(shared),
                    "gaps": gap_rows,
                    "reflowed_old_pairs": sorted(old_gaps.keys() - new_gaps.keys()),
                    "reflowed_new_pairs": sorted(new_gaps.keys() - old_gaps.keys()),
                    "baseline_pixels_exact_to_43bb9000": True,
                }
            )
        if comparison_images:
            comparison_images[0].save(
                run.path / "comparisons/word-gap-before-after.pdf",
                save_all=True,
                append_images=comparison_images[1:],
                resolution=150,
            )
        atomic_json(run.path / "reports/page-comparison.json", reports)
        validation = read(run.path / "rendered/v4/validation_report.json")
        summary = {
            "pages": len(reports),
            "word_gap_scale": SCALE,
            "semantic_identity_all_exact": True,
            "source_changes": sum(len(r["source_changes"]) for r in reports),
            "shape_changes": sum(len(r["shape_changes"]) for r in reports),
            "text_lines_before_after": [r["text_lines_before_after"] for r in reports],
            "font_fallbacks": sum(len(p["math_fallback_glyphs"]) for p in validation["pages"]),
            "author_style_approval": False,
            "production_go": False,
            "all_font_ink_assets_exact": True,
            "source_inputs_unchanged": True,
        }
        run.finish("held", result=summary)
        verify_run(run.path)
        print(json.dumps({"run": str(run.path), **summary}))
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise
    finally:
        PersonalCpuEngine.compose, HandwritingLayoutPlanner.draw_line = old_compose, old_draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "compare"])
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.assets:
            parser.error("--assets required")
        compare(args.assets.resolve())


if __name__ == "__main__":
    main()
