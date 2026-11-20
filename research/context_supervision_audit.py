"""Bounded, append-only visual review of already registered original handwriting.

prepare creates source inventories and RGB contact sheets. finish requires explicit
per-candidate Agent review; text sequences and pixel thresholds never approve joins.
No training, OCR, relabeling, or access to sealed data is performed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "workspace/runs/20261120T103956559524Z-federated-corpus-registry-097d0533"
OLD_WORDS = (
    ROOT
    / "workspace/runs/20261120T023639068683Z-canonical-intact-word-audit-627f2eae"
    / "reports/samples.json"
)
SEQUENCES = ("th", "st", "ll", "ing", "ff", "fi", "ft", "ti")
FORBIDDEN = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
DECISIONS = {"cross-letter-ink", "adjacent-unconnected", "crop-or-label-uncertain"}


def read(path):
    return json.loads(Path(path).read_text())


def occurrences(text, part):
    return [i for i in range(len(text) - len(part) + 1) if text[i : i + len(part)].lower() == part]


def inventory(registry):
    data = read(registry / "reports/registry.json")
    manifests = {
        key: read(registry / "solutions/manifests" / (item["sha256"] + ".json"))
        for key, item in data["manifests"].items()
    }
    views = {
        value["observation"]: value
        for value in data["views"].values()
        if value["kind"] in {"original-word", "original-line"}
    }
    old = read(OLD_WORDS)
    old18 = {row["crop_sha256"] for row in old}
    old13 = {
        row["crop_sha256"]
        for row in old
        if len(row["local_conditions_from_exact_source"]) == len(row["text"])
    }
    result = {"words": [], "lines": []}
    for observation_id, observation in data["observations"].items():
        if observation["kind"] not in {"english-word", "english-line"}:
            continue
        word = observation["kind"] == "english-word"
        collection = "words" if word else "lines"
        view = views[observation_id]
        source_records = []
        source_path = None
        for reference in view["source_records"]:
            original = manifests[reference["manifest"]][reference["row"]]
            source_records.append({"reference": reference, "original_record": original})
            if "word_position" in reference:
                match = [
                    row
                    for row in original["training_word_links"]
                    if row["position"] == reference["word_position"]
                ]
                assert len(match) == 1
                path = match[0]["crop"]
            else:
                path = original["crop"] if word else original["source_crop"]
            if sha256(Path(path)) == view["asset_sha256"]:
                source_path = path
        assert source_path is not None
        document = data["documents"][observation["document"]]
        assert document["source_sha256"] != FORBIDDEN
        label = observation["label"]
        row = {
            "id": f"{'w' if word else 'l'}{len(result[collection]):03d}",
            "observation_id": observation_id,
            "label": label,
            "group_id": observation["group_id"],
            "document": document,
            "source_sha256": document["source_sha256"],
            "page": observation["page"],
            "box_inches": observation["box_inches"],
            "geometry": view["geometry"],
            "source_path": source_path,
            "crop_sha256": view["asset_sha256"],
            "source_records": source_records,
            "sequence_positions": {seq: occurrences(label, seq) for seq in SEQUENCES},
            "adjacent_letter_positions": [
                {"position": i, "pair": label[i : i + 2]}
                for i in range(len(label) - 1)
                if label[i].isalpha() and label[i + 1].isalpha()
            ],
            "fit_positions": [
                {
                    "position": i,
                    "character": ch,
                    "region_position": "initial"
                    if i == 0
                    else "final"
                    if i == len(label) - 1
                    else "medial",
                }
                for i, ch in enumerate(label)
                if ch.lower() in "fit"
            ],
            "position_scope": "within-word" if word else "within-line",
            "old18_reconstruction": view["asset_sha256"] in old18,
            "old13_context_training": view["asset_sha256"] in old13,
            "backbone_document_exposed": True,
            "character_boxes_inferred": False,
        }
        result[collection].append(row)
    assert len(result["words"]) == 174 and len(result["lines"]) == 39
    return result


def coverage(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)

    def summarize(items):
        return {
            "position_scope": items[0]["position_scope"] if items else None,
            "regions": len(items),
            "unique_labels": len({row["label"] for row in items}),
            "sequence_occurrences": {
                seq: sum(len(row["sequence_positions"][seq]) for row in items) for seq in SEQUENCES
            },
            "fit_positions": dict(
                Counter(
                    f"{item['character'].lower()}:{item['region_position']}"
                    for row in items
                    for item in row["fit_positions"]
                )
            ),
            "fit_characters_case_sensitive": dict(
                Counter(item["character"] for row in items for item in row["fit_positions"])
            ),
            "all_adjacent_letter_occurrences": dict(
                sorted(
                    Counter(
                        pair["pair"].lower()
                        for row in items
                        for pair in row["adjacent_letter_positions"]
                    ).items()
                )
            ),
        }

    return {
        "total": summarize(rows),
        "by_group": {key: summarize(items) for key, items in sorted(groups.items())},
    }


def choose(words):
    selected = [
        row
        for row in words
        if "f" in row["label"].lower()
        or any(row["sequence_positions"][seq] for seq in ("st", "ll", "ing"))
    ]
    seen = set()
    for row in words:
        key = (row["group_id"], row["label"].lower())
        if row not in selected and row["sequence_positions"]["th"] and key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) >= 40:
            break
    if len(selected) < 40:
        for row in words:
            if row not in selected and row["label"].lower() == "it":
                selected.append(row)
                break
    assert len(selected) <= 40
    return selected


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def contact(rows, output, title, *, line_mode=False, detail=False):
    columns, per_page = (1, 4) if line_mode else (4, 20)
    cell_width, cell_height = (1900, 270) if line_mode else (550, 240)
    if detail:
        columns, per_page, cell_width, cell_height = 2, 8, 900, 340
    pages = []
    for start in range(0, len(rows), per_page):
        subset = rows[start : start + per_page]
        canvas = Image.new(
            "RGB", (columns * cell_width, (per_page // columns) * cell_height + 70), "white"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 10), title + f" / {start // per_page + 1}", fill="black", font=font(23))
        draw.text(
            (15, 40),
            "Original RGB previews; source labels unchanged; no automatic join approval.",
            fill="black",
            font=font(16),
        )
        for j, row in enumerate(subset):
            x, y = (j % columns) * cell_width, 70 + (j // columns) * cell_height
            draw.text((x + 10, y + 5), f"{row['id']}  {row['label']}", fill="black", font=font(18))
            draw.text(
                (x + 10, y + 30),
                f"{row['source_sha256'][:8]} p{row['page']} | "
                f"old13={row['old13_context_training']}",
                fill="black",
                font=font(13),
            )
            with Image.open(row["source_path"]) as source:
                image = source.convert("RGB")
            max_width, max_height = cell_width - 25, cell_height - 95
            scale = min(max_width / image.width, max_height / image.height, 6 if detail else 3)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.NEAREST,
            )
            canvas.paste(image, (x + 10, y + 58))
            families = [seq for seq in SEQUENCES if row["sequence_positions"][seq]]
            draw.text(
                (x + 10, y + cell_height - 25),
                "text sequences: " + ", ".join(families),
                fill="black",
                font=font(14),
            )
        path = output / f"sheet-{start // per_page + 1:02d}.png"
        canvas.save(path)
        pages.append(canvas)
    pages[0].save(output / "contact.pdf", save_all=True, append_images=pages[1:], resolution=150)


def prepare(registry=REGISTRY):
    verify_run(registry)
    protected = {
        str(registry / "run.json"): sha256(registry / "run.json"),
        str(ROOT / "workspace/latest.json"): sha256(ROOT / "workspace/latest.json"),
        str(OLD_WORDS): sha256(OLD_WORDS),
    }
    run = Run.create(
        "context-supervision-audit",
        config={
            "registry": str(registry),
            "source_hashes": protected,
            "candidate_limit": 40,
            "new_neural_updates": 0,
            "new_neural_inference": 0,
            "sealed_C_used": False,
            "original_labels_modified": False,
            "selection": "all f, st/ll/ing; group+text th; first other it; no image scoring",
            "gate": {
                "reliable_context_words": 20,
                "source_groups": 3,
                "context_families_in_multiple_groups": 2,
            },
        },
    )
    shutil.copyfile(__file__, run.path / "logs/context_supervision_audit.py")
    try:
        data = inventory(registry)
        for collection, rows in data.items():
            folder = run.path / "solutions" / collection
            folder.mkdir()
            for row in rows:
                destination = folder / (row["id"] + "-original.png")
                shutil.copyfile(row["source_path"], destination)
                assert sha256(destination) == row["crop_sha256"]
                protected[row["source_path"]] = row["crop_sha256"]
            output = run.path / "comparisons" / collection
            output.mkdir()
            contact(
                rows,
                output,
                f"All registered original {collection}",
                line_mode=collection == "lines",
            )
        candidates = choose(data["words"])
        output = run.path / "comparisons/candidates"
        output.mkdir()
        contact(candidates, output, "Fixed bounded contextual-ink candidates")
        details = output / "detail"
        details.mkdir()
        contact(candidates, details, "Candidate detail for explicit visual review", detail=True)
        atomic_json(run.path / "reports/inventory.json", data)
        atomic_json(
            run.path / "reports/coverage.json", {key: coverage(rows) for key, rows in data.items()}
        )
        repeated = defaultdict(list)
        for row in data["words"]:
            repeated[row["label"].lower()].append(row)
        atomic_json(
            run.path / "reports/repeated-content.json",
            [
                {
                    "casefolded_text_for_analysis_only": text,
                    "source_groups": sorted({row["group_id"] for row in rows}),
                    "word_ids": [row["id"] for row in rows],
                }
                for text, rows in repeated.items()
                if len({row["group_id"] for row in rows}) > 1
            ],
        )
        atomic_json(run.path / "reports/candidates.json", candidates)
        atomic_json(run.path / "reports/protected.json", protected)
        atomic_json(
            run.path / "reports/review-template.json",
            [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "decision": "pending",
                    "pair_observations": [],
                    "label_visual_status": "pending",
                    "note": "",
                }
                for row in candidates
            ],
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def finish(path, reviews_path):
    run = Run(path)
    assert run.record()["status"] == "running"
    candidates = read(path / "reports/candidates.json")
    submission = read(reviews_path)
    reviews = submission["reviews"]
    assessment = submission["task_assessment"]
    if assessment.get("decision") not in {
        "no-go",
        "data-ready-for-protocol-only",
    } or not assessment.get("reason"):
        raise ValueError("An explicit task assessment is required; counts never approve training")
    expected = {row["id"]: row for row in candidates}
    if len(reviews) != len(expected) or {row["id"] for row in reviews} != set(expected):
        raise ValueError("Exactly one explicit review per fixed candidate is required")
    usable = []
    families = defaultdict(set)
    for row in reviews:
        source = expected[row["id"]]
        if row["label"] != source["label"] or row["decision"] not in DECISIONS or not row["note"]:
            raise ValueError(
                "Source labels cannot be changed; explicit visual decision and note required"
            )
        if row["label_visual_status"] not in {"consistent-with-existing-label", "uncertain"}:
            raise ValueError("An explicit label consistency check is required")
        for pair in row["pair_observations"]:
            start, end = pair["positions"]
            if (
                not 0 <= start < end <= len(row["label"])
                or row["label"][start:end].lower() != pair["sequence"]
            ):
                raise ValueError("Pair positions must preserve the actual original label")
            if pair["status"] not in DECISIONS or not pair["note"]:
                raise ValueError("Explicit pair review is required")
        admitted = [p for p in row["pair_observations"] if p["status"] == "cross-letter-ink"]
        if row["decision"] == "cross-letter-ink" and not admitted:
            raise ValueError("Cross-letter classification needs a concrete reviewed span")
        if (
            row["decision"] == "cross-letter-ink"
            and row["label_visual_status"] == "consistent-with-existing-label"
        ):
            usable.append(source)
            for pair in admitted:
                families[pair["sequence"]].add(source["group_id"])
        row.update(
            reviewer="Codex Agent visual audit",
            human_author_confirmation=False,
            original_review_unchanged=True,
            no_new_transcription=True,
        )
    groups = {row["group_id"] for row in usable}
    replicated = sum(len(value) >= 2 for value in families.values())
    ready = len(usable) >= 20 and len(groups) >= 3 and replicated >= 2
    protected = read(path / "reports/protected.json")
    for source, digest in protected.items():
        if sha256(Path(source)) != digest:
            raise ValueError("Protected source changed during review")
    shutil.copyfile(reviews_path, path / "reports/submitted-review.json")
    shutil.copyfile(__file__, path / "logs/context_supervision_audit_finish.py")
    atomic_json(path / "reports/visual-review.json", reviews)
    atomic_json(path / "reports/task-assessment.json", assessment)
    summary = {
        "words_in_inventory": 174,
        "lines_in_inventory": 39,
        "reviewed_candidates": len(reviews),
        "decision_counts": dict(Counter(r["decision"] for r in reviews)),
        "reliable_context_words": len(usable),
        "context_source_groups": sorted(groups),
        "context_families": {key: sorted(value) for key, value in families.items()},
        "families_present_in_multiple_groups": replicated,
        "heuristic_count_reference_met": ready,
        "count_reference_is_not_a_scientific_boundary": True,
        "H_context_task_gate": assessment["decision"],
        "task_assessment": assessment,
        "whole_word_supervision_remains_valid": True,
        "independent_holdout_claimed": False,
        "new_neural_updates": 0,
        "new_neural_inference": 0,
        "sealed_C_used": False,
        "original_labels_modified": False,
    }
    run.finish("held", result=summary)
    verify_run(path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--registry", type=Path, default=REGISTRY)
    p = sub.add_parser("finish")
    p.add_argument("run", type=Path)
    p.add_argument("reviews", type=Path)
    args = parser.parse_args()
    print(
        prepare(args.registry)
        if args.mode == "prepare"
        else json.dumps(finish(args.run, args.reviews), indent=2)
    )


if __name__ == "__main__":
    main()
