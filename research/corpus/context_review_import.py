"""Bind human review exports to immutable images without promoting models or training data."""

import argparse
import hashlib
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import fmean

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create

FIELDS = ("naturalness", "similarity", "consistency", "connections", "neatness")


def strict_json(text: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant: " + value)

    def unique(items: list) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def validate_review(value: dict, review_id: str, candidate_ids: set[str]) -> dict:
    if value.get("schema") != "context-human-review/1" or value.get("review_id") != review_id:
        raise ValueError("Review schema or identity differs")
    ratings = value.get("ratings")
    if not isinstance(ratings, dict) or set(ratings) - candidate_ids:
        raise ValueError("Unknown or malformed candidate ratings")
    clean = {}
    for key, item in ratings.items():
        if not isinstance(item, dict) or set(item) - {"content", "notes", "scores"}:
            raise ValueError("Invalid rating record")
        content = item.get("content")
        if content not in ("correct", "incorrect", "uncertain", None):
            raise ValueError("Invalid content decision")
        notes, scores = item.get("notes", ""), item.get("scores", {})
        if not isinstance(notes, str) or not isinstance(scores, dict) or set(scores) - set(FIELDS):
            raise ValueError("Invalid score fields or notes")
        for score in scores.values():
            if score is not None and (type(score) is not int or not 1 <= score <= 5):
                raise ValueError("Scores must be integer 1–5 or null")
        if content != "correct" and any(v is not None for v in scores.values()):
            raise ValueError("Style scores require confirmed correct content")
        clean[key] = {
            "content": content,
            "notes": notes,
            "scores": scores if content == "correct" else {},
        }
    return {**value, "ratings": clean}


def summarize(rows: list[dict]) -> tuple[dict, list[dict]]:
    methods = {}
    for method in sorted({r["method"] for r in rows}):
        items = [r for r in rows if r["method"] == method]
        scores = {}
        for field in FIELDS:
            values = [
                r["scores"][field]
                for r in items
                if r["content"] == "correct" and r["scores"].get(field) is not None
            ]
            scores[field] = {"n": len(values), "mean": fmean(values) if values else None}
        positions = dict(Counter(r["position"] for r in items))
        methods[method] = {
            "candidates": len(items),
            "content": dict(Counter(r["content"] or "unreviewed" for r in items)),
            "scores": scores,
            "positions": positions,
            "same_position_all_groups": len(items) > 1 and len(positions) == 1,
        }
    pairs = []
    for group in sorted({r["group_id"] for r in rows}):
        items = [r for r in rows if r["group_id"] == group and r["content"] == "correct"]
        for a, b in combinations(items, 2):
            for field in FIELDS:
                left, right = a["scores"].get(field), b["scores"].get(field)
                if left is None or right is None:
                    continue
                pairs.append(
                    {
                        "group_id": group,
                        "dimension": field,
                        "left_id": a["id"],
                        "right_id": b["id"],
                        "left_method": a["method"],
                        "right_method": b["method"],
                        "left_image_sha256": a["image_sha256"],
                        "right_image_sha256": b["image_sha256"],
                        "left_score": left,
                        "right_score": right,
                        "preferred_id": a["id"]
                        if left > right
                        else b["id"]
                        if right > left
                        else None,
                        "tie": left == right,
                        "source_split": a["source_split"],
                        "human": True,
                        "optimization_eligible": False,
                        "scope": "evaluation-and-research-planning-only",
                    }
                )
    return {
        "methods": methods,
        "candidates": len(rows),
        "reviewed": sum(r["content"] is not None for r in rows),
        "content_confirmed": sum(r["content"] == "correct" for r in rows),
        "groups": len({r["group_id"] for r in rows}),
        "dimensions": list(FIELDS),
        "render_content_confirmation_only": True,
        "not_generalization_or_statistical_significance": True,
        "optimization_eligible": False,
    }, pairs


def bind_export(delivery: Path, export_bytes: bytes) -> tuple[dict, list[dict]]:
    verify_run(delivery)
    review = delivery / "comparisons/author-review"
    match = re.search(
        r'<script id="payload" type="application/json">(.*?)</script>',
        (review / "index.html").read_text(),
        re.S,
    )
    if not match:
        raise ValueError("Missing immutable review payload")
    payload = strict_json(match.group(1))
    if payload["id"] != delivery.name:
        raise ValueError("Delivery identity differs")
    raw_key = json.loads((delivery / "reports/author-review-key.json").read_text())
    key = {r["id"]: r for r in raw_key}
    if len(key) != len(raw_key):
        raise ValueError("Duplicate candidate key")
    bindings = {}
    for group in payload["groups"]:
        for candidate in group["candidates"]:
            cid = candidate["id"]
            if cid not in key or cid in bindings:
                raise ValueError("Candidate/key binding differs")
            path = (review / candidate["image"]).resolve()
            if not path.is_relative_to(review.resolve()) or sha256(path) != key[cid]["sha256"]:
                raise ValueError("Review image differs or escapes review folder")
            if group["target"] != key[cid]["text"]:
                raise ValueError("Displayed text/key differs")
            bindings[cid] = {
                "id": cid,
                "group_id": group["id"],
                "position": candidate["label"],
                "method": key[cid]["method"],
                "image": str(path),
                "image_sha256": key[cid]["sha256"],
                "target": group["target"],
                "source_split": key[cid]["split"],
                "row_index": key[cid]["row_index"],
            }
    if set(bindings) != set(key):
        raise ValueError("Incomplete immutable review bindings")
    accepted = validate_review(
        strict_json(export_bytes.decode("utf-8-sig")), delivery.name, set(key)
    )
    rows = [
        {**binding, **accepted["ratings"].get(cid, {"content": None, "notes": "", "scores": {}})}
        for cid, binding in bindings.items()
    ]
    return accepted, rows


def import_review(delivery: Path, export: Path) -> Path:
    delivery, export = delivery.resolve(), export.resolve()
    original = export.read_bytes()
    accepted, rows = bind_export(delivery, original)
    summary, pairs = summarize(rows)
    source_hash = hashlib.sha256(original).hexdigest()
    run = create(
        "context-author-feedback",
        config={
            "delivery": str(delivery),
            "delivery_sha256": sha256(delivery / "run.json"),
            "export": str(export),
            "export_sha256": source_hash,
            "key_sha256": sha256(delivery / "reports/author-review-key.json"),
            "human_submission": True,
            "training_updates": 0,
            "automatic_publication": False,
            "generated_images_are_not_real_training_samples": True,
            "preference_scope": "evaluation-and-research-planning-only",
        },
    )
    try:
        (run.path / "reports/submitted-review.json").write_bytes(original)
        atomic_json(run.path / "reports/validated-review.json", accepted)
        atomic_json(run.path / "reports/ratings.json", rows)
        atomic_json(run.path / "reports/summary.json", summary)
        atomic_json(run.path / "reports/dimension-preferences.json", pairs)
        atomic_json(
            run.path / "reports/receipt.json",
            {
                "export_sha256": source_hash,
                "image_bindings_verified": len(rows),
                "human": True,
                "training_updates": 0,
                "default_changed": False,
                "latest_updated": False,
                "explicitly_not_scan_labels_or_neural_training": True,
            },
        )
        lines = [
            "# 已接收的作者评分",
            "",
            f"共{summary['groups']}组/{summary['candidates']}候选，内容确认{summary['content_confirmed']}项。",
            "",
            "|方法|自然度|本人相似性|词内协调|连接自然度|工整|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method, item in summary["methods"].items():
            cells = [
                f"{item['scores'][field]['mean']:.2f}"
                if item["scores"][field]["mean"] is not None
                else "未评分"
                for field in FIELDS
            ]
            lines.append("|" + method + "|" + "|".join(cells) + "|")
        lines += [
            "",
            "仅描述本次已评图像，不推断全基准正确率或统计显著性。",
            "偏好对按维度分开且保留原split，只用于评估和研究规划；没有执行RL或将生成图回标为真实样本。",
            "位置分布在summary.json中记录，后续页面应采用均衡位置。原导出与旧交付均未修改。",
        ]
        (run.path / "reports/REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish(
            "held",
            result={
                "groups": summary["groups"],
                "candidates": summary["candidates"],
                "reviewed": summary["reviewed"],
                "content_confirmed": summary["content_confirmed"],
                "dimension_pairs": len(pairs),
                "human": True,
                "training_updates": 0,
                "optimization_eligible": False,
                "default_changed": False,
                "latest_updated": False,
            },
            deliverable=False,
        )
        verify_run(run.path)
        return run.path
    except Exception as exc:
        if run.record()["status"] == "running":
            run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("delivery", type=Path)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    print(import_review(args.delivery, args.export))
