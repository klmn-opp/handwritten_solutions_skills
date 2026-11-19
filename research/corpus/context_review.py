"""Offline anonymous comparison UI; user ratings remain independent of model observations."""

import html
import json
import random
from pathlib import Path


def balanced_candidate_orders(methods: list[str], groups: int, seed: int) -> list[list[str]]:
    """Seeded rotations balance every backend over positions without changing old reviews."""
    if not methods or len(methods) != len(set(methods)) or type(groups) is not int or groups < 0:
        raise ValueError("Unique nonempty methods and a nonnegative group count required")
    rng = random.Random(seed)
    base = list(methods)
    rng.shuffle(base)
    result = []
    while len(result) < groups:
        offsets = list(range(len(base)))
        rng.shuffle(offsets)
        for offset in offsets:
            result.append(base[offset:] + base[:offset])
            if len(result) == groups:
                break
    return result


def build_review(destination: Path, groups: list[dict], review_id: str) -> None:
    """Groups contain opaque candidate IDs/images, never backend names or checkpoint IDs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"id": review_id, "groups": groups}, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    source = (Path(__file__).parent / "web/context-review.html").read_text("utf-8")
    destination.write_text(source.replace("__DATA__", data), encoding="utf-8")
    (destination.parent / "README.txt").write_text(
        html.unescape(
            "双击 index.html。先检查内容，再评分；点击图像可放大。修改自动保存在当前浏览器，"
            "请用‘导出审核数据’保存文件。导出的文件可用‘恢复数据’继续。"
        ),
        encoding="utf-8",
    )
