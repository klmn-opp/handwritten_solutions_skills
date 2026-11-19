"""Offline anonymous comparison UI; user ratings remain independent of model observations."""

import html
import json
from pathlib import Path


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
