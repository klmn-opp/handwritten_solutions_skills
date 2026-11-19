"""Source-linked whole-region clean-ink evidence and anonymous author review materials."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.clean_ink import ink_metrics
from research.corpus.context_report import require_running
from research.corpus.context_review import build_review

METHODS = [
    ("real-reference", "YOUR REAL WRITING - original region, normalized display"),
    ("handnote", "HANDNOTE - unchanged personal font"),
    ("context-previous", "PREVIOUS v0.10.1 - the model you approved"),
    ("context-previous-sharp", "PREVIOUS + DISPLAY CONTRAST ONLY - zero training updates"),
    ("context-control", "NEW TRAINING - continuation control"),
    ("context-distance", "NEW TRAINING - distance-field consistency"),
    ("context-mass", "NEW TRAINING - consistency + ink amount + robust content"),
]


def report(main: Path, comparison: Path) -> None:
    require_running(main)
    verify_run(comparison)
    rows = json.loads((comparison / "reports/observations-input.json").read_text())
    indices = sorted({r["row_index"] for r in rows})
    selected = []
    for split in ["train", "validation", "holdout"]:
        selected += sorted({r["row_index"] for r in rows if r["split"] == split})[
            : 2 if split == "train" else 4
        ]
    index = []
    metric_rows = []
    with (
        PdfPages(main / "comparisons/all-clean-ink-comparisons.pdf") as full,
        PdfPages(main / "rendered/homework_solutions.pdf") as overview,
    ):
        for i in indices:
            current = {r["method"]: r for r in rows if r["row_index"] == i}
            fig, axes = plt.subplots(len(METHODS), 1, figsize=(8.27, 11.69))
            for ax, (method, label) in zip(axes, METHODS, strict=True):
                r = current[method]
                im = Image.open(r["image"])
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
                ax.set_title(label, fontsize=9, loc="left")
                ax.axis("off")
                ink = 1 - np.asarray(im.convert("L"), dtype=np.float32) / 255
                mask = ink > 0.5
                outside = cv2.distanceTransform((~mask).astype("uint8"), cv2.DIST_L2, 5)
                halo = (ink > 0.1) & (outside > 1.5)
                metric_rows.append(
                    {
                        "id": r["id"],
                        "row_index": i,
                        "method": method,
                        "split": r["split"],
                        **ink_metrics(ink),
                        "halo_far_pixels": int(halo.sum()),
                        "halo_far_mass": float(ink[halo].sum()),
                        "sha256": sha256(Path(r["image"])),
                    }
                )
            split = current["handnote"]["split"]
            text = current["handnote"]["text"]
            fig.suptitle(split.upper() + " | " + text, fontsize=10, wrap=True)
            fig.text(
                0.055,
                0.015,
                "Same text and display scale. New neural rows use unchanged output conversion. "
                "No post-hoc cleaning.",
                fontsize=7,
            )
            fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.5)
            full.savefig(fig)
            if i in selected:
                overview.savefig(fig)
                fig.savefig(main / f"rendered/pages/comparison-{i:03d}.png", dpi=150)
            plt.close(fig)
            index.append({"row_index": i, "split": split, "text": text, "overview": i in selected})
    summary = {}
    for method, _ in METHODS:
        summary[method] = {}
        for split in ["train", "validation", "holdout"]:
            subset = [r for r in metric_rows if r["method"] == method and r["split"] == split]
            summary[method][split] = {
                "n": len(subset),
                **{
                    key: float(np.mean([r[key] for r in subset]))
                    for key in [
                        "ink_mass",
                        "soft_fraction",
                        "core_pixels",
                        "halo_far_pixels",
                        "halo_far_mass",
                        "components",
                    ]
                },
                "blank": sum(r["blank"] for r in subset),
                "border_ink": sum(r["border_ink"] for r in subset),
            }
    atomic_json(
        main / "reports/clean-ink-metrics.json",
        {
            "summary": summary,
            "rows": metric_rows,
            "interpretation": "Descriptive only; removal of genuine thin strokes can lower haze. "
            "Check content independently.",
        },
    )
    atomic_json(
        main / "reports/comparison-index.json",
        {
            "comparison": str(comparison),
            "all_pages": len(indices),
            "overview_pages": len(selected),
            "pages": index,
            "research_comparison_not_solved_homework": True,
        },
    )
    paths = json.loads((main / "reports/training-runs.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, p in paths.items():
        if name not in {"control", "distance", "mass"}:
            continue
        curve = json.loads((Path(p) / "reports/curve.json").read_text())
        ev = json.loads((Path(p) / "reports/evaluations.json").read_text())
        axes[0].plot([r["step"] for r in curve], [r["loss"] for r in curve], label=name)
        axes[1].plot([r["step"] for r in ev], [r["soft_fraction"] for r in ev], label=name)
        axes[2].plot([r["step"] for r in ev], [r["ink_mass"] for r in ev], label=name)
    for ax, title in zip(
        axes,
        [
            "Different objectives: losses not directly comparable",
            "Val soft pixels / visible pixels",
            "Val total ink: not a content score",
        ],
        strict=True,
    ):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Updates")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(main / "comparisons/training-curves.png", dpi=160)
    plt.close(fig)
    # Four fixed non-training groups; candidate labels carry no backend hint.
    review = main / "comparisons/author-review"
    (review / "assets").mkdir(parents=True, exist_ok=True)
    rng = random.Random(1207)
    groups = []
    key = []
    review_indices = []
    for split in ["validation", "holdout"]:
        review_indices += sorted({r["row_index"] for r in rows if r["split"] == split})[:2]
    for gi, i in enumerate(review_indices):
        current = {r["method"]: r for r in rows if r["row_index"] == i}
        methods = [
            "handnote",
            "context-previous",
            "context-previous-sharp",
            "context-distance",
            "context-mass",
        ]
        rng.shuffle(methods)
        candidates = []
        for ci, method in enumerate(methods):
            r = current[method]
            opaque = f"g{gi + 1:02d}c{ci + 1:02d}"
            file = f"assets/{opaque}.png"
            shutil.copyfile(r["image"], review / file)
            candidates.append({"id": opaque, "label": chr(65 + ci), "image": file})
            key.append({**r, "source_observation_id": r["id"], "id": opaque})
        reference = f"assets/original-{gi + 1}.png"
        shutil.copyfile(current["real-reference"]["raw_image"], review / reference)
        groups.append(
            {
                "id": f"g{gi + 1:02d}",
                "title": f"去灰影对照 {gi + 1} / {len(review_indices)}",
                "description": "先确认文字完整，特别看 i 的点、标点、细笔和下伸。"
                "自然度评分包含灰影/重影是否消失；备注可写偏好字母及具体缺笔。其他评分可留空。",
                "target": current["handnote"]["text"],
                "reference": reference,
                "candidates": candidates,
            }
        )
    build_review(review / "index.html", groups, main.name)
    atomic_json(main / "reports/author-review-key.json", key)
    (review / "操作说明.txt").write_text(
        "双击 index.html。共4组，不需要重新转写文字或填写LaTeX。\n"
        "1. 展开目标文字，对照每个候选。错字、少字、丢点/断笔选有错误；不确定选有疑问。\n"
        "2. 内容正确才评自然程度（灰影/重影越少越好）与像本人程度，其他项目可留空。\n"
        "3. 备注写“更喜欢A/B…；哪个字的点或下伸丢失”。可展开本人原图参考；点击候选放大。\n"
        "4. 点击导出审核数据，将JSON保存到仓库source_pdf/review_exports/，告知已导出即可。\n"
        "页面只在本机浏览器保存，不上传。导出前不要清除浏览器数据。不要修改封存run文件。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("main", type=Path)
    p.add_argument("comparison", type=Path)
    a = p.parse_args()
    report(a.main, a.comparison)
