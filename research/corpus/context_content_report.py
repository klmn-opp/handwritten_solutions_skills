"""Explicit new/old evidence pages and loss curves; no hidden candidate selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from hwsol.profiles.io import atomic_json
from hwsol.workspace import verify_run
from research.corpus.context_report import require_running


def report(main: Path, comparison: Path):
    require_running(main)
    verify_run(comparison)
    rows = json.loads((comparison / "reports/observations-input.json").read_text())
    methods = [
        ("real-reference", "YOUR ORIGINAL - normalized display"),
        ("handnote", "HANDNOTE - unchanged personal font"),
        ("repair", "OLD v0.9 repair glyph composition"),
        ("context-old-context", "OLD v0.10 context model"),
        ("context-font-warmup", "NEW network - FONT WARMUP ONLY"),
        ("context-raw", "NEW real training - whole-region target"),
        ("context-aligned", "NEW real training - word alignment"),
        ("context-aligned-content", "NEW real training - alignment + content constraint"),
    ]
    indices = sorted({r["row_index"] for r in rows})
    selected = []
    for split in ["train", "validation", "holdout"]:
        selected.extend(
            sorted({r["row_index"] for r in rows if r["split"] == split})[
                : (2 if split == "train" else 4)
            ]
        )
    summaries = []
    with (
        PdfPages(main / "comparisons/all-region-comparisons.pdf") as full,
        PdfPages(main / "rendered/homework_solutions.pdf") as overview,
    ):
        for i in indices:
            current = {r["method"]: r for r in rows if r["row_index"] == i}
            fig, axes = plt.subplots(len(methods), 1, figsize=(8.27, 11.69))
            for ax, (method, label) in zip(axes, methods, strict=True):
                ax.imshow(Image.open(current[method]["image"]), cmap="gray", vmin=0, vmax=255)
                ax.set_title(label, fontsize=9, loc="left")
                ax.axis("off")
            label = current["handnote"]["text"]
            split = current["handnote"]["split"]
            fig.suptitle(f"{split.upper()} | " + label, fontsize=10, wrap=True)
            fig.text(
                0.06,
                0.015,
                "Whole-region neural output. Deterministic content pilot; "
                "no stochastic style or math-generation claim.",
                fontsize=7,
            )
            fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.55)
            full.savefig(fig)
            if i in selected:
                overview.savefig(fig)
                fig.savefig(main / f"rendered/pages/comparison-{i:03d}.png", dpi=150)
            plt.close(fig)
            summaries.append(
                {"row_index": i, "split": split, "text": label, "overview": i in selected}
            )
    atomic_json(
        main / "reports/comparison-index.json",
        {
            "comparison": str(comparison),
            "all_pages": len(indices),
            "overview_pages": len(selected),
            "pages": summaries,
            "overview_is_comparison_not_solved_homework": True,
        },
    )
    paths = json.loads((main / "reports/training-runs.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, p in paths.items():
        history = json.loads((Path(p) / "reports/curve.json").read_text())
        ev = json.loads((Path(p) / "reports/evaluations.json").read_text())
        ax[0].plot([r["step"] for r in history], [r["loss"] for r in history], label=name)
        ax[1].plot([r["step"] for r in ev], [r["validation"]["raw_shape"] for r in ev], label=name)
    ax[0].set_title("Training loss: teacher arm includes additional content term")
    ax[1].set_title("Raw validation shape proxy, NOT content or style score")
    for a in ax:
        a.set_xlabel("Updates")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(main / "comparisons/training-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("main", type=Path)
    p.add_argument("comparison", type=Path)
    a = p.parse_args()
    report(a.main, a.comparison)
