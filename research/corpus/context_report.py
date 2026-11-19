"""Descriptive diagnostics and source-linked visual evidence, with no candidate selection."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from hwsol.profiles.io import atomic_json
from hwsol.workspace import verify_run


def load(path):
    return json.loads(Path(path).read_text())


def require_running(destination: Path):
    if load(destination / "run.json")["status"] != "running":
        raise ValueError("Finalized evidence is immutable; use a new running delivery run")


def diagnose(source: Path, destination: Path):
    require_running(destination)
    verify_run(source)
    images = load(source / "reports/image-diagnostics.json")
    latent = load(source / "reports/latent-diagnostics.json")
    summary = {}
    for name, report in images.items():
        classes = report["per_class"]
        dist = latent[name]["distribution_samples"]
        active = {}
        for split, rows in dist.items():
            means = np.asarray([r["posterior_mean_in_prior_sigma"] for r in rows])
            kl = np.asarray([r["kl_per_dimension"] for r in rows])
            # Within-class centering avoids confusing character identity with style variation.
            residual = means.copy()
            for cls in {r["class"] for r in rows}:
                mask = np.array([r["class"] == cls for r in rows])
                residual[mask] -= means[mask].mean(0)
            variance = residual.var(0)
            active[split] = {
                "count": len(rows),
                "mean_kl_per_dim": kl.mean(0).tolist(),
                "within_class_posterior_variance_in_prior_units": variance.tolist(),
                "active_dimensions_at_001": int((variance > 0.01).sum()),
                "threshold_is_descriptive_not_quality_gate": True,
            }
        sensitivity = latent[name]["latent_sensitivity"]
        bins = {}
        for label, low, high in [("none", 0, 0), ("low", 1, 4), ("higher", 5, 10000)]:
            subset = [c for c in classes if low <= c["own_real_train_count"] <= high]
            bins[label] = {
                "classes": len(subset),
                "pairwise_xor": float(np.mean([c["mean_pairwise_ink_xor"] for c in subset]))
                if subset
                else None,
                "font_shape": float(
                    np.mean([c["nearest_font"]["mean_nearest_shape"] for c in subset])
                )
                if subset
                else None,
                "fragment_proxy": sum(c["fragment_proxy_count"] for c in subset),
            }
        summary[name] = {
            "pairwise_xor": float(np.mean([c["mean_pairwise_ink_xor"] for c in classes])),
            "font_shape": float(
                np.mean([c["nearest_font"]["mean_nearest_shape"] for c in classes])
            ),
            "exact_train_binary": sum(
                (c["nearest_own_real_train"] or {}).get("exact_binary_count", 0) for c in classes
            ),
            "exact_font_binary": sum(c["nearest_font"]["exact_binary_count"] for c in classes),
            "soft_edge_fraction": float(np.mean([c["soft_edge_fraction"] for c in classes])),
            "fragment_proxy": sum(c["fragment_proxy_count"] for c in classes),
            "latent_response": float(np.mean([c["mean_absolute_response"] for c in sensitivity])),
            "pixel_identical_latent_classes": sum(
                c["all_directions_pixel_identical"] for c in sensitivity
            ),
            "same_final_data_shape": {
                s: r["mean_best_of_8_shape"] for s, r in report["same_final_data_coverage"].items()
            },
            "coverage_bins": bins,
            "latent_distributions": active,
        }
    atomic_json(destination / "reports/frozen-diagnosis-summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, key, title in zip(
        axes,
        ["pairwise_xor", "font_shape", "latent_response"],
        ["Pairwise ink XOR", "Distance to font", "Latent sensitivity"],
        strict=True,
    ):
        ax.bar(
            list(summary),
            [v[key] for v in summary.values()],
            color=["#537c8c", "#7a9e92", "#5a82a6", "#b17b5a"],
        )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Frozen v0.9 diagnostics: descriptive proxies, not naturalness scores")
    fig.tight_layout()
    fig.savefig(destination / "comparisons/frozen-diagnosis.png", dpi=180)
    plt.close(fig)
    return summary


def comparisons(source: Path, destination: Path):
    require_running(destination)
    verify_run(source)
    rows = load(source / "reports/observations-input.json")
    methods = [
        "real-reference",
        "handnote",
        "round2",
        "repair",
        "context-context",
        "context-no-context",
        "context-no-style",
        "context-support",
        "context-half-real",
    ]
    pdfpath = destination / "comparisons/same-content-real-references.pdf"
    pages = 0
    with PdfPages(pdfpath) as pdf:
        for index in sorted({r["row_index"] for r in rows}):
            current = {r["method"]: r for r in rows if r["row_index"] == index}
            fig, axes = plt.subplots(9, 1, figsize=(8.27, 11.69))
            for ax, method in zip(axes, methods, strict=True):
                ax.imshow(Image.open(current[method]["image"]), cmap="gray", vmin=0, vmax=255)
                ax.set_title(method, loc="left", fontsize=9)
                ax.axis("off")
            fig.suptitle(current["handnote"]["text"], fontsize=10, wrap=True)
            fig.text(
                0.08,
                0.02,
                "Same whole-region scale. Original crops retained. "
                "New context outputs are unaccepted prototypes.",
                fontsize=7,
            )
            fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.55)
            pdf.savefig(fig)
            fig.savefig(destination / f"comparisons/english-reference-{index:03d}.png", dpi=140)
            plt.close(fig)
            pages += 1
        for j, row in enumerate(
            r for r in load(source / "reports/math-references.json") if r["status"] == "rendered"
        ):
            fig, axes = plt.subplots(4, 1, figsize=(8.27, 11.69))
            for ax, method in zip(axes, methods[:4], strict=True):
                ax.imshow(Image.open(row["images"][method]), cmap="gray")
                ax.set_title(method, loc="left", fontsize=10)
                ax.axis("off")
            fig.suptitle("Same-content mathematical reference " + str(j + 1))
            fig.text(
                0.08,
                0.02,
                "Font / frozen glyph composition only. No new mathematical neural generation.",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
            pages += 1
    atomic_json(
        destination / "reports/comparison-pdf.json",
        {
            "pages": pages,
            "english_regions": 29,
            "math_regions": 8,
            "source": str(source),
            "pdf": str(pdfpath),
        },
    )


def curves(main: Path):
    require_running(main)
    paths = load(main / "reports/context-training-runs.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, path in paths.items():
        rows = load(Path(path) / "reports/curve.json")
        # Field names are the worker's actual logged values.
        axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], label=name)
        val = [load(p) for p in sorted(Path(path).glob("reports/validation-*.json"))]
        axes[1].plot([r["step"] for r in val], [r["validation_shape"] for r in val], label=name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation shape proxy (not transcription)")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(main / "comparisons/context-training-curves.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("main", type=Path)
    p.add_argument("comparison", type=Path)
    p.add_argument("diagnosis", type=Path)
    a = p.parse_args()
    diagnose(a.diagnosis, a.main)
    comparisons(a.comparison, a.main)
