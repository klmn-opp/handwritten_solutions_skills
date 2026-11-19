"""Reproducible all-class sampling and explicit proxy/independent-data separation."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create
from research.corpus.reference_worker import contact_sheet


def sample_worker(request_path: Path):
    import torch

    from research.corpus.reference_generator import ReferenceCVAE, ink_from_sdf

    request = json.loads(request_path.read_text())
    checkpoint = Path(request["checkpoint"])
    if sha256(checkpoint) != request["checkpoint_sha256"]:
        raise ValueError("Frozen model changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA sampling requested")
    torch.set_num_threads(2)
    torch.manual_seed(request["seed"])
    state = torch.load(checkpoint, weights_only=True, map_location="cuda")
    model = ReferenceCVAE(len(state["labels"])).cuda()
    model.load_state_dict(state["model"])
    model.eval()
    arrays = np.load(Path(request["data"]) / "arrays.npz", allow_pickle=False)
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    labels = torch.arange(len(refs), device="cuda").repeat_interleave(request["count"])
    noise = torch.randn(len(labels), 8, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = torch.cat(
            [
                ink_from_sdf(
                    model.generate(refs[labels[i : i + 32]], labels[i : i + 32], noise[i : i + 32])
                )
                for i in range(0, len(labels), 32)
            ]
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    np.savez_compressed(
        Path(request["run_path"]) / "reports/generated.npz",
        ink=outputs[:, 0].cpu().numpy().reshape(len(refs), request["count"], 96, 96),
        noise=noise.cpu().numpy(),
    )
    atomic_json(
        Path(request["run_path"]) / "reports/inference.json",
        {
            "seconds": elapsed,
            "samples": len(labels),
            "milliseconds_per_glyph": elapsed * 1000 / len(labels),
            "device": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "checkpoint_step": state["step"],
            "architecture": state["architecture"],
        },
    )


def shape_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (np.abs(left - right) * (1 + 4 * left + 4 * right)).mean(axis=(-2, -1))


def character_metrics(
    generated: np.ndarray, font: np.ndarray, font_labels: np.ndarray, index: int
) -> dict:
    import cv2

    own = font[font_labels == index]
    distances = shape_distance(generated[:, None], font[None])
    masks = generated > 0.5
    pairwise = [
        np.logical_xor(masks[a], masks[b]).sum() / max(1, np.logical_or(masks[a], masks[b]).sum())
        for a in range(len(masks))
        for b in range(a + 1, len(masks))
    ]
    nearest_binary = [
        min(np.logical_xor(mask, other > 0.5).sum() for other in own) for mask in masks
    ]

    def components(mask):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        return int((stats[1:, cv2.CC_STAT_AREA] >= 3).sum())

    max_components = max(components(x > 0.5) for x in own)
    fragment = [components(mask) > max_components + 1 for mask in masks]
    border = np.concatenate(
        [
            masks[:, :2].reshape(len(masks), -1),
            masks[:, -2:].reshape(len(masks), -1),
            masks[:, :, :2].reshape(len(masks), -1),
            masks[:, :, -2:].reshape(len(masks), -1),
        ],
        1,
    ).any(1)
    return {
        "template_identity_proxy": float((font_labels[distances.argmin(1)] == index).mean()),
        "blank_count": int((~masks.reshape(len(masks), -1).any(1)).sum()),
        "border_count": int(border.sum()),
        "fragment_proxy_count": sum(fragment),
        "unique_binary_outputs": len({m.tobytes() for m in masks}),
        "mean_pairwise_ink_xor": float(np.mean(pairwise)) if pairwise else None,
        "exact_font_mask_copies": int(sum(x == 0 for x in nearest_binary)),
        "nearest_font_differing_pixels_min": int(min(nearest_binary)),
        "nearest_font_shape_mae": float(
            shape_distance(generated[:, None], own[None]).min(1).mean()
        ),
        "semantic_status": "unverified; template proxy is not independent recognition",
        "naturalness_status": "unverified by author",
    }


def panels(run: Path, labels: list[str], reference: np.ndarray, generated: np.ndarray):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    paths, provenance = [], []
    samples = run / "rendered/glyphs"
    samples.mkdir()
    for i, label in enumerate(labels):
        for j, ink in enumerate(generated[i]):
            path = samples / f"u{ord(label):04x}-s{j:02d}.png"
            Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(path)
            provenance.append(
                {
                    "character": label,
                    "glyph_id": path.stem,
                    "path": str(path),
                    "sha256": sha256(path),
                    "source_type": "neural-offline-generated",
                    "is_real_personal_sample": False,
                    "external_fallback": False,
                    "production_eligible": False,
                }
            )
    atomic_json(run / "reports/glyph-provenance.json", provenance)
    for start in range(0, len(labels), 15):
        canvas = Image.new("RGB", (870, 1660), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (20, 12),
            "Character | Handnote reference | Neural samples (fixed seeds)",
            fill="black",
            font=font,
        )
        for row, index in enumerate(range(start, min(start + 15, len(labels)))):
            y = 45 + row * 106
            draw.text(
                (12, y + 28), f"{labels[index]}\n{ord(labels[index]):04X}", fill="black", font=font
            )
            for col, ink in enumerate([reference[index], *generated[index, :7]]):
                canvas.paste(
                    Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)), (64 + col * 100, y)
                )
        path = (
            run
            / f"rendered/pages/characters-{start + 1:02d}-{min(start + 15, len(labels)):02d}.png"
        )
        canvas.save(path)
        paths.append(path)
    # A4 at 150 DPI, using existing Pillow PDF support (no new runtime dependency).
    pages = []
    for path in paths:
        page = Image.new("RGB", (1240, 1754), "white")
        with Image.open(path) as source:
            content = ImageOps.contain(source, (1140, 1654))
            page.paste(content, ((1240 - content.width) // 2, (1754 - content.height) // 2))
        page.save(path)
        pages.append(page)
    pages[0].save(
        run / "rendered/homework_solutions.pdf",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )
    contact_sheet(generated[:, 0], labels, run / "comparisons/all-75.png")


def sample(training: Path, *, checkpoint_name="best-checkpoint.pt", seed=2909, count=8) -> Path:
    from research.corpus.reference_data import validate_views

    if checkpoint_name not in {"best-checkpoint.pt", "last-checkpoint.pt"} or not 2 <= count <= 32:
        raise ValueError("Invalid sampling request")
    verify_run(training)
    train_request = json.loads((training / "reports/request.json").read_text())
    data = Path(train_request["data"])
    manifest = validate_views(data)
    checkpoint = training / "logs" / checkpoint_name
    request = {
        "training": str(training.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "data": str(data),
        "seed": seed,
        "count": count,
    }
    run = create("all-character-sampling", config=request, seed=seed)
    request["run_path"] = str(run.path)
    atomic_json(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/inference.log").open("w") as stream:
            subprocess.run(
                [
                    str(Path("workspace/environments/corpus-generator-v09/bin/python").absolute()),
                    "-m",
                    "research.corpus.reference_evaluate",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        arrays = np.load(data / "arrays.npz", allow_pickle=False)
        generated = np.load(run.path / "reports/generated.npz", allow_pickle=False)["ink"]
        font = arrays["target"][arrays["domains"] == 0]
        font_labels = arrays["classes"][arrays["domains"] == 0]
        metrics = [
            {"label": label, **character_metrics(generated[i], font, font_labels, i)}
            for i, label in enumerate(manifest["labels"])
        ]
        atomic_json(run.path / "reports/per-character.json", metrics)
        summary = {
            "classes": len(metrics),
            "samples": int(generated.shape[0] * generated.shape[1]),
            "template_identity_proxy": float(
                np.mean([m["template_identity_proxy"] for m in metrics])
            ),
            "blank_count": sum(m["blank_count"] for m in metrics),
            "border_count": sum(m["border_count"] for m in metrics),
            "fragment_proxy_count": sum(m["fragment_proxy_count"] for m in metrics),
            "exact_font_mask_copies": sum(m["exact_font_mask_copies"] for m in metrics),
            "collapsed_classes": [m["label"] for m in metrics if m["mean_pairwise_ink_xor"] < 0.02],
            "production_eligible": False,
        }
        atomic_json(run.path / "reports/summary.json", summary)
        panels(run.path, manifest["labels"], arrays["reference"], generated)
        run.finish("held", result=summary)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        sample_worker(Path(sys.argv[2]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("training", type=Path)
        parser.add_argument("--checkpoint-name", default="best-checkpoint.pt")
        parser.add_argument("--seed", type=int, default=2909)
        args = parser.parse_args()
        print(sample(args.training, checkpoint_name=args.checkpoint_name, seed=args.seed))
