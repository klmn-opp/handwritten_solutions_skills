"""Fixed post-training raster diagnostics, not author style or content acceptance."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus.experiment import create
from research.corpus.label_admission import label_text
from research.corpus.training_data import ink_view


def evaluate(training_run: Path, font: Path, python: Path | None = None) -> Path:
    """Allocate provenance in the CPU environment; keep Torch in the optional worker."""
    verify_run(training_run)
    receipt = json.loads((training_run / "reports/training.json").read_text())
    if not receipt["training_executed"] or receipt["parameter_absolute_change"] <= 0:
        raise ValueError("A real learned checkpoint is required")
    run = create(
        "digit1-neural-evaluation",
        seed=1902,
        config={
            "training_run": str(training_run.resolve()),
            "checkpoint_sha256": sha256(training_run / "logs/best-checkpoint.pt"),
            "font": str(font.resolve()),
            "font_baseline": "personal@v1 raw TTF glyph via Pillow; no page-layout comparison",
            "font_sha256": sha256(font),
            "holdout_used_after_checkpoint_frozen": True,
        },
    )
    interpreter = python or Path("workspace/environments/corpus-generator-v09/bin/python")
    try:
        with (run.path / "logs/evaluation.log").open("w") as log:
            subprocess.run(
                [str(interpreter.absolute()), "-m", __spec__.name, "--worker-run", str(run.path)],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        run.finish(
            "held",
            result={
                "real_checkpoint_evaluated": True,
                "generated": 16,
                "production_eligible": False,
                "training_run": str(training_run),
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def worker(run: Run) -> None:
    import torch

    from research.corpus.generator import GlyphCVAE

    config = run.record()["config"]
    training_run, font = Path(config["training_run"]), Path(config["font"])
    verify_run(training_run)
    if sha256(font) != config["font_sha256"]:
        raise ValueError("Baseline font changed")
    receipt = json.loads((training_run / "reports/training.json").read_text())
    if not receipt["training_executed"] or receipt["parameter_absolute_change"] <= 0:
        raise ValueError("A real learned checkpoint is required")
    request = json.loads((training_run / "reports/request.json").read_text())
    checkpoint = training_run / "logs/best-checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    labels = saved["labels"]
    if labels != ["1"]:
        raise ValueError("This bounded pilot evaluates digit 1 only")
    mode = saved.get("preprocessing", "raw-gray")
    rows = request["rows"]
    train = [r for r in rows if r["dataset_split"] == "train" and label_text(r) == "1"]
    held = [r for r in rows if r["dataset_split"] == "holdout" and label_text(r) == "1"]
    model = GlyphCVAE(len(labels)).eval()
    model.load_state_dict(saved["model"], strict=True)
    seed = 1902
    torch.manual_seed(seed)
    with torch.inference_mode():
        generated = model.generate(
            torch.zeros(16, dtype=torch.long),
            torch.zeros(16, dtype=torch.long),
            torch.zeros(16, 4),
            torch.randn(16, saved["architecture"]["latent"]),
        )[:, 0].numpy()
    train_images = np.stack([ink_view(Path(r["image_crop"]), mode) for r in train])
    held_images = np.stack([ink_view(Path(r["image_crop"]), mode) for r in held])
    differences = np.abs(generated[:, None] - train_images[None]).mean(axis=(2, 3))
    nearest = differences.argmin(axis=1)
    pairwise = [
        float(np.abs(a - b).mean()) for i, a in enumerate(generated) for b in generated[i + 1 :]
    ]

    def bitmap(ink):
        return Image.fromarray(np.rint((1 - np.clip(ink, 0, 1)) * 255).astype(np.uint8))

    generated_bytes = []
    for i, ink in enumerate(generated):
        im = bitmap(ink)
        im.save(run.path / f"rendered/pages/generated-{i:02d}.png")
        generated_bytes.append(im.tobytes())
    face = ImageFont.truetype(str(font), 72)
    base = Image.new("L", (120, 120), "white")
    box = face.getbbox("1")
    ImageDraw.Draw(base).text((10 - box[0], 10 - box[1]), "1", font=face, fill=0)
    base = ImageOps.pad(
        base.crop((6, 6, 14 + box[2] - box[0], 14 + box[3] - box[1])), (64, 64), color=255
    )
    base.save(run.path / "comparisons/personal-v1-font.png")
    geometric = bitmap(train_images[0]).rotate(5, resample=Image.Resampling.BICUBIC, fillcolor=255)
    geometric.save(run.path / "comparisons/geometric-control.png")
    sheet = Image.new("RGB", (1000, 690), "white")
    draw = ImageDraw.Draw(sheet)
    groups = [
        ("Real training views (Sol labels)", [bitmap(x) for x in train_images[:5]]),
        ("Real holdout views (not optimized)", [bitmap(x) for x in held_images]),
        ("Font personal@v1 / geometric control", [base, geometric]),
        ("Neural samples 0-7 (fixed latent seed)", [bitmap(x) for x in generated[:8]]),
        ("Neural samples 8-15", [bitmap(x) for x in generated[8:]]),
    ]
    for row, (title, images) in enumerate(groups):
        y = row * 130 + 12
        draw.text((14, y), title, fill="black")
        for col, im in enumerate(images):
            sheet.paste(im.resize((90, 90)), (16 + col * 120, y + 25))
    draw.text(
        (14, 665),
        "Digit 1 pilot only. Pixel diversity does not establish naturalness or author style.",
        fill="black",
    )
    sheet.save(run.path / "comparisons/digit1-comparison.png")
    sheet.save(
        run.path / "rendered/homework_solutions.pdf",
        resolution=150,
        title="Digit 1 research comparison; not homework output",
    )
    report = {
        "classes": labels,
        "preprocessing": mode,
        "source_labels": "Sol model-accepted; not human ground truth",
        "generated_count": 16,
        "unique_generated_8bit_images": len(
            {hashlib.sha256(x).hexdigest() for x in generated_bytes}
        ),
        "exact_training_pixel_copies": sum(
            x in {bitmap(t).tobytes() for t in train_images} for x in generated_bytes
        ),
        "minimum_pairwise_pixel_mae": min(pairwise),
        "mean_pairwise_pixel_mae": float(np.mean(pairwise)),
        "nearest_training_mae": differences.min(axis=1).tolist(),
        "nearest_training_ids": [train[i]["sample_id"] for i in nearest],
        "holdout_nearest_training_mae": np.abs(held_images[:, None] - train_images[None])
        .mean(axis=(2, 3))
        .min(axis=1)
        .tolist(),
        "ink_fraction_above_half": (generated > 0.5).mean(axis=(1, 2)).tolist(),
        "weights_learned": receipt["parameter_absolute_change"] > 0,
        "content_identity": "not independently assessed",
        "style_naturalness": "not assessed",
        "quality_decision": "NO-GO production; one symbol, no independent content/style acceptance",
        "novelty_limit": "Nonidentical pixels do not prove meaningful new handwriting variation",
        "training_source_count": len(train),
        "holdout_count": len(held),
    }
    atomic_json(run.path / "reports/evaluation.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("training_run", type=Path, nargs="?")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--worker-run", type=Path)
    args = parser.parse_args()
    if args.worker_run:
        worker(Run(args.worker_run))
    elif args.training_run and args.font:
        print(evaluate(args.training_run, args.font, args.python))
    else:
        parser.error("training_run and --font are required")
