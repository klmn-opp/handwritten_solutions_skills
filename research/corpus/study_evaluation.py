"""Frozen-model comparison on newly sealed pages; no fitting or threshold search."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import Run, verify_run
from research.corpus.experiment import create


def pairwise_ink_distance(images) -> float:
    """Symmetric foreground-weighted distance, with equal weight for every unordered pair."""
    import numpy as np

    if len(images) < 2:
        return 0.0
    total, count = 0.0, 0
    for i, image in enumerate(images[:-1]):
        others = images[i + 1 :]
        values = (np.abs(image - others) * (1 + 4 * image + 4 * others)).mean(axis=(1, 2))
        total += float(values.sum())
        count += len(values)
    return total / count


def evaluate(study: Path, freeze: Path, test_run: Path) -> Path:
    verify_run(freeze)
    verify_run(test_run)
    checkpoints = json.loads((freeze / "reports/frozen-checkpoints.json").read_text())
    for item in checkpoints:
        source = Path(item["run"])
        verify_run(source)
        if sha256(source / "logs/best-checkpoint.pt") != item["checkpoint_sha256"]:
            raise ValueError("Frozen candidate changed")
    run = create(
        "glyph-study-evaluation",
        seed=2902,
        config={
            "study": str(study.resolve()),
            "freeze": str(freeze.resolve()),
            "test_run": str(test_run.resolve()),
            "new_training_updates": 0,
            "test_label_source": "user-authorized Sol; not independent human ground truth",
            "test_independence": "new pages in original heldout document; not new documents",
        },
    )
    try:
        with (run.path / "logs/evaluation.log").open("w") as log:
            subprocess.run(
                [
                    str(Path("workspace/environments/corpus-generator-v09/bin/python").absolute()),
                    "-m",
                    "research.corpus.study_evaluation",
                    "--worker",
                    str(run.path),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        run.finish(
            "held",
            result={
                "comparison_executed": True,
                "production_eligible": False,
                "content_and_style_acceptance": "not-established",
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def validate_new_test(test_run: Path) -> list[dict]:
    """Recheck exact original pixels and Sol evidence; do not re-transcribe labels."""
    import numpy as np
    from PIL import Image

    verify_run(test_run)
    rows = json.loads((test_run / "reports/proposals.json").read_text())
    if not rows:
        raise ValueError("No new test samples")
    seen = set()
    for row in rows:
        if (
            row["candidate_label"] != "1"
            or row["split"] != "holdout"
            or row["decision"] != "accept"
            or row["quality_flags"]
            or row["ambiguity_flags"]
            or row["transcription_confidence"] != "high"
            or row["human_review"]
            or row["annotation_model_id"] != "gpt-6.2-sol"
            or row["annotation_reasoning_effort"] != "xhigh"
            or not all(
                row.get(k) is True
                for k in ("single_symbol", "complete_crop", "writer_personal", "privacy_safe")
            )
        ):
            raise ValueError("Unadmitted new test annotation")
        if row["crop_sha256"] in seen:
            raise ValueError("Duplicate new test crop")
        seen.add(row["crop_sha256"])
        for key, digest in [
            ("crop_path", "crop_sha256"),
            ("page_image_path", "source_image_sha256"),
        ]:
            if sha256(Path(row[key])) != row[digest]:
                raise ValueError("New test source integrity mismatch")
        source = Path("source_pdf/source_pdf_phy1.pdf")
        if sha256(source) != row["source_pdf_sha256"]:
            raise ValueError("Wrong source document for test")
        with Image.open(row["page_image_path"]) as page, Image.open(row["crop_path"]) as crop:
            box = row["bbox"]
            if len(box) != 4 or not all(type(v) is int for v in box):
                raise ValueError("Invalid test crop coordinates")
            if not (0 <= box[0] < box[2] <= page.width and 0 <= box[1] < box[3] <= page.height):
                raise ValueError("Test crop extends beyond original page")
            if not np.array_equal(np.asarray(page.crop(row["bbox"])), np.asarray(crop)):
                raise ValueError("Test crop differs from original page pixels")
    for i, row in enumerate(rows):
        x0, y0, x1, y1 = row["bbox"]
        for other in rows[i + 1 :]:
            if row["source_image_sha256"] != other["source_image_sha256"]:
                continue
            a, b, c, d = other["bbox"]
            intersection = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
            if intersection / min((x1 - x0) * (y1 - y0), (c - a) * (d - b)) > 0.65:
                raise ValueError("Overlapping physical ink in new test")
    return rows


def worker(path: Path) -> None:
    import hashlib

    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from research.corpus.font_auxiliary import validate_auxiliary
    from research.corpus.generator import GlyphCVAE
    from research.corpus.label_admission import label_text
    from research.corpus.training_data import ink_view

    torch.set_num_threads(2)
    config = Run(path).record()["config"]
    study, freeze, test_run = [Path(config[k]) for k in ("study", "freeze", "test_run")]
    verify_run(freeze)
    frozen = json.loads((freeze / "reports/frozen-checkpoints.json").read_text())
    test = validate_new_test(test_run)
    protocol = json.loads((study / "reports/frozen-protocol.json").read_text())
    font_rows, _ = validate_auxiliary(Path(protocol["auxiliary"]))
    fonts = np.stack(
        [
            1 - np.asarray(Image.open(r["image"]).convert("L"), dtype=np.float32) / 255
            for r in font_rows
            if r["label"] == "1"
        ]
    )
    request = json.loads((Path(frozen[0]["run"]) / "reports/request.json").read_text())
    train_rows = [r for r in request["real_rows"] if r["dataset_split"] == "train"]
    assert all(label_text(r) == "1" for r in train_rows)
    training = np.stack([ink_view(Path(r["image_crop"]), "dark-ink-v1") for r in train_rows])
    testing = np.stack([ink_view(Path(r["crop_path"]), "dark-ink-v1") for r in test])
    atomic_json(
        path / "reports/test-admission.json",
        {
            "count": len(test),
            "pages": sorted({r["page_number"] for r in test}),
            "source_pdf_sha256": test[0]["source_pdf_sha256"],
            "label_source": "Sol model-accepted under existing user directive",
            "human_ground_truth": False,
            "unchanged_raw_crops": True,
            "candidate_freeze": str(freeze),
            "same_original_heldout_document": True,
        },
    )

    def bitmap(ink):
        return Image.fromarray(np.rint((1 - ink.clip(0, 1)) * 255).astype("uint8"))

    def distances(a, b):
        return (np.abs(a[:, None] - b[None]) * (1 + 8 * b[None])).mean(axis=(2, 3))

    def components(mask):
        unseen = set(zip(*np.where(mask), strict=True))
        sizes = []
        while unseen:
            queue = [unseen.pop()]
            count = 0
            while queue:
                y, x = queue.pop()
                count += 1
                for yy in range(y - 1, y + 2):
                    for xx in range(x - 1, x + 2):
                        if (yy, xx) in unseen:
                            unseen.remove((yy, xx))
                            queue.append((yy, xx))
            if count >= 3:
                sizes.append(count)
        return len(sizes)

    def metrics(images):
        masks = images > 0.5
        borders = np.concatenate(
            [
                masks[:, :2].reshape(len(images), -1),
                masks[:, -2:].reshape(len(images), -1),
                masks[:, :, :2].reshape(len(images), -1),
                masks[:, :, -2:].reshape(len(images), -1),
            ],
            axis=1,
        )
        train_distance = distances(images, training)
        near_train = train_distance.min(axis=1)
        hold_distance = distances(images, testing)
        transitions = ((images > 0.1) & (images < 0.9)).sum(axis=(1, 2))
        ink_pixels = (images > 0.1).sum(axis=(1, 2))
        exact_train = {bitmap(x).tobytes() for x in training}
        return {
            "count": len(images),
            "blank_count": int((~masks.reshape(len(images), -1).any(axis=1)).sum()),
            "border_ink_count": int(borders.any(axis=1).sum()),
            "disconnected_count": sum(components(m) > 1 for m in masks),
            "mean_peak_ink": float(images.max(axis=(1, 2)).mean()),
            "mean_dark_fraction": float(masks.mean()),
            "mean_soft_edge_fraction": float((transitions / np.maximum(1, ink_pixels)).mean()),
            "binary_unique_count": len({hashlib.sha256(m.tobytes()).hexdigest() for m in masks}),
            "mean_pairwise_weighted_mae": pairwise_ink_distance(images),
            "exact_training_copies": sum(bitmap(x).tobytes() in exact_train for x in images),
            "nearest_train_weighted_mae_mean": float(near_train.mean()),
            "nearest_train_weighted_mae_min": float(near_train.min()),
            "nearest_font_weighted_mae_mean": float(distances(images, fonts).min(axis=1).mean()),
            "new_test_nearest_generation_weighted_mae": float(hold_distance.min(axis=0).mean()),
            "new_test_nearest_per_sample": hold_distance.min(axis=0).tolist(),
            "content_identity": "unverified; 1/l ambiguity is not resolved by pixel metrics",
            "style_naturalness": "not assessed by author",
        }

    groups = {"font-control": fonts}
    rng = np.random.default_rng(2902)
    groups["geometric-control"] = np.stack(
        [
            1
            - np.asarray(
                bitmap(fonts[i % len(fonts)]).rotate(
                    float(rng.uniform(-5, 5)), resample=Image.Resampling.BICUBIC, fillcolor=255
                ),
                dtype=np.float32,
            )
            / 255
            for i in range(64)
        ]
    )
    sources = [
        {
            "run": str(Path("workspace/runs/v09-sol-digit1-cvae-pilot").resolve()),
            "stage": "original-pilot",
            "seed": 902,
        }
    ] + [r for r in frozen if r["stage"] != "font-pretrain"]
    timings = {}
    import time

    for record in sources:
        source = Path(record["run"])
        verify_run(source)
        file = source / "logs/best-checkpoint.pt"
        if "checkpoint_sha256" in record and sha256(file) != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after freeze")
        saved = torch.load(file, map_location="cpu", weights_only=True)
        model = GlyphCVAE(len(saved["labels"])).eval()
        model.load_state_dict(saved["model"], strict=True)
        generator = torch.Generator().manual_seed(2902)
        z = torch.randn(64, 16, generator=generator)
        label = torch.full((64,), saved["labels"].index("1"), dtype=torch.long)
        start = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(
                label, torch.zeros(64, dtype=torch.long), torch.zeros(64, 4), z
            )[:, 0].numpy()
        key = f"{record['stage']}-s{record['seed']}"
        groups[key] = result
        timings[key] = time.perf_counter() - start
    reports = {key: metrics(values) for key, values in groups.items()}
    for key, values in groups.items():
        folder = path / "comparisons" / key
        folder.mkdir()
        for i, ink in enumerate(values):
            bitmap(ink).save(folder / f"{i:03d}.png")
    pages = []
    for seed in (902, 903):
        rows = [
            ("Real TRAIN; Sol labels", training[:8]),
            ("New sealed-test pages; Sol labels, not human ground truth", testing[:8]),
            (
                "Handnote personal outlines / non-learning rotation",
                np.concatenate([fonts, groups["geometric-control"][:4]]),
            ),
            ("Original 120-update pilot", groups["original-pilot-s902"][:8]),
        ]
        rows += [
            (f"{stage}, training seed {seed}", groups[f"{stage}-s{seed}"][:8])
            for stage in ("real-control", "real-prior", "font-finetune")
        ]
        sheet = Image.new("RGB", (1040, len(rows) * 130 + 30), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (title, images) in enumerate(rows):
            y = 8 + j * 130
            draw.text((12, y), title, fill="black")
            for i, im in enumerate(images):
                sheet.paste(bitmap(im).resize((92, 92)), (15 + i * 128, y + 22))
        draw.text(
            (12, len(rows) * 130 + 8),
            "Digit 1 research only. Darkness and pixel diversity are not content/style approval.",
            fill="black",
        )
        sheet.save(path / f"rendered/pages/{seed}.png")
        pages.append(sheet)
    pages[0].save(
        path / "rendered/homework_solutions.pdf",
        save_all=True,
        append_images=pages[1:],
        resolution=150,
    )
    atomic_json(
        path / "reports/comparison.json",
        {
            "groups": reports,
            "cpu_seconds_for_64_images": timings,
            "evaluation_seed": 2902,
            "test_samples": len(test),
            "new_training_updates": 0,
            "selected_before_test": json.loads(
                (study / "reports/candidate-freeze.json").read_text()
            ),
            "quality_gate": "NO-GO production; independent identity/style acceptance missing",
            "novelty_limit": "Different pixels or masks do not rule out memorization",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--test-run", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.study and args.freeze and args.test_run:
        print(evaluate(args.study, args.freeze, args.test_run))
    else:
        parser.error("Expected --study --freeze --test-run")
