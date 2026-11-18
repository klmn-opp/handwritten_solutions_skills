"""CUDA-only worker; all-class sampling and validation snapshots during real optimization."""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json, sha256


def contact_sheet(images: np.ndarray, labels: list[str], path: Path, columns: int = 10):
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 104, rows * 122), "#dddddd")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for index, (ink, label) in enumerate(zip(images, labels, strict=True)):
        x, y = (index % columns) * 104, (index // columns) * 122
        canvas.paste(Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)), (x + 4, y + 22))
        draw.text((x + 5, y + 2), label, font=font, fill="black")
    canvas.save(path)


def main(request_path: Path):
    request = json.loads(request_path.read_text())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from research.corpus.reference_generator import (
        ReferenceCVAE,
        ink_from_sdf,
        objective,
        prior_support_loss,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA device required")
    data_path = Path(request["data"])
    if sha256(data_path / "arrays.npz") != request["arrays_sha256"]:
        raise ValueError("Frozen training arrays changed")
    arrays = np.load(data_path / "arrays.npz", allow_pickle=False)
    labels, config = request["labels"], request["config"]
    run = Path(request["run_path"])
    seed = request["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    ref = torch.tensor(arrays["reference"][:, None], device="cuda")
    target = torch.tensor(arrays["target"][:, None], device="cuda")
    sdf = torch.tensor(arrays["sdf"][:, None], device="cuda")
    classes = torch.tensor(arrays["classes"], device="cuda")
    model = ReferenceCVAE(len(labels)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if request.get("resume"):
        source = Path(request["resume"])
        if sha256(source) != request["resume_sha256"]:
            raise ValueError("Resume checkpoint changed")
        old = torch.load(source, weights_only=True, map_location="cuda")
        from research.corpus.reference_lineage import verify_continuation

        verify_continuation(request, old)
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"]
    initial = [p.detach().clone() for p in model.parameters()]
    torch.save(model.state_dict(), run / "logs/initial-state.pt")
    bins = [
        np.where((arrays["classes"] == i) & (arrays["domains"] == 0))[0] for i in range(len(labels))
    ]
    if any(not len(indices) for indices in bins):
        raise ValueError("All classes must be sampled")
    real_indices = np.where(arrays["domains"] == 1)[0]
    real_bins = [
        real_indices[arrays["classes"][real_indices] == c]
        for c in np.unique(arrays["classes"][real_indices])
    ]
    templates = target[torch.tensor(arrays["domains"] == 0, device="cuda")]
    template_labels = classes[torch.tensor(arrays["domains"] == 0, device="cuda")]
    supports = None
    if config.get("prior_support_weight", 0):
        # Only optimizer inputs are available here; val is loaded separately below.
        groups = [np.where(arrays["classes"] == i)[0] for i in range(len(labels))]
        maximum = max(map(len, groups))
        indices = [np.pad(g, (0, maximum - len(g)), mode="edge") for g in groups]
        supports = sdf[torch.tensor(np.stack(indices), device="cuda")]
    noise = torch.randn(
        len(labels) * 4,
        8,
        generator=torch.Generator(device="cuda").manual_seed(1909),
        device="cuda",
    )
    eval_labels = torch.arange(len(labels), device="cuda").repeat_interleave(4)
    counts = np.zeros(len(labels), dtype=np.int64)
    val = torch.tensor(arrays["validation"][:, None], device="cuda")
    val_labels = arrays["val_classes"]
    best, best_step = float("inf"), 0
    started = time.monotonic()

    def checkpoint(step, score):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.spec,
            "labels": labels,
            "step": step,
            "config": config,
            "seed": seed,
            "arrays_sha256": request["arrays_sha256"],
            "dataset_manifest_sha256": request["manifest_sha256"],
            "score": score,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "parameter_absolute_change": sum(
                (a - b).abs().sum().item() for a, b in zip(initial, model.parameters(), strict=True)
            ),
        }

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        generated = torch.cat(
            [
                ink_from_sdf(
                    model.generate(
                        ref[eval_labels[i : i + 32]], eval_labels[i : i + 32], noise[i : i + 32]
                    )
                )
                for i in range(0, len(noise), 32)
            ]
        )
        # Template matching is a development proxy, not OCR or proof of text correctness.
        comparison = templates if config.get("all_variant_selection") else ref
        comparison_labels = (
            template_labels
            if config.get("all_variant_selection")
            else torch.arange(len(labels), device="cuda")
        )
        distances = torch.cat(
            [
                (
                    (generated[i : i + 16, None] - comparison[None]).abs()
                    * (1 + 4 * generated[i : i + 16, None] + 4 * comparison[None])
                ).mean((2, 3, 4))
                for i in range(0, len(generated), 16)
            ]
        )
        identity = (comparison_labels[distances.argmin(1)] == eval_labels).float().mean().item()
        mask = generated > 0.5
        blank = (~mask.flatten(1).any(1)).float().mean().item()
        border = (
            torch.cat(
                [
                    mask[:, :, :2].flatten(1),
                    mask[:, :, -2:].flatten(1),
                    mask[:, :, :, :2].flatten(1),
                    mask[:, :, :, -2:].flatten(1),
                ],
                1,
            )
            .any(1)
            .float()
            .mean()
            .item()
        )
        grouped = generated.reshape(len(labels), 4, 1, 96, 96)
        val_error = []
        for image, label in zip(val, val_labels, strict=True):
            error = (
                ((grouped[label] - image).abs() * (1 + 4 * grouped[label] + 4 * image))
                .mean((1, 2, 3))
                .min()
            )
            val_error.append(error.item())
        val_mae = float(np.mean(val_error))
        score = val_mae + (1 - identity) * 0.5 + blank + border
        result = {
            "step": step,
            "template_identity_proxy": identity,
            "blank_rate": blank,
            "border_rate": border,
            "real_validation_nearest_mae": val_mae,
            "score": score,
            "elapsed_seconds": time.monotonic() - started,
        }
        contact_sheet(
            grouped[:, 0, 0].cpu().numpy(), labels, run / f"comparisons/step-{step:05d}.png"
        )
        atomic_json(run / f"reports/validation-{step:05d}.json", result)
        model.train()
        return result

    atomic_json(
        run / "reports/environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "total_memory": torch.cuda.get_device_properties(0).total_memory,
            "parameters": sum(p.numel() for p in model.parameters()),
            "architecture": model.spec,
        },
    )
    first = evaluate(0)
    with (run / "logs/curve.jsonl").open("w") as stream:
        for step in range(1, config["steps"] + 1):
            n_real = config["real_per_batch"]
            chosen_classes = np.random.randint(len(labels), size=config["batch_size"] - n_real)
            indices = [np.random.choice(bins[c]) for c in chosen_classes]
            indices += [
                np.random.choice(
                    real_indices
                    if config.get("real_sampling") == "proportional"
                    else real_bins[np.random.randint(len(real_bins))]
                )
                for _ in range(n_real)
            ]
            idx = torch.tensor(indices, device="cuda")
            y = classes[idx]
            counts += np.bincount(arrays["classes"][indices], minlength=len(labels))
            prediction, distributions = model(ref[y], target[idx], y)
            beta = config["beta"] * min(1, step / config["warmup"])
            loss, parts = objective(prediction, sdf[idx], target[idx], distributions, beta)
            if supports is not None:
                prior = model.generate(ref[y], y, torch.randn(len(y), model.latent, device="cuda"))
                support_loss = prior_support_loss(prior, supports[y])
                loss = loss + config["prior_support_weight"] * support_loss
                parts["prior_support"] = support_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            if step % 25 == 0:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "beta": beta,
                            **{k: v.item() for k, v in parts.items()},
                        }
                    )
                    + "\n"
                )
                stream.flush()
            if step % config["evaluate_every"] == 0 or step == config["steps"]:
                metrics = evaluate(step)
                print(json.dumps(metrics), flush=True)
                if step >= config["selection_starts_at"] and metrics["score"] < best:
                    best, best_step = metrics["score"], step
                    torch.save(checkpoint(step, best), run / "logs/best-checkpoint.pt")
        torch.save(checkpoint(step, metrics["score"]), run / "logs/last-checkpoint.pt")
    atomic_json(
        run / "reports/training.json",
        {
            "steps": config["steps"],
            "best_step": best_step,
            "best_score": best,
            "initial_validation": first,
            "final_validation": metrics,
            "training_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "class_exposures": dict(zip(labels, counts.tolist(), strict=True)),
            "parameter_absolute_change": checkpoint(step, best)["parameter_absolute_change"],
            "best_checkpoint_sha256": sha256(run / "logs/best-checkpoint.pt"),
            "last_checkpoint_sha256": sha256(run / "logs/last-checkpoint.pt"),
            "production_eligible": False,
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
