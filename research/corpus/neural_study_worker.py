"""Real CUDA optimization worker for the frozen Handnote transfer protocol."""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.label_admission import label_text
from research.corpus.neural_study import validate_request
from research.corpus.training_data import ink_view


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    validate_request(request)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from research.corpus.generator import GlyphCVAE, objective

    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA training is required")
    torch.set_num_threads(2)
    seed = request["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    run = Path(request["run_path"])
    protocol, stage, labels = request["protocol"], request["stage"], request["labels"]
    font_stage = stage == "font-pretrain"
    rows = (
        request["font_rows"]
        if font_stage
        else [r for r in request["real_rows"] if r["dataset_split"] == "train"]
    )
    arrays, classes = [], []
    for row in rows:
        if font_stage:
            with Image.open(row["image"]) as im:
                arrays.append(1 - np.asarray(im.convert("L"), dtype=np.float32) / 255)
            classes.append(labels.index(row["label"]))
        else:
            arrays.append(ink_view(Path(row["image_crop"]), protocol["preprocessing"]))
            classes.append(labels.index(label_text(row)))
    train_x = torch.tensor(np.stack(arrays)[:, None], device="cuda")
    train_y = torch.tensor(classes, device="cuda")
    validation = [r for r in request["real_rows"] if r["dataset_split"] == "validation"]
    val_x = torch.tensor(
        np.stack([ink_view(Path(r["image_crop"]), protocol["preprocessing"]) for r in validation])[
            :, None
        ],
        device="cuda",
    )
    # Font pretraining never uses validation for optimization or model selection.
    model = GlyphCVAE(len(labels)).cuda()
    source = Path(request["initial_run"]) / "logs/best-checkpoint.pt"
    initial = torch.load(source, map_location="cuda", weights_only=True)
    state = dict(initial["model"])
    old_embedding = state.pop("character.weight")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["character.weight"] or unexpected:
        raise ValueError("Incompatible initial architecture")
    with torch.no_grad():
        for index, label in enumerate(initial["labels"]):
            model.character.weight[labels.index(label)].copy_(old_embedding[index])
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"])
    torch.save(model.state_dict(), run / "logs/initial-state.pt")
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    steps = protocol["font_steps"] if font_stage else protocol["real_steps"]
    target_beta = (
        protocol["font_beta"]
        if font_stage
        else protocol["control_beta" if stage == "real-control" else "corrected_beta"]
    )
    noise = torch.Generator(device="cuda").manual_seed(protocol["validation_prior_seed"])
    eval_z = torch.randn(protocol["validation_prior_count"], 16, generator=noise, device="cuda")

    def condition(n, values=None):
        char = (
            values
            if values is not None
            else torch.full((n,), labels.index("1"), dtype=torch.long, device="cuda")
        )
        return (
            char,
            torch.zeros(n, dtype=torch.long, device="cuda"),
            torch.zeros(n, 4, device="cuda"),
        )

    def validation_metrics():
        model.eval()
        with torch.inference_mode():
            cond = condition(len(val_x))
            mu, lv = model.posterior(
                torch.cat([model.encoder(val_x), model.condition(*cond)], dim=1)
            ).chunk(2, dim=1)
            prediction = model.generate(*cond, mu)
            _, parts = objective(prediction, val_x, mu, lv.clamp(-12, 8), target_beta)
            prior = model.generate(*condition(len(eval_z)), eval_z)
            differences = (prior[:, None] - val_x[None]).abs() * (1 + 8 * val_x[None])
            nearest = differences.mean(dim=(2, 3, 4)).min(dim=0).values.mean()
            mask = prior[:, 0] > 0.5
            blank = (~mask.flatten(1).any(1)).float().mean()
            border = torch.cat(
                [
                    mask[:, :2].flatten(1),
                    mask[:, -2:].flatten(1),
                    mask[:, :, :2].flatten(1),
                    mask[:, :, -2:].flatten(1),
                ],
                dim=1,
            )
            border = border.any(1).float().mean()
            score = parts["reconstruction"] + nearest + 0.5 * (blank + border)
            return {
                "score": score.item(),
                "reconstruction": parts["reconstruction"].item(),
                "kl_per_dim": parts["kl"].item(),
                "posterior_mean_norm": mu.norm(dim=1).mean().item(),
                "posterior_aggregate_mean_norm": mu.mean(0).norm().item(),
                "prior_nearest_weighted_mae": nearest.item(),
                "prior_blank_rate": blank.item(),
                "prior_border_rate": border.item(),
                "prior_peak_mean": prior.flatten(1).max(1).values.mean().item(),
                "prior_ink_fraction": mask.float().mean().item(),
            }

    def checkpoint(step, score):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.spec,
            "labels": labels,
            "seed": seed,
            "step": step,
            "selection_score": score,
            "stage": stage,
            "protocol": protocol,
            "initial_checkpoint_sha256": sha256(source),
            "dataset_manifest_sha256": sha256(Path(request["dataset"]) / "manifest.json"),
            "auxiliary_manifest_sha256": sha256(Path(request["auxiliary"]) / "manifest.json"),
            "preprocessing": protocol["preprocessing"],
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "parameter_absolute_change": sum(
                (a - b).abs().sum().item()
                for a, b in zip(initial_parameters, model.parameters(), strict=True)
            ),
        }

    atomic_json(
        run / "reports/architecture.json",
        {
            **model.spec,
            "parameters": sum(p.numel() for p in model.parameters()),
            "font_labels": labels,
            "real_supervised_labels": [] if font_stage else ["1"],
            "context": "zeros; no learned connected-writing context claimed",
        },
    )
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    curves, best, best_step = [], float("inf"), None
    order, cursor = torch.randperm(len(rows), device="cuda"), 0
    for step in range(1, steps + 1):
        if cursor >= len(rows):
            order, cursor = torch.randperm(len(rows), device="cuda"), 0
        ids = order[cursor : cursor + protocol["batch_size"]]
        cursor += len(ids)
        model.train()
        pred, mu, lv = model(train_x[ids], *condition(len(ids), train_y[ids]))
        beta = 0.001 + (target_beta - 0.001) * min(1, step / protocol["warmup_steps"])
        loss, parts = objective(pred, train_x[ids], mu, lv, beta)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite optimization loss")
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        row = {
            "step": step,
            "loss": loss.item(),
            "reconstruction": parts["reconstruction"].item(),
            "kl_per_dim": parts["kl"].item(),
            "beta": beta,
            "gradient_norm": norm.item(),
        }
        if step % protocol["evaluate_every"] == 0:
            if not font_stage:
                row["validation"] = validation_metrics()
                if step >= protocol["selection_starts_at"] and row["validation"]["score"] < best:
                    best, best_step = row["validation"]["score"], step
                    torch.save(checkpoint(step, best), run / "logs/best-checkpoint.pt")
            curves.append(row)
            atomic_json(run / "reports/curves.json", curves)
    torch.cuda.synchronize()
    last = checkpoint(steps, None)
    torch.save(last, run / "logs/last-checkpoint.pt")
    if font_stage:
        torch.save(last, run / "logs/best-checkpoint.pt")
        best_step = steps  # Fixed-budget final; no real validation model selection.
    saved = torch.load(run / "logs/best-checkpoint.pt", weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    with torch.inference_mode():
        z = torch.randn(
            16, 16, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1902)
        )
        generated = model.generate(*condition(16), z)[:, 0].cpu().numpy()
    sheet = Image.new("RGB", (960, 290), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{stage}, seed={seed}, digit 1 only; quality not accepted", fill="black")
    for i, values in enumerate(generated):
        im = Image.fromarray(np.rint((1 - values.clip(0, 1)) * 255).astype("uint8"))
        im.save(run / f"rendered/pages/sample-{i:02d}.png")
        sheet.paste(im.resize((100, 100)), (10 + (i % 8) * 118, 34 + (i // 8) * 120))
    sheet.save(run / "comparisons/training-preview.png")
    sheet.save(run / "rendered/homework_solutions.pdf", resolution=150)
    atomic_json(
        run / "reports/training.json",
        {
            "training_executed": True,
            "stage": stage,
            "optimizer_updates": steps,
            "parameter_absolute_change": last["parameter_absolute_change"],
            "selected_parameter_absolute_change": saved["parameter_absolute_change"],
            "best_step": best_step,
            "selection_score": None if font_stage else best,
            "selection_is_content_or_style_acceptance": False,
            "initial_sha256": sha256(run / "logs/initial-state.pt"),
            "checkpoint_sha256": sha256(run / "logs/best-checkpoint.pt"),
            "source_checkpoint_sha256": sha256(source),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "real_train_samples": 0 if font_stage else len(rows),
            "font_derived_train_samples": len(rows) if font_stage else 0,
            "validation_samples_for_selection": 0 if font_stage else len(validation),
            "holdout_used": False,
            "new_default_published": False,
        },
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
