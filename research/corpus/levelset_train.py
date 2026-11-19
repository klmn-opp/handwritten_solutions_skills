"""Finite contour-learning diagnostics and opt-in full-data CUDA experiments."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def train(mode: str, *, stage: str = "capacity") -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if mode not in {"soft", "projected"} or stage != "capacity":
        raise ValueError("Only predeclared capacity stage currently enabled")
    views = Path("workspace/runs/20261119T031307931151Z-context-word-aligned-views-e393b3f7")
    source = Path("workspace/runs/20261119T035904738321Z-context-clean-ink-training-5758634f")
    for p in [views, source]:
        verify_run(p)
    checkpoint = (source / "logs/checkpoint-1500.pt").resolve()
    config = {
        "stage": stage,
        "mode": mode,
        "steps": 1200,
        "seed": 1401,
        "lr": 0.0001,
        "views": str(views.resolve()),
        "arrays_sha256": digest(views / "reports/arrays.npz"),
        "samples_sha256": digest(views / "reports/samples.json"),
        "initial_checkpoint": str(checkpoint),
        "initial_sha256": digest(checkpoint),
        "protocol_sha256": digest("docs/V0104_PROTOCOL.md"),
        "optimizer": "new AdamW, equal data/order/steps across modes",
        "target": "derived distance raster from real aligned scan mask, not a new scan",
        "teacher_used": False,
        "holdout_used": False,
        "production_eligible": False,
    }
    run = create("context-levelset-capacity-training", seed=config["seed"], config=config)
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.levelset_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        run.finish("held", result=json.loads((run.path / "reports/training.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(path: Path) -> None:
    import importlib.metadata
    import os
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.clean_ink import ink_metrics
    from research.corpus.context_content_model import ContentInk
    from research.corpus.levelset_ink import mask_scores, render_field
    from research.corpus.reference_generator import ink_from_sdf

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    r = json.loads(path.read_text())
    run = Path(r["run"])
    views = Path(r["views"])
    for p, h in [
        (views / "reports/arrays.npz", r["arrays_sha256"]),
        (views / "reports/samples.json", r["samples_sha256"]),
        (r["initial_checkpoint"], r["initial_sha256"]),
    ]:
        if digest(p) != h:
            raise ValueError("Changed frozen source")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(r["seed"])
    torch.cuda.manual_seed_all(r["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = np.load(views / "reports/arrays.npz")
    rows = json.loads((views / "reports/samples.json").read_text())
    unique = {}
    for i, row in enumerate(rows):
        label = row["normalized_label"]
        if data["split"][i] == "train" and data["admitted"][i] and 3 <= len(label) <= 32:
            unique.setdefault(label, i)
    rng = np.random.default_rng(r["seed"])
    words = sorted(i for s, i in unique.items() if len(s.split()) == 1 and len(s) <= 12)
    phrases = sorted(
        (i for s, i in unique.items() if 2 <= len(s.split()) <= 4),
        key=lambda i: (len(rows[i]["normalized_label"]), rows[i]["sample_id"]),
    )
    ids = sorted(rng.choice(words, 16, replace=False).tolist() + phrases[:8])
    assert len(ids) == 24 and len(set(ids)) == 24
    write(run / "reports/subset.json", [{"index": i, **rows[i]} for i in ids])
    ref = torch.tensor(data["reference"][ids, None], device="cuda")
    sdf = torch.tensor(data["aligned_sdf"][ids, None], device="cuda")
    target = ink_from_sdf(sdf)
    model = ContentInk().cuda()
    model.load_state_dict(
        torch.load(r["initial_checkpoint"], map_location="cuda", weights_only=True)["model"]
    )
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    curves = []
    evaluations = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        records = []
        images = []
        for n in range(0, 24, 4):
            images.append(render_field(model(ref[n : n + 4]), r["mode"]))
        images = torch.cat(images)[:, 0].cpu().numpy()
        targets = target[:, 0].cpu().numpy()
        for n, img in enumerate(images):
            records.append({"index": ids[n], **ink_metrics(img), **mask_scores(img, targets[n])})
            Image.fromarray(np.uint8((1 - img.clip(0, 1)) * 255)).save(
                run / f"comparisons/step-{step:04d}-{n:02d}.png"
            )
        result = {
            "step": step,
            **{
                k: float(np.mean([q[k] for q in records]))
                for k in [
                    "binary_iou",
                    "binary_dice",
                    "target_core_recall",
                    "generated_core_precision",
                    "ink_mass",
                    "soft_fraction",
                ]
            },
        }
        write(run / f"reports/evaluation-{step:04d}.json", records)
        evaluations.append(result)
        model.train()
        return result

    evaluate(0)
    for step in range(1, r["steps"] + 1):
        chosen = torch.tensor(rng.choice(24, 4, replace=True), device="cuda")
        field = model(ref[chosen])
        ink = render_field(field, r["mode"])
        truth = target[chosen]
        distance = ((field - sdf[chosen]).abs() * (1 + 4 * (sdf[chosen].abs() < 0.35))).mean()
        image = ((ink - truth).square() * (1 + 8 * truth)).mean()
        dice = (
            1
            - (2 * (ink * truth).sum((1, 2, 3)) + 1)
            / (ink.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
        ).mean()
        loss = 0.5 * distance + 0.5 * image + 0.5 * dice
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            curves.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "sdf": distance.item(),
                    "image": image.item(),
                    "dice": dice.item(),
                }
            )
        if step % 400 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "architecture": {**model.spec, "renderer": r["mode"]},
                    "step": step,
                    "request": r,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                    "rng_numpy": json.dumps(rng.bit_generator.state),
                },
                run / f"logs/checkpoint-{step:04d}.pt",
            )
            print(json.dumps(evaluate(step)), flush=True)
            write(run / "reports/curve.json", curves)
            write(run / "reports/evaluations.json", evaluations)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    checkpoint = run / f"logs/checkpoint-{r['steps']:04d}.pt"
    write(
        run / "reports/environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    )
    write(
        run / "reports/training.json",
        {
            "actual_parameter_training": True,
            "stage": r["stage"],
            "mode": r["mode"],
            "executed_updates": r["steps"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_absolute_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "seconds": elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "final_training_metrics": evaluations[-1],
            "memorization_diagnostic_not_generalization": True,
            "holdout_used": False,
            "production_eligible": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "worker"])
    p.add_argument("value")
    a = p.parse_args()
    if a.mode == "worker":
        worker(Path(a.value))
    else:
        print(train(a.value))
