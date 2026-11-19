"""Finite CUDA training of real-supervised, font-constrained contextual contours."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def train(views: Path, mode: str, *, steps: int = 1500) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if mode not in {"local", "context"} or steps <= 0:
        raise ValueError("Invalid finite contour experiment")
    verify_run(views)
    config = {
        "views": str(views.resolve()),
        "arrays_sha256": digest(views / "reports/arrays.npz"),
        "samples_sha256": digest(views / "reports/samples.json"),
        "mode": mode,
        "steps": steps,
        "seed": 1501,
        "lr": 0.0003,
        "bound_pixels": 2.5,
        "size": [1024, 128],
        "batch": 4,
        "new_line_probability": 0.35,
        "teacher_used": False,
        "holdout_used": False,
        "protocol_sha256": digest("docs/V0105_PROTOCOL.md"),
        "architecture": "font-constrained-context-contour/1",
        "production_eligible": False,
        "scope": (
            "learned full-region contour correction; "
            "no random style or free-form ligature generation"
        ),
    }
    run = create("context-contour-training", config=config, seed=config["seed"])
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.context_contour_train",
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

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.context_contour_model import ContourInk
    from research.corpus.levelset_ink import mask_scores

    r = json.loads(path.read_text())
    run, views = Path(r["run"]), Path(r["views"])
    for name, key in [("arrays.npz", "arrays_sha256"), ("samples.json", "samples_sha256")]:
        if digest(views / "reports" / name) != r[key]:
            raise ValueError("Frozen source changed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(r["seed"])
    torch.cuda.manual_seed_all(r["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = np.load(views / "reports/arrays.npz")
    rows = json.loads((views / "reports/samples.json").read_text())
    if data["reference"].shape[1:] != (128, 1024):
        raise ValueError("Native high-resolution views required")
    train_ids = np.where((data["split"] == "train") & data["admitted"])[0]
    val_ids = np.where((data["split"] == "validation") & data["admitted"])[0]
    if any(s not in {"train", "validation"} for s in data["split"]):
        raise ValueError("Holdout cannot enter worker")
    new_ids = np.array(
        [i for i in train_ids if "new-english-sol-review" in rows[i]["source_package"]]
    )
    if not len(new_ids):
        raise ValueError("New genuine English material required")
    old_ids = np.array([i for i in train_ids if i not in new_ids])

    def distance(ink):
        mask = np.uint8(ink > 0.5)
        return (
            cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        ).clip(-16, 16)

    refs = torch.tensor(np.stack([distance(x) for x in data["reference"]])[:, None], device="cuda")
    targets = torch.tensor(data["aligned"][:, None], device="cuda")
    target_sdf = torch.tensor(
        np.stack([distance(x) for x in data["aligned"]])[:, None], device="cuda"
    )
    model = ContourInk(context=r["mode"] == "context", bound=r["bound_pixels"]).cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    curve, evals = [], []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        outputs, deltas = [], []
        for i in range(0, len(rows), 4):
            ink, delta, _ = model(refs[i : i + 4])
            outputs.append(ink[:, 0].cpu().numpy())
            deltas.append(delta[:, 0].cpu().numpy())
        outputs, deltas = np.concatenate(outputs), np.concatenate(deltas)
        result = {"step": step}
        for name, ids in [("train", train_ids), ("validation", val_ids), ("new_train", new_ids)]:
            scores = [mask_scores(outputs[i], data["aligned"][i]) for i in ids]
            result[name] = {
                k: float(np.mean([q[k] for q in scores]))
                for k in ["binary_dice", "binary_iou", "target_core_recall"]
            }
            result[name]["correction_abs"] = float(np.mean(np.abs(deltas[ids])))
            result[name]["blank"] = int(sum(outputs[i].max() < 0.5 for i in ids))
        evals.append(result)
        if step in [0, r["steps"]]:
            np.savez_compressed(
                run / f"reports/generated-{step}.npz", ink=outputs, correction=deltas
            )
            for i in range(len(rows)):
                Image.fromarray(np.uint8((1 - outputs[i].clip(0, 1)) * 255)).save(
                    run / f"rendered/pages/step-{step}-{i:03d}.png"
                )
        write(run / "reports/evaluations.json", evals)
        print(json.dumps(result), flush=True)
        model.train()

    evaluate(0)
    for step in range(1, r["steps"] + 1):
        ids = [
            int(rng.choice(new_ids if rng.random() < r["new_line_probability"] else old_ids))
            for _ in range(r["batch"])
        ]
        optimizer.zero_grad(set_to_none=True)
        ink, delta, field = model(refs[ids])
        target = targets[ids]
        weight = torch.exp(-refs[ids].abs() / 4)
        contour = (
            F.smooth_l1_loss(field / 8, target_sdf[ids] / 8, reduction="none") * weight
        ).sum() / weight.sum().clamp_min(1)
        dice = 1 - (2 * (ink * target).sum((1, 2, 3)) + 1) / (
            ink.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1
        )
        # Small reference-support constraint protects fine dots while fitting real ink.
        support = ((refs[ids] > 0) * F.relu(0.55 - ink)).sum() / (refs[ids] > 0).sum().clamp_min(1)
        smooth = (delta[..., 1:] - delta[..., :-1]).square().mean() + (
            delta[..., 1:, :] - delta[..., :-1, :]
        ).square().mean()
        loss = contour + 0.3 * dice.mean() + 0.15 * support + 0.01 * smooth
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "contour": float(contour.detach()),
                    "dice": float(dice.mean().detach()),
                    "support": float(support.detach()),
                    "seconds": time.perf_counter() - start,
                }
            )
            write(run / "reports/curve.json", curve)
        if step % 500 == 0 or step == r["steps"]:
            evaluate(step)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state,
                },
                run / f"logs/checkpoint-{step:04d}.pt",
            )
    torch.cuda.synchronize()
    checkpoint = run / f"logs/checkpoint-{r['steps']:04d}.pt"
    changed = sum((v - initial[k]).abs().sum().item() for k, v in model.state_dict().items())
    result = {
        "architecture": r["architecture"],
        "actual_parameter_training": True,
        "executed_updates": r["steps"],
        "parameter_absolute_change": changed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_count": len(train_ids),
        "new_train_count": len(new_ids),
        "validation_admitted_count": len(val_ids),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "seconds": time.perf_counter() - start,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "dependencies": {
            k: importlib.metadata.version(k)
            for k in ["torch", "numpy", "pillow", "opencv-python"]
            if k != "opencv-python"
        },
        "cuda": torch.version.cuda,
        "teacher_used": False,
        "holdout_opened": False,
        "production_eligible": False,
        "scope": r["scope"],
    }
    if changed <= 0:
        raise RuntimeError("No actual parameter update")
    write(run / "reports/training.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker", "local", "context"])
    p.add_argument("path", type=Path)
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.path)
    else:
        print(train(a.path, a.mode))
