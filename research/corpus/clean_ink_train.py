"""Bounded fine-tuning of the approved text model, without changing production rendering."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def train(arm: str) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if arm not in {"control", "distance", "mass"}:
        raise ValueError("Unknown fixed arm")
    views = Path(
        "workspace/runs/20261119T031307931151Z-context-word-aligned-views-e393b3f7"
    ).resolve()
    source = Path(
        "workspace/runs/20261119T032117530522Z-context-content-training-6b8f516f"
    ).resolve()
    for path in [views, source]:
        verify_run(path)
    checkpoint = source / "logs/checkpoint-2000.pt"
    teacher = Path("workspace/checkpoints/trocr-base-handwritten-v08-xet").resolve()
    config = {
        "arm": arm,
        "views": str(views),
        "views_sha256": digest(views / "run.json"),
        "arrays_sha256": digest(views / "reports/arrays.npz"),
        "samples_sha256": digest(views / "reports/samples.json"),
        "initial_checkpoint": str(checkpoint),
        "initial_sha256": digest(checkpoint),
        "steps": 1500,
        "seed": 1203,
        "batch": 4,
        "real_per_batch": 3,
        "lr": 0.0001,
        "distance_weight": 0 if arm == "control" else 1.0,
        "mass_weight": 0.15 if arm == "mass" else 0,
        "content_weight": 0.15,
        "content_every": 4,
        "dual_content_view": arm == "mass",
        "teacher": str(teacher),
        "teacher_sha256": {
            str(p.relative_to(teacher)): digest(p) for p in teacher.rglob("*") if p.is_file()
        },
        "content_label_protocol": "content-then-eos/no-predicted-bos/1",
        "protocol_sha256": digest("docs/V0102_PROTOCOL.md"),
        "optimizer": "new AdamW for changed objective; not exact continuation of old optimizer",
        "checkpoint_selection": "fixed final; all candidates frozen before new regions",
        "self_targets_are_real_labels": False,
        "default_output_contrast_changed": False,
    }
    run = create("context-clean-ink-training", seed=1203, config=config)
    request = {**config, "run": str(run.path)}
    write(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.clean_ink_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        result = json.loads((run.path / "reports/training.json").read_text())
        run.finish("held", result=result)
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
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.clean_ink import ink_metrics, redistance_batch
    from research.corpus.context_content_model import ContentInk, FrozenTranscriptionLoss
    from research.corpus.reference_generator import ink_from_sdf

    os.environ["HF_HUB_OFFLINE"] = "1"
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
            raise ValueError("Changed training source")
    for rel, h in r["teacher_sha256"].items():
        if digest(Path(r["teacher"]) / rel) != h:
            raise ValueError("Changed frozen teacher")
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(r["seed"])
    torch.cuda.manual_seed_all(r["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(r["seed"])
    data = np.load(views / "reports/arrays.npz")
    rows = json.loads((views / "reports/samples.json").read_text())
    ids = np.where((data["split"] == "train") & data["admitted"])[0]
    val = np.where(data["split"] == "validation")[0]
    if len(ids) != 171 or len(val) != 25:
        raise ValueError("Fixed comparison dataset differs")
    x = {
        k: torch.tensor(data[k][:, None], device="cuda")
        for k in ["reference", "aligned", "aligned_sdf", "reference_sdf"]
    }
    model = ContentInk().cuda()
    model.load_state_dict(
        torch.load(r["initial_checkpoint"], map_location="cuda", weights_only=True)["model"]
    )
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    teacher = FrozenTranscriptionLoss(r["teacher"])
    labels = teacher.labels([q["normalized_label"] for q in rows])
    teacher_initial = {n: p.detach().cpu().clone() for n, p in teacher.model.named_parameters()}
    curve = []
    evaluations = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def evaluate(step: int) -> None:
        model.eval()
        outputs = []
        for i in range(0, len(val), 8):
            outputs.append(ink_from_sdf(model(x["reference"][val[i : i + 8]])))
        ink = torch.cat(outputs)[:, 0].cpu().numpy()
        metrics = []
        for j, img in enumerate(ink):
            Image.fromarray(np.uint8((1 - img.clip(0, 1)) * 255)).save(
                run / f"comparisons/step-{step:04d}-val-{j:02d}.png"
            )
            metrics.append({"sample_id": rows[val[j]]["sample_id"], **ink_metrics(img)})
        result = {
            "step": step,
            "soft_fraction": float(np.mean([q["soft_fraction"] for q in metrics])),
            "blank": sum(q["blank"] for q in metrics),
            "ink_mass": float(np.mean([q["ink_mass"] for q in metrics])),
        }
        evaluations.append(result)
        write(run / f"reports/validation-{step:04d}.json", metrics)
        model.train()

    evaluate(0)
    for step in range(1, r["steps"] + 1):
        chosen = torch.tensor(rng.choice(ids, 4, replace=True), device="cuda")
        ref = x["reference"][chosen]
        target = x["aligned"][chosen].clone()
        sdf = x["aligned_sdf"][chosen].clone()
        target[3:] = ref[3:]
        sdf[3:] = x["reference_sdf"][chosen[3:]]
        prediction = model(ref)
        ink = ink_from_sdf(prediction)
        boundary = 1 + 4 * (sdf.abs() < 0.35)
        sdf_loss = ((prediction - sdf).abs() * boundary).mean()
        bce = (F.binary_cross_entropy(ink, target, reduction="none") * (1 + 5 * target)).mean()
        dice = (
            1
            - (2 * (ink * target).sum((1, 2, 3)) + 1)
            / (ink.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        ).mean()
        loss = sdf_loss + 0.3 * bce + 0.2 * dice
        distance = torch.zeros((), device="cuda")
        mass = torch.zeros((), device="cuda")
        content = torch.zeros((), device="cuda")
        if r["distance_weight"]:
            re, valid = redistance_batch(prediction.detach().cpu().numpy())
            re = torch.tensor(re, device="cuda")
            valid = torch.tensor(valid, device="cuda")
            per = ((prediction - re).abs() * (1 + 4 * (re.abs() < 0.35))).mean((1, 2, 3))
            distance = (per * valid).sum() / valid.sum().clamp_min(1)
            loss = loss + r["distance_weight"] * distance
        if r["mass_weight"]:
            mass = (
                (ink.sum((1, 2, 3)) - target.sum((1, 2, 3))).abs() / (target.sum((1, 2, 3)) + 1)
            ).mean()
            loss = loss + r["mass_weight"] * mass
        if step % r["content_every"] == 0:
            content = teacher(ink[:2], labels[chosen[:2]])
            if r["dual_content_view"]:
                sharp = torch.sigmoid(prediction[:2] * (8 / 0.35) * 3)
                content = (content + teacher(sharp, labels[chosen[:2]])) / 2
            loss = loss + r["content_weight"] * content
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            curve.append(
                {
                    "step": step,
                    **{
                        k: float(v.detach())
                        for k, v in [
                            ("loss", loss),
                            ("sdf", sdf_loss),
                            ("bce", bce),
                            ("dice", dice),
                            ("distance", distance),
                            ("mass", mass),
                            ("content", content),
                        ]
                    },
                }
            )
        if step % 500 == 0 or step == r["steps"]:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "architecture": model.spec,
                    "request": r,
                    "step": step,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                    "rng_numpy": json.dumps(rng.bit_generator.state),
                },
                run / f"logs/checkpoint-{step:04d}.pt",
            )
            evaluate(step)
            write(run / "reports/curve.json", curve)
            write(run / "reports/evaluations.json", evaluations)
            print(json.dumps(evaluations[-1]), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    frozen = all(
        torch.equal(p.detach().cpu(), teacher_initial[n]) and p.grad is None
        for n, p in teacher.model.named_parameters()
    )
    assert frozen
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
            "arm": r["arm"],
            "steps": r["steps"],
            "executed_updates": r["steps"],
            "train_real_regions": len(ids),
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds": elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "parameter_absolute_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "teacher_parameters_unchanged": frozen,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "final_validation": evaluations[-1],
            "stochastic_style": False,
            "self_targets_are_real_labels": False,
            "production_eligible": False,
            "default_output_contrast_changed": False,
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
