"""Bounded whole-region supervised training, with an optional frozen content teacher."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def train(views: Path, arm: str, warmup: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(views)
    if arm not in {"font-warmup", "raw", "aligned", "aligned-content"}:
        raise ValueError("Unknown fixed arm")
    if (arm == "font-warmup") != (warmup is None):
        raise ValueError("Every real arm needs identical font warmup")
    teacher = Path("workspace/checkpoints/trocr-base-handwritten-v08-xet").resolve()
    config = {
        "views": str(views.resolve()),
        "views_sha256": digest(views / "run.json"),
        "arrays_sha256": digest(views / "reports/arrays.npz"),
        "arm": arm,
        "steps": 1500 if arm == "font-warmup" else 2000,
        "batch": 4,
        "seed": 1103,
        "lr": 0.0003,
        "real_per_batch": 0 if arm == "font-warmup" else 3,
        "content_weight": 0.15 if arm == "aligned-content" else 0,
        "content_every": 4,
        "teacher": str(teacher),
        "warmup": str(warmup.resolve()) if warmup else None,
        "warmup_sha256": digest(warmup) if warmup else None,
        "checkpoint_selection": "fixed final, no winner selection",
        "stochastic_style": False,
        "teacher_sha256": {
            str(p.relative_to(teacher)): digest(p) for p in teacher.rglob("*") if p.is_file()
        }
        if arm == "aligned-content"
        else None,
    }
    run = create("context-content-training", seed=1103, config=config)
    request = {**config, "run": str(run.path)}
    write(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.context_content_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
        result = json.loads((run.path / "reports/training.json").read_text())
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request_path: Path):
    import importlib.metadata
    import os
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.context_content_model import ContentInk, FrozenTranscriptionLoss
    from research.corpus.reference_generator import ink_from_sdf

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    r = json.loads(request_path.read_text())
    run = Path(r["run"])
    views = Path(r["views"])
    if digest(views / "reports/arrays.npz") != r["arrays_sha256"]:
        raise ValueError("Data mutated")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(r["seed"])
    torch.cuda.manual_seed_all(r["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    data = np.load(views / "reports/arrays.npz", allow_pickle=False)
    rows = json.loads((views / "reports/samples.json").read_text())
    ids = np.where((data["split"] == "train") & data["admitted"])[0]
    val = np.where(data["split"] == "validation")[0]
    if len(ids) < 60 or not len(val):
        raise ValueError("Insufficient admitted real regions")
    rng = np.random.default_rng(r["seed"])
    x = {
        k: torch.tensor(data[k][:, None], device="cuda")
        for k in ["reference", "raw", "aligned", "raw_sdf", "aligned_sdf", "reference_sdf"]
    }
    model = ContentInk().cuda()
    if r["warmup"]:
        if digest(r["warmup"]) != r["warmup_sha256"]:
            raise ValueError("Warmup mutated")
        model.load_state_dict(
            torch.load(r["warmup"], map_location="cuda", weights_only=True)["model"]
        )
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    teacher = FrozenTranscriptionLoss(r["teacher"]) if r["content_weight"] else None
    labels = teacher.labels([q["normalized_label"] for q in rows]) if teacher else None
    teacher_initial = (
        {n: p.detach().clone().cpu() for n, p in teacher.model.named_parameters()}
        if teacher
        else None
    )
    history = []
    evaluations = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def save(step):
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

    @torch.inference_mode()
    def evaluate(step):
        model.eval()
        metrics = {}
        for split, index in [("train", ids[:16]), ("validation", val)]:
            outputs = []
            for i in range(0, len(index), 8):
                outputs.append(ink_from_sdf(model(x["reference"][index[i : i + 8]])))
            ink = torch.cat(outputs)
            real = x["raw"][index]
            aligned = x["aligned"][index]
            metrics[split] = {
                "raw_shape": float(((ink - real).abs() * (1 + 4 * ink + 4 * real)).mean()),
                "aligned_shape": float(
                    ((ink - aligned).abs() * (1 + 4 * ink + 4 * aligned)).mean()
                ),
                "blank": int((ink.flatten(1).max(1).values < 0.5).sum()),
            }
            if split == "validation":
                for j, img in enumerate(ink[:, 0].cpu().numpy()):
                    Image.fromarray(np.uint8((1 - img.clip(0, 1)) * 255)).save(
                        run / f"comparisons/step-{step:04d}-val-{j:02d}.png"
                    )
        evaluations.append({"step": step, **metrics})
        write(run / f"reports/validation-{step:04d}.json", metrics)
        model.train()

    evaluate(0)
    for step in range(1, r["steps"] + 1):
        chosen = torch.tensor(rng.choice(ids, 4, replace=True), device="cuda")
        ref = x["reference"][chosen]
        target_key = "aligned" if r["arm"].startswith("aligned") else "raw"
        target = x[target_key][chosen].clone()
        sdf = x[target_key + "_sdf"][chosen].clone()
        real = r["real_per_batch"]
        target[real:] = ref[real:]
        sdf[real:] = x["reference_sdf"][chosen[real:]]
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
        content = torch.zeros((), device="cuda")
        if teacher and step % r["content_every"] == 0:
            content = teacher(ink[:2], labels[chosen[:2]])
            loss = loss + r["content_weight"] * content
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "sdf": sdf_loss.item(),
                    "bce": bce.item(),
                    "dice": dice.item(),
                    "content": content.item(),
                }
            )
        if step % 500 == 0 or step == r["steps"]:
            save(step)
            evaluate(step)
            write(run / "reports/curve.json", history)
            print(json.dumps(evaluations[-1]), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if teacher:
        frozen = all(
            torch.equal(p.detach().cpu(), teacher_initial[n])
            for n, p in teacher.model.named_parameters()
        )
        assert frozen and all(p.grad is None for p in teacher.model.parameters())
    else:
        frozen = None
    write(run / "reports/evaluations.json", evaluations)
    write(
        run / "reports/environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    )
    checkpoint = run / f"logs/checkpoint-{r['steps']:04d}.pt"
    write(
        run / "reports/training.json",
        {
            "actual_parameter_training": True,
            "arm": r["arm"],
            "steps": r["steps"],
            "train_real_regions": len(ids) if real else 0,
            "font_text_conditions": len(ids),
            "seconds": elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "parameter_absolute_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "teacher_parameters_unchanged": frozen,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "production_eligible": False,
            "stochastic_style": False,
            "final_validation": evaluations[-1]["validation"],
            "font_only_warmup": real == 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "worker"])
    p.add_argument("path", type=Path)
    p.add_argument("--arm")
    p.add_argument("--warmup", type=Path)
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.path)
    else:
        print(train(a.path, a.arm, a.warmup))
