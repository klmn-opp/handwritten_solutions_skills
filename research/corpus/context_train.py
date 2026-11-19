"""Bounded equal-update whole-region pilot; each candidate remains held, no winner selection."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create

ARMS = {
    "context": {"context": True, "global_style": True, "real_per_batch": 3, "support": 0.0},
    "no-context": {"context": False, "global_style": True, "real_per_batch": 3, "support": 0.0},
    "no-style": {"context": True, "global_style": False, "real_per_batch": 3, "support": 0.0},
    "support": {"context": True, "global_style": True, "real_per_batch": 3, "support": 0.05},
    "half-real": {"context": True, "global_style": True, "real_per_batch": 2, "support": 0.0},
}


def train(views: Path, arm: str, steps: int = 800) -> Path:
    verify_run(views)
    if arm not in ARMS or not 200 <= steps <= 1200:
        raise ValueError("Outside bounded context experiment")
    run = create(
        "context-region-training",
        seed=1011,
        config={
            "views": str(views.resolve()),
            "views_sha256": sha256(views / "run.json"),
            "arm": arm,
            **ARMS[arm],
            "steps": steps,
            "batch_size": 4,
            "learning_rate": 0.0003,
            "checkpoint_rule": "fixed final step, no best/winner selected",
            "fresh_document_holdout": False,
            "production_eligible": False,
        },
    )
    request = {
        **run.record()["config"],
        "seed": 1011,
        "run": str(run.path),
        "arrays_sha256": sha256(views / "reports/arrays.npz"),
    }
    atomic_json(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/training.log").open("w") as stream:
            subprocess.run(
                [
                    str(Path("workspace/environments/corpus-generator-v09/bin/python").absolute()),
                    "-m",
                    "research.corpus.context_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                timeout=1800,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        summary = json.loads((run.path / "reports/training.json").read_text())
        run.finish("held", result=summary)
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


def worker(request_path: Path) -> None:
    import os
    import time

    import numpy as np
    from PIL import Image

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from research.corpus.context_model import ContextInk
    from research.corpus.reference_generator import ink_from_sdf, objective

    request = json.loads(request_path.read_text())
    run = Path(request["run"])
    views = Path(request["views"])
    if sha256(views / "reports/arrays.npz") != request["arrays_sha256"]:
        raise ValueError("Frozen context views changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(request["seed"])
    data = np.load(views / "reports/arrays.npz", allow_pickle=False)
    train_ids = np.where(data["split"] == "train")[0]
    val_ids = np.where(data["split"] == "validation")[0]
    if not len(val_ids) or set(train_ids) & set(val_ids):
        raise ValueError("Invalid context split")
    tensors = {
        k: torch.tensor(data[k][:, None], device="cuda")
        for k in ["target", "target_sdf", "reference", "reference_sdf"]
    }
    roles = torch.tensor(data["role"], device="cuda")
    model = ContextInk(request["context"], request["global_style"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=request["learning_rate"])
    initial = [p.detach().clone() for p in model.parameters()]
    torch.save(model.state_dict(), run / "logs/initial-state.pt")
    history = []
    started = time.perf_counter()

    def save(step: int):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "architecture": model.spec,
                "request": request,
                "step": step,
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all(),
                "rng_numpy": json.dumps(rng.bit_generator.state),
                "dataset_sha256": request["views_sha256"],
            },
            run / f"logs/checkpoint-{step:04d}.pt",
        )

    @torch.inference_mode()
    def evaluate(step: int) -> dict:
        model.eval()
        generated = []
        for start in range(0, len(val_ids), 4):
            ids = torch.tensor(val_ids[start : start + 4], device="cuda")
            noise = torch.zeros(len(ids), 8, device="cuda")
            generated.append(
                ink_from_sdf(model.generate(tensors["reference"][ids], roles[ids], noise))
            )
        ink = torch.cat(generated)[:, 0]
        truth = tensors["target"][val_ids, 0]
        mask = ink > 0.5
        mae = ((ink - truth).abs() * (1 + 4 * ink + 4 * truth)).mean().item()
        result = {
            "step": step,
            "validation_shape": mae,
            "blank": int((~mask.flatten(1).any(1)).sum()),
            "boundary": int(
                torch.cat(
                    [
                        mask[:, :2].flatten(1),
                        mask[:, -2:].flatten(1),
                        mask[:, :, :2].flatten(1),
                        mask[:, :, -2:].flatten(1),
                    ],
                    1,
                )
                .any(1)
                .sum()
            ),
            "content_correctness": None,
            "author_style": None,
        }
        atomic_json(run / f"reports/validation-{step:04d}.json", result)
        for i, image in enumerate(ink.cpu().numpy()):
            Image.fromarray(np.uint8((1 - image.clip(0, 1)) * 255)).save(
                run / f"comparisons/step-{step:04d}-val-{i:02d}.png"
            )
        model.train()
        return result

    initial_validation = evaluate(0)
    for step in range(1, request["steps"] + 1):
        selected = rng.choice(train_ids, 4, replace=True)
        ids = torch.tensor(selected, device="cuda")
        reference, target = tensors["reference"][ids], tensors["target"][ids].clone()
        sdf = tensors["target_sdf"][ids].clone()
        real = request["real_per_batch"]
        # Font view is explicit auxiliary replay, never counted as extra real labels.
        target[real:] = reference[real:]
        sdf[real:] = tensors["reference_sdf"][ids[real:]]
        noise = torch.randn(4, 8, device="cuda")
        prediction, distribution = model(reference, target, roles[ids], noise)
        beta = min(1, step / 200) * (0.02 if request["global_style"] else 0)
        loss, parts = objective(prediction, sdf, target, distribution, beta)
        support = torch.tensor(0.0, device="cuda")
        if request["support"]:
            generated = model.generate(reference, roles[ids], torch.randn_like(noise))
            support = (generated - tensors["reference_sdf"][ids]).abs().mean()
            loss = loss + request["support"] * support
        if not torch.isfinite(loss):
            raise ValueError("Non-finite context loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            history.append(
                {
                    "step": step,
                    "loss": loss.item(),
                    "support": support.item(),
                    **{k: v.item() for k, v in parts.items()},
                }
            )
        if step % 200 == 0 or step == request["steps"]:
            save(step)
            evaluation = evaluate(step)
            print(json.dumps(evaluation), flush=True)
            atomic_json(run / "reports/curve.json", history)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    delta = sum(
        (a - b).abs().sum().item() for a, b in zip(initial, model.parameters(), strict=True)
    )
    assert delta > 0
    atomic_json(
        run / "reports/environment.json",
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "architecture": model.spec,
        },
    )
    atomic_json(
        run / "reports/training.json",
        {
            "real_cuda_parameter_training": True,
            "arm": request["arm"],
            "steps": request["steps"],
            "parameter_absolute_change": delta,
            "seconds": elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "train_real_regions": len(train_ids),
            "validation_real_regions": len(val_ids),
            "initial_validation": initial_validation,
            "final_validation": evaluation,
            "checkpoint": f"logs/checkpoint-{request['steps']:04d}.pt",
            "checkpoint_sha256": sha256(run / f"logs/checkpoint-{request['steps']:04d}.pt"),
            "selection": "fixed final step; no accepted best model",
            "production_eligible": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("request", type=Path)
    t = sub.add_parser("train")
    t.add_argument("views", type=Path)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.request)
    else:
        print(train(args.views, args.arm, args.steps))
