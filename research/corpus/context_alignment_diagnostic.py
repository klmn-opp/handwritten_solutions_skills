"""Separate posterior reconstruction, prior generation, and small-set learning ability."""

import argparse
import json
import subprocess
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create


def run_diagnostic(views: Path, checkpoint: Path, *, steps: int = 0) -> Path:
    verify_run(views)
    if steps not in {0, 2000}:
        raise ValueError("Fixed diagnostic budget only")
    run = create(
        "context-posterior-prior-diagnostic",
        seed=1101,
        config={
            "views": str(views.resolve()),
            "views_sha256": sha256(views / "run.json"),
            "arrays_sha256": sha256(views / "reports/arrays.npz"),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "steps": steps,
            "diagnostic_only": True,
            "holdout_used": False,
            "training_mode": "random initialization, 16 words + 16 shortest multiword train regions"
            if steps
            else "frozen model, zero updates",
        },
    )
    request = {**run.record()["config"], "run": str(run.path), "seed": 1101}
    atomic_json(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/corpus-generator-v09/bin/python",
                    "-m",
                    __name__
                    if __name__ != "__main__"
                    else "research.corpus.context_alignment_diagnostic",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        result = json.loads((run.path / "reports/diagnostic.json").read_text())
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(path: Path):
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.context_model import ContextInk
    from research.corpus.reference_generator import ink_from_sdf, objective

    request = json.loads(path.read_text())
    run = Path(request["run"])
    views = Path(request["views"])
    if (
        sha256(views / "reports/arrays.npz") != request["arrays_sha256"]
        or sha256(Path(request["checkpoint"])) != request["checkpoint_sha256"]
    ):
        raise ValueError("Frozen evidence changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    torch.manual_seed(request["seed"])
    torch.cuda.manual_seed_all(request["seed"])
    data = np.load(views / "reports/arrays.npz", allow_pickle=False)
    rows = json.loads((views / "reports/samples.json").read_text())
    train = [i for i, r in enumerate(rows) if r["dataset_split"] == "train"]
    words = [i for i in train if len(rows[i]["normalized_label"].split()) == 1]
    phrases = sorted(
        [i for i in train if i not in words],
        key=lambda i: (len(rows[i]["normalized_label"]), rows[i]["sample_id"]),
    )
    rng = np.random.default_rng(1101)
    chosen = sorted(rng.choice(words, 16, replace=False).tolist() + phrases[:16])
    assert len(set(chosen)) == 32
    atomic_json(run / "reports/subset.json", [{"index": i, **rows[i]} for i in chosen])
    x = {
        k: torch.tensor(data[k][:, None], device="cuda")
        for k in ["target", "reference", "target_sdf"]
    }
    role = torch.tensor(data["role"], device="cuda")
    model = ContextInk().cuda()
    steps = request["steps"]
    if not steps:
        model.load_state_dict(
            torch.load(request["checkpoint"], map_location="cuda", weights_only=True)["model"]
        )
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        ids = torch.tensor(rng.choice(chosen, 4, replace=True), device="cuda")
        noise = torch.randn(4, 8, device="cuda")
        pred, dist = model(x["reference"][ids], x["target"][ids], role[ids], noise)
        loss, parts = objective(
            pred, x["target_sdf"][ids], x["target"][ids], dist, 0.02 * min(1, step / 200)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % 25 == 0:
            history.append(
                {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()}}
            )
        if step in {250, 500, 1000, 2000}:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "architecture": model.spec,
                    "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all(),
                    "rng_numpy": json.dumps(rng.bit_generator.state),
                    "request": request,
                },
                run / f"logs/checkpoint-{step:04d}.pt",
            )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    results = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            ref = x["reference"][i : i + 1]
            target = x["target"][i : i + 1]
            roles = role[i : i + 1]
            zero = torch.zeros(1, 8, device="cuda")
            rec, dist = model(ref, target, roles, zero)
            prior = model.generate(ref, roles, zero)
            record = {
                "index": i,
                "sample_id": row["sample_id"],
                "text": row["normalized_label"],
                "split": row["dataset_split"],
                "in_small_train": i in chosen,
            }
            for name, tensor in [
                ("target", target),
                ("reference", ref),
                ("posterior", ink_from_sdf(rec)),
                ("prior", ink_from_sdf(prior)),
            ]:
                image = tensor[0, 0].cpu().numpy()
                dest = run / f"rendered/pages/{i:03d}-{name}.png"
                Image.fromarray(np.uint8((1 - image.clip(0, 1)) * 255)).save(dest)
                record[name + "_image"] = str(dest)
                if name in {"posterior", "prior"}:
                    record[name + "_shape"] = float(
                        ((tensor - target).abs() * (1 + 4 * tensor + 4 * target)).mean()
                    )
            qm, ql, pm, pl = dist
            record["kl"] = float(
                0.5 * (pl - ql + (ql.exp() + (qm - pm).square()) / pl.exp() - 1).mean()
            )
            results.append(record)
    atomic_json(run / "reports/regions.json", results)
    atomic_json(run / "reports/curve.json", history)
    summary = {
        split: {
            key: float(
                np.mean(
                    [
                        r[key]
                        for r in results
                        if (r["in_small_train"] if split == "small_train" else r["split"] == split)
                    ]
                )
            )
            for key in ["posterior_shape", "prior_shape", "kl"]
        }
        for split in ["small_train", "train", "validation"]
    }
    delta = sum(float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items())
    atomic_json(
        run / "reports/diagnostic.json",
        {
            "optimizer_updates": steps,
            "parameter_absolute_change": delta,
            "seconds": elapsed,
            "summary": summary,
            "production_eligible": False,
            "content_accuracy": None,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "peak_bytes": torch.cuda.max_memory_allocated(),
            },
            "memorization_test_is_not_generalization": bool(steps),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "worker"])
    p.add_argument("path", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=0)
    args = p.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(run_diagnostic(args.path, args.checkpoint, steps=args.steps))
