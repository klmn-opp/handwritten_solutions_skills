"""Finite CUDA whole-word experiments with posterior/prior evidence kept separate."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest as sha256
from research.corpus.context_content_train import write as atomic_json


def train(data: Path, mode: str, steps: int, *, warmstart: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if (
        mode not in {"capacity", "deterministic", "variational", "continue", "content"}
        or steps <= 0
    ):
        raise ValueError("Finite registered mode required")
    if (mode in {"continue", "content"}) != (warmstart is not None):
        raise ValueError("Paired continuation needs the same frozen parent checkpoint")
    verify_run(data)
    config = {
        "data": str(data),
        "data_sha256": sha256(data / "run.json"),
        "mode": mode,
        "steps": steps,
        "seed": 1606,
        "batch": 8,
        "lr": 0.0001 if warmstart else 0.0005,
        "latent": 32,
        "kl_max": 0.003 if mode in {"variational", "continue", "content"} else 0,
        "teacher_used": mode == "content",
        "font_target_or_anchor": False,
        "C_used": False,
        "protocol_sha256": sha256(Path("docs/V0106_PROTOCOL.md")),
        "architecture": "whole-word-direct-field-cvae/1",
        "warmstart": str(warmstart) if warmstart else None,
        "warmstart_sha256": sha256(warmstart) if warmstart else None,
        "optimizer": "fresh AdamW, identical in both continuation arms",
    }
    if warmstart:
        config["protocol_sha256"] = sha256(Path("docs/V0106_CONTENT_PROTOCOL.md"))
    if mode == "content":
        teacher = Path("workspace/checkpoints/trocr-base-handwritten-v08-xet").resolve()
        config["teacher"] = str(teacher)
        config["teacher_sha256"] = {
            str(p.relative_to(teacher)): sha256(p) for p in teacher.rglob("*") if p.is_file()
        }
    run = create("word-style-cuda-training", config=config, seed=config["seed"])
    atomic_json(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.word_style_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
        run.finish("held", result=json.loads((run.path / "reports/training.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: Path) -> None:
    import importlib.metadata
    import os
    import time

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.word_style_model import WordStyle, reconstruction_loss

    r = json.loads(request.read_text())
    run, data_path = Path(r["run"]), Path(r["data"])
    if sha256(data_path / "run.json") != r["data_sha256"]:
        raise ValueError("Frozen data changed")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.manual_seed(r["seed"])
    torch.cuda.manual_seed_all(r["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA required")
    arrays = np.load(data_path / "reports/arrays.npz")
    rows = json.loads((data_path / "reports/samples.json").read_text())
    if any(s not in {"train", "validation"} for s in arrays["split"]):
        raise ValueError("Holdout forbidden")
    ids = np.where(arrays["split"] == "train")[0]
    val = np.where(arrays["split"] == "validation")[0]
    rng = np.random.default_rng(r["seed"])
    if r["mode"] == "capacity":
        # Prefer genuine new lines, but choose before any learning or visual selection.
        new = np.array([i for i in ids if rows[i]["new_session"]])
        ids = np.sort(rng.choice(new, min(24, len(new)), replace=False))
        eval_ids = ids
    else:
        eval_ids = np.array(
            sorted(set(val) | set(ids[:64]) | {i for i in ids if rows[i]["new_session"]})
        )
    atomic_json(
        run / "reports/indices.json",
        {"train": ids.tolist(), "validation": val.tolist(), "evaluated": eval_ids.tolist()},
    )
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    targets = torch.tensor(arrays["target"][:, None], device="cuda")

    def sdf(a):
        mask = np.uint8(a > 0.5)
        return (
            cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            - cv2.distanceTransform(1 - mask, cv2.DIST_L2, 5)
        ).clip(-8, 8)

    fields = torch.tensor(np.stack([sdf(a) for a in arrays["target"]])[:, None], device="cuda")
    model = WordStyle(r["latent"]).cuda()
    if r.get("warmstart"):
        if sha256(r["warmstart"]) != r["warmstart_sha256"]:
            raise ValueError("Changed parent checkpoint")
        model.load_state_dict(
            torch.load(r["warmstart"], map_location="cuda", weights_only=False)["model"]
        )
    if r["mode"] == "deterministic":
        model.posterior.requires_grad_(False)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=r["lr"])
    teacher, labels, teacher_initial = None, None, None
    if r["teacher_used"]:
        from research.corpus.context_content_model import FrozenTranscriptionLoss

        for rel, expected in r["teacher_sha256"].items():
            if sha256(Path(r["teacher"]) / rel) != expected:
                raise ValueError("Changed local teacher weights")
        os.environ["HF_HUB_OFFLINE"] = "1"
        before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        teacher = FrozenTranscriptionLoss(r["teacher"])
        torch.set_rng_state(before_cpu)
        torch.cuda.set_rng_state(before_cuda)
        labels = teacher.labels([q["text"] for q in rows])
        teacher_initial = {n: p.detach().clone().cpu() for n, p in teacher.model.named_parameters()}
    curve, evaluations = [], []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def evaluate(step, final=False):
        model.eval()
        results, outputs = {}, {}
        means, logvars = [], []
        for k in range(0, len(eval_ids), 8):
            ii = eval_ids[k : k + 8]
            mu, lv = model.encode(refs[ii], targets[ii])
            means.append(mu.cpu().numpy())
            logvars.append(lv.cpu().numpy())
        means, logvars = np.concatenate(means), np.concatenate(logvars)
        modes = ["posterior", "prior-0"] if r["mode"] != "deterministic" else ["prior-0"]
        if final and r["mode"] not in {"deterministic", "capacity"}:
            modes += ["prior-1", "prior-2", "prior-3"]
        for mode in modes:
            generator = torch.Generator(device="cuda").manual_seed(
                2606 + int(mode[-1]) if mode.startswith("prior") else 2606
            )
            chunks = []
            for k in range(0, len(eval_ids), 8):
                ii = eval_ids[k : k + 8]
                if mode == "posterior":
                    z = torch.tensor(means[k : k + 8], device="cuda")
                elif r["mode"] == "deterministic":
                    z = torch.zeros(len(ii), r["latent"], device="cuda")
                else:
                    z = torch.randn(len(ii), r["latent"], device="cuda", generator=generator)
                ink, _ = model(refs[ii], z)
                chunks.append(ink[:, 0].cpu().numpy())
            out = np.concatenate(chunks)
            a, b = out > 0.5, arrays["target"][eval_ids] > 0.5
            dice = 2 * (a & b).sum((1, 2)) / np.maximum(1, a.sum((1, 2)) + b.sum((1, 2)))
            results[mode] = {
                split: {
                    "count": int(sum(rows[i]["dataset_split"] == split for i in eval_ids)),
                    "dice": float(
                        np.mean(
                            [
                                dice[k]
                                for k, i in enumerate(eval_ids)
                                if rows[i]["dataset_split"] == split
                            ]
                        )
                    ),
                }
                for split in ["train", "validation"]
                if any(rows[i]["dataset_split"] == split for i in eval_ids)
            }
            results[mode]["blank"] = int(sum(not q.any() for q in a))
            if final or step == 0:
                outputs[mode] = out
                for k, i in enumerate(eval_ids):
                    Image.fromarray(np.uint8((1 - out[k]) * 255)).save(
                        run / f"rendered/pages/step-{step}-{mode}-{i:04d}.png"
                    )
        results.update(
            step=step,
            latent_active_dims=int((means.var(0) > 0.01).sum()),
            mean_kl=float(0.5 * np.mean(np.exp(logvars) + means**2 - 1 - logvars)),
        )
        evaluations.append(results)
        atomic_json(run / "reports/evaluations.json", evaluations)
        if outputs:
            np.savez_compressed(run / f"reports/generated-{step}.npz", **outputs)
            np.savez_compressed(run / f"reports/latent-{step}.npz", mu=means, logvar=logvars)
        print(json.dumps(results), flush=True)
        model.train()

    evaluate(0)
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(ids, r["batch"], replace=True)
        optimizer.zero_grad(set_to_none=True)
        if r["mode"] == "deterministic":
            z = torch.zeros(len(ii), r["latent"], device="cuda")
            kl = z.sum()
        else:
            mu, lv = model.encode(refs[ii], targets[ii])
            z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            kl = 0.5 * (lv.exp() + mu.square() - 1 - lv).mean()
        ink, field = model(refs[ii], z)
        reconstruction = reconstruction_loss(ink, field, targets[ii], fields[ii])
        loss = reconstruction + r["kl_max"] * min(1, step / 500) * kl
        content = torch.zeros((), device="cuda")
        if teacher is not None and step % 4 == 0:
            prior_z = torch.randn(
                2,
                r["latent"],
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(7606 + step),
            )
            prior_ink, _ = model(refs[ii[:2]], prior_z)
            content = teacher(prior_ink, labels[ii[:2]])
            loss = loss + 0.2 * content
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite learning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "kl": float(kl.detach()),
                    "content": float(content.detach()),
                    "seconds": time.perf_counter() - start,
                }
            )
            atomic_json(run / "reports/curve.json", curve)
        if step % 500 == 0 or step == r["steps"]:
            evaluate(step, final=step == r["steps"])
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
    teacher_frozen = None
    if teacher is not None:
        teacher_frozen = all(
            torch.equal(p.detach().cpu(), teacher_initial[n]) and p.grad is None
            for n, p in teacher.model.named_parameters()
        )
        if not teacher_frozen:
            raise ValueError("Frozen teacher mutated")
    result = {
        "actual_parameter_training": True,
        "updates": r["steps"],
        "parameter_absolute_change": sum(
            (v - initial[k]).abs().sum().item() for k, v in model.state_dict().items()
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_words": len(ids),
        "seconds": time.perf_counter() - start,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "checkpoint_sha256": sha256(run / f"logs/checkpoint-{r['steps']:04d}.pt"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dependencies": {k: importlib.metadata.version(k) for k in ["numpy", "pillow"]},
        "posterior_is_oracle_not_generation": True,
        "C_used": False,
        "teacher_parameters_unchanged": teacher_frozen,
        "warmstart_sha256": r.get("warmstart_sha256"),
    }
    atomic_json(run / "reports/training.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["worker"])
    p.add_argument("request", type=Path)
    worker(p.parse_args().request)
