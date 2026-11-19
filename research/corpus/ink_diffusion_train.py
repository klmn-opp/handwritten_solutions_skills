"""Finite real-CUDA learning and target-free inference of personal ink fields."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path, *, mode: str, steps: int, warmstart: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if mode not in {"capacity", "expanded"} or steps not in {1000, 2000, 3000, 9000}:
        raise ValueError("Finite predeclared experiment required")
    verify_run(data)
    config = {
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "mode": mode,
        "steps": steps,
        "seed": 1909,
        "batch": 8,
        "lr": 0.0003,
        "reverse_steps": 64,
        "seeds": [1910, 1911],
        "warmstart": str(warmstart) if warmstart else None,
        "warmstart_sha256": digest(warmstart) if warmstart else None,
        "warmstart_optimizer": False,
        "architecture": "real-ink-conditional-v-denoiser/1",
        "C_used": False,
        "font_is_condition_only": True,
        "ocr_teacher": False,
        "protocol_sha256": digest(Path("docs/V0109_PROTOCOL.md")),
    }
    run = create("real-ink-diffusion-training", config=config, seed=config["seed"])
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    print(str(run.path), flush=True)
    try:
        with (run.path / "logs/training.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.ink_diffusion_train",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.ink_diffusion_model import InkDenoiser, diffusion_loss, generate

    r = json.loads(request.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    if digest(data / "run.json") != r["data_sha256"]:
        raise ValueError("Frozen data identity changed")
    rows = json.loads((data / "reports/samples.json").read_text())
    arrays = np.load(data / "reports/arrays.npz")
    if set(arrays["split"]) - {"train", "validation"}:
        raise ValueError("Holdout forbidden")
    forbidden = "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
    if any(q["source_hash"] == forbidden for q in rows):
        raise ValueError("C forbidden")
    ids = np.where(arrays["split"] == "train")[0]
    rng = np.random.default_rng(r["seed"])
    if r["mode"] == "capacity":
        candidates = [i for i in ids if 2 <= len(rows[i]["text"]) <= 10]
        # Fixed coverage choice before training, not visual output selection.
        f = [i for i in candidates if "f" in rows[i]["text"].lower()][:12]
        rest = [i for i in candidates if i not in f]
        ids = np.array(sorted(f + list(rng.choice(rest, 24 - len(f), replace=False))))
        eval_ids = ids
    else:
        val = np.where(arrays["split"] == "validation")[0]
        f = [i for i in ids if "f" in rows[i]["text"].lower()][:12]
        eval_ids = np.array(sorted(set(val) | set(ids[:12]) | set(f)))
    write(run / "reports/indices.json", {"train": ids.tolist(), "evaluated": eval_ids.tolist()})
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    fields = torch.tensor(arrays["field"][:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    model = InkDenoiser().cuda()
    if r["warmstart"]:
        if digest(r["warmstart"]) != r["warmstart_sha256"]:
            raise ValueError("Changed warmstart")
        model.load_state_dict(
            torch.load(r["warmstart"], map_location="cuda", weights_only=False)["model"]
        )
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    auxiliary = [i for i in ids if rows[i].get("kind") == "auxiliary-glyph"]
    regular = [i for i in ids if i not in auxiliary]
    curve, checkpoints = [], []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, r["steps"] + 1):
        ii = list(rng.choice(regular, r["batch"] - bool(auxiliary), replace=True))
        if auxiliary:
            ii += list(rng.choice(auxiliary, 1))
        noise = torch.randn_like(fields[ii])
        t = 0.001 + 0.998 * torch.rand(len(ii), device="cuda")
        loss = diffusion_loss(model, fields[ii], refs[ii], tokens[ii], t, noise)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            curve.append(
                {"step": step, "loss": float(loss.detach()), "seconds": time.perf_counter() - start}
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 1000 == 0 or step == r["steps"]:
            p = run / f"logs/checkpoint-{step:05d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                p,
            )
            checkpoints.append({"step": step, "path": str(p), "sha256": digest(p)})
    # All candidates frozen before reading development targets or generating evaluations.
    model.eval()
    generated, metrics = {}, []
    with torch.inference_mode():
        for seed in r["seeds"]:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            chunks = []
            for k in range(0, len(eval_ids), 4):
                ii = eval_ids[k : k + 4]
                noise = torch.randn(refs[ii].shape, device="cuda", generator=gen)
                ink, _ = generate(model, refs[ii], tokens[ii], noise, r["reverse_steps"])
                chunks.append(ink[:, 0].cpu().numpy())
            generated[str(seed)] = np.concatenate(chunks)
            for j, i in enumerate(eval_ids):
                a = generated[str(seed)][j]
                p = run / f"rendered/pages/prior-{seed}-{i:04d}.png"
                Image.fromarray(np.uint8((1 - a) * 255)).save(p)
                mask, target = a > 0.5, arrays["target"][i] > 0.5
                metrics.append(
                    {
                        "index": int(i),
                        "seed": seed,
                        "split": rows[i]["dataset_split"],
                        "text": rows[i]["text"],
                        "dice": float(
                            2 * (mask & target).sum() / max(1, mask.sum() + target.sum())
                        ),
                        "blank": not bool(mask.any()),
                        "border": int(
                            mask[:2].sum()
                            + mask[-2:].sum()
                            + mask[:, :2].sum()
                            + mask[:, -2:].sum()
                        ),
                        "image": str(p),
                        "sha256": digest(p),
                    }
                )
    np.savez_compressed(run / "reports/generated.npz", **generated)
    write(run / "reports/generation-metrics.json", metrics)
    write(run / "reports/checkpoints.json", checkpoints)
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    torch.cuda.synchronize()
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "updates": r["steps"],
            "train_regions": len(ids),
            "train_labels": len({rows[i]["text"] for i in ids}),
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_absolute_change": sum(
                (v - initial[k]).abs().sum().item() for k, v in model.state_dict().items()
            ),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "checkpoint_sha256": checkpoints[-1]["sha256"],
            "prior_reads_target": False,
            "output_copies_font_pixels": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["worker"])
    p.add_argument("request", type=Path)
    args = p.parse_args()
    worker(args.request)
