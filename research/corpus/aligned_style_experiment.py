"""Finite registration and neural distribution experiments, CPU runtime isolated."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(
    stage: str,
    data: Path,
    *,
    registration: Path | None = None,
    mode: str = "mean",
    monotone: bool = False,
) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if stage not in {"registration", "train"} or mode not in {"mean", "stochastic"}:
        raise ValueError("Unregistered experiment")
    verify_run(data)
    if stage == "train":
        if registration is None:
            raise ValueError("A frozen registration receipt is required")
        verify_run(registration)
        receipt = json.loads((registration / "reports/request.json").read_text())
        if (
            receipt["data_sha256"] != digest(data / "run.json")
            or receipt.get("monotone", False) != monotone
        ):
            raise ValueError("Registration source or geometry differs from learner")
    config = {
        "stage": stage,
        "monotone": monotone,
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "registration": str(registration) if registration else None,
        "registration_sha256": digest(registration / "run.json") if registration else None,
        "mode": mode,
        "seed": 1707,
        "steps": 240 if stage == "registration" else 1000,
        "protocol_sha256": digest(
            Path("docs/V0107_MONOTONE_PROTOCOL.md" if monotone else "docs/V0107_PROTOCOL.md")
        ),
        "architecture": "monotone-word-geometry-cvae/1"
        if monotone
        else "registered-word-geometry-cvae/1",
        "C_used": False,
        "new_ink_topology": False,
        "font_profile": "personal@v1",
        "negative_ink_correction": False,
    }
    run = create("aligned-style-" + stage, config=config, seed=1707)
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/execution.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.aligned_style_experiment",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=2400,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def setup(seed):
    import os

    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA is required")


def dice(a, b):
    import torch

    axes = (1, 2, 3)
    return 2 * (a * b).sum(axes) / torch.clamp(a.sum(axes) + b.sum(axes), min=1)


def topology(flow):
    from research.corpus.context_flow_model import jacobian_determinant

    jac = jacobian_determinant(flow)
    return jac.flatten(1).min(1).values


def registration_worker(r, arrays, rows):
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.aligned_style_model import (
        BOUNDS,
        GRID,
        dense_flow,
        monotone_controls,
        render,
    )
    from research.corpus.context_flow_model import jacobian_determinant

    run = Path(r["run"])
    all_controls, summaries, curve = [], [], []
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    targets = torch.tensor(arrays["target"][:, None], device="cuda")
    bounds = torch.tensor(BOUNDS, device="cuda")[None, :, None, None]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for k in range(0, len(rows), 32):
        ii = list(range(k, min(k + 32, len(rows))))
        raw = torch.zeros(
            (len(ii), 3, 17) if r.get("monotone") else (len(ii), 2, *GRID),
            device="cuda",
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([raw], lr=0.025)
        ref, target = refs[ii], targets[ii]
        base = dice((ref > 0.5).float(), (target > 0.5).float())
        cached = {
            s: (
                F.avg_pool2d(ref, 2 * s + 1, stride=1, padding=s),
                F.avg_pool2d(target, 2 * s + 1, stride=1, padding=s),
            )
            for s in (8, 4, 2)
        }
        for step in range(r["steps"]):
            s = (8, 4, 2)[min(2, step // 80)]
            controls = monotone_controls(raw) if r.get("monotone") else bounds * torch.tanh(raw)
            flow = dense_flow(controls)
            a, b = cached[s]
            from research.corpus.context_flow_model import sample_fixed_field

            warped = sample_fixed_field(a, flow)
            overlap = 1 - (2 * (warped * b).sum((1, 2, 3)) + 0.01) / (
                warped.square().sum((1, 2, 3)) + b.square().sum((1, 2, 3)) + 0.01
            )
            # Positive-Jacobian penalty and control-grid smoothness discourage
            # folding without pretending that numeric registration preserves text.
            jac = jacobian_determinant(flow)
            fold = F.relu(0.25 - jac).square().mean()
            norm = controls / bounds
            smooth = (norm[..., 1:] - norm[..., :-1]).square().mean() + (
                norm[..., 1:, :] - norm[..., :-1, :]
            ).square().mean()
            loss = overlap.mean() + 0.03 * smooth + 2 * fold
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            controls = monotone_controls(raw) if r.get("monotone") else bounds * torch.tanh(raw)
            out = render(ref, controls)
            score = dice((out > 0.5).float(), (target > 0.5).float())
            min_jac = topology(dense_flow(controls))
            border = (
                (out[..., :2, :].flatten(1) > 0.5).sum(1)
                + (out[..., -2:, :].flatten(1) > 0.5).sum(1)
                + (out[..., :, :2].flatten(1) > 0.5).sum(1)
                + (out[..., :, -2:].flatten(1) > 0.5).sum(1)
            )
            for j, i in enumerate(ii):
                ok = bool(
                    min_jac[j] > 0
                    and border[j] == 0
                    and score[j] >= 0.35
                    and score[j] - base[j] >= 0.03
                )
                summaries.append(
                    {
                        "index": i,
                        "text": rows[i]["text"],
                        "split": rows[i]["dataset_split"],
                        "identity_dice": float(base[j]),
                        "registered_dice": float(score[j]),
                        "min_jacobian": float(min_jac[j]),
                        "border_core": int(border[j]),
                        "eligible": ok,
                        "target_assisted_oracle": True,
                    }
                )
                Image.fromarray(np.uint8((1 - out[j, 0].cpu().numpy()) * 255)).save(
                    run / f"rendered/pages/oracle-{i:04d}.png"
                )
            all_controls.append(controls.cpu().numpy())
        curve.append(
            {
                "batch_start": k,
                "steps": r["steps"],
                "loss": float(loss.detach()),
                "seconds": time.perf_counter() - start,
            }
        )
        write(run / "reports/curve.json", curve)
        print(json.dumps(curve[-1]), flush=True)
    np.savez_compressed(run / "reports/controls.npz", controls=np.concatenate(all_controls))
    write(run / "reports/registration.json", summaries)
    eligible = [q for q in summaries if q["eligible"] and q["split"] == "train"]
    write(
        run / "reports/result.json",
        {
            "neural_updates": 0,
            "numerical_registration_steps_per_pair": r["steps"],
            "pairs": len(rows),
            "eligible_train": len(eligible),
            "eligible_labels": len({q["text"] for q in eligible}),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "target_assisted_not_generation": True,
        },
    )


def train_worker(r, arrays, rows):
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.aligned_style_contract import text_tokens, training_indices
    from research.corpus.aligned_style_model import AlignedStyle, dense_flow
    from research.corpus.context_flow_model import jacobian_determinant

    run, registration = Path(r["run"]), Path(r["registration"])
    if digest(registration / "run.json") != r["registration_sha256"]:
        raise ValueError("Registration changed")
    details = json.loads((registration / "reports/registration.json").read_text())
    ids = np.array(training_indices(details, rows))
    if any(rows[i]["dataset_split"] != "train" for i in ids):
        raise ValueError("Leakage into training")
    val = np.where(arrays["split"] == "validation")[0]
    eval_ids = np.array(sorted(set(val) | set(ids[:12])))
    write(
        run / "reports/indices.json",
        {"train": ids.tolist(), "validation": val.tolist(), "evaluated": eval_ids.tolist()},
    )
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    fields = torch.tensor(np.load(registration / "reports/controls.npz")["controls"], device="cuda")
    model = AlignedStyle(
        stochastic=r["mode"] == "stochastic", monotone=r.get("monotone", False)
    ).cuda()
    if r.get("warmstart"):
        if digest(r["warmstart"]) != r["warmstart_sha256"]:
            raise ValueError("Frozen continuation weights changed")
        model.load_state_dict(
            torch.load(r["warmstart"], map_location="cuda", weights_only=False)["model"]
        )
    if not model.stochastic:
        model.prior.requires_grad_(False)
        model.posterior.requires_grad_(False)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=r.get("lr", 0.0007)
    )
    rng = np.random.default_rng(r["seed"])
    curve = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(ids, 32, replace=True)
        condition = model.condition(refs[ii], tokens[ii])
        if model.stochastic:
            pm, pl = model.distribution(condition)
            qm, ql = model.distribution(condition, fields[ii])
            z = qm + (0.5 * ql).exp() * torch.randn_like(qm)
            kl = 0.5 * (pl - ql + ((ql.exp() + (qm - pm).square()) / pl.exp()) - 1).mean()
        else:
            z = torch.zeros((len(ii), model.latent), device="cuda")
            kl = z.sum()
        pred = model.decode(condition, z)
        reconstruction = F.mse_loss(pred / model.bounds, fields[ii] / model.bounds)
        # This differentiable regularizer sees predicted coordinates, not target pixels.
        flow = dense_flow(pred, (32, 128)) / 4
        fold = F.relu(0.25 - jacobian_determinant(flow)).square().mean()
        loss = reconstruction + 0.003 * min(1, step / 200) * kl + 0.1 * fold
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "kl": float(kl.detach()),
                    "seconds": time.perf_counter() - start,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 500 == 0 or step == r["steps"]:
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
    model.eval()
    generated, controls, metrics = {}, {}, []
    with torch.inference_mode():
        for seed in range(1708, 1712) if model.stochastic else [1708]:
            generator = torch.Generator(device="cuda").manual_seed(seed)
            chunks, cc = [], []
            for k in range(0, len(eval_ids), 8):
                ii = eval_ids[k : k + 8]
                epsilon = torch.randn((len(ii), 4), device="cuda", generator=generator)
                out, control = model.generate(refs[ii], tokens[ii], epsilon)
                chunks.append(out[:, 0].cpu().numpy())
                cc.append(control.cpu().numpy())
            generated[str(seed)], controls[str(seed)] = np.concatenate(chunks), np.concatenate(cc)
            jac = (
                topology(dense_flow(torch.tensor(controls[str(seed)], device="cuda"))).cpu().numpy()
            )
            for k, i in enumerate(eval_ids):
                a = generated[str(seed)][k]
                Image.fromarray(np.uint8((1 - a) * 255)).save(
                    run / f"rendered/pages/prior-{seed}-{i:04d}.png"
                )
                metrics.append(
                    {
                        "index": int(i),
                        "seed": seed,
                        "min_jacobian": float(jac[k]),
                        "border_core": int(
                            (a[:2] > 0.5).sum()
                            + (a[-2:] > 0.5).sum()
                            + (a[:, :2] > 0.5).sum()
                            + (a[:, -2:] > 0.5).sum()
                        ),
                        "blank": not bool((a > 0.5).any()),
                        "pixel_mae_from_font": float(np.abs(a - arrays["reference"][i]).mean()),
                    }
                )
    np.savez_compressed(run / "reports/generated.npz", **generated)
    np.savez_compressed(run / "reports/generated-controls.npz", **controls)
    write(run / "reports/generation-metrics.json", metrics)
    torch.cuda.synchronize()
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "updates": r["steps"],
            "parameter_absolute_change": sum(
                (v - initial[k]).abs().sum().item() for k, v in model.state_dict().items()
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_words": len(ids),
            "train_labels": len({rows[i]["text"] for i in ids}),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{r['steps']:04d}.pt"),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "prior_uses_real_target": False,
            "output_is_warped_font_ink": True,
            "C_used": False,
        },
    )


def worker(request):
    import importlib.metadata

    import numpy as np

    r = json.loads(Path(request).read_text())
    setup(r["seed"])
    data = Path(r["data"])
    if digest(data / "run.json") != r["data_sha256"]:
        raise ValueError("Frozen source changed")
    arrays = np.load(data / "reports/arrays.npz")
    rows = json.loads((data / "reports/samples.json").read_text())
    if set(arrays["split"]) - {"train", "validation"}:
        raise ValueError("Holdout forbidden")
    if any(
        q["source_hash"] == "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
        for q in rows
    ):
        raise ValueError("C forbidden")
    write(
        Path(r["run"]) / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    (registration_worker if r["stage"] == "registration" else train_worker)(r, arrays, rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    worker(p.parse_args().request)
