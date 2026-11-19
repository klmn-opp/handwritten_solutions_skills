"""Equal-budget structure control and unpaired real-ink style training, isolated CUDA."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(arm: str, base: Path, resume: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if arm not in ["structure", "adversarial", "structure-relaxed", "adversarial-relaxed"]:
        raise ValueError("Fixed arms required")
    verify_run(base)
    prior = json.loads((base / "reports/request.json").read_text())
    data = Path(prior["data"])
    verify_run(data)
    r = {
        "arm": arm,
        "base": str((base / "logs/checkpoint-04000.pt").resolve()),
        "base_sha256": digest(base / "logs/checkpoint-04000.pt"),
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "steps": 4000 if arm.endswith("-relaxed") else 2000,
        "batch": 8,
        "seed": 2914,
        "generator_lr": 0.0001,
        "critic_lr": 0.0001,
        "adversarial_weight": 0.5 if arm.endswith("-relaxed") else 0.05,
        "relaxed_structure": arm.endswith("-relaxed"),
        "input_patch": 128,
        "loss_patch": 64,
        "receptive_field": 63,
        "uses_unpaired_real_original_ink": arm.startswith("adversarial"),
        "paired_pixel_reconstruction": False,
        "font_structure_is_fixed_constraint": True,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01014_PROTOCOL.md")),
    }
    if resume is not None:
        verify_run(resume.parent.parent)
        old = json.loads((resume.parent.parent / "reports/request.json").read_text())
        for key in ["arm", "base_sha256", "data_sha256", "arrays_sha256", "steps", "seed"]:
            if old[key] != r[key]:
                raise ValueError(f"Resume changes fixed experiment: {key}")
        r.update(resume=str(resume.resolve()), resume_sha256=digest(resume))
    run = create("ink-distribution-" + arm, config=r, seed=r["seed"])
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.adversarial_ink_train",
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

    from research.corpus.adversarial_ink_model import (
        InkPatchCritic,
        generate_local,
        structure_loss,
        word_patches,
    )
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.local_ink_model import LocalInk

    r = json.loads(request.read_text())
    setup(r["seed"])
    run = Path(r["run"])
    data = Path(r["data"])
    for path, expected in [
        (Path(r["base"]), r["base_sha256"]),
        (data / "run.json", r["data_sha256"]),
        (data / "reports/arrays.npz", r["arrays_sha256"]),
    ]:
        if digest(path) != expected:
            raise ValueError("Changed frozen input")
    idx = json.loads((data / "reports/indices.json").read_text())
    train = idx["train"]
    eval_ids = train + idx["development"] + idx["novel"]
    rows = json.loads((data / "reports/samples.json").read_text())
    assert all(rows[i]["dataset_split"] == "train" for i in train) and not set(train) & set(
        idx["development"] + idx["novel"]
    )
    raw = np.load(data / "reports/arrays.npz")
    reference = raw["reference"]
    original = raw["original_target"]
    ref = torch.tensor(reference[train, None], device="cuda")
    real = torch.tensor(original[train, None], device="cuda")
    all_ref = torch.tensor(reference[:, None], device="cuda")
    canonical = raw["target"]
    ref_points = [np.argwhere(x > 0.5) for x in reference[train]]
    real_points = [np.argwhere(x > 0.5) for x in original[train]]
    model = LocalInk().cuda()
    model.load_state_dict(torch.load(r["base"], map_location="cuda", weights_only=False)["model"])
    critic = InkPatchCritic().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_d = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    torch.save({"model": initial, "critic": initial_d}, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["generator_lr"])
    dopt = torch.optim.AdamW(critic.parameters(), lr=r["critic_lr"])
    rng = np.random.default_rng(r["seed"])
    curve = []
    traces = []
    counts = np.zeros(len(train), dtype=np.int64)
    start_step = 0
    if r.get("resume"):
        if digest(Path(r["resume"])) != r["resume_sha256"]:
            raise ValueError("Changed resume checkpoint")
        saved = torch.load(r["resume"], map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        critic.load_state_dict(saved["critic"])
        optimizer.load_state_dict(saved["optimizer"])
        dopt.load_state_dict(saved["critic_optimizer"])
        start_step = saved["generator_steps"]
        if not 0 < start_step < r["steps"]:
            raise ValueError("Resume must precede the fixed budget")
        # Reconstruct sampling receipts without running any neural update.
        for _ in range(start_step):
            ii = rng.choice(len(train), r["batch"], replace=True)
            jj = rng.choice(len(train), r["batch"], replace=True)
            np.add.at(counts, ii, 1)
            centers = [
                (int(i), *map(int, ref_points[i][rng.integers(len(ref_points[i]))])) for i in ii
            ]
            real_centers = [
                (int(j), *map(int, real_points[j][rng.integers(len(real_points[j]))])) for j in jj
            ]
            traces.append([list(a) + list(b) for a, b in zip(centers, real_centers, strict=True)])
        if rng.bit_generator.state != saved["numpy_rng"]:
            raise ValueError("Resume sampling receipt does not reproduce saved RNG")
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([q.cpu() for q in saved["cuda_rng"]])
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        initial_d = {k: v.detach().clone() for k, v in critic.state_dict().items()}
        torch.save({"model": initial, "critic": initial_d}, run / "logs/initial-state.pt")
        write(
            run / "reports/resume.json",
            {"start_step": start_step, "sampling_rng_exact": True, "optimizer_restored": True},
        )
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(run / "reports/indices.json", {**idx, "evaluated": eval_ids})
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    for step in range(start_step + 1, r["steps"] + 1):
        ii = rng.choice(len(train), r["batch"], replace=True)
        jj = rng.choice(len(train), r["batch"], replace=True)
        np.add.at(counts, ii, 1)
        centers = [(int(i), *map(int, ref_points[i][rng.integers(len(ref_points[i]))])) for i in ii]
        real_centers = [
            (int(j), *map(int, real_points[j][rng.integers(len(real_points[j]))])) for j in jj
        ]
        traces.append([list(a) + list(b) for a, b in zip(centers, real_centers, strict=True)])
        condition = word_patches(ref, centers, 64)
        real_patch = word_patches(real, real_centers, 32)
        fake = model(condition).sigmoid()[..., 32:96, 32:96]
        guide = condition[..., 32:96, 32:96]
        d_loss = fake.new_zeros(())
        if r["arm"].startswith("adversarial"):
            critic.requires_grad_(True)
            dopt.zero_grad(set_to_none=True)
            real_patch.requires_grad_(step % 16 == 0)
            real_score = critic(real_patch)
            fake_score = critic(fake.detach())
            d_loss = 0.5 * ((real_score - 1).square().mean() + fake_score.square().mean())
            if step % 16 == 0:
                grad = torch.autograd.grad(real_score.sum(), real_patch, create_graph=True)[0]
                d_loss = d_loss + 0.05 * grad.square().flatten(1).sum(1).mean()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5)
            dopt.step()
        critic.requires_grad_(False)
        content, parts = structure_loss(fake, guide, relaxed=r.get("relaxed_structure", False))
        adversarial = (
            (critic(fake) - 1).square().mean()
            if r["arm"].startswith("adversarial")
            else fake.new_zeros(())
        )
        loss = content + r["adversarial_weight"] * adversarial
        if not torch.isfinite(loss) or not torch.isfinite(d_loss):
            raise ValueError("Nonfinite adversarial objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == start_step + 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "g_loss": float(loss.detach()),
                    "d_loss": float(d_loss.detach()),
                    "adversarial": float(adversarial.detach()),
                    **{k: float(v.detach()) for k, v in parts.items()},
                    "seconds": time.perf_counter() - begin,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "critic_optimizer": dopt.state_dict(),
                    "generator_steps": step,
                    "critic_steps": step if r["arm"].startswith("adversarial") else 0,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    images = []
    with torch.inference_mode():
        for k in range(0, len(eval_ids), 4):
            images.append(generate_local(model, all_ref[eval_ids[k : k + 4]])[:, 0].cpu().numpy())
    images = np.concatenate(images)
    metrics = []
    for j, i in enumerate(eval_ids):
        im = images[j]
        mask = im > 0.5
        truth = canonical[i] > 0.5
        dest = run / f"rendered/pages/{r['arm']}-{i:04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
        metrics.append(
            {
                "index": i,
                "text": rows[i]["text"],
                "image": str(dest),
                "sha256": digest(dest),
                "group": "novel"
                if i in idx["novel"]
                else "development"
                if i in idx["development"]
                else "train",
                "dice": None
                if i in idx["novel"]
                else float(2 * (mask & truth).sum() / max(1, mask.sum() + truth.sum())),
                "blank": not bool(mask.any()),
                "edge": bool(
                    mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                ),
                "target_read_for_output": False,
            }
        )
    np.savez_compressed(run / "reports/generated.npz", ink=images)
    np.savez_compressed(run / "logs/patch-trace.npz", trace=np.array(traces, dtype=np.int32))
    write(run / "reports/generation-metrics.json", metrics)
    write(
        run / "reports/sampling-counts.json",
        {"indices": train, "counts": counts.tolist(), "development_used": 0, "novel_used": 0},
    )
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "generator_updates": r["steps"] - start_step,
            "critic_updates": r["steps"] - start_step if r["arm"].startswith("adversarial") else 0,
            "start_step": start_step,
            "cumulative_optimizer_steps": r["steps"],
            "generator_parameters": sum(p.numel() for p in model.parameters()),
            "critic_parameters": sum(p.numel() for p in critic.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "critic_parameter_l1_change": sum(
                float((v - initial_d[k]).abs().sum()) for k, v in critic.state_dict().items()
            ),
            "blank": sum(q["blank"] for q in metrics),
            "edge": sum(q["edge"] for q in metrics),
            "seconds": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{r['steps']:05d}.pt"),
            "C_used": False,
            "stochastic_output": False,
            "RL_or_preference_training": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["structure", "adversarial", "structure-relaxed", "adversarial-relaxed", "worker"],
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(launch(args.mode, args.path, args.resume))
