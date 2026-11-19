"""Finite real-word latent experiment; posterior reconstruction is never prior generation."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path, warmup: Path):
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "warmup": str(warmup.resolve()),
        "warmup_sha256": digest(warmup),
        "architecture": "semantic-word-latent-ink/1",
        "latent": 8,
        "seed": 3115,
        "steps": 8000,
        "latent_warmup": 1000,
        "batch": 4,
        "points": 2048,
        "decoder_lr": 0.0001,
        "latent_lr": 0.0003,
        "kl_weight": 0.01,
        "posterior_input": "real target and frozen neural semantic base",
        "generation_input": "characters, layout, independent eight-dimensional normal latent",
        "real_samples": 66,
        "font_auxiliary_draws": 0,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01015_PROTOCOL.md")),
    }
    run = create("semantic-word-latent", config=r, seed=r["seed"])
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    Path("/tmp/context-v01015-word-latent").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.semantic_word_latent_train",
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


def worker(request: Path):
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_ink_model import SemanticInk, WordLatentSemanticInk

    r = json.loads(request.read_text())
    run, data = Path(r["run"]), Path(r["data"])
    setup(r["seed"])
    assert digest(data / "run.json") == r["data_sha256"]
    assert digest(data / "reports/arrays.npz") == r["arrays_sha256"]
    assert digest(r["warmup"]) == r["warmup_sha256"]
    with np.load(data / "reports/arrays.npz") as archive:
        a = {k: archive[k] for k in archive.files}
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    train = idx["train"]
    assert len(train) == 66 and not set(train) & set(idx["development"] + idx["novel"])
    assert all(rows[i]["real_scan"] and rows[i]["split"] == "train" for i in train)
    x = {k: torch.tensor(a[k], device="cuda") for k in ["tokens", "boxes", "ink", "sdf"]}
    saved = torch.load(r["warmup"], map_location="cuda", weights_only=False)
    assert saved["step"] == 24000
    base = SemanticInk(context_scale=0).cuda().eval()
    base.load_state_dict(saved["model"])
    base.requires_grad_(False)
    semantic_base = torch.zeros_like(x["ink"])
    # No real target participates in these neural baseline images.
    for i in train + idx["development"]:
        semantic_base[i] = base.image(x["tokens"][i : i + 1], x["boxes"][i : i + 1])[0]
    np.savez_compressed(
        run / "reports/posterior-semantic-base.npz", ink=semantic_base.cpu().numpy()
    )
    del base
    model = WordLatentSemanticInk().cuda()
    model.load_identity(saved["model"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    newly_conditioned, body = [], []
    for name, p in model.named_parameters():
        if name == "context_gate" or name.startswith(("style_projection.", "posterior_encoder.")):
            newly_conditioned.append(p)
        else:
            p.requires_grad_(False)
            body.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": r["decoder_lr"]},
            {"params": newly_conditioned, "lr": r["latent_lr"]},
        ]
    )
    rng = np.random.default_rng(r["seed"])
    curve, counts = [], np.zeros(len(rows), np.int64)
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    for step in range(1, r["steps"] + 1):
        if step == r["latent_warmup"] + 1:
            for p in body:
                p.requires_grad_(True)
        ii = rng.choice(train, r["batch"])
        np.add.at(counts, ii, 1)
        pixels = []
        for i in ii:
            valid = np.flatnonzero(a["tokens"][i, :, 0])
            boxes = a["boxes"][i, rng.choice(valid, r["points"] // 2)]
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            radius = (boxes[:, 2:] - boxes[:, :2]) / 2
            xy = center + rng.uniform(-1.6, 1.6, center.shape) * radius
            px, py = (
                np.rint(xy[:, 0]).astype(int).clip(0, 511),
                np.rint(xy[:, 1]).astype(int).clip(0, 127),
            )
            pixels.append(
                np.concatenate([py * 512 + px, rng.integers(0, 128 * 512, r["points"] // 2)])
            )
        pi = torch.tensor(np.array(pixels), device="cuda")
        points = torch.stack([pi % 512, pi // 512], -1).float()
        mu, lv = model.posterior(x["ink"][ii, None], semantic_base[ii, None])
        z = mu + (0.5 * lv).exp() * torch.randn_like(mu)
        logits, sdf = model(x["tokens"][ii], x["boxes"][ii], points, style=z)
        target = x["ink"][ii].flatten(1).gather(1, pi)
        distance = x["sdf"][ii].flatten(1).gather(1, pi)
        reconstruction = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * target)
        ).mean()
        reconstruction = reconstruction + 0.2 * (sdf - distance).abs().mean()
        kl = 0.5 * (mu.square() + lv.exp() - 1 - lv).mean()
        beta = r["kl_weight"] * min(1, step / r["latent_warmup"])
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite latent objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "reconstruction": float(reconstruction.detach()),
                    "kl": float(kl.detach()),
                    "beta": beta,
                    "seconds": time.perf_counter() - begin,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 1000 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    posterior = {}
    with torch.inference_mode():
        for kind in ["train", "development"]:
            ii = idx[kind]
            mu, lv = model.posterior(x["ink"][ii, None], semantic_base[ii, None])
            posterior[kind] = (mu, lv)
    mean, lv = posterior["train"]
    write(
        run / "reports/latent-diagnostics.json",
        {
            "training_mean_by_dimension": mean.mean(0).tolist(),
            "training_mean_variance_by_dimension": mean.var(0).tolist(),
            "posterior_std_by_dimension": (0.5 * lv).exp().mean(0).tolist(),
            "kl_by_dimension": (0.5 * (mean.square() + lv.exp() - 1 - lv)).mean(0).tolist(),
            "active_dimensions_variance_gt_0_01": int((mean.var(0) > 0.01).sum()),
            "not_proof_of_style_disentanglement": True,
        },
    )
    records, generated = [], []
    selections = train[:12] + idx["development"] + idx["novel"]
    for mode, seed in [("prior", 2012), ("prior", 2013), ("posterior-mean", None)]:
        prior = (
            torch.randn(
                1, 8, generator=torch.Generator(device="cuda").manual_seed(seed), device="cuda"
            )
            if seed
            else None
        )
        for i in selections:
            if mode == "posterior-mean" and i in idx["novel"]:
                continue
            if mode == "posterior-mean":
                kind = "train" if i in train else "development"
                z = posterior[kind][0][idx[kind].index(i) : idx[kind].index(i) + 1]
            else:
                z = prior
            image = (
                model.image(x["tokens"][i : i + 1], x["boxes"][i : i + 1], style=z)[0].cpu().numpy()
            )
            path = run / f"rendered/pages/{mode}-{seed or 0}-{i:04d}.png"
            Image.fromarray(np.uint8((1 - image) * 255)).save(path)
            mask = image > 0.5
            target = a["ink"][i] > 0.5
            records.append(
                {
                    "index": i,
                    "text": rows[i]["text"],
                    "split": rows[i]["split"],
                    "mode": mode,
                    "seed": seed,
                    "latent": z[0].tolist(),
                    "image": str(path),
                    "sha256": digest(path),
                    "target_seen_at_output": mode == "posterior-mean",
                    "blank": not bool(mask.any()),
                    "edge": bool(
                        mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                    ),
                    "dice_to_target": None
                    if i in idx["novel"]
                    else float(2 * (mask & target).sum() / max(1, mask.sum() + target.sum())),
                }
            )
            generated.append(image)
    np.savez_compressed(run / "reports/generated.npz", ink=np.array(generated))
    write(run / "reports/generation-metrics.json", records)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/result.json",
        {
            "new_updates": r["steps"],
            "latent_only_updates": r["latent_warmup"],
            "joint_updates": r["steps"] - r["latent_warmup"],
            "inherited_updates": 24000,
            "actual_parameter_training": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "context_gate": float(model.context_gate.detach()),
            "style_projection_norm": float(model.style_projection.weight.norm().detach()),
            "seconds_training_and_generation": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / "logs/checkpoint-08000.pt"),
            "prior_outputs": sum(q["mode"] == "prior" for q in records),
            "posterior_outputs": sum(q["mode"] == "posterior-mean" for q in records),
            "real_draws": int(counts[train].sum()),
            "development_draws": int(counts[idx["development"]].sum()),
            "novel_draws": int(counts[idx["novel"]].sum()),
            "C_used": False,
            "style_quality_accepted": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "worker"])
    p.add_argument("path", type=Path)
    p.add_argument("--warmup", type=Path)
    args = p.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        launch(args.path, args.warmup)
