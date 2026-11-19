"""Finite shared-writer experiments; production imports neither this module nor Torch."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write

BASE = Path(
    "workspace/runs/20261119T173436830135Z-thin-ink-detail-5faffd79/logs/checkpoint-03000.pt"
)
CODEC = Path(
    "workspace/runs/20261119T170016941957Z-latent-ink-codec-ffb2c521/logs/checkpoint-02000.pt"
)


def launch(mode: str, data: Path, codec_run: Path | None = None) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if mode not in {"decoder", "writer"}:
        raise ValueError("Unknown fixed phase")
    verify_run(data)
    checkpoint = CODEC
    if mode == "writer":
        if codec_run is None:
            raise ValueError("Frozen adapted decoder required")
        verify_run(codec_run)
        result = json.loads((codec_run / "reports/result.json").read_text())
        if result["mean_dice"] < 0.93 or result["p5_dice"] < 0.80 or result["blank"]:
            raise ValueError("Decoder representation gate failed")
        checkpoint = codec_run / "logs/checkpoint-02000.pt"
    r = {
        "mode": mode,
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "base": str(BASE.resolve()),
        "base_sha256": digest(BASE),
        "codec": str(checkpoint.resolve()),
        "codec_sha256": digest(checkpoint),
        "steps": 2000 if mode == "decoder" else 8000,
        "batch": 16,
        "lr": 0.0005 if mode == "decoder" else 0.0002,
        "seed": 2313,
        "generation_seeds": [2012, 2013],
        "reverse_steps": 64,
        "font_condition_dropout": 0.15,
        "pure_noise_probability": 0.02,
        "image_auxiliary_weight": 0.15,
        "global_writer_fixed_across_outputs": True,
        "label_balanced_sampling": True,
        "fresh_optimizer": True,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01013_PROTOCOL.md")),
    }
    run = create("writer-style-" + mode, config=r, seed=r["seed"])
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.writer_style_train",
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
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.ink_diffusion_model import schedule
    from research.corpus.latent_ink_model import (
        InkCodec,
        bounded_code,
        generate,
        reconstruction_loss,
    )
    from research.corpus.writer_style_model import WriterInk

    r = json.loads(request.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    for p, h in [
        (data / "run.json", r["data_sha256"]),
        (data / "reports/arrays.npz", r["arrays_sha256"]),
        (Path(r["base"]), r["base_sha256"]),
        (Path(r["codec"]), r["codec_sha256"]),
    ]:
        if digest(p) != h:
            raise ValueError("Changed immutable input")
    rows = json.loads((data / "reports/samples.json").read_text())
    ids = json.loads((data / "reports/indices.json").read_text())
    train = ids["train"]
    if len(train) != 450 or any(
        rows[i]["dataset_split"] != "train" or not rows[i]["real_personal_pixels"] for i in train
    ):
        raise ValueError("Exactly450 real Train regions required")
    if set(train) & set(ids["validation"] + ids["novel"]):
        raise ValueError("Train/evaluation overlap")
    raw = np.load(data / "reports/arrays.npz")
    target = torch.tensor(raw["target"][train, None], device="cuda")
    reference = torch.tensor(raw["reference"][:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    positions = torch.tensor(raw["positions"], device="cuda")
    codec = InkCodec().cuda()
    codec.load_state_dict(torch.load(r["codec"], map_location="cuda", weights_only=False)["model"])
    if r["mode"] == "decoder":
        codec.encoder.requires_grad_(False)
        model = codec
        mean = std = None
        evaluated = train
    else:
        codec.eval().requires_grad_(False)
        base = torch.load(r["base"], map_location="cuda", weights_only=False)
        model = WriterInk().cuda()
        missing = model.load_state_dict(base["model"], strict=False)
        if missing.unexpected_keys or set(missing.missing_keys) != set(model.state_dict()) - set(
            base["model"]
        ):
            raise ValueError("Unexpected model warm-start keys")
        mean, std = base["mean"], base["std"]
        with torch.no_grad():
            code = (
                torch.cat([codec.encode(target[k : k + 16]) for k in range(0, len(train), 16)])
                - mean
            ) / std
            refcode = (
                torch.cat([codec.encode(reference[k : k + 16]) for k in range(0, len(rows), 16)])
                - mean
            ) / std
        evaluated = ids["evaluated"]
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    noise_rng = torch.Generator(device="cuda").manual_seed(r["seed"])
    groups = {}
    for local, global_id in enumerate(train):
        groups.setdefault(rows[global_id]["text"], []).append(local)
    labels = sorted(groups)
    counts = np.zeros(len(train), dtype=np.int64)
    curve = []
    checkpoints = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/indices.json",
        {
            "train": train,
            "evaluated": evaluated,
            **{k: v for k, v in ids.items() if k not in ["train", "evaluated"]},
        },
    )
    for step in range(1, r["steps"] + 1):
        chosen = rng.integers(len(labels), size=r["batch"])
        ii = np.array([rng.choice(groups[labels[j]]) for j in chosen])
        np.add.at(counts, ii, 1)
        extra = {}
        if r["mode"] == "decoder":
            loss = reconstruction_loss(codec(target[ii]), target[ii])
        else:
            jj = np.array(train)[ii]
            t = 0.005 + 0.99 * torch.rand(len(ii), device="cuda", generator=noise_rng)
            t[
                torch.rand(len(ii), device="cuda", generator=noise_rng)
                < r["pure_noise_probability"]
            ] = 1
            a, s = schedule(t)
            noise = torch.randn(code[ii].shape, device="cuda", generator=noise_rng)
            refs = refcode[jj].clone()
            drop = (
                torch.rand(len(ii), device="cuda", generator=noise_rng)
                < r["font_condition_dropout"]
            )
            refs[drop] = 0
            pred = model(a * code[ii] + s * noise, refs, tokens[jj], t, positions[jj])
            v_loss = (
                (pred - (a * noise - s * code[ii])).square() * (1 + 3 * F.avg_pool2d(target[ii], 8))
            ).mean()
            clean = bounded_code(a * (a * code[ii] + s * noise) - s * pred, mean, std)
            image_loss = reconstruction_loss(codec.decode(clean * std + mean), target[ii])
            loss = v_loss + r["image_auxiliary_weight"] * image_loss
            extra = {"v_loss": float(v_loss.detach()), "image_loss": float(image_loss.detach())}
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "seconds": time.perf_counter() - begin,
                    **extra,
                }
            )
            write(run / "reports/curve.json", curve)
            print(json.dumps(curve[-1]), flush=True)
        if step % 1000 == 0:
            dest = run / f"logs/checkpoint-{step:05d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "mean": mean,
                    "std": std,
                    "numpy_rng": rng.bit_generator.state,
                    "noise_rng": noise_rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                },
                dest,
            )
            checkpoints.append({"step": step, "path": str(dest), "sha256": digest(dest)})
    model.eval()
    generated = {}
    metrics = []
    with torch.inference_mode():
        if r["mode"] == "decoder":
            generated["oracle-reconstruction"] = np.concatenate(
                [
                    codec(target[k : k + 4]).sigmoid()[:, 0].cpu().numpy()
                    for k in range(0, len(train), 4)
                ]
            )
        else:
            for seed in r["generation_seeds"]:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                chunks = []
                for k in range(0, len(evaluated), 4):
                    jj = evaluated[k : k + 4]

                    def conditioned(x, ref, text, t, pos=positions[jj]):
                        return model(x, ref, text, t, pos)

                    output, _ = generate(
                        conditioned,
                        codec,
                        refcode[jj],
                        tokens[jj],
                        torch.randn(refcode[jj].shape, device="cuda", generator=gen),
                        mean,
                        std,
                        r["reverse_steps"],
                    )
                    chunks.append(output[:, 0].cpu().numpy())
                generated[str(seed)] = np.concatenate(chunks)
    real = raw["target"]
    for seed, images in generated.items():
        for j, i in enumerate(evaluated):
            ink = images[j]
            mask = ink > 0.5
            truth = real[i] > 0.5
            has_target = rows[i].get("target_available", True)
            image = run / f"rendered/pages/{seed}-{i:04d}.png"
            Image.fromarray(np.uint8((1 - ink) * 255)).save(image)
            metrics.append(
                {
                    "index": i,
                    "text": rows[i]["text"],
                    "split": rows[i]["dataset_split"],
                    "seed": seed,
                    "image": str(image),
                    "sha256": digest(image),
                    "target_read_for_output": r["mode"] == "decoder",
                    "dice": float(2 * (mask & truth).sum() / max(1, mask.sum() + truth.sum()))
                    if has_target
                    else None,
                    "target_available": has_target,
                    "blank": not mask.any(),
                    "border_pixels": int(
                        mask[:2].sum() + mask[-2:].sum() + mask[:, :2].sum() + mask[:, -2:].sum()
                    ),
                }
            )
    np.savez_compressed(run / "reports/generated.npz", **generated)
    write(run / "reports/generation-metrics.json", metrics)
    write(run / "reports/checkpoints.json", checkpoints)
    write(
        run / "reports/sampling-counts.json",
        {
            "indices": train,
            "counts": counts.tolist(),
            "all_450_used": bool((counts > 0).all()),
            "validation_samples_used": 0,
            "novel_prompts_used": 0,
        },
    )
    dice = [q["dice"] for q in metrics if q["dice"] is not None]
    torch.cuda.synchronize()
    result = {
        "actual_parameter_training": True,
        "updates": r["steps"],
        "train_regions": 450,
        "train_labels": len(labels),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_absolute_change": sum(
            float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
        ),
        "mean_dice": float(np.mean(dice)),
        "p5_dice": float(np.percentile(dice, 5)),
        "blank": sum(q["blank"] for q in metrics),
        "dice_scope": "450 Train oracle reconstruction"
        if r["mode"] == "decoder"
        else "regression plus old document development, excludes novel prompts",
        "seconds": time.perf_counter() - begin,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint_sha256": checkpoints[-1]["sha256"],
        "target_required_for_output": r["mode"] == "decoder",
        "C_used": False,
    }
    if r["mode"] == "decoder":
        result["encoder_exactly_frozen"] = all(
            torch.equal(v, initial[k])
            for k, v in model.state_dict().items()
            if k.startswith("encoder.")
        )
    else:
        result.update(
            writer_vector_l1_change=float((model.writer - initial["writer"]).abs().sum()),
            attention_gain=float(model.attention_gain.detach()),
            single_shared_writer=True,
        )
    write(run / "reports/result.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["decoder", "writer", "worker"])
    p.add_argument("path", type=Path)
    p.add_argument("--codec-run", type=Path)
    args = p.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(launch(args.mode, args.path, args.codec_run))
