"""Autonomous, finite content experiments; no imports from production runtime."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write

DATA = Path("workspace/runs/20261119T180917749565Z-writer-style-data-ffe48c04")
BASE = Path(
    "workspace/runs/20261119T173436830135Z-thin-ink-detail-5faffd79/logs/checkpoint-03000.pt"
)
CODEC = Path(
    "workspace/runs/20261119T181050016066Z-writer-style-decoder-a1346b32/logs/checkpoint-02000.pt"
)


def select_admitted(packet: list[dict], review: list[dict], rows: list[dict]) -> list[int]:
    """Conservative labels are inherited; review can only exclude existing Train."""
    expected = {q["index"]: q for q in packet}
    seen = [q["index"] for q in review]
    if len(expected) != len(packet) or len(set(seen)) != len(seen) or set(seen) != set(expected):
        raise ValueError("Complete unique source review required")
    accepted = []
    for q in review:
        i = q["index"]
        if type(q["accept"]) is not bool or q["sha256"] != expected[i]["sha256"]:
            raise ValueError("Explicit review tied to original pixels required")
        if rows[i]["dataset_split"] != "train" or rows[i]["text"] != expected[i]["text"]:
            raise ValueError("Train-only existing labels required")
        if q["accept"]:
            accepted.append(i)
    return sorted(accepted)


def prepare(parent: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(DATA)
    rows = json.loads((DATA / "reports/samples.json").read_text())
    packet = json.loads((parent / "reports/source-review-packet.json").read_text())
    review = json.loads((parent / "reports/source-review.json").read_text())
    admitted = select_admitted(
        packet, review if isinstance(review, list) else review["items"], rows
    )
    if len(admitted) < 24:
        raise ValueError("At least24 unambiguous real words required")
    for q in packet:
        if (
            digest(Path(q["image"])) != q["sha256"]
            or digest(Path(rows[q["index"]]["crop"])) != q["sha256"]
        ):
            raise ValueError("Changed real crop")
    r = create(
        "content-cycle-data",
        config={
            "source": str(DATA.resolve()),
            "source_sha256": digest(DATA / "run.json"),
            "review_sha256": digest(parent / "reports/source-review.json"),
            "C_used": False,
        },
    )
    for name in ["source-review.json", "source-review-packet.json"]:
        (r.path / "reports" / name).write_bytes((parent / "reports" / name).read_bytes())
    write(
        r.path / "reports/indices.json",
        {
            "capacity": admitted[:24],
            "expanded": admitted,
            "novel": json.loads((DATA / "reports/indices.json").read_text())["novel"],
            "all_admitted_are_train": True,
        },
    )
    r.finish("prepared", result={"capacity": 24, "expanded": len(admitted), "no_new_labels": True})
    return r.path


def launch(arm: str, admission: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    if arm not in ["uniform", "endpoint"]:
        raise ValueError("Only fixed equal-budget noise arms enabled")
    for p in [DATA, admission, BASE.parent.parent, CODEC.parent.parent]:
        verify_run(p)
    r = {
        "arm": arm,
        "data": str(DATA.resolve()),
        "data_sha256": digest(DATA / "run.json"),
        "arrays_sha256": digest(DATA / "reports/arrays.npz"),
        "admission": str(admission.resolve()),
        "admission_sha256": digest(admission / "run.json"),
        "base": str(BASE.resolve()),
        "base_sha256": digest(BASE),
        "codec": str(CODEC.resolve()),
        "codec_sha256": digest(CODEC),
        "steps": 4000,
        "batch": 8,
        "lr": 0.0002,
        "seed": 2414,
        "endpoint_probability": 0.02 if arm == "uniform" else 0.50,
        "font_dropout": 0,
        "detail_weight": 0.35,
        "generation_seeds": [2012, 2013],
        "reverse_steps": 64,
        "C_used": False,
        "protocol_sha256": digest(Path("docs/V01014_PROTOCOL.md")),
    }
    run = create("content-cycle-" + arm, config=r, seed=r["seed"])
    print(run.path, flush=True)
    try:
        import numpy as np

        from research.corpus.thin_ink_data import component_weights

        ids = json.loads((admission / "reports/indices.json").read_text())["capacity"]
        targets = np.load(DATA / "reports/arrays.npz")["target"]
        np.save(
            run.path / "reports/components.npy",
            np.stack([component_weights(targets[i]) for i in ids])[:, None],
        )
        write(
            run.path / "reports/request.json",
            {
                **r,
                "run": str(run.path),
                "components_sha256": digest(run.path / "reports/components.npy"),
            },
        )
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.content_cycle",
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


def worker(path: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.ink_diffusion_model import schedule
    from research.corpus.latent_ink_model import InkCodec, LatentInk, bounded_code, generate
    from research.corpus.thin_ink_model import ink_detail_loss

    r = json.loads(path.read_text())
    setup(r["seed"])
    run = Path(r["run"])
    data = Path(r["data"])
    admission = Path(r["admission"])
    for p, h in [
        (data / "run.json", r["data_sha256"]),
        (data / "reports/arrays.npz", r["arrays_sha256"]),
        (admission / "run.json", r["admission_sha256"]),
        (Path(r["base"]), r["base_sha256"]),
        (Path(r["codec"]), r["codec_sha256"]),
    ]:
        if digest(p) != h:
            raise ValueError("Changed frozen input")
    indices = json.loads((admission / "reports/indices.json").read_text())
    train = indices["capacity"]
    evaluated = train + indices["novel"]
    rows = json.loads((data / "reports/samples.json").read_text())
    a = np.load(data / "reports/arrays.npz")
    targets = a["target"]
    references = a["reference"]
    assert (
        len(train) == 24
        and len(set(train)) == 24
        and all(rows[i]["dataset_split"] == "train" for i in train)
    )
    target = torch.tensor(targets[train, None], device="cuda")
    ref = torch.tensor(references[:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    # Derive component weights in CPU preparation environment, not in Torch runtime.
    if digest(run / "reports/components.npy") != r["components_sha256"]:
        raise ValueError("Changed component supervision")
    weights = torch.tensor(np.load(run / "reports/components.npy"), device="cuda")
    codec = InkCodec().cuda().eval().requires_grad_(False)
    codec.load_state_dict(torch.load(r["codec"], map_location="cuda", weights_only=False)["model"])
    base = torch.load(r["base"], map_location="cuda", weights_only=False)
    mean, std = base["mean"], base["std"]
    model = LatentInk().cuda()
    model.load_state_dict(base["model"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    with torch.no_grad():
        code = (
            torch.cat([codec.encode(target[k : k + 16]) for k in range(0, 24, 16)]) - mean
        ) / std
        refcode = (
            torch.cat([codec.encode(ref[k : k + 16]) for k in range(0, len(rows), 16)]) - mean
        ) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    ng = torch.Generator(device="cuda").manual_seed(r["seed"])
    curve = []
    counts = np.zeros(24, dtype=np.int64)
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/indices.json",
        {"train": train, "evaluated": evaluated, "novel": indices["novel"]},
    )
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(24, r["batch"], replace=True)
        np.add.at(counts, ii, 1)
        jj = np.array(train)[ii]
        t = 0.005 + 0.99 * torch.rand(len(ii), device="cuda", generator=ng)
        t[torch.rand(len(ii), device="cuda", generator=ng) < r["endpoint_probability"]] = 1
        alpha, sigma = schedule(t)
        noise = torch.randn(code[ii].shape, device="cuda", generator=ng)
        x = alpha * code[ii] + sigma * noise
        pred = model(x, refcode[jj], tokens[jj], t)
        velocity = alpha * noise - sigma * code[ii]
        vl = ((pred - velocity).square() * (1 + 3 * F.avg_pool2d(target[ii], 8))).mean()
        clean = bounded_code(alpha * x - sigma * pred, mean, std)
        detail = ink_detail_loss(codec.decode(clean * std + mean), target[ii], weights[ii])
        loss = vl + r["detail_weight"] * detail
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "v_loss": float(vl.detach()),
                    "detail": float(detail.detach()),
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
                    "mean": mean,
                    "std": std,
                    "numpy_rng": rng.bit_generator.state,
                    "noise_rng": ng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs = {}
    metrics = []
    with torch.inference_mode():
        for seed in r["generation_seeds"]:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            chunks = []
            direct = []
            for k in range(0, len(evaluated), 4):
                ii = evaluated[k : k + 4]
                noise = torch.randn(refcode[ii].shape, device="cuda", generator=gen)
                im, _ = generate(
                    model, codec, refcode[ii], tokens[ii], noise, mean, std, r["reverse_steps"]
                )
                chunks.append(im[:, 0].cpu().numpy())
                z = bounded_code(
                    -model(noise, refcode[ii], tokens[ii], torch.ones(len(ii), device="cuda")),
                    mean,
                    std,
                )
                direct.append(codec.decode(z * std + mean).sigmoid()[:, 0].cpu().numpy())
            outputs[f"multistep-{seed}"] = np.concatenate(chunks)
            outputs[f"one-step-{seed}"] = np.concatenate(direct)
    for label, images in outputs.items():
        for j, i in enumerate(evaluated):
            image = images[j]
            mask = image > 0.5
            truth = targets[i] > 0.5
            dest = run / f"rendered/pages/{label}-{i:04d}.png"
            Image.fromarray(np.uint8((1 - image) * 255)).save(dest)
            metrics.append(
                {
                    "method": label,
                    "index": i,
                    "text": rows[i]["text"],
                    "image": str(dest),
                    "sha256": digest(dest),
                    "dice": None
                    if i in indices["novel"]
                    else float(2 * (mask & truth).sum() / max(1, mask.sum() + truth.sum())),
                    "blank": not bool(mask.any()),
                    "edge": bool(
                        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                    ),
                    "target_read_for_output": False,
                }
            )
    np.savez_compressed(run / "reports/generated.npz", **outputs)
    write(run / "reports/generation-metrics.json", metrics)
    write(
        run / "reports/sampling-counts.json",
        {"indices": train, "counts": counts.tolist(), "validation_used": 0, "novel_used": 0},
    )
    write(
        run / "reports/result.json",
        {
            "updates": r["steps"],
            "actual_parameter_training": True,
            "mean_capacity_dice": {
                key: float(
                    np.mean(
                        [q["dice"] for q in metrics if q["method"] == key and q["dice"] is not None]
                    )
                )
                for key in outputs
            },
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{r['steps']:05d}.pt"),
            "gpu": torch.cuda.get_device_name(),
            "blank": sum(q["blank"] for q in metrics),
            "C_used": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "uniform", "endpoint", "worker"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.path))
    elif args.mode == "worker":
        worker(args.path)
    else:
        print(launch(args.mode, args.path))
