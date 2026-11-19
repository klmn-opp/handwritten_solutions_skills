"""Fit uncertain word coordinates with a frozen neural field; NOT generator training."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def adjusted_boxes(base, delta):
    import torch

    shift = delta[..., :2].tanh() * torch.tensor([64.0, 24.0], device=base.device)
    scale = (delta[..., 2:].tanh() * torch.tensor([1.5, 0.6], device=base.device)).exp()
    center = (base[..., :2] + base[..., 2:]) / 2 + shift
    radius = (base[..., 2:] - base[..., :2]) / 2 * scale
    return torch.cat([center - radius, center + radius], -1)


def geometry_penalty(boxes, tokens, delta):
    import torch

    valid = tokens[..., 0] != 0
    center = (boxes[..., 0] + boxes[..., 2]) / 2
    pairs = valid[:, :-1] & valid[:, 1:]
    order = (
        (torch.relu(center[:, :-1] - center[:, 1:] + 0.5).square() * pairs).sum()
        / pairs.sum().clamp_min(1)
        / 100
    )
    lower = torch.relu(-boxes[..., :2]).square()
    upper = torch.relu(boxes[..., 2:] - torch.tensor([512.0, 128.0], device=boxes.device)).square()
    border = ((lower + upper).sum(-1) * valid).sum() / valid.sum().clamp_min(1) / 100
    return order + border + 0.01 * delta.square().mean()


def launch(data: Path, checkpoint: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "seed": 3217,
        "steps": 1000,
        "lr": 0.003,
        "batch": 4,
        "points": 2048,
        "generator_parameter_updates": 0,
        "target_used_to_fit_coordinates": True,
        "coordinates_are_verified_labels": False,
        "C_used": False,
        "protocol_sha256": digest("docs/V01017_PROTOCOL.md"),
    }
    run = create("frozen-word-alignment", seed=r["seed"], config=r)
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    Path("/tmp/context-v01017-alignment").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.whole_word_alignment",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=3600,
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
    from research.corpus.semantic_ink_model import AutoDecodedSemanticInk

    r = json.loads(request.read_text())
    setup(r["seed"])
    run = Path(r["run"])
    data = Path(r["data"])
    assert (
        digest(data / "run.json") == r["data_sha256"]
        and digest(data / "reports/arrays.npz") == r["arrays_sha256"]
    )
    assert digest(r["checkpoint"]) == r["checkpoint_sha256"]
    a = dict(np.load(data / "reports/arrays.npz"))
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    weak = idx["weak"]
    assert len(weak) == 44 and not set(weak) & set(
        idx["development"] + idx["novel"] + idx["strong"]
    )
    state = torch.load(r["checkpoint"], map_location="cuda", weights_only=False)
    assert state["step"] == 24000 and state["config"]["support_weight"] == 0.5
    model = AutoDecodedSemanticInk().cuda().eval()
    model.load_state_dict(state["model"])
    model.requires_grad_(False)
    z = state["training_codes"].mean(0, keepdim=True).detach()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tokens = torch.tensor(a["tokens"][weak], device="cuda")
    base = torch.tensor(a["train_geometry"][weak], device="cuda")
    ink = torch.tensor(a["ink"][weak], device="cuda")
    sdf = torch.tensor(a["sdf"][weak], device="cuda")
    delta = torch.nn.Parameter(torch.zeros_like(base))
    optimizer = torch.optim.Adam([delta], lr=r["lr"])
    rng = np.random.default_rng(r["seed"])
    counts = np.zeros(len(weak), np.int64)
    curve = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    records = []
    images = []

    def output(mode, boxes):
        for j, i in enumerate(weak):
            im = model.image(tokens[j : j + 1], boxes[j : j + 1], style=z)[0].cpu().numpy()
            mask = im > 0.5
            gt = a["ink"][i] > 0.5
            path = run / f"rendered/pages/{mode}-{i:04d}.png"
            Image.fromarray(np.uint8((1 - im) * 255)).save(path)
            records.append(
                {
                    "index": i,
                    "text": rows[i]["text"],
                    "mode": mode,
                    "image": str(path),
                    "sha256": digest(path),
                    "target_seen": True,
                    "not_unconditional_generation": True,
                    "generator_updates": 0,
                    "mean_absolute_error": float(np.abs(im - a["ink"][i]).mean()),
                    "dice": float(2 * (mask & gt).sum() / max(1, mask.sum() + gt.sum())),
                    "blank": not bool(mask.any()),
                    "edge": bool(
                        mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                    ),
                }
            )
            images.append(im)

    output("initial-estimate", base)
    for step in range(1, r["steps"] + 1):
        ii = rng.choice(len(weak), r["batch"])
        np.add.at(counts, ii, 1)
        boxes = adjusted_boxes(base[ii], delta[ii])
        pixel = []
        # Fixed original word frame for the sampler: moving coordinates cannot hide target ink.
        for j in ii:
            valid = np.flatnonzero(a["tokens"][weak[j], :, 0])
            bb = a["train_geometry"][weak[j], rng.choice(valid, r["points"] // 2)]
            center = (bb[:, :2] + bb[:, 2:]) / 2
            radius = (bb[:, 2:] - bb[:, :2]) / 2
            xy = center + rng.uniform(-1.6, 1.6, center.shape) * radius
            px = np.rint(xy[:, 0]).astype(int).clip(0, 511)
            py = np.rint(xy[:, 1]).astype(int).clip(0, 127)
            pixel.append(
                np.concatenate([py * 512 + px, rng.integers(0, 128 * 512, r["points"] // 2)])
            )
        pi = torch.tensor(np.array(pixel), device="cuda")
        points = torch.stack([pi % 512, pi // 512], -1).float()
        logits, dist = model(tokens[ii], boxes, points, style=z.expand(len(ii), -1))
        gt = ink[ii].flatten(1).gather(1, pi)
        ss = sdf[ii].flatten(1).gather(1, pi)
        fit = (
            F.binary_cross_entropy_with_logits(logits, gt, reduction="none") * (1 + 3 * gt)
        ).mean() + 0.2 * (dist - ss).abs().mean()
        penalty = geometry_penalty(boxes, tokens[ii], delta[ii])
        loss = fit + penalty
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite coordinate loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta], 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            q = {
                "step": step,
                "loss": float(loss.detach()),
                "fit": float(fit.detach()),
                "geometry_penalty": float(penalty.detach()),
                "seconds": time.perf_counter() - start,
            }
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
    with torch.no_grad():
        final = adjusted_boxes(base, delta)
    output("optimized-estimate", final)
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    torch.save(
        {
            "delta": delta.detach(),
            "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "step": 1000,
            "config": r,
            "frozen_source_sha256": r["checkpoint_sha256"],
        },
        run / "logs/alignment-step-01000.pt",
    )
    np.savez_compressed(
        run / "reports/estimated-geometry.npz",
        indices=np.array(weak),
        initial=base.cpu().numpy(),
        optimized=final.cpu().numpy(),
        delta=delta.detach().cpu().numpy(),
        shared_style=z.cpu().numpy(),
    )
    np.savez_compressed(run / "reports/generated.npz", ink=np.array(images))
    write(run / "reports/outputs.json", records)
    write(run / "reports/sampling-counts.json", counts.tolist())
    quality = []
    for j, i in enumerate(weak):
        n = int((a["tokens"][i, :, 0] != 0).sum())
        b = final[j, :n].cpu().numpy()
        cx = (b[:, 0] + b[:, 2]) / 2
        before, after = records[j], records[len(weak) + j]
        quality.append(
            {
                "index": i,
                "text": rows[i]["text"],
                "relative_mae_reduction": 1
                - after["mean_absolute_error"] / before["mean_absolute_error"],
                "ordered_centers": bool(np.all(np.diff(cx) > 0)),
                "boxes_inside_page": bool((b[:, :2] >= 0).all() and (b[:, 2:] <= [512, 128]).all()),
                "not_ground_truth": True,
                "content_accuracy_not_established": True,
            }
        )
    write(run / "reports/alignment-quality.json", quality)
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/result.json",
        {
            "coordinate_updates": 1000,
            "generator_parameter_updates": 0,
            "generator_weights_unchanged": True,
            "words": 44,
            "images": 88,
            "median_relative_mae_reduction": float(
                np.median([q["relative_mae_reduction"] for q in quality])
            ),
            "improved_words": sum(q["relative_mae_reduction"] > 0 for q in quality),
            "order_failures": sum(not q["ordered_centers"] for q in quality),
            "box_boundary_failures": sum(not q["boxes_inside_page"] for q in quality),
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "seconds": time.perf_counter() - start,
            "gpu": torch.cuda.get_device_name(),
            "C_used": False,
            "style_quality_passed": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker", "fit"])
    p.add_argument("path", type=Path)
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    worker(a.path) if a.mode == "worker" else launch(a.path, a.checkpoint)
