"""Real-word code fitting and image-free prior generation, kept explicitly separate."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(
    data: Path,
    warmup: Path,
    resume: Path | None = None,
    support: Path | None = None,
    support_weight: float = 0.0,
) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    if support:
        verify_run(support)
        if resume is None or support_weight not in {0.0, 0.5}:
            raise ValueError("Registered prior comparison requires frozen 20k source")
    elif support_weight:
        raise ValueError("Real support data required")
    r = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "warmup": str(warmup.resolve()),
        "warmup_sha256": digest(warmup),
        "seed": 3015,
        "steps": 16000 if resume and not support else 4000,
        "resume": str(resume.resolve()) if resume else None,
        "resume_sha256": digest(resume) if resume else None,
        "support": str(support.resolve()) if support else None,
        "support_sha256": digest(support / "run.json") if support else None,
        "support_weight": support_weight,
        "batch": 4,
        "points": 2048,
        "model_lr": 0.0001,
        "code_lr": 0.001,
        "code_l2_weight": 0.01,
        "latent": 2,
        "architecture": "semantic-word-autodecoder/1",
        "prior": "Train-code mean/covariance plus 1e-4 diagonal, NOT validated population",
        "C_used": False,
        "target_at_generation": False,
        "target_geometry_at_generation": False,
        "protocol_sha256": digest("docs/V01016_PROTOCOL.md"),
    }
    run = create("semantic-word-autodecoder", seed=r["seed"], config=r)
    write(run.path / "reports/request.json", {**r, "run": str(run.path)})
    name = (
        ("prior-real" if support_weight else "prior-control")
        if support
        else "autodecoder-extended"
        if resume
        else "autodecoder"
    )
    Path(f"/tmp/context-v01016-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.semantic_autodecoder_train",
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


def worker(request: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_ink_model import AutoDecodedSemanticInk
    from research.corpus.semantic_raw_data import apply_metrics

    r = json.loads(request.read_text())
    setup(r["seed"])
    run, data = Path(r["run"]), Path(r["data"])
    for path, key in [
        (data / "run.json", "data_sha256"),
        (data / "reports/arrays.npz", "arrays_sha256"),
        (r["warmup"], "warmup_sha256"),
    ]:
        assert digest(path) == r[key]
    with np.load(data / "reports/arrays.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    idx = json.loads((data / "reports/indices.json").read_text())
    rows = json.loads((data / "reports/samples.json").read_text())
    metrics = json.loads((data / "reports/class-metrics.json").read_text())
    train = idx["train"]
    assert all(rows[i]["split"] == "train" and rows[i]["no_inverse_registration"] for i in train)
    assert not set(train) & set(idx["development"] + idx["novel"])
    arrays["generation_boxes"] = apply_metrics(arrays["tokens"], arrays["boxes"], metrics)
    intact = r.get("intact_geometry")
    if intact:
        arrays["anchor_boxes"] = arrays[
            "fitted_geometry" if intact == "fitted" else "train_geometry"
        ]
    x = {k: torch.tensor(v, device="cuda") for k, v in arrays.items()}
    saved = torch.load(r["warmup"], map_location="cuda", weights_only=False)
    assert saved["step"] == 24000 and saved["config"]["arm"] == "identity-extended"
    model = AutoDecodedSemanticInk().cuda()
    model.load_identity(saved["model"])
    codes = torch.nn.Parameter(torch.randn(len(train), r["latent"], device="cuda"))
    teacher = None
    if intact:
        from research.corpus.semantic_ink_model import SemanticInk

        assert digest(r["intact_source"]) == r["intact_source_sha256"]
        source = torch.load(r["intact_source"], map_location="cuda", weights_only=False)
        assert source["step"] == 24000 and source["config"]["support_weight"] == 0.5
        model.load_state_dict(source["model"])
        with torch.no_grad():
            source_mu = source["training_codes"].mean(0)
            source_chol = torch.linalg.cholesky(
                torch.cov(source["training_codes"].T) + torch.eye(2, device="cuda") * 1e-4
            )
            codes.copy_(codes @ source_chol.T + source_mu)
            for j, i in enumerate(source["training_indices"]):
                codes[train.index(i)].copy_(source["training_codes"][j])
        teacher = SemanticInk(context_scale=0).cuda().eval()
        teacher.load_state_dict(saved["model"])
        teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": r["model_lr"]},
            {"params": [codes], "lr": r["code_lr"]},
        ]
    )
    rng = np.random.default_rng(r["seed"])
    start = 0
    if r.get("resume"):
        assert digest(r["resume"]) == r["resume_sha256"]
        previous = torch.load(r["resume"], map_location="cuda", weights_only=False)
        assert previous["step"] == (20000 if r.get("support") else 4000)
        assert previous["config"]["arrays_sha256"] == r["arrays_sha256"]
        assert previous["training_indices"] == train
        model.load_state_dict(previous["model"])
        with torch.no_grad():
            codes.copy_(previous["training_codes"])
        optimizer.load_state_dict(previous["optimizer"])
        rng.bit_generator.state = previous["numpy_rng"]
        torch.set_rng_state(previous["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([q.cpu() for q in previous["cuda_rng"]])
        start = previous["step"]
    support_rng = np.random.default_rng(r.get("prior_seed", 3216))
    support_groups = None
    if r.get("support"):
        from research.corpus.real_ink_support import nearest_real_loss

        path = Path(r["support"])
        assert digest(path / "run.json") == r["support_sha256"]
        with np.load(path / "reports/arrays.npz") as lib:
            support_groups = {
                int(c): torch.tensor(lib["ink"][lib["labels"] == c], device="cuda").flatten(1)
                for c in np.unique(lib["labels"])
            }
        support_classes = sorted(support_groups)
        font_classes = []
        if intact:
            training_classes = sorted(set(arrays["tokens"][train, :, 0].ravel()) - {0})
            teacher_data = Path(saved["config"]["data"])
            assert digest(teacher_data / "run.json") == saved["config"]["data_sha256"]
            teacher_train = json.loads((teacher_data / "reports/indices.json").read_text())["train"]
            with np.load(teacher_data / "reports/arrays.npz") as teacher_arrays:
                teacher_classes = set(teacher_arrays["tokens"][teacher_train, :, 0].ravel()) - {0}
            assert set(support_classes) <= teacher_classes
            font_classes = sorted(teacher_classes - set(support_classes))
            assert font_classes
            write(
                run / "reports/class-supervision.json",
                {
                    "real_glyph_support_classes": [chr(c) for c in support_classes],
                    "whole_word_plus_font_teacher_classes": [
                        chr(c) for c in font_classes if c in training_classes
                    ],
                    "font_auxiliary_retention_only_classes": [
                        chr(c) for c in font_classes if c not in training_classes
                    ],
                    "font_teacher_classes": [chr(c) for c in font_classes],
                    "font_teacher_sha256": r["warmup_sha256"],
                    "font_teacher_not_updated": True,
                    "neural_outputs_are_not_collected_glyphs": True,
                },
            )
        # Fixed prior from the common 20k source; no target- or label-specific code.
        with torch.no_grad():
            support_mu = codes.mean(0)
            support_chol = torch.linalg.cholesky(
                torch.cov(codes.T) + torch.eye(2, device="cuda") * 1e-4
            )
            if intact:
                support_mu, support_chol = source_mu, source_chol
    initial_codes = codes.detach().clone()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial, "codes": initial_codes}, run / "logs/initial-state.pt")
    counts = np.zeros(len(rows), np.int64)
    curve = []
    begin = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    for step in range(start + 1, start + r["steps"] + 1):
        if intact:
            ii = np.concatenate([rng.choice(idx["strong"], 2), rng.choice(idx["weak"], 2)])
            positions = np.array([train.index(i) for i in ii])
        else:
            positions = rng.choice(len(train), r["batch"])
            ii = np.asarray(train)[positions]
        np.add.at(counts, ii, 1)
        pixel = []
        for i in ii:
            valid = np.flatnonzero(arrays["tokens"][i, :, 0])
            sampling_boxes = arrays["train_geometry"] if intact else arrays["anchor_boxes"]
            boxes = sampling_boxes[i, rng.choice(valid, r["points"] // 2)]
            center = (boxes[:, :2] + boxes[:, 2:]) / 2
            radius = (boxes[:, 2:] - boxes[:, :2]) / 2
            xy = center + rng.uniform(-1.6, 1.6, center.shape) * radius
            a = np.rint(xy[:, 1]).astype(int).clip(0, 127) * 512 + np.rint(xy[:, 0]).astype(
                int
            ).clip(0, 511)
            pixel.append(np.concatenate([a, rng.integers(0, 128 * 512, r["points"] - len(a))]))
        pi = torch.tensor(np.array(pixel), device="cuda")
        points = torch.stack([pi % 512, pi // 512], -1).float()
        logits, distance = model(
            x["tokens"][ii], x["anchor_boxes"][ii], points, style=codes[positions]
        )
        gt = x["ink"][ii].flatten(1).gather(1, pi)
        sdf = x["sdf"][ii].flatten(1).gather(1, pi)
        reconstruction = (
            F.binary_cross_entropy_with_logits(logits, gt, reduction="none") * (1 + 3 * gt)
        ).mean()
        reconstruction = reconstruction + 0.2 * (distance - sdf).abs().mean()
        penalty = codes.square().mean()
        loss = reconstruction + r["code_l2_weight"] * penalty
        prior_loss = torch.zeros((), device="cuda")
        if support_groups is not None:
            selected = (
                np.concatenate(
                    [support_rng.choice(support_classes, 2), support_rng.choice(font_classes, 2)]
                )
                if intact
                else support_rng.choice(support_classes, 4)
            )
            # Random legal neighbour identities prevent the code from disabling a
            # character in an unfamiliar word. These are conditions, not fake ink.
            neighbours = support_rng.choice([0] + support_classes + font_classes, (4, 2))
            tt = torch.zeros(4, 2, 3, dtype=torch.long, device="cuda")
            tt[:, 0, 0] = torch.tensor(selected, device="cuda")
            tt[:, 0, 1:] = torch.tensor(neighbours, device="cuda")
            bb = torch.zeros(4, 2, 4, device="cuda")
            bb[:, 0] = torch.tensor([16, 16, 80, 80], device="cuda")
            xy = support_rng.integers(8, 88, (4, 768, 2))
            xy = np.concatenate([xy, support_rng.integers(0, 96, (4, 256, 2))], 1)
            pp = torch.tensor(xy, device="cuda").float()
            zz = (
                torch.tensor(
                    support_rng.standard_normal((4, 2)), device="cuda", dtype=torch.float32
                )
                @ support_chol.T
                + support_mu
            )
            prior_logits = model(tt, bb, pp, style=zz)[0]
            pixel_ids = torch.tensor(xy[..., 1] * 96 + xy[..., 0], device="cuda")
            if teacher is not None:
                with torch.no_grad():
                    teacher_target = teacher(tt, bb, pp)[0].sigmoid()
            prior_terms = []
            for j, c in enumerate(selected):
                if int(c) in support_groups:
                    prior_terms.append(
                        nearest_real_loss(prior_logits[j], support_groups[int(c)][:, pixel_ids[j]])
                    )
                else:
                    target = teacher_target[j]
                    prior_terms.append(
                        (
                            F.binary_cross_entropy_with_logits(
                                prior_logits[j], target, reduction="none"
                            )
                            * (1 + 3 * target)
                        ).mean()
                    )
            prior_loss = torch.stack(prior_terms).mean()
            loss = loss + r["support_weight"] * prior_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite auto-decoder loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), codes], 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            q = {
                "step": step,
                "loss": float(loss.detach()),
                "reconstruction": float(reconstruction.detach()),
                "code_l2": float(penalty.detach()),
                "real_prior_loss": float(prior_loss.detach()),
                "seconds": time.perf_counter() - begin,
            }
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
        if step % 1000 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "training_codes": codes.detach(),
                    "training_indices": train,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": r,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "support_rng": support_rng.bit_generator.state,
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    if teacher is not None:
        assert all(torch.equal(v, saved["model"][k]) for k, v in teacher.state_dict().items())
    with torch.no_grad():
        mu = codes.mean(0)
        cov = torch.cov(codes.T)
        chol = torch.linalg.cholesky(cov + torch.eye(r["latent"], device="cuda") * 1e-4)
    write(
        run / "reports/prior.json",
        {
            "mean": mu.tolist(),
            "covariance": cov.tolist(),
            "cholesky": chol.tolist(),
            "eigenvalues": torch.linalg.eigvalsh(cov).tolist(),
            "estimated_from_train_only": True,
        },
    )
    images, records = [], []
    with torch.inference_mode():
        for seed in [2012, 2013, None]:
            z = (
                mu[None]
                + torch.randn(
                    1,
                    r["latent"],
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
                @ chol.T
                if seed
                else None
            )
            for i in train + idx["development"] + idx["novel"] if seed else train:
                style = z if seed else codes[train.index(i) : train.index(i) + 1]
                boxes = x["generation_boxes"][i : i + 1] if seed else x["anchor_boxes"][i : i + 1]
                im = model.image(x["tokens"][i : i + 1], boxes, style=style)[0].cpu().numpy()
                mode = "prior" if seed else "training-code-reconstruction"
                path = run / f"rendered/pages/{mode}-{seed or 0}-{i:04d}.png"
                Image.fromarray(np.uint8((1 - im) * 255)).save(path)
                mask = im > 0.5
                records.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "index": i,
                        "text": rows[i]["text"],
                        "split": rows[i]["split"],
                        "latent": style[0].tolist(),
                        "image": str(path),
                        "sha256": digest(path),
                        "target_seen": not bool(seed),
                        "target_geometry_used": not bool(seed),
                        "blank": not bool(mask.any()),
                        "edge": bool(
                            mask[:2].any()
                            or mask[-2:].any()
                            or mask[:, :2].any()
                            or mask[:, -2:].any()
                        ),
                    }
                )
                images.append(im)
    np.savez_compressed(run / "reports/generated.npz", ink=np.array(images))
    write(run / "reports/generation-metrics.json", records)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/result.json",
        {
            "new_model_updates": r["steps"],
            "inherited_model_updates": r.get("initial_model_updates", 24000) + start,
            "model_lineage_updates": r.get("initial_model_updates", 24000) + start + r["steps"],
            "actual_parameter_training": True,
            "training_code_parameters": codes.numel(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "training_code_l1_change": float((codes - initial_codes).abs().sum().detach()),
            "seconds_training_and_generation": time.perf_counter() - begin,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{start + r['steps']:05d}.pt"),
            "prior_outputs": sum(q["mode"] == "prior" for q in records),
            "training_reconstruction_outputs": len(train),
            "real_draws": int(counts[train].sum()),
            "development_draws": int(counts[idx["development"]].sum()),
            "novel_draws": int(counts[idx["novel"]].sum()),
            "C_used": False,
            "style_quality_passed": False,
            "support_loss_weight": r.get("support_weight", 0),
            "font_teacher_unchanged": teacher is not None,
            "intact_geometry": intact,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "worker"])
    p.add_argument("path", type=Path)
    p.add_argument("--warmup", type=Path)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        launch(args.path, args.warmup, args.resume)
