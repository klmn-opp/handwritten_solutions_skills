"""Direct verified-glyph neural training; generation accepts no reference pixels."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(
    data: Path,
    teacher: Path,
    words: Path,
    control: Path | None = None,
    modulated=False,
    context_independent=False,
    continuation: Path | None = None,
    distance_regularized=False,
) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [data, teacher.parent.parent, words]:
        verify_run(p)
    if control:
        verify_run(control)
    if modulated and control is None:
        raise ValueError("Layer modulation requires the registered hierarchical control")
    if context_independent and not modulated:
        raise ValueError("Context independence is registered for the modulated control only")
    if continuation:
        verify_run(continuation.parent.parent)
        previous = json.loads((continuation.parent.parent / "reports/request.json").read_text())
        if (
            not context_independent
            or not previous.get("context_independent")
            or previous["data_sha256"] != digest(data / "run.json")
            or previous["steps"] != 8000
            or previous.get("continuation")
        ):
            raise ValueError("Registered continuation requires the same 8k local control/data")
    if distance_regularized and not continuation:
        raise ValueError("Distance diagnostic requires the frozen 8k local control")
    data_config = json.loads((data / "run.json").read_text())["config"]
    class_coordinates = data_config.get("per_class_coordinates", False)
    density_normalized = any(
        "density_gain" in q for q in json.loads((data / "reports/samples.json").read_text())
    )
    cfg = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/arrays.npz"),
        "teacher": str(teacher.resolve()),
        "teacher_sha256": digest(teacher),
        "word_data": str(words.resolve()),
        "word_data_sha256": digest(words / "run.json"),
        "steps": 4000 if continuation else 8000,
        "seed": 3518,
        "batch": 4,
        "points": 1024,
        "model_lr": 0.0001,
        "code_lr": 0.001,
        "code_l2": 0.01,
        "support_weight": 0.5,
        "inherited_model_updates": 32000 if continuation else 24000,
        "continuation": str(continuation.resolve()) if continuation else None,
        "continuation_sha256": digest(continuation) if continuation else None,
        "distance_regularized": bool(distance_regularized),
        "distance_gradient_weight": 0.1 if distance_regularized else 0.0,
        "architecture": "semantic-distance-local-ink/1"
        if distance_regularized
        else "semantic-context-independent-local-ink/1"
        if context_independent
        else "semantic-modulated-local-ink/1"
        if modulated
        else "semantic-hierarchical-local-ink/1"
        if control
        else "semantic-word-autodecoder/1",
        "hierarchical": control is not None,
        "modulated": bool(modulated),
        "context_independent": bool(context_independent),
        "class_coordinates": class_coordinates,
        "density_normalized": density_normalized,
        "capacity_diagnostic": data_config.get("capacity_diagnostic_only", False),
        "initial_code_groups": data_config.get("initial_code_groups"),
        "control_initial": str(control / "logs/initial-state.pt") if control else None,
        "control_initial_sha256": digest(control / "logs/initial-state.pt") if control else None,
        "local_seed_offsets": {"glyph": 100000, "word": 200000} if control else None,
        "no_target_or_reference_at_generation": True,
        "local_component_not_end_to_end": True,
        "C_used": False,
        "protocol_sha256": digest("docs/V01018_PROTOCOL.md"),
    }
    run = create("verified-local-ink-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    name = (
        "ft-distance-gradient"
        if distance_regularized
        else "ft-continuation-control"
        if continuation
        else "ft-context-independent"
        if cfg["capacity_diagnostic"] and context_independent
        else "context-independent"
        if context_independent
        else "ft-capacity"
        if cfg["capacity_diagnostic"]
        else "modulated"
        if modulated
        else "density-hierarchical"
        if density_normalized
        else "class-hierarchical"
        if class_coordinates
        else "hierarchical"
        if control
        else "model"
    )
    Path(f"/tmp/context-v01018-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.local_real_train",
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


def worker(path: Path) -> None:
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.local_sampling import sample_indices, sample_points
    from research.corpus.real_ink_support import nearest_real_loss
    from research.corpus.semantic_ink_model import (
        AutoDecodedSemanticInk,
        ContextIndependentLocalInk,
        HierarchicalSemanticInk,
        ModulatedSemanticInk,
        SemanticInk,
        SignedDistanceLocalInk,
        distance_gradient_loss,
    )
    from research.corpus.semantic_raw_data import apply_metrics

    cfg = json.loads(path.read_text())
    setup(cfg["seed"])
    run, data, words = (Path(cfg[k]) for k in ["run", "data", "word_data"])
    for p, h in [
        (data / "run.json", cfg["data_sha256"]),
        (data / "reports/arrays.npz", cfg["arrays_sha256"]),
        (cfg["teacher"], cfg["teacher_sha256"]),
        (words / "run.json", cfg["word_data_sha256"]),
    ]:
        assert digest(p) == h
    with np.load(data / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    x = {k: torch.tensor(v, device="cuda") for k, v in arrays.items()}
    rows = json.loads((data / "reports/samples.json").read_text())
    group_count = int(arrays["groups"].max()) + 1
    by_class = {int(c): np.flatnonzero(arrays["labels"] == c) for c in np.unique(arrays["labels"])}
    real_classes = sorted(by_class)
    saved = torch.load(cfg["teacher"], map_location="cuda", weights_only=False)
    assert saved["step"] == 24000 and saved["config"]["arm"] == "identity-extended"
    teacher = SemanticInk(context_scale=0).cuda().eval()
    teacher.load_state_dict(saved["model"])
    teacher.requires_grad_(False)
    hierarchical = cfg.get("hierarchical", False)
    model = (
        SignedDistanceLocalInk()
        if cfg.get("distance_regularized")
        else ContextIndependentLocalInk()
        if cfg.get("context_independent")
        else ModulatedSemanticInk()
        if cfg.get("modulated")
        else HierarchicalSemanticInk()
        if hierarchical
        else AutoDecodedSemanticInk()
    ).cuda()
    model.load_identity(saved["model"])
    codes = torch.nn.Parameter(torch.randn(group_count, 2, device="cuda") * 0.1)
    local_priors = {}
    if hierarchical:
        assert digest(cfg["control_initial"]) == cfg["control_initial_sha256"]
        previous = torch.load(cfg["control_initial"], map_location="cuda", weights_only=False)
        initial_word_codes = previous["codes"]
        if cfg.get("initial_code_groups") is not None:
            initial_word_codes = initial_word_codes[cfg["initial_code_groups"]]
        assert initial_word_codes.shape == codes.shape
        with torch.no_grad():
            codes.copy_(initial_word_codes)
        local_priors = {
            int(k): {
                name: torch.tensor(q[name], device="cuda", dtype=torch.float32)
                for name in ["mean", "cholesky"]
            }
            for k, q in json.loads((data / "reports/local-priors.json").read_text()).items()
        }

    def target_style(ii):
        word_style = codes[x["groups"][ii]]
        if not hierarchical:
            return word_style
        local = torch.zeros(len(ii), 2, 8, device="cuda")
        local[:, 0] = x["local_codes"][ii]
        return torch.cat([word_style[:, None].expand(-1, 2, -1), local], -1)

    def sampled_style(tokens, word_style, generator=None):
        if not hierarchical:
            return word_style, None
        if cfg.get("matched_neighborhood"):
            from research.corpus.local_prior import sample_neighborhood

            local = torch.zeros(*tokens.shape[:2], 8, device="cuda")
            style = torch.cat([word_style[:, None].expand(-1, tokens.shape[1], -1), local], -1)
            selected_centers = torch.full(tokens.shape[:2], -1, device="cuda", dtype=torch.long)
            for i in range(len(tokens)):
                for j, c in enumerate(tokens[i, :, 0].tolist()):
                    if c in by_class:
                        ids = by_class[c]
                        joint = torch.cat([codes[x["groups"][ids]], x["local_codes"][ids]], -1)
                        value, selected = sample_neighborhood(
                            joint, cfg["neighborhood_sigma"], generator=generator
                        )
                        style[i, j] = value
                        selected_centers[i, j] = int(ids[selected])
            return style, selected_centers
        local = torch.zeros(*tokens.shape[:2], 8, device="cuda")
        for i in range(len(tokens)):
            for j, c in enumerate(tokens[i, :, 0].tolist()):
                if c in local_priors:
                    q = local_priors[c]
                    local[i, j] = (
                        q["mean"]
                        + torch.randn(8, device="cuda", generator=generator) @ q["cholesky"].T
                    )
        return torch.cat([word_style[:, None].expand(-1, tokens.shape[1], -1), local], -1), None

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": cfg["model_lr"]},
            {"params": [codes], "lr": cfg["code_lr"]},
        ]
    )
    td = Path(saved["config"]["data"])
    assert digest(td / "run.json") == saved["config"]["data_sha256"]
    ti = json.loads((td / "reports/indices.json").read_text())["train"]
    with np.load(td / "reports/arrays.npz") as f:
        alphabet = sorted(set(int(c) for c in f["tokens"][ti, :, 0].ravel()) - {0})
    font_classes = sorted(set(alphabet) - set(real_classes))
    assert font_classes and set(real_classes) <= set(alphabet)
    write(
        run / "reports/class-supervision.json",
        {
            "real_classes": [chr(c) for c in real_classes],
            "font_teacher_only_classes": [chr(c) for c in font_classes],
            "no_verified_data_from_development_or_C": True,
        },
    )
    rng = np.random.default_rng(cfg["seed"])
    if cfg.get("continuation"):
        assert digest(cfg["continuation"]) == cfg["continuation_sha256"]
        previous = torch.load(cfg["continuation"], map_location="cuda", weights_only=False)
        assert previous["step"] == cfg.get("continuation_step", 8000)
        for key in [
            "arrays_sha256",
            "data_sha256",
            "teacher_sha256",
            "word_data_sha256",
            "model_lr",
            "code_lr",
            "code_l2",
            "support_weight",
            "context_independent",
        ]:
            assert previous["config"][key] == cfg[key], key
        model.load_state_dict(previous["model"])
        with torch.no_grad():
            codes.copy_(previous["training_codes"])
        if cfg.get("capacity_comparison"):
            assert cfg.get("reset_optimizer") and cfg["hidden_width"] in {128, 256}
            assert cfg["steps"] == 4000 and previous["step"] == 16000
            assert previous["config"].get("class_balanced")
            conversion = {"width": cfg["hidden_width"], "optimizer_reset": True}
            if cfg["hidden_width"] == 256:
                from research.corpus.local_capacity_model import widen_local_field

                probe = torch.arange(0, 96, 3, device="cuda").float()
                yy, xx = torch.meshgrid(probe, probe, indexing="ij")
                pp = torch.stack([xx.flatten(), yy.flatten()], -1)[None]
                with torch.no_grad():
                    ii = np.arange(len(rows))
                    style = target_style(ii)
                    before = model(
                        x["tokens"], x["boxes"], pp.expand(len(rows), -1, -1), style=style
                    )
                    model = widen_local_field(model, seed=3719, noise_ratio=0.01)
                    after = model(
                        x["tokens"], x["boxes"], pp.expand(len(rows), -1, -1), style=style
                    )
                    errors = [
                        float((a - b).abs().max()) for a, b in zip(before, after, strict=True)
                    ]
                    assert all(
                        torch.allclose(a, b, atol=1e-4, rtol=2e-5)
                        for a, b in zip(before, after, strict=True)
                    )
                conversion.update(initial_max_errors=errors, checked_real_conditions=len(rows))
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.parameters(), "lr": cfg["model_lr"]},
                    {"params": [codes], "lr": cfg["code_lr"]},
                ]
            )
            write(run / "reports/capacity-initialization.json", conversion)
        else:
            assert not cfg.get("reset_optimizer") and cfg.get("hidden_width", 128) == 128
            optimizer.load_state_dict(previous["optimizer"])
        rng.bit_generator.state = previous["numpy_rng"]
        torch.set_rng_state(previous["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([q.cpu() for q in previous["cuda_rng"]])
    start_step = 0
    if cfg.get("recovery_checkpoint"):
        assert cfg.get("capacity_comparison") and cfg["hidden_width"] == 256
        assert digest(cfg["recovery_checkpoint"]) == cfg["recovery_checkpoint_sha256"]
        recovery = torch.load(cfg["recovery_checkpoint"], map_location="cuda", weights_only=False)
        assert recovery["step"] == cfg["recovery_step"] == 3000 and cfg["steps"] == 4000
        for key, value in recovery["config"].items():
            if key not in {"run", "inherited_model_updates", "protocol_sha256"}:
                assert cfg[key] == value, key
        model.load_state_dict(recovery["model"])
        with torch.no_grad():
            codes.copy_(recovery["training_codes"])
        optimizer.load_state_dict(recovery["optimizer"])
        rng.bit_generator.state = recovery["numpy_rng"]
        torch.set_rng_state(recovery["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([q.cpu() for q in recovery["cuda_rng"]])
        start_step = recovery["step"]
        write(
            run / "reports/recovery.json",
            {
                "restored_step": start_step,
                "remaining_updates": cfg["steps"] - start_step,
                "weights_optimizer_codes_all_rng_restored": True,
                "sampling_counts_cover_recovery_segment_only": True,
            },
        )
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    initial_codes = codes.detach().clone()
    torch.save({"model": initial, "codes": initial_codes}, run / "logs/initial-state.pt")
    counts = np.zeros(len(rows), np.int64)
    curve = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )

    def prior():
        with torch.no_grad():
            mu = codes.mean(0)
            chol = torch.linalg.cholesky(torch.cov(codes.T) + torch.eye(2, device="cuda") * 1e-4)
        return mu, chol

    for step in range(start_step + 1, cfg["steps"] + 1):
        batch = cfg["batch"]
        ii = sample_indices(rng, by_class, batch, balanced=cfg.get("class_balanced", False))
        np.add.at(counts, ii, 1)
        xy = sample_points(rng, batch, cfg["points"])
        pp = torch.tensor(xy, device="cuda").float()
        if cfg.get("distance_regularized"):
            pp.requires_grad_(True)
        pi = torch.tensor(xy[..., 1] * 96 + xy[..., 0], device="cuda")
        logits, distance = model(x["tokens"][ii], x["boxes"][ii], pp, style=target_style(ii))
        target = x["ink"][ii].flatten(1).gather(1, pi)
        sdf = x["sdf"][ii].flatten(1).gather(1, pi)
        fit = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * target)
        ).mean() + 0.2 * (distance - sdf).abs().mean()
        selected = np.concatenate(
            [rng.choice(real_classes, batch // 2), rng.choice(font_classes, batch - batch // 2)]
        )
        tt = torch.zeros(batch, 2, 3, dtype=torch.long, device="cuda")
        tt[:, 0, 0] = torch.tensor(selected, device="cuda")
        tt[:, 0, 1:] = torch.tensor(rng.choice([0] + alphabet, (batch, 2)), device="cuda")
        bb = x["boxes"][ii]
        mu, chol = prior()
        zz = torch.tensor(rng.standard_normal((batch, 2)), device="cuda", dtype=torch.float32)
        zz = zz @ chol.T + mu
        prior_style, prior_centers = sampled_style(tt, zz)
        ll, prior_distance = model(tt, bb, pp, style=prior_style)
        with torch.no_grad():
            ft = teacher(tt, bb, pp)[0].sigmoid()
        terms = []
        for j, c in enumerate(selected):
            if int(c) in by_class:
                ids = (
                    [int(prior_centers[j, 0])]
                    if cfg.get("matched_neighborhood")
                    else by_class[int(c)]
                )
                variants = x["ink"][ids].flatten(1)[:, pi[j]]
                terms.append(nearest_real_loss(ll[j], variants))
            else:
                terms.append(
                    (
                        F.binary_cross_entropy_with_logits(ll[j], ft[j], reduction="none")
                        * (1 + 3 * ft[j])
                    ).mean()
                )
        support = torch.stack(terms).mean()
        loss = fit + cfg["code_l2"] * codes.square().mean() + cfg["support_weight"] * support
        if cfg.get("distance_regularized"):
            boundary = 0.5 * (
                distance_gradient_loss(distance, pp, sdf.abs() < 0.5)
                + distance_gradient_loss(prior_distance, pp, prior_distance.abs() < 0.5)
            )
            loss = loss + cfg["distance_gradient_weight"] * boundary
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite direct real glyph training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), codes], 5)
        optimizer.step()
        if step == start_step + 1 or step % 100 == 0:
            q = {
                "step": step,
                "loss": float(loss.detach()),
                "fit": float(fit.detach()),
                "prior": float(support.detach()),
                "seconds": time.perf_counter() - start,
            }
            if cfg.get("distance_regularized"):
                q["distance_gradient"] = float(boundary.detach())
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
        if step % 1000 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "training_codes": codes.detach(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    assert all(torch.equal(v, saved["model"][k]) for k, v in teacher.state_dict().items())
    model.eval()
    mu, chol = prior()
    if cfg.get("matched_neighborhood"):
        write(
            run / "reports/joint-prior.json",
            {
                "sigma": cfg["neighborhood_sigma"],
                "shared_word_variable": False,
                "sampling_is_not_neural_training": True,
                "centers": {
                    str(c): {
                        "sample_indices": by_class[c].tolist(),
                        "values": torch.cat(
                            [codes[x["groups"][by_class[c]]], x["local_codes"][by_class[c]]], -1
                        )
                        .detach()
                        .tolist(),
                    }
                    for c in real_classes
                },
            },
        )
    write(
        run / "reports/prior.json",
        {"mean": mu.tolist(), "cholesky": chol.tolist(), "train_codes_only": True},
    )
    output, images = [], []

    def emit(name, tokens, boxes, style, metadata, height=96, width=96):
        im = model.image(tokens, boxes, height, width, style=style)[0].cpu().numpy()
        path = run / f"rendered/pages/{name}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(path)
        mask = im > 0.5
        output.append(
            {
                **metadata,
                "image": str(path),
                "sha256": digest(path),
                "latent": style[0].tolist(),
                "shape": [height, width],
                "blank": not bool(mask.any()),
                "edge": bool(
                    mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                ),
            }
        )
        images.append(im)

    for i, row in enumerate(rows):
        emit(
            f"reconstruction-{i:04d}",
            x["tokens"][i : i + 1],
            x["boxes"][i : i + 1],
            target_style([i]),
            {
                "mode": "training-code-reconstruction",
                "index": i,
                "text": row["label"],
                "target_seen": True,
            },
        )
    np.savez_compressed(run / "reports/reconstructions.npz", ink=np.stack(images))
    images.clear()
    with np.load(words / "reports/arrays.npz") as f:
        wa = {k: f[k] for k in ["tokens", "boxes"]}
    wi = json.loads((words / "reports/indices.json").read_text())
    wr = json.loads((words / "reports/samples.json").read_text())
    metrics = json.loads((words / "reports/class-metrics.json").read_text())
    wb = torch.tensor(apply_metrics(wa["tokens"], wa["boxes"], metrics), device="cuda")
    wt = torch.tensor(wa["tokens"], device="cuda")
    for seed in range(2012, 2020):
        local_generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
        z = (
            mu[None]
            + torch.randn(
                1, 2, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)
            )
            @ chol.T
        )
        for c in real_classes:
            tt = torch.tensor([[[c, 0, 0], [0, 0, 0]]], device="cuda")
            style, centers = sampled_style(tt, z, local_generator)
            emit(
                f"prior-{seed}-{c}",
                tt,
                x["boxes"][:1],
                style,
                {
                    "mode": "glyph-prior",
                    "seed": seed,
                    "text": chr(c),
                    "target_seen": False,
                    "style_center_indices": centers[0].tolist() if centers is not None else None,
                    "style_center_seen": centers is not None,
                    "pixels_or_outline_as_generator_input": False,
                },
            )
    np.savez_compressed(run / "reports/glyph-priors.npz", ink=np.stack(images))
    images.clear()
    for seed in [2012, 2013]:
        local_generator = torch.Generator(device="cuda").manual_seed(seed + 200000)
        z = (
            mu[None]
            + torch.randn(
                1, 2, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)
            )
            @ chol.T
        )
        for i in wi["development"] + wi["novel"]:
            style, centers = sampled_style(wt[i : i + 1], z, local_generator)
            emit(
                f"word-{seed}-{i:04d}",
                wt[i : i + 1],
                wb[i : i + 1],
                style,
                {
                    "mode": "word-prior",
                    "seed": seed,
                    "index": i,
                    "text": wr[i]["text"],
                    "split": wr[i]["split"],
                    "target_seen": False,
                    "style_center_indices": centers[0].tolist() if centers is not None else None,
                    "style_center_seen": centers is not None,
                    "word_variable_shared": centers is None,
                    "pixels_or_outline_as_generator_input": False,
                },
                128,
                512,
            )
    np.savez_compressed(run / "reports/word-priors.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", output)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/result.json",
        {
            "new_model_updates": cfg["steps"] - start_step,
            "inherited_model_updates": cfg["inherited_model_updates"],
            "model_lineage_updates": cfg["inherited_model_updates"] + cfg["steps"] - start_step,
            "actual_parameter_training": True,
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "training_code_l1_change": float((codes - initial_codes).abs().sum().detach()),
            "real_target_draws": int(counts.sum()),
            "unseen_targets": int((counts == 0).sum()),
            "parameters": sum(p.numel() for p in model.parameters()),
            "code_parameters": codes.numel(),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{cfg['steps']:05d}.pt"),
            "glyph_reconstructions": len(rows),
            "glyph_priors": len(real_classes) * 8,
            "word_priors": 48,
            "font_teacher_unchanged": True,
            "C_used": False,
            "quality_passed": False,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["worker"])
    p.add_argument("request", type=Path)
    args = p.parse_args()
    worker(args.request)
