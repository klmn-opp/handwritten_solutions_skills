"""Finite CUDA learning of shared-word and local neural style distributions."""

import json
import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write


def launch(backbone: Path, real: Path, *, context: bool, support: bool = False) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create
    from research.corpus.hierarchical_style_data import group_conditions

    for p in [backbone, real]:
        verify_run(p)
    rows = json.loads((real / "reports/glyph-samples.json").read_text())
    with np.load(real / "reports/glyphs.npz") as a:
        arrays, groups = group_conditions(rows, a["local_codes"])
    assert len(groups) == 68 and int(arrays["known"].sum()) == 275
    old = json.loads((backbone / "reports/request.json").read_text())
    cp = backbone / f"logs/checkpoint-{old['steps']:05d}.pt"
    cfg = {
        "backbone": str(backbone),
        "backbone_sha256": digest(backbone / "run.json"),
        "real": str(real),
        "real_sha256": digest(real / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "context": context,
        "enforce_component_support": support,
        "seed": 4622,
        "steps": 2000,
        "learning_rate": 0.001,
        "batch_size": 2,
        "code_weight": 0.05,
        "kl_weight": 0.001,
        "kl_warmup": 500,
        "architecture": "hierarchical-neural-ink-style/1",
        "frozen_backbone_updates": 0,
        "group_validation_upstream_exposed": True,
        "C_used": False,
        "protocol_sha256": digest("docs/V01022_PROTOCOL.md"),
    }
    run = create("hierarchical-neural-style-training", seed=cfg["seed"], config=cfg)
    np.savez_compressed(run.path / "solutions/conditions.npz", **arrays)
    write(run.path / "reports/groups.json", groups)
    cfg["conditions_sha256"] = digest(run.path / "solutions/conditions.npz")
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    name = "hierarchical-context" if context else "hierarchical-control"
    if support:
        name += "-support"
    Path(f"/tmp/context-v01022-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    __name__,
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: str) -> None:
    import importlib.metadata
    import time

    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.hierarchical_ink_style import HierarchicalInkStyle
    from research.corpus.hierarchical_style_data import component_support

    cfg = json.loads(Path(request).read_text())
    run, real = Path(cfg["run"]), Path(cfg["real"])
    for key in ["real", "backbone"]:
        assert digest(Path(cfg[key]) / "run.json") == cfg[key + "_sha256"]
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    assert digest(run / "solutions/conditions.npz") == cfg["conditions_sha256"]
    setup(cfg["seed"])
    with np.load(real / "reports/glyphs.npz") as a:
        target = torch.tensor(a["ink"], device="cuda")
        support = torch.tensor(component_support(a["labels"], a["local_codes"]), device="cuda")
    with np.load(run / "solutions/conditions.npz") as a:
        data = {k: torch.tensor(a[k], device="cuda") for k in a.files}
        train = np.flatnonzero(a["train"])
    # Same random initialization in both arms; context toggle contains no parameters.
    model = HierarchicalInkStyle(context=cfg["context"]).cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    ink = CanonicalInk().cuda().eval().requires_grad_(False)
    ink.load_state_dict(
        torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)["model"]
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    counts = np.zeros(len(data["tokens"]), np.int64)
    curve, validation = [], []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def reconstruction(indices, *, stochastic):
        tokens, codes, known = [data[k][indices] for k in ["tokens", "codes", "known"]]
        wm, wv, lm, lv = model.posterior(tokens, codes, known)
        w = wm + (0.5 * wv).exp() * torch.randn_like(wm) if stochastic else wm
        local = lm + (0.5 * lv).exp() * torch.randn_like(lm) if stochastic else lm
        predicted = model(tokens, w, local)
        if cfg.get("enforce_component_support", False):
            predicted = predicted * support[tokens[..., 0]]
        logits = ink(tokens[..., 0][known], predicted[known])
        actual = target[data["glyph_indices"][indices][known]]
        bce = (
            F.binary_cross_entropy_with_logits(logits, actual, reduction="none") * (1 + 3 * actual)
        ).mean()
        mse = F.mse_loss(predicted[known], codes[known])
        kl = model.kl(wm, wv).mean() + model.kl(lm, lv)[known].mean()
        return bce, mse, kl, logits.sigmoid(), predicted

    for step in range(1, cfg["steps"] + 1):
        ids = rng.choice(train, size=cfg["batch_size"], replace=True)
        np.add.at(counts, ids, 1)
        bce, mse, kl, _, _ = reconstruction(ids, stochastic=True)
        loss = (
            bce + cfg["code_weight"] * mse + cfg["kl_weight"] * min(1, step / cfg["kl_warmup"]) * kl
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite hierarchical loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "image_bce": float(bce.detach()),
                "code_mse": float(mse.detach()),
                "kl": float(kl.detach()),
                "seconds": time.perf_counter() - start,
            }
            curve.append(row)
            write(run / "reports/curve.json", curve)
            print(json.dumps(row), flush=True)
        if step % 500 == 0:
            with torch.no_grad():
                vals = []
                for i in np.flatnonzero(~data["train"].cpu().numpy()):
                    x = reconstruction([i], stochastic=False)
                    vals.append([float(v) for v in x[:3]])
                validation.append(
                    {
                        "step": step,
                        "posterior_mean_validation_bce_mse_kl": np.mean(vals, 0).tolist(),
                        "upstream_exposed_not_independent_holdout": True,
                    }
                )
            write(run / "reports/validation.json", validation)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "config": cfg,
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs, images, codes_out = [], [], []
    with torch.no_grad():
        for i in range(len(data["tokens"])):
            _, _, _, generated, predicted = reconstruction([i], stochastic=False)
            known = data["known"][i]
            codes_out.append(predicted[0].cpu().numpy())
            for j, glyph_index in enumerate(data["glyph_indices"][i][known].tolist()):
                im = generated[j].cpu().numpy()
                dest = run / f"rendered/pages/posterior-{glyph_index:03d}.png"
                Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
                truth = target[glyph_index].cpu().numpy() > 0.5
                mask = im > 0.5
                images.append(im)
                outputs.append(
                    {
                        "glyph_index": glyph_index,
                        "group_index": i,
                        "image": str(dest),
                        "sha256": digest(dest),
                        "style_prior_train": bool(data["train"][i]),
                        "dice": float(2 * (truth & mask).sum() / max(1, truth.sum() + mask.sum())),
                        "posterior_reconstruction_not_prior_generation": True,
                    }
                )
        wm, wv, lm, lv = model.posterior(data["tokens"], data["codes"], data["known"])
        np.savez_compressed(
            run / "reports/posterior.npz",
            word_mean=wm.cpu().numpy(),
            word_logvar=wv.cpu().numpy(),
            local_mean=lm.cpu().numpy(),
            local_logvar=lv.cpu().numpy(),
            styles=np.stack(codes_out),
        )
    np.savez_compressed(run / "reports/outputs.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "new_model_updates": cfg["steps"],
            "frozen_backbone_updates": 0,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{cfg['steps']:05d}.pt"),
            "outputs": len(outputs),
            "prior_train_word_groups": len(train),
            "optimization_validation_groups": len(data["tokens"]) - len(train),
            "quality_passed": False,
            "context": cfg["context"],
            "C_used": False,
            "prior_training_not_end_to_end_connected_line_training": True,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
