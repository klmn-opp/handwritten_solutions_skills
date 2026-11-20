"""Expand a passed local capacity study to real classes and explicit font-only targets."""

import json
import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write


def combine_targets(real, font):
    labels = real["labels"].astype(np.int64)
    fonts = font["tokens"][:, 0, 0].astype(np.int64)
    if len(np.unique(fonts)) != len(fonts):
        raise ValueError("Unique font identities required")
    retained = np.flatnonzero(~np.isin(fonts, labels))
    return {
        "ink": np.concatenate([real["ink"], font["ink"][retained]]),
        "labels": np.concatenate([labels, fonts[retained]]),
        "codes": np.concatenate([real["local_codes"], np.zeros((len(retained), 8), np.float32)]),
        "real": np.arange(len(labels) + len(retained)) < len(labels),
        "source_indices": np.concatenate([np.arange(len(labels)), retained]),
    }


def launch(source: Path, real: Path, font: Path, gate: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [source, real, font]:
        verify_run(p)
    approval = json.loads(gate.read_text())
    if not approval.get("go_full_local_capacity") or approval["source_sha256"] != digest(
        source / "run.json"
    ):
        raise ValueError("Frozen local-capacity self-review required")
    cp = source / "logs/checkpoint-01000.pt"
    cfg = {
        "source": str(source),
        "source_sha256": digest(source / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "real": str(real),
        "real_sha256": digest(real / "run.json"),
        "font": str(font),
        "font_sha256": digest(font / "run.json"),
        "gate": str(gate),
        "gate_sha256": digest(gate),
        "steps": 4000,
        "seed": 4221,
        "learning_rate": 0.001,
        "sigma": 0.15,
        "inherited_model_updates": 1000,
        "font_weight": 0.25,
        "batch": "one target per supported class, uniform within class",
        "architecture": "canonical-conditioned-ink/1",
        "C_used": False,
        "protocol_sha256": digest("docs/V01021_PROTOCOL.md"),
        "contextual_model": False,
        "no_image_input": True,
        "fresh_optimizer": True,
    }
    run = create("canonical-ink-full-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01021-full-model").write_text(str(run.path))
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


def worker(request):
    import importlib.metadata
    import time

    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_model import CanonicalInk

    cfg = json.loads(Path(request).read_text())
    run, real, font = [Path(cfg[k]) for k in ["run", "real", "font"]]
    for key in ["source", "real", "font"]:
        assert digest(Path(cfg[key]) / "run.json") == cfg[key + "_sha256"]
    for key in ["checkpoint", "gate"]:
        assert digest(cfg[key]) == cfg[key + "_sha256"]
    setup(cfg["seed"])
    with np.load(real / "reports/glyphs.npz") as g, np.load(font / "reports/font-glyphs.npz") as f:
        a = combine_targets(g, f)
    assert len(a["ink"]) == 319 and len(np.unique(a["labels"])) == 72
    np.savez_compressed(run / "reports/targets.npz", **a)
    original_rows = json.loads((real / "reports/glyph-samples.json").read_text())
    rows = []
    for i, (c, observed, source_index) in enumerate(
        zip(a["labels"], a["real"], a["source_indices"], strict=True)
    ):
        rows.append(
            {
                "index": i,
                "character": chr(c),
                "source_type": "real-scan-derived" if observed else "font-teacher-auxiliary",
                "personally_collected": bool(observed),
                "source_index": int(source_index),
                "original_source": original_rows[source_index] if observed else None,
            }
        )
    write(run / "reports/samples.json", rows)
    target = torch.tensor(a["ink"], device="cuda")
    labels = torch.tensor(a["labels"], device="cuda")
    codes = torch.tensor(a["codes"], device="cuda")
    classes = sorted(np.unique(a["labels"]).tolist())
    by = {c: np.flatnonzero(a["labels"] == c) for c in classes}
    std = {c: codes[ii].std(0, unbiased=False) for c, ii in by.items()}
    scales = torch.stack([std[c] for c in classes])
    weights = torch.tensor(
        [1 if a["real"][by[c][0]] else cfg["font_weight"] for c in classes], device="cuda"
    )
    model = CanonicalInk().cuda()
    saved = torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)
    assert saved["step"] == cfg.get("source_checkpoint_step", 1000)
    model.load_state_dict(saved["model"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    pixel_weights = None
    if cfg.get("pixel_weights"):
        assert digest(cfg["pixel_weights"]) == cfg["pixel_weights_sha256"]
        with np.load(cfg["pixel_weights"]) as archive:
            pixel_weights = torch.tensor(archive["weights"], device="cuda")
        assert pixel_weights.shape == target.shape
    rng = np.random.default_rng(cfg["seed"])
    counts = np.zeros(len(target), np.int64)
    curve = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        ids = np.array([int(rng.choice(by[c])) for c in classes])
        np.add.at(counts, ids, 1)
        noise = torch.tensor(rng.standard_normal((len(classes), 8)), device="cuda").float()
        local = codes[ids] + cfg["sigma"] * scales * noise
        logits = model(labels[ids], local)
        weight = (1 + 3 * target[ids]) if pixel_weights is None else pixel_weights[ids]
        losses = (
            F.binary_cross_entropy_with_logits(logits, target[ids], reduction="none") * weight
        ).mean((1, 2))
        loss = (losses * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            q = {
                "step": step,
                "loss": float(loss.detach()),
                "real_loss": float(losses[weights == 1].mean().detach()),
                "font_loss": float(losses[weights != 1].mean().detach()),
                "seconds": time.perf_counter() - start,
            }
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
        if step % 500 == 0:
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
    images, outputs = [], []

    def emit(c, local, *, reconstruction=None, seed=None, center=None):
        im = model.image(torch.tensor([c], device="cuda"), local[None])[0].cpu().numpy()
        dest = run / f"rendered/pages/ink-{len(images):04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
        images.append(im)
        outputs.append(
            {
                "image": str(dest),
                "sha256": digest(dest),
                "label": chr(c),
                "style": local.tolist(),
                "reconstruction": reconstruction,
                "seed": seed,
                "center": center,
                "source_type": "learned-offline-ink",
                "personally_collected": False,
                "real_training_coverage": bool(a["real"][by[c][0]]),
                "no_reference_image_input": True,
            }
        )

    for i, c in enumerate(a["labels"]):
        emit(int(c), codes[i], reconstruction=i)
    for seed in range(2012, 2020):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        for c in classes:
            ii = by[c]
            if not a["real"][ii[0]]:
                continue
            k = int(torch.randint(len(ii), (1,), device="cuda", generator=gen))
            local = codes[ii[k]] + cfg["sigma"] * std[c] * torch.randn(
                8, device="cuda", generator=gen
            )
            emit(c, local, seed=seed, center=int(ii[k]))
    np.savez_compressed(run / "reports/outputs.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "new_model_updates": cfg["steps"],
            "model_lineage_updates": cfg["inherited_model_updates"] + cfg["steps"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{cfg['steps']:05d}.pt"),
            "outputs": len(outputs),
            "real_classes": 28,
            "font_only_local_classes": 44,
            "quality_passed": False,
            "contextual_word_learning": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
