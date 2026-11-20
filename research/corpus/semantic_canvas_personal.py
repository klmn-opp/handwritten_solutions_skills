"""Mixed real glyph/whole-word learning, isolated from production handwriting."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(source: Path, real_data: Path, gate: Path, *, modulated: bool = False) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [source, real_data]:
        verify_run(p)
    approval = json.loads(gate.read_text())
    if not approval.get("go_real_supervision") or approval["model_sha256"] != digest(
        source / "run.json"
    ):
        raise ValueError("Recorded self-reviewed content gate required")
    previous = json.loads((source / "reports/request.json").read_text())
    if previous["steps"] != 2000 or not previous["content_only_font_auxiliary"]:
        raise ValueError("Registered frozen 2k content initializer required")
    font_data = Path(previous["data"])
    verify_run(font_data)
    cp = source / "logs/checkpoint-02000.pt"
    cfg = {
        "source": str(source),
        "source_sha256": digest(source / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "real_data": str(real_data),
        "real_data_sha256": digest(real_data / "run.json"),
        "font_data": str(font_data),
        "font_data_sha256": digest(font_data / "run.json"),
        "gate": str(gate),
        "gate_sha256": digest(gate),
        "steps": 4000,
        "seed": 4120,
        "learning_rate": 0.0003,
        "glyph_batch": 16,
        "word_batch": 4,
        "font_batch": 16,
        "font_retention_weight": 0.25,
        "local_noise_sigma": 0.15,
        "inherited_model_updates": 2000,
        "fresh_optimizer": True,
        "font_retention_excludes_all_real_word_and_glyph_classes": True,
        "whole_word_conditions_unperturbed": True,
        "no_image_or_outline_as_generator_input": True,
        "global_writer_shared": True,
        "independent_random_word_code": False,
        "architecture": "semantic-canvas-modulated-ink/1" if modulated else "semantic-canvas-ink/1",
        "modulated": modulated,
        "C_used": False,
        "protocol_sha256": digest("docs/V01020_PROTOCOL.md"),
    }
    run = create("semantic-canvas-personal-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    pointer = "personal-modulated" if modulated else "personal-model"
    Path(f"/tmp/context-v01020-{pointer}").write_text(str(run.path))
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


def worker(path):
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_canvas_model import ModulatedSemanticCanvasInk, SemanticCanvasInk
    from research.corpus.semantic_raw_data import apply_metrics

    cfg = json.loads(Path(path).read_text())
    run, data, font = (Path(cfg[k]) for k in ["run", "real_data", "font_data"])
    for p, h in [
        (data / "run.json", cfg["real_data_sha256"]),
        (font / "run.json", cfg["font_data_sha256"]),
        (cfg["checkpoint"], cfg["checkpoint_sha256"]),
        (cfg["gate"], cfg["gate_sha256"]),
    ]:
        assert digest(p) == h
    setup(cfg["seed"])
    arrays = {}
    for name, archive in [
        ("glyphs", data / "reports/glyphs.npz"),
        ("words", data / "reports/words.npz"),
        ("font", font / "reports/font-glyphs.npz"),
        ("evaluation", font / "reports/evaluation.npz"),
    ]:
        with np.load(archive) as f:
            arrays[name] = {k: torch.tensor(f[k], device="cuda") for k in f.files}
    g, w, ft, ev = (arrays[k] for k in ["glyphs", "words", "font", "evaluation"])
    by_class = {int(c): torch.where(g["labels"] == c)[0] for c in g["labels"].unique().tolist()}
    real_classes = sorted(by_class)
    real_word_classes = set(int(c) for c in w["tokens"][..., 0].unique().tolist()) - {0}
    retained = [
        i
        for i, c in enumerate(ft["tokens"][:, 0, 0].tolist())
        if c not in set(real_classes) | real_word_classes
    ]
    assert retained
    local_std = {c: g["local_codes"][ii].std(0, unbiased=False) for c, ii in by_class.items()}
    saved = torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)
    assert saved["step"] == 2000
    if cfg.get("modulated"):
        model = ModulatedSemanticCanvasInk().cuda()
        model.load_base(saved["model"])
    else:
        model = SemanticCanvasInk().cuda()
        model.load_state_dict(saved["model"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    counts = {
        "glyphs": np.zeros(len(g["ink"]), np.int64),
        "words": np.zeros(len(w["ink"]), np.int64),
        "font": np.zeros(len(ft["ink"]), np.int64),
    }
    write(
        run / "reports/coverage.json",
        {
            "real_isolated_classes": [chr(c) for c in real_classes],
            "real_word_classes": [chr(c) for c in sorted(real_word_classes)],
            "font_only_retained_classes": [chr(int(ft["tokens"][i, 0, 0])) for i in retained],
            "word_only_real_classes": [
                chr(c) for c in sorted(real_word_classes - set(real_classes))
            ],
            "reference_image_at_generation": False,
        },
    )
    write(
        run / "reports/local-prior.json",
        {
            str(c): {
                "centers": g["local_codes"][ii].tolist(),
                "std": local_std[c].tolist(),
                "source_indices": ii.tolist(),
                "sigma": cfg["local_noise_sigma"],
            }
            for c, ii in by_class.items()
        },
    )
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    curve = []
    window = {role: [] for role in counts}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, cfg["steps"] + 1):
        role = "glyphs" if step % 2 else "words" if step % 4 == 2 else "font"
        a = arrays[role]
        if role == "glyphs":
            selected = rng.choice(real_classes, cfg["glyph_batch"])
            ids = np.array([int(by_class[c][rng.integers(len(by_class[c]))]) for c in selected])
            style = torch.zeros(len(ids), 2, 8, device="cuda")
            noise = torch.tensor(rng.standard_normal((len(ids), 8)), device="cuda").float()
            scale = torch.stack([local_std[int(c)] for c in selected])
            style[:, 0] = g["local_codes"][ids] + cfg["local_noise_sigma"] * scale * noise
        elif role == "words":
            ids = rng.choice(len(w["ink"]), cfg["word_batch"])
            style = w["style"][ids]
        else:
            ids = rng.choice(retained, cfg["font_batch"])
            style = None
        np.add.at(counts[role], ids, 1)
        target = a["ink"][ids]
        h, width = target.shape[-2:]
        logits = model(a["tokens"][ids], a["boxes"][ids], h, width, style=style)
        loss = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * target)
        ).mean()
        if role == "font":
            loss = loss * cfg["font_retention_weight"]
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite canvas personal loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        window[role].append(float(loss.detach()))
        if step <= 4 or step % 100 == 0:
            q = {
                "step": step,
                "role": role,
                "loss": float(loss.detach()),
                "seconds": time.perf_counter() - start,
                "role_loss_means": {k: float(np.mean(v)) for k, v in window.items() if v},
            }
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
            if step % 100 == 0:
                window = {role: [] for role in counts}
        if step % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs, images = [], []

    def emit(name, group, index, tokens, boxes, style, height, width, seed=None, centers=None):
        im = model.image(tokens, boxes, height, width, style=style)[0].cpu().numpy()
        dest = run / f"rendered/pages/{name}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
        images.append(im)
        outputs.append(
            {
                "group": group,
                "index": index,
                "seed": seed,
                "image": str(dest),
                "sha256": digest(dest),
                "shape": [height, width],
                "style": style[0].tolist() if style is not None else None,
                "selected_prior_centers": centers,
                "no_reference_image_input": True,
                "source_type": "learned-offline-ink",
                "personally_collected": False,
            }
        )

    for i in range(len(g["ink"])):
        style = torch.zeros(1, 2, 8, device="cuda")
        style[0, 0] = g["local_codes"][i]
        emit(
            f"real-glyph-{i:03d}",
            "real-glyph-reconstruction",
            i,
            g["tokens"][i : i + 1],
            g["boxes"][i : i + 1],
            style,
            96,
            96,
        )
    np.savez_compressed(run / "reports/real-glyph-reconstructions.npz", ink=np.stack(images))
    images.clear()
    for i in range(len(w["ink"])):
        emit(
            f"real-word-{i:03d}",
            "real-word-reconstruction",
            i,
            w["tokens"][i : i + 1],
            w["boxes"][i : i + 1],
            w["style"][i : i + 1],
            128,
            512,
        )
    np.savez_compressed(run / "reports/real-word-reconstructions.npz", ink=np.stack(images))
    images.clear()
    word_source = Path(json.loads((font / "reports/request.json").read_text())["words"])
    metrics = json.loads((word_source / "reports/class-metrics.json").read_text())
    boxes = torch.tensor(
        apply_metrics(ev["tokens"].cpu().numpy(), ev["boxes"].cpu().numpy(), metrics), device="cuda"
    )

    def sample(tokens, generator):
        style = torch.zeros(*tokens.shape[:2], 8, device="cuda")
        selected = [-1] * tokens.shape[1]
        for j, c in enumerate(tokens[0, :, 0].tolist()):
            if c in by_class:
                ii = by_class[c]
                k = int(torch.randint(len(ii), (1,), device="cuda", generator=generator))
                selected[j] = int(ii[k])
                style[0, j] = g["local_codes"][ii[k]] + (
                    cfg["local_noise_sigma"]
                    * local_std[c]
                    * torch.randn(8, device="cuda", generator=generator)
                )
        return style, selected

    for seed in range(2012, 2020):
        generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
        for c in real_classes:
            tokens = torch.tensor([[[c, 0, 0], [0, 0, 0]]], device="cuda")
            style, selected = sample(tokens, generator)
            emit(
                f"prior-{seed}-{c}",
                "glyph-prior",
                c,
                tokens,
                g["boxes"][:1],
                style,
                96,
                96,
                seed,
                selected,
            )
    np.savez_compressed(run / "reports/glyph-priors.npz", ink=np.stack(images))
    images.clear()
    for seed in [2012, 2013]:
        generator = torch.Generator(device="cuda").manual_seed(seed + 200000)
        for i in range(len(ev["tokens"])):
            style, selected = sample(ev["tokens"][i : i + 1], generator)
            emit(
                f"word-{seed}-{i:03d}",
                "word-prior",
                i,
                ev["tokens"][i : i + 1],
                boxes[i : i + 1],
                style,
                128,
                512,
                seed,
                selected,
            )
    np.savez_compressed(run / "reports/word-priors.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(run / "reports/sampling-counts.json", {k: v.tolist() for k, v in counts.items()})
    write(
        run / "reports/result.json",
        {
            "new_model_updates": cfg["steps"],
            "inherited_model_updates": 2000,
            "model_lineage_updates": 6000,
            "actual_parameter_training": True,
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "whole_word_optimizer_steps": cfg["steps"] // 4,
            "glyph_optimizer_steps": cfg["steps"] // 2,
            "font_retention_optimizer_steps": cfg["steps"] // 4,
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "outputs": len(outputs),
            "checkpoint_sha256": digest(run / "logs/checkpoint-04000.pt"),
            "context_naturalness_proven": False,
            "quality_passed": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
