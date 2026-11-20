"""Bounded CUDA style learning and frozen decoder comparison for known English content."""

import subprocess
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write
from research.corpus.context_metric_evaluate import read


def launch(grouped: Path, baseline: Path, metric: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [grouped, baseline, metric]:
        verify_run(p)
    old = read(baseline / "reports/request.json")
    group_cfg = read(grouped / "reports/request.json")
    assert old["checkpoint_sha256"] == group_cfg["checkpoint_sha256"]
    cfg = {
        "grouped": str(grouped),
        "grouped_sha256": digest(grouped / "run.json"),
        "baseline": str(baseline),
        "baseline_sha256": digest(baseline / "run.json"),
        "metric": str(metric),
        "metric_sha256": digest(metric / "run.json"),
        "backbone": old["checkpoint"],
        "backbone_sha256": old["checkpoint_sha256"],
        "canonical_data": old["source"],
        "canonical_data_sha256": old["source_sha256"],
        "steps": 2500,
        "seed": 4822,
        "lr": 0.001,
        "C_used": False,
        "protocol_sha256": digest("docs/V01022_DENOISING_PROTOCOL.md"),
    }
    run = create("semantic-denoising-style-prior", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-denoising").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        for mode in ["train", "generate", "replay"]:
            with (run.path / f"logs/{mode}.log").open("w") as log:
                subprocess.run(
                    [
                        "workspace/environments/english-htr-v08/bin/python",
                        "-m",
                        __name__,
                        str(run.path / "reports/request.json"),
                        mode,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=360,
                )
        run.finish(
            "held",
            result={
                **read(run.path / "reports/training.json"),
                **read(run.path / "reports/replay.json"),
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def tensors(cfg):
    import torch

    from research.corpus.hierarchical_style_data import component_support

    with np.load(Path(cfg["grouped"]) / "solutions/conditions.npz") as a:
        positions = a["known"]
        tokens, codes = a["tokens"][positions], a["codes"][positions]
        train = np.broadcast_to(a["train"][:, None], positions.shape)[positions]
    support = component_support(tokens[:, 0], codes)
    return (
        torch.tensor(tokens, device="cuda"),
        torch.tensor(codes, device="cuda"),
        train,
        torch.tensor(support, device="cuda"),
    )


def train(cfg, run: Path) -> None:
    import importlib.metadata
    import time

    import torch

    from research.corpus.denoised_style_prior import DenoisedStyle

    tokens, coords, is_train, support = tensors(cfg)
    char = tokens[:, 0].cpu().numpy()
    eligible = sorted(set(char[is_train]) & set(np.flatnonzero(support.any(1).cpu().numpy())))
    groups = [np.flatnonzero(is_train & (char == c)) for c in eligible]
    validation = torch.tensor(np.flatnonzero(~is_train), device="cuda")
    rng = np.random.default_rng(cfg["seed"])
    model = DenoisedStyle().cuda()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    curve, counts = [], np.zeros(len(coords), np.int64)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    def loss(t, target, sigma, noise):
        enabled = support[t[:, 0]]
        noisy = target + sigma * noise * enabled
        guess = model(t, noisy, sigma, enabled)
        error = (guess - target).square() * (1 + sigma.square()) / sigma.square()
        return (error.sum(1) / enabled.sum(1).clamp_min(1)).mean()

    for step in range(1, cfg["steps"] + 1):
        ids = np.array([rng.choice(group) for group in groups])
        np.add.at(counts, ids, 1)
        t = tokens[ids].clone()
        t[torch.rand(len(t), device="cuda") < 0.5, 1:] = 0
        sigma = 0.03 * (4 / 0.03) ** torch.rand(len(t), 1, device="cuda")
        error = loss(t, coords[ids], sigma, torch.randn_like(coords[ids]))
        assert torch.isfinite(error)
        optimizer.zero_grad(set_to_none=True)
        error.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {"step": step, "training_noise_weighted_mse": float(error.detach())}
            if step % 500 == 0:
                gen = torch.Generator(device="cuda").manual_seed(9822)
                with torch.no_grad():
                    val = coords[validation]
                    noise = torch.randn(val.shape, generator=gen, device="cuda")
                    row["upstream_exposed_validation_weighted_mse"] = [
                        float(
                            loss(
                                tokens[validation],
                                val,
                                torch.full((len(val), 1), s, device="cuda"),
                                noise,
                            )
                        )
                        for s in [0.12, 0.6, 2.0]
                    ]
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
            curve.append(row)
            write(run / "reports/curve.json", curve)
            print(row, flush=True)
    write(run / "reports/sampling-counts.json", counts.tolist())
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/training.json",
        {
            "new_style_parameter_updates": cfg["steps"],
            "new_ink_decoder_updates": 0,
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "trained_variable_classes": [chr(c) for c in eligible],
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / f"logs/checkpoint-{cfg['steps']:05d}.pt"),
            "C_used": False,
        },
    )


def render(cfg, run: Path, mode: str) -> None:
    import torch
    from PIL import Image

    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.denoised_style_prior import DenoisedStyle

    baseline, metric = Path(cfg["baseline"]), Path(cfg["metric"])
    style = DenoisedStyle().cuda().eval()
    cp = run / f"logs/checkpoint-{cfg['steps']:05d}.pt"
    checkpoint = digest(cp)
    assert checkpoint == read(run / "reports/training.json")["checkpoint_sha256"]
    style.load_state_dict(torch.load(cp, map_location="cuda", weights_only=False)["model"])
    ink = CanonicalInk().cuda().eval()
    ink.load_state_dict(
        torch.load(cfg["backbone"], map_location="cuda", weights_only=False)["model"]
    )
    _, _, _, support = tensors(cfg)
    with np.load(Path(cfg["canonical_data"]) / "reports/targets.npz") as a:
        real = a["real"].copy()
        class_labels = a["labels"][real]
        training_ink = a["ink"][real] > 0.5
        true_codes = a["codes"][real]
    rows = read(metric / "reports/outputs.json")
    real_classes = set(class_labels.tolist())
    previous = read(run / "reports/outputs.json") if mode == "replay" else None
    outputs = []

    def emit(ident, record, image, noise, condition):
        array = image.cpu().numpy()
        png = np.uint8((1 - array) * 255)
        dest = run / f"rendered/pages/{ident}.png"
        receipt = run / f"reports/{ident}.npz"
        arrays = {"ink": array, "noise": noise.cpu().numpy(), "style": condition.cpu().numpy()}
        if mode == "generate":
            Image.fromarray(png).save(dest)
            np.savez_compressed(receipt, **arrays)
        else:
            with np.load(receipt) as saved:
                assert all(np.array_equal(v, saved[k]) for k, v in arrays.items())
            assert np.array_equal(png, np.asarray(Image.open(dest)))
            assert digest(dest) == previous[len(outputs)]["sha256"]
        outputs.append(
            {
                **record,
                "id": ident,
                "image": str(dest),
                "sha256": digest(dest),
                "blank": not bool((array > 0.5).any()),
                "edge": bool(
                    (array[:2] > 0.5).any()
                    or (array[-2:] > 0.5).any()
                    or (array[:, :2] > 0.5).any()
                    or (array[:, -2:] > 0.5).any()
                ),
                "style_checkpoint_sha256": checkpoint,
                "ink_checkpoint_sha256": cfg["backbone_sha256"],
                "image_input": False,
                "training_center_lookup": False,
            }
        )

    for row in rows:
        if row["method"] != "anchored":
            continue
        ident = f"{row['id']}-{row['seed']}"
        gen = torch.Generator(device="cuda").manual_seed(row["seed"] + 400000 + len(outputs))
        with (
            np.load(metric / f"reports/{ident}-anchored.npz") as a,
            np.load(baseline / f"reports/{ident}.npz") as b,
        ):
            tokens = torch.tensor(b["tokens"][0], device="cuda")
            boxes = torch.tensor(a["boxes"], device="cuda")
            height, width = a["ink"].shape
        noise = torch.randn(len(tokens), 8, device="cuda", generator=gen)
        for variant in ["neighbors", "no-neighbors"]:
            semantic = tokens.clone()
            if variant == "no-neighbors":
                semantic[:, 1:] = 0
            condition = style.generate(semantic, noise, support[tokens[:, 0]])
            image = compose(ink, tokens, boxes, condition, height, width)
            emit(
                f"{ident}-{variant}",
                {
                    "kind": row["kind"],
                    "text": row["text"],
                    "case": row["id"],
                    "seed": row["seed"],
                    "method": variant,
                    "baseline_image": row["image"],
                    "glyph_provenance": [
                        {
                            "position": pos,
                            "semantic": chr(c),
                            "actual_glyph_id": f"{checkpoint[:12]}:{ident}:{variant}:{pos}",
                            "source_type": "learned-offline-ink-with-denoising-style-prior",
                            "real_scan_training_coverage": c in real_classes,
                            "font_teacher_only_training": c not in real_classes,
                            "personal_collected": False,
                        }
                        for pos, c in enumerate(tokens[:, 0].tolist())
                        if c
                    ],
                },
                image,
                noise,
                condition,
            )
    truth = {int(c): training_ink[class_labels == c] for c in np.unique(class_labels)}
    chars = torch.tensor(sorted(truth), device="cuda")
    tokens = torch.stack([chars, chars * 0, chars * 0], -1)
    for seed in range(2012, 2020):
        gen = torch.Generator(device="cuda").manual_seed(seed + 4822)
        noise = torch.randn(len(chars), 8, device="cuda", generator=gen)
        styles = style.generate(tokens, noise, support[chars])
        images = ink.image(chars, styles)
        for i, c in enumerate(chars.tolist()):
            sample = images[i].cpu().numpy() > 0.5
            near = np.logical_xor(sample, truth[c]).sum(axis=(1, 2))
            d = np.linalg.norm(styles[i].cpu().numpy() - true_codes[class_labels == c], axis=1)
            emit(
                f"glyph-{c}-{seed}",
                {
                    "kind": "glyph",
                    "text": chr(c),
                    "seed": seed,
                    "nearest_real_binary_difference_pixels": int(near.min()),
                    "nearest_real_style_distance": float(d.min()),
                    "real_class_samples": len(near),
                },
                images[i],
                noise[i],
                styles[i],
            )
    if mode == "generate":
        write(run / "reports/outputs.json", outputs)
    else:
        assert outputs == previous
        write(
            run / "reports/replay.json",
            {
                "outputs": len(outputs),
                "all_noise_style_float_ink_png_exact": True,
                "blank": sum(r["blank"] for r in outputs),
            },
        )
    print(mode, len(outputs), flush=True)


if __name__ == "__main__":
    import sys

    from research.corpus.aligned_style_experiment import setup

    config = read(Path(sys.argv[1]))
    for name in ["grouped", "baseline", "metric", "canonical_data"]:
        assert digest(Path(config[name]) / "run.json") == config[name + "_sha256"]
    assert digest(config["backbone"]) == config["backbone_sha256"]
    setup(config["seed"])
    path = Path(config["run"])
    if sys.argv[2] == "train":
        train(config, path)
    else:
        render(config, path, sys.argv[2])
