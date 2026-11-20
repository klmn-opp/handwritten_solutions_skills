"""Text-only hierarchical prior generation and deterministic latent ablations."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(model: Path, baseline: Path, *, name: str) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [model, baseline]:
        verify_run(p)
    old = json.loads((model / "reports/request.json").read_text())
    original = json.loads((baseline / "reports/request.json").read_text())
    assert old["checkpoint_sha256"] == original["checkpoint_sha256"]
    cp = model / f"logs/checkpoint-{old['steps']:05d}.pt"
    cfg = {
        "model": str(model),
        "model_sha256": digest(model / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "baseline": str(baseline),
        "baseline_sha256": digest(baseline / "run.json"),
        "backbone": old["checkpoint"],
        "backbone_sha256": old["checkpoint_sha256"],
        "context": old["context"],
        "seed": 4622,
        "C_used": False,
        "new_neural_updates": 0,
        "reference_image_input": False,
        "lookup_training_centers": False,
        "normal_prior_scale": 1.0,
    }
    run = create("hierarchical-style-text-generation", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path(f"/tmp/context-v01022-evaluation-{name}").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        for mode in ["generate", "replay"]:
            with (run.path / f"logs/{mode}.log").open("w") as log:
                subprocess.run(
                    [
                        "workspace/environments/english-htr-v08/bin/python",
                        "-m",
                        __name__,
                        str(run.path / "reports/request.json"),
                        mode,
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                )
        run.finish("held", result=json.loads((run.path / "reports/replay.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def word_ranges(tokens):
    """Whitespace breaks are encoded by previous-character=0, never crossed."""
    starts = [i for i, t in enumerate(tokens.tolist()) if t[0] and (i == 0 or t[1] == 0)]
    valid = int((tokens[:, 0] != 0).sum())
    return list(zip(starts, starts[1:] + [valid], strict=True))


def worker(request: str, mode: str) -> None:
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.hierarchical_ink_style import HierarchicalInkStyle

    cfg = json.loads(Path(request).read_text())
    run, baseline, trained = [Path(cfg[k]) for k in ["run", "baseline", "model"]]
    for key in ["baseline", "model"]:
        assert digest(Path(cfg[key]) / "run.json") == cfg[key + "_sha256"]
    for key in ["checkpoint", "backbone"]:
        assert digest(cfg[key]) == cfg[key + "_sha256"]
    setup(cfg["seed"])
    local = CanonicalInk().cuda().eval()
    local.load_state_dict(
        torch.load(cfg["backbone"], map_location="cuda", weights_only=False)["model"]
    )
    model = HierarchicalInkStyle(context=cfg["context"]).cuda().eval()
    model.load_state_dict(
        torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)["model"]
    )
    trained_cfg = json.loads((trained / "reports/request.json").read_text())
    with np.load(Path(trained_cfg["real"]) / "reports/glyphs.npz") as a:
        real_classes = set(a["labels"].tolist())
    cases = json.loads((baseline / "reports/cases.json").read_text())
    outputs, effects = [], []
    prior = json.loads((run / "reports/outputs.json").read_text()) if mode == "replay" else None
    methods = ["prior", "word-zero", "context-off"] if cfg["context"] else ["prior"]
    for seed in [2012, 2013]:
        generator = torch.Generator(device="cuda").manual_seed(seed + 300000)
        for case in cases:
            ident = f"{case['id']}-{seed}"
            with np.load(baseline / f"solutions/{case['id']}.npz") as a:
                tokens = torch.tensor(a["tokens"][0], device="cuda")
                boxes = torch.tensor(a["boxes"][0], device="cuda")
            styles = {m: torch.zeros(len(tokens), 8, device="cuda") for m in methods}
            word_latents = torch.zeros(len(tokens), 8, device="cuda")
            local_latents = torch.zeros(len(tokens), 4, device="cuda")
            with torch.no_grad():
                for start, end in word_ranges(tokens):
                    w = torch.randn(1, 8, device="cuda", generator=generator)
                    z = torch.randn(1, end - start, 4, device="cuda", generator=generator)
                    word_latents[start:end] = w
                    local_latents[start:end] = z[0]
                    for method in methods:
                        model.context = cfg["context"] and method != "context-off"
                        styles[method][start:end] = model(
                            tokens[None, start:end], w * 0 if method == "word-zero" else w, z
                        )[0]
                for j, char in enumerate(tokens[:, 0].tolist()):
                    if char not in real_classes:
                        for style in styles.values():
                            style[j] = 0
            model.context = cfg["context"]
            images = {}
            for method in methods:
                w, h = case["size"]
                start_time = time.perf_counter()
                im = compose(local, tokens, boxes, styles[method], h, w).cpu().numpy()
                seconds = time.perf_counter() - start_time
                images[method] = im
                dest = run / f"rendered/pages/{ident}-{method}.png"
                array = run / f"reports/{ident}-{method}.npz"
                style = styles[method].cpu().numpy()
                if mode == "replay":
                    with np.load(array) as a:
                        for key, actual in [
                            ("ink", im),
                            ("style", style),
                            ("word_latent", word_latents.cpu().numpy()),
                            ("local_latent", local_latents.cpu().numpy()),
                        ]:
                            assert np.array_equal(actual, a[key])
                    assert np.array_equal(np.uint8((1 - im) * 255), np.asarray(Image.open(dest)))
                    assert digest(dest) == prior[len(outputs)]["sha256"]
                else:
                    Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
                    np.savez_compressed(
                        array,
                        ink=im,
                        style=style,
                        word_latent=word_latents.cpu().numpy(),
                        local_latent=local_latents.cpu().numpy(),
                    )
                mask = im > 0.5
                outputs.append(
                    {
                        "id": case["id"],
                        "text": case["text"],
                        "kind": case["kind"],
                        "seed": seed,
                        "method": method,
                        "image": str(dest),
                        "sha256": digest(dest),
                        "seconds": seconds,
                        "blank": not bool(mask.any()),
                        "edge": bool(
                            mask[:2].any()
                            or mask[-2:].any()
                            or mask[:, :2].any()
                            or mask[:, -2:].any()
                        ),
                        "reference_image_input": False,
                        "training_center_lookup": False,
                        "neural_style_checkpoint": cfg["checkpoint_sha256"],
                        "neural_ink_checkpoint": cfg["backbone_sha256"],
                        "glyphs": [
                            {
                                "position": j,
                                "semantic_character": chr(c),
                            "actual_glyph_id": (
                                f"{cfg['checkpoint_sha256'][:16]}:{ident}:{method}:{j}"
                            ),
                                "source_type": "learned-offline-ink-with-neural-style-prior",
                                "personally_collected": False,
                                "real_scan_training_coverage": c in real_classes,
                                "font_teacher_only_training": c not in real_classes,
                                "external_font_fallback_at_render": False,
                            }
                            for j, c in enumerate(tokens[:, 0].tolist())
                            if c
                        ],
                        "not_end_to_end_connected_line_generation": True,
                    }
                )
            if cfg["context"]:
                effects.append(
                    {
                        "id": ident,
                        **{
                            m + "_mean_absolute_difference": float(
                                abs(images[m] - images["prior"]).mean()
                            )
                            for m in methods[1:]
                        },
                    }
                )
    # Reproduce every posterior glyph and every stored condition independently.
    with (
        np.load(trained / "solutions/conditions.npz") as a,
        np.load(trained / "reports/posterior.npz") as p,
        np.load(trained / "reports/outputs.npz") as o,
    ):
        records = json.loads((trained / "reports/outputs.json").read_text())
        count = 0
        with torch.no_grad():
            for i in range(len(a["tokens"])):
                t, c, k = [
                    torch.tensor(a[key][i : i + 1], device="cuda")
                    for key in ["tokens", "codes", "known"]
                ]
                wm, wv, lm, lv = model.posterior(t, c, k)
                style = model(t, wm, lm)
                assert np.array_equal(style[0].cpu().numpy(), p["styles"][i])
                generated = local(t[..., 0][k], style[k]).sigmoid().cpu().numpy()
                for im in generated:
                    assert np.array_equal(im, o["ink"][count])
                    assert np.array_equal(
                        np.uint8((1 - im) * 255), np.asarray(Image.open(records[count]["image"]))
                    )
                    assert digest(records[count]["image"]) == records[count]["sha256"]
                    count += 1
    if mode == "replay":
        write(
            run / "reports/replay.json",
            {
                "word_line_outputs": len(outputs),
                "posterior_glyphs": count,
                "all_float_png_resampled_latents_styles_exact": True,
                "new_neural_updates": 0,
                "C_used": False,
            },
        )
    else:
        write(run / "reports/outputs.json", outputs)
        write(run / "reports/latent-effects.json", effects)
    print(mode, len(outputs), "prior outputs;", count, "posterior exact", flush=True)


if __name__ == "__main__":
    import sys

    worker(sys.argv[1], sys.argv[2])
