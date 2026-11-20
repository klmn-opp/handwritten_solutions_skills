"""Frozen-context comparison, removal of neighboring glyphs, and exact replay."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(model: Path, baseline: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [model, baseline]:
        verify_run(p)
    old = json.loads((baseline / "reports/request.json").read_text())
    trained = json.loads((model / "reports/request.json").read_text())
    assert old["checkpoint_sha256"] == trained["checkpoint_sha256"]
    cp = model / f"logs/checkpoint-{trained['steps']:05d}.pt"
    cfg = {
        "model": str(model),
        "model_sha256": digest(model / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "baseline": str(baseline),
        "baseline_sha256": digest(baseline / "run.json"),
        "backbone": trained["checkpoint"],
        "backbone_sha256": trained["checkpoint_sha256"],
        "seed": 4522,
        "C_used": False,
        "new_neural_updates": 0,
    }
    run = create("word-context-neighbor-ablation", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-context-evaluation").write_text(str(run.path))
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


def worker(request: str, mode: str) -> None:
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.word_context_model import WordContextInk

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
    context = WordContextInk().cuda().eval()
    context.load_state_dict(
        torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)["model"]
    )
    rows = json.loads((baseline / "reports/outputs.json").read_text())
    outputs, effects = [], []
    prior = json.loads((run / "reports/outputs.json").read_text()) if mode == "replay" else None
    for row in rows:
        ident = f"{row['id']}-{row['seed']}"
        with np.load(baseline / f"reports/{ident}.npz") as a:
            tokens = torch.tensor(a["tokens"][0], device="cuda")
            boxes = torch.tensor(a["boxes"][0], device="cuda")
            style = torch.tensor(a["style"], device="cuda")
            height, width = a["ink"].shape
            layers = compose(local, tokens, boxes, style, height, width, return_layers=True)
            base = layers.amax(0)
            assert np.array_equal(base.cpu().numpy(), a["ink"])
        start = time.perf_counter()
        full = context.image(base[None, None])[0, 0]
        removed = context.isolated_max(layers)
        effects.append(
            {
                "id": ident,
                "mean_absolute_neighbor_effect": float((full - removed).abs().mean()),
                "max_neighbor_effect": float((full - removed).abs().max()),
                "binary_changed_pixels": int(((full > 0.5) != (removed > 0.5)).sum()),
                "mean_absolute_context_change": float((full - base).abs().mean()),
            }
        )
        for name, tensor in [("context", full), ("isolated", removed)]:
            im = tensor.cpu().numpy()
            dest = run / f"rendered/pages/{ident}-{name}.png"
            array = run / f"reports/{ident}-{name}.npz"
            mask = im > 0.5
            if mode == "replay":
                with np.load(array) as a:
                    assert np.array_equal(im, a["ink"])
                assert np.array_equal(np.uint8((1 - im) * 255), np.asarray(Image.open(dest)))
                assert digest(dest) == prior[len(outputs)]["sha256"]
            else:
                Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
                np.savez_compressed(array, ink=im)
            outputs.append(
                {
                    **row,
                    "image": str(dest),
                    "sha256": digest(dest),
                    "method": name,
                    "blank": not bool(mask.any()),
                    "edge": bool(
                        mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any()
                    ),
                    "local_composition_not_contextual_training": False,
                    "spatial_adapter_not_end_to_end_line_generator": True,
                    "context_checkpoint_sha256": cfg["checkpoint_sha256"],
                    "seconds": time.perf_counter() - start,
                }
            )
    # Recompute the original whole-word audit as well as all trained reconstruction outputs.
    train_cfg = json.loads((trained / "reports/request.json").read_text())
    audit = Path(train_cfg["audit"])
    with (
        np.load(audit / "solutions/original-words.npz") as a,
        np.load(audit / "reports/reconstructions.npz") as b,
        np.load(trained / "reports/reconstructions.npz") as c,
    ):
        for i in range(len(a["ink"])):
            raw = compose(
                local,
                torch.tensor(a["tokens"][i], device="cuda"),
                torch.tensor(a["boxes"][i], device="cuda"),
                torch.tensor(a["style"][i], device="cuda"),
                128,
                512,
            )
            assert np.array_equal(raw.cpu().numpy(), b["ink"][i])
            generated = context.image(raw[None, None])[0, 0].cpu().numpy()
            assert np.array_equal(generated, c["ink"][i])
            assert np.array_equal(
                np.uint8((1 - generated) * 255),
                np.asarray(Image.open(trained / f"rendered/pages/context-{i:02d}.png")),
            )
    if mode == "replay":
        write(
            run / "reports/replay.json",
            {
                "word_line_outputs": len(outputs),
                "audit_outputs": 18,
                "training_reconstructions": 18,
                "all_float_png_exact": True,
                "all_local_baselines_exact": True,
                "new_neural_updates": 0,
                "C_used": False,
            },
        )
    else:
        write(run / "reports/outputs.json", outputs)
        write(run / "reports/neighbor-effects.json", effects)
    print(mode, len(outputs), flush=True)


if __name__ == "__main__":
    import sys

    worker(sys.argv[1], sys.argv[2])
