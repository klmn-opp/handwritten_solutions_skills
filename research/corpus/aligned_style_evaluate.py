"""Independent target-free prior replay and limited geometric observations."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def evaluate(models: dict[str, str], data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    for v in models.values():
        verify_run(Path(v))
    run = create(
        "aligned-style-frozen-evaluation",
        config={"models": models, "data": str(data), "C_used": False, "target_free_prior": True},
    )
    request = {
        "run": str(run.path),
        "models": models,
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
    }
    write(run.path / "reports/request.json", request)
    try:
        with (run.path / "logs/replay.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.aligned_style_evaluate",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        run.finish("held", result=json.loads((run.path / "reports/replay.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request):
    import time

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.aligned_style_model import AlignedStyle

    r = json.loads(Path(request).read_text())
    run = Path(r["run"])
    data = Path(r["data"])
    setup(1707)
    if digest(data / "run.json") != r["data_sha256"]:
        raise ValueError("Source changed")
    arrays = np.load(data / "reports/arrays.npz")
    rows = json.loads((data / "reports/samples.json").read_text())
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    report = {}
    total = 0

    def components(a):
        _, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(a > 0.5), 8)
        return int((stats[1:, cv2.CC_STAT_AREA] >= 4).sum())

    for name, value in r["models"].items():
        source = Path(value)
        cr = json.loads((source / "reports/request.json").read_text())
        cp = source / f"logs/checkpoint-{cr['steps']:04d}.pt"
        checkpoint = torch.load(cp, map_location="cuda", weights_only=False)
        initial = torch.load(
            source / "logs/initial-state.pt", map_location="cuda", weights_only=False
        )
        delta = sum((v - initial[k]).abs().sum().item() for k, v in checkpoint["model"].items())
        opt_steps = sorted({int(v["step"]) for v in checkpoint["optimizer"]["state"].values()})
        assert delta > 0 and opt_steps == [cr["steps"]]
        model = (
            AlignedStyle(stochastic=cr["mode"] == "stochastic", monotone=cr.get("monotone", False))
            .cuda()
            .eval()
        )
        model.load_state_dict(checkpoint["model"])
        ids = json.loads((source / "reports/indices.json").read_text())["evaluated"]
        saved = np.load(source / "reports/generated.npz")
        generated = {}
        checks = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for seed in range(1708, 1712) if model.stochastic else [1708]:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                chunks = []
                for k in range(0, len(ids), 8):
                    ii = ids[k : k + 8]
                    epsilon = torch.randn((len(ii), 4), device="cuda", generator=gen)
                    out, _ = model.generate(refs[ii], tokens[ii], epsilon)
                    chunks.append(out[:, 0].cpu().numpy())
                out = np.concatenate(chunks)
                assert np.array_equal(out, saved[str(seed)])
                generated[str(seed)] = out
                for k, i in enumerate(ids):
                    p = run / f"rendered/pages/{name}-{seed}-{i:04d}.png"
                    Image.fromarray(np.uint8((1 - out[k]) * 255)).save(p)
                    assert digest(p) == digest(source / f"rendered/pages/prior-{seed}-{i:04d}.png")
                    total += 1
                    a, b = out[k] > 0.5, arrays["reference"][i] > 0.5
                    target = arrays["target"][i] > 0.5
                    # Diagnostic only: actual scans have different pen and style topology.
                    checks.append(
                        {
                            "index": i,
                            "seed": seed,
                            "split": rows[i]["dataset_split"],
                            "core_mass_ratio_to_font": float(a.sum() / max(1, b.sum())),
                            "core_components_generated": components(out[k]),
                            "core_components_font": components(arrays["reference"][i]),
                            "shape_dice_to_real": float(
                                2 * (a & target).sum() / max(1, a.sum() + target.sum())
                            ),
                            "font_shape_dice_to_real": float(
                                2 * (b & target).sum() / max(1, b.sum() + target.sum())
                            ),
                            "exact_font_binary_match": bool(np.array_equal(a, b)),
                        }
                    )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        masks = arrays["target"][arrays["split"] == "train", ::2, ::2] > 0.5
        train_ids = np.where(arrays["split"] == "train")[0]
        nearest = []
        for k, i in enumerate(ids):
            if rows[i]["dataset_split"] != "validation":
                continue
            a = generated["1708"][k, ::2, ::2] > 0.5
            scores = 2 * (masks & a).sum((1, 2)) / np.maximum(1, masks.sum((1, 2)) + a.sum())
            j = int(scores.argmax())
            nearest.append(
                {
                    "index": i,
                    "nearest_train_index": int(train_ids[j]),
                    "nearest_train_dice": float(scores[j]),
                    "same_label": rows[i]["text"] == rows[int(train_ids[j])]["text"],
                    "exact_train_binary_match": bool((masks == a).all((1, 2)).any()),
                }
            )
        diversity = []
        seeds = list(generated)
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                diversity.append(float(np.abs(generated[seeds[a]] - generated[seeds[b]]).mean()))
        report[name] = {
            "checkpoint_sha256": digest(cp),
            "parameter_absolute_change": delta,
            "optimizer_steps": opt_steps,
            "replay_seconds_with_io": elapsed,
            "exact_images": len(ids) * len(seeds),
            "pairwise_prior_pixel_mae": diversity,
            "pixel_difference_is_not_naturalness": True,
            "nearest_real_train": nearest,
            "geometry": checks,
        }
    write(
        run / "reports/replay.json",
        {
            "exact_png_replays": total,
            "models": report,
            "C_used": False,
            "prior_uses_target": False,
            "glyph_source": "warped-personal-font-not-new-ink",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    worker(p.parse_args().request)
