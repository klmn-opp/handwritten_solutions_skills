"""Replay frozen whole-word generators without giving prior generation target ink."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def evaluate(models: dict[str, str], data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    for path in models.values():
        verify_run(Path(path))
    config = {
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "models": {
            k: {"run": v, "run_sha256": digest(Path(v) / "run.json")} for k, v in models.items()
        },
        "C_used": False,
        "posterior_diagnostic_only": True,
        "prior_uses_target": False,
    }
    run = create("word-style-frozen-evaluation", config=config)
    write(run.path / "reports/request.json", {**config, "run": str(run.path)})
    try:
        with (run.path / "logs/replay.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.word_style_evaluate",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1800,
            )
        run.finish("held", result=json.loads((run.path / "reports/replay.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request: Path) -> None:
    import os
    import time

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.word_style_model import WordStyle

    r = json.loads(request.read_text())
    run, data = Path(r["run"]), Path(r["data"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    arrays = np.load(data / "reports/arrays.npz")
    refs = torch.tensor(arrays["reference"][:, None], device="cuda")
    rows = json.loads((data / "reports/samples.json").read_text())
    all_report, matches = {}, 0
    for name, cfg in r["models"].items():
        source = Path(cfg["run"])
        cr = json.loads((source / "reports/request.json").read_text())
        ck = source / f"logs/checkpoint-{cr['steps']:04d}.pt"
        checkpoint = torch.load(ck, map_location="cuda", weights_only=False)
        model = WordStyle(cr["latent"]).cuda().eval()
        model.load_state_dict(checkpoint["model"])
        ids = json.loads((source / "reports/indices.json").read_text())["evaluated"]
        saved = np.load(source / f"reports/generated-{cr['steps']}.npz")
        modes = ["prior-0"] if cr["mode"] == "deterministic" else [f"prior-{j}" for j in range(4)]
        generated = {}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for mode in modes:
                generator = torch.Generator(device="cuda").manual_seed(2606 + int(mode[-1]))
                out = []
                for k in range(0, len(ids), 8):
                    ii = ids[k : k + 8]
                    z = (
                        torch.zeros(len(ii), cr["latent"], device="cuda")
                        if cr["mode"] == "deterministic"
                        else torch.randn(len(ii), cr["latent"], device="cuda", generator=generator)
                    )
                    ink, _ = model(refs[ii], z)
                    out.append(ink[:, 0].cpu().numpy())
                out = np.concatenate(out)
                if not np.array_equal(saved[mode], out):
                    raise ValueError("Frozen prior replay differs")
                generated[mode] = out
                for k, i in enumerate(ids):
                    dest = run / f"rendered/pages/{name}-{mode}-{i:04d}.png"
                    Image.fromarray(np.uint8((1 - out[k]) * 255)).save(dest)
                    if digest(dest) != digest(
                        source / f"rendered/pages/step-{cr['steps']}-{mode}-{i:04d}.png"
                    ):
                        raise ValueError("Frozen PNG differs")
                    matches += 1
        torch.cuda.synchronize()
        timing = time.perf_counter() - start
        # Memorization distance is whole-word pixel geometry, not personal style truth.
        train_ids = np.where(arrays["split"] == "train")[0]
        target_masks = arrays["target"][train_ids, ::2, ::2] > 0.5
        scores = []
        for k, i in enumerate(ids):
            if rows[i]["dataset_split"] != "validation":
                continue
            a = generated["prior-0"][k, ::2, ::2] > 0.5
            dice = (
                2
                * (target_masks & a).sum((1, 2))
                / np.maximum(1, target_masks.sum((1, 2)) + a.sum())
            )
            best = int(dice.argmax())
            scores.append(
                {
                    "index": i,
                    "nearest_train": int(train_ids[best]),
                    "same_label": rows[i]["text"] == rows[int(train_ids[best])]["text"],
                    "nearest_train_dice": float(dice[best]),
                    "exact_binary_train_match": bool((target_masks == a).all((1, 2)).any()),
                    "exact_font_binary_match": bool(
                        np.array_equal(generated["prior-0"][k] > 0.5, arrays["reference"][i] > 0.5)
                    ),
                }
            )
        diversity = []
        if len(modes) > 1:
            for a in range(4):
                for b in range(a + 1, 4):
                    diversity.append(
                        float(np.mean(np.abs(generated[f"prior-{a}"] - generated[f"prior-{b}"])))
                    )
        all_report[name] = {
            "checkpoint_sha256": digest(ck),
            "rows": len(ids),
            "prior_modes": modes,
            "seconds_with_png_and_replay_checks": timing,
            "pairwise_prior_pixel_mae": diversity,
            "memorization": scores,
            "content_accuracy_required_before_diversity_interpretation": True,
        }
        np.savez_compressed(run / f"reports/{name}-generated.npz", **generated)
    write(
        run / "reports/replay.json",
        {
            "exact_png_replays": matches,
            "models": all_report,
            "C_used": False,
            "prior_has_no_target_input": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["worker"])
    p.add_argument("request", type=Path)
    worker(p.parse_args().request)
