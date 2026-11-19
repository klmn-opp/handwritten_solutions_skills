"""One-pass original-outline sampling in fixed Handnote sentence positions."""

import argparse
import json
from pathlib import Path

from research.corpus.context_content_train import digest, write


def render_line(source, words, parts, size, *, sampling=4):
    import cv2
    import numpy as np

    from research.corpus.ink_integrity import (
        PAD_X,
        PAD_Y,
        coordinates,
        effective_scale,
        sample_reference,
    )

    result = np.zeros(size[::-1], np.float32)
    for word, mapping in zip(words, parts, strict=True):
        tr = word["transform"]
        scx, scy = effective_scale(tr)
        bx, by = tr["bbox"][:2]
        dx, dy = tr["translation"]
        left = word["left"]
        right = word["right"]
        x0 = max(0, int(np.floor((-PAD_X - dx) / scx + bx + left + 64)) - 2)
        x1 = min(size[0], int(np.ceil((512 + PAD_X - dx) / scx + bx + left + 64)) + 2)
        y0 = max(0, int(np.floor((-PAD_Y - dy) / scy + by + 64)) - 2)
        y1 = min(size[1], int(np.ceil((128 + PAD_Y - dy) / scy + by + 64)) + 2)
        xx = (np.arange((x1 - x0) * sampling) + 0.5) / sampling - 0.5 + x0 - 64
        yy = (np.arange((y1 - y0) * sampling) + 0.5) / sampling - 0.5 + y0 - 64
        u = (xx - left - bx + 0.5) * scx + dx - 0.5
        v = (yy - by + 0.5) * scy + dy - 0.5
        sx, sy = coordinates(mapping, u[None], v[:, None])
        sx = np.broadcast_to(sx, sy.shape)
        sx = ((sx - dx + 0.5) / scx + bx) * 4 - 0.5
        sy = ((sy - dy + 0.5) / scy + by) * 4 - 0.5
        ink = sample_reference(source[:, left * 4 : right * 4], sx, sy)
        patch = cv2.resize(ink, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        result[y0:y1, x0:x1] = np.maximum(result[y0:y1, x0:x1], patch)
    return result


def prepare(old: Path, models: dict[str, str]):
    import numpy as np
    from PIL import Image

    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_handnote_reference import render
    from research.corpus.experiment import create

    verify_run(old)
    for p in models.values():
        verify_run(Path(p))
    profile = profile_store().resolve("personal@v1")
    run = create(
        "ink-integrity-lines",
        config={
            "source": str(old),
            "source_sha256": digest(old / "run.json"),
            "models": models,
            "profile": profile.provenance(),
            "training_updates": 0,
            "C_used": False,
            "single_pass_original_reference": True,
        },
    )
    try:
        groups = json.loads((old / "reports/groups.json").read_text())
        words = json.loads((old / "reports/words.json").read_text())
        for q in groups:
            low, cuts, meta = render(q["text"], profile)
            if not np.array_equal(
                np.asarray(low),
                np.asarray(Image.open(old / f"comparisons/original-handnote-{q['group']}.png")),
            ):
                raise ValueError("Original Handnote reference changed")
            high, _, hm = render(q["text"], profile, raster_scale=4)
            if hm["glyph_provenance"] != meta["glyph_provenance"]:
                raise ValueError("Glyph outlines changed")
            high.save(run.path / f"comparisons/source-{q['group']}-4x.png")
            low.save(run.path / f"comparisons/source-{q['group']}-1x.png")
            for pos, i in enumerate(q["indices"]):
                words[i]["right"] = (
                    words[q["indices"][pos + 1]]["left"]
                    if pos + 1 < len(q["indices"])
                    else low.width
                )
        write(run.path / "reports/words.json", words)
        write(run.path / "reports/groups.json", groups)
        write(
            run.path / "reports/request.json",
            {
                "run": str(run.path),
                "old": str(old),
                "models": models,
                "checkpoints": {
                    k: digest(Path(p) / "logs/checkpoint-1000.pt") for k, p in models.items()
                },
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def replay_controls(request):
    import numpy as np
    import torch

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.aligned_style_model import AlignedStyle

    r = json.loads(Path(request).read_text())
    setup(1707)
    run = Path(r["run"])
    old = Path(r["old"])
    refs = torch.tensor(np.load(old / "reports/reference.npz")["reference"][:, None], device="cuda")
    words = json.loads((old / "reports/words.json").read_text())
    tokens = torch.tensor(text_tokens([q["text"] for q in words]), device="cuda")
    saved = np.load(old / "reports/generated.npz")
    controls = {}
    checked = 0
    for name, p in r["models"].items():
        cp = Path(p) / "logs/checkpoint-1000.pt"
        if digest(cp) != r["checkpoints"][name]:
            raise ValueError("Frozen weight changed")
        model = AlignedStyle(stochastic=name == "stochastic", monotone=True).cuda().eval()
        model.load_state_dict(torch.load(cp, map_location="cuda", weights_only=False)["model"])
        with torch.inference_mode():
            for seed in [1708, 1709] if name == "stochastic" else [1708]:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                out = []
                fields = []
                for k in range(0, len(words), 8):
                    eps = torch.randn((len(words[k : k + 8]), 4), device="cuda", generator=gen)
                    ink, control = model.generate(refs[k : k + 8], tokens[k : k + 8], eps)
                    out.append(ink[:, 0].cpu().numpy())
                    fields.append(control.cpu().numpy())
                key = "mean" if name == "mean" else f"stoch-{seed - 1708}"
                if not np.array_equal(np.concatenate(out), saved[key]):
                    raise ValueError("Old line model output changed")
                controls[key] = np.concatenate(fields)
                checked += len(words)
    np.savez_compressed(run / "reports/frozen-controls.npz", **controls)
    write(
        run / "reports/neural-replay.json",
        {"exact_old_word_arrays": checked, "neural_updates": 0, "weights_unchanged": True},
    )


def compose(run_path: Path):
    import numpy as np
    from PIL import Image

    from hwsol.workspace import Run
    from research.corpus.ink_integrity import (
        field_metrics,
        ink_from_reference,
        map_components,
        untrained_control,
    )
    from research.corpus.ink_integrity_study import pixels

    run = Run(run_path)
    r = json.loads((run_path / "reports/request.json").read_text())
    old = Path(r["old"])
    groups = json.loads((run_path / "reports/groups.json").read_text())
    words = json.loads((run_path / "reports/words.json").read_text())
    controls = np.load(run_path / "reports/frozen-controls.npz")
    obs = []
    metrics = []
    replays = 0
    random_reports = []
    try:
        for g in groups:
            gi = g["group"]
            source = ink_from_reference(Image.open(run_path / f"comparisons/source-{gi}-4x.png"))
            ww = [words[i] for i in g["indices"]]
            mappings = {
                "handnote-hires": [map_components(np.zeros((2, 5, 17)), guarded=False) for _ in ww],
                "single-pass": [
                    map_components(controls["stoch-0"][i], guarded=False) for i in g["indices"]
                ],
                "guarded": [
                    map_components(controls["stoch-0"][i], guarded=True) for i in g["indices"]
                ],
                "mean-guarded": [
                    map_components(controls["mean"][i], guarded=True) for i in g["indices"]
                ],
            }
            random = []
            for i, p in zip(g["indices"], mappings["guarded"], strict=True):
                q, report = untrained_control(p, 1808 + i)
                random_reports.append({"word_index": i, **report})
                if not report["valid_control"]:
                    raise ValueError("Invalid nonlearning comparison; cannot reuse learned map")
                random.append(q)
            mappings["untrained"] = random
            for name, parts in mappings.items():
                a = render_line(source, ww, parts, tuple(g["size"]))
                dest = run_path / f"comparisons/{name}-{gi}.png"
                Image.fromarray(np.uint8((1 - a) * 255)).save(dest)
                if not np.array_equal(a, render_line(source, ww, parts, tuple(g["size"]))):
                    raise ValueError("Line renderer replay differs")
                replays += 1
                item = {
                    "id": f"{name}-line-{gi}",
                    "method": name,
                    "text": g["text"],
                    "split": "exposed-train-composition",
                    "image": str(dest),
                    "sha256": digest(dest),
                    "row_index": g["source_index"],
                    "source_hash": g["source_sha256"],
                    "source_crop": g["source_crop"],
                    "personal_collected": False,
                    "neural_updates": 0,
                    "source_type": "personal-font"
                    if name == "handnote-hires"
                    else "untrained-geometric-control"
                    if name == "untrained"
                    else "frozen-neural-map-render-correction",
                }
                obs.append(item)
                metrics.append({**item, **pixels(a), "field": [field_metrics(p) for p in parts]})
            for name, source_name in [
                ("legacy", "stoch-0"),
                ("legacy-font", "handnote"),
                ("real", "real"),
            ]:
                src = old / f"comparisons/{source_name}-{gi}.png"
                dest = run_path / f"comparisons/{name}-{gi}.png"
                dest.write_bytes(src.read_bytes())
                if name != "real":
                    obs.append(
                        {
                            "id": f"{name}-line-{gi}",
                            "method": name,
                            "text": g["text"],
                            "split": "exposed-train-composition",
                            "image": str(dest),
                            "sha256": digest(dest),
                            "row_index": g["source_index"],
                            "source_hash": g["source_sha256"],
                            "source_crop": g["source_crop"],
                            "personal_collected": False,
                            "neural_updates": 0,
                            "source_type": "historical-frozen-render",
                        }
                    )
        write(run_path / "reports/observations-input.json", obs)
        write(run_path / "reports/metrics.json", metrics)
        write(run_path / "reports/nonlearning-control.json", random_reports)
        run.finish(
            "held",
            result={
                "groups": 4,
                "observations": len(obs),
                "exact_cpu_line_replays": replays,
                "exact_old_neural_word_replays": 72,
                "training_updates": 0,
                "C_used": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    replay_controls(p.parse_args().request)
