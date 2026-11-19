"""Frozen word-geometry composition probe; never claims end-to-end line generation."""

import argparse
import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def restore_word(ink, transform: dict, left: int, canvas_size: tuple[int, int], pad: int = 64):
    """Inverse the known reference normalization, without fitting generated ink bounds."""
    import numpy as np
    from PIL import Image

    scale = transform["scale"]
    x, y = transform["bbox"][:2]
    dx, dy = transform["translation"]
    # Output canvas -> original line -> word source -> normalized model pixels.
    affine = (scale, 0, dx - scale * (left + x + pad), 0, scale, dy - scale * (y + pad))
    image = Image.fromarray(np.uint8(np.clip(ink, 0, 1) * 255))
    return np.asarray(
        image.transform(
            canvas_size, Image.Transform.AFFINE, affine, Image.Resampling.BILINEAR, fillcolor=0
        )
    )


def run_probe(models: dict[str, str], views: Path) -> Path:
    import math

    import numpy as np
    from PIL import Image

    from hwsol.workspace import profile_store, verify_run
    from research.corpus.context_handnote_reference import render
    from research.corpus.experiment import create
    from research.corpus.word_style_data import normalize

    verify_run(views)
    for model in models.values():
        verify_run(Path(model))
    source = json.loads((views / "reports/samples.json").read_text())
    chosen = [
        (i, q)
        for i, q in enumerate(source)
        if "new-english-sol-review" in q["source_package"]
        and len(q["normalized_label"].split()) >= 6
        and q["dataset_split"] == "train"
    ][:4]
    if len(chosen) != 4:
        raise ValueError("Expected four predeclared exposed training sentences")
    profile = profile_store().resolve("personal@v1")
    run = create(
        "aligned-style-sentence-probe",
        config={
            "models": models,
            "views": str(views),
            "view_sha256": digest(views / "run.json"),
            "profile": profile.provenance(),
            "C_used": False,
            "word_composition_not_line_generation": True,
            "training_updates": 0,
        },
    )
    try:
        rows = []
        references = []
        groups = []
        for gi, (si, q) in enumerate(chosen):
            if digest(Path(q["crop"])) != q["crop_sha256"]:
                raise ValueError("Real source changed")
            image, cuts, provenance = render(q["normalized_label"], profile)
            image.save(run.path / f"comparisons/original-handnote-{gi}.png")
            Image.open(q["crop"]).save(run.path / f"comparisons/real-{gi}.png")
            bounds = [0] + [int(math.floor(c)) for c in cuts] + [image.width]
            indices = []
            for word, left, right in zip(
                q["normalized_label"].split(), bounds[:-1], bounds[1:], strict=True
            ):
                arr, tr = normalize(image.crop((left, 0, right, image.height)))
                indices.append(len(rows))
                references.append(arr)
                rows.append({"text": word, "transform": tr, "left": left, "group": gi})
            groups.append(
                {
                    "group": gi,
                    "text": q["normalized_label"],
                    "source_index": si,
                    "source_crop": q["crop"],
                    "source_sha256": q["source_hash"],
                    "source_crop_sha256": q["crop_sha256"],
                    "split": "exposed-train-composition",
                    "size": [image.width + 128, image.height + 128],
                    "indices": indices,
                    "glyph_provenance": provenance,
                }
            )
        np.savez_compressed(run.path / "reports/reference.npz", reference=np.stack(references))
        write(run.path / "reports/groups.json", groups)
        write(run.path / "reports/words.json", rows)
        request = {
            "run": str(run.path),
            "models": models,
            "checkpoints": {
                k: json.loads((Path(v) / "reports/result.json").read_text())["checkpoint_sha256"]
                for k, v in models.items()
            },
        }
        write(run.path / "reports/request.json", request)
        with (run.path / "logs/inference.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    "research.corpus.aligned_style_lines",
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        generated = np.load(run.path / "reports/generated.npz")
        observations = []
        boundaries = []
        for group in groups:
            for name, images in {"handnote": np.stack(references), **dict(generated)}.items():
                canvas = np.zeros(group["size"][::-1], np.uint8)
                for i in group["indices"]:
                    word = rows[i]
                    patch = restore_word(
                        images[i], word["transform"], word["left"], tuple(group["size"])
                    )
                    canvas = np.maximum(canvas, patch)
                p = run.path / f"comparisons/{name}-{group['group']}.png"
                Image.fromarray(255 - canvas).save(p)
                observations.append(
                    {
                        "id": f"{name}-line-{group['group']}",
                        "method": name,
                        "text": group["text"],
                        "split": group["split"],
                        "image": str(p),
                        "sha256": digest(p),
                        "row_index": group["source_index"],
                        "source_crop": group["source_crop"],
                        "source_hash": group["source_sha256"],
                        "source_type": "personal-font-with-shared-resampling"
                        if name == "handnote"
                        else "learned-word-geometry-composed-line",
                        "personal_collected": False,
                    }
                )
                border = int(
                    (canvas[:2] > 128).sum()
                    + (canvas[-2:] > 128).sum()
                    + (canvas[:, :2] > 128).sum()
                    + (canvas[:, -2:] > 128).sum()
                )
                boundaries.append({"group": group["group"], "method": name, "border_core": border})
        write(run.path / "reports/observations-input.json", observations)
        write(run.path / "reports/boundaries.json", boundaries)
        build_pdf(run.path, groups)
        run.finish(
            "held",
            result={
                "pages": 4,
                "word_inferences": len(rows),
                "neural_updates": 0,
                "C_used": False,
                "line_model": False,
                "automatic_publication": False,
                "border_core": sum(q["border_core"] for q in boundaries),
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def build_pdf(run: Path, groups: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    methods = [
        ("real", "REAL personal scan"),
        ("handnote", "HANDNOTE - shared normalization/resampling"),
        ("mean", "NEW mean word geometry"),
        ("stoch-0", "NEW stochastic word geometry - seed 1708"),
        ("stoch-1", "NEW stochastic word geometry - seed 1709"),
    ]
    with PdfPages(run / "rendered/homework_solutions.pdf") as pdf:
        for group in groups:
            fig, axs = plt.subplots(5, 1, figsize=(8.27, 11.69))
            fig.suptitle(group["text"], fontsize=11)
            for ax, (name, label) in zip(axs, methods, strict=True):
                ax.imshow(Image.open(run / f"comparisons/{name}-{group['group']}.png"))
                ax.set_title(label, loc="left", fontsize=9)
                ax.axis("off")
            fig.text(
                0.04,
                0.02,
                "Exposed Train sentences. Per-word neural geometry + fixed composition. "
                "NOT end-to-end neural lines.",
                fontsize=7,
            )
            fig.subplots_adjust(top=0.94, bottom=0.05, hspace=0.5)
            pdf.savefig(fig)
            fig.savefig(run / f"rendered/pages/line-{group['group']}.png", dpi=150)
            plt.close(fig)


def worker(request):
    import time

    import numpy as np
    import torch

    from research.corpus.aligned_style_contract import text_tokens
    from research.corpus.aligned_style_experiment import setup
    from research.corpus.aligned_style_model import AlignedStyle

    r = json.loads(Path(request).read_text())
    run = Path(r["run"])
    setup(1707)
    references = np.load(run / "reports/reference.npz")["reference"]
    rows = json.loads((run / "reports/words.json").read_text())
    ref = torch.tensor(references[:, None], device="cuda")
    tokens = torch.tensor(text_tokens([q["text"] for q in rows]), device="cuda")
    output = {}
    runtime = []
    for name, value in r["models"].items():
        source = Path(value)
        cp = source / "logs/checkpoint-1000.pt"
        if digest(cp) != r["checkpoints"][name]:
            raise ValueError("Frozen model changed")
        model = AlignedStyle(stochastic=name == "stochastic", monotone=True).cuda().eval()
        model.load_state_dict(torch.load(cp, map_location="cuda", weights_only=False)["model"])
        with torch.inference_mode():
            for seed in [1708, 1709] if name == "stochastic" else [1708]:
                gen = torch.Generator(device="cuda").manual_seed(seed)
                chunks = []
                torch.cuda.synchronize()
                start = time.perf_counter()
                for k in range(0, len(rows), 8):
                    eps = torch.randn((len(rows[k : k + 8]), 4), device="cuda", generator=gen)
                    ink, _ = model.generate(ref[k : k + 8], tokens[k : k + 8], eps)
                    chunks.append(ink[:, 0].cpu().numpy())
                torch.cuda.synchronize()
                runtime.append(
                    {
                        "model": name,
                        "seed": seed,
                        "words": len(rows),
                        "seconds": time.perf_counter() - start,
                    }
                )
                output["mean" if name == "mean" else f"stoch-{seed - 1708}"] = np.concatenate(
                    chunks
                )
    np.savez_compressed(run / "reports/generated.npz", **output)
    write(run / "reports/runtime.json", runtime)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("request", type=Path)
    worker(p.parse_args().request)
