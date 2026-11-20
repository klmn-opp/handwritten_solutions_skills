"""Replay frozen neural ink with measured English proportions and spacing."""

import json
import subprocess
from functools import partial
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write


def read(path: Path):
    return json.loads(path.read_text())


def measure_thresholds(audit: Path) -> dict:
    ratios = read(audit / "reports/real-height-ratios.json")
    gaps = read(audit / "reports/real-gaps.json")
    t = [row["ratio"] for row in ratios["t"]]
    return {
        "t_median": float(np.median(t)),
        "t_upper": float(np.quantile(t, 0.75)),
        "l_left_upper": max(
            0.08, float(np.quantile([q["gap"] for q in gaps if q["right"] == "l"], 0.75))
        ),
        "r_right_upper": max(
            0.08, float(np.quantile([q["gap"] for q in gaps if q["left"] == "r"], 0.75))
        ),
    }


def launch(audit: Path, baseline: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for path in [audit, baseline]:
        verify_run(path)
    old = read(baseline / "reports/request.json")
    cfg = {
        "audit": str(audit),
        "audit_sha256": digest(audit / "run.json"),
        "baseline": str(baseline),
        "baseline_sha256": digest(baseline / "run.json"),
        "checkpoint": old["checkpoint"],
        "checkpoint_sha256": old["checkpoint_sha256"],
        "thresholds": measure_thresholds(audit),
        "seed": 4221,
        "new_neural_updates": 0,
        "C_used": False,
    }
    run = create("neural-english-metric-layout", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-metric-layout").write_text(str(run.path))
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
                    timeout=220,
                )
        result = report(run.path, baseline)
        run.finish("held", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def ink_metrics(layers, tokens) -> dict:
    import torch

    from research.corpus.context_metric_layout import BODY

    mask = layers > 0.5
    extents = []
    for layer in mask:
        yy = torch.where(layer.any(1))[0]
        xx = torch.where(layer.any(0))[0]
        extents.append(
            [int(xx[0]), int(yy[0]), int(xx[-1]) + 1, int(yy[-1]) + 1] if len(xx) else None
        )
    heights = [b[3] - b[1] for t, b in zip(tokens, extents, strict=True) if b and chr(t[0]) in BODY]
    scale = float(np.median(heights)) if heights else 1.0
    gaps = {"t_height": [], "l_left": [], "r_right": [], "word_gap": []}
    overlap = []
    for j, (token, b) in enumerate(zip(tokens, extents, strict=True)):
        if b and token[0] == ord("t") and heights:
            gaps["t_height"].append((b[3] - b[1]) / scale)
        if not j:
            continue
        intersection = int((mask[j] & mask[j - 1]).sum())
        overlap.append(intersection)
        if not b or not extents[j - 1] or not heights:
            continue
        gap = (b[0] - extents[j - 1][2]) / scale
        if not token[1]:
            gaps["word_gap"].append(gap)
        else:
            if token[0] == ord("l"):
                gaps["l_left"].append(gap)
            if tokens[j - 1, 0] == ord("r"):
                gaps["r_right"].append(gap)
    return {
        **gaps,
        "empty_glyphs": sum(b is None for b in extents),
        "overlap": overlap,
        "ink_at_edge": bool(
            mask[..., :2, :].any()
            or mask[..., -2:, :].any()
            or mask[..., :, :2].any()
            or mask[..., :, -2:].any()
        ),
    }


def worker(request: Path, mode: str) -> None:
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk
    from research.corpus.context_metric_layout import adjust_boxes

    cfg = read(request)
    run, baseline = Path(cfg["run"]), Path(cfg["baseline"])
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    assert digest(baseline / "run.json") == cfg["baseline_sha256"]
    setup(cfg["seed"])
    model = CanonicalInk().cuda().eval()
    model.load_state_dict(
        torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)["model"]
    )
    outputs = []
    prior = read(run / "reports/outputs.json") if mode == "replay" else None
    for row in read(baseline / "reports/outputs.json"):
        ident = f"{row['id']}-{row['seed']}"
        with np.load(baseline / f"reports/{ident}.npz") as image:
            tokens, boxes, style = image["tokens"][0], image["boxes"][0], image["style"]
            height, width = image["ink"].shape
            token_tensor, condition = [torch.tensor(x, device="cuda") for x in [tokens, style]]

            render = partial(
                compose,
                model,
                token_tensor,
                style=condition,
                height=height,
                width=width,
                return_layers=True,
            )
            layers = render(torch.tensor(boxes, device="cuda"))
            assert np.array_equal(image["ink"], layers.amax(0).cpu().numpy())
        semantic = tokens[tokens[:, 0] != 0]
        original = ink_metrics(layers, semantic)
        for method in ["anchored", "preserve-word-gaps"]:
            candidate = adjust_boxes(
                tokens, boxes, cfg["thresholds"], preserve_word_gaps=method != "anchored"
            )
            new = render(torch.tensor(candidate, device="cuda"))
            metrics = ink_metrics(new, semantic)
            added_overlap = sum(
                max(0, v - u) for u, v in zip(original["overlap"], metrics["overlap"], strict=True)
            )
            array = new.amax(0).cpu().numpy()
            png = np.uint8((1 - array) * 255)
            dest = run / f"rendered/pages/{ident}-{method}.png"
            receipt = run / f"reports/{ident}-{method}.npz"
            if mode == "generate":
                Image.fromarray(png).save(dest)
                np.savez_compressed(receipt, ink=array, boxes=candidate)
            else:
                with np.load(receipt) as saved:
                    assert np.array_equal(array, saved["ink"])
                    assert np.array_equal(candidate, saved["boxes"])
                assert np.array_equal(png, np.asarray(Image.open(dest)))
                assert digest(dest) == prior[len(outputs)]["sha256"]
            outputs.append(
                {
                    "id": row["id"],
                    "seed": row["seed"],
                    "text": row["text"],
                    "kind": row["kind"],
                    "method": method,
                    "image": str(dest),
                    "sha256": digest(dest),
                    "baseline_image": row["image"],
                    "before": original,
                    "after": metrics,
                    "added_overlap_pixels": added_overlap,
                    "source_glyphs": row["glyphs"],
                    "layout_changed": not np.array_equal(boxes, candidate),
                }
            )
    if mode == "generate":
        write(run / "reports/outputs.json", outputs)
    else:
        write(
            run / "reports/replay.json",
            {"count": len(outputs), "all_boxes_float_ink_png_exact": True},
        )
    print(mode, len(outputs), flush=True)


def report(run: Path, baseline: Path) -> dict:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    from PIL import Image

    rows = read(run / "reports/outputs.json")
    replay = read(run / "reports/replay.json")
    summary = {}
    for method in ["anchored", "preserve-word-gaps"]:
        relevant = [q for q in rows if q["method"] == method]
        summary[method] = {
            side: {
                key: float(np.median([n for q in relevant for n in q[side][key]]))
                for key in ["t_height", "l_left", "r_right", "word_gap"]
            }
            for side in ["before", "after"]
        }
        summary[method].update(
            empty_glyphs=sum(q["after"]["empty_glyphs"] for q in relevant),
            edge_cases=sum(q["after"]["ink_at_edge"] for q in relevant),
            added_adjacent_overlap_pixels=sum(q["added_overlap_pixels"] for q in relevant),
        )
    cases = read(baseline / "reports/cases.json")
    case_by_id = {q["id"]: q for q in cases}
    sizes = {"line": (11.69, 8.27), "word": (8.27, 11.69)}
    for kind, filename in [("line", "line-layout.pdf"), ("word", "word-layout.pdf")]:
        with PdfPages(run / "comparisons" / filename) as pdf:
            for row in rows:
                if row["method"] != "anchored" or row["kind"] != kind:
                    continue
                other = next(
                    q
                    for q in rows
                    if q["id"] == row["id"]
                    and q["seed"] == row["seed"]
                    and q["method"] != "anchored"
                )
                panels = [
                    ("Handnote", case_by_id[row["id"]]["baseline"]),
                    ("Frozen neural ink and original layout", row["baseline_image"]),
                    ("Neural ink with original word anchors", row["image"]),
                    ("Neural ink with original word gaps", other["image"]),
                ]
                figure = Figure(figsize=sizes[kind])
                figure.suptitle(f"{row['text']}   (seed {row['seed']})", fontsize=12)
                for ax, (label, image) in zip(figure.subplots(4, 1), panels, strict=True):
                    with Image.open(image) as pic:
                        ax.imshow(np.asarray(pic.convert("L")), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(label, fontsize=9, loc="left")
                    ax.axis("off")
                figure.subplots_adjust(left=0.06, right=0.94, top=0.90, bottom=0.10, hspace=0.55)
                pdf.savefig(figure)
    result = {
        "outputs": len(rows),
        "replay": replay,
        "measurements": summary,
        "new_neural_updates": 0,
        "C_used": False,
        "candidate_quality_requires_image_review": True,
    }
    write(run / "reports/summary.json", result)
    return result


if __name__ == "__main__":
    import sys

    worker(Path(sys.argv[1]), sys.argv[2])
