"""CPU-only frozen-reference experiments; all outputs are private immutable runs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_handnote_reference import render
from research.corpus.experiment import create
from research.corpus.ink_integrity import (
    PAD_X,
    PAD_Y,
    field_metrics,
    ink_from_reference,
    map_components,
    render_word,
)
from research.corpus.word_style_data import normalize


def prepare(data: Path, models: dict[str, str]) -> Path:
    verify_run(data)
    for p in models.values():
        verify_run(Path(p))
    ids = json.loads((Path(models["mean"]) / "reports/indices.json").read_text())["evaluated"]
    rows = json.loads((data / "reports/samples.json").read_text())
    arrays = np.load(data / "reports/arrays.npz")
    profile = profile_store().resolve("personal@v1")
    run = create(
        "ink-integrity-references",
        config={
            "data": str(data),
            "data_sha256": sha256(data / "run.json"),
            "models": models,
            "raster_scale": 4,
            "profile": profile.provenance(),
            "C_used": False,
            "training_updates": 0,
        },
    )
    try:
        records = []
        for i in ids:
            low, _, meta = render(rows[i]["text"], profile)
            current, tr = normalize(low)
            if (
                not np.array_equal(current, arrays["reference"][i])
                or tr != rows[i]["reference_transform"]
            ):
                raise ValueError(f"Frozen reference differs: {i}")
            high, _, high_meta = render(rows[i]["text"], profile, raster_scale=4)
            if high_meta["glyph_provenance"] != meta["glyph_provenance"]:
                raise ValueError("Changed outlines in supersampled reference")
            p = run.path / f"comparisons/reference-{i:04d}-4x.png"
            high.save(p)
            low.save(run.path / f"comparisons/reference-{i:04d}-1x.png")
            records.append(
                {
                    "index": i,
                    "text": rows[i]["text"],
                    "split": rows[i]["dataset_split"],
                    "image": str(p),
                    "sha256": sha256(p),
                    "reference_transform": tr,
                    "glyph_provenance": high_meta,
                    "old_reference_exact": True,
                }
            )
            print("reference", i, flush=True)
        atomic_json(run.path / "reports/references.json", records)
        run.finish(
            "prepared",
            result={
                "references": len(records),
                "old_normalized_references_exact": len(records),
                "C_used": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def pixels(a: np.ndarray) -> dict:
    return {
        "border_core": int(
            (a[:2] > 0.5).sum()
            + (a[-2:] > 0.5).sum()
            + (a[:, :2] > 0.5).sum()
            + (a[:, -2:] > 0.5).sum()
        ),
        "core_pixels": int((a > 0.5).sum()),
        "coverage": float(a.sum()),
        "faint_pixels": int(((a > 0.05) & (a < 0.5)).sum()),
    }


def evaluate(data: Path, references: Path, models: dict[str, str]) -> Path:
    import time

    verify_run(data)
    verify_run(references)
    for p in models.values():
        verify_run(Path(p))
    refs = {q["index"]: q for q in json.loads((references / "reports/references.json").read_text())}
    rows = json.loads((data / "reports/samples.json").read_text())
    run = create(
        "ink-integrity-frozen-render",
        config={
            "data": str(data),
            "references": str(references),
            "models": models,
            "training_updates": 0,
            "C_used": False,
            "protocol_sha256": sha256(Path("docs/V0108_PROTOCOL.md")),
            "arms": ["legacy", "single-pass", "guarded"],
            "padding": [PAD_X, PAD_Y],
            "raster_scale": 4,
            "supersampling": 4,
        },
    )
    try:
        metrics = []
        observations = []
        replays = 0
        start = time.perf_counter()
        sources = {i: ink_from_reference(Image.open(q["image"])) for i, q in refs.items()}
        for model, path in models.items():
            source = Path(path)
            ids = json.loads((source / "reports/indices.json").read_text())["evaluated"]
            fields = np.load(source / "reports/generated-controls.npz")
            old = np.load(source / "reports/generated.npz")
            checkpoint = source / "logs/checkpoint-1000.pt"
            weight_sha = sha256(checkpoint)
            for seed in fields:
                for k, i in enumerate(ids):
                    q = refs[i]
                    control = fields[seed][k]
                    for arm in ["legacy", "single-pass", "guarded"]:
                        parts = map_components(control, guarded=arm == "guarded")
                        if arm == "legacy":
                            a = np.pad(old[seed][k], ((PAD_Y, PAD_Y), (PAD_X, PAD_X)))
                        else:
                            a = render_word(sources[i], q["reference_transform"], parts)
                        dest = run.path / f"rendered/pages/{model}-{seed}-{arm}-{i:04d}.png"
                        Image.fromarray(np.uint8((1 - np.clip(a, 0, 1)) * 255)).save(dest)
                        item = {
                            "id": dest.stem,
                            "method": model + "-" + arm,
                            "model": model,
                            "seed": int(seed),
                            "arm": arm,
                            "row_index": i,
                            "text": q["text"],
                            "split": q["split"],
                            "image": str(dest),
                            "sha256": sha256(dest),
                            "checkpoint_sha256": weight_sha,
                            "source_type": "fixed-neural-field-render-correction",
                            "personal_collected": False,
                            "training_updates": 0,
                            "source_hash": rows[i]["source_hash"],
                        }
                        observations.append(item)
                        metrics.append(
                            {
                                **item,
                                **pixels(a),
                                "field": field_metrics(parts),
                                "pixels_outside_old_canvas": int(
                                    (a > 0.5).sum()
                                    - (a[PAD_Y : PAD_Y + 128, PAD_X : PAD_X + 512] > 0.5).sum()
                                ),
                            }
                        )
                        if arm == "guarded":
                            repeated = render_word(sources[i], q["reference_transform"], parts)
                            if not np.array_equal(a, repeated):
                                raise ValueError("CPU renderer replay differs")
                            replays += 1
                    print(model, seed, i, flush=True)
            if sha256(checkpoint) != weight_sha:
                raise ValueError("Frozen weight changed")
        for i, q in refs.items():
            for name, parts in [
                ("handnote-hires", map_components(np.zeros((2, 5, 17)), guarded=False))
            ]:
                a = render_word(sources[i], q["reference_transform"], parts)
                dest = run.path / f"comparisons/{name}-{i:04d}.png"
                Image.fromarray(np.uint8((1 - a) * 255)).save(dest)
                observations.append(
                    {
                        "id": dest.stem,
                        "method": name,
                        "row_index": i,
                        "text": q["text"],
                        "split": q["split"],
                        "image": str(dest),
                        "sha256": sha256(dest),
                        "personal_collected": False,
                        "source_type": "untrained-personal-font-reference",
                        "training_updates": 0,
                    }
                )
        atomic_json(run.path / "reports/observations-input.json", observations)
        atomic_json(run.path / "reports/metrics.json", metrics)
        run.finish(
            "held",
            result={
                "rendered_candidates": len(metrics),
                "exact_cpu_replays": replays,
                "seconds": time.perf_counter() - start,
                "training_updates": 0,
                "C_used": False,
                "weights_unchanged": True,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
