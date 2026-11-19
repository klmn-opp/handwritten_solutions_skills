"""Versioned scan-density normalization; preserves the nonzero ink support."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance


def normalize_density(ink: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.isfinite(ink).all() or ink.min() < 0 or ink.max() > 1:
        raise ValueError("Finite ink occupancy in [0, 1] required")
    core = ink[ink > 0.1]
    if not len(core):
        raise ValueError("Cannot normalize blank or unsupported faint observations")
    level = float(np.quantile(core, 0.9))
    gain = 1.0 / max(0.25, level) if level < 0.9 else 1.0
    return np.clip(ink * gain, 0, 1), gain


def prepare(parent: Path) -> Path:
    verify_run(parent)
    with np.load(parent / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    rows = json.loads((parent / "reports/samples.json").read_text())
    images, report = [], []
    for i, ink in enumerate(arrays["ink"]):
        im, gain = normalize_density(ink)
        assert np.array_equal(im > 0, ink > 0)
        images.append(im)
        report.append(
            {
                "index": i,
                "label": rows[i]["label"],
                "gain": gain,
                "original_sha256": rows[i]["training_image_sha256"],
                "strong_ink_before": int((ink > 0.5).sum()),
                "strong_ink_after": int((im > 0.5).sum()),
            }
        )
    run = create(
        "real-ink-density-data",
        config={
            "parent": str(parent),
            "parent_sha256": digest(parent / "run.json"),
            "C_used": False,
            "geometry_warp": False,
            "new_nonzero_pixels": 0,
            "paper_zero_unchanged": True,
            "original_observations_preserved": True,
            "thresholded_footprint_can_change": True,
            "new_real_observations": 0,
            "rule": "gain to p90(ink>0.1) if below0.9; maxgain4; clip at1",
        },
    )
    arrays["ink"] = np.array(images)
    arrays["sdf"] = np.stack([signed_distance(im) for im in images])
    for i, im in enumerate(images):
        dest = run.path / f"solutions/glyph-{i:04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
        rows[i] = {
            **rows[i],
            "before_density_image": rows[i]["training_image"],
            "before_density_sha256": rows[i]["training_image_sha256"],
            "training_image": str(dest),
            "training_image_sha256": digest(dest),
            "density_gain": report[i]["gain"],
        }
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays)
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/density.json", report)
    for name in ["coverage", "groups"]:
        write(
            run.path / f"reports/{name}.json",
            json.loads((parent / f"reports/{name}.json").read_text()),
        )
    run.finish(
        "held",
        result={
            "observations": len(rows),
            "adjusted": sum(q["gain"] != 1 for q in report),
            "new_nonzero_pixels": 0,
            "neural_updates": 0,
            "C_used": False,
        },
    )
    return run.path
