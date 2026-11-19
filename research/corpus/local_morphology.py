"""Train-only within-class residual coordinates; never an output-image synthesizer."""

import json
from pathlib import Path

import numpy as np

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def residual_coordinates(ink: np.ndarray, labels: np.ndarray, dimensions=8) -> tuple:
    if len(ink) != len(labels) or dimensions >= len(ink) or not np.isfinite(ink).all():
        raise ValueError("Finite real observations and a smaller residual dimension required")
    classes = np.unique(labels)
    flat = ink.reshape(len(ink), -1).astype(np.float64)
    residual = flat.copy()
    means = []
    for c in classes:
        mean = flat[labels == c].mean(0)
        residual[labels == c] -= mean
        means.append(mean)
    eigenvalues, vectors = np.linalg.eigh(residual @ residual.T)
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    selected = eigenvalues[order].clip(1e-8)
    components = vectors[:, order].T @ residual / np.sqrt(selected[:, None])
    # Stable signs, independent of the eigensolver's arbitrary orientation.
    signs = np.sign(components[np.arange(dimensions), np.abs(components).argmax(1)])
    components *= np.where(signs == 0, 1, signs)[:, None]
    scale = np.sqrt(selected / max(1, len(ink) - 1))
    codes = (residual @ components.T) / scale
    priors = {}
    for c in classes:
        cc = codes[labels == c]
        cov = np.cov(cc.T) if len(cc) > 1 else np.zeros((dimensions, dimensions))
        priors[str(int(c))] = {
            "count": len(cc),
            "mean": cc.mean(0).tolist(),
            "cholesky": np.linalg.cholesky(cov + np.eye(dimensions) * 1e-4).tolist(),
            "variation_supported": len(cc) > 1,
        }
    return (
        codes.astype(np.float32),
        priors,
        {
            "classes": classes,
            "means": np.array(means),
            "components": components,
            "scale": scale,
            "eigenvalues": selected,
            "variance_fraction": selected.sum() / max(1e-8, eigenvalues.clip(0).sum()),
        },
    )


def prepare(parent: Path) -> Path:
    verify_run(parent)
    with np.load(parent / "reports/arrays.npz") as f:
        arrays = {k: f[k] for k in f.files}
    codes, priors, decomposition = residual_coordinates(arrays["ink"], arrays["labels"])
    run = create(
        "train-local-morphology-data",
        config={
            "parent": str(parent),
            "parent_sha256": digest(parent / "run.json"),
            "training_indices_unchanged": True,
            "real_images_unchanged": True,
            "C_used": False,
            "neural_updates": 0,
            "coordinates_are_not_identified_personal_style_factors": True,
            "PCA_inverse_not_used_by_generator": True,
        },
    )
    np.savez_compressed(run.path / "reports/arrays.npz", **arrays, local_codes=codes)
    np.savez_compressed(run.path / "reports/train-decomposition.npz", **decomposition)
    write(run.path / "reports/local-priors.json", priors)
    for name in ["samples", "coverage", "groups"]:
        write(
            run.path / f"reports/{name}.json",
            json.loads((parent / f"reports/{name}.json").read_text()),
        )
    run.finish(
        "held",
        result={
            "observations": len(codes),
            "dimensions": 8,
            "residual_variance_fraction": float(decomposition["variance_fraction"]),
            "neural_updates": 0,
            "new_real_observations": 0,
            "C_used": False,
        },
    )
    return run.path
