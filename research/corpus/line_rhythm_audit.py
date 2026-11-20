"""Measure true word gaps and test a small geometry fit by document."""

import math
from pathlib import Path

import numpy as np

from research.corpus.context_content_train import digest, write
from research.corpus.context_metric_evaluate import read


def features(left: dict, right: dict) -> list[float]:
    a, b = left["ink_bbox"], right["ink_bbox"]
    return [
        math.log1p(len(left["text"])),
        math.log1p(len(right["text"])),
        float(left["text"][-1] in ",;:.!?"),
        float(right["text"][0].isupper()),
        abs(math.log((a[3] - a[1]) / (b[3] - b[1]))),
    ]


def fit(x: np.ndarray, gap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fixed regularized, robust fit in log gap units."""
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 0.1)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    target = np.log(gap)
    regularization = np.diag([0, 8, 8, 8, 8, 8])
    weights = np.ones(len(x))
    for _ in range(8):
        coeff = np.linalg.solve(
            design.T @ (design * weights[:, None]) + regularization, design.T @ (target * weights)
        )
        residual = np.abs(design @ coeff - target)
        weights = 0.25 / np.maximum(residual, 0.25)
    return mean, scale, coeff


def observations(registry: Path) -> list[dict]:
    rows = []
    for line in read(registry / "reports/line-registration.json"):
        words = {x["position"]: x for x in line["training_word_links"]}
        for pair in line["adjacent_pairs"]:
            left, right = words[pair["left_position"]], words[pair["right_position"]]
            rows.append(
                {
                    "source": line["source_hash"],
                    "line": line["candidate_id"],
                    "left": left,
                    "right": right,
                    "features": features(left, right),
                    "gap": pair["gap_over_mean_word_ink_height"],
                }
            )
    return rows


def evaluate(registry: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(registry)
    rows = observations(registry)
    x = np.asarray([r["features"] for r in rows])
    y = np.asarray([r["gap"] for r in rows])
    documents = np.asarray([r["source"] for r in rows])
    run = create(
        "document-separated-word-rhythm-fit",
        config={
            "registration_sha256": digest(registry / "run.json"),
            "registry": str(registry),
            "protocol_sha256": digest("docs/V01023_PROTOCOL.md"),
            "C_used": False,
            "new_neural_updates": 0,
        },
    )
    try:
        folds = []
        for doc in sorted(set(documents)):
            test = documents == doc
            train = ~test
            mean, scale, coeff = fit(x[train], y[train])
            bounds = np.quantile(y[train], [0.1, 0.9])
            predictions = np.exp(coeff[0] + ((x[test] - mean) / scale) @ coeff[1:]).clip(*bounds)
            constant = float(np.median(y[train]))
            folds.append(
                {
                    "document": doc,
                    "train_pairs": int(train.sum()),
                    "test_pairs": int(test.sum()),
                    "constant_mae": float(np.abs(y[test] - constant).mean()),
                    "context_mae": float(np.abs(y[test] - predictions).mean()),
                    "held_group_used_for_fitting": False,
                }
            )
            for i, prediction in zip(np.flatnonzero(test), predictions, strict=True):
                rows[i].update(constant_prediction=constant, fitted_prediction=float(prediction))
        constant = float(np.mean([f["constant_mae"] for f in folds]))
        contextual = float(np.mean([f["context_mae"] for f in folds]))
        gate = contextual <= constant * 0.95 and all(
            f["context_mae"] <= f["constant_mae"] * 1.10 for f in folds if f["test_pairs"] >= 10
        )
        mean, scale, coeff = fit(x, y)
        write(run.path / "reports/pairs.json", rows)
        write(run.path / "reports/folds.json", folds)
        write(
            run.path / "reports/geometry-parameters.json",
            {
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "coefficients": coeff.tolist(),
                "global_median": float(np.median(y)),
                "source_range_p10_p90": np.quantile(y, [0.1, 0.9]).tolist(),
                "deployed": False,
            },
        )
        summary = {
            "documents": len(folds),
            "pairs": len(rows),
            "doc_macro_mae_median": constant,
            "doc_macro_mae_context": contextual,
            "passed_context_gap_gate": bool(gate),
            "micro_mae_median": float(
                np.mean([abs(r["gap"] - r["constant_prediction"]) for r in rows])
            ),
            "micro_mae_context": float(
                np.mean([abs(r["gap"] - r["fitted_prediction"]) for r in rows])
            ),
            "new_neural_updates": 0,
            "historical_documents_seen_upstream": True,
            "C_used": False,
        }
        run.finish("held", result=summary)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path
