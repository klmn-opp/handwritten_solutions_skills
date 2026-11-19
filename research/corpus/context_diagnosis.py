"""Read-only diagnosis of frozen v0.9 models; historical holdout is never called new."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus.experiment import create
from research.corpus.reference_data import align_real
from research.corpus.reference_evaluate import character_metrics, shape_distance


def nearest(query: np.ndarray, targets: np.ndarray) -> dict | None:
    if not len(targets):
        return None
    distance = shape_distance(query[:, None], targets[None])
    binary = (query[:, None] > 0.5) != (targets[None] > 0.5)
    return {
        "mean_nearest_shape": float(distance.min(1).mean()),
        "minimum_shape": float(distance.min()),
        "exact_binary_count": int((binary.sum((-2, -1)).min(1) == 0).sum()),
        "target_count": len(targets),
    }


def worker(path: Path) -> None:
    import torch

    from research.corpus.reference_generator import ReferenceCVAE, ink_from_sdf

    request = json.loads(path.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requires the real device")
    torch.set_num_threads(2)
    torch.manual_seed(1010)
    data = np.load(request["arrays"], allow_pickle=False)
    result = {}
    for name, record in request["models"].items():
        checkpoint = Path(record["checkpoint"])
        if sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed")
        state = torch.load(checkpoint, weights_only=True, map_location="cuda")
        model = ReferenceCVAE(len(state["labels"])).cuda().eval()
        model.load_state_dict(state["model"])
        reference = torch.tensor(data["reference"][:, None], device="cuda")
        all_kl, sensitivity = {}, []
        with torch.inference_mode():
            for split in ["train", "validation", "historical_holdout", "font"]:
                targets = data[split]
                classes = data[split + "_classes"]
                rows = []
                for start in range(0, len(targets), 16):
                    ids = torch.tensor(classes[start : start + 16], device="cuda")
                    target = torch.tensor(targets[start : start + 16, None], device="cuda")
                    _, (qm, ql, pm, pl) = model(reference[ids], target, ids)
                    kl = 0.5 * (pl - ql + (ql.exp() + (qm - pm).square()) / pl.exp() - 1)
                    standardized = (qm - pm) / (pl * 0.5).exp()
                    for i, label in enumerate(ids.tolist()):
                        rows.append(
                            {
                                "class": label,
                                "kl_per_dimension": kl[i].cpu().tolist(),
                                "posterior_mean_in_prior_sigma": standardized[i].cpu().tolist(),
                                "prior_mean": pm[i].cpu().tolist(),
                                "prior_sigma": (pl[i] * 0.5).exp().cpu().tolist(),
                                "posterior_sigma": (ql[i] * 0.5).exp().cpu().tolist(),
                            }
                        )
                all_kl[split] = rows
            for label in range(len(reference)):
                # +/- one conditional-prior sigma, one dimension at a time.
                noises = torch.cat([torch.zeros(1, 8), torch.eye(8), -torch.eye(8)]).cuda()
                ids = torch.full((17,), label, device="cuda", dtype=torch.long)
                ink = ink_from_sdf(model.generate(reference[ids], ids, noises))[:, 0]
                center = ink[0]
                difference = (ink[1:] - center).abs()
                union = ((ink[1:] > 0.5) | (center > 0.5)).sum((1, 2)).clamp_min(1)
                xor = ((ink[1:] > 0.5) != (center > 0.5)).sum((1, 2)) / union
                sensitivity.append(
                    {
                        "label": state["labels"][label],
                        "mean_absolute_response": difference.mean((1, 2)).cpu().tolist(),
                        "ink_xor_response": xor.cpu().tolist(),
                        "all_directions_pixel_identical": bool((difference == 0).all()),
                    }
                )
        result[name] = {"distribution_samples": all_kl, "latent_sensitivity": sensitivity}
        del model
    atomic_json(Path(request["run"]) / "reports/latent-diagnostics.json", result)
    atomic_json(
        Path(request["run"]) / "reports/device.json",
        {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "optimizer_updates": 0,
        },
    )


def diagnose(delivery: Path) -> Path:
    verify_run(delivery)
    frozen = json.loads((delivery / "reports/candidates.json").read_text())
    data = np.load(Path(frozen["data_version"]) / "arrays.npz", allow_pickle=False)
    labels = json.loads((Path(frozen["data_version"]) / "manifest.json").read_text())["labels"]
    package = Path("workspace/runs/20261118T175430545693Z-multichar-holdout-v2-seal-620a64f9")
    verify_run(package)
    held = json.loads((package / "reports/proposals.json").read_text())
    images = [
        align_real(Path(r["crop_path"]), data["reference"][labels.index(r["candidate_label"])])[0]
        for r in held
    ]
    real = data["domains"] == 1
    font_mask = data["domains"] == 0
    arrays = {
        "reference": data["reference"],
        "train": data["target"][real],
        "train_classes": data["classes"][real],
        "validation": data["validation"],
        "validation_classes": data["val_classes"],
        "font": data["target"][font_mask],
        "font_classes": data["classes"][font_mask],
        "historical_holdout": np.stack(images),
        "historical_holdout_classes": np.array([labels.index(r["candidate_label"]) for r in held]),
    }
    run = create(
        "context-frozen-diagnosis",
        seed=1010,
        config={
            "models": frozen["candidates"],
            "historical_holdout": str(package.resolve()),
            "fresh_holdout": False,
            "training": False,
        },
    )
    try:
        np.savez_compressed(run.path / "reports/inputs.npz", **arrays)
        request = {
            "run": str(run.path),
            "arrays": str(run.path / "reports/inputs.npz"),
            "models": frozen["candidates"],
        }
        atomic_json(run.path / "reports/request.json", request)
        with (run.path / "logs/diagnostic.log").open("w") as stream:
            subprocess.run(
                [
                    str(Path("workspace/environments/corpus-generator-v09/bin/python").absolute()),
                    "-m",
                    "research.corpus.context_diagnosis",
                    "worker",
                    str(run.path / "reports/request.json"),
                ],
                check=True,
                timeout=900,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        results = {}
        for name, row in frozen["candidates"].items():
            sample = Path(row["sample_run"])
            verify_run(sample)
            generated = np.load(sample / "reports/generated.npz", allow_pickle=False)["ink"]
            train_run = Path(row["checkpoint"]).parent.parent
            old_data = Path(json.loads((train_run / "reports/request.json").read_text())["data"])
            own = np.load(old_data / "arrays.npz", allow_pickle=False)
            per_class = []
            for i, label in enumerate(labels):
                own_real = own["target"][(own["domains"] == 1) & (own["classes"] == i)]
                font = arrays["font"][arrays["font_classes"] == i]
                active = generated[i] > 0.1
                per_class.append(
                    {
                        "label": label,
                        "own_real_train_count": len(own_real),
                        "nearest_font": nearest(generated[i], font),
                        "nearest_own_real_train": nearest(generated[i], own_real),
                        "soft_edge_fraction": float(
                            ((generated[i] > 0.1) & (generated[i] < 0.9)).sum()
                            / max(1, active.sum())
                        ),
                        **character_metrics(
                            generated[i], arrays["font"], arrays["font_classes"], i
                        ),
                    }
                )
            coverage = {}
            for split in ["train", "validation", "historical_holdout"]:
                distances = [
                    float(shape_distance(generated[c], im).min())
                    for im, c in zip(arrays[split], arrays[split + "_classes"], strict=True)
                ]
                coverage[split] = {
                    "count": len(distances),
                    "mean_best_of_8_shape": float(np.mean(distances)),
                    "distances": distances,
                }
            results[name] = {
                "per_class": per_class,
                "same_final_data_coverage": coverage,
                "limitation": "earlier models did not train on all final Train rows",
            }
        atomic_json(run.path / "reports/image-diagnostics.json", results)
        run.finish(
            "held",
            result={
                "models": 4,
                "optimizer_updates": 0,
                "new_holdout": False,
                "quality_acceptance": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["worker", "diagnose"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.path)
    else:
        print(diagnose(args.path))
