"""Isolated CUDA optimization worker; admission is performed by the CPU coordinator."""

import json
import random
import sys
import time
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256


def admitted_request(request: dict) -> tuple[list[dict], dict]:
    from research.corpus.train import validate_dataset

    rows, gate = validate_dataset(Path(request["dataset"]))
    if gate["decision"] != "GO-limited-prototype":
        raise ValueError("Independent worker refuses a dataset without verified training coverage")
    if rows != request["rows"] or gate != request["gate"]:
        raise ValueError("Training request no longer matches the versioned dataset")
    if not 1 <= request["epochs"] <= 500:
        raise ValueError("Invalid epoch budget")
    return rows, gate


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    dataset = Path(request["dataset"])
    run_path = Path(request["run_path"])
    seed, epochs = request["seed"], request["epochs"]
    rows, gate = admitted_request(request)
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, TensorDataset

    from research.corpus.generator import GlyphCVAE, objective

    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA training required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    labels = sorted(gate["eligible_labels"])
    train_rows = [
        r
        for r in rows
        if r["dataset_split"] == "train"
        and (r.get("verified_text") or r.get("verified_latex")) in labels
    ]
    val_rows = [
        r
        for r in rows
        if r["dataset_split"] == "validation"
        and (r.get("verified_text") or r.get("verified_latex")) in labels
    ]

    def tensors(records):
        images, classes = [], []
        for row in records:
            with Image.open(row["image_crop"]) as im:
                image = ImageOps.pad(im.convert("L"), (64, 64), color=255)
                images.append((1 - np.asarray(image, dtype=np.float32) / 255)[None])
            classes.append(labels.index(row.get("verified_text") or row["verified_latex"]))
        return TensorDataset(torch.tensor(np.array(images)), torch.tensor(classes))

    loader = DataLoader(
        tensors(train_rows),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation = DataLoader(tensors(val_rows), batch_size=32)
    model = GlyphCVAE(len(labels)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    torch.save(model.state_dict(), run_path / "logs/initial-state.pt")
    initial_sha = sha256(run_path / "logs/initial-state.pt")
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    atomic_json(
        run_path / "reports/architecture.json",
        {
            **model.spec,
            "parameters": sum(p.numel() for p in model.parameters()),
            "labels": labels,
            "context": "zeros until verified context measurements exist",
        },
    )
    start = time.perf_counter()
    curves, best = [], float("inf")
    for epoch in range(epochs):
        stats = {}
        for name, batches, training in [
            ("train", loader, True),
            ("validation", validation, False),
        ]:
            model.train(training)
            losses = []
            with torch.set_grad_enabled(training):
                for ink, label in batches:
                    ink, label = ink.cuda(), label.cuda()
                    pred, mu, logvar = model(
                        ink,
                        label,
                        torch.zeros_like(label),
                        torch.zeros((len(label), 4), device="cuda"),
                    )
                    loss, _ = objective(pred, ink, mu, logvar)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss; do not save a valid candidate")
                    if training:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    losses.append(loss.item())
            stats[name] = sum(losses) / len(losses)
        curves.append({"epoch": epoch + 1, **stats})
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": model.spec,
            "labels": labels,
            "seed": seed,
            "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        torch.save(state, run_path / "logs/last-checkpoint.pt")
        if stats["validation"] < best:
            best = stats["validation"]
            torch.save(state, run_path / "logs/best-checkpoint.pt")
        atomic_json(run_path / "reports/curves.json", curves)
    atomic_json(
        run_path / "reports/training.json",
        {
            "training_executed": True,
            "optimizer_updates": epochs * len(loader),
            "parameter_absolute_change": sum(
                (a - p).abs().sum().item()
                for a, p in zip(initial_parameters, model.parameters(), strict=True)
            ),
            "initial_sha256": initial_sha,
            "checkpoint_sha256": sha256(run_path / "logs/best-checkpoint.pt"),
            "seconds": time.perf_counter() - start,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "holdout_used_for_training": False,
        },
    )


if __name__ == "__main__":
    main()
