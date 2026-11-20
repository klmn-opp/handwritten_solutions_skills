"""Finite isolated capacity study on already verified real local images."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    cfg = {
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "arrays_sha256": digest(data / "reports/glyphs.npz"),
        "architecture": "canonical-conditioned-ink/1",
        "seed": 4210,
        "steps": 1000,
        "learning_rate": 0.001,
        "classes": [102, 116],
        "expected_samples": 18,
        "sigma": 0.15,
        "initialization": "fixed random small decoder, no pretrained weights",
        "no_image_input": True,
        "font_auxiliary": False,
        "C_used": False,
        "protocol_sha256": digest("docs/V01021_PROTOCOL.md"),
    }
    run = create("canonical-ink-capacity", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01021-ft-model").write_text(str(run.path))
    print(run.path, flush=True)
    try:
        with (run.path / "logs/worker.log").open("w") as log:
            subprocess.run(
                [
                    "workspace/environments/english-htr-v08/bin/python",
                    "-m",
                    __name__,
                    str(run.path / "reports/request.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request):
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_model import CanonicalInk

    cfg = json.loads(Path(request).read_text())
    run, data = Path(cfg["run"]), Path(cfg["data"])
    assert digest(data / "run.json") == cfg["data_sha256"]
    assert digest(data / "reports/glyphs.npz") == cfg["arrays_sha256"]
    setup(cfg["seed"])
    with np.load(data / "reports/glyphs.npz") as a:
        ids = np.flatnonzero(np.isin(a["labels"], cfg["classes"]))
        assert len(ids) == cfg["expected_samples"]
        target = torch.tensor(a["ink"][ids], device="cuda")
        labels = torch.tensor(a["labels"][ids], device="cuda").long()
        codes = torch.tensor(a["local_codes"][ids], device="cuda")
    rows = json.loads((data / "reports/glyph-samples.json").read_text())
    write(run / "reports/samples.json", [{**rows[i], "source_index": int(i)} for i in ids])
    by_class = {c: torch.where(labels == c)[0] for c in cfg["classes"]}
    std = {c: codes[ii].std(0, unbiased=False) for c, ii in by_class.items()}
    scales = torch.stack([std[int(c)] for c in labels])
    model = CanonicalInk().cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    curve = []
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        noise = torch.tensor(rng.standard_normal(codes.shape), device="cuda").float()
        local = codes + cfg["sigma"] * scales * noise
        logits = model(labels, local)
        loss = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * target)
        ).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite local loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "seconds": time.perf_counter() - start,
            }
            curve.append(row)
            write(run / "reports/curve.json", curve)
            print(json.dumps(row), flush=True)
        if step % 250 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "step": step,
                    "config": cfg,
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs, images = [], []

    def emit(label, local, *, source_index=None, seed=None, selected_center=None):
        image = model.image(torch.tensor([label], device="cuda"), local[None])[0].cpu().numpy()
        dest = run / f"rendered/pages/ink-{len(images):03d}.png"
        Image.fromarray(np.uint8((1 - image) * 255)).save(dest)
        images.append(image)
        outputs.append(
            {
                "image": str(dest),
                "sha256": digest(dest),
                "label": chr(label),
                "style": local.tolist(),
                "source_index": source_index,
                "seed": seed,
                "selected_center": selected_center,
                "source_type": "learned-offline-ink",
                "personally_collected": False,
                "reference_image_input": False,
            }
        )

    for j, i in enumerate(ids):
        emit(int(labels[j]), codes[j], source_index=int(i))
    for seed in range(2012, 2020):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for c, ii in by_class.items():
            k = int(torch.randint(len(ii), (1,), device="cuda", generator=generator))
            local = codes[ii[k]] + cfg["sigma"] * std[c] * torch.randn(
                8, device="cuda", generator=generator
            )
            emit(c, local, seed=seed, selected_center=int(ids[int(ii[k])]))
    np.savez_compressed(run / "reports/outputs.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(
        run / "reports/dependencies.json",
        {k: importlib.metadata.version(k) for k in ["torch", "numpy", "pillow"]},
    )
    write(
        run / "reports/result.json",
        {
            "actual_parameter_training": True,
            "new_model_updates": cfg["steps"],
            "model_lineage_updates": cfg["steps"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "checkpoint_sha256": digest(run / "logs/checkpoint-01000.pt"),
            "outputs": len(outputs),
            "quality_passed": False,
            "independent_generalization_proven": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
