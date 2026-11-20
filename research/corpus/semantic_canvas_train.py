"""Isolated finite-budget canvas content capacity training, not a personal release."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    verify_run(data)
    cfg = {
        "data": str(data.resolve()),
        "data_sha256": digest(data / "run.json"),
        "archives": {
            n: digest(data / f"reports/{n}.npz")
            for n in ["font-glyphs", "font-words", "evaluation"]
        },
        "steps": 2000,
        "seed": 4020,
        "learning_rate": 0.001,
        "glyph_batch": 16,
        "word_batch": 4,
        "real_training_samples": 0,
        "content_only_font_auxiliary": True,
        "inference_reference_image": False,
        "architecture": "semantic-canvas-ink/1",
        "C_used": False,
        "protocol_sha256": digest("docs/V01020_PROTOCOL.md"),
    }
    run = create("semantic-canvas-content-training", seed=cfg["seed"], config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01020-content-model").write_text(str(run.path))
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


def worker(path):
    import importlib.metadata
    import time

    import numpy as np
    import torch
    from PIL import Image
    from torch.nn import functional as F

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.semantic_canvas_model import SemanticCanvasInk

    cfg = json.loads(Path(path).read_text())
    run, data = Path(cfg["run"]), Path(cfg["data"])
    assert digest(data / "run.json") == cfg["data_sha256"]
    setup(cfg["seed"])
    arrays = {}
    for name, hash_ in cfg["archives"].items():
        assert digest(data / f"reports/{name}.npz") == hash_
        with np.load(data / f"reports/{name}.npz") as archive:
            arrays[name] = {k: torch.tensor(archive[k], device="cuda") for k in archive.files}
    model = SemanticCanvasInk().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    rng = np.random.default_rng(cfg["seed"])
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save({"model": initial}, run / "logs/initial-state.pt")
    write(
        run / "reports/dependencies.json",
        {n: importlib.metadata.version(n) for n in ["torch", "numpy", "pillow"]},
    )
    curve = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, cfg["steps"] + 1):
        name = "font-glyphs" if step % 2 else "font-words"
        a = arrays[name]
        batch = cfg["glyph_batch"] if step % 2 else cfg["word_batch"]
        ids = rng.choice(len(a["ink"]), batch)
        target = a["ink"][ids]
        height, width = target.shape[-2:]
        logits = model(a["tokens"][ids], a["boxes"][ids], height, width)
        loss = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (1 + 3 * target)
        ).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite canvas content loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            q = {
                "step": step,
                "loss": float(loss.detach()),
                "batch_source": name,
                "seconds": time.perf_counter() - start,
            }
            curve.append(q)
            write(run / "reports/curve.json", curve)
            print(json.dumps(q), flush=True)
        if step % 500 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": cfg,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                run / f"logs/checkpoint-{step:05d}.pt",
            )
    model.eval()
    outputs = []
    for name in ["font-glyphs", "evaluation"]:
        a = arrays[name]
        images = []
        for i in range(len(a["ink"])):
            h, w = a["ink"].shape[-2:]
            im = model.image(a["tokens"][i : i + 1], a["boxes"][i : i + 1], h, w)[0].cpu().numpy()
            dest = run / f"rendered/pages/{name}-{i:03d}.png"
            Image.fromarray(np.uint8((1 - im) * 255)).save(dest)
            images.append(im)
            outputs.append(
                {
                    "group": name,
                    "index": i,
                    "image": str(dest),
                    "sha256": digest(dest),
                    "reference_pixels_input": False,
                    "source_type": "neural-font-auxiliary-capacity",
                    "real_scan": False,
                }
            )
        np.savez_compressed(run / f"reports/{name}-outputs.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    write(
        run / "reports/result.json",
        {
            "new_model_updates": cfg["steps"],
            "actual_parameter_training": True,
            "parameter_l1_change": sum(
                float((v - initial[k]).abs().sum()) for k, v in model.state_dict().items()
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds_training_and_generation": time.perf_counter() - start,
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "outputs": len(outputs),
            "checkpoint_sha256": digest(run / "logs/checkpoint-02000.pt"),
            "real_training_samples": 0,
            "personal_style_training_completed": False,
            "quality_passed": False,
            "C_used": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
