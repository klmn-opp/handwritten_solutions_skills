"""Optimization unit test on synthetic pixels, explicitly NOT personal generator training."""

import json
import sys
import time
from pathlib import Path

import torch

from hwsol.profiles.io import atomic_json
from research.corpus.generator import GlyphCVAE, objective


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    path = Path(request["run_path"])
    torch.manual_seed(request["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = GlyphCVAE(3).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    target = torch.zeros(4, 1, 64, 64, device="cuda")
    target[:, :, 10:50, 29:34] = 1  # A synthetic test stripe, not a person's character.
    labels = torch.tensor([0, 1, 2, 0], device="cuda")
    before = [p.detach().clone() for p in model.parameters()]
    start = time.perf_counter()
    pred, mu, logvar = model(
        target, labels, torch.zeros_like(labels), torch.zeros(4, 4, device="cuda")
    )
    loss, parts = objective(pred, target, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    difference = sum(
        (a - p).abs().sum().item() for a, p in zip(before, model.parameters(), strict=True)
    )
    assert torch.isfinite(loss) and difference > 0
    with torch.inference_mode():
        a = model.generate(
            labels,
            torch.zeros_like(labels),
            torch.zeros(4, 4, device="cuda"),
            torch.zeros(4, 16, device="cuda"),
        )
        b = model.generate(
            labels,
            torch.zeros_like(labels),
            torch.zeros(4, 4, device="cuda"),
            torch.ones(4, 16, device="cuda"),
        )
    assert a.shape == (4, 1, 64, 64) and not torch.equal(a, b)
    atomic_json(
        path / "reports/optimization-unit.json",
        {
            "unit_test_passed": True,
            "synthetic_fixture": True,
            "personal_training_executed": False,
            "handwriting_quality_evaluated": False,
            "optimizer_updates": 1,
            "parameter_absolute_change": difference,
            "parameters": sum(p.numel() for p in model.parameters()),
            "loss": loss.item(),
            "parts": {k: v.item() for k, v in parts.items()},
            "latent_changes_output": True,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "seconds": time.perf_counter() - start,
        },
    )
    print("CUDA optimization unit test passed; no personal model trained")


if __name__ == "__main__":
    main()
