"""Real DiffBrush weights through pinned Handnote, in an isolated CUDA environment."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import resource
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    request = json.loads(parser.parse_args().request.read_text())
    run = Path(request["run"])
    started = time.perf_counter()
    report = {"status": "failed", "model_executed": False, "mock": False}
    try:
        os.environ["HANDNOTE_CACHE"] = request["cache"]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["MPLCONFIGDIR"] = str(run / "logs/matplotlib")
        import handnote
        import numpy as np
        import torch
        from handnote.linegen import style, weights
        from handnote.linegen.generate import LineGenerator

        # diffusers 0.32.2 accesses this submodule without importing it on this path.
        importlib.import_module("safetensors.torch")

        installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
        (run / "reports/installed-dependencies.json").write_text(json.dumps(installed, indent=2))
        normalized = {name.lower().replace("_", "-"): value for name, value in installed.items()}
        for line in (run / "reports/dependencies.lock").read_text().splitlines():
            if "==" in line and not line.startswith(("#", "--")):
                name, version = line.split("==", 1)
                if normalized.get(name.lower().replace("_", "-")) != version:
                    raise ValueError("Isolated dependency mismatch: " + name)
        for relative, expected in request["handnote_files"].items():
            if digest(Path(handnote.__file__).parent / relative) != expected:
                raise ValueError("Pinned Handnote source differs: " + relative)
        for name, (_, size, expected) in weights.FILES.items():
            path = weights.cache_dir() / name
            if path.stat().st_size != size or digest(path) != expected:
                raise ValueError("Model asset differs: " + name)
        if digest(weights.cache_dir() / "content.npz") != request["content_sha256"]:
            raise ValueError("Content glyph asset differs")
        reference = Path(request["reference"])
        if digest(reference) != request["reference_sha256"]:
            raise ValueError("Personal reference changed")
        if not torch.cuda.is_available():
            raise ValueError("CUDA required; no silent CPU substitution")
        torch.manual_seed(request["seed"])
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        gen = LineGenerator(device="cuda", steps=request["steps"], half=False)
        torch.cuda.synchronize()
        report["model_load_seconds"] = time.perf_counter() - load_started
        ref = style.load_style(reference)
        image = gen.write(request["text"], ref, seed=request["seed"], trim=False)
        torch.cuda.synchronize()
        # Preserve full 64x1024 canvas; provide the normal upstream trimmed view separately.
        image.save(run / "rendered/pages/01.png")
        image.convert("RGB").save(run / "rendered/homework_solutions.pdf")
        from PIL import Image

        Image.fromarray(style.trim(np.asarray(image))).save(run / "comparisons/trimmed.png")
        report.update(
            status="generated-unvalidated",
            model_executed=True,
            device=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            output_sha256=digest(run / "rendered/pages/01.png"),
            timings=gen.timings,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            width=image.width,
            height=image.height,
            target=request["text"],
            seed=request["seed"],
            online_strokes_available=False,
            independent_content_score=None,
            human_style_score=None,
            source_type="model-generated",
            personally_collected=False,
        )
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    report.update(
        worker_wall_seconds=time.perf_counter() - started,
        peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    (run / "reports/inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["model_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
