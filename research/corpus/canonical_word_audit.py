"""Zero-update audit of intact real words against a frozen local generator."""

import json
import subprocess
from pathlib import Path

from research.corpus.context_content_train import digest, write


def launch(model: Path, data: Path) -> Path:
    from hwsol.workspace import verify_run
    from research.corpus.experiment import create

    for p in [model, data]:
        verify_run(p)
    previous = json.loads((model / "reports/request.json").read_text())
    if not previous.get("component_balanced"):
        raise ValueError("Registered frozen local backbone required")
    cp = model / f"logs/checkpoint-{previous['steps']:05d}.pt"
    cfg = {
        "model": str(model),
        "model_sha256": digest(model / "run.json"),
        "checkpoint": str(cp),
        "checkpoint_sha256": digest(cp),
        "data": str(data),
        "data_sha256": digest(data / "run.json"),
        "word_arrays_sha256": digest(data / "reports/words.npz"),
        "new_neural_updates": 0,
        "known_train_conditions": True,
        "input_reference_image": False,
        "C_used": False,
        "protocol_sha256": digest("docs/V01022_PROTOCOL.md"),
    }
    run = create("canonical-intact-word-audit", seed=4222, config=cfg)
    write(run.path / "reports/request.json", {**cfg, "run": str(run.path)})
    Path("/tmp/context-v01022-word-audit").write_text(str(run.path))
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
                timeout=300,
            )
        run.finish("held", result=json.loads((run.path / "reports/result.json").read_text()))
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


def worker(request):
    import shutil

    import numpy as np
    import torch
    from PIL import Image

    from research.corpus.aligned_style_experiment import setup
    from research.corpus.canonical_ink_composer import compose
    from research.corpus.canonical_ink_model import CanonicalInk

    cfg = json.loads(Path(request).read_text())
    run, data = Path(cfg["run"]), Path(cfg["data"])
    for key in ["model", "data"]:
        assert digest(Path(cfg[key]) / "run.json") == cfg[key + "_sha256"]
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    assert digest(data / "reports/words.npz") == cfg["word_arrays_sha256"]
    setup(4222)
    saved = torch.load(cfg["checkpoint"], map_location="cuda", weights_only=False)
    model = CanonicalInk().cuda().eval()
    model.load_state_dict(saved["model"])
    with np.load(data / "reports/words.npz") as source:
        a = {k: source[k] for k in source.files}
    rows = json.loads((data / "reports/word-samples.json").read_text())
    assert len(rows) == len(a["ink"]) == 18
    shutil.copyfile(data / "reports/word-samples.json", run / "reports/samples.json")
    shutil.copyfile(data / "reports/words.npz", run / "solutions/original-words.npz")
    outputs, images = [], []
    for i, row in enumerate(rows):
        assert row["split"] == "train" and row["source_hash"] != (
            "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
        )
        t = torch.tensor(a["tokens"][i], device="cuda")
        b = torch.tensor(a["boxes"][i], device="cuda")
        style = torch.tensor(a["style"][i], device="cuda")
        im = compose(model, t, b, style, 128, 512).cpu().numpy()
        images.append(im)
        target = a["ink"][i]
        target_path, output_path = (
            run / f"solutions/real-{i:02d}.png",
            run / f"rendered/pages/local-{i:02d}.png",
        )
        Image.fromarray(np.uint8((1 - target) * 255)).save(target_path)
        Image.fromarray(np.uint8((1 - im) * 255)).save(output_path)
        tt, pp = target > 0.5, im > 0.5
        outputs.append(
            {
                "index": i,
                "text": row["text"],
                "original_target": str(target_path),
                "target_sha256": digest(target_path),
                "image": str(output_path),
                "sha256": digest(output_path),
                "dice": float(2 * (tt & pp).sum() / max(1, tt.sum() + pp.sum())),
                "mae": float(abs(im - target).mean()),
                "target_foreground_p90": float(np.quantile(target[target > 0.1], 0.9)),
                "generated_foreground_p90": float(np.quantile(im[im > 0.1], 0.9)),
                "false_background_ink_mean": float(im[target == 0].mean())
                if (target == 0).any()
                else None,
                "known_local_positions": len(row["local_conditions_from_exact_source"]),
                "semantic_characters": len(row["text"]),
                "known_train_reconstruction_not_generalization": True,
            }
        )
    np.savez_compressed(run / "reports/reconstructions.npz", ink=np.stack(images))
    write(run / "reports/outputs.json", outputs)
    assert digest(cfg["checkpoint"]) == cfg["checkpoint_sha256"]
    write(
        run / "reports/result.json",
        {
            "new_neural_updates": 0,
            "real_intact_words": 18,
            "target_pixels_changed": False,
            "mean_dice": float(np.mean([q["dice"] for q in outputs])),
            "input_reference_image": False,
            "C_used": False,
            "training_gate_passed": False,
        },
    )


if __name__ == "__main__":
    import sys

    worker(sys.argv[1])
