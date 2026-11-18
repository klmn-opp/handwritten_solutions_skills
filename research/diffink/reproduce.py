"""Run pinned upstream inference with genuine weights and one real online sample.

This isolated Python 3.10 worker intentionally does not import the CPU Runtime.
No random-weight fallback and no generated output counts as independently validated.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = Path(request["run"])
    report = {
        "status": "failed",
        "model_executed": False,
        "mock": False,
        "independent_transcription": None,
        "style_score": None,
        "personal_style": False,
    }
    try:
        checkout = Path(request["checkout"])
        for relative, expected in request["upstream_source_sha256"].items():
            if digest(checkout / relative) != expected:
                raise ValueError("Upstream source changed: " + relative)
        for asset in request["assets"].values():
            path = Path(asset["path"])
            if not path.is_file() or digest(path) != asset["sha256"]:
                raise ValueError("Missing or altered official asset: " + str(path))
        sys.path.insert(0, str(checkout))
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["MPLCONFIGDIR"] = str(output / "logs/matplotlib-cache")
        import numpy as np
        import torch
        import trainer.dit_trainer as trainer
        import yaml
        from dataset.vae_dataset import ValDataset
        from model.dit import DiT
        from model.vae import VAE
        from torch.utils.data import DataLoader, Subset
        from utils.utils import ModelConfig, set_seed

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU substitution is not a CUDA reproduction")
        set_seed(request["seed"])
        torch.backends.cudnn.benchmark = False
        props = torch.cuda.get_device_properties(0)
        report["hardware"] = {
            "gpu": props.name,
            "vram_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "arch_list": torch.cuda.get_arch_list(),
        }
        assets = request["assets"]
        config = yaml.safe_load((checkout / "configs/dit_val_config.yaml").read_text())
        config.update(
            val_file=assets["validation"]["path"],
            text_file=assets["vocabulary"]["path"],
            writer_file=assets["writers"]["path"],
            val_batch_size=1,
            output_base=str(output / "comparisons/official"),
        )
        vocab = json.loads(Path(config["text_file"]).read_text())
        config["num_text_embedding"] = len(vocab) + 1
        dataset = ValDataset(config["val_file"], config["text_file"], config["writer_file"])
        chosen = None
        # Reproduction uses one short real validation line; no invented input trajectories.
        candidates = sorted(zip(dataset.lengths, dataset.keys, strict=True))
        for length, key in candidates:
            text = dataset.hf[key]["line_text"][()].decode("utf-8")
            if 300 <= length <= 800 and 8 <= len(text) <= 24 and all(c in vocab for c in text):
                original_index = dataset.keys.index(key)
                chosen = list(dataset.sorted_indices).index(original_index)
                break
        if chosen is None:
            raise ValueError("No suitable real validation sample found")
        writer_id, points, target_text, character_ends = dataset[chosen]
        write(
            output / "solutions/reference.json",
            {
                "origin": "dataset-recorded-online",
                "personal_collected": False,
                "source_sha256": assets["validation"]["sha256"],
                "dataset_key": key,
                "writer": writer_id,
                "text": target_text,
                "points": points.tolist(),
                "character_end_indices": character_ends.tolist(),
                "license": "CASIA academic research terms",
            },
        )
        write(
            output / "solutions/target.json",
            {
                "text": target_text,
                "official_conditioning_extra_token": "、",
                "seed": request["seed"],
            },
        )
        loader = DataLoader(
            Subset(dataset, [chosen]),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=ValDataset.collate_fn,
        )
        # Safe tensor loading; original upstream can silently use random weights when paths fail.
        vae_checkpoint = torch.load(
            assets["vae"]["path"], map_location="cpu", weights_only=True, mmap=True
        )
        vae_state = {
            k.removeprefix("module."): v for k, v in vae_checkpoint["model_state_dict"].items()
        }
        config["num_writer"] = int(vae_state["style_classifier.mlp.8.weight"].shape[0])
        vae = VAE(ModelConfig(config))
        vae.load_state_dict(vae_state, strict=True)
        vae = vae.to("cuda")
        report["vae_loaded_tensors"] = len(vae_state)
        del vae_state, vae_checkpoint
        gc.collect()
        dit_checkpoint = torch.load(
            assets["dit"]["path"], map_location="cpu", weights_only=True, mmap=True
        )
        dit_state = {
            k.removeprefix("module."): v for k, v in dit_checkpoint["model_state_dict"].items()
        }
        dit = DiT(ModelConfig(config))
        dit.load_state_dict(dit_state, strict=True)
        dit = dit.to("cuda")
        report["dit_loaded_tensors"] = len(dit_state)
        del dit_state, dit_checkpoint
        gc.collect()
        write(output / "reports/effective-config.json", config)
        # Instrument the actual upstream sampler to retain its online stroke output unchanged.
        original_sampler = trainer.sample_from_params
        captures = []

        def capture(*positional, **keywords):
            sequence = original_sampler(*positional, **keywords)
            array = (
                sequence.detach().cpu().numpy()
                if hasattr(sequence, "detach")
                else np.asarray(sequence)
            )
            if not np.isfinite(array).all() or array.ndim != 2 or array.shape[1] != 5:
                raise ValueError("Invalid generated online sequence")
            np.save(output / "solutions/generated-strokes.npy", array)
            captures.append(
                {
                    "points": int(len(array)),
                    "sha256": digest(output / "solutions/generated-strokes.npy"),
                }
            )
            return sequence

        trainer.sample_from_params = capture
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.infer_diffink(dit, vae, loader, torch.device("cuda"), config["output_base"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if len(captures) != 1:
            raise ValueError("Expected one genuine generated sequence")
        from PIL import Image

        preview = output / "comparisons/official/recon_0.png"
        with Image.open(preview) as image:
            image.verify()
        with Image.open(preview) as image:
            rgb = image.convert("RGB")
            rgb.save(output / "rendered/pages/01.png")
            rgb.save(output / "rendered/homework_solutions.pdf")
        report.update(
            status="generated-unvalidated",
            model_executed=True,
            inference_seconds=elapsed,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            target_text=target_text,
            writer=writer_id,
            dataset_key=key,
            input_points=len(points),
            captures=captures,
            output_sha256=digest(preview),
            preserved_prefix_ratio=0.3,
            sampling_steps=20,
            cfg_scale=1.0,
            inference_scope="official Chinese continuation; not personal English generation",
            patches=[
                "batch 1, single short sample, workers 0",
                "checkpoint paths pinned; safe strict CPU loading then CUDA",
                "auxiliary writer dimension inferred from checkpoint",
                "sampler instrumentation saves unchanged online sequence",
            ],
            upstream_algorithm_modified=False,
        )
        dataset.hf.close()
    except Exception as exc:
        report.update(
            error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc()
        )
    write(output / "reports/inference.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "traceback"}, ensure_ascii=False),
        flush=True,
    )
    return 0 if report["model_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
