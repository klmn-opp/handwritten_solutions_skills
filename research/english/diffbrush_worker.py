"""Genuine DiffBrush batch inference and same-input official-code diagnostics."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    req = json.loads(parser.parse_args().request.read_text())
    root = Path(req["run"])
    rows = []
    summary = {"model_executed": False, "cases": rows}
    try:
        os.environ["HANDNOTE_CACHE"] = req["cache"]
        os.environ["HF_HUB_OFFLINE"] = "1"
        import numpy as np
        import torch
        from handnote.linegen import style, weights
        from handnote.linegen.generate import LineGenerator
        from PIL import Image

        importlib.import_module("safetensors.torch")
        for name, expected in req["file_sha256"].items():
            if digest(name) != expected:
                raise ValueError("Source or asset differs: " + name)
        dump(
            root / "reports/dependencies.json",
            {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        )
        if not torch.cuda.is_available():
            raise ValueError("CUDA required; CPU inference is not substituted")
        torch.manual_seed(req["seeds"][0])
        torch.backends.cudnn.benchmark = False
        gen = LineGenerator("cuda", steps=req["steps"], half=False)
        with Image.open(req["reference"]) as image:
            raw_style = np.asarray(image.convert("L"))
        ref = style.prepare_style(raw_style)
        if req["style_mode"] == "fit":
            ratio = min(64 / raw_style.shape[0], 1024 / raw_style.shape[1])
            resized = Image.fromarray(raw_style).resize(
                (round(raw_style.shape[1] * ratio), round(raw_style.shape[0] * ratio)),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("L", (1024, 64), 255)
            canvas.paste(resized, (0, (64 - resized.height) // 2))
            ref = np.asarray(canvas, dtype=np.float32) / 255
        elif req["style_mode"] != "raw":
            raise ValueError("Unknown reference preprocessing")
        Image.fromarray((ref * 255).astype("uint8")).save(
            root / "comparisons/preprocessed-reference.png"
        )
        original_symbols = weights.content_from_unifont(
            (Path(req["checkout"]) / "files/unifont.pickle").read_bytes()
        )
        if not np.array_equal(original_symbols, weights.load_content()):
            raise ValueError("Content glyph mapping differs from pinned official Unifont")
        official = None
        if req.get("official_pairs"):
            sys.path.insert(0, req["checkout"])
            from models.diffusion import Diffusion
            from models.unet import UNetModel

            official = (
                UNetModel(
                    in_channels=4,
                    model_channels=512,
                    out_channels=4,
                    num_res_blocks=1,
                    attention_resolutions=(1, 1),
                    channel_mult=(1, 1),
                    num_heads=4,
                    context_dim=512,
                    nb_classes=496,
                )
                .to("cuda")
                .eval()
            )
            state = torch.load(
                weights.cache_dir() / "diffbrush.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            official.load_state_dict(state, strict=True)
            diffusion = Diffusion(device="cuda")
            del state
        for case in req["cases"]:
            for seed in req["seeds"]:
                row = {
                    **case,
                    "seed": seed,
                    "backend": "diffbrush",
                    "style_mode": req["style_mode"],
                    "sample_id": f"{case['id']}-s{seed}",
                    "status": "failed",
                }
                try:
                    text = style.check_text(case["text"])
                    indices = [weights.LETTERS.index(c) for c in text]
                    assert "".join(weights.LETTERS[i] for i in indices) == case["text"]
                    content = gen._content(text)
                    assert content.shape[1] == len(text)
                    row.update(
                        input_text=text,
                        content_token_indices=indices,
                        content_shape=list(content.shape),
                        content_sha256=hashlib.sha256(content.cpu().numpy().tobytes()).hexdigest(),
                        input_complete=True,
                        in_upstream_length_range=35 <= len(text) <= 61,
                    )
                    torch.cuda.reset_peak_memory_stats()
                    image = gen.write(text, ref, seed=seed, trim=False)
                    name = row["sample_id"] + ".png"
                    path = root / "rendered/pages" / name
                    image.save(path)
                    Image.fromarray(style.trim(np.asarray(image))).save(root / "comparisons" / name)
                    row.update(
                        status="generated",
                        image=str(path.relative_to(root)),
                        image_sha256=digest(path),
                        source_type="model-generated",
                        personally_collected=False,
                        elapsed_seconds=gen.timings["total"],
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        raw_size=list(image.size),
                        adapter_trim_width=style.trim(np.asarray(image)).shape[1],
                    )
                    summary["model_executed"] = True
                    if official is not None:
                        with torch.no_grad():
                            start = time.perf_counter()
                            raw = diffusion.ddim_sample(
                                official,
                                gen.vae,
                                1,
                                gen._noise(seed),
                                gen._style(ref),
                                content,
                                sampling_timesteps=req["steps"],
                                eta=0,
                            )
                            torch.cuda.synchronize()
                            official_seconds = time.perf_counter() - start
                        # Save official floor conversion and Handnote rounding separately.
                        rgb = raw[0].permute(1, 2, 0).cpu().numpy()
                        original = Image.fromarray((rgb * 255).astype("uint8")).convert("L")
                        rounded = Image.fromarray((rgb * 255).round().astype("uint8")).convert("L")
                        original.save(root / "comparisons" / (row["sample_id"] + "-official.png"))
                        rounded.save(
                            root / "comparisons" / (row["sample_id"] + "-official-rounded.png")
                        )
                        delta = np.abs(
                            np.asarray(rounded, dtype=np.int16) - np.asarray(image, dtype=np.int16)
                        )
                        row["official_comparison"] = {
                            "real_model_executed": True,
                            "official_seconds": official_seconds,
                            "same_noise_style_content": True,
                            "rounded_pixel_equal": bool((delta == 0).all()),
                            "maximum_pixel_difference": int(delta.max()),
                            "mean_pixel_difference": float(delta.mean()),
                        }
                except Exception as exc:
                    row.update(error=str(exc), traceback=traceback.format_exc())
                rows.append(row)
                dump(root / "reports/lines.json", summary)
                print(row["sample_id"], row["status"], flush=True)
        summary.update(
            hardware={
                "gpu": torch.cuda.get_device_name(0),
                "cuda": torch.version.cuda,
                "torch": torch.__version__,
            },
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            style_shape=list(ref.shape),
            content_mapping_matches_official=True,
        )
    except Exception as exc:
        summary.update(error=str(exc), traceback=traceback.format_exc())
    dump(root / "reports/lines.json", summary)
    return 0 if summary["model_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
