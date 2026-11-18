"""Bounded real DiffusionPen inference using unmodified official model definitions."""

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
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    req = json.loads(p.parse_args().request.read_text())
    root = Path(req["run"])
    rows = []
    report = {"model_executed": False, "cases": rows}
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from PIL import Image, ImageChops, ImageOps
        from torchvision import transforms
        from transformers import CanineConfig, CanineModel, CanineTokenizer

        importlib.import_module("safetensors.torch")
        sys.path.insert(0, req["checkout"])
        from feature_extractor import ImageEncoder
        from unet import UNetModel

        if not torch.cuda.is_available():
            raise ValueError("CUDA required")
        for path, expected in req["file_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Asset/source changed: " + path)
        torch.manual_seed(42)
        torch.backends.cudnn.benchmark = False
        cfg = CanineConfig.from_pretrained(req["canine"], local_files_only=True)
        encoder = CanineModel(cfg)
        model = UNetModel(
            image_size=(64, 256),
            in_channels=4,
            model_channels=320,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=4,
            num_classes=339,
            context_dim=320,
            vocab_size=79,
            text_encoder=encoder,
            args=SimpleNamespace(interpolation=False, mix_rate=0),
        )
        state = torch.load(req["checkpoint"], map_location="cpu", weights_only=True, mmap=True)
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean, strict=True)
        del state, clean
        model = model.to("cuda").eval().requires_grad_(False)
        style = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=False)
        state = torch.load(
            req["style_checkpoint"], map_location="cpu", weights_only=True, mmap=True
        )
        ignored = {k: list(v.shape) for k, v in state.items() if k.startswith("classifier.")}
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        style.load_state_dict(state, strict=True)
        del state
        style = style.to("cuda").eval().requires_grad_(False)
        vae = AutoencoderKL.from_pretrained(req["vae"], local_files_only=True).to("cuda").eval()
        scheduler = DDIMScheduler.from_pretrained(req["scheduler"], local_files_only=True)
        tokenizer = CanineTokenizer.from_pretrained(req["canine"], local_files_only=True)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        )
        refs = []
        for i, r in enumerate(req["references"]):
            with Image.open(r["image"]) as im:
                im = im.convert("RGB")
                im = im.resize((round(im.width * 64 / im.height), 64))
                if im.width > 256:
                    im.thumbnail((256, 64))
                im = ImageOps.pad(im, (256, 64), color="white")
                im.save(root / "comparisons" / f"style-{i}.png")
                refs.append(transform(im))
        with torch.inference_mode():
            features = style(torch.stack(refs).to("cuda"))
        report.update(
            strict_unet_load=True,
            strict_style_load=True,
            ignored_classifier=ignored,
            pretrained_canine_loaded_from_complete_checkpoint=True,
            source_commit=req["upstream_commit"],
            hardware={
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            dependencies={
                d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
            },
        )
        for case in req["cases"]:
            for seed in req["seeds"]:
                row = {
                    **case,
                    "seed": seed,
                    "sample_id": f"{case['id']}-s{seed}",
                    "backend": "diffusionpen",
                    "status": "failed",
                    "words": [],
                }
                try:
                    words = case["text"].split()
                    if max(map(len, words)) > 38:
                        raise ValueError(
                            "Word exceeds official max_length40 including two special tokens"
                        )
                    torch.manual_seed(seed)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    crops = []
                    for index, word in enumerate(words):
                        tokens = tokenizer(
                            [word],
                            padding="max_length",
                            truncation=False,
                            return_tensors="pt",
                            max_length=40,
                        ).to("cuda")
                        assert tokens.input_ids.shape[1] == 40
                        assert int(tokens.attention_mask.sum()) == len(word) + 2
                        with torch.inference_mode():
                            x = torch.randn((1, 4, 8, 32), device="cuda")
                            scheduler.set_timesteps(50)
                            for t in scheduler.timesteps:
                                residual = model(
                                    x,
                                    torch.full((1,), int(t), device="cuda", dtype=torch.long),
                                    tokens,
                                    style_extractor=features,
                                )
                                x = scheduler.step(residual, t, x).prev_sample
                            pixels = (vae.decode(x / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        array = (
                            (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
                        )
                        im = Image.fromarray(array)
                        path = root / "comparisons" / f"{row['sample_id']}-word{index:02d}.png"
                        im.save(path)
                        bbox = (
                            ImageChops.invert(im.convert("L"))
                            .point(lambda x: 255 if x >= 55 else 0)
                            .getbbox()
                        )
                        # Crop horizontal whitespace; retain vertical canvas and descenders.
                        crop = im.crop((bbox[0], 0, bbox[2], 64)) if bbox else im
                        crops.append(crop)
                        row["words"].append(
                            {
                                "text": word,
                                "image": str(path.relative_to(root)),
                                "sha256": digest(path),
                                "token_ids": tokens.input_ids[0].tolist(),
                                "ink_bbox": bbox,
                                "source_type": "model-generated-including-punctuation",
                            }
                        )
                    line = Image.new(
                        "RGB", (sum(i.width for i in crops) + 16 * (len(crops) - 1), 64), "white"
                    )
                    left = 0
                    for im in crops:
                        line.paste(im, (left, 0))
                        left += im.width + 16
                    path = root / "rendered/pages" / f"{row['sample_id']}.png"
                    line.save(path)
                    torch.cuda.synchronize()
                    row.update(
                        status="generated",
                        image=str(path.relative_to(root)),
                        image_sha256=digest(path),
                        elapsed_seconds=time.perf_counter() - start,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        composition="64x256 words; horizontal whitespace crop; fixed16px spaces",
                        input_complete=True,
                        source_type="model-generated-word-composition",
                    )
                    report["model_executed"] = True
                except Exception as exc:
                    row.update(error=str(exc), traceback=traceback.format_exc())
                rows.append(row)
                (root / "reports/lines.json").write_text(json.dumps(report, indent=2))
                print(row["sample_id"], row["status"], flush=True)
        report["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
    (root / "reports/lines.json").write_text(json.dumps(report, indent=2))
    return 0 if report["model_executed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
