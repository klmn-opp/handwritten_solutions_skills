"""Real, offline TrOCR handwriting recognition. Request contains no target text."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--request", type=Path, required=True)
    req = json.loads(p.parse_args().request.read_text())
    if any(set(row) - {"id", "image", "sha256"} for row in req["images"]):
        raise ValueError("Recognizer must not receive expected text")
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    processor = TrOCRProcessor.from_pretrained(req["checkpoint"], local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(req["checkpoint"], local_files_only=True)
    model = model.to("cuda").eval()
    rows = []
    for row in req["images"]:
        if hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Image digest mismatch")
        with Image.open(row["image"]) as image:
            pixels = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(
                "cuda"
            )
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            tokens = model.generate(pixels, do_sample=False, num_beams=4, max_new_tokens=256)
        torch.cuda.synchronize()
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        rows.append(
            {
                "id": row["id"],
                "text": text,
                "seconds": time.perf_counter() - started,
                "tokens": tokens[0].tolist(),
                "reached_token_limit": tokens.shape[1] >= 257,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        Path(req["output"]).write_text(
            json.dumps(
                {
                    "model_executed": True,
                    "rows": rows,
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "decoder": {"num_beams": 4, "max_new_tokens": 256, "do_sample": False},
                    "dependencies": {
                        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                    },
                },
                indent=2,
            )
        )
        print(row["id"], flush=True)


if __name__ == "__main__":
    main()
