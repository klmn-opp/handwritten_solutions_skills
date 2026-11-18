"""Isolated genuine recognition. Input contract deliberately excludes target labels."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    req = json.loads(parser.parse_args().request.read_text())
    if any(set(r) - {"id", "image", "sha256"} for r in req["images"]):
        raise ValueError("Recognition must not receive expected labels")
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this recorded pilot")
    torch.manual_seed(req["seed"])
    engine = req["engine"]
    if engine == "trocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(req["checkpoint"], local_files_only=True)
        model = (
            VisionEncoderDecoderModel.from_pretrained(req["checkpoint"], local_files_only=True)
            .to("cuda")
            .eval()
        )

        def predict(image):
            pixels = processor(images=image, return_tensors="pt").pixel_values.to("cuda")
            tokens = model.generate(pixels, do_sample=False, num_beams=4, max_new_tokens=256)
            return processor.batch_decode(tokens, skip_special_tokens=True)[0], tokens[0].tolist()

        config = {"num_beams": 4, "max_new_tokens": 256, "do_sample": False}
    elif engine == "unimernet":
        sys.path.insert(0, req["source"])
        from omegaconf import OmegaConf
        from unimernet.models.unimernet.unimernet import UniMERModel
        from unimernet.processors.formula_processor import FormulaImageEvalProcessor

        cfg = OmegaConf.create({"model_name": req["checkpoint"], "max_seq_len": 512})
        model = UniMERModel(
            model_name=None,
            model_config=cfg,
            tokenizer_name=None,
            tokenizer_config=OmegaConf.create({"path": req["checkpoint"]}),
        )
        # Official released file contains tensor state, not user-supplied executable objects.
        weights = torch.load(
            Path(req["checkpoint"]) / "unimernet_tiny.pth", map_location="cpu", weights_only=True
        )
        state = weights.get("model", weights)
        model.load_state_dict(state, strict=True)
        model = model.to("cuda").eval()
        processor = FormulaImageEvalProcessor([192, 672])

        def predict(image):
            result = model.generate(
                {"image": processor(image).unsqueeze(0).to("cuda")},
                temperature=1.0,
                do_sample=False,
            )
            return result["pred_str"][0], result["pred_ids"][0].tolist()

        config = {
            "max_new_tokens": 512,
            "image_size": [192, 672],
            "do_sample": False,
            "official_processor": True,
            "checkpoint_strict": True,
        }
    else:
        raise ValueError("Unknown real recognition backend")
    rows = []
    for row in req["images"]:
        if hashlib.sha256(Path(row["image"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Input digest changed")
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with Image.open(row["image"]) as im, torch.inference_mode():
            text, tokens = predict(im.convert("RGB"))
        torch.cuda.synchronize()
        rows.append(
            {
                "id": row["id"],
                "text": text,
                "tokens": tokens,
                "seconds": time.perf_counter() - start,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "input_sha256": row["sha256"],
                "confidence": None,
                "confidence_calibrated": False,
                "possible_token_truncation": len(tokens) >= config["max_new_tokens"],
            }
        )
        Path(req["output"]).write_text(
            json.dumps(
                {
                    "model_executed": True,
                    "is_generator_training": False,
                    "engine": engine,
                    "config": config,
                    "gpu": torch.cuda.get_device_name(0),
                    "rows": rows,
                    "dependencies": {
                        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(row["id"], flush=True)


if __name__ == "__main__":
    main()
