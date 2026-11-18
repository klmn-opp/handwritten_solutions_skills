"""Recognition adapters exchange crops and candidate text; no source data leaves this host."""

import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.experiment import create
from research.corpus.store import Corpus


def recognize(
    corpus: Corpus,
    engine: str,
    sample_ids: list[str],
    *,
    run_id: str | None = None,
    python: Path | None = None,
    checkpoint: Path | None = None,
    source: Path | None = None,
) -> Path:
    samples = [corpus.get("sample", sid) for sid in sample_ids]
    if not samples:
        raise ValueError("Select at least one real crop")
    weights = {
        str(p.resolve()): sha256(p)
        for p in (checkpoint.rglob("*") if checkpoint else [])
        if p.is_file()
        and p.suffix in {".pth", ".safetensors", ".bin", ".json"}
        and ".cache" not in p.parts
    }
    run = create(
        "corpus-recognition",
        run_id=run_id,
        seed=902,
        config={
            "engine": engine,
            "sample_ids": sample_ids,
            "checkpoints": weights,
            "python": str(python),
            "source": str(source),
            "no_external_upload": True,
        },
    )
    try:
        if engine == "tesseract":
            version = subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[0]
            rows = []
            for s in samples:
                start = time.perf_counter()
                result = subprocess.run(
                    [
                        "tesseract",
                        str(corpus.path(s["image_crop"])),
                        "stdout",
                        "-l",
                        "eng",
                        "--psm",
                        "7",
                        "tsv",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                records = list(csv.DictReader(io.StringIO(result.stdout), delimiter="\t"))
                words = [r for r in records if r["text"].strip()]
                rows.append(
                    {
                        "id": s["sample_id"],
                        "text": " ".join(r["text"] for r in words),
                        "raw_tsv": result.stdout,
                        "seconds": time.perf_counter() - start,
                        "input_sha256": s["crop_sha256"],
                        "confidence": sum(float(r["conf"]) for r in words) / len(words)
                        if words
                        else None,
                        "confidence_calibrated": False,
                    }
                )
            output = {
                "engine": engine,
                "version": version,
                "rows": rows,
                "model_executed": True,
                "is_generator_training": False,
            }
            atomic_json(run.path / "reports/raw.json", output)
        else:
            if engine not in {"trocr", "unimernet"} or python is None or checkpoint is None:
                raise ValueError("Explicit isolated interpreter and local checkpoint required")
            req = {
                "engine": engine,
                "seed": 902,
                "checkpoint": str(checkpoint.resolve()),
                "source": str(source.resolve()) if source else None,
                "output": str(run.path / "reports/raw.json"),
                "images": [
                    {
                        "id": s["sample_id"],
                        "image": str(corpus.path(s["image_crop"])),
                        "sha256": s["crop_sha256"],
                    }
                    for s in samples
                ],
            }
            atomic_json(run.path / "reports/request.json", req)
            with (run.path / "logs/inference.log").open("w") as stream:
                subprocess.run(
                    [
                        str(python.absolute()),
                        str(Path(__file__).with_name("recognition_worker.py").resolve()),
                        "--request",
                        str(run.path / "reports/request.json"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=3600,
                )
            output = json.loads((run.path / "reports/raw.json").read_text())
            version = output["dependencies"]["transformers"]
        for row in output["rows"]:
            corpus.add_recognition(
                row["id"],
                {
                    "engine": engine,
                    "version": version,
                    "text": row["text"],
                    "raw": row,
                    "input_sha256": row["input_sha256"],
                    "confidence": row.get("confidence"),
                    "checkpoint_sha256": weights,
                    "run": str(run.path),
                    "source": str(source),
                },
            )
        run.finish(
            "held",
            result={
                "recognition_executed": True,
                "crops": len(output["rows"]),
                "human_verified": False,
                "generator_training": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    print(
        recognize(
            Corpus(Path(config.pop("corpus"))),
            **{
                k: Path(v) if k in {"python", "checkpoint", "source"} else v
                for k, v in config.items()
            },
        )
    )
