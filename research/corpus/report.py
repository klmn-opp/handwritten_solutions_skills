"""Evidence packet with real crops, unverified transcriptions and explicit training No-Go."""

import json
import shutil
import textwrap
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hwsol.profiles.io import atomic_json
from research.corpus.experiment import create
from research.corpus.store import Corpus
from research.english.artifacts import lossless_pdf


def report(run_id: str):
    corpus = Corpus(Path("workspace/datasets/personal-homework"))
    run = create(
        "corpus-evidence-report",
        run_id=run_id,
        config={"private_local_only": True, "no_human_truth_inferred": True},
    )
    samples = corpus.rows("sample")
    results = corpus.rows("recognition")
    by_sample = {}
    for row in results:
        by_sample.setdefault(row["sample_id"], []).append(row)
    inventory = json.loads(
        Path("workspace/runs/v09-user-pdf-intake/reports/files.json").read_text()
    )
    metrics = {}
    for name in ("trocr", "unimernet", "tesseract"):
        rows = [r for r in results if r["engine"] == name]
        metrics[name] = {
            "recognized_crops": len(rows),
            "inference_seconds_excluding_load": sum(r["raw"]["seconds"] for r in rows),
            "peak_torch_allocated_bytes": max(
                (r["raw"].get("peak_cuda_bytes", 0) for r in rows), default=0
            ),
            "true_CER": None,
            "true_WER": None,
            "true_exact_match": None,
            "reason": "No independent human-verified transcript set yet",
        }
    observations = [
        {
            "id": "s-c464e605308f794274a5d0c1",
            "issue": (
                "Leading minus and negative exponent are lost; related equations confuse B/digits."
            ),
            "type": "symbol-substitution-or-omission",
        },
        {
            "id": "s-af334cd70c3a89c2ccb5ffbd",
            "issue": "Visible integral upper limit x is transcribed as pi.",
            "type": "mathematical-structure-label",
        },
        {
            "id": "s-0b71e552061ef4ba91dc289c",
            "issue": "Red mark crosses ink; recognition replaces G(t) with a different expression.",
            "type": "occluded-or-overwritten-region",
        },
        {
            "id": "s-50f0440624c8e02a9670abe9",
            "issue": "'find f satisfy' becomes 'find f satisfying.' Do not correct author grammar.",
            "type": "recognizer-language-prior",
        },
        {
            "id": "s-07b0c727d697386c60ac6de9",
            "issue": "Plausible recognition on a clean crop; human confirmation still required.",
            "type": "plausible-candidate-not-ground-truth",
        },
    ]
    atomic_json(
        run.path / "reports/observations.json",
        {
            "reviewer_kind": "assistant-visual-audit",
            "not_human_ground_truth": True,
            "observations": observations,
        },
    )
    atomic_json(run.path / "reports/metrics.json", metrics)
    atomic_json(
        run.path / "reports/corpus-summary.json",
        {
            "new_pdf_documents": len(inventory),
            "original_pdf_pages": sum(r["pages"] for r in inventory),
            "rendered_homework_pages": 10,
            "legacy_image_pages": 7,
            "candidate_regions": len(samples),
            "candidate_content_types": dict(Counter(s["content_type"] for s in samples)),
            "recognition_results": len(results),
            "human_verified_labels": sum(s["review_status"] == "verified" for s in samples),
            "training_executed": False,
            "data_volume_sufficiency": "not established; verified coverage is the bottleneck",
        },
    )
    font = ImageFont.truetype("DejaVuSans.ttf", 23)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    images = []
    for i, observation in enumerate(observations, 1):
        sample = corpus.get("sample", observation["id"])
        canvas = Image.new("RGB", (1200, 1000), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (35, 25),
            f"v0.9 real homework evidence {i}/5 | NOT HUMAN-VERIFIED",
            fill="black",
            font=font,
        )
        crop = Image.open(corpus.path(sample["image_crop"])).convert("RGB")
        crop.thumbnail((1120, 330))
        canvas.paste(crop, (35, 95))
        lines = [
            "Assistant observation (not a verified training label):",
            *textwrap.wrap(observation["issue"], 100),
            "",
        ]
        for row in by_sample.get(sample["sample_id"], []):
            lines.extend(textwrap.wrap(row["engine"] + " candidate: " + row["text"], 105))
            lines.append("")
        lines.extend(
            [
                "Source PDF SHA256: " + sample["source_hash"],
                "Page ID: " + sample["page_id"],
                "Original raster box: " + str(sample["bounding_box"]),
                "",
                "No inpainting. No semantic answer correction. No neural handwriting training.",
            ]
        )
        draw.multiline_text((35, 450), "\n".join(lines), fill="black", font=small, spacing=8)
        path = run.path / "rendered/pages" / f"{i:02d}.png"
        canvas.save(path)
        images.append(path)
        shutil.copyfile(
            corpus.path(sample["image_crop"]), run.path / "comparisons" / f"original-{i:02d}.png"
        )
    atomic_json(
        run.path / "reports/pdf-integrity.json",
        lossless_pdf(images, run.path / "rendered/homework_solutions.pdf"),
    )
    return run.finish(
        "held",
        result={
            "recognition_evidence_pages": len(images),
            "personal_model_training": False,
            "decision": "GO corpus review; NO-GO personal training/SFT/RL",
        },
    )


if __name__ == "__main__":
    import sys

    print(report(sys.argv[1]))
