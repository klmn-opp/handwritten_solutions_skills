"""Same-text complete-region baselines, original references, and recognizer inputs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from hwsol.config import RenderConfig
from hwsol.jobs.paper import pdf_image
from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_composer import FrozenBank, layout, replace_ink
from research.corpus.context_evaluation_ink import evaluation_ink
from research.corpus.experiment import create


def strips(
    text: str,
    identity: str,
    profile,
    banks: dict,
    *,
    methods: tuple[str, ...] = ("handnote", "round2", "repair"),
    dpi: int | None = None,
) -> tuple[dict, dict]:
    if not methods or not set(methods) <= {"handnote", "round2", "repair"}:
        raise ValueError("Unknown frozen comparison method")
    cfg = RenderConfig(seed=1010, font_size_pt=12, ink="#000000", semantic_layout=True)
    if dpi is not None:
        cfg = RenderConfig(**{**cfg.model_dump(), "dpi": dpi})
    page, renderer, provenance = layout(text, "Reference", profile, cfg)
    original = dict(renderer._outlines)
    # Explicit title contains exactly nine visible letters; crop only body items.
    body = provenance[len("Reference") :]
    bounds = np.array([r["bounds_px"] for r in body if r["bounds_px"]])
    box = [
        math.floor(bounds[:, 0].min()) - 8,
        math.floor(bounds[:, 1].min()) - 8,
        math.ceil(bounds[:, 2].max()) + 8,
        math.ceil(bounds[:, 3].max()) + 8,
    ]
    images, receipts = {}, {}
    for name in methods:
        renderer._outlines = dict(original)
        if name == "handnote":
            current, records = page, provenance
            result = {"neural_items": 0, "end_to_end_line_generation": False}
        else:
            current, records, result = replace_ink(
                page, renderer, provenance, banks[name], identity
            )
        blob = renderer.to_vector_pdf([current])
        with pdfium.PdfDocument(blob) as doc:
            image = pdf_image(doc, 0, cfg.dpi / 72).crop(box)
        images[name] = image
        receipts[name] = {
            "crop_from_pdf": box,
            "provenance": records,
            "result": result,
            "formula_groups": renderer.formulas,
            "matrix_groups": renderer.matrices,
        }
    return images, receipts


def prepare(evaluation: Path) -> Path:
    verify_run(evaluation)
    samples = json.loads((evaluation / "reports/samples.json").read_text())
    previous = Path("workspace/runs/20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2")
    old = json.loads((previous / "reports/candidates.json").read_text())["candidates"]
    banks = {k: FrozenBank(Path(old[k]["sample_run"])) for k in ["round2", "repair"]}
    models = json.loads((evaluation / "reports/request.json").read_text())["models"]
    profile = profile_store().resolve("personal@v1")
    run = create(
        "context-same-text-comparisons",
        seed=1010,
        config={
            "evaluation": str(evaluation.resolve()),
            "evaluation_sha256": sha256(evaluation / "run.json"),
            "selection": "all validation/holdout English plus eight train references",
            "training": False,
            "old_methods_are_glyph_composition": True,
        },
    )
    try:
        selected, train_count = [], 0
        for i, row in enumerate(samples):
            if row["dataset_split"] == "train":
                if train_count >= 8:
                    continue
                train_count += 1
            selected.append((i, row))
        observations, source_records = [], []
        for i, row in selected:
            images, receipts = strips(row["normalized_label"], row["sample_id"], profile, banks)
            real = Image.open(row["crop"]).convert("RGB")
            images["real-reference"] = real
            atomic_json(
                run.path / f"reports/source-{i:03d}.json", {"row": row, "compositions": receipts}
            )
            for method, image in images.items():
                original = run.path / f"comparisons/{method}-{i:03d}-original.png"
                image.save(original)
                ink, transform = evaluation_ink(image)
                normalized = Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255))
                key = f"{method}-{i:03d}"
                path = run.path / f"rendered/pages/{key}.png"
                normalized.save(path)
                observations.append(
                    {
                        "id": key,
                        "method": method,
                        "row_index": i,
                        "text": row["normalized_label"],
                        "split": row["dataset_split"],
                        "image": str(path),
                        "sha256": sha256(path),
                        "preprocessing": transform,
                        "raw_image": str(original),
                    }
                )
            for method in models:
                source = evaluation / f"rendered/pages/{method}-{i:03d}-s0.png"
                # Preserve the raw fixed model output; do not repair or pick a nicer latent.
                key = f"context-{method}-{i:03d}"
                path = run.path / f"rendered/pages/{key}.png"
                path.write_bytes(source.read_bytes())
                observations.append(
                    {
                        "id": key,
                        "method": "context-" + method,
                        "row_index": i,
                        "text": row["normalized_label"],
                        "split": row["dataset_split"],
                        "image": str(path),
                        "sha256": sha256(path),
                        "preprocessing": None,
                        "raw_image": str(source),
                    }
                )
            source_records.append({"row_index": i, **row})
        atomic_json(run.path / "reports/observations-input.json", observations)
        atomic_json(run.path / "reports/selected-regions.json", source_records)
        # Real same-content mathematical references; unsupported LaTeX stays explicit.
        dataset = Path("workspace/runs/20261119T014221658876Z-continuous-context-dataset-40e4147b")
        regions = [
            json.loads(line)
            for line in (dataset / "reports/samples.jsonl").read_text().splitlines()
        ]
        math_rows = []
        for row in (
            r for r in regions if r["content_type"] in {"math-expression", "multi-line-derivation"}
        ):
            if len([r for r in math_rows if r["status"] == "rendered"]) >= 8:
                break
            tex = row["normalized_label"]
            if tex.startswith(r"\begin{aligned}") and tex.endswith(r"\end{aligned}"):
                body = tex[len(r"\begin{aligned}") : -len(r"\end{aligned}")]
                if r"\begin" in body:
                    continue
                source = "\n\n".join(
                    "$$\n" + part.replace("&", "").strip() + "\n$$" for part in body.split(r"\\")
                )
            else:
                source = "$$\n" + tex + "\n$$"
            number = len(math_rows)
            try:
                images, receipts = strips(source, row["sample_id"], profile, banks)
            except ValueError as exc:
                math_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "status": "unsupported",
                        "reason": str(exc),
                        "raw_latex": tex,
                    }
                )
                continue
            images["real-reference"] = Image.open(row["crop"]).convert("RGB")
            paths = {}
            for method, image in images.items():
                path = run.path / f"comparisons/math-{number:02d}-{method}.png"
                image.save(path)
                paths[method] = str(path)
            math_rows.append(
                {
                    "row": row,
                    "raw_latex": tex,
                    "render_markdown": source,
                    "status": "rendered",
                    "images": paths,
                    "receipts": receipts,
                }
            )
        atomic_json(run.path / "reports/math-references.json", math_rows)
        run.finish(
            "held",
            result={
                "same_content_english_regions": len(selected),
                "recognizer_inputs": len(observations),
                "models": len(models),
                "math_same_content_rendered": sum(r["status"] == "rendered" for r in math_rows),
                "human_review": False,
                "production_eligible": False,
            },
        )
        return run.path
    except Exception as exc:
        run.fail(exc)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation", type=Path)
    print(prepare(parser.parse_args().evaluation))
