"""Versioned raw and conservative word-aligned views for equal-data experiments."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_training_data import reference_image, whole_ink
from research.corpus.context_word_alignment import aligned_view
from research.corpus.experiment import create
from research.corpus.reference_data import signed_distance


def prepare(dataset: Path) -> Path:
    verify_run(dataset)
    profile = profile_store().resolve("personal@v1")
    rows = [json.loads(s) for s in (dataset / "reports/samples.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["content_type"].startswith("english")]
    run = create(
        "context-word-aligned-views",
        config={
            "dataset": str(dataset.resolve()),
            "dataset_sha256": sha256(dataset / "run.json"),
            "profile": profile.provenance(),
            "raw_preserved": True,
            "training_geometry": "natural word gaps, uniform word scale; no character cutting",
            "evaluation_reference_uses_target_geometry": False,
        },
    )
    try:
        data = {
            k: []
            for k in [
                "reference",
                "raw",
                "aligned",
                "raw_sdf",
                "aligned_sdf",
                "reference_sdf",
                "admitted",
                "split",
            ]
        }
        metadata = []
        for i, row in enumerate(rows):
            if sha256(Path(row["crop"])) != row["crop_sha256"]:
                raise ValueError("Changed crop")
            image = Image.open(row["crop"])
            raw, raw_transform = whole_ink(image)
            ref, ref_transform = whole_ink(reference_image(row["normalized_label"], profile.font))
            reason = None
            try:
                aligned_ref, aligned, transform = aligned_view(
                    row["normalized_label"], image, profile.font
                )
                if not np.array_equal(ref, aligned_ref):
                    raise ValueError("reference mismatch")
            except ValueError as exc:
                aligned = raw.copy()
                transform = None
                reason = str(exc)
            for key, value in [
                ("reference", ref),
                ("raw", raw),
                ("aligned", aligned),
                ("raw_sdf", signed_distance(raw)),
                ("aligned_sdf", signed_distance(aligned)),
                ("reference_sdf", signed_distance(ref)),
                ("admitted", reason is None),
                ("split", row["dataset_split"]),
            ]:
                data[key].append(value)
            metadata.append(
                {
                    **row,
                    "alignment": transform,
                    "raw_transform": raw_transform,
                    "reference_transform": ref_transform,
                    "alignment_rejected_reason": reason,
                    "reference_fallback_codepoints": sorted(
                        set(row["normalized_label"])
                        - {chr(c) for c in profile.manifest.coverage}
                        - {" "}
                    ),
                }
            )
            for label, ink in [("real", raw), ("aligned", aligned), ("reference", ref)]:
                Image.fromarray(np.uint8((1 - ink.clip(0, 1)) * 255)).save(
                    run.path / f"comparisons/{i:03d}-{label}.png"
                )
        np.savez_compressed(
            run.path / "reports/arrays.npz", **{k: np.asarray(v) for k, v in data.items()}
        )
        atomic_json(run.path / "reports/samples.json", metadata)
        result = {
            "regions": len(rows),
            "by_split": dict(Counter(r["dataset_split"] for r in rows)),
            "aligned_by_split": dict(
                Counter(r["dataset_split"] for r in metadata if r["alignment"] is not None)
            ),
            "rejects": dict(
                Counter(r["alignment_rejected_reason"] for r in metadata if r["alignment"] is None)
            ),
            "label_accuracy": None,
            "no_new_real_samples_created": True,
        }
        atomic_json(
            run.path / "reports/manifest.json",
            {
                "dataset": str(dataset.resolve()),
                "arrays_sha256": sha256(run.path / "reports/arrays.npz"),
                "samples_sha256": sha256(run.path / "reports/samples.json"),
                "result": result,
            },
        )
        run.finish("prepared", result=result)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    print(prepare(p.parse_args().dataset))
