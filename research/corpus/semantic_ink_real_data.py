"""Assemble reviewed real whole words and separate font auxiliaries, without relabeling."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance
from research.corpus.semantic_ink_data import conditions, pad_conditions


def prepare(canonical: Path, extra: Path, auxiliary: Path):
    for p in [canonical, extra, auxiliary]:
        verify_run(p)
    old = json.loads((canonical / "reports/samples.json").read_text())
    idx = json.loads((canonical / "reports/indices.json").read_text())
    # NpzFile reloads/decompresses an entire member on EACH __getitem__.
    # Keeping slices of repeated loads would retain one full corpus per word.
    with np.load(canonical / "reports/arrays.npz") as archive:
        arrays = {key: archive[key] for key in ["target", "reference"]}
    new = json.loads((extra / "reports/processed-words.json").read_text())
    reviewed = json.loads((extra / "reports/processed-word-review.json").read_text())
    admitted = {q["index"]: q for q in reviewed if q["status"] == "accept"}
    extra_rows = {q["index"]: q for q in new if q["index"] in admitted}
    assert not set(extra_rows) & set(idx["train"] + idx["development"] + idx["novel"])
    records, ink, reference, fields = [], [], [], []
    for i in idx["train"] + sorted(extra_rows) + idx["development"] + idx["novel"]:
        row = old[i]
        if i in extra_rows:
            q = extra_rows[i]
            if digest(q["image"]) != admitted[i]["image_sha256"]:
                raise ValueError("Reviewed processed image changed")
            target = np.load(extra / f"solutions/word-{i:04d}.npy")
            # Require the actual neural target array to reproduce the reviewed PNG.
            with Image.open(q["image"]) as image:
                assert np.array_equal(np.uint8((1 - target) * 255), np.asarray(image))
        else:
            target = arrays["target"][i]
        group = (
            "development" if i in idx["development"] else "novel" if i in idx["novel"] else "train"
        )
        records.append(
            {
                **row,
                "index": len(records),
                "old_index": i,
                "split": group,
                "source_type": "real-reviewed-whole-word"
                if group != "novel"
                else "fixed-text-prompt",
                "real_scan": group != "novel",
                "synthetic_repair": False,
                "source_data_sha256": digest(
                    (extra if i in extra_rows else canonical) / "run.json"
                ),
            }
        )
        ink.append(target)
        reference.append(arrays["reference"][i])
        fields.append(conditions(row["text"], row["reference_glyphs"], row["reference_transform"]))
    aux_rows = json.loads((auxiliary / "reports/samples.json").read_text())
    with np.load(auxiliary / "reports/arrays.npz") as archive:
        aux = {"ink": archive["ink"]}
    for i, row in enumerate(aux_rows):
        if row["split"] != "train":
            continue
        records.append(
            {
                **row,
                "index": len(records),
                "split": "auxiliary",
                "source_data_sha256": digest(auxiliary / "run.json"),
            }
        )
        ink.append(aux["ink"][i])
        reference.append(aux["ink"][i])
        fields.append(conditions(row["text"], row["reference_glyphs"], row["reference_transform"]))
    run = create(
        "semantic-real-word-data",
        config={
            "sources": {
                str(p.resolve()): digest(p / "run.json") for p in [canonical, extra, auxiliary]
            },
            "whole_words_not_character_cuts": True,
            "real_vs_font_domains_separate": True,
            "C_used": False,
            "new_holdout": False,
        },
    )
    tokens, boxes = pad_conditions(fields)
    sdf = []
    for im in ink:
        sdf.append(signed_distance(im) if (im > 0.5).any() else np.zeros_like(im))
    np.savez_compressed(
        run.path / "reports/arrays.npz",
        ink=np.array(ink),
        sdf=np.array(sdf),
        reference=np.array(reference),
        tokens=tokens,
        boxes=boxes,
    )
    splits = {
        kind: [q["index"] for q in records if q["split"] == kind]
        for kind in ["train", "auxiliary", "development", "novel"]
    }
    write(run.path / "reports/samples.json", records)
    write(run.path / "reports/indices.json", splits)
    run.finish(
        "held",
        result={**{k: len(v) for k, v in splits.items()}, "C_used": False, "neural_updates": 0},
    )
    print(run.path)
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["canonical", "extra", "auxiliary"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.canonical, args.extra, args.auxiliary)
