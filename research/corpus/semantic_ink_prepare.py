"""Versioned font-only content pretraining, separated from verified real supervision."""

import argparse
import json
import string
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.workspace import profile_store, verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.context_handnote_reference import render
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance
from research.corpus.semantic_ink_data import conditions, pad_conditions
from research.corpus.word_style_data import normalize


def prepare(parent: Path):
    verify_run(parent)
    old = json.loads((parent / "reports/samples.json").read_text())
    splits = json.loads((parent / "reports/indices.json").read_text())
    tests = [(old[i]["text"], kind, i) for kind in ["development", "novel"] for i in splits[kind]]
    forbidden = {text.lower() for text, _, _ in tests}
    train = sorted(
        {old[i]["text"] for i in splits["train"] if old[i]["text"].lower() not in forbidden}
    )
    rng = np.random.default_rng(3015)
    alphabet = string.ascii_letters + string.digits + ".,:;!?-()/"
    # Balanced explicit coverage plus randomized local contexts, all font-derived.
    for _ in range(8):
        chars = rng.permutation(list(alphabet))
        train.extend("".join(chars[j : j + 8]) for j in range(0, len(chars), 8))
    train = sorted(set(train) - forbidden)
    profile = profile_store().resolve("personal@v1")
    run = create(
        "semantic-font-data",
        seed=3015,
        config={
            "parent": str(parent.resolve()),
            "parent_sha256": digest(parent / "run.json"),
            "profile": profile.provenance(),
            "all_training_targets": "personal-font-derived auxiliary only, zero real scans",
            "canonical_glyphs": True,
            "C_used": False,
        },
    )
    try:
        rows, images, fields = [], [], []
        for text, split, old_index in [(w, "train", None) for w in train] + tests:
            image, _, provenance = render(text, profile, canonical_glyphs=True)
            ink, transform = normalize(image)
            tokens, boxes = conditions(text, provenance, transform)
            index = len(rows)
            path = run.path / f"solutions/auxiliary-{index:04d}.png"
            Image.fromarray(np.uint8((1 - ink) * 255)).save(path)
            rows.append(
                {
                    "index": index,
                    "text": text,
                    "split": split,
                    "old_index": old_index,
                    "source_type": "personal-font-derived",
                    "real_scan": False,
                    "reference_glyphs": provenance,
                    "reference_transform": transform,
                    "image": str(path),
                    "image_sha256": digest(path),
                    "external_fallback_count": sum(
                        g["external_fallback"] for g in provenance["glyph_provenance"]
                    ),
                }
            )
            images.append(ink)
            fields.append((tokens, boxes))
        tokens, boxes = pad_conditions(fields)
        np.savez_compressed(
            run.path / "reports/arrays.npz",
            ink=np.array(images),
            sdf=np.array([signed_distance(im) for im in images]),
            tokens=tokens,
            boxes=boxes,
        )
        write(run.path / "reports/samples.json", rows)
        write(
            run.path / "reports/indices.json",
            {
                kind: [q["index"] for q in rows if q["split"] == kind]
                for kind in ["train", "development", "novel"]
            },
        )
        run.finish(
            "held",
            result={
                "train": len(train),
                "evaluation": len(tests),
                "real_scans": 0,
                "fallback_glyph_occurrences": sum(q["external_fallback_count"] for q in rows),
                "C_used": False,
            },
        )
    except Exception as exc:
        run.fail(exc)
        raise
    print(run.path)
    return run.path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    prepare(parser.parse_args().parent)
