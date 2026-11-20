"""Versioned real glyphs plus intact positioned words, with separate label certainty."""

import json
from pathlib import Path

import numpy as np

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create


def match_conditions(word, tokens, glyph_rows, glyph_codes):
    """Only exact original-word hash AND character position can supply a local code."""
    lookup = {(q["source_crop_sha256"], q["position"]): i for i, q in enumerate(glyph_rows)}
    if len(lookup) != len(glyph_rows) or len(glyph_codes) != len(glyph_rows):
        raise ValueError("Unique source-position glyphs and matching codes required")
    actual = "".join(chr(int(t[0])) for t in tokens if t[0])
    if actual != word["text"]:
        raise ValueError("Semantic input must match the verified whole-word text")
    result = np.zeros((len(tokens), 8), np.float32)
    matches = []
    for pos, char in enumerate(word["text"]):
        k = lookup.get((word["crop_sha256"], pos))
        if k is not None:
            if glyph_rows[k]["label"] != char:
                raise ValueError("Source position label disagrees")
            result[pos] = glyph_codes[k]
            matches.append({"position": pos, "glyph_index": k})
    return result, matches


def prepare(glyphs: Path, words: Path) -> Path:
    for source in [glyphs, words]:
        verify_run(source)
    gs = json.loads((glyphs / "reports/samples.json").read_text())
    ws = json.loads((words / "reports/samples.json").read_text())
    indices = json.loads((words / "reports/indices.json").read_text())["strong"]
    with np.load(glyphs / "reports/arrays.npz") as archive:
        g = {k: archive[k] for k in archive.files}
    with np.load(words / "reports/arrays.npz") as archive:
        w = {k: archive[k] for k in archive.files}
    if len(gs) != 275 or len(indices) != 18 or not w["known_anchors"][indices].all():
        raise ValueError("Registered clean glyphs and positioned whole words required")
    if not np.array_equal(w["train_geometry"][indices], w["anchor_boxes"][indices]):
        raise ValueError("First whole-word study must not replace annotated positions by estimates")
    local = np.zeros((*w["tokens"][indices].shape[:2], 8), np.float32)
    mapped = []
    for j, i in enumerate(indices):
        q = ws[i]
        if q["split"] != "train" or q["source_hash"] == (
            "ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d"
        ):
            raise ValueError("Only existing Train, never C")
        if digest(q["crop"]) != q["crop_sha256"]:
            raise ValueError("Original whole word changed")
        local[j], matches = match_conditions(q, w["tokens"][i], gs, g["local_codes"])
        mapped.append(
            {
                **q,
                "training_index": i,
                "local_conditions_from_exact_source": matches,
                "unmatched_conditions": "zero; no invented glyph label or pixel repair",
                "whole_word_original_ink_preserved": True,
            }
        )
    run = create(
        "semantic-canvas-real-data",
        config={
            "glyphs": str(glyphs.resolve()),
            "glyphs_sha256": digest(glyphs / "run.json"),
            "words": str(words.resolve()),
            "words_sha256": digest(words / "run.json"),
            "word_indices": indices,
            "new_real_observations": 0,
            "new_character_cuts": 0,
            "C_used": False,
            "whole_word_targets_unmodified": True,
            "unreliable_single_glyph_crops_not_reintroduced": True,
        },
    )
    np.savez_compressed(
        run.path / "reports/glyphs.npz",
        ink=g["ink"],
        tokens=g["tokens"],
        boxes=g["boxes"],
        local_codes=g["local_codes"],
        labels=g["labels"],
    )
    np.savez_compressed(
        run.path / "reports/words.npz",
        ink=w["ink"][indices],
        tokens=w["tokens"][indices],
        boxes=w["anchor_boxes"][indices],
        style=local,
    )
    write(run.path / "reports/glyph-samples.json", gs)
    write(run.path / "reports/word-samples.json", mapped)
    glyph_classes = set(int(c) for c in g["labels"])
    word_classes = set(int(c) for c in w["tokens"][indices, :, 0].ravel()) - {0}
    write(
        run.path / "reports/coverage.json",
        {
            "real_isolated_classes": [chr(c) for c in sorted(glyph_classes)],
            "real_word_supervised_classes": [chr(c) for c in sorted(word_classes)],
            "word_only_classes": [chr(c) for c in sorted(word_classes - glyph_classes)],
            "independent_observations_not_sum_of_overlapping_views": True,
        },
    )
    run.finish(
        "held",
        result={
            "glyph_views": 275,
            "intact_words": 18,
            "word_only_classes": [chr(c) for c in sorted(word_classes - glyph_classes)],
            "new_observations": 0,
            "new_neural_updates": 0,
            "C_used": False,
        },
    )
    Path("/tmp/context-v01020-real-data").write_text(str(run.path))
    return run.path
