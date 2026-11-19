"""Versioned lexical generalization inputs; private real pixels retain original splits."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import profile_store, verify_run
from research.corpus.context_handnote_reference import render
from research.corpus.experiment import create
from research.corpus.latent_ink_train import DATA, FORBIDDEN
from research.corpus.thin_ink_data import token_positions
from research.corpus.word_style_data import normalize

NOVEL = [
    "coherent",
    "consistent",
    "handwriting",
    "reasoning",
    "variation",
    "solution",
    "integrate",
    "function",
    "smooth",
    "student",
    "vector",
    "matrix",
]
OLD = Path("workspace/runs/20261119T170441302262Z-latent-ink-latent-b9aec932")


def split_indices(rows: list[dict], regression: list[int]) -> dict:
    eligible = [
        i
        for i, r in enumerate(rows)
        if r["text"].isascii() and r["text"].isalpha() and 2 <= len(r["text"]) <= 18
    ]
    train = [i for i in eligible if rows[i]["dataset_split"] == "train"]
    validation = [i for i in eligible if rows[i]["dataset_split"] == "validation"]
    if any(r["source_hash"] == FORBIDDEN for r in rows):
        raise ValueError("Sealed C forbidden")
    if {rows[i]["source_hash"] for i in train} & {rows[i]["source_hash"] for i in validation}:
        raise ValueError("Document split overlap")
    all_train_labels = {r["text"] for r in rows if r["dataset_split"] == "train"}
    if set(NOVEL) & all_train_labels:
        raise ValueError("Novel prompt exists in historical Train")
    chars = set("".join(rows[i]["text"] for i in train))
    if set("".join(NOVEL)) - chars:
        raise ValueError("Novel text has untrained character")
    if set(regression) - set(range(len(rows))):
        raise ValueError("Unknown regression ID")
    return {"train": train, "validation": validation, "regression": regression}


def prepare() -> Path:
    verify_run(DATA)
    verify_run(OLD)
    original = json.loads((DATA / "reports/samples.json").read_text())
    old_ids = json.loads((OLD / "reports/indices.json").read_text())["train"]
    split = split_indices(original, old_ids)
    chosen = sorted(set(sum(split.values(), [])))
    mapping = {i: j for j, i in enumerate(chosen)}
    profile = profile_store().resolve("personal@v1")
    run = create(
        "writer-style-data",
        config={
            "source": str(DATA.resolve()),
            "source_sha256": sha256(DATA / "run.json"),
            "C_used": False,
            "profile": profile.provenance(),
            "new_labels": False,
            "novel_prompts_are_not_training_data": True,
        },
    )
    try:
        source = np.load(DATA / "reports/arrays.npz")
        targets = list(source["target"][chosen])
        refs = list(source["reference"][chosen])
        rows = [{**original[i], "original_index": i} for i in chosen]
        for r in rows:
            if sha256(Path(r["crop"])) != r["crop_sha256"]:
                raise ValueError("Changed real source crop")
        indices = {name: [mapping[i] for i in ii] for name, ii in split.items()}
        indices["novel"] = []
        for word in NOVEL:
            raw, _, glyphs = render(word, profile)
            ref, transform = normalize(raw)
            indices["novel"].append(len(rows))
            refs.append(ref)
            targets.append(np.zeros_like(ref))
            rows.append(
                {
                    "text": word,
                    "dataset_split": "novel-prompt",
                    "source_hash": None,
                    "target_available": False,
                    "real_personal_pixels": False,
                    "label_source": "fixed-evaluation-prompt",
                    "reference_transform": transform,
                    "reference_glyphs": glyphs,
                }
            )
        indices["evaluated"] = sorted(
            set(indices["regression"] + indices["validation"] + indices["novel"])
        )
        positions = np.stack([token_positions(q) for q in rows])
        np.savez_compressed(
            run.path / "reports/arrays.npz",
            reference=np.stack(refs),
            target=np.stack(targets),
            positions=positions,
        )
        atomic_json(run.path / "reports/samples.json", rows)
        atomic_json(run.path / "reports/indices.json", indices)
        report = {
            "samples": len(rows),
            "train": len(indices["train"]),
            "train_labels": len({rows[i]["text"] for i in indices["train"]}),
            "old_document_validation": len(indices["validation"]),
            "validation_unseen_labels": sum(
                rows[i]["text"] not in {rows[j]["text"] for j in indices["train"]}
                for i in indices["validation"]
            ),
            "novel_prompts": NOVEL,
            "train_by_document": dict(Counter(rows[i]["source_hash"] for i in indices["train"])),
            "C_used": False,
            "old_validation_is_not_new_holdout": True,
            "synthetic_training_images": 0,
            "novel_zero_target_is_placeholder_only": True,
        }
        atomic_json(run.path / "reports/audit.json", report)
        run.finish("prepared", result=report)
    except Exception as exc:
        run.fail(exc)
        raise
    return run.path


if __name__ == "__main__":
    print(prepare())
