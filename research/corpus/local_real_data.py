"""Verified English glyph targets, with original whole-word provenance retained."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hwsol.workspace import verify_run
from research.corpus.context_content_train import digest, write
from research.corpus.experiment import create
from research.corpus.ink_geometry import signed_distance
from research.corpus.partial_glyph_support import validate_partial


def prepare(support: Path, rare: Path) -> Path:
    for p in [support, rare]:
        verify_run(p)
    with np.load(support / "reports/arrays.npz") as f:
        images, labels = list(f["ink"]), list(f["labels"])
    rows = json.loads((support / "reports/samples.json").read_text())
    inputs = json.loads((rare / "reports/inputs.json").read_text())
    sources = json.loads((rare / "reports/source-private.json").read_text())
    reviews = {q["id"]: q for q in json.loads((rare / "reports/review.json").read_text())}
    with np.load(rare / "reports/arrays.npz") as f:
        raw = f["ink"]
    for s, source, ink in zip(inputs, sources, raw, strict=True):
        q = reviews[s["id"]]
        if q["status"] != "accept":
            continue
        assert q["observed_text"] == s["text"] == source["text"]
        assert digest(s["image"]) == s["image_sha256"] == q["image_sha256"]
        assert digest(source["crop"]) == source["crop_sha256"]
        for g in validate_partial(q, s["text"], ink):
            x0, y0, x1, y1 = g["bbox"]
            im = np.zeros((96, 96), np.float32)
            im[16:80, 16:80] = np.asarray(
                Image.fromarray(ink[y0:y1, x0:x1]).resize((64, 64), Image.Resampling.BILINEAR)
            )
            images.append(im)
            labels.append(ord(g["char"]))
            rows.append(
                {
                    "id": f"{s['id']}-{g['position']}",
                    "word": source["text"],
                    "label": g["char"],
                    "position": g["position"],
                    "source_crop": source["crop"],
                    "source_crop_sha256": source["crop_sha256"],
                    "source_hash": source["source_hash"],
                    "page_number": source["page_number"],
                    "raw_normalized_bbox": g["bbox"],
                    "transform": source["transform"],
                    "review_run": str(rare),
                    "review_id": q["id"],
                    "source_role": "verified real English glyph; original word retained",
                }
            )
    tokens = np.zeros((len(rows), 2, 3), np.int64)
    groups, group_lookup = [], {}
    seen = set()
    for i, q in enumerate(rows):
        key = q["source_crop_sha256"], q["position"]
        if key in seen or digest(q["source_crop"]) != q["source_crop_sha256"]:
            raise ValueError("Duplicate glyph or changed original crop")
        seen.add(key)
        word, pos = q["word"], q["position"]
        assert word[pos] == q["label"] == chr(labels[i])
        tokens[i, 0] = [
            ord(word[pos]),
            ord(word[pos - 1]) if pos else 0,
            ord(word[pos + 1]) if pos + 1 < len(word) else 0,
        ]
        group_lookup.setdefault(key[0], len(group_lookup))
        groups.append(group_lookup[key[0]])
    images = np.array(images)
    boxes = np.zeros((len(rows), 2, 4), np.float32)
    boxes[:, 0] = [16, 16, 80, 80]
    run = create(
        "verified-local-real-data",
        config={
            "sources": {str(p): digest(p / "run.json") for p in [support, rare]},
            "C_used": False,
            "new_real_observations": 0,
            "synthetic_targets": 0,
            "target_normalization": "reviewed bounding box to central64 in96, training only",
            "split": "existing Train only; document holdout untouched",
            "group": "shared word code keyed by original crop hash, not target pixel input",
        },
    )
    for i, im in enumerate(images):
        path = run.path / f"solutions/glyph-{i:04d}.png"
        Image.fromarray(np.uint8((1 - im) * 255)).save(path)
        rows[i].update(training_image=str(path), training_image_sha256=digest(path))
    np.savez_compressed(
        run.path / "reports/arrays.npz",
        ink=images,
        sdf=np.stack([signed_distance(im) for im in images]),
        labels=np.array(labels),
        tokens=tokens,
        boxes=boxes,
        groups=np.array(groups),
    )
    write(run.path / "reports/samples.json", rows)
    write(run.path / "reports/coverage.json", dict(sorted(Counter(chr(c) for c in labels).items())))
    write(run.path / "reports/groups.json", group_lookup)
    run.finish(
        "held",
        result={
            "real_derived_views": len(rows),
            "classes": len(set(labels)),
            "word_groups": len(group_lookup),
            "f_views": int(sum(c == ord("f") for c in labels)),
            "new_real_observations": 0,
            "neural_updates": 0,
            "C_used": False,
        },
    )
    return run.path
