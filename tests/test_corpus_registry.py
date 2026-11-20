"""Source lineage invariants for the private research registry (no model mocks)."""

import copy
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from hwsol.profiles.io import atomic_json, sha256
from hwsol.workspace import verify_run
from research.corpus_registry import SEALED, read, register


@pytest.fixture
def corpus(tmp_path):
    original = tmp_path / "original.pdf"
    original.write_bytes(b"%PDF-1.4\noriginal immutable fixture document\n")
    parent = tmp_path / "parent.png"
    image = Image.new("RGB", (20, 10), "white")
    ImageDraw.Draw(image).line((3, 2, 8, 7), fill=(30, 40, 50), width=2)
    image.save(parent)
    crop = tmp_path / "word.png"
    image.crop((2, 1, 10, 9)).save(crop)
    derived = tmp_path / "glyph.png"
    image.crop((2, 1, 6, 9)).resize((8, 16)).save(derived)
    source_hash = sha256(original)
    documents = [
        {
            "source_hash": source_hash,
            "source_path": str(original),
            "split_group": "session:fixture",
            "writer_provenance": {"writer_id": "personal"},
        }
    ]
    line = {
        "source_hash": source_hash,
        "page_number": 1,
        "dpi": 300,
        "source_crop": str(parent),
        "source_crop_sha256": sha256(parent),
        "parent_bbox": [100, 200, 120, 210],
        "verified_text": "ab",
        "review": {"status": "accept", "transcription": "ab"},
        "training_word_links": [
            {
                "position": 0,
                "text": "ab",
                "crop": str(crop),
                "sha256": sha256(crop),
                "source_bbox": [2, 1, 10, 9],
            }
        ],
    }
    word = {
        "source_hash": source_hash,
        "page_number": 1,
        "source_crop": str(parent),
        "source_crop_sha256": sha256(parent),
        "parent_bbox": line["parent_bbox"],
        "box_in_source_crop": [2, 1, 10, 9],
        "text": "ab",
        "crop": str(crop),
        "crop_sha256": sha256(crop),
        "dataset_split": "train",
        "label_source": "fixture accepted original transcription",
    }
    glyph = {
        "source_hash": source_hash,
        "page_number": 1,
        "source_crop": str(crop),
        "source_crop_sha256": sha256(crop),
        "word": "ab",
        "position": 0,
        "label": "a",
        "training_image": str(derived),
        "training_image_sha256": sha256(derived),
        "raw_normalized_bbox": [0, 0, 4, 8],
        "transform": {"scale": 2},
        "legacy_review": {"status": "accept"},
    }
    rows = {"documents": documents, "lines": [line], "words": [word], "glyphs": [glyph]}
    specs = []
    for kind, data in rows.items():
        path = tmp_path / f"{kind}.json"
        atomic_json(path, data)
        specs.append(
            {
                "kind": kind,
                "path": str(path),
                "dataset_id": f"fixture-{kind}",
                "split": "train",
                "source_dpi": 300,
                "exposure": ["fixture-development; not holdout"],
            }
        )
    return {"specs": specs, "rows": rows, "workspace": tmp_path / "workspace", "root": tmp_path}


def initial(corpus):
    return register(corpus["specs"], workspace=corpus["workspace"])


def test_provenance_replay_and_distinct_derivatives(corpus):
    first = initial(corpus)
    before = sha256(first / "run.json")
    data = read(first / "reports/registry.json")
    assert len(data["documents"]) == 1
    assert len(data["observations"]) == 2  # One original line and its word, not a new glyph.
    assert len(data["views"]) == 3
    assert all(o["group_id"] == "session:fixture" for o in data["observations"].values())
    obs = next(o for o in data["observations"].values() if o["kind"] == "english-word")
    assert obs["box_inches"] == [round(x / 300, 10) for x in [102, 201, 110, 209]]
    for manifest in data["manifests"].values():
        snapshot = first / "solutions/manifests" / (manifest["sha256"] + ".json")
        assert sha256(snapshot) == manifest["sha256"]
        assert read(snapshot) == read(Path(manifest["original_path"]))
    replay = register(corpus["specs"], parent=first, workspace=corpus["workspace"])
    assert all(n == 0 for n in read(replay / "reports/result.json")["added"].values())
    assert read(replay / "reports/registry.json") == data
    assert sha256(first / "run.json") == before
    assert verify_run(first)["status"] == "passed"
    assert not (corpus["workspace"] / "latest.json").exists()

    # A distinct preprocessing view is preserved without inventing a new original observation.
    glyph = copy.deepcopy(corpus["rows"]["glyphs"][0])
    new_image = corpus["root"] / "new-derived.png"
    with Image.open(glyph["training_image"]) as image:
        image.resize((9, 18)).save(new_image)
    glyph.update(training_image=str(new_image), training_image_sha256=sha256(new_image))
    manifest = corpus["root"] / "new-glyph.json"
    atomic_json(manifest, [glyph])
    spec = {**corpus["specs"][-1], "path": str(manifest)}
    version = register([spec], parent=replay, workspace=corpus["workspace"])
    added = read(version / "reports/result.json")["added"]
    assert added["observations"] == added["documents"] == 0
    assert added["views"] == 1
    assert set(data["views"]) <= set(read(version / "reports/registry.json")["views"])


@pytest.mark.parametrize("conflict", ["label", "group", "split"])
def test_conflicting_historical_identity_is_rejected(corpus, conflict):
    parent = initial(corpus)
    before = sha256(parent / "run.json")
    kind = "documents" if conflict == "group" else "words"
    rows = copy.deepcopy(corpus["rows"][kind])
    if conflict == "label":
        rows[0]["text"] = "ac"
    elif conflict == "group":
        rows[0]["split_group"] = "another-session"
    else:
        rows[0]["dataset_split"] = "validation"
    path = corpus["root"] / "conflict.json"
    atomic_json(path, rows)
    spec = next(s for s in corpus["specs"] if s["kind"] == kind)
    with pytest.raises(ValueError, match="Conflicting|conflicts"):
        register([{**spec, "path": str(path)}], parent=parent, workspace=corpus["workspace"])
    assert sha256(parent / "run.json") == before
    assert verify_run(parent)["status"] == "passed"


@pytest.mark.parametrize("corruption", ["missing", "hash", "pixels", "manifest"])
def test_source_provenance_failure_rejects_registration(corpus, corruption):
    parent = initial(corpus)
    before = sha256(parent / "run.json")
    specs = copy.deepcopy(corpus["specs"])
    if corruption == "missing":
        Path(corpus["rows"]["words"][0]["crop"]).unlink()
    elif corruption == "hash":
        Path(corpus["rows"]["words"][0]["crop"]).write_bytes(b"changed source")
    elif corruption == "pixels":
        word = copy.deepcopy(corpus["rows"]["words"][0])
        bad = corpus["root"] / "different-pixels.png"
        Image.new("RGB", (8, 8), "red").save(bad)
        word.update(crop=str(bad), crop_sha256=sha256(bad))
        path = corpus["root"] / "bad-word.json"
        atomic_json(path, [word])
        specs = [{**corpus["specs"][2], "path": str(path)}]
    else:
        specs[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Missing|mismatch|pixels"):
        register(specs, parent=parent, workspace=corpus["workspace"])
    assert sha256(parent / "run.json") == before
    assert verify_run(parent)["status"] == "passed"  # External assets checked by registration.


def test_parent_snapshot_tampering_cannot_be_inherited(corpus):
    parent = initial(corpus)
    snapshot = next((parent / "solutions/manifests").glob("*.json"))
    snapshot.write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        register(corpus["specs"], parent=parent, workspace=corpus["workspace"])


def test_sealed_source_and_undeclared_dpi_are_not_opened(corpus):
    spec = corpus["specs"][0]
    path = corpus["root"] / "sealed.json"
    atomic_json(path, [{"source_hash": next(iter(SEALED)), "source_path": "/must-not-read.pdf"}])
    with pytest.raises(ValueError, match="Sealed source"):
        register([{**spec, "path": str(path)}], workspace=corpus["workspace"])
    specs = [dict(s) for s in corpus["specs"]]
    for s in specs:
        s.pop("source_dpi")
    with pytest.raises(ValueError, match="DPI"):
        register(specs, workspace=corpus["workspace"])


def test_held_math_review_retained_without_approval(corpus):
    row = corpus["rows"]["words"][0]
    math = {
        "source_sha256": row["source_hash"],
        "pdf": corpus["rows"]["documents"][0]["source_path"],
        "page": 1,
        "capture_session": "fixture",
        "character": "∑",
        "box": [102, 201, 110, 209],
        "parent_box": row["parent_bbox"],
        "render_dpi": 300,
        "parent_300dpi_crop": row["source_crop"],
        "parent_300dpi_sha256": row["source_crop_sha256"],
        "crop": row["crop"],
        "crop_sha256": row["crop_sha256"],
        "crop_to_pdf": [[0.24, 0, 24.48], [0, -0.24, 800], [0, 0, 1]],
        "raster_to_pdf": [[0.24, 0, 0], [0, -0.24, 842], [0, 0, 1]],
        "review_status": "held",
        "human_review": False,
    }
    path = corpus["root"] / "math.json"
    atomic_json(path, [math])
    run = register(
        [{"kind": "math", "path": str(path), "split": "calibration"}], workspace=corpus["workspace"]
    )
    raw = read(next((run / "solutions/manifests").glob("*.json")))[0]
    assert raw == math
    assert raw["review_status"] == "held" and raw["human_review"] is False
    assert read(run / "run.json")["config"]["label_approval_performed"] is False
    data = read(run / "reports/registry.json")
    view = next(iter(data["views"].values()))
    assert view["geometry"]["crop_to_pdf"] == math["crop_to_pdf"]
    assert all(not member["independent_holdout_claimed"] for member in data["memberships"].values())
    quality = read(run / "reports/quality.json")
    assert quality["math_label_review_counts"] == {"held-or-unresolved": 1}
    assert not next(iter(quality["views"].values()))["training_ready"]


def test_distinct_words_from_one_group_expose_split_leakage(corpus):
    parent = initial(corpus)
    old_hash = sha256(parent / "run.json")
    word = copy.deepcopy(corpus["rows"]["words"][0])
    crop = corpus["root"] / "second-word.png"
    with Image.open(word["source_crop"]) as image:
        image.crop((11, 1, 19, 9)).save(crop)
    word.update(
        text="cd",
        box_in_source_crop=[11, 1, 19, 9],
        crop=str(crop),
        crop_sha256=sha256(crop),
        dataset_split="validation",
    )
    path = corpus["root"] / "validation-word.json"
    atomic_json(path, [word])
    # Different view/word, same source group and historical dataset identity.
    spec = {**corpus["specs"][2], "path": str(path), "split": "validation"}
    run = register([spec], parent=parent, workspace=corpus["workspace"])
    quality = read(run / "reports/quality.json")
    assert quality["group_split_gate"] == "no-go"
    assert quality["group_split_leakage"]
    assert all(
        set(item["splits"]) == {"train", "validation"} for item in quality["group_split_leakage"]
    )
    assert all(not v["training_ready"] for v in quality["views"].values())
    data = read(run / "reports/registry.json")
    assert {m["split"] for m in data["memberships"].values()} == {"train", "validation"}
    assert sha256(parent / "run.json") == old_hash
    assert read(run / "run.json")["status"] == "held"
