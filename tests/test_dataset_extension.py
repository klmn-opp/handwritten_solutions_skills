import json

import numpy as np
import pytest
from PIL import Image

from hwsol.profiles.io import atomic_json, sha256
from research.corpus.dataset_extension import certify_extension, overlaps_same_ink
from research.corpus.reference_lineage import verify_continuation


def test_physical_dedup_across_render_dpi():
    old = {"source_hash": "pdf", "page_id": "legacy", "bounding_box": [10, 20, 30, 60]}
    new = {
        "source_hash": "pdf",
        "source_page_number": 3,
        "source_page_size": [200, 400],
        "bounding_box": [20, 40, 60, 120],
    }
    pages = {"legacy": {"page_number": 3, "width": 100, "height": 200}}
    assert overlaps_same_ink(old, new, pages)
    new["source_page_number"] = 4
    assert not overlaps_same_ink(old, new, pages)


def dataset_view(path, *, extra=False):
    path.mkdir()
    image = path / "font.png"
    ink = np.zeros((96, 96), np.float32)
    ink[20:80, 45:50] = 1
    Image.fromarray(np.uint8((1 - ink) * 255)).save(image)
    records = [
        {
            "sample_id": "font-1",
            "label": "1",
            "source_type": "personal-font-derived",
            "dataset_split": "auxiliary-train-only",
            "external_fallback": False,
            "is_real_scan": False,
            "image": str(image),
            "sha256": sha256(image),
        }
    ]
    targets = [ink]
    if extra:
        records.append(
            {
                "label": "1",
                "source_type": "personal-real",
                "dataset_split": "train",
                "original_record": {"sample_id": "new-real", "label": "1"},
            }
        )
        targets.append(np.roll(ink, 1, axis=1))
    np.savez_compressed(
        path / "arrays.npz",
        reference=ink[None],
        target=np.stack(targets),
        validation=np.zeros((0, 96, 96), np.float32),
    )
    atomic_json(path / "records.json", records)
    atomic_json(path / "validation.json", [])
    manifest = {
        "schema": "reference-glyph-views/1",
        "holdout_in_arrays": False,
        "labels": ["1"],
        "profile": "test@v1",
        "source_files": {},
        "files": {p.name: sha256(p) for p in path.iterdir() if p.name != "font.png"},
    }
    atomic_json(path / "manifest.json", manifest)


def test_certified_growth_allows_only_matching_checkpoint_parent(tmp_path):
    parent, child = tmp_path / "old", tmp_path / "new"
    dataset_view(parent)
    dataset_view(child, extra=True)
    certify_extension(parent, child)
    certify_extension(parent, child, write=False)
    request = {
        "labels": ["1"],
        "data": str(child),
        "arrays_sha256": sha256(child / "arrays.npz"),
        "manifest_sha256": sha256(child / "manifest.json"),
        "allow_data_extension": False,
    }
    checkpoint = {
        "labels": ["1"],
        "arrays_sha256": sha256(parent / "arrays.npz"),
        "dataset_manifest_sha256": sha256(parent / "manifest.json"),
    }
    with pytest.raises(ValueError, match="without explicit"):
        verify_continuation(request, checkpoint)
    request["allow_data_extension"] = True
    verify_continuation(request, checkpoint)
    checkpoint["dataset_manifest_sha256"] = "different-parent"
    with pytest.raises(ValueError, match="certified parent"):
        verify_continuation(request, checkpoint)


def test_growth_rejects_old_ink_or_label_replacement_even_with_new_hashes(tmp_path):
    parent, child = tmp_path / "old", tmp_path / "new"
    dataset_view(parent)
    dataset_view(child, extra=True)
    arrays = dict(np.load(child / "arrays.npz"))
    arrays["target"][0] = np.roll(arrays["target"][0], 1, axis=0)
    np.savez_compressed(child / "arrays.npz", **arrays)
    manifest = json.loads((child / "manifest.json").read_text())
    manifest["files"]["arrays.npz"] = sha256(child / "arrays.npz")
    atomic_json(child / "manifest.json", manifest)
    with pytest.raises(ValueError, match="reprocessed"):
        certify_extension(parent, child)
