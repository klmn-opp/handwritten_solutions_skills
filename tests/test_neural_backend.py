from __future__ import annotations

import io

import numpy as np
from pypdf import PdfReader

from hwsol.jobs.schema import Background
from hwsol.neural.bundle import load_bundle, sha256
from hwsol.neural.english import EnglishGenerator
from hwsol.neural.math import MathGenerator, topology
from hwsol.profiles.store import ProfileStore


def test_frozen_bundle_inventory_and_hashes():
    bundle = load_bundle()
    assert bundle.manifest["schema"] == "nextgen-model-bundle/1"
    assert bundle.manifest["training_updates_on_export"] == 0
    assert len(bundle.english) == 34 and len(bundle.math) == 38
    assert len(bundle.english_glyphs["images"]) == 951
    assert len(bundle.math_glyphs) == 755
    for name, digest in bundle.manifest["files"].items():
        assert sha256(bundle.root / name) == digest


def test_english_draw_is_seeded_and_shape_changing():
    bundle = load_bundle()
    generator = EnglishGenerator(bundle)
    base = bundle.english_glyphs["images"][0]
    label = str(bundle.english_glyphs["labels"][0])
    first = generator.draw(base, label, 12345)
    repeat = generator.draw(base, label, 12345)
    other = generator.draw(base, label, 54321)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, other)
    assert first.shape == (80, 80) and first.dtype == np.uint8


def test_math_warp_preserves_topology_or_falls_back():
    bundle = load_bundle()
    generator = MathGenerator(bundle)
    row = next(r for r in bundle.math_glyphs if r["role"] == "ordinary" and r["label"] == "ρ")
    from PIL import Image

    with Image.open(bundle.root / "glyphs" / row["file"]) as image:
        gray = np.asarray(image.convert("L")).copy()
    warped, diagnostics = generator.warp_native(np.pad(gray, 3, constant_values=255), "ρ", 7)
    cropped = warped[3:-3, 3:-3]
    assert diagnostics["guard"] is True
    assert topology(cropped) == topology(gray)
    assert generator.predict(gray)["label"] == "ρ"


def test_neural_engine_renders_one_page_with_routes():
    profile = ProfileStore().resolve("author-default@v1")
    from hwsol.neural.engine import NextGenEngine

    engine = NextGenEngine(profile)
    source = "The result is $x=1$."
    blob, diagnostics = engine.render_one(
        source, "1", profile, profile.manifest.render, Background(kind="plain"), None
    )
    assert len(PdfReader(io.BytesIO(blob), strict=True).pages) == 1
    candidate = diagnostics["candidate"]
    assert candidate["backend"] == "personal-neural@1"
    assert candidate["model_bundle"] == "nextgen-v1"
    assert candidate["neural_weight_updates"] == 0
    assert sum(candidate["glyph_routes"].values()) == candidate["items"] > 0
    assert candidate["glyph_routes"].get("english", 0) > 0
    assert set(candidate["page_contours"]) == {"128", "160", "192"}
    assert candidate["items_detail"]


def test_overlapping_tiles_keep_per_item_ownership():
    import numpy as np

    from hwsol.neural.compose import areas_from_page

    page = np.full((12, 12), 255, np.uint8)
    left = np.full((4, 8), 255, np.uint8)
    left[:, :5] = 0
    right = np.full((4, 8), 255, np.uint8)
    right[:, 3:] = 0
    page[4:8, 2:10] = np.minimum(left, right)
    items = [
        {
            "id": 0,
            "box": [2, 4, 10, 8],
            "stable": left,
            "placed": left,
            "kind": "math",
            "role": "ordinary",
            "semantic": "0",
            "latex": "0",
        },
        {
            "id": 1,
            "box": [2, 4, 10, 8],
            "stable": right,
            "placed": right,
            "kind": "math",
            "role": "ordinary",
            "semantic": "1",
            "latex": "1",
        },
    ]
    areas = areas_from_page(page, items)
    # The left-only and right-only pixels stay exclusive; shared pixels are excluded.
    assert int(areas[0]["mask"].sum()) == 12
    assert int(areas[1]["mask"].sum()) == 12


def test_neural_engine_renders_radical_after_ownership_fix():
    profile = ProfileStore().resolve("author-default@v1")
    from hwsol.neural.engine import NextGenEngine

    engine = NextGenEngine(profile)
    source = "We evaluate the square root directly.\n\n$$\\sqrt{0.36}=0.6$$\n"
    blob, diagnostics = engine.render_one(
        source, "2", profile, profile.manifest.render, Background(kind="plain"), None
    )
    assert len(PdfReader(io.BytesIO(blob), strict=True).pages) == 1
    candidate = diagnostics["candidate"]
    assert candidate["backend"] == "personal-neural@1"
    assert candidate["neural_weight_updates"] == 0
    labels = {item["label"]: item["route"] for item in candidate["items_detail"]}
    assert labels["0"] in {"layout", "identity-rejected", "topology-fallback"}
    assert labels["√"] == "layout"
    assert candidate["glyph_routes"].get("math-neural", 0) >= 1


def test_foreign_math_and_text_symbols_become_personal():
    profile = ProfileStore().resolve("author-default@v1")
    from hwsol.neural.engine import NextGenEngine

    engine = NextGenEngine(profile)
    source = "A relation $a \\neq b$ and $50\\%$ of a sample; prose 50% is here.\n"
    blob, diagnostics = engine.render_one(
        source, "4", profile, profile.manifest.render, Background(kind="plain"), None
    )
    assert len(PdfReader(io.BytesIO(blob), strict=True).pages) == 1
    candidate = diagnostics["candidate"]
    assert candidate["foreign_sources"] == []
    assert candidate["neural_weight_updates"] == 0
    by_label = {}
    for item in candidate["items_detail"]:
        by_label.setdefault(item["label"], []).append(item)
    assert by_label["≠"][0]["source_type"] == "personal-o-atom"
    assert by_label["≠"][0]["route"] in {"math-neural", "personal-atom"}
    percents = by_label["%"]
    assert len(percents) == 2
    assert all(item["source_type"] == "personal-o-atom" for item in percents)
    assert all(
        item["route"] in {"math-neural", "personal-atom", "personal-atom-text"}
        for item in percents
    )
    assert any(item["kind"] == "math" for item in percents)
    assert any(item["route"] == "personal-atom-text" for item in percents)
