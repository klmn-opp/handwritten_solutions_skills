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
