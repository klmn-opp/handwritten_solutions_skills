"""Synthetic contour security/roundtrip checks; no personal data in test fixtures."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from hwsol.personal_ink.bundle import Recipe
from hwsol.personal_ink.compiled import compile_glyph, load_compiled_outline
from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph
from hwsol.profiles.io import sha256
from hwsol.profiles.schema import ProfileError


@pytest.fixture
def source_glyph(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    image = Image.new("RGB", (50, 60), "white")
    ink = ImageDraw.Draw(image)
    ink.ellipse((8, 7, 40, 52), fill=(20, 30, 40))
    ink.ellipse((16, 15, 32, 44), fill="white")
    image.save(original / "source.png")
    row = {
        "id": "synthetic-ring",
        "character": "O",
        "review_status": "agent-accepted",
        "clean_isolated_ink": True,
        "original_pixels_edited": False,
        "crop": "source.png",
        "crop_sha256": sha256(original / "source.png"),
        "contour_luminance": 160,
        "source_sha256": "0" * 64,
        "pdf": "/home/private/source.pdf",
        "parent_text": "Never distribute this note",
        "source_page": {"raw_path": "/home/private/page.png"},
    }
    return original, row


def test_compilation_preserves_every_contour_and_excludes_raw_context(tmp_path, source_glyph):
    original, row = source_glyph
    output = tmp_path / "compiled"
    output.mkdir()
    before = ScanGlyph.load(original, row)
    compiled = compile_glyph(original, row, output, 1)
    after = ScanGlyph.load(output, compiled)
    assert np.array_equal(before.outline.verts, after.outline.verts)
    assert np.array_equal(before.outline.codes, after.outline.codes)
    assert list(output.rglob("*.png")) == []
    text = json.dumps(compiled)
    assert "/home/" not in text and "Never distribute" not in text
    assert "source_page" not in compiled and "crop" not in compiled
    assert compiled["crop_sha256"] == row["crop_sha256"]
    assert compiled["raw_source_included"] is False
    # A hole is preserved, not replaced by a filled bounding rectangle.
    assert np.count_nonzero(after.outline.codes == 1) == 2
    assert sha256(original / "source.png") == row["crop_sha256"]


def test_compiled_hash_is_checked_before_loading(tmp_path, source_glyph):
    original, row = source_glyph
    output = tmp_path / "compiled"
    output.mkdir()
    compiled = compile_glyph(original, row, output, 1)
    with (output / compiled["outline"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ProfileError, match="integrity"):
        load_compiled_outline(output, compiled)


def test_compiled_paths_cannot_escape_the_bank(tmp_path, source_glyph):
    original, row = source_glyph
    output = tmp_path / "compiled"
    output.mkdir()
    compiled = compile_glyph(original, row, output, 1)
    compiled["outline"] = "../outside.npz"
    with pytest.raises(ProfileError, match="Unsafe"):
        load_compiled_outline(output, compiled)


@pytest.mark.parametrize("defect", ["nan", "codes", "dimensions", "object", "extra"])
def test_compiled_invalid_arrays_are_rejected(tmp_path, source_glyph, defect):
    original, row = source_glyph
    output = tmp_path / "compiled"
    output.mkdir()
    compiled = compile_glyph(original, row, output, 1)
    path = output / compiled["outline"]
    outline = load_compiled_outline(output, compiled)
    arrays = {"vertices": outline.verts.copy(), "codes": outline.codes.copy()}
    if defect == "nan":
        arrays["vertices"][2, 0] = np.nan
    elif defect == "codes":
        arrays["codes"][1] = 200
    elif defect == "dimensions":
        arrays["vertices"] = arrays["vertices"].ravel()
    elif defect == "object":
        arrays["vertices"] = np.array([{"unsafe": "object"}], dtype=object)
    else:
        arrays["unknown"] = np.zeros(3)
    np.savez_compressed(path, **arrays)
    compiled["outline_sha256"] = sha256(path)
    with pytest.raises(ValueError):
        load_compiled_outline(output, compiled)


def test_representation_version_cannot_silently_change_the_original_contract():
    values = {
        "schema": "personal-cpu-recipe/1",
        "recipe": "compiled-outlines/1",
        "base_profile": "base",
        "base_profile_sha256": "0" * 64,
        "stroke_profile": "strokes",
        "stroke_profile_sha256": "0" * 64,
        "calibrated_font_sha256": "0" * 64,
        "banks": {
            key: f"banks/{key}"
            for key in ("legacy", "sessions", "historical", "rules", "f_variants", "session_o")
        },
        "bindings": {
            "font_sha256": "0" * 64,
            "digit_sample_ids": ["a", "b"],
            "horizontal_source_ids": ["bar"],
            "horizontal_source_sha256": "0" * 64,
            "single_bar_source_id": "bar",
        },
        "admission": {},
    }
    with pytest.raises(ValueError, match="representation/version"):
        Recipe.model_validate(values)
    values["schema"] = "personal-cpu-recipe/2"
    assert Recipe.model_validate(values).recipe == "compiled-outlines/1"


def test_large_declared_array_is_rejected_before_numpy_load(tmp_path, monkeypatch):
    import io
    import zipfile

    path = tmp_path / "oversized.npz"
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": "<f8",
            "fortran_order": False,
            "shape": (2**40, 2),
        },
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("vertices.npy", header.getvalue())
        archive.writestr("codes.npy", b"unused")
    source = {
        "representation": "compiled-contour/1",
        "outline": path.name,
        "outline_sha256": sha256(path),
    }

    def must_not_allocate(*args, **kwargs):
        raise AssertionError("np.load must not be called for oversized declared shapes")

    monkeypatch.setattr(np, "load", must_not_allocate)
    with pytest.raises(ProfileError, match="array header"):
        load_compiled_outline(tmp_path, source)


def test_historical_compiled_aliases_cannot_hide_an_undeclared_live_asset(tmp_path):
    import shutil

    from hwsol.personal_ink.bundle import load_recipe
    from hwsol.profiles.io import atomic_json
    from hwsol.profiles.store import ProfileStore, copy_snapshot, load_snapshot

    base = ProfileStore(tmp_path / "store").resolve("bundled-default@v1")
    folder = tmp_path / "candidate"
    folder.mkdir()
    copy_snapshot(base, folder / "base")
    copy_snapshot(base, folder / "strokes")
    shutil.copyfile(base.font, folder / "font.ttf")
    bank = folder / "banks/historical"
    bank.mkdir(parents=True)
    atomic_json(
        bank / "manifest.json",
        {
            "sources": [
                {
                    "representation": "compiled-contour/1",
                    "outline": "undeclared.npz",
                    "outline_sha256": "1" * 64,
                    "contour_archive": "declared.npz",
                    "contour_archive_sha256": "2" * 64,
                    "reviewed_component_selection": "selection.json",
                }
            ]
        },
    )
    recipe = {
        "schema": "personal-cpu-recipe/2",
        "recipe": "compiled-outlines/1",
        "base_profile": "base",
        "base_profile_sha256": sha256(folder / "base/profile.json"),
        "stroke_profile": "strokes",
        "stroke_profile_sha256": sha256(folder / "strokes/profile.json"),
        "calibrated_font_sha256": sha256(base.font),
        "banks": {
            key: "banks/historical"
            for key in ("legacy", "sessions", "historical", "rules", "f_variants", "session_o")
        },
        "bindings": {
            "font_sha256": sha256(base.font),
            "digit_sample_ids": ["a", "b"],
            "horizontal_source_ids": ["bar"],
            "horizontal_source_sha256": "0" * 64,
            "single_bar_source_id": "bar",
        },
        "admission": {},
    }
    # Earlier banks must be traversable before reaching the historical case.
    atomic_json(bank / "sources.json", [])
    atomic_json(folder / "ink.json", recipe)
    manifest = base.manifest.model_dump(mode="json")
    manifest.update(
        profile="synthetic-compiled",
        distribution="local-only",
        font="font.ttf",
        ink_bundle="ink.json",
    )
    manifest["assets"] = [
        {
            "path": path.relative_to(folder).as_posix(),
            "sha256": sha256(path),
            "role": "font"
            if path.name == "font.ttf"
            else ("personal-ink" if path.name == "ink.json" else "provenance"),
        }
        for path in folder.rglob("*")
        if path.is_file()
    ]
    atomic_json(folder / "profile.json", manifest)
    with pytest.raises(ProfileError, match="aliases differ"):
        load_recipe(load_snapshot(folder))
