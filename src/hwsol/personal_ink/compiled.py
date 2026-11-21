"""Lossless runtime contours without original scans or private operational metadata.

Compilation is a local data transformation. It grants no publication, activation,
redistribution or qualification authority. The original recipe remains immutable.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from handnote.vector import Outline
from matplotlib.path import Path as MplPath

from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import ProfileError, name, reference
from hwsol.profiles.store import LoadedProfile, load_snapshot

REPRESENTATION = "compiled-contour/1"
# These fields describe glyph identity, geometry and source hashes, never documents.
SOURCE_FIELDS = frozenset(
    {
        "id",
        "character",
        "review_status",
        "clean_isolated_ink",
        "original_pixels_edited",
        "crop_sha256",
        "source_sha256",
        "source_pdf_sha256",
        "contour_luminance",
        "role",
        "seam_x",
        "capture_session",
        "actual_visible_label",
        "review_approval_sha256",
        "prompt_is_ground_truth",
    }
)
PRIVATE_REFERENCE = re.compile(r"/home/|/Users/|workspace/runs/|source_pdf/|file://|[A-Za-z]:\\")


def load_compiled_outline(directory: Path, source: dict) -> Outline:
    """Read a hash-bound finite contour; NPZ never enables pickle/object loading."""
    if source.get("representation") != REPRESENTATION:
        raise ProfileError("Expected an explicit compiled contour representation")
    path = inside(directory, source["outline"])
    if sha256(path) != source["outline_sha256"]:
        raise ProfileError("Compiled contour integrity check failed")
    with zipfile.ZipFile(path) as archive:
        if (
            sorted(archive.namelist()) != ["codes.npy", "vertices.npy"]
            or sum(item.file_size for item in archive.infolist()) > 32 * 1024**2
        ):
            raise ProfileError("Invalid or oversized compiled contour archive")
        counts = []
        for filename in ("vertices.npy", "codes.npy"):
            with archive.open(filename) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    raise ProfileError("Unsupported compiled contour array format")
                shape, _, dtype = reader(stream, max_header_size=4096)
                expected_dims = 2 if filename == "vertices.npy" else 1
                expected_dtype = np.float64 if expected_dims == 2 else np.uint8
                if (
                    dtype != expected_dtype
                    or len(shape) != expected_dims
                    or not 3 <= shape[0] <= 1_000_000
                    or (expected_dims == 2 and shape[1] != 2)
                ):
                    raise ProfileError("Invalid compiled contour array header")
                expected_bytes = shape[0] * (2 if expected_dims == 2 else 1) * dtype.itemsize
                if stream.tell() + expected_bytes != archive.getinfo(filename).file_size:
                    raise ProfileError("Compiled contour array size differs from its header")
                counts.append(shape[0])
        if counts[0] != counts[1]:
            raise ProfileError("Compiled vertices and codes have different lengths")
    with np.load(path, allow_pickle=False) as arrays:
        vertices, codes = arrays["vertices"], arrays["codes"]
    if (
        vertices.dtype != np.float64
        or vertices.ndim != 2
        or vertices.shape[1] != 2
        or not 3 <= len(vertices) <= 1_000_000
        or codes.dtype != np.uint8
        or codes.shape != (len(vertices),)
        or not np.isfinite(vertices).all()
        or not np.isin(codes, [1, 2, 3, 4, 79]).all()
        or codes[0] != MplPath.MOVETO
        or codes[-1] != MplPath.CLOSEPOLY
    ):
        raise ProfileError("Invalid compiled contour geometry")
    return Outline(vertices, codes, stroke=False)


def _source_fields(row: dict) -> dict:
    result = {key: row[key] for key in sorted(SOURCE_FIELDS & row.keys())}
    if row.get("body_reference"):
        body = row["body_reference"]
        result["body_reference"] = {
            "role": body["role"],
            "bounds_crop_px": body["bounds_crop_px"],
        }
    result["representation"] = REPRESENTATION
    result["raw_source_included"] = False
    return result


def compile_glyph(directory: Path, row: dict, output: Path, index: int) -> dict:
    """Freeze the exact contour returned by the established original-crop loader."""
    from hwsol.personal_ink.kernels.math_glyph_resolver import ScanGlyph

    glyph = ScanGlyph.load(directory, row)
    result = _source_fields(row)
    relative = f"outlines/{index:04d}.npz"
    target = inside(output, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        np.savez_compressed(stream, vertices=glyph.outline.verts, codes=glyph.outline.codes)
    result.update(outline=relative, outline_sha256=sha256(target))
    restored = load_compiled_outline(output, result)
    if not (
        np.array_equal(restored.verts, glyph.outline.verts)
        and np.array_equal(restored.codes, glyph.outline.codes)
    ):
        raise ProfileError("Compilation changed an original contour")
    return result


def _copy(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, output.open("xb") as dst:
        shutil.copyfileobj(src, dst)


def _nested(loaded: LoadedProfile, output: Path) -> None:
    output.mkdir(parents=True)
    data = loaded.manifest.model_dump(mode="json")
    assets = []
    for asset in loaded.manifest.assets:
        if asset.role not in {"font", "calibration", "license"}:
            continue
        _copy(inside(loaded.directory, asset.path), inside(output, asset.path))
        assets.append(asset.model_dump(mode="json"))
    data.update(
        assets=assets,
        description="Runtime font and measured layout; original samples retained privately",
        build={"kind": "compiled-runtime-font/1", "source_manifest_sha256": loaded.manifest_sha256},
    )
    atomic_json(output / "profile.json", data)
    load_snapshot(output)


def _historical(directory: Path, output: Path) -> None:
    from hwsol.personal_ink.kernels.historical_math_candidate import load_sources

    # Verify the private originals and reviewed selection before excluding them.
    load_sources(directory)
    original = json.loads((directory / "manifest.json").read_text())
    rows = []
    for index, row in enumerate(original["sources"]):
        result = _source_fields(row)
        contour = f"outlines/{index:04d}.npz"
        _copy(inside(directory, row["contour_archive"]), inside(output, contour))
        result.update(
            outline=contour,
            outline_sha256=sha256(inside(output, contour)),
            contour_archive=contour,
            contour_archive_sha256=sha256(inside(output, contour)),
        )
        selection_name = f"selections/{index:04d}.json"
        selection = json.loads(inside(directory, row["reviewed_component_selection"]).read_text())
        # Geometry/hash-only audit; the original full review stays private.
        public_selection = {
            key: selection[key]
            for key in ("level", "selected_components", "selected_component_ink_pixels")
            if key in selection
        }
        public_selection.update(
            original_review_sha256=row["reviewed_component_selection_sha256"],
            compiled_contour_sha256=result["outline_sha256"],
            original_pixels_included=False,
        )
        atomic_json(inside(output, selection_name), public_selection)
        result.update(
            reviewed_component_selection=selection_name,
            reviewed_component_selection_sha256=sha256(inside(output, selection_name)),
        )
        load_compiled_outline(output, result)
        rows.append(result)
    atomic_json(
        output / "manifest.json",
        {
            "schema": "compiled-historical-contours/1",
            "sources": rows,
            "selected": original["selected"],
        },
    )


def _variants(directory: Path, output: Path) -> None:
    rows = []
    _copy(directory / "source-sample.ttf", output / "source-sample.ttf")
    for index, row in enumerate(json.loads((directory / "candidates.json").read_text())):
        source = row["source"]
        fields = {
            key: source[key]
            for key in (
                "id",
                "kind",
                "original_label",
                "label",
                "original_sha256",
                "font_source_sha256",
                "source_glyph",
                "font_variant",
                "source_hash",
                "source_crop_sha256",
                "sha256",
            )
            if key in source
        }
        contour = f"outlines/{index:04d}.npz"
        path = inside(directory, row["outline"])
        if sha256(path) != row["outline_sha256"]:
            raise ProfileError("Source variant contour changed")
        _copy(path, inside(output, contour))
        # Keep only applied transform measurements. Full source-region descriptions
        # are private evidence and are not required by the runtime.
        geometry = {
            key: row["geometry"][key]
            for key in (
                "scale",
                "y_translation",
                "fit_rmse_font_units",
                "uniform_scale_no_stroke_dilation",
            )
            if key in row["geometry"]
        }
        rows.append(
            {
                "id": row["id"],
                "source": fields,
                "geometry": geometry,
                "outline": contour,
                "outline_sha256": sha256(inside(output, contour)),
            }
        )
    atomic_json(output / "candidates.json", rows)


def audit_compiled(directory: Path) -> dict:
    """Inventory a runtime-only derivative and reject raw images or private locators."""
    from hwsol.personal_ink.bundle import load_recipe

    loaded = load_snapshot(directory)
    recipe = load_recipe(loaded)
    if recipe.description.recipe != "compiled-outlines/1":
        raise ProfileError("Expected a compiled runtime recipe")
    allowed = {"profile.json", *(asset.path for asset in loaded.manifest.assets)}
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if actual != allowed:
        raise ProfileError("Compiled directory differs from its declared asset inventory")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ProfileError("Compiled assets cannot be symlinks")
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".npz", ".ttf", ".txt"}:
            raise ProfileError("Raw source or unexpected runtime asset")
        if path.suffix.lower() in {".json", ".txt"} and PRIVATE_REFERENCE.search(path.read_text()):
            raise ProfileError("Private operational locator in compiled metadata")
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema": "compiled-personal-runtime-audit/1",
        "profile": loaded.manifest.ref,
        "manifest_sha256": loaded.manifest_sha256,
        "files": files,
        "raw_scan_files": 0,
        "private_operational_locators": 0,
        "neural_weights": 0,
        "qualification_transferred": False,
    }


def compile_profile(
    source: Path, output: Path, *, profile: str = "author-default", version: str = "v1"
) -> dict:
    """Create a new local-only snapshot, preserving all original files unchanged."""
    from hwsol.personal_ink.bundle import load_recipe

    name(profile)
    reference(f"{profile}@{version}")
    loaded = load_snapshot(source)
    bundle = load_recipe(loaded)
    if bundle.description.recipe != "reviewed-outlines/1":
        raise ProfileError("Compile from an original reviewed recipe, not another derivative")
    output = output.expanduser().absolute()
    if output.exists():
        raise ProfileError("Compiled destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compiled-", dir=output.parent) as temporary:
        stage = Path(temporary) / "profile"
        stage.mkdir()
        _nested(bundle.base, stage / "ink/base-profile")
        _nested(bundle.strokes, stage / "ink/stroke-profile")
        banks = {}
        for key, directory in bundle.directories.items():
            relative = f"ink/banks/{key}"
            target = stage / relative
            target.mkdir(parents=True)
            banks[key] = relative
            if key == "historical":
                _historical(directory, target)
            elif key == "f_variants":
                _variants(directory, target)
            else:
                original = json.loads((directory / "sources.json").read_text())
                rows = [
                    compile_glyph(directory, row, target, index)
                    for index, row in enumerate(original)
                    if row["review_status"] == "agent-accepted"
                ]
                atomic_json(target / "sources.json", rows)
        recipe = bundle.description.model_dump(mode="json", by_alias=True)
        recipe.update(
            schema="personal-cpu-recipe/2",
            recipe="compiled-outlines/1",
            banks=banks,
            base_profile="ink/base-profile",
            stroke_profile="ink/stroke-profile",
            base_profile_sha256=sha256(stage / "ink/base-profile/profile.json"),
            stroke_profile_sha256=sha256(stage / "ink/stroke-profile/profile.json"),
            admission={
                "kind": "lossless-runtime-compilation/1",
                "source_profile_sha256": loaded.manifest_sha256,
                "source_recipe_sha256": bundle.sha256,
                "raw_sources_included": False,
                "qualification_transferred": False,
                "neural_parameter_updates": 0,
            },
        )
        atomic_json(stage / "ink/recipe.json", recipe)
        data = loaded.manifest.model_dump(mode="json")
        data.update(
            profile=profile,
            version=version,
            parent=None,
            description="Compiled personal runtime; exact reviewed contours, no original scans",
            distribution="local-only",
            font="ink/base-profile/font.ttf",
            ink_bundle="ink/recipe.json",
            build={
                "kind": "compiled-personal-runtime/1",
                "source_profile_sha256": loaded.manifest_sha256,
                "source_recipe_sha256": bundle.sha256,
                "neural_updates": 0,
                "publication_authority": False,
            },
        )
        data["assets"] = [
            {
                "path": path.relative_to(stage).as_posix(),
                "sha256": sha256(path),
                "role": "font"
                if path.relative_to(stage).as_posix() == data["font"]
                else (
                    "personal-ink"
                    if path.relative_to(stage).as_posix() == data["ink_bundle"]
                    else "provenance"
                ),
            }
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        ]
        atomic_json(stage / "profile.json", data)
        receipt = audit_compiled(stage)
        stage.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", default="author-default")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    receipt = compile_profile(args.source, args.output, profile=args.profile, version=args.version)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
