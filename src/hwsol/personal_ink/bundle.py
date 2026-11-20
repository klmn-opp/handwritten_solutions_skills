"""Validate a data-only, asset-inventoried CPU recipe before any ink is loaded."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hwsol.profiles.io import inside, sha256
from hwsol.profiles.schema import Digest, ProfileError, relative
from hwsol.profiles.store import LoadedProfile, load_snapshot


class RecipeBindings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    font_sha256: Digest
    digit_sample_ids: list[str] = Field(min_length=2, max_length=2)
    horizontal_source_ids: list[str] = Field(min_length=1, max_length=16)
    horizontal_source_sha256: Digest
    single_bar_source_id: str = Field(min_length=1)


class Banks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    legacy: str
    sessions: str
    historical: str
    rules: str
    f_variants: str
    session_o: str

    _paths = field_validator("*")(relative)


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_: Literal["personal-cpu-recipe/1"] = Field(alias="schema")
    recipe: Literal["reviewed-outlines/1"]
    base_profile: str
    base_profile_sha256: Digest
    stroke_profile: str
    stroke_profile_sha256: Digest
    calibrated_font_sha256: Digest
    banks: Banks
    bindings: RecipeBindings
    admission: dict

    _profiles = field_validator("base_profile", "stroke_profile")(relative)


@dataclass(frozen=True)
class LoadedRecipe:
    description: Recipe
    candidate: LoadedProfile
    base: LoadedProfile
    strokes: LoadedProfile
    directories: dict[str, Path]
    sha256: str


def load_recipe(profile: LoadedProfile) -> LoadedRecipe:
    """Do not execute data or follow source-provenance paths outside the Profile.

    The kernel loaders retain historic manifests, so all their live path fields
    are checked here, before loading a contour or nested Profile. Other original
    source paths are historical provenance, never runtime dependencies.
    """
    if not profile.manifest.ink_bundle:
        raise ProfileError("Selected candidate has no personal ink recipe")
    root = profile.directory
    inventory = {a.path: a.sha256 for a in profile.manifest.assets}

    def registered(path: str) -> Path:
        target = inside(root, path)
        if path not in inventory:
            raise ProfileError(f"Ink dependency is absent from the Profile inventory: {path}")
        if not target.is_file() or sha256(target) != inventory[path]:
            raise ProfileError(f"Ink dependency failed integrity: {path}")
        return target

    descriptor = registered(profile.manifest.ink_bundle)
    spec = Recipe.model_validate_json(descriptor.read_text("utf-8"))

    def nested(folder: str, digest: str) -> LoadedProfile:
        metadata = registered(f"{folder}/profile.json")
        if sha256(metadata) != digest:
            raise ProfileError("Nested Profile identity changed")
        data = json.loads(metadata.read_text("utf-8"))
        if data.get("ink_bundle"):
            raise ProfileError("Nested personal ink recipes are not supported")
        for asset in data["assets"]:
            registered(f"{folder}/{relative(asset['path'])}")
        return load_snapshot(inside(root, folder))

    base = nested(spec.base_profile, spec.base_profile_sha256)
    strokes = nested(spec.stroke_profile, spec.stroke_profile_sha256)
    if not (
        sha256(profile.font)
        == sha256(base.font)
        == spec.calibrated_font_sha256
        == spec.bindings.font_sha256
    ):
        raise ProfileError("Personal ink recipe and selected font calibration do not match")
    if (
        profile.manifest.coverage != base.manifest.coverage
        or profile.manifest.variants != base.manifest.variants
        or profile.manifest.personal_codepoints != base.manifest.personal_codepoints
    ):
        raise ProfileError("Candidate font identity differs from calibrated base")

    directories = {}
    for name, folder in spec.banks.model_dump().items():
        directories[name] = inside(root, folder)
        if name == "historical":
            data = json.loads(registered(f"{folder}/manifest.json").read_text())
            for row in data["sources"]:
                for key in ("crop", "contour_archive", "reviewed_component_selection"):
                    registered(f"{folder}/{relative(row[key])}")
        elif name == "f_variants":
            data = json.loads(registered(f"{folder}/candidates.json").read_text())
            registered(f"{folder}/source-sample.ttf")
            for row in data:
                registered(f"{folder}/{relative(row['outline'])}")
        else:
            data = json.loads(registered(f"{folder}/sources.json").read_text())
            for row in data:
                if row["review_status"] == "agent-accepted":
                    registered(f"{folder}/{relative(row['crop'])}")
                    if not row.get("clean_isolated_ink") or row.get("original_pixels_edited"):
                        raise ProfileError("Only reviewed unedited isolated source ink is admitted")
    return LoadedRecipe(spec, profile, base, strokes, directories, sha256(descriptor))
