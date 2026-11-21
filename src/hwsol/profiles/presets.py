"""Installed author preset admission, separate from private Profile qualification.

Only the immutable package location can receive this release authority. A copied
or imported Profile, even one with the same name and bytes, remains subject to
the ordinary local qualification lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from hwsol.identity import build_identity, digest, runtime_text_capabilities
from hwsol.profiles.io import atomic_json, inside, sha256
from hwsol.profiles.schema import Digest, FrozenModel, ProfileError, reference

if TYPE_CHECKING:
    from hwsol.profiles.store import LoadedProfile

AUTHOR = "author-default"
AUTHOR_REF = f"{AUTHOR}@v1"


def author_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "assets/presets/author-default"


class ReleaseReceipt(FrozenModel):
    schema_: Literal["hwsol-bundled-author-release/1"] = Field(alias="schema")
    profile: Literal["author-default@v1"]
    profile_sha256: Digest
    recipe_sha256: Digest
    source_sha256: Digest
    handnote_source_sha256: Digest
    packages: dict[str, str] = Field(min_length=1)
    environments: list[dict] = Field(min_length=1, max_length=16)
    evidence_sha256: list[Digest] = Field(min_length=1, max_length=64)
    scope: Literal["packaged-compiled-author-preset"]

    _profile = field_validator("profile")(reference)


def is_installed_author(profile: LoadedProfile) -> bool:
    """A reserved name or admission flag is not evidence of package membership."""
    return (
        profile.manifest.ref == AUTHOR_REF
        and profile.directory == author_directory()
        and not profile.directory.is_symlink()
    )


def validate_author(profile: LoadedProfile) -> dict:
    from hwsol.personal_ink.bundle import load_recipe
    from hwsol.profiles.qualification import Environment
    from hwsol.profiles.store import load_snapshot

    if not is_installed_author(profile):
        raise ProfileError("Only the installed author preset can use bundled release authority")
    current = load_snapshot(author_directory())
    if current.manifest_sha256 != profile.manifest_sha256:
        raise ProfileError("Installed author Profile changed after resolution")
    receipt_path = inside(author_directory(), "release.json")
    if not receipt_path.is_file():
        raise ProfileError("Author preset has no release receipt; use a verified distribution")
    receipt = ReleaseReceipt.model_validate_json(receipt_path.read_bytes())
    recipe = load_recipe(current)
    if (
        recipe.description.recipe != "compiled-outlines/1"
        or receipt.profile_sha256 != current.manifest_sha256
        or receipt.recipe_sha256 != recipe.sha256
    ):
        raise ProfileError("Installed author preset differs from its compiled release receipt")
    build = build_identity()
    if (
        digest(build["source"]) != receipt.source_sha256
        or digest(build["handnote_source"]) != receipt.handnote_source_sha256
        or build["packages"] != receipt.packages
    ):
        raise ProfileError("Author preset runtime changed; install the matching verified release")
    environment = Environment.model_validate(runtime_text_capabilities(current.font))
    supported = [Environment.model_validate(row) for row in receipt.environments]
    if len({digest(row.model_dump()) for row in supported}) != len(supported):
        raise ProfileError("Author preset release repeats an environment")
    if environment not in supported:
        raise ProfileError("Author preset text environment is not verified by this release")
    return {
        "strategy": "personal-cpu-qualified",
        "profile_sha256": current.manifest_sha256,
        "recipe_sha256": recipe.sha256,
        "build_commit": build["commit"],
        "environment": environment.model_dump(),
        "scope": receipt.scope,
        "release_receipt_sha256": sha256(receipt_path),
    }


def write_release_receipt(
    profile: LoadedProfile,
    *,
    environments: list[dict],
    evidence_sha256: list[str],
    output: Path,
) -> dict:
    """Seal maintainer-supplied actual probes after reviewing their evidence.

    This records evidence identities, not proof that the observations are true.
    It does not perform native tests or approve a release. The caller must freeze
    source and review real platform results before distributing the receipt.
    """
    from hwsol.personal_ink.bundle import load_recipe
    from hwsol.profiles.qualification import Environment

    if profile.manifest.ref != AUTHOR_REF:
        raise ProfileError("Release receipt must name author-default@v1")
    if output.exists():
        raise ProfileError("Release receipt already exists; preserve the previous evidence")
    recipe = load_recipe(profile)
    if recipe.description.recipe != "compiled-outlines/1":
        raise ProfileError("Only compiled outlines can enter the bundled author release")
    measured = [Environment.model_validate(row).model_dump() for row in environments]
    if len({digest(row) for row in measured}) != len(measured):
        raise ProfileError("Duplicate release environment")
    build = build_identity()
    receipt = ReleaseReceipt(
        schema="hwsol-bundled-author-release/1",
        profile=AUTHOR_REF,
        profile_sha256=profile.manifest_sha256,
        recipe_sha256=recipe.sha256,
        source_sha256=digest(build["source"]),
        handnote_source_sha256=digest(build["handnote_source"]),
        packages=build["packages"],
        environments=measured,
        evidence_sha256=evidence_sha256,
        scope="packaged-compiled-author-preset",
    )
    result = receipt.model_dump(mode="json", by_alias=True)
    atomic_json(output, result)
    return result
