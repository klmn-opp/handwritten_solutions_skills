"""Versioned, portable Profile contracts. Publication state lives outside snapshots."""

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hwsol.config import RenderConfig

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Version = Annotated[str, Field(pattern=r"^v[1-9][0-9]*$")]


class ProfileError(ValueError):
    """Profile input, integrity, or lifecycle error."""


def name(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value):
        raise ProfileError("Profile name must contain lowercase letters, digits or hyphens")
    if value in {
        "con",
        "prn",
        "aux",
        "nul",
        *[f"com{i}" for i in range(10)],
        *[f"lpt{i}" for i in range(10)],
    }:
        raise ProfileError("Profile name is reserved on Windows")
    return value


def relative(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or re.search(r'[<>:"|?*\x00-\x1f]', value)
        or any(
            part.endswith((".", " "))
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[0-9¹²³]|LPT[0-9¹²³])(?:\..*)?", part, re.I)
            for part in value.split("/")
        )
        or any(p in {"", ".", ".."} for p in value.split("/"))
    ):
        raise ProfileError(f"Unsafe asset path: {value}")
    return value


def reference(value: str) -> str:
    parts = value.split("@")
    if len(parts) != 2 or not re.fullmatch(r"v[1-9][0-9]*", parts[1]):
        raise ProfileError("Expected a pinned profile@vN reference")
    name(parts[0])
    return value


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Asset(FrozenModel):
    path: str
    sha256: Digest
    role: Literal[
        "font",
        "license",
        "provenance",
        "calibration",
        "trajectory",
        "glyph-metrics",
        "model-adapter",
        "personal-ink",
    ]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        if value == "profile.json":
            raise ProfileError("profile.json cannot be an asset")
        return relative(value)


class SampleIndex(FrozenModel):
    id: str = Field(pattern=r"^s-[a-f0-9]{32}$")
    sha256: Digest
    included: bool


class ProfileVersion(FrozenModel):
    schema_version: Literal[1] = 1
    profile: str
    version: Version
    parent: str | None = None
    base: str | None = None
    created_at: str
    description: str
    source_kind: Literal["test-font", "personal-assets", "derived"]
    distribution: Literal["OFL-1.1", "local-only"] = "local-only"
    assets: tuple[Asset, ...]
    font: str
    coverage: tuple[int, ...]
    variants: dict[str, int] = Field(default_factory=dict)
    personal_codepoints: tuple[int, ...] = ()
    samples: tuple[SampleIndex, ...] = ()
    render: RenderConfig = Field(default_factory=RenderConfig)
    build: dict = Field(default_factory=dict)
    ink_bundle: str | None = None

    _name = field_validator("profile")(name)
    _font = field_validator("font")(relative)

    @field_validator("ink_bundle")
    @classmethod
    def safe_ink_bundle(cls, value: str | None) -> str | None:
        return relative(value) if value is not None else None

    @field_validator("parent", "base")
    @classmethod
    def pinned(cls, value: str | None) -> str | None:
        return reference(value) if value is not None else None

    @model_validator(mode="after")
    def consistency(self) -> "ProfileVersion":
        paths = [asset.path for asset in self.assets]
        if sum(a.role == "calibration" for a in self.assets) > 1:
            raise ProfileError("Only one cumulative calibration asset per snapshot")
        if len(set(p.casefold() for p in paths)) != len(paths):
            raise ProfileError("Duplicate asset paths")
        if not any(a.path == self.font and a.role == "font" for a in self.assets):
            raise ProfileError("Profile font must be a hashed font asset")
        ink_assets = [a for a in self.assets if a.role == "personal-ink"]
        if [a.path for a in ink_assets] != ([self.ink_bundle] if self.ink_bundle else []):
            raise ProfileError("Personal ink requires exactly one matching hashed recipe asset")
        if self.ink_bundle and self.distribution != "local-only":
            raise ProfileError("Experimental personal ink assets must remain local-only")
        if list(self.coverage) != sorted(set(self.coverage)) or any(
            cp < 0 or cp > 0x10FFFF or 0xD800 <= cp <= 0xDFFF for cp in self.coverage
        ):
            raise ProfileError("Coverage must be sorted unique Unicode scalar values")
        if not set(self.personal_codepoints) <= set(self.coverage):
            raise ProfileError("Personal coverage must be a subset of font coverage")
        if any(
            not key.isdecimal() or int(key) not in self.coverage or count < 1
            for key, count in self.variants.items()
        ):
            raise ProfileError("Invalid variant counts")
        return self

    @property
    def ref(self) -> str:
        return f"{self.profile}@{self.version}"
