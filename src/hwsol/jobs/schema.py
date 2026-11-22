"""The editable, replayable contract shared by the Agent and local Runtime."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from hwsol.config import RenderConfig
from hwsol.profiles.schema import Digest, relative

Realism = Literal["baseline", "layout-enhanced", "stroke-enhanced", "combined"]

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Record):
    path: Text
    page: int | None = Field(None, ge=1)
    location: str = ""
    sha256: Digest | None = None


class Background(Record):
    kind: Literal["plain", "lined", "grid", "dots", "image", "pdf"] = "plain"
    path: str | None = None
    page: int = Field(1, ge=1)
    fit: Literal["contain", "cover", "stretch"] = "contain"

    @model_validator(mode="after")
    def valid_asset(self):
        if bool(self.path) != (self.kind in {"image", "pdf"}):
            raise ValueError("Provide a background file for image or pdf paper")
        if self.page != 1 and self.kind != "pdf":
            raise ValueError("Only PDF backgrounds accept a page selection")
        return self


class Equality(Record):
    lhs: Text
    rhs: Text


class Verification(Record):
    status: Literal["not_checked", "inconclusive", "passed", "failed"] = "not_checked"
    evidence: str = ""
    checks: list[Equality] = Field(default_factory=list)

    @model_validator(mode="after")
    def explain_status(self):
        if self.status != "not_checked" and not self.evidence.strip() and not self.checks:
            raise ValueError("Verification status needs evidence")
        return self


class Problem(Record):
    id: Text | None = None
    number: Text
    sources: list[Source] = Field(min_length=1)
    statement: Text
    complete: bool = True
    solution: Text
    verification: Verification = Field(default_factory=Verification)
    background: Background | None = None


class Job(Record):
    schema_version: Literal[1]
    backend: Literal[
        "font-cpu", "personal-cpu-experimental", "personal-neural", "diffink-experimental"
    ] = "font-cpu"
    realism: Realism = "baseline"
    generated_by: Text
    language: Text = "en"
    profile: str | None = None
    profile_sha256: Digest | None = None
    background: Background | None = None
    render: RenderConfig = Field(default_factory=RenderConfig)
    problems: list[Problem] = Field(min_length=1)

    @model_validator(mode="after")
    def identities(self):
        if self.backend == "personal-cpu-experimental":
            if not self.profile or "@" not in self.profile:
                raise ValueError("Personal CPU candidate requires an explicit pinned Profile")
            if self.realism != "baseline":
                raise ValueError("Personal CPU candidate uses its reviewed ink recipe")
        ids = [p.id or p.number for p in self.problems]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate problem identity. Use distinct ids for repeated source numbers."
            )
        for p in self.problems:
            relative(p.solution)
            if not p.solution.lower().endswith(".md"):
                raise ValueError("Solution files must be Markdown")
            for src in p.sources:
                relative(src.path)
            if "\n" in p.number or any(c in p.number for c in "$\\"):
                raise ValueError("Problem numbers must be plain single-line labels")
        return self
