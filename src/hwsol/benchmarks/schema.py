import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hwsol.jobs.checks import arithmetic
from hwsol.jobs.schema import Equality
from hwsol.profiles.io import sha256


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Expected(Strict):
    outcome: Literal["render", "reject"]
    error_contains: str | None
    page_count: int = Field(ge=0, le=1)
    no_clipping: bool
    structures: list[str]


class ReferenceVerification(Strict):
    method: str
    checks: list[Equality]


class Case(Strict):
    id: str = Field(pattern=r"^(smoke|core|stress|holdout)-[a-z0-9-]+$")
    suite: Literal["smoke", "core", "stress", "holdout"]
    category: Literal[
        "linear-algebra", "calculus", "statistics", "physics", "english", "layout", "edge"
    ]
    complexity: Literal["medium", "high", "extreme"]
    source: dict[str, str]
    question: str = Field(min_length=1)
    input_format: Literal["text", "image", "pdf"]
    reference_solution: str = Field(min_length=1)
    reference_verification: ReferenceVerification
    expected: Expected
    paper: Literal["plain", "lined", "grid", "dots", "image", "pdf"]
    isolation: dict[str, bool]

    @model_validator(mode="after")
    def invariants(self):
        if not self.id.startswith(self.suite + "-"):
            raise ValueError("Case ID and suite disagree")
        if self.suite == "holdout" and self.isolation.get("policy_tuning"):
            raise ValueError("Holdout cannot be used for tuning")
        if self.isolation.get("model_training"):
            raise ValueError("Benchmark reference solutions are not training data")
        if (self.expected.outcome == "reject") != bool(self.expected.error_contains):
            raise ValueError("Rejection cases need an exact expected error class/message")
        for check in self.reference_verification.checks:
            if arithmetic(check.lhs) != arithmetic(check.rhs):
                raise ValueError(f"Incorrect reference numeric check: {check}")
        return self


class Corpus(Strict):
    schema_version: Literal[1]
    benchmark_version: str
    license: str
    cases: list[Case]

    @model_validator(mode="after")
    def inventory(self):
        if Counter(c.suite for c in self.cases) != dict(smoke=8, core=24, stress=16, holdout=12):
            raise ValueError("Benchmark v1 requires 8/24/16/12 cases")
        if len({c.id for c in self.cases}) != 60:
            raise ValueError("Case IDs must be unique")
        return self


def corpus_path() -> Path:
    return Path(__file__).parents[1] / "assets/benchmarks/v1/cases.json"


def load_corpus(path: Path | None = None) -> Corpus:
    path = path or corpus_path()
    expected = (path.parent / "SHA256SUMS").read_text().split()[0]
    if sha256(path) != expected:
        raise ValueError("Frozen benchmark checksum changed; create a new version")
    return Corpus.model_validate(json.loads(path.read_bytes()))


def select(corpus: Corpus, suite: str, purpose: str = "evaluation") -> list[Case]:
    if suite not in {"smoke", "core", "stress", "holdout", "development", "all"}:
        raise ValueError("Unknown suite")
    if purpose not in {"evaluation", "tuning"}:
        raise ValueError("Unknown benchmark purpose")
    cases = [
        c
        for c in corpus.cases
        if suite == "all" or c.suite == suite or (suite == "development" and c.suite != "holdout")
    ]
    if purpose == "tuning" and any(c.suite == "holdout" for c in cases):
        raise ValueError("Holdout is evaluation-only; it cannot tune a policy or model")
    return cases
