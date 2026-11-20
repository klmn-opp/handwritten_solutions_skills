"""Portable renderer policies; no Profile-store or training imports."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class StrokeAction(PolicyModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,31}$")
    legacy: bool = False
    ink_balance: float = Field(default=0, ge=0, le=8)
    continuity: float = Field(default=0, ge=0, le=4)
    repeat_penalty: float = Field(default=0, ge=0, le=3)
    lean: float = Field(default=0.075, ge=-0.15, le=0.15)
    lean_variation: float = Field(default=0, ge=0, le=0.04)


class RenderPolicy(PolicyModel):
    schema_version: Literal[1] = 1
    profile_sha256: Digest
    reward_source: Literal["untrained", "proxy", "human-preference"] = "untrained"
    training_sha256: Digest | None = None
    actions: tuple[StrokeAction, ...] = Field(min_length=2, max_length=12)
    probabilities: tuple[Annotated[float, Field(ge=0, le=1)], ...]

    @model_validator(mode="after")
    def distribution(self):
        if len(self.actions) != len(self.probabilities):
            raise ValueError("Each action needs a probability")
        if len({a.name for a in self.actions}) != len(self.actions):
            raise ValueError("Duplicate policy actions")
        if abs(sum(self.probabilities) - 1) > 1e-8:
            raise ValueError("Policy probabilities must sum to one")
        return self
