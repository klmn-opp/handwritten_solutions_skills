from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from hwsol.config import RenderConfig


class RenderError(ValueError):
    """An input or backend cannot produce a trustworthy render probe."""


@dataclass
class RenderResult:
    page_count: int
    warnings: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    backend_details: dict = field(default_factory=dict)


class RendererAdapter(Protocol):
    name: str

    def render(
        self, source: str, font: Path, config: RenderConfig, output: Path
    ) -> RenderResult: ...
