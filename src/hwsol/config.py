"""Strict, backend-independent physical units; no training configuration here."""

import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def data_dir() -> Path:
    """Resolve local user data without creating directories during import."""
    if configured := os.environ.get("HWSOL_DATA_DIR"):
        return Path(configured).expanduser().resolve()
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "handwritten-solutions"


def test_font() -> Path:
    return Path(__file__).parent / "assets/presets/patrick-hand/PatrickHand-Regular.ttf"


class RenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dpi: int = Field(default=150, ge=72, le=300)
    page_width_mm: float = Field(default=210, ge=100, le=300)
    page_height_mm: float = Field(default=297, ge=100, le=400)
    margin_mm: float = Field(default=18, ge=5, le=50)
    font_size_pt: float = Field(default=13, ge=8, le=24)
    line_spacing: float = Field(default=1.6, ge=1.1, le=2.5)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    paper: Literal["plain", "lined", "grid", "dots"] = "plain"
    ink: str = Field(default="#202838", pattern=r"^#[0-9a-fA-F]{6}$")

    @model_validator(mode="after")
    def content_area(self) -> "RenderConfig":
        if min(self.page_width_mm, self.page_height_mm) - 2 * self.margin_mm < 40:
            raise ValueError("Margins must leave at least 40 mm of content area")
        return self

    def mm(self, value: float) -> int:
        return round(value * self.dpi / 25.4)

    def pt(self, value: float) -> int:
        return round(value * self.dpi / 72)
