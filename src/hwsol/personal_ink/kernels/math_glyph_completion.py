"""CPU kernel extracted from research.corpus.math_glyph_completion; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

SELECTION = set("λ∂βαρ∑∞[]∩≥→′πτεδ:()") | {"fraction-bar", "radical-overbar"}
