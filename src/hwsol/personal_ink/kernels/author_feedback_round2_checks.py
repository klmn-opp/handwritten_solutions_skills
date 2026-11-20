"""CPU kernel extracted from research.author_feedback_round2_checks; CPU recipe only.

Geometry and operation order retain the reviewed implementation. Source identity
values are provided by an explicit local data bundle; they are not public defaults."""

from __future__ import annotations

from hwsol.personal_ink.kernels.math_latin_feedback import filled_path


def page_geometry(page, renderer):
    return [filled_path(item, renderer) for item in page.items]
