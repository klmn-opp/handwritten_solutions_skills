"""Explicit, context-local asset identities for the calibrated CPU recipe.

No author identity or private file path is an executable package default.
The caller validates and hashes the containing data bundle before binding it.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_RECIPE: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "hwsol_personal_ink_recipe", default=None
)
_KEYS = frozenset(
    {
        "font_sha256",
        "digit_sample_ids",
        "horizontal_source_ids",
        "horizontal_source_sha256",
        "single_bar_source_id",
    }
)


@contextmanager
def bind_recipe(values: Mapping[str, Any]) -> Iterator[None]:
    """Bind a validated bundle for one pipeline invocation, without global mutation."""
    if not isinstance(values, Mapping) or not set(values) <= _KEYS:
        raise ValueError("Unknown personal ink recipe identity binding")
    token = _RECIPE.set(copy.deepcopy(dict(values)))
    try:
        yield
    finally:
        _RECIPE.reset(token)


def recipe_value(key: str) -> Any:
    """Require an explicit identity; missing data is never filled from local history."""
    values = _RECIPE.get()
    if key not in _KEYS or values is None or key not in values:
        raise ValueError(f"Personal ink recipe binding is required: {key}")
    return values[key]
