"""Thin wrapper around DataFrameModel's internal formula evaluation.

This isolates the engine so UI/model code can simply delegate, enabling future
independent testing and optimisation.
"""
from __future__ import annotations

from typing import Set


def compute_value(model, row: int, col: int, visiting: Set[tuple[int, int]] | None = None):
    """Evaluate cell value in *model* at (row, col) using its internal engine."""
    if visiting is None:
        visiting = set()
    return model._internal_compute_value(row, col, visiting)
