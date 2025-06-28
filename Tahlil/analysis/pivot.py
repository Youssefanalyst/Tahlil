"""Utility helpers for building pivot tables.

Isolating pandas pivot logic here keeps UI code cleaner and allows future unit
testing.
"""
from __future__ import annotations

import warnings
from typing import List, Sequence, Any, Hashable, Iterable

import pandas as pd


__all__ = ["pivot_table"]


def pivot_table(
    df: pd.DataFrame,
    *,
    index: Sequence[Hashable] | None = None,
    columns: Sequence[Hashable] | None = None,
    values: Sequence[Hashable] | None = None,
    aggs: Any = "sum",
    margins: bool = True,
    margins_name: str = "Total",
    try_numeric: bool = True,
) -> pd.DataFrame:
    """Wrapper around :pyfunc:`pandas.pivot_table` with extra robustness.

    Features compared to calling pandas directly:
    • ``dropna`` set to *False* so that categories with no data still appear.
    • Suppresses *FutureWarning* noise from pandas.
    • If initial attempt fails (e.g., *mean* over non-numeric cols) and
      *try_numeric* is *True*, attempts to convert *values* columns to numeric
      (coerce errors to NaN) and retries once.
    """

    def _create(_df: pd.DataFrame) -> pd.DataFrame:
        return pd.pivot_table(
            _df,
            index=index if index else None,
            columns=columns if columns else None,
            values=values,
            aggfunc=aggs,
            dropna=False,
            margins=margins,
            margins_name=margins_name,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            return _create(df)
        except Exception:
            if not try_numeric or values is None:
                raise
            # Attempt numeric conversion of value columns, then retry once
            df_temp = df.copy()
            for v in values:
                df_temp[v] = pd.to_numeric(df_temp[v], errors="coerce")
            return _create(df_temp)
