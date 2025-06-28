"""Utility helpers for loading and saving DataFrames in multiple formats.

Moved from project root into ``utils`` package for cleaner project structure.
Existing public API is preserved and re-exported in the old ``io_utils`` module,
so that legacy imports keep working without code changes.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

SQL_TABLE_NAME = "sheet"


class UnsupportedFormatError(ValueError):
    """Raised when the file extension is not recognised."""


def _suffix(path: str | Path) -> str:
    return Path(path).suffix.lower()


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load DataFrame from *path* based on its suffix."""
    suf = _suffix(path)
    path = Path(path)
    if suf == ".csv":
        return pd.read_csv(path, header=0)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, header=0)
    if suf == ".json":
        try:
            return pd.read_json(path, orient="records")
        except ValueError as exc:
            # fallback to JSON Lines (one record per line)
            if "Trailing data" in str(exc):
                return pd.read_json(path, lines=True)
            raise
    if suf in (".db", ".sqlite"):
        conn = sqlite3.connect(path)
        try:
            return pd.read_sql_query(f"SELECT * FROM {SQL_TABLE_NAME}", conn)
        finally:
            conn.close()
    raise UnsupportedFormatError(f"Unsupported file type: {path}")


def preview_dataframe(path: str | Path, nrows: int = 100, header: bool = True) -> pd.DataFrame:
    """Load only *nrows* rows for quick preview without reading whole file."""
    suf = _suffix(path)
    path = Path(path)
    if suf == ".csv":
        return pd.read_csv(path, header=0 if header else None, nrows=nrows)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path, header=0 if header else None, nrows=nrows)
    if suf == ".json":
        try:
            return pd.read_json(path, orient="records", lines=not header).head(nrows)
        except ValueError:
            return pd.read_json(path, lines=True).head(nrows)
    if suf in (".db", ".sqlite"):
        conn = sqlite3.connect(path)
        try:
            return pd.read_sql_query(f"SELECT * FROM {SQL_TABLE_NAME} LIMIT {nrows}", conn)
        finally:
            conn.close()
    raise UnsupportedFormatError(f"Unsupported file type: {path}")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save *df* to *path* in format inferred from suffix."""
    suf = _suffix(path)
    path = Path(path)
    if suf == ".csv":
        df.to_csv(path, header=False, index=False)
        return
    if suf in (".xlsx", ".xls"):
        df.to_excel(path, header=False, index=False, engine="openpyxl")
        return
    if suf == ".json":
        df.to_json(path, orient="records", force_ascii=False)
        return
    if suf in (".db", ".sqlite"):
        conn = sqlite3.connect(path)
        try:
            df.to_sql(SQL_TABLE_NAME, conn, if_exists="replace", index=False)
        finally:
            conn.close()
        return
    raise UnsupportedFormatError(f"Unsupported file type: {path}")
