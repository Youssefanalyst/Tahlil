"""Compatibility wrapper.

Implementation moved to ``utils.io_utils``. This stub re-exports the public
API so that legacy imports (``import io_utils``) continue to work without
modification. Remove once all code is updated to the new package path.
"""

from importlib import import_module as _import_mod

_mod = _import_mod("utils.io_utils")
globals().update(_mod.__dict__)

# Explicit re-export for static analysers
load_dataframe = _mod.load_dataframe  # type: ignore
preview_dataframe = _mod.preview_dataframe  # type: ignore
save_dataframe = _mod.save_dataframe  # type: ignore
UnsupportedFormatError = _mod.UnsupportedFormatError  # type: ignore

__all__ = [
    "load_dataframe",
    "preview_dataframe",
    "save_dataframe",
    "UnsupportedFormatError",
]

"""
Supported formats determined by file extension:
    • .csv                → CSV via pandas
    • .xlsx / .xls        → Excel via pandas (openpyxl/xlrd engines)
    • .json               → JSON (records orient)
    • .db / .sqlite       → SQLite database – uses table name 'sheet'

The API intentionally keeps parameters minimal; advanced options can be added
later (e.g., selecting table name, JSON orient, compression, etc.).
"""
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
