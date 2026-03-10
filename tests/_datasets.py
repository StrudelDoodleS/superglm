"""Cached loader for freMTPL2 datasets used by real-data parity tests.

Searches for parquet files in order:
  1. $SUPERGLM_DATA_DIR (if set)
  2. ~/.cache/superglm/
  3. <project_root>/data/

Returns None when the file is not found, so callers can skip
gracefully.  No auto-download — place the parquet files in any
of the above directories.  Source: CASdatasets R package or
https://www.openml.org (freMTPL2freq / freMTPL2sev).
"""

import os
from pathlib import Path

import pandas as pd

_SEARCH_DIRS = [
    Path(d)
    for d in [
        os.environ.get("SUPERGLM_DATA_DIR", ""),
        Path.home() / ".cache" / "superglm",
        Path(__file__).resolve().parent.parent / "data",
    ]
    if d
]


def find(name: str) -> Path | None:
    """Return the path to *name* if it exists, else None."""
    for d in _SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def load_freq() -> pd.DataFrame | None:
    """Load freMTPL2freq.parquet, or None if not found."""
    p = find("freMTPL2freq.parquet")
    return pd.read_parquet(p) if p else None


def load_sev() -> pd.DataFrame | None:
    """Load freMTPL2sev.parquet, or None if not found."""
    p = find("freMTPL2sev.parquet")
    return pd.read_parquet(p) if p else None
