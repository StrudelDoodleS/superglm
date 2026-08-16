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

import importlib.util
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


def usable(name: str) -> Path | None:
    """The path to *name* when it is present AND an engine can load it.

    Deliberately NOT "can actually be read": this checks that the path exists
    and that a parquet engine is importable, and never opens the file.  A
    truncated parquet, a permission-denied file, or a ``pyarrow`` whose shared
    object fails at import time all still raise from inside the test body.

    That is still the right pair of questions, because it is the pair that was
    wrong.  :func:`find` answers only "is the file there", so with the file
    present and no engine installed a guard built on it let the test run and
    raise ``ImportError`` from the body -- a developer holding the data got no
    coverage, and the failure named pandas rather than the missing extra.
    """
    p = find(name)
    if p is None:
        return None
    if any(importlib.util.find_spec(e) is not None for e in ("pyarrow", "fastparquet")):
        return p
    return None


#: Marker prefix on skip reasons that :func:`require_data` may escalate.
SKIP_SENTINEL = "[dataset]"


def skip_reason(name: str) -> str | None:
    """``None`` when *name* is usable, else why it is not.

    Pure: it never raises, so importing a module that calls it at module scope
    can never collapse that module's collection.  Escalation is the collection
    hook's job -- see ``tests/conftest.py`` -- because it has to be scoped to
    the marked ITEMS.  An earlier revision raised here, which turned an
    unreadable dataset into zero tests collected for the whole module: 33
    synthetic tests in one suite have no stake in the parquet and were being
    deleted from coverage along with the 2 that do.
    """
    if usable(name) is not None:
        return None
    return (
        f"{SKIP_SENTINEL} {name} is not readable: put it in $SUPERGLM_DATA_DIR, "
        f"~/.cache/superglm/ or <project>/data/, and install a parquet engine"
    )


def require_data() -> bool:
    """Whether ``SUPERGLM_REQUIRE_DATA`` asks for a dataset skip to be a FAILURE.

    Without it these suites skip silently EVERYWHERE -- CI included, since the
    parquet is gitignored -- and a skip reads identically to a pass in the
    summary.  That is not hypothetical: the guide anchor drifted out of
    tolerance on 2026-08-07 and nothing reported it until someone ran it by
    hand.  This switch is what makes "did this actually run?" answerable.

    Parsed as an explicit allow-list, so ``=no`` and ``=off`` mean off rather
    than on.
    """
    return os.environ.get("SUPERGLM_REQUIRE_DATA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_freq() -> pd.DataFrame | None:
    """Load freMTPL2freq.parquet, or None if not found."""
    p = find("freMTPL2freq.parquet")
    return pd.read_parquet(p) if p else None


def load_sev() -> pd.DataFrame | None:
    """Load freMTPL2sev.parquet, or None if not found."""
    p = find("freMTPL2sev.parquet")
    return pd.read_parquet(p) if p else None
