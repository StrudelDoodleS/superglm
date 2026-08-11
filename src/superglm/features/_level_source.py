"""Normalize a user-supplied level universe into a plain list of labels.

Accepted shapes are deliberately exactly three (spec 2026-08-11, §3.1):
an explicit sequence, a data column (Series/array), or a CategoricalDtype.
Encoder objects are rejected with the one-line recipe instead of being
half-supported.
"""

from __future__ import annotations

from typing import Any, NoReturn

import numpy as np


def _fail(msg: str, context: str, exc: type[Exception] = ValueError) -> NoReturn:
    raise exc(f"[{context}] {msg}" if context else msg)


def _declared_categories(dtype: Any) -> list | None:
    """Return categories a dtype DECLARES, or None when it declares none.

    Only an Enum declares on the polars/narwhals side. A plain polars
    `Categorical` also carries a `.categories` attribute, but that is the
    process-wide string cache -- it can hold labels contributed by unrelated
    columns, in insertion order, so reading it would invent levels.
    """
    if type(dtype).__name__ != "Enum":
        return None
    declared = getattr(dtype, "categories", None)
    return None if declared is None else list(declared)


def resolve_level_source(source: Any, *, context: str = "") -> list:
    import pandas as pd

    if hasattr(source, "categories_"):
        _fail(
            "levels= does not accept fitted encoder objects; pass the "
            "vocabulary itself, e.g. levels=encoder.categories_[0] for a "
            "single-feature encoder.",
            context,
            TypeError,
        )

    if isinstance(source, pd.CategoricalDtype):
        labels = source.categories.tolist()
    elif isinstance(source, list | tuple):
        labels = list(source)
    elif isinstance(source, pd.Series | np.ndarray) or (
        hasattr(source, "to_numpy") and hasattr(source, "dtype")
    ):
        dtype = getattr(source, "dtype", None)
        if isinstance(dtype, pd.CategoricalDtype):
            labels = dtype.categories.tolist()
        else:
            declared = _declared_categories(dtype)
            if declared is not None:
                labels = declared
            else:
                values = np.asarray(
                    source.to_numpy() if hasattr(source, "to_numpy") else source
                ).ravel()
                labels = sorted(pd.unique(values).tolist(), key=str)
    else:
        _fail(
            "levels= accepts a list/tuple of labels, a data column "
            "(pandas/polars Series or numpy array), or a pandas "
            f"CategoricalDtype; got {type(source).__name__}.",
            context,
            TypeError,
        )

    # Per-element and exact: pandas decides nullness by asking `v != v`, so the
    # narrow float test is not redundant with `pd.isna`, which also catches
    # pd.NA / pd.NaT.
    if any(v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) for v in labels):
        _fail("levels= contains a missing value; a level cannot be NaN or None.", context)
    seen: set = set()
    dupes = [v for v in labels if v in seen or seen.add(v)]
    if dupes:
        _fail(f"levels= contains duplicate labels: {sorted(set(dupes), key=str)}.", context)
    if len(labels) < 2:
        _fail(f"levels= needs >= 2 labels, got {len(labels)}.", context)
    return labels
