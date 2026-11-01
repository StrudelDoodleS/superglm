"""Small helpers shared by the distributional records and their payloads.

Each helper here has one definition. Records freeze what they hold with the
read-only copies, and payloads write JSON with the leaf helpers, so that every
module encodes a number the same way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike


def readonly(values: NDArray, dtype: type | np.dtype | None = None) -> NDArray:
    """Return an owned, write-protected copy of ``values`` (``dtype=None`` keeps its dtype)."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def readonly_float(values: NDArray) -> NDArray[np.float64]:
    """Return an owned, write-protected float64 copy of ``values``."""
    return readonly(values, np.float64)


def readonly_finite(values: NDArray, *, name: str) -> NDArray[np.float64]:
    """Return a write-protected float64 copy of ``values``, refusing non-finite or non-numeric input."""
    try:
        result = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values")
    result.setflags(write=False)
    return result


def frozen_float_mapping(
    values: Mapping[str, float],
    *,
    name: str,
    nonnegative: bool = False,
) -> Mapping[str, float]:
    """Return a read-only mapping of non-empty names to finite floats (optionally non-negative)."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be numeric, not bool")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} values must be finite numeric values") from exc
        if not math.isfinite(numeric) or (nonnegative and numeric < 0.0):
            qualification = " finite non-negative" if nonnegative else " finite"
            raise ValueError(f"{name} values must be{qualification}")
        result[key] = numeric
    return MappingProxyType(result)


def json_number(value: Any) -> float | None:
    """One number of a JSON payload: a float, or ``None`` where it is not finite."""
    number = float(value)
    return number if math.isfinite(number) else None


def json_floats(values: Any) -> list[Any]:
    """Return nested lists of floats in which every non-finite value is ``None``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 1:
        return [json_floats(row) for row in array]
    return [None if not np.isfinite(value) else float(value) for value in array]


def row_index(X: FrameLike | EagerFrame, n_observations: int) -> pd.Index:
    """Name payload rows as the caller's frame does, or by position for any other input."""
    if isinstance(X, pd.DataFrame):
        return X.index.copy()
    if isinstance(X, EagerFrame) and isinstance(X.native, pd.DataFrame):
        return X.native.index.copy()
    return pd.RangeIndex(n_observations)
