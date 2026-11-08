"""Owned link-scale offset sources and their original-row resolution.

Declarations belong to predictor templates. Resolved sums belong to one fit or
prediction call; there is no cache or inference from the training row count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame


@dataclass(frozen=True, eq=False)
class _IndexedOffset:
    values: NDArray
    index: pd.Index


def _copy_source(source: Any) -> str | NDArray | _IndexedOffset:
    if isinstance(source, str):
        if not source:
            raise ValueError("offset column names must be non-empty")
        return source
    index = None
    if isinstance(source, _IndexedOffset):
        index, source = source.index.copy(deep=True), source.values
    elif isinstance(source, pd.Series):
        index, source = source.index.copy(deep=True), source.to_numpy()
    try:
        # Preserve each raw Python value until row selection: one invalid
        # string/complex/bool must not coerce all the otherwise numeric rows.
        values = np.array(
            source, dtype=object if isinstance(source, (list, tuple)) else None, copy=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "an offset source must be a column name or a one-dimensional array"
        ) from exc
    if values.ndim != 1:
        raise ValueError("an offset source must be a column name or a one-dimensional array")
    values.setflags(write=False)
    return values if index is None else _IndexedOffset(values, index)


def normalize_offset(offset: Any) -> tuple | None:
    """Copy/freeze sources; a flat numeric sequence is one row vector.

    A sequence of columns or vectors supplies multiple sources (including a
    mixture). Scalar/vector mixtures and two-dimensional arrays are refused.
    Values are checked only after the fit's zero-weight row selection.
    """
    if offset is None:
        return None
    if isinstance(offset, (list, tuple)):
        if not offset:
            raise ValueError("offset sources must not be empty")
        if all(np.isscalar(value) and not isinstance(value, str) for value in offset):
            sources = (_copy_source(offset),)
        else:
            sources = tuple(_copy_source(value) for value in offset)
    else:
        sources = (_copy_source(offset),)
    lengths = {
        len(source.values if isinstance(source, _IndexedOffset) else source)
        for source in sources
        if not isinstance(source, str)
    }
    if len(lengths) > 1:
        raise ValueError("offset array sources must have the same length")
    return sources


def sources_of(predictor: Any) -> tuple:
    """Old saved templates lack this field and declare no source."""
    return getattr(predictor, "offset", None) or ()


def offset_columns(predictors: Sequence) -> tuple[str, ...]:
    return tuple(dict.fromkeys(s for p in predictors for s in sources_of(p) if isinstance(s, str)))


def _shaped(source: Any, frame: EagerFrame, name: str) -> NDArray:
    if isinstance(source, str):
        frame.require_columns((source,))
        source = frame.column_array(source)
    if isinstance(source, pd.Series):
        source = _IndexedOffset(source.to_numpy(), source.index)
    if isinstance(source, _IndexedOffset):
        index = getattr(frame.native, "index", None)
        if index is not None and not source.index.equals(index):
            raise ValueError(f"offset for {name!r} index must match X on the original rows")
        source = source.values
    try:
        values = np.asarray(source, dtype=object if isinstance(source, (list, tuple)) else None)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"offset for {name!r} must be one-dimensional with length {len(frame)}"
        ) from exc
    if values.shape != (len(frame),):
        raise ValueError(f"offset for {name!r} must be one-dimensional with length {len(frame)}")
    return values


def shape_fit_offsets(predictors: Sequence, frame: EagerFrame) -> dict[str, tuple[NDArray, ...]]:
    """Check every source's original-row shape/labels before selecting rows."""
    return {
        predictor.name: tuple(
            _shaped(source, frame, predictor.name) for source in sources_of(predictor)
        )
        for predictor in predictors
    }


def _numeric(values: NDArray, name: str, positions: NDArray) -> NDArray[np.float64]:
    retained = values[positions]
    if retained.dtype.kind in "iuf":
        valid = np.isfinite(retained)
    elif retained.dtype.kind == "O":
        valid = np.array(
            [
                isinstance(v, Real) and not isinstance(v, (bool, np.bool_, np.timedelta64))
                for v in retained
            ]
        )
    else:
        valid = np.zeros(len(retained), dtype=bool)
    if not np.all(valid):
        raise ValueError(
            f"offset for {name!r} must contain finite real numeric values; original rows {positions[~valid].tolist()}"
        )
    try:
        result = np.asarray(retained, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"offset for {name!r} must contain finite real numeric values") from exc
    valid = np.isfinite(result)
    if not np.all(valid):
        raise ValueError(
            f"offset for {name!r} must contain finite real numeric values; original rows {positions[~valid].tolist()}"
        )
    return result


def sum_offset_sources(
    shaped: Mapping[str, tuple[NDArray, ...]], positions: NDArray
) -> dict[str, NDArray]:
    """Sum finite retained sources with fixed coefficient one, in declared order."""
    result = {}
    for name, sources in shaped.items():
        total = np.zeros(len(positions), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for source in sources:
                total += _numeric(source, name, positions)
        if not np.all(np.isfinite(total)):
            raise ValueError(
                f"offset sum for {name!r} must be finite; original rows {positions[~np.isfinite(total)].tolist()}"
            )
        total.setflags(write=False)
        result[name] = total
    return result


def shape_prediction_offsets(
    predictors: Sequence, frame: EagerFrame, offsets: Mapping | None = None
) -> dict[str, tuple[NDArray, ...]]:
    """Check original-row sources; replacements supply the sum of declared arrays.

    Column-only declarations reject a second source even if its values agree.
    Undeclared predictors retain the legacy optional prediction mapping.
    """
    supplied = {} if offsets is None else offsets
    if not isinstance(supplied, Mapping):
        raise TypeError("offsets must be a predictor-keyed mapping")
    names = tuple(p.name for p in predictors)
    unknown = [str(name) for name in supplied if name not in names]
    if unknown:
        raise ValueError(f"unknown offset predictor name: {', '.join(unknown)}")
    shaped = {}
    for predictor in predictors:
        name, sources = predictor.name, sources_of(predictor)
        columns = tuple(s for s in sources if isinstance(s, str))
        arrays = len(columns) != len(sources)
        if arrays and name not in supplied:
            raise ValueError(
                f"offset for {name!r} needs replacement new-row values for its array sources"
            )
        if columns and not arrays and name in supplied:
            raise ValueError(f"offset for {name!r} duplicates its declared column sources")
        shaped[name] = tuple(_shaped(s, frame, name) for s in columns)
        if name in supplied:
            if isinstance(supplied[name], str):
                raise ValueError(
                    f"replacement offset for {name!r} must be a row array, not a column name"
                )
            shaped[name] += (_shaped(supplied[name], frame, name),)
    return shaped


def prediction_offsets(
    predictors: Sequence,
    frame: EagerFrame,
    offsets: Mapping | None = None,
    *,
    positions: NDArray | None = None,
) -> dict[str, NDArray]:
    """Resolve a new-row sum, validating values only at retained positions."""
    shaped = shape_prediction_offsets(predictors, frame, offsets)
    return sum_offset_sources(shaped, np.arange(len(frame)) if positions is None else positions)


def take_prediction_replacements(
    predictors: Sequence, frame: EagerFrame, offsets: Mapping | None, positions: NDArray
) -> dict[str, NDArray] | None:
    """Take only explicit replacements; downstream prediction resolves columns.

    Validate labels and retained values against the original rows before
    slicing, including column sources. The replacement, when supplied, is
    the last shaped source; it never includes the columns' contributions.
    """
    shaped = shape_prediction_offsets(predictors, frame, offsets)
    sum_offset_sources(shaped, positions)
    return None if offsets is None else {name: shaped[name][-1][positions] for name in offsets}


def take_offset(offset: Any, positions: NDArray) -> tuple | None:
    """Take array declarations positionally for a fold; columns stay declarative."""
    if offset is None:
        return None
    return tuple(
        source
        if isinstance(source, str)
        else (
            _IndexedOffset(source.values[positions], source.index.take(positions))
            if isinstance(source, _IndexedOffset)
            else source[positions]
        )
        for source in offset
    )
