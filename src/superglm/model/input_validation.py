"""Pure validation and normalization for public fit entry points."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._frame import EagerFrame, as_eager_frame
from superglm._utils import _validate_strict_prior_weights
from superglm.distributions import Distribution, Tweedie, validate_response
from superglm.solvers.dispersion import PRIOR_WEIGHTS


@dataclass(frozen=True)
class ValidatedFitInput:
    """Normalized arrays that are safe to pass into design construction."""

    X: EagerFrame
    y: NDArray[np.float64]
    sample_weight: NDArray[np.float64]
    offset: NDArray[np.float64] | None


def _finite_vector(
    name: str,
    value,
    n_rows: int,
    *,
    require_nonempty: bool = False,
    check_finite: bool = True,
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if require_nonempty and raw.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if getattr(raw.dtype, "kind", None) in {"M", "m"}:
        raise ValueError(f"{name} must contain only real numeric values")
    if len(raw) != n_rows:
        raise ValueError(f"{name} must have length {n_rows}, got {len(raw)}")
    try:
        normalized = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain only real numeric values") from exc
    if check_finite and not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def _reject_unhashable_values(name: object, values: NDArray) -> None:
    """Raise unless every element of an object column can serve as a level.

    Everything downstream keys on these values -- level tables, encoders,
    interaction keys -- so "does it hash" is the whole question, and asking it
    with `hash` itself is the only answer that cannot drift from what those
    callers will do. Column-wide dtype inference is not a substitute: pandas
    labels by `isinstance`, so a `string` column may still hold a `str`
    subclass that sets `__hash__ = None`, and a tuple of scalars may still
    nest an unhashable member.

    How the answer is collected is what makes it affordable. `deque(...,
    maxlen=0)` drives the map in C and discards each result, so a column costs
    one `hash` call per element and runs no Python bytecode at all, where the
    scan this replaced ran a Python-level predicate per row. A power search
    re-validates every configured column on every candidate fit, which is
    enough repetition for that difference to outweigh the fits it guards.
    """
    try:
        deque(map(hash, values), maxlen=0)
    except (TypeError, ValueError) as exc:
        # `hash` reports refusal as TypeError for the unhashable types and as
        # ValueError for the few objects that hash conditionally, such as a
        # memoryview over a writable buffer. Both mean the same thing here.
        raise ValueError(
            f"X column {name!r} must contain only scalar values or hashable tuple levels"
        ) from exc


def validate_x_columns(frame: EagerFrame, columns: Iterable[object]) -> None:
    """Validate configured model columns before feature construction or scoring."""
    for name in tuple(dict.fromkeys(columns)):
        values = frame.column_array(name)
        dtype_kind = getattr(values.dtype, "kind", None)
        inferred_dtype = ""
        object_has_complex = False
        if dtype_kind == "O":
            _reject_unhashable_values(name, values)
            inferred_dtype = pd.api.types.infer_dtype(values, skipna=True)
            object_has_complex = inferred_dtype == "complex" or (
                inferred_dtype.startswith("mixed")
                and any(isinstance(value, complex | np.complexfloating) for value in values)
            )
        if dtype_kind == "c" or object_has_complex:
            raise ValueError(f"X column {name!r} must be real-valued")
        numeric_like_object = inferred_dtype in {
            "decimal",
            "floating",
            "integer",
            "mixed-integer-float",
        }
        physical_numeric = dtype_kind in {"b", "i", "u", "f"}
        if (
            frame.column_kind(name) in {"numeric", "boolean"}
            or physical_numeric
            or numeric_like_object
        ):
            try:
                numeric = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"X column {name!r} must contain only real numeric values"
                ) from exc
            if not np.all(np.isfinite(numeric)):
                raise ValueError(
                    f"X column {name!r} must contain only finite values; "
                    "non-finite or missing values are not allowed"
                )


def validate_prediction_offset(value, n_rows: int) -> NDArray[np.float64] | None:
    """Validate a public prediction offset against the scored frame."""
    if value is None:
        return None
    return _finite_vector("offset", value, n_rows)


def validate_fit_input(
    X,
    y,
    sample_weight,
    offset,
    *,
    family: Distribution,
    weight_semantics: str,
    required_columns: Iterable[object],
    check_all_columns: bool = False,
) -> ValidatedFitInput:
    """Validate a public fit call before any feature is built or learned."""
    frame = as_eager_frame(X)
    if len(frame) == 0:
        raise ValueError("X must be non-empty")

    required = tuple(dict.fromkeys(required_columns))
    frame.require_columns(required)
    columns_to_check = frame.columns if check_all_columns else required
    validate_x_columns(frame, columns_to_check)

    n_rows = len(frame)
    # validate_response() performs the universal finite check together with the
    # family-domain check, so avoid scanning the response twice here.
    y_arr = _finite_vector("y", y, n_rows, require_nonempty=True, check_finite=False)
    if sample_weight is None:
        weight_arr = np.ones(n_rows, dtype=np.float64)
    elif isinstance(family, Tweedie) and weight_semantics == PRIOR_WEIGHTS:
        # Strict positivity is a Tweedie density requirement, not a statement
        # about the weight contract: the compound-Poisson normalizer carries
        # ``log w``, so a zero prior weight is an unevaluable density rather
        # than an uninformative row.  Read as a replication count the weight
        # never enters that normalizer, and zero means what it means for every
        # other family -- a row that appears no times.
        weight_arr = _validate_strict_prior_weights(sample_weight, n_rows)
    else:
        weight_arr = _finite_vector("sample_weight", sample_weight, n_rows)
    if np.any(weight_arr < 0.0):
        raise ValueError("sample_weight must be nonnegative")
    if not np.any(weight_arr > 0.0):
        raise ValueError("sample_weight must not be all zero")
    offset_arr = None if offset is None else _finite_vector("offset", offset, n_rows)
    validate_response(y_arr, family)
    return ValidatedFitInput(frame, y_arr, weight_arr, offset_arr)
