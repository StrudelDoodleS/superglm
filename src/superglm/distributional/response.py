"""Structural response shape and labels, separate from each family's support law.

Keep the result of :func:`unvalidated_response` through row selection and pass
those raw retained values to the family's binder/validator. Only then use
:func:`validated_response_array` where a generic finite float64 array is needed.
Neither metadata nor a successful float conversion certifies family support,
exact count representability, or a composition's normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResponseSpec:
    """A scalar response, or an explicitly named vector in component order."""

    shape: tuple[int, ...] = ()
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) > 1
            or any(type(size) is not int or size <= 0 for size in self.shape)
        ):
            raise ValueError("response shape must be () or a tuple containing a positive integer")
        if (
            not isinstance(self.component_names, tuple)
            or any(not isinstance(name, str) or not name.strip() for name in self.component_names)
            or len(set(self.component_names)) != len(self.component_names)
            or len(self.component_names) != (self.shape[0] if self.shape else 0)
        ):
            raise ValueError("response components must have one unique nonempty name per component")


_SCALAR_RESPONSE = ResponseSpec()


def response_spec(family: object) -> ResponseSpec:
    """Read structural opt-in metadata; families without it remain scalar."""
    spec = getattr(family, "response_spec", _SCALAR_RESPONSE)
    if not isinstance(spec, ResponseSpec):
        raise TypeError("family.response_spec must be a ResponseSpec")
    return spec


def require_scalar_response(family: object, operation: str) -> None:
    """Refuse a scalar-only consumer before it converts or allocates response rows."""
    if response_spec(family).shape:
        raise NotImplementedError(
            f"{operation} is not supported for vector responses by {type(family).__name__}; "
            "use joint log scores or a scalar share callable for posterior quantities"
        )


def _uncoerced_values(values: object) -> NDArray:
    """Preserve nullable pandas integers before NumPy can promote them to float."""
    if isinstance(values, pd.Series | pd.Index | pd.api.extensions.ExtensionArray) and isinstance(
        values.dtype, pd.api.extensions.ExtensionDtype
    ):
        return values.to_numpy(dtype=object)
    return np.asarray(values, dtype=object if isinstance(values, list | tuple) else None)


def unvalidated_response(
    y: object, *, family: object, n_rows: int, input_index: object = None
) -> NDArray:
    """Check original row-first shape/labels without coercing response values.

    Labeled vector columns are reordered by identity. Pandas row indices must
    match the input index positionally, including duplicate labels. Index-less
    frames and arrays remain positional. Mixed columns and Python sequences
    retain individual value types so selection cannot hide a bool, complex
    number, string, or inexact integer behind an earlier numeric promotion.
    """
    spec = response_spec(family)
    if isinstance(n_rows, bool) or not isinstance(n_rows, int | np.integer) or n_rows < 1:
        raise ValueError("response shape requires a positive row count")
    index = getattr(y, "index", None)
    if input_index is not None and isinstance(index, pd.Index):
        if not index.equals(pd.Index(input_index)):
            raise ValueError("response index must match the input index positionally")
    columns = getattr(y, "columns", None)
    if columns is not None and spec.shape:
        labels = tuple(columns)
        if (
            len(labels) != len(spec.component_names)
            or any(not isinstance(label, str) for label in labels)
            or len(set(labels)) != len(labels)
            or set(labels) != set(spec.component_names)
        ):
            raise ValueError("response columns must contain each declared component exactly once")
        parts = [_uncoerced_values(y[name]) for name in spec.component_names]
        if all(part.dtype == parts[0].dtype for part in parts):
            array = np.column_stack(parts)
        else:
            array = np.empty((len(parts[0]), len(parts)), dtype=object)
            for column, part in enumerate(parts):
                array[:, column] = part
    else:
        try:
            array = _uncoerced_values(y)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "response must have row-first shape (n, *response_spec.shape)"
            ) from exc
    expected = (n_rows, *spec.shape)
    if array.shape != expected:
        raise ValueError(f"response shape {array.shape} does not match {expected}")
    return array


def validated_response_array(y: object, *, family: object, n_rows: int) -> NDArray[np.float64]:
    """Check retained shape and real finite values; family validation is still required.

    Numeric object arrays are supported after discarding zero-weight rows, but
    bools, complex values and coercible strings are not real response data.
    Callers must retain the raw array for family checks before this conversion.
    """
    raw = unvalidated_response(y, family=family, n_rows=n_rows)
    message = "response must contain only finite real numeric values (not bool or complex)"
    if raw.dtype.kind not in "fiuO":
        raise ValueError(message)
    if raw.dtype.kind == "O" and any(
        isinstance(value, bool | np.bool_ | complex | np.complexfloating)
        or not isinstance(value, int | float | np.integer | np.floating)
        for value in raw.flat
    ):
        raise ValueError(message)
    try:
        with np.errstate(over="raise", invalid="raise"):
            values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(message)
    return values
