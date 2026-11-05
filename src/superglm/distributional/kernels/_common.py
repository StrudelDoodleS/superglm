"""Primitive helpers shared by the numerical kernels.

A sibling numerical module is the one package-internal dependency the
primitive-kernel rule permits; nothing here imports a distributional
contract.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from superglm._parallel import pooled_row_blocks

WeightSemantics = Literal["prior", "frequency"]
DerivativeOrder = Literal[0, 1, 2]
FLOAT = np.float64

_DERIVATIVE_ORDER_MESSAGE = "derivative_order must be an integer from zero through two"
# Below this many rows a batch runs as one serial block; above it, blocks of
# PARALLEL_BLOCK rows run on the worker pool.
PARALLEL_MIN_ROWS = 5_000
PARALLEL_BLOCK = 1024


class _NumericalEvaluationError(ValueError):
    """Valid primitive inputs have no representable numerical result."""


def readonly(
    values: object,
    *,
    dtype: np.dtype | type = FLOAT,
    shape: tuple[int, ...] | None = None,
) -> NDArray:
    """A C-ordered read-only copy, optionally checked against an expected shape."""
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if shape is not None and result.shape != shape:
        raise ValueError(f"result array shape {result.shape} does not match {shape}")
    result.setflags(write=False)
    return result


def readonly_bool(values: object, *, shape: tuple[int, ...] | None = None) -> NDArray[np.bool_]:
    return readonly(values, dtype=np.bool_, shape=shape)


def validated_semantics(value: object) -> WeightSemantics:
    if value not in ("prior", "frequency"):
        raise ValueError("semantics must be 'prior' or 'frequency'")
    return cast(WeightSemantics, value)


def float64_vector(values: object, name: str) -> NDArray[np.float64]:
    """The caller's non-empty finite one-dimensional float64 array, uncopied."""
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype(FLOAT)
        or values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional float64 NumPy array")
    return values


def positive_weights(values: object, weight_semantics: WeightSemantics) -> NDArray[np.float64]:
    weights = float64_vector(values, "weights")
    if np.any(weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive")
    if weight_semantics == "frequency" and np.any(weights != np.floor(weights)):
        raise ValueError("frequency weights must be exact positive integers")
    return weights


def stacked_direction(
    direction: object,
    n_rows: int,
    k_parameters: int,
    error: type[Exception],
    message: str,
) -> NDArray[np.float64]:
    """A finite float64 direction ``(n, K)``, or ``(m, n, K)`` for ``m`` stacked directions."""
    values = np.asarray(direction, dtype=FLOAT)
    if (
        values.ndim not in (2, 3)
        or values.shape[-2:] != (n_rows, k_parameters)
        or not np.all(np.isfinite(values))
    ):
        raise error(message)
    return values


def validated_derivative_order(value: object) -> DerivativeOrder:
    if isinstance(value, bool | np.bool_):
        raise ValueError(_DERIVATIVE_ORDER_MESSAGE)
    try:
        order = operator.index(value)  # ty: ignore[invalid-argument-type] -- runtime protocol check
    except TypeError as exc:
        raise ValueError(_DERIVATIVE_ORDER_MESSAGE) from exc
    if order not in (0, 1, 2):
        raise ValueError(_DERIVATIVE_ORDER_MESSAGE)
    return order


def pooled_rows(kernel: Callable[[int, int], tuple[int, int]], n_rows: int) -> tuple[int, int]:
    """Run ``kernel(start, stop)`` over the rows: one block below ``PARALLEL_MIN_ROWS``,
    else blocks of ``PARALLEL_BLOCK`` on the worker pool.  The first failure in
    row order, ``(status, row)``, or ``(0, -1)``; status 0 is success."""
    if n_rows < PARALLEL_MIN_ROWS:
        return kernel(0, n_rows) if n_rows else (0, -1)
    for status, row in pooled_row_blocks(kernel, n_rows, PARALLEL_BLOCK):
        if status != 0:
            return status, row
    return 0, -1


__all__ = [
    "FLOAT",
    "PARALLEL_BLOCK",
    "PARALLEL_MIN_ROWS",
    "DerivativeOrder",
    "WeightSemantics",
    "float64_vector",
    "pooled_rows",
    "positive_weights",
    "readonly",
    "readonly_bool",
    "stacked_direction",
    "validated_derivative_order",
    "validated_semantics",
]
