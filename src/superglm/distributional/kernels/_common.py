"""Primitive helpers shared by the numerical kernels.

A sibling numerical module is the one package-internal dependency the
primitive-kernel rule permits; nothing here imports a distributional
contract.
"""

from __future__ import annotations

import operator
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

WeightSemantics = Literal["prior", "frequency"]
DerivativeOrder = Literal[0, 1, 2]
FLOAT = np.float64

_DERIVATIVE_ORDER_MESSAGE = "derivative_order must be an integer from zero through two"


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


__all__ = [
    "FLOAT",
    "DerivativeOrder",
    "WeightSemantics",
    "float64_vector",
    "positive_weights",
    "readonly",
    "readonly_bool",
    "validated_derivative_order",
    "validated_semantics",
]
