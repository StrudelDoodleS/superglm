"""Plumbing shared by the family adapters: immutability, digests, plan checks."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
)

FLOAT = np.float64


def immutable(values: NDArray) -> NDArray[np.float64]:
    """A read-only float64 copy whose buffer cannot be re-armed by a caller."""
    array = np.ascontiguousarray(values, dtype=FLOAT)
    return np.frombuffer(array.tobytes(order="C"), dtype=FLOAT).reshape(array.shape)


def readonly(values: NDArray) -> NDArray[np.float64]:
    result = np.array(values, dtype=FLOAT, copy=True)
    result.setflags(write=False)
    return result


def array_digest(domain: bytes, values: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(repr(values.shape).encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def response_row_count(y: NDArray) -> int:
    shape = np.shape(y)
    if len(shape) != 1 or shape[0] == 0:
        raise ValueError("y must be a non-empty row vector")
    return shape[0]


def validated_float_response(
    y: NDArray,
    *,
    message: str,
    lower: float | None = None,
    lower_inclusive: bool = True,
) -> NDArray[np.float64]:
    """A finite float64 response vector satisfying an optional lower bound.

    ``message`` is the family's own wording; every rejection raises it.
    """
    try:
        response = np.asarray(y, dtype=FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if response.ndim != 1 or len(response) == 0 or not np.all(np.isfinite(response)):
        raise ValueError(message)
    if lower is not None:
        below = response < lower if lower_inclusive else response <= lower
        if np.any(below):
            raise ValueError(message)
    return response


def typed_plan[PlanT](
    plan: object,
    plan_type: type[PlanT],
    weights_length: int,
    *,
    family_name: str,
) -> PlanT:
    if not isinstance(plan, plan_type):
        raise UnsupportedLikelihoodContractError(
            f"{family_name} received a likelihood prepared for another family"
        )
    weights = getattr(plan, "weights", None)
    if not isinstance(weights, ResolvedLikelihoodWeights) or weights.values.shape != (
        weights_length,
    ):
        raise UnsupportedLikelihoodContractError(
            f"{family_name} likelihood rows do not match the fitted response"
        )
    return plan


__all__ = [
    "FLOAT",
    "array_digest",
    "immutable",
    "readonly",
    "response_row_count",
    "typed_plan",
    "validated_float_response",
]
