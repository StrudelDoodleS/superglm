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


def natural_curvature_direction(
    family,
    y: NDArray,
    eta: NDArray,
    eta_direction: NDArray,
    links,
    plan,
    hessian_direction,
) -> NDArray[np.float64]:
    """Packed predictor-curvature direction from a family's natural ``dH(v)``.

    ``hessian_direction(theta, v)`` returns the packed derivative of the
    family's weighted natural Hessian along the natural direction ``v``.
    Every inverse link must supply three derivatives.
    """
    from superglm.distributional.kernels._curvature_direction import (
        predictor_curvature_direction,
    )

    try:
        predictors = np.asarray(eta, dtype=FLOAT)
        direction = np.asarray(eta_direction, dtype=FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("predictor state and direction must be finite arrays") from exc
    link_tuple = tuple(links)
    k_parameters = len(family.parameters)
    expected = (response_row_count(y), k_parameters)
    if (
        predictors.shape != expected
        or direction.shape != expected
        or len(link_tuple) != k_parameters
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(direction))
    ):
        raise ValueError("predictor state and direction must be finite n-by-K arrays")
    derivatives = []
    for index, link in enumerate(link_tuple):
        third = getattr(link, "deriv3_inverse", None)
        if not callable(third):
            raise UnsupportedLikelihoodContractError(
                f"link {index} has no third inverse-link derivative"
            )
        column = predictors[:, index]
        derivatives.append(
            tuple(
                np.asarray(values, dtype=FLOAT)
                for values in (
                    link.inverse(column),
                    link.deriv_inverse(column),
                    link.deriv2_inverse(column),
                    third(column),
                )
            )
        )
    theta, first, second, third_derivative = (
        np.column_stack([row[order] for row in derivatives]) for order in range(4)
    )
    evaluation = family.evaluate_natural(y, theta, plan, derivative_order=2)
    natural = first * direction
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = predictor_curvature_direction(
            np.asarray(evaluation.score, dtype=FLOAT),
            np.asarray(evaluation.hessian_packed, dtype=FLOAT),
            np.asarray(hessian_direction(theta, natural), dtype=FLOAT),
            first,
            second,
            third_derivative,
            direction,
        )
    # A row the family flags invalid publishes zero curvature, so its
    # derivative is zero too.
    valid = np.asarray(evaluation.valid, dtype=np.bool_)
    result[~valid] = 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError("the predictor-curvature derivative is not representable at this state")
    return readonly(result)


__all__ = [
    "FLOAT",
    "array_digest",
    "immutable",
    "natural_curvature_direction",
    "readonly",
    "response_row_count",
    "typed_plan",
    "validated_float_response",
]
