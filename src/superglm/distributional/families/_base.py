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


def second_direction_arrays(
    y: NDArray,
    eta: NDArray,
    direction_a: NDArray,
    direction_b: NDArray,
    k_parameters: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validated ``(eta, a, b)`` for a second curvature direction; ``a`` and ``b`` stacked.

    ``a`` and ``b`` are ``(n, K)`` or ``(m, n, K)``, a pair the same on every
    row also ``(m, 1, K)``; both are returned stacked, ``(m, n, K)`` or
    ``(m, 1, K)`` (``m = 1`` for a single pair).
    """
    try:
        predictors = np.asarray(eta, dtype=FLOAT)
        first = np.asarray(direction_a, dtype=FLOAT)
        second = np.asarray(direction_b, dtype=FLOAT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("predictor state and directions must be finite arrays") from exc
    expected = (response_row_count(y), k_parameters)
    if (
        predictors.shape != expected
        or first.shape != second.shape
        or first.shape[-1] != k_parameters
        or first.ndim not in (2, 3)
        or first.shape[-2] not in ((expected[0],) if first.ndim == 2 else (expected[0], 1))
        or not np.all(np.isfinite(predictors))
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError(
            "predictor state must be a finite n-by-K array and the directions finite "
            "n-by-K or m-by-n-by-K arrays of one shape"
        )
    if first.ndim == 2:
        first, second = first[None], second[None]
    return predictors, first, second


def link_derivative_columns(
    predictors: NDArray[np.float64], links, order: int
) -> tuple[NDArray[np.float64], ...]:
    """``(theta, g', ..., g^(order))`` column-stacked from each link, each ``(n, K)``.

    A link without the derivative a family needs is refused with
    ``UnsupportedLikelihoodContractError``, so the caller differences instead.
    """
    names = ("deriv_inverse", "deriv2_inverse", "deriv3_inverse", "deriv4_inverse")
    columns = []
    for index, link in enumerate(links):
        methods = [link.inverse]
        for name in names[:order]:
            method = getattr(link, name, None)
            if not callable(method):
                raise UnsupportedLikelihoodContractError(f"link {index} has no {name}")
            methods.append(method)
        column = predictors[:, index]
        columns.append(tuple(np.asarray(method(column), dtype=FLOAT) for method in methods))
    return tuple(np.column_stack([row[k] for row in columns]) for k in range(order + 1))


def contract_second_direction(
    fourth: NDArray[np.float64], first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Packed second curvature direction ``-sum_kl l_ijkl a_k b_l`` from predictor fourth derivatives."""
    from superglm.distributional.kernels._curvature_direction import curvature_second_direction

    return curvature_second_direction(fourth, first, second)


def composed_second_direction(
    outer: dict[int, NDArray[np.float64]],
    inner,
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The second curvature direction of ``L(phi(eta))`` from ``L``'s derivatives and ``phi``'s.

    ``outer`` holds ``L``'s packed derivatives of orders one to four and
    ``inner`` the map's structural nonzeros (``compose_derivatives``).
    """
    from superglm.distributional.kernels._curvature_direction import (
        compose_derivatives,
        curvature_second_direction,
    )

    fourth = compose_derivatives(outer, inner, 4, first.shape[-1])
    return curvature_second_direction(fourth, first, second)


def link_inner_derivatives(*derivatives: NDArray[np.float64]):
    """``inner`` for the separable links from their derivatives, each ``(n, K)``."""
    from superglm.distributional.kernels._curvature_direction import separable_inner_derivatives

    return separable_inner_derivatives(*derivatives)


def natural_third_from_direction(hessian_direction, n_rows: int, k_parameters: int):
    """Packed natural third derivatives from a family's ``dH(v)`` on the natural axes."""
    from superglm.distributional.kernels._curvature_direction import (
        third_from_hessian_direction,
    )

    return third_from_hessian_direction(hessian_direction, n_rows, k_parameters)


def natural_curvature_second_direction(
    family,
    y: NDArray,
    eta: NDArray,
    direction_a: NDArray,
    direction_b: NDArray,
    links,
    plan,
    natural_derivatives,
) -> NDArray[np.float64]:
    """Packed second direction of predictor curvature from a family's natural derivatives.

    ``natural_derivatives(theta, natural_a, natural_b)`` returns the packed
    ``(third, fourth)`` derivatives of the family's weighted natural
    log-likelihood; ``natural_a = g' a`` and ``natural_b = g' b`` (stacked
    ``(m, n, K)``) let a family refuse rows the path cannot cross.  Every
    inverse link must supply four derivatives.  Rows the family flags invalid
    publish zero curvature, so their derivatives are zero; the result has the
    directions' leading shape.
    """
    from superglm.distributional.kernels._curvature_direction import (
        predictor_curvature_second_direction,
    )

    link_tuple = tuple(links)
    k_parameters = len(family.parameters)
    if len(link_tuple) != k_parameters:
        raise ValueError("one link per natural parameter is required")
    single = np.ndim(direction_a) == 2
    predictors, first, second = second_direction_arrays(
        y, eta, direction_a, direction_b, k_parameters
    )
    theta, *derivatives = link_derivative_columns(predictors, link_tuple, 4)
    evaluation = family.evaluate_natural(y, theta, plan, derivative_order=2)
    third, fourth = natural_derivatives(theta, derivatives[0] * first, derivatives[0] * second)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = predictor_curvature_second_direction(
            (
                np.asarray(evaluation.score, dtype=FLOAT),
                np.asarray(evaluation.hessian_packed, dtype=FLOAT),
                np.asarray(third, dtype=FLOAT),
                np.asarray(fourth, dtype=FLOAT),
            ),
            tuple(derivatives),
            first,
            second,
        )
    valid = np.asarray(evaluation.valid, dtype=np.bool_)
    result[:, ~valid] = 0.0
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "the predictor-curvature second derivative is not representable at this state"
        )
    return readonly(result[0] if single else result)


__all__ = [
    "FLOAT",
    "array_digest",
    "composed_second_direction",
    "contract_second_direction",
    "immutable",
    "link_inner_derivatives",
    "link_derivative_columns",
    "natural_curvature_direction",
    "natural_curvature_second_direction",
    "natural_third_from_direction",
    "readonly",
    "response_row_count",
    "second_direction_arrays",
    "typed_plan",
    "validated_float_response",
]
