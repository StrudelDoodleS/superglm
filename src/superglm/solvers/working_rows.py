"""Working-response geometry for direct coefficient fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import (
    _VARIANCE_FLOOR,
    Distribution,
    Gamma,
    Gaussian,
    Poisson,
    initial_mean,
)
from superglm.links import IdentityLink, Link, LogLink, SqrtLink


@dataclass(frozen=True)
class CoefficientWorkingRows:
    """One coherent quadratic model for a direct coefficient update."""

    weights: NDArray
    response: NDArray
    curvature_source: Literal["fisher", "observed"]
    fallback_reason: str | None = None


def supports_observed_newton(distribution: object, link: object) -> bool:
    """Return whether an exact, positive observed-Newton row kernel is approved."""
    # Exact types are intentional: a subclass can change either likelihood or
    # inverse-link derivatives and must not inherit an unproved Hessian.
    return type(distribution) is Gamma and type(link) is LogLink


def coefficient_initial_intercept(
    *,
    distribution: Distribution,
    link: Link,
    y: NDArray,
    sample_weight: NDArray,
) -> float:
    """Return a link-appropriate intercept before offsets are applied."""
    if type(distribution) is Poisson and type(link) is SqrtLink:
        # Unlike log, sqrt represents zero and arbitrarily small non-negative
        # means directly.  Do not inherit the positive-family initialization
        # floor needed by singular-at-zero links.
        mean = float(np.average(y, weights=sample_weight))
        return float(np.sqrt(max(mean, 0.0)))
    mean = initial_mean(y, sample_weight, distribution)
    return float(link.link(np.atleast_1d(mean))[0])


def _fisher_rows(
    *,
    distribution: object,
    link: object,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    fallback_reason: str | None = None,
) -> CoefficientWorkingRows:
    if type(distribution) is Poisson and type(link) is SqrtLink:
        # Analytically, μ=η² and V(μ)=μ cancel from the Fisher geometry:
        #
        #   W = w (2η)² / η² = 4w,
        #   z = η + (y-η²)/(2η) = (η + y/η)/2.
        #
        # Evaluate that identity directly so generic variance/mean floors do
        # not distort genuinely tiny nonzero means.  At the sole singular
        # point η=0, retain the exact 4w limit and enter the branch selected by
        # signed zero at its saturated predictor ±sqrt(y).
        weights = 4.0 * sample_weight
        response = np.empty_like(eta)
        nonzero = eta != 0.0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            response[nonzero] = 0.5 * (eta[nonzero] + y[nonzero] / eta[nonzero])
        response[~nonzero] = np.copysign(
            np.sqrt(y[~nonzero]),
            eta[~nonzero],
        )
        with np.errstate(invalid="ignore", over="ignore"):
            weighted_response = weights * response
            weighted_response_sum = float(np.sum(weighted_response, dtype=np.float64))
        if (
            not np.all(np.isfinite(response[nonzero]))
            or not np.all(np.isfinite(weighted_response[nonzero]))
            or not np.isfinite(weighted_response_sum)
        ):
            # A nonzero subnormal η can make the exact Fisher response exceed
            # float64 even though its sign branch and optimum are well
            # defined.  In that arithmetic-only case use the branch-preserving
            # finite fixed-point response.  This is a trust response, not an
            # epsilon neighbourhood: every representable Fisher system above
            # remains unchanged.
            response = np.copysign(np.sqrt(y), eta)
        return CoefficientWorkingRows(
            weights=np.asarray(weights, dtype=np.float64),
            response=np.asarray(response, dtype=np.float64),
            curvature_source="fisher",
            fallback_reason=fallback_reason,
        )

    # For the exact Gaussian/identity pair the Fisher rows reduce algebraically
    # to the supplied weights and y. Preserve that identity bit-for-bit: the
    # generic ``eta + (y - eta)`` expression introduces iteration-dependent
    # roundoff and defeats constant-geometry factor-certificate reuse.
    if type(distribution) is Gaussian and type(link) is IdentityLink:
        return CoefficientWorkingRows(
            weights=np.array(sample_weight, dtype=np.float64, copy=True),
            response=np.array(y, dtype=np.float64, copy=True),
            curvature_source="fisher",
            fallback_reason=fallback_reason,
        )
    variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    dmu_deta = link.deriv_inverse(eta)
    weights = sample_weight * dmu_deta**2 / variance
    response = eta + (y - mu) / dmu_deta
    return CoefficientWorkingRows(
        weights=np.asarray(weights, dtype=np.float64),
        response=np.asarray(response, dtype=np.float64),
        curvature_source="fisher",
        fallback_reason=fallback_reason,
    )


def coefficient_working_rows(
    *,
    distribution: object,
    link: object,
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    prefer_observed: bool,
) -> CoefficientWorkingRows:
    """Return Fisher rows or a guarded exact observed-Newton quadratic model.

    Gamma/log has positive rowwise observed curvature
    ``w * y / mu`` and score ``w * (y / mu - 1)``.  Applying their ratio in
    the working response avoids forming a large score separately.  Any
    non-finite or non-positive active row rejects the *whole* observed model;
    mixing Fisher and observed rows would no longer be a Newton step for a
    defined objective.
    """
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if not prefer_observed or not supports_observed_newton(distribution, link):
        return _fisher_rows(
            distribution=distribution,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=sample_weight,
        )

    active = sample_weight > 0.0
    observed_weights = np.zeros_like(sample_weight)
    response = eta.copy()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        observed_weights[active] = sample_weight[active] * y[active] / mu[active]
        response[active] += (y[active] - mu[active]) / y[active]
        total_observed_weight = float(np.sum(observed_weights, dtype=np.float64))
    valid = bool(
        np.all(np.isfinite(observed_weights))
        and np.all(np.isfinite(response))
        and np.all(observed_weights[active] > 0.0)
        and total_observed_weight > 0.0
    )
    if not valid:
        return _fisher_rows(
            distribution=distribution,
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=sample_weight,
            fallback_reason="invalid_observed_rows",
        )
    return CoefficientWorkingRows(
        weights=observed_weights,
        response=response,
        curvature_source="observed",
    )
