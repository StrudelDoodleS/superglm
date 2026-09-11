"""Working-response geometry for direct coefficient fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import (
    _VARIANCE_FLOOR,
    Binomial,
    Distribution,
    Gamma,
    Gaussian,
    NegativeBinomial,
    Poisson,
    Tweedie,
    _weighted_residual_square,
    initial_mean,
)
from superglm.links import IdentityLink, Link, LogitLink, LogLink, SqrtLink


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


def fisher_working_weights(
    *,
    distribution: object,
    link: object,
    mu: NDArray,
    eta: NDArray,
    sample_weight: NDArray,
    dmu_deta: NDArray | None = None,
    variance: NDArray | None = None,
) -> NDArray:
    """Return expected curvature without premature weighted-product rounding.

    Built-in log pairs use the fitted mean, which is the inverse of stabilized
    eta. Their algebraic reductions keep unweighted curvature in range and
    avoid recomputing the inverse link. Exact types preserve custom variance
    and link overrides. Poisson/sqrt uses the exact 4w curvature, including at
    zero, with the unfloored working response in ``_fisher_rows``. Other pairs
    retain the variance floor and their actual inverse-link derivative,
    including when the mean has been clipped.
    """
    family_type, link_type = type(distribution), type(link)
    if (family_type is Gaussian and link_type is IdentityLink) or (
        family_type is Gamma and link_type is LogLink
    ):
        return np.array(sample_weight, dtype=np.float64, copy=True)
    if family_type is Poisson and link_type is SqrtLink:
        return 4.0 * sample_weight
    if link_type is LogLink:
        if family_type is Poisson:
            return sample_weight * mu
        if family_type is Tweedie:
            return sample_weight * np.power(mu, 2.0 - distribution.p)
        if family_type is NegativeBinomial:
            # mu*theta/(mu+theta), with neither an overflowing mu/theta nor
            # an overflowing product. Both temporaries are owned here.
            curvature = np.minimum(mu, distribution.theta)
            ratio = np.maximum(mu, distribution.theta)
            np.divide(curvature, ratio, out=ratio)
            ratio += 1.0
            np.divide(curvature, ratio, out=curvature)
            curvature *= sample_weight
            return curvature
    if dmu_deta is None:
        dmu_deta = link.deriv_inverse(eta)
    if variance is None:
        variance = distribution.variance(mu)
    variance = np.maximum(variance, _VARIANCE_FLOOR)
    if family_type is Binomial and link_type is LogitLink:
        # The clipped mean need not equal expit(eta), so dmu != V(mu).
        # For the stabilized logit range, dmu/V and this unit curvature are
        # normal. Applying mass last avoids subnormal weighted numerators.
        return sample_weight * (dmu_deta * (dmu_deta / variance))

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        squared = dmu_deta**2
        numerator = sample_weight * squared
        result = numerator / variance
    tiny = np.finfo(np.float64).tiny
    repair = (
        (~np.isfinite(squared) | (squared < tiny))
        | (~np.isfinite(numerator) | (np.abs(numerator) < tiny))
    ) & (
        (sample_weight != 0.0)
        & (dmu_deta != 0.0)
        & np.isfinite(sample_weight)
        & np.isfinite(dmu_deta)
        & np.isfinite(variance)
    )
    if np.any(repair):
        from superglm.distributional.kernels._common import _NumericalEvaluationError
        from superglm.distributional.kernels.gamma import _binary_product_divide

        for index in np.flatnonzero(repair):
            derivative = float(dmu_deta[index])
            try:
                result[index] = _binary_product_divide(
                    (float(sample_weight[index]), derivative, derivative),
                    (float(variance[index]),),
                )
            except _NumericalEvaluationError:
                result[index] = np.inf
    result[sample_weight == 0.0] = 0.0
    return result


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
    if type(distribution) is Gamma and type(link) is LogLink:
        # V(mu)=mu**2 and dmu/deta=mu cancel exactly. Multiplying the
        # supplied weight by mu**2 first can overflow or underflow before
        # that cancellation, even though W=w is representable.
        response = eta.copy()
        active = sample_weight > 0.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            response[active] += (y[active] - mu[active]) / mu[active]
        return CoefficientWorkingRows(
            weights=np.array(sample_weight, dtype=np.float64, copy=True),
            response=response,
            curvature_source="fisher",
            fallback_reason=fallback_reason,
        )
    reuse_mean = type(link) is LogLink and type(distribution) in (
        Poisson,
        NegativeBinomial,
        Tweedie,
    )
    dmu_deta = mu if reuse_mean else link.deriv_inverse(eta)
    weights = fisher_working_weights(
        distribution=distribution,
        link=link,
        mu=mu,
        eta=eta,
        sample_weight=sample_weight,
        dmu_deta=dmu_deta,
    )
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
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        product = sample_weight[active] * y[active]
        observed_weights[active] = product / mu[active]
        response[active] += (y[active] - mu[active]) / y[active]

    # Preserve ordinary vector arithmetic. Only an intermediate outside the
    # normal float range needs exponent composition; a subnormal intermediate
    # can lose relative accuracy before division restores a normal result.
    repair = (~np.isfinite(product) | (product < np.finfo(np.float64).tiny)) & (
        np.isfinite(sample_weight[active])
        & np.isfinite(y[active])
        & (y[active] > 0.0)
        & np.isfinite(mu[active])
        & (mu[active] > 0.0)
    )
    if np.any(repair):
        from superglm.distributional.kernels._common import _NumericalEvaluationError
        from superglm.distributional.kernels.gamma import _binary_product_divide

        for index in np.flatnonzero(active)[repair]:
            try:
                observed_weights[index] = _binary_product_divide(
                    (float(sample_weight[index]), float(y[index])), (float(mu[index]),)
                )
            except _NumericalEvaluationError:
                # The requested curvature itself does not fit. Leave the
                # whole-model fallback below in charge of choosing Fisher.
                observed_weights[index] = np.inf

    with np.errstate(over="ignore", invalid="ignore"):
        total_observed_weight = float(np.sum(observed_weights, dtype=np.float64))
    valid = bool(
        np.all(np.isfinite(observed_weights))
        and np.all(np.isfinite(response))
        and np.all(observed_weights[active] > 0.0)
        and np.isfinite(total_observed_weight)
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


def pearson_chi2(
    *,
    distribution: Distribution,
    y: NDArray,
    mu: NDArray,
    sample_weight: NDArray,
    variance_floor: float = _VARIANCE_FLOOR,
) -> float:
    """Return the weighted Pearson sum used by retained-fit dispersion."""
    with np.errstate(over="ignore", invalid="ignore"):
        variance = np.maximum(distribution.variance(mu), variance_floor)
    contributions = _weighted_residual_square(y, mu, sample_weight, variance)
    # NB2 can have an unrepresentable unweighted V(mu) even at a clipped mean.
    # Its factored denominator retains the weighted statistic in that case.
    repair = ~np.isfinite(variance) & (sample_weight != 0.0) & (y != mu)
    if type(distribution) is NegativeBinomial and np.any(repair):
        from superglm.distributional.kernels.gamma import _binary_product_divide

        for index in np.flatnonzero(repair):
            delta = float(y[index]) - float(mu[index])
            try:
                contributions[index] = _binary_product_divide(
                    (float(sample_weight[index]), delta, delta, float(distribution.theta)),
                    (float(mu[index]), float(mu[index]) + float(distribution.theta)),
                )
            except ValueError:
                contributions[index] = np.inf
    if type(distribution) in (Gamma, Tweedie):
        # Unfloored reporting can be requested outside the fitted-mean guard.
        # A square-root variance remains representable when V itself does not.
        repair = (
            (~np.isfinite(variance) | (variance == 0.0))
            & (sample_weight != 0.0)
            & (y != mu)
            & (mu > 0.0)
        )
        if np.any(repair):
            from superglm.distributional.kernels.gamma import _binary_product_divide

            roots = (
                mu[repair]
                if type(distribution) is Gamma
                else np.power(mu[repair], distribution.p / 2.0)
            )
            for index, root in zip(np.flatnonzero(repair), roots, strict=True):
                delta = float(y[index]) - float(mu[index])
                try:
                    contributions[index] = _binary_product_divide(
                        (float(sample_weight[index]), delta, delta), (float(root), float(root))
                    )
                except ValueError:
                    contributions[index] = np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.sum(contributions))
