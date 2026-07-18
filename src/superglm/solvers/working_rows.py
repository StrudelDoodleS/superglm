"""Working-response geometry for direct coefficient fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR, Gamma
from superglm.links import LogLink


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
