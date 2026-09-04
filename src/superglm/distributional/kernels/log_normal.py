"""Primitive row numerics for the two-parameter log-normal.

Coordinates: location form ``(mu, sigma)`` with ``log Y ~ N(mu, sigma^2)`` on
``y > 0``; the mean form replaces ``mu`` by ``m = E[Y] = exp(mu) C(sigma)``
with the mean loading ``log C(sigma) = sigma^2 / 2``.  The mean always exists,
so this family has no invalid region and no validity mask beyond ``y > 0``,
which the adapter refuses at ``bind_likelihood``.

The optimising log-likelihood is ``-log sigma - w^2 / 2`` with
``w = (log y - mu)/sigma``; the carrier ``-log y - log(2 pi)/2`` is the
caller's.  That split is the generalized gamma's, not the Gaussian's, so the
two families' optimising channels coincide exactly at shape zero.

In ``(mu, sigma)`` the score and Hessian are the Gaussian kernel's at
``z = log y``, so ``location_rows`` delegates to the Gaussian sibling (the one
package-internal dependency the primitive-kernel rule permits) and moves the
constant.  ``multiplier`` is a replication mass and must be exact positive
integers, which the sibling validates.

Overflow: ``sigma`` is floored and its link is stabilised, so every channel
here stays inside float64 for any state the solver can reach, and the Gaussian
sibling's finiteness check never fires.  The family therefore carries exactly
``GaussianLS``'s numerical risk profile, on the log scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.kernels._common import readonly, readonly_bool
from superglm.distributional.kernels.gaussian import (
    evaluate_gaussian_rows,
    gaussian_expected_information,
)

Parametrisation = Literal["mean", "location"]
_FLOAT = np.float64
_EPS = np.finfo(_FLOAT).eps
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_TAIL_INVERSE_RTOL = 64.0 * math.sqrt(_EPS)


class LogNormalDomainError(ValueError):
    """A log-normal row argument left the family's domain."""


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=_FLOAT)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise LogNormalDomainError(f"log-normal {name} must be a finite non-empty vector")
    return array


def _positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array <= 0.0):
        raise LogNormalDomainError(f"log-normal {name} must be strictly positive")
    return array


def _aligned(reference: NDArray[np.float64], *others: NDArray[np.float64]) -> None:
    if any(values.shape != reference.shape for values in others):
        raise LogNormalDomainError("log-normal row arrays must have the same shape")


@dataclass(frozen=True)
class LogNormalKernelEvaluation:
    """Immutable row evaluation with read-only outputs."""

    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n_rows = len(self.optimizing_log_likelihood)
        object.__setattr__(
            self,
            "optimizing_log_likelihood",
            readonly(self.optimizing_log_likelihood, shape=(n_rows,)),
        )
        object.__setattr__(self, "valid", readonly_bool(self.valid, shape=(n_rows,)))
        for name, width in (("score", 2), ("hessian_packed", 3)):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, readonly(values, shape=(n_rows, width)))


def location_rows(
    response: NDArray,
    mu: NDArray,
    sigma: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> LogNormalKernelEvaluation:
    """Log-normal optimising log density, score and packed Hessian in ``(mu, sigma)``."""
    y = _positive_vector(response, name="response")
    location = _vector(mu, name="mu")
    scale = _positive_vector(sigma, name="sigma")
    mass = _positive_vector(multiplier, name="multiplier")
    _aligned(y, location, scale, mass)
    evaluated = evaluate_gaussian_rows(
        np.log(y), location, scale, mass, "frequency", derivative_order=derivative_order
    )
    return LogNormalKernelEvaluation(
        optimizing_log_likelihood=evaluated.optimizing_log_likelihood + mass * _HALF_LOG_TWO_PI,
        score=evaluated.score,
        hessian_packed=evaluated.hessian_packed,
        valid=evaluated.valid,
    )


def location_expected_information(sigma: NDArray, multiplier: NDArray) -> NDArray[np.float64]:
    """Packed ``-E[Hessian]`` in ``(mu, sigma)``: ``diag(1/sigma^2, 2/sigma^2)`` per row."""
    scale = _positive_vector(sigma, name="sigma")
    mass = _positive_vector(multiplier, name="multiplier")
    _aligned(scale, mass)
    return gaussian_expected_information(scale, mass, "frequency")


def log_normal_cdf(y: NDArray, mu: NDArray, sigma: NDArray) -> NDArray[np.float64]:
    """``P(Y <= y) = Phi((log y - mu)/sigma)`` in location coordinates."""
    response = _vector(y, name="y")
    location = _vector(mu, name="mu")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(response, location, scale)
    out = np.zeros_like(response)
    interior = response > 0.0
    if np.any(interior):
        out[interior] = special.ndtr(
            (np.log(response[interior]) - location[interior]) / scale[interior]
        )
    return readonly(out)


def log_normal_quantile(p: NDArray, mu: NDArray, sigma: NDArray) -> NDArray[np.float64]:
    """``exp(mu + sigma Phi^-1(p))`` for ``p`` strictly inside ``(0, 1)``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise LogNormalDomainError("quantile probabilities must lie strictly inside (0, 1)")
    location = _vector(mu, name="mu")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(probabilities, location, scale)
    with np.errstate(over="ignore"):
        return readonly(np.exp(location + scale * special.ndtri(probabilities)))


def log_normal_expected_shortfall(p: NDArray, mu: NDArray, sigma: NDArray) -> NDArray[np.float64]:
    """``E[Y | Y > q_p]`` in location coordinates, using the stable normal tail."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise LogNormalDomainError(
            "expected-shortfall probabilities must lie strictly inside (0, 1)"
        )
    location = _vector(mu, name="mu")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(probabilities, location, scale)
    z, log_tail_ratio, log_quantile_ratio = log_normal_tail_log_factors(probabilities, scale)
    with np.errstate(over="ignore"):
        result = np.exp(location + 0.5 * scale * scale + log_tail_ratio)
        quantile = np.exp(location + scale * z)
    log_gap = log_tail_ratio - log_quantile_ratio
    return _certified_expected_shortfall_output(result, quantile, log_gap=log_gap)


def log_normal_tail_log_factors(
    p: NDArray, sigma: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``z_p``, log tail-to-mean ratio, and log quantile-to-mean ratio."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise LogNormalDomainError(
            "expected-shortfall probabilities must lie strictly inside (0, 1)"
        )
    scale = _positive_vector(sigma, name="sigma")
    _aligned(probabilities, scale)
    z = special.ndtri(probabilities)
    requested_log_survival = np.log1p(-probabilities)
    achieved_log_survival = special.log_ndtr(-z)
    inverse_residual = np.abs(np.expm1(achieved_log_survival - requested_log_survival))
    if np.any(inverse_residual > _TAIL_INVERSE_RTOL):
        rows = np.flatnonzero(inverse_residual > _TAIL_INVERSE_RTOL).tolist()
        raise LogNormalDomainError(
            f"log-normal expected shortfall cannot be certified in float64 for rows {rows}"
        )

    log_tail_ratio = special.log_ndtr(scale - z) - requested_log_survival
    small_shift = scale <= math.sqrt(_EPS) / np.maximum(1.0, np.abs(z))
    if np.any(small_shift):
        selected_z = z[small_shift]
        log_inverse_mills = (
            -0.5 * selected_z * selected_z - _HALF_LOG_TWO_PI - achieved_log_survival[small_shift]
        )
        inverse_mills = np.exp(log_inverse_mills)
        selected_scale = scale[small_shift]
        log_tail_ratio[small_shift] = (
            selected_scale * inverse_mills
            + 0.5 * selected_scale * selected_scale * inverse_mills * (selected_z - inverse_mills)
        )
    log_quantile_ratio = scale * z - 0.5 * scale * scale
    return readonly(z), readonly(log_tail_ratio), readonly(log_quantile_ratio)


def _certified_expected_shortfall_output(
    result: NDArray[np.float64],
    quantile: NDArray[np.float64],
    *,
    log_gap: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Require the represented expected shortfall to dominate its quantile."""
    certified = ~np.isnan(result) & (result >= quantile) & (log_gap >= 0.0)
    if not np.all(certified):
        rows = np.flatnonzero(~certified).tolist()
        raise LogNormalDomainError(
            f"log-normal expected shortfall cannot be certified in float64 for rows {rows}"
        )
    return readonly(result)


def log_normal_expected_shortfall_from_mean(
    p: NDArray, mean: NDArray, sigma: NDArray
) -> NDArray[np.float64]:
    """``E[Y | Y > q_p]`` using an exactly supplied mean parameter."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    mean_values = _positive_vector(mean, name="mean")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(probabilities, mean_values, scale)
    z, log_tail_ratio, log_quantile_ratio = log_normal_tail_log_factors(probabilities, scale)
    with np.errstate(over="ignore"):
        result = mean_values * np.exp(log_tail_ratio)
        location = np.log(mean_values) - 0.5 * scale * scale
        quantile = np.exp(location + scale * z)
    log_gap = log_tail_ratio - log_quantile_ratio
    return _certified_expected_shortfall_output(
        result,
        quantile,
        log_gap=log_gap,
    )


def log_mean_loading(
    sigma: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(log C, dlog C/dsigma, d2 log C/dsigma2)`` for ``E[Y] = exp(mu) C(sigma)``.

    ``log C(sigma) = sigma^2 / 2`` is finite at every scale, so the log-normal
    mean form has no infinite-mean region.
    """
    scale = _positive_vector(sigma, name="sigma")
    return 0.5 * scale * scale, scale, np.ones_like(scale)


def location_of_mean(mean: NDArray, sigma: NDArray) -> NDArray[np.float64]:
    """``mu = log m - sigma^2 / 2``."""
    first = _positive_vector(mean, name="mean")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(first, scale)
    return readonly(np.log(first) - 0.5 * scale * scale)


def mean_of_location(mu: NDArray, sigma: NDArray) -> NDArray[np.float64]:
    """``E[Y] = exp(mu + sigma^2 / 2)``; ``inf`` when that exceeds float64."""
    location = _vector(mu, name="mu")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(location, scale)
    with np.errstate(over="ignore"):
        return readonly(np.exp(location + 0.5 * scale * scale))


def mean_rows(
    response: NDArray,
    mean: NDArray,
    sigma: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> LogNormalKernelEvaluation:
    """The same channels in ``(m, sigma)``, chained through ``mu = log m - log C``."""
    first = _positive_vector(mean, name="mean")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(first, scale)
    _, c_sigma, c_sigma_2 = log_mean_loading(scale)
    base = location_rows(
        response,
        location_of_mean(first, scale),
        scale,
        multiplier,
        derivative_order=derivative_order,
    )
    score = None
    hessian = None
    if base.score is not None:
        s_mu, s_sigma = base.score[:, 0], base.score[:, 1]
        score = np.column_stack((s_mu / first, s_sigma - s_mu * c_sigma))
        if base.hessian_packed is not None:
            h_mumu, h_musigma, h_sigmasigma = (base.hessian_packed[:, i] for i in range(3))
            hessian = np.column_stack(
                (
                    (h_mumu - s_mu) / (first * first),
                    (h_musigma - h_mumu * c_sigma) / first,
                    h_sigmasigma
                    - 2.0 * h_musigma * c_sigma
                    + h_mumu * c_sigma * c_sigma
                    - s_mu * c_sigma_2,
                )
            )
    return LogNormalKernelEvaluation(
        optimizing_log_likelihood=base.optimizing_log_likelihood,
        score=score,
        hessian_packed=hessian,
        valid=base.valid,
    )


def mean_expected_information(
    mean: NDArray, sigma: NDArray, multiplier: NDArray
) -> NDArray[np.float64]:
    """Packed mean-form expected information ``J^T I J``."""
    first = _positive_vector(mean, name="mean")
    scale = _positive_vector(sigma, name="sigma")
    _aligned(first, scale)
    packed = location_expected_information(scale, multiplier)
    _, c_sigma, _ = log_mean_loading(scale)
    i_mumu, i_musigma, i_sigmasigma = (packed[:, i] for i in range(3))
    return readonly(
        np.column_stack(
            (
                i_mumu / (first * first),
                (i_musigma - i_mumu * c_sigma) / first,
                i_sigmasigma - 2.0 * i_musigma * c_sigma + i_mumu * c_sigma * c_sigma,
            )
        )
    )


def initialize_log_normal(
    response: NDArray,
    mass: NDArray,
    *,
    parametrisation: Parametrisation,
    scale_floor: float,
) -> NDArray[np.float64]:
    """Constant start from the weighted log moments.

    ``mass`` is the replication weight per row (ones under unit prior weights).
    The floor is applied before the mean is formed, so the mean-form start maps
    back to exactly the location-form start.
    """
    y = _positive_vector(response, name="response")
    weights = _positive_vector(mass, name="mass")
    _aligned(y, weights)
    if parametrisation not in ("mean", "location"):
        raise LogNormalDomainError(f"unsupported parametrisation: {parametrisation!r}")
    z = np.log(y)
    total = float(np.sum(weights, dtype=_FLOAT))
    location = float(np.sum(weights * z, dtype=_FLOAT) / total)
    centred = z - location
    variance = float(np.sum(weights * centred * centred, dtype=_FLOAT) / total)
    margin = max(1.0e-8, math.sqrt(np.finfo(_FLOAT).eps) * max(1.0, abs(location)))
    scale = max(math.sqrt(max(variance, 0.0)), float(scale_floor) + margin)
    first = (
        location
        if parametrisation == "location"
        else float(mean_of_location(np.array([location]), np.array([scale]))[0])
    )
    return readonly(np.column_stack((np.full(len(y), first), np.full(len(y), scale))))


__all__ = [
    "LogNormalDomainError",
    "LogNormalKernelEvaluation",
    "Parametrisation",
    "initialize_log_normal",
    "location_expected_information",
    "location_of_mean",
    "location_rows",
    "log_mean_loading",
    "log_normal_cdf",
    "log_normal_expected_shortfall",
    "log_normal_expected_shortfall_from_mean",
    "log_normal_quantile",
    "log_normal_tail_log_factors",
    "mean_expected_information",
    "mean_of_location",
    "mean_rows",
]
