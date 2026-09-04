"""Primitive row numerics for the epsilon-skew two-piece normal.

Location coordinates ``(mu, sigma, eps)`` with ``T = mu + sigma W`` and

    f_W(w) = phi(w / (1 - eps))   w <  0
             phi(w / (1 + eps))   w >= 0

so the RIGHT piece is the wide one for ``eps > 0`` and positive ``eps`` is
positive skew (Mudholkar and Hutson 2000; Arellano-Valle, Gomez and Quintana
2005; Rubio and Steel 2014, whose Theorem 3 and Corollary 2 give the two
orthogonal blocks of the information below).  ``T = log Y`` is
``TwoPieceLogNormalLSS`` and ``T = Y`` is ``TwoPieceNormalLSS``: one kernel,
two carriers.  The mean form replaces ``mu`` by ``m = E[Y] = exp(mu) K``.

Every row formula is elementary; the only cancellation-sensitive object is the
mean loading, which is evaluated as a log-sum-exp of its two log pieces.  The
derivatives and limiting cases are checked against independent references in
the family kernel tests.

Imports only numpy, scipy, the standard library and the sibling primitive
helpers (primitive-kernel rule).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special

from superglm.distributional.kernels._common import (
    readonly,
    readonly_bool,
    validated_derivative_order,
)

Parametrisation = Literal["mean", "location"]

_FLOAT = np.float64
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_SKEW_INVERSION_MARGIN = 1.0e-6


class TwoPieceDomainError(ValueError):
    """A two-piece row argument left the family's support."""


class TwoPieceInitializationWarning(UserWarning):
    """The moment start had to be clamped or floored to a supported point."""


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=_FLOAT)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise TwoPieceDomainError(f"two-piece {name} must be a non-empty finite vector")
    return array


def _positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array <= 0.0):
        raise TwoPieceDomainError(f"two-piece {name} must be strictly positive")
    return array


def _skew_vector(values: object, *, name: str = "skew") -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(np.abs(array) >= 1.0):
        raise TwoPieceDomainError("two-piece skew must lie strictly inside (-1, 1)")
    return array


def _same_shape(reference: NDArray[np.float64], *arrays: NDArray[np.float64]) -> None:
    if any(array.shape != reference.shape for array in arrays):
        raise TwoPieceDomainError("two-piece row arrays must have the same shape")


@dataclass(frozen=True)
class TwoPieceKernelEvaluation:
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
        for name, width in (("score", 3), ("hessian_packed", 6)):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, readonly(values, shape=(n_rows, width)))


def _placeholders_where_invalid(
    optimizing: NDArray[np.float64],
    score: NDArray[np.float64] | None,
    hessian: NDArray[np.float64] | None,
    valid: NDArray[np.bool_],
) -> TwoPieceKernelEvaluation:
    """Zero every channel of a row that left float64 and flag it invalid.

    A row whose standardised residual overflows has a log density below
    ``-1e308``: the step is infeasible, which the solver answers by shortening
    it, whereas an exception would abort the fit.
    """
    valid = valid & np.isfinite(optimizing)
    for values in (score, hessian):
        if values is not None:
            valid = valid & np.all(np.isfinite(values), axis=1)
    if not np.all(valid):
        invalid = ~valid
        optimizing = np.where(invalid, 0.0, optimizing)
        if score is not None:
            score[invalid] = 0.0
        if hessian is not None:
            hessian[invalid] = 0.0
    return TwoPieceKernelEvaluation(optimizing, score, hessian, valid)


def _pieces(
    variate: NDArray[np.float64],
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    skew: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(u, s, e)``: the standardised residual, its piece scale and ``ds/deps``."""
    z = (variate - mu) / sigma
    left = z < 0.0
    s = np.where(left, 1.0 - skew, 1.0 + skew)
    e = np.where(left, -1.0, 1.0)
    return z / s, s, e


def location_rows(
    variate: NDArray,
    location: NDArray,
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> TwoPieceKernelEvaluation:
    """Log density, score and packed Hessian in ``(mu, sigma, eps)`` per row.

    ``variate`` is ``log y`` for the log-normal law and ``y`` for the real-line
    law; the parameter-free carrier is the caller's.  Every quantity is
    multiplied by ``multiplier`` (the frequency weight; ones under unit prior
    weights).  The log-likelihood is C1 and its Hessian jumps at ``z = 0``.
    """
    t = _vector(variate, name="variate")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    m = _positive_vector(multiplier, name="multiplier")
    _same_shape(t, mu, sigma, eps, m)
    order = validated_derivative_order(derivative_order)
    score = None
    hessian = None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u, s, e = _pieces(t, mu, sigma, eps)
        u2 = u * u
        optimizing = m * (-np.log(sigma) - 0.5 * u2)
        if order >= 1:
            # every channel is formed at unit mass and scaled last, so a
            # frequency weight multiplies the row exactly rather than to
            # within an associativity error.
            score = np.empty((len(t), 3), dtype=_FLOAT)
            score[:, 0] = u / (sigma * s)
            score[:, 1] = (u2 - 1.0) / sigma
            score[:, 2] = u2 * e / s
            if order == 2:
                sigma2 = sigma * sigma
                hessian = np.empty((len(t), 6), dtype=_FLOAT)
                hessian[:, 0] = -1.0 / (sigma * s) ** 2
                hessian[:, 1] = -2.0 * u / (sigma2 * s)
                hessian[:, 2] = -2.0 * u * e / (sigma * s * s)
                hessian[:, 3] = (1.0 - 3.0 * u2) / sigma2
                hessian[:, 4] = -2.0 * u2 * e / (sigma * s)
                hessian[:, 5] = -3.0 * u2 / (s * s)
                hessian *= m[:, None]
            score *= m[:, None]
    return _placeholders_where_invalid(optimizing, score, hessian, np.isfinite(u2))


def two_piece_cdf(
    variate: NDArray, location: NDArray, scale: NDArray, skew: NDArray
) -> NDArray[np.float64]:
    """``P(T <= t)``, piecewise closed form; the split is at ``t = mu``."""
    t = _vector(variate, name="variate")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    _same_shape(t, mu, sigma, eps)
    w = (t - mu) / sigma
    left = w < 0.0
    lower = (1.0 - eps) * special.ndtr(np.where(left, w, -1.0) / (1.0 - eps))
    upper = 0.5 * (1.0 - eps) + (1.0 + eps) * (
        special.ndtr(np.where(left, 1.0, w) / (1.0 + eps)) - 0.5
    )
    return readonly(np.where(left, lower, upper))


def two_piece_quantile(
    p: NDArray, location: NDArray, scale: NDArray, skew: NDArray
) -> NDArray[np.float64]:
    """Quantile of ``T`` for ``p`` strictly inside ``(0, 1)``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise TwoPieceDomainError("quantile probabilities must lie strictly inside (0, 1)")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    _same_shape(mu, probabilities, sigma, eps)
    split = 0.5 * (1.0 - eps)
    below = probabilities < split
    lower = (1.0 - eps) * special.ndtri(np.where(below, probabilities, 0.25) / (1.0 - eps))
    upper = (1.0 + eps) * special.ndtri(
        (np.where(below, 0.75, probabilities) - split) / (1.0 + eps) + 0.5
    )
    return readonly(mu + sigma * np.where(below, lower, upper))


def location_expected_information(
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed ``-E[H]`` in ``(mu, sigma, eps)`` per row (channel order as the Hessian).

    ``U = W/s(W)`` has density ``(1-eps) phi(u)`` on ``u < 0`` and
    ``(1+eps) phi(u)`` on ``u >= 0``, so each entry is a half-normal moment and
    ``h = 1/(1-eps) + 1/(1+eps) = 2/(1-eps^2)`` collects all of them.  The two
    zeros are exact: Rubio and Steel (2014) Theorem 3 gives ``mu _|_ sigma`` for
    every two-piece parametrisation, and Corollary 2 gives ``sigma _|_ eps``
    exactly when ``a(eps) + b(eps)`` is constant, which the epsilon-skew form
    satisfies with ``a + b = 2``.
    """
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    m = _positive_vector(multiplier, name="multiplier")
    _same_shape(sigma, eps, m)
    h = 2.0 / (1.0 - eps * eps)
    sigma2 = sigma * sigma
    information = np.empty((len(sigma), 6), dtype=_FLOAT)
    information[:, 0] = 0.5 * h / sigma2
    information[:, 1] = 0.0
    information[:, 2] = _SQRT_2_OVER_PI * h / sigma
    information[:, 3] = 2.0 / sigma2
    information[:, 4] = 0.0
    information[:, 5] = 1.5 * h
    information = information * m[:, None]
    if not np.all(np.isfinite(information)):
        raise TwoPieceDomainError("two-piece expected information is not representable")
    return readonly(information)


# ---------------------------------------------------------------- mean form


def log_mean_loading(scale: NDArray, skew: NDArray) -> tuple[NDArray[np.float64], ...]:
    """``log K`` and its five derivatives in ``(sigma, eps)``; always finite.

    ``K(sigma, eps) = E[e^(sigma W)]`` is the mean loading ``E[Y]/exp(mu)`` of
    the log-normal variant.  Both pieces are evaluated in the log domain, where
    ``a^2/2 + log Phi(-a)`` stays bounded instead of overflowing against an
    underflowing tail; the derivatives then follow the log-sum-exp (cumulant)
    rule over the two pieces, which is what keeps the mixture weights and the
    per-piece derivatives separately well scaled.

    Returns ``(log K, K_sigma, K_eps, K_sigma_sigma, K_sigma_eps, K_eps_eps)``.
    At ``eps = 0``: ``log K = sigma^2/2``, ``K_sigma = sigma``,
    ``K_eps = (1 + sigma^2)(2 Phi(sigma) - 1) + 2 sigma phi(sigma)``,
    ``K_sigma_sigma = 1``, and the two mixed limits, which have no shorter
    closed form.  ``K_eps`` tends to ``2 sigma sqrt(2/pi)`` as ``sigma -> 0``,
    not to ``sigma sqrt(2/pi)``: ``E[W] = 2 eps sqrt(2/pi)`` carries the two.
    """
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    _same_shape(sigma, eps)
    left_width, right_width = 1.0 - eps, 1.0 + eps
    a1, a2 = sigma * left_width, sigma * right_width
    log_left = np.log1p(-eps) + 0.5 * a1 * a1 + special.log_ndtr(-a1)
    log_right = np.log1p(eps) + 0.5 * a2 * a2 + special.log_ndtr(a2)
    logk = np.logaddexp(log_left, log_right)
    p_left = np.exp(log_left - logk)
    p_right = np.exp(log_right - logk)
    # inverse Mills ratios, formed in the log domain so the far tail survives
    r1 = np.exp(-0.5 * a1 * a1 - _HALF_LOG_TWO_PI - special.log_ndtr(-a1))
    r2 = np.exp(-0.5 * a2 * a2 - _HALF_LOG_TWO_PI - special.log_ndtr(a2))
    g1, g1d = a1 - r1, 1.0 + a1 * r1 - r1 * r1
    g2, g2d = a2 + r2, 1.0 - a2 * r2 - r2 * r2
    d1s, d1e = left_width * g1, -1.0 / left_width - sigma * g1
    d2s, d2e = right_width * g2, 1.0 / right_width + sigma * g2
    d1ss = left_width * left_width * g1d
    d1se = -g1 - sigma * left_width * g1d
    d1ee = -1.0 / (left_width * left_width) + sigma * sigma * g1d
    d2ss = right_width * right_width * g2d
    d2se = g2 + sigma * right_width * g2d
    d2ee = -1.0 / (right_width * right_width) + sigma * sigma * g2d
    k_s = p_left * d1s + p_right * d2s
    k_e = p_left * d1e + p_right * d2e
    k_ss = p_left * (d1ss + d1s * d1s) + p_right * (d2ss + d2s * d2s) - k_s * k_s
    k_se = p_left * (d1se + d1s * d1e) + p_right * (d2se + d2s * d2e) - k_s * k_e
    k_ee = p_left * (d1ee + d1e * d1e) + p_right * (d2ee + d2e * d2e) - k_e * k_e
    values = (logk, k_s, k_e, k_ss, k_se, k_ee)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise TwoPieceDomainError("two-piece mean loading is not representable")
    return values


def location_of_mean(mean: NDArray, scale: NDArray, skew: NDArray) -> NDArray[np.float64]:
    """``mu = log m - log K(sigma, eps)``."""
    m = _positive_vector(mean, name="mean")
    logk = log_mean_loading(scale, skew)[0]
    _same_shape(m, logk)
    return readonly(np.log(m) - logk)


def mean_of_location(location: NDArray, scale: NDArray, skew: NDArray) -> NDArray[np.float64]:
    """``E[Y] = exp(mu) K(sigma, eps)`` for the log-normal variant."""
    mu = _vector(location, name="location")
    logk = log_mean_loading(scale, skew)[0]
    _same_shape(mu, logk)
    with np.errstate(over="ignore"):
        return readonly(np.exp(mu + logk))


def real_line_mean(location: NDArray, scale: NDArray, skew: NDArray) -> NDArray[np.float64]:
    """``E[T] = mu + 2 eps sigma sqrt(2/pi)`` for the real-line variant."""
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    _same_shape(mu, sigma, eps)
    return readonly(mu + 2.0 * eps * sigma * _SQRT_2_OVER_PI)


def mean_rows(
    response: NDArray,
    mean: NDArray,
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> TwoPieceKernelEvaluation:
    """Mean-form rows ``(m, sigma, eps)`` chained through ``mu = log m - log K``.

    Unlike the generalized gamma there is no infinite-mean region: ``K`` is
    finite for every ``(sigma, eps)`` in the support, so every row that is
    representable is valid.  The chain rule runs at unit mass and the
    multiplier is applied last, so a frequency weight scales the row exactly.
    """
    y = _positive_vector(response, name="response")
    m = _positive_vector(mean, name="mean")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    weight = _positive_vector(multiplier, name="multiplier")
    _same_shape(y, m, sigma, eps, weight)
    order = validated_derivative_order(derivative_order)
    logk, k_s, k_e, k_ss, k_se, k_ee = log_mean_loading(sigma, eps)
    base = location_rows(
        np.log(y), np.log(m) - logk, sigma, eps, np.ones_like(weight), derivative_order=order
    )
    optimizing = weight * np.asarray(base.optimizing_log_likelihood)
    score = None
    hessian = None
    if base.score is not None:
        s_mu, s_sigma, s_eps = (base.score[:, i] for i in range(3))
        score = np.empty_like(base.score)
        score[:, 0] = s_mu / m
        score[:, 1] = s_sigma - s_mu * k_s
        score[:, 2] = s_eps - s_mu * k_e
        if base.hessian_packed is not None:
            h_mm, h_ms, h_me, h_ss, h_se, h_ee = (base.hessian_packed[:, i] for i in range(6))
            hessian = np.empty_like(base.hessian_packed)
            hessian[:, 0] = (h_mm - s_mu) / (m * m)
            hessian[:, 1] = (h_ms - h_mm * k_s) / m
            hessian[:, 2] = (h_me - h_mm * k_e) / m
            hessian[:, 3] = h_ss - 2.0 * h_ms * k_s + h_mm * k_s * k_s - s_mu * k_ss
            hessian[:, 4] = h_se - h_ms * k_e - h_me * k_s + h_mm * k_s * k_e - s_mu * k_se
            hessian[:, 5] = h_ee - 2.0 * h_me * k_e + h_mm * k_e * k_e - s_mu * k_ee
            hessian *= weight[:, None]
        score *= weight[:, None]
    return _placeholders_where_invalid(optimizing, score, hessian, np.asarray(base.valid))


def mean_expected_information(
    mean: NDArray,
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed mean-form expected information ``J^T I J``."""
    m = _positive_vector(mean, name="mean")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    weight = _positive_vector(multiplier, name="multiplier")
    _same_shape(m, sigma, eps, weight)
    packed = location_expected_information(sigma, eps, np.ones_like(weight))
    _, k_s, k_e = log_mean_loading(sigma, eps)[:3]
    i_mm, i_ms, i_me, i_ss, i_se, i_ee = (packed[:, i] for i in range(6))
    out = np.empty_like(packed)
    out[:, 0] = i_mm / (m * m)
    out[:, 1] = (i_ms - i_mm * k_s) / m
    out[:, 2] = (i_me - i_mm * k_e) / m
    out[:, 3] = i_ss - 2.0 * i_ms * k_s + i_mm * k_s * k_s
    out[:, 4] = i_se - i_ms * k_e - i_me * k_s + i_mm * k_s * k_e
    out[:, 5] = i_ee - 2.0 * i_me * k_e + i_mm * k_e * k_e
    out *= weight[:, None]
    if not np.all(np.isfinite(out)):
        raise TwoPieceDomainError("two-piece expected information is not representable")
    return readonly(out)


# ---------------------------------------------------------------- initialiser

_A1 = 1.0 / math.sqrt(2.0 * math.pi)


def standard_variance(skew: NDArray) -> NDArray[np.float64]:
    """``Var[W] = 1 + 3 eps^2 - 8 eps^2/pi`` (Wallis 2014's two-piece variance)."""
    eps = np.asarray(skew, dtype=_FLOAT)
    return 1.0 + 3.0 * eps * eps - 8.0 * eps * eps / math.pi


def standard_skewness(skew: NDArray) -> NDArray[np.float64]:
    """Population skewness of ``W``: strictly increasing, reaching +/-0.9655 at 0.899."""
    eps = np.asarray(skew, dtype=_FLOAT)
    third = _A1 * eps * (4.0 - 20.0 * eps * eps) + 128.0 * _A1**3 * eps**3
    return third / standard_variance(eps) ** 1.5


def skew_from_sample_skewness(value: float, *, bound: float) -> tuple[float, bool]:
    """Invert the monotone skewness map inside ``(-bound, bound)``; ``(eps, clamped)``."""
    edge = bound - _SKEW_INVERSION_MARGIN
    reach = float(standard_skewness(np.array([edge]))[0])
    clamped = not math.isfinite(value) or abs(value) > reach
    if clamped:
        target = math.copysign(reach, value if math.isfinite(value) else 0.0)
        warnings.warn(
            f"sample skewness {value:.4f} lies outside the two-piece range "
            f"(+/-{reach:.4f}) at skew_bound {bound}; starting from the clamp "
            f"{target:+.4f}",
            TwoPieceInitializationWarning,
            stacklevel=3,
        )
    else:
        target = float(value)
    if abs(target) < 1.0e-12:
        return 0.0, clamped
    root = optimize.brentq(
        lambda e: float(standard_skewness(np.array([e]))[0]) - target, -edge, edge, xtol=1e-14
    )
    return float(root), clamped


def _weighted_moments(
    variate: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Weighted mean, variance and skewness of the variate (weights as replication)."""
    total = float(np.sum(weights, dtype=_FLOAT))
    mean = float(np.sum(weights * variate, dtype=_FLOAT) / total)
    centred = variate - mean
    variance = float(np.sum(weights * centred * centred, dtype=_FLOAT) / total)
    if variance <= 0.0 or not math.isfinite(variance):
        return mean, max(variance, 1.0e-12), 0.0
    skewness = float(np.sum(weights * centred**3, dtype=_FLOAT) / total / variance**1.5)
    return mean, variance, skewness


def initialize_two_piece(
    variate: NDArray,
    mass: NDArray,
    *,
    parametrisation: Parametrisation,
    scale_floor: float,
    skew_bound: float,
) -> NDArray[np.float64]:
    """Constant start from the weighted moments of the variate.

    ``variate`` is always the quantity the location parameter lives on:
    ``log y`` for either form of the log-normal family, ``y`` for the real-line
    family.  Taking the logarithm is the caller's job, so the kernel never has
    to guess which family it is serving.  ``parametrisation`` decides only what
    the first column carries: ``mu`` for the location form, ``m = exp(mu) K``
    for the mean form.  Skewness fixes ``eps``, the variance then fixes
    ``sigma``, and the mean fixes ``mu``.
    """
    if parametrisation not in ("mean", "location"):
        raise TwoPieceDomainError(f"unsupported parametrisation: {parametrisation!r}")
    values = _vector(variate, name="variate")
    weights = _positive_vector(mass, name="mass")
    _same_shape(values, weights)
    mean, variance, skewness = _weighted_moments(values, weights)
    eps, _ = skew_from_sample_skewness(skewness, bound=skew_bound)
    sigma = math.sqrt(variance / float(standard_variance(np.array([eps]))[0]))
    margin = max(1.0e-8, math.sqrt(np.finfo(_FLOAT).eps) * max(1.0, abs(mean)))
    sigma = max(sigma, scale_floor + margin)
    mu = mean - 2.0 * eps * sigma * _SQRT_2_OVER_PI
    if parametrisation == "mean":
        first = float(mean_of_location(np.array([mu]), np.array([sigma]), np.array([eps]))[0])
    else:
        first = mu
    n = len(values)
    return readonly(np.column_stack((np.full(n, first), np.full(n, sigma), np.full(n, eps))))


__all__ = [
    "Parametrisation",
    "TwoPieceDomainError",
    "TwoPieceInitializationWarning",
    "TwoPieceKernelEvaluation",
    "initialize_two_piece",
    "location_expected_information",
    "location_of_mean",
    "location_rows",
    "log_mean_loading",
    "mean_expected_information",
    "mean_of_location",
    "mean_rows",
    "real_line_mean",
    "skew_from_sample_skewness",
    "standard_skewness",
    "standard_variance",
    "two_piece_cdf",
    "two_piece_quantile",
]
