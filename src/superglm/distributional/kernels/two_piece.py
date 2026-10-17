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
from superglm.distributional.kernels._curvature_direction import (
    compose_derivatives,
    mean_form_hessian_direction,
    packed_tuples,
    stack_columns,
)
from superglm.distributional.kernels._fourth_generated import (
    two_piece_left_mills,
    two_piece_location_third_fourth,
    two_piece_piece_loading,
    two_piece_right_mills,
)
from superglm.distributional.kernels._weighted import weighted_natural_channel

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
                hessian = np.empty((len(t), 6), dtype=_FLOAT)
                hessian[:, 0] = ((-1.0 / s) / sigma / s) / sigma
                hessian[:, 1] = ((-2.0 * u / s) / sigma) / sigma
                hessian[:, 2] = ((-2.0 * u * e / s) / s) / sigma
                hessian[:, 3] = ((1.0 - 3.0 * u2) / sigma) / sigma
                hessian[:, 4] = (-2.0 * u2 * e / s) / sigma
                hessian[:, 5] = -3.0 * u2 / (s * s)
                factors = (
                    ((-1.0,), (sigma, s, sigma, s)),
                    ((-2.0, u), (sigma, sigma, s)),
                    ((-2.0, u, e), (sigma, s, s)),
                    ((1.0 - 3.0 * u2,), (sigma, sigma)),
                    ((-2.0, u2, e), (sigma, s)),
                    ((-3.0, u2), (s, s)),
                )
                for column, (numerators, denominators) in enumerate(factors):
                    hessian[:, column] = weighted_natural_channel(
                        hessian[:, column], m, numerators, denominators
                    )
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
    log_cdf_left = special.log_ndtr(-a1)
    log_cdf_right = special.log_ndtr(a2)
    log_left = np.log1p(-eps) + 0.5 * a1 * a1 + log_cdf_left
    log_right = np.log1p(eps) + 0.5 * a2 * a2 + log_cdf_right
    logk = np.logaddexp(log_left, log_right)
    p_left = np.exp(log_left - logk)
    p_right = np.exp(log_right - logk)
    # inverse Mills ratios, formed in the log domain so the far tail survives
    r1 = np.exp(-0.5 * a1 * a1 - _HALF_LOG_TWO_PI - log_cdf_left)
    r2 = np.exp(-0.5 * a2 * a2 - _HALF_LOG_TWO_PI - log_cdf_right)
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


def location_hessian_direction(
    variate: NDArray,
    location: NDArray,
    scale: NDArray,
    skew: NDArray,
    direction: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed derivative of the location-form Hessian along a natural direction ``v``.

    Within a piece (``u = (t - mu)/(sigma s)``, ``s = 1 + e eps``, ``e = +-1``)
    the third derivatives of ``-log sigma - u^2/2`` are monomials in ``u``,
    ``1/sigma`` and ``1/s``:

    * ``T_mms = 2/(sigma^3 s^2)``, ``T_mme = 2e/(sigma^2 s^3)``, ``T_mmm = 0``,
    * ``T_mss = 6u/(sigma^3 s)``, ``T_mse = 4ue/(sigma^2 s^2)``, ``T_mee = 6u/(sigma s^3)``,
    * ``T_sss = (12u^2 - 2)/sigma^3``, ``T_sse = 6u^2 e/(sigma^2 s)``,
      ``T_see = 6u^2/(sigma s^2)``, ``T_eee = 12u^2 e/s^3``.

    The Hessian jumps at ``u = 0``; there the right piece's derivative is
    returned, as the Hessian's own piece selection does.  Each entry has
    relative error ``gamma_8``; each output channel sums three of them.
    """
    t = _vector(variate, name="variate")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    m = _positive_vector(multiplier, name="multiplier")
    _same_shape(t, mu, sigma, eps, m)
    v = np.asarray(direction, dtype=_FLOAT)
    if v.shape != (len(t), 3) or not np.all(np.isfinite(v)):
        raise TwoPieceDomainError("two-piece natural direction must be a finite n-by-3 array")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u, s, e = _pieces(t, mu, sigma, eps)
        u2 = u * u
        inv_sigma, inv_s = 1.0 / sigma, 1.0 / s
        t_mms = 2.0 * inv_sigma**3 * inv_s**2
        t_mme = 2.0 * e * inv_sigma**2 * inv_s**3
        t_mss = 6.0 * u * inv_sigma**3 * inv_s
        t_mse = 4.0 * u * e * inv_sigma**2 * inv_s**2
        t_mee = 6.0 * u * inv_sigma * inv_s**3
        t_sss = (12.0 * u2 - 2.0) * inv_sigma**3
        t_sse = 6.0 * u2 * e * inv_sigma**2 * inv_s
        t_see = 6.0 * u2 * inv_sigma * inv_s**2
        t_eee = 12.0 * u2 * e * inv_s**3
        vm, vs, ve = v[:, 0], v[:, 1], v[:, 2]
        result = np.empty((len(t), 6), dtype=_FLOAT)
        result[:, 0] = t_mms * vs + t_mme * ve
        result[:, 1] = t_mms * vm + t_mss * vs + t_mse * ve
        result[:, 2] = t_mme * vm + t_mse * vs + t_mee * ve
        result[:, 3] = t_mss * vm + t_sss * vs + t_sse * ve
        result[:, 4] = t_mse * vm + t_sse * vs + t_see * ve
        result[:, 5] = t_mee * vm + t_see * vs + t_eee * ve
        result *= m[:, None]
    return result


def join_crossing_rows(
    variate: NDArray,
    location: NDArray,
    scale: NDArray,
    skew: NDArray,
    moved: NDArray,
) -> NDArray[np.bool_]:
    """Rows on the join ``z = 0`` whose Hessian has no derivative along the path.

    With nonzero skew the Hessian jumps at ``z = 0`` (``s`` changes from
    ``1 + eps`` to ``1 - eps``), and at zero skew its ``(mu, eps)`` entry
    ``-2 (t - mu) e / (sigma^2 s^3)`` is ``-2 |t - mu| / sigma^2``, a kink:
    ``T_mme = 2e/(sigma^2 s^3)`` changes sign across the join at every skew.
    So a path that moves the location there has only one-sided derivatives,
    the right piece's when ``mu`` falls and the left piece's when it rises,
    whatever the skew; a path that keeps ``mu`` fixed keeps ``z = 0`` and the
    right piece, and is differentiable.  ``moved`` is nonzero where
    the path can move ``mu``; the caller decides that, since only a location
    linear in the predictor is held fixed by a zero first-order change.  ``z``
    is formed exactly as the Hessian's piece selection forms it.
    """
    t = _vector(variate, name="variate")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    moved = np.asarray(moved, dtype=_FLOAT)
    _same_shape(t, mu, sigma, eps, moved)
    return ((t - mu) / sigma == 0.0) & (moved != 0.0)


def _log_mean_loading_third(
    sigma: NDArray[np.float64], eps: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """``log K``'s derivatives through third order in ``(sigma, eps)``.

    The same two log pieces as :func:`log_mean_loading`; with
    ``g = a -+ r`` (``r`` the inverse Mills ratio of the piece) the piece
    derivatives use ``g'' = r + (a - 2r) r (r - a)`` (left) and
    ``g'' = -r + (a + 2r) r (a + r)`` (right), from ``r' = r (r - a)`` and
    ``r' = -r (a + r)``.

    ``log K = log(e^L1 + e^L2)`` is a softplus of ``L1 - L2``, so with the
    piece weights ``w_j = e^(L_j - log K)``, ``E[.]`` their average and
    ``D = L2 - L1`` differentiated channel by channel:

    * ``K_ab = E[L_ab] + w1 w2 D_a D_b``,
    * ``K_abc = E[L_abc] + w1 w2 (D_ab D_c + D_ac D_b + D_bc D_a)
      + w1 w2 (w1 - w2) D_a D_b D_c``.

    These are the mixture cumulants formed from centred piece derivatives,
    since ``L_j,a - E[L_a]`` is ``-w2 D_a`` on the left piece and ``w1 D_a`` on
    the right.  The raw-moment identity ``R_abc - 3 S M + 2 M^3`` cancels terms
    of order ``E[L_a]^3`` when one piece dominates (16 against 0.305 at
    ``sigma = 401.93``, ``eps = 0.8703``); here ``w1 w2`` vanishes there and
    the one-piece limit is what remains.  ``w1 - w2`` is ``tanh((L1 - L2)/2)``,
    accurate when the weights are equal.

    Returns ``(K_s, K_e, K_ss, K_se, K_ee, K_sss, K_sse, K_see, K_eee)``.
    """
    left_width, right_width = 1.0 - eps, 1.0 + eps
    a1, a2 = sigma * left_width, sigma * right_width
    log_cdf_left = special.log_ndtr(-a1)
    log_cdf_right = special.log_ndtr(a2)
    log_left = np.log1p(-eps) + 0.5 * a1 * a1 + log_cdf_left
    log_right = np.log1p(eps) + 0.5 * a2 * a2 + log_cdf_right
    logk = np.logaddexp(log_left, log_right)
    weights = (np.exp(log_left - logk), np.exp(log_right - logk))
    r1 = np.exp(-0.5 * a1 * a1 - _HALF_LOG_TWO_PI - log_cdf_left)
    r2 = np.exp(-0.5 * a2 * a2 - _HALF_LOG_TWO_PI - log_cdf_right)
    g1, g1d = a1 - r1, 1.0 + a1 * r1 - r1 * r1
    g2, g2d = a2 + r2, 1.0 - a2 * r2 - r2 * r2
    g1dd = r1 + (a1 - 2.0 * r1) * r1 * (r1 - a1)
    g2dd = -r2 + (a2 + 2.0 * r2) * r2 * (a2 + r2)
    pieces = (
        (
            left_width * g1,
            -1.0 / left_width - sigma * g1,
            left_width * left_width * g1d,
            -g1 - sigma * left_width * g1d,
            -1.0 / (left_width * left_width) + sigma * sigma * g1d,
            left_width**3 * g1dd,
            -2.0 * left_width * g1d - sigma * left_width * left_width * g1dd,
            2.0 * sigma * g1d + sigma * sigma * left_width * g1dd,
            -2.0 / left_width**3 - sigma**3 * g1dd,
        ),
        (
            right_width * g2,
            1.0 / right_width + sigma * g2,
            right_width * right_width * g2d,
            g2 + sigma * right_width * g2d,
            -1.0 / (right_width * right_width) + sigma * sigma * g2d,
            right_width**3 * g2dd,
            2.0 * right_width * g2d + sigma * right_width * right_width * g2dd,
            2.0 * sigma * g2d + sigma * sigma * right_width * g2dd,
            2.0 / right_width**3 + sigma**3 * g2dd,
        ),
    )

    w1, w2 = weights
    spread = w1 * w2
    tilt = np.tanh(0.5 * (log_left - log_right))  # w1 - w2
    (ls1, le1, lss1, lse1, lee1, lsss1, lsse1, lsee1, leee1) = pieces[0]
    (ls2, le2, lss2, lse2, lee2, lsss2, lsse2, lsee2, leee2) = pieces[1]
    d_s, d_e = ls2 - ls1, le2 - le1
    d_ss, d_se, d_ee = lss2 - lss1, lse2 - lse1, lee2 - lee1
    m_s = w1 * ls1 + w2 * ls2
    m_e = w1 * le1 + w2 * le2
    k_ss = w1 * lss1 + w2 * lss2 + spread * d_s * d_s
    k_se = w1 * lse1 + w2 * lse2 + spread * d_s * d_e
    k_ee = w1 * lee1 + w2 * lee2 + spread * d_e * d_e
    spread_tilt = spread * tilt
    k_sss = w1 * lsss1 + w2 * lsss2 + spread * 3.0 * d_ss * d_s + spread_tilt * d_s**3
    k_sse = (
        w1 * lsse1
        + w2 * lsse2
        + spread * (d_ss * d_e + 2.0 * d_se * d_s)
        + spread_tilt * d_s * d_s * d_e
    )
    k_see = (
        w1 * lsee1
        + w2 * lsee2
        + spread * (d_ee * d_s + 2.0 * d_se * d_e)
        + spread_tilt * d_s * d_e * d_e
    )
    k_eee = w1 * leee1 + w2 * leee2 + spread * 3.0 * d_ee * d_e + spread_tilt * d_e**3
    return m_s, m_e, k_ss, k_se, k_ee, k_sss, k_sse, k_see, k_eee


def mean_hessian_direction(
    response: NDArray,
    mean: NDArray,
    scale: NDArray,
    skew: NDArray,
    direction: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed derivative of the mean-form Hessian in ``(m, sigma, eps)`` along ``v``.

    The location-form direction chained through ``mu = log m - log K(sigma, eps)``
    (see ``mean_form_hessian_direction``).
    """
    y = _positive_vector(response, name="response")
    m = _positive_vector(mean, name="mean")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    weight = _positive_vector(multiplier, name="multiplier")
    _same_shape(y, m, sigma, eps, weight)
    v = np.asarray(direction, dtype=_FLOAT)
    if v.shape != (len(y), 3) or not np.all(np.isfinite(v)):
        raise TwoPieceDomainError("two-piece natural direction must be a finite n-by-3 array")
    logk = log_mean_loading(sigma, eps)[0]
    k_s, k_e, k_ss, k_se, k_ee, k_sss, k_sse, k_see, k_eee = _log_mean_loading_third(sigma, eps)
    location = np.log(m) - logk
    unit = np.ones_like(weight)
    variate = np.log(y)
    base = location_rows(variate, location, sigma, eps, unit, derivative_order=2)
    assert base.score is not None and base.hessian_packed is not None
    result = mean_form_hessian_direction(
        m,
        base.score[:, 0],
        base.hessian_packed,
        lambda w: location_hessian_direction(variate, location, sigma, eps, w, unit),
        (k_s, k_e),
        (k_ss, k_se, k_ee),
        (k_sss, k_sse, k_see, k_eee),
        v,
    )
    return result * weight[:, None]


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


# ---------------------------------------------------------------- fourth order


def location_third_fourth(
    variate: NDArray,
    location: NDArray,
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Packed natural third ``(n, 10)`` and fourth ``(n, 15)`` derivatives in ``(mu, sigma, eps)``.

    Within a piece every derivative of ``-log sigma - u^2/2`` is a monomial in
    ``u``, ``1/sigma`` and ``1/s`` (generated; ``two_piece_location_third_fourth``);
    at ``u = 0`` the right piece's, as the Hessian's own piece selection.
    Each is multiplied by ``multiplier``.  A path through the join has no
    such derivative; the family refuses it before calling here.
    """
    t = _vector(variate, name="variate")
    mu = _vector(location, name="location")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    m = _positive_vector(multiplier, name="multiplier")
    _same_shape(t, mu, sigma, eps, m)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u, s, e = _pieces(t, mu, sigma, eps)
        values = stack_columns(two_piece_location_third_fourth(u, 1.0 / sigma, 1.0 / s, e), len(t))
        values *= m[:, None]
    return values[:, :10], values[:, 10:]


def _log_mean_loading_fourth(
    sigma: NDArray[np.float64], eps: NDArray[np.float64]
) -> dict[int, NDArray[np.float64]]:
    """``log K``'s derivatives of orders one to four in ``(sigma, eps)``, packed by order.

    The pieces are those of :func:`_log_mean_loading_third`, now through
    ``G4`` (each piece's derivatives generated from its inverse Mills ratio,
    ``two_piece_left_mills`` and ``two_piece_right_mills``, and chained
    through ``a = sigma s`` by ``two_piece_piece_loading``).  ``log K`` is
    ``L2 + f(L1 - L2)`` with ``f`` the softplus, whose derivatives are
    ``w1``, ``w1 w2``, ``w1 w2 (w2 - w1)`` and ``w1 w2 (1 - 6 w1 w2)``; order
    ``m`` is the weighted piece average ``w1 L1 + w2 L2`` plus the softplus
    chain rule over the partitions with at least two blocks, which carry the
    factor ``w1 w2`` and vanish when one piece dominates.
    """
    left_width, right_width = 1.0 - eps, 1.0 + eps
    a1, a2 = sigma * left_width, sigma * right_width
    log_cdf_left = special.log_ndtr(-a1)
    log_cdf_right = special.log_ndtr(a2)
    log_left = np.log1p(-eps) + 0.5 * a1 * a1 + log_cdf_left
    log_right = np.log1p(eps) + 0.5 * a2 * a2 + log_cdf_right
    logk = np.logaddexp(log_left, log_right)
    w1, w2 = np.exp(log_left - logk), np.exp(log_right - logk)
    r1 = np.exp(-0.5 * a1 * a1 - _HALF_LOG_TWO_PI - log_cdf_left)
    r2 = np.exp(-0.5 * a2 * a2 - _HALF_LOG_TWO_PI - log_cdf_right)
    n_rows = len(sigma)
    left = stack_columns(
        two_piece_piece_loading(sigma, left_width, -1.0, *two_piece_left_mills(a1, r1)), n_rows
    )
    right = stack_columns(
        two_piece_piece_loading(sigma, right_width, 1.0, *two_piece_right_mills(a2, r2)), n_rows
    )
    offsets = {1: slice(0, 2), 2: slice(2, 5), 3: slice(5, 9), 4: slice(9, 14)}
    spread = w1 * w2
    tilt = np.tanh(0.5 * (log_right - log_left))  # w2 - w1
    outer = {
        1: np.zeros((n_rows, 1)),
        2: spread[:, None],
        3: (spread * tilt)[:, None],
        4: (spread * (1.0 - 6.0 * spread))[:, None],
    }
    difference = {order: left[:, block] - right[:, block] for order, block in offsets.items()}
    inner = {
        order: {
            (0, key): difference[order][:, column]
            for column, key in enumerate(packed_tuples(2, order))
        }
        for order in offsets
    }
    loading = {}
    for order, block in offsets.items():
        average = w1[:, None] * left[:, block] + w2[:, None] * right[:, block]
        if order == 1:
            loading[order] = average
        else:
            loading[order] = average + compose_derivatives(outer, inner, order, 2, components=1)
    return loading


def _mean_inner_derivatives(
    mean: NDArray[np.float64], loading: dict[int, NDArray[np.float64]]
) -> dict[int, dict[tuple[int, tuple[int, ...]], NDArray[np.float64] | float]]:
    """``mu = log m - log K(sigma, eps)`` with ``sigma`` and ``eps`` unchanged, as ``inner``."""
    inner: dict[int, dict[tuple[int, tuple[int, ...]], NDArray[np.float64] | float]] = {
        1: {(0, (0,)): 1.0 / mean, (1, (1,)): 1.0, (2, (2,)): 1.0},
        2: {(0, (0, 0)): -1.0 / (mean * mean)},
        3: {(0, (0, 0, 0)): 2.0 / (mean * mean * mean)},
        4: {(0, (0, 0, 0, 0)): -6.0 / (mean * mean * mean * mean)},
    }
    for order, values in loading.items():
        for column, key in enumerate(packed_tuples(2, order)):
            inner[order][(0, tuple(index + 1 for index in key))] = -values[:, column]
    return inner


def mean_third_fourth(
    response: NDArray,
    mean: NDArray,
    scale: NDArray,
    skew: NDArray,
    multiplier: NDArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Packed mean-form third ``(n, 10)`` and fourth ``(n, 15)`` derivatives in ``(m, sigma, eps)``.

    The location form's derivatives composed with
    ``mu = log m - log K(sigma, eps)`` by the set-partition chain rule
    (``compose_derivatives``), the loading's own derivatives from
    :func:`_log_mean_loading_fourth`.
    """
    y = _positive_vector(response, name="response")
    m = _positive_vector(mean, name="mean")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    weight = _positive_vector(multiplier, name="multiplier")
    _same_shape(y, m, sigma, eps, weight)
    logk = log_mean_loading(sigma, eps)[0]
    loading = _log_mean_loading_fourth(sigma, eps)
    location = np.log(m) - logk
    variate = np.log(y)
    unit = np.ones_like(weight)
    base = location_rows(variate, location, sigma, eps, unit, derivative_order=2)
    assert base.score is not None and base.hessian_packed is not None
    third, fourth = location_third_fourth(variate, location, sigma, eps, unit)
    outer = {1: np.asarray(base.score), 2: np.asarray(base.hessian_packed), 3: third, 4: fourth}
    inner = _mean_inner_derivatives(m, loading)
    with np.errstate(over="ignore", invalid="ignore"):
        mean_third = compose_derivatives(outer, inner, 3, 3) * weight[:, None]
        mean_fourth = compose_derivatives(outer, inner, 4, 3) * weight[:, None]
    return mean_third, mean_fourth


#: The reach of the analytic Hessian's local model near the join, in
#: predictor units along a unit direction: the engine's derivative step
#: (``smoothing.endpoint_direction.DEFAULT_STEP``), the span its second
#: differences cover.
JOIN_REACH = 1.0e-3


def location_join_jumps(
    scale: NDArray, skew: NDArray, multiplier: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The location form's Hessian ``(n, 6)`` and third-derivative ``(n, 10)`` jumps at the join.

    Right piece minus left piece at ``u = 0``.  The Hessian jumps only in
    ``(mu, mu)``, by ``1/(sigma^2 (1 - eps)^2) - 1/(sigma^2 (1 + eps)^2)``; the
    third derivatives jump in ``(mu, mu, sigma)`` and ``(mu, mu, eps)``, the
    latter even at zero skew, where ``T_mme = 2e/(sigma^2 s^3)`` changes sign.
    The score is continuous.  Each is multiplied by ``multiplier``.
    """
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    m = _positive_vector(multiplier, name="multiplier")
    _same_shape(sigma, eps, m)
    n_rows = len(sigma)
    zero = np.zeros_like(sigma)
    inverse_sigma = 1.0 / sigma
    right = stack_columns(
        two_piece_location_third_fourth(zero, inverse_sigma, 1.0 / (1.0 + eps), 1.0), n_rows
    )
    left = stack_columns(
        two_piece_location_third_fourth(zero, inverse_sigma, 1.0 / (1.0 - eps), -1.0), n_rows
    )
    hessian = np.zeros((n_rows, 6), dtype=_FLOAT)
    hessian[:, 0] = (1.0 / ((1.0 - eps) * (1.0 - eps)) - 1.0 / ((1.0 + eps) * (1.0 + eps))) * (
        inverse_sigma * inverse_sigma
    )
    return hessian * m[:, None], (right[:, :10] - left[:, :10]) * m[:, None]


def mean_join_jumps(
    mean: NDArray, scale: NDArray, skew: NDArray, multiplier: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The mean form's Hessian and third-derivative jumps at the join, in ``(m, sigma, eps)``.

    The location form's jumps composed with ``mu = log m - log K(sigma, eps)``;
    the score is continuous, so only the jumps of order two and three enter.
    """
    m = _positive_vector(mean, name="mean")
    sigma = _positive_vector(scale, name="scale")
    eps = _skew_vector(skew)
    weight = _positive_vector(multiplier, name="multiplier")
    _same_shape(m, sigma, eps, weight)
    hessian, third = location_join_jumps(sigma, eps, weight)
    inner = _mean_inner_derivatives(m, _log_mean_loading_fourth(sigma, eps))
    outer = {1: np.zeros((len(m), 3), dtype=_FLOAT), 2: hessian, 3: third}
    return compose_derivatives(outer, inner, 2, 3), compose_derivatives(outer, inner, 3, 3)


def join_certificate(
    distance: NDArray,
    location_gradient: NDArray,
    natural_jumps: tuple[NDArray, NDArray],
    link_derivatives: tuple[NDArray, NDArray, NDArray],
    first: NDArray,
    second: NDArray,
    *,
    reach: float = JOIN_REACH,
    location_hessian: NDArray | None = None,
) -> NDArray[np.float64]:
    """The analytic second curvature direction's trust signal on rows near the join.

    The analytic fourth derivatives are exact on each piece, but the
    curvature jumps at the join (``natural_jumps``, the Hessian and third
    derivatives of :func:`location_join_jumps` or :func:`mean_join_jumps`),
    so as a local model over a displacement ``r`` they are wrong by about
    ``|jump of C| / r^2 + |jump of dC| / r`` on a row whose location the
    displacement carries across the join.  A row is near when ``|t - mu|``
    (``distance``) is at most ``reach`` times the location's rate along
    either unit direction or their normalised sum, the direction a mixed
    pair's second differences step along (``location_gradient`` is
    ``d mu / d eta``); there its certificate is that error at ``r = reach``
    times ``|a| |b|``, which is the size of what second differences over
    ``reach`` record across a kink; elsewhere it is zero.  Directions as for the second direction,
    ``(m, n, K)`` or ``(m, 1, K)``.  The jumps reach the predictors through
    the links' first three derivatives.

    ``location_hessian`` (``(n, K, K)``, ``d^2 mu / d eta^2``) adds the
    second-order displacement ``reach |u' H u| / 2`` to each rate, so a row
    the location's curvature carries across is near too.  A near row on
    which a link is close to stationary (``|g''| reach > |g'|``) is refused
    with ``ValueError``: the jumps composed through its derivatives at the
    row miss what the step meets, so the caller differences instead, as for
    a row on the join.
    """
    g1, g2, g3 = link_derivatives
    n_rows, k_parameters = g1.shape
    inner = {
        order: {(k, (k,) * order): values[:, k] for k in range(k_parameters)}
        for order, values in enumerate((g1, g2, g3), start=1)
    }
    hessian_jump, third_jump = natural_jumps
    outer = {1: np.zeros((n_rows, k_parameters), dtype=_FLOAT), 2: hessian_jump, 3: third_jump}
    value_jump = np.abs(compose_derivatives(outer, inner, 2, k_parameters))
    slope_jump = np.abs(compose_derivatives(outer, inner, 3, k_parameters))
    norm_a = np.linalg.norm(first, axis=-1)
    norm_b = np.linalg.norm(second, axis=-1)
    unit_a = first / np.where(norm_a > 0.0, norm_a, 1.0)[..., None]
    unit_b = second / np.where(norm_b > 0.0, norm_b, 1.0)[..., None]
    combined = unit_a + unit_b
    norm_c = np.linalg.norm(combined, axis=-1)
    unit_c = combined / np.where(norm_c > 0.0, norm_c, 1.0)[..., None]
    rates = []
    for unit in (unit_a, unit_b, unit_c):
        rate_u = np.abs(np.sum(location_gradient * unit, axis=-1))
        if location_hessian is not None:
            curvature = np.sum(unit * np.matmul(location_hessian, unit[..., None])[..., 0], axis=-1)
            rate_u = rate_u + 0.5 * reach * np.abs(curvature)
        rates.append(rate_u)
    rate = np.maximum.reduce(rates)
    near = (rate > 0.0) & (np.asarray(distance, dtype=_FLOAT) <= reach * rate)
    stationary = np.any(np.abs(g2) * reach > np.abs(g1), axis=-1)
    if np.any(near & stationary):
        raise ValueError("a link is near stationary on a row near the two-piece join")
    pairs = packed_tuples(k_parameters, 2)
    triples = {key: column for column, key in enumerate(packed_tuples(k_parameters, 3))}
    reach_a = np.maximum.reduce([np.abs(unit_a), np.abs(unit_b), np.abs(unit_c)])
    certificate = np.zeros((*near.shape, len(pairs)), dtype=_FLOAT)
    for column, (i, j) in enumerate(pairs):
        slope = np.zeros(near.shape, dtype=_FLOAT)
        for k in range(k_parameters):
            slope = slope + slope_jump[:, triples[tuple(sorted((i, j, k)))]] * reach_a[..., k]
        certificate[..., column] = value_jump[:, column] / (reach * reach) + slope / reach
    certificate *= (near * norm_a * norm_b)[..., None]
    return certificate


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
    "JOIN_REACH",
    "initialize_two_piece",
    "join_certificate",
    "join_crossing_rows",
    "location_expected_information",
    "location_of_mean",
    "location_join_jumps",
    "location_rows",
    "location_third_fourth",
    "log_mean_loading",
    "mean_expected_information",
    "mean_join_jumps",
    "mean_of_location",
    "mean_rows",
    "mean_third_fourth",
    "real_line_mean",
    "skew_from_sample_skewness",
    "standard_skewness",
    "standard_variance",
    "two_piece_cdf",
    "two_piece_quantile",
]
