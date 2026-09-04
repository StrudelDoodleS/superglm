"""Primitive row numerics for the Prentice (1974) generalized gamma.

Coordinates: location form ``(mu, sigma, Q)`` with ``log Y = mu + sigma W``,
``W = log(Q^2 G)/Q``, ``G ~ Gamma(Q^-2, 1)``; ``Q = 0`` is the log-normal,
``Q = 1`` the Weibull, ``Q = sigma`` the gamma.  The mean form replaces ``mu``
by ``m = E[Y] = exp(mu) C(sigma, Q)``.

Every formula is written without catastrophic cancellation: the log density is
``-log sigma - S(k) - w^2 R2(Q w)`` with ``S`` the Stirling remainder of
``lgamma`` at ``k = Q^-2`` and ``R2(u) = (expm1(u) - u)/u^2``.  The limiting
cases and cancellation-sensitive branches are checked against independent
high-precision references in the family kernel tests.

Imports only numpy, scipy, the standard library and the sibling primitive
helpers (primitive-kernel rule).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
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
from superglm.distributional.kernels.gamma import gamma_shape_plus_one_log_increment
from superglm.distributional.kernels.log_normal import log_normal_tail_log_factors

Parametrisation = Literal["mean", "location"]
_Evaluator = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_FLOAT = np.float64
_EPS = np.finfo(_FLOAT).eps
_STIRLING_SWITCH = 8.0
_U_SERIES_RADIUS = 1.0
_U_SERIES_TERMS = 25
_V_SERIES_RADIUS = 0.25
_V_SERIES_TERMS = 40
_M3_SERIES_RADIUS = 0.8
_M3_SERIES_TERMS = 48
_ZERO_SHAPE = 1.0e-8
_TAIL_INVERSE_RTOL = 64.0 * math.sqrt(_EPS)
_SHIFT_INCREMENT_RTOL = 16.0 * _EPS
_M3_A_COEFFICIENTS = tuple(1.0 / (2 * j + 3) for j in range(_M3_SERIES_TERMS))
_M3_B_COEFFICIENTS = tuple((2 * j + 2) / (2 * j + 5) for j in range(_M3_SERIES_TERMS))
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_BERNOULLI = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
    -3617.0 / 122400.0,
    43867.0 / 244188.0,
    -174611.0 / 125400.0,
)


class GeneralizedGammaDomainError(ValueError):
    """Raised when a generalized gamma row is outside the numerical domain."""


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=_FLOAT)
    if array.ndim != 1 or array.size == 0:
        raise GeneralizedGammaDomainError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise GeneralizedGammaDomainError(f"{name} must be finite")
    return array


def _positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array <= 0.0):
        raise GeneralizedGammaDomainError(f"{name} must be strictly positive")
    return array


# ---------------------------------------------------------------- Stirling remainders


def _split(x: NDArray[np.float64], switch: float) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    big = x >= switch
    return big, ~big


def stirling_remainder(x: NDArray) -> NDArray[np.float64]:
    """``lgamma(x) - (x - 1/2) log x + x - log(2 pi)/2``; series for ``x >= 8``."""
    x = np.asarray(x, dtype=_FLOAT)
    out = np.empty_like(x)
    big, small = _split(x, _STIRLING_SWITCH)
    if np.any(big):
        xb = x[big]
        inv2 = 1.0 / (xb * xb)
        acc = np.zeros_like(xb)
        power = 1.0 / xb
        for coefficient in _BERNOULLI:
            acc = acc + coefficient * power
            power = power * inv2
        out[big] = acc
    if np.any(small):
        xs = x[small]
        out[small] = special.gammaln(xs) - (xs - 0.5) * np.log(xs) + xs - _HALF_LOG_TWO_PI
    return out


def stirling_remainder_d1(x: NDArray) -> NDArray[np.float64]:
    """``digamma(x) - log x + 1/(2x)``; series for ``x >= 8``."""
    x = np.asarray(x, dtype=_FLOAT)
    out = np.empty_like(x)
    big, small = _split(x, _STIRLING_SWITCH)
    if np.any(big):
        xb = x[big]
        inv2 = 1.0 / (xb * xb)
        acc = np.zeros_like(xb)
        power = inv2
        for j, coefficient in enumerate(_BERNOULLI):
            acc = acc - (2 * j + 1) * coefficient * power
            power = power * inv2
        out[big] = acc
    if np.any(small):
        xs = x[small]
        out[small] = special.digamma(xs) - np.log(xs) + 0.5 / xs
    return out


def _stirling_d2_series(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """The Bernoulli series for ``trigamma(x) - 1/x - 1/(2x^2)``, valid for ``x >= 8``."""
    inv = 1.0 / x
    inv2 = inv * inv
    acc = np.zeros_like(x)
    power = inv2 * inv
    for j, coefficient in enumerate(_BERNOULLI):
        acc = acc + (2 * j + 1) * (2 * j + 2) * coefficient * power
        power = power * inv2
    return acc


def stirling_remainder_d2(x: NDArray) -> NDArray[np.float64]:
    """``trigamma(x) - 1/x - 1/(2x^2)``; series for ``x >= 8``.

    Below the switch the remainder's own recurrence
    ``S''(x) = S''(x + 1) + 1 / (2 x^2 (x + 1)^2)`` (from
    ``trigamma(x) - trigamma(x + 1) = 1/x^2``) walks the argument onto the
    series with positive terms only, so nothing cancels; SciPy's ``polygamma``
    goes through the Hurwitz zeta and is two orders slower.
    """
    x = np.asarray(x, dtype=_FLOAT)
    out = np.empty_like(x)
    big, small = _split(x, _STIRLING_SWITCH)
    if np.any(big):
        out[big] = _stirling_d2_series(x[big])
    if np.any(small):
        xs = x[small]
        steps = np.zeros_like(xs)
        for j in range(int(_STIRLING_SWITCH)):
            lower, upper = xs + j, xs + j + 1.0
            steps = steps + 0.5 / (lower * lower * upper * upper)
        out[small] = _stirling_d2_series(xs + _STIRLING_SWITCH) + steps
    return out


def _stirling_triplet(
    k: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(S(k), S'(k), S''(k))``, evaluated once when every row shares one ``k``."""
    if k.size > 1 and k.min() == k.max():
        head = k[:1]
        return (
            np.full_like(k, stirling_remainder(head)[0]),
            np.full_like(k, stirling_remainder_d1(head)[0]),
            np.full_like(k, stirling_remainder_d2(head)[0]),
        )
    return stirling_remainder(k), stirling_remainder_d1(k), stirling_remainder_d2(k)


# ---------------------------------------------------------------- series functions


def _horner(arg: NDArray[np.float64], coefficients: tuple[float, ...]) -> NDArray[np.float64]:
    """``sum_j c_j arg^j`` by Horner's rule, highest coefficient first."""
    acc = np.full_like(arg, coefficients[-1])
    for coefficient in coefficients[-2::-1]:
        acc = acc * arg + coefficient
    return acc


def _series(coefficient: Callable[[int], float], terms: int) -> _Evaluator:
    """The power series ``sum_j c_j arg^j`` truncated after ``terms`` terms."""
    coefficients = tuple(coefficient(j) for j in range(terms))
    return lambda arg: _horner(arg, coefficients)


def _piecewise(
    arg: NDArray,
    radius: float,
    inner_evaluator: _Evaluator,
    direct: _Evaluator,
) -> NDArray[np.float64]:
    """``inner_evaluator`` inside ``|arg| < radius``, ``direct`` outside."""
    arg = np.asarray(arg, dtype=_FLOAT)
    inner = np.abs(arg) < radius
    if inner.all():
        return inner_evaluator(arg)
    if not inner.any():
        return direct(arg)
    out = np.empty_like(arg)
    out[inner] = inner_evaluator(arg[inner])
    outer = ~inner
    out[outer] = direct(arg[outer])
    return out


def series_r2(u: NDArray) -> NDArray[np.float64]:
    """``(expm1(u) - u)/u^2``, ``1/2`` at zero."""
    return _piecewise(
        u,
        _U_SERIES_RADIUS,
        _series(lambda j: 1.0 / math.factorial(j + 2), _U_SERIES_TERMS),
        lambda a: (np.expm1(a) - a) / (a * a),
    )


def series_e1(u: NDArray) -> NDArray[np.float64]:
    """``expm1(u)/u``, ``1`` at zero."""
    return _piecewise(
        u,
        _U_SERIES_RADIUS,
        _series(lambda j: 1.0 / math.factorial(j + 1), _U_SERIES_TERMS),
        lambda a: np.expm1(a) / a,
    )


def series_u2(u: NDArray) -> NDArray[np.float64]:
    """``(u e^u - expm1(u))/u^2``, ``1/2`` at zero."""
    return _piecewise(
        u,
        _U_SERIES_RADIUS,
        _series(lambda j: (j + 1) / math.factorial(j + 2), _U_SERIES_TERMS),
        lambda a: (a * np.exp(a) - np.expm1(a)) / (a * a),
    )


def series_t3(u: NDArray) -> NDArray[np.float64]:
    """``(2(expm1(u) - u) - u expm1(u))/u^3``, ``-1/6`` at zero."""

    def direct(a: NDArray[np.float64]) -> NDArray[np.float64]:
        e = np.expm1(a)
        return (2.0 * (e - a) - a * e) / (a * a * a)

    return _piecewise(
        u,
        _U_SERIES_RADIUS,
        _series(lambda j: -(j + 1) / math.factorial(j + 3), _U_SERIES_TERMS),
        direct,
    )


def series_t3_d1(u: NDArray) -> NDArray[np.float64]:
    """Derivative of ``series_t3``, ``-1/12`` at zero."""

    def direct(a: NDArray[np.float64]) -> NDArray[np.float64]:
        e = np.expm1(a)
        return (4.0 * a * e - a * a * np.exp(a) - 6.0 * e + 6.0 * a) / (a * a * a * a)

    return _piecewise(
        u,
        _U_SERIES_RADIUS,
        _series(lambda j: -(j + 1) * (j + 2) / math.factorial(j + 4), _U_SERIES_TERMS),
        direct,
    )


def series_l1(v: NDArray) -> NDArray[np.float64]:
    """``log1p(v)/v``, ``1`` at zero."""
    return _piecewise(
        v,
        _V_SERIES_RADIUS,
        _series(lambda j: (-1.0) ** j / (j + 1), _V_SERIES_TERMS),
        lambda a: np.log1p(a) / a,
    )


def series_l1_d1(v: NDArray) -> NDArray[np.float64]:
    """Derivative of ``series_l1``, ``-1/2`` at zero."""
    return _piecewise(
        v,
        _V_SERIES_RADIUS,
        _series(lambda j: (-1.0) ** (j + 1) * (j + 1) / (j + 2), _V_SERIES_TERMS),
        lambda a: (a / (1.0 + a) - np.log1p(a)) / (a * a),
    )


def series_l2(v: NDArray) -> NDArray[np.float64]:
    """``(log1p(v) - v)/v^2``, ``-1/2`` at zero."""
    return _piecewise(
        v,
        _V_SERIES_RADIUS,
        _series(lambda j: (-1.0) ** (j + 1) / (j + 2), _V_SERIES_TERMS),
        lambda a: (np.log1p(a) - a) / (a * a),
    )


def _m3_series(v: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(M3, M3')`` for ``|v| < 0.8`` from the ``atanh`` form of ``log1p``.

    With ``t = v/(2 + v)``, ``log1p(v) = 2 atanh(t)`` and the numerator
    ``v - (1 + v/2) log1p(v)`` becomes ``-(2/(1 - t)) (atanh(t) - t)``, so
    ``M3 = -(1 - t)^2 A(t)/4`` with ``A(t) = sum_j t^(2j)/(2j + 3)`` and
    ``M3' = (1 - t)^3 A/4 - (1 - t)^4 A'/8``.  The series in ``t^2 <= 0.45``
    reaches float64 in 48 terms; the direct formulas lose ``eps/v^3`` and
    ``eps/v^4`` there (measured ``3e-15`` and ``4e-14`` at ``|v| = 0.25``).
    """
    t = v / (2.0 + v)
    t2 = t * t
    a = _horner(t2, _M3_A_COEFFICIENTS)
    b = _horner(t2, _M3_B_COEFFICIENTS)  # A'(t) / t
    one = 1.0 - t
    one2 = one * one
    m3 = -0.25 * one2 * a
    m3_d1 = 0.25 * one2 * one * a - 0.125 * one2 * one2 * t * b
    return m3, m3_d1


def series_m3(v: NDArray) -> NDArray[np.float64]:
    """``(v - (1 + v/2) log1p(v))/v^3``, ``-1/12`` at zero."""
    return _piecewise(
        v,
        _M3_SERIES_RADIUS,
        lambda a: _m3_series(a)[0],
        lambda a: (a - (1.0 + 0.5 * a) * np.log1p(a)) / (a * a * a),
    )


def series_m3_d1(v: NDArray) -> NDArray[np.float64]:
    """Derivative of ``series_m3``, ``1/12`` at zero.

    Beyond the series radius it is ``(L1'(v)/2 - 3 M3(v))/v``: the derivative
    of the numerator is exactly ``v^2 L1'(v)/2``, which keeps the rounding at
    ``eps/v`` instead of the ``eps/v^4`` of the expanded quotient rule.
    """
    return _piecewise(
        v,
        _M3_SERIES_RADIUS,
        lambda a: _m3_series(a)[1],
        lambda a: (0.5 * series_l1_d1(a) - 3.0 * series_m3(a)) / a,
    )


# ---------------------------------------------------------------- location form


@dataclass(frozen=True)
class GeneralizedGammaKernelEvaluation:
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


def _shape_terms(
    shape: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(k, S(k), k^2 S'(k), k^3 S''(k))`` with the exact ``Q = 0`` limits."""
    zero = np.abs(shape) < _ZERO_SHAPE
    safe = np.where(zero, 1.0, shape)
    k = 1.0 / (safe * safe)
    s0_all, s1_all, s2_all = _stirling_triplet(k)
    s0 = np.where(zero, 0.0, s0_all)
    k2s1 = np.where(zero, -1.0 / 12.0, k * k * s1_all)
    k3s2 = np.where(zero, 1.0 / 6.0, k * k * k * s2_all)
    return k, s0, k2s1, k3s2


def _finite_or_raise(name: str, values: NDArray | None) -> None:
    if values is not None and not np.all(np.isfinite(values)):
        raise GeneralizedGammaDomainError(f"generalized gamma {name} is not representable")


def location_rows(
    response: NDArray,
    mu: NDArray,
    sigma: NDArray,
    shape: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> GeneralizedGammaKernelEvaluation:
    """Prentice log density, score and packed Hessian in ``(mu, sigma, Q)`` per row.

    Every quantity is multiplied by ``multiplier`` (the frequency weight; ones
    under unit prior weights).  The carrier ``-log y - log(2 pi)/2`` is left to
    the caller.
    """
    y = _positive_vector(response, name="response")
    mu_values = _vector(mu, name="mu")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    m = _positive_vector(multiplier, name="multiplier")
    if any(values.shape != y.shape for values in (mu_values, sigma_values, q, m)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    order = validated_derivative_order(derivative_order)

    w = (np.log(y) - mu_values) / sigma_values
    zero = np.abs(q) < _ZERO_SHAPE
    u = np.where(zero, 0.0, q * w)
    _, s0, k2s1, k3s2 = _shape_terms(q)
    score = None
    hessian = None
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(u)
        optimizing = m * (-np.log(sigma_values) - s0 - w * w * series_r2(u))
        if order >= 1:
            e1 = series_e1(u)
            t3 = series_t3(u)
            score = np.empty((len(y), 3), dtype=_FLOAT)
            score[:, 0] = m * w * e1 / sigma_values
            score[:, 1] = m * (w * w * e1 - 1.0) / sigma_values
            score[:, 2] = m * (2.0 * q * k2s1 + w**3 * t3)
            if order == 2:
                u2 = series_u2(u)
                t3d = series_t3_d1(u)
                s2 = sigma_values * sigma_values
                hessian = np.empty((len(y), 6), dtype=_FLOAT)
                hessian[:, 0] = m * (-e / s2)
                hessian[:, 1] = m * (-w * (e + e1) / s2)
                hessian[:, 2] = m * (w * w * u2 / sigma_values)
                hessian[:, 3] = m * ((1.0 - 2.0 * w * w * e1 - w * w * e) / s2)
                hessian[:, 4] = m * (w**3 * u2 / sigma_values)
                hessian[:, 5] = m * (-4.0 * k3s2 - 6.0 * k2s1 + w**4 * t3d)
    return _placeholders_where_invalid(optimizing, score, hessian, np.isfinite(e))


def _placeholders_where_invalid(
    optimizing: NDArray[np.float64],
    score: NDArray[np.float64] | None,
    hessian: NDArray[np.float64] | None,
    valid: NDArray[np.bool_],
) -> GeneralizedGammaKernelEvaluation:
    """Zero every channel of a row whose evaluation left float64; flag it invalid.

    A row where ``exp(Q w)`` overflows has a log density below ``-1e308``: the
    step is infeasible, which the solver learns from ``valid`` and answers by
    shortening the step, whereas an exception would abort the fit.
    """
    valid = valid & np.isfinite(optimizing)
    for values in (score, hessian):
        if values is not None:
            valid &= np.all(np.isfinite(values), axis=1)
    if not np.all(valid):
        invalid = ~valid
        optimizing = np.where(invalid, 0.0, optimizing)
        if score is not None:
            score[invalid] = 0.0
        if hessian is not None:
            hessian[invalid] = 0.0
    return GeneralizedGammaKernelEvaluation(optimizing, score, hessian, valid)


def location_expected_information(
    sigma: NDArray,
    shape: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed ``-E[Hessian]`` in ``(mu, sigma, Q)`` per row (channel order as the Hessian).

    Every entry is a Gamma moment of ``gamma = k e^(Q w)`` and ``log gamma``.  The
    ``I_sQ`` and ``I_QQ`` channels are written in the Stirling remainders
    ``S'(k)`` and ``S''(k)`` rather than in ``psi`` and ``psi'`` directly, because
    the textbook forms cancel their leading orders as ``Q -> 0``; below
    ``_ZERO_SHAPE`` all three ``O(Q)`` channels take their small-``Q``
    expansions, whose truncation error there is below float64 rounding.
    """
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    m = _positive_vector(multiplier, name="multiplier")
    if any(values.shape != q.shape for values in (sigma_values, m)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    zero = np.abs(q) < _ZERO_SHAPE
    safe = np.where(zero, 1.0, q)
    k = 1.0 / (safe * safe)
    s1 = np.where(zero, 0.0, stirling_remainder_d1(k))
    s2 = np.where(zero, 0.0, stirling_remainder_d2(k))
    q2 = safe * safe
    a = s1 + q2 / 2.0  # psi(k + 1) - log k
    # psi'(k + 1) + a^2 - Q^2, with its O(Q^2) terms cancelled analytically against
    # psi'(k + 1) = S''(k) + Q^2 - Q^4/2 and a = S'(k) + Q^2/2.  Formed directly it
    # cancels four leading digits per decade of Q: the difference is -Q^4/4 + O(Q^6)
    # while the terms are O(Q^2), which drove I_QQ 550% wrong and negative near 1e-8.
    cross = s2 + s1 * s1 + s1 * q2 - 0.25 * q2 * q2
    sig2 = sigma_values * sigma_values
    information = np.empty((len(q), 6), dtype=_FLOAT)
    information[:, 0] = 1.0 / sig2
    information[:, 1] = np.where(zero, q / (2.0 * sig2), a / (safe * sig2))
    information[:, 2] = np.where(zero, -1.0 / (2.0 * sigma_values), -a / (q2 * sigma_values))
    information[:, 3] = np.where(zero, 2.0 / sig2, (2.0 + cross / q2) / sig2)
    information[:, 4] = np.where(
        zero, q / (4.0 * sigma_values), -cross / (q2 * safe * sigma_values)
    )
    # 4k^3 S'' + k^2 S'' + k^2 S'^2 + k S' - 1/4: the same identity, with the 4k^2
    # and k terms of the textbook form cancelled by hand.  Series below the switch:
    # 5/12 + Q^2/12 + O(Q^4), whose next term is 1.3e-33 at |Q| = 1e-8.
    information[:, 5] = np.where(
        zero,
        5.0 / 12.0 + q * q / 12.0,
        4.0 * k**3 * s2 + k * k * s2 + k * k * s1 * s1 + k * s1 - 0.25,
    )
    information = information * m[:, None]
    _finite_or_raise("expected information", information)
    return readonly(information)


# ---------------------------------------------------------------- mean form


def mean_exists(sigma: NDArray, shape: NDArray) -> NDArray[np.bool_]:
    """``E[Y]`` is finite iff ``Q > 0`` or ``sigma |Q| < 1`` (``Q = 0`` counts)."""
    sigma_values, q = np.broadcast_arrays(
        np.asarray(sigma, dtype=_FLOAT), np.asarray(shape, dtype=_FLOAT)
    )
    exists = np.array(q >= 0.0, copy=True)
    negative = ~exists
    if np.any(negative):
        with np.errstate(over="ignore"):
            exists[negative] = sigma_values[negative] * -q[negative] < 1.0
    return exists


def log_mean_loading(sigma: NDArray, shape: NDArray) -> tuple[NDArray[np.float64], ...]:
    """``log C`` and its five derivatives in ``(sigma, Q)``; caller guarantees ``mean_exists``.

    ``C(sigma, Q) = (Q^2)^(sigma/Q) Gamma(k + sigma/Q)/Gamma(k)`` is ``E[Y]/exp(mu)``.
    Returns ``(log C, C_sigma, C_Q, C_sigma_sigma, C_sigma_Q, C_QQ)``, each with
    its exact ``Q = 0`` limit (``sigma^2/2``, ``sigma``, ``-sigma^3/6 - sigma/2``,
    ``1``, ``-sigma^2/2 - 1/2``, ``sigma^4/6 + sigma^2/2``).
    """
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    if sigma_values.shape != q.shape:
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    if not np.all(mean_exists(sigma_values, q)):
        raise GeneralizedGammaDomainError("generalized gamma mean does not exist for every row")
    zero = np.abs(q) < _ZERO_SHAPE
    safe = np.where(zero, 1.0, q)
    v = sigma_values * q
    k = 1.0 / (safe * safe)
    x = np.where(zero, 1.0, k + sigma_values / safe)
    s0k_all, s1k_all, s2k_all = _stirling_triplet(k)
    s0k = np.where(zero, 0.0, s0k_all)
    s1k = np.where(zero, 0.0, s1k_all)
    s2k = np.where(zero, 0.0, s2k_all)
    s0x = np.where(zero, 0.0, stirling_remainder(x))
    s1x = np.where(zero, 0.0, stirling_remainder_d1(x))
    s2x = np.where(zero, 0.0, stirling_remainder_d2(x))
    d_s1 = s1x - s1k
    sig2 = sigma_values * sigma_values
    one_v = 1.0 + v
    half_v = 1.0 + 0.5 * v
    q2 = safe * safe
    inv_q = np.where(zero, 0.0, 1.0 / safe)
    inv_q2 = inv_q * inv_q
    inv_q3 = inv_q2 * inv_q
    logc = sig2 * (series_l2(v) + series_l1(v)) - 0.5 * np.log1p(v) + s0x - s0k
    c_s = sigma_values * series_l1(v) - q / (2.0 * one_v) + s1x * inv_q
    c_q = (
        2.0 * sig2 * sigma_values * series_m3(v)
        - sigma_values * half_v / one_v
        - 2.0 * inv_q3 * half_v * d_s1
        + 0.5 * sigma_values
        - sigma_values * s1k * inv_q2
    )
    c_ss = 1.0 / one_v + np.where(zero, 0.0, q2 / (2.0 * one_v * one_v)) + s2x * inv_q2
    c_sq = (
        sig2 * series_l1_d1(v)
        - 1.0 / (2.0 * one_v * one_v)
        - s2x * (2.0 + v) * inv_q2 * inv_q2
        - s1x * inv_q2
    )
    c_qq = (
        2.0 * sig2 * sig2 * series_m3_d1(v)
        + sig2 / (2.0 * one_v * one_v)
        + 6.0 * half_v * d_s1 * inv_q2 * inv_q2
        - sigma_values * d_s1 * inv_q3
        + 2.0 * inv_q3 * inv_q3 * half_v * ((2.0 + v) * s2x - 2.0 * s2k)
        + 2.0 * sigma_values * s2k * inv_q3 * inv_q2
        + 2.0 * sigma_values * s1k * inv_q3
    )
    values = (logc, c_s, c_q, c_ss, c_sq, c_qq)
    for value in values:
        _finite_or_raise("mean loading", value)
    return tuple(readonly(value) for value in values)


def location_of_mean(mean: NDArray, sigma: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``mu = log m - log C(sigma, Q)``."""
    m = _positive_vector(mean, name="mean")
    logc = log_mean_loading(sigma, shape)[0]
    return readonly(np.log(m) - logc)


def mean_of_location(mu: NDArray, sigma: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``E[Y] = exp(mu) C(sigma, Q)``, ``inf`` where the mean does not exist."""
    mu_values = _vector(mu, name="mu")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    exists = mean_exists(sigma_values, q)
    result = np.full(mu_values.shape, np.inf, dtype=_FLOAT)
    if np.any(exists):
        logc = log_mean_loading(sigma_values[exists], q[exists])[0]
        with np.errstate(over="ignore"):
            result[exists] = np.exp(mu_values[exists] + logc)
    return readonly(result)


def mean_rows(
    response: NDArray,
    mean: NDArray,
    sigma: NDArray,
    shape: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> GeneralizedGammaKernelEvaluation:
    """Mean-form rows ``(m, sigma, Q)`` chained through ``mu = log m - log C``.

    Rows whose mean does not exist (``Q < 0`` and ``sigma |Q| >= 1``) get finite
    zero placeholders and ``valid = False``.
    """
    y = _positive_vector(response, name="response")
    m = _positive_vector(mean, name="mean")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    if any(values.shape != y.shape for values in (m, sigma_values, q, weight)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    order = validated_derivative_order(derivative_order)
    exists = mean_exists(sigma_values, q)
    n = len(y)
    optimizing = np.zeros(n, dtype=_FLOAT)
    score = np.zeros((n, 3), dtype=_FLOAT) if order >= 1 else None
    hessian = np.zeros((n, 6), dtype=_FLOAT) if order == 2 else None
    valid = np.array(exists, dtype=bool)
    if np.any(exists):
        logc, c_s, c_q, c_ss, c_sq, c_qq = log_mean_loading(sigma_values[exists], q[exists])
        me = m[exists]
        base = location_rows(
            y[exists],
            np.log(me) - logc,
            sigma_values[exists],
            q[exists],
            weight[exists],
            derivative_order=order,
        )
        optimizing[exists] = base.optimizing_log_likelihood
        valid[exists] = base.valid
        if score is not None and base.score is not None:
            s_mu, s_sig, s_q = base.score[:, 0], base.score[:, 1], base.score[:, 2]
            score[exists, 0] = s_mu / me
            score[exists, 1] = s_sig - s_mu * c_s
            score[exists, 2] = s_q - s_mu * c_q
            if hessian is not None and base.hessian_packed is not None:
                h = base.hessian_packed
                h_mm, h_ms, h_mq, h_ss, h_sq, h_qq = (h[:, i] for i in range(6))
                hessian[exists, 0] = (h_mm - s_mu) / (me * me)
                hessian[exists, 1] = (h_ms - h_mm * c_s) / me
                hessian[exists, 2] = (h_mq - h_mm * c_q) / me
                hessian[exists, 3] = h_ss - 2.0 * h_ms * c_s + h_mm * c_s * c_s - s_mu * c_ss
                hessian[exists, 4] = h_sq - h_ms * c_q - h_mq * c_s + h_mm * c_s * c_q - s_mu * c_sq
                hessian[exists, 5] = h_qq - 2.0 * h_mq * c_q + h_mm * c_q * c_q - s_mu * c_qq
    return _placeholders_where_invalid(optimizing, score, hessian, valid)


def mean_expected_information(
    mean: NDArray,
    sigma: NDArray,
    shape: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed mean-form expected information ``J^T I J``; caller guarantees ``mean_exists``."""
    m = _positive_vector(mean, name="mean")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    if any(values.shape != m.shape for values in (sigma_values, q, weight)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    packed = location_expected_information(sigma_values, q, weight)
    _, c_s, c_q, _, _, _ = log_mean_loading(sigma_values, q)
    i_mm, i_ms, i_mq, i_ss, i_sq, i_qq = (packed[:, i] for i in range(6))
    out = np.empty_like(packed)
    out[:, 0] = i_mm / (m * m)
    out[:, 1] = (i_ms - i_mm * c_s) / m
    out[:, 2] = (i_mq - i_mm * c_q) / m
    out[:, 3] = i_ss - 2.0 * i_ms * c_s + i_mm * c_s * c_s
    out[:, 4] = i_sq - i_ms * c_q - i_mq * c_s + i_mm * c_s * c_q
    out[:, 5] = i_qq - 2.0 * i_mq * c_q + i_mm * c_q * c_q
    _finite_or_raise("expected information", out)
    return readonly(out)


# ---------------------------------------------------------------- functionals


def _location_arguments(
    mu: NDArray, sigma: NDArray, shape: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mu_values = _vector(mu, name="mu")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    if any(values.shape != mu_values.shape for values in (sigma_values, q)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    return mu_values, sigma_values, q


def generalized_gamma_cdf(
    y: NDArray, mu: NDArray, sigma: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """``P(Y <= y)`` in location coordinates.

    ``P(k, k e^(Q w))`` for ``Q > 0``, ``Q(k, k e^(Q w))`` for ``Q < 0`` (``W``
    decreases in the gamma variate there) and ``Phi(w)`` at ``Q = 0``.
    """
    y_values = _vector(y, name="y")
    mu_values, sigma_values, q = _location_arguments(mu, sigma, shape)
    if y_values.shape != mu_values.shape:
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    out = np.zeros_like(y_values)
    interior = y_values > 0.0
    if np.any(interior):
        y_inside = y_values[interior]
        mu_inside = mu_values[interior]
        sigma_inside = sigma_values[interior]
        q_inside = q[interior]
        w = (np.log(y_inside) - mu_inside) / sigma_inside
        zero = np.abs(q_inside) < _ZERO_SHAPE
        safe = np.where(zero, 1.0, q_inside)
        k = 1.0 / (safe * safe)
        with np.errstate(over="ignore", invalid="ignore"):
            argument = k * np.exp(safe * w)
            lower = special.gammainc(k, argument)
            upper = special.gammaincc(k, argument)
        inside_cdf = np.where(safe > 0.0, lower, upper)
        out[interior] = np.where(zero, special.ndtr(w), inside_cdf)
    _finite_or_raise("cdf", out)
    return readonly(out)


def generalized_gamma_quantile(
    p: NDArray, mu: NDArray, sigma: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """Quantile function in location coordinates for ``p`` strictly inside ``(0, 1)``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise GeneralizedGammaDomainError("quantile probabilities must lie strictly inside (0, 1)")
    mu_values, sigma_values, q = _location_arguments(mu, sigma, shape)
    if probabilities.shape != mu_values.shape:
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    zero = np.abs(q) < _ZERO_SHAPE
    safe = np.where(zero, 1.0, q)
    k = 1.0 / (safe * safe)
    gamma_quantile = np.where(
        safe > 0.0,
        special.gammaincinv(k, probabilities),
        special.gammainccinv(k, probabilities),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(zero, special.ndtri(probabilities), np.log(gamma_quantile / k) / safe)
        out = np.exp(mu_values + sigma_values * w)
    _finite_or_raise("quantile", out)
    return readonly(out)


def generalized_gamma_expected_shortfall(
    p: NDArray, mu: NDArray, sigma: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """Upper conditional first moment in Prentice location coordinates.

    With ``kappa = Q^-2``, ``r = sigma / Q`` and ``a = kappa + r``, the
    truncated first moment uses the upper regularized gamma tail for ``Q > 0``
    and the lower tail for ``Q < 0``.  The latter is positive infinity when
    ``a <= 0`` because the upper tail of ``Y`` then has no first moment.
    """
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise GeneralizedGammaDomainError(
            "expected-shortfall probabilities must lie strictly inside (0, 1)"
        )
    mu_values, sigma_values, q = _location_arguments(mu, sigma, shape)
    if probabilities.shape != mu_values.shape:
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")

    result = np.full(probabilities.shape, np.inf, dtype=_FLOAT)
    exists = mean_exists(sigma_values, q)
    finite_rows = np.flatnonzero(exists)
    if finite_rows.size:
        log_loading, log_tail_ratio, log_quantile_ratio = _expected_shortfall_factors(
            probabilities[finite_rows], sigma_values[finite_rows], q[finite_rows]
        )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result[finite_rows] = np.exp(mu_values[finite_rows] + log_loading + log_tail_ratio)
            quantile = np.exp(mu_values[finite_rows] + log_loading + log_quantile_ratio)
        certified = ~np.isnan(result[finite_rows]) & (result[finite_rows] >= quantile)
        if not np.all(certified):
            rows = finite_rows[~certified].tolist()
            raise GeneralizedGammaDomainError(
                "generalized gamma expected shortfall cannot be certified in float64 "
                f"for rows {rows}"
            )
    return readonly(result)


def generalized_gamma_expected_shortfall_from_mean(
    p: NDArray, mean: NDArray, sigma: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """Upper conditional first moment using an exactly supplied mean parameter."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.any(~np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise GeneralizedGammaDomainError(
            "expected-shortfall probabilities must lie strictly inside (0, 1)"
        )
    mean_values = _positive_vector(mean, name="mean")
    sigma_values = _positive_vector(sigma, name="sigma")
    q = _vector(shape, name="shape")
    if any(values.shape != mean_values.shape for values in (probabilities, sigma_values, q)):
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    if not np.all(mean_exists(sigma_values, q)):
        raise GeneralizedGammaDomainError("generalized gamma mean does not exist for every row")

    _, log_tail_ratio, log_quantile_ratio = _expected_shortfall_factors(
        probabilities, sigma_values, q
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = mean_values * np.exp(log_tail_ratio)
        quantile = mean_values * np.exp(log_quantile_ratio)
    certified = ~np.isnan(result) & (result >= quantile)
    if not np.all(certified):
        rows = np.flatnonzero(~certified).tolist()
        raise GeneralizedGammaDomainError(
            f"generalized gamma expected shortfall cannot be certified in float64 for rows {rows}"
        )
    return readonly(result)


def _log_positive_ratio(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Log of a positive ratio, preserving both near-one and far-tail resolution."""
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    result = np.empty_like(ratio)
    close = (ratio >= 0.5) & (ratio <= 1.5)
    relative = (numerator[close] - denominator[close]) / denominator[close]
    result[close] = np.log1p(relative)
    far = ~close
    result[far] = np.log(numerator[far]) - np.log(denominator[far])
    return result


def _expected_shortfall_factors(
    probabilities: NDArray[np.float64],
    sigma: NDArray[np.float64],
    shape: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return log mean loading, tail ratio, and quantile-to-mean ratio."""
    log_loading = np.asarray(log_mean_loading(sigma, shape)[0])
    log_tail_ratio = np.empty_like(probabilities)
    log_quantile_ratio = np.empty_like(probabilities)
    log_survival = np.log1p(-probabilities)
    zero = np.abs(shape) < _ZERO_SHAPE
    if np.any(zero):
        _, zero_log_tail_ratio, zero_log_quantile_ratio = log_normal_tail_log_factors(
            probabilities[zero], sigma[zero]
        )
        log_tail_ratio[zero] = zero_log_tail_ratio
        log_quantile_ratio[zero] = zero_log_quantile_ratio

    survival_all = 1.0 - probabilities
    survival_one = (~zero) & (survival_all == 1.0)
    log_tail_ratio[survival_one] = 0.0
    boundary_rows = np.flatnonzero(survival_one)
    if boundary_rows.size:
        boundary_shape = shape[boundary_rows]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            boundary_kappa = 1.0 / (boundary_shape * boundary_shape)
        boundary_threshold = np.empty_like(boundary_kappa)
        positive_boundary = boundary_shape > 0.0
        if np.any(positive_boundary):
            boundary_threshold[positive_boundary] = special.gammaincinv(
                boundary_kappa[positive_boundary], probabilities[boundary_rows[positive_boundary]]
            )
        negative_boundary = ~positive_boundary
        if np.any(negative_boundary):
            boundary_threshold[negative_boundary] = special.gammainccinv(
                boundary_kappa[negative_boundary], probabilities[boundary_rows[negative_boundary]]
            )
        boundary_at_support = (positive_boundary & (boundary_threshold == 0.0)) | (
            negative_boundary & np.isposinf(boundary_threshold)
        )
        boundary_interior = np.isfinite(boundary_threshold) & (boundary_threshold > 0.0)
        boundary_resolved = boundary_at_support | boundary_interior
        if not np.all(boundary_resolved):
            rows = boundary_rows[~boundary_resolved].tolist()
            raise GeneralizedGammaDomainError(
                "generalized gamma expected shortfall cannot be certified in float64 "
                f"for rows {rows}"
            )
        log_quantile_ratio[boundary_rows[boundary_at_support]] = -np.inf
        if np.any(boundary_interior):
            interior_rows = boundary_rows[boundary_interior]
            log_quantile_ratio[interior_rows] = (
                sigma[interior_rows]
                * _log_positive_ratio(
                    boundary_threshold[boundary_interior], boundary_kappa[boundary_interior]
                )
                / boundary_shape[boundary_interior]
                - log_loading[interior_rows]
            )

    active_rows = np.flatnonzero((~zero) & (~survival_one))
    if active_rows.size:
        active_shape = shape[active_rows]
        active_sigma = sigma[active_rows]
        survival = survival_all[active_rows]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            kappa = 1.0 / (active_shape * active_shape)
            shift = active_sigma / active_shape
            shifted_shape = kappa + shift
        threshold = np.empty_like(kappa)
        achieved_survival = np.empty_like(kappa)
        positive = active_shape > 0.0
        if np.any(positive):
            threshold[positive] = special.gammainccinv(kappa[positive], survival[positive])
            achieved_survival[positive] = special.gammaincc(kappa[positive], threshold[positive])
        negative = ~positive
        if np.any(negative):
            threshold[negative] = special.gammaincinv(kappa[negative], survival[negative])
            achieved_survival[negative] = special.gammainc(kappa[negative], threshold[negative])

        inverse_resolved = (
            np.isfinite(threshold)
            & (threshold > 0.0)
            & np.isfinite(achieved_survival)
            & (np.abs(achieved_survival - survival) <= _TAIL_INVERSE_RTOL * survival)
        )
        if not np.all(inverse_resolved):
            rows = active_rows[~inverse_resolved].tolist()
            raise GeneralizedGammaDomainError(
                "generalized gamma expected shortfall cannot be certified in float64 "
                f"for rows {rows}"
            )

        unit_recurrence = positive & (active_sigma == active_shape)
        generic = ~unit_recurrence
        with np.errstate(over="ignore", invalid="ignore"):
            represented_increment = shifted_shape - kappa
        increment_resolved = (
            np.isfinite(shifted_shape)
            & (shifted_shape > 0.0)
            & np.isfinite(shift)
            & (
                np.abs(represented_increment - shift)
                <= _SHIFT_INCREMENT_RTOL * np.maximum(1.0, np.abs(shift))
            )
        )
        increment_resolved[unit_recurrence] = True
        if not np.all(increment_resolved):
            rows = active_rows[~increment_resolved].tolist()
            raise GeneralizedGammaDomainError(
                "generalized gamma expected shortfall cannot be certified in float64 "
                f"for rows {rows}: the shifted gamma shape increment is unresolved"
            )

        active_log_tail_ratio = np.empty_like(survival)
        if np.any(unit_recurrence):
            log_increment = gamma_shape_plus_one_log_increment(
                kappa[unit_recurrence], threshold[unit_recurrence]
            )
            active_log_tail_ratio[unit_recurrence] = np.log1p(
                np.exp(log_increment - np.log(survival[unit_recurrence]))
            )

        if np.any(generic):
            generic_rows = np.flatnonzero(generic)
            moment_share = np.empty(generic_rows.size, dtype=_FLOAT)
            generic_positive = positive[generic_rows]
            if np.any(generic_positive):
                selected = generic_rows[generic_positive]
                moment_share[generic_positive] = special.gammaincc(
                    shifted_shape[selected], threshold[selected]
                )
            generic_negative = ~generic_positive
            if np.any(generic_negative):
                selected = generic_rows[generic_negative]
                moment_share[generic_negative] = special.gammainc(
                    shifted_shape[selected], threshold[selected]
                )
            moment_resolved = np.isfinite(moment_share) & (moment_share > 0.0)
            if not np.all(moment_resolved):
                rows = active_rows[generic_rows[~moment_resolved]].tolist()
                raise GeneralizedGammaDomainError(
                    "generalized gamma expected shortfall cannot be certified in float64 "
                    f"for rows {rows}"
                )
            active_log_tail_ratio[generic_rows] = (
                np.log(moment_share) - log_survival[active_rows[generic_rows]]
            )
        log_tail_ratio[active_rows] = active_log_tail_ratio

        log_threshold_ratio = _log_positive_ratio(threshold, kappa)
        log_quantile_ratio[active_rows] = (
            active_sigma * log_threshold_ratio / active_shape - log_loading[active_rows]
        )

    dominates_quantile = log_tail_ratio >= log_quantile_ratio
    if not np.all(dominates_quantile):
        rows = np.flatnonzero(~dominates_quantile).tolist()
        raise GeneralizedGammaDomainError(
            f"generalized gamma expected shortfall cannot be certified in float64 for rows {rows}"
        )
    return log_loading, log_tail_ratio, log_quantile_ratio


# ---------------------------------------------------------------- initialiser


class GeneralizedGammaInitializationWarning(UserWarning):
    """The log-moment start had to be clamped or shrunk to a supported point."""


_MAX_ABS_LOG_SKEW = 1.9


def _log_skew_of_shape(t: float) -> float:
    """``skew(log Y)`` for ``Q = t > 0``: ``psi''(k)/psi'(k)^1.5`` (negative, in (-2, 0))."""
    k = 1.0 / (t * t)
    return float(special.polygamma(2, k) / special.polygamma(1, k) ** 1.5)


def _shape_from_log_skewness(skew_z: float) -> tuple[float, bool]:
    """Invert the monotone log-skewness map; returns ``(Q, clamped)``."""
    clamped = abs(skew_z) > _MAX_ABS_LOG_SKEW
    if clamped:
        warnings.warn(
            f"sample log-skewness {skew_z:.3f} lies outside the generalized gamma range "
            f"(-2, 2); starting from the clamp {math.copysign(_MAX_ABS_LOG_SKEW, skew_z):+.1f}",
            GeneralizedGammaInitializationWarning,
            stacklevel=3,
        )
        skew_z = math.copysign(_MAX_ABS_LOG_SKEW, skew_z)
    if abs(skew_z) < 1.0e-6:
        return 0.0, clamped
    magnitude = optimize.brentq(
        lambda t: _log_skew_of_shape(t) + abs(skew_z), 1.0e-6, 50.0, xtol=1e-12
    )
    return -math.copysign(magnitude, skew_z), clamped  # Q > 0 gives negative log-skew


def _weighted_log_moments(
    y: NDArray[np.float64], weights: NDArray[np.float64]
) -> tuple[float, float, float]:
    """Weighted mean, variance and skewness of ``log y`` (weights as replication)."""
    z = np.log(y)
    total = float(np.sum(weights, dtype=_FLOAT))
    mean_z = float(np.sum(weights * z, dtype=_FLOAT) / total)
    centred = z - mean_z
    var_z = float(np.sum(weights * centred * centred, dtype=_FLOAT) / total)
    if var_z <= 0.0 or not math.isfinite(var_z):
        return mean_z, max(var_z, 1.0e-12), 0.0
    skew_z = float(np.sum(weights * centred**3, dtype=_FLOAT) / total / var_z**1.5)
    return mean_z, var_z, skew_z


def initialize_generalized_gamma(
    response: NDArray,
    mass: NDArray,
    *,
    parametrisation: Parametrisation,
    scale_floor: float,
) -> NDArray[np.float64]:
    """Constant start from the weighted log moments: skewness -> Q, variance -> sigma, mean -> mu.

    ``mass`` is the replication weight per row (ones under unit prior weights).
    Warns when the sample log-skewness is clamped into the family's ``(-2, 2)``
    range or when ``Q`` had to be shrunk toward zero for the mean form to start
    at a finite mean.
    """
    y = _positive_vector(response, name="response")
    weights = _positive_vector(mass, name="mass")
    if weights.shape != y.shape:
        raise GeneralizedGammaDomainError("generalized gamma row arrays must have the same shape")
    if parametrisation not in ("mean", "location"):
        raise GeneralizedGammaDomainError(f"unsupported parametrisation: {parametrisation!r}")
    mean_z, var_z, skew_z = _weighted_log_moments(y, weights)
    q, warned = _shape_from_log_skewness(skew_z)

    def sigma_mu(shape: float) -> tuple[float, float]:
        if abs(shape) < _ZERO_SHAPE:
            return math.sqrt(var_z), mean_z
        k = 1.0 / (shape * shape)
        sigma = abs(shape) * math.sqrt(var_z / float(special.polygamma(1, k)))
        expected_w = (float(stirling_remainder_d1(np.array([k]))[0]) - shape * shape / 2.0) / shape
        return sigma, mean_z - sigma * expected_w

    def floored_sigma_mu(shape: float) -> tuple[float, float]:
        """``sigma_mu`` with the scale floor already applied.

        The floor has to be inside the mean-existence loop, not after it: a
        floor above the sample scale raises ``sigma`` again, and on the
        ``Q < 0`` side that walks the start back into ``sigma |Q| >= 1``, where
        the mean is infinite.
        """
        sigma, mu = sigma_mu(shape)
        margin = max(1.0e-8, math.sqrt(np.finfo(_FLOAT).eps) * max(1.0, abs(mu)))
        return max(sigma, scale_floor + margin), mu

    sigma, mu = floored_sigma_mu(q)
    if parametrisation == "mean":
        shrunk = False
        while not bool(mean_exists(np.array([sigma]), np.array([q]))[0]):
            q *= 0.5
            sigma, mu = floored_sigma_mu(q)
            shrunk = True
        if shrunk and not warned:
            warnings.warn(
                "the log-moment start lies in the infinite-mean region; the shape was shrunk "
                "toward the log-normal so that the mean form starts at a finite mean",
                GeneralizedGammaInitializationWarning,
                stacklevel=2,
            )
        first = float(mean_of_location(np.array([mu]), np.array([sigma]), np.array([q]))[0])
    else:
        first = mu
    theta = np.column_stack((np.full(len(y), first), np.full(len(y), sigma), np.full(len(y), q)))
    return readonly(theta)


__all__ = [
    "GeneralizedGammaDomainError",
    "GeneralizedGammaInitializationWarning",
    "GeneralizedGammaKernelEvaluation",
    "generalized_gamma_cdf",
    "generalized_gamma_expected_shortfall",
    "generalized_gamma_expected_shortfall_from_mean",
    "generalized_gamma_quantile",
    "initialize_generalized_gamma",
    "location_expected_information",
    "location_of_mean",
    "location_rows",
    "log_mean_loading",
    "mean_exists",
    "mean_expected_information",
    "mean_of_location",
    "mean_rows",
    "Parametrisation",
    "series_e1",
    "series_l1",
    "series_l1_d1",
    "series_l2",
    "series_m3",
    "series_m3_d1",
    "series_r2",
    "series_t3",
    "series_t3_d1",
    "series_u2",
    "stirling_remainder",
    "stirling_remainder_d1",
    "stirling_remainder_d2",
]
