"""Primitive row numerics for the generalized Pareto distribution of excesses.

Coordinates: ``(psi, xi)`` = (scale, shape), response the excess ``y >= 0`` over
a threshold chosen outside the family (the splice recipe lives in
``docs/how-to/fit-a-distributional-model.md``):

    log f = -log psi - (1 + 1/xi) log1p(xi y / psi)

with the exponential as the ``xi -> 0`` limit.  Every formula is written without
catastrophic cancellation, using ``t = y/psi``, ``z = xi t``, ``s = 1 + z`` and
the identity ``z / xi = t``:

    log f    = -log psi - t L1(z) - log1p(z)
    s_psi    = (t - 1)/(psi s)
    s_xi     = t^2 D(z) - t/s
    H_psipsi = (1 - t (2 + z))/(psi^2 s^2)
    H_psixi  = -t (t - 1)/(psi s^2)
    H_xixi   = t^3 E(z) + t^2/s^2

    L1(z) = log1p(z)/z, D(z) = (log1p(z) - z/(1+z))/z^2, E(z) = (1/(1+z)^2 - 2 D(z))/z

``D`` and ``E`` lose everything in their direct form as ``z -> 0`` (``E`` is a
total loss by ``z = 1e-10``), so both switch to an alternating power series
inside ``|z| < 0.5``.  The switch and truncation are checked against
high-precision references in the family kernel tests.

Imports only numpy, scipy and the standard library (primitive-kernel rule).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import readonly, readonly_bool
from superglm.distributional.kernels._response_tails import (
    NARROW_WINDOW,
    TailChannels,
    absent_tail,
    log_window_channels,
    quadrature_window_channels,
)
from superglm.distributional.kernels._weighted import weighted_natural_channel

_FLOAT = np.float64
_Z_SERIES_RADIUS = 0.5
_Z_SERIES_TERMS = 70
_MOMENT_WALL_MARGIN = 0.02
_FALLBACK_SHAPE = 0.25


class GeneralizedParetoDomainError(ValueError):
    """Raised when a generalized Pareto row is outside the numerical domain."""


class GeneralizedParetoInitializationWarning(UserWarning):
    """The method-of-moments start was not usable inside the configured shape walls."""


def _vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=_FLOAT)
    if array.ndim != 1 or array.size == 0:
        raise GeneralizedParetoDomainError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise GeneralizedParetoDomainError(f"{name} must be finite")
    return array


def _positive_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array <= 0.0):
        raise GeneralizedParetoDomainError(f"{name} must be strictly positive")
    return array


def _nonnegative_vector(values: object, *, name: str) -> NDArray[np.float64]:
    array = _vector(values, name=name)
    if np.any(array < 0.0):
        raise GeneralizedParetoDomainError(f"{name} must be non-negative")
    return array


def _matching(reference: NDArray[np.float64], *others: NDArray[np.float64]) -> None:
    if any(other.shape != reference.shape for other in others):
        raise GeneralizedParetoDomainError("generalized Pareto row arrays must have the same shape")


# ---------------------------------------------------------------- series functions of z


def _power_series(arg: NDArray[np.float64], coefficient, terms: int) -> NDArray[np.float64]:
    acc = np.zeros_like(arg)
    power = np.ones_like(arg)
    for j in range(terms):
        acc = acc + coefficient(j) * power
        power = power * arg
    return acc


def _piecewise(
    arg: NDArray,
    coefficient,
    direct,
    *,
    radius: float = _Z_SERIES_RADIUS,
    terms: int = _Z_SERIES_TERMS,
) -> NDArray[np.float64]:
    values = np.asarray(arg, dtype=_FLOAT)
    out = np.empty_like(values)
    inner = np.abs(values) < radius
    if np.any(inner):
        out[inner] = _power_series(values[inner], coefficient, terms)
    outer = ~inner
    if np.any(outer):
        out[outer] = direct(values[outer])
    return out


def series_l1(z: NDArray) -> NDArray[np.float64]:
    """``log1p(z)/z``, ``1`` at zero."""
    return _piecewise(z, lambda j: (-1.0) ** j / (j + 1), lambda a: np.log1p(a) / a)


def series_d(z: NDArray) -> NDArray[np.float64]:
    """``(log1p(z) - z/(1 + z))/z^2``, ``1/2`` at zero."""
    return _piecewise(
        z,
        lambda j: (-1.0) ** j * (j + 1) / (j + 2),
        lambda a: (np.log1p(a) - a / (1.0 + a)) / (a * a),
    )


def series_e(z: NDArray) -> NDArray[np.float64]:
    """``(1/(1 + z)^2 - 2 D(z))/z``, ``-2/3`` at zero."""

    def direct(a):
        d = (np.log1p(a) - a / (1.0 + a)) / (a * a)
        return (1.0 / ((1.0 + a) * (1.0 + a)) - 2.0 * d) / a

    return _piecewise(z, lambda j: (-1.0) ** (j + 1) * (j + 2) * (j + 1) / (j + 3), direct)


def _series_l1_d_e(z: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Evaluate the Hessian's three series with one mask and power sequence.

    Each accumulator keeps the standalone series' coefficient and addition
    order. Only the identical powers and outer D value are shared.
    """
    values = np.asarray(z, dtype=_FLOAT)
    l1, d, e = (np.empty_like(values) for _ in range(3))
    inner = np.abs(values) < _Z_SERIES_RADIUS
    if np.any(inner):
        arg = values[inner]
        l1_acc, d_acc, e_acc = (np.zeros_like(arg) for _ in range(3))
        power = np.ones_like(arg)
        for j in range(_Z_SERIES_TERMS):
            l1_acc = l1_acc + ((-1.0) ** j / (j + 1)) * power
            d_acc = d_acc + ((-1.0) ** j * (j + 1) / (j + 2)) * power
            e_acc = e_acc + ((-1.0) ** (j + 1) * (j + 2) * (j + 1) / (j + 3)) * power
            power = power * arg
        l1[inner], d[inner], e[inner] = l1_acc, d_acc, e_acc
    outer = ~inner
    if np.any(outer):
        arg = values[outer]
        l1[outer] = np.log1p(arg) / arg
        direct_d = (np.log1p(arg) - arg / (1.0 + arg)) / (arg * arg)
        d[outer] = direct_d
        e[outer] = (1.0 / ((1.0 + arg) * (1.0 + arg)) - 2.0 * direct_d) / arg
    return l1, d, e


def series_e1(a: NDArray) -> NDArray[np.float64]:
    """``expm1(a)/a``, ``1`` at zero; used by the quantile."""
    return _piecewise(a, lambda j: 1.0 / math.factorial(j + 1), lambda b: np.expm1(b) / b)


@dataclass(frozen=True)
class GeneralizedParetoKernelEvaluation:
    """Immutable row evaluation with read-only outputs."""

    optimizing_log_likelihood: NDArray[np.float64]
    score: NDArray[np.float64] | None
    hessian_packed: NDArray[np.float64] | None
    valid: NDArray[np.bool_]


def scale_rows(
    response: NDArray,
    scale: NDArray,
    shape: NDArray,
    multiplier: NDArray,
    *,
    derivative_order: int,
) -> GeneralizedParetoKernelEvaluation:
    """Generalized Pareto log density, score and packed Hessian in ``(psi, xi)`` per row."""
    y = _nonnegative_vector(response, name="response")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(y, psi, xi, weight)
    if derivative_order not in (0, 1, 2):
        raise GeneralizedParetoDomainError("derivative_order must be an integer from zero to two")

    with np.errstate(over="ignore", invalid="ignore"):
        t = y / psi
        z = xi * t
    support = 1.0 + z
    inside = np.isfinite(support) & (support > 0.0)
    # Positive shape has no finite right endpoint. Use bounded ratios in
    # that tail instead of materializing y/psi, its square or its cube.
    positive_tail = (xi > 0.0) & (z >= _Z_SERIES_RADIUS)
    inside |= positive_tail
    n = len(y)
    optimizing = np.zeros(n, dtype=_FLOAT)
    score = np.zeros((n, 2), dtype=_FLOAT) if derivative_order >= 1 else None
    hessian = np.zeros((n, 3), dtype=_FLOAT) if derivative_order == 2 else None
    ordinary = inside & ~positive_tail
    if np.any(ordinary):
        ti, zi, si = t[ordinary], z[ordinary], support[ordinary]
        pi, wi = psi[ordinary], weight[ordinary]
        if hessian is not None:
            l1, d, e = _series_l1_d_e(zi)
        else:
            l1, d = series_l1(zi), None
        optimizing[ordinary] = wi * (-np.log(pi) - ti * l1 - np.log1p(zi))
        if score is not None:
            if d is None:
                d = series_d(zi)
            score[ordinary, 0] = wi * (ti - 1.0) / (pi * si)
            score[ordinary, 1] = wi * (ti * ti * d - ti / si)
        if hessian is not None:
            s2 = si * si
            hessian[ordinary, 0] = wi * (1.0 - ti * (2.0 + zi)) / (pi * pi * s2)
            hessian[ordinary, 1] = -wi * ti * (ti - 1.0) / (pi * s2)
            hessian[ordinary, 2] = wi * (ti**3 * e + ti * ti / s2)
    if np.any(positive_tail):
        yi, pi, xi_i, wi = (value[positive_tail] for value in (y, psi, xi, weight))
        inverse_t = pi / yi
        # L=log(1+xi*t), v=t/(1+xi*t), a=1/(1+xi*t).
        # Retain log1p for a represented argument. Only the exceptional
        # tail needs the log sum, which remains finite when t overflows.
        zi = z[positive_tail]
        log_support = np.log1p(zi)
        overflowed = ~np.isfinite(zi)
        if np.any(overflowed):
            log_support[overflowed] = (
                np.log(xi_i[overflowed])
                + np.log(yi[overflowed])
                - np.log(pi[overflowed])
                + np.log1p(inverse_t[overflowed] / xi_i[overflowed])
            )
        v = 1.0 / (xi_i + inverse_t)
        a = inverse_t * v
        optimizing[positive_tail] = wi * (-np.log(pi) - log_support - log_support / xi_i)
        if score is not None:
            score[positive_tail, 0] = wi * ((v - a) / pi)
            score[positive_tail, 1] = wi * (((log_support - xi_i * v) / xi_i) / xi_i - v)
        if hessian is not None:
            scale_numerator = a * a - v * (1.0 + a)
            hessian[positive_tail, 0] = weighted_natural_channel(
                (scale_numerator / pi) / pi, wi, (scale_numerator,), (pi, pi)
            )
            hessian[positive_tail, 1] = wi * ((-v * (v - a)) / pi)
            u = xi_i * v
            hessian[positive_tail, 2] = wi * (
                (((-2.0 * log_support + 2.0 * u + u * u) / xi_i) / xi_i) / xi_i + v * v
            )
    for name, values in (("likelihood", optimizing), ("score", score), ("Hessian", hessian)):
        if values is not None and not np.all(np.isfinite(values)):
            raise GeneralizedParetoDomainError(f"generalized Pareto {name} is not representable")
    return GeneralizedParetoKernelEvaluation(
        readonly(optimizing),
        None if score is None else readonly(score),
        None if hessian is None else readonly(hessian),
        readonly_bool(inside),
    )


def log_survival_rows(
    points: NDArray,
    scale: NDArray,
    shape: NDArray,
    *,
    derivative_order: int,
) -> TailChannels:
    """``log S(c) = -log1p(xi c/psi)/xi`` with its score and packed Hessian in ``(psi, xi)``.

    Unit mass, for finite ``c >= 0`` (``log S(0) = 0``).  It is the density's
    ``-t L1(z)`` term, so the channels are the density's less those of
    ``-log psi - log1p(z)``::

        log S    = -t L1(z)
        s_psi    = t/(psi s),           s_xi    = t^2 D(z)
        H_psipsi = -t (2 + z)/(psi^2 s^2)
        H_psixi  = -t^2/(psi s^2),      H_xixi  = t^3 E(z)

    and in the positive-shape tail, with the bounded ratios of
    :func:`scale_rows` (``L = log1p(z)``, ``v = t/s``, ``a = 1/s``, ``u = xi v``),
    ``s_psi = v/psi``, ``s_xi = (L - u)/xi^2``, ``H_psipsi = -v (1 + a)/psi^2``,
    ``H_psixi = -v^2/psi`` and ``H_xixi = (u^2 + 2u - 2L)/xi^3``.
    """
    c = _nonnegative_vector(points, name="points")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(c, psi, xi)
    with np.errstate(over="ignore", invalid="ignore"):
        t = c / psi
        z = xi * t
    positive_tail = (xi > 0.0) & (z >= _Z_SERIES_RADIUS)
    ordinary = ~positive_tail
    n = len(c)
    value = np.zeros(n, dtype=_FLOAT)
    score = np.zeros((n, 2), dtype=_FLOAT) if derivative_order >= 1 else None
    hessian = np.zeros((n, 3), dtype=_FLOAT) if derivative_order == 2 else None
    if np.any(ordinary):
        ti, zi, pi = t[ordinary], z[ordinary], psi[ordinary]
        si = 1.0 + zi
        l1, d, e = _series_l1_d_e(zi)
        value[ordinary] = -ti * l1
        if score is not None:
            score[ordinary, 0] = ti / (pi * si)
            score[ordinary, 1] = ti * ti * d
        if hessian is not None:
            s2 = si * si
            hessian[ordinary, 0] = -ti * (2.0 + zi) / (pi * pi * s2)
            hessian[ordinary, 1] = -ti * ti / (pi * s2)
            hessian[ordinary, 2] = ti**3 * e
    if np.any(positive_tail):
        ci, pi, xi_i = c[positive_tail], psi[positive_tail], xi[positive_tail]
        inverse_t = pi / ci
        zi = z[positive_tail]
        log_support = np.log1p(zi)
        overflowed = ~np.isfinite(zi)
        if np.any(overflowed):
            log_support[overflowed] = (
                np.log(xi_i[overflowed])
                + np.log(ci[overflowed])
                - np.log(pi[overflowed])
                + np.log1p(inverse_t[overflowed] / xi_i[overflowed])
            )
        v = 1.0 / (xi_i + inverse_t)
        a = inverse_t * v
        u = xi_i * v
        value[positive_tail] = -log_support / xi_i
        if score is not None:
            score[positive_tail, 0] = v / pi
            score[positive_tail, 1] = ((log_support - u) / xi_i) / xi_i
        if hessian is not None:
            hessian[positive_tail, 0] = (-v * (1.0 + a) / pi) / pi
            hessian[positive_tail, 1] = -v * v / pi
            hessian[positive_tail, 2] = (
                ((u * u + 2.0 * u - 2.0 * log_support) / xi_i) / xi_i
            ) / xi_i
    return value, score, hessian


def window_rows(
    lower: NDArray,
    upper: NDArray,
    scale: NDArray,
    shape: NDArray,
    *,
    derivative_order: int,
) -> TailChannels:
    """``log P(lower < Y <= upper) = log(S(lower) - S(upper))`` per row, unit mass.

    ``lower`` at or below zero means the support's start (``S = 1``) and
    ``upper = +inf`` an open window.  Both ends come from the survival
    function, which no window on ``y >= 0`` rounds to one at its far end.
    The log density varies on the scale ``(psi + xi y)/(1 + |xi|)``; a window
    at most :data:`NARROW_WINDOW` of that scale wide at its steeper end is
    integrated by quadrature instead.
    """
    lo = np.maximum(np.asarray(lower, dtype=_FLOAT), 0.0)
    hi = np.asarray(upper, dtype=_FLOAT)
    psi = np.asarray(scale, dtype=_FLOAT)
    xi = np.asarray(shape, dtype=_FLOAT)
    near = log_survival_rows(lo, psi, xi, derivative_order=derivative_order)
    far = absent_tail(len(psi), 2, derivative_order=derivative_order)
    present = np.isfinite(hi)
    if np.any(present):
        channels = log_survival_rows(
            hi[present], psi[present], xi[present], derivative_order=derivative_order
        )
        for target, source in zip(far, channels, strict=True):
            if target is not None and source is not None:
                target[present] = source
    window = log_window_channels(near, far, derivative_order=derivative_order)
    span = hi - lo
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        support = psi + xi * np.where(xi < 0.0, hi, lo)
        reach = span * (1.0 + np.abs(xi)) / support
    rows = np.flatnonzero((support > 0.0) & (reach <= NARROW_WINDOW))
    if len(rows) == 0:
        return window

    def log_density(fraction: float) -> TailChannels:
        evaluation = scale_rows(
            lo[rows] + fraction * span[rows],
            psi[rows],
            xi[rows],
            np.ones(len(rows)),
            derivative_order=derivative_order,
        )
        return evaluation.optimizing_log_likelihood, evaluation.score, evaluation.hessian_packed

    quadrature = quadrature_window_channels(
        0.5 * span[rows], log_density, derivative_order=derivative_order
    )
    for target, source in zip(window, quadrature, strict=True):
        if target is not None and source is not None:
            target[rows] = source
    return window


def series_f(z: NDArray) -> NDArray[np.float64]:
    """``E'(z) = -(2/(1 + z)^3 + 3 E(z))/z``, ``3/2`` at zero (``D' = E``, ``E' = F``)."""

    def direct(a):
        return -(2.0 / ((1.0 + a) * (1.0 + a) * (1.0 + a)) + 3.0 * series_e(a)) / a

    return _piecewise(z, lambda j: (-1.0) ** j * (j + 1) * (j + 2) * (j + 3) / (j + 4), direct)


_TAIL_SERIES_LIMIT = 0.8
_TAIL_SERIES_TERMS = 200


def _tail_third_numerator(log_support: NDArray, u: NDArray) -> NDArray[np.float64]:
    """``6L - 6u - 3u^2 - 2u^3`` with ``L = log1p(z) = -log(1 - u)``, ``u = z/(1 + z)``.

    It equals ``6 sum_{k >= 4} u^k / k``, a sum of positive terms, which is
    summed below ``u = 0.8`` (200 terms leave a tail under ``0.8^200``);
    the direct form there would cancel by a factor near 100 at ``z = 1/2``.
    Above, the direct form cancels by at most a factor of five.
    """
    result = 6.0 * log_support - 6.0 * u - 3.0 * u * u - 2.0 * u**3
    inner = u < _TAIL_SERIES_LIMIT
    if np.any(inner):
        base = u[inner]
        power = base**4
        total = np.zeros_like(base)
        for k in range(4, 4 + _TAIL_SERIES_TERMS):
            total = total + power / k
            power = power * base
        result[inner] = 6.0 * total
    return result


def scale_hessian_direction(
    response: NDArray,
    scale: NDArray,
    shape: NDArray,
    direction: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed derivative of the ``(psi, xi)`` Hessian along a natural direction ``v``.

    With ``v = t/(1 + z)`` and ``a = 1/(1 + z)`` (both bounded on the support):

    * ``T_ppp = 2 (v (1 + a + a^2) - a^3)/psi^3``,
    * ``T_ppx = -v (2 a^2 - v (1 + 2a))/psi^2``,
    * ``T_pxx = 2 v^2 (v - a)/psi``,
    * ``T_xxx = t^4 F(z) - 2 v^3`` below ``z = 1/2`` (``F`` by its series) and
      ``(6L - 6u - 3u^2 - 2u^3)/xi^4 - 2 v^3`` in the positive tail, with
      ``L = log1p(z)`` and ``u = xi v``, the Hessian's own tail split.

    Rows outside the support (published as zero) return zero.
    """
    y = _nonnegative_vector(response, name="response")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(y, psi, xi, weight)
    nu = np.asarray(direction, dtype=_FLOAT)
    if nu.shape != (len(y), 2) or not np.all(np.isfinite(nu)):
        raise GeneralizedParetoDomainError("natural direction must be a finite n-by-2 array")
    with np.errstate(over="ignore", invalid="ignore"):
        t = y / psi
        z = xi * t
    support = 1.0 + z
    positive_tail = (xi > 0.0) & (z >= _Z_SERIES_RADIUS)
    ordinary = np.isfinite(support) & (support > 0.0) & ~positive_tail
    result = np.zeros((len(y), 3), dtype=_FLOAT)
    for rows in (ordinary, positive_tail):
        if not np.any(rows):
            continue
        yi, pi, xi_i, wi, ni = y[rows], psi[rows], xi[rows], weight[rows], nu[rows]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if rows is ordinary:
                ti, zi = t[rows], z[rows]
                a = 1.0 / (1.0 + zi)
                v = ti * a
                t_xxx = ti**4 * series_f(zi) - 2.0 * v**3
            else:
                inverse_t = pi / yi
                zi = z[rows]
                log_support = np.log1p(zi)
                overflowed = ~np.isfinite(zi)
                if np.any(overflowed):
                    log_support[overflowed] = (
                        np.log(xi_i[overflowed])
                        + np.log(yi[overflowed])
                        - np.log(pi[overflowed])
                        + np.log1p(inverse_t[overflowed] / xi_i[overflowed])
                    )
                v = 1.0 / (xi_i + inverse_t)
                a = inverse_t * v
                u = xi_i * v
                t_xxx = _tail_third_numerator(log_support, u) / xi_i**4 - 2.0 * v**3
            t_ppp = 2.0 * (v * (1.0 + a + a * a) - a**3) / pi**3
            t_ppx = -v * (2.0 * a * a - v * (1.0 + 2.0 * a)) / (pi * pi)
            t_pxx = 2.0 * v * v * (v - a) / pi
            vp, vx = ni[:, 0], ni[:, 1]
            result[rows, 0] = wi * (t_ppp * vp + t_ppx * vx)
            result[rows, 1] = wi * (t_ppx * vp + t_pxx * vx)
            result[rows, 2] = wi * (t_pxx * vp + t_xxx * vx)
    if not np.all(np.isfinite(result)):
        raise GeneralizedParetoDomainError(
            "generalized Pareto Hessian direction is not representable"
        )
    return result


def series_g(z: NDArray) -> NDArray[np.float64]:
    """``F'(z) = (6/(1 + z)^4 - 4 F(z))/z``, ``-24/5`` at zero.

    The pure-shape fourth derivative is ``t^5 F'(z) + 6 v^4`` below the tail
    split, as the third is ``t^4 F(z) - 2 v^3``; ``F'`` takes the series of
    ``F`` differentiated term by term inside ``|z| < 1/2``.
    """

    def direct(a):
        return (6.0 / ((1.0 + a) ** 4) - 4.0 * series_f(a)) / a

    return _piecewise(
        z, lambda j: (-1.0) ** (j + 1) * (j + 1) * (j + 2) * (j + 3) * (j + 4) / (j + 5), direct
    )


def _tail_fourth_numerator(log_support: NDArray, u: NDArray) -> NDArray[np.float64]:
    """``24 L - 24 u - 12 u^2 - 8 u^3 - 6 u^4 = 24 sum_{k >= 5} u^k / k``.

    The tail's pure-shape fourth derivative is ``-N5/xi^5 + 6 v^4``, the
    derivative of ``N4/xi^4 - 2 v^3`` (``dN4/dxi = 6 u^4/xi``).  Summed below
    ``u = 0.8`` like ``N4``, where the direct form would cancel.
    """
    result = 24.0 * log_support - 24.0 * u - 12.0 * u * u - 8.0 * u**3 - 6.0 * u**4
    inner = u < _TAIL_SERIES_LIMIT
    if np.any(inner):
        base = u[inner]
        power = base**5
        total = np.zeros_like(base)
        for k in range(5, 5 + _TAIL_SERIES_TERMS):
            total = total + power / k
            power = power * base
        result[inner] = 24.0 * total
    return result


def scale_fourth(
    response: NDArray,
    scale: NDArray,
    shape: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed natural fourth derivatives ``(pppp, pppx, ppxx, pxxx, xxxx)`` in ``(psi, xi)``.

    The four with a ``psi`` are polynomials in ``v`` and ``a`` over powers of
    ``psi`` (generated, ``generalized_pareto_fourth_closed``); ``T_xxxx`` is
    ``t^5 F'(z) + 6 v^4`` below ``z = 1/2`` and ``-N5/xi^5 + 6 v^4`` in the
    positive tail, the Hessian's own split.  Rows outside the support (published
    as zero) return zero.
    """
    from superglm.distributional.kernels._curvature_direction import stack_columns
    from superglm.distributional.kernels._fourth_generated import (
        generalized_pareto_fourth_closed,
    )

    y = _nonnegative_vector(response, name="response")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(y, psi, xi, weight)
    with np.errstate(over="ignore", invalid="ignore"):
        t = y / psi
        z = xi * t
    support = 1.0 + z
    positive_tail = (xi > 0.0) & (z >= _Z_SERIES_RADIUS)
    ordinary = np.isfinite(support) & (support > 0.0) & ~positive_tail
    result = np.zeros((len(y), 5), dtype=_FLOAT)
    for rows in (ordinary, positive_tail):
        if not np.any(rows):
            continue
        yi, pi, xi_i, wi = y[rows], psi[rows], xi[rows], weight[rows]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if rows is ordinary:
                ti, zi = t[rows], z[rows]
                a = 1.0 / (1.0 + zi)
                v = ti * a
                t_xxxx = ti**5 * series_g(zi) + 6.0 * v**4
            else:
                inverse_t = pi / yi
                zi = z[rows]
                log_support = np.log1p(zi)
                overflowed = ~np.isfinite(zi)
                if np.any(overflowed):
                    log_support[overflowed] = (
                        np.log(xi_i[overflowed])
                        + np.log(yi[overflowed])
                        - np.log(pi[overflowed])
                        + np.log1p(inverse_t[overflowed] / xi_i[overflowed])
                    )
                v = 1.0 / (xi_i + inverse_t)
                a = inverse_t * v
                u = xi_i * v
                t_xxxx = -_tail_fourth_numerator(log_support, u) / xi_i**5 + 6.0 * v**4
            closed = stack_columns(generalized_pareto_fourth_closed(v, a, pi), len(yi))
            result[rows, :4] = closed * wi[:, None]
            result[rows, 4] = wi * t_xxxx
    if not np.all(np.isfinite(result)):
        raise GeneralizedParetoDomainError(
            "generalized Pareto fourth derivatives are not representable"
        )
    return result


def expected_information(
    scale: NDArray,
    shape: NDArray,
    multiplier: NDArray,
) -> NDArray[np.float64]:
    """Packed ``-E[Hessian]`` in ``(psi, xi)`` per row (channel order as the Hessian).

    Yoshida (arXiv:2303.02402, Appendix B) in ``(log psi, xi)``, converted by
    ``I_psi. = I_eta. / psi``.  Finite and positive definite for ``xi > -1/2``
    (Smith 1985).
    """
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    weight = _positive_vector(multiplier, name="multiplier")
    _matching(xi, psi, weight)
    if np.any(2.0 * xi + 1.0 <= 0.0):
        raise GeneralizedParetoDomainError(
            "the generalized Pareto expected information exists only for shape > -1/2"
        )
    information = np.empty((len(xi), 3), dtype=_FLOAT)
    information[:, 0] = 1.0 / (psi * psi * (2.0 * xi + 1.0))
    information[:, 1] = 1.0 / (psi * (2.0 * xi + 1.0) * (xi + 1.0))
    information[:, 2] = 2.0 / ((2.0 * xi + 1.0) * (xi + 1.0))
    information = information * weight[:, None]
    if not np.all(np.isfinite(information)):
        raise GeneralizedParetoDomainError(
            "generalized Pareto expected information is not representable"
        )
    return readonly(information)


def generalized_pareto_cdf(y: NDArray, scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``P(Y <= y)`` for an excess: zero at ``y <= 0``, one past a finite upper endpoint."""
    values = _vector(y, name="y")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(values, psi, xi)
    out = np.zeros(len(values), dtype=_FLOAT)
    interior = values > 0.0
    if np.any(interior):
        with np.errstate(over="ignore", invalid="ignore"):
            t = values[interior] / psi[interior]
            z = xi[interior] * t
        interior_cdf = np.ones(len(t), dtype=_FLOAT)
        below_upper_endpoint = np.isfinite(z) & (1.0 + z > 0.0)
        if np.any(below_upper_endpoint):
            interior_cdf[below_upper_endpoint] = -np.expm1(
                -t[below_upper_endpoint] * series_l1(z[below_upper_endpoint])
            )
        out[interior] = interior_cdf
    return readonly(np.clip(out, 0.0, 1.0))


def generalized_pareto_quantile(p: NDArray, scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """Quantile of the excess for ``p`` strictly inside ``(0, 1)``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise GeneralizedParetoDomainError("quantile probabilities must lie strictly inside (0, 1)")
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(probabilities, psi, xi)
    log_survival = np.log1p(-probabilities)
    return readonly(psi * (-log_survival) * series_e1(-xi * log_survival))


def generalized_pareto_expected_shortfall(
    p: NDArray, scale: NDArray, shape: NDArray
) -> NDArray[np.float64]:
    """``E[Y | Y > q_p] = (q_p + psi) / (1 - xi)``; infinity for ``xi >= 1``."""
    probabilities = np.asarray(p, dtype=_FLOAT)
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(probabilities, psi, xi)
    quantile = generalized_pareto_quantile(probabilities, psi, xi)
    result = np.full(psi.shape, np.inf, dtype=_FLOAT)
    finite = xi < 1.0
    if np.any(finite):
        with np.errstate(over="ignore", invalid="ignore"):
            result[finite] = (quantile[finite] + psi[finite]) / (1.0 - xi[finite])
    return readonly(result)


def generalized_pareto_mean(scale: NDArray, shape: NDArray) -> NDArray[np.float64]:
    """``E[Y] = psi/(1 - xi)``; ``inf`` for ``xi >= 1``."""
    psi = _positive_vector(scale, name="scale")
    xi = _vector(shape, name="shape")
    _matching(psi, xi)
    result = np.full(psi.shape, np.inf, dtype=_FLOAT)
    finite = xi < 1.0
    if np.any(finite):
        result[finite] = psi[finite] / (1.0 - xi[finite])
    return readonly(result)


def initialize_generalized_pareto(
    response: NDArray,
    mass: NDArray,
    *,
    shape_lower: float,
    shape_upper: float,
) -> NDArray[np.float64]:
    """Constant start from the weighted first two moments of the excesses.

    ``E[Y] = psi/(1 - xi)`` and ``Var[Y] = psi^2/((1 - xi)^2 (1 - 2 xi))`` give
    ``ybar^2 / s2 = 1 - 2 xi``, so ``xi0 = (1 - ybar^2/s2)/2`` and
    ``psi0 = ybar (1 - xi0)``.  Falls back to a quarter shape, clamped inside the
    walls, when the moments are unusable or the estimate leaves the walls.
    """
    y = _nonnegative_vector(response, name="response")
    weights = _positive_vector(mass, name="mass")
    _matching(y, weights)
    if not (
        math.isfinite(shape_lower) and math.isfinite(shape_upper) and shape_lower < shape_upper
    ):
        raise GeneralizedParetoDomainError("shape walls must be finite and strictly ordered")
    margin = _MOMENT_WALL_MARGIN * (shape_upper - shape_lower)
    fallback = min(max(_FALLBACK_SHAPE, shape_lower + margin), shape_upper - margin)
    total = float(np.sum(weights, dtype=_FLOAT))
    mean = float(np.sum(weights * y, dtype=_FLOAT) / total)
    centred = y - mean
    variance = float(np.sum(weights * centred * centred, dtype=_FLOAT) / total)
    usable = math.isfinite(mean) and mean > 0.0 and math.isfinite(variance) and variance > 0.0
    estimate = 0.5 * (1.0 - mean * mean / variance) if usable else math.nan
    if (
        not usable
        or not math.isfinite(estimate)
        or estimate <= shape_lower + margin
        or estimate >= shape_upper - margin
    ):
        warnings.warn(
            "the method-of-moments generalized Pareto start is not usable inside the shape walls "
            f"({shape_lower!r}, {shape_upper!r}); starting from shape {fallback:.4g}",
            GeneralizedParetoInitializationWarning,
            stacklevel=2,
        )
        estimate = fallback
    start_scale = mean * (1.0 - estimate) if (math.isfinite(mean) and mean > 0.0) else 1.0
    if not math.isfinite(start_scale) or start_scale <= 0.0:
        start_scale = 1.0
    theta = np.column_stack(
        (np.full(len(y), start_scale, dtype=_FLOAT), np.full(len(y), estimate, dtype=_FLOAT))
    )
    return readonly(theta)
