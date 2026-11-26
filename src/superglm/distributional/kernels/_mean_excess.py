"""Row-wise mean excess ``e(v) = E[Y - v | Y > v]`` of the distributional families.

``e(v) = int_v^inf S(t) dt / S(v)``.  Every form here is assembled from
positive terms, or from a difference whose two sides differ by a bounded
factor, so no result changes sign through rounding and none is formed from
``1 - F(v)``.  Amounts are inside the law's support; the family methods give
``E[Y] - v`` below it.

Building blocks:

- ``_normal_excess(z) = phi(z)/Q(z) - z``, the standard normal mean excess.
  At ``z <= 0`` it is a sum of positive terms; on ``(0, 3]`` it is
  ``sqrt(2/pi)/erfcx(z/sqrt 2) - z``, which loses at most a factor 11.6 at
  ``z = 3``; above, it is the continued fraction ``1/(z + 2/(z + 3/(z + ...)))``
  of the Mills ratio (DLMF 7.9.3), every term positive.
- ``_gamma_excess(a, d)``, the mean excess of a unit-scale gamma at ``x = a + d``,
  ``d > 0``: ``1 - T`` with ``T = c_1/(b_1 - c_2/(b_2 - ...))``, ``c_n = n (n - a)``,
  ``b_n = d + (2n + 1)``, the tail of Legendre's continued fraction for
  ``Gamma(a, x)`` (DLMF 8.9.2), whose leading ``x - a`` cancels exactly.  Every
  denominator exceeds ``d + n``.
- ``_gamma_deficit(a, x, d)``, the mean deficit ``E[x - G | G < x]`` of the same
  law at ``x = a - d``, ``d > -1``: ``rho_1`` of ``rho_n = n x/(d + n + rho_(n+1))``,
  the ratios of ``int_0^x (x - t)^n t^(a-1) e^-t dt``, which is the minimal
  solution of ``A_(n+1) = (x - a - n) A_n + n x A_(n-1)``.  Every term is positive.

Both gamma fractions are swept backward from a zero tail at 32 levels, then at
doubled depths until two successive values agree to ``4u``.

Imports only numpy, scipy and the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import special

Rows = NDArray[np.intp]
Vector = NDArray[np.float64]

_U = 2.0**-53
_SQRT_HALF = math.sqrt(0.5)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_PHI_0 = 1.0 / math.sqrt(2.0 * math.pi)
# Above z = 3 the Mills fraction reaches 2u in 32 + 320/z^2 levels from its fixed-point start.
_MILLS_FROM = 3.0
_MILLS_LEVELS = 68
# Ten Gauss-Legendre nodes on the unit interval.  The integrands below are analytic off the
# zeros of erfc, the nearest at -1.916 +/- 2.816i (DLMF table 7.13.2, times sqrt 2), so a panel
# of width at most one has Bernstein parameter at least 11.3 and the rule's error is near 11.3^-20.
_NODES, _WEIGHTS = (
    0.5 * (values + offset)
    for values, offset in zip(np.polynomial.legendre.leggauss(10), (1.0, 0.0), strict=True)
)
# exp and expm1 are finite below this; above it an excess is assembled in logs.
_EXP_SAFE = 700.0
# The log Mills drop comes from its closed form at or above this, by quadrature below.
_DROP_CLOSED = 0.25
# Gamma fraction sweeps: first depth, the cap on doubling, and the agreement that stops it.
_FRACTION_LEVELS = 32
_FRACTION_CAP = 1 << 16
_FRACTION_RTOL = 4.0 * _U
# x above a + 1.5 sqrt(a) + 1.5 reads the upper fraction; the direct form loses under 5x below.
_TAIL_SD = 1.5
_TAIL_OFFSET = 1.5
# |Q| below this is the log-normal, as in the generalised gamma's own kernels.
_ZERO_SHAPE = 1.0e-8
# Log-logistic: (s/v)^a at or below 2^-6 reads the right-tail series, nine terms to u.
_SERIES_LOGIT = 6.0 * math.log(2.0)
_SERIES_TERMS = 9
# Two-piece log-normal, left of the join: log(m/v) at or above this reads the closed form.
_WINDOW_CLOSED = 2.0
# The unit normal's mass beyond this is under 2e-19 of any window integral that reaches it.
_NORMAL_REACH = 9.0


def finite_amounts(v: object, n_rows: int) -> Vector:
    """``v`` broadcast to the rows; refuses a missing or infinite amount."""
    amount = np.broadcast_to(np.asarray(v, dtype=np.float64), (n_rows,))
    if not np.all(np.isfinite(amount)):
        raise ValueError("mean excess amounts must be finite")
    return amount


def on_positive_support(
    amount: Vector, mean: Callable[[Rows], Vector], inside: Callable[[Rows], Vector]
) -> NDArray[np.float64]:
    """``inside(rows)`` where ``amount > 0``; ``mean(rows) - v`` at or below zero, where ``S(v) = 1``.

    The result is read-only.
    """
    result = np.empty(amount.shape)
    above = amount > 0.0
    for rows, value in ((np.flatnonzero(above), inside), (np.flatnonzero(~above), mean)):
        if rows.size:
            result[rows] = value(rows)
    result[~above] -= amount[~above]
    result.setflags(write=False)
    return result


def _mills_fraction(z: Vector) -> Vector:
    """``rho_1`` of ``rho_n = n/(z + rho_(n+1))``, which is ``phi(z)/Q(z) - z``, for ``z > 3``."""
    top = _MILLS_LEVELS + 1.0
    rho = 2.0 * top / (np.hypot(z, 2.0 * math.sqrt(top)) + z)
    for n in range(_MILLS_LEVELS, 0, -1):
        rho = n / (z + rho)
    return rho


def _normal_excess(z: Vector) -> Vector:
    """``phi(z)/Q(z) - z``: the standard normal's mean excess at ``z``, positive.

    Relative error a few ``u``, times at most 11.6 as ``z`` rises to 3.
    """
    z = np.asarray(z, dtype=np.float64)
    result = np.empty_like(z)
    left = z <= 0.0
    middle = (z > 0.0) & (z <= _MILLS_FROM)
    right = z > _MILLS_FROM
    with np.errstate(under="ignore", over="ignore"):
        t = z[left]
        result[left] = _SQRT_2_OVER_PI * np.exp(-0.5 * t * t) / special.erfc(t * _SQRT_HALF) - t
        t = z[middle]
        result[middle] = _SQRT_2_OVER_PI / special.erfcx(t * _SQRT_HALF) - t
        result[right] = _mills_fraction(z[right])
    return result


def _log_mills_drop(z: Vector, s: Vector) -> tuple[Vector, Vector]:
    """``I = log M(z - s) - log M(z) = int_(z-s)^z (phi/Q - t) dt`` and ``log[Q(z - s)/Q(z)]``.

    ``M = Q/phi`` is the Mills ratio.  Where ``I >= 1/4`` it is the closed form in ``erfc`` and
    ``erfcx``, a sum of two non-negative terms with absolute error a few ``u``.  Below, where
    that form would subtract nearly equal logs, it is ``s int_0^1`` of the integrand at
    ``z - s t`` by one ten-point panel: ``I < 1/4`` holds only for ``s <= 1`` or ``z > 4 s``,
    where one panel is enough.  The second value is exact only where ``z < 0``.
    """
    low = z - s
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        above_low = np.where(
            low < 0.0,
            special.erfc(low * _SQRT_HALF),
            special.erfcx(np.maximum(low, 0.0) * _SQRT_HALF),
        )
        above_z = np.where(
            z < 0.0, special.erfc(z * _SQRT_HALF), special.erfcx(np.maximum(z, 0.0) * _SQRT_HALF)
        )
        log_ratio = np.log(above_low / above_z)
        base = np.where(z < 0.0, s * (0.5 * s - z), 0.5 * np.minimum(low, 0.0) ** 2)
        drop = base + log_ratio
    rows = np.flatnonzero(~(drop >= _DROP_CLOSED))
    if rows.size:
        total = np.zeros(rows.size)
        for node, weight in zip(_NODES, _WEIGHTS, strict=True):
            total += weight * _normal_excess(z[rows] - s[rows] * node)
        drop[rows] = s[rows] * total
    return drop, log_ratio


def _log_normal_excess(amount: Vector, location: Vector, z: Vector, s: Vector) -> Vector:
    """``v expm1(I)`` for a log-normal tail of log-scale ``s`` at ``z = (log v - location)/s``.

    ``E[Y; Y > v]/(v S(v)) = M(z - s)/M(z)``, so ``e(v) = v expm1(I)`` with ``I`` the log Mills
    drop.  Below the median with ``I > 1``, and wherever ``I`` passes 700, it is
    ``(E[Y; Y > v]/S(v)) (1 - e^-I)`` with the first factor assembled in logs: there
    ``log v`` and ``I`` are large and cancel, and their sum ``mu + s^2/2 + log[Q(z-s)/Q(z)]``
    is not.
    """
    drop, log_ratio = _log_mills_drop(z, s)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = amount * np.expm1(np.minimum(drop, _EXP_SAFE))
        rows = np.flatnonzero(((z < 0.0) & (drop > 1.0)) | (drop > _EXP_SAFE))
        if rows.size:
            log_tail_mean = np.where(
                z[rows] < 0.0,
                location[rows] + 0.5 * s[rows] * s[rows] + log_ratio[rows],
                np.log(amount[rows]) + drop[rows],
            )
            result[rows] = np.exp(log_tail_mean) * -np.expm1(-drop[rows])
    return result


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def normal_mean_excess(v: Vector, location: Vector, scale: Vector) -> Vector:
    """``sigma (phi(z)/Q(z) - z)``, ``z = (v - mu)/sigma``; accuracy as ``_normal_excess``."""
    return scale * _normal_excess((v - location) / scale)


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def two_piece_normal_mean_excess(
    v: Vector, location: Vector, scale: Vector, skew: Vector
) -> Vector:
    """Epsilon-skew normal: scales ``sigma (1 -+ eps)`` left and right of ``mu``.

    At or above ``mu`` it is the normal form at the right scale; the piece's mass cancels.
    Below, with ``t = (mu - v)/sigma_l`` and ``h = 1/2 - Phi(-t)``, numerator and survival are
    sums of the left window's and the right piece's positive shares::

        int_v^inf S = (1 + eps) [(mu - v)/2 + sigma_r phi(0)] + (1 - eps) sigma_l R(t)
        S(v)        = (1 + eps)/2 + (1 - eps) h

    ``R(t) = int_0^t h = t h - (phi(0) - phi(t))`` loses at most a factor 3.
    """
    left, right = scale * (1.0 - skew), scale * (1.0 + skew)
    distance = v - location
    result = np.empty_like(distance)
    above = distance >= 0.0
    result[above] = right[above] * _normal_excess(distance[above] / right[above])
    below = ~above
    t = -distance[below] / left[below]
    with np.errstate(under="ignore", over="ignore"):
        window = 0.5 * special.erf(t * _SQRT_HALF)
        ramp = t * window + _PHI_0 * np.expm1(-0.5 * t * t)
    eps = skew[below]
    numerator = (1.0 + eps) * (-0.5 * distance[below] + right[below] * _PHI_0) + (1.0 - eps) * left[
        below
    ] * ramp
    result[below] = numerator / (0.5 * (1.0 + eps) + (1.0 - eps) * window)
    return result


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def log_normal_mean_excess(v: Vector, location: Vector, scale: Vector) -> Vector:
    """``v expm1(I)``, ``I = int_(z-sigma)^z (phi/Q - t) dt``, ``z = (log v - mu)/sigma``.

    Relative error a few ``u`` in ``I`` (times ``1/I`` where the closed form applies, at most 4),
    plus the conditioning of ``z`` in ``log v - mu``.
    """
    return _log_normal_excess(v, location, (np.log(v) - location) / scale, scale)


def _window_moment(t: Vector, s: Vector) -> Vector:
    """``int_0^t expm1(s (t - u)) phi(u) du`` for ``s t < 2``: ten-point panels of width <= 1."""
    reach = np.minimum(t, _NORMAL_REACH)
    panels = np.maximum(np.ceil(reach), 1.0).astype(np.intp)
    result = np.empty_like(t)
    for count in np.unique(panels):
        rows = np.flatnonzero(panels == count)
        width = reach[rows] / count
        total = np.zeros(rows.size)
        for panel in range(count):
            for node, weight in zip(_NODES, _WEIGHTS, strict=True):
                u = width * (panel + node)
                total += weight * np.expm1(s[rows] * (t[rows] - u)) * np.exp(-0.5 * u * u)
        result[rows] = _PHI_0 * width * total
    return result


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def two_piece_log_normal_mean_excess(
    v: Vector, location: Vector, scale: Vector, skew: Vector
) -> Vector:
    """Epsilon-skew log-normal: log-scales ``sigma (1 -+ eps)`` left and right of ``m = exp(mu)``.

    At or above ``m`` it is the log-normal form at the right scale.  Below, with
    ``t = log(m/v)/sigma_l`` and ``h = 1/2 - Phi(-t)``, numerator and survival are sums of
    positive shares::

        int_v^inf S = (1 - eps) L + (1 + eps)/2 [(m - v) + e_r(m)]
        S(v)        = (1 + eps)/2 + (1 - eps) h

    ``e_r(m)`` is the right piece's log-normal excess at its median, and
    ``L = E[(Y - v); v < Y < m]`` under the left piece.  For ``log(m/v) >= 2``,
    ``L = m E(sigma_l) - v [exp(-t^2/2) E(sigma_l + t) + h]`` with ``E(x) = erfcx(x/sqrt 2)/2``,
    which never overflows and loses under a factor 2.5; nearer the join ``L`` is ``v`` times
    the quadrature of ``expm1(sigma_l (t - u)) phi(u)`` over ``[0, t]``.
    """
    left, right = scale * (1.0 - skew), scale * (1.0 + skew)
    log_ratio = np.log(v) - location
    result = np.empty_like(log_ratio)
    above = log_ratio >= 0.0
    result[above] = _log_normal_excess(
        v[above], location[above], log_ratio[above] / right[above], right[above]
    )
    rows = np.flatnonzero(~above)
    if not rows.size:
        return result
    amount, s, eps, depth = v[rows], left[rows], skew[rows], -log_ratio[rows]
    t = depth / s
    with np.errstate(over="ignore", under="ignore"):
        median = np.exp(location[rows])
        window = 0.5 * special.erf(t * _SQRT_HALF)
        closed = depth >= _WINDOW_CLOSED
        left_share = np.empty_like(t)
        left_share[closed] = median[closed] * (0.5 * special.erfcx(s[closed] * _SQRT_HALF)) - (
            amount[closed]
            * (
                np.exp(-0.5 * t[closed] ** 2)
                * (0.5 * special.erfcx((s[closed] + t[closed]) * _SQRT_HALF))
                + window[closed]
            )
        )
        left_share[~closed] = amount[~closed] * _window_moment(t[~closed], s[~closed])
        at_median = _log_normal_excess(median, location[rows], np.zeros_like(t), right[rows])
        numerator = (1.0 - eps) * left_share + 0.5 * (1.0 + eps) * (
            -median * np.expm1(-depth) + at_median
        )
    result[rows] = numerator / (0.5 * (1.0 + eps) + (1.0 - eps) * window)
    return result


def _settled(sweep: Callable[[Rows, int], Vector], n_rows: int) -> tuple[Vector, Rows]:
    """``sweep(rows, levels)`` at doubling depths until it agrees to ``4u``; value and depth used."""
    rows = np.arange(n_rows)
    levels = _FRACTION_LEVELS
    value = sweep(rows, levels)
    used = np.full(n_rows, levels, dtype=np.intp)
    while rows.size and levels < _FRACTION_CAP:
        levels *= 2
        deeper = sweep(rows, levels)
        agree = np.abs(deeper - value[rows]) <= _FRACTION_RTOL * np.abs(deeper)
        value[rows] = deeper
        used[rows] = levels
        rows = rows[~agree]
    return value, used


def _gamma_excess(shape: Vector, gap: Vector) -> tuple[Vector, Rows]:
    """Mean excess of a unit-scale gamma at ``x = shape + gap``, ``gap > 0``, and the depth used."""

    def sweep(rows: Rows, levels: int) -> Vector:
        a, d = shape[rows], gap[rows]
        tail = np.zeros(rows.size)
        with np.errstate(over="ignore", invalid="ignore"):
            for n in range(levels, 0, -1):
                tail = n * (n - a) / ((d + (2 * n + 1)) - tail)
        return 1.0 - tail

    return _settled(sweep, shape.size)


def _gamma_deficit(shape_gap: Vector, x: Vector) -> tuple[Vector, Rows]:
    """Mean deficit of a unit-scale gamma at ``x = shape - shape_gap``, ``shape_gap > -1``."""

    def sweep(rows: Rows, levels: int) -> Vector:
        d, at = shape_gap[rows], x[rows]
        rho = np.zeros(rows.size)
        for n in range(levels, 0, -1):
            rho = n * at / (d + n + rho)
        return rho

    return _settled(sweep, x.size)


def gamma_in_tail(shape: Vector, gap: Vector) -> NDArray[np.bool_]:
    """Whether ``x = shape + gap`` lies above ``shape + 1.5 sqrt(shape) + 1.5``."""
    return gap > _TAIL_SD * np.sqrt(shape) + _TAIL_OFFSET


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def gamma_mean_excess(
    v: Vector, mean: Vector, shape: Vector, log_hazard: Callable[[Rows, Vector], Vector]
) -> NDArray[np.float64]:
    """Gamma of shape ``a`` and mean ``m``: ``(m/a) e_a(x)`` at ``x = a v/m``, ``v > 0``.

    Above ``a + 1.5 sqrt(a) + 1.5``: the continued fraction for ``e_a``, relative error a
    few ``u``.  At or below: ``e_a = (a - x) + D/Q``, ``D = x^a e^-x/Gamma(a)``, with
    ``log_hazard(rows, x) = log D - log Q`` supplied for those rows; a sum of positive terms
    for ``x <= a`` and a loss under a factor 5 above.  ``x - a`` is formed as
    ``a (v - m)/m``, which keeps a small distance from the mean.
    """
    x = shape * (v / mean)
    gap = shape * ((v - mean) / mean)
    unit = np.empty_like(gap)
    tail = gamma_in_tail(shape, gap)
    unit[tail] = _gamma_excess(shape[tail], gap[tail])[0]
    rows = np.flatnonzero(~tail)
    if rows.size:
        unit[rows] = np.exp(log_hazard(rows, x[rows])) - gap[rows]
    return mean * (unit / shape)


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def generalized_gamma_mean_excess(
    v: Vector,
    location: Vector,
    scale: Vector,
    shape: Vector,
    log_mean: Vector,
    log_survival: Callable[[Rows, Vector, Vector, Vector], Vector],
) -> NDArray[np.float64]:
    """Prentice generalised gamma ``(mu, sigma, Q)``: ``log Y = mu + sigma W``, ``Q^2 G = e^(Q W)``.

    With ``G`` gamma of shape ``k = 1/Q^2`` and ``Y`` proportional to ``G^(sigma/Q)``, the law
    weighted by ``y`` is the same family at gamma shape ``k* = k + sigma/Q``, and
    ``e(v)/v = h_k(x)/h_k*(x) - 1`` at ``x = k e^(Q w)``, a ratio of gamma hazards.

    - ``|Q| < 1e-8``: the log-normal form.
    - ``Q < 0`` and ``sigma |Q| >= 1``: no mean, ``inf``.
    - ``x`` in the tail of both gamma laws (above ``k* + 1.5 sqrt(k*) + 1.5`` for ``Q > 0``;
      below ``k*`` by ``1.5 sqrt(k*)``, or below ``min(k*, 8)``, for ``Q < 0``): with ``e_a`` the
      gamma mean excess (``Q > 0``) or deficit (``Q < 0``) and ``r = sigma/|Q|``,

          e(v)/v = [r - (e_k* - e_k)] / [|x - k*| + e_k*],

      from the two continued fractions.  ``e_k* - e_k`` is under ``0.9 r``, so the numerator
      keeps its sign; its relative error is a few ``u (1 + 1/sigma)``.
    - elsewhere: ``v expm1(D)``, ``D = log E[Y] - log v + log S*(v) - log S(v)``, with
      ``log_survival(rows, mu, sigma, Q)`` the log survival at ``v[rows]`` of the law with
      those coordinates, read at the row's and at the weighted law's
      (``mu + sigma log1p(sigma Q)/Q``, ``sigma/sqrt(1 + sigma Q)``, ``Q/sqrt(1 + sigma Q)``).
      Absolute error of ``D`` a few ``u (1 + |log S|)``, so relative error that over
      ``1 - e^-D``; ``|log S|`` stays moderate here because the far tail is read above.

    ``log_mean`` is ``log E[Y]`` (unused where there is no mean).
    """
    result = np.full(v.shape, np.inf)
    lognormal = np.abs(shape) < _ZERO_SHAPE
    result[lognormal] = log_normal_mean_excess(v[lognormal], location[lognormal], scale[lognormal])
    finite = ~lognormal & ((shape > 0.0) | (scale * -shape < 1.0))
    rows = np.flatnonzero(finite)
    if not rows.size:
        return result
    amount, mu, sigma, q = v[rows], location[rows], scale[rows], shape[rows]
    k = 1.0 / (q * q)
    r = sigma / np.abs(q)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        qw = q * ((np.log(amount) - mu) / sigma)
        x = k * np.exp(qw)
        # The distance of x from the two gamma shapes, on the side the tail of Y lies.
        own = np.sign(q) * k * np.expm1(qw)
        tilted_shape = k + np.sign(q) * r
        tilted = own - r
        upper = (q > 0.0) & gamma_in_tail(tilted_shape, tilted)
        lower = (q < 0.0) & (
            (tilted > _TAIL_SD * np.sqrt(tilted_shape)) | ((tilted > 0.0) & (x < 8.0))
        )
        inner = np.empty_like(amount)
        outer = np.empty_like(amount)
        inner[upper] = _gamma_excess(k[upper], own[upper])[0]
        outer[upper] = _gamma_excess(tilted_shape[upper], tilted[upper])[0]
        inner[lower] = _gamma_deficit(own[lower], x[lower])[0]
        outer[lower] = _gamma_deficit(tilted[lower], x[lower])[0]
        tail = upper | lower
        ratio = (r[tail] - (outer[tail] - inner[tail])) / (tilted[tail] + outer[tail])
        excess = np.empty_like(amount)
        excess[tail] = amount[tail] * ratio
        body = np.flatnonzero(~tail)
        if body.size:
            sq = sigma[body] * q[body]
            root = np.sqrt(1.0 + sq)
            at = rows[body]
            log_tilt = log_survival(
                at,
                mu[body] + sigma[body] * np.log1p(sq) / q[body],
                sigma[body] / root,
                q[body] / root,
            )
            log_tail_mean = (
                log_mean[at] + log_tilt - log_survival(at, mu[body], sigma[body], q[body])
            )
            # D is positive; rounding can take it below zero only where it is under about 1e-14
            # (sigma or sigma/|Q| that small), and the excess there reads zero.
            drop = np.maximum(log_tail_mean - np.log(amount[body]), 0.0)
            excess[body] = np.where(
                drop > _EXP_SAFE,
                np.exp(log_tail_mean),
                amount[body] * np.expm1(np.minimum(drop, _EXP_SAFE)),
            )
    result[rows] = excess
    return result


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def generalized_pareto_mean_excess(v: Vector, scale: Vector, shape: Vector) -> Vector:
    """``(scale + shape v)/(1 - shape)`` for ``0 <= shape < 1``; exact to rounding."""
    return (scale + shape * v) / (1.0 - shape)


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def lomax_mean_excess(v: Vector, scale: Vector, shape: Vector) -> NDArray[np.float64]:
    """``(scale + v)/(shape - 1)``, ``inf`` for ``shape <= 1``; exact to rounding."""
    with np.errstate(divide="ignore"):
        return np.where(shape > 1.0, (scale + v) / np.maximum(shape - 1.0, 0.0), np.inf)


@np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore")
def log_logistic_mean_excess(v: Vector, scale: Vector, shape: Vector) -> NDArray[np.float64]:
    """Log-logistic (Fisk) with ``S(v) = 1/(1 + (v/s)^a)``; ``inf`` for ``a <= 1``.

    With ``l = a log(v/s)`` (the log odds of ``F``, from a centred log ratio),
    ``r = e^-l = (s/v)^a`` and ``C = E[Y]/s = (pi/a)/sin(pi/a)``:

    - ``r <= 2^-6``: ``v (1 + r) [1/(a-1) - r/(2a-1) + r^2/(3a-1) - ...]``, nested so that every
      partial difference is positive; a few ``u``.
    - ``l < -700``, where ``F`` leaves float64 but ``v/s = F^(1/a)`` need not be small:
      ``E[Y] - v = s (C - 1) + (s - v)``, two positive terms; a few ``u (1 + a/700)``.
    - otherwise ``s C I_S(1 - 1/a, 1/a)/S``, from ``int_v^inf S = (s/a) B_S(1 - 1/a, 1/a)``
      (substitute ``u = S(t)``).  At or below the median the regularised incomplete beta is
      read at ``F``, so neither argument rounds to one; a few ``u``.
    """
    result = np.full(v.shape, np.inf)
    rows = np.flatnonzero(shape > 1.0)
    if not rows.size:
        return result
    amount, s, a = v[rows], scale[rows], shape[rows]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        near = (amount >= 0.5 * s) & (amount - s <= s)
        log_ratio = np.where(near, np.log1p((amount - s) / s), np.log(amount) - np.log(s))
        logit = a * log_ratio
        # 1 - 1/a as (a - 1)/a, and sin(pi/a) as sin(pi (a - 1)/a) below a = 2: both keep the
        # distance from a = 1, where the mean diverges.
        first, second = (a - 1.0) / a, 1.0 / a
        loading = np.pi * second / np.sin(np.pi * np.where(a < 2.0, first, second))
        excess = np.empty_like(amount)
        right = logit >= _SERIES_LOGIT
        r = np.exp(-logit[right])
        nested = np.zeros_like(r)
        for j in range(_SERIES_TERMS, 0, -1):
            nested = 1.0 / (j * a[right] - 1.0) - r * nested
        excess[right] = amount[right] * (1.0 + r) * nested
        left = logit < -_EXP_SAFE
        excess[left] = s[left] * (
            (np.maximum(loading[left], 1.0) - 1.0) - np.expm1(log_ratio[left])
        )
        beta = ~(right | left)
        z, p, q = logit[beta], first[beta], second[beta]
        survival = special.expit(-z)
        cdf = special.expit(np.minimum(z, 0.0))
        below = special.betainc(q, p, cdf)
        share = np.where(
            z > 0.0,
            special.betainc(p, q, survival),
            # 1 - I_F loses one bit at most while I_F <= 1/2; past that, the complement itself.
            np.where(below <= 0.5, 1.0 - below, special.betaincc(q, p, cdf)),
        )
        excess[beta] = s[beta] * loading[beta] * share / survival
    result[rows] = excess
    return result


__all__ = [
    "finite_amounts",
    "gamma_in_tail",
    "gamma_mean_excess",
    "generalized_gamma_mean_excess",
    "generalized_pareto_mean_excess",
    "log_logistic_mean_excess",
    "log_normal_mean_excess",
    "lomax_mean_excess",
    "normal_mean_excess",
    "on_positive_support",
    "two_piece_log_normal_mean_excess",
    "two_piece_normal_mean_excess",
]
