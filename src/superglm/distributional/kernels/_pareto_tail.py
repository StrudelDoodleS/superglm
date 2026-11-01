"""The generalized Pareto log survival as a two-argument engine (plan interface I2).

``G(t, xi) = log S = -log1p(xi t)/xi`` at the scaled point ``t = y/psi``, with
the exponential ``-t`` at ``xi = 0``, and its partials in ``(t, xi)`` to order
four.  With ``z = xi t``, ``a = 1/(1 + z)`` and ``v = t a``:

- every partial with a ``t`` is closed form, from ``G_t = -a`` and
  ``da/dxi = -t a^2``: ``G_tt = xi a^2``, ``G_t xi = v a``, ``G_ttt =
  -2 xi^2 a^3``, ``G_tt xi = (1 - z) a^3``, ``G_t xi xi = -2 v^2 a``, ``G_tttt =
  6 xi^3 a^4``, ``G_ttt xi = -2 xi (2 - z) a^4``, ``G_tt xi xi = -2 v (2 - z)
  a^3`` and ``G_t xi xi xi = 6 v^3 a``, each a product of positive factors and
  one exact difference;
- the pure shape partials are ``-t^(k+1) L^(k)(z)`` for ``L(z) = log1p(z)/z``,
  whose derivatives ``L' = -D``, ``D' = E``, ``E' = F``, ``F' = F_1`` are
  ``kernels.generalized_pareto``'s series functions: the alternating Taylor
  series inside ``|z| < 1/2``, where their direct forms cancel completely as
  ``z -> 0`` (the removable singularity at ``xi = 0``), and the direct
  recursion below ``z = -1/2``.  For ``z >= 1/2`` they are ``N_(k+1)(u)/xi^(k+1)``
  up to sign, ``u = z/(1 + z)``, with ``N_m = (m - 1)! sum_(j >= m) u^j/j`` the
  remainder of ``-log(1 - u) = log1p(z)`` (Abramowitz and Stegun 4.1.24).
  To order two it is formed directly, as the complete-row Hessian is; the
  third and fourth orders sum the remainder while ``u < 0.8`` (200 terms
  leave under ``0.8^200``) and form it directly above, where it cancels by
  at most a factor of eight.

``upper`` must be true: the family reads every window from the survival
function.  ``t = inf`` is ``log S = -inf``, and a row outside ``t >= 0``,
``1 + z > 0`` writes NaN.  Plain ``njit``:
the generated kernels that call it hold the GIL.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SERIES_RADIUS = 0.5
_SERIES_TERMS = 70
_TAIL_SERIES_LIMIT = 0.8
_TAIL_SERIES_TERMS = 200


_SERIES_COEFFICIENTS = np.array(
    [
        [
            (1.0 if j % 2 == 0 else -1.0)
            * (-1.0) ** k
            * math.prod(range(j + 1, j + k + 1))
            / (j + k + 1.0)
            for j in range(_SERIES_TERMS)
        ]
        for k in range(5)
    ]
)
#: ``2^-56 / (j + 6)^order``: the series stops once ``|z|^j`` is below it.
_SERIES_STOP = np.array(
    [[2.0**-56 / (j + 6.0) ** order for j in range(_SERIES_TERMS)] for order in range(5)]
)


@njit(cache=True)
def _shape_factors(z, order):
    """``L(z) = log1p(z)/z`` and its derivatives to ``order``; higher entries are zero.

    Inside ``|z| < 1/2`` each is the term-by-term derivative of ``L``'s
    alternating Taylor series, ``sum_j (-1)^j z^j/(j + 1)``, summed until the
    largest coefficient times ``|z|^j`` falls below ``2^-56`` of the series'
    ``O(1)`` values (at most 70 terms, at ``|z| = 1/2``).
    """
    l0 = l1 = l2 = l3 = l4 = 0.0
    if abs(z) < _SERIES_RADIUS:
        power = 1.0
        for j in range(_SERIES_TERMS):
            l0 += _SERIES_COEFFICIENTS[0, j] * power
            if order >= 1:
                l1 += _SERIES_COEFFICIENTS[1, j] * power
            if order >= 2:
                l2 += _SERIES_COEFFICIENTS[2, j] * power
            if order >= 3:
                l3 += _SERIES_COEFFICIENTS[3, j] * power
            if order >= 4:
                l4 += _SERIES_COEFFICIENTS[4, j] * power
            power *= z
            if abs(power) < _SERIES_STOP[order, j]:
                break
        return l0, l1, l2, l3, l4
    a = 1.0 / (1.0 + z)
    l0 = math.log1p(z) / z
    d = (l0 - a) / z
    e = (a * a - 2.0 * d) / z
    f = -(2.0 * a * a * a + 3.0 * e) / z
    return l0, -d, -e, -f, -(6.0 * a**4 - 4.0 * f) / z


@njit(cache=True)
def _tail_numerators(z, order):
    """``(L, N_2, .., N_(order+1))`` at ``u = z/(1 + z)``, ``L = log1p(z)``, for ``z >= 1/2``.

    To order two the direct forms ``L - u`` and ``2L - 2u - u^2`` cancel by at
    most a factor of 25, as in the complete-row Hessian.  For the third and
    fourth orders, below ``u = 0.8``, the remainder ``sum_(j >= m) u^j/j`` is
    summed from ``j = order + 1`` until the next term is below ``2^-54`` of the
    sum, at most 200 terms, and the lower ``N`` add their leading terms.
    """
    log_support = math.log1p(z)
    u = z / (1.0 + z)
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    if order >= 3 and u < _TAIL_SERIES_LIMIT:
        first = order + 1
        power = u**first
        total = 0.0
        for k in range(first, first + _TAIL_SERIES_TERMS):
            total += power / k
            power *= u
            if power < 2.0**-54 * total * (1.0 - u):
                break
        s5 = 0.0
        if first == 5:
            s5 = total
            s4 = s5 + 0.25 * u4
        else:
            s4 = total
        s3 = s4 + u3 / 3.0
        s2 = s3 + 0.5 * u2
        return log_support, s2, 2.0 * s3, 6.0 * s4, 24.0 * s5
    return (
        log_support,
        log_support - u,
        2.0 * log_support - 2.0 * u - u2,
        6.0 * log_support - 6.0 * u - 3.0 * u2 - 2.0 * u3,
        24.0 * log_support - 24.0 * u - 12.0 * u2 - 8.0 * u3 - 6.0 * u4,
    )


@njit(cache=True)
def gpd_log_survival_into(t, xi, upper, order, out):
    """``log S`` of the generalized Pareto at scaled point ``t``, shape ``xi``, with partials."""
    z = xi * t
    count = (order + 1) * (order + 2) // 2
    if t == math.inf and xi >= 0.0:
        # S = 0 at an infinite scaled point: an absent far end, or no mass.
        out[0] = -math.inf
        for index in range(1, count):
            out[index] = 0.0
        return
    if not upper or not (t >= 0.0) or not (1.0 + z > 0.0) or not math.isfinite(z):
        for index in range(count):
            out[index] = math.nan
        return
    if z >= _SERIES_RADIUS:
        log_support, n2, n3, n4, n5 = _tail_numerators(z, order)
        pure = (
            -log_support / xi,
            n2 / xi / xi,
            -n3 / xi / xi / xi,
            n4 / xi / xi / xi / xi,
            -n5 / xi / xi / xi / xi / xi,
        )
    else:
        l0, l1, l2, l3, l4 = _shape_factors(z, order)
        t2 = t * t
        pure = (-t * l0, -t2 * l1, -t2 * t * l2, -t2 * t2 * l3, -t2 * t2 * t * l4)
    out[0] = pure[0]
    if order < 1:
        return
    a = 1.0 / (1.0 + z)
    v = t * a
    a2 = a * a
    out[1] = -a
    out[2] = pure[1]
    if order < 2:
        return
    out[3] = xi * a2
    out[4] = v * a
    out[5] = pure[2]
    if order < 3:
        return
    a3 = a2 * a
    out[6] = -2.0 * xi * xi * a3
    out[7] = (1.0 - z) * a3
    out[8] = -2.0 * v * v * a
    out[9] = pure[3]
    if order < 4:
        return
    out[10] = 6.0 * xi * xi * xi * a2 * a2
    out[11] = -2.0 * xi * (2.0 - z) * a2 * a2
    out[12] = -2.0 * v * (2.0 - z) * a3
    out[13] = 6.0 * v * v * v * a
    out[14] = pure[4]


__all__ = ["gpd_log_survival_into"]
