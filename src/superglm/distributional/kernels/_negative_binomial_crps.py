"""The NB2 CRPS at zero, ``mu - E|Y - Y'| / 2 = sum_k P(Y > k)^2``, by quadrature.

Also the shortfall ``E[(y - Y)^+]`` below a small response, by its pmf sum.

Wei and Held (2014, *TEST* 23(4)) write the NB2 CRPS with the mean absolute
difference ``E|Y - Y'| / 2 = (mu / p) 2F1(theta + 1, 1/2; 2; -4(1 - p) / p^2)``
(Jordan, Krueger and Lerch 2019, *J. Stat. Softw.* 90(12), appendix A.5).  That
form subtracts two numbers of order ``mu`` to leave a score at zero of order
``mu^2`` when claims are rare, and binary64 evaluations of this ``2F1`` lose
digits for a large size or a large ``mu / theta``.  The quadrature recommended
for ``c > b > 0`` (Pearson, Olver and Porter 2017, *Numer. Algorithms* 74(3),
section 4.3) is applied instead, to an integrand that is already the score.

With ``r = mu / (mu + theta)``, ``q = 1 - r``, ``w = 4r / (1 + r)^2`` and
``rho = (q / (1 + r))^2``, the lattice Fourier identity
``|d| = (1 / 2 pi) int (1 - cos dt) / (1 - cos t) dt`` and
``v = tan(t / 2) (1 + r) / q`` take Euler's integral of that ``2F1`` to

    E|Y - Y'| / 2 = C int_0^inf (1 - R(v)^theta) / v^2 dv,
    R(v) = (1 + rho v^2) / (1 + v^2),   C = (1 + r) / (2 pi q).

Because ``C int -theta log R / v^2 dv = C theta pi (1 - sqrt(rho)) = mu``, the
score at zero is

    CRPS(0) = C int_0^inf e2(theta log R(v)) / v^2 dv,   e2(z) = e^z - 1 - z >= 0,

an integrand that is never negative, so nothing cancels.  In ``x = log v`` it
decays like ``e^{3x}`` to the left and ``e^{-x}`` to the right, with its
features at ``x = -log(1 + theta w) / 2`` (where ``theta log R`` reaches one),
``0`` and ``log((1 + r) / q)`` (where ``R`` reaches ``rho``).  A sinh-mapped
trapezoid rule (Takahasi and Mori 1974) centred between them, with its step
scaled to their spread, spaces its nodes about 0.3 apart in ``x`` across them,
well inside the integrand's strip of analyticity. It agrees with a direct
lattice sum to about ``1e-14`` relative from ``mu = 1e-8`` to ``1e3`` and from
``theta = 1e-4`` to ``1e12``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]

#: The step in the sinh variable at unit feature spread.
_STEP = 0.2
#: How far past the outer features the rule runs, in ``x = log v``: the
#: integrand decays like ``e^{3x}`` on the left and ``e^{-x}`` on the right.
_LEFT_SPAN = 14.0
_RIGHT_SPAN = 40.0
#: Below this many cells the shortfall is the pmf sum, whose terms are all positive.
SHORTFALL_CELLS = 64
#: ``1 / n!`` for ``n = 2, ..., 16``: ``e2`` on ``|z| < 1/2`` to within ``1e-17``
#: relative; past it ``expm1(z) - z`` loses at most a factor five to cancellation.
_E2_SERIES = np.array([1.0 / math.factorial(n) for n in range(16, 1, -1)])
#: A tail node past the outer features whose term is below this fraction of the
#: sum so far ends its side: the terms there fall double-exponentially.
_NEGLIGIBLE = 2.0**-60
_TWO_PI = 2.0 * math.pi


@njit(cache=True)
def _expm1_less_identity(z: float) -> float:
    """``e^z - 1 - z``, by its Taylor series where the difference would cancel."""
    if abs(z) < 0.5:
        acc = 0.0
        for coefficient in _E2_SERIES:
            acc = acc * z + coefficient
        return acc * z * z
    return math.expm1(z) - z


@njit(cache=True)
def _node_term(
    s: float, centre: float, spread: float, size: float, w: float, rho: float
) -> tuple[float, float]:
    """``(x, e2(size log R) e^{-x} dx/ds)`` at the sinh-mapped node ``s``."""
    grow = math.exp(s)
    shrink = 1.0 / grow
    x = centre + spread * 0.5 * (grow - shrink)
    inverse_v = math.exp(-x)
    decay = inverse_v * inverse_v  # 1 / v^2
    cosine = decay / (1.0 + decay)  # 1 / (1 + v^2)
    drop = w / (1.0 + decay)  # 1 - R = w v^2 / (1 + v^2)
    log_ratio = math.log1p(-drop) if drop < 0.5 else math.log(rho + w * cosine)
    jacobian = spread * 0.5 * (grow + shrink)
    return x, _expm1_less_identity(size * log_ratio) * inverse_v * jacobian


@njit(cache=True)
def _zero_score(mean: float, size: float) -> float:
    total = mean + size
    r = mean / total
    q = size / total
    w = 4.0 * r / ((1.0 + r) * (1.0 + r))
    sqrt_rho = q / (1.0 + r)
    rho = sqrt_rho * sqrt_rho
    left = -0.5 * math.log1p(size * w)
    right = max(math.log1p(r) - math.log(q), 0.0)
    centre = 0.5 * (left + right)
    spread = 0.5 * (right - left) + 1.0
    step = _STEP / spread
    first = int(math.ceil(math.asinh((spread + _LEFT_SPAN) / spread) / step))
    last = int(math.ceil(math.asinh((spread + _RIGHT_SPAN) / spread) / step))
    acc = 0.0
    for node in range(0, last + 1):
        x, term = _node_term(node * step, centre, spread, size, w, rho)
        acc += term
        if x > right + 1.0 and term <= _NEGLIGIBLE * acc:
            break
    for node in range(1, first + 1):
        x, term = _node_term(-node * step, centre, spread, size, w, rho)
        acc += term
        if x < left - 1.0 and term <= _NEGLIGIBLE * acc:
            break
    return acc * step * (1.0 + r) / (_TWO_PI * q)


@njit(cache=True)
def _shortfall(mean: float, size: float, response: float, cells: int) -> float:
    """``sum_{j < cells} (y - j) f(j)`` from ``f(0) = (size / (mean + size))^size``."""
    rate = mean / (mean + size)
    mass = math.exp(-size * math.log1p(mean / size))
    acc = response * mass
    for j in range(1, cells):
        mass *= rate * (j - 1.0 + size) / j
        acc += (response - j) * mass
    return acc


@njit(cache=True)
def negative_binomial_crps_rows(mean, size, response, zero, shortfall, start, stop):
    """Write ``CRPS(NB2(mean, size), 0)`` and ``E[(y - Y)^+]`` for rows ``start:stop``.

    The caller passes contiguous float64 rows (positive finite ``mean`` and
    ``size``, finite ``response``) and the two outputs, and runs the row blocks.
    The shortfall is NaN where the response reaches ``SHORTFALL_CELLS``, for the
    caller to take from the incomplete beta function.  Returns ``(0, -1)``.
    """
    for row in range(start, stop):
        zero[row] = _zero_score(mean[row], size[row])
        cells = math.floor(response[row]) + 1.0
        if cells <= 0.0:
            shortfall[row] = 0.0
        elif cells <= SHORTFALL_CELLS:
            shortfall[row] = _shortfall(mean[row], size[row], response[row], int(cells))
        else:
            shortfall[row] = np.nan
    return 0, -1


__all__ = ["SHORTFALL_CELLS", "negative_binomial_crps_rows"]
