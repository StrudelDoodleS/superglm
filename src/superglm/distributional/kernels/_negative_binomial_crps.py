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

Where ``CRPS(0)`` exceeds ``mu / 2`` it is of order ``mu`` and keeps only
``mu``'s digits, which a large central count's score, of order its standard
deviation, cannot spare.  The same nodes then sum ``(1 - R^theta) e^{-x}``,
whose total is ``E|Y - Y'| / 2`` without the subtraction from ``mu``, and the
midpoints between them, halving the step.  Its left tail decays like
``theta w e^x``, so that side runs ``_HALF_LEFT_SPAN`` past the feature instead
of ``_LEFT_SPAN``.  Rare claims, whose ``CRPS(0)`` is of order ``mu^2``, never
take this route.
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
#: The left span for ``E|Y - Y'| / 2``, whose integrand decays only like ``e^x``.
_HALF_LEFT_SPAN = 44.0
#: Below this many cells the shortfall is the pmf sum, whose terms are all positive.
SHORTFALL_CELLS = 64
#: ``1 / n!`` for ``n = 2, ..., 16``: ``e2`` on ``|z| < 1/2`` to within ``1e-17``
#: relative; past it ``expm1(z) - z`` loses at most a factor five to cancellation.
_E2_SERIES = np.array([1.0 / math.factorial(n) for n in range(16, 1, -1)])
#: A tail node past the outer features whose term is below this fraction of the
#: sum so far ends its side: the terms there fall double-exponentially.
_NEGLIGIBLE = 2.0**-60
_TWO_PI = 2.0 * math.pi
#: The smallest normal binary64 number: below it ``rho`` has lost its digits.
_TINY = 2.0**-1022
#: Below this ``d``, ``-log1p(-d) = d (1 + d / 2 + ...)`` is ``d`` to within ``u``.
_SMALL_DROP = 2.0**-53


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
def _node(
    s: float, centre: float, spread: float, size: float, w: float, rho: float, size_w: float
) -> tuple[float, float, float, float]:
    """``(x, z = size log R, e^{-x}, dx/ds)`` at the sinh-mapped node ``s``.

    Where ``1 - R`` is below ``_SMALL_DROP``, ``z = -size w / (1 + v^-2)`` from
    ``size_w``: ``w`` itself underflows near the Poisson limit, ``size >> mean``.
    """
    grow = math.exp(s)
    shrink = 1.0 / grow
    x = centre + spread * 0.5 * (grow - shrink)
    inverse_v = math.exp(-x)
    decay = inverse_v * inverse_v  # 1 / v^2
    cosine = decay / (1.0 + decay)  # 1 / (1 + v^2)
    drop = w / (1.0 + decay)  # 1 - R = w v^2 / (1 + v^2)
    if drop < _SMALL_DROP:
        z = -size_w / (1.0 + decay)
    elif drop < 0.5:
        z = size * math.log1p(-drop)
    else:
        z = size * math.log(rho + w * cosine)
    return x, z, inverse_v, spread * 0.5 * (grow + shrink)


@njit(cache=True)
def _node_term(
    s: float, centre: float, spread: float, size: float, w: float, rho: float, size_w: float
) -> tuple[float, float]:
    """``(x, e2(size log R) e^{-x} dx/ds)`` at the sinh-mapped node ``s``."""
    x, z, inverse_v, jacobian = _node(s, centre, spread, size, w, rho, size_w)
    return x, _expm1_less_identity(z) * inverse_v * jacobian


@njit(cache=True)
def _half_side(
    sign: float,
    offset: float,
    first: int,
    last: int,
    step: float,
    edge: float,
    before: float,
    centre: float,
    spread: float,
    size: float,
    w: float,
    rho: float,
    size_w: float,
) -> float:
    """``1 - e^z``'s terms at ``s = sign (j + offset) step`` for ``j = first..last``.

    A node past ``edge`` whose term is negligible against the sum plus
    ``before`` ends the side.
    """
    half = 0.0
    for j in range(first, last + 1):
        x, z, inverse_v, jacobian = _node(
            sign * (j + offset) * step, centre, spread, size, w, rho, size_w
        )
        lost = -math.expm1(z) * inverse_v * jacobian  # 1 - e^z, without cancellation
        half += lost
        if (x - edge) * sign > 0.0 and lost <= _NEGLIGIBLE * (half + before):
            break
    return half


@njit(cache=True)
def _zero_score(mean: float, size: float) -> tuple[float, float]:
    """``(CRPS(0), E|Y - Y'| / 2)``; the second is NaN unless it is the smaller.

    Both are NaN where ``rho`` is not a normal number (a size below about
    ``1e-154`` of the mean): the rule's features then lie past binary64's range.

    Where ``CRPS(0) = mean - E|Y - Y'| / 2`` exceeds ``mean / 2`` it carries the
    rounding of ``mean`` and loses the score's digits at a large mean, so the
    caller takes ``E|Y - Y'| / 2`` instead, integrating ``1 - R^size >= 0``.
    That sum adds the midpoints, halving the step: at a large mean and a large
    ``size w`` the integrand grows fast off the real axis, and the step that
    serves ``CRPS(0)`` to ``1e-14`` leaves ``E|Y - Y'| / 2`` about ``1e-10`` off.
    """
    total = mean + size
    r = mean / total
    q = size / total
    w = 4.0 * r / ((1.0 + r) * (1.0 + r))
    # size w = 4 mean q / (1 + r)^2, formed without r, which underflows where size >> mean.
    size_w = 4.0 * mean * q / ((1.0 + r) * (1.0 + r))
    sqrt_rho = q / (1.0 + r)
    rho = sqrt_rho * sqrt_rho
    if rho < _TINY:
        return np.nan, np.nan
    left = -0.5 * math.log1p(size_w)
    right = max(math.log1p(r) - math.log(q), 0.0)
    centre = 0.5 * (left + right)
    spread = 0.5 * (right - left) + 1.0
    step = _STEP / spread
    first = int(math.ceil(math.asinh((spread + _LEFT_SPAN) / spread) / step))
    last = int(math.ceil(math.asinh((spread + _RIGHT_SPAN) / spread) / step))
    acc = 0.0
    for node in range(0, last + 1):
        x, term = _node_term(node * step, centre, spread, size, w, rho, size_w)
        acc += term
        if x > right + 1.0 and term <= _NEGLIGIBLE * acc:
            break
    for node in range(1, first + 1):
        x, term = _node_term(-node * step, centre, spread, size, w, rho, size_w)
        acc += term
        if x < left - 1.0 and term <= _NEGLIGIBLE * acc:
            break
    zero = acc * step * (1.0 + r) / (_TWO_PI * q)
    if not zero > 0.5 * mean:
        return zero, np.nan
    # Each side runs until the half mean difference's own tail is negligible;
    # on the left it decays only like ``size w e^x``.  The left side's ``j``
    # counts nodes from ``s = -step``.
    law = (centre, spread, size, w, rho, size_w)
    reach = int(math.ceil(math.asinh((spread + _HALF_LEFT_SPAN) / spread) / step))
    half = _half_side(1.0, 0.0, 0, last, step, right + 1.0, 0.0, *law)
    half += _half_side(-1.0, 1.0, 0, reach - 1, step, left - 1.0, half, *law)
    middle = _half_side(1.0, 0.5, 0, last, step, right + 1.0, half, *law)
    middle += _half_side(-1.0, 0.5, 0, reach - 1, step, left - 1.0, half + middle, *law)
    return zero, 0.5 * (half + middle) * step * (1.0 + r) / (_TWO_PI * q)


@njit(cache=True)
def _shortfall(mean: float, size: float, response: float, cells: int) -> float:
    """``sum_{j < cells} (y - j) f(j)`` from ``f(0) = (size / (mean + size))^size``.

    Neither ``mean / (mean + size)`` nor ``mean / size`` is formed alone: both
    underflow near the Poisson limit, where ``size log1p(mean / size)`` is ``mean``.
    """
    total = mean + size
    ratio = mean / size
    mass = math.exp(-mean if ratio < _SMALL_DROP else -size * math.log1p(ratio))
    acc = response * mass
    for j in range(1, cells):
        mass *= (mean / j) * ((j - 1.0 + size) / total)
        acc += (response - j) * mass
    return acc


@njit(cache=True, nogil=True)
def negative_binomial_crps_rows(mean, size, response, zero, half, shortfall, start, stop):
    """Write ``CRPS(NB2(mean, size), 0)``, ``E|Y - Y'| / 2`` and ``E[(y - Y)^+]``, rows ``start:stop``.

    The caller passes contiguous float64 rows (positive finite ``mean`` and
    ``size``, finite ``response``) and the three outputs, and runs the row blocks.
    ``half`` is NaN where ``CRPS(0)`` is the smaller and the score is taken from it.
    The shortfall is NaN where the response reaches ``SHORTFALL_CELLS``, for the
    caller to take from the partial moment's closed form.  Returns ``(0, -1)``.
    """
    for row in range(start, stop):
        zero[row], half[row] = _zero_score(mean[row], size[row])
        cells = math.floor(response[row]) + 1.0
        if cells <= 0.0:
            shortfall[row] = 0.0
        elif cells <= SHORTFALL_CELLS:
            shortfall[row] = _shortfall(mean[row], size[row], response[row], int(cells))
        else:
            shortfall[row] = np.nan
    return 0, -1


def _warmup_negative_binomial_crps() -> None:
    """Compile the CRPS row kernel with the scorer's operands (no rows)."""
    rows = np.zeros(0)
    negative_binomial_crps_rows(rows, rows, rows, rows, rows, rows, 0, 0)


# The kernel the worker pool runs (``nogil``, never ``prange``), writing only its
# own rows' outputs: the pool's threads need the GIL released to run at once.
_POOLED_ROW_KERNELS = (negative_binomial_crps_rows,)

__all__ = ["SHORTFALL_CELLS", "negative_binomial_crps_rows"]
