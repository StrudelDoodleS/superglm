"""Log probabilities of response windows, for truncated and censored rows.

A truncated row divides its density by ``P(a < Y <= b)``, and a censored row
contributes ``log P(l < Y <= r)`` alone.  Both are ``log(T(near) - T(far))``
for one tail function ``T`` (the CDF, or the survival function under
reflection) taken at the window's two ends, with ``T(far) <= T(near)``.  The
family supplies ``log T`` at each end with its score and packed Hessian in the
natural parameters; :func:`log_window_channels` combines them.

With ``Delta = log T(far) - log T(near) <= 0``, ``kappa = 1/expm1(-Delta)``
and ``d = g_near - g_far``::

    log D = log T(near) + log(-expm1(Delta))
    grad  = g_near + kappa d
    hess  = H_near + kappa (H_near - H_far) - kappa (1 + kappa) d d^T

An absent far end (``Delta = -inf``), or one whose tail is negligible beside
the near end's (``kappa`` rounds to zero), returns the near end's channels
unchanged, so a one-sided window is exactly its tail.  The difference of two
log tails keeps only about ``u/Delta`` of relative accuracy as the window
narrows, so a window narrower than :data:`NARROW_WINDOW` on its density's own
scale is integrated instead by three-point Gauss-Legendre quadrature of the
density (:func:`quadrature_window_channels`).  The quadrature's relative error
there is about ``5e-7`` times the sixth power of that width (Abramowitz and
Stegun 25.4.30), below ``1e-13``.

The normal tail, :func:`normal_log_cdf_channels`, differentiates
``log Phi(x)`` through the inverse Mills ratio ``r = phi(x)/Phi(x)``:
``d log Phi/dx = r`` and ``d2 log Phi/dx2 = -r (x + r)``.  Below ``x = -3``
the gap ``x + r`` cancels (``r`` tends to ``-x``), so it comes from the
Laplace continued fraction ``x + r = 1/(t + 2/(t + 3/(t + ...)))`` with
``t = -x``, which has no subtraction (DLMF 7.9.2 for the complementary error
function).  Values come from ``scipy.special.log_ndtr``.

Imports only numpy, scipy and the shared primitive helpers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.kernels._common import FLOAT

_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
# Below this the Mills-ratio gap comes from the continued fraction; its
# convergence at t = 3 needs about 80 levels for float64, so 120 has margin.
_MILLS_SWITCH = -3.0
_MILLS_DEPTH = 120
NARROW_WINDOW = 0.05
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(3)

TailChannels = tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]


def packed_pairs(k_parameters: int) -> tuple[tuple[int, int], ...]:
    """Packed Hessian column order: ``combinations_with_replacement`` of the parameters."""
    return tuple(combinations_with_replacement(range(k_parameters), 2))


def _outer_packed(left: NDArray, right: NDArray) -> NDArray[np.float64]:
    """Packed symmetric ``(left right^T + right left^T)/2`` per row."""
    pairs = packed_pairs(left.shape[1])
    return np.column_stack(
        [0.5 * (left[:, a] * right[:, b] + left[:, b] * right[:, a]) for a, b in pairs]
    )


def inverse_mills(x: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray]:
    """``(log Phi(x), r, x + r)`` with ``r = phi(x)/Phi(x)``, tail-safe for finite ``x``."""
    values = np.asarray(x, dtype=FLOAT)
    log_cdf = special.log_ndtr(values)
    ratio = np.empty_like(values)
    gap = np.empty_like(values)
    lower = values < _MILLS_SWITCH
    direct = ~lower
    if np.any(direct):
        upper_values = values[direct]
        with np.errstate(under="ignore", over="ignore"):
            ratio[direct] = np.exp(
                -0.5 * upper_values * upper_values - _HALF_LOG_TWO_PI - log_cdf[direct]
            )
        gap[direct] = upper_values + ratio[direct]
    if np.any(lower):
        t = -values[lower]
        level = t.copy()
        for k in range(_MILLS_DEPTH - 1, 0, -1):
            level = t + (k + 1.0) / level
        fraction = 1.0 / level
        ratio[lower] = t + fraction
        gap[lower] = fraction
    return log_cdf, ratio, gap


def normal_log_cdf_channels(
    x: NDArray,
    x_score: NDArray,
    x_hessian: NDArray,
    *,
    derivative_order: int,
) -> TailChannels:
    """``log Phi(x(theta))`` with its score and packed Hessian in ``theta``.

    ``x_score`` is ``dx/dtheta`` per row and ``x_hessian`` the packed second
    derivatives of ``x``.
    """
    log_cdf, ratio, gap = inverse_mills(x)
    if derivative_order == 0:
        return log_cdf, None, None
    # Where Phi(x) rounds to one the tail is flat in float64 and its
    # derivatives are zero, though x's own derivatives may have overflowed.
    flat = ratio == 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        score = ratio[:, None] * x_score
        hessian = (
            ratio[:, None] * x_hessian - (ratio * gap)[:, None] * _outer_packed(x_score, x_score)
            if derivative_order == 2
            else None
        )
    score[flat] = 0.0
    if hessian is not None:
        hessian[flat] = 0.0
    return log_cdf, score, hessian


def log_window_channels(
    near: TailChannels,
    far: TailChannels,
    *,
    derivative_order: int,
) -> TailChannels:
    """``log(T(near) - T(far))`` and its derivatives from the two ends' log tails.

    A far end with value ``-inf`` is absent, and one with ``kappa`` rounding
    to zero is negligible; their score and Hessian rows, which may have
    overflowed, are then ignored.  A window of zero probability returns
    ``-inf``.
    """
    near_value, near_score, near_hessian = near
    far_value, far_score, far_hessian = far
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        delta = far_value - near_value
        value = near_value + np.log(-np.expm1(delta))
        kappa = 1.0 / np.expm1(-delta)
    weighted = (far_value > -np.inf) & (kappa > 0.0)
    kappa = np.where(weighted, kappa, 0.0)
    if derivative_order == 0:
        return value, None, None
    assert near_score is not None and far_score is not None
    with np.errstate(invalid="ignore", over="ignore"):
        difference = np.where(weighted[:, None], near_score - far_score, 0.0)
    score = near_score + kappa[:, None] * difference
    if derivative_order == 1:
        return value, score, None
    assert near_hessian is not None and far_hessian is not None
    with np.errstate(invalid="ignore", over="ignore"):
        spread = np.where(weighted[:, None], near_hessian - far_hessian, 0.0)
    hessian = (
        near_hessian
        + kappa[:, None] * spread
        - (kappa * (1.0 + kappa))[:, None] * _outer_packed(difference, difference)
    )
    return value, score, hessian


def quadrature_window_channels(
    lower: NDArray,
    half_width: NDArray,
    log_density: Callable[[NDArray], TailChannels],
    *,
    derivative_order: int,
) -> TailChannels:
    """``log P(lower < Y <= lower + 2 half_width)`` by Gauss-Legendre quadrature.

    ``log_density(points)`` gives the log density at ``points`` with its score
    and packed Hessian.  With node weights ``p_i`` proportional to
    ``omega_i f(y_i)``, the score is ``sum p_i g_i`` and the Hessian
    ``sum p_i (H_i + (g_i - g)(g_i - g)^T)``.
    """
    nodes = [log_density(lower + half_width * (1.0 + t)) for t in _LEGENDRE_NODES]
    logs = np.column_stack(
        [np.log(omega) + node[0] for omega, node in zip(_LEGENDRE_WEIGHTS, nodes, strict=True)]
    )
    top = np.max(logs, axis=1)
    share = np.exp(logs - np.where(np.isfinite(top), top, 0.0)[:, None])
    total = np.sum(share, axis=1)
    share /= np.where(total > 0.0, total, 1.0)[:, None]
    with np.errstate(divide="ignore"):
        value = np.log(half_width) + top + np.log(total)
    if derivative_order == 0:
        return value, None, None
    score = sum(share[:, [i]] * node[1] for i, node in enumerate(nodes))
    if derivative_order == 1:
        return value, score, None
    hessian = sum(
        share[:, [i]] * (node[2] + _outer_packed(node[1] - score, node[1] - score))
        for i, node in enumerate(nodes)
    )
    return value, score, hessian


def absent_tail(n_rows: int, k_parameters: int, *, derivative_order: int) -> TailChannels:
    """Channels of an absent far end: ``-inf`` with zero derivatives."""
    n_pairs = k_parameters * (k_parameters + 1) // 2
    return (
        np.full(n_rows, -np.inf, dtype=FLOAT),
        np.zeros((n_rows, k_parameters), dtype=FLOAT) if derivative_order >= 1 else None,
        np.zeros((n_rows, n_pairs), dtype=FLOAT) if derivative_order == 2 else None,
    )


def normal_window_channels(
    lower: NDArray,
    upper: NDArray,
    location: NDArray,
    scale: NDArray,
    precision_root: NDArray,
    *,
    derivative_order: int,
    width: NDArray | None = None,
) -> TailChannels:
    """``log P(lower < X <= upper)`` for ``X ~ N(location, (scale/precision_root)^2)``.

    Score and packed Hessian are in ``(location, scale)``.  ``lower`` may be
    ``-inf`` and ``upper`` ``+inf``, not both.  The window is read from the
    lower tail (``Phi``) when its midpoint is below the location, else from
    the upper tail by reflection, so neither end's tail rounds to one.  A
    window whose standardised width, times ``max(1, |midpoint|)``, is at most
    :data:`NARROW_WINDOW` is integrated by quadrature.  ``width`` is
    ``upper - lower`` when the caller has it more accurately than that
    difference.
    """
    lo = np.asarray(lower, dtype=FLOAT)
    hi = np.asarray(upper, dtype=FLOAT)
    mu = np.asarray(location, dtype=FLOAT)
    sigma = np.asarray(scale, dtype=FLOAT)
    root = np.asarray(precision_root, dtype=FLOAT)
    inverse = 1.0 / sigma
    with np.errstate(invalid="ignore", over="ignore"):
        z_lo = (lo - mu) * root * inverse
        z_hi = (hi - mu) * root * inverse
        from_lower_tail = z_lo + z_hi < 0.0
    sign = np.where(from_lower_tail, 1.0, -1.0)
    near_z = np.where(from_lower_tail, z_hi, z_lo)
    far_z = np.where(from_lower_tail, z_lo, z_hi)
    near = normal_log_cdf_channels(
        *_standardized(near_z, sign, root, inverse), derivative_order=derivative_order
    )
    far_present = np.isfinite(far_z)
    far = absent_tail(len(mu), 2, derivative_order=derivative_order)
    if np.any(far_present):
        present_channels = normal_log_cdf_channels(
            *_standardized(
                far_z[far_present], sign[far_present], root[far_present], inverse[far_present]
            ),
            derivative_order=derivative_order,
        )
        for target, source in zip(far, present_channels, strict=True):
            if target is not None and source is not None:
                target[far_present] = source
    channels = log_window_channels(near, far, derivative_order=derivative_order)
    span = hi - lo if width is None else np.asarray(width, dtype=FLOAT)
    with np.errstate(invalid="ignore", over="ignore"):
        reach = span * root * inverse * np.maximum(1.0, 0.5 * np.abs(z_lo + z_hi))
    rows = np.flatnonzero(reach <= NARROW_WINDOW)
    if len(rows) == 0:
        return channels

    def log_density(points: NDArray) -> TailChannels:
        return _normal_log_density(
            points, mu[rows], root[rows], inverse[rows], derivative_order=derivative_order
        )

    quadrature = quadrature_window_channels(
        lo[rows], 0.5 * span[rows], log_density, derivative_order=derivative_order
    )
    for target, source in zip(channels, quadrature, strict=True):
        if target is not None and source is not None:
            target[rows] = source
    return channels


def _standardized(
    z: NDArray, sign: NDArray, root: NDArray, inverse: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """``x = sign z`` and its derivatives in ``(mu, sigma)`` for ``z = (c - mu) k/sigma``.

    ``x_mu = -sign k/sigma``, ``x_sigma = -x/sigma``, ``x_mumu = 0``,
    ``x_musigma = sign k/sigma^2`` and ``x_sigmasigma = 2 x/sigma^2``.
    """
    x = sign * z
    with np.errstate(over="ignore", invalid="ignore"):
        x_score = np.column_stack((-sign * root * inverse, -x * inverse))
        x_hessian = np.column_stack(
            (np.zeros(len(x)), sign * root * inverse * inverse, 2.0 * x * inverse * inverse)
        )
    return x, x_score, x_hessian


def _normal_log_density(
    points: NDArray, location: NDArray, root: NDArray, inverse: NDArray, *, derivative_order: int
) -> TailChannels:
    """``log`` of the ``N(location, (scale/root)^2)`` density at ``points``, in ``(location, scale)``."""
    z = (points - location) * root * inverse
    value = np.log(root * inverse) - _HALF_LOG_TWO_PI - 0.5 * z * z
    if derivative_order == 0:
        return value, None, None
    _, z_score, z_hessian = _standardized(z, np.ones_like(z), root, inverse)
    score = -z[:, None] * z_score
    score[:, 1] -= inverse
    if derivative_order == 1:
        return value, score, None
    hessian = -z[:, None] * z_hessian - _outer_packed(z_score, z_score)
    hessian[:, 2] += inverse * inverse
    return value, score, hessian
