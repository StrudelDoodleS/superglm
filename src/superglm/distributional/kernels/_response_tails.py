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

An absent far end (``Delta = -inf``) gives ``kappa = 0`` and returns the near
end's channels unchanged, so a one-sided window is exactly its tail.  The
form cancels only as the window narrows (``kappa`` large), where the
probability itself is ill-conditioned in the parameters.

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
        with np.errstate(under="ignore"):
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
    score = ratio[:, None] * x_score
    if derivative_order == 1:
        return log_cdf, score, None
    hessian = ratio[:, None] * x_hessian - (ratio * gap)[:, None] * _outer_packed(x_score, x_score)
    return log_cdf, score, hessian


def log_window_channels(
    near: TailChannels,
    far: TailChannels,
    *,
    derivative_order: int,
) -> TailChannels:
    """``log(T(near) - T(far))`` and its derivatives from the two ends' log tails.

    A far end with value ``-inf`` is absent; its score and Hessian rows are
    then ignored.  A window of zero probability returns ``-inf``.
    """
    near_value, near_score, near_hessian = near
    far_value, far_score, far_hessian = far
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        delta = far_value - near_value
        value = near_value + np.log(-np.expm1(delta))
        kappa = 1.0 / np.expm1(-delta)
    present = far_value > -np.inf
    kappa = np.where(present, kappa, 0.0)
    if derivative_order == 0:
        return value, None, None
    assert near_score is not None and far_score is not None
    difference = np.where(present[:, None], near_score - far_score, 0.0)
    score = near_score + kappa[:, None] * difference
    if derivative_order == 1:
        return value, score, None
    assert near_hessian is not None and far_hessian is not None
    hessian = (
        near_hessian
        + kappa[:, None] * np.where(present[:, None], near_hessian - far_hessian, 0.0)
        - (kappa * (1.0 + kappa))[:, None] * _outer_packed(difference, difference)
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
) -> TailChannels:
    """``log P(lower < X <= upper)`` for ``X ~ N(location, (scale/precision_root)^2)``.

    Score and packed Hessian are in ``(location, scale)``.  ``lower`` may be
    ``-inf`` and ``upper`` ``+inf``, not both.  The window is read from the
    lower tail (``Phi``) when its midpoint is below the location, else from
    the upper tail by reflection, so neither end's tail rounds to one.
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
    return log_window_channels(near, far, derivative_order=derivative_order)


def _standardized(
    z: NDArray, sign: NDArray, root: NDArray, inverse: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """``x = sign z`` and its derivatives in ``(mu, sigma)`` for ``z = (c - mu) k/sigma``.

    ``x_mu = -sign k/sigma``, ``x_sigma = -x/sigma``, ``x_mumu = 0``,
    ``x_musigma = sign k/sigma^2`` and ``x_sigmasigma = 2 x/sigma^2``.
    """
    x = sign * z
    x_score = np.column_stack((-sign * root * inverse, -x * inverse))
    x_hessian = np.column_stack(
        (np.zeros(len(x)), sign * root * inverse * inverse, 2.0 * x * inverse * inverse)
    )
    return x, x_score, x_hessian
