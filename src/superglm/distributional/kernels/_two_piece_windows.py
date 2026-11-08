"""Core natural-coordinate window jets for epsilon-skew two-piece laws.

Same-side endpoints are fixed and may use the normal window engine. Across
the join, differentiate ``s_- B(d_-) + s_+ B(d_+)``, with
``B(d) = Phi(d) - 1/2`` and ``d_-/d_+`` the nonnegative distances from the
moving join. The raw endpoint derivatives are summed *before* taking log
derivatives: individual log-half scores diverge as a half shrinks to zero.
This retains its boundary flux instead of encoding it as an absent mixture.
Only orders zero through two exist here; a tail is generally not C3 at a join.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import special

from superglm.distributional.kernels._common import validated_derivative_order
from superglm.distributional.kernels._declared_bounds import normal_window_jets
from superglm.distributional.kernels._response_tails import (
    NARROW_WINDOW,
    CoreLogMixtureAccumulator,
    Jet,
    log_mixture_jets,
    quadrature_window_jets,
)
from superglm.distributional.kernels.two_piece import log_mean_loading

_HALF_LOG_TWO_PI = 0.5 * np.log(2.0 * np.pi)
_PHI_ZERO = 1.0 / np.sqrt(2.0 * np.pi)
_U = np.finfo(np.float64).eps / 2.0
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _endpoints(lower: NDArray, upper: NDArray, transform: str):
    lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if transform == "identity":
            return lo, hi, hi - lo
        log_lo, log_hi = np.log(np.maximum(lo, 0.0)), np.log(np.maximum(hi, 0.0))
        relative = (hi - lo) / lo
        width = np.where(
            (lo > 0.0) & (relative <= 1.0),
            np.log1p(np.minimum(relative, 1.0)),
            log_hi - log_lo,
        )
    return log_lo, log_hi, width


def _unresolved_log_join(lo, hi, width, mu, theta, loading):
    """Decline a curvature whose side fraction cannot be resolved after logs.

    Log-endpoint/location rounding is O(u) times the map's magnitude. At a
    crossing its amplification is the curvature jump divided by the window
    width. This is a numerical resolution guard, not a join certificate or
    a claim that the optional higher derivatives exist. Value/score remain
    meaningful because the density is C1.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        magnitude = np.abs(lo) + np.abs(hi) + np.abs(mu)
        if loading is not None:
            magnitude += np.abs(np.log(theta[:, 0])) + np.abs(loading[0])
        uncertainty = 64.0 * _U * magnitude
        near = (lo - mu <= uncertainty) & (hi - mu >= -uncertainty)
        skew = np.abs(theta[:, 2])
        jump = 4.0 * skew / (1.0 + skew) ** 2
        return np.isfinite(width) & near & (jump * uncertainty / width > 64.0 * np.sqrt(_U))


def _mean_map(jet: Jet, theta: NDArray, loading: tuple[NDArray, ...]) -> Jet:
    """Chain ``mu=log(mean)-log K(scale,skew)``, including its mixed partial."""
    if len(jet) == 1:
        return jet
    _, k_s, k_e, k_ss, k_se, k_ee = loading
    m = theta[:, 0]
    g_m, g_s, g_e = jet[1].T
    score = np.column_stack((g_m / m, g_s - g_m * k_s, g_e - g_m * k_e))
    if len(jet) == 2:
        return jet[0], score
    h_mm, h_ms, h_me, h_ss, h_se, h_ee = jet[2].T
    hessian = np.column_stack(
        (
            (h_mm - g_m) / m / m,
            (h_ms - h_mm * k_s) / m,
            (h_me - h_mm * k_e) / m,
            h_ss - 2 * h_ms * k_s + h_mm * k_s**2 - g_m * k_ss,
            h_se - h_ms * k_e - h_me * k_s + h_mm * k_s * k_e - g_m * k_se,
            h_ee - 2 * h_me * k_e + h_mm * k_e**2 - g_m * k_ee,
        )
    )
    return jet[0], score, hessian


def _same_side(lo, hi, width, mu, sigma, skew, sign, order) -> Jet:
    s = 1.0 + sign * skew
    normal = normal_window_jets(lo, hi, mu, sigma * s, np.ones(len(mu)), order=order, width=width)
    value = normal[0] + np.log(s)
    if order == 0:
        return (value,)
    g_mu, g_w = normal[1].T
    score = np.column_stack((g_mu, s * g_w, sign * (sigma * g_w + 1.0 / s)))
    if order == 1:
        return value, score
    h_mm, h_mw, h_ww = normal[2].T
    hessian = np.column_stack(
        (
            h_mm,
            s * h_mw,
            sign * sigma * h_mw,
            s * s * h_ww,
            sign * (sigma * s * h_ww + g_w),
            sigma * sigma * h_ww - 1.0 / s**2,
        )
    )
    return value, score, hessian


def _half_skew_mass(d, half, phi):
    """``B(d)-d phi(d) = integral_0^d x^2 phi(x) dx``, stable near zero.

    For d <= 1/2 the alternating exponential series has remainder bounded
    by its first omitted term. Twelve terms leave < 4e-22 relative error;
    ordinary binary64 rounding dominates. No extended arithmetic is used.
    """
    finite_d = np.where(np.isfinite(d), d, 0.0)
    result = half - finite_d * phi
    small = d <= 0.5
    x = d[small]
    term = np.ones(len(x))
    total = term / 3.0
    for k in range(1, 12):
        term *= -0.5 * x * x / k
        total += term / (2 * k + 3)
    result[small] = _PHI_ZERO * x * x * x * total
    return result


def _crossing(lo, hi, width, mu, sigma, skew, order) -> Jet:
    sm, sp = 1.0 - skew, 1.0 + skew
    close = np.isfinite(width) & (width <= 0.5 * np.maximum(np.abs(lo), np.abs(hi)))
    residual_hi = np.where(close, (lo - mu) + width, hi - mu)
    dl, dr = (mu - lo) / (sigma * sm), residual_hi / (sigma * sp)
    bl, br = 0.5 * special.erf(dl / np.sqrt(2.0)), 0.5 * special.erf(dr / np.sqrt(2.0))
    with np.errstate(divide="ignore"):
        value = log_mixture_jets(
            np.column_stack((np.log(sm) + np.log(bl), np.log(sp) + np.log(br))), order=0
        )[0]
    if order == 0:
        return (value,)
    # Raw half-tail derivatives, divided by the *total* window mass. The
    # flux at a zero half is finite and nonzero; it is never dropped.
    pl, pr = np.exp(-0.5 * dl**2 - _HALF_LOG_TWO_PI), np.exp(-0.5 * dr**2 - _HALF_LOG_TWO_PI)
    rl, rr = (
        np.exp(-0.5 * dl**2 - _HALF_LOG_TWO_PI - value),
        np.exp(-0.5 * dr**2 - _HALF_LOG_TWO_PI - value),
    )
    left, right = np.where(rl > 0.0, dl, 0.0), np.where(rr > 0.0, dr, 0.0)
    # expm1 preserves phi(dl)-phi(dr) when both endpoints approach the join.
    smaller, larger = np.minimum(dl, dr), np.maximum(dl, dr)
    exponent = -0.5 * (larger - smaller) * (larger + smaller)
    flux = (
        np.sign(dr - dl)
        * np.exp(-0.5 * smaller**2 - _HALF_LOG_TWO_PI - value)
        * (-np.expm1(exponent))
    )
    flux = np.where(np.isinf(dl) & np.isinf(dr), 0.0, flux)
    skew_mass = _half_skew_mass(dr, br, pr) - _half_skew_mass(dl, bl, pl)
    score = np.column_stack(
        (flux / sigma, -(sm * left * rl + sp * right * rr) / sigma, skew_mass * np.exp(-value))
    )
    if order == 1:
        return value, score
    raw = np.column_stack(
        (
            -(left * rl / sm + right * rr / sp) / sigma / sigma,
            (-flux + left * left * rl - right * right * rr) / sigma / sigma,
            -(left * left * rl / sm + right * right * rr / sp) / sigma,
            (sm * left * rl * (2.0 - left * left) + sp * right * rr * (2.0 - right * right))
            / sigma
            / sigma,
            (left**3 * rl - right**3 * rr) / sigma,
            -(left**3) * rl / sm - right**3 * rr / sp,
        )
    )
    return value, score, raw - np.column_stack([score[:, i] * score[:, j] for i, j in _PAIRS])


def two_piece_window_jets(
    lower: NDArray,
    upper: NDArray,
    theta: NDArray,
    *,
    order: int,
    response_transform: str,
    parametrisation: str,
) -> Jet:
    """Unit-mass ``log P(lower < Y <= upper)`` and natural score/Hessian.

    ``theta`` has columns (location/mean, scale, skew). Nonpositive log-law
    limits map to the support boundary. Empty windows have value -inf;
    unresolved requested channels remain NaN for the family trial controller.
    """
    order = validated_derivative_order(order)
    if response_transform not in ("identity", "log") or parametrisation not in ("location", "mean"):
        raise ValueError("unknown two-piece response transform or parametrisation")
    if response_transform == "identity" and parametrisation != "location":
        raise ValueError("the real-line two-piece law uses location coordinates")
    theta = np.asarray(theta, dtype=np.float64)
    lo, hi, width = _endpoints(lower, upper, response_transform)
    n = len(theta)
    if theta.shape != (n, 3) or lo.shape != (n,) or hi.shape != (n,):
        raise ValueError("two-piece windows and parameters must have matching row shapes")
    loading = log_mean_loading(theta[:, 1], theta[:, 2]) if parametrisation == "mean" else None
    mu = theta[:, 0] if loading is None else np.log(theta[:, 0]) - loading[0]
    sigma, skew = theta[:, 1], theta[:, 2]
    result = (np.full(n, np.nan),) + tuple(np.full((n, k), np.nan) for k in (3, 6)[:order])
    whole = np.isneginf(lo) & np.isposinf(hi)
    empty = width <= 0.0
    result[0][empty] = -np.inf
    for channel in result:
        channel[whole] = 0.0
    active = ~whole & ~empty
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        for mask, sign in (
            (active & (hi <= mu), -1),
            (active & (lo >= mu), 1),
            (active & (lo < mu) & (hi > mu), 0),
        ):
            rows = np.flatnonzero(mask)
            if not len(rows):
                continue
            args = (lo[rows], hi[rows], width[rows], mu[rows], sigma[rows], skew[rows])
            jet = (
                _same_side(
                    lo[rows], hi[rows], width[rows], mu[rows], sigma[rows], skew[rows], sign, order
                )
                if sign
                else _crossing(*args, order)
            )
            for target, source in zip(result, jet, strict=True):
                target[rows] = source
        result = result if loading is None else _mean_map(result, theta, loading)
        if order == 2 and response_transform == "log":
            result[2][_unresolved_log_join(lo, hi, width, mu, theta, loading)] = np.nan
        return result


def two_piece_narrow_rows(lower, upper, theta, *, response_transform, parametrisation):
    """Finite same-side truncations admitted by the normal short-window bound.

    The Gaussian exponential and requested density derivatives vary by at
    most the normal engine's NARROW_WINDOW scale. No cache outlives this
    parameter state. Crossing windows use the explicit half-tail engine.
    """
    lo, hi, width = _endpoints(lower, upper, response_transform)
    loading = log_mean_loading(theta[:, 1], theta[:, 2]) if parametrisation == "mean" else None
    mu = theta[:, 0] if loading is None else np.log(theta[:, 0]) - loading[0]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sign = np.where(hi <= mu, -1.0, 1.0)
        spread = theta[:, 1] * (1.0 + sign * theta[:, 2])
        a = (lo - mu) / spread
        extent = width / spread
        reach = extent * np.maximum(1.0, np.abs(a + 0.5 * extent))
    resolved = (
        np.ones(len(lo), dtype=bool)
        if response_transform == "identity"
        else ~_unresolved_log_join(lo, hi, width, mu, theta, loading)
    )
    return (
        resolved
        & np.isfinite(lo)
        & np.isfinite(hi)
        & (width > 0.0)
        & ((hi <= mu) | (lo >= mu))
        & (reach <= NARROW_WINDOW)
    )


def two_piece_normalized_jets(
    response, lower, upper, side, theta, *, order, response_transform, parametrisation
) -> Jet:
    """Full conditional contributions in admitted same-side narrow windows.

    Center the log density and its derivatives on the fixed lower endpoint.
    Differences of squared residuals are factored using the small displacement
    rather than subtracting large completed channels. A common anchor cancels
    from numerator and denominator *before* rounding. The quadrature uses the
    normal engine's density/derivative variation admission (see narrow_rows).
    """
    order = validated_derivative_order(order)
    lo, hi, width = _endpoints(lower, upper, response_transform)
    loading = log_mean_loading(theta[:, 1], theta[:, 2]) if parametrisation == "mean" else None
    mu = theta[:, 0] if loading is None else np.log(theta[:, 0]) - loading[0]
    sigma, skew = theta[:, 1], theta[:, 2]
    sign = np.where(hi <= mu, -1.0, 1.0)
    s = 1.0 + sign * skew
    spread = sigma * s
    anchor = (lo - mu) / spread

    def centered(offset):
        delta = offset / spread
        square_difference = delta * (2.0 * anchor + delta)
        value = -0.5 * square_difference
        if order == 0:
            return (value,)
        g_mu = (delta / sigma) / s
        score = np.column_stack((g_mu, square_difference / sigma, sign * square_difference / s))
        if order == 1:
            return value, score
        hessian = np.column_stack(
            (
                np.zeros(len(lo)),
                -2.0 * g_mu / sigma,
                -2.0 * sign * g_mu / s,
                -3.0 * square_difference / sigma / sigma,
                -2.0 * sign * square_difference / sigma / s,
                -3.0 * square_difference / s / s,
            )
        )
        return value, score, hessian

    def average(start, span):
        # Integrate over a unit interval. log(span) is added separately, so
        # half a subnormal span is never formed and equal log widths cancel.
        # Two half-panels reduce the three-point rule's sixth-order error by
        # 64. Conditional Hessians can be small differences of polynomial
        # density moments; they need this margin beyond the density alone.
        accumulator = CoreLogMixtureAccumulator(3, order)
        for panel in (0.0, 0.5):
            accumulator.add(
                quadrature_window_jets(
                    np.full(len(lo), 0.25),
                    lambda f: centered(start + (panel + 0.5 * f) * span),
                    order=order,
                )
            )
        return accumulator.finish()

    denominator = average(np.zeros(len(lo)), width)
    offset = _endpoints(lower, response, response_transform)[2]
    exact = side == 0
    # Exact rows use a harmless positive numerator span which is overwritten.
    start = np.where(side > 0, offset, 0.0)
    span = np.where(side > 0, _endpoints(response, upper, response_transform)[2], offset)
    span = np.where(exact, width, span)
    numerator = list(average(start, span))
    exact_jet = centered(offset)
    for target, source in zip(numerator, exact_jet, strict=True):
        target[exact] = source[exact]
    result = tuple(a - b for a, b in zip(numerator, denominator, strict=True))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_ratio = np.log(span / width)
        result[0][:] += np.where(exact, -np.log(width), log_ratio)
        if response_transform == "log":
            result[0][exact] -= np.log(response[exact])
        if loading is not None:
            result = _mean_map(result, theta, loading)
    return result
