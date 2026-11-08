"""Finite NB2 windows and centered conditional likelihoods, orders zero--two.

The row law always has finite size, including exact numerators of bounded
rows. Small windows use full PMF seeds and neighboring-count recurrences;
wide windows use differentiated beta tails. Ill-conditioned differences or
unresolved fractions fall back to a count sum with derivative tail admission.
An infinite upper endpoint is never replaced by the exact-count encoding cap.

Scratch belongs to this evaluation only. There is no retained parameter,
weight, precision, or basis cache. Work is counted in recurrence/fraction
cells, with an aggregate 200,000,000-cell cap, as for complete NB rows.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

import superglm.distributional.kernels._incomplete_beta as beta
from superglm.distributional.kernels._incomplete_beta import _beta_log_shape_row, _log1pmx
from superglm.distributional.kernels._negative_binomial_numba import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    _fold,
    _ratio_remainder_row,
)
from superglm.distributional.kernels._response_tails import Jet, _checked_core_order
from superglm.distributional.kernels._severity_primitives import gnorm, psires_0123

# Compiled callees' source/constant digest, including the beta fraction. This
# invalidates Numba's caller cache when an imported numerical helper changes.
_CALLEE_DIGEST = "31ee28822a67e5e487bec0c7c573383c7c303aacb9a7f0dbbcb987309f60e46d"

_MAX_EXACT_COUNT = 2**53
_MAX_RECURRENCE_CELLS = 200_000_000
_SMALL_WINDOW = 128
_CONDITIONAL_WINDOW = 4096
_U = 2.0**-53


class CountWindowRepresentationError(ValueError):
    """A finite recorded-rate boundary has no distinguishable count encoding."""


def _boundary_count(c: float, w: float) -> float:
    if c < 0.0:
        return -1.0
    if c == np.inf:
        return np.inf
    # Exponent scaling avoids overflow in w*c before testing the encoding
    # range. The final product has the ordinary binary64 product's rounding.
    cm, ce = math.frexp(c)
    wm, we = math.frexp(w)
    if ce + we > 55:
        raise CountWindowRepresentationError("NB2 finite boundary counts must be in [0, 2**53)")
    product = math.ldexp(cm * wm, ce + we)
    if product >= _MAX_EXACT_COUNT:
        raise CountWindowRepresentationError("NB2 finite boundary counts must be in [0, 2**53)")
    k = math.floor(product)
    # Rounding w*c can place the seed on either side. Membership is defined
    # by recorded k/w, not by that rounded product or by a tolerance.
    while k > 0 and k / w > c:
        k -= 1
    while (k + 1) / w <= c:
        k += 1
        if k >= _MAX_EXACT_COUNT:
            raise CountWindowRepresentationError("NB2 boundary count reaches 2**53")
    recorded = k / w
    below = (k - 1) / w if k else -1.0
    above = (k + 1) / w
    if not (math.isfinite(recorded) and below < recorded < above):
        raise CountWindowRepresentationError(
            "NB2 boundary has indistinguishable adjacent recorded counts"
        )
    return float(k)


def count_window_endpoints(lower, upper, prior_weights):
    """Inclusive ``(L,U,empty)`` for the recorded-rate window ``lower < N/w <= upper``.

    Use ones for frequency rows. Negative support limits map to -1; infinite
    ends remain sentinels. Finite boundaries obey the existing binary64 count
    encoding range and distinguishability rule, independently of the NB tail.
    """
    lo, hi, weights = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, dtype=float)) for v in (lower, upper, prior_weights))
    )
    if lo.ndim != 1 or np.any(np.isnan(lo) | np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("count windows need row vectors with lower <= upper and no NaN")
    if np.any(~np.isfinite(weights) | (weights <= 0)):
        raise ValueError("prior weights must be finite and positive")
    L, U = np.empty_like(lo), np.empty_like(hi)
    for row, (a, b, w) in enumerate(zip(lo, hi, weights, strict=True)):
        try:
            L[row] = _boundary_count(float(a), float(w)) + 1.0
            U[row] = _boundary_count(float(b), float(w))
        except CountWindowRepresentationError as exc:
            raise CountWindowRepresentationError(f"{exc} (boundary row {row})") from exc
    return L, U, (L > U) | (lo == hi)


@njit(cache=True)
def _pmf_log_jet(k, mean, size, order):
    """Full log PMF, with derivatives in log(mean), log(size); no surrogate."""
    out = np.zeros(6)
    total = mean + size
    p, q = size / total, mean / total
    logp = -math.log1p(mean / size)
    if k == 0:
        out[0] = size * logp
    else:
        # p/(size/(size+k))-1 and q/(k/(size+k))-1. Form the
        # displacement from k-mean itself: subtracting logarithms first
        # would lose central deviance at a huge count. The linear terms
        # size*d1+k*d2 cancel algebraically, before any rounded reduction.
        d1 = (k - mean) / total
        d2 = (size / total) * ((mean - k) / k)
        r1 = _log1pmx(d1) if d1 > -0.5 else math.log((size + k) / total) - d1
        r2 = _log1pmx(d2) if d2 > -0.5 else math.log(mean / k) + math.log((size + k) / total) - d2
        deviance = -size * r1 - k * r2
        out[0] = -deviance + gnorm(size) + gnorm(k) - gnorm(size + k) - math.log(k)
    if order == 0:
        return out
    out[1] = p * (k - mean)
    if k <= _SMALL_WINDOW:
        # Compensated harmonic differences, with a cancellation-free ratio
        # remainder even when size is many orders larger than mean/count.
        rem, _, _ = _ratio_remainder_row(mean / size)
        gs, gc, hs, hc = 0.0, 0.0, 0.0, 0.0
        for j in range(int(k)):
            v = (size / (size + j)) * ((mean - j) / total)
            gs, gc = _fold(gs, gc, v)
            hs, hc = _fold(hs, hc, v * (1.0 - size / (size + j) - p))
        out[2] = (gs + gc) + size * rem
        out[5] = (hs + hc) + size * (q * q + rem)
    else:
        # psi differences split into their logarithm and small residual.
        # This form cancels the first-order displacement analytically.
        rs, rt = psires_0123(size), psires_0123(size + k)
        delta = (k - mean) / total
        if delta > -0.5:
            log_remainder = _log1pmx(delta)
        else:
            log_remainder = math.log((size + k) / total) - delta
        out[2] = size * (log_remainder + rt[0] - rs[0])
        out[5] = out[2] + size * (size * (delta * delta / (size + k) + rt[1] - rs[1]))
    if order == 2:
        out[3] = -(p * q) * (k + size)
        out[4] = (p * q) * (k - mean)
    return out


@njit(cache=True)
def _log_ratio(k, mean, size, order):
    """Jet of log(P(k+1)/P(k)), keeping small mean/size derivatives."""
    total = mean + size
    p, q = size / total, mean / total
    out = np.zeros(6)
    delta = (k - mean) / total
    if delta > -0.5:
        out[0] = math.log(mean / (k + 1.0)) + math.log1p(delta)
    else:
        out[0] = math.log(q) + math.log((size + k) / (k + 1.0))
    if order >= 1:
        out[1] = p
        out[2] = (size / (size + k)) * ((mean - k) / total)
    if order == 2:
        out[3], out[4] = -p * q, p * q
        out[5] = out[2] * (1.0 - size / (size + k) - p)
    return out


@njit(cache=True)
def _log_add(a, b, order):
    if a[0] == -np.inf:
        return b.copy()
    if b[0] > a[0]:
        a, b = b, a
    ratio = math.exp(b[0] - a[0])
    small = ratio / (1.0 + ratio)
    # sqrt-share protects representable score covariance even if share
    # itself underflows; no component with unresolved jets is discarded.
    root = math.exp(0.5 * (b[0] - a[0])) / math.sqrt(1.0 + ratio)
    out = a.copy()
    out[0] += math.log1p(ratio)
    if order >= 1:
        delta = b[1:3] - a[1:3]
        out[1:3] += (root * delta) * root
    if order == 2:
        centered = root * delta
        for col, (i, j) in enumerate(((0, 0), (0, 1), (1, 1))):
            out[3 + col] += (root * (b[3 + col] - a[3 + col])) * root + (1.0 - small) * centered[
                i
            ] * centered[j]
    return out


@njit(cache=True)
def _upper_remainder_small(term, accumulated, k, anchor, mean, size, order):
    """Geometric omitted mass and degree-1/2 derivative moment envelopes.

    Each adjacent log PMF score/Hessian change is bounded by one in log
    coordinates. At distance d from the anchor, |g|<=d, |H|<=d, hence
    |P''/P|<=d+d**2. Geometric moments bound every requested raw channel.
    A factor 1+the accumulated derivatives also bounds normalization error.
    The 64u target is a convergence tolerance, not an outward certificate.
    """
    rho = max(mean / (mean + size), math.exp(_log_ratio(k, mean, size, 0)[0]))
    if not 0 < rho < 1:
        return False
    inverse = 1.0 / (1.0 - rho)
    B0 = rho * inverse
    d = abs(k - anchor)
    B1 = d * B0 + B0 * inverse
    B2 = d * d * B0 + 2 * d * B0 * inverse + B0 * (1 + rho) * inverse * inverse
    envelope = B0
    if order >= 1:
        envelope += B1
    if order == 2:
        envelope += B2
    n = 1 if order == 0 else (3 if order == 1 else 6)
    scale = 1.0 + np.max(np.abs(accumulated[1:n])) if order else 1.0
    return term[0] - accumulated[0] + math.log(envelope * scale) < math.log(64 * _U)


@njit(cache=True)
def _relative_sums(L, U, OL, OU, mean, size, order, budget):
    """Denominator and subset numerator, using one shared zero log-PMF anchor."""
    # Keep an unbounded mode floating until it is clipped to the finite
    # window. A machine-integer conversion can wrap a large finite mean.
    mode = np.floor(mean * (1 - 1 / size)) if size > 1 else 0.0
    anchor = min(max(float(mode), L), U)
    if anchor >= _MAX_EXACT_COUNT or budget < 1 or anchor - L + 1 > budget:
        return np.full(6, np.nan), np.full(6, np.nan), anchor, 0
    den = np.zeros(6)
    num = np.zeros(6)
    num[0] = 0.0 if OL <= anchor <= OU else -np.inf
    cells = 1
    for direction in (-1, 1):
        term = np.zeros(6)
        k = anchor
        while k > L if direction < 0 else k < U:
            if cells >= budget or k + direction >= _MAX_EXACT_COUNT:
                return np.full(6, np.nan), np.full(6, np.nan), anchor, cells
            term += (
                -_log_ratio(k - 1, mean, size, order)
                if direction < 0
                else _log_ratio(k, mean, size, order)
            )
            k += direction
            den = _log_add(den, term, order)
            if OL <= k <= OU:
                num = _log_add(num, term, order)
            cells += 1
            if direction > 0 and _upper_remainder_small(term, den, k, anchor, mean, size, order):
                # A conditional numerator may have a much smaller mass; its
                # own omitted derivative tail must also meet the target.
                if k >= OU:
                    break
                if k >= OL and _upper_remainder_small(term, num, k, anchor, mean, size, order):
                    break
    return den, num, anchor, cells


@njit(cache=True)
def _beta_tail(k, mean, size, upper, order, max_terms):
    if k < 0:
        out = np.zeros(6)
        out[0] = 0.0 if upper else -np.inf
        return out, 0
    jet, work = _beta_log_shape_row(
        size, k + 1.0, math.log(size) - math.log(mean), upper, order, max_terms
    )
    out = np.zeros(6)
    out[0] = jet[0]
    if order >= 1:
        out[1], out[2] = -jet[3], jet[1] + jet[3]
    if order == 2:
        out[3], out[4], out[5] = jet[9], -jet[6] - jet[9], jet[4] + 2 * jet[6] + jet[9]
    # Composing a large beta-shape and logit derivative can cancel almost
    # completely in the size direction. Admit that composition only while
    # its conditioning preserves the measured core absolute/relative scale.
    # This is an engineering admission test, not a global error certificate;
    # a declined fraction falls back to the finite-law count recurrence.
    if order >= 1 and 64 * _U * (abs(jet[1]) + abs(jet[3])) > 2e-12 * max(1.0, abs(out[2])):
        return np.full(6, np.nan), work
    if order == 2 and (
        64 * _U * (abs(jet[6]) + abs(jet[9])) > 2e-12 * max(1.0, abs(out[4]))
        or 64 * _U * (abs(jet[4]) + 2 * abs(jet[6]) + abs(jet[9])) > 2e-12 * max(1.0, abs(out[5]))
    ):
        return np.full(6, np.nan), work
    return out, work


@njit(cache=True)
def _log_difference(near, far, order):
    if far[0] == -np.inf:
        return near.copy()
    delta = far[0] - near[0]
    if not delta < 0:
        return np.full(6, np.nan)
    v = math.log(-math.expm1(delta)) if delta > -math.log(2.0) else math.log1p(-math.exp(delta))
    ratio = math.exp(delta - v)
    out = near.copy()
    out[0] += v
    if order >= 1:
        d = far[1:3] - near[1:3]
        out[1:3] -= ratio * d
    if order == 2:
        for col, (i, j) in enumerate(((0, 0), (0, 1), (1, 1))):
            out[3 + col] -= (
                ratio * (far[3 + col] - near[3 + col]) + ratio * (1 + ratio) * d[i] * d[j]
            )
    return out


@njit(cache=True)
def _survival_from_zero(k, mean, size, order, budget):
    """P(N>k) from -expm1(log P(0)) and a positive finite body.

    Small size puts almost all mass at zero; complementing a beta CDF then
    loses shape information. This route removes that cancellation at zero.
    A poorly conditioned subsequent body subtraction is refused.
    """
    if budget < 1:
        return np.full(6, np.nan), 0
    tail = _log_difference(np.zeros(6), _pmf_log_jet(0.0, mean, size, order), order)
    if k == 0:
        return tail, 1
    den, _, anchor, cells = _relative_sums(1.0, k, 1.0, k, mean, size, order, budget - 1)
    seed_work = int(anchor) if anchor <= _SMALL_WINDOW else 1
    if cells + seed_work + 1 > budget:
        return np.full(6, np.nan), cells + 1
    body = den + _pmf_log_jet(anchor, mean, size, order)
    if body[0] - tail[0] < -0.1:
        return _log_difference(tail, body, order), cells + seed_work + 1
    return np.full(6, np.nan), cells + seed_work + 1


@njit(cache=True)
def _window_log_row(L, U, mean, size, order, budget, max_terms):
    """Log jet, work, route (1 sum, 2 beta, 3 fallback, 4 stable complement)."""
    if L > U:
        out = np.full(6, np.nan)
        out[0] = -np.inf
        return out, 0, 0
    if L == 0 and U == np.inf:
        return np.zeros(6), 0, 0
    work = 0
    if L == 1 and U == np.inf:
        jet, work = _survival_from_zero(0.0, mean, size, order, budget)
        return jet, work, 4
    if size < 0.1 and (
        (U == np.inf and L <= _CONDITIONAL_WINDOW) or (L == 0 and U < _CONDITIONAL_WINDOW)
    ):
        jet, work = _survival_from_zero(L - 1 if U == np.inf else U, mean, size, order, budget)
        if np.all(np.isfinite(jet)):
            if L == 0:
                jet = _log_difference(np.zeros(6), jet, order)
            return jet, work, 4
    small = U - L < _SMALL_WINDOW
    # When size dwarfs the whole region of appreciable mass, composing beta
    # shape and logit partials would cancel the finite size information.
    # The count sum retains it; no Poisson law is selected.
    near_poisson = size > 1e6 * max(mean, 1.0) and mean < _CONDITIONAL_WINDOW
    if not small and not near_poisson and budget > work:
        limit = min(max_terms, budget - work)
        if U == np.inf:
            jet, cells = _beta_tail(L - 1, mean, size, True, order, limit)
            work += cells
        elif L == 0:
            jet, cells = _beta_tail(U, mean, size, False, order, limit)
            work += cells
        else:
            upper = L > mean
            near, cells = _beta_tail(L - 1 if upper else U, mean, size, upper, order, limit)
            work += cells
            far, cells = _beta_tail(
                U if upper else L - 1,
                mean,
                size,
                upper,
                order,
                min(max_terms, max(0, budget - work)),
            )
            work += cells
            # Avoid amplification beyond roughly twenty times the measured
            # tail error. Narrow differences use the direct count sum.
            jet = (
                _log_difference(near, far, order) if far[0] - near[0] < -0.1 else np.full(6, np.nan)
            )
        if np.all(np.isfinite(jet)):
            return jet, work, 2
    if size < 0.1 and ((U == np.inf and L > 1) or (L == 0 and np.isfinite(U))):
        jet, cells = _survival_from_zero(
            L - 1 if U == np.inf else U, mean, size, order, budget - work
        )
        work += cells
        if np.all(np.isfinite(jet)):
            if L == 0:
                jet = _log_difference(np.zeros(6), jet, order)
            return jet, work, 4
    den, _, anchor, cells = _relative_sums(L, U, L, U, mean, size, order, budget - work)
    work += cells
    if np.all(np.isfinite(den)):
        seed_work = int(anchor) if anchor <= _SMALL_WINDOW else 1
        if work + seed_work <= budget:
            jet = den + _pmf_log_jet(anchor, mean, size, order)
            work += seed_work
            if L == 0 and jet[0] > -0.01:
                # Recover a near-one CDF from its small upper complement,
                # including its tiny derivatives. Never clip a positive log
                # probability produced by cancelling the seed and sum.
                tail, _, anchor, cells = _relative_sums(
                    U + 1, np.inf, U + 1, np.inf, mean, size, order, budget - work
                )
                work += cells
                seed_work = int(anchor) if anchor <= _SMALL_WINDOW else 1
                if work + seed_work <= budget and np.all(np.isfinite(tail)):
                    tail += _pmf_log_jet(anchor, mean, size, order)
                    return _log_difference(np.zeros(6), tail, order), work + seed_work, 4
                return np.full(6, np.nan), work, 4
            return jet, work, 1 if small or near_poisson else 3
    return np.full(6, np.nan), work, 3


@njit(cache=True)
def _normalized_log_row(L, U, OL, OU, mean, size, order, budget, max_terms):
    if L == OL and U == OU:
        return np.zeros(6), 0
    near_poisson = (
        size > 1e6 * max(mean, 1.0)
        and mean < _CONDITIONAL_WINDOW
        and OL <= max(L, mean) + _CONDITIONAL_WINDOW
    )
    if U - L < _CONDITIONAL_WINDOW or near_poisson:
        den, num, _, work = _relative_sums(L, U, OL, OU, mean, size, order, budget)
        return num - den, work
    numerator, work, _ = _window_log_row(OL, OU, mean, size, order, budget, max_terms)
    denominator, cells, _ = _window_log_row(L, U, mean, size, order, budget - work, max_terms)
    return numerator - denominator, work + cells


def _inputs(theta, prior_weights, order):
    order = _checked_core_order(order)
    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("NB2 parameters must have shape (n, 2)")
    weights = np.broadcast_to(np.asarray(prior_weights, dtype=float), (len(parameters),))
    if np.any(~np.isfinite(weights) | (weights <= 0)):
        raise ValueError("prior weights must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        effective = parameters * weights[:, None]
    valid = np.all(
        np.isfinite(parameters) & (parameters >= DOMAIN_MIN) & (parameters <= DOMAIN_MAX), axis=1
    )
    valid &= np.all(
        np.isfinite(effective) & (effective >= DOMAIN_MIN) & (effective <= DOMAIN_MAX), axis=1
    )
    valid &= (weights >= DOMAIN_MIN) & (weights <= DOMAIN_MAX)
    return parameters, weights, effective, valid, order


def _natural_jets(out, parameters, order):
    if order == 0:
        return (out[:, 0],)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        score = out[:, 1:3] / parameters
        if order == 1:
            return out[:, 0], score
        hessian = out[:, 3:].copy()
        hessian[:, 0] -= out[:, 1]
        hessian[:, 2] -= out[:, 2]
        for col, (i, j) in enumerate(((0, 0), (0, 1), (1, 1))):
            hessian[:, col] = (hessian[:, col] / parameters[:, i]) / parameters[:, j]
    return out[:, 0], score, hessian


def negative_binomial_window_jets(lower, upper, theta, prior_weights, *, order: int) -> Jet:
    """Unit-mass log window jets in natural ``(mean,theta)`` coordinates.

    Empty count sets have log probability -inf. An unresolved numerical state
    has NaNs, allowing the solver to backtrack/disclose it. Infinite support
    tails are evaluated by beta or a derivative-admitted sum, never clipped.
    """
    parameters, weights, effective, valid, order = _inputs(theta, prior_weights, order)
    L, U, empty = count_window_endpoints(lower, upper, weights)
    out = np.full((len(parameters), 6), np.nan)
    budget = _MAX_RECURRENCE_CELLS
    for row, (mean, size) in enumerate(effective):
        if empty[row]:
            out[row, 0] = -np.inf
        elif valid[row]:
            out[row], work, _ = _window_log_row(
                L[row], U[row], mean, size, order, budget, beta.MAX_TERMS
            )
            budget -= work
    return _natural_jets(out, parameters, order)


def negative_binomial_normalized_jets(L, U, OL, OU, theta, prior_weights, *, order: int) -> Jet:
    """Complete unit-mass bounded contributions from inclusive count sets.

    Exact numerators are singleton sets. Finite small conditioning sets share
    a zero PMF anchor, so huge factorials and common scores never enter the
    subtraction. Inputs are the binder's validated, support-clipped count sets.
    """
    parameters, _, effective, valid, order = _inputs(theta, prior_weights, order)
    out = np.full((len(parameters), 6), np.nan)
    budget = _MAX_RECURRENCE_CELLS
    for row, (mean, size) in enumerate(effective):
        if valid[row]:
            out[row], work = _normalized_log_row(
                L[row], U[row], OL[row], OU[row], mean, size, order, budget, beta.MAX_TERMS
            )
            budget -= work
    return _natural_jets(out, parameters, order)
