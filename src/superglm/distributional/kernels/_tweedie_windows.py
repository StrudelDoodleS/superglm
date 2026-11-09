"""Atom-aware compound Poisson/gamma windows, with window-relative admission.

There is no approximation to the observation law. The CDF's Chernoff range
starts the sum; omitted Poisson moments bound every requested raw derivative
over ANY response window. Bounds use log tails (never underflowed CDFs),
positive centered-moment majorants, and outward positive arithmetic. The
gamma engine's measured envelope is additional numerical evidence, not a
rigorous universal certificate. Independent normalized-likelihood tests cover
the composed core channels and the narrow/rare/small-shape routes.

Support: representable positive rates/scales, 1 < p < 2, at most 20,000 as
the largest Poisson index, and the call-local aggregate gamma work budget.
No extra power/shape wall is imposed. Failed special functions, unresolved
window-relative remainders, or nonrepresentable channels return NaNs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy import special

from superglm.distributional.kernels._gamma_windows import (
    GammaWindowWorkspace,
    gamma_shape_scale_window_jets,
)
from superglm.distributional.kernels._response_tails import (
    CoreLogMixtureAccumulator,
    Jet,
    _checked_core_order,
)
from superglm.distributional.kernels._severity_primitives import (
    gdev,
    gnorm,
    polygamma_0123,
    psires_0123,
)

_U = 2.0**-53
_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_MAX_POISSON_INDEX = 20_000
_PROBABILITY_TARGET = 1e-12
_DERIVATIVE_TARGET = 1e-10
_ZERO_SCORE = np.zeros(3)
_ZERO_HESSIAN = np.zeros(6)


@dataclass
class TweedieWindowWorkspace:
    """Instrumentation and reuse for ONE parameter state, released by caller."""

    gamma: GammaWindowWorkspace = field(default_factory=GammaWindowWorkspace)
    evaluated_terms: int = 0
    largest_index: int = 0
    initial_largest_index: int = 0
    route: str = "compound_poisson_gamma"
    refusal_reason: str | None = None


def _initial_range(rate):
    # Same Chernoff window as the complete CDF. This is only a starting range.
    level = -math.log(0.5e-12)
    spread = np.sqrt(2 * level * rate)
    high = rate + spread + level
    for _ in range(8):
        slope = np.log(high) - np.log(rate)
        high = np.maximum(high - (high * slope - high + rate - level) / slope, rate * (1 + 1e-6))
    low = np.clip(rate - spread - level, 1e-12, rate * (1 - 1e-6))
    for _ in range(8):
        slope = np.log(low) - np.log(rate)
        low = np.clip(low - (low * slope - low + rate - level) / slope, 1e-12, rate * (1 - 1e-6))
    first = np.floor(np.where(rate > level, low, 0)).astype(int) + 1
    last = np.maximum(1, np.ceil(high)).astype(int)
    return first, last


@njit(cache=True, error_model="numpy")
def _up(x):
    return np.nextafter(x, np.inf)


@njit(cache=True, error_model="numpy")
def _add(x, y):
    return _up(x + y)


@njit(cache=True, error_model="numpy")
def _mul(x, y):
    return _up(x * y)


@njit(cache=True, error_model="numpy")
def _log_tail(rate, k, upper):
    """Chernoff upper bound on P(N>=k) or P(N<=k), including rounding.

    The exponent k log(k/rate)-k+rate is evaluated away from the mode;
    a gamma_32 allowance over its terms covers arithmetic and the libm
    log envelope (the latter is numerical evidence on supported platforms).
    Unlike a probability-valued incomplete gamma, it cannot falsely certify
    zero after underflow. No subtractive raw moments are formed.
    """
    if k < 0:
        return 0.0 if upper else -np.inf
    if upper and k <= rate or not upper and k >= rate:
        return 0.0
    if k == 0:
        return 0.0 if upper else _up(-rate)
    term = k * (math.log(k) - math.log(rate))
    value = -term + k - rate
    allowance = 32 * _U / (1 - 32 * _U) * (abs(term) + k + rate + 1)
    return min(0.0, _up(value + _up(allowance)))


@njit(cache=True, error_model="numpy")
def _log_omitted_moments(rate, first, last):
    logs = []
    for degree in range(3):
        # E[(N)_d 1_O] = lambda^d P(N+d in O).
        lower = _log_tail(rate, first - 1 - degree, False) if first > 1 else -np.inf
        upper = _log_tail(rate, last + 1 - degree, True)
        logs.append(_up(np.logaddexp(lower, upper) + degree * math.log(rate)))
    logs[2] = _up(np.logaddexp(logs[2], logs[1]))
    return np.asarray(logs)


@njit(cache=True, error_model="numpy")
def _relative_remainder(rate, a, first, last, log_probability):
    """M0, |D P|, |D2 P| divided by the retained probability, in (L,a,B)."""
    moments = _log_omitted_moments(rate, first, last)
    # All further sums/products are positive and rounded outward, one
    # nextafter per operation. Underflows round UP to a positive subnormal.
    m0, m1, m2 = _up(np.exp(_up(moments - log_probability)))
    vl = _add(_add(m2, _mul(2 * rate, m1)), _mul(_mul(rate, rate), m0))
    va = _add(_up(m1 / a), _up(_up(m0 / a) / a))
    vb = _mul(a, m1)
    variance = np.array([vl, va, vb])
    # sqrt(m0)*sqrt(V), instead of sqrt(m0*V), avoids premature underflow.
    first_bound = _mul(_up(np.sqrt(m0)), _up(np.sqrt(variance)))
    h = np.array([_mul(rate, m0), 0, 0, va, m1, vb])
    second = np.empty(6)
    for c, (i, j) in enumerate(_PAIRS):
        second[c] = _add(h[c], _mul(_up(np.sqrt(variance[i])), _up(np.sqrt(variance[j]))))
    return m0, first_bound, second


def _map(theta):
    """z=(log rate,a,log scale) differentiated in t=(log mean,log phi,p)."""
    mu, phi, p = theta.T
    r, q = p - 1, 2 - p
    lm = np.log(mu)
    jac = np.zeros((len(theta), 3, 3))
    jac[:, 0, :] = np.column_stack((q, -np.ones(len(theta)), -lm + 1 / q))
    jac[:, 1, 2] = -1 / r**2
    jac[:, 2, :] = np.column_stack((r, np.ones(len(theta)), lm + 1 / r))
    second = np.zeros((len(theta), 3, 6))
    second[:, 0, 2], second[:, 2, 2] = -1, 1
    second[:, 0, 5], second[:, 1, 5], second[:, 2, 5] = 1 / q**2, 2 / r**3, -1 / r**2
    return jac, second


def _compose(jet, theta, order):
    if order == 0:
        return jet
    jac, second = _map(theta)
    score = np.einsum("ni,nij->nj", jet[1], jac)
    scale = np.column_stack((theta[:, 0], theta[:, 1], np.ones(len(theta))))
    result = [jet[0], score / scale]
    if order == 2:
        hessian = np.empty((len(theta), 3, 3))
        for c, (i, j) in enumerate(_PAIRS):
            hessian[:, i, j] = hessian[:, j, i] = jet[2][:, c]
        composed = np.einsum("nai,nab,nbj->nij", jac, hessian, jac)
        packed = np.column_stack([composed[:, i, j] for i, j in _PAIRS])
        packed += np.einsum("na,naq->nq", jet[1], second)
        packed[:, 0] -= score[:, 0]
        packed[:, 3] -= score[:, 1]
        result.append(
            np.column_stack(
                [packed[:, c] / scale[:, i] / scale[:, j] for c, (i, j) in enumerate(_PAIRS)]
            )
        )
    return tuple(result)


def _admitted(jet, row, rate, a, first, last, jac, second, order):
    return _admitted_core(
        jet[0][row],
        jet[1][row] if order else _ZERO_SCORE,
        jet[2][row] if order == 2 else _ZERO_HESSIAN,
        rate,
        a,
        first,
        last,
        jac,
        second,
        order,
    )


@njit(cache=True, error_model="numpy")
def _admitted_core(value, g, h, rate, a, first, last, jac, second, order):
    """The same outward-rounded bound, without per-operation Python dispatch.

    Strict IEEE arithmetic (no fastmath) retains every nextafter, including
    the positive subnormal on underflow. Only constant-sized row work lives
    here; the enclosing tile still owns expansion and the aggregate budget.
    """
    if not np.isfinite(value):
        return False
    if order and not np.isfinite(g).all():
        return False
    if order == 2 and not np.isfinite(h).all():
        return False
    mass, d1, d2 = _relative_remainder(rate, a, first, last, value)
    # Reserve most of the derivative budget for the measured gamma/composition
    # envelope. This is a truncation bound, not a claim of certified rounding.
    if not mass <= _PROBABILITY_TARGET / 8:
        return False
    if order == 0:
        return True
    eg = _add(d1, _mul(mass, np.abs(g)))
    mapped_g = np.abs(jac).T @ eg
    scale_g = np.maximum(1.0, np.abs(g @ jac))
    if np.any(mapped_g * (1 / (1 - 16 * _U)) > _DERIVATIVE_TARGET / 8 * scale_g):
        return False
    if order == 1:
        return True
    eh = np.empty((3, 3))
    hh = np.empty((3, 3))
    for c, (i, j) in enumerate(_PAIRS):
        raw = _add(abs(h[c]), _mul(abs(g[i]), abs(g[j])))
        bound = _add(d2[c], _mul(mass, raw))
        bound = _add(bound, _add(_mul(abs(g[i]), eg[j]), _mul(abs(g[j]), eg[i])))
        bound = _add(bound, _mul(eg[i], eg[j]))
        eh[i, j] = eh[j, i] = bound
        hh[i, j] = hh[j, i] = h[c]
    # Three terms in each dot: gamma_32 inflation covers both products,
    # reductions and the following additions. Inputs are nonnegative.
    inflation = 1 / (1 - 32 * _U)
    mapped_h = np.abs(jac).T @ eh @ np.abs(jac)
    reference_h = jac.T @ hh @ jac
    for c, (i, j) in enumerate(_PAIRS):
        error_second, reference_second = 0.0, 0.0
        for k in range(3):
            error_second += eg[k] * abs(second[k, c])
            reference_second += g[k] * second[k, c]
        error = (mapped_h[i, j] + error_second) * inflation
        scale = max(1.0, abs(reference_h[i, j] + reference_second))
        if not error <= _DERIVATIVE_TARGET / 8 * scale:
            return False
    return True


def _density_anchor(response, theta, prior_weights, order):
    """One joint Poisson/gamma density, shared by conditional numerator/denominator.

    The integer component is fixed during differentiation. It may change on
    another trial: the same anchor cancels algebraically from that trial's
    normalized likelihood. Both intermediate and natural jets are returned.
    """
    values = np.empty(len(theta))
    score, hessian = np.zeros((len(theta), 3)), np.zeros((len(theta), 6))
    for row, ((mu, phi, p), w, y) in enumerate(zip(theta, prior_weights, response, strict=True)):
        ll = math.log(w) + (2 - p) * math.log(mu) - math.log(phi) - math.log(2 - p)
        rate = math.exp(ll)
        a = (2 - p) / (p - 1)
        bb = math.log(phi) + math.log(p - 1) + (p - 1) * math.log(mu) - math.log(w)
        j = max(1, int(round(rate)))
        shape, x = j * a, math.exp(math.log(y) - bb)
        values[row] = (
            gnorm(float(j))
            - math.log(j)
            - j * gdev(rate, float(j))
            + gnorm(shape)
            - shape * gdev(x, shape)
            - math.log(y)
        )
        score[row] = (j - rate, j * (math.log(x / shape) - psires_0123(shape)[0]), x - shape)
        hessian[row, 0], hessian[row, 3], hessian[row, 4], hessian[row, 5] = (
            -rate,
            -j * j * polygamma_0123(shape)[1],
            -j,
            -x,
        )
    jet = (values, score, hessian)[: order + 1]
    return jet, _compose(jet, theta, order)


def tweedie_window_jets(
    lower, upper, theta, prior_weights, *, order, _workspace=None, _anchor=None
) -> Jet:
    """Natural (mean,dispersion,power) log-window jets, orders 0--2.

    Frequency is deliberately absent: weights here define the prior law.
    The optional private workspace combines numerator/denominator endpoint
    reuse and diagnostics, and must be discarded at the end of the call.
    """
    order = _checked_core_order(order)
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[1] != 3:
        raise ValueError("Tweedie window theta must have shape (n,3)")
    n = len(theta)
    lo, hi, weights = (
        np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in (lower, upper, prior_weights)
    )
    work = TweedieWindowWorkspace() if _workspace is None else _workspace
    out = (np.full(n, np.nan), np.full((n, 3), np.nan), np.full((n, 6), np.nan))[: order + 1]
    # Bound the live row/term working set. The endpoint memo and aggregate
    # budget still belong to the enclosing call, including its other window.
    if n > 128:
        for start in range(0, n, 128):
            tile = slice(start, min(start + 128, n))
            result = tweedie_window_jets(
                lo[tile],
                hi[tile],
                theta[tile],
                weights[tile],
                order=order,
                _workspace=work,
                _anchor=None if _anchor is None else tuple(c[tile] for c in _anchor),
            )
            for target, source in zip(out, result, strict=True):
                target[tile] = source
        return out
    mu, phi, p = theta.T
    valid = (
        np.isfinite(theta).all(axis=1)
        & (mu > 0)
        & (phi > 0)
        & (p > 1)
        & (p < 2)
        & np.isfinite(weights)
        & (weights > 0)
        & ~np.isnan(lo)
        & ~np.isnan(hi)
    )
    with np.errstate(all="ignore"):
        ll = np.log(weights) + (2 - p) * np.log(mu) - np.log(phi) - np.log(2 - p)
        a = (2 - p) / (p - 1)
        bb = np.log(phi) + np.log(p - 1) + (p - 1) * np.log(mu) - np.log(weights)
        rate = np.exp(ll)
    valid &= np.isfinite(rate) & (rate > 0) & np.isfinite(a) & np.isfinite(bb)
    empty = valid & ((hi <= lo) | (hi < 0) | ((hi == 0) & (lo >= 0)))
    for channel in out:
        channel[empty] = 0.0
    out[0][empty] = -np.inf
    whole = valid & (lo < 0) & (hi == np.inf)
    for channel in out:
        channel[whole] = 0.0
    special_rows = np.flatnonzero(
        valid & ~empty & ~whole & ((hi == 0) | ((lo == 0) & (hi == np.inf)))
    )
    if len(special_rows):
        rr = rate[special_rows]
        atom = hi[special_rows] == 0
        value = np.where(atom, -rr, np.log(-np.expm1(-rr)))
        score = np.zeros((len(rr), 3))
        hessian = np.zeros((len(rr), 6))
        with np.errstate(over="ignore", invalid="ignore"):
            g = rr / np.expm1(rr)
            h = g * (1 - rr / (-np.expm1(-rr)))
        score[:, 0] = np.where(atom, -rr, g)
        hessian[:, 0] = np.where(atom, -rr, h)
        converted = _compose((value, score, hessian)[: order + 1], theta[special_rows], order)
        for target, source in zip(out, converted, strict=True):
            target[special_rows] = source
    if _anchor is not None:
        # Special/whole events are analytic before centering. Mixture rows
        # below instead center EACH component before the positive sum.
        natural_anchor = _compose(_anchor, theta, order)
        for target, shift in zip(out, natural_anchor, strict=True):
            target -= shift
    active = valid & ~empty & ~whole
    active[special_rows] = False
    active &= rate < _MAX_POISSON_INDEX
    if np.any(valid & ~empty & ~whole & (rate >= _MAX_POISSON_INDEX)):
        work.refusal_reason = "Poisson upper-index budget"
    rows = np.flatnonzero(active)
    if not len(rows):
        return out
    r, aa = rate[rows], a[rows]
    first, last = _initial_range(r)
    last = np.minimum(last, _MAX_POISSON_INDEX)
    work.initial_largest_index = max(work.initial_largest_index, int(last.max()))
    accumulator = CoreLogMixtureAccumulator(3, order)
    atom = (lo[rows] < 0) & (hi[rows] >= 0)
    values = np.where(atom, -r, -np.inf)
    scores, hessians = np.zeros((len(rows), 3)), np.zeros((len(rows), 6))
    scores[:, 0], hessians[:, 0] = -r, -r
    anchor = None if _anchor is None else tuple(c[rows] for c in _anchor)

    def centered(jet):
        if anchor is None:
            return jet
        return tuple(c - shift for c, shift in zip(jet, anchor, strict=True))

    def absolute(jet):
        if anchor is None:
            return jet
        return tuple(c + shift for c, shift in zip(jet, anchor, strict=True))

    accumulator.add(centered((values, scores, hessians)[: order + 1]))
    jac, second = _map(theta[rows])
    settled = np.zeros(len(rows), dtype=bool)

    def add_range(start, end):
        for j in range(int(start.min()), int(end.max()) + 1):
            selected = np.flatnonzero(~settled & (start <= j) & (j <= end))
            if not len(selected):
                continue
            original = rows[selected]
            gamma = gamma_shape_scale_window_jets(
                lo[original],
                hi[original],
                j * aa[selected],
                bb[original],
                order=order,
                _workspace=work.gamma,
            )
            v = np.full(len(rows), -np.inf)
            s = np.zeros((len(rows), 3))
            h = np.zeros((len(rows), 6))
            v[selected] = j * ll[original] - r[selected] - special.gammaln(j + 1) + gamma[0]
            if order:
                s[selected, 0], s[selected, 1], s[selected, 2] = (
                    j - r[selected],
                    j * gamma[1][:, 0],
                    gamma[1][:, 1],
                )
            if order == 2:
                h[selected, 0], h[selected, 3], h[selected, 4], h[selected, 5] = (
                    -r[selected],
                    j * j * gamma[2][:, 0],
                    j * gamma[2][:, 1],
                    gamma[2][:, 2],
                )
            accumulator.add(centered((v, s, h)[: order + 1]))
            work.evaluated_terms += len(selected)
            work.largest_index = max(work.largest_index, j)

    add_range(first, last)
    while True:
        jet = accumulator.finish()
        absolute_jet = absolute(jet)
        for row in np.flatnonzero(~settled):
            settled[row] = _admitted(
                absolute_jet,
                row,
                r[row],
                aa[row],
                first[row],
                last[row],
                jac[row],
                second[row],
                order,
            )
        failed = ~np.isfinite(jet[0])
        for channel in jet[1:]:
            failed |= ~np.isfinite(channel).all(axis=1)
        if settled.all() or np.all(settled | failed | (last >= _MAX_POISSON_INDEX)):
            break
        settled |= failed  # retain NaNs; do not waste work after an engine refusal
        step = np.maximum(8, (last - first + 1) // 4)
        new_first = np.where(settled, first, np.maximum(1, first - step))
        new_last = np.where(settled, last, np.minimum(_MAX_POISSON_INDEX, last + step))
        add_range(new_first, first - 1)
        add_range(last + 1, new_last)
        first, last = new_first, new_last
    for row in range(len(rows)):
        if not _admitted(
            absolute_jet, row, r[row], aa[row], first[row], last[row], jac[row], second[row], order
        ):
            for channel in jet:
                channel[row] = np.nan
            work.refusal_reason = (
                work.gamma.refusal_reason
                or "window-relative Poisson derivative remainder / index budget"
            )
    converted = _compose(jet, theta[rows], order)
    for target, source in zip(out, converted, strict=True):
        target[rows] = source
    return out
