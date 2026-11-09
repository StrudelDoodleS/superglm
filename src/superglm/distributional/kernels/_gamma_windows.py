"""Core gamma windows in (shape, log scale), on the shipping tail engine.

The endpoint memo belongs to ONE likelihood evaluation (including both
normalizing and observed windows). Its keys contain endpoint, shape, log scale,
tail and order; it is never stored on a plan/model. Precision is binary64.

Tail accuracy uses the measured order-0--2 envelope of
``test_incomplete_gamma_engine``, not a universal error certificate. Narrow
positive windows instead integrate in log response. The five-node rule is
admitted only when width * max(1, |a-x|, sqrt(x)) <= .05. Derivatives of the
log-density in this variable are a-x, -x, ...; differentiating twice in the
parameters adds at most a quadratic score factor. The degree-ten remainder
therefore has the same small-width bound, with a conservative factor 13!
(5-point Gauss coefficient < 4e-13, giving < 3e-16 after scaling). Endpoints
at zero always use tails; smooth quadrature is never applied at a singularity.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from superglm.distributional.kernels._incomplete_gamma import (
    MAX_TERMS,
    WORK_SIZE,
    log_gamma_tail_jet_into,
)
from superglm.distributional.kernels._response_tails import (
    Jet,
    _checked_core_order,
)
from superglm.distributional.kernels._severity_primitives import (
    gdev,
    gnorm,
    polygamma_0123,
    psires_0123,
)

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(5)
_U = 2.0**-53
_CALLEE_DIGEST = "e5808dafa97122ff434d474bac404b304b59fe0ad6b27a3f62cfa1080e2d8d2c"


@dataclass
class GammaWindowWorkspace:
    """Call-local endpoint reuse and bounded work; no parameter cache survives."""

    # A call has a hard aggregate endpoint budget as well as the primitive's
    # 100,000-iteration per-endpoint budget. See the stage timing receipt.
    max_evaluations: int = 250_000
    evaluations: int = 0
    quadrature_nodes: int = 0
    cache_hits: int = 0
    # The primitive returns convergence, not its iteration count. An optional
    # untimed observer can replay its unchanged recurrence and report that
    # count; timed calls never pay for replay. None means NOT measured.
    iteration_observer: Callable | None = None
    gamma_iterations: int | None = None
    refusal_reason: str | None = None
    endpoints: dict = field(default_factory=dict)
    scratch: np.ndarray = field(default_factory=lambda: np.empty(WORK_SIZE))

    @property
    def iteration_upper_bound(self):
        # The small-shape branch can visit both its Q series and the P series.
        return self.evaluations * (2 * MAX_TERMS)


def _tail(point, shape, log_scale, upper, order, workspace):
    key = (point, shape, log_scale, upper, order)
    if key in workspace.endpoints:
        workspace.cache_hits += 1
        return workspace.endpoints[key]
    result = np.full(6, np.nan)
    if workspace.evaluations >= workspace.max_evaluations:
        workspace.refusal_reason = "aggregate gamma endpoint work budget"
        return result
    log_x = math.log(point) - log_scale
    if not -708.0 < log_x < 709.0:
        workspace.refusal_reason = "gamma argument outside normal float64 range"
        return result
    x = math.exp(log_x)
    jet = np.empty(6)
    workspace.evaluations += 1
    # The fixed-ratio coordinates keep the prefactor stable near x/a=1.
    along = abs(x - shape) < 0.5 * shape
    success = log_gamma_tail_jet_into(shape, x, upper, order, along, jet, workspace.scratch)
    if workspace.iteration_observer is not None:
        count = workspace.iteration_observer(shape, x, upper, order, along)
        workspace.gamma_iterations = (workspace.gamma_iterations or 0) + count
    if not success:
        workspace.refusal_reason = "incomplete gamma did not converge"
        return result
    result[0] = jet[0]
    if order:
        # x_actual/x0 = exp(-dB). In the fixed-ratio chart this means
        # e = a/(a+da)*exp(-dB)-1, so e_a=-1/a, e_B=-1,
        # e_aa=2/a^2, e_aB=1/a, e_BB=1. Jet entries are factorial divided.
        inv = 1.0 / shape if along else 0.0
        result[1] = jet[1] - inv * jet[2]
        result[2] = -jet[2]
        if order == 2:
            result[3] = 2 * jet[3] - 2 * inv * jet[4] + 2 * inv**2 * (jet[5] + jet[2])
            result[4] = -jet[4] + inv * (2 * jet[5] + jet[2])
            result[5] = 2 * jet[5] + jet[2]
    # Bound the transient memo: reuse is an optimization, never authority.
    if len(workspace.endpoints) < 50_000:
        workspace.endpoints[key] = result
    return result


def _as_jet(flat, order):
    return (flat[:, 0], flat[:, 1:3], flat[:, 3:6])[: order + 1]


@njit(cache=True, error_model="numpy")
def _log_window_core(near, far, order):
    """Packed two-parameter specialization of the log-tail difference.

    The Python caller retains the same cancellation admission and quadrature
    fallback. In particular a failed far tail stays NaN, never an absent end.
    """
    result = near.copy()
    delta = far[0] - near[0]
    kappa = 1.0 / np.expm1(-delta)
    if far[0] == -np.inf or abs(kappa) == 0.0:
        return result
    result[0] += np.log(-np.expm1(delta))
    if order:
        d0, d1 = far[1] - near[1], far[2] - near[2]
        result[1] += -kappa * d0
        result[2] += -kappa * d1
        if order == 2:
            outer = -kappa * (1.0 + kappa)
            result[3] += -kappa * (far[3] - near[3]) + outer * d0 * d0
            result[4] += -kappa * (far[4] - near[4]) + outer * d0 * d1
            result[5] += -kappa * (far[5] - near[5]) + outer * d1 * d1
    return result


@njit(cache=True, error_model="numpy")
def _quadrature_mix_add(state, term, order):
    """Two-score centered positive mixture, with the shared accumulator's scaling.

    Ten scalars hold top, mass, anchor, centered mean, packed Hessian and
    invalidity. Root scaling and promotion retain small weighted derivatives
    even when a component's mass underflows; no raw score squares are formed.
    """
    if term[0] == -np.inf or state[9]:
        return
    size = 1 if order == 0 else (3 if order == 1 else 6)
    for c in range(size):
        if not np.isfinite(term[c]):
            state[9] = 1.0
            return
    if state[1] == 0.0:
        state[0], state[1] = term[0], 1.0
        if order:
            state[2:4] = term[1:3]
        if order == 2:
            state[6:9] = term[3:6]
        return
    top = max(state[0], term[0])
    log_left = np.log(state[1]) + (state[0] - top)
    log_right = term[0] - top
    mass = state[1] * np.exp(state[0] - top) + np.exp(log_right)
    state[0], state[1] = top, mass
    if order == 0:
        return
    log_p, log_q = log_left - np.log(mass), log_right - np.log(mass)
    root_p, root_q = np.exp(0.5 * log_p), np.exp(0.5 * log_q)
    delta = np.empty(2)
    for i in range(2):
        centered = term[1 + i] - state[2 + i]
        delta[i] = centered - state[4 + i]
        left_centered = state[4 + i]
        if log_right > log_left:
            left_centered, centered = -delta[i], 0.0
            state[2 + i] = term[1 + i]
        state[4 + i] = (left_centered * root_p) * root_p + (centered * root_q) * root_q
    if order == 2:
        fourth_root = np.exp(0.25 * log_p + 0.25 * log_q)
        scaled = (delta * fourth_root) * fourth_root
        for c, (i, j) in enumerate(((0, 0), (0, 1), (1, 1))):
            state[6 + c] = (
                (state[6 + c] * root_p) * root_p
                + (term[3 + c] * root_q) * root_q
                + scaled[i] * scaled[j]
            )


# The generated callee digest invalidates this module's code cache when a
# shared primitive changes; Numba alone does not track imported callees.
@njit(cache=True, error_model="numpy")
def _quadrature_core(lower, upper, shape, log_scale, order):
    width = math.log1p((upper - lower) / lower)
    log_left = math.log(lower) - log_scale
    state = np.zeros(10)
    state[0] = -np.inf
    term = np.empty(6)
    psi = psires_0123(shape)[0] if order else 0.0
    trigamma = polygamma_0123(shape)[1] if order == 2 else 0.0
    norm = gnorm(shape)
    for i in range(5):
        log_x = log_left + 0.5 * (_NODES[i] + 1) * width
        x = math.exp(log_x)
        term[0] = norm - shape * gdev(x, shape) + math.log(_WEIGHTS[i] * width / 2)
        if order:
            term[1], term[2] = log_x - math.log(shape) - psi, x - shape
        if order == 2:
            term[3], term[4], term[5] = -trigamma, -1.0, -x
        _quadrature_mix_add(state, term, order)
    result = np.full(6, np.nan)
    if state[9]:
        return result
    result[0] = state[0] + np.log(state[1])
    if order and state[1] != 0.0:
        result[1:3] = state[2:4] + state[4:6]
    if order == 2 and state[1] != 0.0:
        result[3:6] = state[6:9]
    size = 1 if order == 0 else (3 if order == 1 else 6)
    if state[1] != 0.0 and not np.isfinite(result[:size]).all():
        result[:] = np.nan
    return result


def _quadrature(lower, upper, shape, log_scale, order, workspace):
    if workspace.quadrature_nodes + 5 > 5 * workspace.max_evaluations:
        workspace.refusal_reason = "aggregate gamma quadrature work budget"
        return _as_jet(np.full((1, 6), np.nan), order)
    flat = _quadrature_core(lower, upper, shape, log_scale, order)
    workspace.quadrature_nodes += 5
    return _as_jet(flat[None], order)


def gamma_shape_scale_window_jets(lower, upper, shape, log_scale, *, order, _workspace=None) -> Jet:
    """Unit log-window jets in (shape, log scale); unresolved rows are NaN.

    Orders are exactly integers 0, 1, 2. Empty windows have log mass -inf.
    ``_workspace`` is private, shared only by one compound-law evaluation.
    """
    order = _checked_core_order(order)
    lo, hi, aa, bb = np.broadcast_arrays(
        *map(lambda x: np.atleast_1d(np.asarray(x, dtype=float)), (lower, upper, shape, log_scale))
    )
    if lo.ndim != 1:
        raise ValueError("gamma window inputs must be vectors")
    work = GammaWindowWorkspace() if _workspace is None else _workspace
    result = np.full((len(lo), 6), np.nan)
    for row, (left, right, a, b) in enumerate(zip(lo, hi, aa, bb, strict=True)):
        if not (np.isfinite(a) and a > 0 and np.isfinite(b)) or np.isnan(left) or np.isnan(right):
            continue
        if right <= max(left, 0.0):
            result[row] = 0.0
            result[row, 0] = -np.inf
            continue
        if left <= 0 and right == np.inf:
            result[row] = 0.0
            continue
        if left <= 0:
            result[row] = _tail(right, a, b, False, order, work)
            continue
        if right == np.inf:
            result[row] = _tail(left, a, b, True, order, work)
            continue
        log_l, log_h = math.log(left) - b, math.log(right) - b
        if not (-708 < log_l <= log_h < 709):
            work.refusal_reason = "gamma window outside normal float64 argument range"
            continue
        xl, xh = math.exp(log_l), math.exp(log_h)
        ratio = (right - left) / left
        width = math.log1p(ratio) if np.isfinite(ratio) else math.log(right) - math.log(left)
        rate = max(1.0, abs(a - xl), abs(a - xh), math.sqrt(xh))
        if width * rate <= 0.05:
            jet = _quadrature(left, right, a, b, order, work)
        else:
            upper_tail = xl >= a
            near = _tail(left if upper_tail else right, a, b, upper_tail, order, work)
            far = _tail(right if upper_tail else left, a, b, upper_tail, order, work)
            if not np.isfinite(near[0]) or far[0] - near[0] > -0.01:
                # A positive finite window whose tails cannot be subtracted
                # accurately: subdivision on log response retains the same
                # density/derivative admission, with an explicit work limit.
                required = width * rate / 0.05
                if not required <= 128:
                    work.refusal_reason = "gamma window cancellation / quadrature work budget"
                    continue
                panels = math.ceil(required)
                from superglm.distributional.kernels._response_tails import (
                    CoreLogMixtureAccumulator,
                )

                accumulator = CoreLogMixtureAccumulator(2, order)
                for panel in range(panels):
                    ll = left if panel == 0 else left * math.exp(width * panel / panels)
                    hh = (
                        right
                        if panel + 1 == panels
                        else left * math.exp(width * (panel + 1) / panels)
                    )
                    accumulator.add(_quadrature(ll, hh, a, b, order, work))
                jet = accumulator.finish()
            else:
                jet = _as_jet(_log_window_core(near, far, order)[None], order)
        for degree, values in enumerate(jet):
            start, end = ((0, 1), (1, 3), (3, 6))[degree]
            result[row, start:end] = values.reshape(-1)
    return _as_jet(result, order)
