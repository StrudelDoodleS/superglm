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

from superglm.distributional.kernels._incomplete_gamma import (
    MAX_TERMS,
    WORK_SIZE,
    log_gamma_tail_jet_into,
)
from superglm.distributional.kernels._response_tails import (
    Jet,
    _checked_core_order,
    log_mixture_jets,
    log_window_jets,
)
from superglm.distributional.kernels._severity_primitives import (
    gdev,
    gnorm,
    polygamma_0123,
    psires_0123,
)

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(5)
_U = 2.0**-53


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


def _quadrature(lower, upper, shape, log_scale, order, workspace):
    if workspace.quadrature_nodes + 5 > 5 * workspace.max_evaluations:
        workspace.refusal_reason = "aggregate gamma quadrature work budget"
        return _as_jet(np.full((1, 6), np.nan), order)
    width = math.log1p((upper - lower) / lower)
    log_left = math.log(lower) - log_scale
    logs = np.empty((1, 5))
    scores = np.empty((1, 5, 2))
    hessians = np.empty((1, 5, 3))
    psi = psires_0123(shape)[0]
    trigamma = polygamma_0123(shape)[1]
    for i, (node, weight) in enumerate(zip(_NODES, _WEIGHTS, strict=True)):
        log_x = log_left + 0.5 * (node + 1) * width
        x = math.exp(log_x)
        logs[0, i] = gnorm(shape) - shape * gdev(x, shape) + math.log(weight * width / 2)
        scores[0, i] = (log_x - math.log(shape) - psi, x - shape)
        hessians[0, i] = (-trigamma, -1.0, -x)
    workspace.quadrature_nodes += 5
    return log_mixture_jets(logs, scores, hessians, order=order)


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
                jet = log_window_jets(
                    _as_jet(near[None], order), _as_jet(far[None], order), order=order
                )
        for degree, values in enumerate(jet):
            start, end = ((0, 1), (1, 3), (3, 6))[degree]
            result[row, start:end] = values.reshape(-1)
    return _as_jet(result, order)
