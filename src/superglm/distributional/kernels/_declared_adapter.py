"""Mechanical invocation of generated natural-coordinate scalar laws.

No family contracts or symbolic machinery belong at this numerical boundary.
"""

from dataclasses import dataclass
from math import comb
from operator import index
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import readonly
from superglm.distributional.kernels._declared import _column, law_jets
from superglm.distributional.kernels._declared_bounds import declared_window_jets
from superglm.distributional.kernels._response_tails import NARROW_WINDOW, quadrature_window_jets


@dataclass(frozen=True)
class ScalarJetResult:
    jets: tuple[NDArray[np.float64], ...]
    carrier: NDArray[np.float64]
    valid: NDArray[np.bool_]


def jet_order(value: object) -> int:
    message = "jet order must be an integer from zero through four"
    if isinstance(value, bool | np.bool_):
        raise ValueError(message)
    try:
        order = index(value)
    except TypeError as exc:
        raise ValueError(message) from exc
    if not 0 <= order <= 4:
        raise ValueError(message)
    return order


def generated_density(module: ModuleType, y, theta, multiplier, *, order: int) -> ScalarJetResult:
    """Call one generated law, including its separately multiplied carrier.

    There is no inferred array fallback. Nonrepresentable channels mark their
    row invalid, while law-specific range/accuracy admission belongs upstream.
    """
    order = jet_order(order)
    if module.COORDINATES != "natural" or module.DATA or "density" not in module.LAWS:
        raise ValueError("scalar adapter requires a natural-coordinate density without row data")
    y = np.asarray(y, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if y.ndim != 1 or theta.shape != (len(y), len(module.PARAMETERS)):
        raise ValueError("density response and natural parameter rows must agree")
    if multiplier.shape != y.shape or np.any(~np.isfinite(multiplier) | (multiplier <= 0)):
        raise ValueError("density multipliers must be finite positive row masses")
    jets, valid = law_jets(module, "density", order, y, theta, multiplier=multiplier)
    carrier = np.empty(len(y))
    module.carrier(_column(y, len(y)), _column(multiplier, len(y)), carrier)
    return ScalarJetResult(jets, carrier, valid & np.isfinite(carrier))


def narrow_windows(lower, upper, rate):
    """The same log-response reach used by the shared quadrature helper."""
    lo, hi = np.asarray(lower), np.asarray(upper)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        relative = (hi - lo) / lo
        width = np.where(
            relative <= 1.0, np.log1p(np.minimum(relative, 1.0)), np.log(hi) - np.log(lo)
        )
    return (lo > 0) & np.isfinite(hi) & (width * rate <= NARROW_WINDOW)


def positive_window_jets(
    module, lower, upper, theta, *, order, rate, linear_narrow=None, density=None
):
    """Shared support endpoints and unit-mass declared-window composition.

    ``linear_narrow`` is a law-owned, justified admission mask for quadrature
    in y (needed when the density is regular at zero). It never creates a
    carrier or frequency multiplier on a probability window.
    """
    order = jet_order(order)
    theta = np.asarray(theta, dtype=np.float64)
    n, k = theta.shape
    lo, hi = np.broadcast_to(lower, (n,)).astype(float), np.broadcast_to(upper, (n,)).astype(float)
    if np.any(np.isnan(lo) | np.isnan(hi)) or np.any(lo > hi):
        raise ValueError("window endpoints require lower <= upper and no NaN")
    lo = np.maximum(lo, 0.0)
    empty = lo >= hi
    result = (np.where(empty, -np.inf, 0.0),) + tuple(
        np.zeros((n, comb(k + d - 1, d))) for d in range(1, order + 1)
    )
    active = ~empty & ((lo > 0) | np.isfinite(hi))
    linear = np.zeros(n, bool) if linear_narrow is None else active & linear_narrow
    rows = np.flatnonzero(active & ~linear)
    if len(rows):
        sub = declared_window_jets(
            module,
            lo[rows],
            hi[rows],
            theta[rows],
            (),
            order=order,
            from_lower=np.zeros(len(rows), bool),
            rate=np.asarray(rate)[rows],
        )
        for target, source in zip(result, sub, strict=True):
            target[rows] = source
    rows = np.flatnonzero(linear)
    if len(rows):
        span = hi[rows] - lo[rows]

        def nodes(fraction):
            evaluated = density(
                lo[rows] + fraction * span, theta[rows], np.ones(len(rows)), order=order
            )
            return (evaluated.jets[0] + evaluated.carrier, *evaluated.jets[1:])

        sub = quadrature_window_jets(0.5 * span, nodes, order=order)
        for target, source in zip(result, sub, strict=True):
            target[rows] = source
    valid = ~empty & np.all(np.isfinite(theta) & (theta > 0), axis=1)
    if order:
        valid &= np.all((theta >= 2.0**-200) & (theta <= 2.0**200), axis=1)
    for values in result:
        valid &= np.isfinite(values) if values.ndim == 1 else np.all(np.isfinite(values), axis=1)
    return result, valid


def positive_parameters(theta):
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError("scale and shape must be finite positive natural parameters")
    return values


def probabilities(p, n):
    p = np.broadcast_to(np.asarray(p, dtype=np.float64), (n,))
    if np.any(~np.isfinite(p) | (p <= 0) | (p >= 1)):
        raise ValueError("probabilities must lie strictly inside (0, 1)")
    return p


def log_response_ratio(y, scale):
    """Centred log(y/scale), preserving near-median response resolution."""
    y, scale = np.broadcast_arrays(y, scale)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        near = (y >= 0.5 * scale) & ((y - scale) <= scale)
        value = np.log(y) - np.log(scale)
        value[near] = np.log1p((y[near] - scale[near]) / scale[near])
    return value


def finite_positive_exp(log_value, quantity):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.exp(log_value)
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError(f"finite {quantity} is unresolved in the numerical range")
    return readonly(values)


def log_expm1(t):
    t = np.asarray(t)
    result = np.empty_like(t)
    small = t <= np.log(2.0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        result[small] = np.log(np.expm1(t[small]))
        result[~small] = t[~small] + np.log1p(-np.exp(-t[~small]))
    return result
