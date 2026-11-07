"""A differential sweep of the compiled row routes near the edges of their input ranges.

A compiled route and its array route evaluate one law by different formulas, and they part where a
formula cancels: forward error of a closed form near a singular or cancelling limit, which grows
with the conditioning of the *formula* and not of the problem (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., sections 1.7 and 1.14).  Nothing non-finite marks such a row, so
the sweep compares entries.

Each :class:`Route` registers how to call it, the units that make its entries dimensionless, and
its grid: products of geometric walks toward every boundary of its declared domain and every limit
its formulas treat specially.  Entries are compared at orders 0 to 2, in the route's units,
relative to the row's largest entry at that order.  A row where the routes differ by more than
its bar is settled against mpmath: the compiled error may be at most ``FACTOR`` times the array
route's error, or ``BAR`` times the row's conditioning.  The conditioning is ``1`` except where a
route registers the limit its law cannot evaluate more accurately than: there both routes carry
rounding of that size, and which is the smaller is a matter of the platform's libm.  Routes with no
array route to differ from are held against mpmath alone.

Reference laws are written here from the published densities, with the oracle modules' loadings.
Each is differentiated by central differences at four times the working precision, with step
``2^-bits`` on the scaled parameter, so its truncation and rounding are both about ``2^-2bits``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace
from unittest import mock

import mpmath as mp
import numpy as np

from superglm.distributional.families.tweedie import BoundedPowerLink, TweedieLSS
from superglm.distributional.family import (
    NaturalLikelihoodEvaluation,
    validated_parameter_matrix,
)
from superglm.distributional.kernels import gamma as gamma_kernel
from superglm.distributional.kernels import generalized_gamma as gg_kernel
from superglm.distributional.kernels import generalized_pareto as pareto_kernel
from superglm.distributional.kernels import log_normal as log_normal_kernel
from superglm.distributional.kernels import negative_binomial as nb_kernel
from superglm.distributional.kernels import tweedie as tweedie_kernel
from superglm.distributional.kernels import two_piece as two_piece_kernel
from superglm.distributional.kernels._declared import gamma as declared_gamma
from superglm.distributional.kernels._declared import generalized_pareto as declared_pareto
from superglm.distributional.kernels._declared import log_normal_mean as declared_log_normal
from superglm.distributional.kernels._declared import two_piece_location as declared_location
from superglm.distributional.kernels._declared import two_piece_mean as declared_mean
from superglm.distributional.solver.derivatives import transform_natural_derivatives
from superglm.links import LogLink
from tests import _two_piece_lss_oracles as two_piece_oracle

U = 2.0**-53
# The generated kernels' own gate (``tests/test_declared_severity_families.py``).
BAR = 2048.0 * U
FACTOR = 8.0
# Where the two pieces of a two-piece row are both within rounding of the join.
JOIN_TIE = 1.0e-13
# Tweedie's series stops ``37`` log units below its mode, a point truncation its moments amplify.
TWEEDIE_BAR = 1.0e-9


@dataclass(frozen=True)
class Jet:
    """Value, score and packed Hessian per row, and which rows the route calls valid."""

    value: np.ndarray
    score: np.ndarray
    hessian: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class Route:
    """One compiled route: its columns are ``(response, *parameters, weight)``."""

    name: str
    call: Callable[[tuple], object]
    gate: tuple[ModuleType, str, ModuleType | None] | None
    units: Callable[[tuple], tuple]
    law: Callable[[tuple], Callable]
    grid: Callable[[bool], tuple]
    array: Callable[[tuple], object] | None = None
    bits: Callable[[tuple], int] | None = None
    conditioning: Callable[[tuple], np.ndarray] = lambda columns: np.ones_like(columns[0])
    curvature: Callable[[tuple], np.ndarray] | None = None
    bar: float = BAR


def _jet(evaluated) -> Jet:
    value = getattr(evaluated, "optimizing_log_likelihood", None)
    value = evaluated.log_likelihood if value is None else value
    return Jet(
        np.asarray(value), np.asarray(evaluated.score), np.asarray(evaluated.hessian_packed),
        np.asarray(evaluated.valid),
    )  # fmt: skip


def _empty_like(columns: tuple, route_width: int, packed: int) -> Jet:
    n = len(columns[0])
    return Jet(
        np.full(n, np.nan), np.full((n, route_width), np.nan), np.full((n, packed), np.nan),
        np.zeros(n, dtype=bool),
    )  # fmt: skip


class _DeclinedError(Exception):
    """A compiled route declined its chunk, so its array route need not run for it."""


def _bisected(call: Callable[[tuple], Jet | None], columns: tuple, width: int, packed: int):
    """``call`` over rows, split until each call holds rows it serves.

    ``None`` or a refusal marks the rows: a call that declines one row declines its chunk, so a row
    reads what it reads in a chunk of rows its route takes.
    """
    n = len(columns[0])
    if n == 0:
        return _empty_like(columns, width, packed), np.zeros(0, dtype=bool)
    try:
        jet = call(columns)
    except (_DeclinedError, ValueError, ArithmeticError, tweedie_kernel.TweedieNumericalRefusal):
        jet = None
    if jet is not None:
        return jet, np.ones(n, dtype=bool)
    if n == 1:
        return _empty_like(columns, width, packed), np.zeros(1, dtype=bool)
    half = n // 2
    left, served_left = _bisected(call, tuple(c[:half] for c in columns), width, packed)
    right, served_right = _bisected(call, tuple(c[half:] for c in columns), width, packed)
    jet = Jet(*(np.concatenate(pair) for pair in zip(_fields(left), _fields(right), strict=True)))
    return jet, np.concatenate((served_left, served_right))


def _fields(jet: Jet) -> tuple:
    return jet.value, jet.score, jet.hessian, jet.valid


def _decline(*args, **kwargs):
    return None


def compiled_rows(route: Route, columns: tuple, width: int, packed: int) -> tuple[Jet, np.ndarray]:
    """The compiled route's rows, and which of them it served; a served row is compiled."""
    if route.gate is None:
        return _bisected(lambda c: _jet(route.call(c)), columns, width, packed)
    module, name, kernel = route.gate
    original = getattr(module, name)

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        # A mean form's array route calls the location kernel: that is not this route's kernel.
        if result is None and (kernel is None or args[0] is kernel):
            raise _DeclinedError
        return result

    def call(chunk):
        with mock.patch.object(module, name, recorded):
            return _jet(route.call(chunk))

    return _bisected(call, columns, width, packed)


def array_rows(route: Route, columns: tuple, width: int, packed: int) -> tuple[Jet, np.ndarray]:
    """The array route's rows, and which of them it did not refuse."""
    if route.array is not None:
        return _bisected(lambda c: _jet(route.array(c)), columns, width, packed)
    if route.gate is None:
        return _empty_like(columns, width, packed), np.zeros(len(columns[0]), dtype=bool)
    module, name, _ = route.gate

    def call(chunk):
        with mock.patch.object(module, name, _decline):
            return _jet(route.call(chunk))

    return _bisected(call, columns, width, packed)


# ---------------------------------------------------------------- the references


def _default_bits(point: tuple) -> int:
    largest = max((abs(math.frexp(float(v))[1]) for v in point if v != 0.0), default=0)
    return 256 + 2 * largest


def reference_jet(route: Route, row: tuple, units: tuple) -> tuple[float, list, list]:
    """``(value, scaled score, scaled packed Hessian)`` of ``route``'s law at ``row``."""
    bits = (route.bits or _default_bits)(row)
    law = route.law(row)
    k = len(units)
    with mp.workprec(4 * bits):
        h = mp.ldexp(mp.mpf(1), -bits)
        start = [mp.mpf(float(v)) for v in row[1 : k + 1]]
        scaled = [mp.mpf(float(v)) for v in units]

        def at(*steps):
            return law(*(p + d * unit for p, d, unit in zip(start, steps, scaled, strict=True)))

        def shift(*pairs):
            steps = [mp.mpf(0)] * k
            for index, sign in pairs:
                steps[index] += sign * h
            return at(*steps)

        centre = at(*([mp.mpf(0)] * k))
        score = [(shift((i, 1)) - shift((i, -1))) / (2 * h) for i in range(k)]
        hessian = []
        for i, j in itertools.combinations_with_replacement(range(k), 2):
            if i == j:
                hessian.append((shift((i, 1)) - 2 * centre + shift((i, -1))) / (h * h))
            else:
                hessian.append(
                    (
                        shift((i, 1), (j, 1))
                        - shift((i, 1), (j, -1))
                        - shift((i, -1), (j, 1))
                        + shift((i, -1), (j, -1))
                    )
                    / (4 * h * h)
                )
        return float(centre), [float(v) for v in score], [float(v) for v in hessian]


def _gamma_law(prior: bool):
    def bound(row):
        y, _, _, weight = (mp.mpf(float(v)) for v in row)

        def law(mean, cv):
            shape = weight / (cv * cv) if prior else 1 / (cv * cv)
            ratio = y / mean
            value = shape * mp.log(shape) - shape - mp.loggamma(shape)
            value -= shape * (ratio - 1 - mp.log(ratio))
            return value if prior else weight * value

        return law

    return bound


def _pareto_law(row):
    y, _, _, weight = (mp.mpf(float(v)) for v in row)

    def law(psi, xi):
        t = y / psi
        z = xi * t
        l1 = mp.log1p(z) / z if z != 0 else mp.mpf(1)
        return weight * (-mp.log(psi) - t * l1 - mp.log1p(z))

    return law


def _log_normal_law(row):
    y, _, _, weight = (mp.mpf(float(v)) for v in row)

    def law(mean, sigma):
        z = (mp.log(y / mean) + sigma * sigma / 2) / sigma
        return weight * (-mp.log(sigma) - z * z / 2)

    return law


def _two_piece_location_law(row, right: bool | None = None):
    y, mu0, sigma0, _, weight = (mp.mpf(float(v)) for v in row)
    piece = (1 if y >= mu0 else -1) if right is None else (1 if right else -1)

    def law(mu, sigma, eps):
        u = (y - mu) / sigma / (1 + piece * eps)
        return weight * (-mp.log(sigma) - u * u / 2)

    return law


def _two_piece_mean_law(row, right: bool | None = None):
    y, mean0, sigma0, eps0, weight = (mp.mpf(float(v)) for v in row)
    if right is None:
        right = mp.log(y / mean0) + two_piece_oracle.mp_log_mean_loading(sigma0, eps0) >= 0
    piece = 1 if right else -1

    def law(mean, sigma, eps):
        z = (mp.log(y / mean) + two_piece_oracle.mp_log_mean_loading(sigma, eps)) / sigma
        u = z / (1 + piece * eps)
        return weight * (-mp.log(sigma) - u * u / 2)

    return law


def _nb_law(prior: bool):
    def bound(row):
        count, _, _, weight = (mp.mpf(float(v)) for v in row)
        scale = weight if prior else mp.mpf(1)
        mass = mp.mpf(1) if prior else weight

        def law(mean, theta):
            mu, size = scale * mean, scale * theta
            return mass * (
                mp.loggamma(count + size)
                - mp.loggamma(size)
                - count * mp.log(mu + size)
                + count * mp.log(mu)
                - size * mp.log1p(mu / size)
            )

        return law

    return bound


def _gg_log_location_density(y, mu, sigma, q):
    """Prentice's density without the carrier ``-log y - log(2 pi)/2``; ``q`` is kept off zero."""
    q = q if abs(q) >= mp.ldexp(mp.mpf(1), -100) else mp.ldexp(mp.mpf(1), -100)
    k = 1 / (q * q)
    w = (mp.log(y) - mu) / sigma
    u = q * w
    return (
        mp.log(abs(q))
        + k * mp.log(k)
        - mp.loggamma(k)
        - mp.log(sigma)
        + k * (u - mp.exp(u))
        + mp.log(2 * mp.pi) / 2
    )


def _gg_location_law(row):
    y, _, _, _, weight = (mp.mpf(float(v)) for v in row)
    return lambda mu, sigma, q: weight * _gg_log_location_density(y, mu, sigma, q)


def _gg_mean_law(row):
    y, _, _, _, weight = (mp.mpf(float(v)) for v in row)

    def law(mean, sigma, q):
        q = q if abs(q) >= mp.ldexp(mp.mpf(1), -100) else mp.ldexp(mp.mpf(1), -100)
        k = 1 / (q * q)
        loading = (sigma / q) * mp.log(q * q) + mp.loggamma(k + sigma / q) - mp.loggamma(k)
        return weight * _gg_log_location_density(y, mp.log(mean) - loading, sigma, q)

    return law


def _gg_bits(row) -> int:
    """``k (u - e^u)`` cancels ``k = 1/q^2`` down to about one."""
    q = abs(float(row[3]))
    return _default_bits(row) + 2 * (abs(math.frexp(q)[1]) if q >= 2.0**-100 else 100)


def _tweedie_law(prior: bool):
    def bound(row):
        y, _, _, _, weight = (mp.mpf(float(v)) for v in row)
        scale = weight if prior else mp.mpf(1)
        mass = mp.mpf(1) if prior else weight

        def law(mean, phi0, p):
            phi = phi0 / scale
            lam = mean ** (2 - p) / (phi * (2 - p))
            if y == 0:
                return -mass * lam
            alpha = (2 - p) / (p - 1)
            spread = phi * (p - 1) * mean ** (p - 1)
            log_lam, log_y, log_spread = mp.log(lam), mp.log(y), mp.log(spread)

            def term(j):
                return (
                    j * log_lam
                    - mp.loggamma(j + 1)
                    + (j * alpha - 1) * log_y
                    - j * alpha * log_spread
                    - mp.loggamma(j * alpha)
                )

            start = max(1, int(mp.floor(mp.power(y, 2 - p) / (phi * (2 - p)))))
            terms = {start: term(start)}
            peak = terms[start]
            for direction in (1, -1):
                j = start + direction
                while j >= 1:
                    current = term(j)
                    terms[j] = current
                    peak = max(peak, current)
                    if current < peak - 80:
                        break
                    j += direction
            total = mp.fsum(mp.exp(value - peak) for value in terms.values())
            return mass * (-lam - y / spread + peak + mp.log(total))

        return law

    return bound


_POWER_LINK = BoundedPowerLink()
_PREDICTOR_LINKS = (LogLink(), LogLink(), _POWER_LINK)


def _tweedie_predictor_law(prior: bool):
    """``_tweedie_law`` on the predictors: log mean, log dispersion and logit power."""
    natural = _tweedie_law(prior)

    def bound(row):
        law = natural(row)
        lower, span = mp.mpf(_POWER_LINK.lower), mp.mpf(_POWER_LINK.upper - _POWER_LINK.lower)

        def at(eta_mean, eta_phi, eta_power):
            share = mp.exp(eta_power) / (1 + mp.exp(eta_power))
            return law(mp.exp(eta_mean), mp.exp(eta_phi), lower + span * share)

        return at

    return bound


# ---------------------------------------------------------------- the grids


def walk(low: float, high: float, large: bool, step: float = 1.0) -> list[float]:
    """Powers of ten from ``low`` to ``high``; the large search walks ten times as finely."""
    count = int(round((math.log10(high) - math.log10(low)) / step * (10 if large else 1)))
    return [float(v) for v in np.geomspace(low, high, count + 1)]


# The large search draws this many rows of a product it cannot hold.
LARGE_ROWS = 2_000_000


def _product(*axes) -> tuple:
    """Every combination of ``axes``, or a fixed random draw of them where there are too many."""
    arrays = [np.array(axis, dtype=np.float64) for axis in axes]
    if math.prod(len(axis) for axis in arrays) <= LARGE_ROWS:
        return tuple(column.ravel() for column in np.meshgrid(*arrays, indexing="ij"))
    rng = np.random.default_rng(20261106)
    return tuple(axis[rng.integers(len(axis), size=LARGE_ROWS)] for axis in arrays)


# The gate's own limits, reached by the large search.
_GATE = (2.0**-200, 2.0**200)
_SKEW_GAPS = (0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
# ``gdev`` and ``logratio`` switch at a ratio of 1/2 and 2, and tie at 1.
_RATIOS = (1e-30, 1e-8, 1e-3, 0.5 * (1.0 - 1e-12), 0.5, 0.5 * (1.0 + 1e-12), 1.0 - 1e-12, 1.0,
           1.0 + 1e-12, 2.0 * (1.0 - 1e-12), 2.0, 2.0 * (1.0 + 1e-12), 1e3, 1e8, 1e30)  # fmt: skip


def _positive_axis(large: bool, *, low: float = 1e-30, high: float = 1e30, step: float = 10.0):
    return [*walk(low, high, large, step), *(_GATE if large else ())]


def _weights(large: bool) -> list[float]:
    return _positive_axis(large, step=30.0)


# Frequency weights are exact positive integers.
_FREQUENCY = (1.0, 3.0, 1e6, 1e15, 2.0**53)


def _frequency(grid: Callable[[bool], tuple]) -> Callable[[bool], tuple]:
    """``grid`` with its weights walked over exact integers."""

    def walked(large: bool) -> tuple:
        columns = grid(large)
        n = len(columns[0])
        repeats = np.tile(np.arange(n), len(_FREQUENCY))
        weights = np.repeat(np.array(_FREQUENCY), n)
        keep = columns[-1][repeats] == columns[-1][0]
        return (*(c[repeats][keep] for c in columns[:-1]), weights[keep])

    return walked


def _gamma_grid(large: bool) -> tuple:
    ratio = np.array(_RATIOS)
    mean = _positive_axis(large, step=30.0)
    # A unit weight's shape crosses the switches of ``sres`` (1/4, 10) and ``polygamma`` (16).
    switches = [math.sqrt(1.0 / a) for a in (0.25, 10.0, 16.0)]
    cv = [*_positive_axis(large, step=5.0), *switches, *(v * (1.0 + 1e-12) for v in switches)]
    y, m, c, w = _product(ratio, mean, cv, _weights(large))
    keep = (y * m > 0.0) & np.isfinite(y * m)
    return y[keep] * m[keep], m[keep], c[keep], w[keep]


def _pareto_grid(large: bool) -> tuple:
    shapes = (-2.0, -1.5, -1.0, -1.0 + 1e-9, -1.0 + 1e-3, -0.9, -0.5, -0.25, -0.1, -1e-3, -1e-8, -1e-12,
              -1e-200, 0.0, 1e-200, 1e-12, 1e-8, 1e-3, 0.1, 0.25, 0.5, 0.9, 0.999, 1.0, 2.0, 1e3)  # fmt: skip
    steps = (0.0, 1e-30, 1e-8, 1e-3, 0.5, 1.0 - 1e-12, 1.0, 1.0 + 1e-12, 2.0, 10.0, 1e3, 1e8, 1e30)
    gaps = (0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
    psi = _positive_axis(large, step=30.0)
    rows = []
    for shape in shapes:
        t = [v for v in steps if 1.0 + shape * v > 0.0]
        if shape < 0.0:
            t += [(1.0 - gap) / -shape for gap in gaps]
        if abs(shape) > 1e-30:
            # The series of ``z`` ends at ``|z| = 1/2``, and the tail's at ``z = 4``.
            t += [v / abs(shape) for v in (0.5 * (1.0 - 1e-12), 0.5, 0.5 * (1.0 + 1e-12))]
            t += [v / shape for v in (4.0 * (1.0 - 1e-12), 4.0) if shape > 0.0]
        rows.append(_product(t, [shape], psi, _weights(large)))
    t, xi, p, w = (np.concatenate(c) for c in zip(*rows, strict=True))
    y = t * p
    keep = np.isfinite(y) & (1.0 + xi * (y / p) > 0.0)
    return y[keep], p[keep], xi[keep], w[keep]


def _log_normal_grid(large: bool) -> tuple:
    sigma = _positive_axis(large, low=1e-30, high=1e30, step=5.0)
    mean = _positive_axis(large, step=30.0)
    logs = (-1e3, -30.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 30.0, 1e3)
    offsets = (0.0, -0.5)
    rows = []
    for s in sigma:
        for shift in offsets:
            for log_ratio in logs:
                value = log_ratio + shift * s * s
                if abs(value) < 300.0:
                    rows.append((math.exp(value), s))
    y0, s0 = np.array(rows).T
    index = np.arange(len(y0))
    i, m, w = _product(index, mean, _weights(large))
    i = i.astype(int)
    return y0[i] * m, m, s0[i], w


def _skews(large: bool) -> list[float]:
    gaps = walk(1e-14, 0.5, True, 1.0) if large else _SKEW_GAPS
    return [0.0, 1e-8, -1e-8, *(1.0 - gap for gap in gaps), *(gap - 1.0 for gap in gaps)]


def _two_piece_location_grid(large: bool) -> tuple:
    skew = _skews(large)
    sigma = _positive_axis(large, low=1e-30, high=1e30, step=10.0)
    z = (0.0, 1e-300, -1e-300, 1e-8, -1e-8, 1.0, -1.0, 1e3, -1e3, 1e8, -1e8, 1e30, -1e30)
    location = (0.0, 1.0, -1.0, 1e6, -1e30)
    zs, mu, s, e, w = _product(z, location, sigma, skew, _weights(large))
    y = mu + zs * s
    keep = np.isfinite(y)
    return y[keep], mu[keep], s[keep], e[keep], w[keep]


def _two_piece_mean_grid(large: bool) -> tuple:
    """Skew to both ends and the scale both ways, ties at the mean and at the join ``y = m/K``."""
    skew = _skews(large)
    sigma = (
        1e-30,
        1e-18,
        1e-8,
        1e-4 * (1.0 - 1e-12),
        1e-4,
        1e-3,
        0.01,
        0.1,
        0.5,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )
    mean = (1e-30, 1.0, 1e30)
    ratio = (1e-30, 1e-8, 0.5, 1.0, 2.0, 1e8, 1e30)
    r, m, s, e, w = _product(ratio, mean, sigma, skew, _weights(large))
    # ``log Phi`` switches at -1 and -3.
    s2, e2 = _product(sigma, skew)
    switches = [(1.0 / (1.0 - e2)), (3.0 / (1.0 - e2)), (1.0 / (1.0 + e2)), (3.0 / (1.0 + e2))]
    s2 = np.concatenate((s2, *switches))
    e2 = np.tile(e2, 5)
    representable = s2 <= 1e3
    s2, e2 = s2[representable], e2[representable]
    with np.errstate(over="ignore"):
        at_join = np.exp(-two_piece_kernel.log_mean_loading(s2, e2)[0])
    keep = np.isfinite(at_join) & (at_join > 0.0)
    s2, e2, at_join = s2[keep], e2[keep], at_join[keep]
    near = (1.0, 1.0 - 1e-14, 1.0 + 1e-14, 1.0 - 1e-8, 1.0 + 1e-8)
    r2 = np.concatenate([at_join * v for v in near])
    s2, e2 = np.tile(s2, len(near)), np.tile(e2, len(near))
    r, m, s, e, w = (
        np.concatenate((r, r2)),
        np.concatenate((m, np.ones_like(r2))),
        np.concatenate((s, s2)),
        np.concatenate((e, e2)),
        np.concatenate((w, np.ones_like(r2))),
    )
    y = r * m
    keep = np.isfinite(y) & (y > 0.0)
    return y[keep], m[keep], s[keep], e[keep], w[keep]


def _nb_grid(large: bool) -> tuple:
    """Counts into the recurrence's blocks, with sizes to both ends and across its two limits.

    The theta series starts at ``reach/theta = 1/16`` and the Poisson limit at ``2^-52``, for
    ``reach = max(mean, count - 1)``; a mean ties with its count.
    """
    counts = (0.0, 1.0, 2.0, 3.0, 10.0, 100.0, 255.0, 256.0, 257.0, 513.0, 1e3, 1e4, 1e5 if large else 1e4)  # fmt: skip
    mean = _positive_axis(large, low=1e-30, high=1e30, step=10.0)
    theta = [*walk(1e-30, 1e30, large, 5.0), *(1e100, 1e300)]
    c, m, t, w = _product(sorted(set(counts)), mean, theta, _weights(large))
    c2, ties = _product([v for v in counts if v > 1.0], (0.5, 1.0, 1.0 + 1e-12, 2.0))
    m2 = c2 * ties
    reach = np.maximum(m2, c2 - 1.0)
    limits = [
        reach * ratio * nudge
        for ratio in (16.0, 2.0**52)
        for nudge in (1.0 - 1e-12, 1.0, 1.0 + 1e-12)
    ]
    t2 = np.concatenate((m2, *limits))
    c2, m2 = np.tile(c2, len(limits) + 1), np.tile(m2, len(limits) + 1)
    return (
        np.concatenate((c, c2)),
        np.concatenate((m, m2)),
        np.concatenate((t, t2)),
        np.concatenate((w, np.ones_like(t2))),
    )


def _gg_shapes() -> tuple:
    """Through zero and to both extremes, and either side of the shape term's ``k = 30`` switch."""
    switch = math.sqrt(1.0 / 30.0)
    return (0.0, 1e-200, 1e-30, 1e-12, 1e-6, 1e-3, 0.1, switch, switch * (1.0 + 1e-12), 0.5, 1.0, 3.0, 10.0, 30.0,
            -1e-200, -1e-30, -1e-12, -1e-6, -1e-3, -0.1, -switch, -0.5, -1.0, -3.0, -10.0, -30.0)  # fmt: skip


def _gg_location_grid(large: bool) -> tuple:
    w = (0.0, 1e-8, -1e-8, 1.0, -1.0, 10.0, -10.0, 1e3, -1e3)
    sigma = _positive_axis(large, low=1e-8, high=1e3, step=2.0)
    q = _gg_shapes()
    location = (0.0, 5.0, -5.0)
    a, s, shape, mu, m = _product(w, sigma, q, location, (1e-30, 1.0, 1e30))
    # The series in ``u = Q w`` ends at ``|u| = 1``.
    s2, q2 = _product((1e-3, 1.0, 3.0), [v for v in q if abs(v) >= 1e-3])
    for u in (1.0 - 1e-12, 1.0, 1.0 + 1e-12, -1.0 + 1e-12, -1.0, -1.0 - 1e-12):
        a, s, shape = np.append(a, u / q2), np.append(s, s2), np.append(shape, q2)
        mu, m = np.append(mu, np.zeros_like(q2)), np.append(m, np.ones_like(q2))
    log_y = mu + a * s
    keep = np.abs(log_y) < 690.0
    return np.exp(log_y[keep]), mu[keep], s[keep], shape[keep], m[keep]


def _gg_mean_grid(large: bool) -> tuple:
    """The scale toward the wall ``sigma |Q| = 1`` where the mean stops existing, and ``w = 0``."""
    q = _gg_shapes()
    edge = [(1.0 - gap) / abs(v) for v in q if v < -1e-3 for gap in (0.5, 1e-3, 1e-8, 1e-12)]
    # The loading's series in ``v = sigma Q`` end at ``|v| = 1/4`` and ``4/5``.
    edge += [
        c / abs(v)
        for v in (0.5, 1.0, 3.0)
        for c in (0.25, 0.25 * (1.0 + 1e-12), 0.8, 0.8 * (1.0 + 1e-12))
    ]
    sigma = sorted({*_positive_axis(large, low=1e-8, high=1e3, step=2.0), *edge})
    r, m, s, shape, w = _product(
        (1e-30, 1e-3, 1.0, 2.0, 1e3, 1e30), (1e-30, 1.0, 1e30), sigma, q, (1.0, 1e30)
    )
    s2, q2 = _product(sigma, q)
    exists = gg_kernel.mean_exists(s2, q2)
    s2, q2 = s2[exists], q2[exists]
    with np.errstate(all="ignore"):
        at_location = np.array([_location_of_unit_mean(a, b) for a, b in zip(s2, q2, strict=True)])
    keep = np.isfinite(at_location) & (at_location > 0.0)
    s2, q2, r2 = s2[keep], q2[keep], at_location[keep]
    return (
        np.concatenate((r * m, r2)),
        np.concatenate((m, np.ones_like(r2))),
        np.concatenate((s, s2)),
        np.concatenate((shape, q2)),
        np.concatenate((w, np.ones_like(r2))),
    )


def _location_of_unit_mean(sigma: float, q: float) -> float:
    """``y`` at ``w = 0`` for a unit mean; NaN where the loading is refused."""
    try:
        location = gg_kernel.location_of_mean(np.ones(1), np.array([sigma]), np.array([q]))
    except gg_kernel.GeneralizedGammaDomainError:
        return math.nan
    return float(np.exp(location[0]))


def _tweedie_grid(large: bool) -> tuple:
    power = (1.05, 1.05 + 1e-12, 1.1, 1.5, 1.9, 1.95 - 1e-12, 1.95)
    mean = (1e-6, 1e-2, 1.0, 1e2, 1e6)
    phi = (1e-2, 0.1, 1.0, 10.0, 1e2)
    y = (0.0, 1e-8, 1e-2, 1.0, 1e2, 1e6)
    weight = (1.0, 1e-3, 1e3) if large else (1.0, 1e3)
    y, m, d, p, w = _product(y, mean, phi, power, weight)
    # The reference sums the series by hand: rows of at most a few hundred terms about its mode.
    mode = y ** (2.0 - p) / (d / w * (2.0 - p))
    keep = mode <= 200.0
    return y[keep], m[keep], d[keep], p[keep], w[keep]


def _tweedie_predictor_grid(large: bool) -> tuple:
    """Zero rows out to the compiled window on every predictor and weight, and series rows.

    Beyond the walk, zero rows where ``eta_mean = 1 / s``: the power's score there is a difference
    of two terms the natural route rounds separately.
    """
    power = (-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0)
    near = (-1e-8, 0.0, 1e-8)
    reach = (1.0, 10.0, 50.0, 100.0)
    if large:
        power = (-36.0, -30.0, -20.0, -10.0, -5.0, -3.0, -1.0, -1e-8, 0.0, 1e-8, 1.0, 3.0, 5.0,
                 10.0, 20.0, 30.0, 36.0)  # fmt: skip
        near = (-1e-12, -1e-8, -1e-3, 0.0, 1e-12, 1e-8, 1e-3)
        reach = (0.5, 1.0, 3.0, 10.0, 30.0, 50.0, 90.0, 100.0)
    axis = sorted({*near, *reach, *(-value for value in reach)})
    y = (0.0, 1e-2, 1.0, 1e2) if not large else (0.0, 1e-8, 1e-2, 1.0, 1e2, 1e6)
    weight = (math.exp(-100.0), 1e-3, 1.0, 1e3, math.exp(100.0))
    if large:
        weight = (math.exp(-100.0), 1e-30, 1e-3, 1.0, 1e3, 1e30, math.exp(100.0))
    columns = _product(y, axis, axis, power, weight)
    gaps = (0.0, -1e-12, 1e-12, -1e-8, 1e-8)
    p, gap, phi, w = _product(power, gaps, (-1.0, 0.0, 1.0), (1.0, 1e3))
    tied = np.zeros_like(p), (1.0 + gap) / (2.0 - _POWER_LINK.inverse(p)), phi, p, w
    y, m, d, p, w = (np.concatenate(pair) for pair in zip(columns, tied, strict=True))
    # The reference sums the series by hand, as ``_tweedie_grid``'s does.
    with np.errstate(all="ignore"):
        rho = d - np.log(w)
        mode = np.exp((2.0 - _POWER_LINK.inverse(p)) * np.log(np.where(y > 0.0, y, 1.0)) - rho)
    keep = (y == 0.0) | (mode / (2.0 - _POWER_LINK.inverse(p)) <= 200.0)
    # Positive rows whose mean or dispersion underflows, is subnormal or overflows: declined.
    out = (-746.0, -740.0, 710.0)
    rim = _product((1e-300, 2.0), out, (0.0,), (-10.0, 0.0), (math.exp(-100.0), 1.0))
    rim_phi = _product((1e-300, 2.0), (0.0,), out, (-10.0, 0.0), (math.exp(-100.0), 1.0))
    kept = (y[keep], m[keep], d[keep], p[keep], w[keep])
    return tuple(np.concatenate(parts) for parts in zip(kept, rim, rim_phi, strict=True))


def _tweedie_predictor_call(semantics: str):
    """The compiled rows one at a time, for each row's value, as a block's kernel call reads them."""

    def call(columns):
        y, eta_mean, eta_phi, eta_power, weight = columns
        rows = tweedie_kernel.tweedie_predictor_rows(y, weight, semantics, _POWER_LINK, False)
        if rows is None:
            raise _DeclinedError
        stack = np.empty((9, len(y)))
        stack[:3] = eta_mean, eta_phi, eta_power
        value = np.empty(len(y))
        for i in range(len(y)):
            try:
                completed, value[i] = rows.evaluate(0, i, i + 1, stack, 2)
            except ZeroDivisionError as error:
                # A trial's backtracking does not catch this, and the sweep reads it as a refusal.
                raise AssertionError("a compiled row must decline, not divide by zero") from error
            if not completed:
                raise _DeclinedError
        return SimpleNamespace(
            log_likelihood=value, score=stack[6:9].T, hessian_packed=-stack[:6].T,
            valid=np.ones(len(y), dtype=bool),
        )  # fmt: skip

    return call


def _tweedie_predictor_array(semantics: str):
    """The family's natural rows through the chain rule: what a chunk reads without compiled rows."""

    def call(columns):
        y, eta_mean, eta_phi, eta_power, weight = columns
        eta = np.column_stack((eta_mean, eta_phi, eta_power))
        theta = validated_parameter_matrix(
            np.column_stack((np.exp(eta_mean), np.exp(eta_phi), _POWER_LINK.inverse(eta_power))),
            n_observations=len(y),
            parameters=TweedieLSS().parameters,
            family_name="TweedieLSS",
        )
        natural = tweedie_kernel.evaluate_tweedie_rows(
            y, *theta.T, weight, semantics, derivative_order=2
        )
        chained = transform_natural_derivatives(
            NaturalLikelihoodEvaluation(
                optimizing_log_likelihood=natural.log_likelihood,
                parameter_independent_carrier=np.zeros(len(y)),
                score=natural.score,
                hessian_packed=natural.hessian_packed,
                valid=natural.valid,
            ),
            eta,
            _PREDICTOR_LINKS,
        )
        return SimpleNamespace(
            log_likelihood=chained.optimizing_log_likelihood, score=chained.score_eta,
            hessian_packed=-chained.curvature_packed, valid=natural.valid,
        )  # fmt: skip

    return call


def _tweedie_predictor_units(columns: tuple) -> tuple:
    """Per unit of the power itself: the link's factor would hide the power's own entries."""
    return (
        np.ones_like(columns[1]),
        np.ones_like(columns[2]),
        1.0 / _POWER_LINK.deriv_inverse(columns[3]),
    )


def _tweedie_predictor_conditioning(columns: tuple) -> np.ndarray:
    """A zero row's closed form holds the generated kernels' bar and a series row Tweedie's."""
    return np.where(columns[0] == 0.0, 1.0, TWEEDIE_BAR / BAR)


def _gg_array_location(columns):
    y, mu, sigma, q, m = columns
    return gg_kernel._location_rows(y, mu, sigma, q, m, derivative_order=2)


def _gg_array_mean(columns):
    from tests.test_generalized_gamma_compiled_rows import _array_mean_rows

    y, mean, sigma, q, m = columns
    return _array_mean_rows(y, mean, sigma, q, m, 2)[0]


def _ones(column):
    return np.ones_like(column)


def _two_piece_mean_conditioning(columns: tuple) -> np.ndarray:
    """``log K`` cancels between its pieces to about ``sigma``.  Near the join a score divides that
    rounding by ``(sigma s)^2``, ``s = 1 - |skew|`` the narrow piece, in either route."""
    narrow = 1.0 - np.abs(columns[3])
    return np.maximum(1.0, (columns[2] * narrow) ** -2.0)


def _two_piece_mean_curvature(columns: tuple) -> np.ndarray:
    """The loading's curvature reads ``u/s`` in the array rows, whose pieces carry their weights."""
    return 1.0 / (1.0 - np.abs(columns[3]))


def _gg_mean_conditioning(columns: tuple) -> np.ndarray:
    """The loading cancels its gamma functions as ``sigma -> 0``, grows with ``sigma``, and meets a
    pole of ``log Gamma`` at the wall ``sigma |Q| = 1``."""
    sigma, q = columns[2], columns[3]
    wall = np.where(q < 0.0, 1.0 - sigma * np.abs(q), 1.0)
    return np.maximum.reduce((np.ones_like(sigma), sigma, sigma**-2.0, 1.0 / np.maximum(wall, U)))


def _nb_conditioning(columns: tuple) -> np.ndarray:
    """A count's recurrence sums that many terms, which the remainder then cancels."""
    return np.maximum(1.0, columns[0])


ROUTES = (
    Route(
        "gamma_prior",
        lambda c: gamma_kernel.evaluate_gamma_rows(*c, "prior", derivative_order=2),
        (gamma_kernel, "density_rows", declared_gamma),
        lambda c: (c[1], c[2]),
        _gamma_law(True),
        _gamma_grid,
    ),
    Route(
        "gamma_frequency",
        lambda c: gamma_kernel.evaluate_gamma_rows(*c, "frequency", derivative_order=2),
        (gamma_kernel, "density_rows", declared_gamma),
        lambda c: (c[1], c[2]),
        _gamma_law(False),
        _frequency(_gamma_grid),
    ),
    Route(
        "generalized_pareto",
        lambda c: pareto_kernel.scale_rows(*c, derivative_order=2),
        (pareto_kernel, "density_rows", declared_pareto),
        lambda c: (c[1], _ones(c[2])),
        _pareto_law,
        _pareto_grid,
    ),
    Route(
        "log_normal_mean",
        lambda c: log_normal_kernel.mean_rows(*c, derivative_order=2),
        (log_normal_kernel, "density_rows", declared_log_normal),
        lambda c: (c[1], c[2]),
        _log_normal_law,
        _log_normal_grid,
    ),
    Route(
        "two_piece_location",
        lambda c: two_piece_kernel.location_rows(*c, derivative_order=2),
        (two_piece_kernel, "density_rows", declared_location),
        lambda c: (c[2], c[2], _ones(c[3])),
        _two_piece_location_law,
        _two_piece_location_grid,
    ),
    Route(
        "two_piece_mean",
        lambda c: two_piece_kernel.mean_rows(*c, derivative_order=2),
        (two_piece_kernel, "density_rows", declared_mean),
        lambda c: (c[1], c[2], _ones(c[3])),
        _two_piece_mean_law,
        _two_piece_mean_grid,
        bits=lambda row: 160,
        conditioning=_two_piece_mean_conditioning,
        curvature=_two_piece_mean_curvature,
    ),
    Route(
        "negative_binomial_prior",
        lambda c: nb_kernel.evaluate_negative_binomial_rows(*c, "prior", derivative_order=2),
        (nb_kernel, "_compiled_rows", None),
        lambda c: (c[1], c[2]),
        _nb_law(True),
        _nb_grid,
        conditioning=_nb_conditioning,
    ),
    Route(
        "negative_binomial_frequency",
        lambda c: nb_kernel.evaluate_negative_binomial_rows(*c, "frequency", derivative_order=2),
        (nb_kernel, "_compiled_rows", None),
        lambda c: (c[1], c[2]),
        _nb_law(False),
        _frequency(_nb_grid),
        conditioning=_nb_conditioning,
    ),
    Route(
        "generalized_gamma_location",
        lambda c: gg_kernel.location_rows(*c, derivative_order=2),
        None,
        lambda c: (c[2], c[2], _ones(c[3])),
        _gg_location_law,
        _gg_location_grid,
        array=_gg_array_location,
        bits=_gg_bits,
    ),
    Route(
        "generalized_gamma_mean",
        lambda c: gg_kernel.mean_rows(*c, derivative_order=2),
        None,
        lambda c: (c[1], c[2], _ones(c[3])),
        _gg_mean_law,
        _gg_mean_grid,
        array=_gg_array_mean,
        bits=_gg_bits,
        conditioning=_gg_mean_conditioning,
    ),
    Route(
        "tweedie_prior",
        lambda c: tweedie_kernel.evaluate_tweedie_rows(*c[:4], c[4], "prior", derivative_order=2),
        None,
        lambda c: (c[1], c[2], _ones(c[3])),
        _tweedie_law(True),
        _tweedie_grid,
        bits=lambda row: 160,
        bar=TWEEDIE_BAR,
    ),
    Route(
        "tweedie_frequency",
        lambda c: tweedie_kernel.evaluate_tweedie_rows(
            *c[:4], c[4], "frequency", derivative_order=2
        ),
        None,
        lambda c: (c[1], c[2], _ones(c[3])),
        _tweedie_law(False),
        _frequency(_tweedie_grid),
        bits=lambda row: 160,
        bar=TWEEDIE_BAR,
    ),
    Route(
        "tweedie_predictor_prior",
        _tweedie_predictor_call("prior"),
        None,
        _tweedie_predictor_units,
        _tweedie_predictor_law(True),
        _tweedie_predictor_grid,
        array=_tweedie_predictor_array("prior"),
        bits=lambda row: 192,
        conditioning=_tweedie_predictor_conditioning,
    ),
    Route(
        "tweedie_predictor_frequency",
        _tweedie_predictor_call("frequency"),
        None,
        _tweedie_predictor_units,
        _tweedie_predictor_law(False),
        _frequency(_tweedie_predictor_grid),
        array=_tweedie_predictor_array("frequency"),
        bits=lambda row: 192,
        conditioning=_tweedie_predictor_conditioning,
    ),
)


# ---------------------------------------------------------------- the comparison


def _scaled(jet: Jet, units: tuple) -> tuple:
    k = len(units)
    pairs = list(itertools.combinations_with_replacement(range(k), 2))
    score = np.column_stack([jet.score[:, i] * units[i] for i in range(k)])
    hessian = np.column_stack(
        [jet.hessian[:, a] * units[i] * units[j] for a, (i, j) in enumerate(pairs)]
    )
    return jet.value, score, hessian


def _relative_gaps(first: tuple, second: tuple, mass: np.ndarray) -> np.ndarray:
    """Each of the three orders' ``max |difference| / row scale``, per row."""
    tiny = np.finfo(float).tiny
    with np.errstate(all="ignore"):
        return np.column_stack(
            (
                np.abs(first[0] - second[0]) / np.maximum(np.abs(second[0]), mass),
                np.max(np.abs(first[1] - second[1]), axis=1)
                / np.maximum(np.max(np.abs(second[1]), axis=1), tiny),
                np.max(np.abs(first[2] - second[2]), axis=1)
                / np.maximum(np.max(np.abs(second[2]), axis=1), tiny),
            )
        )


def route_bars(route: Route, columns: tuple) -> np.ndarray:
    """The bar of each row at orders 0, 1 and 2."""
    conditioning = route.conditioning(columns)
    curvature = conditioning if route.curvature is None else route.curvature(columns)
    return route.bar * np.column_stack((conditioning, conditioning, curvature))


@dataclass(frozen=True)
class Divergence:
    """One row a route reads worse than the rule allows."""

    route: str
    row: tuple
    compiled_error: float
    array_error: float
    order: int


def _order_errors(
    jet: tuple, index: int, reference: tuple, mass: float
) -> tuple[float, float, float]:
    """Error of one row against the reference, relative to the reference's row scale, per order.

    A value's scale is at least the row's mass: its terms are that size, whatever they cancel to.
    """
    value, score, hessian = reference
    errors = [abs(float(jet[0][index]) - value) / max(abs(value), mass)]
    for level, truth in ((1, score), (2, hessian)):
        truth = np.array(truth)
        size = np.max(np.abs(truth)) if truth.size else 0.0
        errors.append(
            float(np.max(np.abs(jet[level][index] - truth)) / max(size, np.finfo(float).tiny))
        )
    return tuple(errors)


def _reference_choices(route: Route, row: tuple, units: tuple) -> list:
    """Each reference a row may read: both pieces where a two-piece row is within rounding of its join."""
    if route.name == "two_piece_location":
        z = (float(row[0]) - float(row[1])) / float(row[2])
        pieces = (None,) if abs(z) > JOIN_TIE * max(1.0, abs(row[1]) / row[2]) else (True, False)
        return [_with_piece(route, row, units, _two_piece_location_law, right) for right in pieces]
    if route.name == "two_piece_mean":
        with mp.workdps(60):
            z = (mp.log(mp.mpf(float(row[0])) / mp.mpf(float(row[1])))
                 + two_piece_oracle.mp_log_mean_loading(float(row[2]), float(row[3]))) / mp.mpf(float(row[2]))  # fmt: skip
        pieces = (None,) if abs(z) > JOIN_TIE / min(1.0, row[2]) else (True, False)
        return [_with_piece(route, row, units, _two_piece_mean_law, right) for right in pieces]
    return [reference_jet(route, row, units)]


def _with_piece(route, row, units, law, right):
    return reference_jet(replace(route, law=lambda r: law(r, right)), row, units)


def sweep(route: Route, columns: tuple, *, max_references: int | None = None, audit: int = 0):
    """Rows the rule fails, and how many rows each outcome held.

    A row is settled by mpmath when the routes differ by more than its bar, when it has no array
    route, or when the array route refuses it.  ``audit`` adds that many rows drawn at random from
    those the routes agree on, for an error they share.  ``max_references`` keeps half its budget
    for the rows that differ most and draws the rest.
    """
    n, k = len(columns[0]), len(route.units(tuple(c[:1] for c in columns)))
    packed = k * (k + 1) // 2
    with np.errstate(all="ignore"):
        compiled, served = compiled_rows(route, columns, k, packed)
        array, answered = array_rows(route, columns, k, packed)
    units = route.units(columns)
    scaled_compiled = _scaled(compiled, units)
    scaled_array = _scaled(array, units)
    bars = route_bars(route, columns)
    with np.errstate(all="ignore"):
        over = np.max(_relative_gaps(scaled_compiled, scaled_array, columns[-1]) / bars, axis=1)
    both = served & answered & compiled.valid & array.valid
    # Routes within the bar of each other cannot fail the rule against any reference.
    settle = served & compiled.valid & (~both | ~(over <= 1.0))
    invalid_disagreement = served & answered & (compiled.valid != array.valid)
    failures = []
    tightest = 0.0
    rows = np.flatnonzero(settle)
    rng = np.random.default_rng(20261106)
    if max_references is not None and len(rows) > max_references:
        order = rows[np.argsort(-np.nan_to_num(over[rows], nan=0.0))]
        kept = order[: max_references // 2]
        drawn = rng.choice(order[max_references // 2 :], max_references - len(kept), replace=False)
        rows = np.concatenate((kept, drawn))
    if audit:
        agree = np.flatnonzero(served & compiled.valid & ~settle)
        rows = np.concatenate((rows, rng.choice(agree, min(audit, len(agree)), replace=False)))
    for index in rows:
        row = tuple(float(c[index]) for c in columns)
        unit = tuple(float(u[index]) for u in units)
        best = None
        for reference in _reference_choices(route, row, unit):
            ours = _order_errors(scaled_compiled, index, reference, row[-1])
            theirs = _order_errors(scaled_array, index, reference, row[-1])
            levels = [
                (
                    mine / (FACTOR * (their if math.isfinite(their) else 0.0) + bar),
                    level,
                    mine,
                    their,
                )
                for level, (mine, their, bar) in enumerate(
                    zip(ours, theirs, bars[index], strict=True)
                )
            ]
            if best is None or max(levels)[0] < best[0]:
                best = max(levels)
        tightest = max(tightest, best[0])
        if best[0] > 1.0:
            failures.append(Divergence(route.name, row, best[2], best[3], best[1]))
    for index in np.flatnonzero(invalid_disagreement):
        row = tuple(float(c[index]) for c in columns)
        failures.append(Divergence(route.name, row, math.nan, math.nan, -1))
    counts = {
        "rows": n,
        "compiled": int(served.sum()),
        "declined": int((~served).sum()),
        "settled": int(len(rows)),
        "tightest": tightest,
    }
    return failures, counts
