"""Window jets to order four against a SymPy/mpmath oracle, and the bounded-row adjustment.

Two synthetic two-parameter laws supply the oracle: the normal in
``(mu, sigma)``, and the Weibull in ``(a, b) = (log scale, log shape)``, whose
tails are asymmetric (``log S = -t`` and ``log F = log(-expm1(-t))`` with
``t = exp(e^b (s - a))`` at ``s = log y``).  SymPy differentiates each law's
log tails, log density and log window probability to order four; mpmath
evaluates them at 80 significant digits (450 for the Weibull, whose far-left
tail is linear in ``a`` up to ``t ~ 1e-304``, so its exact second
derivatives there sit 300 digits below the terms that form them).  The end
jets, rounded once to float64, are the combiners' inputs; the oracle's
derivatives of ``log(T(near) - T(far))`` are the reference.

Error analysis, in the unit roundoff ``u = 2**-53`` (Higham 2002, 2.2, 3.1),
first order, doubled to cover second-order terms.  Each bound is formed per
entry from the combiner's own float64 inputs (the convention of
``tests/test_family_derivatives_symbolic.py``'s derived bounds); the report
divides errors by ``u`` times the row's largest same-order oracle entry.

Combiner (:func:`log_window_jets`).  ``F_I = N_I + sum_pi phi^(m)(Delta)
prod_{B in pi} Delta_B`` over the set partitions ``pi`` of ``I`` with ``m``
blocks, ``Delta_B = F_B - N_B`` (far minus near).

- ``Delta_B`` carries ``2u (|F_B| + |N_B|)``: one rounding per input and one
  for the subtraction.
- ``Delta``'s error ``2u (|F_0| + |N_0|)`` reaches ``expm1(-Delta)`` times
  ``1 + kappa``; with ``expm1`` within one ulp and the reciprocal,
  ``kappa``'s relative error is ``rho = 2u (1 + kappa)(|F_0| + |N_0|) + 3u``.
  This is the combiner's ``u/|Delta|`` loss: as the window narrows,
  ``kappa ~ 1/|Delta|``.
- ``phi^(m)`` is a degree-``m`` polynomial in ``kappa`` with positive terms,
  formed in at most six roundings, so its relative error is ``m rho + 6u``.
- Each term is ``m`` further products, and at most fifteen terms plus ``N_I``
  are summed: ``4u + 16u``, so ``26u`` per term with ``phi^(m)``'s own, and
  ``17u |N_I|`` for the near entry.

So ``tol_I = 2 [17u |N_I| + sum_pi (|phi^(m)| prod |Delta_B| (26u + m rho) +
|phi^(m)| sum_B 2u (|F_B| + |N_B|) prod_{B' != B} |Delta_B'|)]``, and at
order zero ``2 [2u |N_0| + 3u |phi| + 2u kappa (|F_0| + |N_0|)]``.  A far end
absent or negligible returns the near jets, whose error is their rounding.

Quadrature (:func:`quadrature_window_jets`).  With node shares ``p_i`` and
centred node derivatives ``c_B,i = l_B,i - E_p l_B``:

- The shares come from ``exp(log omega_i + l_0,i - top)``, normalised, so
  their relative error is ``eps_p = 2u (max |l_0| + 4)``.
- ``E_p l_I`` carries ``8u max |l_I|`` (input, three products, two sums and
  the normalisation) plus ``eps_p max |c_I|``.
- ``c_B`` carries ``6u max |l_B| + eps_p max |c_B|``; a cumulant term is a
  product of centred factors, each rounded, and the fourth cumulant has four
  such products.
- The partition terms and the mean are summed: ``16u`` of their absolute sum.
- The value ``log(h) + top + log(total)`` carries ``u (3|log h| + 1) +
  u (max |l_0| + 3 |top|) + 2u (max |l_0| + 2) + 4u + 3u |log total|``.

The three-point Gauss-Legendre error on a window of width ``w`` on the
density's own scale is about ``5e-7 w^6`` relative (Abramowitz and Stegun
25.4.30), below ``1e-24`` for the widths here (``1e-3`` and ``1e-8``), and
not counted.  Every bound adds ``64 * 2**-1074`` for gradual underflow.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest

sp = pytest.importorskip("sympy")
mp = pytest.importorskip("mpmath")

from sympy.codegen.cfunctions import expm1 as sp_expm1  # noqa: E402
from sympy.utilities.iterables import multiset_partitions  # noqa: E402

from superglm.distributional.families._response_bounds import (  # noqa: E402
    bounded_higher_derivatives,
)
from superglm.distributional.kernels._response_tails import (  # noqa: E402
    log_window_jets,
    quadrature_window_jets,
)
from superglm.distributional.observation import (  # noqa: E402
    LEFT_CENSORED,
    RIGHT_CENSORED,
    ResponseBounds,
)

_U = 2.0**-53
_TINY = 64 * 2.0**-1074
_K = 2
_ORDER = 4
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(3)


def _tuples(order: int) -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations_with_replacement(range(_K), order))


@functools.cache
def _partitions(order: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    return tuple(
        tuple(tuple(block) for block in p) for p in multiset_partitions(list(range(order)))
    )


def _blocks(target: tuple[int, ...], partition) -> list[tuple[int, ...]]:
    return [tuple(sorted(target[position] for position in block)) for block in partition]


def _column(key: tuple[int, ...]) -> int:
    return _tuples(len(key)).index(key)


# --- the oracle -------------------------------------------------------------


def _jet_functions(expression, params, arguments):
    levels = [{(): expression}]
    for order in range(1, _ORDER + 1):
        previous = levels[-1]
        levels.append({key: sp.diff(previous[key[:-1]], params[key[-1]]) for key in _tuples(order)})
    modules = [{"expm1": mp.expm1}, "mpmath"]
    return [
        [sp.lambdify((*params, *arguments), level[key], modules) for key in _tuples(order)]
        for order, level in enumerate(levels)
    ]


class _Law:
    """A two-parameter law's symbolic log tails, log density and log windows."""

    def __init__(self, params, values, point, cdf, sf, log_density, *, dps, to_y, from_y):
        self.values = values
        self.dps = dps
        self.to_y, self.from_y = to_y, from_y
        lo, hi = sp.symbols("lo hi", real=True)
        self.tail = {
            "lower": _jet_functions(sp.log(cdf(point)), params, (point,)),
            "upper": _jet_functions(sp.log(sf(point)), params, (point,)),
        }
        self.window = {
            "lower": _jet_functions(sp.log(cdf(hi) - cdf(lo)), params, (lo, hi)),
            "upper": _jet_functions(sp.log(sf(lo) - sf(hi)), params, (lo, hi)),
        }
        self.density = _jet_functions(log_density, params, (point,))

    def evaluate(self, functions, *points):
        with mp.workdps(self.dps):
            arguments = [mp.mpf(v) for v in self.values] + [mp.mpf(p) for p in points]
            return [[f(*arguments) for f in level] for level in functions]


@functools.cache
def _law(name: str) -> _Law:
    if name == "normal":
        mu, sigma, y = sp.symbols("mu sigma y", real=True)

        def cdf(x):
            return sp.erfc(-(x - mu) / (sigma * sp.sqrt(2))) / 2

        def sf(x):
            return sp.erfc((x - mu) / (sigma * sp.sqrt(2))) / 2

        density = -sp.log(sigma) - sp.log(2 * sp.pi) / 2 - (y - mu) ** 2 / (2 * sigma**2)
        return _Law(
            (mu, sigma), _NORMAL, y, cdf, sf, density, dps=80, to_y=lambda p: p, from_y=lambda y: y
        )
    a, b, s = sp.symbols("a b s", real=True)

    def t(x):
        return sp.exp(sp.exp(b) * (x - a))

    def cdf(x):
        return -sp_expm1(-t(x))

    def sf(x):
        return sp.exp(-t(x))

    density = b - a + (sp.exp(b) - 1) * (s - a) - t(s)
    return _Law((a, b), _WEIBULL, s, cdf, sf, density, dps=450, to_y=mp.exp, from_y=mp.log)


_NORMAL = (0.3, 1.7)
_WEIBULL = (0.4, math.log(1.5))


def _normal_point(z: float) -> float:
    return _NORMAL[0] + _NORMAL[1] * z


def _weibull_point(w: float) -> float:
    return _WEIBULL[0] + w / math.exp(_WEIBULL[1])


def _rounded(exact) -> tuple[np.ndarray, ...]:
    return (np.array([float(exact[0][0])]),) + tuple(
        np.array([[float(v) for v in level]]) for level in exact[1:]
    )


def _absent() -> tuple[np.ndarray, ...]:
    return (np.array([-np.inf]),) + tuple(
        np.zeros((1, len(_tuples(k)))) for k in range(1, _ORDER + 1)
    )


def _errors(computed, oracle) -> list[np.ndarray]:
    with mp.workdps(60):
        return [
            np.array(
                [float(abs(mp.mpf(float(c)) - o)) for c, o in zip(np.ravel(got), want, strict=True)]
            )
            for got, want in zip(computed, oracle, strict=True)
        ]


# --- the bounds of the module docstring ---------------------------------------


def _kappa(near, far) -> float:
    with np.errstate(over="ignore", divide="ignore"):
        return float(1.0 / np.expm1(-(far[0][0] - near[0][0])))


def _phi_magnitudes(kappa: float) -> dict[int, float]:
    second = kappa * (1.0 + kappa)
    return {
        1: kappa,
        2: second,
        3: second * (1.0 + 2.0 * kappa),
        4: second * (1.0 + 6.0 * kappa * (1.0 + kappa)),
    }


def _combiner_terms(near, far, order: int):
    """Per entry of ``order``: ``(|N_I|, [(m, |phi^(m)|, [|Delta_B|], [|F_B| + |N_B|])])``."""
    phi = _phi_magnitudes(_kappa(near, far))
    entries = []
    for column, target in enumerate(_tuples(order)):
        terms = []
        for partition in _partitions(order):
            keys = _blocks(target, partition)
            deltas = [abs(far[len(k)][0, _column(k)] - near[len(k)][0, _column(k)]) for k in keys]
            scales = [
                abs(far[len(k)][0, _column(k)]) + abs(near[len(k)][0, _column(k)]) for k in keys
            ]
            terms.append((len(keys), phi[len(keys)], deltas, scales))
        entries.append((abs(near[order][0, column]), terms))
    return entries


def _combiner_bounds(near, far) -> list[np.ndarray]:
    n0, f0 = near[0][0], far[0][0]
    kappa = 0.0 if f0 == -np.inf else _kappa(near, far)
    if kappa == 0.0:
        return [2 * _U * np.abs(np.ravel(j)) + _TINY for j in near]
    ends = abs(n0) + abs(f0)
    rho = 2 * _U * (1.0 + kappa) * ends + 3 * _U
    phi0 = abs(math.log(-math.expm1(f0 - n0)))
    bounds = [np.array([2 * (2 * _U * abs(n0) + 3 * _U * phi0 + 2 * _U * kappa * ends) + _TINY])]
    for order in range(1, _ORDER + 1):
        entries = _combiner_terms(near, far, order)
        tol = []
        for near_entry, terms in entries:
            total = 17 * _U * near_entry
            for m, phi, deltas, scales in terms:
                total += phi * math.prod(deltas) * (26 * _U + m * rho)
                for j, scale in enumerate(scales):
                    total += phi * 2 * _U * scale * math.prod(deltas[:j] + deltas[j + 1 :])
            tol.append(2 * total + _TINY)
        bounds.append(np.array(tol))
    return bounds


def _combiner_scales(near, far) -> list[np.ndarray]:
    """``|N_I| + sum_pi |phi^(m)| prod |Delta_B|``: the term scale two roundings apart compare on."""
    scales = [np.array([abs(near[0][0]) + abs(math.log(-math.expm1(far[0][0] - near[0][0])))])]
    for order in range(1, _ORDER + 1):
        entries = _combiner_terms(near, far, order)
        scales.append(
            np.array(
                [n + sum(phi * math.prod(d) for _, phi, d, _ in terms) for n, terms in entries]
            )
        )
    return scales


def _quadrature_scales(nodes, half_width: float):
    """``(tol, scale)`` per order: the docstring's bound and the term scale."""
    logs = np.array([math.log(w) + node[0][0] for w, node in zip(_LEGENDRE_WEIGHTS, nodes)])
    top = float(np.max(logs))
    shares = np.exp(logs - top)
    log_total = math.log(float(np.sum(shares)))
    shares /= np.sum(shares)
    l0 = max(abs(node[0][0]) for node in nodes)
    eps_p = 2 * _U * (l0 + 4)
    lh = abs(math.log(half_width))
    value_tol = (
        _U * (3 * lh + 1) + _U * (l0 + 3 * abs(top)) + 2 * _U * (l0 + 2) + 4 * _U
    ) + 3 * _U * abs(log_total)
    tols, scales = [np.array([2 * value_tol + _TINY])], [np.array([lh + abs(top)])]

    def stats(key):
        values = np.array([node[len(key)][0, _column(key)] for node in nodes])
        centred = values - np.sum(shares * values)
        return float(np.max(np.abs(values))), float(np.max(np.abs(centred)))

    for order in range(1, _ORDER + 1):
        tol, scale = [], []
        for target in _tuples(order):
            big, small = stats(target)
            total, terms = 8 * _U * big + eps_p * small, big
            for partition in _partitions(order):
                if len(partition) < 2:
                    continue
                keys = _blocks(target, partition)
                pairs = [stats(k) for k in keys]
                weight = 4 if len(keys) == 4 else 1
                product = math.prod(c for _, c in pairs)
                terms += weight * product
                total += weight * (12 * _U + eps_p) * product
                for j, (b, c) in enumerate(pairs):
                    rest = math.prod(cc for _, cc in pairs[:j] + pairs[j + 1 :])
                    total += weight * (6 * _U * b + eps_p * c) * rest
            tol.append(2 * (total + 16 * _U * terms) + _TINY)
            scale.append(terms)
        tols.append(np.array(tol))
        scales.append(np.array(scale))
    return tols, scales


# --- the combiner -------------------------------------------------------------

_Z = _normal_point
_W = _weibull_point
_WINDOWS = [
    ("normal", "body, lower tail", _Z(-0.8), _Z(1.3), "lower"),
    ("normal", "body, upper tail", _Z(-0.8), _Z(1.3), "upper"),
    ("normal", "far left tail", _Z(-37.5), _Z(-37.2), "lower"),
    ("normal", "far right tail", _Z(37.2), _Z(37.5), "upper"),
    ("normal", "width 0.05", _Z(0.40), _Z(0.45), "upper"),
    ("normal", "width 0.05, left tail", _Z(-5.0), _Z(-5.0 + 0.01), "lower"),
    ("normal", "lower one-sided", -np.inf, _Z(1.0), "lower"),
    ("normal", "upper one-sided", _Z(-0.5), np.inf, "upper"),
    ("normal", "negligible far end", _Z(-40.0), _Z(0.5), "lower"),
    ("weibull", "body, lower tail", _W(-0.7), _W(0.3), "lower"),
    ("weibull", "body, upper tail", _W(-0.7), _W(0.3), "upper"),
    ("weibull", "far left tail", _W(-701.0), _W(-700.0), "lower"),
    ("weibull", "far right tail", _W(math.log(700.0)), _W(math.log(705.0)), "upper"),
    ("weibull", "width 0.05", _W(0.10), _W(0.13), "upper"),
    ("weibull", "lower one-sided", -np.inf, _W(0.2), "lower"),
    ("weibull", "upper one-sided", _W(0.5), np.inf, "upper"),
    ("weibull", "negligible far end", _W(-800.0), _W(0.3), "lower"),
]


def _window_inputs(law_name, lo, hi, reading):
    law = _law(law_name)
    near_point, far_point = (hi, lo) if reading == "lower" else (lo, hi)
    near_exact = law.evaluate(law.tail[reading], near_point)
    if np.isfinite(far_point):
        far = _rounded(law.evaluate(law.tail[reading], far_point))
        oracle = law.evaluate(law.window[reading], lo, hi)
    else:
        far, oracle = _absent(), near_exact
    return _rounded(near_exact), far, oracle


@pytest.mark.parametrize(
    ("law_name", "lo", "hi", "reading"),
    [case[:1] + case[2:] for case in _WINDOWS],
    ids=[f"{case[0]}: {case[1]}" for case in _WINDOWS],
)
def test_window_jets_match_the_oracle_to_order_four(law_name, lo, hi, reading):
    near, far, oracle = _window_inputs(law_name, lo, hi, reading)
    computed = log_window_jets(near, far, order=_ORDER)
    for order, (error, tol) in enumerate(
        zip(_errors(computed, oracle), _combiner_bounds(near, far), strict=True)
    ):
        assert np.all(error <= tol), (order, error, tol)


@pytest.mark.parametrize(
    ("law_name", "lo", "hi", "reading"),
    [case[:1] + case[2:] for case in _WINDOWS if np.isfinite(case[2]) and np.isfinite(case[3])],
)
def test_window_jets_of_a_lower_order_are_the_prefix(law_name, lo, hi, reading):
    near, far, _ = _window_inputs(law_name, lo, hi, reading)
    jets = log_window_jets(near, far, order=_ORDER)
    for order in range(_ORDER):
        prefix = log_window_jets(near, far, order=order)
        assert len(prefix) == order + 1
        for got, want in zip(prefix, jets, strict=False):
            np.testing.assert_array_equal(got, want)


def _synthetic_jets(rng, value):
    return (np.asarray(value, dtype=float),) + tuple(
        rng.normal(size=(len(value), len(_tuples(k)))) for k in range(1, _ORDER + 1)
    )


def test_absent_negligible_and_equal_ends_keep_the_order_two_semantics():
    rng = np.random.default_rng(544)
    near = _synthetic_jets(rng, [-1.0, -1.0, -2.0, -0.5])
    far = _synthetic_jets(rng, [-np.inf, -900.0, -2.0, -0.7])
    for k in range(1, _ORDER + 1):
        # Overflowed far-end derivatives of the absent and negligible ends are ignored.
        far[k][0] = np.nan
        far[k][1] = np.inf
        far[k][2] = near[k][2]
    jets = log_window_jets(near, far, order=_ORDER)
    for got, want in zip(jets, near, strict=True):
        np.testing.assert_array_equal(got[:2], want[:2])
    assert jets[0][2] == -np.inf
    assert np.all(np.isfinite(np.concatenate([np.ravel(j[3]) for j in jets])))
    with pytest.raises(ValueError, match="order 0 through 4"):
        log_window_jets(near, far, order=5)


def test_a_nan_far_end_is_a_failed_window_not_an_absent_one():
    """Only ``-inf`` marks an absent end; an engine's NaN far tail must leave the row NaN."""
    rng = np.random.default_rng(565)
    near = _synthetic_jets(rng, [-1.0, -1.0])
    far = _synthetic_jets(rng, [np.nan, -np.inf])
    jets = log_window_jets(near, far, order=_ORDER)
    assert np.isnan(jets[0][0])
    for got, want in zip(jets, near, strict=True):
        np.testing.assert_array_equal(got[1], want[1])


# --- the quadrature route -----------------------------------------------------

_NARROW = [
    ("normal", "body", 0.37, 1.0, "upper"),
    ("normal", "left tail", -9.0, 9.0, "lower"),
    ("weibull", "body", 0.2, 1.33, "upper"),
    ("weibull", "right tail", math.log(50.0), 75.0, "upper"),
]


def _narrow_inputs(law_name, where, scale, width, reading):
    """A window ``width`` wide on the density's own scale, ``scale`` its log-density slope there."""
    law = _law(law_name)
    if law_name == "normal":
        lo = _normal_point(where)
        hi = lo + _NORMAL[1] * width / scale
    else:
        lo = _weibull_point(where)
        hi = lo + width / scale
    with mp.workdps(law.dps):
        y_lo, y_hi = law.to_y(mp.mpf(lo)), law.to_y(mp.mpf(hi))
        half_width = float((y_hi - y_lo) / 2)

    def node(fraction: float):
        with mp.workdps(law.dps):
            point = law.from_y(y_lo + mp.mpf(fraction) * (y_hi - y_lo))
        return _rounded(law.evaluate(law.density, point))

    nodes = {float(0.5 * (1.0 + t)): node(float(0.5 * (1.0 + t))) for t in _LEGENDRE_NODES}
    oracle = law.evaluate(law.window[reading], lo, hi)
    return np.array([half_width]), (lambda fraction: nodes[fraction]), list(nodes.values()), oracle


_NARROW_CASES = [(*case, width) for case in _NARROW for width in (1e-3, 1e-8)]


@pytest.mark.parametrize(
    ("law_name", "where", "scale", "reading", "width"),
    [case[:1] + case[2:] for case in _NARROW_CASES],
    ids=[f"{case[0]}: {case[1]}, width {case[-1]:g}" for case in _NARROW_CASES],
)
def test_quadrature_jets_match_the_oracle_to_order_four(law_name, where, scale, reading, width):
    half_width, density, nodes, oracle = _narrow_inputs(law_name, where, scale, width, reading)
    computed = quadrature_window_jets(half_width, density, order=_ORDER)
    tols, _ = _quadrature_scales(nodes, float(half_width[0]))
    for order, (error, tol) in enumerate(zip(_errors(computed, oracle), tols, strict=True)):
        assert np.all(error <= tol), (order, error, tol)
    for order in range(_ORDER):
        prefix = quadrature_window_jets(half_width, density, order=order)
        for got, want in zip(prefix, computed, strict=False):
            np.testing.assert_array_equal(got, want)


# --- bounded rows -------------------------------------------------------------


def test_bounded_higher_derivatives_adjust_censored_and_truncated_rows():
    response = np.array([1.0, 2.0, 1.5, 0.8, 1.2])
    bounds = ResponseBounds.from_arrays(
        lower=np.array([-np.inf, -np.inf, 0.5, 0.25, 0.2]),
        upper=np.array([np.inf, np.inf, 3.0, 4.0, np.inf]),
        side=np.array([0, RIGHT_CENSORED, 0, 0, LEFT_CENSORED]),
    )
    multiplier = np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    rng = np.random.default_rng(7)
    third = rng.normal(size=(5, 4))
    fourth = rng.normal(size=(5, 5))
    valid = np.array([True, True, True, True, False])

    def jets(lower, upper, rows, order):
        assert order == 4
        base = np.arctan(lower) + 2.0 * np.arctan(upper) + rows
        out = (base,) + tuple(
            np.outer(base, np.arange(1.0, len(_tuples(k)) + 1.0)) ** (1 + k % 2)
            for k in range(1, 5)
        )
        out[3][rows == 3] = np.nan
        return out

    got_third, got_fourth, got_valid = bounded_higher_derivatives(
        third,
        fourth,
        valid,
        response=response,
        bounds=bounds,
        multiplier=multiplier,
        window_jets=jets,
    )
    row = np.arange(5)
    want_third, want_fourth = third.copy(), fourth.copy()
    for k, want in ((3, want_third), (4, want_fourth)):
        observed = jets(np.array([2.0, 0.2]), np.array([np.inf, 1.2]), row[[1, 4]], 4)[k]
        want[[1, 4]] = multiplier[[1, 4], None] * observed
        truncation = jets(bounds.lower[2:], bounds.upper[2:], row[2:], 4)[k]
        want[2:] -= multiplier[2:, None] * truncation
    want_third[3] = want_fourth[3] = 0.0
    np.testing.assert_array_equal(got_third, want_third)
    np.testing.assert_array_equal(got_fourth, want_fourth)
    np.testing.assert_array_equal(got_valid, [True, True, True, False, False])
    np.testing.assert_array_equal(third[0], got_third[0])
