"""mpmath oracle for the log incomplete-gamma tail engine and its partials.

``log_gamma_tail(a, x, upper, order, out)`` writes ``log Q(a, x)`` or
``log P(a, x)`` and every partial in ``(a, x)`` to order 4.  The oracle works
at 60 significant digits and shares no formula with the engine: the pure
``a`` derivatives of the smaller tail come from ``mpmath.diffs`` on
``mpmath.gammainc``, every partial with an ``x`` in it from ``mpmath.diff``
of the gamma density (``dP/dx`` is the density), and the log by the series
of ``log(1 + H)`` on the bivariate Taylor polynomial.

Each order's error is measured against the row's order-``k`` scale, the
largest oracle entry of that order in the row (the convention of
``tests/test_family_derivatives_symbolic.py``).

Error model.  Inside ``1e-2 <= a <= 1e4`` the error has three sources:

- the value's rounding, at most a few ``u`` times the magnitudes of the
  terms forming the primary ``log T``.  A tail read by complement multiplies
  it by ``T/(1 - T)``, and a complement of a tail far below one carries that
  tail's relative error, which is the absolute error of its log (``u |log
  T|``);
- for orders 3 and 4 near the diagonal, cancellation in forming the log's
  Taylor coefficients from those of the series or continued fraction, which
  grows like ``sqrt(a)`` (the ratio of the shape to the law's width);
- below ``a ~ 1e-2``, cancellation in the value itself: near ``x ~ 1`` the
  log of the series tail is ``O(a)`` but is formed from ``O(x)`` terms, so
  its relative error grows like ``u/a``.

Measured worst errors over the grid below, in ``u`` of the row scale, on
x86-64 Linux: 330, 330, 580, 5.4e3 and 4.4e4 at orders 0-4 (the last two at
``a = 1e4`` on the switch line).  The tolerances are ``1e-12`` (about
4,500 ``u``) at orders 0-2, the stage gate of issue 564 stage 4a, which the
measurement meets with a margin of eight; and ``1e-11`` and ``1e-10`` at
orders 3 and 4, about twice and ten times the measurement.  Outside the
range the same measurement gives up to ``6e-9`` at ``a = 1e-6`` (the
``u/a`` term, at every order) and ``3e-10`` at order 4 for ``a = 1e6`` (the
``sqrt(a)`` term).

Along fixed ``x/a``.  ``log_gamma_tail_jet_into(..., along_ratio=True, ...)``
gives the Taylor coefficients ``c_ij`` of ``log T(a + da, (a + da) (x/a) (1 +
e))``; the oracle is the fixed-``x`` one above, composed exactly with ``dx = x
(da/a + e + da e/a)``.  Coefficients of mixed sign can vanish, so each is
measured against ``max(|c_ij|, F_ij, M_k/a^i)``: ``F_ij`` the largest ``|c_p
c_r|`` with ``p + r = (i, j)`` (its scale in the log), ``M_k`` the row's
largest ``|c| a^i`` of order ``k`` (``a^i`` puts ``da`` in units of ``a``).
Measured worst over the grid: 1.65e3 ``u`` at orders 0-2 (the value's rounding
at ``a = 1e3``, far upper tail) and 6.7e3 ``u`` at orders 3-4 (``a = 0.01``,
the ``u/a`` term); the tolerances are 8e3 and 3e4 ``u``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from scipy.special import gammainc, gammaincc  # noqa: E402
from scipy.stats import gamma as gamma_law  # noqa: E402

from superglm.distributional.kernels._incomplete_gamma import (  # noqa: E402
    MAX_ORDER,
    WORK_SIZE,
    log_gamma_tail,
    log_gamma_tail_jet_into,
    log_gamma_tail_jets,
    output_size,
)

_DIGITS = 60
_TOLERANCE = (1e-12, 1e-12, 1e-12, 1e-11, 1e-10)
_BLOCKS = tuple((k * (k + 1) // 2, (k + 1) * (k + 2) // 2) for k in range(MAX_ORDER + 1))
_ALONG_TOLERANCE = np.array([8e3, 8e3, 8e3, 3e4, 3e4]) * 2.0**-53
_KEYS = tuple((k - j, j) for k in range(MAX_ORDER + 1) for j in range(k + 1))
_SHAPES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 12.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4)


def _switch_line(a: float) -> float:
    return a + max(1.0, math.sqrt(a))


def _points(a: float) -> dict[str, float]:
    s = max(1.0, math.sqrt(a))
    return {
        "far lower tail": a * 1e-3 if a >= 1.0 else 1e-30,
        "lower tail": a * 0.3 if a >= 1.0 else 1e-6,
        "lower body": a - 2.0 * s if a > 2.0 * s else a / 4.0,
        "diagonal": a,
        "just above the diagonal": a + 1.5,
        "between diagonal and switch": a + 0.5 * (1.0 + s),
        "switch line, series side": _switch_line(a),
        "switch line, fraction side": float(np.nextafter(_switch_line(a), np.inf)),
        "upper body": a + 3.0 * s,
        "far upper tail": a + 40.0 * s + 700.0,
        "x = 1": 1.0,
        "x = 5": 5.0,
    }


def _times(p: dict, q: dict) -> dict:
    product: dict = {}
    for (i1, j1), v1 in p.items():
        for (i2, j2), v2 in q.items():
            if i1 + i2 + j1 + j2 <= MAX_ORDER:
                key = (i1 + i2, j1 + j2)
                product[key] = product.get(key, 0) + v1 * v2
    return product


def _oracle(a: float, x: float) -> dict[bool, list]:
    """Partials of ``log P`` (``False``) and ``log Q`` (``True``) in I2 order."""
    with mp.workdps(_DIGITS):
        shape, point = mp.mpf(a), mp.mpf(x)
        small_is_lower = mp.gammainc(shape, 0, point, regularized=True) < 0.5
        if small_is_lower:
            sign = 1

            def tail(s):
                return mp.gammainc(s, 0, point, regularized=True)

        else:
            sign = -1

            def tail(s):
                return mp.gammainc(s, point, mp.inf, regularized=True)

        def density(s, y):
            return mp.exp((s - 1) * mp.log(y) - y - mp.loggamma(s))

        taylor = {}
        for i, value in enumerate(mp.diffs(tail, shape, MAX_ORDER)):
            taylor[(i, 0)] = value / mp.factorial(i)
        for j in range(1, MAX_ORDER + 1):
            for i in range(MAX_ORDER + 1 - j):
                derivative = mp.diff(density, (shape, point), (i, j - 1))
                taylor[(i, j)] = sign * derivative / (mp.factorial(i) * mp.factorial(j))
        t0 = taylor[(0, 0)]
        result = {}
        for upper in (False, True):
            if small_is_lower != upper:
                log0 = mp.log(t0)
                h = {key: v / t0 for key, v in taylor.items() if key != (0, 0)}
            else:
                log0 = mp.log1p(-t0)
                h = {key: -v / (1 - t0) for key, v in taylor.items() if key != (0, 0)}
            log_tail = {(0, 0): log0}
            power = h
            for k in range(1, MAX_ORDER + 1):
                for key, v in power.items():
                    log_tail[key] = log_tail.get(key, 0) + (-1) ** (k + 1) * v / k
                power = _times(power, h)
            result[upper] = [
                mp.factorial(k - j) * mp.factorial(j) * log_tail.get((k - j, j), mp.mpf(0))
                for k in range(MAX_ORDER + 1)
                for j in range(k + 1)
            ]
        return result


def _along_ratio(plain: list, a: float, x: float) -> np.ndarray:
    """Fixed-``x`` plain partials to Taylor coefficients along fixed ``x/a``."""
    with mp.workdps(_DIGITS):
        dx = {(1, 0): mp.mpf(x) / a, (0, 1): mp.mpf(x), (1, 1): mp.mpf(x) / a}
        powers = [{(0, 0): mp.mpf(1)}]
        for _ in range(MAX_ORDER):
            powers.append(_times(powers[-1], dx))
        taylor: dict = {}
        for value, (i, j) in zip(plain, _KEYS, strict=True):
            coefficient = value / (mp.factorial(i) * mp.factorial(j))
            for (k, m), v in powers[j].items():
                if k + i + m <= MAX_ORDER:
                    taylor[(k + i, m)] = taylor.get((k + i, m), 0) + coefficient * v
        return np.array([float(taylor.get(key, 0)) for key in _KEYS])


def _along_scale(c: np.ndarray, a: float) -> np.ndarray:
    """``max(|c_ij|, F_ij, M_k/a^i)`` (module docstring)."""
    weight = np.array([a**i for i, _ in _KEYS])
    scale = np.abs(c)
    for r, (i, j) in enumerate(_KEYS):
        for p, (ip, jp) in enumerate(_KEYS):
            if 0 < p != r and ip <= i and jp <= j:
                scale[r] = max(scale[r], abs(c[p] * c[_KEYS.index((i - ip, j - jp))]))
    for start, stop in _BLOCKS:
        order_scale = np.max(np.abs(c[start:stop] * weight[start:stop]))
        scale[start:stop] = np.maximum(scale[start:stop], order_scale / weight[start:stop])
    return scale


def _engine(a: float, x: float, upper: bool, order: int) -> np.ndarray:
    out = np.empty(output_size(order))
    log_gamma_tail(a, x, upper, order, out)
    return out


@pytest.mark.parametrize("a", _SHAPES)
def test_every_partial_matches_the_mpmath_oracle(a: float) -> None:
    """Each order of each call, both tails, across both far tails and the diagonal.

    The order-4 jet along fixed ``x/a`` is checked on the same rows, and at
    fixed ``x`` the jet entry is the engine's partials before scaling.
    """
    failures = []
    for label, x in _points(a).items():
        oracle = _oracle(a, x)
        for upper in (False, True):
            reference = np.array([float(v) for v in oracle[upper]])
            work = np.empty(WORK_SIZE)
            jet = np.empty(output_size(MAX_ORDER))
            assert log_gamma_tail_jet_into(a, x, upper, MAX_ORDER, False, jet, work)
            factor = np.array([math.factorial(i) * math.factorial(j) / x**j for i, j in _KEYS])
            np.testing.assert_allclose(factor * jet, _engine(a, x, upper, MAX_ORDER), rtol=1e-15)
            along = _along_ratio(oracle[upper], a, x)
            if np.all(np.isfinite(along)):
                assert log_gamma_tail_jet_into(a, x, upper, MAX_ORDER, True, jet, work)
                scale = _along_scale(along, a)
                for order, (start, stop) in enumerate(_BLOCKS):
                    usable = scale[start:stop] > 1e-280
                    error = np.abs(jet[start:stop] - along[start:stop])[usable]
                    if not np.all(error <= _ALONG_TOLERANCE[order] * scale[start:stop][usable]):
                        tail = "Q" if upper else "P"
                        failures.append(f"x={x:.17g} ({label}) along x/a log {tail} order {order}")
            for order in range(MAX_ORDER + 1):
                start, stop = _BLOCKS[order]
                got = _engine(a, x, upper, order)[start:stop]
                scale = np.max(np.abs(reference[start:stop]))
                if scale < 1e-280:
                    # The partials underflow in truth; the engine must not invent mass.
                    error = float(np.max(np.abs(got)))
                    bound = 1e-280
                else:
                    error = float(np.max(np.abs(got - reference[start:stop])) / scale)
                    bound = _TOLERANCE[order]
                if not error <= bound:
                    tail = "Q" if upper else "P"
                    failures.append(f"x={x:.17g} ({label}) log {tail} order {order}: {error:.2e}")
    assert failures == []


@pytest.mark.parametrize("a", (*_SHAPES, 1e5))
@pytest.mark.parametrize("upper", [False, True])
def test_series_and_continued_fraction_agree_across_the_switch_line(a: float, upper: bool) -> None:
    below = _switch_line(a)
    above = float(np.nextafter(below, np.inf))
    series = _engine(a, below, upper, MAX_ORDER)
    fraction = _engine(a, above, upper, MAX_ORDER)
    for order, (start, stop) in enumerate(_BLOCKS):
        scale = np.max(np.abs(series[start:stop]))
        gap = np.max(np.abs(series[start:stop] - fraction[start:stop]))
        assert gap <= _TOLERANCE[order] * scale, (order, gap / scale)


def test_values_agree_with_scipy_where_scipy_is_accurate() -> None:
    """Relative ``1e-12`` on the probability wherever scipy's value is normal."""
    shapes = np.geomspace(0.01, 1e4, 25)
    worst = 0.0
    for a in shapes:
        s = max(1.0, math.sqrt(a))
        for x in np.concatenate([a + s * np.linspace(-6.0, 12.0, 37), [1e-3, 0.5, 2.0, 30.0]]):
            if x <= 0.0:
                continue
            for upper, reference in ((False, gammainc(a, x)), (True, gammaincc(a, x))):
                if not 1e-300 < reference <= 1.0:
                    continue
                value = math.exp(_engine(float(a), float(x), upper, 0)[0])
                worst = max(worst, abs(value - reference) / reference)
    assert worst <= 1e-12


@pytest.mark.parametrize(
    ("a", "x", "upper", "scipy_log"),
    [
        (10.0, 800.0, True, lambda: gamma_law.logsf(800.0, 10.0)),
        (1e4, 10.0, False, lambda: gamma_law.logcdf(10.0, 1e4)),
    ],
)
def test_far_tails_stay_finite_where_scipy_underflows(a, x, upper, scipy_log) -> None:
    assert scipy_log() == -np.inf
    with mp.workdps(_DIGITS):
        if upper:
            reference = mp.log(mp.gammainc(a, x, mp.inf, regularized=True))
        else:
            reference = mp.log(mp.gammainc(a, 0, x, regularized=True))
    got = _engine(a, x, upper, MAX_ORDER)
    assert np.all(np.isfinite(got))
    assert abs(got[0] - float(reference)) <= 1e-14 * abs(float(reference))


@pytest.mark.parametrize(
    ("a", "x", "order"),
    [
        (0.0, 1.0, 4),
        (-1.0, 1.0, 4),
        (np.nan, 1.0, 4),
        (np.inf, 1.0, 4),
        (2.0, 0.0, 4),
        (2.0, -1.0, 4),
        (2.0, np.nan, 4),
        (2.0, np.inf, 4),
        (1e12, 1e12, 0),  # the series needs about 9 sqrt(a) terms, past MAX_TERMS
        (2.0, 1.0, 5),
        (2.0, 1.0, -1),
    ],
)
@pytest.mark.parametrize("upper", [False, True])
def test_rows_outside_the_domain_write_nan(a: float, x: float, order: int, upper: bool) -> None:
    out = np.zeros(output_size(MAX_ORDER))
    log_gamma_tail(a, x, upper, order, out)
    assert np.all(np.isnan(out))


def test_jets_wrapper_packs_the_scalar_engine_rows() -> None:
    a = np.array([0.5, 2.0, 2.0, 50.0, 1e4])
    x = np.array([0.1, 1.0, 9.0, 80.0, 1e4])
    upper = np.array([False, True, True, False, True])
    for order in range(MAX_ORDER + 1):
        jets = log_gamma_tail_jets(a, x, upper, order)
        assert len(jets) == order + 1
        assert jets[0].shape == (a.size,)
        for k in range(1, order + 1):
            assert jets[k].shape == (a.size, k + 1)
        for row in range(a.size):
            packed = np.concatenate([np.atleast_1d(jet[row]) for jet in jets])
            np.testing.assert_array_equal(packed, _engine(a[row], x[row], upper[row], order))
    broadcast = log_gamma_tail_jets(3.0, x, True, 1)
    assert broadcast[1].shape == (x.size, 2)
    with pytest.raises(ValueError, match="order"):
        log_gamma_tail_jets(a, x, upper, 5)
