"""The generalized gamma's log tail at every shape against an mpmath oracle, to order 4.

``kernels/_gg_log_tail.py`` routes each row to the uniform expansion
(``_gg_uniform_tail``, tested in ``tests/test_gg_uniform_tail.py``) inside
``gg_uniform_in_domain`` and to the incomplete gamma at ``a = 1/q^2``,
``x = a e^(q w)`` with its ``(da, e)`` jet along fixed ``x/a`` composed into
``(w, q)`` elsewhere.

Oracle.  ``mpmath.diff`` of ``log gammainc`` as a function of ``(w, q)``, at 40
digits (``mp.diff`` raises the working precision for its own stencil), with the
lower and upper incomplete gamma swapped for ``q < 0`` and the larger tail as
``log1p`` of minus the smaller, which keeps it where the smaller tail is far
below ``10^-40``.  It shares no formula with either engine and is checked
below against the uniform tail's ``q``-stencil reference.

Error model of the composed route.  Order ``d`` is compared relative to the
row's largest same-order oracle entry ``scale_d``.  The incomplete-gamma engine's
coefficient ``c_ij`` along fixed ``x/a`` carries at most ``rho_k`` times its
scale ``max(|c_ij|, F_ij, M_k/a^i)`` (``F_ij`` the largest ``|c_p c_r|`` with
``p + r = (i, j)``, ``M_k`` the row's largest ``|c| a^i`` of order ``k``; the
envelope is stated in ``tests/test_incomplete_gamma_engine.py``), plus ``4 u
|c_ij|``.  Linearised through the composition ``sum c_ij da^i e^j``, each error
enters with the jet ``|da|^i |e|^j`` of absolute coefficients.  The rounding of
``a = 1/q^2`` (``2 u a``) and of ``x`` (``(3 + |q w|) u`` relative, so ``(5 + |q
w|) u`` in ``x/a``) shifts the expansion point, which enters as ``sum |c_ij|
((|da| + 2 u a)^i (|e| + delta)^j - |da|^i |e|^j)``.  The bound is that sum's
order-``d`` partials, times 4 for libm variation.

Near the diagonal (``|eta| < 1``) the composition's terms still exceed its
result by about ``a^(d/2)`` (at fixed ``x`` it was ``a^d``), so the body at
``0.2887 < |q| < 0.5`` (``4 < a < 12``) is the composed route's least accurate
region: measured worst 1.6e-13, 6.2e-12 and 6.4e-11 of the scale at orders 2,
3 and 4 (1.2e-12, 5.7e-11 and 1.2e-8 at fixed ``x``).  Composing the exact
``c_ij`` rounded once to float64 gives up to 6.5e-12 at order 4; the rest is
the engine's coefficient error of a few tens of ``u``, amplified by the same
terms.  ``test_body_between_the_engines_keeps_order_four_within_2e_10`` holds
the region to about three times the measurement.

Continuity.  At adjacent floats either side of the uniform domain's edge (one
row per engine), every partial agrees within the sum of the two error models.

Entries below float64's normal range compare absolutely.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from scipy.special import gammainc, gammaincc  # noqa: E402

from superglm.distributional.kernels import _gg_log_tail as engine  # noqa: E402
from superglm.distributional.kernels import _gg_uniform_tail as uniform  # noqa: E402
from superglm.distributional.kernels.generalized_gamma import (  # noqa: E402
    generalized_gamma_cdf,
)
from tests.test_gg_uniform_tail import (  # noqa: E402
    _float_mul,
    edge_t,
    gammainc_reference,
    plain_partials,
    rounding_bound,
    truncation_majorant,
    zeta_jet,
)

U = 2.0**-53
ORDER = 4
TINY = 2.0**-1022
DIGITS = 40
KEYS = tuple((d - j, j) for d in range(ORDER + 1) for j in range(d + 1))
INDEX = {key: r for r, key in enumerate(KEYS)}
BLOCKS = tuple(slice(d * (d + 1) // 2, (d + 1) * (d + 2) // 2) for d in range(ORDER + 1))
FACTORIALS = np.array([math.factorial(i) * math.factorial(j) for i, j in KEYS], dtype=float)
RHO_ALONG_RATIO = np.array([4e3, 4e3, 4e3, 3e4, 3e4]) * U
LIBM_SLACK = 4.0


# --- oracle -------------------------------------------------------------------------


def reference(w, q):
    """Plain ``(w, q)`` partials of ``(log F, log S)`` by ``mp.diff`` of ``log gammainc``."""
    memo = {}

    def tails(w_value, q_value):
        key = (w_value, q_value, mp.mp.prec)
        if key not in memo:
            a = 1 / (q_value * q_value)
            x = a * mp.exp(q_value * w_value)
            lower = mp.gammainc(a, 0, x, regularized=True)
            upper = mp.gammainc(a, x, mp.inf, regularized=True)
            if q_value < 0:
                lower, upper = upper, lower
            if lower <= upper:
                memo[key] = (mp.log(lower), mp.log1p(-lower))
            else:
                memo[key] = (mp.log1p(-upper), mp.log(upper))
        return memo[key]

    with mp.workdps(DIGITS):
        point = (mp.mpf(w), mp.mpf(q))
        return tuple(
            np.array([float(mp.diff(lambda u, v: tails(u, v)[side], point, key)) for key in KEYS])
            for side in (0, 1)
        )


def partials(w, q, upper, order=ORDER):
    out = np.empty((order + 1) * (order + 2) // 2)
    engine.gg_log_tail(w, q, upper, order, out, np.empty(engine.SCRATCH_SHAPE))
    return out


def composed_partials(w, q, upper):
    """The incomplete-gamma route forced on any row, and its ``(da, e)`` jet."""
    out = np.empty(len(KEYS))
    flat = np.empty(engine.SCRATCH_SHAPE[0] * engine.SCRATCH_SHAPE[1])
    engine._composed_log_tail(w, q, upper, ORDER, out, flat)
    return out, flat[engine._JET : engine._JET + len(KEYS)].copy()


# --- error models -------------------------------------------------------------------


def _powers(jet):
    powers = [np.eye(len(KEYS))[0]]
    for _ in range(ORDER):
        powers.append(_float_mul(powers[-1], jet))
    return powers


def coefficient_scale(a, coefficients):
    """``max(|c_ij|, F_ij, M_k/a^i)``, the scale of the engine's envelope (module docstring)."""
    weight = np.array([a**i for i, _ in KEYS])
    products = np.zeros(len(KEYS))
    for r, (i, j) in enumerate(KEYS):
        for p, (ip, jp) in enumerate(KEYS):
            if 0 < p != r and ip <= i and jp <= j:
                partner = coefficients[INDEX[(i - ip, j - jp)]]
                products[r] = max(products[r], abs(coefficients[p] * partner))
    order_scale = np.empty(len(KEYS))
    for block in BLOCKS:
        order_scale[block] = np.max(np.abs(coefficients[block] * weight[block]))
    return np.maximum(np.maximum(np.abs(coefficients), products), order_scale / weight)


def composed_bound(w, q, coefficients):
    """Majorant of the composed route's error, plain ``(w, q)`` partials (module docstring)."""
    a = 1.0 / (q * q)
    rho = np.array([RHO_ALONG_RATIO[i + j] for i, j in KEYS])
    carried = rho * coefficient_scale(a, coefficients) + 4 * U * np.abs(coefficients)
    da, e = np.zeros(len(KEYS)), np.zeros(len(KEYS))
    for k in range(1, ORDER + 1):
        da[INDEX[(0, k)]] = a * (k + 1) / abs(q) ** k
    for i, j in KEYS:
        e[INDEX[(i, j)]] = sum(
            abs(q) ** (i - m)
            * abs(w) ** (j - m)
            / FACTORIALS[INDEX[(i - m, 0)]]
            / FACTORIALS[INDEX[(0, j - m)]]
            / math.factorial(m)
            for m in range(min(i, j) + 1)
        )
    e[0] = 0.0
    da_shifted, e_shifted = da.copy(), e.copy()
    da_shifted[0], e_shifted[0] = 2 * U * a, (5 + abs(q * w)) * U
    p_da, p_e, s_da, s_e = _powers(da), _powers(e), _powers(da_shifted), _powers(e_shifted)
    total = np.zeros(len(KEYS))
    for r, (i, j) in enumerate(KEYS):
        exact = _float_mul(p_da[i], p_e[j])
        total += carried[r] * exact + abs(coefficients[r]) * (_float_mul(s_da[i], s_e[j]) - exact)
    return LIBM_SLACK * total * FACTORIALS


def uniform_bound(w, q, upper, lower_values, upper_values):
    """The uniform tail's error model (``tests/test_gg_uniform_tail.py``) on its own values."""
    small_is_upper = abs(upper_values[0]) > abs(lower_values[0])
    log_small = float(upper_values[0] if small_is_upper else lower_values[0])
    zeta = float(zeta_jet(w, q)[0])
    log_bracket = log_small + 0.5 * zeta * zeta + 0.5 * math.log(2 * math.pi)
    omitted = truncation_majorant(w, q) / math.exp(log_bracket)
    widen = 1.0 if upper == small_is_upper else 1.0 + abs(log_small)
    values = upper_values if upper else lower_values
    bound = np.empty(len(KEYS))
    for order, block in enumerate(BLOCKS):
        scale = float(np.max(np.abs(values[block])))
        bound[block] = (
            rounding_bound(order, zeta) * scale + 10 * float(np.max(omitted[block]))
        ) * widen
    return bound


def failures(w, q, upper, actual, expected, bound):
    found = []
    for order, block in enumerate(BLOCKS):
        error = float(np.max(np.abs(actual[block] - expected[block])))
        allowed = float(np.max(bound[block])) + TINY
        if not error <= allowed:
            scale = float(np.max(np.abs(expected[block])))
            found.append(
                f"w={w!r} q={q!r} upper={upper} order {order}: error {error:.2e} > "
                f"{allowed:.2e} (scale {scale:.2e})"
            )
    return found


# --- grids --------------------------------------------------------------------------

BODY_Q = (0.2888, 0.32, 0.4, 0.5, 0.75, 1.0, 1.5, 5.0, -0.32, -1.5, -5.0)
BODY_W = (-20.0, -8.0, -3.0, -1.0, 0.0, 0.3, 1.0, 3.0, 8.0, 20.0)
FAR_Q = (1e-3, 0.01, -0.01, 0.1, -0.1, 0.2, -0.2)


def far_points(q):
    return [t * fraction / q for t in edge_t() for fraction in (1.0001, 3.0)]


def boundary_pair(w, q, along_w):
    """Adjacent floats straddling the uniform domain's edge from ``(w, q)`` inside outward."""
    inside, outside = (w, q), (w * 2.0, q) if along_w else (w, math.copysign(0.3, q))
    for _ in range(200):
        mid = tuple(0.5 * (p + r) for p, r in zip(inside, outside, strict=True))
        if mid in (inside, outside):
            break
        if uniform.gg_uniform_in_domain(*mid):
            inside = mid
        else:
            outside = mid
    assert uniform.gg_uniform_in_domain(*inside) and not uniform.gg_uniform_in_domain(*outside)
    return inside, outside


# --- tests --------------------------------------------------------------------------


def test_the_oracle_agrees_with_the_uniform_tails_stencil_reference():
    worst = 0.0
    for w, q in ((0.3, 0.29), (-8.0, 0.5), (3.0, -1.5)):
        mine = reference(w, q)
        theirs = tuple(plain_partials(jet) for jet in gammainc_reference(w, q))
        for a, b in zip(mine, theirs, strict=True):
            for block in BLOCKS:
                worst = max(
                    worst, float(np.max(np.abs(a[block] - b[block])) / np.max(np.abs(b[block])))
                )
    assert worst <= 4 * U


@pytest.mark.parametrize("q", BODY_Q)
def test_composed_route_matches_the_oracle_in_the_body_and_tails(q):
    found = []
    for w in BODY_W:
        assert not uniform.gg_uniform_in_domain(w, q)
        lower, upper = reference(w, q)
        for side, expected in ((False, lower), (True, upper)):
            actual, coefficients = composed_partials(w, q, side)
            assert np.array_equal(actual, partials(w, q, side))
            found += failures(w, q, side, actual, expected, composed_bound(w, q, coefficients))
    assert not found, "\n".join(found)


BETWEEN_Q = (0.2888, 0.32, 0.4, 0.5, -0.2888, -0.45)
BETWEEN_W = (-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.0)
BETWEEN_BUDGET = (1e-13, 1e-13, 5e-13, 2e-11, 2e-10)


def test_body_between_the_engines_keeps_order_four_within_2e_10():
    """``0.2887 < |q| <= 0.5`` across the diagonal, relative to the row's same-order scale."""
    worst = np.zeros(ORDER + 1)
    for q in BETWEEN_Q:
        for w in BETWEEN_W:
            for side, expected in zip((False, True), reference(w, q), strict=True):
                actual = partials(w, q, side)
                for order, block in enumerate(BLOCKS):
                    error = np.max(np.abs(actual[block] - expected[block]))
                    worst[order] = max(worst[order], error / np.max(np.abs(expected[block])))
    assert np.all(worst <= BETWEEN_BUDGET), worst


@pytest.mark.parametrize("q", FAR_Q)
def test_composed_route_matches_the_oracle_in_far_tails_at_small_shape(q):
    found = []
    for w in far_points(q):
        assert not uniform.gg_uniform_in_domain(w, q)
        lower, upper = reference(w, q)
        for side, expected in ((False, lower), (True, upper)):
            actual, coefficients = composed_partials(w, q, side)
            found += failures(w, q, side, actual, expected, composed_bound(w, q, coefficients))
    assert not found, "\n".join(found)


def _leading_term_reference(w, q):
    """``(log F, log S)`` partials where ``x = a e^(q w)`` is below the normal range.

    There ``log P = a log x - ln Gamma(1 + a) + O(x)`` (DLMF 8.7.1), and ``x``
    is below ``1e-300``, so the leading term is exact at 40 digits; each value
    is checked against ``mpmath.gammainc`` too.
    """

    def tail(w_value, q_value, upper):
        a = 1 / (q_value * q_value)
        log_p = a * (mp.log(a) + q_value * w_value) - mp.loggamma(1 + a)
        return log_p if upper != (q_value > 0) else mp.log(-mp.expm1(log_p))

    with mp.workdps(DIGITS):
        point = (mp.mpf(w), mp.mpf(q))
        a = 1 / point[1] ** 2
        x = a * mp.exp(point[0] * point[1])
        lower = mp.gammainc(a, 0, x, regularized=True)
        if q < 0:
            lower = 1 - lower
        assert abs(mp.log(lower) - tail(*point, False)) < mp.mpf(10) ** -30
        return tuple(
            np.array([float(mp.diff(lambda u, v: tail(u, v, side), point, key)) for key in KEYS])
            for side in (False, True)
        )


@pytest.mark.parametrize(
    ("w", "q"),
    [(-40.0, 20.0), (40.0, -20.0), (-36.0, 20.0), (-0.0069, 1e5), (-2000.0, 0.5)],
)
def test_a_point_below_the_normal_range_keeps_log_x(w, q):
    """``x = a e^(q w)`` underflows or is subnormal while both tails are substantial.

    At ``(-40, 20)`` ``x = e^-800/400``: ``log F = -2.0135`` and ``log S =
    -0.1433``.  ``(-0.0069, 1e5)`` has a normal ``e^(q w)`` and a subnormal
    ``x``; ``(-2000, 0.5)`` a subnormal ``e^(q w)``.  The route keeps ``log x``
    and its error is the composition's (``composed_bound``).
    """
    expected = _leading_term_reference(w, q)
    for side in (False, True):
        actual, coefficients = composed_partials(w, q, side)
        assert np.all(np.isfinite(actual))
        assert not failures(w, q, side, actual, expected[side], composed_bound(w, q, coefficients))
        np.testing.assert_array_equal(partials(w, q, side), actual)


@pytest.mark.parametrize(
    ("w", "q", "along_w"),
    [
        (-3.0, 0.2887, False),
        (0.0, 0.2887, False),
        (2.0, -0.2887, False),
        (-1.0, -0.2887, False),
        *(
            (t * 0.99 / q, q, True)
            for q in (0.01, -0.01, 0.2, -0.2, 0.2887)
            for t in (-1.198, 0.85)
        ),
    ],
)
def test_partials_are_continuous_across_the_routing_boundary(w, q, along_w):
    """Adjacent floats either side of the uniform domain's edge, one per engine, agree."""
    inside, outside = boundary_pair(w, q, along_w)
    found = []
    lower_in, upper_in = partials(*inside, False), partials(*inside, True)
    for side, inner in ((False, lower_in), (True, upper_in)):
        out = np.empty(len(KEYS))
        uniform.gg_uniform_log_tail(*inside, side, ORDER, out)
        assert np.array_equal(inner, out)
        outer, coefficients = composed_partials(*outside, side)
        assert np.array_equal(outer, partials(*outside, side))
        bound = composed_bound(*outside, coefficients) + uniform_bound(
            *inside, side, lower_in, upper_in
        )
        found += failures(*outside, side, outer, inner, bound)
    assert not found, "\n".join(found)


def test_routing_is_the_uniform_engine_inside_its_domain():
    rng = np.random.default_rng(5)
    q = rng.uniform(-0.2887, 0.2887, 400)
    w = rng.uniform(-30.0, 30.0, 400)
    inside = uniform.gg_uniform_domain(w, q)
    assert 50 < inside.sum() < 350
    for upper in (False, True):
        ours = engine.gg_log_tail_jets(w[inside], q[inside], upper=upper, order=ORDER)
        theirs = uniform.gg_uniform_log_tail_jets(w[inside], q[inside], upper=upper, order=ORDER)
        for a, b in zip(ours, theirs, strict=True):
            assert np.array_equal(a, b)


def test_reflection_swaps_the_tails_with_partial_signs():
    """``W(-q) = -W(q)``: ``log F(w, q) = log S(-w, -q)``, partial ``(i, j)`` times ``(-1)^(i+j)``."""
    signs = np.array([(-1.0) ** (i + j) for i, j in KEYS])
    for w, q in ((0.4, 0.6), (-3.0, 1.5), (12.0, 0.1), (-2.0, 4.0)):
        assert not uniform.gg_uniform_in_domain(w, q)
        for upper in (False, True):
            assert np.array_equal(partials(w, q, upper), signs * partials(-w, -q, not upper))


def test_values_match_scipy_incomplete_gamma():
    rng = np.random.default_rng(11)
    q = rng.uniform(-3.0, 3.0, 3000)
    w = rng.uniform(-6.0, 6.0, 3000)
    a = 1.0 / (q * q)
    x = a * np.exp(q * w)
    lower, upper = gammainc(a, x), gammaincc(a, x)
    lower, upper = np.where(q > 0, lower, upper), np.where(q > 0, upper, lower)
    for side, tail in ((False, lower), (True, upper)):
        value = engine.gg_log_tail_jets(w, q, upper=side, order=0)[0]
        usable = tail > 1e-300
        assert usable.sum() > 2500
        error = np.abs(value[usable] - np.log(tail[usable]))
        assert np.max(error / np.maximum(1.0, np.abs(value[usable]))) <= 1e-12


def test_cdf_matches_the_generalized_gamma_cdf():
    rng = np.random.default_rng(3)
    q = rng.uniform(-2.0, 2.0, 2000)
    w = rng.uniform(-5.0, 5.0, 2000)
    y = np.exp(w)
    cdf = generalized_gamma_cdf(y, np.zeros_like(w), np.ones_like(w), q)
    ours = np.exp(engine.gg_log_tail_jets(np.log(y), q, upper=False, order=0)[0])
    usable = (cdf > 1e-200) & (np.abs(q) > 1e-6)
    assert usable.sum() > 1500
    assert np.max(np.abs(ours[usable] - cdf[usable]) / cdf[usable]) <= 1e-12


@pytest.mark.parametrize(
    ("w", "q", "order"),
    [
        (math.nan, 0.5, 4),
        (1.0, math.nan, 4),
        (math.inf, 0.5, 2),
        (1.0, -math.inf, 2),
        (2.0**101, 0.0, 4),
        (1e200, 1e-160, 4),
        (200.0, 5.0, 4),
        (-200.0, -5.0, 4),
        (1.0, 0.5, 5),
        (1.0, 0.5, -1),
    ],
)
def test_rows_neither_engine_evaluates_write_nan(w, q, order):
    for upper in (False, True):
        out = np.zeros(15)
        engine.gg_log_tail(w, q, upper, order, out, np.empty(engine.SCRATCH_SHAPE))
        assert np.all(np.isnan(out))


def test_vector_wrapper_is_an_i1_jet_of_the_scalar_engine():
    w = np.array([-20.0, -1.0, 0.0, 0.5, 3.0, 40.0])
    q = np.array([0.6, -0.1, 0.0, 1.5, -2.0, 0.05])
    upper = np.array([True, False, True, False, True, False])
    for order in range(ORDER + 1):
        jets = engine.gg_log_tail_jets(w, q, upper=upper, order=order)
        assert len(jets) == order + 1
        assert jets[0].shape == (len(w),)
        for k in range(1, order + 1):
            assert jets[k].shape == (len(w), k + 1)
        for row in range(len(w)):
            scalar = partials(w[row], q[row], bool(upper[row]), order)
            assert np.array_equal(np.concatenate([np.atleast_1d(j[row]) for j in jets]), scalar)
    with pytest.raises(ValueError, match="order"):
        engine.gg_log_tail_jets(w, q, upper=False, order=5)
