"""The generalized gamma's uniform log tail against two independent mpmath references.

The engine (``kernels/_gg_uniform_tail.py``) evaluates Temme's uniform expansion
in the GG's own variables.  Neither reference uses that expansion:

- **The q-expansion of the density** (``|q| <= 1e-4``, including ``q = 0``).
  Prentice's density is ``f_W(v; q) = phi(v) exp(-R(v, q))/Gamma*(1/q^2)`` with
  ``R = sum_{j>=1} q^j v^(j+2)/(j+2)!``.  Expanding ``exp(-R)`` and Stirling's
  series for ``1/Gamma*`` in powers of ``q`` and integrating the Gaussian
  moments exactly gives ``F(w; q) = Phi(w) + phi(w) sum_k q^k Q_k(w)`` with exact
  rational polynomials ``Q_k`` (``Q_1 = (w^2 + 2)/6``); the ``Phi`` coefficient of
  every ``k >= 1`` cancels exactly, which the builder asserts.  At fixed ``w``
  the terms fall like ``(|q| |w|^3/6)^k/k!``, so 40 terms are used where that
  ratio is at most 1.2.
- **The incomplete gamma** (every other point).  ``mpmath.gammainc`` gives the
  smaller tail at ``a = 1/q^2``, ``x = e^(q w)/q^2`` on nine nodes ``q + k h``.
  At each node the closed-form density gives the tail's Taylor series in ``w``,
  whose logarithm is taken in mpmath, and the degree-8 interpolant through the
  nodes gives the ``q`` coefficients of ``log T``.  The step
  ``h = 1e-5 min(|q|, 1/(1 + |w|))`` sits well inside the ``q``-radius of
  ``log T`` (``|q|``, from the essential singularity at ``q = 0``, and ``1/|w|``,
  from ``t = q w``); ``90 + 2 log10(a)`` digits cover the stencil's ``h^-4`` and
  gammainc's own cancellation at large ``a``.

The two references agree to better than ``1e-25`` where both apply (checked below).

Error model.  Order ``d`` is compared relative to the row's largest same-order
reference entry ``scale_d`` (the convention of
``tests/test_family_derivatives_symbolic.py``).  For the smaller tail::

    |error_d| <= rho_d scale_d + 10 max_order_d |partial of q R_N| / B,
    rho_d = 32 (d + 1) u (1 + min(|zeta|, 4)^(2 d)/d!).

``rho_d`` is rounding: a few ``u`` for each of some thirty jet operations, a
growing share of which cancel as the order rises (the fourth derivative of
``log M`` at ``zeta = 0`` is 0.12, from terms near 5), and below ``t = 4`` the forward
Mills recurrence's loss, the factor in brackets.  The largest measured over this
grid is ``4 u`` at order 0 and ``48 u`` at order 4.  The second term is
truncation: ``R_N = sum_{m=N+1}^{N+3} d_{m+1} sum_k m!!/(m - 2k)!! s^k eta^(m-2k)``
(``s = q^2``, ``eta = q zeta``) is the first omitted part of GST's series, whose
partials are bounded by evaluating it on jets with absolute coefficients (a
majorant), and ``B = M(xi) - eps q S`` is the bracket it enters, read from the
reference.  Measured against this grid at ``N = 25``, 30 and 35, every error is
within 0.3 of its bound, and where truncation dominates the bound is 9 to 14
times the error.  ``N = 35`` is the documented term count, written here rather
than read from the engine, so a shorter recursion fails.  The larger tail is
``log(1 - e^L)``, which inherits ``L``'s absolute error, so its bound is
multiplied by ``1 + |L|``.  Entries below float64's normal range compare
absolutely.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import cache

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

from superglm.distributional.kernels import _gg_uniform_tail as engine  # noqa: E402
from superglm.distributional.kernels._response_tails import (  # noqa: E402
    normal_log_cdf_channels,
)

U = 2.0**-53
ORDER = 4
DOCUMENTED_TERMS = 35
KEYS = tuple((d - j, j) for d in range(ORDER + 1) for j in range(d + 1))
BLOCKS = tuple(slice(d * (d + 1) // 2, (d + 1) * (d + 2) // 2) for d in range(ORDER + 1))
EXPANSION_TERMS = 40
TINY = 2.0**-1022


# --- exact coefficients -------------------------------------------------------------


def _series_mul(a, b):
    out = [Fraction(0)] * len(a)
    for i, x in enumerate(a):
        if x:
            for j in range(len(a) - i):
                out[i + j] += x * b[j]
    return out


def _series_exp(a):
    """exp of a series with zero constant: ``n e_n = sum_k k a_k e_{n-k}``."""
    e = [Fraction(1)] + [Fraction(0)] * (len(a) - 1)
    for n in range(1, len(a)):
        e[n] = sum(k * a[k] * e[n - k] for k in range(1, n + 1)) / n
    return e


def _series_log(a):
    """log of a series with ``a_0 = 1``: ``n l_n = n a_n - sum_k k l_k a_{n-k}``."""
    out = [Fraction(0)] * len(a)
    for n in range(1, len(a)):
        out[n] = (n * a[n] - sum(k * out[k] * a[n - k] for k in range(1, n))) / n
    return out


@cache
def exact_coefficients(n):
    """``(d_0..d_{n-1}, G_0..G_{n-1})``: Lagrange-Buermann inversion of ``eta = t G(t)``.

    ``eta/(lambda - 1) = h(t) = G(t) t/(e^t - 1)``, and for ``n >= 1``
    ``[eta^n] h(t(eta)) = [t^(n-1)] h'(t) G(t)^(-n) / n``.
    """
    log_g = _series_log([Fraction(2, math.factorial(j + 2)) for j in range(n)])
    big_g = _series_exp([x / 2 for x in log_g])
    expm1_over_t = [Fraction(1, math.factorial(j + 1)) for j in range(n)]
    bernoulli = [Fraction(1)] + [Fraction(0)] * (n - 1)
    for k in range(1, n):
        bernoulli[k] = -sum(expm1_over_t[i] * bernoulli[k - i] for i in range(1, k + 1))
    h = _series_mul(big_g, bernoulli)
    h_prime = [h[k + 1] * (k + 1) for k in range(n - 1)] + [Fraction(0)]
    d = [h[0]]
    for m in range(1, n):
        power = _series_exp([-Fraction(m, 2) * x for x in log_g])
        d.append(sum(h_prime[i] * power[m - 1 - i] for i in range(m)) / m)
    return d, big_g


# --- reference A: the q-expansion of the density at fixed w -------------------------


def _poly_add(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def _poly_mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _gaussian_operator(poly):
    """``Q' - w Q``: ``d/dw [phi(w) Q(w)] = phi(w) (Q' - w Q)``."""
    derivative = [poly[i + 1] * (i + 1) for i in range(len(poly) - 1)] or [Fraction(0)]
    return _poly_add(derivative, [Fraction(0)] + [-c for c in poly])


@cache
def q_expansion(terms):
    """``chains[i][k] = D^i Q_k`` (``D`` the Gaussian operator), ``ones[i] = D^i 1``."""
    remainder = {
        j: [Fraction(0)] * (j + 2) + [Fraction(1, math.factorial(j + 2))]
        for j in range(1, terms + 1)
    }
    density = [[Fraction(1)]]
    for k in range(1, terms + 1):
        total = [Fraction(0)]
        for j in range(1, k + 1):
            total = _poly_add(total, [c * j for c in _poly_mul(remainder[j], density[k - j])])
        density.append([-c / k for c in total])
    # 1/Gamma*(1/q^2) = exp(-sum_k B_2k/(2k(2k-1)) q^(4k-2)), Stirling's series in q.
    bernoulli = [Fraction(1)] + [Fraction(0)] * (terms + 2)
    for m in range(1, terms + 3):
        bernoulli[m] = -sum(Fraction(math.comb(m + 1, j)) * bernoulli[j] for j in range(m)) / (
            m + 1
        )
    stirling = [Fraction(0)] * (terms + 1)
    for k in range(1, terms):
        if 4 * k - 2 <= terms:
            stirling[4 * k - 2] = bernoulli[2 * k] / (2 * k * (2 * k - 1))
    inverse_gamma_star = _series_exp([-x for x in stirling])
    degree = 3 * terms + 2
    moment_phi = [Fraction(0)] * (degree + 1)
    moment_poly = [[Fraction(0)] for _ in range(degree + 1)]
    moment_phi[0], moment_poly[1] = Fraction(1), [Fraction(-1)]
    for i in range(2, degree + 1):
        moment_phi[i] = (i - 1) * moment_phi[i - 2]
        moment_poly[i] = _poly_add(
            [Fraction(0)] * (i - 1) + [Fraction(-1)], [c * (i - 1) for c in moment_poly[i - 2]]
        )
    polys = []
    for k in range(terms + 1):
        phi_coefficient, poly = Fraction(0), [Fraction(0)]
        for ell in range(k + 1):
            factor = inverse_gamma_star[k - ell]
            if factor:
                for i, c in enumerate(density[ell]):
                    if c:
                        phi_coefficient += factor * c * moment_phi[i]
                        poly = _poly_add(poly, [factor * c * x for x in moment_poly[i]])
        assert phi_coefficient == (1 if k == 0 else 0)
        polys.append(poly)
    assert polys[1] == [Fraction(1, 3), 0, Fraction(1, 6)]
    chains, ones = [polys], [[Fraction(1)]]
    for _ in range(ORDER):
        chains.append([_gaussian_operator(p) for p in chains[-1]])
        ones.append(_gaussian_operator(ones[-1]))
    return chains, ones


def _evaluate(poly, w):
    total = mp.mpf(0)
    for c in reversed(poly):
        total = total * w + mp.mpf(c.numerator) / c.denominator
    return total


def _jet_mul(a, b):
    out = {key: mp.mpf(0) for key in KEYS}
    for (i1, j1), x in a.items():
        for (i2, j2), y in b.items():
            if i1 + i2 + j1 + j2 <= ORDER:
                out[(i1 + i2, j1 + j2)] += x * y
    return out


def _jet_log(jet, log_value):
    """Taylor coefficients of ``log`` of a bivariate jet whose value's log is ``log_value``."""
    ratio = {
        key: (value / jet[(0, 0)] if key != (0, 0) else mp.mpf(0)) for key, value in jet.items()
    }
    out = {key: mp.mpf(0) for key in KEYS}
    power = {key: mp.mpf(1 if key == (0, 0) else 0) for key in KEYS}
    for n in range(1, ORDER + 1):
        power = _jet_mul(power, ratio)
        for key in KEYS:
            out[key] += (-1) ** (n + 1) * power[key] / n
    out[(0, 0)] = log_value
    return out


def expansion_reference(w, q):
    """``(log F, log S)`` Taylor coefficients in ``(dw, dq)`` from reference A."""
    with mp.workdps(70):
        return _expansion_reference(w, q)


def _expansion_reference(w, q):
    chains, ones = q_expansion(EXPANSION_TERMS)
    w, q = mp.mpf(w), mp.mpf(q)
    phi = mp.npdf(w)
    sums = {}
    for i, j in KEYS:
        total = sum(
            mp.ff(k, j) * q ** (k - j) * _evaluate(chains[i][k], w)
            for k in range(max(j, 1), EXPANSION_TERMS + 1)
        )
        if j == 0 and i >= 1:
            total += _evaluate(ones[i - 1], w)
        sums[(i, j)] = phi * total / (math.factorial(i) * math.factorial(j))
    lower = dict(sums)
    lower[(0, 0)] = mp.ncdf(w) + sums[(0, 0)]
    upper = {key: -value for key, value in sums.items()}
    upper[(0, 0)] = mp.ncdf(-w) - sums[(0, 0)]
    if lower[(0, 0)] <= upper[(0, 0)]:
        logs = mp.log(lower[(0, 0)]), mp.log1p(-lower[(0, 0)])
    else:
        logs = mp.log1p(-upper[(0, 0)]), mp.log(upper[(0, 0)])
    return _jet_log(lower, logs[0]), _jet_log(upper, logs[1])


# --- reference B: gammainc on a q-stencil of log T ----------------------------------


@cache
def _stencil_inverse():
    """Exact inverse of the Vandermonde matrix on the nodes ``-4..4``."""
    n = 9
    rows = [
        [Fraction(x) ** j for j in range(n)] + [Fraction(int(i == r)) for i in range(n)]
        for r, x in enumerate(range(-4, 5))
    ]
    for c in range(n):
        pivot = next(r for r in range(c, n) if rows[r][c] != 0)
        rows[c], rows[pivot] = rows[pivot], rows[c]
        rows[c] = [x / rows[c][c] for x in rows[c]]
        for r in range(n):
            if r != c and rows[r][c] != 0:
                rows[r] = [x - rows[r][c] * y for x, y in zip(rows[r], rows[c], strict=True)]
    return [row[n:] for row in rows]


def _univariate_log(series, log_value):
    out = [mp.mpf(0)] * len(series)
    ratio = [c / series[0] for c in series]
    for k in range(1, len(series)):
        out[k] = (k * ratio[k] - sum(m * out[m] * ratio[k - m] for m in range(1, k))) / k
    out[0] = log_value
    return out


def _tails(w, q):
    a = 1 / (q * q)
    x = mp.exp(q * w) * a
    lower = mp.gammainc(a, 0, x, regularized=True)
    upper = mp.gammainc(a, x, mp.inf, regularized=True)
    return (lower, upper) if q > 0 else (upper, lower)


def _density_taylor_in_w(w, q):
    """Taylor coefficients in ``dw`` of the Prentice density ``f_W(w + dw; q)``, orders 0-3."""
    a = 1 / (q * q)
    growth = mp.exp(q * w)
    log_density = mp.log(abs(q)) + a * mp.log(a) - mp.loggamma(a) + a * (q * w - growth)
    log_series = [mp.mpf(0), (1 - growth) / q] + [
        -growth * q ** (m - 2) / math.factorial(m) for m in range(2, ORDER)
    ]
    series = [mp.mpf(1)] + [mp.mpf(0)] * (ORDER - 1)
    for k in range(1, ORDER):
        series[k] = sum(m * log_series[m] * series[k - m] for m in range(1, k + 1)) / k
    return [mp.exp(log_density) * c for c in series]


def gammainc_reference(w, q):
    """``(log F, log S)`` Taylor coefficients in ``(dw, dq)`` from reference B."""
    q_value = mp.mpf(q)
    with mp.workdps(90 + int(2 * mp.log10(1 / q_value**2))):
        w, q = mp.mpf(w), mp.mpf(q)
        h = mp.mpf("1e-5") * min(abs(q), 1 / (1 + abs(w)))
        lower0, upper0 = _tails(w, q)
        small_is_lower = lower0 <= upper0
        nodes_small, nodes_large = [], []
        for k in range(-4, 5):
            node = q + k * h
            lower, upper = _tails(w, node)
            small = lower if small_is_lower else upper
            sign = 1 if small_is_lower else -1
            density = _density_taylor_in_w(w, node)
            series = [small] + [sign * density[i - 1] / i for i in range(1, ORDER + 1)]
            nodes_small.append(_univariate_log(series, mp.log(small)))
            large = [1 - small] + [-c for c in series[1:]]
            nodes_large.append(_univariate_log(large, mp.log1p(-small)))
        inverse = _stencil_inverse()

        def stencil(nodes):
            return {
                (i, j): sum(
                    mp.mpf(inverse[j][k].numerator) / inverse[j][k].denominator * nodes[k][i]
                    for k in range(9)
                )
                / h**j
                for i, j in KEYS
            }

        small, large = stencil(nodes_small), stencil(nodes_large)
        return (small, large) if small_is_lower else (large, small)


def reference(w, q):
    if q == 0.0 or (abs(q) <= 1e-4 and abs(q) * abs(w) ** 3 / 6 <= 1.2):
        return expansion_reference(w, q)
    return gammainc_reference(w, q)


def plain_partials(jet):
    return np.array([float(jet[(i, j)] * math.factorial(i) * math.factorial(j)) for i, j in KEYS])


# --- the grid and the error model ---------------------------------------------------

Q_GRID = (
    0.0,
    1e-8,
    -1e-8,
    1e-4,
    -1e-4,
    0.01,
    -0.01,
    0.05,
    -0.05,
    0.1,
    -0.1,
    0.2,
    -0.2,
    0.2887,
    -0.2887,
)
W_GRID = (-200.0, -37.0, -20.0, -8.0, -3.0, -1.0, 0.0, 0.7, 2.0, 5.0, 12.0, 25.0, 37.0, 200.0)


@cache
def edge_t():
    """``t = q w`` at ``eta = -1`` and ``eta = +1``: ``e^t - 1 - t = 1/2``."""
    return tuple(
        float(mp.findroot(lambda t: mp.expm1(t) - t - mp.mpf(1) / 2, start))
        for start in (-1.2, 0.86)
    )


def grid_points(q):
    low, high = edge_t()
    points = []
    for w in W_GRID:
        inside = q == 0.0 or low < q * w < high
        reachable = abs(q) >= 0.01 or abs(q) * abs(w) ** 3 / 6 <= 1.2 or abs(q) == 1e-8
        if inside and reachable:
            points.append(w)
    if q != 0.0:
        points += [t * fraction / q for t in (low, high) for fraction in (1 - 1e-9, 0.9)]
    return points


def _float_jet_tables():
    triples = [
        (KEYS.index((i1 + i2, j1 + j2)), n1, n2)
        for n1, (i1, j1) in enumerate(KEYS)
        for n2, (i2, j2) in enumerate(KEYS)
        if i1 + i2 + j1 + j2 <= ORDER
    ]
    out, left, right = (np.array(column) for column in zip(*triples, strict=True))
    factorials = np.array([math.factorial(i) * math.factorial(j) for i, j in KEYS], dtype=float)
    return out, left, right, factorials


_FLOAT_OUT, _FLOAT_LEFT, _FLOAT_RIGHT, _FACTORIALS = _float_jet_tables()


def _float_mul(a, b):
    return np.bincount(_FLOAT_OUT, weights=a[_FLOAT_LEFT] * b[_FLOAT_RIGHT], minlength=len(KEYS))


def zeta_jet(w, q):
    """Taylor coefficients of ``zeta = w G(q w)`` in ``(dw, dq)``, from the exact ``G`` series."""
    _, exact_g = exact_coefficients(40)
    g_series = [float(x) for x in exact_g]
    t = q * w
    taylor = [
        sum(math.comb(n, k) * g_series[n] * t ** (n - k) for n in range(k, 40))
        for k in range(ORDER + 1)
    ]
    delta = np.zeros(len(KEYS))
    delta[KEYS.index((1, 0))], delta[KEYS.index((0, 1))], delta[KEYS.index((1, 1))] = q, w, 1.0
    g_jet, power = np.zeros(len(KEYS)), np.eye(len(KEYS))[0]
    for k in range(ORDER + 1):
        g_jet += taylor[k] * power
        power = _float_mul(power, delta)
    w_jet = np.zeros(len(KEYS))
    w_jet[0], w_jet[KEYS.index((1, 0))] = w, 1.0
    return _float_mul(w_jet, g_jet)


def truncation_majorant(w, q, terms=DOCUMENTED_TERMS):
    """Majorant of the plain ``(w, q)`` partials of the first omitted terms of ``q T``.

    ``q T_exact - q T_N`` is led by ``q sum_{m=N+1}^{N+3} d_{m+1} sum_k m!!/(m - 2k)!!
    s^k eta^(m-2k)`` (``s = q^2``, ``eta = q zeta``); replacing ``q``, ``zeta`` and
    ``d_{m+1}`` by jets and values with absolute coefficients bounds every partial.
    """
    d, _ = exact_coefficients(terms + 6)
    q_jet = np.zeros(len(KEYS))
    q_jet[0], q_jet[KEYS.index((0, 1))] = abs(q), 1.0
    eta = _float_mul(q_jet, np.abs(zeta_jet(w, q)))
    s = _float_mul(q_jet, q_jet)
    eta_powers, s_powers = [np.eye(len(KEYS))[0]], [np.eye(len(KEYS))[0]]
    for _ in range(terms + 3):
        eta_powers.append(_float_mul(eta_powers[-1], eta))
        s_powers.append(_float_mul(s_powers[-1], s))
    total = np.zeros(len(KEYS))
    for m in range(terms + 1, terms + 4):
        falling = 1.0
        for k in range(m // 2 + 1):
            term = falling * _float_mul(s_powers[k], eta_powers[m - 2 * k])
            total += abs(float(d[m + 1])) * term
            falling *= m - 2 * k
    return _float_mul(q_jet, total) * _FACTORIALS


def rounding_bound(order, zeta):
    """``rho_d``: the rounding term of the error model."""
    return 32 * (order + 1) * U * (1 + min(abs(zeta), 4.0) ** (2 * order) / math.factorial(order))


def engine_partials(w, q, upper, order=ORDER):
    out = np.empty((order + 1) * (order + 2) // 2)
    engine.gg_uniform_log_tail(w, q, upper, order, out)
    return out


def row_failures(w, q, lower_jet, upper_jet):
    lower_ref, upper_ref = plain_partials(lower_jet), plain_partials(upper_jet)
    small_is_upper = abs(upper_ref[0]) > abs(lower_ref[0])
    log_small = float((upper_jet if small_is_upper else lower_jet)[(0, 0)])
    zeta = float(zeta_jet(w, q)[0])
    # B = (smaller tail)/phi(zeta), so log B = log_small + zeta^2/2 + log(2 pi)/2.
    log_bracket = log_small + 0.5 * zeta * zeta + 0.5 * math.log(2 * math.pi)
    omitted = truncation_majorant(w, q) / math.exp(log_bracket)
    failures = []
    for upper, expected in ((False, lower_ref), (True, upper_ref)):
        actual = engine_partials(w, q, upper)
        widen = 1.0 if upper == small_is_upper else 1.0 + abs(log_small)
        for order, block in enumerate(BLOCKS):
            scale = float(np.max(np.abs(expected[block])))
            error = float(np.max(np.abs(actual[block] - expected[block])))
            truncation = 10 * float(np.max(omitted[block]))
            bound = (rounding_bound(order, zeta) * scale + truncation) * widen + TINY
            if not error <= bound:
                failures.append(
                    f"w={w!r} q={q!r} upper={upper} order {order}: error {error:.2e} > "
                    f"{bound:.2e} (scale {scale:.2e})"
                )
    return failures


# --- tests --------------------------------------------------------------------------


def test_coefficients_are_the_exact_series_rounded_once():
    d, big_g = exact_coefficients(len(engine._D))
    assert [float(x) for x in d] == engine._D.tolist()
    assert [float(x) for x in big_g[: len(engine._G)]] == engine._G.tolist()
    assert d[:5] == [1, Fraction(-1, 3), Fraction(1, 12), Fraction(-2, 135), Fraction(1, 864)]
    assert len(engine._D) == engine.TERMS + 2


@pytest.mark.parametrize("a", [12.0, 30.0, 1000.0])
def test_d2_is_plus_one_twelfth_because_gamma_star_is_one_plus_beta1_over_a(a):
    """GST's ``Gamma*(a) = 1 + beta_1/a`` with the engine's ``d_n`` and recursion, against mpmath.

    ``beta_1 = sum_k d_{2k+2} (2k+1)!! a^-k``, so ``d_2`` is Stirling's ``1/12``;
    GST's printed ``d_2 = -1/12`` would miss by ``1/(6 a)``.
    """
    beta = [0.0] * (engine.TERMS + 3)
    for n in range(engine.TERMS, -1, -1):
        beta[n] = engine._D[n + 1] + (n + 2) * beta[n + 2] / a
    with mp.workdps(40):
        gamma_star = mp.gamma(a) / (mp.sqrt(2 * mp.pi / a) * (a / mp.e) ** a)
    assert abs(1.0 + beta[1] / a - float(gamma_star)) <= 8 * U


def test_the_two_references_agree():
    worst = 0.0
    for w, q in ((-3.0, 0.01), (0.5, 0.01), (2.5, -0.01), (-1.0, -0.01)):
        for a, b in zip(expansion_reference(w, q), gammainc_reference(w, q), strict=True):
            pa, pb = plain_partials(a), plain_partials(b)
            for block in BLOCKS:
                worst = max(
                    worst, float(np.max(np.abs(pa[block] - pb[block])) / np.max(np.abs(pb[block])))
                )
    assert worst <= 1e-25


@pytest.mark.parametrize("q", Q_GRID)
def test_log_tails_match_the_references_to_order_four(q):
    failures = []
    for w in grid_points(q):
        failures += row_failures(w, q, *reference(w, q))
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("w", [-37.0, -8.0, -2.5, 0.0, 0.7, 1.9, 6.0, 37.0])
def test_q_derivative_at_zero_is_phi_w_squared_plus_two_over_six(w):
    """``dF/dq = phi(w)(w^2 + 2)/6`` at ``q = 0``: read as ``d log F/dq`` and ``d log S/dq``."""
    with mp.workdps(40):
        slope = mp.npdf(w) * (w * w + 2) / 6
        expected_lower = float(slope / mp.ncdf(w))
        expected_upper = float(-slope / mp.ncdf(-w))
    for upper, expected in ((False, expected_lower), (True, expected_upper)):
        actual = engine_partials(w, 0.0, upper, order=1)[2]
        widen = 1.0 + abs(engine_partials(w, 0.0, not upper, 0)[0])
        assert abs(actual - expected) <= 64 * U * widen * abs(expected) + TINY


def _log_phi(x):
    """``log Phi(x)``, the larger tail through ``log1p`` so it keeps its digits."""
    return mp.log(mp.ncdf(x)) if x <= 0 else mp.log1p(-mp.ncdf(-x))


@pytest.mark.parametrize("w", [-200.0, -37.0, -3.5, -0.4, 0.0, 1.3, 4.5, 37.0])
def test_q_zero_is_the_log_normal(w):
    """At ``q = 0`` the value and every ``w`` partial are ``log Phi``'s, and match the log-normal kernel.

    Each order is held to the grid's error model, relative to the row's largest
    same-order entry (a ``q`` partial from order 2 on).
    """
    pure_w = [d * (d + 1) // 2 for d in range(ORDER + 1)]
    with mp.workdps(50):
        log_small = float(abs(mp.log(mp.ncdf(-abs(w)))))
    for upper in (False, True):
        sign = -1.0 if upper else 1.0
        widen = 1.0 if sign * w <= 0 else 1.0 + log_small
        partials = engine_partials(w, 0.0, upper)
        actual = partials[pure_w]
        with mp.workdps(50):
            expected = [
                float(mp.diff(lambda x, sign=sign: _log_phi(sign * x), w, d))
                for d in range(ORDER + 1)
            ]
        for d in range(ORDER + 1):
            scale = max(abs(expected[d]), float(np.max(np.abs(partials[BLOCKS[d]]))))
            bound = rounding_bound(d, w) * widen * scale + TINY
            assert abs(actual[d] - expected[d]) <= bound, (w, upper, d, actual[d], expected[d])
        value, score, hessian = normal_log_cdf_channels(
            np.array([sign * w]), np.array([[sign]]), np.zeros((1, 1)), derivative_order=2
        )
        kernel = np.array([value[0], score[0, 0], hessian[0, 0]])
        np.testing.assert_allclose(actual[:3], kernel, rtol=64 * U * widen, atol=TINY)


@pytest.mark.parametrize("w", [-30.0, -2.0, 0.3, 4.0])
def test_partials_are_analytic_through_q_zero(w):
    """Every partial at ``q = +-h`` is the engine's own Taylor polynomial from ``q = 0``.

    ``P_ij(q) - sum_{m <= 4-d} P_{i,j+m}(0) q^m/m!`` is the order-``5 - d``
    remainder, at most ``|q|^(5-d) 100 (1 + |w|)^3 scale_4/(5-d)!`` (each further
    ``q``-derivative of ``log F`` at ``q = 0`` grows by about ``|w|^3/6``), plus the
    rounding of the terms.  A switch to the log-normal near ``q = 0``, whose
    ``q``-slope is zero, leaves a residual of ``|q|`` times the true slope, and a jump
    leaves the jump.
    """
    growth = 100 * (1 + abs(w)) ** 3
    for upper in (False, True):
        base = engine_partials(w, 0.0, upper)
        widen = 1.0 + abs(engine_partials(w, 0.0, not upper, 0)[0])
        scales = [float(np.max(np.abs(base[block]))) for block in BLOCKS]
        for h in (1e-5, 1e-8, 1e-11):
            for q in (h, -h):
                moved = engine_partials(w, q, upper)
                for index, (i, j) in enumerate(KEYS):
                    d = i + j
                    steps = range(ORDER - d + 1)
                    taylor = sum(
                        base[KEYS.index((i, j + m))] * q**m / math.factorial(m) for m in steps
                    )
                    rounding = sum(
                        rounding_bound(d + m, w) * scales[d + m] * h**m / math.factorial(m)
                        for m in steps
                    )
                    remainder = h ** (5 - d) * growth * scales[ORDER] / math.factorial(5 - d)
                    bound = remainder + 2 * widen * rounding + TINY
                    assert abs(moved[index] - taylor) <= bound, (w, q, upper, (i, j))


@pytest.mark.parametrize(
    ("w", "q", "order"),
    [
        (0.5, 0.29, 2),
        (0.5, -0.3, 4),
        (-4.2, 0.2887, 4),
        (3.0, 0.2887, 4),
        (-121.0, 0.01, 1),
        (86.0, 0.01, 3),
        (2.0**101, 0.0, 0),
        (math.inf, 0.0, 2),
        (math.nan, 0.1, 2),
        (1.0, math.nan, 4),
        (0.0, math.inf, 4),
        (0.5, 0.1, 5),
        (0.5, 0.1, -1),
    ],
)
def test_rows_outside_the_domain_write_nan(w, q, order):
    out = np.zeros(15)
    for upper in (False, True):
        engine.gg_uniform_log_tail(w, q, upper, order, out)
        size = (order + 1) * (order + 2) // 2 if 0 <= order <= ORDER else 15
        assert np.all(np.isnan(out[:size]))
        assert not engine.gg_uniform_in_domain(w, q) or not 0 <= order <= ORDER


def test_domain_edges_are_inside():
    low, high = edge_t()
    for q in (0.01, -0.2, 0.2887, -0.2887):
        for t in (low, high):
            assert engine.gg_uniform_in_domain(t * (1 - 1e-9) / q, q)
            assert not engine.gg_uniform_in_domain(t * (1 + 1e-9) / q, q)
    assert engine.gg_uniform_in_domain(1e30, 0.0) and engine.gg_uniform_in_domain(-1e30, 1e-31)


def test_vector_wrapper_is_an_i1_jet_of_the_scalar_engine():
    w = np.array([-37.0, -2.0, 0.0, 1.5, 30.0, 0.5])
    q = np.array([0.01, -0.2, 0.0, 0.2887, -1e-8, 0.5])
    upper = np.array([False, True, False, True, True, False])
    jets = engine.gg_uniform_log_tail_jets(w, q, upper=upper, order=ORDER)
    assert [block.shape for block in jets] == [(6,), (6, 2), (6, 3), (6, 4), (6, 5)]
    packed = np.column_stack([jets[0], *jets[1:]])
    for row in range(len(w)):
        np.testing.assert_array_equal(
            packed[row], engine_partials(w[row], q[row], bool(upper[row]))
        )
    np.testing.assert_array_equal(engine.gg_uniform_domain(w, q), [True] * 5 + [False])
    assert np.all(np.isnan(packed[-1]))
    low_order = engine.gg_uniform_log_tail_jets(w, q, upper=upper, order=1)
    np.testing.assert_array_equal(np.column_stack(low_order), packed[:, :3])
    with pytest.raises(ValueError, match="order"):
        engine.gg_uniform_log_tail_jets(w, q, upper=True, order=5)
