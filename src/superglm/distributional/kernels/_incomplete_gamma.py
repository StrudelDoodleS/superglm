"""Log regularised incomplete gamma tails with partials to order 4 in shape and point.

:func:`log_gamma_tail` writes ``log Q(a, x)`` (``upper``) or ``log P(a, x)``
(lower), with ``P + Q = 1`` the regularised lower and upper incomplete gamma
functions, then every partial in ``(a, x)`` by total order ``1..order``
(``order <= 4``), each block in ``combinations_with_replacement(range(2), k)``
order: ``(a, x)``, then ``(aa, ax, xx)``, and so on.  Entries are plain
partials, not divided by factorials; :func:`log_gamma_tail_scaled_into`
writes them times ``x^j`` (``j`` the order in ``x``), the form the generated
kernels compose.  At ``x = inf`` it writes the tails' limits, ``log Q = -inf``
and ``log P = 0`` with every partial zero.  A row otherwise outside the domain
(``a`` not finite and positive, ``x`` not positive), or one whose series or
continued fraction does not converge within :data:`MAX_TERMS`, writes NaN
everywhere.

Every quantity is a truncated bivariate Taylor jet in ``(da, e)`` with
``x -> x (1 + e)``, coefficients ``d^(i+j) f/(da^i de^j)/(i! j!)``, packed in
the output order; the ``x`` partials are the ``e`` coefficients divided by
``x^j``.  One primary tail is evaluated on jets (Gautschi 1979; Gil, Segura
and Temme 2012, sections 2-3):

- ``x <= a + max(1, sqrt(a))``: ``P = D S`` with ``log D = a log x - x -
  lgamma(a + 1)`` and the all-positive series ``S = sum_n t_n``,
  ``t_n = x^n/((a + 1) ... (a + n))``.  The ``e`` coefficient ``j`` of
  ``t_n`` is ``C(n, j) t_n``, so every coefficient of ``S`` is a sum of
  terms of one sign;
- above that line: ``Q = (x^a e^-x/Gamma(a)) h`` with Legendre's continued
  fraction ``h = int_0^inf (1 + s)^(a-1) e^(-x s) ds``, evaluated by modified
  Lentz.  No Taylor coefficient of ``h`` vanishes, and iteration stops when
  every coefficient's own relative change is below ``u``; a stop on the
  value's change alone stops too early for orders 3 and 4.  Just above the
  diagonal the continued fraction's order-4 coefficients are ill-conditioned
  at large ``a``, so the line sits ``sqrt(a)`` above the diagonal rather than
  at ``a + 1``;
- below shape :data:`SMALL_SHAPE`, on the series side, ``Q = u + v`` (Gil,
  Segura and Temme 2012, section 2.3; Gautschi 1979), ``u = 1 -
  x^a/Gamma(1 + a)`` by ``expm1`` and ``v`` the series of ``P``'s DLMF
  8.7.1 form without its first term, so a ``Q`` far below one is not read
  as ``1 - P`` (which costs ``u/Q``).  While ``Q <= 1/2`` it is the primary
  tail and ``log P`` its complement; above, ``P`` is, by the series.
  ``ln Gamma(1 + a)`` there comes from DLMF 5.7.3, since ``1 + a`` rounds;
- the other tail is ``log(1 - T)`` on the jet, its value by Maechler's (2012)
  switch at ``-log 2``.  The factor ``a`` of ``1/Gamma(a)`` stays outside the
  exponent there, because the Taylor coefficients ``~ a^-k`` of ``log a``
  cancel in ``exp``.

``log D`` stays on the log scale, so ``log Q`` near ``-750`` and below is exact
to rounding where ``Q`` underflows.  For ``a >= 10`` it is formed as ``gnorm(a)
- log a - a gdev(x, a)``, that is ``a log1pmx((x - a)/a) - log(2 pi a)/2 -
stirling(a)`` (the ``eta``/``Gamma*`` form of Gil, Segura and Temme, section
2.2), from the shared primitives; it removes the ``u a |log x|`` cancellation
of ``a log x - lgamma(a + 1)`` near the diagonal.  The jet of
``lgamma`` uses polygamma at ``a + 1`` and ``lgamma(a) = lgamma(a + 1) - log
a``.  The measured error envelope is stated with the tests in
``tests/test_incomplete_gamma_engine.py``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray
from scipy import special

from superglm.distributional.kernels._severity_primitives import (
    gdev,
    gnorm,
    polygamma_0123,
    psires_0123,
)

# Digest of the primitives above, stamped by ``tools/generate_severity_families.py``:
# Numba's on-disk cache keys a compiled caller on its own file only.
_CALLEE_DIGEST = "1b2c811fe9c0609eb9e42cd4b034a7df3ed410c3c142085d8c6fcb0740a48c24"

MAX_ORDER = 4
MAX_TERMS = 100_000

_U = 2.0**-53
_LOG_HALF = -math.log(2.0)
_FACTORIAL = (1.0, 1.0, 2.0, 6.0, 24.0)
# From here log D takes the Stirling forms of the primitives gnorm and psires.
_STIRLING_FROM = 10.0
# Per-iteration relative change at which the continued fraction along x/a stops.
_ALONG_RATIO_STOP = 16.0 * _U
# Below this shape the upper tail beside the series is Q = u + v (Gil, Segura
# and Temme 2012, section 2.3), not 1 - P: complementing a P within u of one
# costs u/Q.  It is the bottom of the measured envelope.
SMALL_SHAPE = 1e-2
_EULER = 0.5772156649015329
# zeta(k) - 1, k = 2..41, for ln Gamma(1 + a) below a = 1/2 (DLMF 5.7.3): the
# terms fall like (a/2)^k, under u of the sum by k = 41.
_ZETA_MINUS_ONE = tuple(float(z) for z in special.zetac(np.arange(2, 42)))


def output_size(order: int) -> int:
    """Number of entries :func:`log_gamma_tail` writes at ``order``."""
    return (order + 1) * (order + 2) // 2


def _jet_tables():
    """Index tables of the packed bivariate jet, total degree <= MAX_ORDER.

    Entry ``r`` holds ``da^i e^j`` with ``i + j = k`` at ``k (k + 1)/2 + j``.
    The product triples ``(p, q, r)`` list every pair of entries whose
    product lands on ``r``, grouped by ``r``.
    """
    size = output_size(MAX_ORDER)
    a_degree = np.empty(size, np.int64)
    x_degree = np.empty(size, np.int64)
    index = {}
    for k in range(MAX_ORDER + 1):
        for j in range(k + 1):
            r = k * (k + 1) // 2 + j
            a_degree[r], x_degree[r] = k - j, j
            index[(k - j, j)] = r
    first, second, start = [], [], [0]
    down_a = np.full(size, -1, np.int64)
    down_x = np.full(size, -1, np.int64)
    for r in range(size):
        i, j = int(a_degree[r]), int(x_degree[r])
        for p in range(size):
            ip, jp = int(a_degree[p]), int(x_degree[p])
            if ip <= i and jp <= j:
                first.append(p)
                second.append(index[(i - ip, j - jp)])
        start.append(len(first))
        if i > 0:
            down_a[r] = index[(i - 1, j)]
        if j > 0:
            down_x[r] = index[(i, j - 1)]
    return (
        a_degree,
        x_degree,
        a_degree + x_degree,
        np.array(first, np.int64),
        np.array(second, np.int64),
        np.array(start, np.int64),
        down_a,
        down_x,
    )


_A_DEG, _X_DEG, _DEG, _FIRST, _SECOND, _START, _DOWN_A, _DOWN_X = _jet_tables()

# Work layout: bivariate jets of output_size(MAX_ORDER) entries.
_N = output_size(MAX_ORDER)
_PRE = 0  # log D
_SUM = _N  # series S, or continued fraction h
_TERM = 2 * _N  # series a-jet t_n, or Lentz d
_LENTZ_C = 3 * _N
_TMP1 = 4 * _N
_TMP2 = 5 * _N
_LOG = 6 * _N  # log T
_SMOOTH = 7 * _N  # log D + log S, or log D + log h
WORK_SIZE = 8 * _N


@njit(cache=True)
def _jet_mul(w, f, g, h, n):
    for r in range(n):
        total = 0.0
        for t in range(_START[r], _START[r + 1]):
            total += w[f + _FIRST[t]] * w[g + _SECOND[t]]
        w[h + r] = total


# The packed jet product ``h = f g`` (offsets into one flat array), public for
# callers composing :func:`log_gamma_tail_jet_into`'s jet.
jet_mul = _jet_mul


@njit(cache=True)
def _jet_inv(w, f, g, n):
    inverse = 1.0 / w[f]
    w[g] = inverse
    for r in range(1, n):
        total = 0.0
        for t in range(_START[r], _START[r + 1]):
            if _FIRST[t] != 0:
                total += w[f + _FIRST[t]] * w[g + _SECOND[t]]
        w[g + r] = -total * inverse


@njit(cache=True)
def _jet_log(w, f, g, n):
    """``g = log f`` by the Euler-operator recurrence ``k g_r f_0 = k f_r - ...``."""
    w[g] = math.log(w[f])
    for r in range(1, n):
        total = _DEG[r] * w[f + r]
        for t in range(_START[r], _START[r + 1]):
            p = _FIRST[t]
            q = _SECOND[t]
            if p != 0 and q != 0:
                total -= _DEG[q] * w[f + p] * w[g + q]
        w[g + r] = total / (_DEG[r] * w[f])


@njit(cache=True)
def _jet_exp(w, f, g, n):
    w[g] = math.exp(w[f])
    for r in range(1, n):
        total = 0.0
        for t in range(_START[r], _START[r + 1]):
            p = _FIRST[t]
            if p != 0:
                total += _DEG[p] * w[f + p] * w[g + _SECOND[t]]
        w[g + r] = total / _DEG[r]


@njit(cache=True)
def _jet_linear(w, f, h, c0, ca, ce, n):
    """``h = (c0 + ca da + ce e) f``; ``h`` may alias ``f``."""
    for r in range(n - 1, -1, -1):
        total = c0 * w[f + r]
        if _DOWN_A[r] >= 0:
            total += ca * w[f + _DOWN_A[r]]
        if _DOWN_X[r] >= 0:
            total += ce * w[f + _DOWN_X[r]]
        w[h + r] = total


@njit(cache=True)
def _log1mexp(value):
    if value > _LOG_HALF:
        return math.log(-math.expm1(value))
    return math.log1p(-math.exp(value))


@njit(cache=True)
def _log_scale(w, f, r):
    """``max |f_p f_q|/|f_0|`` over ``p + q = r``, ``p, q != 0``: the scale of ``r`` in ``log f``."""
    best = 0.0
    for t in range(_START[r], _START[r + 1]):
        p = _FIRST[t]
        q = _SECOND[t]
        if p != 0 and q != 0:
            best = max(best, abs(w[f + p] * w[f + q]))
    return best / abs(w[f])


@njit(cache=True)
def _log_prefactor(a, x, k_max, n, w, along_ratio):
    """``log D = a log x - x - lgamma(a + 1)`` on the jet, into ``w[_PRE]``."""
    for r in range(n):
        w[_PRE + r] = 0.0
    # log(x/a), with t = (x - a)/a, is log1p(t) near the diagonal and
    # log x - log a away from it, where log1p of the rounded t would carry u/(1 + t).
    t = (x - a) / a
    near = abs(t) <= 0.5
    log_ratio = math.log1p(t) if near else math.log(x) - math.log(a)
    if a < _STIRLING_FROM:
        w[_PRE] = a * math.log(x) - x - math.lgamma(a + 1.0)
    else:
        # a (log(x/a) - t) is -a gdev(x, a), and -log(2 pi a)/2 - stirling(a)
        # is gnorm(a) - log a.
        w[_PRE] = gnorm(a) - math.log(a) - a * gdev(x, a)
    if k_max == 0:
        return
    p0, p1, p2, p3 = polygamma_0123(a + 1.0)
    if a < _STIRLING_FROM:
        w[_PRE + 1] = math.log(x) - p0
    else:
        # psi(a + 1) - log a = psires(0, a) + 1/a.
        w[_PRE + 1] = log_ratio - (psires_0123(a)[0] + 1.0 / a)
    polygamma = (p1, p2, p3)
    for k in range(2, k_max + 1):
        w[_PRE + k * (k + 1) // 2] = -polygamma[k - 2] / _FACTORIAL[k]
    if along_ratio:
        _along_ratio_prefactor(a, x, p0, polygamma, k_max, w)
    # a log(1 + e) - x e: e coefficient j is (a + da) (-1)^(j+1)/j, less x at j = 1.
    for j in range(1, k_max + 1):
        log_coefficient = (1.0 if j % 2 == 1 else -1.0) / j
        w[_PRE + j * (j + 1) // 2 + j] = a * log_coefficient
        if j < k_max:
            w[_PRE + (j + 1) * (j + 2) // 2 + j] = log_coefficient
    w[_PRE + 2] = a - x
    if along_ratio and k_max >= 2:
        w[_PRE + 4] = -t


@njit(cache=True)
def _along_ratio_prefactor(a, x, p0, polygamma, k_max, w):
    """The ``da^k`` coefficients of ``log D`` at fixed ``x/a``.

    ``a log x - lgamma(a + 1)`` at ``x = a lambda`` is ``g(a) = a log a + a log
    lambda - lgamma(a + 1)``, with ``g' = -gdev(x, a) - (psi(a + 1) - log a)``
    and ``g^(k) = (-1)^k (k - 2)!/a^(k - 1) - psi^(k - 1)(a + 1)`` for ``k >= 2``,
    which for ``a >= 10`` is ``-psires(k - 1, a) + (-1)^k (k - 1)!/a^k``.
    """
    if a < _STIRLING_FROM:
        w[_PRE + 1] = -gdev(x, a) - (p0 - math.log(a))
        power = 1.0
        for k in range(2, k_max + 1):
            power /= a
            w[_PRE + k * (k + 1) // 2] = (
                (1.0 if k % 2 == 0 else -1.0) * _FACTORIAL[k - 2] * power - polygamma[k - 2]
            ) / _FACTORIAL[k]
        return
    residual = psires_0123(a)
    w[_PRE + 1] = -gdev(x, a) - (residual[0] + 1.0 / a)
    power = 1.0
    for k in range(1, k_max + 1):
        power /= a
        if k >= 2:
            w[_PRE + k * (k + 1) // 2] = (
                -residual[k - 1] + (1.0 if k % 2 == 0 else -1.0) * _FACTORIAL[k - 1] * power
            ) / _FACTORIAL[k]


@njit(cache=True)
def _series_log_sum(a, x, k_max, n, w, along_ratio):
    """``log S`` into ``w[_TMP2]``; returns the term count, or -1 at the cap.

    Along ``x/a`` the term ratio is ``lambda (a + da)/(a + n + da)`` and the
    ``da`` coefficients of a term grow by up to ``(1 + 2/n)^k``, so the tail
    estimate squares the growth factor; coefficients of mixed sign may sum to
    near zero, so each is held to ``u`` times its scale in ``log S`` too.
    """
    binomial = _TMP1
    ratio_x = x / a
    for r in range(n):
        w[_SUM + r] = 0.0
    for k in range(k_max + 1):
        w[_TERM + k] = 0.0
    w[_TERM] = 1.0
    w[_SUM] = 1.0
    w[binomial] = 1.0
    for n_term in range(1, MAX_TERMS + 1):
        denominator = a + n_term
        previous = 0.0
        for k in range(k_max + 1):
            quotient = (w[_TERM + k] - previous) / denominator
            w[_TERM + k] = x * quotient + ratio_x * previous if along_ratio else x * quotient
            previous = quotient
        for j in range(1, k_max + 1):
            w[binomial + j] = (w[binomial + j - 1] * (n_term - j + 1)) / j
        # Successive terms C(n, j) t_n shrink by (n + 1)/(n + 1 - j) x/(a + n + 1).
        ratio = 2.0
        if n_term + 1 > k_max:
            growth = (n_term + 1.0) / (n_term + 1.0 - k_max)
            ratio = x / (denominator + 1.0) * (growth * growth if along_ratio else growth)
        done = ratio < 1.0
        tail = max(1.0, ratio / (1.0 - ratio)) if done else 0.0
        for r in range(n):
            term = w[binomial + _X_DEG[r]] * w[_TERM + _A_DEG[r]]
            w[_SUM + r] += term
            if done and abs(term) * tail > _U * abs(w[_SUM + r]):
                done = along_ratio and abs(term) * tail <= _U * _log_scale(w, _SUM, r)
        if done:
            _jet_log(w, _SUM, _TMP2, n)
            return n_term
    return -1


@njit(cache=True)
def _continued_fraction_log_h(a, x, n, w, along_ratio):
    """``log h`` into ``w[_TMP2]``; returns the iteration count, or -1 at the cap.

    Along ``x/a`` mixed coefficients of ``h`` change sign, so each is held to
    its scale in ``log h`` as well as its own size, and the fixed point keeps
    a few ``u`` of rounding noise in every coefficient (``b``'s ``da``
    coefficient is ``x/a - 1``), so the change allowed is
    :data:`_ALONG_RATIO_STOP`.
    """
    d = _TERM
    c = _LENTZ_C
    h = _SUM
    # b_i = x (1 + e) + 1 + 2 i - a and a_i = -i (i - a), linear in (da, e); along
    # x/a, x = (a + da) lambda (1 + e) adds lambda - 1 to da and lambda to da e.
    b_da = (x - a) / a if along_ratio else -1.0
    b_mixed = x / a if along_ratio else 0.0
    for r in range(n):
        w[_TMP1 + r] = 0.0
    b0 = x + 1.0 - a
    w[_TMP1] = b0
    if n > 1:
        w[_TMP1 + 1] = b_da
        w[_TMP1 + 2] = x
    if along_ratio and n > 4:
        w[_TMP1 + 4] = b_mixed
    _jet_inv(w, _TMP1, d, n)
    for r in range(n):
        w[h + r] = w[d + r]
    for i in range(1, MAX_TERMS + 1):
        an0 = -i * (i - a)
        an1 = float(i)
        b0 += 2.0
        _jet_linear(w, d, d, an0, an1, 0.0, n)
        w[d] += b0
        if n > 1:
            w[d + 1] += b_da
            w[d + 2] += x
        if along_ratio and n > 4:
            w[d + 4] += b_mixed
        if i == 1:
            for r in range(n):
                w[c + r] = 0.0
        else:
            _jet_inv(w, c, _TMP1, n)
            _jet_linear(w, _TMP1, c, an0, an1, 0.0, n)
        w[c] += b0
        if n > 1:
            w[c + 1] += b_da
            w[c + 2] += x
        if along_ratio and n > 4:
            w[c + 4] += b_mixed
        _jet_inv(w, d, _TMP1, n)
        for r in range(n):
            w[d + r] = w[_TMP1 + r]
        _jet_mul(w, d, c, _TMP1, n)
        _jet_mul(w, h, _TMP1, _TMP2, n)
        done = i > 2
        for r in range(n):
            change = abs(w[_TMP2 + r] - w[h + r])
            if done and change > _U * abs(w[_TMP2 + r]):
                limit = max(abs(w[_TMP2 + r]), _log_scale(w, _TMP2, r))
                done = along_ratio and change <= _ALONG_RATIO_STOP * limit
            w[h + r] = w[_TMP2 + r]
        if done:
            _jet_log(w, h, _TMP2, n)
            return i
    return -1


@njit(cache=True)
def _jet_complement(w, a, scaled, n):
    """``log(1 - T)`` into ``w[_TMP2]``, with ``log T`` in ``w[_LOG]``.

    ``T = exp(w[_SMOOTH])``, times ``a + da`` when ``scaled``.
    """
    _jet_exp(w, _SMOOTH, _TERM, n)
    if scaled:
        _jet_linear(w, _TERM, _TERM, a, 1.0, 0.0, n)
    for r in range(1, n):
        w[_TMP1 + r] = -w[_TERM + r]
    w[_TMP1] = -math.expm1(w[_LOG])
    _jet_log(w, _TMP1, _TMP2, n)
    w[_TMP2] = _log1mexp(w[_LOG])


@njit(cache=True)
def _fill_nan(out):
    for i in range(out.shape[0]):
        out[i] = np.nan


@njit(cache=True)
def _lgamma1p(a):
    """``ln Gamma(1 + a)``, relative-accurate as ``a`` goes to zero, where ``1 + a`` rounds."""
    if a >= 0.5:
        return math.lgamma(1.0 + a)
    # DLMF 5.7.3: -log1p(a) + a (1 - gamma) + sum_k (-1)^k (zeta(k) - 1) a^k/k.
    total = 0.0
    for index in range(len(_ZETA_MINUS_ONE) - 1, -1, -1):
        k = index + 2.0
        sign = 1.0 if index % 2 == 0 else -1.0
        total = (total + sign * _ZETA_MINUS_ONE[index] / k) * a
    return a * total + a * (1.0 - _EULER) - math.log1p(a)


@njit(cache=True)
def _log_power_jet(a, log_x, k_max, n, w, f, along_ratio):
    """Jet of ``L = a log x - ln Gamma(a + 1)`` into ``w[f:]``.

    ``x -> x (1 + e)``, or along ``x/a`` ``x -> (a + da)(x/a)(1 + e)``, which
    adds ``(a + da) log(1 + da/a)``, whose ``da^k`` coefficient is ``1`` at
    ``k = 1`` and ``(-1)^k/(k (k - 1) a^(k - 1))`` above.  ``L`` is ``log P``
    to leading order in ``x`` (DLMF 8.7.1), ``log D + x``.
    """
    for r in range(n):
        w[f + r] = 0.0
    w[f] = a * log_x - _lgamma1p(a)
    if k_max == 0:
        return
    p0, p1, p2, p3 = polygamma_0123(a + 1.0)
    polygamma = (p1, p2, p3)
    w[f + 1] = log_x - p0 + (1.0 if along_ratio else 0.0)
    power = 1.0
    for k in range(2, k_max + 1):
        power /= a
        coefficient = -polygamma[k - 2] / _FACTORIAL[k]
        if along_ratio:
            coefficient += (1.0 if k % 2 == 0 else -1.0) * power / (k * (k - 1.0))
        w[f + k * (k + 1) // 2] = coefficient
    # (a + da) log(1 + e): e^j has a (-1)^(j+1)/j, da e^j has (-1)^(j+1)/j.
    for j in range(1, k_max + 1):
        log_coefficient = (1.0 if j % 2 == 1 else -1.0) / j
        w[f + j * (j + 1) // 2 + j] = a * log_coefficient
        if j < k_max:
            w[f + (j + 1) * (j + 2) // 2 + j] = log_coefficient


@njit(cache=True)
def _small_shape_log_q(a, x, k_max, n, w, along_ratio):
    """``log Q`` into ``w[_LOG]``, and ``Q`` into ``w[_TMP1]``, by ``Q = u + v`` for small ``a``.

    ``False`` if the series does not settle or ``Q`` is not positive.

    Gil, Segura and Temme (2012), eqs. (2.20)-(2.24), after Gautschi (1979)
    and DiDonato and Morris (1986): with ``E = x^a/Gamma(1 + a) = exp(L)``,
    ``u = -expm1(L)`` and ``v = -E a V``, ``V = sum_{n>=1} (-x)^n/(n! (a +
    n))``.  ``u`` and ``v`` cancel only mildly for ``x <= 1.5`` (by about
    ``(|log x| + gamma + Ein(x))/E1(x)``, under 7 at ``x = 1``), where the
    series alternates with terms falling from ``n = 1``.  Every quantity is a
    jet, as in the series and the continued fraction; ``x`` is ``X``'s jet.
    """
    _log_power_jet(a, math.log(x), k_max, n, w, _PRE, along_ratio)
    log_power = w[_PRE]
    _jet_exp(w, _PRE, _SMOOTH, n)
    point = _LENTZ_C
    for r in range(n):
        w[point + r] = 0.0
        w[_SUM + r] = 0.0
        w[_TMP2 + r] = 0.0
    w[point] = x
    if n > 1:
        w[point + 2] = x
        if along_ratio:
            w[point + 1] = x / a
            if n > 4:
                w[point + 4] = x / a
    settled = False
    for n_term in range(1, MAX_TERMS + 1):
        # t_n = (-x)^n/n! on the jet, into _TERM.
        if n_term == 1:
            for r in range(n):
                w[_TERM + r] = -w[point + r]
        else:
            _jet_mul(w, _TERM, point, _TMP1, n)
            for r in range(n):
                w[_TERM + r] = -w[_TMP1 + r] / n_term
        # t_n/(a + n + da): the da^k coefficient of the inverse is (-1)^k/(a + n)^(k + 1).
        inverse = 1.0 / (a + n_term)
        done = n_term > k_max and n_term > x
        for r in range(n):
            i = _A_DEG[r]
            j = _X_DEG[r]
            total = 0.0
            power = inverse
            for k in range(i + 1):
                source = (i - k + j) * (i - k + j + 1) // 2 + j
                total += w[_TERM + source] * power
                power *= -inverse
            w[_SUM + r] += total
            w[_TMP2 + r] += abs(total)
            if abs(total) > _U * w[_TMP2 + r]:
                done = False
        if done:
            settled = True
            break
    if not settled:
        return False
    _jet_linear(w, _SUM, _TMP1, a, 1.0, 0.0, n)
    _jet_mul(w, _SMOOTH, _TMP1, _TMP2, n)
    for r in range(1, n):
        w[_TMP1 + r] = -w[_SMOOTH + r] - w[_TMP2 + r]
    w[_TMP1] = -math.expm1(log_power) - w[_TMP2]
    if not w[_TMP1] > 0.0:
        return False
    _jet_log(w, _TMP1, _LOG, n)
    return True


@njit(cache=True)
def log_gamma_tail_small_x_jet_into(a, log_x, upper, order, along_ratio, jet, w):
    """:func:`log_gamma_tail_jet_into` at ``x = exp(log_x)`` too small to form, from ``log x``.

    ``log P = L + log(e^-x S) = L - a x/(a + 1) + O(x^2)``, ``L = a log x -
    ln Gamma(a + 1)`` (DLMF 8.7.1); every jet coefficient of the remainder is
    at most ``x``, or ``x/a`` along ``x/a``, so where those are below ``u``
    the jet is ``L``'s, and ``log Q`` its complement.  The caller holds the
    remainder below ``u`` of the row's scale; ``False`` (and NaN) off the
    domain.
    """
    if not (0 <= order <= MAX_ORDER and jet.shape[0] >= (order + 1) * (order + 2) // 2):
        _fill_nan(jet)
        return False
    if not (0.0 < a < math.inf and -math.inf < log_x < math.inf):
        _fill_nan(jet)
        return False
    n = (order + 1) * (order + 2) // 2
    _log_power_jet(a, log_x, order, n, w, _SMOOTH, along_ratio)
    for r in range(n):
        w[_LOG + r] = w[_SMOOTH + r]
    if upper:
        _jet_complement(w, a, False, n)
        for r in range(n):
            w[_LOG + r] = w[_TMP2 + r]
    for r in range(n):
        jet[r] = w[_LOG + r]
    return True


@njit(cache=True)
def log_gamma_tail_jet_into(a, x, upper, order, along_ratio, jet, w):
    """Taylor coefficients of the log tail into ``jet``; ``False`` (and NaN) off the domain.

    ``jet[r]`` is the coefficient of ``da^i e^j`` (``d^(i+j) log T/(da^i de^j)/(i! j!)``),
    packed as the output.  With ``along_ratio`` false ``x -> x (1 + e)`` at fixed
    ``x``; with it true ``x -> (a + da) (x/a) (1 + e)``, along fixed ``x/a``,
    where near the diagonal the tail depends on ``(x - a)/sqrt(a)`` and the
    ``da`` coefficients stay of the size of the result.  ``w`` is caller-owned
    work of :data:`WORK_SIZE`.
    """
    if not (0 <= order <= MAX_ORDER and jet.shape[0] >= (order + 1) * (order + 2) // 2):
        _fill_nan(jet)
        return False
    if not (0.0 < a < math.inf and 0.0 < x <= math.inf):
        _fill_nan(jet)
        return False
    if x == math.inf:
        # Past the top of the range log Q is below float64's and log P is 0, every
        # partial zero: a window's far end there is absent, not unknown.
        for r in range((order + 1) * (order + 2) // 2):
            jet[r] = 0.0
        jet[0] = -math.inf if upper else 0.0
        return True
    k_max = order
    n = (k_max + 1) * (k_max + 2) // 2
    series = x <= a + max(1.0, math.sqrt(a))
    if series and a < SMALL_SHAPE:
        # Q is the primary tail while Q <= 1/2; past that P is, by its own series.
        if not _small_shape_log_q(a, x, k_max, n, w, along_ratio):
            _fill_nan(jet)
            return False
        if upper or w[_LOG] <= _LOG_HALF:
            if not upper:
                # log(1 - Q) from Q's own jet: rebuilding Q as exp(log Q) would
                # cancel log a's da coefficients, of order a^-k.
                for r in range(1, n):
                    w[_TMP2 + r] = -w[_TMP1 + r]
                w[_TMP2] = 1.0 - w[_TMP1]
                _jet_log(w, _TMP2, _SMOOTH, n)
                w[_SMOOTH] = _log1mexp(w[_LOG])
                for r in range(n):
                    w[_LOG + r] = w[_SMOOTH + r]
            for r in range(n):
                jet[r] = w[_LOG + r]
            return True
    _log_prefactor(a, x, k_max, n, w, along_ratio)
    if series:
        status = _series_log_sum(a, x, k_max, n, w, along_ratio)
    else:
        status = _continued_fraction_log_h(a, x, n, w, along_ratio)
    if status < 0:
        _fill_nan(jet)
        return False
    for r in range(n):
        w[_SMOOTH + r] = w[_PRE + r] + w[_TMP2 + r]
        w[_LOG + r] = w[_SMOOTH + r]
    if not series:
        # log Q = log D + log a + log h; the da^k coefficient of log a is (-1)^(k+1)/(k a^k).
        w[_LOG] += math.log(a)
        power = 1.0
        for k in range(1, k_max + 1):
            power /= a
            w[_LOG + k * (k + 1) // 2] += (1.0 if k % 2 == 1 else -1.0) * power / k
    if upper == series:
        _jet_complement(w, a, not series, n)
        for r in range(n):
            w[_LOG + r] = w[_TMP2 + r]
    for r in range(n):
        jet[r] = w[_LOG + r]
    return True


@njit(cache=True)
def log_gamma_tail_scaled_into(a, x, upper, order, out, w):
    """The scaled partials ``x^j d^(i+j) log T/(da^i dx^j)`` into ``out``, packed as the output.

    They are the jet's coefficients times ``i! j!``, with no power of ``x``:
    a kernel that composes them through ``x = a y/mean`` meets the factor
    ``x^j`` in the chain rule's own ``x``-derivatives, so neither ``x^-j``
    nor ``x^j`` is ever formed.  ``w`` is caller-owned work of :data:`WORK_SIZE`.
    Returns ``False`` (and NaN) off the domain.
    """
    if not log_gamma_tail_jet_into(a, x, upper, order, False, out, w):
        return False
    for r in range((order + 1) * (order + 2) // 2):
        out[r] *= _FACTORIAL[_A_DEG[r]] * _FACTORIAL[_X_DEG[r]]
    return True


@njit(cache=True)
def log_gamma_tail_into(a, x, upper, order, out, w):
    """:func:`log_gamma_tail` with caller-owned work of :data:`WORK_SIZE`."""
    if not log_gamma_tail_scaled_into(a, x, upper, order, out, w):
        return
    # One division per power of x: it overflows only where the partial itself does.
    for r in range((order + 1) * (order + 2) // 2):
        for _ in range(_X_DEG[r]):
            out[r] /= x


@njit(cache=True)
def log_gamma_tail(a, x, upper, order, out):
    """``log Q(a, x)`` (``upper``) or ``log P(a, x)`` and its partials into ``out``."""
    log_gamma_tail_into(a, x, upper, order, out, np.empty(WORK_SIZE))


@njit(cache=True)
def _log_gamma_tail_rows(a, x, upper, order, packed):
    w = np.empty(WORK_SIZE)
    for row in range(a.shape[0]):
        log_gamma_tail_into(a[row], x[row], upper[row], order, packed[row], w)


def log_gamma_tail_jets(
    a: ArrayLike, x: ArrayLike, upper: ArrayLike, order: int
) -> tuple[NDArray[np.float64], ...]:
    """Packed jet ``(J0, ..., J_order)`` of the log tail over rows.

    ``a``, ``x`` and ``upper`` broadcast to one dimension.  ``J0`` has shape
    ``(n,)`` and ``Jk`` shape ``(n, k + 1)``, columns in
    ``combinations_with_replacement(range(2), k)`` order over ``(a, x)``.
    """
    if not (isinstance(order, (int, np.integer)) and 0 <= order <= MAX_ORDER):
        raise ValueError(f"order must be an integer from 0 to {MAX_ORDER}; got {order!r}.")
    shape_a, point, tail = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(x, dtype=np.float64), np.asarray(upper, bool)
    )
    if shape_a.ndim > 1:
        raise ValueError("a, x and upper must broadcast to one dimension.")
    shape_a = np.ascontiguousarray(np.atleast_1d(shape_a))
    point = np.ascontiguousarray(np.atleast_1d(point))
    tail = np.ascontiguousarray(np.atleast_1d(tail))
    packed = np.empty((shape_a.shape[0], output_size(int(order))))
    _log_gamma_tail_rows(shape_a, point, tail, int(order), packed)
    jets = [packed[:, 0].copy()]
    for k in range(1, int(order) + 1):
        start = k * (k + 1) // 2
        jets.append(np.ascontiguousarray(packed[:, start : start + k + 1]))
    return tuple(jets)
