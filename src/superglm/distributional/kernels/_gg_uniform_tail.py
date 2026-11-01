"""Log tails of the Prentice generalized gamma near its log-normal limit, to order 4.

Coordinates: ``log Y = mu + sigma W``, ``W = log(q^2 G)/q`` with
``G ~ Gamma(1/q^2, 1)``, and ``w = (log y - mu)/sigma``.  For ``q > 0`` the CDF
is ``F = P(1/q^2, e^(q w)/q^2)``; for ``q < 0`` the lower and upper regularised
incomplete gamma functions swap; ``q = 0`` is the log-normal, ``F = Phi(w)``.

Method.  Temme's uniform asymptotic expansion of the incomplete gamma function
(Temme 1979; DLMF 8.12), as Gil, Segura and Temme implement it (2012, SIAM J.
Sci. Comput. 34(6), section 2.5, eq. 5.1), written in the GG's own variables.
With ``a = 1/q^2``, ``lambda = e^t``, ``t = q w`` and
``eta^2/2 = lambda - 1 - log lambda``, ``eta = q zeta`` where::

    zeta = w G(t),   G = sqrt(g),   g(t) = 2 (e^t - 1 - t)/t^2,
    F      = phi(zeta) [M(zeta)  - q S],
    S_surv = phi(zeta) [M(-zeta) + q S],

``M = Phi/phi`` is the Mills ratio and ``S = T/(1 + s beta_1)`` is GST's series
(their 2.40) with ``s = 1/a = q^2``: ``T = sum_n beta_n eta^n``,
``beta_n = d_{n+1} + (n + 2) s beta_{n+2}`` run backward from
``beta_{N+1} = beta_{N+2} = 0``, and ``d_n`` the Taylor coefficients of
``eta/(lambda - 1)``.  Nothing divides by ``q``: the expression is analytic in
``(w, q)`` through ``q = 0`` for both signs of ``q``, and at ``q = 0`` it is
``log Phi(w)``.  ``d F/d q`` there is ``phi(w)(w^2 + 2)/6``, not zero, so no
switch to the log-normal is made.

The coefficients are exact rationals rounded once to float64: ``d_n`` by
Lagrange-Buermann inversion of ``eta = t G(t)``, since
``eta/(lambda - 1) = G(t) t/(e^t - 1)`` is a function of ``t`` alone, and
``G_n`` as the square root of ``g``'s series.  ``d_2 = +1/12``, which is what
``Gamma*(a) = 1 + beta_1/a = 1 + 1/(12 a) + ...`` requires (GST's printed
``-1/12`` is a sign error).

Derivatives.  Every intermediate is a bivariate Taylor polynomial (a jet) of
total degree ``order``, stored as coefficients
``c[i, j] = (d^(i+j) f/dx^i dy^j)/(i! j!)`` at ``(i+j)(i+j+1)/2 + j``, the packed
order of the I1 jets, so the output is ``c i! j!``.  A function ``f`` of a jet
``u`` is ``sum_k f^(k)(u_0)/k! (u - u_0)^k``, with the powers of ``u - u_0``
formed by truncated Cauchy products that skip their known-zero low degrees.
``zeta``, ``M``, ``log phi`` and the logarithms are jets in ``(w, q)``.  The
recursion and the Horner sum for ``T`` run on jets in ``(ds, d eta)``, where each step is a shift by ``eta_0 + d eta``, and the result
is composed once into ``(w, q)`` through ``s = q^2`` and ``eta = q zeta``.

Cancellation.
- The tail that is at most about 1/2 (the lower one when ``zeta <= 0``, the
  upper one otherwise) is computed directly as ``log phi(xi) + log B`` with
  ``xi = eps zeta <= 0`` and ``B = M(xi) - eps q S``; the other is
  ``log(1 - e^L)`` (Maechler 2012, *Accurately computing log(1 - exp(-|a|))*).
  ``|q S/M(xi)|`` is about ``|c_0(eta) eta| <= 0.44`` (``S ~ c_0``, ``M(xi) ~ 1/|zeta|``), so
  in the domain ``B/M(xi)`` stays between 0.7 and 1.5 and the bracket never cancels.
- ``M``'s derivatives at ``xi = -t <= 0`` are the positive integrals
  ``J_k(t) = int_0^inf x^k e^(-t x - x^2/2) dx``.  Up to ``t = 4``,
  ``J_0 = sqrt(pi/2) erfc(t/sqrt 2) e^(t^2/2)`` and the forward recurrence
  ``J_k = (k - 1) J_{k-2} - t J_{k-1}`` (``J_1 = 1 - t J_0``) loses at most a
  factor ``t^(2k)/k!`` (about 3e3 for ``J_4`` at ``t = 4``).  Beyond, the
  ratios ``rho_k = J_k/J_{k-1} = k/(t + rho_{k+1})`` are the tails of the
  Laplace continued fraction (DLMF 7.9.2), the minimal solution of the same
  recurrence (Gautschi 1967, SIAM Rev. 9), evaluated bottom-up from depth
  ``10 + 160/t``, which reaches float64 accuracy for every ``t > 4``.  No
  subtraction is involved.
- ``G`` comes from its Taylor series at 0 (radius 7.7, the nearest complex zero
  of ``e^t - 1 - t``), never from ``expm1(t) - t``.

Truncation.  ``T`` truncated after ``beta_N`` equals
``sum_{m <= N} d_{m+1} q^m H_m(zeta)`` with ``H_m = zeta^m + m H_{m-2}``, so the
remainder is led by ``d_{N+2} q^{N+1} H_{N+1}(zeta)``, and a derivative of order
``k`` amplifies it by about ``(N + 2)^k``.  With ``TERMS = N = 35``, measured
against mpmath at the domain's worst corner (``|q| = 0.2887``, ``|eta| = 1``),
orders 0-2 are within ``2e-14``, order 3 within ``1e-12`` and order 4 within
``6e-11`` of their scale, and far better inside the domain; 25 terms (GST's
count for values) leave ``9e-8`` at order 4.

Domain: ``|q| <= 0.2887`` (``a >= 12`` to rounding, GST's bound), ``|eta| <= 1``
(GST's bound), ``|t| <= 1.25`` (where 29 terms of ``G`` reach float64 accuracy
for its fourth derivative), and ``|w| <= 2^100`` (order-4 intermediates grow
like ``|w|^8``).  A row outside it writes NaN everywhere.

Imports only numpy and numba.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

ORDER_MAX = 4
Q_MAX = 0.2887
ETA_MAX = 1.0
T_MAX = 1.25
W_MAX = 2.0**100
TERMS = 35

_N_COEF = (ORDER_MAX + 1) * (ORDER_MAX + 2) // 2
_MILLS_SERIES_SWITCH = 4.0
_SQRT_HALF = math.sqrt(0.5)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_LOG_TWO = math.log(2.0)
_FACTORIAL = np.array([1.0, 1.0, 2.0, 6.0, 24.0])

# d_k, the Taylor coefficients of eta/(lambda - 1) in eta, k = 0..TERMS + 1.
_D = np.array(
    [
        1.0,
        -0.3333333333333333,
        0.08333333333333333,
        -0.014814814814814815,
        0.0011574074074074073,
        0.0003527336860670194,
        -0.0001787551440329218,
        3.919263178522438e-05,
        -2.185448510679992e-06,
        -1.85406221071516e-06,
        8.296711340953087e-07,
        -1.7665952736826078e-07,
        6.707853543401498e-09,
        1.0261809784240309e-08,
        -4.382036018453353e-09,
        9.14769958223679e-10,
        -2.5514193994946248e-11,
        -5.830772132550426e-11,
        2.4361948020667415e-11,
        -5.0276692801141755e-12,
        1.1004392031956135e-13,
        3.371763262400985e-13,
        -1.392388722418162e-13,
        2.8534893807047445e-14,
        -5.139111834242572e-16,
        -1.9752288294349442e-15,
        8.099521156704561e-16,
        -1.6522531216398162e-16,
        2.5305430097478883e-18,
        1.1686939738559576e-17,
        -4.770037049820485e-18,
        9.699126059056237e-19,
        -1.2932565538038175e-20,
        -6.969230253185693e-20,
        2.835145432176937e-20,
        -5.7509821590070474e-21,
        6.792953783488915e-23,
    ]
)

# G_n, the Taylor coefficients of G(t) = sqrt(2 (e^t - 1 - t)/t^2), n = 0..28.
_G = np.array(
    [
        1.0,
        0.16666666666666666,
        0.027777777777777776,
        0.003703703703703704,
        0.00038580246913580245,
        3.123162845385068e-05,
        2.020870076425632e-06,
        1.224769743288262e-07,
        8.930612711476909e-09,
        6.275913983389978e-10,
        1.4022160410722694e-11,
        -2.623008629823793e-12,
        -1.1856906386313562e-13,
        3.685136557722098e-14,
        4.102912849464845e-15,
        -2.449127157828315e-16,
        -7.086079800525314e-17,
        -8.977250071541828e-19,
        9.409590319636145e-19,
        7.115296746106628e-20,
        -8.992834454564342e-21,
        -1.5900623609170354e-21,
        2.9556691283957894e-23,
        2.5052400175094895e-23,
        1.1694454993132205e-24,
        -2.9385637201590607e-25,
        -3.6257236272955543e-26,
        2.021231495039928e-27,
        6.723618532860443e-28,
    ]
)


def _packed(i: int, j: int) -> int:
    """Index of the ``dx^i dy^j`` coefficient in a packed jet."""
    return (i + j) * (i + j + 1) // 2 + j


def _power_tables() -> tuple[NDArray[np.intp], ...]:
    """CSR tables of ``u^k = u^(k-1) u`` for a jet ``u`` with zero constant, ``k = 2..4``.

    ``u^(k-1)`` has no entry below degree ``k - 1`` and ``u`` none below degree 1, so only
    those blocks are multiplied; output ``o`` sums its pairs
    ``pointer[k, o] <= p < pointer[k, o + 1]``, and entries below degree ``k`` sum none.
    """
    pointer = np.zeros((ORDER_MAX + 1, _N_COEF + 1), dtype=np.intp)
    left: list[int] = []
    right: list[int] = []
    for k in range(2, ORDER_MAX + 1):
        for d in range(ORDER_MAX + 1):
            for j in range(d + 1):
                pointer[k, _packed(d - j, j)] = len(left)
                for d1 in range(k - 1, d):
                    for j1 in range(max(0, j - (d - d1)), min(j, d1) + 1):
                        left.append(_packed(d1 - j1, j1))
                        right.append(_packed(d - d1 - (j - j1), j - j1))
        pointer[k, _N_COEF] = len(left)
    return pointer, np.array(left, dtype=np.intp), np.array(right, dtype=np.intp)


def _eta_shift_table() -> tuple[NDArray[np.intp], ...]:
    """``(destination, source)`` of ``T <- (eta_0 + d eta) T`` on the entries with an ``eta`` power.

    In descending degree, so each source is read before it is updated; the entries of
    degree at most ``o`` are the table's last ``count[o]``.
    """
    pairs = [
        (_packed(d - b, b), _packed(d - b, b - 1))
        for d in range(ORDER_MAX, 0, -1)
        for b in range(d, 0, -1)
    ]
    count = [sum(1 for dst, _ in pairs if dst < _packed(o + 1, 0)) for o in range(ORDER_MAX + 1)]
    dst, src = (np.array(column, dtype=np.intp) for column in zip(*pairs, strict=True))
    return dst, src, np.array(count, dtype=np.intp)


_POWER_POINTER, _POWER_LEFT, _POWER_RIGHT = _power_tables()
_SHIFT_DST, _SHIFT_SRC, _SHIFT_COUNT = _eta_shift_table()

# Rows of the scratch matrix.  Rows 1..4 hold the powers of the jet being composed.
_DELTA, _ZETA, _XI, _ETA, _T, _V, _W, _R = 0, 5, 6, 7, 8, 9, 10, 11
_BRACKET, _LOG_TAIL, _LOG_BRACKET, _RESULT = 12, 13, 14, 15
_COEF, _BETA_EVEN, _BETA_ODD, _INVERSE = 16, 17, 18, 19
_SCRATCH_ROWS = 20
SCRATCH_SHAPE = (_SCRATCH_ROWS, _N_COEF)


@njit(cache=True, nogil=True, inline="always")
def _shift(work, row, x0, along_y, order):
    """Row ``row`` times ``(x0 + dx)``, ``x`` the second coordinate if ``along_y``, in place."""
    for d in range(order, -1, -1):
        base = d * (d + 1) // 2
        previous = base - d
        for j in range(d + 1):
            value = x0 * work[row, base + j]
            if along_y and j >= 1:
                value += work[row, previous + j - 1]
            elif not along_y and j < d:
                value += work[row, previous + j]
            work[row, base + j] = value


@njit(cache=True, nogil=True, inline="always")
def _powers(work, row, order):
    """Rows ``1..order`` = ``(u - u_0)^k`` for the jet ``u`` in row ``row``."""
    n_coef = (order + 1) * (order + 2) // 2
    for i in range(n_coef):
        work[1, i] = work[row, i]
    work[1, 0] = 0.0
    for k in range(2, order + 1):
        for o in range(n_coef):
            total = 0.0
            for p in range(_POWER_POINTER[k, o], _POWER_POINTER[k, o + 1]):
                total += work[k - 1, _POWER_LEFT[p]] * work[1, _POWER_RIGHT[p]]
            work[k, o] = total


@njit(cache=True, nogil=True, inline="always")
def _series(work, out, order):
    """Row ``out`` = ``c_0 + sum_k c_k (u - u_0)^k``, ``c`` in row ``_COEF``, powers in rows 1..4."""
    work[out, 0] = work[_COEF, 0]
    for i in range(1, (order + 1) * (order + 2) // 2):
        total = 0.0
        for k in range(order, 0, -1):
            total += work[_COEF, k] * work[k, i]
        work[out, i] = total


@njit(cache=True, nogil=True, inline="always")
def _sqrt_g(t):
    value = 0.0
    for n in range(_G.shape[0] - 1, -1, -1):
        value = value * t + _G[n]
    return value


@njit(cache=True, nogil=True, inline="always")
def _sqrt_g_taylor(work, t, order):
    """Row ``_COEF`` = ``G^(k)(t)/k!``, ``k <= order``, by Horner shifts of ``G``'s series."""
    for k in range(order + 1):
        work[_COEF, k] = 0.0
    for n in range(_G.shape[0] - 1, -1, -1):
        for k in range(order, 0, -1):
            work[_COEF, k] = work[_COEF, k] * t + work[_COEF, k - 1]
        work[_COEF, 0] = work[_COEF, 0] * t + _G[n]


@njit(cache=True, nogil=True, inline="always")
def _mills_taylor(work, t, order):
    """Row ``_COEF`` = ``M^(k)(-t)/k!``, ``k <= order``, for ``t >= 0``, ``M = Phi/phi``."""
    if t <= _MILLS_SERIES_SWITCH:
        work[_COEF, 0] = _SQRT_HALF_PI * math.erfc(t * _SQRT_HALF) * math.exp(0.5 * t * t)
        if order >= 1:
            work[_COEF, 1] = 1.0 - t * work[_COEF, 0]
        for k in range(2, order + 1):
            work[_COEF, k] = (k - 1) * work[_COEF, k - 2] - t * work[_COEF, k - 1]
    else:
        keep = max(order, 1)
        ratio = 0.0
        for k in range(int(10.0 + 160.0 / t), 0, -1):
            ratio = k / (t + ratio)
            if k <= keep:
                work[_COEF, k] = ratio
        work[_COEF, 0] = 1.0 / (t + work[_COEF, 1])
        for k in range(1, order + 1):
            work[_COEF, k] *= work[_COEF, k - 1]
    for k in range(2, order + 1):
        work[_COEF, k] /= _FACTORIAL[k]


@njit(cache=True, nogil=True)
def gg_uniform_in_domain(w, q):
    """Whether ``(w, q)`` lies in the uniform expansion's domain (NaN and infinities do not)."""
    if not (abs(q) <= Q_MAX and abs(w) <= W_MAX):
        return False
    t = q * w
    if not abs(t) <= T_MAX:
        return False
    return abs(t * _sqrt_g(t)) <= ETA_MAX


@njit(cache=True, nogil=True, inline="always")
def _gst_series(work, s0, eta0, order, terms):
    """Row ``_V`` = ``S(s, eta)`` as a jet in ``(ds, d eta)``, ``s = q^2``.

    ``T = sum_n beta_n eta^n`` by Horner, ``beta_n = d_{n+1} + (n + 2) s beta_{n+2}``
    univariate jets in ``ds``; then ``S = T/(1 + s beta_1)``.
    """
    n_coef = (order + 1) * (order + 2) // 2
    for k in range(order + 1):
        work[_BETA_EVEN, k] = 0.0
        work[_BETA_ODD, k] = 0.0
    for i in range(n_coef):
        work[_T, i] = 0.0
    first_shift = _SHIFT_DST.shape[0] - _SHIFT_COUNT[order]
    for n in range(terms, -1, -1):
        beta = _BETA_EVEN if n % 2 == 0 else _BETA_ODD
        for a in range(order, 0, -1):
            work[beta, a] = (n + 2) * (s0 * work[beta, a] + work[beta, a - 1])
        work[beta, 0] = (n + 2) * s0 * work[beta, 0] + _D[n + 1]
        for p in range(first_shift, _SHIFT_DST.shape[0]):
            work[_T, _SHIFT_DST[p]] = eta0 * work[_T, _SHIFT_DST[p]] + work[_T, _SHIFT_SRC[p]]
        for a in range(order + 1):
            work[_T, a * (a + 1) // 2] = eta0 * work[_T, a * (a + 1) // 2] + work[beta, a]
    # 1/(1 + s beta_1) as a univariate jet in ds.
    for a in range(order, 0, -1):
        work[_COEF, a] = s0 * work[_BETA_ODD, a] + work[_BETA_ODD, a - 1]
    work[_COEF, 0] = 1.0 + s0 * work[_BETA_ODD, 0]
    work[_INVERSE, 0] = 1.0 / work[_COEF, 0]
    for a in range(1, order + 1):
        total = 0.0
        for i in range(1, a + 1):
            total += work[_COEF, i] * work[_INVERSE, a - i]
        work[_INVERSE, a] = -total * work[_INVERSE, 0]
    for d in range(order + 1):
        base = d * (d + 1) // 2
        for b in range(d + 1):
            total = 0.0
            for a2 in range(d - b + 1):
                total += work[_INVERSE, a2] * work[_T, (d - a2) * (d - a2 + 1) // 2 + b]
            work[_V, base + b] = total


@njit(cache=True, nogil=True, inline="always")
def _compose_in_w_q(work, q, order):
    """Row ``_R`` = row ``_V`` (a jet in ``(ds, d eta)``) at ``s = q^2``, ``eta = q zeta``.

    ``sum_a ds^a W_a`` with ``W_a = sum_b V_ab d eta^b``, by Horner in
    ``ds = 2 q dq + dq^2``; the powers of ``d eta`` are in rows 1..4.
    """
    n_coef = (order + 1) * (order + 2) // 2
    two_q = 2.0 * q
    for i in range(n_coef):
        work[_R, i] = 0.0
    for a in range(order, -1, -1):
        for b in range(order + 1):
            work[_COEF, b] = work[_V, (a + b) * (a + b + 1) // 2 + b] if a + b <= order else 0.0
        _series(work, _W, order)
        for d in range(order, -1, -1):
            base = d * (d + 1) // 2
            for j in range(d + 1):
                value = work[_W, base + j]
                if j >= 1:
                    value += two_q * work[_R, base - d + j - 1]
                if j >= 2:
                    value += work[_R, base - 2 * d + 1 + j - 2]
                work[_R, base + j] = value


@njit(cache=True, nogil=True, inline="always")
def _fill_nan(out, n_out):
    for i in range(min(n_out, out.shape[0])):
        out[i] = np.nan


@njit(cache=True, nogil=True)
def _log_tail_into(w, q, upper, order, out, work, terms):
    """The engine of :func:`gg_uniform_log_tail` with scratch ``work`` and ``terms = N``."""
    valid_order = 0 <= order <= ORDER_MAX
    n_out = (order + 1) * (order + 2) // 2 if valid_order else out.shape[0]
    t = q * w
    if not (valid_order and abs(q) <= Q_MAX and abs(w) <= W_MAX and abs(t) <= T_MAX):
        _fill_nan(out, n_out)
        return
    # The domain's eta test reads G(t) from its Taylor coefficients: the same Horner
    # sum as gg_uniform_in_domain, so the two decide alike.
    _sqrt_g_taylor(work, t, order)
    if not abs(t * work[_COEF, 0]) <= ETA_MAX:
        _fill_nan(out, n_out)
        return
    n_coef = n_out

    # zeta = w G(q w), through the jet of t = q w = (w + dw)(q + dq).
    for i in range(n_coef):
        work[_DELTA, i] = 0.0
    if order >= 1:
        work[_DELTA, 1] = q
        work[_DELTA, 2] = w
    if order >= 2:
        work[_DELTA, 4] = 1.0
    _powers(work, _DELTA, order)
    _series(work, _ZETA, order)
    _shift(work, _ZETA, w, False, order)

    # The smaller tail: lower (eps = 1) when zeta <= 0, upper (eps = -1) otherwise.
    eps = 1.0 if work[_ZETA, 0] <= 0.0 else -1.0
    for i in range(n_coef):
        work[_XI, i] = eps * work[_ZETA, i]
        work[_ETA, i] = work[_ZETA, i]
    _shift(work, _ETA, q, True, order)

    # M(xi) and log phi(xi), both series in xi - xi_0.
    xi0 = work[_XI, 0]
    _powers(work, _XI, order)
    _mills_taylor(work, -xi0, order)
    _series(work, _BRACKET, order)
    for k in range(order + 1):
        work[_COEF, k] = 0.0
    work[_COEF, 0] = -0.5 * xi0 * xi0 - _HALF_LOG_TWO_PI
    if order >= 1:
        work[_COEF, 1] = -xi0
    if order >= 2:
        work[_COEF, 2] = -0.5
    _series(work, _LOG_TAIL, order)

    # U = q S(q^2, q zeta), and B = M(xi) - eps U > 0.
    _gst_series(work, q * q, work[_ETA, 0], order, terms)
    _powers(work, _ETA, order)
    _compose_in_w_q(work, q, order)
    _shift(work, _R, q, True, order)
    for i in range(n_coef):
        work[_BRACKET, i] -= eps * work[_R, i]
    bracket0 = work[_BRACKET, 0]
    if not bracket0 > 0.0:
        _fill_nan(out, n_out)
        return

    # The smaller tail L = log phi(xi) + log B.
    work[_COEF, 0] = math.log(bracket0)
    power = 1.0
    for k in range(1, order + 1):
        power /= bracket0
        work[_COEF, k] = power / k if k % 2 == 1 else -power / k
    _powers(work, _BRACKET, order)
    _series(work, _LOG_BRACKET, order)
    for i in range(n_coef):
        work[_LOG_TAIL, i] += work[_LOG_BRACKET, i]

    result = _LOG_TAIL
    if upper != (eps < 0.0):
        # log(1 - e^L): derivatives -kappa, -kappa(1 + kappa), ... with kappa = 1/expm1(-L).
        x0 = work[_LOG_TAIL, 0]
        work[_COEF, 0] = math.log(-math.expm1(x0)) if x0 > -_LOG_TWO else math.log1p(-math.exp(x0))
        kappa = 1.0 / math.expm1(-x0)
        rising = kappa * (1.0 + kappa)
        if order >= 1:
            work[_COEF, 1] = -kappa
        if order >= 2:
            work[_COEF, 2] = -0.5 * rising
        if order >= 3:
            work[_COEF, 3] = -rising * (1.0 + 2.0 * kappa) / 6.0
        if order >= 4:
            work[_COEF, 4] = -rising * (1.0 + 6.0 * kappa * (1.0 + kappa)) / 24.0
        _powers(work, _LOG_TAIL, order)
        _series(work, _RESULT, order)
        result = _RESULT

    for d in range(order + 1):
        base = d * (d + 1) // 2
        for j in range(d + 1):
            out[base + j] = work[result, base + j] * _FACTORIAL[d - j] * _FACTORIAL[j]


@njit(cache=True, nogil=True)
def gg_uniform_log_tail_with_scratch(w, q, upper, order, out, scratch):
    """:func:`gg_uniform_log_tail` with caller-owned ``scratch`` of :data:`SCRATCH_SHAPE`."""
    _log_tail_into(w, q, upper, order, out, scratch, TERMS)


@njit(cache=True, nogil=True)
def gg_uniform_log_tail(w, q, upper, order, out):
    """``log S`` (``upper``) or ``log F`` of the Prentice GG at ``(w, q)``, then its partials.

    Writes ``out[: (order + 1)(order + 2)/2]``: the value, then the plain partials
    in ``(w, q)`` by total order ``1..order`` (``order <= 4``), each block in
    ``combinations_with_replacement(range(2), k)`` order.  A row outside
    :func:`gg_uniform_in_domain`, or an ``order`` outside ``0..4``, writes NaN.
    """
    scratch = np.empty((_SCRATCH_ROWS, _N_COEF))
    _log_tail_into(w, q, upper, order, out, scratch, TERMS)


@njit(cache=True, nogil=True)
def _log_tail_rows(w, q, upper, order, out):
    scratch = np.empty((_SCRATCH_ROWS, _N_COEF))
    for row in range(w.shape[0]):
        _log_tail_into(w[row], q[row], upper[row], order, out[row], scratch, TERMS)


@njit(cache=True, nogil=True)
def _domain_rows(w, q, out):
    for row in range(w.shape[0]):
        out[row] = gg_uniform_in_domain(w[row], q[row])


def _rows(w: object, q: object) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    w_rows, q_rows = np.broadcast_arrays(
        np.atleast_1d(np.asarray(w, dtype=np.float64)),
        np.atleast_1d(np.asarray(q, dtype=np.float64)),
    )
    if w_rows.ndim != 1:
        raise ValueError("w and q must broadcast to one dimension")
    return np.ascontiguousarray(w_rows), np.ascontiguousarray(q_rows)


def gg_uniform_domain(w: object, q: object) -> NDArray[np.bool_]:
    """:func:`gg_uniform_in_domain` per row."""
    w_rows, q_rows = _rows(w, q)
    out = np.empty(w_rows.shape[0], dtype=np.bool_)
    _domain_rows(w_rows, q_rows, out)
    return out


def gg_uniform_log_tail_jets(
    w: object, q: object, *, upper: object, order: int
) -> tuple[NDArray[np.float64], ...]:
    """I1 jet ``(J0, J1, ..., Jorder)`` of :func:`gg_uniform_log_tail` over rows.

    ``J0`` has shape ``(n,)`` and ``Jk`` shape ``(n, k + 1)``, columns in
    ``combinations_with_replacement(range(2), k)`` order of ``(w, q)``.  ``upper``
    is a bool or a per-row bool array.  Rows outside the domain are NaN.
    """
    if (
        isinstance(order, bool)
        or not isinstance(order, int | np.integer)
        or not 0 <= order <= ORDER_MAX
    ):
        raise ValueError("order must be an integer from zero through four")
    w_rows, q_rows = _rows(w, q)
    upper_rows = np.ascontiguousarray(
        np.broadcast_to(np.asarray(upper, dtype=np.bool_), w_rows.shape)
    )
    order = int(order)
    packed = np.empty((w_rows.shape[0], (order + 1) * (order + 2) // 2))
    _log_tail_rows(w_rows, q_rows, upper_rows, order, packed)
    blocks = [np.ascontiguousarray(packed[:, 0])]
    for k in range(1, order + 1):
        start = k * (k + 1) // 2
        blocks.append(np.ascontiguousarray(packed[:, start : start + k + 1]))
    return tuple(blocks)


__all__ = [
    "ETA_MAX",
    "ORDER_MAX",
    "Q_MAX",
    "SCRATCH_SHAPE",
    "TERMS",
    "T_MAX",
    "W_MAX",
    "gg_uniform_domain",
    "gg_uniform_in_domain",
    "gg_uniform_log_tail",
    "gg_uniform_log_tail_jets",
    "gg_uniform_log_tail_with_scratch",
]
