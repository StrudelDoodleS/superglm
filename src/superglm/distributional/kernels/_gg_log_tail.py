"""Log tails of the Prentice generalized gamma at every shape, with partials to order 4.

Coordinates: ``log Y = mu + sigma W``, ``w = (log y - mu)/sigma``, shape ``q``
unrestricted (``q = 0`` the log-normal).  :func:`gg_log_tail` writes ``log S``
(``upper``) or ``log F`` and every partial in ``(w, q)`` by total order
``1..order`` (``order <= 4``), each block in
``combinations_with_replacement(range(2), k)`` order (interface I2).

Routing, by the domain of each engine:

- inside :func:`gg_uniform_in_domain` (``|q| <= 0.2887``, ``|eta| <= 1``,
  ``|w| <= 2^100``): Temme's uniform expansion in the GG's own variables
  (``kernels/_gg_uniform_tail.py``), analytic through ``q = 0``;
- everywhere else: the regularised incomplete gamma (``kernels/_incomplete_gamma.py``)
  at ``a = 1/q^2``, ``x = a e^(q w)``.  For ``q > 0``, ``F = P(a, x)`` and
  ``S = Q(a, x)``; for ``q < 0`` they swap, ``F = Q`` and ``S = P``.

The second route is a composition.  The incomplete-gamma engine's
:func:`log_gamma_tail_jet_into` gives the Taylor jet of ``log T`` in ``(da,
e)`` along fixed ``lambda = x/a = e^(q w)``, ``x -> (a + da) lambda (1 + e)``.
With ``r = dq/q`` and ``q, w`` the row's values::

    da    = a ((1 + r)^-2 - 1)                       (a function of dq alone)
    1 + e = exp(q dw + w dq + dq dw)

and both are truncated bivariate Taylor jets in ``(dw, dq)`` with zero
constant term, so ``log T(w + dw, q + dq) = sum_{i+j <= order} c_ij da^i e^j``
truncated at total degree ``order`` is the multivariate Faa di Bruno formula
evaluated in truncated Taylor arithmetic (Griewank and Walther, *Evaluating
Derivatives*, 2nd ed., 2008, ch. 13), here by Horner in ``e`` then in ``da``.
Reading the ``e``-jet rather than the ``x`` partials keeps ``x^-k`` out of the
arithmetic, so a far lower tail with ``x`` below ``1e-77`` does not overflow.

Conditioning.  Near the diagonal (``|eta| < 1``, ``eta = q zeta`` Temme's
variable) the tail depends on ``(x - a)/sqrt(a)``.  At fixed ``x`` the
``da`` and ``e`` terms of the composition then cancel, by about ``a^k`` at
order ``k``; along fixed ``x/a`` the ``da`` coefficients are of the size of
the result and nothing cancels, so in the body at ``0.2887 < |q| < 0.5``
(``4 < a < 12``), the part of the diagonal the first route leaves, the
composition keeps the engine's accuracy.  The error is the incomplete-gamma
engine's, carried through the terms' absolute values;
``tests/test_gg_log_tail.py`` states the bound.

Rows neither engine evaluates write NaN everywhere: a non-finite ``w`` or
``q``, ``q`` so small that ``1/q^2`` overflows outside the first route,
``x = a e^(q w)`` that overflows or underflows to zero, or an ``order``
outside ``0..4``.
For ``|q| > 10`` (``a < 0.01``) the incomplete-gamma engine's error grows like
``u/a`` (its documented envelope).

Imports only numpy and numba.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.distributional.kernels._gg_uniform_tail import (
    SCRATCH_SHAPE as _UNIFORM_SCRATCH_SHAPE,
)
from superglm.distributional.kernels._gg_uniform_tail import (
    gg_uniform_in_domain,
    gg_uniform_log_tail_with_scratch,
)
from superglm.distributional.kernels._incomplete_gamma import (
    WORK_SIZE,
    jet_mul,
    log_gamma_tail_jet_into,
)

# Digest of the compiled callees above, stamped by ``tools/generate_severity_families.py``:
# Numba's on-disk cache keys a compiled caller on its own file only.
_CALLEE_DIGEST = "b60441c1792b0d1ae62f665c09ca5500f977b59084d9025d3d740c8680ccb3dd"

ORDER_MAX = 4
_N = (ORDER_MAX + 1) * (ORDER_MAX + 2) // 2
_FACTORIAL = (1.0, 1.0, 2.0, 6.0, 24.0)

# Flat scratch layout for the composition route: the incomplete-gamma work,
# then six jets of _N entries.
_GAMMA_ROWS = -(-WORK_SIZE // _N)
_DA = _GAMMA_ROWS * _N
_E = _DA + _N
_INNER = _E + _N
_OUTER = _INNER + _N
_TMP = _OUTER + _N
_JET = _TMP + _N
_ROWS = max(_UNIFORM_SCRATCH_SHAPE[0], _GAMMA_ROWS + 6)
SCRATCH_SHAPE = (_ROWS, _N)
if _UNIFORM_SCRATCH_SHAPE[1] != _N:
    raise ImportError("the uniform engine's scratch rows must hold one order-4 jet")


@njit(cache=True, inline="always")
def _index(i, j):
    return (i + j) * (i + j + 1) // 2 + j


@njit(cache=True)
def _fill_nan(out):
    for r in range(out.shape[0]):
        out[r] = np.nan


@njit(cache=True)
def _map_jets(flat, w, q, a, order, n):
    """Jets of ``da`` and ``e`` in ``(dw, dq)`` into ``flat[_DA]`` and ``flat[_E]``."""
    for r in range(n):
        flat[_DA + r] = 0.0
        flat[_E + r] = 0.0
    minus_inverse_q = -1.0 / q
    # (1 + r)^-2 = sum_k (k + 1) (-dq/q)^k.
    power = 1.0
    for k in range(1, order + 1):
        power *= minus_inverse_q
        flat[_DA + _index(0, k)] = a * (k + 1) * power
    # e = exp(q dw + w dq + dq dw) - 1: the (i, j) coefficient is
    # sum_m q^(i-m) w^(j-m) / ((i-m)! (j-m)! m!).
    for i in range(order + 1):
        for j in range(order + 1 - i):
            total = 0.0
            for m in range(min(i, j) + 1):
                total += (
                    q ** (i - m)
                    * w ** (j - m)
                    / (_FACTORIAL[i - m] * _FACTORIAL[j - m] * _FACTORIAL[m])
                )
            flat[_E + _index(i, j)] = total
    flat[_E] = 0.0


@njit(cache=True)
def _composed_log_tail(w, q, upper, order, out, flat):
    """The incomplete-gamma route for any row, its ``(da, e)`` jet composed into ``(w, q)``."""
    a = 1.0 / (q * q)
    x = a * math.exp(q * w)
    jet = flat[_JET : _JET + _N]
    if not log_gamma_tail_jet_into(a, x, upper == (q > 0.0), order, np.bool_(True), jet, flat):
        _fill_nan(out)
        return
    if not math.isfinite(jet[0]):
        _fill_nan(out)
        return
    _compose_into(flat, w, q, a, order, out)


@njit(cache=True)
def _compose_into(flat, w, q, a, order, out):
    """Compose the ``(da, e)`` jet in ``flat[_JET:]`` into plain ``(w, q)`` partials in ``out``."""
    n = (order + 1) * (order + 2) // 2
    _map_jets(flat, w, q, a, order, n)
    # Horner: P_i = sum_j c_ij e^j, then sum_i P_i da^i.
    for i in range(order, -1, -1):
        for r in range(n):
            flat[_INNER + r] = 0.0
        flat[_INNER] = flat[_JET + _index(i, order - i)]
        for j in range(order - i - 1, -1, -1):
            jet_mul(flat, _INNER, _E, _TMP, n)
            for r in range(n):
                flat[_INNER + r] = flat[_TMP + r]
            flat[_INNER] += flat[_JET + _index(i, j)]
        if i == order:
            for r in range(n):
                flat[_OUTER + r] = flat[_INNER + r]
        else:
            jet_mul(flat, _OUTER, _DA, _TMP, n)
            for r in range(n):
                flat[_OUTER + r] = flat[_TMP + r] + flat[_INNER + r]
    for k in range(order + 1):
        for j in range(k + 1):
            r = k * (k + 1) // 2 + j
            out[r] = flat[_OUTER + r] * _FACTORIAL[k - j] * _FACTORIAL[j]


@njit(cache=True)
def gg_log_tail(w, q, upper, order, out, scratch):
    """``log S`` (``upper``) or ``log F`` of the Prentice GG at ``(w, q)``, then its partials.

    Writes ``out[: (order + 1)(order + 2)/2]``: the value, then the plain partials
    in ``(w, q)`` by total order ``1..order`` (``order <= 4``), each block in
    ``combinations_with_replacement(range(2), k)`` order.  ``scratch`` is
    caller-owned, of :data:`SCRATCH_SHAPE`.  A row neither engine evaluates
    writes NaN everywhere; nothing raises.
    """
    if not (0 <= order <= ORDER_MAX and out.shape[0] >= (order + 1) * (order + 2) // 2):
        _fill_nan(out)
        return
    if gg_uniform_in_domain(w, q):
        gg_uniform_log_tail_with_scratch(w, q, upper, order, out, scratch)
        return
    if not (math.isfinite(w) and math.isfinite(q)) or q == 0.0:
        _fill_nan(out)
        return
    _composed_log_tail(w, q, upper, order, out, scratch.reshape(_ROWS * _N))


@njit(cache=True)
def _composed_rows(w, q, upper, order, out):
    scratch = np.empty((_ROWS, _N))
    flat = scratch.reshape(_ROWS * _N)
    for row in range(w.shape[0]):
        if math.isfinite(w[row]) and math.isfinite(q[row]) and q[row] != 0.0:
            _composed_log_tail(w[row], q[row], upper[row], order, out[row], flat)
        else:
            _fill_nan(out[row])


@njit(cache=True)
def _log_tail_rows(w, q, upper, order, out):
    scratch = np.empty((_ROWS, _N))
    for row in range(w.shape[0]):
        gg_log_tail(w[row], q[row], upper[row], order, out[row], scratch)


def _jet_rows(w, q, upper, order, kernel):
    if (
        isinstance(order, bool)
        or not isinstance(order, int | np.integer)
        or not 0 <= order <= ORDER_MAX
    ):
        raise ValueError("order must be an integer from zero through four")
    w_rows, q_rows = np.broadcast_arrays(
        np.atleast_1d(np.asarray(w, dtype=np.float64)),
        np.atleast_1d(np.asarray(q, dtype=np.float64)),
    )
    if w_rows.ndim != 1:
        raise ValueError("w and q must broadcast to one dimension")
    # Writable C-contiguous rows whatever the inputs, so the kernel compiles one
    # signature: numba types a read-only array (a broadcast view, a read-only
    # input) apart from a writable one.
    w_rows = np.require(w_rows, np.float64, ("C", "W"))
    q_rows = np.require(q_rows, np.float64, ("C", "W"))
    upper_rows = np.empty(w_rows.shape, dtype=np.bool_)
    upper_rows[:] = np.asarray(upper, dtype=np.bool_)
    order = int(order)
    packed = np.empty((w_rows.shape[0], (order + 1) * (order + 2) // 2))
    kernel(w_rows, q_rows, upper_rows, order, packed)
    blocks = [np.ascontiguousarray(packed[:, 0])]
    for k in range(1, order + 1):
        start = k * (k + 1) // 2
        blocks.append(np.ascontiguousarray(packed[:, start : start + k + 1]))
    return tuple(blocks)


def gg_log_tail_jets(
    w: object, q: object, *, upper: object, order: int
) -> tuple[NDArray[np.float64], ...]:
    """I1 jet ``(J0, J1, ..., Jorder)`` of :func:`gg_log_tail` over rows.

    ``J0`` has shape ``(n,)`` and ``Jk`` shape ``(n, k + 1)``, columns in
    ``combinations_with_replacement(range(2), k)`` order of ``(w, q)``.  ``upper``
    is a bool or a per-row bool array.  Rows neither engine evaluates are NaN.
    """
    return _jet_rows(w, q, upper, order, _log_tail_rows)


__all__ = ["ORDER_MAX", "SCRATCH_SHAPE", "gg_log_tail", "gg_log_tail_jets"]
