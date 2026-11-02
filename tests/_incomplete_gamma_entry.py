"""Plain-partial entry points of the incomplete-gamma engine, for its tests.

Fits read the engine through ``log_gamma_tail_scaled_into`` from the declared
kernels; these divide the scaled partials back to plain ones.
"""

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from superglm.distributional.kernels._incomplete_gamma import (
    _X_DEG,
    MAX_ORDER,
    WORK_SIZE,
    log_gamma_tail_scaled_into,
    output_size,
)


@njit
def log_gamma_tail_into(a, x, upper, order, out, w):
    """:func:`log_gamma_tail` with caller-owned work of :data:`WORK_SIZE`."""
    if not log_gamma_tail_scaled_into(a, x, upper, order, out, w):
        return
    # One division per power of x: it overflows only where the partial itself does.
    for r in range((order + 1) * (order + 2) // 2):
        for _ in range(_X_DEG[r]):
            out[r] /= x


@njit
def log_gamma_tail(a, x, upper, order, out):
    """``log Q(a, x)`` (``upper``) or ``log P(a, x)`` and its partials into ``out``."""
    log_gamma_tail_into(a, x, upper, order, out, np.empty(WORK_SIZE))


@njit
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
