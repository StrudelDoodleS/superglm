"""Log probabilities of response windows, for truncated and censored rows, on packed jets.

A truncated row divides its density by ``P(a < Y <= b)``, and a censored row
contributes ``log P(l < Y <= r)`` alone.  Both are ``log(T(near) - T(far))``
for one tail function ``T`` (the CDF, or the survival function under
reflection) taken at the window's two ends, with ``T(far) <= T(near)``.  The
family supplies ``log T`` at each end as a packed jet in its parameters;
:func:`log_window_jets` combines them.  With ``Delta = log T(far) - log
T(near) <= 0`` and ``kappa = 1/expm1(-Delta)``::

    log D = log T(near) + log(-expm1(Delta))
    grad  = g_near + kappa d
    hess  = H_near + kappa (H_near - H_far) - kappa (1 + kappa) d d^T

with ``d = g_near - g_far``, and every higher derivative of ``log(-expm1(Delta))``
a polynomial in ``kappa``, composed with ``Delta``'s derivatives by the
set-partition chain rule.  An absent far end (``Delta = -inf``), or one whose
tail is negligible beside the near end's (``kappa`` rounds to zero), returns
the near end's jet unchanged, so a one-sided window is exactly its tail.  The
difference of two log tails keeps only about ``u/Delta`` of relative accuracy
as the window narrows, so a window narrower than :data:`NARROW_WINDOW` on its
density's own scale is integrated instead by three-point Gauss-Legendre
quadrature of the density (:func:`quadrature_window_jets`).  The quadrature's
relative error there is about ``5e-7`` times the sixth power of that width
(Abramowitz and Stegun 25.4.30), below ``1e-13``.

Imports only numpy and the shared chain-rule helpers.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.kernels._common import FLOAT
from superglm.distributional.kernels._curvature_direction import (
    _set_partitions,
    compose_derivatives,
    packed_tuples,
)

NARROW_WINDOW = 0.05
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(3)

TailChannels = tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None]
#: A packed jet ``(J0, J1, ..., Jp)``: ``J0`` per row, ``Jk`` the ``k``-th partial
#: derivatives ``(n, C(K + k - 1, k))`` in :func:`packed_tuples` order.
Jet = tuple[NDArray[np.float64], ...]


def _window_log_derivatives(kappa: NDArray, order: int) -> dict[int, NDArray[np.float64]]:
    """``phi^(m)(Delta)``, ``m = 1..order``, for ``phi(Delta) = log(-expm1(Delta))``.

    ``phi' = -kappa`` and ``d kappa/d Delta = kappa (1 + kappa)``, so
    ``phi'' = -kappa (1 + kappa)``, ``phi''' = phi'' (1 + 2 kappa)`` and
    ``phi'''' = phi'' (1 + 6 kappa (1 + kappa))``.  For ``kappa > 0`` every
    factor is a sum of positive terms, so each is formed to a few ``u``
    relative to ``kappa``'s own error.
    """
    derivatives = {1: -kappa}
    if order >= 2:
        derivatives[2] = -kappa * (1.0 + kappa)
    if order >= 3:
        derivatives[3] = derivatives[2] * (1.0 + 2.0 * kappa)
    if order >= 4:
        derivatives[4] = derivatives[2] * (1.0 + 6.0 * kappa * (1.0 + kappa))
    return derivatives


def _checked_order(order: int) -> int:
    if order not in (0, 1, 2, 3, 4):
        raise ValueError(f"window jets are formed to order 0 through 4, not {order}")
    return order


def log_window_jets(near: Jet, far: Jet, *, order: int) -> Jet:
    """``log(T(near) - T(far))`` and its packed derivatives to ``order <= 4``.

    ``near`` and ``far`` are the two ends' log-tail jets ``(J0, J1, ...)``,
    ``J0`` per row and ``Jk`` the packed ``k``-th partial derivatives in
    :func:`packed_tuples` order, each with at least ``order + 1`` entries.
    With ``Delta = J0_far - J0_near``, ``log D = J0_near + phi(Delta)``, and
    the derivatives of ``phi(Delta(theta))`` follow from the set-partition
    chain rule (Constantine and Savits, *A multivariate Faa di Bruno formula
    with applications*, Trans. AMS 348, 1996; Hardy 2006) with the far-minus-
    near jets as ``Delta``'s derivatives.  A far end with value ``-inf`` is
    absent, one with ``kappa`` rounding to zero is negligible, and both return
    the near jets with the far entries ignored; ends whose tails round equal
    give ``-inf`` with non-finite derivatives.
    """
    order = _checked_order(order)
    near_value, far_value = near[0], far[0]
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        delta = far_value - near_value
        kappa = 1.0 / np.expm1(-delta)
    result = [np.array(entry, dtype=FLOAT, copy=True) for entry in near[: order + 1]]
    # Only -inf marks an absent end; a NaN far end (an engine failure) stays NaN.
    rows = np.flatnonzero(~(far_value == -np.inf) & ~(np.abs(kappa) == 0.0))
    if len(rows) == 0:
        return tuple(result)
    with np.errstate(divide="ignore", invalid="ignore"):
        result[0][rows] += np.log(-np.expm1(delta[rows]))
    if order == 0:
        return tuple(result)
    k_parameters = near[1].shape[1]
    outer = {
        m: values[:, None] for m, values in _window_log_derivatives(kappa[rows], order).items()
    }
    with np.errstate(invalid="ignore", over="ignore"):
        inner = {
            m: {
                (0, key): far[m][rows, column] - near[m][rows, column]
                for column, key in enumerate(packed_tuples(k_parameters, m))
            }
            for m in range(1, order + 1)
        }
        for m in range(1, order + 1):
            result[m][rows] += compose_derivatives(outer, inner, m, k_parameters, components=1)
    return tuple(result)


@functools.cache
def _block_keys(
    k_parameters: int, order: int
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]:
    """Per packed entry of ``order``, the block index tuples of each partition with two or more blocks."""
    partitions = [p for p in _set_partitions(list(range(order))) if len(p) >= 2]
    return tuple(
        tuple(
            tuple(tuple(sorted(target[position] for position in block)) for block in partition)
            for partition in partitions
        )
        for target in packed_tuples(k_parameters, order)
    )


def _node_cumulant(share: NDArray, factors: list[list[NDArray]]) -> NDArray[np.float64]:
    """Joint cumulant over the node mixture of two to four centred node variables.

    ``factors[j][i]`` is variable ``j`` at node ``i``, already centred at its
    mixture mean, so the second and third cumulants are plain moments and the
    fourth subtracts the three pairings' covariance products.
    """

    def moment(variables: list[list[NDArray]]) -> NDArray[np.float64]:
        return sum(
            share[:, i] * functools.reduce(np.multiply, [v[i] for v in variables])
            for i in range(share.shape[1])
        )

    value = moment(factors)
    if len(factors) == 4:
        a, b, c, d = factors
        value = value - (
            moment([a, b]) * moment([c, d])
            + moment([a, c]) * moment([b, d])
            + moment([a, d]) * moment([b, c])
        )
    return value


def quadrature_window_jets(
    half_width: NDArray,
    log_density_jets: Callable[[float], Jet],
    *,
    order: int,
) -> Jet:
    """Jets of ``log P`` of a window ``2 half_width`` wide by Gauss-Legendre quadrature.

    ``log_density_jets(fraction)`` gives the log density's jet (at least
    ``order + 1`` entries) at the point that far across each row's window,
    ``0 < fraction < 1``; the caller forms that point in coordinates where it
    does not round away.  ``F = log sum_i omega_i exp(l_i(theta))`` has, with node
    weights ``p_i`` proportional to ``omega_i f(y_i)``, the derivatives

        F_I = sum over set partitions pi of I of kappa_p(l_B : B in pi),

    the joint cumulants under ``p`` of the blocks' node derivatives ``l_B``:
    the cumulant expansion ``log E_p exp(X) = sum_r kappa_r(X)/r!`` of the
    Taylor increment ``X`` of ``l``.  So ``F_ij = E[l_ij] + Cov(l_i, l_j)``,
    order 3 adds ``3 sym Cov(l_ij, l_k)`` and the third central moment of the
    score, and order 4 the covariances of third with first and second with
    second derivatives, the third cumulants of a second derivative with two
    scores and the scores' fourth cumulant.  Each cumulant is formed from
    node values centred at their mixture means, as the narrow window makes
    them nearly equal.
    """
    order = _checked_order(order)
    nodes = [log_density_jets(0.5 * (1.0 + float(t)))[: order + 1] for t in _LEGENDRE_NODES]
    logs = np.column_stack(
        [np.log(omega) + node[0] for omega, node in zip(_LEGENDRE_WEIGHTS, nodes, strict=True)]
    )
    top = np.max(logs, axis=1)
    share = np.exp(logs - np.where(np.isfinite(top), top, 0.0)[:, None])
    total = np.sum(share, axis=1)
    share /= np.where(total > 0.0, total, 1.0)[:, None]
    with np.errstate(divide="ignore"):
        value = np.log(half_width) + top + np.log(total)
    if order == 0:
        return (value,)
    k_parameters = nodes[0][1].shape[1]
    means = {
        m: sum(share[:, [i]] * node[m] for i, node in enumerate(nodes)) for m in range(1, order + 1)
    }
    centred = {m: [node[m] - means[m] for node in nodes] for m in range(1, order)}
    columns = {
        m: {key: column for column, key in enumerate(packed_tuples(k_parameters, m))}
        for m in range(1, order)
    }
    jet = [value, means[1]]
    for m in range(2, order + 1):
        derivative = np.array(means[m], dtype=FLOAT, copy=True)
        for column, partitions in enumerate(_block_keys(k_parameters, m)):
            for keys in partitions:
                factors = [
                    [node[:, columns[len(key)][key]] for node in centred[len(key)]] for key in keys
                ]
                derivative[:, column] += _node_cumulant(share, factors)
        jet.append(derivative)
    return tuple(jet)
