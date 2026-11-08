"""Log probabilities of response windows, for truncated and censored rows, on packed jets.

A truncated row divides its density by ``P(a < Y <= b)``; a censored row
contributes ``log P(l < Y <= r)``. Both are ``log(T(near) - T(far))`` for one
tail function ``T`` at the window's ends. With
``Delta = log T(far) - log T(near) <= 0`` and ``kappa = 1/expm1(-Delta)``::

    log D = log T(near) + log(-expm1(Delta))
    grad  = g_near + kappa d
    hess  = H_near + kappa (H_near - H_far) - kappa (1 + kappa) d d^T

with ``d = g_near - g_far``. The difference of two log tails keeps only about
``u/Delta`` relative accuracy, so a window narrower than :data:`NARROW_WINDOW`
on its density's scale uses three-point Gauss-Legendre quadrature instead
(error about ``5e-7`` times the sixth power of the width; Abramowitz and
Stegun 25.4.30). Imports only numpy and the shared chain-rule helpers.
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


def _checked_core_order(order: int) -> int:
    if (
        isinstance(order, bool | np.bool_)
        or not isinstance(order, int | np.integer)
        or order not in (0, 1, 2)
    ):
        raise ValueError("core mixture order must be an integer from 0 through 2")
    return int(order)


class CoreLogMixtureAccumulator:
    """Streaming positive log-sum jets, with O(n * (K + K(K+1)/2)) state.

    Each ``add`` consumes one component's row-first log jet. Scaled positive
    masses determine the shares; centered score moments give
    ``H = E[H_component] + Cov(score_component)``. Score means are stored
    relative to a per-row anchor, so a large common score cannot erase a small
    variance; a newly dominant component becomes the anchor. Only this
    accumulator owns the temporary state; it is never
    cached or reused across parameter states except by explicitly adding terms.

    A ``-inf`` log value declares an *identically absent* component: its
    derivatives are ignored. It must not denote a moving split piece with
    boundary derivatives. NaN/+inf values or nonfinite requested derivatives
    of any finite component poison that row, even if its scaled mass underflows.
    ``finish`` returns NaNs for unresolved rows; all-absent rows have value
    ``-inf`` and undefined (NaN) log derivatives. Finishing before any add
    raises, since no row count has been established. Returned arrays are copies.
    """

    def __init__(self, k_parameters: int, order: int):
        self.order = _checked_core_order(order)
        if (
            isinstance(k_parameters, bool | np.bool_)
            or not isinstance(k_parameters, int | np.integer)
            or k_parameters < 1
        ):
            raise ValueError("k_parameters must be a positive integer")
        self.k_parameters = int(k_parameters)
        self._top: NDArray | None = None

    def add(self, jet: Jet) -> None:
        """Add one component; channels use the existing ``packed_tuples`` order."""
        if len(jet) < self.order + 1:
            raise ValueError("component jet is missing a requested derivative order")
        channels = tuple(np.asarray(j, dtype=FLOAT) for j in jet[: self.order + 1])
        value = channels[0]
        if value.ndim != 1:
            raise ValueError("component values must have shape (n,)")
        n = len(value)
        for degree in range(1, self.order + 1):
            expected = (n, len(packed_tuples(self.k_parameters, degree)))
            if channels[degree].shape != expected:
                raise ValueError(f"component order-{degree} shape must be {expected}")
        if self._top is None:
            self._top = np.full(n, -np.inf)
            self._mass = np.zeros(n)
            self._invalid = np.zeros(n, dtype=bool)
            if self.order >= 1:
                self._anchor = np.zeros((n, self.k_parameters))
                self._mean = np.zeros_like(self._anchor)
            if self.order == 2:
                self._hessian = np.zeros((n, len(packed_tuples(self.k_parameters, 2))))
        elif value.shape != self._top.shape:
            raise ValueError("component row shape must match previous components")

        absent = value == -np.inf
        finite = np.isfinite(value)
        for channel in channels[1:]:
            finite &= np.all(np.isfinite(channel), axis=1)
        self._invalid |= ~absent & ~finite
        present = ~absent & ~self._invalid
        fresh = present & (self._mass == 0.0)
        combined = np.flatnonzero(present & ~fresh)
        self._top[fresh], self._mass[fresh] = value[fresh], 1.0
        if self.order >= 1:
            self._anchor[fresh] = channels[1][fresh]
        if self.order == 2:
            self._hessian[fresh] = channels[2][fresh]
        if len(combined) == 0:
            return

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            top = np.maximum(self._top[combined], value[combined])
            log_left = np.log(self._mass[combined]) + (self._top[combined] - top)
            log_right = value[combined] - top
            left = self._mass[combined] * np.exp(self._top[combined] - top)
            right = np.exp(log_right)
            mass = left + right
            self._top[combined], self._mass[combined] = top, mass
            if self.order == 0:
                return

            log_p, log_q = log_left - np.log(mass), log_right - np.log(mass)
            root_p, root_q = np.exp(0.5 * log_p)[:, None], np.exp(0.5 * log_q)[:, None]
            centered = channels[1][combined] - self._anchor[combined]
            delta = centered - self._mean[combined]
            left_centered = self._mean[combined].copy()
            promote = log_right > log_left
            left_centered[promote] = -delta[promote]
            centered[promote] = 0.0
            self._anchor[combined[promote]] = channels[1][combined[promote]]
            # Form p*x as (sqrt(p)*x)*sqrt(p): p may underflow while p*x
            # is still representable. Reanchoring also avoids canceling a
            # huge old anchor when its remaining weighted score is tiny.
            self._mean[combined] = (left_centered * root_p) * root_p + (centered * root_q) * root_q
            if self.order == 2:
                # Split sqrt(p*q) into two fourth roots before scaling delta.
                # Even sqrt(p*q) may underflow although p*q*delta**2 does not.
                fourth_root = np.exp(0.25 * log_p + 0.25 * log_q)[:, None]
                scaled = (delta * fourth_root) * fourth_root
                covariance = np.column_stack(
                    [scaled[:, i] * scaled[:, j] for i, j in packed_tuples(self.k_parameters, 2)]
                )
                self._hessian[combined] = (
                    (self._hessian[combined] * root_p) * root_p
                    + (channels[2][combined] * root_q) * root_q
                    + covariance
                )

    def finish(self) -> Jet:
        """Return the accumulated core jet without changing the accumulator."""
        if self._top is None:
            raise ValueError("an empty accumulator needs a component to establish its rows")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            value = self._top + np.log(self._mass)
            result = [value]
            if self.order >= 1:
                result.append(self._anchor + self._mean)
            if self.order == 2:
                result.append(self._hessian.copy())
        empty = self._mass == 0.0
        invalid = self._invalid | (~empty & ~np.isfinite(value))
        for channel in result[1:]:
            invalid |= ~empty & ~np.all(np.isfinite(channel), axis=1)
            channel[empty] = np.nan
        for channel in result:
            channel[invalid] = np.nan
        return tuple(result)


def log_mixture_jets(
    log_terms: NDArray,
    term_score: NDArray | None = None,
    term_hessian: NDArray | None = None,
    *,
    order: int,
) -> Jet:
    """Positive-sum jets for a small batch in row-first ``(n, terms)`` layout.

    Scores are ``(n, terms, K)`` and Hessians ``(n, terms, K(K+1)/2)`` in
    ``packed_tuples(K, 2)`` order. Only requested channels are read. Empty sums
    and absent/unresolved components follow :class:`CoreLogMixtureAccumulator`;
    the batch uses that same centered-moment implementation, one term at a time.
    """
    order = _checked_core_order(order)
    logs = np.asarray(log_terms, dtype=FLOAT)
    if logs.ndim != 2:
        raise ValueError("log_terms must have two-dimensional shape (n, terms)")
    channels = [logs]
    k_parameters = 1
    if order >= 1:
        score = np.asarray(term_score, dtype=FLOAT)
        if score.ndim != 3 or score.shape[:2] != logs.shape or score.shape[2] < 1:
            raise ValueError("term_score must have shape (n, terms, K), with K positive")
        k_parameters = score.shape[2]
        channels.append(score)
    if order == 2:
        hessian = np.asarray(term_hessian, dtype=FLOAT)
        if hessian.shape != (*logs.shape, k_parameters * (k_parameters + 1) // 2):
            raise ValueError("term_hessian must have shape (n, terms, K*(K+1)//2)")
        channels.append(hessian)
    accumulator = CoreLogMixtureAccumulator(k_parameters, order)
    if logs.shape[1] == 0:
        accumulator.add(
            (np.full(logs.shape[0], -np.inf),)
            + tuple(np.zeros((logs.shape[0], c.shape[2])) for c in channels[1:])
        )
    for term in range(logs.shape[1]):
        accumulator.add(tuple(c[:, term] for c in channels))
    return accumulator.finish()


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

    ``near`` and ``far`` are log-tail jets ``(J0, J1, ...)`` in
    :func:`packed_tuples` order. With ``Delta = J0_far - J0_near``,
    ``log D = J0_near + phi(Delta)``, differentiated by the set-partition chain
    rule (Constantine and Savits, Trans. AMS 348, 1996; Hardy 2006). An absent
    (``-inf``) or negligible (``kappa`` rounds to zero) far end returns the
    near jets; ends whose tails round equal give ``-inf``.
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

    ``log_density_jets(fraction)`` gives the log density's jet at that
    fraction across each row's window (the caller forms the point so it does
    not round away). With node weights ``p_i`` proportional to
    ``omega_i f(y_i)``, ``F = log sum_i omega_i exp(l_i)`` has derivatives

        F_I = sum over set partitions pi of I of kappa_p(l_B : B in pi),

    the joint cumulants under ``p`` of the node derivatives (e.g.
    ``F_ij = E[l_ij] + Cov(l_i, l_j)``), formed from values centred at their
    mixture means since the narrow window makes them nearly equal.
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
