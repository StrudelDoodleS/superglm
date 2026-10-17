"""Directional derivative of predictor curvature from natural-coordinate derivatives.

With natural parameters ``theta_k = g_k(eta_k)`` the predictor Hessian is
``l_eta[l, r] = H[l, r] g'_l g'_r + delta_lr s_l g''_l``.  Differentiating it
along a predictor direction ``d`` with ``v = g' * d`` (the natural direction)
gives

    dl_eta[l, r] = g'_l g'_r dH[l, r](v)
                   + H[l, r] (g''_l d_l g'_r + g'_l g''_r d_r)
                   + delta_lr (g''_l (H v)_l + s_l g'''_l d_l),

where ``dH(v) = sum_k T[:, :, k] v_k`` is the natural Hessian's derivative
along ``v``, the one family-specific input.  The packed curvature is
``-l_eta``, so its derivative is the negation.  Every output is a sum of at
most ``K + 4`` products of at most four rounded factors, so it carries a
relative error of ``gamma_{2K + 12}`` of its absolute-term scale on top of
the inputs' own errors.

One order up, the curvature's second directional derivative is the
negated fourth derivative of the predictor log-likelihood contracted with
two directions.  :func:`compose_derivatives` forms any order of a
composition by the set-partition chain rule (Hardy 2006; for one univariate
link, Wood, Pya and Saefken 2016, Appendix A, eq. 14), which serves both the
links and a mean form's reparametrisation; :func:`curvature_second_direction`
contracts the result.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

#: ``inner[m][(c, idx)]`` is ``d^m phi_c / d x_idx`` for a sorted index tuple
#: ``idx`` of length ``m``; entries left out are structurally zero.
InnerDerivatives = Mapping[int, Mapping[tuple[int, tuple[int, ...]], NDArray[np.float64] | float]]


def _pairs(k_parameters: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (left, right) for left in range(k_parameters) for right in range(left, k_parameters)
    )


def predictor_curvature_direction(
    score: NDArray[np.float64],
    hessian: NDArray[np.float64],
    hessian_direction: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    third: NDArray[np.float64],
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Packed ``d curvature / d eta`` along ``direction``; inputs are natural and per row.

    ``score`` is ``(n, K)``, ``hessian`` and ``hessian_direction`` are packed
    ``(n, K(K+1)/2)`` in the canonical upper-triangular order, and ``first``,
    ``second`` and ``third`` are the inverse links' derivatives ``(n, K)``.
    """
    n_rows, k_parameters = score.shape
    pairs = _pairs(k_parameters)
    index = {pair: column for column, pair in enumerate(pairs)}

    def entry(values: NDArray[np.float64], left: int, right: int) -> NDArray[np.float64]:
        return values[:, index[(left, right) if left <= right else (right, left)]]

    natural_direction = first * direction
    hessian_times_direction = np.zeros((n_rows, k_parameters), dtype=np.float64)
    for left in range(k_parameters):
        for right in range(k_parameters):
            hessian_times_direction[:, left] += (
                entry(hessian, left, right) * natural_direction[:, right]
            )
    result = np.empty((n_rows, len(pairs)), dtype=np.float64)
    for column, (left, right) in enumerate(pairs):
        value = first[:, left] * first[:, right] * hessian_direction[:, column] + hessian[
            :, column
        ] * (
            second[:, left] * direction[:, left] * first[:, right]
            + first[:, left] * second[:, right] * direction[:, right]
        )
        if left == right:
            value = (
                value
                + second[:, left] * hessian_times_direction[:, left]
                + score[:, left] * third[:, left] * direction[:, left]
            )
        result[:, column] = -value
    return result


def mean_form_hessian_direction(
    mean: NDArray[np.float64],
    location_score: NDArray[np.float64],
    location_hessian: NDArray[np.float64],
    location_direction,
    loading_first: tuple[NDArray[np.float64], NDArray[np.float64]],
    loading_second: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    loading_third: tuple[NDArray[np.float64], ...],
    direction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean-form ``dH(v)`` in ``(m, a, b)`` from the location form in ``(mu, a, b)``.

    ``mu = log m - log C(a, b)`` has Jacobian ``J`` (identity except the ``mu``
    row ``(1/m, -C_a, -C_b)``) and second and third derivatives only in ``mu``.
    With ``w = J v``, ``Z_i = sum_k mu_ik v_k`` and ``mu_ij(v) = sum_k mu_ijk v_k``:

        dH(v) = J' dH_loc(w) J + Z (H J)_mu' + (H J)_mu Z' + mu_ij (H w)_mu
                + s_mu mu_ij(v).

    ``location_hessian`` is packed (3 parameters, unit mass), ``location_direction(w)``
    returns the location form's packed ``dH_loc(w)``, ``loading_third`` is
    ``(C_aaa, C_aab, C_abb, C_bbb)``.
    """
    c_a, c_b = loading_first
    c_aa, c_ab, c_bb = loading_second
    c_aaa, c_aab, c_abb, c_bbb = loading_third
    n_rows = len(mean)
    jacobian = np.zeros((n_rows, 3, 3), dtype=np.float64)
    jacobian[:, 0, 0] = 1.0 / mean
    jacobian[:, 0, 1] = -c_a
    jacobian[:, 0, 2] = -c_b
    jacobian[:, 1, 1] = 1.0
    jacobian[:, 2, 2] = 1.0
    second = np.zeros((n_rows, 3, 3), dtype=np.float64)
    second[:, 0, 0] = -1.0 / (mean * mean)
    second[:, 1, 1] = -c_aa
    second[:, 1, 2] = second[:, 2, 1] = -c_ab
    second[:, 2, 2] = -c_bb
    vm, va, vb = direction[:, 0], direction[:, 1], direction[:, 2]
    third = np.zeros((n_rows, 3, 3), dtype=np.float64)
    third[:, 0, 0] = 2.0 / (mean * mean * mean) * vm
    third[:, 1, 1] = -(c_aaa * va + c_aab * vb)
    third[:, 1, 2] = third[:, 2, 1] = -(c_aab * va + c_abb * vb)
    third[:, 2, 2] = -(c_abb * va + c_bbb * vb)
    w = np.matmul(jacobian, direction[:, :, None])[:, :, 0]
    pairs = _pairs(3)
    hessian = np.empty((n_rows, 3, 3), dtype=np.float64)
    location = np.empty((n_rows, 3, 3), dtype=np.float64)
    packed_location = location_direction(w)
    for column, (left, right) in enumerate(pairs):
        hessian[:, left, right] = hessian[:, right, left] = location_hessian[:, column]
        location[:, left, right] = location[:, right, left] = packed_location[:, column]
    z = np.matmul(second, direction[:, :, None])[:, :, 0]
    hessian_jacobian_mu = np.matmul(hessian[:, 0:1, :], jacobian)[:, 0, :]
    hessian_w_mu = np.sum(hessian[:, 0, :] * w, axis=1)
    full = (
        np.matmul(np.matmul(np.swapaxes(jacobian, 1, 2), location), jacobian)
        + z[:, :, None] * hessian_jacobian_mu[:, None, :]
        + hessian_jacobian_mu[:, :, None] * z[:, None, :]
        + second * hessian_w_mu[:, None, None]
        + location_score[:, None, None] * third
    )
    return np.column_stack([full[:, left, right] for left, right in pairs])


def packed_tuples(k_parameters: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Sorted index tuples of a symmetric tensor's distinct entries, in packed order.

    Order two is the canonical upper-triangular packing of the Hessian; every
    order lists ``itertools.combinations_with_replacement`` order, so the
    third derivatives of three parameters are ``000, 001, 002, 011, ..., 222``.
    """
    return tuple(itertools.combinations_with_replacement(range(k_parameters), order))


def _set_partitions(positions: list[int]) -> list[list[list[int]]]:
    if not positions:
        return [[]]
    first, rest = positions[0], positions[1:]
    partitions = []
    for partition in _set_partitions(rest):
        for index in range(len(partition)):
            partitions.append(
                partition[:index] + [[first, *partition[index]]] + partition[index + 1 :]
            )
        partitions.append([[first], *partition])
    return partitions


def compose_derivatives(
    outer: Mapping[int, NDArray[np.float64]],
    inner: InnerDerivatives,
    order: int,
    k_parameters: int,
    *,
    components: int | None = None,
) -> NDArray[np.float64]:
    """Packed ``order``-th derivatives of ``L(phi(x))`` by the set-partition chain rule.

    ``outer[m]`` (``m = 1..order``) holds the packed ``m``-th derivatives of
    ``L`` at ``phi(x)``, ``(n, C(K + m - 1, m))`` in :func:`packed_tuples`
    order, and ``inner`` the structurally nonzero derivatives of ``phi``.  For
    each distinct entry ``(i_1..i_order)`` of the result and each partition of
    its positions into blocks ``B``, the term is the outer derivative over one
    component ``c_B`` per block times ``prod_B d^|B| phi_{c_B} / d x_B``
    (Hardy, *Combinatorics of partial derivatives*, Electron. J. Combin. 13,
    2006, R1; for one univariate link this is Wood, Pya and Saefken 2016,
    Appendix A).  Only the structural nonzeros are visited, so a separable
    link map costs at most one term per partition.  ``components`` is the
    number of ``phi``'s components when it differs from its ``K`` inputs.
    """
    n_components = k_parameters if components is None else components
    columns = {
        m: {key: column for column, key in enumerate(packed_tuples(n_components, m))}
        for m in range(1, order + 1)
    }
    targets = packed_tuples(k_parameters, order)
    n_rows = outer[1].shape[0]
    result = np.zeros((n_rows, len(targets)), dtype=np.float64)
    partitions = _set_partitions(list(range(order)))
    for column, target in enumerate(targets):
        for partition in partitions:
            options = []
            for block in partition:
                key = tuple(sorted(target[position] for position in block))
                level = inner.get(len(block), {})
                choices = [
                    (component, level[(component, key)])
                    for component in range(n_components)
                    if (component, key) in level
                ]
                if not choices:
                    break
                options.append(choices)
            else:
                for choice in itertools.product(*options):
                    chosen = tuple(sorted(component for component, _ in choice))
                    term = outer[len(chosen)][:, columns[len(chosen)][chosen]]
                    for _, factor in choice:
                        term = term * factor
                    result[:, column] += term
    return result


def stack_columns(values, n_rows: int) -> NDArray[np.float64]:
    """Column-stack generated outputs, broadcasting the scalar zeros, as ``(n, len(values))``."""
    result = np.empty((n_rows, len(values)), dtype=np.float64)
    for column, value in enumerate(values):
        result[:, column] = value
    return result


def separable_inner_derivatives(
    *derivatives: NDArray[np.float64],
) -> dict[int, dict[tuple[int, tuple[int, ...]], NDArray[np.float64]]]:
    """``inner`` for ``theta_k = g_k(eta_k)`` from the link derivatives, each ``(n, K)``."""
    k_parameters = derivatives[0].shape[1]
    return {
        order: {(k, (k,) * order): values[:, k] for k in range(k_parameters)}
        for order, values in enumerate(derivatives, start=1)
    }


def third_from_hessian_direction(
    hessian_direction: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_rows: int,
    k_parameters: int,
) -> NDArray[np.float64]:
    """Packed natural third derivatives from a family's ``dH(v)`` on the natural axes.

    ``dH(e_k)[i, j]`` is ``L_ijk``, so ``K`` evaluations of the existing
    direction give every distinct third derivative.
    """
    pairs = packed_tuples(k_parameters, 2)
    triples = {key: column for column, key in enumerate(packed_tuples(k_parameters, 3))}
    third = np.empty((n_rows, len(triples)), dtype=np.float64)
    for k in range(k_parameters):
        axis = np.zeros((n_rows, k_parameters), dtype=np.float64)
        axis[:, k] = 1.0
        packed = np.asarray(hessian_direction(axis), dtype=np.float64)
        for column, (i, j) in enumerate(pairs):
            if j <= k:
                third[:, triples[(i, j, k)]] = packed[:, column]
    return third


def curvature_second_direction(
    fourth: NDArray[np.float64],
    direction_a: NDArray[np.float64],
    direction_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Packed ``-sum_kl l_ijkl a_k b_l`` from the packed fourth derivatives ``l``.

    The directions are ``(n, K)``, or ``(m, n, K)`` for ``m`` pairs at once,
    where a pair the same on every row may be ``(m, 1, K)``; the result is
    ``(n, C)`` or ``(m, n, C)`` with ``C = K(K+1)/2`` packed channels.
    """
    k_parameters = direction_a.shape[-1]
    quadruples = {key: column for column, key in enumerate(packed_tuples(k_parameters, 4))}
    pairs = packed_tuples(k_parameters, 2)
    leading = (*direction_a.shape[:-2], fourth.shape[0])
    result = np.zeros((*leading, len(pairs)), dtype=np.float64)
    if direction_a.ndim == 3 and direction_a.shape[1] == 1:
        # Pairs the same on every row (the LAML Hessian's axis pairs): only the
        # (k, l) whose coefficient a_k b_l is nonzero touch the rows.
        for index in range(direction_a.shape[0]):
            weights = np.outer(direction_a[index, 0], direction_b[index, 0])
            terms = [(k, ell) for k in range(k_parameters) for ell in range(k_parameters)]
            live = [(k, ell) for k, ell in terms if weights[k, ell] != 0.0]
            for column, (i, j) in enumerate(pairs):
                for k, ell in live:
                    entry = fourth[:, quadruples[tuple(sorted((i, j, k, ell)))]]
                    result[index, :, column] -= weights[k, ell] * entry
        return result
    for column, (i, j) in enumerate(pairs):
        for k in range(k_parameters):
            for ell in range(k_parameters):
                entry = fourth[:, quadruples[tuple(sorted((i, j, k, ell)))]]
                result[..., column] -= entry * direction_a[..., k] * direction_b[..., ell]
    return result


def predictor_curvature_second_direction(
    natural: tuple[NDArray[np.float64], ...],
    links: tuple[NDArray[np.float64], ...],
    direction_a: NDArray[np.float64],
    direction_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Packed second directional derivative of predictor curvature from natural derivatives.

    ``natural`` is ``(score, hessian, third, fourth)``, packed, and ``links``
    the inverse links' first four derivatives, each ``(n, K)``.  The
    predictor log-likelihood's fourth derivatives are the natural ones
    composed with the separable links (:func:`compose_derivatives`); the
    curvature is their negation contracted with ``a`` and ``b``.
    """
    k_parameters = links[0].shape[1]
    outer = dict(enumerate(natural, start=1))
    fourth = compose_derivatives(outer, separable_inner_derivatives(*links), 4, k_parameters)
    return curvature_second_direction(fourth, direction_a, direction_b)


__all__ = [
    "InnerDerivatives",
    "compose_derivatives",
    "curvature_second_direction",
    "mean_form_hessian_direction",
    "packed_tuples",
    "predictor_curvature_direction",
    "predictor_curvature_second_direction",
    "separable_inner_derivatives",
    "stack_columns",
    "third_from_hessian_direction",
]
