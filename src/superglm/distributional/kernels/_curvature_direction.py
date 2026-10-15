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
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    w = np.einsum("nai,ni->na", jacobian, direction)
    pairs = _pairs(3)
    hessian = np.empty((n_rows, 3, 3), dtype=np.float64)
    location = np.empty((n_rows, 3, 3), dtype=np.float64)
    packed_location = location_direction(w)
    for column, (left, right) in enumerate(pairs):
        hessian[:, left, right] = hessian[:, right, left] = location_hessian[:, column]
        location[:, left, right] = location[:, right, left] = packed_location[:, column]
    z = np.einsum("nij,nj->ni", second, direction)
    hessian_jacobian_mu = np.einsum("nb,nbj->nj", hessian[:, 0, :], jacobian)
    hessian_w_mu = np.einsum("nb,nb->n", hessian[:, 0, :], w)
    full = (
        np.einsum("nai,nab,nbj->nij", jacobian, location, jacobian)
        + z[:, :, None] * hessian_jacobian_mu[:, None, :]
        + hessian_jacobian_mu[:, :, None] * z[:, None, :]
        + second * hessian_w_mu[:, None, None]
        + location_score[:, None, None] * third
    )
    return np.column_stack([full[:, left, right] for left, right in pairs])


__all__ = ["mean_form_hessian_direction", "predictor_curvature_direction"]
