"""Per-pair sufficient statistics for interaction screening.

A candidate tensor block over covariates (a, b) needs only two quantities to
form its score statistic against a fitted mains model: the null score vector
aggregated over the pair's joint cells, and the working weights aggregated the
same way.  Both come from one fused O(n) pass; everything downstream is dense
algebra over ``(n_a, n_b)`` cells and never touches rows again.

Exactness contract: the cell-space assembly must reproduce the dense
row-Kronecker assembly to floating-point reordering, because the same
sufficient-statistic identity underpins lossless support compression.  The
release pin lives in tests/test_interaction_screening.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_kernels import _fused_bincount_2
from superglm.distributions import _VARIANCE_FLOOR


def working_score(
    y: NDArray,
    mu: NDArray,
    eta: NDArray,
    weights: NDArray,
    distribution,
    link,
) -> NDArray:
    """Score of the unnormalised log-likelihood with respect to eta, per row.

    ``s = weights * (dmu/deta) * (y - mu) / V(mu)`` — the exact quantity the
    solver's KKT threshold uses (``compute_lambda_max`` calls this with the
    null-model ``mu``), and the residual signal screening aggregates over a
    candidate pair's cells (with the fitted mains' ``mu``).  Reduces to
    ``weights * (y - mu)`` for canonical links.
    """
    dmu_deta = link.deriv_inverse(eta)
    variance = np.maximum(distribution.variance(mu), _VARIANCE_FLOOR)
    return weights * dmu_deta * (y - mu) / variance


def pair_cell_moments(
    codes_a: NDArray,
    codes_b: NDArray,
    n_a: int,
    n_b: int,
    score: NDArray,
    working_weights: NDArray,
) -> tuple[NDArray, NDArray]:
    """Aggregate score and working weights over the pair's joint cells.

    One fused O(n) pass; returns ``(S_cell, W_cell)`` with shape
    ``(n_a, n_b)``.  Codes must already be dense 0-based level indices —
    support-compressed groups store exactly these, and categoricals store
    their level codes.
    """
    codes_a = np.asarray(codes_a, dtype=np.intp)
    codes_b = np.asarray(codes_b, dtype=np.intp)
    if codes_a.shape != codes_b.shape:
        raise ValueError("pair codes must share one row dimension")
    joint = codes_a * np.intp(n_b) + codes_b
    w_flat, s_flat = _fused_bincount_2(
        joint,
        np.ascontiguousarray(working_weights, dtype=np.float64),
        np.ascontiguousarray(score, dtype=np.float64),
        int(n_a) * int(n_b),
    )
    return s_flat.reshape(n_a, n_b), w_flat.reshape(n_a, n_b)


def pair_score_curvature(
    B_a: NDArray,
    B_b: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
) -> tuple[NDArray, NDArray]:
    """Score vector and unadjusted curvature of the pair's tensor block.

    With ``X_T`` the row-Kronecker design ``kron(B_a[i_a[r]], B_b[i_b[r]])``:

    ``U = X_T' s  = vec(B_a' S_cell B_b)``
    ``V = X_T' diag(W) X_T``, assembled from ``W_cell`` without forming rows:
    ``V[(p,q),(r,s)] = sum_i B_a[i,p] B_a[i,r] * (sum_j W[i,j] B_b[j,q] B_b[j,s])``

    Flattening is C-order, matching ``np.kron`` column ordering.
    """
    B_a = np.asarray(B_a, dtype=np.float64)
    B_b = np.asarray(B_b, dtype=np.float64)
    k_a, k_b = B_a.shape[1], B_b.shape[1]
    U = (B_a.T @ S_cell @ B_b).reshape(k_a * k_b)
    inner = np.einsum("ij,jq,js->iqs", W_cell, B_b, B_b, optimize=True)
    V = np.einsum("ip,ir,iqs->pqrs", B_a, B_a, inner, optimize=True)
    return U, V.reshape(k_a * k_b, k_a * k_b)
