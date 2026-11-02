"""Per-pair sufficient statistics for interaction screening.

A candidate tensor block over covariates (a, b) needs the null score vector and the
working weights aggregated over the pair's joint cells, from one fused O(n) pass;
everything downstream is dense algebra over ``(n_a, n_b)`` cells. The cell-space
assembly must reproduce the dense row-Kronecker assembly to floating-point
reordering (pinned in tests/test_interaction_screening.py). The Gram producers the
factors are graded against live in ``tests/_screening_oracle.py`` and
``tests/_overlap_oracle.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_kernels import _fused_bincount_2, _native
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

    ``s = weights * (dmu/deta) * (y - mu) / V(mu)``, the quantity the solver's KKT
    threshold uses and the residual signal screening aggregates over a pair's cells;
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
    *,
    max_cells: int = 5_000_000,
) -> tuple[NDArray, NDArray]:
    """Aggregate score and working weights over the pair's joint cells.

    One fused O(n) pass; returns ``(S_cell, W_cell)`` of shape ``(n_a, n_b)``. Codes must
    be dense 0-based indices in ``[0, n_a)`` / ``[0, n_b)`` (checked here, since the
    kernel indexes without bounds checks). ``CategoricalGroupMatrix`` codes carry a sink
    bin at ``n_levels`` for the base level, so callers strip base-level rows or widen the
    grid to ``n_levels + 1``.
    """
    codes_a = np.asarray(codes_a, dtype=np.intp)
    codes_b = np.asarray(codes_b, dtype=np.intp)
    if codes_a.shape != codes_b.shape:
        raise ValueError("pair codes must share one row dimension")
    score = np.ascontiguousarray(score, dtype=np.float64)
    working_weights = np.ascontiguousarray(working_weights, dtype=np.float64)
    if score.shape != codes_a.shape or working_weights.shape != codes_a.shape:
        raise ValueError("score and working weights must match the pair codes row-for-row")
    n_cells = int(n_a) * int(n_b)
    if n_cells > max_cells:
        raise ValueError(
            f"pair grid has {n_cells} cells, above the {max_cells} ceiling; "
            "bin the wide margin or raise max_cells explicitly"
        )
    if codes_a.size:
        if int(codes_a.min()) < 0 or int(codes_a.max()) >= n_a:
            raise ValueError("codes_a fall outside [0, n_a)")
        if int(codes_b.min()) < 0 or int(codes_b.max()) >= n_b:
            raise ValueError("codes_b fall outside [0, n_b)")
    joint = codes_a * np.intp(n_b) + codes_b
    w_flat, s_flat = _fused_bincount_2(joint, _native(working_weights), _native(score), n_cells)
    return s_flat.reshape(n_a, n_b), w_flat.reshape(n_a, n_b)
