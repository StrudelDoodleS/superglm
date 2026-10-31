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

``pair_cell_moments`` and ``working_score`` are live: the factor builders of
:mod:`superglm.screening._pair_factor` consume exactly the cell tables the
first produces.  The Gram producers the dense route used to assemble from them
are the definition those factors are graded against, and live in
``tests/_screening_oracle.py`` and ``tests/_overlap_oracle.py``.
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
    *,
    max_cells: int = 5_000_000,
) -> tuple[NDArray, NDArray]:
    """Aggregate score and working weights over the pair's joint cells.

    One fused O(n) pass; returns ``(S_cell, W_cell)`` with shape
    ``(n_a, n_b)``.  Codes must be dense 0-based indices in ``[0, n_a)`` /
    ``[0, n_b)``.  Support-compressed groups store exactly these.
    ``CategoricalGroupMatrix`` codes do NOT satisfy this as stored: they
    carry a sink bin at ``n_levels`` for the base level, so callers must
    strip base-level rows or widen the grid to ``n_levels + 1``.  The bounds
    are enforced here because the kernel below indexes without checks — an
    out-of-range code would be silent memory corruption, not an exception.
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
