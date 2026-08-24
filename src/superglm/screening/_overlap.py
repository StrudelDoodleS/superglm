"""Overlap moments for PSST profiling, assembled from the pair's cell tables.

The overlap span is what the candidate pair's own mains already cover:
``[intercept | A | B]`` with ``A``/``B`` the centered marginal menus evaluated
on each covariate's support.  Profiling (Task 2) needs three quantities —
``M = X_o' W X_o``, ``C = X_o' W X_T`` and ``u_m = X_o' s`` — and every one of
them reduces to row-sums, column-sums and small contractions of the same
``(n_a, n_b)`` cell tables Task 1 builds, so no additional data pass exists.

Column order of the tensor block is C-order ``p * k_b + q``, matching
``features/interaction.py``'s row-Kronecker and ``pair_score_curvature``.

``tensor_penalty_root`` is live and is what the dense ladder is handed.
``pair_overlap_moments`` and ``tensor_penalty`` are not.

**NO LONGER THE PRODUCTION ROUTE, AND KEPT DELIBERATELY.**  Issue #257 moved
the dense screening path onto design factors -- ``_pair_factor`` reduces the
same cell tables to one triangular factor and the ladder reads ``V_eff`` off a
block of it, so no Gram is formed and the spectrum the statistic works in is
the design's rather than its square.  What is below is the DEFINITION that
factor is graded against: every exactness pin in
``tests/test_pair_design_factor.py`` compares the two, and
``test_the_dense_path_s_ceiling_is_its_gram_and_not_its_arithmetic`` counts the
directions the Gram cannot resolve, which needs the Gram.  Precedent for
keeping a live-tested non-caller is ``_reference_edf`` in the structured suite.
An import guard keeps it from becoming production again by accident:
``test_no_production_module_imports_a_retired_gram_producer``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from superglm.screening._factor_kernels import _penalty_spectrum


def tensor_penalty(S1: NDArray, S2: NDArray) -> NDArray:
    """``kron(S1, I) + kron(I, S2)`` — interaction.py's ti() penalty order.

    **NO LONGER THE PRODUCTION ROUTE, AND KEPT AS AN ARBITER.**  The dense
    ladder is handed :func:`tensor_penalty_root` instead, which never assembles
    this matrix; what is left here is the definition that root is graded
    against, and the assembled form the guide and the tests quote.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    return np.kron(S1, np.eye(S2.shape[0])) + np.kron(np.eye(S1.shape[0]), S2)


def tensor_penalty_root(S1: NDArray, S2: NDArray) -> NDArray:
    """``rootS`` with ``rootS' rootS`` the tensor penalty, from the MARGINS.

    ``kron(S1, I) + kron(I, S2)`` is a Kronecker SUM, so with
    ``S1 = Qa diag(a) Qa'`` and ``S2 = Qb diag(b) Qb'`` its eigenvectors are
    ``kron(Qa, Qb)`` and its eigenvalues are ``a_p + b_q`` -- exactly, and in
    the C-order ``p * k_b + q`` the tensor block already uses.  The root is
    therefore assembled rather than computed, and the only eigendecompositions
    taken are of the two MARGIN penalties.

    **THAT IS WHERE THE DENSE LADDER'S HIGH-EDGE ACCURACY COMES FROM, AND IT
    IS MEASURED.**  ``eigh`` is accurate to ``n eps ||S||_2`` ABSOLUTELY, so a
    direction that is exactly null in exact arithmetic comes back at the
    dimension's round-off -- and the ladder's high edge multiplies it by
    ``1e10``.  On ``moderate_pair`` the penalty is ``kron(S_a, I_23)`` with
    ``S_a`` nine wide and one exactly-null direction; its assembly is 207 wide
    and ``eigh`` resolves that direction only to 6.94e-13 of ``||S_ti||_2``,
    where on ``S_a`` itself it resolves to 4.3e-17.  Through the ladder, the
    dense arm's high-edge ``edf`` reads 8.26e-07 from the stacked-QR arbiter
    when the assembly is rooted and 4.99e-13 when the margins are -- and the
    statistic 7.38e-07 against 3.13e-11.  The same six orders separate the
    dense arm from the structured one, which has always rooted ``S_a``.

    ONE ROOTING POLICY, and it is
    :func:`superglm.screening._factor_kernels._penalty_root`'s: an eigenvalue
    inside ``eigh``'s own bar is taken at its MAGNITUDE, one outside it and
    negative is dropped.  Applied to each margin separately, which is what the
    structured path already does to ``S_a`` -- so the two paths root the same
    matrices by the same rule, and issue #323's "two policies deciding on
    different bars" cannot reappear between them.  The dense path does NOT
    inherit :func:`superglm.screening._structured._profile`'s refusal on a
    material drop: it has no numerical-certificate refusal contract, only a
    budget one, and adding one would delete rows the moment route published.
    A dropped direction is reported FREE, which is the direction
    :func:`_penalty_root` documents.

    A margin that carries no penalty passes a zero block of its own width, the
    same contract :func:`tensor_penalty` has -- a categorical contrast block is
    unpenalized and its zeros then contribute nothing to any eigenvalue sum.
    """
    a, Qa, _, _ = _penalty_spectrum(np.asarray(S1, dtype=np.float64))
    b, Qb, _, _ = _penalty_spectrum(np.asarray(S2, dtype=np.float64))
    values = (a[:, None] + b[None, :]).reshape(-1)
    keep = values > 0.0
    if not np.any(keep):
        return np.zeros((0, a.size * b.size), dtype=np.float64)
    vectors = np.kron(Qa, Qb)
    return (vectors[:, keep] * np.sqrt(values[keep])).T


def pair_overlap_moments(
    A: NDArray,
    B: NDArray,
    S_cell: NDArray,
    W_cell: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Return ``(M, C, u_m)`` for the overlap span ``[1 | A | B]``.

    ``A`` is ``(n_a, k_a)``, ``B`` is ``(n_b, k_b)``; the cell tables are
    ``(n_a, n_b)``.  Shapes out: ``M (q, q)``, ``C (q, k_a * k_b)``,
    ``u_m (q,)`` with ``q = 1 + k_a + k_b``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    S_cell = np.asarray(S_cell, dtype=np.float64)
    W_cell = np.asarray(W_cell, dtype=np.float64)
    k_a, k_b = A.shape[1], B.shape[1]
    q = 1 + k_a + k_b

    w_row = W_cell.sum(axis=1)
    w_col = W_cell.sum(axis=0)
    s_row = S_cell.sum(axis=1)
    s_col = S_cell.sum(axis=0)

    M = np.empty((q, q), dtype=np.float64)
    sl_a = slice(1, 1 + k_a)
    sl_b = slice(1 + k_a, q)
    M[0, 0] = W_cell.sum()
    M[0, sl_a] = w_row @ A
    M[0, sl_b] = w_col @ B
    M[sl_a, 0] = M[0, sl_a]
    M[sl_b, 0] = M[0, sl_b]
    M[sl_a, sl_a] = A.T @ (A * w_row[:, None])
    M[sl_b, sl_b] = B.T @ (B * w_col[:, None])
    M[sl_a, sl_b] = A.T @ W_cell @ B
    M[sl_b, sl_a] = M[sl_a, sl_b].T

    u_m = np.empty(q, dtype=np.float64)
    u_m[0] = S_cell.sum()
    u_m[sl_a] = A.T @ s_row
    u_m[sl_b] = B.T @ s_col

    C = np.empty((q, k_a * k_b), dtype=np.float64)
    C[0] = (A.T @ W_cell @ B).reshape(k_a * k_b)
    C[sl_a] = np.einsum("ij,ic,ip,jq->cpq", W_cell, A, A, B, optimize=True).reshape(k_a, k_a * k_b)
    C[sl_b] = np.einsum("ij,ip,jd,jq->dpq", W_cell, A, B, B, optimize=True).reshape(k_b, k_a * k_b)
    return M, C, u_m
