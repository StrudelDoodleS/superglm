"""Reference producers of the overlap Gram that the dense route used to assemble.

They are the definition the design factors in ``superglm.screening._pair_factor``
are graded against: the exactness tests compare the two, and the ceiling test
needs the Gram itself.  Nothing in ``src`` calls them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tensor_penalty(S1: NDArray, S2: NDArray) -> NDArray:
    """``kron(S1, I) + kron(I, S2)`` — interaction.py's ti() penalty order.

    The definition ``tensor_penalty_root`` is graded against, and the assembled form
    the guide quotes.
    """
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    return np.kron(S1, np.eye(S2.shape[0])) + np.kron(np.eye(S1.shape[0]), S2)


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
