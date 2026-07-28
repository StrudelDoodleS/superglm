"""Lossless row-support detection for factored SSP group matrices.

Exact-path spline and tensor bases repeat rows whenever the underlying covariate
is integer-valued or otherwise low-cardinality, which is the common case for
insurance rating variables.  Storing one copy per distinct row turns an O(n)
weighted gram into an O(n) bincount plus an O(n_support) dense gram.

This is deduplication, not binning: it introduces no discretisation error and is
unrelated to ``discrete=True``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

# Above this distinct-row fraction the bookkeeping costs more than the saved
# arithmetic, so detection declines and the caller keeps its dense-row path.
DEFAULT_MAX_SUPPORT_RATIO = 0.5


def detect_row_support(
    B_csr: sp.spmatrix, max_ratio: float = DEFAULT_MAX_SUPPORT_RATIO
) -> tuple[NDArray, NDArray] | None:
    """Return ``(B_unique, row_index)`` when row compression pays, else ``None``.

    ``B_unique[row_index]`` reproduces the input basis exactly.
    """
    dense = np.asarray(B_csr.toarray(), dtype=np.float64)
    n_rows = dense.shape[0]
    if n_rows == 0:
        return None
    b_unique, row_index = np.unique(dense, axis=0, return_inverse=True)
    if b_unique.shape[0] > max_ratio * n_rows:
        return None
    return b_unique, np.asarray(row_index, dtype=np.intp).ravel()
