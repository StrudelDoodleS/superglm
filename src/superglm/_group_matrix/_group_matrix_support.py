"""Lossless row-support detection for factored SSP group matrices.

Exact-path spline and tensor bases repeat rows whenever the underlying covariate
is integer-valued or otherwise low-cardinality, which is the common case for
insurance rating variables.  Storing one copy per distinct row turns an O(n)
weighted gram into an O(n) bincount plus an O(n_support) dense gram.

This is deduplication, not binning: it introduces no discretisation error and is
unrelated to ``discrete=True``.

Two entry points, differing only in how the row grouping is obtained:

``plan_row_support``
    The production path.  The caller already knows which rows are identical --
    a single-covariate spline basis has identical rows exactly where the
    covariate value repeats -- so it supplies ``row_index`` and this module
    never touches the full basis.

``detect_row_support``
    Convenience for callers without a grouping.  Densifies the basis to derive
    one, which costs several times the dense basis in transient memory, so it
    is unsuitable for the hot path.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

# Require a real win before paying the bookkeeping and the (n_support, p_b)
# dense buffer that the compressed gram allocates each iteration.
DEFAULT_MIN_SPEEDUP = 1.5

# Ratio of realised speedup to the flop-count ratio below.  The flop count alone
# badly under-predicts, because the compressed side is a BLAS dense gram while
# the current side is a numba scalar loop over rows.  Measured at n=200_000,
# median of 5 after warm-up (see docs/audit/2026-07-28/):
#
#   p_b  nnz_row  support ratio  flop ratio  measured  implied factor
#     9        4          0.400       0.309     1.68x            5.4
#     9        4          0.950       0.130     0.74x            5.7
#    20        4          0.400       0.063     0.73x           11.7
#    81       81          0.400       1.266     9.96x            7.9
#    81       81          0.950       0.533     4.46x            8.4
#
# The low end of the observed range is used so the gate stays conservative.
_BLAS_ADVANTAGE = 6.0

# Cap on the dense unique-row buffer, matching the byte budgets used elsewhere
# in this package.
DEFAULT_MAX_SUPPORT_BYTES = 64 << 20

# Floor below which the speedup model is not applied: the calibration was
# measured on large blocks, and a gram over fewer rows than this is negligible
# whichever path it takes.
_MIN_CALIBRATED_ROWS = 1_000


def _estimated_speedup(n_rows: int, n_support: int, p_b: int, nnz: int) -> float:
    """Estimated gram speedup from deduplicating rows.

    The current path (``_csr_weighted_gram``) accumulates only nonzero pairs
    within each row, so it costs about ``n * r(r+1)/2`` for ``r`` nonzeros per
    row -- cheap for a locally-supported B-spline, expensive for a tensor whose
    row-Kronecker rows are dense.  The compressed path costs one bincount over
    ``n`` plus a dense ``(n_support, p_b)`` gram.

    The flop ratio is scaled by ``_BLAS_ADVANTAGE`` because the two sides run on
    very different hardware paths; the raw ratio under-predicts by 5-12x.
    """
    if n_rows <= 0 or n_support <= 0:
        return 0.0
    nnz_per_row = nnz / n_rows
    current = n_rows * nnz_per_row * (nnz_per_row + 1.0) / 2.0
    compressed = n_rows + n_support * float(p_b) ** 2
    if compressed <= 0.0:
        return 0.0
    return _BLAS_ADVANTAGE * current / compressed


def plan_row_support(
    B_csr: sp.spmatrix,
    row_index: NDArray,
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    max_support_bytes: int = DEFAULT_MAX_SUPPORT_BYTES,
) -> tuple[NDArray, NDArray] | None:
    """Return ``(B_unique, row_index)`` when compression pays, else ``None``.

    ``row_index`` maps each observation to its distinct-row group and must
    satisfy ``B_unique[row_index] == B``; callers derive it from the covariate
    that generated the basis, which is an O(n) scan of a one-dimensional array.
    Only the first occurrence of each group is materialised, so the full basis
    is never densified.
    """
    n_rows, p_b = B_csr.shape
    row_index = np.asarray(row_index, dtype=np.intp).ravel()
    if n_rows == 0 or row_index.shape[0] != n_rows:
        return None
    n_support = int(row_index.max()) + 1 if n_rows else 0
    # Strict inequality: equal counts mean no row actually repeats, so there is
    # nothing to deduplicate and the compressed form is pure overhead.
    if n_support <= 0 or n_support >= n_rows:
        return None
    # The speedup model is calibrated on large blocks; below this the gram is
    # negligible either way and the calibration does not apply.
    if n_rows < _MIN_CALIBRATED_ROWS:
        return None
    if n_support * p_b * 8 > max_support_bytes:
        return None
    if _estimated_speedup(n_rows, n_support, p_b, int(B_csr.nnz)) < min_speedup:
        return None

    # First occurrence of each group, taken without densifying the whole basis.
    first_occurrence = np.full(n_support, -1, dtype=np.intp)
    first_occurrence[row_index[::-1]] = np.arange(n_rows - 1, -1, -1, dtype=np.intp)
    if np.any(first_occurrence < 0):
        return None
    b_unique = np.asarray(B_csr[first_occurrence].todense(), dtype=np.float64)
    return b_unique, row_index


def detect_row_support(
    B_csr: sp.spmatrix,
    *,
    min_speedup: float = DEFAULT_MIN_SPEEDUP,
    max_support_bytes: int = DEFAULT_MAX_SUPPORT_BYTES,
) -> tuple[NDArray, NDArray] | None:
    """Derive the row grouping from the basis itself, then plan compression.

    Densifies ``B_csr`` and sorts it, so transient memory is several times the
    dense basis and the cost is paid even when compression is declined.  Prefer
    :func:`plan_row_support` wherever the caller knows the grouping.

    Rows containing NaN never compare equal, so a basis with NaNs simply fails
    to compress rather than compressing incorrectly.
    """
    n_rows = B_csr.shape[0]
    if n_rows == 0:
        return None
    dense = np.asarray(B_csr.toarray(), dtype=np.float64)
    _, row_index = np.unique(dense, axis=0, return_inverse=True)
    return plan_row_support(
        B_csr,
        row_index,
        min_speedup=min_speedup,
        max_support_bytes=max_support_bytes,
    )
