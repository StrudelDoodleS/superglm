"""Lossless row-support detection for factored SSP group matrices.

Exact-path spline and tensor bases repeat rows whenever the underlying covariate
is integer-valued or otherwise low-cardinality, which is the common case for
insurance rating variables.  Storing one copy per distinct row turns an O(n)
weighted gram into an O(n) bincount plus an O(n_support) dense gram.

This is deduplication, not binning: it introduces no discretisation error and is
unrelated to ``discrete=True``.

Two entry points, differing only in how the row grouping is obtained:

``detect_row_support``
    The production path (``dm_builder._build_ssp_group``).  Derives the
    grouping from the basis itself, chunk by chunk, so only bit-identical
    rows merge; transients are bounded chunks plus one dense
    ``(n_support, p_b)`` representative block (which approaches the dense
    basis only when nothing repeats).  Needs no assumption that equal
    covariate values produce bit-identical basis rows.

``plan_row_support``
    Core gate + materialisation for callers that already know the grouping --
    a single-covariate spline basis repeats rows exactly where the covariate
    value repeats -- letting them skip the detection scan entirely.
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
    return _BLAS_ADVANTAGE * current / compressed


def _row_index_chunked(B_csr: sp.spmatrix, chunk_rows: int = 65_536) -> NDArray:
    """Exact row grouping with bounded transient memory.

    Densifying the whole basis and sorting it peaks at several times the dense
    size.  Working a chunk at a time keeps the transient at
    ``chunk_rows * p_b`` while remaining exact: rows are keyed by their raw
    bytes, so only bit-identical rows are merged.
    """
    n_rows = B_csr.shape[0]
    row_index = np.empty(n_rows, dtype=np.intp)
    seen: dict[bytes, int] = {}
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        block = np.ascontiguousarray(B_csr[start:stop].toarray(), dtype=np.float64)
        # Deduplicate within the chunk first, so the Python-level dictionary
        # lookups below run once per distinct row rather than once per row.
        block_unique, block_inverse = np.unique(block, axis=0, return_inverse=True)
        mapped = np.empty(block_unique.shape[0], dtype=np.intp)
        for position, row in enumerate(block_unique):
            key = row.tobytes()
            group = seen.get(key)
            if group is None:
                group = len(seen)
                seen[key] = group
            mapped[position] = group
        row_index[start:stop] = mapped[block_inverse.ravel()]
    return row_index


# Fixed odd multipliers for the row-hash mix, one per basis column.  A fixed
# seed keeps grouping deterministic across runs and machines; correctness never
# depends on hash quality because every grouping is verified bitwise below.
_ROW_HASH_SEED = 0x5EED_0001


def _row_hash_multipliers(p_b: int) -> NDArray:
    mults = np.random.default_rng(_ROW_HASH_SEED).integers(
        1, np.iinfo(np.uint64).max, size=max(p_b, 1), dtype=np.uint64
    )
    return mults | np.uint64(1)


def _row_hashes(B_csr: sp.spmatrix, chunk_rows: int) -> NDArray:
    """Mix each dense row's raw float bits column-wise into one uint64."""
    n_rows, p_b = B_csr.shape
    mults = _row_hash_multipliers(p_b)
    hashes = np.empty(n_rows, dtype=np.uint64)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        block = np.ascontiguousarray(B_csr[start:stop].toarray(), dtype=np.float64)
        # Per-column multipliers make the mix position-dependent; uint64
        # arithmetic wraps, which is what a mix wants.
        hashes[start:stop] = (block.view(np.uint64) * mults).sum(axis=1, dtype=np.uint64)
    return hashes


def _verified_representatives(
    B_csr: sp.spmatrix,
    first_occurrence: NDArray,
    row_index: NDArray,
    chunk_rows: int,
) -> NDArray | None:
    """Densify the representative rows and verify the grouping bitwise.

    Returns the float64 representative block on success, or None on a true
    64-bit hash collision (caller falls back to byte keying).  Comparing raw
    bits keeps NaN rows mergeable only when bit-identical and keeps ``-0.0``
    distinct from ``0.0``, matching the byte-keyed semantics.
    """
    n_rows = B_csr.shape[0]
    representatives = np.ascontiguousarray(
        B_csr[np.asarray(first_occurrence, dtype=np.intp)].toarray(), dtype=np.float64
    )
    rep_bits = representatives.view(np.uint64)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        block = np.ascontiguousarray(B_csr[start:stop].toarray(), dtype=np.float64)
        if not np.array_equal(rep_bits[row_index[start:stop]], block.view(np.uint64)):
            return None
    return representatives


def _row_index_hashed(B_csr: sp.spmatrix, chunk_rows: int = 65_536) -> NDArray:
    """Exact row grouping via vectorized 64-bit mixing, verified bitwise.

    Grouping needs a single 8-byte sort instead of a lexicographic sort of
    ``p_b``-wide records — the cost that dominates :func:`_row_index_chunked`.
    A verification failure (a true 64-bit collision) falls back to the
    byte-keyed path, so exactness never rests on the hash.
    """
    hashes = _row_hashes(B_csr, chunk_rows)
    _, first_occurrence, row_index = np.unique(hashes, return_index=True, return_inverse=True)
    row_index = np.asarray(row_index, dtype=np.intp).ravel()
    if _verified_representatives(B_csr, first_occurrence, row_index, chunk_rows) is None:
        return _row_index_chunked(B_csr, chunk_rows=chunk_rows)
    return row_index


def _passes_support_gates(
    n_rows: int,
    n_support: int,
    p_b: int,
    nnz: int,
    min_speedup: float,
    max_support_bytes: int,
) -> bool:
    """Cheap accept/decline gates, evaluated before anything is densified."""
    # Strict inequality: equal counts mean no row actually repeats, so there is
    # nothing to deduplicate and the compressed form is pure overhead.
    if n_support <= 0 or n_support >= n_rows:
        return False
    # The speedup model is calibrated on large blocks; below this the gram is
    # negligible either way and the calibration does not apply.
    if n_rows < _MIN_CALIBRATED_ROWS:
        return False
    if n_support * p_b * 8 > max_support_bytes:
        return False
    return _estimated_speedup(n_rows, n_support, p_b, nnz) >= min_speedup


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
    n_support = int(row_index.max()) + 1
    if not _passes_support_gates(
        n_rows, n_support, p_b, int(B_csr.nnz), min_speedup, max_support_bytes
    ):
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

    Rows are grouped by a vectorized 64-bit mix of their raw float bits and the
    grouping is verified bitwise, falling back to byte keying on a collision.
    The scan is O(n * p_b) in bounded chunks; the accept/decline gates run on
    the hash grouping alone, so a declined block (nothing repeats, support
    over budget, speedup below threshold) never materialises the dense
    ``(n_support, p_b)`` representative block — the worst case for that
    allocation is exactly the no-repeats case where compression is refused.
    On accept the verified representative block is returned directly, so the
    support rows are densified exactly once.  Callers that already know the
    grouping can skip the scan via :func:`plan_row_support`.

    Only bit-identical densified rows merge, so reconstruction is exact even
    for non-finite values: NaN rows merge only when their bit patterns match.
    (Explicitly stored ``-0.0`` entries densify to ``+0.0`` before grouping,
    identically for grouping and reconstruction, so exactness is unaffected.)
    """
    n_rows, p_b = B_csr.shape
    if n_rows == 0:
        return None
    chunk_rows = 65_536
    hashes = _row_hashes(B_csr, chunk_rows)
    _, first_occurrence, row_index = np.unique(hashes, return_index=True, return_inverse=True)
    row_index = np.asarray(row_index, dtype=np.intp).ravel()
    if not _passes_support_gates(
        n_rows, len(first_occurrence), p_b, int(B_csr.nnz), min_speedup, max_support_bytes
    ):
        return None
    representatives = _verified_representatives(B_csr, first_occurrence, row_index, chunk_rows)
    if representatives is None:
        # A true 64-bit collision: regroup byte-keyed and re-plan on the
        # correct grouping (the hash-based gates may have undercounted).
        return plan_row_support(
            B_csr,
            _row_index_chunked(B_csr, chunk_rows=chunk_rows),
            min_speedup=min_speedup,
            max_support_bytes=max_support_bytes,
        )
    return representatives, row_index
