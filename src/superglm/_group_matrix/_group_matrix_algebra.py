"""Private algebra helpers for group-matrix block operations."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _cat_cat_weighted_crosstab,
    _cat_weighted_bincount,
    _cell_hist_raw_kron,
    _csr_row_chunk,
    _csr_weighted_bincount,
    _disc_disc_2d_hist,
    _disc_disc_2d_hist_channels,
    _fused_2d_bincount_2,
    _gather_cell_order,
    _operand_exponent_bounds,
    _tensor_operand_in_reassociation_range,
    _weighted_bincount_2d,
)

if TYPE_CHECKING:
    from ..group_matrix import (
        DenseGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        FactorSmoothGroupMatrix,
        GroupMatrix,
        RandomEffectGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )
else:
    GroupMatrix = Any

_MAX_DISC_DISC_HIST_CELLS = 5_000_000
_MAX_DISC_DISC_CHANNEL_HIST_CELLS = 5_000_000

# Transient ceiling for the row-expanded cross-gram fallback, matching the byte
# budgets used elsewhere in this package.  The histogram cap above bounds CELLS;
# this bounds the ROWS the fallback expands, which the cell cap cannot.
_MAX_CROSS_EXPANSION_BYTES = 64 << 20

# Ceiling on a CROSS-SHAPED aggregate: an array whose row count comes from one
# block and whose column count comes from the other.  Every gate in this
# subsystem bounds one block against its OWN width -- the support gate bounds
# n_support * p_b, the histogram caps bound n_bins_i * n_bins_j -- and none of
# them bounds a row count from one block against a width from the other.  That
# product is what the aggregate allocates, so it needs its own ceiling.
_MAX_AGGREGATE_CELLS = _MAX_CROSS_EXPANSION_BYTES // 8


def _aggregate_column_chunk(n_bins: int, n_cols: int) -> int:
    """Output columns per pass of a cross-shaped aggregate, sized in CELLS."""
    return max(1, min(int(n_cols), int(_MAX_AGGREGATE_CELLS // max(int(n_bins), 1))))


def _profile_add(profile: dict[str, Any] | None, key: str, value: float) -> None:
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + value


def _profile_count(profile: dict[str, Any] | None, key: str, value: int = 1) -> None:
    if profile is not None:
        profile[key] = int(profile.get(key, 0)) + value


def _profile_elapsed(profile: dict[str, Any] | None, key: str, start: float) -> None:
    if profile is not None:
        _profile_add(profile, key, perf_counter() - start)


def _cross_factors_in_range(*operands: NDArray, cache=None) -> bool:
    """Bound every partial cross product and reduction, not just its result.

    Sum negative/positive exponent bounds separately to bound every partial
    product, not just the final product. The product of operand dimensions
    overbounds reduction lengths in either association. Non-cancelling terms
    then stay normal; cancellation still follows ordinary rounded arithmetic.

    Look up weight_range optionally: centered assembly uses a weight-grid-only
    cache without that method.
    """
    weight_range = getattr(cache, "weight_range", None)
    lower = upper = 0
    for operand in operands:
        if operand.dtype != np.float64:
            return False
        values = operand if operand.ndim == 2 else operand[:, None]
        lo, hi = (
            weight_range(operand)[1]
            if weight_range is not None and operand.ndim == 1
            else _operand_exponent_bounds(values)
        )
        lower += min(0, lo)
        upper += max(0, hi) + sum((max(1, size) - 1).bit_length() for size in operand.shape)
    return lower >= -1022 and upper <= 1022


def _cross_support(
    gm: DiscretizedSSPGroupMatrix, cache, *partners: NDArray
) -> tuple[NDArray, NDArray | None]:
    """Project support when the remaining cross factors retain range.

    Mixed routes pass their already-weighted aggregate. Tensor/histogram
    routes pass W and all pending factors; declined inputs keep raw association.
    """
    if not _cross_factors_in_range(gm.B_unique, gm.R_inv, *partners, cache=cache):
        return gm.B_unique, gm.R_inv
    # Centered tensor assembly supplies a weight-grid-only cache.
    project = getattr(cache, "solver_support", None)
    return (gm.B_unique @ gm.R_inv if project is None else project(gm)), None


class _BlockWeightCache:
    """Per-block-assembly cache for weighted discrete summaries."""

    __slots__ = (
        "_hist2d",
        "_profile",
        "_channel_scratch",
        "_cell_weights",
        "_supports",
        "_sparse_grams",
        "_weight_ranges",
        "_cell_orders",
    )

    def __init__(self, profile: dict[str, Any] | None = None) -> None:
        self._hist2d: dict[tuple[int, int, int, int, int], NDArray] = {}
        self._profile = profile
        self._channel_scratch = np.empty(0)
        self._cell_weights: dict[tuple[int, int], NDArray] = {}
        self._supports: dict[DiscretizedSSPGroupMatrix, NDArray] = {}
        self._sparse_grams: dict[SparseSSPGroupMatrix, tuple[NDArray, bool]] = {}
        self._weight_ranges: dict[int, tuple[NDArray, bool, tuple[int, int]]] = {}
        self._cell_orders: dict[DiscretizedTensorGroupMatrix, tuple[NDArray, NDArray]] = {}

    def weight_range(self, W: NDArray) -> tuple[bool, tuple[int, int]]:
        """Reuse both original range decisions for this assembly's row weights.

        Retain the array, so an identity key cannot outlive its owner. The
        legacy boolean is separate: exponent bounds round powers of two up.
        """
        entry = self._weight_ranges.get(id(W))
        if entry is None:
            values = W[:, None]
            entry = (
                W,
                _tensor_operand_in_reassociation_range(values),
                _operand_exponent_bounds(values),
            )
            self._weight_ranges[id(W)] = entry
        return entry[1], entry[2]

    def cell_csr(self, gm: DiscretizedTensorGroupMatrix) -> tuple[NDArray, NDArray]:
        """Validate each grid once within an assembly with unchanged inputs.

        Keys own the grids. A fresh assembly and every uncached call validate
        live indices again; no cross-fit validation result is retained.
        """
        result = self._cell_orders.get(gm)
        if result is None:
            result = gm.cell_csr()
            self._cell_orders[gm] = result
        return result

    def release_channel_buffers(self) -> None:
        """Drop derived channel workspace before another route spends its budget."""
        self._channel_scratch = np.empty(0)
        self._cell_weights.clear()

    def sparse_gram(self, gm: SparseSSPGroupMatrix, weights: NDArray) -> tuple[NDArray, bool]:
        """Share a sparse diagonal's cancellation decision with its crosses.

        This cache belongs to one weighted assembly. A cross may request the
        right diagonal early; the diagonal loop then reuses that same result.
        """
        result = self._sparse_grams.get(gm)
        if result is None:
            result = gm._gram_with_projection(weights)
            self._sparse_grams[gm] = result
        return result

    def solver_support(self, gm: DiscretizedSSPGroupMatrix) -> NDArray:
        """Project each live support once within this synchronous assembly.

        Keys retain the owning blocks. A new moments call creates a new cache,
        so changed factors, weights, subsets and reparameterizations cannot
        reuse a previous assembly's projection.
        """
        support = self._supports.get(gm)
        if support is None:
            support = gm.B_unique @ gm.R_inv
            self._supports[gm] = support
            _profile_count(self._profile, "block_solver_support_builds")
        else:
            _profile_count(self._profile, "block_solver_support_reuses")
        return support

    def cell_weights(self, order: NDArray, W: NDArray) -> NDArray:
        """``W`` in a grid tensor's cell order, permuted once per grid per build.

        Every partner of a grid tensor reads the same permutation -- ``order``
        is the grid's cached cell-CSR and ``W`` the build's weights -- so the
        entry is keyed on their identity, as ``disc_disc_hist``'s is.
        """
        key = (id(order), id(W))
        permuted = self._cell_weights.get(key)
        if permuted is not None:
            _profile_count(self._profile, "block_cell_weight_reuses")
            return permuted
        permuted = W[order]
        self._cell_weights[key] = permuted
        return permuted

    def channel_accumulator(self, n_cells: int, width: int) -> NDArray:
        """One ``(n_cells, width)`` scratch per Gram build, grown to the largest request.

        The raw-band kernel writes every row, so the scratch is never zeroed
        and its pages are faulted in once per build rather than once per
        block (a fresh accumulator's first touch measured 15.6% of the dense
        kernel's time).  Blocks are assembled one after another, so one
        buffer serves them all.
        """
        cells = n_cells * width
        if self._channel_scratch.size < cells:
            # No consumer keeps the borrowed scratch between cross blocks.
            # Release it before growth so old and new capacities do not overlap.
            self._channel_scratch = np.empty(0)
            self._channel_scratch = np.empty(cells)
        return self._channel_scratch[:cells].reshape(n_cells, width)

    @staticmethod
    def _key(idx_a: NDArray, idx_b: NDArray, W: NDArray, n_a: int, n_b: int):
        return (id(idx_a), id(idx_b), id(W), int(n_a), int(n_b))

    def disc_disc_hist(
        self,
        idx_a: NDArray,
        idx_b: NDArray,
        W: NDArray,
        n_a: int,
        n_b: int,
    ) -> NDArray:
        key = self._key(idx_a, idx_b, W, n_a, n_b)
        cached = self._hist2d.get(key)
        if cached is not None:
            _profile_count(self._profile, "block_hist2d_reuses")
            return cached

        rev_key = self._key(idx_b, idx_a, W, n_b, n_a)
        rev_cached = self._hist2d.get(rev_key)
        if rev_cached is not None:
            hist = rev_cached.T
            self._hist2d[key] = hist
            _profile_count(self._profile, "block_hist2d_reuses")
            return hist

        t0 = perf_counter() if self._profile is not None else 0.0
        hist = _disc_disc_2d_hist(idx_a, idx_b, W, n_a, n_b)
        _profile_elapsed(self._profile, "block_hist2d_s", t0)
        _profile_count(self._profile, "block_hist2d_builds")
        self._hist2d[key] = hist
        return hist

    def tensor_w_grid(self, gm: DiscretizedTensorGroupMatrix, W: NDArray) -> NDArray:
        return self.disc_disc_hist(gm.idx1, gm.idx2, W, gm.n_bins1, gm.n_bins2)

    def tensor_w_wz_grid(
        self, gm: DiscretizedTensorGroupMatrix, W: NDArray, Wz: NDArray
    ) -> tuple[NDArray, NDArray]:
        w_key = self._key(gm.idx1, gm.idx2, W, gm.n_bins1, gm.n_bins2)
        wz_key = self._key(gm.idx1, gm.idx2, Wz, gm.n_bins1, gm.n_bins2)
        w_grid = self._hist2d.get(w_key)
        wz_grid = self._hist2d.get(wz_key)
        if w_grid is None and wz_grid is None:
            t0 = perf_counter() if self._profile is not None else 0.0
            w_grid, wz_grid = _fused_2d_bincount_2(gm.idx1, gm.idx2, W, Wz, gm.n_bins1, gm.n_bins2)
            _profile_elapsed(self._profile, "block_hist2d_s", t0)
            _profile_count(self._profile, "block_hist2d_builds")
            self._hist2d[w_key] = w_grid
            self._hist2d[wz_key] = wz_grid
            return w_grid, wz_grid

        if w_grid is None:
            w_grid = self.tensor_w_grid(gm, W)
        if wz_grid is None:
            wz_grid = self.disc_disc_hist(gm.idx1, gm.idx2, Wz, gm.n_bins1, gm.n_bins2)
        return w_grid, wz_grid


def _runtime_group_matrix_types():
    """Import group-matrix runtime classes lazily to avoid circular imports."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    return (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )


def _agg_by_bin_fits(gm: GroupMatrix, n_bins: int) -> bool:
    """Whether ``_agg_by_bin``'s output is small enough to materialise.

    Its result is ``(n_bins, gm.shape[1])`` -- the row count from the block
    supplying the bins, the width from the block being aggregated.  Those are
    DIFFERENT blocks at every caller, so this is the cross shape, and no gate in
    the subsystem bounds the product.  An earlier version of the invariant test
    exempted these calls on the stated ground that both dimensions came from the
    same block; that was simply wrong, and a narrow million-row support beside a
    wide sparse term is the counterexample.
    """
    return int(n_bins) * _agg_by_bin_width(gm) <= _MAX_AGGREGATE_CELLS


def _agg_by_bin_width(gm: GroupMatrix) -> int:
    """The width ``_agg_by_bin`` actually ALLOCATES at, not the width it returns.

    The SSP branches aggregate in basis space and only then apply ``R_inv``, so
    the intermediate is ``_p_b`` wide while ``shape[1]`` is the post-transform
    width -- 600 against 4 on the pairing that exposed this.  Budgeting against
    the returned width silently permits the allocation it is meant to stop.
    """
    width = getattr(gm, "_p_b", None)
    if width is not None:
        return int(width)
    matrix = getattr(gm, "M", None)
    if matrix is not None:
        return int(matrix.shape[1])
    unique = getattr(gm, "B_unique", None)
    if unique is not None:
        return max(int(unique.shape[1]), int(gm.shape[1]))
    return int(gm.shape[1])


def _agg_by_bin(
    gm: GroupMatrix,
    bin_idx: NDArray,
    W: NDArray,
    n_bins: int,
    cache: _BlockWeightCache | None = None,
) -> NDArray:
    """Aggregate W * gm's columns by bin index → (n_bins, p_g) dense array.

    Dispatches to the most efficient kernel for each GroupMatrix type:
    - SparseGroupMatrix: CSR-aware kernel (avoids toarray, O(nnz) not O(n*p))
    - SparseSSPGroupMatrix: CSR kernel in B-spline space + R_inv transform
    - DenseGroupMatrix / other: fused dense kernel (avoids W-broadcast alloc)
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import FactorSmoothGroupMatrix

    if isinstance(gm, CategoricalGroupMatrix):
        return _cat_weighted_bincount(gm.codes, bin_idx, W, n_bins, gm.n_levels)
    if isinstance(gm, SparseGroupMatrix):
        return _csr_weighted_bincount(
            np.asarray(gm.M.data, dtype=np.float64),
            gm.M.indices,
            gm.M.indptr,
            gm.M.shape[1],
            bin_idx,
            W,
            n_bins,
        )
    if isinstance(gm, SparseSSPGroupMatrix):
        if type(gm) is SparseSSPGroupMatrix:
            local_cache = _BlockWeightCache() if cache is None else cache
            if local_cache.sparse_gram(gm, W)[1]:
                return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
        B_agg = _csr_weighted_bincount(
            gm._data, gm._indices, gm._indptr, gm._p_b, bin_idx, W, n_bins
        )
        return B_agg @ gm.R_inv
    if isinstance(gm, DiscretizedSplineCategoricalGroupMatrix):
        rows = gm.row_idx
        # Chunked, and it is the only branch here that needed saying so: the
        # SCOP, SSP and factor-smooth branches all decline into
        # ``_aggregate_group_matrix_columns``, which is column-at-a-time and
        # bounded, while this one expanded the level per observation row.
        # Reached when ``_cross_gram_discrete_spline_categorical`` declines at
        # its cell cap and dispatch falls through to the disc-x-non-disc branch,
        # which is newly reachable because a lossless support makes ``n_bins``
        # large on both sides.  ``B_unique @ R_inv`` is ``(n_support, p_g)``,
        # bounded by the support gate; the gather off it was not.
        return _chunked_support_bincount_2d(
            bin_idx[rows], W[rows], gm.B_unique @ gm.R_inv, gm.bin_idx_level, n_bins
        )
    if isinstance(gm, SplineCategoricalGroupMatrix):
        rows = gm.row_idx
        B_agg = _csr_weighted_bincount(
            gm._data,
            gm._indices,
            gm._indptr,
            gm._p_b,
            bin_idx[rows],
            W[rows],
            n_bins,
        )
        return B_agg @ gm.R_inv
    if isinstance(gm, DiscretizedSCOPGroupMatrix):
        n_cells = n_bins * gm.n_bins
        if n_cells <= _MAX_DISC_DISC_HIST_CELLS:
            weight_grid = _disc_disc_2d_hist(
                bin_idx,
                gm.bin_idx,
                W,
                n_bins,
                gm.n_bins,
            )
            return weight_grid @ gm.B_scop_unique
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    if isinstance(gm, DiscretizedSSPGroupMatrix):
        n_cells = n_bins * gm.n_bins
        if n_cells <= _MAX_DISC_DISC_HIST_CELLS:
            weight_grid = _disc_disc_2d_hist(
                bin_idx,
                gm.bin_idx,
                W,
                n_bins,
                gm.n_bins,
            )
            return (weight_grid @ gm.B_unique) @ gm.R_inv
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    if isinstance(gm, FactorSmoothGroupMatrix):
        return _aggregate_group_matrix_columns(gm, bin_idx, W, n_bins)
    X = gm.toarray()
    return _weighted_bincount_2d(bin_idx, W, X, n_bins)


def _aggregate_group_matrix_columns(
    gm: GroupMatrix,
    bin_idx: NDArray,
    W: NDArray,
    n_bins: int,
) -> NDArray:
    """Aggregate a compact matrix by bins without an observation-by-column temporary."""
    result = np.empty((n_bins, gm.shape[1]), dtype=np.float64)
    unit = np.zeros(gm.shape[1], dtype=np.float64)
    for column in range(gm.shape[1]):
        unit[column] = 1.0
        values = gm.matvec(unit)
        result[:, column] = np.bincount(
            bin_idx,
            weights=W * values,
            minlength=n_bins,
        )
        unit[column] = 0.0
    return result


def _random_effect_cross_gram(
    random_effect: RandomEffectGroupMatrix,
    other: GroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray:
    """Return ``X_re.T @ diag(W) @ X_other`` by direct level aggregation."""
    # Guarded like every other _agg_by_bin caller.  The output is
    # (n_levels, width-of-other) -- cross-shaped, and a high-cardinality random
    # effect beside a wide raw-basis SSP term is the case that reaches it.  An
    # earlier audit only inspected functions named _cross_gram, so this call sat
    # outside its scope entirely.
    if not _agg_by_bin_fits(other, random_effect.n_levels):
        return _cross_gram_by_columns(random_effect, other, W)
    return _agg_by_bin(
        other,
        random_effect.codes,
        W,
        random_effect.n_levels,
        cache,
    )


def _cross_gram_tensor_tensor(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray:
    """Cross-gram between two tensor groups sharing the same marginals.

    Used for decomposed tensor subgroups (bilinear × wiggly) that share
    the same B1_unique, B2_unique, idx1, idx2 but have different R_inv.
    """
    if cache is None:
        w_grid = _disc_disc_2d_hist(gm_i.idx1, gm_i.idx2, W, gm_i.n_bins1, gm_i.n_bins2)
    else:
        cache.release_channel_buffers()
        w_grid = cache.tensor_w_grid(gm_i, W)
    G_raw = gm_i._factored_gram_raw(w_grid)
    return gm_i.R_inv.T @ G_raw @ gm_j.R_inv


def _tensor_margin_parts(
    gm: DiscretizedTensorGroupMatrix,
    margin: int,
) -> tuple[NDArray, NDArray, int, NDArray, NDArray, int, bool]:
    if margin == 1:
        return (
            gm.B1_unique_t,
            gm.idx1,
            gm.n_bins1,
            gm.B2_unique_t,
            gm.idx2,
            gm.n_bins2,
            True,
        )
    return (
        gm.B2_unique_t,
        gm.idx2,
        gm.n_bins2,
        gm.B1_unique_t,
        gm.idx1,
        gm.n_bins1,
        False,
    )


def _margin_bins(gm: DiscretizedTensorGroupMatrix, margin: int) -> tuple[int, int]:
    """Bin counts of ``margin`` and of the other margin."""
    return (gm.n_bins1, gm.n_bins2) if margin == 1 else (gm.n_bins2, gm.n_bins1)


def _shared_margin_fits(
    gm_i: DiscretizedTensorGroupMatrix,
    margin_i: int,
    gm_j: DiscretizedTensorGroupMatrix,
    margin_j: int,
) -> bool:
    """O(1) admission of one pairing: equal bin counts and the three-way cell cap.

    Checked BEFORE the O(n) index comparison of ``_same_discrete_margin``: at
    the default 256 bins every pairing is over the cap, so comparing first
    cost four full row passes per tensor pair per Gram build for a route the
    pair could never take (0.375 s of a ten-pair fit).
    """
    n_shared, n_other_i = _margin_bins(gm_i, margin_i)
    n_shared_j, n_other_j = _margin_bins(gm_j, margin_j)
    return n_shared == n_shared_j and n_shared * n_other_i * n_other_j <= _MAX_DISC_DISC_HIST_CELLS


def _same_discrete_margin(
    gm_i: DiscretizedTensorGroupMatrix,
    margin_i: int,
    gm_j: DiscretizedTensorGroupMatrix,
    margin_j: int,
) -> bool:
    """The O(n) row-index comparison, for a pairing ``_shared_margin_fits`` admitted."""
    _B_i, idx_i, *_ = _tensor_margin_parts(gm_i, margin_i)
    _B_j, idx_j, *_ = _tensor_margin_parts(gm_j, margin_j)
    return np.array_equal(idx_i, idx_j)


def _cross_gram_tensor_tensor_shared_margin(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray | None:
    """Cross-Gram for two tensor terms sharing one marginal index under the cell cap.

    A pair sharing both margins takes this route when exactly one pairing is
    under the cap and declines (to the channel route) when both are.
    """
    matches = [
        (margin_i, margin_j)
        for margin_i in (1, 2)
        for margin_j in (1, 2)
        if _shared_margin_fits(gm_i, margin_i, gm_j, margin_j)
        and _same_discrete_margin(gm_i, margin_i, gm_j, margin_j)
    ]
    if len(matches) != 1:
        return None
    if cache is not None:
        cache.release_channel_buffers()

    margin_i, margin_j = matches[0]
    (
        B_shared_i,
        idx_shared,
        n_shared,
        B_other_i,
        idx_other_i,
        n_other_i,
        i_shared_first,
    ) = _tensor_margin_parts(gm_i, margin_i)
    (
        B_shared_j,
        _idx_shared_j,
        _n_shared_j,
        B_other_j,
        idx_other_j,
        n_other_j,
        j_shared_first,
    ) = _tensor_margin_parts(gm_j, margin_j)

    n_cells = n_shared * n_other_i * n_other_j
    flat = (idx_shared * n_other_i + idx_other_i) * n_other_j + idx_other_j
    joint = np.bincount(flat, weights=W, minlength=n_cells).reshape(
        n_shared,
        n_other_i,
        n_other_j,
    )

    K_shared_i = B_shared_i.shape[1]
    K_other_i = B_other_i.shape[1]
    K_shared_j = B_shared_j.shape[1]
    K_other_j = B_other_j.shape[1]
    raw4 = np.zeros(
        (K_shared_i, K_other_i, K_shared_j, K_other_j),
        dtype=np.float64,
    )
    for idx in range(n_shared):
        other_cross = B_other_i.T @ joint[idx] @ B_other_j
        raw4 += np.einsum(
            "p,q,rs->prqs",
            B_shared_i[idx],
            B_shared_j[idx],
            other_cross,
            optimize=True,
        )

    axes = (
        (0, 1) if i_shared_first else (1, 0),
        (2, 3) if j_shared_first else (3, 2),
    )
    raw = raw4.transpose(*axes[0], *axes[1]).reshape(
        gm_i.R_inv.shape[0],
        gm_j.R_inv.shape[0],
    )
    return gm_i.R_inv.T @ raw @ gm_j.R_inv


def _tensor_channel_workspace_bytes(
    grid: DiscretizedTensorGroupMatrix,
    chan: DiscretizedTensorGroupMatrix,
    width: int,
    cache: _BlockWeightCache | None,
    *,
    raw: bool,
    retain_buffers: bool = True,
) -> int:
    """Bound active channel workspace, including retained reusable arrays.

    This is not a whole-model/RSS limit: other groups' cell indexes and input
    factors remain model storage. Include the active grid's index even when
    retained, and every permuted weight vector retained by this assembly.
    """
    n, cells = len(grid.idx1), grid.n_bins1 * grid.n_bins2
    index_bytes = np.dtype(np.intp).itemsize
    retained_index = 0 if grid._cell_csr is None else sum(a.nbytes for a in grid._cell_csr)
    if raw:
        # Invalidated indexes are released before rebuilding. Reserve the
        # counting-sort fill even on warm calls to cover live-index mutation.
        retained_index = max(retained_index, 8 * (cells + 1) + index_bytes * n)
    scratch = 0 if cache is None or not retain_buffers else cache._channel_scratch.nbytes
    weights = (
        0
        if cache is None or not retain_buffers
        else sum(a.nbytes for a in cache._cell_weights.values())
    )
    histogram = 8 * cells * width
    base = retained_index + weights + (max(scratch, histogram) if raw else scratch + histogram)
    # Reserve a new W permutation even if an old cell order has cached one:
    # live-index invalidation can require a fresh permutation in this call.
    if raw and cache is not None:
        base += 8 * n  # A new permutation stays cached through stage two.
    # The 8 * cells term reserves the counting-sort fill array during a rebuild.
    stage1 = (
        max(8 * cells, (2 * index_bytes + (8 if cache is None else 0)) * n + 8 * width)
        if raw
        else 0
    )
    k1, k2 = grid.B1_unique_t.shape[1], grid.B2_unique_t.shape[1]
    p_grid, p_chan = grid.shape[1], chan.shape[1]
    stage2 = 8 * (
        k1 * grid.n_bins2 * width  # tmp
        + k1 * k2 * width  # raw
        + k2 * width  # one contraction result while assigning into raw
        + p_grid * width
        + p_grid * p_chan  # the two final products
        + (width * p_chan if raw else 0)  # projected channel map
    )
    # Marginal snapshots are model storage. Their array_equal comparison uses
    # one byte per entry before allocating H, alongside any retained buffers.
    validation = (
        retained_index + weights + scratch + max(chan.B1_unique_t.size, chan.B2_unique_t.size)
        if raw
        else 0
    )
    return max(validation, base + max(stage1, stage2))


def _tensor_channel_histogram(
    grid: DiscretizedTensorGroupMatrix,
    chan: DiscretizedTensorGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None,
    profile: dict[str, Any] | None,
) -> tuple[NDArray, NDArray] | None:
    """Stage 1 of the channel route: ``H`` over the cells of ``grid``, and the
    map carrying its channels onto ``chan``'s solver columns.

    Raw bands accumulate in cell order and project after contraction. The
    dense stage gathers stored joint rows. Both must fit the per-operation
    byte budget, including active retained storage and subsequent contraction
    workspace. Clear derived build-cache buffers under memory pressure;
    decline to the existing row route only if neither stage fits on its own.
    """
    n1, n2 = grid.n_bins1, grid.n_bins2
    band = chan.raw_channels
    for raw in (True, False):
        if raw:
            if band is None or not _cross_factors_in_range(
                W,
                grid.B1_unique_t,
                grid.B2_unique_t,
                grid.R_inv,
                band.values1,
                band.values2,
                band.projection,
                chan.R_inv,
                cache=cache,
            ):
                continue
            width = band.projection.shape[0]
        else:
            width = chan.B_unique.shape[1]
        if n1 * n2 * width > _MAX_AGGREGATE_CELLS:
            continue
        workspace = _tensor_channel_workspace_bytes(grid, chan, width, cache, raw=raw)
        if cache is not None and workspace > _MAX_CROSS_EXPANSION_BYTES:
            workspace = _tensor_channel_workspace_bytes(
                grid, chan, width, cache, raw=raw, retain_buffers=False
            )
            if workspace <= _MAX_CROSS_EXPANSION_BYTES:
                cache.release_channel_buffers()
        if workspace > _MAX_CROSS_EXPANSION_BYTES:
            continue
        if raw and chan._current_raw_channels() is None:
            continue
        if not raw:
            H = _disc_disc_2d_hist_channels(
                grid.idx1, grid.idx2, chan.bin_idx, W, chan.B_unique, n1, n2
            )
            return H, chan.R_inv
        assert band is not None
        ptr, order = grid.cell_csr() if cache is None else cache.cell_csr(grid)
        H = (
            np.empty((n1 * n2, width))
            if cache is None
            else cache.channel_accumulator(n1 * n2, width)
        )
        bin1, bin2 = _gather_cell_order(order, chan.idx1, chan.idx2)
        w = W[order] if cache is None else cache.cell_weights(order, W)
        _cell_hist_raw_kron(
            ptr,
            bin1,
            bin2,
            w,
            band.offsets1,
            band.values1,
            band.offsets2,
            band.values2,
            band.k2_raw,
            H,
        )
        del bin1, bin2, w
        _profile_count(profile, "block_cross_tensor_tensor_channel_raw")
        return H, band.projection @ chan.R_inv
    return None


def _cross_gram_tensor_tensor_channels(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
    profile: dict[str, Any] | None = None,
) -> NDArray | None:
    """Cross-Gram of two tensor terms with distinct ids, staged on one term's grid.

    ``X_i.T @ diag(W) @ X_j`` where row ``r`` of tensor ``i`` is the Kronecker
    row ``B1_i[idx1_r] (x) B2_i[idx2_r]`` (column order ``a * K2 + b``, as
    ``_row_kron_dense`` stores it) and row ``r`` of tensor ``j`` is its stored
    joint row ``B_joint_j[bin_idx_r]``.  The raw product factors through the
    cell index of tensor ``i``::

        H[i1 * n2 + i2, cd] = sum_{r in cell (i1, i2)} W_r * B_joint_j[bin_idx_r, cd]
        raw[a * K2 + b, cd] = sum_{i1, i2} B1_i[i1, a] * B2_i[i2, b] * H[i1 * n2 + i2, cd]

    Accumulate channels over the grid, contract with its two margins, then
    apply the coefficient maps. Choose the smaller stored-width histogram;
    ties use the left grid, preserving the established summation orientation.

    Both stages require float64 operands and a histogram below the aggregate
    cell cap. The stored-margin range guard bounds five factors and three
    reductions; the raw stage also checks its actual band and map factors.
    Its byte admission includes active retained buffers and the contraction
    peak, not just H. If neither stage fits, return None for the bounded row
    fallback. Other tensors' model-owned cell indexes are not charged to this
    per-operation workspace limit.

    Stored joint rows, products of centered margins, and projected raw bands
    are related by rounded matrix products. Reassociation changes both that
    representation error and accumulation error; it does not promise bit
    identity. Numerical tests check the dimension/epsilon absolute-product
    bound against the stored-row oracle, separately from backend dispatch.
    """
    if gm_i.tensor_id == gm_j.tensor_id:
        return None
    cells_i = gm_i.n_bins1 * gm_i.n_bins2 * int(gm_j.B_unique.shape[1])
    cells_j = gm_j.n_bins1 * gm_j.n_bins2 * int(gm_i.B_unique.shape[1])
    if min(cells_i, cells_j) > _MAX_AGGREGATE_CELLS:
        return None
    grid, chan, transposed = (gm_i, gm_j, False) if cells_i <= cells_j else (gm_j, gm_i, True)
    margins = (grid.B1_unique_t, grid.B2_unique_t, chan.B1_unique_t, chan.B2_unique_t)
    if any(operand.dtype != np.float64 for operand in (*margins, chan.B_unique, W)):
        return None
    if not all(_tensor_operand_in_reassociation_range(v) for v in margins):
        return None
    weights_in_range = (
        _tensor_operand_in_reassociation_range(W[:, None])
        if cache is None
        else cache.weight_range(W)[0]
    )
    if not weights_in_range:
        return None

    B1, B2 = grid.B1_unique_t, grid.B2_unique_t
    K1, K2 = B1.shape[1], B2.shape[1]
    n1, n2 = grid.n_bins1, grid.n_bins2
    histogram = _tensor_channel_histogram(grid, chan, W, cache, profile)
    if histogram is None:
        return None
    H, chan_map = histogram
    width = H.shape[1]
    tmp = (B1.T @ H.reshape(n1, n2 * width)).reshape(K1, n2, width)
    raw = np.empty((K1 * K2, width))
    for a in range(K1):
        raw[a * K2 : (a + 1) * K2, :] = B2.T @ tmp[a]
    result = grid.R_inv.T @ raw @ chan_map
    if transposed:
        _profile_count(profile, "block_cross_tensor_tensor_channel_transposed")
        return result.T
    return result


def _cross_gram_tensor_main(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray:
    """Blocked cross-gram between a tensor and a main-effect discretized group.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, shape (p_main, p_tensor).

    Chooses the cheaper channel orientation. If the second tensor margin is
    narrower, aggregate channels over B2 and contract with B1. If the first
    margin is narrower, aggregate channels over B1 and contract with B2.

    Column ordering: j1 * K2 + j2, matching _row_kron_dense().
    """
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    # The tensor margins are already projected. Project the main support
    # before the channel contraction so its small columns do not cancel raw moments.
    B_main, R_main = _cross_support(gm_main, cache, W, B1, B2, gm_tensor.R_inv)
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]

    n_cells_b2 = gm_main.n_bins * gm_tensor.n_bins1 * K2
    n_cells_b1 = gm_main.n_bins * gm_tensor.n_bins2 * K1
    channel_over_b2 = n_cells_b2 <= n_cells_b1
    n_cells = n_cells_b2 if channel_over_b2 else n_cells_b1
    if n_cells <= _MAX_DISC_DISC_CHANNEL_HIST_CELLS:
        if not channel_over_b2:
            H_flat = _disc_disc_2d_hist_channels(
                gm_main.bin_idx,
                gm_tensor.idx2,
                gm_tensor.idx1,
                W,
                B1,
                gm_main.n_bins,
                gm_tensor.n_bins2,
            )
            tmp = B_main.T @ H_flat.reshape(gm_main.n_bins, gm_tensor.n_bins2 * K1)
            tmp_3d = tmp.reshape(K_main_raw, gm_tensor.n_bins2, K1)

            result_raw = np.empty((K_main_raw, K1 * K2))
            for j1 in range(K1):
                result_raw[:, j1 * K2 : (j1 + 1) * K2] = tmp_3d[:, :, j1] @ B2
            return (result_raw if R_main is None else R_main.T @ result_raw) @ gm_tensor.R_inv

        H_flat = _disc_disc_2d_hist_channels(
            gm_main.bin_idx,
            gm_tensor.idx1,
            gm_tensor.idx2,
            W,
            B2,
            gm_main.n_bins,
            gm_tensor.n_bins1,
        )
        tmp = B_main.T @ H_flat.reshape(gm_main.n_bins, gm_tensor.n_bins1 * K2)
        tmp_3d = tmp.reshape(K_main_raw, gm_tensor.n_bins1, K2)

        result_raw = np.empty((K_main_raw, K1 * K2))
        for j2 in range(K2):
            result_raw[:, j2::K2] = tmp_3d[:, :, j2] @ B1
        return (result_raw if R_main is None else R_main.T @ result_raw) @ gm_tensor.R_inv

    result_raw = np.zeros((K_main_raw, K1 * K2))
    if not channel_over_b2:
        for j1 in range(K1):
            w_col = W * B1[gm_tensor.idx1, j1]
            H = _disc_disc_2d_hist(
                gm_main.bin_idx,
                gm_tensor.idx2,
                w_col,
                gm_main.n_bins,
                gm_tensor.n_bins2,
            )
            result_raw[:, j1 * K2 : (j1 + 1) * K2] = B_main.T @ H @ B2
        return (result_raw if R_main is None else R_main.T @ result_raw) @ gm_tensor.R_inv

    for j2 in range(K2):
        # Weight observations by B2[idx2[obs], j2]
        w_col = W * B2[gm_tensor.idx2, j2]
        # 2D histogram: (n_bins_main, n_bins1)
        H = _disc_disc_2d_hist(
            gm_main.bin_idx,
            gm_tensor.idx1,
            w_col,
            gm_main.n_bins,
            gm_tensor.n_bins1,
        )
        # Contract: (K_main, n_bins_main) × (n_bins_main, n_bins1) × (n_bins1, K1)
        result_raw[:, j2::K2] = B_main.T @ H @ B1

    return (result_raw if R_main is None else R_main.T @ result_raw) @ gm_tensor.R_inv


def _cross_gram_tensor_own_margin(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray | None:
    """Cross-Gram for tensor × one of its own discretized marginal smooths.

    mgcv bam(discrete=TRUE) stores tensor marginals as compact matrices plus
    row index arrays and forms X'WX from that packed representation via XWXd.
    For tensor × own-margin main-effect cross-blocks, reuse the tensor 2D
    W-grid instead of rescanning observations through the generic tensor-main
    channel histogram.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, or None if the main
    group is not exactly one unambiguous tensor margin.
    """
    cache_key = (id(gm_main), id(gm_main.bin_idx), gm_main.n_bins)
    margin = gm_tensor._own_margin_cache.get(cache_key, -1)
    if margin == -1:
        same_margin1 = gm_main.n_bins == gm_tensor.n_bins1 and np.array_equal(
            gm_main.bin_idx, gm_tensor.idx1
        )
        same_margin2 = gm_main.n_bins == gm_tensor.n_bins2 and np.array_equal(
            gm_main.bin_idx, gm_tensor.idx2
        )
        margin = None if same_margin1 == same_margin2 else (1 if same_margin1 else 2)
        gm_tensor._own_margin_cache[cache_key] = margin
    if margin is None:
        return None

    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    B_main, R_main = _cross_support(gm_main, cache, W, B1, B2, gm_tensor.R_inv)
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]
    result_raw = np.empty((K_main_raw, K1 * K2), dtype=np.float64)
    if cache is None:
        w_grid = _disc_disc_2d_hist(
            gm_tensor.idx1,
            gm_tensor.idx2,
            W,
            gm_tensor.n_bins1,
            gm_tensor.n_bins2,
        )
    else:
        w_grid = cache.tensor_w_grid(gm_tensor, W)

    if margin == 1:
        weighted_margin2 = w_grid @ B2  # (n_bins1, K2)
        for j2 in range(K2):
            result_raw[:, j2::K2] = B_main.T @ (B1 * weighted_margin2[:, j2][:, None])
    else:
        weighted_margin1 = w_grid.T @ B1  # (n_bins2, K1)
        for j1 in range(K1):
            result_raw[:, j1 * K2 : (j1 + 1) * K2] = B_main.T @ (
                B2 * weighted_margin1[:, j1][:, None]
            )

    return (result_raw if R_main is None else R_main.T @ result_raw) @ gm_tensor.R_inv


def _cross_gram_tensor_spline_categorical(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram X_tensor.T W X_spline_cat without materialising joint support."""
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    K1, K2 = B1.shape[1], B2.shape[1]
    K_cat = gm_spline_cat._p_b
    result_raw = np.empty((K1 * K2, K_cat), dtype=np.float64)
    rows = gm_spline_cat.row_idx
    W_rows = W[rows]
    idx1 = gm_tensor.idx1[rows]
    idx2 = gm_tensor.idx2[rows]

    if hasattr(gm_spline_cat, "B_unique"):
        bin_cat = gm_spline_cat.bin_idx_level
        # A lossless spline_cat support is bounded by its row count, not by a
        # bin count, so the joint histogram is not automatically small; expand
        # the level's rows once and aggregate on them when it is not.
        if gm_tensor.n_bins1 * gm_spline_cat.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
            for j2 in range(K2):
                w_col = W_rows * B2[idx2, j2]
                H = _disc_disc_2d_hist(
                    idx1,
                    bin_cat,
                    w_col,
                    gm_tensor.n_bins1,
                    gm_spline_cat.n_bins,
                )
                result_raw[j2::K2, :] = B1.T @ H @ gm_spline_cat.B_unique
            return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

        # Over the cell cap.  Chunk the row expansion rather than materialising
        # the whole level: the cap bounds CELLS, and a lossless support leaves
        # the row count unbounded underneath it.  The chunk loop is outermost so
        # each row is still expanded exactly once across all K2 passes.
        n_level = int(W_rows.shape[0])
        chunk = min(
            max(n_level, 1),
            _cross_expansion_chunk_rows(K_cat, 0, _MAX_CROSS_EXPANSION_BYTES),
        )
        # Holding every channel at once lets each row be expanded exactly once,
        # but the accumulator is (K2, n_bins1, K_cat) -- itself cross-shaped, and
        # n_bins1 follows a per-feature n_bins that nothing caps.  So the fast
        # nesting is used only while that buffer fits, and otherwise the loops
        # invert: one channel at a time, one (n_bins1, K_cat) accumulator, at the
        # cost of re-expanding the rows per channel.  Bounded memory bought with
        # time, declared rather than assumed.
        if K2 * gm_tensor.n_bins1 * K_cat <= _MAX_AGGREGATE_CELLS:
            agg = np.zeros((K2, gm_tensor.n_bins1, K_cat), dtype=np.float64)
            for start in range(0, n_level, chunk):
                stop = min(start + chunk, n_level)
                block = _expand_support_rows(gm_spline_cat.B_unique, bin_cat[start:stop])
                idx1_chunk = idx1[start:stop]
                for j2 in range(K2):
                    w_col = W_rows[start:stop] * B2[idx2[start:stop], j2]
                    agg[j2] += _weighted_bincount_2d(idx1_chunk, w_col, block, gm_tensor.n_bins1)
                del block  # next expansion is evaluated before the rebind
            for j2 in range(K2):
                result_raw[j2::K2, :] = B1.T @ agg[j2]
            return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

        # One channel at a time is still not enough: a single (n_bins1, K_cat)
        # channel is itself cross-shaped, and n_bins1 follows a configured
        # n_bins.  So the basis dimension is tiled inside the channel as well,
        # and only the columns of that tile are ever expanded.  What is left at
        # the smallest tile is (n_bins1, 1), which is the least an aggregate
        # over n_bins1 bins can occupy.
        tile = _aggregate_column_chunk(gm_tensor.n_bins1, K_cat)
        for j2 in range(K2):
            for first in range(0, K_cat, tile):
                last = min(first + tile, K_cat)
                columns = gm_spline_cat.B_unique[:, first:last]
                channel = np.zeros((gm_tensor.n_bins1, last - first), dtype=np.float64)
                for start in range(0, n_level, chunk):
                    stop = min(start + chunk, n_level)
                    block = _expand_support_rows(columns, bin_cat[start:stop])
                    w_col = W_rows[start:stop] * B2[idx2[start:stop], j2]
                    channel += _weighted_bincount_2d(
                        idx1[start:stop], w_col, block, gm_tensor.n_bins1
                    )
                    del block
                result_raw[j2::K2, first:last] = B1.T @ channel
        return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv

    for j2 in range(K2):
        w_col = W_rows * B2[idx2, j2]
        B_cat_agg = _csr_weighted_bincount(
            gm_spline_cat._data,
            gm_spline_cat._indices,
            gm_spline_cat._indptr,
            K_cat,
            idx1,
            w_col,
            gm_tensor.n_bins1,
        )
        result_raw[j2::K2, :] = B1.T @ B_cat_agg

    return gm_tensor.R_inv.T @ result_raw @ gm_spline_cat.R_inv


def _expand_support_rows(B_unique: NDArray, bin_idx: NDArray) -> NDArray:
    """Materialise a support block on its observation rows.

    Named so the chunking above it can be pinned by row count in a test rather
    than asserted about.
    """
    # take avoids fancy-index row overhead for contiguous supports. On a
    # strided support it first copies the entire input, violating panel bounds.
    if B_unique.flags.c_contiguous:
        return np.take(B_unique, bin_idx, axis=0)
    return B_unique[bin_idx]


def _cross_expansion_chunk_rows(p_i: int, p_j: int, max_bytes: int) -> int:
    """Rows per chunk of the expanded cross-gram, sized in BYTES.

    Two expanded blocks are live at once and the right one is scaled in place,
    so a chunk costs ``rows * (p_i + p_j) * 8``.  Pass ``p_j=0`` where only one
    block is expanded.
    """
    return max(1, int(max_bytes // max((p_i + p_j) * 8, 1)))


def _chunked_support_bincount_2d(
    bin_idx: NDArray,
    weights: NDArray,
    B_unique: NDArray,
    support_idx: NDArray,
    n_bins: int,
    max_bytes: int | None = None,
) -> NDArray:
    """``_weighted_bincount_2d`` over support-indexed rows, expanded in chunks.

    The aggregation is a sum over rows, so partitioning the rows partitions the
    sum.  Chunking matters here for the same reason it does in
    :func:`_support_support_raw_cross`: a lossless support bounds the number of
    DISTINCT rows, not the number of observation rows the level owns, so
    materialising the level in one go is unbounded in ``n``.

    Below the threshold the loop runs once and the result is bit-identical to
    the unchunked form, so ordinary fits are unaffected.
    """
    # Resolved at call time, not bound as a default: the budget is a module
    # global so that it can be lowered in a test, and a default argument would
    # freeze it at import.
    budget = _MAX_CROSS_EXPANSION_BYTES if max_bytes is None else max_bytes
    p_b = int(B_unique.shape[1])
    out = np.zeros((int(n_bins), p_b), dtype=np.float64)
    n_rows = int(np.size(support_idx))
    if n_rows == 0:
        return out
    chunk = min(n_rows, _cross_expansion_chunk_rows(p_b, 0, budget))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = _expand_support_rows(B_unique, support_idx[start:stop])
        out += _weighted_bincount_2d(bin_idx[start:stop], weights[start:stop], block, int(n_bins))
        # See _support_support_raw_cross: the next expansion is evaluated before
        # the name is rebound, so without this the ceiling is 2x the budget.
        del block
    return out


def _mixed_chunk_stop(
    indptr: NDArray, start_row: int, n_rows: int, dense_rows: int, max_nnz: int
) -> int:
    """End of a mixed-pair row chunk, budgeting the SPARSE payload as well.

    Sizing by the dense expansion alone budgets only ``rows * p_b * 8``.  The
    weighted CSR slice beside it costs ``nnz-in-range * 12`` (float64 plus a
    32-bit index), and ``nnz`` per row is a property of the OTHER block -- for a
    cardinal-CR basis the rows are structurally dense.  A narrow compressed side
    therefore permits a huge row count, and the sparse payload follows it: five
    compressed columns admit ~1.68M rows, which against a 20-column dense CSR is
    tens of millions of stored entries.  Boundaries come off ``indptr``, so the
    bound is the range's ACTUAL nonzeros rather than an average that a skewed
    row density would defeat.
    """
    stop_row = min(start_row + dense_rows, n_rows)
    limit = int(indptr[start_row]) + max_nnz
    # Largest row boundary whose cumulative nnz is still inside the budget.
    by_payload = int(np.searchsorted(indptr, limit, side="right")) - 1
    stop_row = min(stop_row, by_payload)
    # One row can never be split, so always make progress.
    return max(stop_row, start_row + 1)


def _weighted_row_chunk(csr, W_rows: NDArray, start_row: int, stop_row: int):
    """Weight CSR rows using one owned value buffer and shared column indices."""
    lo = int(csr.indptr[start_row])
    hi = int(csr.indptr[stop_row])
    row_ptr = csr.indptr[start_row : stop_row + 1]
    data = np.repeat(W_rows[start_row:stop_row], np.diff(row_ptr))
    data *= csr.data[lo:hi]
    return _csr_row_chunk(csr, start_row, stop_row, data=data)


def _support_csr_raw_cross(
    B_unique: NDArray,
    support_idx: NDArray,
    B_csr,
    W_rows: NDArray,
) -> NDArray:
    """``B_support.T @ diag(W) @ B_csr`` with NO observation-level temporary.

    The mixed pairing -- one ``spline_cat`` block compressed, the other still
    CSR -- is created by the compression gate itself, so it is a regression
    rather than a pre-existing path: before, two exact blocks contracted sparse
    against sparse.  Expanding the compressed side and densifying the weighted
    CSR side made BOTH sides worse than they had been.

    Aggregating the CSR side onto the support bins first avoids the choice.
    The only dense array is ``(n_support, p_csr)``, bounded by the same support
    gate that bounds ``B_unique``; the CSR side is never densified and the
    observation rows are never materialised, so no chunking is needed.
    """
    # Row-chunked, in ONE pass over the data.  Column chunking bounded the
    # memory but made every pass re-walk all n row pointers -- measured at 88
    # passes for n_support=1e6 against a 700-column term, and the CSC rewrite
    # that removes the slicing cost still leaves that traversal, so it bought
    # 1.10x rather than fixing it.  Contracting over row chunks instead touches
    # each nonzero once, keeps the CSR side sparse throughout, and never forms
    # the (n_support, p_csr) aggregate at all -- so the cross-shaped array this
    # helper existed to bound is simply not built.
    csr = B_csr.tocsr()
    p_b = int(B_unique.shape[1])
    p_csr = int(csr.shape[1])
    support_idx = np.asarray(support_idx, dtype=np.intp)
    W_rows = np.asarray(W_rows, dtype=np.float64)
    out = np.zeros((p_b, p_csr), dtype=np.float64)
    n_rows = int(W_rows.shape[0])
    if n_rows == 0:
        return out
    dense_rows = min(n_rows, _cross_expansion_chunk_rows(p_b, 0, _MAX_CROSS_EXPANSION_BYTES))
    # float64 payload plus a 32-bit column index per stored entry.
    max_nnz = max(1, _MAX_CROSS_EXPANSION_BYTES // 12)
    start_row = 0
    while start_row < n_rows:
        stop_row = _mixed_chunk_stop(csr.indptr, start_row, n_rows, dense_rows, max_nnz)
        left = _expand_support_rows(B_unique, support_idx[start_row:stop_row])
        right = _weighted_row_chunk(csr, W_rows, start_row, stop_row)
        # sparse.T @ dense keeps the CSR side sparse; the product is (p_csr, p_b).
        out += np.asarray(right.T @ left, dtype=np.float64).T
        del left, right
        start_row = stop_row
    return out


def _support_support_raw_cross(
    B_unique_i: NDArray,
    bin_idx_i: NDArray,
    B_unique_j: NDArray,
    bin_idx_j: NDArray,
    W_rows: NDArray,
    max_bytes: int | None = None,
) -> NDArray:
    """``B_i.T @ diag(W) @ B_j`` for two support-indexed blocks over shared rows.

    The 2-D weight histogram both callers prefer costs ``n_bins_i * n_bins_j``
    cells, which is bounded only when the supports are bins.  A lossless
    support is bounded by the row count instead, so on wide supports this
    contracts over the shared rows and lets BLAS do the work.

    Chunked over rows, because the row count is exactly what a lossless support
    does NOT bound: expanding both sides in one go costs
    ``n_rows * (p_i + p_j) * 8`` bytes, which for a dominant level on a large
    book runs to hundreds of MB per call, inside solver iterations.  The cap
    that routes here bounds cells; without this it would only move the memory
    rather than bound it.  Accumulating in chunks keeps the transient at
    ``max_bytes`` regardless of row count, and the contraction is a sum over
    rows so partitioning it changes nothing but summation order.
    """
    budget = _MAX_CROSS_EXPANSION_BYTES if max_bytes is None else max_bytes
    p_i = int(B_unique_i.shape[1])
    p_j = int(B_unique_j.shape[1])
    out = np.zeros((p_i, p_j), dtype=np.float64)
    n_rows = int(W_rows.shape[0])
    if n_rows == 0:
        return out
    chunk = min(n_rows, _cross_expansion_chunk_rows(p_i, p_j, budget))
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        left = _expand_support_rows(B_unique_i, bin_idx_i[start:stop])
        right = _expand_support_rows(B_unique_j, bin_idx_j[start:stop])
        # Fancy indexing already returned a fresh array, so scaling it in place
        # keeps the live count at two blocks rather than three.
        right *= W_rows[start:stop, None]
        out += left.T @ right
        # Released explicitly: the next iteration's expansion is evaluated
        # BEFORE its name is rebound, so without this the previous pair is
        # still referenced at the allocation instant and the real ceiling is
        # 1.5x the budget rather than 1x.
        del left, right
    return out


def _cross_gram_categorical_spline_categorical(
    gm_cat: GroupMatrix,
    gm_spline_cat: GroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram X_cat.T W X_spline_cat via one categorical aggregation."""
    if hasattr(gm_spline_cat, "B_unique"):
        rows = gm_spline_cat.row_idx
        # Chunked: this expands the level to observation rows, which no cap
        # above it bounds.  Pre-dates support compression -- the binned path
        # reaches it too -- but compression puts it on the hot path of every
        # model pairing a Categorical main effect with a spline_cat term,
        # which is every model this compression targets.
        B_agg = _chunked_support_bincount_2d(
            gm_cat.codes[rows],
            W[rows],
            gm_spline_cat.B_unique,
            gm_spline_cat.bin_idx_level,
            gm_cat.n_levels + 1,
        )
        return B_agg[: gm_cat.n_levels] @ gm_spline_cat.R_inv

    B_agg = _csr_weighted_bincount(
        gm_spline_cat._data,
        gm_spline_cat._indices,
        gm_spline_cat._indptr,
        gm_spline_cat._p_b,
        gm_cat.codes[gm_spline_cat.row_idx],
        W[gm_spline_cat.row_idx],
        gm_cat.n_levels + 1,
    )
    return B_agg[: gm_cat.n_levels] @ gm_spline_cat.R_inv


def _cross_gram_spline_categorical_spline_categorical(
    gm_i: SplineCategoricalGroupMatrix | DiscretizedSplineCategoricalGroupMatrix,
    gm_j: SplineCategoricalGroupMatrix | DiscretizedSplineCategoricalGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram between compact spline-category level groups."""
    same_cat_parent = getattr(gm_i, "spline_cat_feature", None) is not None and getattr(
        gm_i, "spline_cat_feature", None
    ) == getattr(gm_j, "spline_cat_feature", None)
    if same_cat_parent and np.array_equal(gm_i.row_idx, gm_j.row_idx):
        rows = gm_i.row_idx
        i_discrete = hasattr(gm_i, "B_unique")
        j_discrete = hasattr(gm_j, "B_unique")
        if i_discrete and j_discrete:
            if gm_i.n_bins * gm_j.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
                H = _disc_disc_2d_hist(
                    gm_i.bin_idx_level,
                    gm_j.bin_idx_level,
                    W[rows],
                    gm_i.n_bins,
                    gm_j.n_bins,
                )
                raw = gm_i.B_unique.T @ H @ gm_j.B_unique
            else:
                raw = _support_support_raw_cross(
                    gm_i.B_unique,
                    gm_i.bin_idx_level,
                    gm_j.B_unique,
                    gm_j.bin_idx_level,
                    W[rows],
                )
            return gm_i.R_inv.T @ raw @ gm_j.R_inv
        if i_discrete:
            raw = _support_csr_raw_cross(gm_i.B_unique, gm_i.bin_idx_level, gm_j.B_level, W[rows])
            return gm_i.R_inv.T @ raw @ gm_j.R_inv
        if j_discrete:
            raw = _support_csr_raw_cross(gm_j.B_unique, gm_j.bin_idx_level, gm_i.B_level, W[rows]).T
            return gm_i.R_inv.T @ raw @ gm_j.R_inv

        raw = gm_i.B_level.T @ gm_j.B_level.multiply(W[rows][:, None])
        if hasattr(raw, "toarray"):
            raw = raw.toarray()
        return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv

    common_rows, pos_i, pos_j = np.intersect1d(
        gm_i.row_idx,
        gm_j.row_idx,
        assume_unique=True,
        return_indices=True,
    )
    if common_rows.size == 0:
        return np.zeros((gm_i.shape[1], gm_j.shape[1]))

    i_discrete = hasattr(gm_i, "B_unique")
    j_discrete = hasattr(gm_j, "B_unique")
    if i_discrete and j_discrete:
        if gm_i.n_bins * gm_j.n_bins <= _MAX_DISC_DISC_HIST_CELLS:
            H = _disc_disc_2d_hist(
                gm_i.bin_idx_level[pos_i],
                gm_j.bin_idx_level[pos_j],
                W[common_rows],
                gm_i.n_bins,
                gm_j.n_bins,
            )
            raw = gm_i.B_unique.T @ H @ gm_j.B_unique
        else:
            raw = _support_support_raw_cross(
                gm_i.B_unique,
                gm_i.bin_idx_level[pos_i],
                gm_j.B_unique,
                gm_j.bin_idx_level[pos_j],
                W[common_rows],
            )
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    if i_discrete:
        raw = _support_csr_raw_cross(
            gm_i.B_unique, gm_i.bin_idx_level[pos_i], gm_j.B_level[pos_j], W[common_rows]
        )
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    if j_discrete:
        raw = _support_csr_raw_cross(
            gm_j.B_unique, gm_j.bin_idx_level[pos_j], gm_i.B_level[pos_i], W[common_rows]
        ).T
        return gm_i.R_inv.T @ raw @ gm_j.R_inv

    B_i = gm_i.B_level[pos_i]
    B_j = gm_j.B_level[pos_j]
    raw = B_i.T @ B_j.multiply(W[common_rows][:, None])
    if hasattr(raw, "toarray"):
        raw = raw.toarray()
    return gm_i.R_inv.T @ np.asarray(raw, dtype=np.float64) @ gm_j.R_inv


def _cross_gram_discrete_spline_categorical(
    gm_disc: DiscretizedSSPGroupMatrix,
    gm_spline_cat: DiscretizedSplineCategoricalGroupMatrix,
    W: NDArray,
) -> NDArray | None:
    """Cross-gram X_disc.T W X_spline_cat from compressed support bins."""
    if not hasattr(gm_spline_cat, "B_unique"):
        return None
    rows = gm_spline_cat.row_idx
    n_joint = gm_disc.n_bins * gm_spline_cat.n_bins
    if n_joint > _MAX_DISC_DISC_HIST_CELLS:
        return None
    H = _disc_disc_2d_hist(
        gm_disc.bin_idx[rows],
        gm_spline_cat.bin_idx_level,
        W[rows],
        gm_disc.n_bins,
        gm_spline_cat.n_bins,
    )
    raw = gm_disc.B_unique.T @ H @ gm_spline_cat.B_unique
    return gm_disc.R_inv.T @ raw @ gm_spline_cat.R_inv


def _support_support_cross_gram(
    gm_disc: DiscretizedSSPGroupMatrix,
    gm_spline_cat: DiscretizedSplineCategoricalGroupMatrix,
    W: NDArray,
) -> NDArray:
    """The same product as :func:`_cross_gram_discrete_spline_categorical`,
    for the case that one declines.

    Two lossless supports can make the joint histogram exceed the cell cap.
    Falling through from there used to reach ``_agg_by_bin``, whose output is
    ``(gm_disc.n_bins, p_spline_cat)`` -- cross-shaped, with the row count from
    one block and the width from the other, so neither support gate bounds it.
    Contracting over the shared rows instead reuses the row-chunked path, whose
    transient is the byte budget regardless of either support size.
    """
    rows = gm_spline_cat.row_idx
    raw = _support_support_raw_cross(
        gm_disc.B_unique,
        gm_disc.bin_idx[rows],
        gm_spline_cat.B_unique,
        gm_spline_cat.bin_idx_level,
        W[rows],
    )
    return gm_disc.R_inv.T @ raw @ gm_spline_cat.R_inv


def _cross_gram_factor_smooth_dense(
    factor: FactorSmoothGroupMatrix, dense: DenseGroupMatrix, W: NDArray
) -> NDArray | None:
    """Batch a narrow dense partner, retaining the legacy route outside its envelope."""
    from ..factor_smooth_geometry import adjoint_sum_to_zero_blocks

    q = dense.shape[1]
    # A singleton already scans the factor once; a wider partner should keep
    # the fallback's smaller-width loop. No observation design is expanded.
    if not 2 <= q < factor.shape[1]:
        return None
    raw_cells = factor.n_levels * factor.raw_width * q
    mapped_cells = factor.n_levels * factor.block_size * q
    # Include einsum's possible raw-layout copy and public-coordinate/reshape
    # copies, in addition to its raw and mapped outputs. Decline before any of
    # those cross-shaped allocations; a level-by-support grid is never needed.
    if 8 * (2 * raw_cells + 3 * mapped_cells) > _MAX_CROSS_EXPANSION_BYTES:
        return None
    basis = factor.B_unique if factor.is_discrete else factor._data
    if (
        type(W) is not np.ndarray
        or W.shape != (factor.shape[0],)
        or dense.shape[0] != factor.shape[0]
        or any(
            type(value) is not np.ndarray or value.dtype != np.float64
            for value in (W, dense.M, basis, factor.natural_map)
        )
    ):
        return None
    # Check the raw source before indexing: an ndarray subclass can override
    # view creation and return an ordinary array with changed values.
    basis = cast(NDArray, basis)
    if not factor.is_discrete:
        basis = basis[:, None]
    # The native scan computes (W*basis)*dense, while the legacy transpose
    # product computes basis*(W*dense). Four factors and two reductions fit
    # comfortably in float64's exponent range under this existing guard.
    # Recheck mutable operands each time, including the raw natural map.
    if not all(
        _tensor_operand_in_reassociation_range(value)
        for value in (W[:, None], dense.M, basis, factor.natural_map)
    ):
        return None
    blocks = factor.factor_smooth_dense_cross_gram(W, dense.M)
    if factor.factor_basis == "sz":
        blocks = adjoint_sum_to_zero_blocks(blocks)
    return blocks.reshape(factor.shape[1], q)


def _full_csr_values(basis) -> NDArray | None:
    """View canonical, fully stored live CSR values as the raw dense basis."""
    rows, cols = basis.shape
    if (
        basis.data.size == rows * cols
        and basis.data.flags.c_contiguous
        and np.all(np.diff(basis.indptr) == cols)
        and np.all(basis.indices[basis.indptr[:-1]] == 0)
        and np.all(basis.indices[basis.indptr[1:] - 1] == cols - 1)
    ):
        return basis.data.reshape(rows, cols)
    return None


def _cross_gram_sparse_ssp(
    gm_i: SparseSSPGroupMatrix,
    gm_j: SparseSSPGroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
) -> NDArray | None:
    """Contract live raw SSP bases before mapping the small cross product."""
    B_i, B_j = gm_i.B, gm_j.B
    R_i, R_j = gm_i.R_inv, gm_j.R_inv
    n = gm_i.shape[0]
    if (
        type(B_i) is not sp.csr_matrix
        or type(B_j) is not sp.csr_matrix
        or any(
            type(value) is not np.ndarray or value.dtype != np.float64
            for value in (W, B_i.data, B_j.data, R_i, R_j)
        )
        or any(
            type(value) is not np.ndarray
            or value.dtype not in (np.dtype(np.int32), np.dtype(np.int64))
            for basis in (B_i, B_j)
            for value in (basis.indices, basis.indptr)
        )
        or W.shape != (n,)
        or B_i.shape[0] != n
        or B_j.shape[0] != n
        or gm_j.shape[0] != n
        or R_i.shape != (B_i.shape[1], gm_i.shape[1])
        or R_j.shape != (B_j.shape[1], gm_j.shape[1])
        or min(n, B_i.shape[1], B_j.shape[1], gm_i.shape[1], gm_j.shape[1]) == 0
    ):
        return None
    k_i, k_j = B_i.shape[1], B_j.shape[1]
    p_i, p_j = gm_i.shape[1], gm_j.shape[1]
    # Conservatively allow two weighted payloads and an index copy, four
    # row-pointer/work arrays, three raw value/index buffers including dense
    # conversion, and both mapped outputs. No n-by-solver-width array exists.
    transient_bytes = (
        24 * max(B_i.data.size, B_j.data.size)
        + 32 * (n + 1)
        + 48 * k_i * k_j
        + 8 * (p_i * k_j + p_i * p_j)
    )
    # Reassociation has five factors and three reductions. This existing
    # interval keeps their products and sums in binary64's exponent range.
    if not all(
        _tensor_operand_in_reassociation_range(value)
        for value in (W[:, None], B_i.data[:, None], B_j.data[:, None], R_i, R_j)
    ):
        return None
    # Every SSP operation reads the same live B buffers.
    # Fresh views also avoid cached canonical flags after index mutations.
    # The sparse-object constructor preserves even int64 buffers; the tuple
    # constructor may downcast and retain observation-sized index copies.
    left = sp.csr_matrix(B_i, copy=False)
    right = sp.csr_matrix(B_j, copy=False)
    if not left.has_canonical_format or not right.has_canonical_format:
        return None
    cache = _BlockWeightCache() if cache is None else cache
    if cache.sparse_gram(gm_i, W)[1] or cache.sparse_gram(gm_j, W)[1]:
        # Retain bounded row products only for cancellation-sensitive blocks.
        # The ordinary sparse raw cross and tensor channel routes stay intact.
        pointer_bytes = max(left.indptr.dtype.itemsize, right.indptr.dtype.itemsize)
        chunk = max(
            1,
            (_MAX_CROSS_EXPANSION_BYTES - 8 * np.getbufsize() - 2 * pointer_bytes)
            // (8 * (p_i + p_j) + 2 * pointer_bytes),
        )
        result = np.zeros((p_i, p_j))
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            first = _csr_row_chunk(left, start, stop) @ R_i
            second = _csr_row_chunk(right, start, stop) @ R_j
            second *= W[start:stop, None]
            result += first.T @ second
            del first, second
        return result
    if transient_bytes > _MAX_CROSS_EXPANSION_BYTES:
        return None
    dense_i, dense_j = _full_csr_values(left), _full_csr_values(right)
    if dense_j is not None and (dense_i is None or left.nnz <= right.nnz):
        weighted = _weighted_row_chunk(left, W, 0, n)
        raw = np.asarray(weighted.T @ dense_j)
    elif dense_i is not None:
        weighted = _weighted_row_chunk(right, W, 0, n)
        raw = np.asarray(weighted.T @ dense_i).T
    elif left.nnz <= right.nnz:
        weighted = _weighted_row_chunk(left, W, 0, n)
        raw = (weighted.T @ right).toarray()
    else:
        weighted = _weighted_row_chunk(right, W, 0, n)
        raw = (weighted.T @ left).toarray().T
    return R_i.T @ raw @ R_j


def _cross_gram_by_columns(
    gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray, *, project_i=False, project_j=False
) -> NDArray:
    """Form a cross-product one generated column at a time.

    This is the bounded-memory fallback for factored support-space groups.
    It preserves the same smaller-width loop count as the generic fallback
    without materializing either effective observation-level design block.
    """
    p_i = gm_i.shape[1]
    p_j = gm_j.shape[1]
    if project_j and not project_i:
        return _cross_gram_by_columns(gm_j, gm_i, W, project_i=True).T
    if project_i:
        # Match the sensitive diagonal's represented solver columns. Sending
        # a partner column through R'@(B'@v) would reintroduce cancellation.
        result = np.empty((p_i, p_j))
        unit_i, unit_j = np.zeros(p_i), np.zeros(p_j)
        for column in range(p_i):
            unit_i[column] = 1.0
            weighted = W * gm_i.matvec(unit_i)
            unit_i[column] = 0.0
            if not project_j:
                result[column] = gm_j.rmatvec(weighted)
            else:
                for partner in range(p_j):
                    unit_j[partner] = 1.0
                    result[column, partner] = weighted @ gm_j.matvec(unit_j)
                    unit_j[partner] = 0.0
        return result
    if p_i <= p_j:
        result = np.empty((p_i, p_j), dtype=np.float64)
        unit = np.zeros(p_i, dtype=np.float64)
        for column in range(p_i):
            unit[column] = 1.0
            result[column] = gm_j.rmatvec(W * gm_i.matvec(unit))
            unit[column] = 0.0
        return result

    result = np.empty((p_i, p_j), dtype=np.float64)
    unit = np.zeros(p_j, dtype=np.float64)
    for column in range(p_j):
        unit[column] = 1.0
        result[:, column] = gm_i.rmatvec(W * gm_j.matvec(unit))
        unit[column] = 0.0
    return result


def _cross_gram(
    gm_i: GroupMatrix,
    gm_j: GroupMatrix,
    W: NDArray,
    cache: _BlockWeightCache | None = None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compute X_i.T @ diag(W) @ X_j efficiently.

    For two DiscretizedSSPGroupMatrix, uses a 2D weight histogram to avoid
    materializing either (n, p) matrix. For disc × non-disc, aggregates by
    disc bins in a single compiled pass (fused W-weighting, no toarray for
    sparse groups). Otherwise falls back to materializing the smaller group
    and using rmatvec on the larger.
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        _SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import DenseGroupMatrix, FactorSmoothGroupMatrix

    SplineCatTypes = (SplineCategoricalGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)

    if isinstance(gm_i, SplineCatTypes) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_spline_categorical_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_spline_cat_spline_cat_s", t0)
        return result
    # Tensor × tensor (same marginals, e.g. decomposed bilinear/wiggly)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and gm_i.tensor_id == gm_j.tensor_id
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_tensor(gm_i, gm_j, W, cache)
        _profile_elapsed(profile, "block_cross_tensor_tensor_s", t0)
        return result
    if isinstance(gm_i, DiscretizedTensorGroupMatrix) and isinstance(
        gm_j, DiscretizedTensorGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_tensor_shared_margin(gm_i, gm_j, W, cache)
        if result is not None:
            _profile_elapsed(profile, "block_cross_tensor_tensor_s", t0)
            return result
        # Distinct margins, or a shared margin above the compact helper's cap
        # (every pair at the default 256 bins): stage the block through the
        # channel histogram. A decline is counted so that a fit which backs
        # off to the quadratic row route below says so in its profile.
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_tensor_channels(gm_i, gm_j, W, cache, profile)
        if result is not None:
            _profile_elapsed(profile, "block_cross_tensor_tensor_channel_s", t0)
            _profile_count(profile, "block_cross_tensor_tensor_channel_calls")
            return result
        _profile_count(profile, "block_cross_tensor_tensor_channel_declines")
        if cache is not None:
            # Every channel decline, including its early gates, reaches here.
            # The following routes budget their own workspace independently.
            cache.release_channel_buffers()

    if isinstance(gm_i, DiscretizedTensorGroupMatrix) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_tensor_spline_cat_s", t0)
        return result
    if isinstance(gm_j, DiscretizedTensorGroupMatrix) and isinstance(gm_i, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_tensor_spline_categorical(gm_j, gm_i, W).T
        _profile_elapsed(profile, "block_cross_tensor_spline_cat_s", t0)
        return result

    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_categorical_spline_categorical(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_cat_spline_cat_s", t0)
        return result
    if isinstance(gm_j, CategoricalGroupMatrix) and isinstance(gm_i, SplineCatTypes):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_categorical_spline_categorical(gm_j, gm_i, W).T
        _profile_elapsed(profile, "block_cross_cat_spline_cat_s", t0)
        return result

    if (
        isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSplineCategoricalGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_discrete_spline_categorical(gm_i, gm_j, W)
        if result is None:
            # Declined at the cell cap.  Falling through would reach the
            # disc-x-non-disc branch and _agg_by_bin, whose output is
            # cross-shaped (gm_i.n_bins, p_j) and bounded by neither support
            # gate.  Contract over rows instead: already chunked, already
            # bounded, and it is the same product.
            result = _support_support_cross_gram(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result
    if (
        isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSplineCategoricalGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_discrete_spline_categorical(gm_j, gm_i, W)
        if result is None:
            result = _support_support_cross_gram(gm_j, gm_i, W)
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result.T

    # Tensor × discretized main-effect (not tensor × tensor with different ids)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        own_margin = _cross_gram_tensor_own_margin(gm_i, gm_j, W, cache)
        if own_margin is not None:
            _profile_elapsed(profile, "block_cross_tensor_own_margin_s", t0)
            return own_margin.T
        result = _cross_gram_tensor_main(gm_i, gm_j, W, cache).T
        _profile_elapsed(profile, "block_cross_tensor_main_s", t0)
        return result
    if (
        isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
    ):
        t0 = perf_counter() if profile is not None else 0.0
        own_margin = _cross_gram_tensor_own_margin(gm_j, gm_i, W, cache)
        if own_margin is not None:
            _profile_elapsed(profile, "block_cross_tensor_own_margin_s", t0)
            return own_margin
        result = _cross_gram_tensor_main(gm_j, gm_i, W, cache)
        _profile_elapsed(profile, "block_cross_tensor_main_s", t0)
        return result

    DiscTypes = (DiscretizedSSPGroupMatrix, DiscretizedSCOPGroupMatrix)
    if isinstance(gm_i, DiscTypes) and isinstance(gm_j, DiscTypes):
        t0 = perf_counter() if profile is not None else 0.0
        # Preserve the established histogram association if projecting a
        # factor first could overflow/underflow. Ordinary support products use
        # solver coordinates; this gate changes arithmetic, not rank policy.
        parts = []
        for gm, partner in ((gm_i, gm_j), (gm_j, gm_i)):
            if not isinstance(gm, DiscretizedSSPGroupMatrix):
                parts.append((gm.B_scop_unique, None))
            else:
                factors = (
                    (partner.B_unique, partner.R_inv)
                    if isinstance(partner, DiscretizedSSPGroupMatrix)
                    else (partner.B_scop_unique,)
                )
                parts.append(_cross_support(gm, cache, W, *factors))
        (B_i, R_i), (B_j, R_j) = parts
        n_i, p_i = B_i.shape
        n_j, p_j = B_j.shape
        n_joint = n_i * n_j
        # Histogram contraction is efficient BLAS on the bin grid; row panels
        # also pay NumPy setup, indexed gathering, and weighting costs. Generic
        # support-grid probes put those overheads at about 128k fixed and 512
        # per-row multiply-add equivalents. Require a 25% estimated advantage
        # to change routes; retain histogram reuse near the crossover. Maps are
        # common to both routes, and the cell ceiling remains a hard limit.
        hist_work = n_joint * (1 + p_i) + n_j * p_i * p_j
        row_work = 128_000 + len(W) * (512 + p_i + p_j + p_i * p_j)
        # The row helper weights a panel in place with a float64 byte budget.
        # Keep other support dtypes on their established histogram/fallback
        # routes so integer casting and float32 accumulation do not change.
        use_rows = (
            B_i.dtype == np.float64
            and B_j.dtype == np.float64
            and (n_joint > _MAX_DISC_DISC_HIST_CELLS or 4 * row_work < 3 * hist_work)
        )
        # Speculative rows change (B_i.T @ histogram(W)) @ B_j into
        # B_i.T @ (W * B_j). Preserve the histogram's exponent behavior when
        # its allocation fits; above the cap retain the bounded row fallback.
        if use_rows and n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            use_rows = all(
                _tensor_operand_in_reassociation_range(value) for value in (B_i, B_j, W[:, None])
            )
        if use_rows or n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            if use_rows:
                raw = _support_support_raw_cross(B_i, gm_i.bin_idx, B_j, gm_j.bin_idx, W)
                _profile_count(profile, "block_cross_disc_disc_rows_calls")
            else:
                W_2d = (
                    _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, n_i, n_j)
                    if cache is None
                    else cache.disc_disc_hist(gm_i.bin_idx, gm_j.bin_idx, W, n_i, n_j)
                )
                raw = B_i.T @ W_2d @ B_j
                _profile_count(profile, "block_cross_disc_disc_hist_calls")
            if R_i is not None:
                raw = R_i.T @ raw
            if R_j is not None:
                raw = raw @ R_j
            _profile_elapsed(profile, "block_cross_disc_disc_s", t0)
            return raw

    if isinstance(gm_i, DiscretizedSCOPGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_j, gm_i.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins, cache)
        result = gm_i.B_scop_unique.T @ WX_agg
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSCOPGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_i, gm_j.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins, cache)
        result = (gm_j.B_scop_unique.T @ WX_agg).T
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    # Disc × non-disc: batch aggregate by disc bins, then dense matmuls.
    # Avoids per-column rmatvec loop, toarray() for sparse groups, and
    # the (n, p) W-broadcast allocation.
    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_j, DiscretizedSSPGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_j, gm_i.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins, cache)
        support, transform = _cross_support(gm_i, cache, WX_agg)
        result = support.T @ WX_agg
        if transform is not None:
            result = transform.T @ result
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    if isinstance(gm_j, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_i, DiscretizedSSPGroupMatrix
    ):
        t0 = perf_counter() if profile is not None else 0.0
        if not _agg_by_bin_fits(gm_i, gm_j.n_bins):
            result = _cross_gram_by_columns(gm_i, gm_j, W)
            _profile_elapsed(profile, "block_cross_fallback_s", t0)
            return result
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins, cache)
        support, transform = _cross_support(gm_j, cache, WX_agg)
        result = support.T @ WX_agg
        if transform is not None:
            result = transform.T @ result
        result = result.T
        _profile_elapsed(profile, "block_cross_disc_other_s", t0)
        return result

    # Cat × cat: weighted crosstab — O(n) with no dense allocation.
    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, CategoricalGroupMatrix):
        t0 = perf_counter() if profile is not None else 0.0
        result = _cat_cat_weighted_crosstab(gm_i.codes, gm_j.codes, W, gm_i.n_levels, gm_j.n_levels)
        _profile_elapsed(profile, "block_cross_cat_cat_s", t0)
        return result

    # Restrict to the concrete built-ins: subclasses may override their public
    # matvec/rmatvec semantics independently of the stored matrix arrays.
    factor_dense = type(gm_i) is FactorSmoothGroupMatrix and type(gm_j) is DenseGroupMatrix
    dense_factor = type(gm_j) is FactorSmoothGroupMatrix and type(gm_i) is DenseGroupMatrix
    if factor_dense or dense_factor:
        t0 = perf_counter() if profile is not None else 0.0
        factor, dense = (gm_i, gm_j) if factor_dense else (gm_j, gm_i)
        result = _cross_gram_factor_smooth_dense(factor, dense, W)
        if result is not None:
            _profile_count(profile, "block_cross_factor_smooth_dense_calls")
            _profile_elapsed(profile, "block_cross_factor_smooth_dense_s", t0)
            return result if factor_dense else result.T

    if type(gm_i) is _SparseSSPGroupMatrix and type(gm_j) is _SparseSSPGroupMatrix:
        t0 = perf_counter() if profile is not None else 0.0
        result = _cross_gram_sparse_ssp(gm_i, gm_j, W, cache)
        if result is not None:
            _profile_count(profile, "block_cross_ssp_ssp_calls")
            _profile_elapsed(profile, "block_cross_ssp_ssp_s", t0)
            return result

    # Factored support-space groups must never be selected for the generic
    # observation-matrix materialization below. Generate the narrower side a
    # column at a time and retain O(n) working memory instead.
    support_space_types = (_SparseSSPGroupMatrix, FactorSmoothGroupMatrix, *SplineCatTypes)
    if isinstance(gm_i, support_space_types) or isinstance(gm_j, support_space_types):
        t0 = perf_counter() if profile is not None else 0.0
        local_cache = _BlockWeightCache() if cache is None else cache
        project_i, project_j = (
            type(gm) is _SparseSSPGroupMatrix
            and type(W) is np.ndarray
            and W.shape == (gm.shape[0],)
            and local_cache.sparse_gram(gm, W)[1]
            for gm in (gm_i, gm_j)
        )
        if project_i or project_j:
            result = _cross_gram_by_columns(gm_i, gm_j, W, project_i=project_i, project_j=project_j)
        else:
            result = _cross_gram_by_columns(gm_i, gm_j, W)
        _profile_elapsed(profile, "block_cross_fallback_s", t0)
        return result

    # Non-disc × non-disc: materialize smaller side, rmatvec larger side.
    t0 = perf_counter() if profile is not None else 0.0
    if gm_i.shape[1] <= gm_j.shape[1]:
        X_i = gm_i.toarray()
        WX_i = W[:, None] * X_i
        result = np.vstack([gm_j.rmatvec(WX_i[:, k]) for k in range(WX_i.shape[1])])
        _profile_elapsed(profile, "block_cross_fallback_s", t0)
        return result

    X_j = gm_j.toarray()
    WX_j = W[:, None] * X_j
    result = np.column_stack([gm_i.rmatvec(WX_j[:, k]) for k in range(WX_j.shape[1])])
    _profile_elapsed(profile, "block_cross_fallback_s", t0)
    return result


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X'diag(W)X for arbitrary-sign weights.

    SSP and Discretized groups handle any-sign W natively (they never use
    sqrt(W)).  Dense and Sparse groups use sqrt(W) internally, which fails
    for negative W, so we fall back to explicit W[:, None] * X for those.
    """
    (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        _DiscretizedTensorGroupMatrix,
        _SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    ) = _runtime_group_matrix_types()
    from ..group_matrix import FactorSmoothGroupMatrix

    if isinstance(
        gm,
        SparseSSPGroupMatrix
        | SplineCategoricalGroupMatrix
        | DiscretizedSplineCategoricalGroupMatrix
        | DiscretizedSSPGroupMatrix
        | DiscretizedSCOPGroupMatrix,
    ):
        return gm.gram(W)
    if isinstance(gm, FactorSmoothGroupMatrix):
        return gm.gram(W)
    if isinstance(gm, CategoricalGroupMatrix):
        return gm.gram(W)  # bincount-based diagonal, handles any-sign W
    X = gm.toarray()
    return (W[:, None] * X).T @ X


def _execution_plan_for_blocks(gms, groups, W: NDArray, tabmat_split):
    """Build the compatibility plan and verify legacy solver-column spans."""
    from ._group_matrix_execution import MatrixExecutionPlan

    plan = MatrixExecutionPlan(
        gms,
        n=len(W),
        ordinary_tabmat=tabmat_split is not None,
        prepared_ordinary_split=tabmat_split,
    )
    plan.validate_group_spans(groups)
    return plan


def _block_xtwx(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compatibility entry point for a weighted design Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=False, profile=profile).gram


def _block_xtwx_rhs(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    Wz: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compatibility entry point for a Gram, ``X'W``, and ``X'Wz``."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    moments = plan.moments(
        W,
        rhs=(Wz,),
        include_xtw=True,
        signed=False,
        profile=profile,
    )
    if moments.xtw is None:  # pragma: no cover - guaranteed by include_xtw
        raise RuntimeError("execution plan did not return X'W")
    return moments.gram, moments.xtw, moments.xt_rhs[0]


def _block_xtwx_signed(
    gms: list[GroupMatrix],
    groups: list,
    W: NDArray,
    *,
    tabmat_split=None,
    profile: dict[str, Any] | None = None,
) -> NDArray:
    """Compatibility entry point for an arbitrary-sign weighted Gram."""
    plan = _execution_plan_for_blocks(gms, groups, W, tabmat_split)
    return plan.moments(W, signed=True, profile=profile).gram
