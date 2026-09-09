"""Private discretized group-matrix class implementations."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _disc_disc_2d_hist,
    _fused_2d_bincount_2,
    _fused_bincount_2,
    _indexed_row_dot,
    _tensor_operand_in_reassociation_range,
)
from ._row_lookup import build_row_lookup


class DiscretizedSSPGroupMatrix:
    """Discretized SSP group matrix: stores dense B_unique + bin index.

    Instead of a sparse (n, K) basis matrix, stores a dense (n_bins, K) matrix
    evaluated at bin centers plus an (n,) index array mapping observations to bins.
    All operations aggregate weights by bin first, reducing O(n) matrix work
    to O(n_bins) + O(n) scatter/gather.
    """

    __slots__ = (
        "B_unique",
        "R_inv",
        "bin_idx",
        "n_bins",
        "shape",
        "omega",
        "projection",
        "omega_components",
        "component_types",
        "lambda_policies",
    )

    def __init__(self, B_unique: NDArray, R_inv: NDArray, bin_idx: NDArray):
        self.B_unique = np.asarray(B_unique)  # (n_bins, K)
        self.R_inv = np.asarray(R_inv)  # (K, p_g)
        self.bin_idx = np.asarray(bin_idx, dtype=np.intp)  # (n,)
        self.n_bins = self.B_unique.shape[0]
        self.shape = (len(bin_idx), self.R_inv.shape[1])
        self.omega = None  # (K, K) B-spline-space penalty, set externally
        self.projection = None  # (K, n_sub) projection matrix, set externally
        self.omega_components = None  # list[(suffix, omega)] for multi-penalty, set externally
        self.component_types = None  # dict[suffix, type] for multi-penalty, set externally
        self.lambda_policies = None  # dict[suffix, LambdaPolicy] for multi-penalty, set externally

    def matvec(self, v: NDArray) -> NDArray:
        # B_unique @ (R_inv @ v) is (n_bins,), scatter to (n,)
        vals = self.B_unique @ (self.R_inv @ v)
        return vals[self.bin_idx]

    def rmatvec(self, w: NDArray) -> NDArray:
        # Aggregate w by bin, then dense rmatvec
        w_agg = np.bincount(self.bin_idx, weights=w, minlength=self.n_bins)
        return self.R_inv.T @ (self.B_unique.T @ w_agg)

    def gram(self, W: NDArray) -> NDArray:
        # Aggregate W by bin, then dense gram, then sandwich with R_inv
        W_agg = np.bincount(self.bin_idx, weights=W, minlength=self.n_bins)
        BtWB = self.B_unique.T @ (self.B_unique * W_agg[:, None])
        return self.R_inv.T @ BtWB @ self.R_inv

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Compute gram(W), rmatvec(W), rmatvec(Wz) with shared bincount.

        Returns (gram, XtW, XtWz) — single O(n) pass for both aggregations.
        """
        W_agg, Wz_agg = _fused_bincount_2(self.bin_idx, W, Wz, self.n_bins)
        BtW_agg = self.B_unique.T @ W_agg  # (K,)
        BtWz_agg = self.B_unique.T @ Wz_agg  # (K,)
        BtWB = self.B_unique.T @ (self.B_unique * W_agg[:, None])  # (K, K)
        gram = self.R_inv.T @ BtWB @ self.R_inv
        xtw = self.R_inv.T @ BtW_agg
        xtwz = self.R_inv.T @ BtWz_agg
        return gram, xtw, xtwz

    def toarray(self) -> NDArray:
        return (self.B_unique @ self.R_inv)[self.bin_idx]

    def row_subset(self, idx: NDArray) -> DiscretizedSSPGroupMatrix:
        # type(self) rather than the parent: a subclass carries semantics the
        # parent does not (lossless support vs binning), and subsetting must not
        # silently downcast it.
        sub = type(self)(self.B_unique, self.R_inv, self.bin_idx[idx])
        sub.omega = self.omega
        sub.projection = self.projection
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        return sub


class DiscretizedSCOPGroupMatrix:
    """Discretized SCOP group matrix: bin-level centered SCOP design.

    Stores the centered SCOP design matrix evaluated at bin centers ``(n_bins, q_eff)``
    plus a bin-index array ``(n,)`` mapping observations to bins.  SCOP terms bypass
    SSP reparametrisation, so there is no ``R_inv`` — the columns are already in
    solver space (the centered B @ Sigma block with column 0 dropped).

    Operations follow the same scatter/gather pattern as DiscretizedSSPGroupMatrix
    but without the R_inv sandwich.
    """

    __slots__ = (
        "B_scop_unique",
        "bin_idx",
        "n_bins",
        "shape",
    )

    def __init__(self, B_scop_unique: NDArray, bin_idx: NDArray):
        self.B_scop_unique = np.asarray(B_scop_unique)  # (n_bins, q_eff)
        self.bin_idx = np.asarray(bin_idx, dtype=np.intp)  # (n,)
        self.n_bins = self.B_scop_unique.shape[0]
        self.shape = (len(bin_idx), self.B_scop_unique.shape[1])

    def matvec(self, v: NDArray) -> NDArray:
        vals = self.B_scop_unique @ v  # (n_bins,)
        return vals[self.bin_idx]

    def rmatvec(self, w: NDArray) -> NDArray:
        w_agg = np.bincount(self.bin_idx, weights=w, minlength=self.n_bins)
        return self.B_scop_unique.T @ w_agg

    def gram(self, W: NDArray) -> NDArray:
        W_agg = np.bincount(self.bin_idx, weights=W, minlength=self.n_bins)
        return self.B_scop_unique.T @ (self.B_scop_unique * W_agg[:, None])

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        W_agg, Wz_agg = _fused_bincount_2(self.bin_idx, W, Wz, self.n_bins)
        BtW_agg = self.B_scop_unique.T @ W_agg
        BtWz_agg = self.B_scop_unique.T @ Wz_agg
        BtWB = self.B_scop_unique.T @ (self.B_scop_unique * W_agg[:, None])
        return BtWB, BtW_agg, BtWz_agg

    def toarray(self) -> NDArray:
        return self.B_scop_unique[self.bin_idx]

    def row_subset(self, idx: NDArray) -> DiscretizedSCOPGroupMatrix:
        return DiscretizedSCOPGroupMatrix(self.B_scop_unique, self.bin_idx[idx])


class DiscretizedSplineCategoricalGroupMatrix:
    """One spline-by-category level using compressed spline support.

    The effective full matrix is zero outside ``row_idx`` and equals
    ``B_unique[bin_idx_level] @ R_inv`` on rows in the category level.  This
    keeps fit algebra on the spline support grid instead of the observation
    row subset used by :class:`SplineCategoricalGroupMatrix`.
    """

    __slots__ = (
        "B_unique",
        "R_inv",
        "bin_idx_level",
        "row_idx",
        "_row_order",
        "_sorted_rows",
        "_row_lookup_certificate",
        "n_bins",
        "n_rows",
        "shape",
        "_p_b",
        "omega",
        "projection",
        "omega_components",
        "component_types",
        "lambda_policies",
        "spline_cat_level",
        "spline_cat_feature",
    )

    def __init__(
        self,
        B_unique: NDArray,
        R_inv: NDArray,
        bin_idx: NDArray,
        row_idx: NDArray,
        *,
        n_rows: int | None = None,
        bin_idx_is_level: bool = False,
    ):
        self.B_unique = np.asarray(B_unique, dtype=np.float64)
        self.R_inv = np.asarray(R_inv, dtype=np.float64)
        # Own the category indices so a caller cannot invalidate the cached
        # row lookup by mutating the constructor argument.
        self.row_idx = np.array(row_idx, dtype=np.intp, copy=True)
        self.row_idx.flags.writeable = False
        self._row_order = None
        self._sorted_rows = None
        self._row_lookup_certificate = None
        bin_idx_arr = np.asarray(bin_idx, dtype=np.intp)
        self.bin_idx_level = (
            bin_idx_arr if bin_idx_is_level else bin_idx_arr[self.row_idx]
        ).astype(np.intp, copy=False)
        self.n_bins = self.B_unique.shape[0]
        if n_rows is not None:
            self.n_rows = int(n_rows)
        elif bin_idx_is_level:
            self.n_rows = int(self.row_idx.max()) + 1 if self.row_idx.size else 0
        else:
            self.n_rows = len(bin_idx_arr)
        self.shape = (self.n_rows, self.R_inv.shape[1])
        self._p_b = self.B_unique.shape[1]
        self.omega = None
        self.projection = None
        self.omega_components = None
        self.component_types = None
        self.lambda_policies = None
        self.spline_cat_level = None
        self.spline_cat_feature = None

    def __getstate__(self):
        dict_state, slot_state = cast(
            tuple[dict[str, object] | None, dict[str, object]], object.__getstate__(self)
        )
        slot_state.pop("_row_order", None)
        slot_state.pop("_sorted_rows", None)
        slot_state.pop("_row_lookup_certificate", None)
        return dict_state, slot_state

    def __setstate__(self, state):
        # Older learned matrices have no lookup slots. Rebuild lazily after
        # restoring owned indices; NumPy pickle does not retain readonly flags.
        dict_state, slot_state = state
        if dict_state is not None:
            self.__dict__.update(dict_state)
        for name, value in slot_state.items():
            setattr(self, name, value)
        self.row_idx = np.array(self.row_idx, dtype=np.intp, copy=True)
        self.row_idx.flags.writeable = False
        self._row_order = None
        self._sorted_rows = None
        self._row_lookup_certificate = None

    def matvec(self, v: NDArray) -> NDArray:
        out = np.zeros(self.n_rows, dtype=np.float64)
        if self.row_idx.size:
            vals = self.B_unique @ (self.R_inv @ v)
            out[self.row_idx] = vals[self.bin_idx_level]
        return out

    def rmatvec(self, w: NDArray) -> NDArray:
        if self.row_idx.size:
            w_agg = np.bincount(
                self.bin_idx_level,
                weights=w[self.row_idx],
                minlength=self.n_bins,
            )
        else:
            w_agg = np.zeros(self.n_bins, dtype=np.float64)
        return self.R_inv.T @ (self.B_unique.T @ w_agg)

    def gram(self, W: NDArray) -> NDArray:
        if self.row_idx.size:
            W_agg = np.bincount(
                self.bin_idx_level,
                weights=W[self.row_idx],
                minlength=self.n_bins,
            )
        else:
            W_agg = np.zeros(self.n_bins, dtype=np.float64)
        BtWB = self.B_unique.T @ (self.B_unique * W_agg[:, None])
        return self.R_inv.T @ BtWB @ self.R_inv

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        if self.row_idx.size:
            W_agg, Wz_agg = _fused_bincount_2(
                self.bin_idx_level,
                W[self.row_idx],
                Wz[self.row_idx],
                self.n_bins,
            )
        else:
            W_agg = np.zeros(self.n_bins, dtype=np.float64)
            Wz_agg = np.zeros(self.n_bins, dtype=np.float64)
        BtW_agg = self.B_unique.T @ W_agg
        BtWz_agg = self.B_unique.T @ Wz_agg
        BtWB = self.B_unique.T @ (self.B_unique * W_agg[:, None])
        gram = self.R_inv.T @ BtWB @ self.R_inv
        xtw = self.R_inv.T @ BtW_agg
        xtwz = self.R_inv.T @ BtWz_agg
        return gram, xtw, xtwz

    def toarray(self) -> NDArray:
        out = np.zeros(self.shape, dtype=np.float64)
        if self.row_idx.size:
            out[self.row_idx] = (self.B_unique @ self.R_inv)[self.bin_idx_level]
        return out

    def row_subset(self, idx: NDArray) -> DiscretizedSplineCategoricalGroupMatrix:
        idx_raw = np.asarray(idx)
        if np.issubdtype(idx_raw.dtype, np.bool_):
            idx_arr = np.flatnonzero(idx_raw).astype(np.intp, copy=False)
        else:
            idx_arr = idx_raw.astype(np.intp, copy=False)
        if self.row_idx.size and idx_arr.size:
            # Chunked fits revisit this parent many times. Sort once, retaining
            # the original level order used by bin_idx_level and its algebra.
            if self._sorted_rows is None:
                self._sorted_rows, self._row_order, self._row_lookup_certificate = build_row_lookup(
                    self.row_idx, with_order=True
                )
            pos = np.searchsorted(self._sorted_rows, idx_arr)
            in_bounds = pos < self._sorted_rows.size
            matched = np.zeros(idx_arr.size, dtype=bool)
            matched[in_bounds] = self._sorted_rows[pos[in_bounds]] == idx_arr[in_bounds]
            pos_sub = np.flatnonzero(matched).astype(np.intp, copy=False)
            pos_self = cast(NDArray[np.intp], self._row_order)[pos[matched]]
            bin_idx_level = self.bin_idx_level[pos_self]
        else:
            pos_sub = np.empty(0, dtype=np.intp)
            bin_idx_level = np.empty(0, dtype=np.intp)
        # type(self) rather than the parent: a subclass carries semantics the
        # parent does not (lossless support vs binning), and subsetting must not
        # silently downcast it.
        sub = type(self)(
            self.B_unique,
            self.R_inv,
            bin_idx_level,
            pos_sub,
            n_rows=len(idx_arr),
            bin_idx_is_level=True,
        )
        sub.omega = self.omega
        sub.projection = self.projection
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        sub.lambda_policies = self.lambda_policies
        sub.spline_cat_level = self.spline_cat_level
        sub.spline_cat_feature = self.spline_cat_feature
        return sub


class DiscretizedTensorGroupMatrix(DiscretizedSSPGroupMatrix):
    """Discretized tensor interaction with factored Kronecker structure.

    Like DiscretizedSSPGroupMatrix but stores the factored marginal bases
    (B1_unique, B2_unique) and marginal bin indices (idx1, idx2) instead
    of only the materialized Kronecker product B_joint.  Gram, matvec,
    and rmatvec operations exploit the product structure for O(n_bins1 *
    K1^2 * K2^2) instead of O(n_pairs * (K1*K2)^2).

    The materialized B_joint is still kept as ``self.B_unique`` (inherited)
    for fallback compatibility in any code path that doesn't know about
    the factored representation.
    """

    __slots__ = (
        "B1_unique_t",
        "B2_unique_t",
        "idx1",
        "idx2",
        "n_bins1",
        "n_bins2",
        "tensor_id",
        "_own_margin_cache",
    )

    def __init__(
        self,
        B1_unique: NDArray,
        B2_unique: NDArray,
        idx1: NDArray,
        idx2: NDArray,
        B_joint: NDArray,
        R_inv: NDArray,
        pair_idx: NDArray,
        tensor_id: int,
    ):
        super().__init__(B_joint, R_inv, pair_idx)
        self.B1_unique_t = np.asarray(B1_unique)
        self.B2_unique_t = np.asarray(B2_unique)
        self.idx1 = np.asarray(idx1, dtype=np.intp)
        self.idx2 = np.asarray(idx2, dtype=np.intp)
        self.n_bins1 = self.B1_unique_t.shape[0]
        self.n_bins2 = self.B2_unique_t.shape[0]
        self.tensor_id = tensor_id
        self._own_margin_cache: dict[tuple[int, int, int], int | None] = {}

    def _factored_gram_raw(self, w_grid: NDArray) -> NDArray:
        """Compute the raw tensor Gram, ordered by j1 * K2 + j2.

        Contract the stored marginal row outer products. The usual route uses
        two ordinary GEMMs without a bin-grid-by-basis weighted workspace.
        """
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        K1, K2 = B1.shape[1], B2.shape[1]
        n1, n2 = B1.shape[0], B2.shape[0]
        # B1's outer table and the raw coefficient Gram occur in both routes.
        # Retain the old contraction when the second outer table would cost
        # more workspace, as can happen with a wide second marginal basis.
        left_cells = n2 * K1 * K1
        right_cells = n1 * K2 * K2
        outer_cells = n2 * K2 * K2
        old_cells = n1 * n2 * K2 + right_cells
        B1_outer = (B1[:, :, None] * B1[:, None, :]).reshape(n1, K1 * K1)
        # Forming B2's outer product before weighting changes integer/float32
        # promotion. Keep the original arithmetic for other operand dtypes.
        float64_inputs = B1.dtype == B2.dtype == w_grid.dtype == np.float64
        # Five original factors contribute to each raw Gram term. Bounding
        # nonzero magnitudes by 2**(+/-128) leaves exponent headroom even for
        # two sums with 64-bit index-sized dimensions (640 + 126 < 1024).
        # This prevents new range failures from reassociation; cancellation
        # still follows ordinary floating-point arithmetic. Extreme inputs
        # retain the previous weighting order, including its finite results.
        reassociate = (
            float64_inputs
            and outer_cells + min(left_cells, right_cells) <= old_cells
            and _tensor_operand_in_reassociation_range(B1)
            and _tensor_operand_in_reassociation_range(B2)
            and _tensor_operand_in_reassociation_range(w_grid)
        )
        if reassociate:
            # Marginal tables can be mutable aliases. Recompute these small
            # products instead of retaining a cache that could become stale.
            B2_outer = (B2[:, :, None] * B2[:, None, :]).reshape(n2, K2 * K2)
            if right_cells <= left_cells:
                G_K1K1_K2K2 = B1_outer.T @ (w_grid @ B2_outer)
            else:
                G_K1K1_K2K2 = (B1_outer.T @ w_grid) @ B2_outer
        else:
            WB2 = w_grid[:, :, None] * B2[None, :, :]
            C = WB2.transpose(0, 2, 1) @ B2[None, :, :]
            G_K1K1_K2K2 = B1_outer.T @ C.reshape(n1, K2 * K2)
        return G_K1K1_K2K2.reshape(K1, K1, K2, K2).transpose(0, 2, 1, 3).reshape(K1 * K2, K1 * K2)

    def gram(self, W: NDArray) -> NDArray:
        w_grid = _disc_disc_2d_hist(self.idx1, self.idx2, W, self.n_bins1, self.n_bins2)
        G_raw = self._factored_gram_raw(w_grid)
        return self.R_inv.T @ G_raw @ self.R_inv

    def gram_rmatvec_from_grids(
        self, w_grid: NDArray, wz_grid: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Factored gram + rmatvec from precomputed tensor weight grids."""
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        G_raw = self._factored_gram_raw(w_grid)
        gram = self.R_inv.T @ G_raw @ self.R_inv
        xtw = self.R_inv.T @ (B1.T @ w_grid @ B2).ravel()
        xtwz = self.R_inv.T @ (B1.T @ wz_grid @ B2).ravel()
        return gram, xtw, xtwz

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Factored gram + rmatvec with shared 2D bincount."""
        w_grid, wz_grid = _fused_2d_bincount_2(
            self.idx1, self.idx2, W, Wz, self.n_bins1, self.n_bins2
        )
        return self.gram_rmatvec_from_grids(w_grid, wz_grid)

    def matvec(self, v: NDArray) -> NDArray:
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        n_pairs = self.B_unique.shape[0]
        n_obs = self.shape[0]
        p_g = self.shape[1]
        K1, K2 = B1.shape[1], B2.shape[1]

        # When the observed tensor support is small, evaluating on the unique
        # support pairs and scattering back to observations is much cheaper
        # than building an (n, K2) temporary via the factored observation path.
        direct_pair_cost = n_pairs * p_g
        factored_obs_cost = B1.shape[0] * p_g + n_obs * K2
        if direct_pair_cost <= factored_obs_cost:
            vals = self.B_unique @ (self.R_inv @ v)
            return vals[self.bin_idx]

        u = (self.R_inv @ v).reshape(K1, K2)
        B1u = B1 @ u  # (n_bins1, K2)
        # Sequential accumulation in the fused kernel has a different range
        # from NumPy's pairwise sum. Preserve the old sum for extreme operands.
        if (
            B1u.dtype == B2.dtype == np.float64
            and _tensor_operand_in_reassociation_range(B1u)
            and _tensor_operand_in_reassociation_range(B2)
        ):
            return _indexed_row_dot(B1u, B2, self.idx1, self.idx2)
        return np.sum(B1u[self.idx1] * B2[self.idx2], axis=1)

    def rmatvec(self, w: NDArray) -> NDArray:
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        w_grid = _disc_disc_2d_hist(self.idx1, self.idx2, w, self.n_bins1, self.n_bins2)
        return self.R_inv.T @ (B1.T @ w_grid @ B2).ravel()

    def row_subset(self, idx: NDArray) -> DiscretizedTensorGroupMatrix:
        sub = DiscretizedTensorGroupMatrix(
            self.B1_unique_t,
            self.B2_unique_t,
            self.idx1[idx],
            self.idx2[idx],
            self.B_unique,  # B_joint stays the same (covers all pairs)
            self.R_inv,
            self.bin_idx[idx],
            tensor_id=self.tensor_id,
        )
        sub.omega = self.omega
        sub.projection = self.projection
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        return sub


class SupportCompressedSplineCategoricalGroupMatrix(DiscretizedSplineCategoricalGroupMatrix):
    """Exact ``spline_cat`` basis stored one row per distinct row.

    Numerically identical to :class:`SplineCategoricalGroupMatrix` over the same
    basis; only the storage differs.  Kept distinct from its binned parent for
    the same reason :class:`SupportCompressedSSPGroupMatrix` is: ``discrete=True``
    is a lossy fREML path and this is not, so ``bin_idx_level`` here indexes
    exact distinct rows of the shared spline basis rather than bins.

    The support is shared across the levels of the categorical parent -- one
    dense ``(n_support, p_b)`` block for the whole term -- while each level
    keeps its own row index into it.
    """

    __slots__ = ()

    @property
    def is_lossless_support(self) -> bool:
        return True


class SupportCompressedSSPGroupMatrix(DiscretizedSSPGroupMatrix):
    """Exact SSP basis stored one row per distinct row.

    Numerically identical to :class:`SparseSSPGroupMatrix` over the same basis;
    only the storage differs.  Kept distinct from its binned parent because
    ``discrete=True`` is a lossy fREML path and this is not: no binning occurs
    and no discretisation error is introduced, so ``bin_idx`` here indexes exact
    distinct rows rather than bins.
    """

    __slots__ = ()

    @property
    def is_lossless_support(self) -> bool:
        return True
