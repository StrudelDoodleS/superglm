"""Per-group matrix wrappers for sparse/dense BCD operations.

Four wrapper types with the same interface:
- DenseGroupMatrix: numeric features (single column) or dense fallback
- SparseGroupMatrix: categoricals, non-SSP splines
- SparseSSPGroupMatrix: SSP splines (factored: sparse B + dense R_inv)
- DiscretizedSSPGroupMatrix: discretized SSP splines (binned B_unique + index)

DesignMatrix holds the list and provides full-matrix matvec/rmatvec.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import njit
from numpy.typing import NDArray


@njit(cache=True)
def _csr_weighted_gram(data, indices, indptr, W, p):
    """B.T @ diag(W) @ B exploiting CSR sparsity (symmetric accumulation)."""
    result = np.zeros((p, p))
    n = len(W)
    for i in range(n):
        w = W[i]
        start = indptr[i]
        end = indptr[i + 1]
        for a in range(start, end):
            ja = indices[a]
            va = data[a] * w
            for b in range(a, end):
                jb = indices[b]
                prod = va * data[b]
                result[ja, jb] += prod
                if a != b:
                    result[jb, ja] += prod
    return result


@njit(cache=True)
def _weighted_bincount_2d(bin_idx, W, M, n_bins):
    """Fused W-weighted multi-column bincount for dense M.

    result[b, c] = sum W[i] * M[i, c] for all i where bin_idx[i] == b.
    Avoids the (n, p) intermediate allocation of W[:, None] * M.
    """
    n = len(bin_idx)
    n_cols = M.shape[1]
    result = np.zeros((n_bins, n_cols))
    for i in range(n):
        b = bin_idx[i]
        w = W[i]
        for c in range(n_cols):
            result[b, c] += w * M[i, c]
    return result


@njit(cache=True)
def _csr_weighted_bincount(data, indices, indptr, n_cols, bin_idx, W, n_bins):
    """Fused CSR-aware W-weighted bincount — avoids materialising sparse to dense.

    result[b, c] = sum W[row] * X_sparse[row, c]  for rows where bin_idx[row] == b.
    For one-hot categoricals (1 non-zero/row) this is O(n), not O(n * n_cols).
    """
    n = len(bin_idx)
    result = np.zeros((n_bins, n_cols))
    for row in range(n):
        b = bin_idx[row]
        w = W[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            col = indices[ptr]
            result[b, col] += w * data[ptr]
    return result


@njit(cache=True)
def _disc_disc_2d_hist(bin_idx_i, bin_idx_j, W, n_bins_i, n_bins_j):
    """Fused 2D histogram for disc-disc cross-gram.

    Replaces ``joint_idx = bin_i * n_bins_j + bin_j`` followed by
    ``np.bincount(joint_idx, W).reshape(n_bins_i, n_bins_j)``.
    Avoids two (n,) intermediate allocations per call.
    """
    n = len(W)
    result = np.zeros((n_bins_i, n_bins_j))
    for obs in range(n):
        result[bin_idx_i[obs], bin_idx_j[obs]] += W[obs]
    return result


@njit(cache=True)
def _fused_bincount_2(bin_idx, W, Wz, n_bins):
    """Fused dual bincount: aggregate W and Wz by bin in one O(n) pass."""
    n = len(bin_idx)
    W_agg = np.zeros(n_bins)
    Wz_agg = np.zeros(n_bins)
    for i in range(n):
        b = bin_idx[i]
        W_agg[b] += W[i]
        Wz_agg[b] += Wz[i]
    return W_agg, Wz_agg


@njit(cache=True)
def _fused_2d_bincount_2(idx1, idx2, W, Wz, n_bins1, n_bins2):
    """Fused dual 2D bincount for tensor gram_rmatvec.

    Aggregates both W and Wz by (idx1, idx2) marginal bins in a single O(n) pass.
    Returns (W_grid, Wz_grid), each (n_bins1, n_bins2).
    """
    n = len(idx1)
    W_grid = np.zeros((n_bins1, n_bins2))
    Wz_grid = np.zeros((n_bins1, n_bins2))
    for i in range(n):
        a = idx1[i]
        b = idx2[i]
        W_grid[a, b] += W[i]
        Wz_grid[a, b] += Wz[i]
    return W_grid, Wz_grid


@njit(cache=True)
def _cat_weighted_bincount(codes, bin_idx, W, n_bins, n_levels):
    """Scatter W into (n_bins, n_levels) by (bin_idx, codes) simultaneously.

    Base-level observations have codes == n_levels (sink bin) and are skipped.
    """
    result = np.zeros((n_bins, n_levels))
    for i in range(len(codes)):
        c = codes[i]
        if c < n_levels:
            result[bin_idx[i], c] += W[i]
    return result


@njit(cache=True)
def _cat_cat_weighted_crosstab(codes_i, codes_j, W, n_levels_i, n_levels_j):
    """Weighted crosstab: X_i.T @ diag(W) @ X_j for two categoricals.

    Sink-bin observations (codes == n_levels) are excluded from both sides.
    Result is (n_levels_i, n_levels_j).
    """
    result = np.zeros((n_levels_i, n_levels_j))
    for k in range(len(W)):
        ci = codes_i[k]
        cj = codes_j[k]
        if ci < n_levels_i and cj < n_levels_j:
            result[ci, cj] += W[k]
    return result


class DenseGroupMatrix:
    """Dense group matrix wrapper."""

    __slots__ = ("M", "shape")

    def __init__(self, M: NDArray):
        self.M = np.asarray(M)
        if self.M.ndim == 1:
            self.M = self.M[:, None]
        self.shape = self.M.shape

    def matvec(self, v: NDArray) -> NDArray:
        return self.M @ v

    def rmatvec(self, w: NDArray) -> NDArray:
        return self.M.T @ w

    def gram(self, W: NDArray) -> NDArray:
        Mw = self.M * np.sqrt(W)[:, None]
        return Mw.T @ Mw

    def toarray(self) -> NDArray:
        return self.M

    def row_subset(self, idx: NDArray) -> DenseGroupMatrix:
        return DenseGroupMatrix(self.M[idx])


class SparseGroupMatrix:
    """Sparse CSR group matrix wrapper."""

    __slots__ = ("M", "shape")

    def __init__(self, M: sp.spmatrix):
        self.M = sp.csr_matrix(M)
        self.shape = self.M.shape

    def matvec(self, v: NDArray) -> NDArray:
        return np.asarray(self.M @ v).ravel()

    def rmatvec(self, w: NDArray) -> NDArray:
        return np.asarray(self.M.T @ w).ravel()

    def gram(self, W: NDArray) -> NDArray:
        sqrtW = np.sqrt(W)
        Mw = self.M.multiply(sqrtW[:, None])
        return np.asarray((Mw.T @ Mw).todense())

    def toarray(self) -> NDArray:
        return self.M.toarray()

    def row_subset(self, idx: NDArray) -> SparseGroupMatrix:
        return SparseGroupMatrix(self.M[idx])


class CategoricalGroupMatrix:
    """One-hot categorical stored as integer codes — no scipy overhead.

    For a categorical with K non-base levels and n observations, stores
    codes (n,) with values in {0, ..., K} where K is the "sink" bin for
    the base level (absorbed into intercept).  All operations use a full
    bincount of K+1 bins and discard the last bin — no boolean masking.
    """

    __slots__ = ("codes", "n_levels", "shape")

    def __init__(self, codes: NDArray, n_levels: int):
        # Remap -1 (base level) → n_levels so bincount/indexing is mask-free
        c = np.asarray(codes, dtype=np.intp)
        c = np.where(c == -1, n_levels, c)
        self.codes = c
        self.n_levels = n_levels
        self.shape = (len(codes), n_levels)

    def matvec(self, v: NDArray) -> NDArray:
        """X @ v: scatter v[codes] to observations, base level → 0."""
        # Pad v with 0 for the sink bin, then pure fancy-index
        v_ext = np.empty(self.n_levels + 1)
        v_ext[: self.n_levels] = v
        v_ext[self.n_levels] = 0.0
        return v_ext[self.codes]

    def rmatvec(self, w: NDArray) -> NDArray:
        """X.T @ w: aggregate w by level via bincount, discard sink bin."""
        return np.bincount(self.codes, weights=w, minlength=self.n_levels + 1)[: self.n_levels]

    def gram(self, W: NDArray) -> NDArray:
        """X.T @ diag(W) @ X: diagonal for one-hot encoding."""
        diag = np.bincount(self.codes, weights=W, minlength=self.n_levels + 1)[: self.n_levels]
        return np.diag(diag)

    def toarray(self) -> NDArray:
        """Materialize to dense (n, K) one-hot matrix."""
        out = np.zeros(self.shape, dtype=np.float64)
        mask = self.codes < self.n_levels
        out[np.where(mask)[0], self.codes[mask]] = 1.0
        return out

    def row_subset(self, idx: NDArray) -> CategoricalGroupMatrix:
        # Must pass original -1-coded form to __init__ for re-remapping
        c = self.codes[idx].copy()
        c[c == self.n_levels] = -1
        return CategoricalGroupMatrix(c, self.n_levels)


class SparseSSPGroupMatrix:
    """Factored SSP group matrix: stores sparse B + dense R_inv separately.

    Effective matrix is B @ R_inv, but we never form it explicitly.
    """

    __slots__ = (
        "B",
        "_data",
        "_indices",
        "_indptr",
        "_p_b",
        "R_inv",
        "shape",
        "omega",
        "projection",
    )

    def __init__(self, B_csr: sp.spmatrix, R_inv: NDArray):
        self.B = sp.csr_matrix(B_csr)
        self._data = self.B.data.astype(np.float64)
        self._indices = self.B.indices
        self._indptr = self.B.indptr
        self._p_b = self.B.shape[1]
        self.R_inv = np.asarray(R_inv)
        self.shape = (self.B.shape[0], self.R_inv.shape[1])
        self.omega = None  # (K, K) B-spline-space penalty, set externally
        self.projection = None  # (K, n_sub) projection matrix, set externally

    def matvec(self, v: NDArray) -> NDArray:
        # B @ (R_inv @ v): tiny dense first, then sparse matvec
        return np.asarray(self.B @ (self.R_inv @ v)).ravel()

    def rmatvec(self, w: NDArray) -> NDArray:
        # R_inv.T @ (B.T @ w): sparse rmatvec, then tiny dense
        return self.R_inv.T @ np.asarray(self.B.T @ w).ravel()

    def gram(self, W: NDArray) -> NDArray:
        # R_inv.T @ (B.T @ diag(W) @ B) @ R_inv: numba CSR gram
        raw_gram = _csr_weighted_gram(self._data, self._indices, self._indptr, W, self._p_b)
        return self.R_inv.T @ raw_gram @ self.R_inv

    def toarray(self) -> NDArray:
        return np.asarray(self.B @ self.R_inv)

    def row_subset(self, idx: NDArray) -> SparseSSPGroupMatrix:
        sub = SparseSSPGroupMatrix(self.B[idx], self.R_inv)
        sub.omega = self.omega
        sub.projection = self.projection
        return sub


def _discretize_column(x: NDArray, n_bins: int = 256) -> tuple[NDArray, NDArray]:
    """Compress a continuous variable to exact support or equal-width bins.

    If the observed support has ``<= n_bins`` unique values, this returns the
    sorted unique values and an exact inverse index. Otherwise it falls back to
    equal-width bin centers across the observed range.

    Returns ``(bin_centers, bin_idx)`` where ``bin_idx[i]`` is the support/bin
    index for ``x[i]``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    unique_vals, inverse = np.unique(x, return_inverse=True)
    if len(unique_vals) <= n_bins:
        return unique_vals.astype(np.float64), inverse.astype(np.intp)

    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        return np.array([lo]), np.zeros(len(x), dtype=np.intp)
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, bin_idx


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
    )

    def __init__(self, B_unique: NDArray, R_inv: NDArray, bin_idx: NDArray):
        self.B_unique = np.asarray(B_unique)  # (n_bins, K)
        self.R_inv = np.asarray(R_inv)  # (K, p_g)
        self.bin_idx = np.asarray(bin_idx, dtype=np.intp)  # (n,)
        self.n_bins = self.B_unique.shape[0]
        self.shape = (len(bin_idx), self.R_inv.shape[1])
        self.omega = None  # (K, K) B-spline-space penalty, set externally
        self.projection = None  # (K, n_sub) projection matrix, set externally

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
        sub = DiscretizedSSPGroupMatrix(self.B_unique, self.R_inv, self.bin_idx[idx])
        sub.omega = self.omega
        sub.projection = self.projection
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

    def _factored_gram_raw(self, w_grid: NDArray) -> NDArray:
        """Compute B_joint.T @ diag(w) @ B_joint via Kronecker factorization.

        Given w_grid (n_bins1, n_bins2) = 2D weight histogram on marginal bins,
        returns the raw (K1*K2, K1*K2) gram matrix in the centered marginal space.

        Column ordering: j1 * K2 + j2, matching _row_kron_dense().
        """
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        K1, K2 = B1.shape[1], B2.shape[1]
        n1 = B1.shape[0]

        # Step 1: C[a, m, n] = sum_b w_grid[a,b] * B2[b,m] * B2[b,n]
        #   = (B2.T @ diag(w_grid[a,:]) @ B2) per a — use batch BLAS
        WB2 = w_grid[:, :, None] * B2[None, :, :]  # (n1, n2, K2)
        C = WB2.transpose(0, 2, 1) @ B2[None, :, :]  # batch: (n1, K2, K2)

        # Step 2: G[(j1,j3), (j2,j4)] = sum_a B1[a,j1]*B1[a,j3] * C[a,j2,j4]
        #   = B1_outer_flat.T @ C_flat — single BLAS gemm
        B1_outer = B1[:, :, None] * B1[:, None, :]  # (n1, K1, K1)
        G_K1K1_K2K2 = B1_outer.reshape(n1, K1 * K1).T @ C.reshape(n1, K2 * K2)

        # Reindex: G_K1K1_K2K2[j1*K1+j3, j2*K2+j4] → G[j1*K2+j2, j3*K2+j4]
        return G_K1K1_K2K2.reshape(K1, K1, K2, K2).transpose(0, 2, 1, 3).reshape(K1 * K2, K1 * K2)

    def gram(self, W: NDArray) -> NDArray:
        w_grid = _disc_disc_2d_hist(self.idx1, self.idx2, W, self.n_bins1, self.n_bins2)
        G_raw = self._factored_gram_raw(w_grid)
        return self.R_inv.T @ G_raw @ self.R_inv

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Factored gram + rmatvec with shared 2D bincount."""
        B1, B2 = self.B1_unique_t, self.B2_unique_t
        w_grid, wz_grid = _fused_2d_bincount_2(
            self.idx1, self.idx2, W, Wz, self.n_bins1, self.n_bins2
        )
        # Factored gram
        G_raw = self._factored_gram_raw(w_grid)
        gram = self.R_inv.T @ G_raw @ self.R_inv
        # Factored rmatvec(W)
        xtw = self.R_inv.T @ (B1.T @ w_grid @ B2).ravel()
        # Factored rmatvec(Wz)
        xtwz = self.R_inv.T @ (B1.T @ wz_grid @ B2).ravel()
        return gram, xtw, xtwz

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
        return sub


GroupMatrix = (
    DenseGroupMatrix
    | SparseGroupMatrix
    | CategoricalGroupMatrix
    | SparseSSPGroupMatrix
    | DiscretizedSSPGroupMatrix
    | DiscretizedTensorGroupMatrix
)

_MAX_DISC_DISC_HIST_CELLS = 5_000_000


def _agg_by_bin(gm: GroupMatrix, bin_idx: NDArray, W: NDArray, n_bins: int) -> NDArray:
    """Aggregate W * gm's columns by bin index → (n_bins, p_g) dense array.

    Dispatches to the most efficient kernel for each GroupMatrix type:
    - SparseGroupMatrix: CSR-aware kernel (avoids toarray, O(nnz) not O(n*p))
    - SparseSSPGroupMatrix: CSR kernel in B-spline space + R_inv transform
    - DenseGroupMatrix / other: fused dense kernel (avoids W-broadcast alloc)
    """
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
        B_agg = _csr_weighted_bincount(
            gm._data, gm._indices, gm._indptr, gm._p_b, bin_idx, W, n_bins
        )
        return B_agg @ gm.R_inv
    X = gm.toarray()
    return _weighted_bincount_2d(bin_idx, W, X, n_bins)


def _cross_gram_tensor_tensor(
    gm_i: DiscretizedTensorGroupMatrix,
    gm_j: DiscretizedTensorGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Cross-gram between two tensor groups sharing the same marginals.

    Used for decomposed tensor subgroups (bilinear × wiggly) that share
    the same B1_unique, B2_unique, idx1, idx2 but have different R_inv.
    """
    w_grid = _disc_disc_2d_hist(gm_i.idx1, gm_i.idx2, W, gm_i.n_bins1, gm_i.n_bins2)
    G_raw = gm_i._factored_gram_raw(w_grid)
    return gm_i.R_inv.T @ G_raw @ gm_j.R_inv


def _cross_gram_tensor_main(
    gm_tensor: DiscretizedTensorGroupMatrix,
    gm_main: DiscretizedSSPGroupMatrix,
    W: NDArray,
) -> NDArray:
    """Blocked cross-gram between a tensor and a main-effect discretized group.

    Returns X_main.T @ diag(W) @ X_tensor in SSP space, shape (p_main, p_tensor).

    Uses K2 passes through 2D histograms — O(n*K2) total observation passes
    with O(n_bins_main * n_bins1) memory per pass.  No 3D histogram.

    Column ordering: j1 * K2 + j2, matching _row_kron_dense().
    """
    B1 = gm_tensor.B1_unique_t
    B2 = gm_tensor.B2_unique_t
    B_main = gm_main.B_unique
    K1, K2 = B1.shape[1], B2.shape[1]
    K_main_raw = B_main.shape[1]

    result_raw = np.zeros((K_main_raw, K1 * K2))
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

    return gm_main.R_inv.T @ result_raw @ gm_tensor.R_inv


def _cross_gram(gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X_i.T @ diag(W) @ X_j efficiently.

    For two DiscretizedSSPGroupMatrix, uses a 2D weight histogram to avoid
    materializing either (n, p) matrix. For disc × non-disc, aggregates by
    disc bins in a single compiled pass (fused W-weighting, no toarray for
    sparse groups). Otherwise falls back to materializing the smaller group
    and using rmatvec on the larger.
    """
    # Tensor × tensor (same marginals, e.g. decomposed bilinear/wiggly)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and gm_i.tensor_id == gm_j.tensor_id
    ):
        return _cross_gram_tensor_tensor(gm_i, gm_j, W)

    # Tensor × discretized main-effect (not tensor × tensor with different ids)
    if (
        isinstance(gm_i, DiscretizedTensorGroupMatrix)
        and isinstance(gm_j, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_j, DiscretizedTensorGroupMatrix)
    ):
        return _cross_gram_tensor_main(gm_i, gm_j, W).T
    if (
        isinstance(gm_j, DiscretizedTensorGroupMatrix)
        and isinstance(gm_i, DiscretizedSSPGroupMatrix)
        and not isinstance(gm_i, DiscretizedTensorGroupMatrix)
    ):
        return _cross_gram_tensor_main(gm_j, gm_i, W)

    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            # Fused 2D histogram: single O(n) pass, no (n,) temp allocations.
            W_2d = _disc_disc_2d_hist(gm_i.bin_idx, gm_j.bin_idx, W, gm_i.n_bins, gm_j.n_bins)
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_unique
            return gm_i.R_inv.T @ BtWB @ gm_j.R_inv

    # Disc × non-disc: batch aggregate by disc bins, then dense matmuls.
    # Avoids per-column rmatvec loop, toarray() for sparse groups, and
    # the (n, p) W-broadcast allocation.
    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_j, DiscretizedSSPGroupMatrix
    ):
        WX_agg = _agg_by_bin(gm_j, gm_i.bin_idx, W, gm_i.n_bins)
        return gm_i.R_inv.T @ (gm_i.B_unique.T @ WX_agg)

    if isinstance(gm_j, DiscretizedSSPGroupMatrix) and not isinstance(
        gm_i, DiscretizedSSPGroupMatrix
    ):
        WX_agg = _agg_by_bin(gm_i, gm_j.bin_idx, W, gm_j.n_bins)
        return (gm_j.R_inv.T @ (gm_j.B_unique.T @ WX_agg)).T

    # Cat × cat: weighted crosstab — O(n) with no dense allocation.
    if isinstance(gm_i, CategoricalGroupMatrix) and isinstance(gm_j, CategoricalGroupMatrix):
        return _cat_cat_weighted_crosstab(gm_i.codes, gm_j.codes, W, gm_i.n_levels, gm_j.n_levels)

    # Non-disc × non-disc: materialize smaller side, rmatvec larger side.
    if gm_i.shape[1] <= gm_j.shape[1]:
        X_i = gm_i.toarray()
        WX_i = W[:, None] * X_i
        return np.vstack([gm_j.rmatvec(WX_i[:, k]) for k in range(WX_i.shape[1])])

    X_j = gm_j.toarray()
    WX_j = W[:, None] * X_j
    return np.column_stack([gm_i.rmatvec(WX_j[:, k]) for k in range(WX_j.shape[1])])


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X'diag(W)X for arbitrary-sign weights.

    SSP and Discretized groups handle any-sign W natively (they never use
    sqrt(W)).  Dense and Sparse groups use sqrt(W) internally, which fails
    for negative W, so we fall back to explicit W[:, None] * X for those.
    """
    if isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
        return gm.gram(W)
    if isinstance(gm, CategoricalGroupMatrix):
        return gm.gram(W)  # bincount-based diagonal, handles any-sign W
    X = gm.toarray()
    return (W[:, None] * X).T @ X


def _block_xtwx(gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None) -> NDArray:
    """Compute X.T @ diag(W) @ X block-by-block.

    Uses gm.gram(W) for diagonal blocks (O(n_bins) for discretized groups)
    and _cross_gram for off-diagonal blocks (2D histogram for disc-disc pairs).
    Avoids materializing the full (n, p_total) matrix.
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich.
    """
    if tabmat_split is not None:
        return np.asarray(tabmat_split.sandwich(W))
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block
        XtWX[sl_i, sl_i] = gm_i.gram(W)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX


def _block_xtwx_rhs(
    gms: list[GroupMatrix], groups: list, W: NDArray, Wz: NDArray, *, tabmat_split=None
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute X'WX, X'W, and X'Wz in a single pass over the data.

    For DiscretizedSSPGroupMatrix, shares the O(n) bincount between gram and
    rmatvec operations.  Returns (XtWX, XtW1, XtWz) where XtW1 = X.T @ W
    and XtWz = X.T @ Wz.
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.
    """
    if tabmat_split is not None:
        XtWX = np.asarray(tabmat_split.sandwich(W))
        XtW1 = np.asarray(tabmat_split.transpose_matvec(W))
        XtWz_out = np.asarray(tabmat_split.transpose_matvec(Wz))
        return XtWX, XtW1, XtWz_out
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))
    XtW1 = np.zeros(p_total)
    XtWz_out = np.zeros(p_total)

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        # Diagonal block + rmatvecs via shared bincount
        if isinstance(gm_i, DiscretizedSSPGroupMatrix):
            gram_i, xtw_i, xtwz_i = gm_i.gram_rmatvec(W, Wz)
            XtWX[sl_i, sl_i] = gram_i
            XtW1[sl_i] = xtw_i
            XtWz_out[sl_i] = xtwz_i
        else:
            XtWX[sl_i, sl_i] = gm_i.gram(W)
            XtW1[sl_i] = gm_i.rmatvec(W)
            XtWz_out[sl_i] = gm_i.rmatvec(Wz)

        # Cross blocks with subsequent groups
        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX, XtW1, XtWz_out


def _block_xtwx_signed(
    gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None
) -> NDArray:
    """Like _block_xtwx but safe for arbitrary-sign weights.

    Uses _gram_any_sign for diagonal blocks (avoids sqrt(W) in Dense/Sparse
    groups) and _cross_gram for off-diagonals (already sign-safe).
    When *tabmat_split* is provided, delegates to tabmat.SplitMatrix.sandwich
    (which handles any-sign weights natively).
    """
    if tabmat_split is not None:
        return np.asarray(tabmat_split.sandwich(W))
    p_total = sum(g.end - g.start for g in groups)
    XtWX = np.zeros((p_total, p_total))

    for i, (gm_i, g_i) in enumerate(zip(gms, groups)):
        sl_i = slice(g_i.start, g_i.end)
        XtWX[sl_i, sl_i] = _gram_any_sign(gm_i, W)

        for j in range(i + 1, len(gms)):
            gm_j = gms[j]
            g_j = groups[j]
            sl_j = slice(g_j.start, g_j.end)
            cross = _cross_gram(gm_i, gm_j, W)
            XtWX[sl_i, sl_j] = cross
            XtWX[sl_j, sl_i] = cross.T

    return XtWX


def _build_tabmat_split(gms: list[GroupMatrix]):
    """Build a tabmat SplitMatrix from non-discrete group matrices.

    Returns None if any group is discretized (tabmat can't handle binned ops).
    Uses native tabmat types to avoid unnecessary densification:
      - CategoricalGroupMatrix → tabmat.CategoricalMatrix (codes only, no dense)
      - SparseGroupMatrix → tabmat.SparseMatrix (CSR, no dense)
      - SparseSSPGroupMatrix → tabmat.DenseMatrix (must materialize B @ R_inv)
      - DenseGroupMatrix → tabmat.DenseMatrix (already dense)
    """
    import tabmat

    if any(isinstance(gm, DiscretizedSSPGroupMatrix) for gm in gms):
        return None

    matrices = []
    for gm in gms:
        if isinstance(gm, CategoricalGroupMatrix):
            if gm.n_levels > 100:
                # High-cardinality: use native CategoricalMatrix to avoid
                # O(n × n_levels) dense allocation.  Remap to tabmat
                # convention: base=0, non-base j→j+1, drop_first=True.
                codes = gm.codes.copy().astype(np.int32)
                base_mask = codes == gm.n_levels
                codes[~base_mask] += 1  # non-base: 0..K-1 → 1..K
                codes[base_mask] = 0  # base: K → 0
                # Pin the full category universe so row subsets (CV folds)
                # preserve column count even when some levels are absent.
                categories = np.arange(gm.n_levels + 1)
                matrices.append(
                    tabmat.CategoricalMatrix(codes, categories=categories, drop_first=True)
                )
            else:
                # Low-cardinality: DenseMatrix is faster for sandwich
                # (uniform BLAS, no categorical cross-term overhead).
                matrices.append(tabmat.DenseMatrix(gm.toarray()))
        elif isinstance(gm, SparseGroupMatrix):
            matrices.append(tabmat.SparseMatrix(gm.M))
        elif isinstance(gm, SparseSSPGroupMatrix):
            # Factored B @ R_inv — must materialize (no tabmat equivalent)
            matrices.append(tabmat.DenseMatrix(gm.toarray()))
        else:
            # DenseGroupMatrix
            arr = gm.toarray()
            if arr.ndim == 1:
                arr = arr[:, None]
            matrices.append(tabmat.DenseMatrix(arr))
    return tabmat.SplitMatrix(matrices)


class DesignMatrix:
    """Container for per-group matrices. Provides full-matrix operations."""

    def __init__(self, group_matrices: list[GroupMatrix], n: int, p: int):
        self.group_matrices = group_matrices
        self.n = n
        self.p = p
        self.shape = (n, p)
        self._tabmat_split = None  # lazily built
        self._tabmat_built = False

    @property
    def tabmat_split(self):
        """Lazily build a tabmat SplitMatrix for non-discrete paths."""
        if not self._tabmat_built:
            self._tabmat_split = _build_tabmat_split(self.group_matrices)
            self._tabmat_built = True
        return self._tabmat_split

    def matvec(self, beta: NDArray) -> NDArray:
        """X @ beta via per-group matvecs."""
        result = np.zeros(self.n)
        col = 0
        for gm in self.group_matrices:
            p_g = gm.shape[1]
            result += gm.matvec(beta[col : col + p_g])
            col += p_g
        return result

    def rmatvec(self, w: NDArray) -> NDArray:
        """X.T @ w via per-group rmatvecs."""
        result = np.zeros(self.p)
        col = 0
        for gm in self.group_matrices:
            p_g = gm.shape[1]
            result[col : col + p_g] = gm.rmatvec(w)
            col += p_g
        return result

    def toarray(self) -> NDArray:
        """Concatenate per-group arrays into full (n, p) dense matrix."""
        return np.hstack([gm.toarray() for gm in self.group_matrices])

    def row_subset(self, idx: NDArray) -> DesignMatrix:
        """Return a new DesignMatrix with only the rows at idx."""
        return DesignMatrix(
            [gm.row_subset(idx) for gm in self.group_matrices],
            len(idx),
            self.p,
        )
