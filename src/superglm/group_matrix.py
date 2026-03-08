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

    def toarray(self) -> NDArray:
        return (self.B_unique @ self.R_inv)[self.bin_idx]

    def row_subset(self, idx: NDArray) -> DiscretizedSSPGroupMatrix:
        sub = DiscretizedSSPGroupMatrix(self.B_unique, self.R_inv, self.bin_idx[idx])
        sub.omega = self.omega
        sub.projection = self.projection
        return sub


GroupMatrix = (
    DenseGroupMatrix | SparseGroupMatrix | SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix
)

_MAX_DISC_DISC_HIST_CELLS = 5_000_000


def _cross_gram(gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray) -> NDArray:
    """Compute X_i.T @ diag(W) @ X_j efficiently.

    For two DiscretizedSSPGroupMatrix, uses a 2D weight histogram to avoid
    materializing either (n, p) matrix. Otherwise falls back to materializing
    the smaller group and using rmatvec on the larger.
    """
    if isinstance(gm_i, DiscretizedSSPGroupMatrix) and isinstance(gm_j, DiscretizedSSPGroupMatrix):
        n_joint = gm_i.n_bins * gm_j.n_bins
        if n_joint <= _MAX_DISC_DISC_HIST_CELLS:
            # 2D histogram: bin weights by (bin_i, bin_j) jointly
            joint_idx = gm_i.bin_idx * gm_j.n_bins + gm_j.bin_idx
            W_2d = np.bincount(joint_idx, weights=W, minlength=n_joint).reshape(
                gm_i.n_bins, gm_j.n_bins
            )
            BtWB = gm_i.B_unique.T @ W_2d @ gm_j.B_unique
            return gm_i.R_inv.T @ BtWB @ gm_j.R_inv

    if gm_i.shape[1] <= gm_j.shape[1]:
        X_i = gm_i.toarray()
        WX_i = W[:, None] * X_i
        return np.vstack([gm_j.rmatvec(WX_i[:, k]) for k in range(WX_i.shape[1])])

    # Fall back: materialize group j, use rmatvec of group i per column
    X_j = gm_j.toarray()
    WX_j = W[:, None] * X_j
    return np.column_stack([gm_i.rmatvec(WX_j[:, k]) for k in range(WX_j.shape[1])])


def _block_xtwx(gms: list[GroupMatrix], groups: list, W: NDArray) -> NDArray:
    """Compute X.T @ diag(W) @ X block-by-block.

    Uses gm.gram(W) for diagonal blocks (O(n_bins) for discretized groups)
    and _cross_gram for off-diagonal blocks (2D histogram for disc-disc pairs).
    Avoids materializing the full (n, p_total) matrix.
    """
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


class DesignMatrix:
    """Container for per-group matrices. Provides full-matrix operations."""

    def __init__(self, group_matrices: list[GroupMatrix], n: int, p: int):
        self.group_matrices = group_matrices
        self.n = n
        self.p = p
        self.shape = (n, p)

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
