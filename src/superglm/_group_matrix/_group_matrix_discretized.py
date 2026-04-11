"""Private discretized group-matrix class implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_kernels import (
    _disc_disc_2d_hist,
    _fused_2d_bincount_2,
    _fused_bincount_2,
)


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
        sub = DiscretizedSSPGroupMatrix(self.B_unique, self.R_inv, self.bin_idx[idx])
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
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        return sub
