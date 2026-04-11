"""Private core group-matrix class implementations."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._group_matrix_kernels import _csr_weighted_gram


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
        out: NDArray = np.zeros(self.shape, dtype=np.float64)
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
        "omega_components",
        "component_types",
        "lambda_policies",
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
        self.omega_components = None  # list[(suffix, omega)] for multi-penalty, set externally
        self.component_types = None  # dict[suffix, type] for multi-penalty, set externally
        self.lambda_policies = None  # dict[suffix, LambdaPolicy] for multi-penalty, set externally

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
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        return sub
