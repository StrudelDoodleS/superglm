"""Per-group matrix wrappers for sparse/dense BCD operations.

Three wrapper types with the same interface:
- DenseGroupMatrix: numeric features (single column) or dense fallback
- SparseGroupMatrix: categoricals, non-SSP splines
- SparseSSPGroupMatrix: SSP splines (factored: sparse B + dense R_inv)

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


class SparseSSPGroupMatrix:
    """Factored SSP group matrix: stores sparse B + dense R_inv separately.

    Effective matrix is B @ R_inv, but we never form it explicitly.
    """

    __slots__ = ("B", "_data", "_indices", "_indptr", "_p_b", "R_inv", "shape")

    def __init__(self, B_csr: sp.spmatrix, R_inv: NDArray):
        self.B = sp.csr_matrix(B_csr)
        self._data = self.B.data.astype(np.float64)
        self._indices = self.B.indices
        self._indptr = self.B.indptr
        self._p_b = self.B.shape[1]
        self.R_inv = np.asarray(R_inv)
        self.shape = (self.B.shape[0], self.R_inv.shape[1])

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


GroupMatrix = DenseGroupMatrix | SparseGroupMatrix | SparseSSPGroupMatrix


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
