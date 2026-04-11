"""Per-group matrix wrappers for sparse/dense BCD operations.

Five wrapper types with the same interface:
- DenseGroupMatrix: numeric features (single column) or dense fallback
- SparseGroupMatrix: categoricals, non-SSP splines
- SparseSSPGroupMatrix: SSP splines (factored: sparse B + dense R_inv)
- DiscretizedSSPGroupMatrix: discretized SSP splines (binned B_unique + index)
- DiscretizedSCOPGroupMatrix: discretized SCOP monotone splines (centered design at bin centers)

DesignMatrix holds the list and provides full-matrix matvec/rmatvec.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._group_matrix import _group_matrix_algebra
from ._group_matrix._group_matrix_bins import discretize_column
from ._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)
from ._group_matrix._group_matrix_discretized import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
)
from ._group_matrix._group_matrix_kernels import (
    _disc_disc_2d_hist as _kernel_disc_disc_2d_hist,
)
from ._group_matrix._group_matrix_tabmat import _build_tabmat_split

DenseGroupMatrix.__module__ = __name__
SparseGroupMatrix.__module__ = __name__
CategoricalGroupMatrix.__module__ = __name__
SparseSSPGroupMatrix.__module__ = __name__
DiscretizedSSPGroupMatrix.__module__ = __name__
DiscretizedSCOPGroupMatrix.__module__ = __name__
DiscretizedTensorGroupMatrix.__module__ = __name__


def _discretize_column(x: NDArray, n_bins: int = 256) -> tuple[NDArray, NDArray]:
    """Compatibility wrapper for the private discretization helper."""
    return cast(tuple[NDArray, NDArray], discretize_column(x, n_bins))


GroupMatrix = (
    DenseGroupMatrix
    | SparseGroupMatrix
    | CategoricalGroupMatrix
    | SparseSSPGroupMatrix
    | DiscretizedSSPGroupMatrix
    | DiscretizedSCOPGroupMatrix
    | DiscretizedTensorGroupMatrix
)

_MAX_DISC_DISC_HIST_CELLS = 5_000_000
_MAX_DISC_DISC_CHANNEL_HIST_CELLS = 5_000_000


def _agg_by_bin(gm: GroupMatrix, bin_idx: NDArray, W: NDArray, n_bins: int) -> NDArray:
    """Compatibility wrapper for the private algebra helper."""
    return _group_matrix_algebra._agg_by_bin(gm, bin_idx, W, n_bins)


def _cross_gram_tensor_tensor(gm_i, gm_j, W: NDArray) -> NDArray:
    """Compatibility wrapper for tensor×tensor cross-gram helper."""
    return _group_matrix_algebra._cross_gram_tensor_tensor(gm_i, gm_j, W)


def _disc_disc_2d_hist(
    bin_idx_i: NDArray, bin_idx_j: NDArray, W: NDArray, n_bins_i: int, n_bins_j: int
) -> NDArray:
    """Compatibility wrapper for the fused discretized 2D histogram kernel."""
    return _kernel_disc_disc_2d_hist(bin_idx_i, bin_idx_j, W, n_bins_i, n_bins_j)


def _cross_gram_tensor_main(gm_tensor, gm_main, W: NDArray) -> NDArray:
    """Compatibility wrapper for tensor×main-effect cross-gram helper."""
    return _group_matrix_algebra._cross_gram_tensor_main(gm_tensor, gm_main, W)


def _cross_gram(gm_i: GroupMatrix, gm_j: GroupMatrix, W: NDArray) -> NDArray:
    """Compatibility wrapper for the private cross-gram helper."""
    return _group_matrix_algebra._cross_gram(gm_i, gm_j, W)


def _gram_any_sign(gm: GroupMatrix, W: NDArray) -> NDArray:
    """Compatibility wrapper for any-sign diagonal gram helper."""
    return _group_matrix_algebra._gram_any_sign(gm, W)


def _block_xtwx(gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None) -> NDArray:
    """Compatibility wrapper for block XtWX assembly."""
    return _group_matrix_algebra._block_xtwx(gms, groups, W, tabmat_split=tabmat_split)


def _block_xtwx_rhs(
    gms: list[GroupMatrix], groups: list, W: NDArray, Wz: NDArray, *, tabmat_split=None
) -> tuple[NDArray, NDArray, NDArray]:
    """Compatibility wrapper for block XtWX/XtW/XtWz assembly."""
    return cast(
        tuple[NDArray, NDArray, NDArray],
        _group_matrix_algebra._block_xtwx_rhs(
            gms,
            groups,
            W,
            Wz,
            tabmat_split=tabmat_split,
        ),
    )


def _block_xtwx_signed(
    gms: list[GroupMatrix], groups: list, W: NDArray, *, tabmat_split=None
) -> NDArray:
    """Compatibility wrapper for arbitrary-sign block XtWX assembly."""
    return _group_matrix_algebra._block_xtwx_signed(
        gms,
        groups,
        W,
        tabmat_split=tabmat_split,
    )


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
