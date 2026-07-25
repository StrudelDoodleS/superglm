"""Private core group-matrix class implementations."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.factor_smooth_geometry import (
    adjoint_sum_to_zero_blocks,
    expand_sum_to_zero_blocks,
)

from ._group_matrix_discretized import DiscretizedSSPGroupMatrix
from ._group_matrix_kernels import (
    _csr_weighted_gram,
    _factor_smooth_csr_dense_cross,
    _factor_smooth_csr_matvec,
    _factor_smooth_csr_rmatvec,
    _factor_smooth_csr_sufficient_stats,
    _factor_smooth_support_cell_aggregates,
    _factor_smooth_support_dense_cell_aggregates,
    _factor_smooth_support_dense_cross,
    _factor_smooth_support_matvec,
    _factor_smooth_support_rmatvec,
)


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


class RandomEffectGroupMatrix(CategoricalGroupMatrix):
    """All-level categorical matrix used by structured random effects."""

    __slots__ = ("lambda_policies",)

    def __init__(
        self,
        codes: NDArray,
        n_levels: int,
        *,
        lambda_policies=None,
    ):
        c = np.asarray(codes, dtype=np.intp)
        if n_levels < 1:
            raise ValueError(f"n_levels must be positive, got {n_levels}")
        if np.any((c < 0) | (c >= n_levels)):
            raise ValueError("RandomEffect codes must be between 0 and n_levels - 1.")
        super().__init__(c, n_levels)
        self.lambda_policies = lambda_policies

    def row_subset(self, idx: NDArray) -> RandomEffectGroupMatrix:
        return RandomEffectGroupMatrix(
            self.codes[idx],
            self.n_levels,
            lambda_policies=self.lambda_policies,
        )


class FactorSmoothGroupMatrix:
    """Compact all-level factor-by-spline matrix.

    FS coefficients have one natural-basis block per level. SZ coefficients
    have ``K-1`` free blocks and reconstruct the final raw level by contrast.
    The observation-level interaction matrix is never retained: exact matrices
    store one shared CSR marginal basis, while discrete matrices store one
    support basis and an observation-to-bin index.
    """

    __slots__ = (
        "B",
        "B_unique",
        "_data",
        "_indices",
        "_indptr",
        "bin_idx",
        "codes",
        "n_levels",
        "coefficient_levels",
        "block_size",
        "raw_width",
        "natural_map",
        "levels",
        "shape",
        "is_discrete",
        "repeated_penalty_components",
        "lambda_policies",
        "omega",
        "omega_components",
        "component_types",
        "projection",
        "structured_kind",
        "factor_basis",
    )

    def __init__(
        self,
        basis: sp.spmatrix | NDArray,
        codes: NDArray,
        n_levels: int,
        *,
        natural_map: NDArray,
        levels: tuple[object, ...] | list[object],
        repeated_penalty_components: tuple[tuple[str, NDArray], ...],
        factor_basis: Literal["fs", "sz"] = "fs",
        lambda_policies=None,
        bin_idx: NDArray | None = None,
    ):
        if n_levels < 1:
            raise ValueError(f"n_levels must be positive, got {n_levels}")
        if factor_basis not in ("fs", "sz"):
            raise ValueError(f"factor_basis must be 'fs' or 'sz', got {factor_basis!r}")
        if factor_basis == "sz" and n_levels < 2:
            raise ValueError("factor_basis='sz' requires at least two levels")
        level_codes = np.asarray(codes, dtype=np.intp)
        if level_codes.ndim != 1:
            raise ValueError("FactorSmooth codes must be one-dimensional.")
        if np.any((level_codes < 0) | (level_codes >= n_levels)):
            raise ValueError("FactorSmooth codes must be between 0 and n_levels - 1.")

        transform = np.asarray(natural_map, dtype=np.float64)
        if transform.ndim != 2:
            raise ValueError("natural_map must be a two-dimensional matrix")
        raw_width, block_size = transform.shape
        if raw_width < 1 or block_size < 1:
            raise ValueError("natural_map dimensions must be positive")
        fitted_levels = tuple(levels)
        if len(fitted_levels) != n_levels:
            raise ValueError("levels length must equal n_levels")

        self.codes = level_codes
        self.n_levels = int(n_levels)
        self.factor_basis = factor_basis
        self.coefficient_levels = self.n_levels if factor_basis == "fs" else self.n_levels - 1
        self.block_size = int(block_size)
        self.raw_width = int(raw_width)
        self.natural_map = transform
        self.levels = fitted_levels
        self.repeated_penalty_components = repeated_penalty_components
        self.lambda_policies = lambda_policies
        self.omega = None
        self.omega_components = None
        self.component_types = None
        self.projection = None
        self.structured_kind = "factor_smooth"

        if bin_idx is None:
            exact_basis = sp.csr_matrix(basis, dtype=np.float64)
            if exact_basis.shape != (len(level_codes), raw_width):
                raise ValueError(
                    "exact FactorSmooth basis shape must match row count and natural_map"
                )
            self.B = exact_basis
            self.B_unique = None
            self._data = exact_basis.data.astype(np.float64, copy=False)
            self._indices = exact_basis.indices
            self._indptr = exact_basis.indptr
            self.bin_idx = None
            n_rows = exact_basis.shape[0]
            self.is_discrete = False
        else:
            support_basis = np.asarray(basis, dtype=np.float64)
            support_index = np.asarray(bin_idx, dtype=np.intp)
            if support_basis.ndim != 2 or support_basis.shape[1] != raw_width:
                raise ValueError("discrete FactorSmooth support basis has the wrong width")
            if support_index.shape != level_codes.shape:
                raise ValueError("FactorSmooth bin_idx must match the code row count")
            if np.any((support_index < 0) | (support_index >= support_basis.shape[0])):
                raise ValueError("FactorSmooth bin_idx contains an out-of-range support row")
            self.B = None
            self.B_unique = np.ascontiguousarray(support_basis)
            self._data = None
            self._indices = None
            self._indptr = None
            self.bin_idx = support_index
            n_rows = len(level_codes)
            self.is_discrete = True

        for suffix, component in repeated_penalty_components:
            block = np.asarray(component)
            if block.shape != (block_size, block_size):
                raise ValueError(
                    f"repeated penalty component {suffix!r} has shape {block.shape}, "
                    f"expected {(block_size, block_size)}"
                )
        self.shape = (n_rows, self.coefficient_levels * self.block_size)

    def matvec(self, v: NDArray) -> NDArray:
        """Apply the implicit level-major design to a coefficient vector."""
        coefficients = np.asarray(v, dtype=np.float64)
        if coefficients.shape != (self.shape[1],):
            raise ValueError(f"coefficient vector must have shape {(self.shape[1],)}")
        natural_coefficients = coefficients.reshape(self.coefficient_levels, self.block_size)
        if self.factor_basis == "sz":
            natural_coefficients = expand_sum_to_zero_blocks(natural_coefficients)
        raw_coefficients = natural_coefficients @ self.natural_map.T
        if self.is_discrete:
            return _factor_smooth_support_matvec(
                self.B_unique,
                self.bin_idx,
                self.codes,
                raw_coefficients,
            )
        return _factor_smooth_csr_matvec(
            self._data,
            self._indices,
            self._indptr,
            self.codes,
            raw_coefficients,
        )

    def rmatvec(self, w: NDArray) -> NDArray:
        """Aggregate observations into implicit level-major coordinates."""
        values = np.asarray(w, dtype=np.float64)
        if values.shape != (self.shape[0],):
            raise ValueError(f"observation vector must have shape {(self.shape[0],)}")
        if self.is_discrete:
            raw = _factor_smooth_support_rmatvec(
                self.B_unique,
                self.bin_idx,
                self.codes,
                values,
                self.n_levels,
            )
        else:
            raw = _factor_smooth_csr_rmatvec(
                self._data,
                self._indices,
                self._indptr,
                self.codes,
                values,
                self.n_levels,
                self.raw_width,
            )
        natural = np.asarray(raw @ self.natural_map, dtype=np.float64)
        if self.factor_basis == "sz":
            natural = adjoint_sum_to_zero_blocks(natural)
        return natural.ravel()

    def factor_smooth_sufficient_stats(
        self,
        W: NDArray,
        rhs: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Return per-level natural-basis Gram, ``X'W``, and ``X'rhs``."""
        weights = np.asarray(W, dtype=np.float64)
        rhs_values = np.asarray(rhs, dtype=np.float64)
        expected = (self.shape[0],)
        if weights.shape != expected or rhs_values.shape != expected:
            raise ValueError("weights and rhs must match the FactorSmooth row count")
        if self.is_discrete:
            _cell_weights, local_gram, xtw_nat, rhs_nat = self.factor_smooth_discrete_cell_moments(
                weights, rhs_values
            )
            return local_gram, xtw_nat, rhs_nat
        raw_gram, raw_xtw, raw_rhs = _factor_smooth_csr_sufficient_stats(
            self._data,
            self._indices,
            self._indptr,
            self.codes,
            weights,
            rhs_values,
            self.n_levels,
            self.raw_width,
        )
        local_gram = np.einsum(
            "ai,kab,bj->kij",
            self.natural_map,
            raw_gram,
            self.natural_map,
            optimize=True,
        )
        # The raw kernels update symmetric entries identically.  BLAS
        # contraction through the natural map can still leave round-off at
        # opposite triangle entries, large enough to trip exact structured
        # factor validation on production-sized data.
        local_gram = 0.5 * (local_gram + local_gram.transpose(0, 2, 1))
        return (
            local_gram,
            raw_xtw @ self.natural_map,
            raw_rhs @ self.natural_map,
        )

    def factor_smooth_discrete_cell_moments(
        self,
        W: NDArray,
        rhs: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Return compact cells and natural-basis moments for a discrete block."""
        weights = np.asarray(W, dtype=np.float64)
        rhs_values = np.asarray(rhs, dtype=np.float64)
        expected = (self.shape[0],)
        if weights.shape != expected or rhs_values.shape != expected:
            raise ValueError("weights and rhs must match the FactorSmooth row count")
        if not self.is_discrete:
            raise ValueError("cell moments require a discrete FactorSmooth matrix")

        basis = self.B_unique
        support_index = self.bin_idx
        if basis is None or support_index is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("discrete FactorSmooth support is unavailable")

        cell_weights, cell_rhs = _factor_smooth_support_cell_aggregates(
            support_index,
            self.codes,
            weights,
            rhs_values,
            self.n_levels,
            basis.shape[0],
        )
        effective_basis = np.ascontiguousarray(
            basis @ self.natural_map,
            dtype=np.float64,
        )
        weighted_basis = cell_weights[:, :, None] * effective_basis[None, :, :]
        local_gram = effective_basis.T[None, :, :] @ weighted_basis
        local_gram = 0.5 * (local_gram + local_gram.transpose(0, 2, 1))
        return (
            np.ascontiguousarray(cell_weights),
            np.ascontiguousarray(local_gram),
            np.ascontiguousarray(cell_weights @ effective_basis),
            np.ascontiguousarray(cell_rhs @ effective_basis),
        )

    def gram(self, W: NDArray) -> NDArray:
        """Materialize the block-diagonal Gram for small dense-reference fits."""
        zeros = np.zeros(self.shape[0], dtype=np.float64)
        local_gram, _xtw, _rhs = self.factor_smooth_sufficient_stats(W, zeros)
        return self._public_gram(local_gram)

    def _public_gram(self, local_gram: NDArray) -> NDArray[np.float64]:
        """Convert raw independent level Grams to public FS/SZ coordinates."""
        result = np.zeros((self.shape[1], self.shape[1]), dtype=np.float64)
        if self.factor_basis == "fs":
            for level, block in enumerate(local_gram):
                start = level * self.block_size
                result[start : start + self.block_size, start : start + self.block_size] = block
            return result

        last = local_gram[-1]
        for left in range(self.coefficient_levels):
            left_sl = slice(left * self.block_size, (left + 1) * self.block_size)
            result[left_sl, left_sl] += local_gram[left]
            for right in range(self.coefficient_levels):
                right_sl = slice(right * self.block_size, (right + 1) * self.block_size)
                result[left_sl, right_sl] += last
        return result

    def factor_smooth_dense_cross_gram(
        self,
        W: NDArray,
        dense_small: NDArray,
    ) -> NDArray:
        """Return ``(K, k, q)`` cross blocks against one narrow dense matrix."""
        weights = np.asarray(W, dtype=np.float64)
        small = np.asarray(dense_small, dtype=np.float64)
        if weights.shape != (self.shape[0],):
            raise ValueError("weights must match the FactorSmooth row count")
        if small.ndim != 2 or small.shape[0] != self.shape[0]:
            raise ValueError("dense_small must be a row-aligned two-dimensional matrix")
        if self.is_discrete:
            raw = _factor_smooth_support_dense_cross(
                self.B_unique,
                self.bin_idx,
                self.codes,
                weights,
                small,
                self.n_levels,
            )
        else:
            raw = _factor_smooth_csr_dense_cross(
                self._data,
                self._indices,
                self._indptr,
                self.codes,
                weights,
                small,
                self.n_levels,
                self.raw_width,
            )
        return np.einsum(
            "ai,kaq->kiq",
            self.natural_map,
            raw,
            optimize=True,
        )

    def factor_smooth_discrete_dense_cell_cross_gram(
        self,
        W: NDArray,
        dense_small: NDArray,
    ) -> NDArray:
        """Return raw level crosses after one dense-small cell aggregation."""
        weights = np.asarray(W, dtype=np.float64)
        small = np.asarray(dense_small, dtype=np.float64)
        if weights.shape != (self.shape[0],):
            raise ValueError("weights must match the FactorSmooth row count")
        if small.ndim != 2 or small.shape[0] != self.shape[0]:
            raise ValueError("dense_small must be a row-aligned two-dimensional matrix")
        if not self.is_discrete:
            raise ValueError("dense cell crosses require a discrete FactorSmooth matrix")
        basis = self.B_unique
        support_index = self.bin_idx
        if basis is None or support_index is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("discrete FactorSmooth support is unavailable")

        cells = _factor_smooth_support_dense_cell_aggregates(
            support_index,
            self.codes,
            weights,
            small,
            self.n_levels,
            basis.shape[0],
        )
        raw = basis.T[None, :, :] @ cells
        return np.ascontiguousarray(
            self.natural_map.T[None, :, :] @ raw,
            dtype=np.float64,
        )

    def factor_smooth_discrete_shared_bin_cross_gram(
        self,
        cell_weights: NDArray,
        other: object,
    ) -> NDArray | None:
        """Return raw level crosses for an SSP sharing the exact support map."""
        basis = self.B_unique
        support_index = self.bin_idx
        if (
            not self.is_discrete
            or basis is None
            or support_index is None
            or not isinstance(other, DiscretizedSSPGroupMatrix)
            or type(other) is not DiscretizedSSPGroupMatrix
        ):
            return None
        if support_index.shape != other.bin_idx.shape or not np.array_equal(
            support_index,
            other.bin_idx,
        ):
            return None

        cells = np.asarray(cell_weights, dtype=np.float64)
        expected_cells = (self.n_levels, basis.shape[0])
        if cells.shape != expected_cells:
            raise ValueError(f"cell_weights must have shape {expected_cells}")
        other_basis = np.asarray(other.B_unique, dtype=np.float64)
        other_transform = np.asarray(other.R_inv, dtype=np.float64)
        if (
            other_basis.ndim != 2
            or other_transform.ndim != 2
            or other_basis.shape[0] != basis.shape[0]
            or other_basis.shape[1] != other_transform.shape[0]
        ):
            return None

        other_support = np.ascontiguousarray(
            other_basis @ other_transform,
            dtype=np.float64,
        )
        weighted_other = cells[:, :, None] * other_support[None, :, :]
        raw = basis.T[None, :, :] @ weighted_other
        return np.ascontiguousarray(
            self.natural_map.T[None, :, :] @ raw,
            dtype=np.float64,
        )

    def gram_rmatvec(
        self,
        W: NDArray,
        Wz: NDArray,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Fuse the dense-reference Gram and two transpose products."""
        local_gram, local_xtw, local_rhs = self.factor_smooth_sufficient_stats(W, Wz)
        if self.factor_basis == "sz":
            local_xtw = adjoint_sum_to_zero_blocks(local_xtw)
            local_rhs = adjoint_sum_to_zero_blocks(local_rhs)
        return self._public_gram(local_gram), local_xtw.ravel(), local_rhs.ravel()

    def toarray(self) -> NDArray:
        """Materialize the implicit matrix as an explicit small-model oracle."""
        if self.is_discrete:
            natural_basis = self.B_unique[self.bin_idx] @ self.natural_map
        else:
            natural_basis = np.asarray(self.B @ self.natural_map)
        result = np.zeros(self.shape, dtype=np.float64)
        blocks = result.reshape(self.shape[0], self.coefficient_levels, self.block_size)
        free_rows = np.flatnonzero(self.codes < self.coefficient_levels)
        blocks[free_rows, self.codes[free_rows]] = natural_basis[free_rows]
        if self.factor_basis == "sz":
            final_rows = np.flatnonzero(self.codes == self.n_levels - 1)
            blocks[final_rows] = -natural_basis[final_rows, None, :]
        return result

    def row_subset(self, idx: NDArray) -> FactorSmoothGroupMatrix:
        """Subset observations while retaining the global fitted level layout."""
        row_index = np.asarray(idx, dtype=np.intp)
        if self.is_discrete:
            return FactorSmoothGroupMatrix(
                self.B_unique,
                self.codes[row_index],
                self.n_levels,
                natural_map=self.natural_map,
                levels=self.levels,
                repeated_penalty_components=self.repeated_penalty_components,
                factor_basis=self.factor_basis,
                lambda_policies=self.lambda_policies,
                bin_idx=self.bin_idx[row_index],
            )
        return FactorSmoothGroupMatrix(
            self.B[row_index],
            self.codes[row_index],
            self.n_levels,
            natural_map=self.natural_map,
            levels=self.levels,
            repeated_penalty_components=self.repeated_penalty_components,
            factor_basis=self.factor_basis,
            lambda_policies=self.lambda_policies,
        )


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


class SplineCategoricalGroupMatrix:
    """One spline-by-category level without materialising a masked spline block."""

    __slots__ = (
        "B",
        "B_level",
        "_data",
        "_indices",
        "_indptr",
        "_p_b",
        "R_inv",
        "row_idx",
        "n_rows",
        "shape",
        "omega",
        "projection",
        "omega_components",
        "component_types",
        "lambda_policies",
        "spline_cat_level",
        "spline_cat_feature",
    )

    def __init__(self, B_csr: sp.spmatrix, R_inv: NDArray, mask_or_idx: NDArray):
        self.B = sp.csr_matrix(B_csr)
        self.n_rows = self.B.shape[0]
        arr = np.asarray(mask_or_idx)
        if arr.dtype == bool:
            if arr.shape != (self.n_rows,):
                raise ValueError("mask length must match spline basis row count")
            row_idx = np.flatnonzero(arr)
        else:
            row_idx = arr.astype(np.intp, copy=False)
            if row_idx.ndim != 1:
                raise ValueError("row index array must be one-dimensional")
            if row_idx.size and (int(row_idx.min()) < 0 or int(row_idx.max()) >= self.n_rows):
                raise ValueError("row index array contains rows outside the spline basis")

        self.row_idx = np.asarray(row_idx, dtype=np.intp)
        self.B_level = self.B[self.row_idx].tocsr()
        self._data = self.B_level.data.astype(np.float64)
        self._indices = self.B_level.indices
        self._indptr = self.B_level.indptr
        self._p_b = self.B_level.shape[1]
        self.R_inv = np.asarray(R_inv)
        self.shape = (self.n_rows, self.R_inv.shape[1])
        self.omega = None
        self.projection = None
        self.omega_components = None
        self.component_types = None
        self.lambda_policies = None
        self.spline_cat_level = None
        self.spline_cat_feature = None

    def matvec(self, v: NDArray) -> NDArray:
        out = np.zeros(self.shape[0], dtype=np.float64)
        if self.row_idx.size == 0:
            return out
        raw_beta = self.R_inv @ v
        out[self.row_idx] = np.asarray(self.B_level @ raw_beta).ravel()
        return out

    def rmatvec(self, w: NDArray) -> NDArray:
        return self.R_inv.T @ np.asarray(self.B_level.T @ w[self.row_idx]).ravel()

    def gram(self, W: NDArray) -> NDArray:
        raw_gram = _csr_weighted_gram(
            self._data,
            self._indices,
            self._indptr,
            W[self.row_idx],
            self._p_b,
        )
        return self.R_inv.T @ raw_gram @ self.R_inv

    def gram_rmatvec(self, W: NDArray, Wz: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        W_sub = W[self.row_idx]
        Wz_sub = Wz[self.row_idx]
        raw_gram = _csr_weighted_gram(
            self._data,
            self._indices,
            self._indptr,
            W_sub,
            self._p_b,
        )
        gram = self.R_inv.T @ raw_gram @ self.R_inv
        xtw = self.R_inv.T @ np.asarray(self.B_level.T @ W_sub).ravel()
        xtwz = self.R_inv.T @ np.asarray(self.B_level.T @ Wz_sub).ravel()
        return gram, xtw, xtwz

    def toarray(self) -> NDArray:
        out = np.zeros(self.shape, dtype=np.float64)
        if self.row_idx.size:
            out[self.row_idx] = np.asarray(self.B_level @ self.R_inv)
        return out

    def row_subset(self, idx: NDArray) -> SplineCategoricalGroupMatrix:
        idx_arr = np.asarray(idx)
        if idx_arr.dtype == bool:
            idx_arr = np.flatnonzero(idx_arr)
        else:
            idx_arr = idx_arr.astype(np.intp, copy=False)
        sub_row_idx = np.flatnonzero(np.isin(idx_arr, self.row_idx))
        sub = SplineCategoricalGroupMatrix(self.B[idx_arr], self.R_inv, sub_row_idx)
        sub.omega = self.omega
        sub.projection = self.projection
        sub.omega_components = self.omega_components
        sub.component_types = self.component_types
        sub.lambda_policies = self.lambda_policies
        sub.spline_cat_level = self.spline_cat_level
        sub.spline_cat_feature = self.spline_cat_feature
        return sub
