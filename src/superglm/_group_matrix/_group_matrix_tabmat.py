"""Private tabmat construction helpers for group matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import tabmat  # type: ignore[import-untyped]
from numpy.typing import NDArray

_MIN_RAW_SPLINE_TABMAT_ROWS = 8_000
_MAX_RAW_SPLINE_TABMAT_CELLS = 4_000_000
_MAX_RAW_SPLINE_TABMAT_NNZ_PER_ROW = 4
_MAX_RAW_SPLINE_TABMAT_DENSITY_NUMERATOR = 1
_MAX_RAW_SPLINE_TABMAT_DENSITY_DENOMINATOR = 3
_MIN_RAW_SPLINE_TABMAT_REUSE_ROWS = 50_000
_MIN_COLD_CONSTANT_WEIGHT_RAW_WIDTH = 30
_MIN_RETAINED_VECTOR_ROWS = 10_000
_MIN_RETAINED_VECTOR_SCALAR_BLOCKS = 8
_MAX_RETAINED_VECTOR_CATEGORICAL_WORK_RATIO = 96


@dataclass(frozen=True)
class RawSplineTabmatPlan:
    """One raw-basis Tabmat split plus solver-coordinate transforms."""

    split: object
    raw_slices: tuple[slice, ...]
    solver_slices: tuple[slice, ...]
    transforms: tuple[NDArray, ...]
    raw_width: int
    solver_width: int

    def transform_vector(self, raw: NDArray) -> NDArray:
        """Map a raw-basis transpose product into solver coordinates."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.raw_width,):
            raise ValueError("raw transpose product has the wrong width")
        result = np.empty(self.solver_width, dtype=np.float64)
        for raw_slice, solver_slice, transform in zip(
            self.raw_slices,
            self.solver_slices,
            self.transforms,
            strict=True,
        ):
            result[solver_slice] = transform.T @ raw[raw_slice]
        return result

    def transform_gram(self, raw: NDArray) -> NDArray:
        """Map a raw-basis Gram into solver coordinates block by block."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (self.raw_width, self.raw_width):
            raise ValueError("raw Gram has the wrong shape")
        result = np.empty((self.solver_width, self.solver_width), dtype=np.float64)
        blocks = tuple(zip(self.raw_slices, self.solver_slices, self.transforms, strict=True))
        for left_index, (raw_left, solver_left, transform_left) in enumerate(blocks):
            for raw_right, solver_right, transform_right in blocks[left_index:]:
                block = transform_left.T @ raw[raw_left, raw_right] @ transform_right
                result[solver_left, solver_right] = block
                if solver_left != solver_right:
                    result[solver_right, solver_left] = block.T
        return 0.5 * (result + result.T)

    @property
    def retained_bytes(self) -> int:
        """Estimate Tabmat's retained CSC/CSR storage through its public API."""
        arrays: list[NDArray] = []
        for matrix in self.split.matrices:
            sparse_array = matrix.array_csc
            arrays.extend((sparse_array.data, sparse_array.indices, sparse_array.indptr))
        arrays.extend(np.asarray(indices) for indices in self.split.indices)
        unique = {id(array): array for array in arrays}
        csc_and_indices = int(sum(array.nbytes for array in unique.values()))
        # ``sandwich`` lazily materializes a CSR representation. Estimate its
        # public sparse-array footprint without forcing that allocation merely
        # because profiling was enabled.
        csr_bytes = sum(
            matrix.data.nbytes
            + matrix.indices.nbytes
            + (matrix.shape[0] + 1) * matrix.indices.dtype.itemsize
            for matrix in self.split.matrices
        )
        return csc_and_indices + int(csr_bytes)


def _is_raw_spline_tabmat_centering_candidate(gms, *, n: int) -> bool:
    """Return whether a measured raw-spline Tabmat crossover admits a layout."""
    from ..group_matrix import SparseSSPGroupMatrix

    splines = tuple(group for group in gms if group.shape[1] > 0)
    if (
        len(splines) < 2
        or any(type(group) is not SparseSSPGroupMatrix for group in splines)
        or any(group.shape[0] != n for group in splines)
        or any(group.B.nnz > _MAX_RAW_SPLINE_TABMAT_NNZ_PER_ROW * n for group in splines)
        or any(
            _MAX_RAW_SPLINE_TABMAT_DENSITY_DENOMINATOR * group.B.nnz
            > _MAX_RAW_SPLINE_TABMAT_DENSITY_NUMERATOR * n * group.B.shape[1]
            for group in splines
        )
        or n < _MIN_RAW_SPLINE_TABMAT_ROWS
    ):
        return False
    raw_width = sum(int(group.B.shape[1]) for group in splines)
    return raw_width > 0 and n * raw_width <= _MAX_RAW_SPLINE_TABMAT_CELLS


def _defer_raw_spline_tabmat_plan(
    *,
    n: int,
    raw_width: int,
    constant_weights: bool,
    repeated_fit: bool,
) -> bool:
    """Return whether cold CSC construction loses to one stable data pass."""
    return bool(
        constant_weights
        and not repeated_fit
        and (
            raw_width < _MIN_COLD_CONSTANT_WEIGHT_RAW_WIDTH
            or n >= _MIN_RAW_SPLINE_TABMAT_REUSE_ROWS
        )
    )


def _build_raw_spline_tabmat_plan(gms, *, n: int) -> RawSplineTabmatPlan | None:
    """Build one combined CSC Tabmat split without materializing ``B @ R_inv``."""
    if not _is_raw_spline_tabmat_centering_candidate(gms, n=n):
        return None
    splines = tuple(group for group in gms if group.shape[1] > 0)
    raw_basis = sp.hstack(
        [group.B.astype(np.float64, copy=False) for group in splines],
        format="csc",
        dtype=np.float64,
    )
    raw_matrix = tabmat.SparseMatrix(raw_basis, copy=False)
    split = tabmat.SplitMatrix([raw_matrix])
    raw_widths = tuple(int(group.B.shape[1]) for group in splines)
    solver_widths = tuple(int(group.shape[1]) for group in splines)
    raw_starts = np.cumsum((0, *raw_widths), dtype=np.intp)
    solver_starts = np.cumsum((0, *solver_widths), dtype=np.intp)
    return RawSplineTabmatPlan(
        split=split,
        raw_slices=tuple(
            slice(int(raw_starts[index]), int(raw_starts[index + 1]))
            for index in range(len(splines))
        ),
        solver_slices=tuple(
            slice(int(solver_starts[index]), int(solver_starts[index + 1]))
            for index in range(len(splines))
        ),
        transforms=tuple(np.asarray(group.R_inv, dtype=np.float64) for group in splines),
        raw_width=int(raw_starts[-1]),
        solver_width=int(solver_starts[-1]),
    )


def _dense_float64(values):
    """Return a solver-compatible dense Tabmat block."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return tabmat.DenseMatrix(array)


def _tabmat_vector(values):
    """Return the writable contiguous float64 buffer Tabmat kernels require."""
    array = np.asarray(values, dtype=np.float64)
    if not array.flags.c_contiguous or not array.flags.writeable:
        array = np.array(array, dtype=np.float64, order="C", copy=True)
    return array


def _native_categorical_matrix(codes, n_levels: int):
    """Return a native Tabmat block matching SuperGLM's base-sentinel encoding."""
    remapped = np.asarray(codes, dtype=np.int32).copy()
    base_mask = remapped == n_levels
    remapped[~base_mask] += 1
    remapped[base_mask] = 0
    return tabmat.CategoricalMatrix(
        remapped,
        categories=np.arange(n_levels + 1, dtype=np.int32),
        drop_first=True,
        dtype=np.float64,
    )


def _is_tabmat_centering_candidate(gms) -> bool:
    """Return whether centering can use a native categorical Tabmat block."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    unsupported = (
        SparseSSPGroupMatrix
        | SplineCategoricalGroupMatrix
        | DiscretizedSplineCategoricalGroupMatrix
        | DiscretizedSSPGroupMatrix
        | DiscretizedSCOPGroupMatrix
    )
    return (
        not any(isinstance(gm, unsupported) for gm in gms)
        and any(not isinstance(gm, CategoricalGroupMatrix) for gm in gms)
        and any(isinstance(gm, CategoricalGroupMatrix) and gm.n_levels > 100 for gm in gms)
    )


def _is_retained_tabmat_vector_candidate(gms, *, n: int) -> bool:
    """Return whether a retained split beats grouped vector operations.

    This gate is driven by the repeated scalar blocks and their ratio to native
    categorical components. A single dense slab is already efficient, while
    repeated scalar groups allocate one row-sized temporary apiece. Category-
    heavy layouts remain on the grouped transpose path because their measured
    SplitMatrix dispatch overhead can exceed that saved by the scalar blocks.
    """
    from ..group_matrix import CategoricalGroupMatrix, DenseGroupMatrix

    dense_groups = tuple(group for group in gms if type(group) is DenseGroupMatrix)
    categorical_groups = tuple(group for group in gms if type(group) is CategoricalGroupMatrix)
    return bool(
        n >= _MIN_RETAINED_VECTOR_ROWS
        and len(dense_groups) >= _MIN_RETAINED_VECTOR_SCALAR_BLOCKS
        and all(group.shape[1] == 1 for group in dense_groups)
        and all(group.n_levels > 100 for group in categorical_groups)
        and 3 * len(categorical_groups) <= len(dense_groups)
        and _MAX_RETAINED_VECTOR_CATEGORICAL_WORK_RATIO
        * sum(group.n_levels for group in categorical_groups)
        <= n * len(dense_groups)
        and len(dense_groups) + len(categorical_groups) == len(gms)
    )


def _build_tabmat_split(gms):
    """Build a tabmat SplitMatrix from non-discrete group matrices."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSCOPGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        DiscretizedSSPGroupMatrix,
        SparseGroupMatrix,
        SparseSSPGroupMatrix,
        SplineCategoricalGroupMatrix,
    )

    if any(
        isinstance(
            gm,
            SparseSSPGroupMatrix
            | SplineCategoricalGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix
            | DiscretizedSCOPGroupMatrix,
        )
        for gm in gms
    ):
        return None

    if all(isinstance(gm, CategoricalGroupMatrix) for gm in gms) and all(
        gm.n_levels <= 100 for gm in gms if isinstance(gm, CategoricalGroupMatrix)
    ):
        return None

    matrices = []
    for gm in gms:
        if isinstance(gm, CategoricalGroupMatrix):
            if gm.n_levels > 100:
                matrices.append(_native_categorical_matrix(gm.codes, gm.n_levels))
            else:
                matrices.append(_dense_float64(gm.toarray()))
        elif isinstance(gm, SparseGroupMatrix):
            matrices.append(tabmat.SparseMatrix(gm.M.astype(np.float64, copy=False)))
        elif isinstance(gm, SparseSSPGroupMatrix):
            matrices.append(_dense_float64(gm.toarray()))
        elif isinstance(gm, DenseGroupMatrix):
            matrices.append(_dense_float64(gm.toarray()))
        else:
            matrices.append(_dense_float64(gm.toarray()))
    return tabmat.SplitMatrix(matrices)
