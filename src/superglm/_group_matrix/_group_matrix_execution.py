"""Backend-neutral execution plans for grouped design matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_algebra import (
    _BlockWeightCache,
    _cross_gram,
    _gram_any_sign,
    _runtime_group_matrix_types,
)
from ._group_matrix_tabmat import _build_tabmat_split, _tabmat_vector


@dataclass(frozen=True)
class GroupSpan:
    """One group's immutable location in global solver coordinates."""

    index: int
    start: int
    end: int

    @property
    def columns(self) -> slice:
        return slice(self.start, self.end)


@dataclass(frozen=True)
class WeightedMoments:
    """Weighted products returned in global solver-column order."""

    gram: NDArray
    xtw: NDArray | None
    xt_rhs: tuple[NDArray, ...]


class MatrixExecutionPlan:
    """Partition ordinary and compressed groups behind one moment interface.

    The group layout is immutable. The ordinary Tabmat partition is built
    lazily and contains only observation-level dense, sparse, and categorical
    blocks. Discretized/support-space groups remain compressed and use their
    specialized group and cross-product kernels.
    """

    _IMMUTABLE_LAYOUT_FIELDS = frozenset({"group_matrices", "n", "p", "shape", "group_spans"})

    def __setattr__(self, name, value):
        if getattr(self, "_layout_frozen", False) and name in self._IMMUTABLE_LAYOUT_FIELDS:
            raise AttributeError(f"{name} is immutable after plan construction")
        super().__setattr__(name, value)

    def __init__(
        self,
        group_matrices,
        *,
        n: int,
        ordinary_tabmat: bool | None = None,
    ):
        self.group_matrices = tuple(group_matrices)
        self.n = int(n)
        self._ordinary_tabmat_policy = ordinary_tabmat
        widths: list[int] = []
        for group in self.group_matrices:
            if group.shape[0] != self.n:
                raise ValueError("every group matrix must match the plan row count")
            widths.append(int(group.shape[1]))
        starts = np.cumsum([0, *widths], dtype=np.intp)
        self.group_spans = tuple(
            GroupSpan(index, int(starts[index]), int(starts[index + 1]))
            for index in range(len(widths))
        )
        self.p = int(starts[-1])
        self.shape = (self.n, self.p)

        ordinary_indices = self._eligible_ordinary_indices()
        self._ordinary_indices = frozenset(ordinary_indices)
        ordinary_columns = (
            np.concatenate(
                [
                    np.arange(
                        self.group_spans[index].start,
                        self.group_spans[index].end,
                        dtype=np.intp,
                    )
                    for index in ordinary_indices
                ]
            )
            if ordinary_indices
            else np.empty(0, dtype=np.intp)
        )
        ordinary_columns.setflags(write=False)
        self._ordinary_columns = ordinary_columns
        self._ordinary_split = None
        self._ordinary_split_built = False
        self._layout_frozen = True

    def _eligible_ordinary_indices(self) -> tuple[int, ...]:
        from ..group_matrix import (
            CategoricalGroupMatrix,
            DenseGroupMatrix,
            SparseGroupMatrix,
        )

        ordinary_types = (DenseGroupMatrix, SparseGroupMatrix, CategoricalGroupMatrix)
        candidates = tuple(
            index
            for index, group in enumerate(self.group_matrices)
            if isinstance(group, ordinary_types)
        )
        if self._ordinary_tabmat_policy is False:
            return ()
        if self._ordinary_tabmat_policy is True:
            return candidates

        # The measured automatic crossover is currently limited to wholly
        # ordinary mixed designs with a native high-cardinality categorical
        # block.  For mixed compressed designs, the specialized grouped
        # assembler beats a partial Tabmat partition, sometimes substantially.
        if len(candidates) != len(self.group_matrices):
            return ()
        has_native_categorical = any(
            isinstance(group, CategoricalGroupMatrix) and group.n_levels > 100
            for group in self.group_matrices
        )
        has_numeric_or_sparse = any(
            isinstance(group, DenseGroupMatrix | SparseGroupMatrix) for group in self.group_matrices
        )
        if not has_native_categorical or not has_numeric_or_sparse:
            return ()
        return candidates

    def _get_ordinary_split(self):
        if not self._ordinary_split_built:
            ordinary_groups = [
                self.group_matrices[index] for index in sorted(self._ordinary_indices)
            ]
            self._ordinary_split = _build_tabmat_split(ordinary_groups) if ordinary_groups else None
            self._ordinary_split_built = True
        return self._ordinary_split

    @staticmethod
    def _scatter_vector(target: NDArray, columns: NDArray, values: NDArray) -> None:
        target[columns] = np.asarray(values, dtype=np.float64)

    def moments(
        self,
        weights: NDArray,
        *,
        rhs: tuple[NDArray, ...] = (),
        include_xtw: bool = False,
        signed: bool = False,
    ) -> WeightedMoments:
        """Return a Gram and transpose products through one hybrid plan."""
        W = np.asarray(weights, dtype=np.float64)
        rhs_vectors = tuple(np.asarray(vector, dtype=np.float64) for vector in rhs)
        if W.shape != (self.n,) or any(vector.shape != (self.n,) for vector in rhs_vectors):
            raise ValueError("weights and right-hand-side vectors must match the plan row count")
        if not np.all(np.isfinite(W)) or any(
            not np.all(np.isfinite(vector)) for vector in rhs_vectors
        ):
            raise ValueError("weights and right-hand-side vectors must be finite")
        if not signed and np.any(W < 0.0):
            raise ValueError("negative weights require signed=True")

        gram = np.zeros((self.p, self.p), dtype=np.float64)
        xtw = np.zeros(self.p, dtype=np.float64) if include_xtw else None
        xt_rhs = [np.zeros(self.p, dtype=np.float64) for _vector in rhs_vectors]
        cache = _BlockWeightCache()
        (
            _CategoricalGroupMatrix,
            DiscretizedSCOPGroupMatrix,
            DiscretizedSplineCategoricalGroupMatrix,
            DiscretizedSSPGroupMatrix,
            DiscretizedTensorGroupMatrix,
            _SparseGroupMatrix,
            _SparseSSPGroupMatrix,
            SplineCategoricalGroupMatrix,
        ) = _runtime_group_matrix_types()
        fused_group_types = (
            DiscretizedSCOPGroupMatrix
            | DiscretizedSplineCategoricalGroupMatrix
            | DiscretizedSSPGroupMatrix
            | SplineCategoricalGroupMatrix
        )

        ordinary_split = self._get_ordinary_split()
        if ordinary_split is not None:
            tabmat_W = _tabmat_vector(W)
            ordinary_gram = np.asarray(ordinary_split.sandwich(tabmat_W), dtype=np.float64)
            gram[np.ix_(self._ordinary_columns, self._ordinary_columns)] = ordinary_gram
            if xtw is not None:
                self._scatter_vector(
                    xtw,
                    self._ordinary_columns,
                    ordinary_split.transpose_matvec(tabmat_W),
                )
            for target, vector in zip(xt_rhs, rhs_vectors, strict=True):
                self._scatter_vector(
                    target,
                    self._ordinary_columns,
                    ordinary_split.transpose_matvec(_tabmat_vector(vector)),
                )

        for span, group in zip(self.group_spans, self.group_matrices, strict=True):
            if span.index in self._ordinary_indices and ordinary_split is not None:
                continue
            columns = span.columns
            fusion_vector = rhs_vectors[0] if rhs_vectors else (W if xtw is not None else None)
            if fusion_vector is not None and isinstance(group, fused_group_types):
                if isinstance(group, DiscretizedTensorGroupMatrix):
                    w_grid, rhs_grid = cache.tensor_w_wz_grid(group, W, fusion_vector)
                    group_gram, group_xtw, group_rhs = group.gram_rmatvec_from_grids(
                        w_grid, rhs_grid
                    )
                else:
                    group_gram, group_xtw, group_rhs = group.gram_rmatvec(W, fusion_vector)
                gram[columns, columns] = group_gram
                if xtw is not None:
                    xtw[columns] = group_xtw
                if rhs_vectors:
                    xt_rhs[0][columns] = group_rhs
                remaining_rhs = zip(xt_rhs[1:], rhs_vectors[1:], strict=True)
            else:
                gram[columns, columns] = _gram_any_sign(group, W) if signed else group.gram(W)
                if xtw is not None:
                    xtw[columns] = group.rmatvec(W)
                remaining_rhs = zip(xt_rhs, rhs_vectors, strict=True)
            for target, vector in remaining_rhs:
                target[columns] = group.rmatvec(vector)

        for left_index, (left_span, left_group) in enumerate(
            zip(self.group_spans, self.group_matrices, strict=True)
        ):
            for right_span, right_group in zip(
                self.group_spans[left_index + 1 :],
                self.group_matrices[left_index + 1 :],
                strict=True,
            ):
                if (
                    ordinary_split is not None
                    and left_span.index in self._ordinary_indices
                    and right_span.index in self._ordinary_indices
                ):
                    continue
                cross = _cross_gram(left_group, right_group, W, cache)
                gram[left_span.columns, right_span.columns] = cross
                gram[right_span.columns, left_span.columns] = cross.T

        return WeightedMoments(gram=gram, xtw=xtw, xt_rhs=tuple(xt_rhs))
