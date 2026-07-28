"""Backend-neutral execution plans for grouped design matrices."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ._group_matrix_algebra import (
    _BlockWeightCache,
    _cross_gram,
    _gram_any_sign,
    _profile_count,
    _profile_elapsed,
    _runtime_group_matrix_types,
)
from ._group_matrix_tabmat import _build_tabmat_split, _tabmat_vector

_MIN_AUTO_TABMAT_MOMENT_ROWS = 50_000


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


@dataclass(frozen=True)
class OrdinaryPartitionDecision:
    """Construction-time ordinary Tabmat partition and its stable reason."""

    indices: tuple[int, ...]
    reason: str


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
        prepared_ordinary_split=None,
        ordinary_split_factory=None,
    ):
        self.group_matrices = tuple(group_matrices)
        self.n = int(n)
        self._ordinary_tabmat_policy = ordinary_tabmat
        self._ordinary_split_factory = ordinary_split_factory
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

        ordinary_decision = self._ordinary_partition_decision()
        ordinary_indices = ordinary_decision.indices
        self._ordinary_indices = frozenset(ordinary_indices)
        self._ordinary_partition_reason = ordinary_decision.reason
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
        if prepared_ordinary_split is not None and prepared_ordinary_split.shape != (
            self.n,
            len(self._ordinary_columns),
        ):
            raise ValueError("prepared ordinary split does not match the planned partition")
        self._ordinary_split = prepared_ordinary_split
        self._ordinary_split_built = prepared_ordinary_split is not None
        runtime_types = _runtime_group_matrix_types()
        from ..group_matrix import FactorSmoothGroupMatrix

        self._tensor_group_type = runtime_types[4]
        self._fused_group_types = (
            runtime_types[1],
            runtime_types[2],
            runtime_types[3],
            runtime_types[7],
            FactorSmoothGroupMatrix,
        )
        self._group_entries = tuple(zip(self.group_spans, self.group_matrices, strict=True))
        self._n_groups = len(self._group_entries)
        self._group_columns = tuple(span.columns for span in self.group_spans)
        self._ordinary_mask = tuple(
            index in self._ordinary_indices for index in range(len(self.group_matrices))
        )
        self._tensor_mask = tuple(
            isinstance(group, self._tensor_group_type) for group in self.group_matrices
        )
        self._fused_mask = tuple(
            isinstance(group, self._fused_group_types) for group in self.group_matrices
        )
        self._layout_frozen = True

    def _ordinary_partition_decision(self) -> OrdinaryPartitionDecision:
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
            return OrdinaryPartitionDecision((), "policy-disabled")
        if self._ordinary_tabmat_policy is True:
            return OrdinaryPartitionDecision(candidates, "policy-forced")

        # The measured automatic crossover is deliberately narrow.  Multiple
        # categoricals (especially a <=100-level block materialized dense), one
        # dense column, sparse blocks, and smaller row counts all have measured
        # counterexamples.  Compressed designs likewise favor their specialized
        # grouped assembler over a partial Tabmat partition.
        if len(candidates) != len(self.group_matrices):
            return OrdinaryPartitionDecision((), "contains-compressed-group")
        categorical_groups = tuple(
            group for group in self.group_matrices if isinstance(group, CategoricalGroupMatrix)
        )
        if len(categorical_groups) != 1 or categorical_groups[0].n_levels <= 100:
            return OrdinaryPartitionDecision((), "categorical-layout")
        dense_width = sum(
            group.shape[1] for group in self.group_matrices if isinstance(group, DenseGroupMatrix)
        )
        if dense_width < 3:
            return OrdinaryPartitionDecision((), "dense-width")
        has_sparse = any(isinstance(group, SparseGroupMatrix) for group in self.group_matrices)
        if has_sparse:
            return OrdinaryPartitionDecision((), "contains-sparse-group")
        if self.n < _MIN_AUTO_TABMAT_MOMENT_ROWS:
            return OrdinaryPartitionDecision((), "row-threshold")
        return OrdinaryPartitionDecision(candidates, "auto-certified")

    @property
    def ordinary_indices(self) -> tuple[int, ...]:
        """Return the immutable ordinary Tabmat group indices without building it."""
        return tuple(sorted(self._ordinary_indices))

    @property
    def ordinary_partition_reason(self) -> str:
        """Return the stable construction-time reason for the ordinary partition."""
        return self._ordinary_partition_reason

    def _get_ordinary_split(self):
        if not self._ordinary_split_built:
            ordinary_groups = [
                self.group_matrices[index] for index in sorted(self._ordinary_indices)
            ]
            if ordinary_groups and self._ordinary_split_factory is not None:
                self._ordinary_split = self._ordinary_split_factory()
            else:
                self._ordinary_split = (
                    _build_tabmat_split(ordinary_groups) if ordinary_groups else None
                )
            if self._ordinary_split is not None and self._ordinary_split.shape != (
                self.n,
                len(self._ordinary_columns),
            ):
                raise ValueError("ordinary split factory returned the wrong partition")
            self._ordinary_split_built = True
        return self._ordinary_split

    def validate_group_spans(self, groups) -> None:
        """Validate external solver metadata against immutable plan coordinates."""
        if len(groups) != len(self.group_spans) or any(
            group.start != span.start or group.end != span.end
            for group, span in zip(groups, self.group_spans, strict=False)
        ):
            raise ValueError("group slices must be contiguous and match matrix widths")

    @staticmethod
    def _scatter_vector(target: NDArray, columns: NDArray, values: NDArray) -> None:
        target[columns] = np.asarray(values, dtype=np.float64)

    def _compressed_signed_gram(self, weights: NDArray, *, profile: dict | None) -> NDArray:
        """Return a signed Gram without the general weighted-moment dispatch."""
        gram = np.zeros((self.p, self.p), dtype=np.float64)
        cache = _BlockWeightCache(profile)
        for left_index, (left_span, left_group) in enumerate(self._group_entries):
            left_columns = self._group_columns[left_index]
            diagonal_start = perf_counter() if profile is not None else 0.0
            gram[left_columns, left_columns] = _gram_any_sign(left_group, weights)
            if profile is not None:
                if self._tensor_mask[left_index]:
                    diagonal_profile_key = "block_diag_tensor_s"
                elif self._fused_mask[left_index]:
                    diagonal_profile_key = "block_diag_discrete_ssp_s"
                else:
                    diagonal_profile_key = "block_diag_other_s"
                _profile_elapsed(profile, diagonal_profile_key, diagonal_start)

            for right_index in range(left_index + 1, self._n_groups):
                right_span, right_group = self._group_entries[right_index]
                right_columns = self._group_columns[right_index]
                cross = _cross_gram(left_group, right_group, weights, cache, profile)
                gram[left_columns, right_columns] = cross
                gram[right_columns, left_columns] = cross.T
        return gram

    def moments(
        self,
        weights: NDArray,
        *,
        rhs: tuple[NDArray, ...] = (),
        include_xtw: bool = False,
        signed: bool = False,
        profile: dict | None = None,
    ) -> WeightedMoments:
        """Validate inputs and return weighted products through this plan."""
        return self._moments_impl(
            weights,
            rhs=rhs,
            include_xtw=include_xtw,
            signed=signed,
            profile=profile,
            validate_inputs=True,
        )

    def _moments_prevalidated(
        self,
        weights: NDArray,
        *,
        rhs: tuple[NDArray, ...] = (),
        include_xtw: bool = False,
        signed: bool = False,
        profile: dict | None = None,
    ) -> WeightedMoments:
        """Return moments for fit-internal arrays with established domains."""
        if signed and not rhs and not include_xtw and not self._ordinary_indices:
            _profile_count(profile, "block_calls")
            return WeightedMoments(
                gram=self._compressed_signed_gram(weights, profile=profile),
                xtw=None,
                xt_rhs=(),
            )
        return self._moments_impl(
            weights,
            rhs=rhs,
            include_xtw=include_xtw,
            signed=signed,
            profile=profile,
            validate_inputs=False,
        )

    def _moments_impl(
        self,
        weights: NDArray,
        *,
        rhs: tuple[NDArray, ...],
        include_xtw: bool,
        signed: bool,
        profile: dict | None,
        validate_inputs: bool,
    ) -> WeightedMoments:
        """Return a Gram and transpose products through one hybrid plan.

        The public route coerces and validates its inputs. The private route
        trusts fit-internal arrays and skips coercion plus domain scans.
        """
        if validate_inputs:
            W = np.asarray(weights, dtype=np.float64)
            rhs_vectors = tuple(np.asarray(vector, dtype=np.float64) for vector in rhs)
            if W.shape != (self.n,) or any(vector.shape != (self.n,) for vector in rhs_vectors):
                raise ValueError(
                    "weights and right-hand-side vectors must match the plan row count"
                )
            if not np.all(np.isfinite(W)) or any(
                not np.all(np.isfinite(vector)) for vector in rhs_vectors
            ):
                raise ValueError("weights and right-hand-side vectors must be finite")
            if not signed and np.any(W < 0.0):
                raise ValueError("negative weights require signed=True")
        else:
            W = weights
            rhs_vectors = rhs
        _profile_count(profile, "block_calls")

        if signed and not rhs_vectors and not include_xtw and not self._ordinary_indices:
            return WeightedMoments(
                gram=self._compressed_signed_gram(W, profile=profile),
                xtw=None,
                xt_rhs=(),
            )

        ordinary_split = self._get_ordinary_split()
        ordinary_is_full = ordinary_split is not None and len(self._ordinary_indices) == len(
            self.group_matrices
        )
        gram: NDArray
        xtw: NDArray | None
        if ordinary_is_full:
            gram = np.empty((0, 0), dtype=np.float64)
            xtw = None
            xt_rhs: list[NDArray] = []
        else:
            gram = np.zeros((self.p, self.p), dtype=np.float64)
            xtw = np.zeros(self.p, dtype=np.float64) if include_xtw else None
            xt_rhs = [np.zeros(self.p, dtype=np.float64) for _vector in rhs_vectors]
        if ordinary_split is not None:
            ordinary_start = perf_counter() if profile is not None else 0.0
            tabmat_W = _tabmat_vector(W)
            ordinary_gram = np.asarray(ordinary_split.sandwich(tabmat_W), dtype=np.float64)
            if ordinary_is_full:
                gram = ordinary_gram
                xtw = (
                    np.asarray(ordinary_split.transpose_matvec(tabmat_W), dtype=np.float64)
                    if include_xtw
                    else None
                )
                xt_rhs = [
                    np.asarray(
                        ordinary_split.transpose_matvec(_tabmat_vector(vector)),
                        dtype=np.float64,
                    )
                    for vector in rhs_vectors
                ]
            else:
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
            _profile_elapsed(profile, "block_tabmat_s", ordinary_start)
            if ordinary_is_full:
                return WeightedMoments(gram=gram, xtw=xtw, xt_rhs=tuple(xt_rhs))

        cache = _BlockWeightCache(profile)
        for left_index, (left_span, left_group) in enumerate(self._group_entries):
            left_is_ordinary = ordinary_split is not None and self._ordinary_mask[left_index]
            if not left_is_ordinary:
                columns = left_span.columns
                diagonal_start = perf_counter() if profile is not None else 0.0
                fusion_vector = rhs_vectors[0] if rhs_vectors else (W if xtw is not None else None)
                if fusion_vector is not None and self._fused_mask[left_index]:
                    if self._tensor_mask[left_index]:
                        w_grid, rhs_grid = cache.tensor_w_wz_grid(left_group, W, fusion_vector)
                        group_gram, group_xtw, group_rhs = left_group.gram_rmatvec_from_grids(
                            w_grid, rhs_grid
                        )
                    else:
                        group_gram, group_xtw, group_rhs = left_group.gram_rmatvec(W, fusion_vector)
                    gram[columns, columns] = group_gram
                    if xtw is not None:
                        xtw[columns] = group_xtw
                    if rhs_vectors:
                        xt_rhs[0][columns] = group_rhs
                    remaining_rhs = zip(xt_rhs[1:], rhs_vectors[1:], strict=True)
                else:
                    gram[columns, columns] = (
                        _gram_any_sign(left_group, W) if signed else left_group.gram(W)
                    )
                    if xtw is not None:
                        xtw[columns] = left_group.rmatvec(W)
                    remaining_rhs = zip(xt_rhs, rhs_vectors, strict=True)
                for target, vector in remaining_rhs:
                    target[columns] = left_group.rmatvec(vector)
                if profile is not None:
                    if self._tensor_mask[left_index]:
                        diagonal_profile_key = "block_diag_tensor_s"
                    elif self._fused_mask[left_index]:
                        diagonal_profile_key = "block_diag_discrete_ssp_s"
                    else:
                        diagonal_profile_key = "block_diag_other_s"
                    _profile_elapsed(profile, diagonal_profile_key, diagonal_start)

            for right_index in range(left_index + 1, self._n_groups):
                right_span, right_group = self._group_entries[right_index]
                if left_is_ordinary and self._ordinary_mask[right_index]:
                    continue
                cross = _cross_gram(left_group, right_group, W, cache, profile)
                gram[left_span.columns, right_span.columns] = cross
                gram[right_span.columns, left_span.columns] = cross.T

        return WeightedMoments(gram=gram, xtw=xtw, xt_rhs=tuple(xt_rhs))
