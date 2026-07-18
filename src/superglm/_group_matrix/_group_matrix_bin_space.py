"""Public-Tabmat weighted moments for mixed observation/bin-space designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tabmat  # type: ignore[import-untyped]
from numpy.typing import NDArray

from ._group_matrix_execution import WeightedMoments
from ._group_matrix_tabmat import _native_categorical_matrix, _tabmat_vector

_MAX_DENSE_SLAB_BYTES = 64 * 1024 * 1024
_MAX_BIN_CODE_BYTES = 64 * 1024 * 1024
_MAX_AUGMENTED_GRAM_BYTES = 64 * 1024 * 1024
_MAX_SUPPORT_BYTES = 64 * 1024 * 1024
_MAX_RETAINED_AUXILIARY_BYTES = 64 * 1024 * 1024
_MAX_CONSTRUCTION_AUXILIARY_BYTES = 64 * 1024 * 1024
_MAX_TRANSIENT_AUXILIARY_BYTES = 64 * 1024 * 1024

_FLOAT64_BYTES = np.dtype(np.float64).itemsize
_CODE_BYTES = np.dtype(np.int32).itemsize


def _frozen_indices(values) -> NDArray:
    indices = np.asarray(values, dtype=np.intp)
    indices.setflags(write=False)
    return indices


@dataclass(frozen=True)
class CompressedBinSpaceBlock:
    """One immutable mapping from augmented bin indicators to solver columns."""

    group: Any
    augmented_indices: NDArray
    solver_indices: NDArray
    support: NDArray


@dataclass(frozen=True)
class MixedBinSpaceCenteringPlan:
    """Cached augmented Tabmat matrix and its bin-to-solver transformations."""

    split: Any
    dense_slab: NDArray | None
    dense_augmented_indices: NDArray
    ordinary_augmented_indices: NDArray
    ordinary_solver_indices: NDArray
    compressed_blocks: tuple[CompressedBinSpaceBlock, ...]
    n: int
    p: int
    retained_bytes_estimate: int
    construction_bytes_estimate: int
    transient_bytes_estimate: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.p)

    def augmented_location_scale(
        self,
        normalized_weights: NDArray,
    ) -> tuple[NDArray, NDArray | None]:
        """Return call-local augmented summaries from Tabmat standardization."""
        if np.shape(normalized_weights) != (self.n,):
            raise ValueError("normalized weights must match the bin-space plan rows")
        _standardized, mean, scale = self.split.standardize(
            _tabmat_vector(normalized_weights),
            center_predictors=True,
            scale_predictors=True,
        )
        augmented_mean = np.asarray(mean, dtype=np.float64)
        augmented_scale = None if scale is None else np.asarray(scale, dtype=np.float64)
        augmented_shape = (self.split.shape[1],)
        if augmented_mean.shape != augmented_shape or (
            augmented_scale is not None and augmented_scale.shape != augmented_shape
        ):
            raise ValueError("Tabmat standardization returned invalid augmented summaries")
        return augmented_mean, augmented_scale

    def moments(
        self,
        W: NDArray,
        weighted_z: NDArray,
        *,
        augmented_xtw: NDArray | None = None,
    ) -> WeightedMoments:
        """Assemble solver-space moments with one sandwich and one RHS transpose."""
        if np.shape(W) != (self.n,) or np.shape(weighted_z) != (self.n,):
            raise ValueError("weights and weighted_z must match the bin-space plan rows")
        augmented_shape = (self.split.shape[1],)
        if augmented_xtw is not None:
            augmented_xtw = np.asarray(augmented_xtw, dtype=np.float64)
            if augmented_xtw.shape != augmented_shape or not np.all(np.isfinite(augmented_xtw)):
                raise ValueError("augmented_xtw must be a finite augmented-column vector")
        tabmat_W = _tabmat_vector(W)
        tabmat_weighted_z = _tabmat_vector(weighted_z)
        augmented_gram = np.asarray(self.split.sandwich(tabmat_W), dtype=np.float64)
        if augmented_xtw is None:
            # Every non-dense augmented column is a one-hot indicator, so its
            # weighted mass is the corresponding raw-Gram diagonal. Dense
            # columns require their first moment, computed from the one
            # bounded slab already retained by this plan.
            augmented_xtw = np.diag(augmented_gram).copy()
            if self.dense_slab is not None:
                augmented_xtw[self.dense_augmented_indices] = self.dense_slab.T @ tabmat_W
        augmented_rhs = np.asarray(
            self.split.transpose_matvec(tabmat_weighted_z),
            dtype=np.float64,
        )

        gram = np.zeros((self.p, self.p), dtype=np.float64)
        xtw = np.zeros(self.p, dtype=np.float64)
        rhs = np.zeros(self.p, dtype=np.float64)

        ordinary_augmented = self.ordinary_augmented_indices
        ordinary_solver = self.ordinary_solver_indices
        gram[np.ix_(ordinary_solver, ordinary_solver)] = augmented_gram[
            np.ix_(ordinary_augmented, ordinary_augmented)
        ]
        xtw[ordinary_solver] = augmented_xtw[ordinary_augmented]
        rhs[ordinary_solver] = augmented_rhs[ordinary_augmented]

        for block_index, block in enumerate(self.compressed_blocks):
            augmented = block.augmented_indices
            solver = block.solver_indices
            support = block.support

            raw_diagonal = augmented_gram[np.ix_(augmented, augmented)]
            gram[np.ix_(solver, solver)] = support.T @ raw_diagonal @ support
            # Keep transform scratch operations disjoint so the construction-time
            # max-of-operation estimate is also a bound on live temporaries.
            del raw_diagonal
            xtw[solver] = support.T @ augmented_xtw[augmented]
            rhs[solver] = support.T @ augmented_rhs[augmented]

            ordinary_cross = augmented_gram[np.ix_(ordinary_augmented, augmented)] @ support
            gram[np.ix_(ordinary_solver, solver)] = ordinary_cross
            gram[np.ix_(solver, ordinary_solver)] = ordinary_cross.T
            del ordinary_cross

            for previous in self.compressed_blocks[:block_index]:
                raw_cross = augmented_gram[np.ix_(previous.augmented_indices, augmented)]
                cross = previous.support.T @ raw_cross @ support
                gram[np.ix_(previous.solver_indices, solver)] = cross
                gram[np.ix_(solver, previous.solver_indices)] = cross.T
                del raw_cross, cross

        return WeightedMoments(gram=gram, xtw=xtw, xt_rhs=(rhs,))


def build_mixed_bin_space_centering_plan(
    group_matrices,
    *,
    n: int,
    p: int,
) -> MixedBinSpaceCenteringPlan | None:
    """Build the bounded mixed plan, or return ``None`` for unsupported layouts."""
    from ..group_matrix import (
        CategoricalGroupMatrix,
        DenseGroupMatrix,
        DiscretizedSSPGroupMatrix,
    )

    groups = tuple(group_matrices)
    grouped_columns = sum(group.shape[1] for group in groups)
    if any(group.shape[0] != n for group in groups) or grouped_columns != p:
        grouped_rows = groups[0].shape[0] if groups else n
        raise ValueError(
            f"declared design shape {(n, p)} does not match grouped shape "
            f"{(grouped_rows, grouped_columns)}"
        )
    compressed_groups = tuple(group for group in groups if type(group) is DiscretizedSSPGroupMatrix)
    dense_groups = tuple(
        group for group in groups if type(group) is DenseGroupMatrix and group.shape[1] > 0
    )
    categorical_groups = tuple(
        group for group in groups if type(group) is CategoricalGroupMatrix and group.shape[1] > 0
    )
    if (
        not compressed_groups
        or not (dense_groups or categorical_groups)
        or len(categorical_groups) > 1
        or any(
            not (
                type(group) in {DenseGroupMatrix, CategoricalGroupMatrix}
                or type(group) is DiscretizedSSPGroupMatrix
            )
            for group in groups
        )
    ):
        return None

    dense_columns = sum(group.shape[1] for group in dense_groups)
    dense_slab_bytes = n * dense_columns * _FLOAT64_BYTES
    bin_code_bytes = n * (len(compressed_groups) + len(categorical_groups)) * _CODE_BYTES
    augmented_columns = sum(
        group.n_bins if type(group) is DiscretizedSSPGroupMatrix else group.shape[1]
        for group in groups
    )
    augmented_gram_bytes = augmented_columns * augmented_columns * _FLOAT64_BYTES
    support_bytes = sum(
        group.n_bins * group.shape[1] * _FLOAT64_BYTES for group in compressed_groups
    )
    category_metadata_bytes = (
        sum(group.n_bins for group in compressed_groups)
        + sum(group.n_levels + 1 for group in categorical_groups)
    ) * _CODE_BYTES
    ordinary_columns = dense_columns + sum(group.shape[1] for group in categorical_groups)
    plan_index_bytes = (2 * augmented_columns + 2 * p) * np.dtype(np.intp).itemsize
    retained_bytes = (
        dense_slab_bytes
        + bin_code_bytes
        + support_bytes
        + category_metadata_bytes
        + plan_index_bytes
    )

    max_dense_conversion_bytes = max(
        (n * group.shape[1] * _FLOAT64_BYTES for group in dense_groups),
        default=0,
    )
    max_category_metadata_block_bytes = max(
        [group.n_bins * _CODE_BYTES for group in compressed_groups]
        + [(group.n_levels + 1) * _CODE_BYTES for group in categorical_groups],
        default=0,
    )
    categorical_construction_bytes = n * _CODE_BYTES + max_category_metadata_block_bytes
    split_validation_bytes = 3 * augmented_columns * np.dtype(np.intp).itemsize
    construction_bytes = retained_bytes + max(
        max_dense_conversion_bytes,
        categorical_construction_bytes,
        split_validation_bytes,
    )

    max_transform_scratch_bytes = ordinary_columns * ordinary_columns * _FLOAT64_BYTES
    for block_index, group in enumerate(compressed_groups):
        bins = group.n_bins
        width = group.shape[1]
        diagonal_scratch = (bins * bins + bins * width + width * width) * _FLOAT64_BYTES
        ordinary_scratch = (ordinary_columns * bins + ordinary_columns * width) * _FLOAT64_BYTES
        max_transform_scratch_bytes = max(
            max_transform_scratch_bytes,
            diagonal_scratch,
            ordinary_scratch,
        )
        for previous in compressed_groups[:block_index]:
            pair_scratch = (
                previous.n_bins * bins + previous.shape[1] * bins + previous.shape[1] * width
            ) * _FLOAT64_BYTES
            max_transform_scratch_bytes = max(max_transform_scratch_bytes, pair_scratch)

    solver_moment_bytes = (p * p + 2 * p) * _FLOAT64_BYTES
    augmented_moment_bytes = (
        augmented_columns * augmented_columns + 2 * augmented_columns
    ) * _FLOAT64_BYTES
    transform_phase_bytes = (
        solver_moment_bytes + augmented_moment_bytes + max_transform_scratch_bytes
    )
    component_widths = sorted(
        ([dense_columns] if dense_columns else [])
        + [group.n_levels for group in categorical_groups]
        + [group.n_bins for group in compressed_groups],
        reverse=True,
    )
    largest_width = component_widths[0]
    second_width = component_widths[1] if len(component_widths) > 1 else 0
    # SplitMatrix.sandwich keeps its previous local ``res`` alive while
    # evaluating the next block result. Conservatively treat the largest
    # diagonal and largest cross result as simultaneous with the output.
    sandwich_result_liveness_bytes = (
        largest_width * largest_width + largest_width * second_width
    ) * _FLOAT64_BYTES
    sandwich_phase_bytes = augmented_moment_bytes + sandwich_result_liveness_bytes
    transient_bytes = max(transform_phase_bytes, sandwich_phase_bytes)
    if (
        dense_slab_bytes > _MAX_DENSE_SLAB_BYTES
        or bin_code_bytes > _MAX_BIN_CODE_BYTES
        or augmented_gram_bytes > _MAX_AUGMENTED_GRAM_BYTES
        or support_bytes > _MAX_SUPPORT_BYTES
        or retained_bytes > _MAX_RETAINED_AUXILIARY_BYTES
        or construction_bytes > _MAX_CONSTRUCTION_AUXILIARY_BYTES
        or transient_bytes > _MAX_TRANSIENT_AUXILIARY_BYTES
    ):
        return None

    solver_cursor = 0
    augmented_cursor = 0
    dense_cursor = 0
    dense_augmented_indices: list[int] = []
    ordinary_augmented_indices: list[int] = []
    ordinary_solver_indices: list[int] = []
    components = []
    component_indices = []
    compressed_blocks: list[CompressedBinSpaceBlock] = []
    dense_slab = np.empty((n, dense_columns), dtype=np.float64)

    for group in groups:
        solver = _frozen_indices(range(solver_cursor, solver_cursor + group.shape[1]))
        if type(group) is DenseGroupMatrix:
            augmented = _frozen_indices(range(augmented_cursor, augmented_cursor + group.shape[1]))
            dense_slab[:, dense_cursor : dense_cursor + group.shape[1]] = np.asarray(
                group.M,
                dtype=np.float64,
            )
            dense_augmented_indices.extend(augmented.tolist())
            ordinary_augmented_indices.extend(augmented.tolist())
            ordinary_solver_indices.extend(solver.tolist())
            dense_cursor += group.shape[1]
            augmented_cursor += group.shape[1]
        elif type(group) is CategoricalGroupMatrix:
            augmented = _frozen_indices(range(augmented_cursor, augmented_cursor + group.n_levels))
            if group.n_levels:
                components.append(_native_categorical_matrix(group.codes, group.n_levels))
                component_indices.append(augmented.copy())
                ordinary_augmented_indices.extend(augmented.tolist())
                ordinary_solver_indices.extend(solver.tolist())
            augmented_cursor += group.n_levels
        else:
            augmented = _frozen_indices(range(augmented_cursor, augmented_cursor + group.n_bins))
            bin_codes = np.asarray(group.bin_idx, dtype=np.int32)
            if (
                bin_codes.shape != (n,)
                or np.any(bin_codes < 0)
                or np.any(bin_codes >= group.n_bins)
            ):
                return None
            components.append(
                tabmat.CategoricalMatrix(
                    bin_codes,
                    categories=np.arange(group.n_bins, dtype=np.int32),
                    drop_first=False,
                    dtype=np.float64,
                )
            )
            del bin_codes
            component_indices.append(augmented.copy())
            support = np.matmul(group.B_unique, group.R_inv, dtype=np.float64)
            support.setflags(write=False)
            compressed_blocks.append(
                CompressedBinSpaceBlock(
                    group=group,
                    augmented_indices=augmented,
                    solver_indices=solver,
                    support=support,
                )
            )
            augmented_cursor += group.n_bins
        solver_cursor += group.shape[1]

    if dense_columns:
        dense_slab.setflags(write=False)
        components.insert(0, tabmat.DenseMatrix(dense_slab))
        component_indices.insert(0, np.asarray(dense_augmented_indices, dtype=np.intp))

    split = tabmat.SplitMatrix(components, indices=component_indices)
    if split.shape != (n, augmented_columns):
        return None
    return MixedBinSpaceCenteringPlan(
        split=split,
        dense_slab=dense_slab if dense_columns else None,
        dense_augmented_indices=_frozen_indices(dense_augmented_indices),
        ordinary_augmented_indices=_frozen_indices(ordinary_augmented_indices),
        ordinary_solver_indices=_frozen_indices(ordinary_solver_indices),
        compressed_blocks=tuple(compressed_blocks),
        n=n,
        p=p,
        retained_bytes_estimate=retained_bytes,
        construction_bytes_estimate=construction_bytes,
        transient_bytes_estimate=transient_bytes,
    )
