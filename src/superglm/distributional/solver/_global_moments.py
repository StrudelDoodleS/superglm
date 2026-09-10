"""Bounded signed global moments for ordinary discretized predictors.

A geometry owns small solver-support tables and sufficient statistics across
likelihood chunks. Histograms are initialized once, updated in place, and
contracted once. Only ordinary columns receive a bounded observation panel.
Source authority and numerical-domain refusals are checked before native writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import cast

import numpy as np
from numba import njit

from superglm._group_matrix._group_matrix_range import _category_rows
from superglm.distributional.layout import PredictorState, StackedLayout
from superglm.distributional.predictor import PredictorExecutionPlan
from superglm.distributional.solver._batched_moments import (
    _batch_workspace_bytes,
    _BatchedMomentReducers,
    _warmup_batched_moments,
)
from superglm.distributional.solver._ordinary_packing import (
    _copy_dense_checked,
    _warmup_ordinary_packing,
)
from superglm.distributional.solver._ordinary_packing import (
    _write_categorical_block as _pack_categorical,
)
from superglm.distributional.solver.assembly import (
    DenseJointGeometry,
    validated_dense_penalty,
)
from superglm.distributional.solver.packing import packed_pairs
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
)

# Bounds apply to original solver-support, ordinary and derivative operands,
# never to signed accumulated moments. Their degree-three products stay in the
# normal exponent range under the separately bounded row/support dimensions;
# cancellation may produce much smaller intermediate or final values.
_MIN_ABS = 2.0**-128
_MAX_ABS = 2.0**128
_ORDINARY_TYPES = (DenseGroupMatrix, CategoricalGroupMatrix)
_SUPPORT_TYPES = (DiscretizedSSPGroupMatrix, DiscretizedSplineCategoricalGroupMatrix)


class GlobalMomentRefusalError(ValueError):
    """The whole current geometry must be discarded after this exception."""

    def __init__(self, reason: str, *, recoverable: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.recoverable = recoverable


@dataclass(frozen=True)
class GlobalMomentBuild:
    """An admitted owned workspace, or an explained refusal before streaming."""

    plan: GlobalMomentPlan | None
    reason: str | None
    estimated_peak_bytes: int


@njit(cache=True)
def _finite_bounded_1d(values):
    for i in range(values.shape[0]):
        value = values[i]
        if value != 0.0 and not _MIN_ABS <= abs(value) <= _MAX_ABS:
            return False
    return True


@njit(cache=True)
def _finite_bounded_2d(values):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value != 0.0 and not _MIN_ABS <= abs(value) <= _MAX_ABS:
                return False
    return True


@njit(cache=True)
def _accumulate_vector(out, bins, weights):
    active = 0
    for i in range(bins.shape[0]):
        index = bins[i]
        if index >= 0:
            out[index] += weights[i]
            active += 1
    return active


@njit(cache=True)
def _accumulate_histogram(out, left_bins, right_bins, weights):
    active = 0
    for i in range(left_bins.shape[0]):
        left = left_bins[i]
        right = right_bins[i]
        if left >= 0 and right >= 0:
            out[left, right] += weights[i]
            active += 1
    return active


@njit(cache=True)
def _accumulate_directional(out, bins, ordinary, weights):
    active = 0
    for i in range(bins.shape[0]):
        index = bins[i]
        if index >= 0:
            weight = weights[i]
            for j in range(ordinary.shape[1]):
                out[index, j] += weight * ordinary[i, j]
            active += 1
    return active


def _warmup_global_moments():
    """Compile bounded writer/reducer signatures without constructing a plan."""
    _warmup_batched_moments()
    _warmup_ordinary_packing()
    for layout in ("C", "F", "A"):
        for readonly in (False, True):
            values = (
                np.ones((3, 4), dtype=np.float64)[:, ::2]
                if layout == "A"
                else np.ones((3, 2), dtype=np.float64, order=layout)
            )
            values.flags.writeable = not readonly
            _finite_bounded_2d(values)
    bins = np.array([0, -1, 1], dtype=np.intp)
    panel = np.ones((3, 2), dtype=np.float64)
    for strided in (False, True):
        step = 2 if strided else 1
        for readonly in (False, True):
            weights = np.ones(3 * step, dtype=np.float64)[::step]
            codes = np.zeros(3 * step, dtype=np.intp)[::step]
            weights.flags.writeable = not readonly
            codes.flags.writeable = not readonly
            _finite_bounded_1d(weights)
            _pack_categorical(panel, codes, 0, 2)
            _accumulate_vector(np.zeros(2), bins, weights)
            _accumulate_histogram(np.zeros((2, 2)), bins, bins, weights)
            _accumulate_directional(np.zeros((2, 2)), bins, panel, weights)


@dataclass
class _Support:
    predictor: int
    group_index: int
    group_type: type
    columns: np.ndarray
    basis: np.ndarray
    transform: np.ndarray
    solver_support: np.ndarray

    @property
    def n_bins(self):
        return self.basis.shape[0]


@dataclass
class _Predictor:
    intercept: bool
    width: int
    ordinary_columns: np.ndarray
    groups: tuple
    support_slots: tuple[int, ...]


def _array(values, shape, dtype, label):
    if type(values) is not np.ndarray or values.dtype != dtype or values.shape != shape:
        raise GlobalMomentRefusalError(f"{label}: expected ndarray {shape} with dtype {dtype}")
    return values


def _small_finite(values, label):
    if not np.all(np.isfinite(values)):
        raise GlobalMomentRefusalError(f"{label}: values must be finite")


def _index_values(values, count, upper, label):
    _array(values, (count,), np.dtype(np.intp), label)
    if values.size and (int(values.min()) < 0 or int(values.max()) >= upper):
        raise GlobalMomentRefusalError(f"{label}: index outside [0, {upper})")
    return values


def _metadata(layout, *, check_values=True):
    if type(layout) is not StackedLayout or not 1 <= len(layout.predictors) <= 2:
        raise GlobalMomentRefusalError("global moments accept one or two StackedLayout predictors")
    if any(
        type(state) is not PredictorState or type(state.design) is not DesignMatrix
        for state in layout.predictors
    ):
        raise GlobalMomentRefusalError("predictors and designs must have exact built-in types")
    n = layout.predictors[0].design.n
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= 2**32:
        raise GlobalMomentRefusalError("row count outside global moment domain")
    predictors, supports = [], []
    expected_start = 0
    for a, state in enumerate(layout.predictors):
        design = state.design
        # Bound Python descriptors before constructing quadratic pair specs.
        # This initial envelope is intentionally narrower than DesignMatrix.
        if len(design.group_matrices) > 64:
            raise GlobalMomentRefusalError("global moments accept at most 64 groups per predictor")
        intercept = state.intercept_index is not None
        start, stop = state.coefficient_slice.start, state.coefficient_slice.stop
        if (
            design.n != n
            or start != expected_start
            or state.parameter_index != a
            or stop - start != design.p + int(intercept)
            or (intercept and state.intercept_index != start)
        ):
            raise GlobalMomentRefusalError("inconsistent predictor layout")
        ordinary_columns = [start] if intercept else []
        groups, slots = [], []
        offset, ordinary_start = start + int(intercept), int(intercept)
        for group_index, group in enumerate(design.group_matrices):
            group_type = type(group)
            if group_type not in _ORDINARY_TYPES + _SUPPORT_TYPES:
                raise GlobalMomentRefusalError(
                    f"unsupported exact group type {group_type.__name__}"
                )
            if len(group.shape) != 2 or group.shape[0] != n:
                raise GlobalMomentRefusalError("group row count disagrees with layout")
            width = group.shape[1]
            if not isinstance(width, (int, np.integer)) or not 0 <= width <= 32:
                raise GlobalMomentRefusalError("group solver width outside [0, 32]")
            columns = tuple(range(offset, offset + width))
            if group_type in _ORDINARY_TYPES:
                if group_type is DenseGroupMatrix:
                    group = cast(DenseGroupMatrix, group)
                    _array(group.M, (n, width), np.dtype(np.float64), "dense source")
                else:
                    group = cast(CategoricalGroupMatrix, group)
                    if group.n_levels != width:
                        raise GlobalMomentRefusalError("categorical width disagrees with n_levels")
                    _array(group.codes, (n,), np.dtype(np.intp), "categorical source codes")
                groups.append((group_type, width, ordinary_start, None))
                ordinary_columns.extend(columns)
                ordinary_start += width
            else:
                group = cast(
                    "DiscretizedSSPGroupMatrix | DiscretizedSplineCategoricalGroupMatrix", group
                )
                basis, transform = group.B_unique, group.R_inv
                if (
                    type(basis) is not np.ndarray
                    or basis.ndim != 2
                    or type(transform) is not np.ndarray
                    or transform.ndim != 2
                ):
                    raise GlobalMomentRefusalError("support and solver map must be matrices")
                m, raw_width = basis.shape
                if not 1 <= m <= 4096 or not 1 <= raw_width <= 64 or width == 0:
                    raise GlobalMomentRefusalError(
                        "support dimensions outside global moment domain"
                    )
                _array(basis, (m, raw_width), np.dtype(np.float64), "support basis")
                _array(transform, (raw_width, width), np.dtype(np.float64), "solver map")
                if group.n_bins != m:
                    raise GlobalMomentRefusalError("n_bins disagrees with support basis")
                if check_values:
                    _small_finite(basis, "support basis")
                    _small_finite(transform, "solver map")
                if group_type is DiscretizedSSPGroupMatrix:
                    group = cast(DiscretizedSSPGroupMatrix, group)
                    _array(group.bin_idx, (n,), np.dtype(np.intp), "source support bins")
                else:
                    group = cast(DiscretizedSplineCategoricalGroupMatrix, group)
                    rows = group.row_idx
                    if type(rows) is not np.ndarray or rows.ndim != 1 or rows.size > n:
                        raise GlobalMomentRefusalError(
                            "source activity rows must be an exact vector"
                        )
                    _array(rows, rows.shape, np.dtype(np.intp), "source activity rows")
                    _array(
                        group.bin_idx_level,
                        rows.shape,
                        np.dtype(np.intp),
                        "source active support bins",
                    )
                slot = len(supports)
                supports.append((a, group_index, group_type, columns, basis, transform))
                groups.append((group_type, width, None, slot))
                slots.append(slot)
            offset += width
        if offset != stop:
            raise GlobalMomentRefusalError("group columns disagree with predictor width")
        predictors.append(
            (intercept, stop - start, tuple(ordinary_columns), tuple(groups), tuple(slots))
        )
        expected_start = stop
    if expected_start != layout.n_coefficients or not 1 <= expected_start <= 4096:
        raise GlobalMomentRefusalError("global coefficient width outside global moment domain")
    return int(n), predictors, supports


def _workspace_spec(n, p, predictor_meta, support_meta, *, chunk_size, byte_budget):
    """Describe construction and peak bytes using only validated dimensions."""
    c = int(chunk_size)
    k, s = len(predictor_meta), len(support_meta)
    ordinary_widths = [len(meta[2]) for meta in predictor_meta]
    # Specs contain support slots / predictor slots and packed channel index.
    hist_specs, mass_specs, directional_specs, ordinary_specs = [], [], [], []
    for channel, (a, b) in enumerate(packed_pairs(k)):
        left_slots, right_slots = predictor_meta[a][4], predictor_meta[b][4]
        ordinary_specs.append((a, b, channel))
        if a == b:
            hist_specs.extend((g, h, channel) for g, h in combinations(left_slots, 2))
            mass_specs.extend((g, channel) for g in left_slots)
        else:
            hist_specs.extend((g, h, channel) for g in left_slots for h in right_slots)
        if ordinary_widths[b]:
            directional_specs.extend((g, b, channel) for g in left_slots)
        if a != b and ordinary_widths[a]:
            directional_specs.extend((h, a, channel) for h in right_slots)
    bins = [meta[4].shape[0] for meta in support_meta]
    hist_bytes = 8 * sum(bins[g] * bins[h] for g, h, _ in hist_specs)
    mass_bytes = 8 * sum(bins[g] for g, _ in mass_specs)
    direction_bytes = 8 * sum(bins[g] * ordinary_widths[a] for g, a, _ in directional_specs)
    score_bytes = 8 * (sum(bins) + sum(ordinary_widths))
    ordinary_bytes = 8 * sum(ordinary_widths[a] * ordinary_widths[b] for a, b, _ in ordinary_specs)
    support_bytes = sum(meta[4].nbytes + meta[5].nbytes for meta in support_meta)
    solver_support_bytes = 8 * sum(meta[4].shape[0] * meta[5].shape[1] for meta in support_meta)
    int_bytes = np.dtype(np.intp).itemsize
    scratch_bytes = (
        8 * c * (sum(ordinary_widths) + max(ordinary_widths, default=0)) + int_bytes * c * s
    )
    indices_bytes = int_bytes * (sum(ordinary_widths) + sum(len(meta[3]) for meta in support_meta))
    accumulator_bytes = hist_bytes + mass_bytes + direction_bytes + score_bytes + ordinary_bytes
    coefficient_bytes = 8 * (p * p + p)
    batch_bytes, batch_metadata_reserve = _batch_workspace_bytes(
        len(hist_specs), len(directional_specs), k, vector_count=s
    )
    persistent_bytes = (
        accumulator_bytes
        + support_bytes
        + solver_support_bytes
        + scratch_bytes
        + indices_bytes
        + coefficient_bytes
        + batch_bytes
    )
    # Includes simultaneous local and readonly result copies, penalty
    # validation, exact-symmetry checks, and worst NumPy boolean temporaries.
    output_reserve = 8 * (12 * p * p + 12 * p)
    finish_scratch = 0
    construction_scratch = 0
    for meta in support_meta:
        m, d = meta[4].shape
        r = meta[5].shape[1]
        o = max(ordinary_widths, default=0)
        # Conservative allowance for operand packing during the one-time
        # B @ R transform; the resulting support is persistent state.
        construction_scratch = max(construction_scratch, 8 * (m * d + d * r + m * r))
        finish_scratch = max(
            finish_scratch,
            8 * (2 * m * r + 4 * r * r + 2 * r * o),
        )
    for g, h, _ in hist_specs:
        left, right = support_meta[g], support_meta[h]
        m_right = right[4].shape[0]
        r_left, r_right = left[5].shape[1], right[5].shape[1]
        # T_left.T @ H first produces solver_width_left by m_right;
        # unequal supports cannot be bounded from either marginal alone.
        finish_scratch = max(
            finish_scratch,
            8 * (2 * r_left * m_right + 4 * r_left * r_right),
        )
    finish_scratch += max((bins[g] * bins[h] for g, h, _ in hist_specs), default=0)
    group_count = sum(len(meta[3]) for meta in predictor_meta)
    metadata_reserve = 65536 + 4096 * (
        k + s + group_count + len(hist_specs) + len(directional_specs) + len(mass_specs)
    )
    metadata_reserve += batch_metadata_reserve
    # Index validation / array_equal use bounded temporary booleans.  A
    # masked assignment may also form a chunk-length advanced-index buffer.
    validation_reserve = 24 * c + max(
        (meta[4].size + meta[5].size for meta in support_meta), default=0
    )
    estimate = int(
        persistent_bytes
        + output_reserve
        + construction_scratch
        + finish_scratch
        + metadata_reserve
        + validation_reserve
    )
    accounting = dict(
        accumulator_bytes=accumulator_bytes,
        histogram_bytes=hist_bytes,
        diagonal_mass_bytes=mass_bytes,
        directional_bytes=direction_bytes,
        score_accumulator_bytes=score_bytes,
        ordinary_curvature_bytes=ordinary_bytes,
        support_authority_bytes=support_bytes,
        solver_support_bytes=solver_support_bytes,
        row_scratch_bytes=scratch_bytes,
        coefficient_state_bytes=coefficient_bytes,
        layout_index_bytes=indices_bytes,
        batched_reducer_bytes=batch_bytes,
        persistent_allocated_bytes=persistent_bytes,
        output_reserve_bytes=output_reserve,
        construction_scratch_bytes=construction_scratch,
        finalization_scratch_bytes=finish_scratch,
        metadata_reserve_bytes=metadata_reserve,
        validation_reserve_bytes=validation_reserve,
        estimated_peak_bytes=estimate,
        published_output_bytes=8 * (2 * p + 3 * p * p),
        histogram_count=len(hist_specs),
        directional_count=len(directional_specs),
        diagonal_count=len(mass_specs),
        support_count=s,
        ordinary_widths=ordinary_widths,
        chunk_size=c,
        byte_budget=int(byte_budget),
    )
    return (
        n,
        p,
        c,
        predictor_meta,
        support_meta,
        hist_specs,
        mass_specs,
        directional_specs,
        ordinary_specs,
        accounting,
    )


def build_global_moment_plan(layout, *, byte_budget=64 << 20, chunk_size):
    """Estimate all owned array state and conservative peak scratch before allocation.

    The cap covers this assembler's extra arrays, output copies and metadata
    reserve.  It excludes caller-owned likelihood chunks/designs, interpreter,
    native libraries/JIT compiler and their global caches. No fallback runs.
    The explicit row bound is independent of this additional workspace cap.
    """
    estimate = 0
    try:
        if (
            isinstance(byte_budget, (bool, np.bool_))
            or not isinstance(byte_budget, (int, np.integer))
            or byte_budget <= 0
            or isinstance(chunk_size, (bool, np.bool_))
            or not isinstance(chunk_size, (int, np.integer))
            or chunk_size <= 0
        ):
            raise GlobalMomentRefusalError("positive integer byte_budget and chunk_size required")
        n, predictor_meta, support_meta = _metadata(layout)
        spec = _workspace_spec(
            n,
            layout.n_coefficients,
            predictor_meta,
            support_meta,
            chunk_size=chunk_size,
            byte_budget=byte_budget,
        )
        estimate = spec[-1]["estimated_peak_bytes"]
        if estimate > byte_budget:
            raise GlobalMomentRefusalError(
                f"estimated peak {estimate} exceeds byte_budget {byte_budget}"
            )
        plan = GlobalMomentPlan(*spec)
        return GlobalMomentBuild(plan, None, estimate)
    except (GlobalMomentRefusalError, MemoryError) as exc:
        return GlobalMomentBuild(None, str(exc) or "allocation failed", estimate)


def global_moment_chunk_size(layout, *, byte_budget, minimum_chunk_size, maximum_chunk_size=65536):
    """Increase an admitted automatic row bound only within the owned budget.

    This metadata-only selector owns no plan and scans no numeric arrays. Its
    caller supplies the existing row bound and handles family/scope admission.
    The builder independently revalidates all sources before allocation; this
    estimate grants no numerical authority or persistent source certificate.
    Caller-owned likelihood arrays remain outside the assembler's byte cap.
    """
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
        for value in (byte_budget, minimum_chunk_size, maximum_chunk_size)
    ):
        return minimum_chunk_size
    minimum, maximum = int(minimum_chunk_size), int(maximum_chunk_size)
    if maximum <= minimum:
        return minimum
    try:
        n, predictor_meta, support_meta = _metadata(layout, check_values=False)
        maximum = min(n, maximum)
        if maximum <= minimum:
            return minimum

        def peak(rows):
            return _workspace_spec(
                n,
                layout.n_coefficients,
                predictor_meta,
                support_meta,
                chunk_size=rows,
                byte_budget=byte_budget,
            )[-1]["estimated_peak_bytes"]

        # Every row-dependent workspace term is affine in the chunk size.
        # Derive both terms from the shared construction estimate, so changing
        # its scratch accounting cannot leave a duplicate selector formula.
        one_row = peak(1)
        bytes_per_row = peak(2) - one_row
        selected = min(maximum, 1 + (int(byte_budget) - one_row) // bytes_per_row)
        return max(minimum, selected)
    except (GlobalMomentRefusalError, MemoryError):
        return minimum


class GlobalMomentPlan:
    """Own one geometry's sufficient statistics and bounded row workspace.

    Construct through ``build_global_moment_plan``. The caller supplies each
    observation once; row identity/order belongs to the likelihood stream.
    Reset clears every global accumulator. No partial result survives refusal.
    """

    def __init__(
        self,
        n,
        width,
        chunk_size,
        predictor_meta,
        support_meta,
        hist_specs,
        mass_specs,
        directional_specs,
        ordinary_specs,
        accounting,
    ):
        self._n, self._width, self._chunk_size = n, width, chunk_size
        self._predictors = tuple(
            _Predictor(m[0], m[1], np.array(m[2], dtype=np.intp), m[3], m[4])
            for m in predictor_meta
        )
        supports = []
        for m in support_meta:
            basis = np.array(m[4], copy=True, order="C")
            transform = np.array(m[5], copy=True, order="C")
            # Evaluate the stored solver design on marginal support once.
            # Delaying R until after raw weighted moments can underflow before
            # a large solver-map scale restores a representable final value.
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                table = basis @ transform
            if not _finite_bounded_2d(table):
                raise GlobalMomentRefusalError(
                    "solver support outside the numerical domain", recoverable=True
                )
            supports.append(
                _Support(m[0], m[1], m[2], np.array(m[3], dtype=np.intp), basis, transform, table)
            )
        self._supports = tuple(supports)
        for support in self._supports:
            support.columns.setflags(write=False)
            support.basis.setflags(write=False)
            support.transform.setflags(write=False)
            support.solver_support.setflags(write=False)
        for predictor in self._predictors:
            predictor.ordinary_columns.setflags(write=False)
        self._histograms = [
            (g, h, channel, np.empty((self._supports[g].n_bins, self._supports[h].n_bins)))
            for g, h, channel in hist_specs
        ]
        self._masses = [
            (g, channel, np.empty(self._supports[g].n_bins)) for g, channel in mass_specs
        ]
        self._directions = [
            (
                g,
                a,
                channel,
                np.empty((self._supports[g].n_bins, len(self._predictors[a].ordinary_columns))),
            )
            for g, a, channel in directional_specs
        ]
        self._ordinary_blocks = [
            (
                a,
                b,
                channel,
                np.empty(
                    (
                        len(self._predictors[a].ordinary_columns),
                        len(self._predictors[b].ordinary_columns),
                    )
                ),
            )
            for a, b, channel in ordinary_specs
        ]
        self._support_scores = [np.empty(g.n_bins) for g in self._supports]
        self._ordinary_scores = [np.empty(len(p.ordinary_columns)) for p in self._predictors]
        self._ordinary = [np.empty((chunk_size, len(p.ordinary_columns))) for p in self._predictors]
        self._weighted = np.empty(
            (chunk_size, max(len(p.ordinary_columns) for p in self._predictors))
        )
        self._bins = np.empty((len(self._supports), chunk_size), dtype=np.intp)
        self._batched_moments = _BatchedMomentReducers(
            self._histograms,
            self._directions,
            self._ordinary,
            self._bins,
            support_sizes=tuple(support.n_bins for support in self._supports),
            n_channels=len(self._predictors) * (len(self._predictors) + 1) // 2,
            vectors=tuple(
                (g, self._supports[g].predictor, channel, self._support_scores[g], mass)
                for g, channel, mass in self._masses
            ),
            n_score_channels=len(self._predictors),
        )
        self._coefficients = np.empty(width)
        self._penalty = np.empty((width, width))
        self._accumulators = (
            [
                x[-1]
                for x in self._histograms + self._masses + self._directions + self._ordinary_blocks
            ]
            + self._support_scores
            + self._ordinary_scores
        )
        self._stats = dict(accounting)
        owned_arrays = (
            self._accumulators
            + self._ordinary
            + [self._weighted, self._bins, self._coefficients, self._penalty]
            + list(self._batched_moments.owned_arrays)
            + [p.ordinary_columns for p in self._predictors]
            + [
                array
                for g in self._supports
                for array in (g.columns, g.basis, g.transform, g.solver_support)
            ]
        )
        actual_bytes = sum(array.nbytes for array in owned_arrays)
        if actual_bytes != accounting["persistent_allocated_bytes"]:
            raise GlobalMomentRefusalError(
                "owned array allocation disagrees with workspace estimate"
            )
        self._stats["persistent_allocated_bytes"] = actual_bytes
        self._stats["persistent_array_count"] = len(owned_arrays)
        self._stats.update(
            {
                name: 0
                for name in (
                    "reset_count",
                    "chunks",
                    "rows",
                    "histogram_zeroings",
                    "histogram_update_calls",
                    "histogram_row_visits",
                    "histogram_active_updates",
                    "directional_update_calls",
                    "directional_row_width_work",
                    "mass_update_calls",
                    "score_update_calls",
                    "ordinary_curvature_products",
                    "ordinary_score_products",
                    "support_pair_finalizations",
                    "diagonal_finalizations",
                    "directional_finalizations",
                    "finish_count",
                    "global_zeroed_bytes",
                    "refusal_count",
                    "support_authority_checks",
                    "batched_moment_calls",
                    "native_moment_workers",
                )
            }
        )
        self._stats["constructor_count"] = 1
        self._stats["solver_support_builds"] = len(self._supports)
        self._state = "built"
        self._rows = 0

    @property
    def stats(self):
        result = dict(
            self._stats,
            state=self._state,
            current_geometry_rows=self._rows,
            current_owned_bytes=0
            if self._state == "closed"
            else self._stats["persistent_allocated_bytes"],
        )
        result["ordinary_widths"] = list(cast("list[int]", result["ordinary_widths"]))
        return result

    def _refuse(self, reason, *, recoverable=False):
        if self._state != "closed":
            self._state = "refused"
        self._stats["refusal_count"] += 1
        raise GlobalMomentRefusalError(reason, recoverable=recoverable)

    def reset(self, *, coefficients, penalty):
        if self._state == "closed":
            self._refuse("closed plan cannot reset")
        try:
            _array(coefficients, (self._width,), np.dtype(np.float64), "coefficients")
            _small_finite(coefficients, "coefficients")
            _array(penalty, (self._width, self._width), np.dtype(np.float64), "penalty")
            _small_finite(penalty, "penalty")
            checked_penalty = validated_dense_penalty(penalty, self._width)
        except GlobalMomentRefusalError as exc:
            self._refuse(exc.reason, recoverable=exc.recoverable)
        except ValueError as exc:
            self._refuse(str(exc))
        cast(np.ndarray, self._coefficients)[:] = coefficients
        cast(np.ndarray, self._penalty)[:] = checked_penalty
        for accumulator in self._accumulators:
            accumulator.fill(0.0)
        self._rows = 0
        self._state = "accumulating"
        self._stats["reset_count"] += 1
        self._stats["histogram_zeroings"] += len(self._histograms)
        self._stats["global_zeroed_bytes"] += self._stats["accumulator_bytes"]

    def _prepare_chunk(self, plans, score, curvature, *, row_range=None):
        k = len(self._predictors)
        if not isinstance(plans, (tuple, list)) or len(plans) != k:
            raise GlobalMomentRefusalError("one chunk plan per predictor required")
        if type(score) is not np.ndarray or score.ndim != 2:
            raise GlobalMomentRefusalError("score must be a float64 matrix")
        n = score.shape[0]
        if not 1 <= n <= self._chunk_size or self._rows + n > self._n:
            raise GlobalMomentRefusalError("chunk exceeds workspace or remaining geometry rows")
        source_n = n
        selection = slice(None)
        if row_range is not None:
            start, stop = row_range
            if (
                type(start) is not int
                or type(stop) is not int
                or start != self._rows
                or not 0 <= start < stop <= self._n
                or stop - start != n
            ):
                raise GlobalMomentRefusalError(
                    "row range must be contiguous and match channel rows"
                )
            source_n = self._n
            selection = slice(start, stop)
        _array(score, (n, k), np.dtype(np.float64), "score channels")
        _array(curvature, (n, k * (k + 1) // 2), np.dtype(np.float64), "curvature channels")
        numerical_refusal = None
        if not _finite_bounded_2d(score) or not _finite_bounded_2d(curvature):
            numerical_refusal = "nonfinite or out-of-domain likelihood channel"
        # Complete validation/preparation before writing ANY persistent moment.
        # A recoverable numeric refusal must not conceal a hard source error
        # later in this same chunk. Keep their priority in this single pass.
        for a, (live, owned) in enumerate(zip(plans, self._predictors, strict=True)):
            if (
                type(live) is not PredictorExecutionPlan
                or type(live.design) is not DesignMatrix
                or live.intercept != owned.intercept
                or live.width != owned.width
                or live.design.n != source_n
                or len(live.design.group_matrices) != len(owned.groups)
            ):
                raise GlobalMomentRefusalError("chunk predictor layout/type/order mismatch")
            panel = self._ordinary[a][:n]
            if owned.intercept:
                panel[:, 0] = 1.0
            for group, meta in zip(live.design.group_matrices, owned.groups, strict=True):
                group_type, width, ordinary_start, slot = meta
                if type(group) is not group_type or group.shape != (source_n, width):
                    raise GlobalMomentRefusalError("chunk group type/order/width mismatch")
                if group_type is DenseGroupMatrix:
                    group = cast(DenseGroupMatrix, group)
                    values = _array(
                        group.M, (source_n, width), np.dtype(np.float64), "dense source"
                    )[selection]
                    if not _copy_dense_checked(panel, values, ordinary_start):
                        numerical_refusal = (
                            numerical_refusal or "nonfinite or out-of-domain ordinary value"
                        )
                elif group_type is CategoricalGroupMatrix:
                    group = cast(CategoricalGroupMatrix, group)
                    if group.n_levels != width:
                        raise GlobalMomentRefusalError("categorical n_levels mismatch")
                    codes = _array(
                        group.codes, (source_n,), np.dtype(np.intp), "categorical source codes"
                    )[selection]
                    _index_values(codes, n, width + 1, "categorical codes")
                    _pack_categorical(panel, codes, ordinary_start, width)
                else:
                    group = cast(
                        "DiscretizedSSPGroupMatrix | DiscretizedSplineCategoricalGroupMatrix", group
                    )
                    support = self._supports[slot]
                    basis, transform = support.basis, support.transform
                    _array(group.B_unique, basis.shape, basis.dtype, "live support basis")
                    _array(group.R_inv, transform.shape, transform.dtype, "live solver map")
                    if (
                        group.n_bins != support.n_bins
                        or not np.array_equal(group.B_unique, basis)
                        or not np.array_equal(group.R_inv, transform)
                    ):
                        raise GlobalMomentRefusalError(
                            "live support basis or solver map mismatches authority"
                        )
                    self._stats["support_authority_checks"] += 1
                    target = cast(np.ndarray, self._bins)[slot, :n]
                    if group_type is DiscretizedSSPGroupMatrix:
                        group = cast(DiscretizedSSPGroupMatrix, group)
                        bins = _array(
                            group.bin_idx, (source_n,), np.dtype(np.intp), "source support bins"
                        )[selection]
                        _index_values(bins, n, support.n_bins, "support bins")
                        target[:] = bins
                    else:
                        group = cast(DiscretizedSplineCategoricalGroupMatrix, group)
                        rows = group.row_idx
                        if type(rows) is not np.ndarray or rows.ndim != 1 or rows.size > source_n:
                            raise GlobalMomentRefusalError(
                                "activity rows must be a vector bounded by the source rows"
                            )
                        _array(rows, rows.shape, np.dtype(np.intp), "source activity rows")
                        bins = _array(
                            group.bin_idx_level,
                            rows.shape,
                            np.dtype(np.intp),
                            "source active support bins",
                        )
                        if row_range is not None:
                            extracted = _category_rows(group, start, stop)
                            if extracted is None:
                                # Replay the ordinary row_subset interpretation
                                # after releasing this workspace. Keep checking
                                # later sources so a hard error wins over this
                                # recoverable dispatch refusal.
                                numerical_refusal = (
                                    numerical_refusal or "uncertified category row lookup"
                                )
                                continue
                            rows, bins = extracted
                            del extracted
                            if type(rows) is not np.ndarray or rows.ndim != 1 or rows.size > n:
                                raise GlobalMomentRefusalError(
                                    "activity rows must be a vector bounded by the chunk rows"
                                )
                        _index_values(rows, len(rows), n, "activity rows")
                        if len(rows) > 1 and np.any(rows[1:] <= rows[:-1]):
                            raise GlobalMomentRefusalError(
                                "activity rows must be strictly increasing"
                            )
                        _index_values(bins, len(rows), support.n_bins, "active support bins")
                        target.fill(-1)
                        target[rows] = bins
                        # Release the two bounded extraction arrays before
                        # the next category allocates its row/bin pair.
                        del rows, bins
        if numerical_refusal is not None:
            raise GlobalMomentRefusalError(numerical_refusal, recoverable=True)
        return n

    def add_chunk(self, plans, score_eta, curvature_packed):
        if self._state != "accumulating":
            self._refuse("add_chunk requires a reset and an unrefused, unfinished geometry")
        try:
            n = self._prepare_chunk(plans, score_eta, curvature_packed)
        except GlobalMomentRefusalError as exc:
            self._refuse(exc.reason, recoverable=exc.recoverable)
        except (ValueError, IndexError, FloatingPointError) as exc:
            self._refuse(str(exc))
        self._accumulate_prepared(n, score_eta, curvature_packed)

    def add_row_range(self, plans, start, stop, score_eta, curvature_packed):
        """Ingest contiguous original-design rows using the owned chunk buffers.

        Live numeric arrays are checked before slicing. Small B/R authority
        tables are compared exactly on each call; source-sized row lookup
        caches retain the ordinary row_subset contract and belong to the
        caller's groups. The usual certified path constructs no child designs.
        """
        if self._state != "accumulating":
            self._refuse("add_row_range requires a reset and an unrefused, unfinished geometry")
        try:
            n = self._prepare_chunk(plans, score_eta, curvature_packed, row_range=(start, stop))
        except GlobalMomentRefusalError as exc:
            self._refuse(exc.reason, recoverable=exc.recoverable)
        except (ValueError, IndexError, FloatingPointError) as exc:
            self._refuse(str(exc))
        self._accumulate_prepared(n, score_eta, curvature_packed)

    def _accumulate_prepared(self, n, score_eta, curvature_packed):
        """Update moments only after complete source and channel validation."""
        try:
            panels = [panel[:n] for panel in self._ordinary]
            for a, panel in enumerate(panels):
                self._ordinary_scores[a] += panel.T @ score_eta[:, a]
                self._stats["ordinary_score_products"] += 1
            for a, b, channel, accumulator in self._ordinary_blocks:
                width = panels[b].shape[1]
                weighted = cast(np.ndarray, self._weighted)[:n, :width]
                np.multiply(panels[b], curvature_packed[:, channel, None], out=weighted)
                accumulator += panels[a].T @ weighted
                self._stats["ordinary_curvature_products"] += 1
            batched_moments = cast(_BatchedMomentReducers, self._batched_moments)
            active, directional_work = batched_moments.accumulate(
                curvature_packed, n, score=score_eta
            )
            self._stats["score_update_calls"] += len(self._supports)
            self._stats["mass_update_calls"] += len(self._masses)
            self._stats["batched_moment_calls"] += 1
            self._stats["native_moment_workers"] = max(
                self._stats["native_moment_workers"], batched_moments.last_worker_count
            )
            self._stats["histogram_update_calls"] += len(self._histograms)
            self._stats["histogram_row_visits"] += n * len(self._histograms)
            self._stats["histogram_active_updates"] += active
            self._stats["directional_update_calls"] += len(self._directions)
            self._stats["directional_row_width_work"] += directional_work
        except GlobalMomentRefusalError as exc:
            self._refuse(exc.reason, recoverable=exc.recoverable)
        except (ValueError, IndexError, FloatingPointError) as exc:
            self._refuse(str(exc))
        self._rows += n
        self._stats["chunks"] += 1
        self._stats["rows"] += n

    def finish(self):
        if self._state != "accumulating" or self._rows != self._n:
            self._refuse("finish requires one complete, unrefused geometry")
        try:
            for accumulator in self._accumulators:
                if not np.all(np.isfinite(accumulator)):
                    raise GlobalMomentRefusalError("nonfinite accumulated moment", recoverable=True)
            score = np.zeros(self._width)
            curvature = np.zeros((self._width, self._width))

            def put(left, right, block, diagonal=False):
                if diagonal:
                    # Averaging the two arithmetic orders agrees with existing
                    # dense/grouped assembly and ensures exact output symmetry.
                    block = 0.5 * (block + block.T)
                curvature[np.ix_(left, right)] = block
                if not diagonal:
                    curvature[np.ix_(right, left)] = block.T

            for a, predictor in enumerate(self._predictors):
                score[predictor.ordinary_columns] = self._ordinary_scores[a]
            for g, support in enumerate(self._supports):
                score[support.columns] = support.solver_support.T @ self._support_scores[g]
            for a, b, _, accumulator in self._ordinary_blocks:
                put(
                    self._predictors[a].ordinary_columns,
                    self._predictors[b].ordinary_columns,
                    accumulator,
                    diagonal=a == b,
                )
            for g, _, accumulator in self._masses:
                support = self._supports[g]
                table = support.solver_support
                block = table.T @ (accumulator[:, None] * table)
                put(support.columns, support.columns, block, diagonal=True)
                self._stats["diagonal_finalizations"] += 1
            for g, h, _, accumulator in self._histograms:
                left, right = self._supports[g], self._supports[h]
                block = left.solver_support.T @ accumulator @ right.solver_support
                put(left.columns, right.columns, block)
                self._stats["support_pair_finalizations"] += 1
            for g, a, _, accumulator in self._directions:
                support = self._supports[g]
                block = support.solver_support.T @ accumulator
                put(support.columns, self._predictors[a].ordinary_columns, block)
                self._stats["directional_finalizations"] += 1
            penalty = cast(np.ndarray, self._penalty)
            coefficients = cast(np.ndarray, self._coefficients)
            score_penalized = score - penalty @ coefficients
            penalized = curvature + penalty
            if not all(
                np.all(np.isfinite(x)) for x in (score, score_penalized, curvature, penalized)
            ):
                raise GlobalMomentRefusalError(
                    "nonfinite coefficient-space geometry", recoverable=True
                )
            result = DenseJointGeometry(score, score_penalized, curvature, penalty, penalized)
        except GlobalMomentRefusalError as exc:
            self._refuse(exc.reason, recoverable=exc.recoverable)
        except (ValueError, FloatingPointError) as exc:
            self._refuse(str(exc))
        self._state = "finished"
        self._stats["finish_count"] += 1
        return result

    def close(self):
        """Release owned arrays; returned DenseJointGeometry owns independent copies."""
        self._batched_moments = None
        self._accumulators = []
        self._histograms = self._masses = self._directions = self._ordinary_blocks = []
        self._support_scores = self._ordinary_scores = self._ordinary = []
        self._supports = self._predictors = ()
        self._weighted = self._bins = self._coefficients = self._penalty = None
        self._state = "closed"
