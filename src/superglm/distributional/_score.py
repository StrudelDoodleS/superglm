"""Read a coefficient score per unit change in the linear predictors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES, DenseGroupMatrix
from superglm._group_matrix._group_matrix_range import group_row_range
from superglm.distributional.layout import StackedLayout


def _group_limits(group, chunk_size, weights, with_intercept):
    """Column extrema after scaling each prior-weighted contrast by sqrt(w/max w)."""
    if type(group) is DenseGroupMatrix and weights is None:
        return np.min(group.M, axis=0), np.max(group.M, axis=0)
    rows, width = group.shape
    if type(group) in ONE_HOT_CODE_TYPES:
        # An indicator has a declared contrast (absence to presence) in either
        # weighting. Count its support without expanding the categorical design.
        counts = np.bincount(group.codes, minlength=width + 1)[:width]
        return (counts == rows).astype(float), (counts > 0).astype(float)
    chunk_size = rows if chunk_size is None else chunk_size

    def blocks():
        for start in range(0, rows, chunk_size):
            stop = min(start + chunk_size, rows)
            block = (
                group if stop - start == rows else group_row_range(group, start, stop, copy=False)
            )
            if block is None:
                block = group.row_subset(np.arange(start, stop))
            yield slice(start, stop), block

    centred = weights is not None and with_intercept
    origin = np.zeros(width)
    if centred:
        for section, block in blocks():
            origin += block.rmatvec(weights.values[section] / weights.weight_sum)
    low, high = np.full(width, np.inf), np.full(width, -np.inf)
    raw_low, raw_high = low.copy(), high.copy()
    unit = np.zeros(width)
    for section, block in blocks():
        factor = (
            None
            if weights is None
            else np.sqrt(weights.values[section]) / np.sqrt(weights.provenance.max_weight)
        )
        for index in range(width):
            unit[index] = 1.0
            column = block.M[:, index] if type(block) is DenseGroupMatrix else block.matvec(unit)
            if factor is not None:
                if centred:
                    raw_low[index] = min(raw_low[index], np.min(column))
                    raw_high[index] = max(raw_high[index], np.max(column))
                # Interpolate about the weighted mean (zero without an intercept),
                # so changing a numeric column's origin cannot change its radius.
                column = factor * column + (1.0 - factor) * origin[index]
            low[index] = min(low[index], np.min(column))
            high[index] = max(high[index], np.max(column))
            unit[index] = 0.0
    if centred:
        # Every interpolation lies in the column's convex hull. In particular
        # a rounded weighted mean must not give a constant column a spread.
        return np.clip(low, raw_low, raw_high), np.clip(high, raw_low, raw_high)
    return low, high


@dataclass(frozen=True, eq=False)
class ScoreUnits:
    centre: NDArray[np.float64]
    spread: NDArray[np.float64]
    intercept: NDArray[np.intp]

    @classmethod
    def from_layout(cls, layout: StackedLayout, dense_matrices, chunk_size, weights) -> ScoreUnits:
        """Representative units of the fixed design, shared within one fit session.

        The compiler whitens an SSP spline using its normalized Gram plus penalty.
        Keep those coordinates: a rare large spline row should not shrink its
        score. Read a numeric column's radius in the weighted design: multiply
        deviations from its weighted mean by sqrt(w/max w). Equal prior weights
        recover the ordinary range. Frequency weights repeat observations, which
        leaves their extrema unchanged. Without an intercept the origin is zero.
        """
        centre = np.zeros(layout.n_coefficients)
        spread = np.ones(layout.n_coefficients)
        intercept = np.arange(layout.n_coefficients)
        # Fixed fits validate the complete weight plan. Its extrema and sum
        # already describe these rows; scale only the block currently in use.
        provenance = weights.provenance
        prior = (
            weights
            if provenance.contract.semantics == "prior"
            and provenance.min_weight != provenance.max_weight
            else None
        )
        for q, state in enumerate(layout.predictors):
            offset = int(state.intercept_index is not None)
            start = state.coefficient_slice.start + offset
            for group, block in zip(state.groups, state.design.group_matrices, strict=True):
                if group.ssp_spline:
                    continue
                slopes = slice(start + group.start, start + group.end)
                if dense_matrices is not None and type(block) not in ONE_HOT_CODE_TYPES:
                    columns = dense_matrices[q][:, offset + group.start : offset + group.end]
                    block = DenseGroupMatrix(columns)
                low, high = _group_limits(block, chunk_size, prior, bool(offset))
                low = np.asarray(low, dtype=np.float64)
                high = np.asarray(high, dtype=np.float64)
                if state.intercept_index is not None:
                    intercept[slopes] = state.intercept_index
                    centre[slopes] = 0.5 * low + 0.5 * high
                radius = np.maximum(high - centre[slopes], centre[slopes] - low)
                # A constant column has no data variation. Keep its own units for
                # a penalty score along that alias. An all-zero column uses one.
                scale = np.where(radius > 0.0, radius, np.abs(low))
                spread[slopes] = np.where(scale > 0.0, scale, 1.0)
        for values in (centre, spread, intercept):
            values.setflags(write=False)
        return cls(centre, spread, intercept)

    def gradient(self, score):
        """Chain rule for centred and rescaled coefficients, for vectors or columns."""
        if score.ndim == 1:
            return (score - self.centre * score[self.intercept]) / self.spread
        return (score - self.centre[:, None] * score[self.intercept]) / self.spread[:, None]

    def face_normals(self, face):
        """An orthonormal basis of the constraints in the standardised coordinates."""
        return np.linalg.qr(self.gradient(face.constraint_basis))[0]

    def step_relative(self, step, coefficients):
        """Measure both vectors in the coordinates dual to the scaled score."""

        def size(values):
            intercept_shift = np.bincount(
                self.intercept, weights=self.centre * values, minlength=len(values)
            )
            return np.max(np.abs(self.spread * values + intercept_shift))

        return float(size(step) / (1.0 + size(coefficients)))

    def relative(self, score, objective, *, face=None, normals=None):
        """Relative gradient with representative parameter scales (Dennis & Schnabel §7.2).

        On a face only tangential derivatives count. A constraint normal is a
        covector and transforms exactly as a score does. Remove a large normal
        component in the original coordinates before measuring the tangent
        in standardised units.
        """
        if face is not None:
            score = face.lift_vector(face.reduce_vector(score))
        scaled = self.gradient(score)
        if face is not None:
            if normals is None:
                normals = self.face_normals(face)
            scaled -= normals @ (normals.T @ scaled)
        return float(np.max(np.abs(scaled), initial=0.0) / (1.0 + abs(objective)))
