"""Read a coefficient score per unit change in the linear predictors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES, DenseGroupMatrix
from superglm._group_matrix._group_matrix_range import group_row_range
from superglm.distributional.layout import StackedLayout


def _group_limits(group, chunk_size):
    """Column extrema using stored values or products within the fit's row bound."""
    if type(group) is DenseGroupMatrix:
        return np.min(group.M, axis=0), np.max(group.M, axis=0)
    rows, width = group.shape
    if type(group) in ONE_HOT_CODE_TYPES:
        counts = np.bincount(group.codes, minlength=width + 1)[:width]
        return (counts == rows).astype(float), (counts > 0).astype(float)
    low, high = np.full(width, np.inf), np.full(width, -np.inf)
    unit = np.zeros(width)
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        block = group if stop - start == rows else group_row_range(group, start, stop, copy=False)
        if block is None:
            block = group.row_subset(np.arange(start, stop))
        for index in range(width):
            unit[index] = 1.0
            column = block.matvec(unit)
            low[index] = min(low[index], np.min(column))
            high[index] = max(high[index], np.max(column))
            unit[index] = 0.0
    return low, high


@dataclass(frozen=True, eq=False)
class ScoreUnits:
    centre: NDArray[np.float64]
    spread: NDArray[np.float64]
    intercept: NDArray[np.intp]

    @classmethod
    def from_layout(cls, layout: StackedLayout, dense_matrices, chunk_size) -> ScoreUnits:
        """Representative units of the fixed design, shared within one fit session.

        The compiler whitens an SSP spline using its normalized Gram plus penalty.
        Keep those coordinates: a rare large spline row should not shrink its
        score. For columns supplied in their own units, a slope of 1 / radius
        moves the centred predictor by one. Without an intercept its origin is
        fixed at zero. Extrema avoid subtracting nearly equal column moments.
        """
        centre = np.zeros(layout.n_coefficients)
        spread = np.ones(layout.n_coefficients)
        intercept = np.arange(layout.n_coefficients)
        for q, state in enumerate(layout.predictors):
            offset = int(state.intercept_index is not None)
            start = state.coefficient_slice.start + offset
            for group, block in zip(state.groups, state.design.group_matrices, strict=True):
                if group.ssp_spline:
                    continue
                slopes = slice(start + group.start, start + group.end)
                if dense_matrices is None:
                    low, high = _group_limits(block, chunk_size)
                else:
                    columns = dense_matrices[q][:, offset + group.start : offset + group.end]
                    low, high = np.min(columns, axis=0), np.max(columns, axis=0)
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
