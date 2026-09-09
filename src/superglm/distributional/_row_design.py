"""Bounded dense row views of retained grouped predictor designs.

The intercept is implicit in the retained design. Only a requested row chunk
is expanded; no observation-by-coefficient matrix is retained by this adapter.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.layout import PredictorState, StackedLayout


@dataclass(frozen=True)
class BoundedPredictorMatrix:
    state: PredictorState
    chunk_size: int = 4096

    @property
    def shape(self) -> tuple[int, int]:
        block = self.state.coefficient_slice
        return self.state.design.n, block.stop - block.start

    def __getitem__(self, rows: slice) -> NDArray[np.float64]:
        if not isinstance(rows, slice):
            raise TypeError("bounded predictor matrices require a row slice")
        start, stop, step = rows.indices(self.shape[0])
        if step != 1 or stop - start > self.chunk_size:
            raise ValueError("predictor row request exceeds its bounded chunk size")
        indices = np.arange(start, stop, dtype=np.intp)
        matrix = np.empty((len(indices), self.shape[1]), dtype=np.float64)
        intercept = int(self.state.intercept_index is not None)
        if intercept:
            matrix[:, 0] = 1.0
        if self.state.design.p:
            matrix[:, intercept:] = self.state.design.row_subset(indices).toarray()
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"design for predictor {self.state.name!r} must be finite")
        return matrix


PredictorMatrix = NDArray[np.float64] | BoundedPredictorMatrix


def bounded_predictor_matrices(
    layout: StackedLayout, *, chunk_size: int = 4096
) -> tuple[BoundedPredictorMatrix, ...]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    return tuple(BoundedPredictorMatrix(state, chunk_size) for state in layout.predictors)


def matrix_row_chunk(matrices: tuple[PredictorMatrix, ...], default: int) -> int:
    return min(
        [default]
        + [matrix.chunk_size for matrix in matrices if isinstance(matrix, BoundedPredictorMatrix)]
    )
