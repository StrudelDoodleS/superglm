"""Memory-bounded design operations for post-fit evaluation diagnostics."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.group_matrix import DesignMatrix

# Diagnostic algebra can keep the dense block, a shifted copy, a weighted copy,
# and matrix-product workspaces live at the same time.  Budget five row-sized
# buffers for those arrays.  Exact feature transforms also return one term-width
# block; feature-specific transform internals (for example a raw spline basis)
# are separate opaque workspaces, but share this bounded row dimension.
_MAX_DESIGN_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_DESIGN_CHUNK_ROWS = 8192
_LIVE_ALGEBRA_ROW_BUFFERS = 5


def _bounded_chunk_rows(*, algebra_width: int, transform_width: int = 0) -> int:
    """Return a row count bounded for explicit dense buffers in this module."""
    cells_per_row = _LIVE_ALGEBRA_ROW_BUFFERS * max(algebra_width, 1)
    cells_per_row += max(transform_width, 0)
    bytes_per_row = np.dtype(np.float64).itemsize * cells_per_row
    return max(
        1,
        min(_MAX_DESIGN_CHUNK_ROWS, _MAX_DESIGN_CHUNK_BYTES // bytes_per_row),
    )


def _as_dense_block(values, *, n_rows: int, n_columns: int, term_name: str) -> NDArray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    block = np.asarray(values, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, None]
    expected = (n_rows, n_columns)
    if block.shape != expected:
        raise ValueError(
            f"runtime transform for {term_name!r} returned {block.shape}, expected {expected}"
        )
    return block


def _exact_runtime_design_block(
    model,
    X: EagerFrame,
    selected_columns: NDArray,
) -> NDArray:
    """Evaluate selected frozen prediction-plan columns on one row chunk."""
    from superglm.model import base

    plan = base._prediction_plan(model)
    selected_columns = np.asarray(selected_columns, dtype=np.intp)
    selected_positions = {int(column): position for position, column in enumerate(selected_columns)}
    block = np.zeros((len(X), len(selected_columns)), dtype=np.float64)
    assigned = np.zeros(len(selected_columns), dtype=bool)

    for term in plan["features"]:
        indices = np.asarray(term["beta_idx"], dtype=np.intp)
        projection = [
            (term_column, selected_positions[int(column)])
            for term_column, column in enumerate(indices)
            if int(column) in selected_positions
        ]
        if not projection:
            continue
        transformed = term["spec"].transform(X.column_array(term["name"]))
        transformed = _as_dense_block(
            transformed,
            n_rows=len(X),
            n_columns=len(indices),
            term_name=term["name"],
        )
        term_columns, output_columns = map(np.asarray, zip(*projection, strict=True))
        block[:, output_columns] = transformed[:, term_columns]
        assigned[output_columns] = True

    for term in plan["interactions"]:
        indices = np.asarray(term["beta_idx"], dtype=np.intp)
        projection = [
            (term_column, selected_positions[int(column)])
            for term_column, column in enumerate(indices)
            if int(column) in selected_positions
        ]
        if not projection:
            continue
        left_name, right_name = term["parent_names"]
        transformed = term["spec"].transform(
            X.column_array(left_name),
            X.column_array(right_name),
        )
        transformed = _as_dense_block(
            transformed,
            n_rows=len(X),
            n_columns=len(indices),
            term_name=term["name"],
        )
        term_columns, output_columns = map(np.asarray, zip(*projection, strict=True))
        block[:, output_columns] = transformed[:, term_columns]
        assigned[output_columns] = True

    if not np.all(assigned):
        missing = selected_columns[~assigned]
        raise RuntimeError(f"prediction plan did not define fitted columns {missing.tolist()}")
    return block


class EvaluationDesign:
    """Lazy exact design on evaluation rows in public fitted coordinates."""

    def __init__(self, model, X: FrameLike | EagerFrame, selected_columns: NDArray):
        self._model = model
        self._X = as_eager_frame(X)
        self._selected_columns = np.asarray(selected_columns, dtype=np.intp)
        self.n = len(self._X)
        self.p = len(self._selected_columns)
        self.shape = (self.n, self.p)
        self._transform_width = self._selected_transform_width()

    def _selected_transform_width(self) -> int:
        """Largest transform block among terms represented in this design."""
        if self.p == 0:
            return 0
        try:
            from superglm.model import base

            plan = base._prediction_plan(self._model)
        except (AttributeError, RuntimeError):
            return len(self._model.result.beta)
        selected = set(map(int, self._selected_columns))
        max_width = 0
        for term in (*plan["features"], *plan["interactions"]):
            indices = np.asarray(term["beta_idx"], dtype=np.intp)
            if selected.intersection(map(int, indices)):
                max_width = max(max_width, len(indices))
        return max_width

    @property
    def chunk_rows(self) -> int:
        return _bounded_chunk_rows(
            algebra_width=self.p,
            transform_width=self._transform_width,
        )

    def iter_dense_chunks(self) -> Iterator[tuple[int, int, NDArray]]:
        for start in range(0, self.n, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.n)
            rows = np.arange(start, stop, dtype=np.intp)
            X_chunk = as_eager_frame(self._X.take_rows(rows))
            block = _exact_runtime_design_block(
                self._model,
                X_chunk,
                self._selected_columns,
            )
            yield start, stop, block

    def toarray(self) -> NDArray:
        result = np.empty(self.shape, dtype=np.float64)
        for start, stop, block in self.iter_dense_chunks():
            result[start:stop] = block
        return result

    def weighted_moments(self, W: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (self.n,):
            raise ValueError("working weights must match evaluation design rows")
        gram = np.zeros((self.p, self.p), dtype=np.float64)
        xtw1 = np.zeros(self.p, dtype=np.float64)
        anchored_gram = np.zeros((self.p, self.p), dtype=np.float64)
        anchored_xtw1 = np.zeros(self.p, dtype=np.float64)
        anchor = None
        for start, stop, block in self.iter_dense_chunks():
            weights = W[start:stop]
            gram += block.T @ (weights[:, None] * block)
            xtw1 += block.T @ weights
            if anchor is None and len(block):
                anchor = block[0].copy()
            shifted = block if anchor is None else block - anchor
            anchored_gram += shifted.T @ (weights[:, None] * shifted)
            anchored_xtw1 += shifted.T @ weights
        gram = 0.5 * (gram + gram.T)
        centered = centered_gram_from_moments(
            anchored_gram,
            anchored_xtw1,
            float(np.sum(W)),
        )
        return gram, xtw1, centered


MetricsDesign = DesignMatrix | EvaluationDesign | NDArray


def centered_gram_from_moments(gram: NDArray, xtw1: NDArray, sum_w: float) -> NDArray:
    """Profile the intercept from raw moments with extended-precision subtraction."""
    if sum_w <= 0.0:
        raise ValueError("working weights must have positive total weight")
    wide = np.longdouble
    centered = np.asarray(gram, dtype=wide) - np.outer(
        np.asarray(xtw1, dtype=wide),
        np.asarray(xtw1, dtype=wide),
    ) / wide(sum_w)
    centered = np.asarray(centered, dtype=np.float64)
    return 0.5 * (centered + centered.T)


def iter_dense_chunks(design: MetricsDesign) -> Iterator[tuple[int, int, NDArray]]:
    """Yield bounded dense row blocks from grouped or evaluation designs."""
    if isinstance(design, EvaluationDesign):
        yield from design.iter_dense_chunks()
        return

    if isinstance(design, np.ndarray):
        n, p = design.shape
        chunk_rows = _bounded_chunk_rows(algebra_width=p)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            yield start, stop, np.asarray(design[start:stop], dtype=np.float64)
        return

    chunk_rows = _bounded_chunk_rows(algebra_width=design.p)
    for start in range(0, design.n, chunk_rows):
        stop = min(start + chunk_rows, design.n)
        rows = np.arange(start, stop, dtype=np.intp)
        yield start, stop, np.asarray(design.row_subset(rows).toarray(), dtype=np.float64)


def quadratic_form_diagonal(design: MetricsDesign, matrix: NDArray) -> NDArray:
    """Return row-wise ``x @ matrix @ x`` without retaining a dense design."""
    result = np.empty(design.shape[0], dtype=np.float64)
    for start, stop, block in iter_dense_chunks(design):
        result[start:stop] = np.sum((block @ matrix) * block, axis=1)
    return result


def weighted_moments(design: MetricsDesign, W: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Return raw Gram, intercept cross-product, and centered data Gram."""
    W = np.asarray(W, dtype=np.float64)
    if isinstance(design, EvaluationDesign):
        return design.weighted_moments(W)

    if isinstance(design, np.ndarray):
        if W.shape != (design.shape[0],):
            raise ValueError("working weights must match dense design rows")
        p = design.shape[1]
        gram = np.zeros((p, p), dtype=np.float64)
        xtw1 = np.zeros(p, dtype=np.float64)
        anchored_gram = np.zeros((p, p), dtype=np.float64)
        anchored_xtw1 = np.zeros(p, dtype=np.float64)
        anchor = None
        for start, stop, block in iter_dense_chunks(design):
            weights = W[start:stop]
            gram += block.T @ (weights[:, None] * block)
            xtw1 += block.T @ weights
            if anchor is None and len(block):
                anchor = block[0].copy()
            shifted = block if anchor is None else block - anchor
            anchored_gram += shifted.T @ (weights[:, None] * shifted)
            anchored_xtw1 += shifted.T @ weights
        gram = 0.5 * (gram + gram.T)
        sum_w = float(np.sum(W))
        centered = centered_gram_from_moments(anchored_gram, anchored_xtw1, sum_w)
        return gram, xtw1, centered

    from superglm.solvers.centered_system import build_centered_system

    system = build_centered_system(
        dm=design,
        W=W,
        z_off=np.zeros(design.n, dtype=np.float64),
        penalty=np.zeros((design.p, design.p), dtype=np.float64),
    )
    gram, xtw1, _, _ = system.raw_weighted_moments()
    return gram, xtw1, system.data_gram


def factor_from_gram(data_gram: NDArray) -> NDArray:
    """Return a square factor R satisfying ``R.T @ R == data_gram``."""
    if data_gram.shape == (0, 0):
        return np.empty((0, 0), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (data_gram + data_gram.T))
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * np.sqrt(eigenvalues)).T
