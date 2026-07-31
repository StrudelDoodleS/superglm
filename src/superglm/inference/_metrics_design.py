"""Memory-bounded design operations for post-fit evaluation diagnostics."""

from __future__ import annotations

import operator
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from superglm._frame import EagerFrame, FrameLike, as_eager_frame
from superglm.features.ordered_categorical import resolve_interaction_parent_of
from superglm.group_matrix import DesignMatrix

# Diagnostic algebra can keep the dense block, a shifted copy, a weighted copy,
# and matrix-product workspaces live at the same time.  Budget five row-sized
# buffers for those arrays.  Exact feature transforms also return one term-width
# block; feature-specific transform internals (for example a raw spline basis)
# are separate opaque workspaces, but share this bounded row dimension.
_MAX_DESIGN_CHUNK_BYTES = 16 * 1024 * 1024
_MAX_DESIGN_CHUNK_ROWS = 8192
_LIVE_ALGEBRA_ROW_BUFFERS = 5


class MappedColumnFactor:
    """Rectangular factor stored only on a narrow set of global columns."""

    def __init__(
        self,
        local_factor: NDArray,
        column_indices: NDArray,
        width: int,
    ):
        local = np.asarray(local_factor, dtype=np.float64)
        indices = np.asarray(column_indices, dtype=np.intp)
        if local.ndim != 2 or indices.shape != (local.shape[1],):
            raise ValueError("local factor columns must match mapped column indices")
        if width < 0 or np.any((indices < 0) | (indices >= width)):
            raise ValueError("mapped factor columns must lie within its global width")
        if len(indices) and not np.array_equal(indices, np.unique(indices)):
            raise ValueError("mapped factor columns must be sorted and unique")
        self.local_factor = np.array(local, copy=True)
        self.column_indices = np.array(indices, copy=True)
        self.width = int(width)
        self.shape = (local.shape[0], self.width)
        self.ndim = 2
        self.dtype = self.local_factor.dtype

    @property
    def storage_nbytes(self) -> int:
        """Bytes owned by the compact representation."""
        return int(self.local_factor.nbytes + self.column_indices.nbytes)

    def _selected_columns(self, selector) -> tuple[NDArray[np.intp], bool]:
        if isinstance(selector, slice):
            start, stop, step = selector.indices(self.width)
            selected = np.arange(start, stop, step, dtype=np.intp)
            scalar = False
        elif isinstance(selector, (int, np.integer)):
            column = operator.index(selector)
            if column < 0:
                column += self.width
            if not 0 <= column < self.width:
                raise IndexError("mapped factor column is out of bounds")
            selected = np.array([column], dtype=np.intp)
            scalar = True
        else:
            scalar = False
            selected = np.asarray(selector)
            if selected.dtype == bool:
                if selected.shape != (self.width,):
                    raise IndexError("boolean mapped factor index has the wrong width")
                selected = np.flatnonzero(selected)
            else:
                selected = np.asarray(selected, dtype=np.intp)
                selected = np.where(selected < 0, selected + self.width, selected)
                if np.any((selected < 0) | (selected >= self.width)):
                    raise IndexError("mapped factor column is out of bounds")
        return np.asarray(selected, dtype=np.intp), bool(scalar)

    def selected_columns(self, selector) -> NDArray:
        """Materialize only the requested global columns."""
        selected, _scalar = self._selected_columns(selector)
        result = np.zeros((self.shape[0], len(selected)), dtype=np.float64)
        if len(selected) and len(self.column_indices):
            positions = np.searchsorted(self.column_indices, selected)
            present = positions < len(self.column_indices)
            present[present] &= self.column_indices[positions[present]] == selected[present]
            result[:, present] = self.local_factor[:, positions[present]]
        return result

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError("mapped factors require two-dimensional indexing")
        rows, columns = key
        selected, scalar_column = self._selected_columns(columns)
        result = self.selected_columns(selected)[rows]
        if scalar_column:
            result = result[..., 0]
        return result

    def __array__(self, dtype=None, copy=None) -> NDArray:
        """Materialize the global rectangle only on explicit NumPy conversion."""
        result = np.zeros(self.shape, dtype=np.float64)
        result[:, self.column_indices] = self.local_factor
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result

    @property
    def T(self) -> NDArray:
        """Dense transpose for explicit legacy matrix algebra."""
        return np.asarray(self).T


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
        # An interaction transform consumes the geometry its parents CONTRIBUTE,
        # not their raw columns: an OrderedCategorical parent contributes its
        # mapped level scores, and the transform would read its labels as
        # float64.  Resolve exactly as the fit and exact-predict paths do --
        # including the FactorSmooth exception, whose grouping parent stays as
        # the fit factorized it.  A plan cached before parent specs were
        # stashed carries no "parent_specs" key; ``None`` then resolves through
        # the identity path.
        left_spec, right_spec = term.get("parent_specs", (None, None))
        ispec = term["spec"]
        _, left = resolve_interaction_parent_of(ispec, left_spec, X.column_array(left_name))
        _, right = resolve_interaction_parent_of(ispec, right_spec, X.column_array(right_name))
        transformed = ispec.transform(left, right)
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
