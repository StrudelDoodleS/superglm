"""Optional chunk-local panels for small ordinary grouped designs.

The caller chooses the row bound and byte budget. No full-design conversion,
support-wide transformed table, or persistent row cache is constructed here.
Refusal leaves the existing grouped contraction available to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from superglm._group_matrix._group_matrix_kernels import _tensor_operand_in_reassociation_range
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)

_SUPPORTED = {
    DenseGroupMatrix,
    CategoricalGroupMatrix,
    RandomEffectGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    FactorSmoothGroupMatrix,
    SupportCompressedSSPGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
}


class _PanelRefusalError(Exception):
    pass


def _in_range(values):
    """Bound exponents, including nonfinite values, without large temporaries.

    Five original factors (two bases, two maps, and weights), each within
    2**[-128, 128], leave ample normal-range headroom for reassociation.
    Dimensions are additionally bounded by 2**20 below. This is deliberately
    conservative: cancellation is allowed, unsafe input ranges fall back.
    """
    if type(values) is not np.ndarray or values.dtype != np.float64:
        return False
    if values.ndim == 1:
        return _tensor_operand_in_reassociation_range(values[:, None])
    if values.ndim == 2:
        return _tensor_operand_in_reassociation_range(values)
    # Preserve the generic-rank predicate with bounded buffers. All current
    # panel, raw-basis, map and weight callers use the allocation-free paths.
    iterator = np.nditer(
        values,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=1024,
    )
    for block in iterator:
        magnitude = np.abs(block)
        if not np.all((magnitude == 0) | ((magnitude >= 2.0**-128) & (magnitude <= 2.0**128))):
            return False
    return True


def _require_range(values):
    if not _in_range(values):
        raise _PanelRefusalError("numerical-domain")
    return values


def _sorted_unique(values):
    # Bound the validation mask independently of the source row count.
    for start in range(1, len(values), 1024):
        stop = min(start + 1024, len(values))
        if np.any(values[start:stop] <= values[start - 1 : stop - 1]):
            return False
    return True


def _csr_product(basis, rows, transform, cached):
    if type(basis) is not sp.csr_matrix:
        raise _PanelRefusalError("unsupported-group")
    data, indices, indptr = cached
    if any(
        type(array) is not np.ndarray
        for array in (basis.data, basis.indices, basis.indptr, data, indices, indptr)
    ):
        raise _PanelRefusalError("unsupported-group")
    # Group kernels use cached CSR buffers. Fresh row subsets keep these
    # coherent; conservatively refuse replaced index storage and compare only
    # selected values when the float64 cache is a separate copy.
    if indices is not basis.indices or indptr is not basis.indptr:
        raise _PanelRefusalError("unsupported-group")
    # SciPy normalizes both entire source index buffers before fancy row
    # slicing. Mixed widths or byte orders would copy beyond our row budget.
    if indices.dtype != indptr.dtype or indices.dtype not in (
        np.dtype(np.int32),
        np.dtype(np.int64),
    ):
        raise _PanelRefusalError("unsupported-group")
    if basis.data.dtype != np.float64:
        raise _PanelRefusalError("numerical-domain")
    # Bound the copy before slicing; canonicality is checked on our owned
    # slice so reading SciPy's lazy property never mutates borrowed state or
    # scans the whole source matrix. Duplicate CSR entries conservatively
    # fall back even when they happen to fit the raw-width allocation bound.
    counts = basis.indptr[rows + 1] - basis.indptr[rows]
    if np.any((counts < 0) | (counts > basis.shape[1])):
        raise _PanelRefusalError("unsupported-group")
    selected = basis[rows]
    if not selected.has_canonical_format:
        raise _PanelRefusalError("unsupported-group")
    if data is not basis.data:
        if data.shape != basis.data.shape or data.dtype != np.float64:
            raise _PanelRefusalError("unsupported-group")
        # Gather only the selected cached entries. Constructing a source-size
        # CSR wrapper can silently copy/downcast its full index buffers.
        positions = np.repeat(basis.indptr[rows] - selected.indptr[:-1], counts)
        positions += np.arange(selected.nnz, dtype=positions.dtype)
        if not np.array_equal(data[positions], selected.data):
            raise _PanelRefusalError("unsupported-group")
    _require_range(selected.data)
    _require_range(transform)
    return selected @ transform


def _support_product(basis, bins, transform):
    if any(type(array) is not np.ndarray for array in (basis, bins, transform)):
        raise _PanelRefusalError("unsupported-group")
    selected = _require_range(basis[bins])
    _require_range(transform)
    return selected @ transform


def _render_group(group, rows, out):
    """Render directly into one panel's column view from stored coordinates."""
    kind = type(group)
    # Check borrowed arrays before gathering: an ndarray subclass may replace
    # __getitem__ with arbitrary arithmetic and return an ordinary ndarray.
    for name in (
        "M",
        "codes",
        "B_unique",
        "bin_idx",
        "bin_idx_level",
        "row_idx",
        "R_inv",
        "natural_map",
    ):
        array = getattr(group, name, None)
        if array is not None and type(array) is not np.ndarray:
            raise _PanelRefusalError("unsupported-group")
    if kind is DenseGroupMatrix:
        out[:] = _require_range(group.M[rows])
    elif kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
        codes = group.codes[rows]
        if np.any((codes < 0) | (codes > group.n_levels)):
            raise _PanelRefusalError("unsupported-group")
        out.fill(0)
        active = np.flatnonzero(codes < group.n_levels)
        out[active, codes[active]] = 1
    elif kind in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
        out[:] = _support_product(group.B_unique, group.bin_idx[rows], group.R_inv)
    elif kind is SparseSSPGroupMatrix:
        out[:] = _csr_product(
            group.B, rows, group.R_inv, (group._data, group._indices, group._indptr)
        )
    elif kind in (
        SplineCategoricalGroupMatrix,
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    ):
        # Unsorted/duplicate level layouts have different cached algebra. Do
        # not repair them, sort source-sized arrays, or infer their semantics.
        if not _sorted_unique(group.row_idx):
            raise _PanelRefusalError("unsupported-group")
        positions = np.searchsorted(group.row_idx, rows)
        active = np.flatnonzero(positions < len(group.row_idx))
        active = active[group.row_idx[positions[active]] == rows[active]]
        level_rows = positions[active]
        out.fill(0)
        if kind is SplineCategoricalGroupMatrix:
            # Materialized Gram caches may no longer agree with mutable CSR
            # data. Fresh chunk subsets have no such cache.
            if group._dense_level is not False and group._dense_level is not None:
                raise _PanelRefusalError("unsupported-group")
            out[active] = _csr_product(
                group.B_level, level_rows, group.R_inv, (group._data, group._indices, group._indptr)
            )
        else:
            out[active] = _support_product(
                group.B_unique, group.bin_idx_level[level_rows], group.R_inv
            )
    elif kind is FactorSmoothGroupMatrix:
        if group.factor_basis not in ("fs", "sz"):
            raise _PanelRefusalError("unsupported-group")
        if group.is_discrete:
            natural = _support_product(group.B_unique, group.bin_idx[rows], group.natural_map)
        else:
            natural = _csr_product(
                group.B, rows, group.natural_map, (group._data, group._indices, group._indptr)
            )
        codes = group.codes[rows]
        if np.any((codes < 0) | (codes >= group.n_levels)):
            raise _PanelRefusalError("unsupported-group")
        out.fill(0)
        blocks = out.reshape(len(rows), group.coefficient_levels, group.block_size)
        active = np.flatnonzero(codes < group.coefficient_levels)
        blocks[active, codes[active], :] = natural[active]
        if group.factor_basis == "sz":
            final = np.flatnonzero(codes == group.coefficient_levels)
            blocks[final, :, :] = -natural[final, None, :]
    else:  # Explicit type dispatch prevents custom subclass semantic changes.
        raise _PanelRefusalError("unsupported-group")


class SmallGroupPanelWorkspace:
    """Own immutable panels and one reusable weighted panel until ``close``.

    Cross moments are owned by the caller; the peak estimate allows one live
    output. Callers retaining multiple outputs must budget those separately.
    """

    def __init__(self, panels, columns, scratch):
        self.panels = tuple(panels)
        self.column_indices = tuple(columns)
        self._scratch = scratch
        self._closed = False

    @property
    def retained_bytes(self):
        if self._closed:
            return 0
        return sum(array.nbytes for array in (*self.panels, *self.column_indices, self._scratch))

    def cross_moment(self, left, right, signed_weights):
        if self._closed:
            raise RuntimeError("small-group panel workspace is closed")
        a, b = self.panels[left], self.panels[right]
        if np.shape(signed_weights) != (a.shape[0],):
            raise ValueError(f"signed weights must have shape {(a.shape[0],)}")
        if not _in_range(signed_weights):
            return None
        weighted = self._scratch[:, : b.shape[1]]
        np.multiply(b, signed_weights[:, None], out=weighted)
        return a.T @ weighted

    def close(self):
        self.panels = ()
        self.column_indices = ()
        self._scratch = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("small-group panel workspace is closed")
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class SmallGroupPanelBuild:
    workspace: SmallGroupPanelWorkspace | None
    reason: str | None
    estimated_peak_bytes: int


def build_small_group_panels(plans, rows, *, byte_budget, group_indices=None):
    """Build bounded ordinary-group panels, or return an explained refusal.

    ``rows`` is a slice or a one-dimensional integer array (duplicates allowed).
    ``group_indices`` selects groups per predictor, always retaining intercepts
    and original column order. Budget includes live panels, row/column indices,
    weighted scratch, one returned moment, bounded renderer/validation buffers,
    and conservative Python/SciPy allocation overhead. Source storage is borrowed
    only while building; neither plans nor row selections are retained.
    """
    plans = tuple(plans)
    if not plans:
        raise ValueError("at least one predictor plan is required")
    if isinstance(byte_budget, bool) or not isinstance(byte_budget, int) or byte_budget < 0:
        raise ValueError("byte_budget must be a nonnegative integer")
    n = plans[0].design.n
    if any(plan.design.n != n for plan in plans):
        raise ValueError("predictor plans must have the same row count")
    if isinstance(rows, slice):
        start, stop, step = rows.indices(n)
        row_count = len(range(start, stop, step))
    else:
        if type(rows) is not np.ndarray or rows.ndim != 1 or rows.dtype.kind not in "iu":
            raise ValueError("rows must be a slice or one-dimensional integer array")
        row_count = len(rows)
        if row_count and (int(rows.min()) < 0 or int(rows.max()) >= n):
            raise ValueError("rows are outside the predictor design")
    if group_indices is None:
        group_indices = tuple(tuple(range(len(plan.design.group_matrices))) for plan in plans)
    if len(group_indices) != len(plans):
        raise ValueError("group_indices must select groups for every predictor")
    selected, widths = [], []
    max_raw = max_group = 0
    for plan, indices in zip(plans, group_indices, strict=True):
        groups = plan.design.group_matrices
        indices = tuple(indices)
        if any(type(index) is not int or not 0 <= index < len(groups) for index in indices):
            raise ValueError("group_indices contains an invalid group index")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("group_indices must be unique and increasing")
        selection = tuple(groups[index] for index in indices)
        for group in selection:
            if type(group) in (DiscretizedTensorGroupMatrix, DiscretizedSCOPGroupMatrix):
                return SmallGroupPanelBuild(None, "specialized-group", 0)
            if type(group) not in _SUPPORTED:
                return SmallGroupPanelBuild(None, "unsupported-group", 0)
            raw = getattr(group, "R_inv", getattr(group, "natural_map", None))
            max_raw = max(max_raw, raw.shape[0] if raw is not None else group.shape[1])
            max_group = max(max_group, group.shape[1])
        selected.append(selection)
        widths.append(int(plan.intercept) + sum(group.shape[1] for group in selection))
    flat = tuple(group for selection in selected for group in selection)
    if flat and all(
        type(group) in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix)
        for group in flat
    ):
        return SmallGroupPanelBuild(None, "specialized-histogram-layout", 0)
    maximum = max(widths)
    # Canonical CSR has at most raw_width entries per selected row. The factor
    # 64 covers its data/indices/indptr, gathers, transforms, masked assignment,
    # and validation buffers simultaneously, even with 64-bit CSR indices.
    estimate = (
        65536
        + 4096 * (len(plans) + len(flat))
        + 8 * (row_count * (sum(widths) + maximum) + maximum**2 + sum(widths))
        + 64 * row_count * (max_raw + max_group + 8)
        + 16 * max_raw * max_group
    )
    if estimate > byte_budget:
        return SmallGroupPanelBuild(None, "byte-budget", estimate)
    if max(row_count, maximum, max_raw, max_group) > 2**20:
        return SmallGroupPanelBuild(None, "numerical-domain", estimate)
    row_indices = (
        np.arange(start, stop, step, dtype=np.intp)
        if isinstance(rows, slice)
        else np.array(rows, dtype=np.intp, copy=True)
    )
    panels, columns = [], []
    try:
        for plan, indices, selection, width in zip(
            plans, group_indices, selected, widths, strict=True
        ):
            panel = np.empty((row_count, width), dtype=np.float64)
            offset = int(plan.intercept)
            if offset:
                panel[:, 0] = 1
            column_values = [0] if offset else []
            global_offset = offset
            for index, group in enumerate(plan.design.group_matrices):
                if index in indices:
                    column_values.extend(range(global_offset, global_offset + group.shape[1]))
                global_offset += group.shape[1]
            for group in selection:
                _render_group(group, row_indices, panel[:, offset : offset + group.shape[1]])
                offset += group.shape[1]
            _require_range(panel)
            panel.flags.writeable = False
            column = np.asarray(column_values, dtype=np.intp)
            column.flags.writeable = False
            panels.append(panel)
            columns.append(column)
    except _PanelRefusalError as refusal:
        return SmallGroupPanelBuild(None, str(refusal), estimate)
    scratch = np.empty((row_count, maximum), dtype=np.float64)
    return SmallGroupPanelBuild(SmallGroupPanelWorkspace(panels, columns, scratch), None, estimate)
