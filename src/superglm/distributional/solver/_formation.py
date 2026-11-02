"""How far forming the joint curvature from its rows can have moved it.

The terminal rank certificate (``certificate.py``) bounds the formed
penalised curvature ``G`` against the exact matrix ``H = sum_r X_r' C_r X_r +
S`` of the stored design, the computed channels and the stored penalty.  Any
kernel forms an entry of the data curvature as sums of products of the
design's stored factors and the row channels, in some association, so its
error is at most ``gamma_m`` times the same sums taken in absolute value
(Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002,
eq. 3.13).  The absolute sums are those of the *absolute design* ``X^abs``:
the product of the absolute values of a group's finest stored factors
(``|B_unique| |R_inv|`` for a support, ``kron(|B_1|, |B_2|) |R_inv|`` for a
tensor, ``|M|`` for a dense block, the one-hot rows for a categorical), which
dominates every representation a kernel contracts, materialised or not.

Each assembly records, from the channels it already holds:

``majorant[i] >= sum_r w_rk (X^abs_ri)^2`` for column ``i`` of predictor
``k``, with the row weights of :func:`channel_weights`, which dominate the
row's ``K x K`` block ``|C_r|`` by a diagonal one and take absolute values, so
a channel of either sign needs no special case.  A chunk adds one bin, level
or cell sum of the weights per group (a dense block: one squared product);
each group closes its sums once per assembly, a factored support through
Cauchy-Schwarz, ``(sum_a b_a r_a)^2 <= (sum_a b_a) (sum_a b_a r_a^2)`` for
``b, r >= 0``, so no support table is formed.

Groups with no absolute design here (factor smooths, unknown subclasses,
designs that are not a plain ``DesignMatrix``) leave the statistics
unavailable, and the terminal decision is then the formed one, as before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_core import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm._group_matrix._group_matrix_discretized import (
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm._group_matrix._group_matrix_range import _category_rows
from superglm.distributional.layout import StackedLayout
from superglm.distributional.solver.packing import packed_pairs
from superglm.group_matrix import DesignMatrix

_U = 2.0**-53


def gamma(count: float) -> float:
    """``gamma_count = count u / (1 - count u)``, rounded up; ``inf`` once ``count u >= 1``."""
    product = count * _U
    if not product < 1.0:
        return math.inf
    return float(np.nextafter(product / (1.0 - product), np.inf))


@dataclass(frozen=True, eq=False)
class CurvatureFormation:
    """The data curvature's formation statistics, for :func:`certificate.formation_bound`.

    ``rows`` counts the rows summed and ``chunks`` the partial sums the
    assembly added into its accumulator (one per chunk, or per contracted
    block); ``inner`` is the largest inner length of a group's factor
    chain (the basis width a support or spline row sums over). ``route`` is
    ``"dense"`` when the dense predictor matrices formed the curvature, so the
    absolute design is ``|toarray()|`` itself; ``range_geometry`` records how
    the chunk stream evaluated its predictors, so a row pass replays it.
    """

    rows: int
    chunks: int
    inner: int
    majorant: NDArray[np.float64] = field(repr=False)
    route: Literal["dense", "chunked"]
    range_geometry: bool = False

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CurvatureFormation):
            return NotImplemented
        scalars = (
            "rows",
            "chunks",
            "inner",
            "route",
            "range_geometry",
        )
        return all(getattr(self, name) == getattr(other, name) for name in scalars) and (
            np.array_equal(self.majorant, other.majorant)
        )


def _abs_csr(matrix: sp.csr_matrix) -> sp.csr_matrix:
    return sp.csr_matrix(
        (np.abs(matrix.data), matrix.indices, matrix.indptr), shape=matrix.shape, copy=False
    )


def _bin_sums(index: NDArray, values: NDArray, length: int) -> NDArray:
    """``sums[b] = sum of values[r] over rows with index[r] == b``, column by column."""
    sums = np.empty((length, values.shape[1]))
    for column in range(values.shape[1]):
        sums[:, column] = np.bincount(index, weights=values[:, column], minlength=length)
    return sums


class _AbsDense:
    """A dense or CSR block: ``X^abs = |M|``; a row sums nothing before its product."""

    inner = 0

    def __init__(self, matrix):
        self._matrix = matrix

    def _rows(self, start, stop):
        rows = self._matrix[start:stop]
        return _abs_csr(sp.csr_matrix(rows)) if sp.issparse(rows) else np.abs(rows)

    def square_partial(self, start, stop, weights):
        rows = self._rows(start, stop)
        squared = rows.multiply(rows) if sp.issparse(rows) else rows * rows
        return np.asarray(squared.T @ weights).ravel()

    def square_close(self, partial):
        return partial

    def matvec(self, start, stop, values):
        return np.asarray(self._rows(start, stop) @ values)

    def rmatvec(self, start, stop, values):
        return np.asarray(self._rows(start, stop).T @ values)


class _AbsCategorical:
    """One-hot rows, exact; the sink code (the base level) is the zero row."""

    inner = 0

    def __init__(self, group):
        self._codes = np.asarray(group.codes, dtype=np.intp)
        self._levels = int(group.n_levels)

    def square_partial(self, start, stop, weights):
        codes = self._codes[start:stop]
        return np.bincount(codes, weights=weights, minlength=self._levels + 1)[: self._levels]

    def square_close(self, partial):
        return partial

    def matvec(self, start, stop, values):
        table = np.vstack((values, np.zeros((1, values.shape[1]))))
        return table[self._codes[start:stop]]

    def rmatvec(self, start, stop, values):
        return _bin_sums(self._codes[start:stop], values, self._levels + 1)[: self._levels]


class _AbsTable:
    """Rows that index a support ``X^abs = basis[index] transform``; level groups use some rows only.

    With a transform the squared statistic takes Cauchy-Schwarz, as
    :class:`_AbsFactoredCSR` does, so a chunk adds one bin sum and the table
    ``basis transform`` is formed only for a row pass.
    """

    def __init__(self, basis, index, inner, transform=None, level_group=None):
        self._basis = np.ascontiguousarray(basis, dtype=np.float64)
        self._transform = transform
        self._table = None
        self._index = index
        self.inner = int(inner)
        self._level = level_group

    def _full(self):
        if self._table is None:
            self._table = self._basis if self._transform is None else self._basis @ self._transform
        return self._table

    def _rows(self, start, stop):
        """``(local rows or None for every row, the table rows they index)``."""
        if self._level is None:
            return None, self._index[start:stop]
        located = _category_rows(self._level, start, stop)
        if located is not None and located[1] is not None:
            return located
        row_idx = np.asarray(self._level.row_idx, dtype=np.intp)
        positions = np.flatnonzero((row_idx >= start) & (row_idx < stop))
        return row_idx[positions] - start, np.asarray(self._level.bin_idx_level)[positions]

    def square_partial(self, start, stop, weights):
        local, bins = self._rows(start, stop)
        values = weights if local is None else weights[local]
        return np.bincount(bins, weights=values, minlength=len(self._basis))

    def square_close(self, sums):
        if self._transform is None:
            return (self._basis * self._basis).T @ sums
        aggregate = self._basis.T @ (sums * self._basis.sum(axis=1))
        return (self._transform * self._transform).T @ aggregate

    def matvec(self, start, stop, values):
        local, bins = self._rows(start, stop)
        gathered = (self._full() @ values)[bins]
        if local is None:
            return gathered
        out = np.zeros((stop - start, values.shape[1]))
        out[local] = gathered
        return out

    def rmatvec(self, start, stop, values):
        local, bins = self._rows(start, stop)
        rows = values if local is None else values[local]
        return self._full().T @ _bin_sums(bins, rows, len(self._basis))


class _AbsFactoredCSR:
    """A CSR basis ``B`` times ``R_inv``: ``X^abs = |B| |R_inv|``, used without forming it.

    The squared statistic takes Cauchy-Schwarz, ``(sum_a |b_a| |r_a|)^2 <=
    (sum_a |b_a|) (sum_a |b_a| r_a^2)``, so it costs one sparse transpose
    product, as the score's does.
    """

    def __init__(self, group):
        self._group = group
        self._transform = np.abs(np.asarray(group.R_inv, dtype=np.float64))
        self.inner = int(self._transform.shape[0])

    def _basis(self, start, stop):
        group = self._group
        if type(group) is SparseSSPGroupMatrix:
            return None, _abs_csr(group.B[start:stop])
        located = _category_rows(group, start, stop)
        if located is None:
            row_idx = np.asarray(group.row_idx, dtype=np.intp)
            local = np.sort(row_idx[(row_idx >= start) & (row_idx < stop)]) - start
        else:
            local = located[0]
        return local, _abs_csr(group.B[start:stop][local])

    def square_partial(self, start, stop, weights):
        local, basis = self._basis(start, stop)
        values = weights if local is None else weights[local]
        row_sums = np.asarray(basis.sum(axis=1)).ravel()
        return np.asarray(basis.T @ (values * row_sums)).ravel()

    def square_close(self, aggregate):
        return (self._transform * self._transform).T @ aggregate

    def matvec(self, start, stop, values):
        local, basis = self._basis(start, stop)
        rows = np.asarray(basis @ (self._transform @ values))
        if local is None:
            return rows
        out = np.zeros((stop - start, values.shape[1]))
        out[local] = rows
        return out

    def rmatvec(self, start, stop, values):
        local, basis = self._basis(start, stop)
        rows = values if local is None else values[local]
        return self._transform.T @ np.asarray(basis.T @ rows)


class _AbsTensor:
    """A tensor's absolute support ``kron(M_1[i], M_2[j]) A`` on the bin grid.

    Kernels contract the marginals ``B_1``, ``B_2`` (or, with a raw band, the
    raw margins and ``projection = kron(P_1, P_2)``) and then ``R_inv``, so
    the grid row ``(i, j)`` is ``kron(|B_1[i]|, |B_2[j]|) |R|``, or with the
    band ``kron(|raw_1[i]|, |raw_2[j]|) |projection| |R|``, which dominates the
    first: ``|B_k| = |raw_k P_k| <= |raw_k| |P_k|`` and ``|kron(P_1, P_2)| =
    kron(|P_1|, |P_2|)``.  The joint table ``B_unique = fl(kron(B_1, B_2))``
    is within ``1 + u`` of the first, so every row carries ``1 + 4u``.

    The squared statistic takes Cauchy-Schwarz over the separable rows: the
    cell sums ``S_ij`` weighted by the margins' row sums give ``sum_ij S_ij
    kron(M_1[i], M_2[j]) = vec(M_1' S M_2)``, then ``(A o A)'`` of that.  The
    grid is formed only for a row pass.
    """

    def __init__(self, group: DiscretizedTensorGroupMatrix):
        self._group = group
        self._n2 = int(group.n_bins2)
        transform = np.abs(np.asarray(group.R_inv, dtype=np.float64))
        band = group._current_raw_channels()
        if band is None:
            self._first = np.abs(np.asarray(group.B1_unique_t, dtype=np.float64))
            self._second = np.abs(np.asarray(group.B2_unique_t, dtype=np.float64))
            self._transform = transform
            chain = 0
        else:

            def placed(offsets, values, width):
                # Each window lies inside the raw columns (``TensorRawChannels``).
                rows = np.zeros((len(offsets), width))
                for column in range(values.shape[1]):
                    rows[np.arange(len(offsets)), offsets + column] = np.abs(values[:, column])
                return rows

            projection = np.abs(np.asarray(band.projection, dtype=np.float64))
            k2_raw = int(band.k2_raw)
            offsets1 = np.asarray(band.offsets1, dtype=np.intp)
            offsets2 = np.asarray(band.offsets2, dtype=np.intp)
            self._first = placed(offsets1, band.values1, projection.shape[0] // k2_raw)
            self._second = placed(offsets2, band.values2, k2_raw)
            self._transform = projection @ transform
            chain = 2 * projection.shape[1]
        width1, width2 = self._first.shape[1], self._second.shape[1]
        # The kron row (and the projection with a band), the margins' row
        # sums, and Cauchy-Schwarz's product.
        self.inner = width1 * width2 + chain + width1 + width2 + 4
        self._grid = None

    def _cells(self, start, stop):
        group = self._group
        first = np.asarray(group.idx1[start:stop], dtype=np.intp)
        return first * self._n2 + np.asarray(group.idx2[start:stop], dtype=np.intp)

    def square_partial(self, start, stop, weights):
        cells = len(self._first) * self._n2
        return np.bincount(self._cells(start, stop), weights=weights, minlength=cells)

    def square_close(self, sums):
        cells = sums.reshape(len(self._first), self._n2)
        cells = cells * np.outer(self._first.sum(axis=1), self._second.sum(axis=1))
        aggregate = (self._first.T @ cells @ self._second).reshape(-1)
        return (self._transform * self._transform).T @ aggregate * (1.0 + 16.0 * _U)

    def _table(self):
        if self._grid is None:
            first, second = self._first, self._second
            grid = (first[:, None, :, None] * second[None, :, None, :]).reshape(
                first.shape[0] * second.shape[0], first.shape[1] * second.shape[1]
            )
            self._grid = grid @ self._transform * (1.0 + 4.0 * _U)
        return self._grid

    def matvec(self, start, stop, values):
        return (self._table() @ values)[self._cells(start, stop)]

    def rmatvec(self, start, stop, values):
        table = self._table()
        return table.T @ _bin_sums(self._cells(start, stop), values, len(table))


def abs_group(group):
    """The absolute design of one group matrix, or ``None`` when none is defined here."""
    kind = type(group)
    if kind is DenseGroupMatrix:
        return _AbsDense(np.asarray(group.M, dtype=np.float64))
    if kind is SparseGroupMatrix:
        return _AbsDense(sp.csr_matrix(group.M, dtype=np.float64))
    if kind in (CategoricalGroupMatrix, RandomEffectGroupMatrix):
        return _AbsCategorical(group)
    if kind in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
        basis = np.abs(np.asarray(group.B_unique, dtype=np.float64))
        transform = np.abs(np.asarray(group.R_inv, dtype=np.float64))
        index = np.asarray(group.bin_idx)
        return _AbsTable(basis, index, basis.shape[1], transform=transform)
    if kind is DiscretizedTensorGroupMatrix:
        return _AbsTensor(group)
    if kind is DiscretizedSCOPGroupMatrix:
        return _AbsTable(np.abs(group.B_scop_unique), np.asarray(group.bin_idx), 0)
    if kind in (
        DiscretizedSplineCategoricalGroupMatrix,
        SupportCompressedSplineCategoricalGroupMatrix,
    ):
        basis = np.abs(np.asarray(group.B_unique, dtype=np.float64))
        transform = np.abs(np.asarray(group.R_inv, dtype=np.float64))
        return _AbsTable(basis, None, basis.shape[1], transform=transform, level_group=group)
    if kind in (SparseSSPGroupMatrix, SplineCategoricalGroupMatrix):
        return _AbsFactoredCSR(group)
    return None


class AbsDesign:
    """Every predictor's absolute design over row ranges, intercepts included."""

    def __init__(self, layout: StackedLayout):
        self.layout = layout
        self.supported = True
        self.inner = 0
        self.predictors: list[tuple[int | None, list[tuple[slice, object]]]] = []
        for state in layout.predictors:
            design = state.design
            if type(design) is not DesignMatrix:
                self.supported = False
                return
            column = state.coefficient_slice.start
            intercept = None
            if state.intercept_index is not None:
                intercept = column
                column += 1
            groups = []
            for group in design.group_matrices:
                width = int(group.shape[1])
                absolute = abs_group(group)
                if absolute is None:
                    self.supported = False
                    return
                groups.append((slice(column, column + width), absolute))
                self.inner = max(self.inner, absolute.inner)
                column += width
            self.predictors.append((intercept, groups))

    def matvec(self, k: int, start: int, stop: int, values: NDArray) -> NDArray:
        """``X^abs_k values`` on rows ``[start, stop)``; ``values`` nonnegative, global columns."""
        intercept, groups = self.predictors[k]
        out = np.zeros((stop - start, values.shape[1]))
        if intercept is not None:
            out += values[intercept]
        for columns, absolute in groups:
            out += absolute.matvec(start, stop, values[columns])
        return out

    def rmatvec(self, k: int, start: int, stop: int, values: NDArray, out: NDArray) -> None:
        """Add ``X^abs_k' values`` (rows ``[start, stop)``) into ``out``'s global rows."""
        intercept, groups = self.predictors[k]
        if intercept is not None:
            out[intercept] += np.sum(values, axis=0)
        for columns, absolute in groups:
            out[columns] += absolute.rmatvec(start, stop, values)


class FormationAccumulator:
    """Collect the formation statistics of one chunked assembly, chunk by chunk."""

    def __init__(self, layout: StackedLayout, *, range_geometry: bool = False):
        self._design = AbsDesign(layout)
        self._width = layout.n_coefficients
        k_parameters = len(layout.predictors)
        pairs = packed_pairs(k_parameters)
        self._diagonal = [pairs.index((k, k)) for k in range(k_parameters)]
        self._cross = [(index, k, j) for index, (k, j) in enumerate(pairs) if k != j]
        self._majorant = np.zeros(self._width)
        # Each group's chunk partials, summed over the assembly and closed once.
        self._partials = [[None] * len(groups) for _, groups in self._design.predictors]
        self._rows = 0
        self._chunks = 0
        self._range_geometry = bool(range_geometry)

    def add(self, start: int, stop: int, curvature_packed: NDArray) -> None:
        self._rows += stop - start
        self._chunks += 1
        if not self._design.supported:
            return
        channels = np.asarray(curvature_packed, dtype=np.float64)
        row_weights = channel_weights(channels, self._diagonal, self._cross)
        for k, weights in enumerate(row_weights):
            intercept, groups = self._design.predictors[k]
            if intercept is not None:
                self._majorant[intercept] += float(np.sum(weights))
            partials = self._partials[k]
            for position, (_, absolute) in enumerate(groups):
                partial = absolute.square_partial(start, stop, weights)
                if partials[position] is None:
                    partials[position] = partial
                else:
                    partials[position] += partial

    def finish(self) -> CurvatureFormation | None:
        if not self._design.supported:
            return None
        majorant = np.array(self._majorant, dtype=np.float64)
        for (_, groups), partials in zip(self._design.predictors, self._partials, strict=True):
            for (columns, absolute), partial in zip(groups, partials, strict=True):
                if partial is not None:
                    majorant[columns] += absolute.square_close(partial)
        majorant.setflags(write=False)
        return CurvatureFormation(
            rows=self._rows,
            chunks=self._chunks,
            inner=self._design.inner,
            majorant=majorant,
            route="chunked",
            range_geometry=self._range_geometry,
        )


def channel_weights(channels: NDArray, diagonal: list[int], cross) -> list[NDArray[np.float64]]:
    """Row weights ``w_rk`` with ``t' |C_r| t <= sum_k w_rk t_k^2`` for every ``t >= 0``.

    ``2 |c_kl| t_k t_l <= |c_kl| (rho t_k^2 + t_l^2 / rho)`` for any ``rho >
    0``, so ``w_rk = |c_kk| + sum_(l != k) |c_kl| rho_kl`` with ``rho_kl
    rho_lk = 1``.  ``rho_kl = sqrt(|c_kk|) / sqrt(|c_ll|)`` where both
    diagonals are positive, which gives ``w_rk = |c_kk| (1 + sum_l kappa_rkl)``
    with ``kappa_rkl = |c_kl| / sqrt(|c_kk c_ll|)`` and is invariant under
    rescaling a predictor, and ``1`` where either is zero, so a row whose
    block is indefinite with a zero diagonal (observed curvature at an exact
    residual) still has a finite weight.  Each term is two square roots, a
    quotient and a product, and a weight sums ``K`` terms: within
    ``gamma_(K + 4)`` of its exact value, which
    :func:`certificate.formation_bound` applies.
    """
    totals = [np.abs(channels[:, index]) for index in diagonal]
    if not cross:
        return totals
    roots = [np.sqrt(total) for total in totals]
    # A predictor whose diagonal is positive on every row needs no per-row
    # masks, which is the common case; NaN fails the test and takes the masks.
    positive = [bool(np.min(root, initial=np.inf) > 0.0) for root in roots]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for index, k, j in cross:
            numerator = np.abs(channels[:, index])
            if positive[k] and positive[j]:
                balance = roots[k] / roots[j]
            else:
                balanced = (roots[k] > 0.0) & (roots[j] > 0.0)
                balance = np.where(balanced, roots[k] / roots[j], 1.0)
            totals[k] += numerator * balance
            totals[j] += numerator / balance
    return totals


def dense_formation(
    layout: StackedLayout, matrices: tuple[NDArray, ...], curvature_packed: NDArray
) -> CurvatureFormation:
    """The statistics of a dense assembly, whose absolute design is ``|toarray()|``."""
    k_parameters = len(matrices)
    pairs = packed_pairs(k_parameters)
    diagonal = [pairs.index((k, k)) for k in range(k_parameters)]
    cross = [(index, k, j) for index, (k, j) in enumerate(pairs) if k != j]
    channels = np.asarray(curvature_packed, dtype=np.float64)
    majorant = np.zeros(layout.n_coefficients)
    row_weights = channel_weights(channels, diagonal, cross)
    for state, matrix, weights in zip(layout.predictors, matrices, row_weights, strict=True):
        majorant[state.coefficient_slice] = (matrix * matrix).T @ weights
    majorant.setflags(write=False)
    return CurvatureFormation(
        rows=int(channels.shape[0]),
        chunks=1,
        inner=0,
        majorant=majorant,
        route="dense",
    )


__all__ = [
    "AbsDesign",
    "CurvatureFormation",
    "FormationAccumulator",
    "abs_group",
    "channel_weights",
    "dense_formation",
    "gamma",
]
