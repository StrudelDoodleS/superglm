"""Stable intercept-profiled systems shared by fitting and inference."""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import cast

import numpy as np
import scipy.sparse
from numpy.typing import NDArray

from superglm._group_matrix._column_local_centering import column_local_centering
from superglm._group_matrix._group_matrix_centered import (
    RawMomentRejection,
    _compensated_add,
    _try_mixed_discrete_centering,
    _try_raw_spline_tabmat_centering,
    _try_tabmat_centering,
    anchor_support_centered_gram_rhs,
    centered_gram_rhs,
    packed_centered_gram_rhs,
    try_raw_moment_centering,
)
from superglm._group_matrix._group_matrix_core import ONE_HOT_CODE_TYPES
from superglm._group_matrix._group_matrix_kernels import (
    _native,
    _ssp_gram_needs_exact,
    _weighted_bincount_2d,
)
from superglm._group_matrix._lookup_tables import table_centred_system, table_gram_plan
from superglm._roundoff import EPS as _EPS
from superglm._roundoff import UNIT_ROUNDOFF as _UNIT
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSSPGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SupportCompressedSSPGroupMatrix,
)
from superglm.solvers.mode_score import (
    corrected_two_pass_pair,
    dense_centred_rmatvec,
    dense_columns,
    rounded_weighted_mean,
)

_FACTOR_CHUNK_BYTES = 16 * 1024 * 1024
_GREY_LEAF_ROWS = 2048
_GREY_SPAN_ROWS = 65536
_FACTOR_CHUNK_ROWS = 8192


def _freeze(values: NDArray) -> NDArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CenteredSystem:
    """Complete weighted system after profiling the intercept."""

    sum_w: float
    mean_x: NDArray
    mean_z: float
    data_gram: NDArray
    rhs: NDArray
    penalty: NDArray
    hessian: NDArray
    # The centre the rows were taken about as an exact pair, ``(x - mean_hi) -
    # mean_lo`` (``weighted_mean_pair``): a dense column's anchor and its small
    # remainder, ``mean_x`` and zero elsewhere.  ``None`` for a system formed
    # from raw moments, whose rows are centred about ``mean_x`` alone.
    mean_hi: NDArray | None = None
    mean_lo: NDArray | None = None

    def centre_pair(self) -> tuple[NDArray, NDArray | None]:
        """``(hi, lo)`` for centring rows as ``(x - hi) - lo`` (``lo`` ``None``: ``x - mean_x``)."""
        if self.mean_hi is None or self.mean_lo is None:
            return self.mean_x, None
        return self.mean_hi, self.mean_lo

    def raw_weighted_moments(self) -> tuple[NDArray, NDArray, NDArray, float]:
        """Recover raw Gram/RHS moments from the stable centered system."""
        xtw1 = self.sum_w * self.mean_x
        sum_wz = self.sum_w * self.mean_z
        gram = self.data_gram + self.sum_w * np.outer(self.mean_x, self.mean_x)
        xtwz = self.rhs + self.mean_x * sum_wz
        return gram, xtw1, xtwz, sum_wz


@dataclass
class TabmatCenteringState:
    """Fit-local safety decision for accelerated raw-moment centering."""

    eligible: bool | None = None
    raw_spline_eligible: bool | None = None
    raw_moment_eligible: bool | None = None
    # The tensor raw rungs of ``packed_centered_gram_rhs`` (pattern, factored):
    # ``False`` once their certificate rejected and the anchor-centred supports
    # served the build, so later iterations go straight to those supports
    # instead of repeating the work.
    tensor_raw_eligible: bool | None = None
    _raw_moment_owners: tuple = ()

    def seed_raw_rejection(self, owners: tuple) -> bool | None:
        """Carry only a refusal within one fixed-coordinate optimizer owner."""
        if len(owners) != len(self._raw_moment_owners) or any(
            current is not previous for current, previous in zip(owners, self._raw_moment_owners)
        ):
            self.raw_moment_eligible = None
        self._raw_moment_owners = owners
        return False if owners and self.raw_moment_eligible is False else None


@dataclass
class _InitialDataReuse:
    """One initial data system, owned by a fixed-design synchronous line search.

    The owner must discard this entry before changing source coordinates or
    invoking user callbacks. No penalty, Hessian, or fitted state is retained.
    """

    key: tuple | None = None
    weights: NDArray | None = None
    response: NDArray | None = None
    before: TabmatCenteringState | None = None
    after: TabmatCenteringState | None = None
    data: tuple | None = None

    def take(self, key, W, z_off, state, penalty):
        if self.key != key or self.before != state or self.data is None:
            return None
        for actual, saved in ((W, self.weights), (z_off, self.response)):
            if (
                actual.dtype != np.float64
                or actual.shape != saved.shape
                or not np.array_equal(actual.view(np.uint64), saved.view(np.uint64))
            ):
                return None
        state.eligible = self.after.eligible
        state.raw_spline_eligible = self.after.raw_spline_eligible
        state.raw_moment_eligible = self.after.raw_moment_eligible
        state.tensor_raw_eligible = self.after.tensor_raw_eligible
        *data, mean_hi, mean_lo = self.data
        return _attach_centered_penalty(*data, penalty, mean_hi=mean_hi, mean_lo=mean_lo)

    def remember(self, key, W, z_off, before, after, system):
        self.key = key
        self.weights, self.response = _freeze(W), _freeze(z_off)
        self.before, self.after = replace(before), replace(after)
        self.data = (
            system.sum_w,
            system.mean_x,
            system.mean_z,
            system.data_gram,
            system.rhs,
            system.mean_hi,
            system.mean_lo,
        )


@dataclass
class _FisherDataReuse:
    """Unpenalized data owned by one synchronous, fixed-coordinate optimizer."""

    owners: tuple = ()
    weights: NDArray | None = None
    data: tuple | None = None

    def clear(self):
        self.owners = ()
        self.weights = self.data = None

    def take(self, owners, W):
        if (
            len(owners) != len(self.owners)
            or any(current is not saved for current, saved in zip(owners, self.owners))
            or self.weights is None
            or W.dtype != np.float64
            or W.shape != self.weights.shape
            or not np.array_equal(W.view(np.uint64), self.weights.view(np.uint64))
        ):
            self.clear()
        self.owners = owners
        return self.data

    def remember(self, W, system):
        self.weights = _freeze(W)
        self.data = (system.sum_w, system.mean_x, system.data_gram, system.mean_hi, system.mean_lo)


def iter_grouped_design_chunks(dm: DesignMatrix) -> Iterator[tuple[int, int, NDArray]]:
    """Yield bounded dense row blocks from a grouped design."""
    bytes_per_row = 3 * np.dtype(np.float64).itemsize * max(dm.p, 1)
    chunk_rows = max(1, min(_FACTOR_CHUNK_ROWS, _FACTOR_CHUNK_BYTES // bytes_per_row))
    for start in range(0, dm.n, chunk_rows):
        stop = min(start + chunk_rows, dm.n)
        rows = np.arange(start, stop, dtype=np.intp)
        yield start, stop, np.asarray(dm.row_subset(rows).toarray(), dtype=np.float64)


def grouped_augmented_row_quadratic_forms(
    dm: DesignMatrix, forms: Callable, active: NDArray | None = None
) -> NDArray:
    """Return ``[1, x_i] H_aug^+ [1, x_i]'`` per row of a grouped design.

    ``forms`` is an augmented factor's ``row_quadratic_forms``, whose first
    coordinate is the intercept.  Rows reach it as CSR blocks with a leading
    column of ones; a categorical or random-effect group stays one-hot from its
    codes, so a block is bounded by the entries stored per row, not the design
    width: the block, its copy with the ones and three working copies.  With
    an ``active`` mask only those rows are formed and every other row reads
    ``0``, so a row outside it cannot overflow into a caller's sums.
    """
    stored = sum(
        1 if isinstance(gm, CategoricalGroupMatrix) else gm.shape[1] for gm in dm.group_matrices
    )
    bytes_per_row = 5 * np.dtype(np.float64).itemsize * max(stored, 1)
    chunk_rows = max(1, min(_FACTOR_CHUNK_ROWS, _FACTOR_CHUNK_BYTES // bytes_per_row))
    result = np.zeros(dm.n, dtype=np.float64)
    for start in range(0, dm.n, chunk_rows):
        stop = min(start + chunk_rows, dm.n)
        rows = np.arange(start, stop)
        if active is not None:
            rows = rows[active[start:stop]]
            if not rows.size:
                continue
        block = scipy.sparse.hstack(
            [_sparse_row_block(gm, rows) for gm in dm.group_matrices], format="csr"
        )
        ones = scipy.sparse.csr_array(np.ones((rows.size, 1)))
        result[rows] = forms(scipy.sparse.hstack((ones, block), format="csr"))
    return result


def _sparse_row_block(gm, rows: NDArray) -> scipy.sparse.csr_array:
    """The ``rows`` of one design block; a one-hot block stays sparse from its codes."""
    if isinstance(gm, CategoricalGroupMatrix):
        codes = gm.codes[rows]
        # the base-level sink code stays an empty row
        present = codes < gm.n_levels
        return scipy.sparse.csr_array(
            (np.ones(np.count_nonzero(present)), (np.flatnonzero(present), codes[present])),
            shape=(len(codes), gm.n_levels),
        )
    return scipy.sparse.csr_array(gm.row_subset(rows).toarray())


_NO_CENTRE = np.empty(0, dtype=np.float64)


def _writable_codes(codes: NDArray) -> NDArray:
    codes = np.ascontiguousarray(codes, dtype=np.intp)
    return codes if codes.flags.writeable else codes.copy()


@dataclass(frozen=True)
class _LeafParts:
    """How a design's column blocks write a leaf's rows (``_design_leaf_parts``).

    ``held`` is the bytes of the support tables the parts hold for as long as
    they live; ``per_leaf`` the largest table a ``rows`` block forms in each
    leaf.  ``rank._tsqr_weighted_factor`` charges both to ``max_memory``.
    """

    parts: tuple
    held: int
    per_leaf: int


def _design_leaf_parts(dm: DesignMatrix) -> _LeafParts:
    """How each column block of ``dm`` writes a leaf's rows: ``(column, kind, a, b)``.

    ``one_hot`` (codes, levels) and ``gather`` (table, index) write
    ``toarray()``'s entries from the compact form without the GIL; ``rows``
    (a matrix) materialises its rows by ``row_subset(...).toarray()``, the
    whole design's when ``dm`` is not a plain ``DesignMatrix``.  A gather
    holds its support's table ``B_unique @ R_inv``; while those tables would
    take more than half of ``max_memory`` together, the supports materialise
    their leaves' rows instead, one table at a time, as ``toarray`` does.
    Inside a fit the parts are formed once (``_DataFactorReuse``).
    """
    from superglm._blas_threads import pooled_blas_threads
    from superglm._group_matrix._group_matrix_core import (
        CategoricalGroupMatrix,
        RandomEffectGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedSSPGroupMatrix,
        DiscretizedTensorGroupMatrix,
    )
    from superglm._parallel import resolve_max_memory

    if type(dm).toarray is not DesignMatrix.toarray or (
        type(dm).row_subset is not DesignMatrix.row_subset
    ):
        return _LeafParts(((0, "rows", dm, None),), 0, 0)
    reuse = _DATA_FACTOR_REUSE.get()
    if reuse is not None:
        cached = reuse.leaf_parts.get(id(dm))
        if cached is not None and cached[0] is dm and cached[1] is dm.group_matrices:
            return cached[2]
        # Only the design being factored keeps its tables: a fit factors a
        # superseded design (a REML rebuild's predecessor) no more.
        reuse.leaf_parts.clear()

    def supports(gm) -> bool:
        kind = type(gm)
        return kind.toarray is DiscretizedSSPGroupMatrix.toarray and kind.row_subset in (
            DiscretizedSSPGroupMatrix.row_subset,
            DiscretizedTensorGroupMatrix.row_subset,
        )

    table_bytes = [
        8 * len(cast(DiscretizedSSPGroupMatrix, gm).B_unique) * int(gm.shape[1])
        for gm in dm.group_matrices
        if supports(gm)
    ]
    gather = 2 * sum(table_bytes) <= resolve_max_memory()
    parts: list[tuple] = []
    column = 0
    # The table is the product ``toarray`` forms, under the pin its leaves had.
    with pooled_blas_threads():
        for gm in dm.group_matrices:
            kind = type(gm)
            if kind.toarray is CategoricalGroupMatrix.toarray and kind.row_subset in (
                CategoricalGroupMatrix.row_subset,
                RandomEffectGroupMatrix.row_subset,
            ):
                one_hot = cast(CategoricalGroupMatrix, gm)
                parts.append(
                    (column, "one_hot", _writable_codes(one_hot.codes), int(one_hot.n_levels))
                )
            elif gather and supports(gm):
                support = cast(DiscretizedSSPGroupMatrix, gm)
                table = np.ascontiguousarray(support.B_unique @ support.R_inv, dtype=np.float64)
                parts.append(
                    (column, "gather", table, _writable_codes(cast(NDArray, support.bin_idx)))
                )
            else:
                parts.append((column, "rows", gm, None))
            column += int(gm.shape[1])
    held = sum(table_bytes) if gather else 0
    per_leaf = 0 if gather else max(table_bytes, default=0)
    result = _LeafParts(tuple(parts), held, per_leaf)
    if reuse is not None:
        reuse.leaf_parts[id(dm)] = (dm, dm.group_matrices, result)
    return result


class _DesignLeaf:
    """Rows ``[start, stop)`` of a grouped design, written by the worker that factors them.

    ``fill_weighted_rows`` writes ``sqrt(w_i) * ((x_ij - c_j) - c_lo_j)``,
    the entries ``rank._tsqr_leaf`` forms from ``row_subset(...).toarray()``,
    in the same order of operations, so the factor is bitwise the same.
    """

    __slots__ = ("_parts", "held_bytes", "leaf_bytes", "shape", "start", "stop")

    def __init__(self, parts: _LeafParts, start: int, stop: int, width: int) -> None:
        self._parts = parts.parts
        self.held_bytes = parts.held
        self.leaf_bytes = parts.per_leaf
        self.start = start
        self.stop = stop
        self.shape = (stop - start, width)

    def fill_weighted_rows(self, out, sqrt_weights, center, center_lo) -> None:
        from superglm._group_matrix._group_matrix_kernels import (
            _weighted_centred_one_hot,
            _weighted_centred_rows,
        )

        centre = _NO_CENTRE if center is None else np.array(center, dtype=np.float64)
        centre_lo = _NO_CENTRE if center_lo is None else np.array(center_lo, dtype=np.float64)
        scale = np.ascontiguousarray(sqrt_weights, dtype=np.float64)
        for column, kind, first, second in self._parts:
            if kind == "one_hot":
                _weighted_centred_one_hot(
                    out, column, first, second, self.start, centre, centre_lo, scale
                )
            elif kind == "gather":
                _weighted_centred_rows(
                    out, column, first, second, self.start, centre, centre_lo, scale
                )
            else:
                rows = np.arange(self.start, self.stop, dtype=np.intp)
                values = np.array(first.row_subset(rows).toarray(), dtype=np.float64, order="C")
                _weighted_centred_rows(
                    out, column, values, rows - self.start, 0, centre, centre_lo, scale
                )


def iter_grouped_design_leaves(dm: DesignMatrix) -> Iterator[tuple[int, int, _DesignLeaf]]:
    """Yield the TSQR leaves of a grouped design: ``rank.tsqr_leaf_rows(p)`` rows each.

    The partition depends only on ``(n, p)``, which fixes the factor's
    reduction tree (``rank._tsqr_weighted_factor``).  Each leaf is deferred
    (:class:`_DesignLeaf`): the pool worker that factors it writes its
    weighted, centred rows from the design's compact form, so no thread
    materialises the rows of every leaf.
    """
    from superglm.solvers.rank import tsqr_leaf_rows

    leaf_rows = tsqr_leaf_rows(dm.p)
    if not dm.n:
        return
    parts = _design_leaf_parts(dm)
    for start in range(0, dm.n, leaf_rows):
        stop = min(start + leaf_rows, dm.n)
        yield start, stop, _DesignLeaf(parts, start, stop, dm.p)


@dataclass
class _DataFactorReuse:
    """The weighted data factors one fit has formed, keyed by the bits of their inputs.

    Owner: one ``fit_reml`` call (:func:`reuse_data_factors`); lifetime: that
    call.  A factor is a function of the design, the leaf partition
    (``rank.tsqr_leaf_rows``), and the float64 bits of the weights, the
    centre pair and the response: the penalty never enters it, the leaves
    run with BLAS on one thread at any worker count, so the same inputs give
    the same bits and an entry needs no other invalidation.  A REML fit asks
    again for factors it has formed (the observed geometry of a line-search
    trial and of the candidate it accepts; one PIRLS start shared by
    successive smoothing states).  The ``_ENTRIES`` most recently used are
    kept (a few factors of ``(p + 1)^2`` doubles and their ``n``-vectors); a
    hit returns copies, so no caller can change an entry.  ``leaf_parts``
    keeps the leaf parts (``_design_leaf_parts``, its support tables
    included) of the design the fit factored last, a function of that design
    alone, keyed by the design and its group matrices: at most one design's
    tables are retained (``retained_bytes``).
    """

    _ENTRIES = 4
    entries: list = field(default_factory=list)
    leaf_parts: dict = field(default_factory=dict)

    def _index(self, key: tuple) -> int | None:
        for index, (entry_key, _) in enumerate(self.entries):
            if len(entry_key) == len(key) and all(
                _same_input(saved, given) for saved, given in zip(entry_key, key)
            ):
                return index
        return None

    def find(self, key: tuple):
        index = self._index(key)
        if index is None:
            return None
        entry_key, value = self.entries.pop(index)
        self.entries.append((entry_key, value))
        return tuple(np.array(part) for part in value)

    def keep(self, key: tuple, value) -> None:
        frozen = tuple(part if not isinstance(part, np.ndarray) else _freeze(part) for part in key)
        self.entries.append((frozen, tuple(_freeze(part) for part in value)))
        del self.entries[: -self._ENTRIES]

    def retained_bytes(self) -> int:
        """The support tables this fit keeps (``leaf_parts``), charged by every pooled section."""
        return sum(entry[2].held for entry in list(self.leaf_parts.values()))


def _same_input(saved, given) -> bool:
    if not isinstance(saved, np.ndarray) or not isinstance(given, np.ndarray):
        return saved is given if not isinstance(saved, int) else saved == given
    return (
        saved.shape == given.shape
        and given.dtype == np.float64
        and np.array_equal(saved.view(np.uint64), given.view(np.uint64))
    )


_DATA_FACTOR_REUSE: ContextVar[_DataFactorReuse | None] = ContextVar(
    "superglm_data_factor_reuse", default=None
)


@contextmanager
def reuse_data_factors() -> Iterator[None]:
    """Reuse the weighted data factors formed inside this context (:class:`_DataFactorReuse`)."""
    from superglm import _parallel

    reuse = _DataFactorReuse()
    token = _DATA_FACTOR_REUSE.set(reuse)
    retained = _parallel._retained.set(reuse.retained_bytes)
    try:
        yield
    finally:
        _parallel._retained.reset(retained)
        _DATA_FACTOR_REUSE.reset(token)


def _factor_key(dm, W, center, center_lo, response) -> tuple:
    from superglm.solvers.rank import tsqr_leaf_rows

    def bits(values):
        return None if values is None else np.asarray(values, dtype=np.float64)

    return (dm, tsqr_leaf_rows(dm.p), bits(W), bits(center), bits(center_lo), bits(response))


def _data_factor(dm, _leaf_rows, W, center, center_lo, response) -> tuple:
    """The TSQR of :func:`grouped_weighted_factor` (``response`` ``None``) or ``_rhs``, as a tuple."""
    from superglm.solvers.rank import streamed_weighted_factor, streamed_weighted_factor_rhs

    leaves = iter_grouped_design_leaves(dm)
    if response is None:
        return (streamed_weighted_factor(leaves, W, center=center, center_lo=center_lo),)
    return streamed_weighted_factor_rhs(leaves, W, response, center=center, center_lo=center_lo)


def _reused_data_factor(dm, W, center, center_lo, response) -> tuple:
    reuse = _DATA_FACTOR_REUSE.get()
    if reuse is None:
        return _data_factor(dm, None, W, center, center_lo, response)
    key = _factor_key(dm, W, center, center_lo, response)
    found = reuse.find(key)
    if found is None:
        found = _data_factor(*key)
        reuse.keep(key, found)
    return found


def grouped_weighted_factor(
    dm: DesignMatrix,
    W: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> NDArray:
    """Return the weighted QR factor, a TSQR over the design's leaves, without retaining all rows."""
    return _reused_data_factor(dm, W, center, center_lo, None)[0]


def grouped_weighted_factor_rhs(
    dm: DesignMatrix,
    W: NDArray,
    response: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return a bounded weighted QR factor and its transformed response."""
    factor, transformed = _reused_data_factor(dm, W, center, center_lo, response)
    return factor, transformed


def penalty_factor(penalty: NDArray) -> NDArray:
    """Return a factor ``R`` with ``R'R`` the PSD penalty ``S`` to within its own resolution.

    Every rank decision downstream counts a row of ``R`` along a data-null
    direction as identifying it, so ``R`` keeps only the curvature ``S``
    certifies.  An exactly zero row of ``S`` is an exact null and is left out.
    The rest splits exactly into its contiguous diagonal blocks (no entry
    couples them; a penalty is block diagonal by term).  Each block is
    Jacobi-equilibrated, ``A_b = D S_b D`` with ``D = diag(S_b)^(-1/2)``, keeps
    the eigenpairs of ``A_b`` above its eigensolver resolution ``n_b eps
    ||A_b||_2`` (*LAPACK Users' Guide*, 3rd ed., sec. 4.7;
    ``rank._eigensolver_relative_bar``, the floor of the Gram route's rank cut,
    which equilibrates the same way; accumulated outward, ``_outward_cut``),
    and maps the root back, ``R_b = W^(1/2) V' D^(-1)``.  ``R_b'R_b = D^(-1)
    (A_b)_+ D^(-1)`` is the projection of
    ``S_b`` onto the PSD cone in the norm ``||D X D||_F`` (Higham 2002, IMA J.
    Numer. Anal. 22, Thm 3.2), less the eigenvalues below the bar.

    Below the bar an eigenvalue's magnitude and sign are rounding; keeping it
    (the former ``> 0.0`` test) invented curvature along an exact null.
    Measured: an ``sz`` term's unpenalized natural coordinates (exactly zero
    rows) came out of one ``eigh`` of the whole matrix at up to ``+5.3e-12``
    against ``||S||_2 = 3.9e4``, which made a thin level's exact data-null
    alias identified on gram (rank +1 against the structured solver).

    The cut is on ``A_b``, not ``S_b``, so no coordinate's units decide its
    rank.  A sum of PSD terms (``lambda_j D_j'D_j``, a natural
    parameterization, a Kronecker sum) is rounded entrywise within ``gamma_n
    sqrt(S_ii S_jj)`` (the dot-product bound and Cauchy-Schwarz), the scaled
    perturbation under which ``S``'s eigenvalues are determined to relative
    accuracy ``||A^-1||_2`` times its size, however graded ``S`` is (Demmel &
    Veselic 1992, SIAM J. Matrix Anal. Appl. 13(4); Drmac 2020,
    arXiv:2006.02753, Thms 3.7 and 3.11), and Jacobi scaling is within a
    factor ``n`` of the best diagonal scaling (van der Sluis 1969, Numer.
    Math. 14; Drmac, Thm 3.12).  The unscaled cut ``n_b eps ||S_b||_2``
    dropped the ``1e-6`` mode of ``[[1e10, 1e-10], [1e-10, 1e-6]]`` (bar
    ``4.4e-6``), whose ``A_b`` is the identity to ``1e-12``, and gram then
    rejected every step of a fit with that penalty.

    A block that is not PSD at its scaled resolution has no scaled model:
    a nonzero row on a non-positive diagonal, or an eigenvalue of ``A_b``
    below ``decompose_gram``'s materially-indefinite bar.  Its small entries
    are not known to be data rather than rounding (Drmac, sec. 3.3), so it
    keeps the eigenpairs above ``n_b eps ||S_b||_2`` of ``S_b`` itself.
    """
    width = penalty.shape[0]
    symmetric = 0.5 * (penalty + penalty.T)
    if penalty.shape == (0, 0) or not np.any(symmetric):
        return np.empty((0, width))
    support = np.flatnonzero(np.any(symmetric != 0.0, axis=1))
    coupled = symmetric[np.ix_(support, support)] != 0.0
    order = np.arange(len(support))
    last = (len(support) - 1) - np.argmax(coupled[:, ::-1], axis=1)
    ends = np.flatnonzero(np.maximum.accumulate(np.maximum(last, order)) == order) + 1
    starts = np.concatenate(([0], ends[:-1]))
    single = ends - starts == 1
    # 1 x 1 blocks are their own eigenpairs, exactly: positive is above the bar
    diagonal = support[starts[single]]
    values = symmetric[diagonal, diagonal]
    diagonal = diagonal[values > 0.0]
    factor = np.zeros((len(diagonal), width))
    factor[np.arange(len(diagonal)), diagonal] = np.sqrt(symmetric[diagonal, diagonal])
    rows = [factor]
    for start, stop in zip(starts[~single], ends[~single], strict=True):
        columns = support[start:stop]
        block = symmetric[np.ix_(columns, columns)]
        root = _equilibrated_block_root(block)
        if root is None:
            root = _block_root(block)
        embedded = np.zeros((root.shape[0], width))
        embedded[:, columns] = root
        rows.append(embedded)
    return np.vstack(rows)


def _block_root(block: NDArray) -> NDArray:
    """The eigenpairs of ``block`` above ``n eps ||block||_2``, as a root."""
    from superglm.solvers.rank import _eigensolver_relative_bar

    eigenvalues, eigenvectors = np.linalg.eigh(block)
    bar = _outward_cut(_eigensolver_relative_bar(len(block)), np.max(np.abs(eigenvalues)))
    kept = eigenvalues > bar
    return np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T


def _outward_cut(relative: float, norm: float) -> float:
    """``relative * norm`` rounded up: an upper bound on the exact product.

    ``relative`` (``n eps``) is exact, so the product is one rounding, and
    one ``nextafter`` toward ``+inf`` covers it (Codex review of #440): an
    eigenvalue at the bar is never kept on the product's downward rounding.
    """
    return float(np.nextafter(relative * float(norm), np.inf))


def _equilibrated_block_root(block: NDArray) -> NDArray | None:
    """``penalty_factor``'s root of a coupled block, cut on its Jacobi equilibration.

    The cut is the eigensolver's resolution ``n_b eps ||A_b||_2``, accumulated
    outward (``_outward_cut``).  The block's formation error is controlled
    where the block is formed, not in the cut: every penalty this package
    forms is a sum of ``lambda_t G_t`` with ``G_t`` the Gram of a root
    (``ssp_penalty_matrix``, ``_enclosed_root_gram``), an eigen
    reconstruction ``V Lambda V'`` (``_canonicalize_ssp_penalty``) or an exact
    integer or Kronecker form, so an exact null of the construction is an
    exact null of the rank-``r`` Gram and only rounding, entrywise within
    ``gamma_m sqrt(S_ii S_jj)`` (Higham 2002, eq. 3.5 and Cauchy-Schwarz),
    lifts it.  The worst case of that rounding, ``n_b gamma_m`` in ``||dA||_2``
    (coherent rounding in every entry), is not what the arithmetic delivers:
    across 7,000 random and structured Gram-form blocks the largest rounding
    eigenvalue was 0.15 of the eigensolver's cut, while a cut at the worst
    case dropped real curvature the eigensolver resolves (Opus review of
    #440: a coupled second-difference block at ``1e10`` beside ``1e-2`` kept
    58 of 60 rows, a 529-column REML tensor 507 of 528, ``D_3'D_3`` at
    ``k = 300`` 296 of 297), and so did ``sqrt(n_b) gamma_m`` (137 of 138
    on a Kronecker sum at ``1e11``).

    ``None`` where the block is not PSD at that resolution: a non-positive
    diagonal, or an eigenvalue of ``D S_b D`` below ``-max(100 eps ||D S_b
    D||_2, cut)``, the bar ``rank.decompose_gram`` refuses as materially
    indefinite.
    """
    from superglm.solvers.rank import _eigensolver_relative_bar

    diagonal = np.diag(block)
    if not np.all(diagonal > 0.0):
        return None
    scale = np.sqrt(diagonal)
    # |S_ij| / s_i <= s_j on a PSD block, so only an indefinite one overflows
    with np.errstate(over="ignore"):
        equilibrated = (block / scale[:, None]) / scale[None, :]
    equilibrated = 0.5 * (equilibrated + equilibrated.T)
    if not np.all(np.isfinite(equilibrated)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    norm = float(np.max(np.abs(eigenvalues)))
    cut = _outward_cut(_eigensolver_relative_bar(len(block)), norm)
    if eigenvalues[0] < -max(100.0 * _EPS * norm, cut):
        return None
    kept = eigenvalues > cut
    return np.sqrt(eigenvalues[kept])[:, None] * eigenvectors[:, kept].T * scale[None, :]


def grouped_augmented_factor(
    dm: DesignMatrix,
    W: NDArray,
    penalty: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> NDArray:
    """Return the bounded weighted-design factor augmented by ``sqrt(S)``."""
    data_factor = grouped_weighted_factor(dm, W, center=center, center_lo=center_lo)
    smooth_factor = penalty_factor(penalty)
    return data_factor if smooth_factor.shape[0] == 0 else np.vstack((data_factor, smooth_factor))


def grouped_augmented_factor_rhs(
    dm: DesignMatrix,
    W: NDArray,
    penalty: NDArray,
    *,
    response: NDArray,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Return one compact QR of the weighted data, penalty, and response."""
    data_factor, transformed_rhs = grouped_weighted_factor_rhs(
        dm,
        W,
        response,
        center=center,
        center_lo=center_lo,
    )
    smooth_factor = penalty_factor(penalty)
    if smooth_factor.shape[0] == 0:
        return data_factor, transformed_rhs
    joint = np.column_stack((data_factor, transformed_rhs))
    smooth_joint = np.column_stack((smooth_factor, np.zeros(smooth_factor.shape[0])))
    joint_factor = np.linalg.qr(np.vstack((joint, smooth_joint)), mode="r")
    return np.asarray(joint_factor[:, :-1]), np.asarray(joint_factor[:, -1])


def _centred_support(values: NDArray, center, center_lo, columns: slice) -> NDArray:
    """``(values - c) - c_lo`` on ``columns``: a support's rows centred as a TSQR leaf centres them."""
    centred = values - (0.0 if center is None else center[columns])
    if center_lo is not None:
        centred = centred - center_lo[columns]
    return centred


def grouped_structural_nulls(
    dm: DesignMatrix,
    W: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> list[tuple[slice, NDArray, float]]:
    """Exact nulls of the weighted, centred design read off each group's occupied support rows.

    A group whose rows are rows of a small support table ``S`` (an indexed
    support, or a spline-by-level block, zero off its level) has ``X_g v = 0``
    on every row for each ``v`` with ``S_occ v = 0``, ``S_occ`` the support
    rows that carry weight; then ``c_g' v``, a weighted mean of those rows
    times ``v``, is zero too, so ``A v = 0`` with no pass over the rows.  The
    candidates are the right singular vectors of ``S_occ`` below ``n_s eps``
    of its largest singular value, and a candidate is kept only when its
    residual is exact to the rounding that measures it: the weighted
    ``||(S - c) n||`` over the occupied rows and the rows off a level no
    larger than ``gamma_(w + 3)`` of ``(|S| + |c|) |n|`` weighted the same
    way.  That is a componentwise form of the rule LAPACK's symmetric
    tridiagonal QR applies (``dsteqr`` splits only when an off-diagonal is
    within ``eps sqrt(|d_i d_(i+1)|)``): drop only what is at the
    algorithm's own rounding.  (``dlaed2``'s divide-and-conquer deflation is
    normwise, ``8 eps max(|D|, |Z|)``; a normwise allowance is a recorded
    follow-up.)  So the rotation a
    deflated null causes in the null basis, ``||A n||`` over the weakest
    retained singular value (Wedin, *BIT* 12, 1972), is the full factor's
    own backward error over that value.  A near null, ``n_s eps`` small but
    above its rounding, stays in the Gram, where the grey subspace resolves
    its coupling to other weak directions at row resolution.  Each group
    returns ``(columns, N_g, bound)`` with ``bound >= ||A N_g||_F``, formed
    from the support rows' weight sums: the residual of every occupied row
    and of the rows off a level, plus the rounding of forming them
    (``gamma_(w + 3)`` of ``(|S| + |c|) |N_g|``, Higham 2002, sec. 3.5).
    """
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
    )

    weights = np.asarray(W, dtype=np.float64)
    total = float(np.sum(weights))
    parts = _design_leaf_parts(dm).parts
    if len(parts) != len(dm.group_matrices):
        return []
    found: list[tuple[slice, NDArray, float]] = []
    level_type = DiscretizedSplineCategoricalGroupMatrix
    for (column, kind, first, second), group in zip(parts, dm.group_matrices, strict=True):
        if kind == "gather":
            support = np.asarray(first, dtype=np.float64)
            mass = np.bincount(second, weights=weights, minlength=support.shape[0])
            outside = 0.0
        elif kind == "rows" and _is_plain_level(group, level_type):
            level = cast(DiscretizedSplineCategoricalGroupMatrix, group)
            support = np.asarray(level.B_unique, dtype=np.float64) @ np.asarray(
                level.R_inv, dtype=np.float64
            )
            rows = np.asarray(level.row_idx)
            mass = np.bincount(
                np.asarray(level.bin_idx_level), weights=weights[rows], minlength=support.shape[0]
            )
            outside = max(total - float(np.sum(mass)), 0.0)
        else:
            continue
        width = support.shape[1]
        occupied = support[mass > 0.0]
        if occupied.shape[0] == 0:
            continue
        if occupied.shape[0] > width:
            # The same right singular vectors from the w x w triangle.
            occupied = np.linalg.qr(occupied, mode="r")
        _, singular, right = np.linalg.svd(occupied, full_matrices=True)
        cut = max(support.shape) * _UNIT * 2.0 * (float(singular[0]) if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > cut))
        null = right[rank:].T
        if null.shape[1] == 0:
            continue
        columns = slice(column, column + width)
        # ``(0 - c) - c_lo``: the centred value of a row off the support
        shift = _centred_support(np.zeros((1, width)), center, center_lo, columns)[0]
        residual = np.abs((support + shift[None, :]) @ null)
        scale = (np.abs(support) + np.abs(shift)[None, :]) @ np.abs(null)
        gamma = (width + 3) * _UNIT / (1.0 - (width + 3) * _UNIT)
        rounding = gamma * scale
        off, off_rounding = np.abs(shift @ null), gamma * (np.abs(shift) @ np.abs(null))
        # Deflate a candidate only when its residual is within its own
        # rounding; a near null, which the rows can tell from zero, stays in
        # the Gram for the grey subspace to turn with its coupling.
        exact = (
            mass @ residual**2 + outside * off**2 <= mass @ rounding**2 + outside * off_rounding**2
        )
        if not np.any(exact):
            continue
        row_bound, off = (residual + rounding)[:, exact], (off + off_rounding)[exact]
        squared = float(mass @ np.sum(row_bound**2, axis=1)) + outside * float(np.sum(off**2))
        found.append((columns, null[:, exact], float(np.nextafter(math.sqrt(squared), np.inf))))
    return found


def _is_plain_level(group, level_type) -> bool:
    """A spline-by-level block whose rows are its support's rows on its level, and zero elsewhere."""
    kind = type(group)
    return (
        isinstance(group, level_type)
        and kind.toarray is level_type.toarray
        and kind.row_subset is level_type.row_subset
    )


def grouped_projected_factor(
    dm: DesignMatrix,
    W: NDArray,
    basis: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray, int, NDArray]:
    """``R`` of ``B = A basis`` (a TSQR), ``A'B``, its rounding count and column magnitudes.

    ``A = sqrt(W) ((X - c) - c_lo)``.

    One pass over the rows.

    ``basis`` is ``p x k``.  Each group block of the design writes its part of
    ``X basis`` from its compact form (``_design_leaf_parts``): a categorical
    and an indexed support form one table ``((S - c) - c_lo) basis_g`` of
    their support rows ``S`` and add the row of it each observation indexes,
    so a row costs one ``k``-vector add per group, ``n T k`` in all, against
    ``2 n p^2`` for the TSQR of ``A`` itself; another group writes its dense
    rows a block at a time, as a TSQR leaf does.  ``A'B`` sums each block's
    weighted rows of ``B`` per support row, one span of rows at a time, and multiplies
    the sums by the centred support once.  A spline-by-level block is zero
    off its level: its raw support's table is added on the level's rows and
    its centre on every row (``_leverage_rows``' support route), so its
    entries are bounded by ``(|x| + |c|)' |basis|`` rather than ``|x - c|'
    |basis|``.

    **Error.**  A table entry is a centring and a dot product of length
    ``w_g`` (``gamma_(w_g + 2)`` of ``|S - c|' |basis_g|``, Higham 2002,
    secs. 3.1 and 3.5) and a row of ``B`` sums at most ``T`` groups and is
    scaled once, so every entry of ``B`` lies within ``gamma_(p + T + 3)``
    of ``|a_r|' |basis|``; the TSQR of ``B`` is columnwise backward stable
    (``rank._tsqr_weighted_factor``).  ``A'B`` adds a sum over the rows of one
    support row's sum over a span's rows, one add per span and a
    product of length ``s_g``; the third value returned counts every rounding
    an entry of ``A'B`` carries, so ``gamma`` of it times ``|A|' |B|`` bounds
    its error (``rank.decompose_gram_deflated``).
    """
    from superglm._group_matrix._group_matrix_discretized import (
        DiscretizedSplineCategoricalGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_kernels import (
        _gather_table_rows,
        _gather_table_rows_at,
        _scatter_table_rows,
        _scatter_table_rows_at,
    )
    from superglm.solvers.rank import streamed_weighted_factor, tsqr_leaf_rows

    basis = np.ascontiguousarray(basis, dtype=np.float64)
    n, p = dm.n, dm.p
    width = basis.shape[1]
    sqrt_weights = np.sqrt(np.asarray(W, dtype=np.float64))
    # Each plan: (columns, kind, table, accumulator, a, b).  "full": every row
    # indexes a centred support row (a: codes).  "level": a spline-by-level
    # block, zero outside its level; its raw support table is added on the
    # level's rows (a: rows, b: codes, both in row order) and its centre
    # enters every row once through ``base``.  "rows": dense rows a block at
    # a time (a: the group matrix).
    plans: list[tuple] = []
    supports: dict[int, NDArray] = {}
    magnitudes = np.zeros(p)
    base = np.zeros(width)
    level_type = DiscretizedSplineCategoricalGroupMatrix
    parts = _design_leaf_parts(dm).parts
    groups = dm.group_matrices if len(parts) == len(dm.group_matrices) else [None] * len(parts)
    for (column, kind, first, second), group in zip(parts, groups, strict=True):
        if kind == "rows" and _is_plain_level(group, level_type):
            level = cast(DiscretizedSplineCategoricalGroupMatrix, group)
            columns = slice(column, column + int(level.shape[1]))
            raw = np.asarray(level.B_unique, dtype=np.float64) @ np.asarray(
                level.R_inv, dtype=np.float64
            )
            order = np.argsort(level.row_idx, kind="stable")
            rows = np.ascontiguousarray(np.asarray(level.row_idx)[order], dtype=np.intp)
            codes = np.ascontiguousarray(np.asarray(level.bin_idx_level)[order], dtype=np.intp)
            centre = _centred_support(np.zeros((1, raw.shape[1])), center, center_lo, columns)
            base += (centre @ basis[columns])[0]
            level_weight = np.bincount(
                codes, weights=np.asarray(W, dtype=np.float64)[rows], minlength=raw.shape[0]
            )
            magnitudes[columns] = np.sqrt(level_weight @ (raw * raw)) + np.abs(
                centre[0]
            ) * math.sqrt(float(np.sum(W)))
            table = np.ascontiguousarray(raw @ basis[columns])
            supports[column] = raw
            plans.append((columns, "level", table, np.zeros((raw.shape[0], width)), rows, codes))
            continue
        if kind == "rows":
            columns = slice(column, column + int(first.shape[1]))
            plans.append((columns, "rows", None, None, first, None))
            continue
        if kind == "one_hot":
            levels = int(second)
            columns = slice(column, column + levels)
            # code ``levels`` is no level: the centred zero row
            support = _centred_support(
                np.vstack((np.eye(levels), np.zeros((1, levels)))), center, center_lo, columns
            )
            codes = first
        else:
            columns = slice(column, column + first.shape[1])
            support = _centred_support(first, center, center_lo, columns)
            codes = second
        table = np.ascontiguousarray(support @ basis[columns])
        supports[column] = support
        plans.append((columns, "full", table, np.zeros((support.shape[0], width)), codes, None))
    cross = np.zeros((p, width))
    weighted_total = np.zeros(width)
    # Narrow blocks: a leaf of a few thousand rows keeps LAPACK's panel in
    # cache (measured 1.5x to 2.6x faster than ``tsqr_leaf_rows`` at k = 55
    # to 330, 540,000 rows); the reduction tree is the same TSQR's.
    block_rows = min(tsqr_leaf_rows(width), max(_GREY_LEAF_ROWS, 2 * width))
    span_rows = max(block_rows, _GREY_SPAN_ROWS)
    locals_ = {id(plan[3]): np.zeros_like(plan[3]) for plan in plans if plan[3] is not None}

    def blocks() -> Iterator[tuple[int, int, NDArray]]:
        for start in range(0, n, block_rows):
            stop = min(start + block_rows, n)
            projected = np.zeros((stop - start, width))
            projected += base
            dense = []
            spans = []
            for columns, kind, table, _, first, second in plans:
                if kind == "full":
                    _gather_table_rows(projected, table, first, start)
                elif kind == "level":
                    low, high = np.searchsorted(first, (start, stop))
                    spans.append((int(low), int(high)))
                    _gather_table_rows_at(projected, table, first, second, low, high, start)
                else:
                    rows = np.arange(start, stop, dtype=np.intp)
                    values = _centred_support(
                        np.asarray(first.row_subset(rows).toarray(), dtype=np.float64),
                        center,
                        center_lo,
                        columns,
                    )
                    projected += values @ basis[columns]
                    dense.append((columns, values))
            projected *= sqrt_weights[start:stop, None]
            weighted = projected * sqrt_weights[start:stop, None]
            weighted_total[:] += weighted.sum(axis=0)
            level = iter(spans)
            flush = stop == n or stop // span_rows != start // span_rows
            for _, kind, _, accumulator, first, second in plans:
                if kind == "rows":
                    continue
                # A span's own sums first, so a support row's total rounds
                # over one span's rows plus one add per span.
                local = locals_[id(accumulator)]
                if kind == "full":
                    _scatter_table_rows(local, weighted, first, start)
                else:
                    low, high = next(level)
                    _scatter_table_rows_at(local, weighted, first, second, low, high, start)
                if flush:
                    accumulator += local
                    local[...] = 0.0
            for columns, values in dense:
                cross[columns] += values.T @ weighted
            yield start, stop, projected
        for columns, kind, _, accumulator, _, _ in plans:
            if kind == "full":
                cross[columns] += supports[columns.start].T @ accumulator
            elif kind == "level":
                cross[columns] += supports[columns.start].T @ accumulator
                centre = _centred_support(
                    np.zeros((1, columns.stop - columns.start)), center, center_lo, columns
                )
                cross[columns] += centre.T @ weighted_total[None, :]

    factor = streamed_weighted_factor(blocks(), np.ones(n))
    # Roundings an entry of ``A'B`` can carry beyond ``|A|' |B|``'s scale:
    # the centring and table product (``p + 2``), at most ``T`` group adds and
    # two scalings per row of ``B``, one span's rows plus one add per span
    # (one add per block for the dense rows and the weighted total), and the
    # product with the widest support.
    widest = max((plan[2].shape[0] for plan in plans if plan[2] is not None), default=0)
    roundings = (
        (p + 2) + len(plans) + 3 + span_rows + -(-n // span_rows) + -(-n // block_rows) + widest + 1
    )
    return factor, cross, roundings, magnitudes


def chunked_projected_factor(
    chunks: Callable[[], Iterator[tuple[int, int, NDArray]]],
    n: int,
    W: NDArray,
    basis: NDArray,
    *,
    center: NDArray | None = None,
    center_lo: NDArray | None = None,
) -> tuple[NDArray, NDArray, int, NDArray]:
    """:func:`grouped_projected_factor` for rows given as dense blocks ``(start, stop, X)``.

    Each block is centred as a TSQR leaf centres it, ``((X - c) - c_lo)``,
    and is one leaf of the TSQR of ``B = A basis``; ``A'B`` adds the blocks'
    products in order.  **Error.**  An entry of ``B`` is a centring (two
    roundings), a product of length ``p`` and a scaling by a rounded
    ``sqrt(w)``; an entry of ``A'B`` adds one more scaling, a block's ``L``
    rows and one add per block, so ``gamma`` of the count returned times
    ``|A|' |B|`` bounds it (``rank.decompose_gram_deflated``).  The rows are
    centred first, so no column magnitude exceeds its weighted norm.
    """
    from superglm.solvers.rank import streamed_weighted_factor

    basis = np.ascontiguousarray(basis, dtype=np.float64)
    p = basis.shape[0]
    sqrt_weights = np.sqrt(np.asarray(W, dtype=np.float64))
    cross = np.zeros((p, basis.shape[1]))
    longest, count = 0, 0

    def blocks() -> Iterator[tuple[int, int, NDArray]]:
        nonlocal longest, count
        for start, stop, rows in chunks():
            values = _centred_support(
                np.asarray(rows, dtype=np.float64), center, center_lo, slice(0, p)
            )
            projected = (values @ basis) * sqrt_weights[start:stop, None]
            np.add(cross, values.T @ (projected * sqrt_weights[start:stop, None]), out=cross)
            longest, count = max(longest, stop - start), count + 1
            yield start, stop, projected

    factor = streamed_weighted_factor(blocks(), np.ones(n))
    return factor, cross, p + 5 + longest + count, np.zeros(p)


def refresh_centered_rhs(
    *,
    system: CenteredSystem,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
) -> CenteredSystem:
    """Reuse an invariant centered Gram while refreshing its working RHS."""
    sum_w, mean_x, mean_z, data_gram, rhs, mean_hi, mean_lo = _refresh_centered_data_rhs(
        dm=dm,
        W=W,
        z_off=z_off,
        data=(system.sum_w, system.mean_x, system.data_gram, system.mean_hi, system.mean_lo),
    )
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=mean_x,
        mean_z=mean_z,
        data_gram=data_gram,
        rhs=rhs,
        penalty=system.penalty,
        hessian=system.hessian,
        mean_hi=mean_hi,
        mean_lo=mean_lo,
    )


def _refresh_centered_data_rhs(*, dm, W, z_off, data):
    """``(X - 1 m')' W (z - mean_z)`` for a cached Gram, by column type.

    Every column takes its transpose product less ``mean_x`` times the sum of
    ``W (z - mean_z)``, which is zero to its rounding; a ``DenseGroupMatrix``
    column, whose entries its type does not bound, is centred row by row about
    the system's exact centre pair (``centre_pair``) instead, so a column's
    offset never multiplies that rounding (issue #430).
    """
    sum_w, mean_x, data_gram, mean_hi, mean_lo = data
    mean_z = float(np.dot(W, z_off) / sum_w)
    z_centered = z_off - mean_z
    weighted_z = W * z_centered
    rhs = dm.rmatvec(weighted_z) - mean_x * float(np.sum(weighted_z))
    dense = dense_columns(dm)
    if np.any(dense):
        centre = mean_x if mean_hi is None else mean_hi
        centred = dense_centred_rmatvec(dm, weighted_z, centre, mean_lo)
        rhs = np.where(dense, centred, rhs)
    return sum_w, mean_x, mean_z, data_gram, _freeze(rhs), mean_hi, mean_lo


def build_centered_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_off: NDArray,
    penalty: NDArray,
    tabmat_split=None,
    tabmat_state: TabmatCenteringState | None = None,
    profile: dict | None = None,
    _force_chunked: bool = False,
    _data: tuple | None = None,
) -> CenteredSystem:
    """Build a stably centered data Gram, RHS, and penalized Hessian."""
    n, p = dm.shape
    W = np.asarray(W, dtype=float)
    z_off = np.asarray(z_off, dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if W.shape != (n,) or z_off.shape != (n,):
        raise ValueError("W and z_off must match the design row count")
    if penalty.shape != (p, p):
        raise ValueError("penalty must have shape (p, p)")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("working weights must be finite and non-negative")

    sum_w = float(np.sum(W, dtype=np.float64))
    if not np.isfinite(sum_w) or sum_w <= 0.0:
        raise ValueError("working weights must have a positive finite sum")
    if _data is not None:
        *refreshed, mean_hi, mean_lo = _refresh_centered_data_rhs(
            dm=dm, W=W, z_off=z_off, data=_data
        )
        return _attach_centered_penalty(*refreshed, penalty, mean_hi=mean_hi, mean_lo=mean_lo)
    mean_z = float(np.dot(W, z_off) / sum_w)
    z_centered = z_off - mean_z
    # A design of lookups and dense columns: one weighted histogram (or
    # accumulator) per term pair, every table centred before its product
    # (``_lookup_tables``; Li & Wood 2020, Algorithm 1).  No raw moment is
    # subtracted, so no rung's value certificate is needed.
    plan = None if _force_chunked else table_gram_plan(dm)
    table = None if plan is None else table_centred_system(plan, dm, W, z_centered, sum_w)
    if table is not None:
        mean_x, data_gram, rhs, mean_hi, mean_lo = table
        if profile is not None:
            profile["centered_table_gram_hits"] = profile.get("centered_table_gram_hits", 0) + 1
        return _attach_centered_penalty(
            sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_hi, mean_lo=mean_lo
        )
    # A ``DenseGroupMatrix`` column, the one type whose entries its type does
    # not bound, never enters a raw-moment rung: each subtracts ``sum W x x'``
    # and ``(X'W)(X'W)' / sum W`` under a value certificate
    # (``_raw_centering_well_scaled``), so the arithmetic changed with the
    # column's location (offset 0: the raw-moment rung; offset 10: the exact
    # pair).  The rest of the design keeps the rungs: they run on its bounded
    # columns, and the dense columns join them centred about their exact pair
    # (``_attach_dense_split``).  A design with no bounded column, or whose
    # bounded part every rung rejects, takes the exact pair throughout.
    if not np.any(dense_columns(dm)):
        packed = _raw_rung_system(
            dm=dm,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            tabmat_split=tabmat_split,
            tabmat_state=tabmat_state,
            profile=profile,
            force_chunked=_force_chunked,
        )
        if packed is not None:
            mean_x, data_gram, rhs, _ = packed
            return _attach_centered_penalty(sum_w, mean_x, mean_z, data_gram, rhs, penalty)
    else:
        split = _dense_split(dm)
        if split.bounded.p:
            packed = _raw_rung_system(
                dm=split.bounded,
                W=W,
                z_centered=z_centered,
                sum_w=sum_w,
                tabmat_split=(
                    None if tabmat_split is None else split.bounded.tabmat_centering_split
                ),
                tabmat_state=tabmat_state,
                profile=profile,
                force_chunked=_force_chunked,
            )
            if packed is not None:
                return _attach_dense_split(
                    split,
                    W=W,
                    z_centered=z_centered,
                    sum_w=sum_w,
                    mean_z=mean_z,
                    packed=packed,
                    penalty=penalty,
                )

    dense = dense_columns(dm)
    if not np.any(dense):
        mean_x = dm.rmatvec(W) / sum_w
        data_gram, rhs = centered_gram_rhs(dm=dm, W=W, mean_x=mean_x, z_centered=z_centered)
        return _attach_centered_penalty(
            sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_x.copy()
        )
    # Beside a dense column the whole design is centred in one pass of rows:
    # every other column about its weighted mean, as above, and every dense
    # column by the corrected two-pass algorithm, as in ``_attach_dense_split``
    # -- rows about the rounded mean ``a`` (``dense_anchor``), its remainder
    # from the same rows' first moments by Björck's correction.  A design of
    # dense columns only has no other mean to read, so its raw ``X'W``, which
    # would be discarded, is not formed.
    anchor = dense_anchor(dm, W, sum_w)
    mean_hi = anchor if np.all(dense) else np.where(dense, anchor, dm.rmatvec(W) / sum_w)
    first = np.zeros(dm.p, dtype=np.float64)
    data_gram, rhs = centered_gram_rhs(
        dm=dm, W=W, mean_x=mean_hi, z_centered=z_centered, first=first
    )
    mean_lo = np.where(dense, first / sum_w, 0.0)
    data_gram = two_pass_centred_gram(first, data_gram, mean_lo, sum_w)
    rhs = rhs - mean_lo * float(np.dot(W, z_centered))
    mean_x = np.where(dense, mean_hi + mean_lo, mean_hi)
    return _attach_centered_penalty(
        sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_hi, mean_lo=mean_lo
    )


def _raw_rung_system(
    *,
    dm: DesignMatrix,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    tabmat_split,
    tabmat_state: TabmatCenteringState | None,
    profile: dict | None,
    force_chunked: bool,
) -> tuple[NDArray, NDArray, NDArray, tuple] | None:
    """``(mean_x, data_gram, rhs, recentred)`` from the first raw rung that accepts ``dm``, else ``None``.

    After the raw rungs, the compact anchor-support fallback, which subtracts
    no raw moment, and then column-local centring of the last raw moments a
    certificate rejected in this build (``column_local_centering``), which
    recentres only the failing columns, each inside its own group.  A build
    reaches that repair only where it used to take the chunked ``O(n p^2)``
    pass, so every other route is unchanged.  The raw-moment rung's refusal
    latches as before unless the repair served the build from that rung's
    own moments, which the next build needs again; after a factored-rung
    rejection it stays latched, which costs nothing, as the factored rung is
    never latched and forms the moments at every build.  ``recentred`` is
    the repair's recentred columns (``RecentredColumns``), empty on every
    other route.  Called only with a design free of ``DenseGroupMatrix``
    columns.
    """
    rejected = RawMomentRejection()
    packed = packed_centered_gram_rhs(
        dm=dm, W=W, z_centered=z_centered, state=tabmat_state, rejected=rejected
    )
    if packed is None and (tabmat_state is None or tabmat_state.eligible is not False):
        mixed_attempted, mixed = _try_mixed_discrete_centering(
            dm=dm,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            preflight=tabmat_state is None or tabmat_state.eligible is None,
        )
        if mixed_attempted:
            packed = mixed
            if tabmat_state is not None:
                # Only the first call pays for Tabmat's location/scale
                # preflight. Every changed weight vector still receives the
                # authoritative full-moment certificate, and a rejection
                # permanently selects stable chunks for this inner fit.
                tabmat_state.eligible = mixed is not None
    # Raw-spline CSC construction is worthwhile only for a caller that owns a
    # reusable fit-local policy state (direct PIRLS and REML do). One-shot
    # inference/finalization calls retain the bounded stable-chunk path.
    if (
        packed is None
        and tabmat_state is not None
        and tabmat_state.raw_spline_eligible is not False
    ):
        raw_spline_plan = dm.get_raw_spline_tabmat_centering_plan(profile=profile)
        if raw_spline_plan is not None:
            packed = _try_raw_spline_tabmat_centering(
                plan=raw_spline_plan,
                W=W,
                z_centered=z_centered,
                sum_w=sum_w,
                preflight=tabmat_state.raw_spline_eligible is None,
                profile=profile,
            )
            tabmat_state.raw_spline_eligible = packed is not None
            if packed is None and profile is not None:
                profile["centered_spline_tabmat_stable_fallbacks"] = (
                    profile.get("centered_spline_tabmat_stable_fallbacks", 0) + 1
                )
    if (
        packed is None
        and tabmat_split is not None
        and (tabmat_state is None or tabmat_state.eligible is not False)
    ):
        packed = _try_tabmat_centering(
            tabmat_split=tabmat_split,
            W=W,
            z_centered=z_centered,
            sum_w=sum_w,
            preflight=tabmat_state is None or tabmat_state.eligible is None,
        )
        if tabmat_state is not None:
            # A rejection is permanent for this fit.  Later IRLS weights can
            # change the centering ratio, but the stable path remains correct
            # and avoids repeating rejected raw work.
            tabmat_state.eligible = packed is not None
    # General raw-moment rung: the rungs above each require particular group
    # types, so ordinary spline designs reach the chunked pass below even though
    # the per-block moment dispatch can produce the same quantities far more
    # cheaply.  A rejected certificate latches for the fit, as the other
    # accelerated rungs do, so the moments are not recomputed every iteration.
    if (
        packed is None
        and not force_chunked
        and (
            tabmat_state is None
            # `eligible is False` means a preflight already certified this
            # design's raw moments as unsafe.  That verdict is about raw-moment
            # subtraction itself, not about tabmat, so this rung must honour it
            # rather than re-attempting the route that was just locked out.
            or (
                tabmat_state.eligible is not False and tabmat_state.raw_moment_eligible is not False
            )
        )
    ):
        # The factored tensor rung forms these very moments, by the same
        # execution-plan call on the same vectors, and certifies them the same
        # way: its rejection in this build is this rung's, and recomputing them
        # cost every rejected build a second raw-moment pass.  (The pattern
        # rung forms them by another summation, so its rejection does not
        # decide this rung's certificate.)
        packed = (
            None
            if rejected.source == "factored"
            else try_raw_moment_centering(
                dm=dm,
                W=W,
                weighted_z=W * z_centered,
                sum_w=sum_w,
                rejected=rejected,
            )
        )
        if tabmat_state is not None:
            tabmat_state.raw_moment_eligible = packed is not None
        if packed is not None and profile is not None:
            profile["centered_raw_moment_hits"] = profile.get("centered_raw_moment_hits", 0) + 1
    # Every raw rung declined: centre compact supports first rather than the
    # chunked rows below.  Only a design ``packed_centered_gram_rhs`` turned
    # away reaches it with something to do -- a discretized SCOP or
    # spline-by-category group, which that rung's tensor stages do not handle
    # -- and it subtracts no raw moment, so no certificate applies.
    if packed is None and not force_chunked:
        packed = anchor_support_centered_gram_rhs(dm=dm, W=W, z_centered=z_centered)
        if packed is not None and profile is not None:
            profile["centered_anchor_support_hits"] = (
                profile.get("centered_anchor_support_hits", 0) + 1
            )
    recentred = ()
    if packed is None and not force_chunked and rejected.source is not None:
        repaired = column_local_centering(dm=dm, W=W, rejected=rejected, sum_w=sum_w)
        if profile is not None:
            key = "declines" if repaired is None else "hits"
            profile[f"centered_column_local_{key}"] = (
                profile.get(f"centered_column_local_{key}", 0) + 1
            )
        if repaired is not None:
            *system, recentred = repaired
            packed = tuple(system)
            if profile is not None:
                profile["centered_column_local_columns"] = profile.get(
                    "centered_column_local_columns", 0
                ) + sum(len(group.columns) for group in recentred)
            if tabmat_state is not None and rejected.source == "raw_moment":
                # The rung's moments served this build through the repair, so
                # its refusal does not latch: the next build needs them again,
                # and a latched rung sent every later build to the chunked pass.
                tabmat_state.raw_moment_eligible = None
    return None if packed is None else (*packed, recentred)


@dataclass(frozen=True)
class _DenseSplit:
    """A design's ``DenseGroupMatrix`` groups and its other (bounded) groups, as two designs."""

    dense: DesignMatrix
    bounded: DesignMatrix
    dense_index: NDArray
    bounded_index: NDArray


# One split per design, built on first use.  The bounded design owns its own
# rung caches (execution plan, Tabmat split, bin-space plan, raw-spline Tabmat
# plan), so they persist across a fit's iterations.  Owner: the fit workspace
# that holds the design.  Lifetime: until the fit is published, when
# ``release_dense_split`` drops it with the full design's raw-spline plan
# (``capture_fit_state``), or until the design itself is collected.  The split
# depends only on the design's groups, never on weights, parameters, penalty or
# precision, so nothing else invalidates it; a later fit rebuilds it on use.
_DENSE_SPLITS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def release_dense_split(dm: DesignMatrix) -> None:
    """Drop ``dm``'s dense/bounded split and with it every cache of its bounded design."""
    _DENSE_SPLITS.pop(dm, None)


def _dense_split(dm: DesignMatrix) -> _DenseSplit:
    split = _DENSE_SPLITS.get(dm)
    if split is not None:
        return split
    dense_groups, bounded_groups = [], []
    dense_index: list[int] = []
    bounded_index: list[int] = []
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        columns = range(offset, offset + width)
        if type(matrix) is DenseGroupMatrix:
            dense_groups.append(matrix)
            dense_index.extend(columns)
        else:
            bounded_groups.append(matrix)
            bounded_index.extend(columns)
        offset += width
    split = _DenseSplit(
        dense=DesignMatrix(dense_groups, n=dm.n, p=len(dense_index)),
        bounded=DesignMatrix(bounded_groups, n=dm.n, p=len(bounded_index)),
        dense_index=np.asarray(dense_index, dtype=np.intp),
        bounded_index=np.asarray(bounded_index, dtype=np.intp),
    )
    _DENSE_SPLITS[dm] = split
    return split


def _attach_dense_split(
    split: _DenseSplit,
    *,
    W: NDArray,
    z_centered: NDArray,
    sum_w: float,
    mean_z: float,
    packed: tuple[NDArray, NDArray, NDArray, tuple],
    penalty: NDArray,
) -> CenteredSystem:
    """The centred system from a raw rung on the bounded columns and the corrected two-pass on the dense ones.

    Every dense column takes the corrected two-pass algorithm (Chan, Golub &
    LeVeque 1983, eq. 1.7: the textbook formula on rows shifted by the
    computed mean, with Björck's correction) about its working-weighted mean:

    - pass one rounds the mean, ``a``, from rows shifted by the first row that
      carries weight (``dense_anchor``, as ``corrected_two_pass_pair``'s pass
      one);
    - pass two forms the rows ``x - a`` once and from them ``e = sum W (x -
      a)``, ``G = sum W (x - a)(x - a)'`` and ``r = sum W z (x - a)``
      (``anchored_dense_moments``).

    The Gram is ``G - e l' - l e' + l l' sum W``, ``l = e / sum W``
    (``two_pass_centred_gram``), and the right-hand side ``r - l sum W z``.
    ``(a, l)`` is the exact pair of ``dense_mean_pair``, its remainder formed
    in pass two instead of a pass of its own.  The cross block is ``N~' W D~
    = N' (W (x - a)) - m_N e'`` (``dense_split_cross``); ``l``'s share, ``l
    (N'W - m_N sum W)``, is a product of two roundings.  The bounded columns'
    raw values never meet a dense column's offset, and the bounded block is
    the rung's own.
    """
    mean_bounded, gram_bounded, rhs_bounded, recentred = packed
    anchor = dense_anchor(split.dense, W, sum_w)
    first, gram, response = anchored_dense_moments(split.dense, W, anchor, z_centered)
    assert response is not None
    lo = first / sum_w
    gram_dense = two_pass_centred_gram(first, gram, lo, sum_w)
    rhs_dense = response - lo * float(np.dot(W, z_centered))
    cross = dense_split_cross(split, W, anchor, mean_bounded, first, recentred)
    data_gram = dense_split_gram(split, gram_bounded, gram_dense, cross)
    hi = anchor
    p = split.dense.p + split.bounded.p
    dense_index, bounded_index = split.dense_index, split.bounded_index
    rhs = np.empty(p, dtype=np.float64)
    rhs[bounded_index] = rhs_bounded
    rhs[dense_index] = rhs_dense
    mean_x = np.empty(p, dtype=np.float64)
    mean_x[bounded_index] = mean_bounded
    mean_x[dense_index] = hi + lo
    mean_hi = mean_x.copy()
    mean_hi[dense_index] = hi
    mean_lo = np.zeros(p, dtype=np.float64)
    mean_lo[dense_index] = lo
    return _attach_centered_penalty(
        sum_w, mean_x, mean_z, data_gram, rhs, penalty, mean_hi=mean_hi, mean_lo=mean_lo
    )


def dense_split_cross(
    split: _DenseSplit,
    W: NDArray,
    anchor: NDArray,
    mean_bounded: NDArray,
    first: NDArray,
    recentred: tuple = (),
    formed: dict | None = None,
) -> NDArray:
    """``(p_b, p_d)``: ``N' (W (x - a)) - m_N e'``, ``e = sum W (x - a)`` the dense pass's ``first``.

    The bounded design ``N`` centred about ``mean_bounded`` against the dense
    columns' rows about their ``anchor``; ``W`` may be signed.  The rows are
    formed in blocks of columns of at most ``_FACTOR_CHUNK_BYTES``, and each
    block meets every bounded group in one pass
    (``_bounded_transpose_products``), O(n) per group and block, a support
    group's projected support before any product.  A bounded column the raw
    rung recentred (``column_local_centering``) failed the raw-moment
    certificate, so its raw values would bring its ``kappa`` into the
    product: it meets ``W (x - a)`` through its centred support instead,
    ``sum_b c_j[b] sum_(r in b) W (x - a)``, the repair's own product with a
    failing partner (``recentred_products``), within the repair's stated
    envelope with ``A_k = |x - a|``.  ``formed`` holds the projected supports
    for a caller that forms the block again with the same factors.
    """
    formed = {} if formed is None else formed
    columns = list(_dense_columns_of(split.dense, anchor))
    n = split.dense.n
    cross = np.empty((split.bounded.p, len(columns)), dtype=np.float64)
    width = max(1, _FACTOR_CHUNK_BYTES // (8 * max(n, 1)))
    for start in range(0, len(columns), width):
        block = columns[start : start + width]
        stop = start + len(block)
        rows = np.empty((n, len(block)), dtype=np.float64)
        for column, (values, column_anchor) in enumerate(block):
            np.subtract(values, column_anchor, out=rows[:, column])
        cross[:, start:stop] = _bounded_transpose_products(split.bounded, rows, W, formed)
        cross[:, start:stop] -= np.outer(mean_bounded, first[start:stop])
        for group in recentred:
            sums = _weighted_bincount_2d(group.codes, _native(W), rows, group.values.shape[0])
            cross[group.columns, start:stop] = group.values.T @ sums
    return cross


def _bounded_transpose_products(
    dm: DesignMatrix, rows: NDArray, W: NDArray, formed: dict
) -> NDArray:
    """``X' diag(W) rows`` for an ``(n, k)`` block of rows, one pass of each group.

    A discretized SSP group (``B_unique``, ``R_inv`` float64 and in range)
    meets ``W rows`` summed by bin through its projected support ``B_unique
    R_inv``, formed once into ``formed``, as its raw moments do: its
    ``rmatvec`` sums ``B_unique`` first, so the projection multiplies that
    sum's rounding, ``gamma_n |R_inv|' |B_unique|' |W rows|``, which cancels
    where ``|B_unique| |R_inv|`` dwarfs the column (Higham 2002, eq. 3.13);
    projected first, the rows' sums meet only the column's own values.  A
    one-hot group sums by level, a SCOP group by bin against its support,
    and a sparse group multiplies its sparse basis by the block; any other
    group takes its ``rmatvec`` of each weighted column.
    """
    W = _native(W)
    products = np.empty((dm.p, rows.shape[1]), dtype=np.float64)
    weighted = None
    offset = 0
    for gm in dm.group_matrices:
        target = products[offset : offset + gm.shape[1]]
        offset += gm.shape[1]
        if type(gm) in (DiscretizedSSPGroupMatrix, SupportCompressedSSPGroupMatrix):
            if gm not in formed:
                in_range = (
                    gm.B_unique.dtype == gm.R_inv.dtype == np.float64
                    and not _ssp_gram_needs_exact(gm.B_unique, gm.R_inv)
                )
                formed[gm] = gm.B_unique @ gm.R_inv if in_range else None
            if formed[gm] is not None:
                target[:] = formed[gm].T @ _weighted_bincount_2d(gm.bin_idx, W, rows, gm.n_bins)
                continue
        elif type(gm) in ONE_HOT_CODE_TYPES:
            target[:] = _weighted_bincount_2d(gm.codes, W, rows, gm.n_levels + 1)[: gm.n_levels]
            continue
        elif type(gm) is DiscretizedSCOPGroupMatrix:
            sums = _weighted_bincount_2d(gm.bin_idx, W, rows, gm.n_bins)
            target[:] = gm.B_scop_unique.T @ sums
            continue
        if weighted is None:
            weighted = rows * W[:, None]
        if type(gm) is SparseSSPGroupMatrix:
            target[:] = gm.R_inv.T @ np.asarray(gm.B.T @ weighted)
        elif type(gm) is SparseGroupMatrix:
            target[:] = np.asarray(gm.M.T @ weighted)
        else:
            for column in range(rows.shape[1]):
                target[:, column] = gm.rmatvec(np.ascontiguousarray(weighted[:, column]))
    return products


def dense_split_gram(
    split: _DenseSplit, bounded: NDArray, dense: NDArray, cross: NDArray
) -> NDArray:
    """The ``(p, p)`` Gram in design order from its bounded, dense and cross (bounded by dense) blocks."""
    p = split.dense.p + split.bounded.p
    dense_index, bounded_index = split.dense_index, split.bounded_index
    gram = np.empty((p, p), dtype=np.float64)
    gram[np.ix_(bounded_index, bounded_index)] = bounded
    gram[np.ix_(dense_index, dense_index)] = dense
    gram[np.ix_(bounded_index, dense_index)] = cross
    gram[np.ix_(dense_index, bounded_index)] = cross.T
    return gram


def _dense_chunks(values: NDArray, n: int):
    """``corrected_two_pass_pair``'s chunk source over one dense block's rows."""

    def chunks():
        for start in range(0, n, _FACTOR_CHUNK_ROWS):
            stop = min(start + _FACTOR_CHUNK_ROWS, n)
            yield start, stop, values[start:stop]

    return chunks


def dense_anchor(dm: DesignMatrix, W: NDArray, sum_w: float) -> NDArray:
    """``a``: each dense column's working-weighted mean, rounded; 0 on every other column.

    Pass one of the corrected two-pass algorithm
    (``mode_score.rounded_weighted_mean``): ``fl(x_ref + sum W (x - x_ref) /
    sum W)`` about the first row that carries weight, so a column constant on
    its weighted rows has ``a`` exactly that constant and centres to exact
    zeros, and ``a`` lies within ``u |m| + gamma_k sum |W| |x - x_ref| / |sum
    W|`` of the mean ``m``.
    """
    W = np.asarray(W, dtype=np.float64)
    anchor = np.zeros(dm.p, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            anchor[offset : offset + width] = rounded_weighted_mean(
                _dense_chunks(matrix.M, dm.n), W, sum_w, width
            )
        offset += width
    return anchor


def _dense_sources(dm: DesignMatrix, anchor: NDArray) -> list[tuple[NDArray, NDArray]]:
    """Per dense block ``(values, anchor)``, the anchor read from ``anchor``."""
    sources = []
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            centre = np.asarray(anchor[offset : offset + width], dtype=np.float64)
            sources.append((matrix.M, centre))
        offset += width
    return sources


def _dense_columns_of(dm: DesignMatrix, anchor: NDArray) -> Iterator[tuple[NDArray, float]]:
    """``(x_k, a_k)`` for each dense column of ``dm`` in order: its values and its anchor."""
    for values, centre in _dense_sources(dm, anchor):
        for index in range(values.shape[1]):
            yield values[:, index], float(centre[index])


def anchored_dense_moments(
    dm: DesignMatrix,
    W: NDArray,
    anchor: NDArray,
    response: NDArray | None = None,
    out: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """``(e, G, r)`` over the dense columns of ``dm``, rows ``fl(x - anchor)``, in one pass.

    ``e = sum W (x - a)``, ``G = sum W (x - a)(x - a)'`` and ``r = sum W z (x
    - a)`` (``z`` the ``response``; ``r`` is ``None`` without one), each
    accumulated across row chunks with Kahan's compensation, in
    ``dense_columns`` order.  Each row is formed by one subtraction.  ``out``
    (``(n, p_d)``, a design of one dense block) receives the rows.
    """
    W = np.asarray(W, dtype=np.float64)
    sources = _dense_sources(dm, np.asarray(anchor, dtype=np.float64))
    width = sum(values.shape[1] for values, _ in sources)
    first = np.zeros(width, dtype=np.float64)
    first_compensation = np.zeros_like(first)
    gram = np.zeros((width, width), dtype=np.float64)
    gram_compensation = np.zeros_like(gram)
    rhs = None if response is None else np.zeros(width, dtype=np.float64)
    rhs_compensation = np.zeros(width, dtype=np.float64)
    if out is not None and len(sources) != 1:
        raise ValueError("out holds the rows of a design with one dense block")
    for start in range(0, dm.n, _FACTOR_CHUNK_ROWS):
        stop = min(start + _FACTOR_CHUNK_ROWS, dm.n)
        if out is not None:
            values, centre = sources[0]
            rows = out[start:stop]
            np.subtract(values[start:stop], centre, out=rows)
        else:
            blocks = [values[start:stop] - centre for values, centre in sources]
            rows = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
        weights = W[start:stop]
        weighted = rows * weights[:, None]
        _compensated_add(first, first_compensation, rows.T @ weights)
        _compensated_add(gram, gram_compensation, rows.T @ weighted)
        if rhs is not None and response is not None:
            _compensated_add(rhs, rhs_compensation, weighted.T @ response[start:stop])
    return first, 0.5 * (gram + gram.T), rhs


def two_pass_centred_gram(
    first: NDArray, gram: NDArray, remainder: NDArray, sum_w: float
) -> NDArray:
    """``sum W (x - a - l)(x - a - l)'`` from the moments about ``a``, symmetrized.

    ``G - e l' - l e' + l l' sum W`` with ``e = sum W (x - a)`` and ``G = sum
    W (x - a)(x - a)'`` (``anchored_dense_moments``, ``centered_gram_rhs``'s
    ``first``) and ``l`` the remainder, ``e / sum W`` on a dense column and 0
    on any other.  On the dense columns this is Björck's correction ``G - e e'
    / sum W`` of the corrected two-pass algorithm (Chan, Golub & LeVeque 1983,
    eq. 1.7); about the rounded mean ``a`` the remainder is pass one's
    rounding, so the correction is a second-order term and the Gram keeps the
    two-pass bound, ``n u (1 + n^2 u^2 kappa^2)`` relative (their Table 1).
    """
    centred = gram - np.outer(first, remainder) - np.outer(remainder, first)
    centred = centred + sum_w * np.outer(remainder, remainder)
    return 0.5 * (centred + centred.T)


def dense_mean_pair(dm: DesignMatrix, W: NDArray, sum_w: float) -> tuple[NDArray, NDArray] | None:
    """``(hi, lo)`` over all ``p`` columns, zero off the ``DenseGroupMatrix`` ones; ``None`` without one.

    Each dense column's weighted mean ``sum W x / sum W`` as an exact pair,
    by the corrected two-pass algorithm (``mode_score.corrected_two_pass_pair``):
    ``hi`` the rounded mean, ``lo`` the remainder formed on rows differenced
    from it.  ``W`` may be signed (observed geometry); ``sum_w`` is the
    caller's own.
    """
    if not any(type(matrix) is DenseGroupMatrix for matrix in dm.group_matrices):
        return None
    W = np.asarray(W, dtype=np.float64)
    hi = np.zeros(dm.p, dtype=np.float64)
    lo = np.zeros(dm.p, dtype=np.float64)
    offset = 0
    for matrix in dm.group_matrices:
        width = matrix.shape[1]
        if type(matrix) is DenseGroupMatrix:
            hi[offset : offset + width], lo[offset : offset + width] = corrected_two_pass_pair(
                _dense_chunks(matrix.M, dm.n), W, sum_w, width
            )
        offset += width
    return hi, lo


def weighted_mean_pair(
    dm: DesignMatrix, W: NDArray, sum_w: float
) -> tuple[NDArray, NDArray, NDArray | None]:
    """``(mean_x, hi, lo)``: the weighted column mean and the exact pair it rounds.

    One-engine design §3.2 applied to gram's centred system: a
    ``DenseGroupMatrix`` column's mean is ``hi + lo`` (``dense_mean_pair``),
    ``hi`` the rounded mean from a pass shifted by the first row that
    carries weight and ``lo`` the remainder formed on rows differenced from
    ``hi`` (Chan, Golub & LeVeque 1983).  A column constant on its weighted
    rows centres to exact zeros (the unshifted ``sum W x / sum W`` left a
    rounding-level centred diagonal whose rank decision moved the REML
    objective by ~30 between iterates: stage-1 verifier, the Gamma/log
    constant-column fit), and a far row of negligible weight no longer sets
    the remainder's scale, as it did as the anchor.  Rows are centred as
    ``(x - hi) - lo``, which rounds at the column's spread; ``x - mean_x``
    with ``mean_x = fl(hi + lo)`` rounds at ``u |mean|`` and adds ``sum W d
    d'`` (``d`` that rounding) to the profiled Gram, which at a 1e16 offset
    moved a Gaussian fit's eta by 1e-3 (issue #430).  Every other column,
    bounded by its type, has ``hi = mean_x = X'W / sum W`` and ``lo = 0``;
    ``lo`` is ``None`` for a design without a dense column.
    """
    mean = dm.rmatvec(W) / sum_w
    pair = dense_mean_pair(dm, W, sum_w)
    if pair is None:
        return mean, mean.copy(), None
    dense = dense_columns(dm)
    hi = np.where(dense, pair[0], mean)
    lo = np.where(dense, pair[1], 0.0)
    return np.where(dense, pair[0] + pair[1], mean), hi, lo


def _attach_centered_penalty(
    sum_w, mean_x, mean_z, data_gram, rhs, penalty, *, mean_hi=None, mean_lo=None
):
    """Attach a fresh trial penalty using the same numerical checks on every path."""
    penalty_symmetric = 0.5 * (penalty + penalty.T)
    hessian = data_gram + penalty_symmetric
    # Both terms are mathematically PSD. Degenerate spline
    # reparameterizations can introduce visible negative round-off, so project
    # only this declared-PSD system back onto its valid cone before rank work.
    # Keep structurally empty coordinates outside the eigensolve: reconstructing
    # the full matrix can scatter round-off into exact zero rows, manufacture
    # numerical rank, and obscure an empty-support spline direction.
    active = np.flatnonzero(np.any(hessian != 0.0, axis=0) | np.any(hessian != 0.0, axis=1))
    active_hessian = hessian[np.ix_(active, active)]
    try:
        np.linalg.cholesky(active_hessian)
    except np.linalg.LinAlgError:
        hessian_eigenvalues, hessian_eigenvectors = np.linalg.eigh(active_hessian)
        if hessian_eigenvalues.size and hessian_eigenvalues[0] < 0.0:
            active_hessian = (
                hessian_eigenvectors * np.maximum(hessian_eigenvalues, 0.0)[None, :]
            ) @ hessian_eigenvectors.T
            active_hessian = 0.5 * (active_hessian + active_hessian.T)
            hessian = np.zeros_like(hessian)
            hessian[np.ix_(active, active)] = active_hessian
    return CenteredSystem(
        sum_w=sum_w,
        mean_x=_freeze(mean_x),
        mean_z=mean_z,
        data_gram=_freeze(data_gram),
        rhs=_freeze(rhs),
        penalty=_freeze(penalty_symmetric),
        hessian=_freeze(hessian),
        mean_hi=None if mean_hi is None else _freeze(mean_hi),
        mean_lo=None if mean_lo is None else _freeze(mean_lo),
    )
