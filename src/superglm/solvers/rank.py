"""Versioned numerical-rank policy and retained-subspace operations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


@dataclass(frozen=True)
class RankPolicy:
    version: int
    factor_rcond: float
    gram_rcond: float
    certification_band: float
    warning_condition: float
    severe_condition: float


_EPS = np.finfo(float).eps
SHARED_RANK_POLICY = RankPolicy(
    version=1,
    factor_rcond=float(np.sqrt(_EPS)),
    gram_rcond=float(_EPS),
    certification_band=32.0,
    warning_condition=float(1.0 / np.sqrt(_EPS)),
    severe_condition=float(1.0 / _EPS),
)


# Largest entry magnitude for which ``M + M.T`` provably cannot overflow: the
# sum is bounded entrywise by twice this.  See ``_symmetric_part``.
_HALF_MAX = float(np.finfo(float).max) / 2.0


def _symmetric_part(values: NDArray) -> NDArray:
    """``0.5 * (M + M.T)``, computed so a finite ``M`` cannot overflow to ``inf``.

    ``M + M.T`` is formed at full magnitude, so a finite ``M`` whose entries
    exceed half the float range overflows before the halving can bring it back.
    ``[[1e308]]`` sums to ``inf``; ``decompose_gram`` then either refuses the
    matrix outright or -- once the caller has pre-symmetrized -- equilibrates
    ``inf / inf`` to ``nan`` and returns a silently wrong answer.  Halving each
    operand first, ``0.5 * M + 0.5 * M.T``, cannot overflow at all: both terms
    are bounded by ``max / 2``, so their sum is bounded by ``max``.

    The two forms are **not** interchangeable, which is why this is a branch
    rather than a rewrite.  Halving is exact only while the halved value stays
    normal, so the split form rounds where the joint form does not.  Swept over
    1.05e6 exhaustive subnormal pairs, 1.05e6 pairs straddling the
    normal/subnormal boundary, 8e3 random subnormal pairs and 1e4 random normal
    pairs spanning the full exponent range, the two forms differ on 393726,
    393728, 2866 and **0** of those respectively.  The normal-range count is the
    load-bearing one -- the forms agree bitwise whenever both operands are
    normal and the sum does not overflow -- but the subnormal disagreement is
    real, and it costs the guarantee that an exactly symmetric ``M`` is
    reproduced bitwise: at ``M = [[3 * 5e-324]]`` the split form returns
    ``4 * 5e-324`` because ``0.5 * M`` rounds to even, while the joint form is
    exact.

    So the joint form is kept verbatim wherever it is provably safe --
    ``max|M| <= max / 2`` bounds ``|M + M.T|`` by ``max`` entrywise -- and the
    split form is taken only in the regime the joint form cannot represent.
    Every in-tree Gram matrix (``XtWX + S``, ``X'X + lambda*P``) is many orders
    below that bound, so this is bitwise inert for all of them.

    Non-finite input needs no special handling: ``max|M|`` is then ``inf`` or
    ``nan``, neither of which satisfies the bound, and the split form
    propagates the ``inf``/``nan`` exactly as the joint form does.
    """
    if float(np.abs(values).max(initial=0.0)) <= _HALF_MAX:
        return 0.5 * (values + values.T)
    return 0.5 * values + 0.5 * values.T


def diagonal_of_square(matrix: NDArray) -> NDArray:
    """Return ``diag(matrix @ matrix)`` with an O(p²) contraction."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return np.einsum("ij,ji->i", matrix, matrix, optimize=True)


def streamed_weighted_factor(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    *,
    center: NDArray | None = None,
) -> NDArray:
    """Build a compact QR factor from bounded weighted row chunks."""
    weights = np.asarray(weights, dtype=float)
    factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        block = np.sqrt(weights[start:stop])[:, None] * block
        stacked = block if factor is None else np.vstack((factor, block))
        factor = np.linalg.qr(stacked, mode="r")
    return np.empty((0, width)) if factor is None else np.asarray(factor)


def streamed_weighted_factor_rhs(
    chunks: Iterable[tuple[int, int, NDArray]],
    weights: NDArray,
    response: NDArray,
    *,
    center: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Build a compact weighted QR factor and its consistently transformed RHS.

    Appending the response to every bounded design chunk preserves ``Q.T @ b``
    without retaining either the observation matrix or the observation-length
    orthogonal factor.  The returned factor has at most ``p + 1`` rows.
    """
    weights = np.asarray(weights, dtype=float)
    response = np.asarray(response, dtype=float)
    if weights.ndim != 1 or response.shape != weights.shape:
        raise ValueError("weights and response must be matching vectors")
    joint_factor: NDArray | None = None
    width = 0 if center is None else len(center)
    for start, stop, values in chunks:
        block = np.asarray(values, dtype=float)
        width = block.shape[1]
        if center is not None:
            block = block - center
        sqrt_weights = np.sqrt(weights[start:stop])
        joint_block = np.column_stack(
            (sqrt_weights[:, None] * block, sqrt_weights * response[start:stop])
        )
        stacked = joint_block if joint_factor is None else np.vstack((joint_factor, joint_block))
        joint_factor = np.linalg.qr(stacked, mode="r")
    if joint_factor is None:
        return np.empty((0, width)), np.empty(0)
    return np.asarray(joint_factor[:, :width]), np.asarray(joint_factor[:, width])


def _certification_required(
    *,
    method: str,
    width: int,
    rank: int,
    pre_truncation_condition: float,
    resolution_limited: bool,
    policy: RankPolicy,
) -> bool:
    """The certification predicate, over the five fields that decide it.

    ``decompose_gram`` knows all five before it builds the retained subspace,
    so the predicate is kept callable without a decomposition in hand --
    otherwise the eager path and the deferring path would each carry their own
    copy of the band, free to drift apart.
    """
    if method == "qr_svd":
        # A factor decomposition is already the authoritative certificate;
        # never stream and factor the same rows again merely because the
        # factor policy itself truncated a nonzero singular value.
        return False
    certification_condition = policy.warning_condition / np.sqrt(policy.certification_band)
    return bool(
        width > 0
        and (
            (rank == width and pre_truncation_condition >= certification_condition)
            or (rank < width and resolution_limited)
        )
    )


def needs_factor_certification(
    decomposition: RankDecomposition,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
) -> bool:
    """Whether Gram geometry lies inside a band requiring factor certification.

    A certificate governs the retained subspace as well as the integer rank.
    Normal equations can erase a factor-scale direction at the numerical
    boundary, or retain a different direction while reporting the same rank.
    """
    return _certification_required(
        method=decomposition.method,
        width=decomposition.width,
        rank=decomposition.rank,
        pre_truncation_condition=decomposition.pre_truncation_condition,
        resolution_limited=decomposition.resolution_limited,
        policy=policy,
    )


def _freeze(values: NDArray, *, dtype=float) -> NDArray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RankDecomposition:
    policy_version: int
    method: Literal["empty", "cholesky", "pivoted_cholesky", "gram_eigh", "qr_svd"]
    column_scale: NDArray
    active_columns: NDArray
    rank: int
    pre_truncation_condition: float
    cutoff: float
    rank_truncated: bool
    used_svd_fallback: bool
    resolution_limited: bool
    log_pdet: float
    cholesky_factor: NDArray | None = None
    pivots: NDArray | None = None
    solution_basis: NDArray | None = None
    parameter_null_basis: NDArray | None = None
    estimable_functional_basis: NDArray | None = None
    structural_aliases: NDArray | None = None
    retained_values: NDArray | None = None
    factor_rhs_left_basis: NDArray | None = None
    factor_rhs_triangular: NDArray | None = None

    @property
    def width(self) -> int:
        return int(self.column_scale.size)

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != (self.width,):
            raise ValueError("rhs width does not match decomposition")
        if self.rank == 0:
            return np.zeros_like(rhs)
        if self.cholesky_factor is not None:
            active_rhs = rhs[self.active_columns] / self.column_scale[self.active_columns]
            active_solution = scipy.linalg.cho_solve(
                (self.cholesky_factor, True), active_rhs, check_finite=False
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution / self.column_scale[self.active_columns]
            return result
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        return self.solution_basis @ ((self.solution_basis.T @ rhs) / self.retained_values)

    def solve_factor_rhs(self, transformed_rhs: NDArray) -> NDArray:
        """Solve from a response transformed with the certified factor's QR.

        This path avoids re-forming normal equations at the factor-rank
        boundary.  It is available only when ``decompose_factor`` was asked to
        retain the bounded factor solve.
        """
        if self.factor_rhs_left_basis is None:
            raise RuntimeError("factor-RHS solve was not retained")
        transformed_rhs = np.asarray(transformed_rhs, dtype=float)
        if transformed_rhs.shape != (self.factor_rhs_left_basis.shape[0],):
            raise ValueError("transformed RHS length does not match the certified factor")
        if self.rank == 0:
            return np.zeros(self.width)
        projected_rhs = self.factor_rhs_left_basis.T @ transformed_rhs
        if self.factor_rhs_triangular is not None:
            active_solution = scipy.linalg.solve_triangular(
                self.factor_rhs_triangular,
                projected_rhs,
                lower=False,
                check_finite=False,
            )
            result = np.zeros(self.width)
            result[self.active_columns] = active_solution
            return result
        if self.solution_basis is None:
            raise RuntimeError("retained factor solution basis is unavailable")
        return self.solution_basis @ projected_rhs

    def pseudo_inverse(self) -> NDArray:
        if self.rank == 0:
            return np.zeros((self.width, self.width))
        if self.cholesky_factor is not None:
            inverse_equilibrated = scipy.linalg.cho_solve(
                (self.cholesky_factor, True),
                np.eye(len(self.active_columns)),
                check_finite=False,
            )
            inverse = np.zeros((self.width, self.width))
            scale = self.column_scale[self.active_columns]
            inverse[np.ix_(self.active_columns, self.active_columns)] = (
                inverse_equilibrated / np.outer(scale, scale)
            )
            return 0.5 * (inverse + inverse.T)
        if self.solution_basis is None or self.retained_values is None:
            raise RuntimeError("retained spectral basis is unavailable")
        inverse = (self.solution_basis / self.retained_values) @ self.solution_basis.T
        return 0.5 * (inverse + inverse.T)

    def retained_parameter_basis(self) -> NDArray:
        if self.solution_basis is not None:
            return self.solution_basis.copy()
        basis = np.zeros((self.width, self.rank))
        if self.rank:
            basis[self.active_columns, :] = np.diag(1.0 / self.column_scale[self.active_columns])
        return basis

    def null_basis(self) -> NDArray:
        if self.parameter_null_basis is None:
            return np.zeros((self.width, 0))
        return self.parameter_null_basis.copy()

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != (self.width,):
            raise ValueError("contrast width does not match decomposition")
        scaled_columns = self.column_scale > 0.0
        contrast_norm = float(np.linalg.norm(contrast))
        structural_tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            contrast_norm,
            np.finfo(float).tiny,
        )
        if np.linalg.norm(contrast[~scaled_columns]) > structural_tolerance:
            return False
        null = self.null_basis()
        if null.shape[1] == 0:
            return True

        # Test orthogonality in the equilibrated dual coordinates used by the
        # rank decision.  Comparing ``contrast @ parameter_null_basis`` against
        # an unscaled absolute tolerance makes exact aliases appear estimable
        # when one design column is multiplied by a large constant.
        scaled_contrast = contrast[scaled_columns] / self.column_scale[scaled_columns]
        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            return True
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        projection = scaled_contrast @ normalized_null
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(
            float(np.linalg.norm(scaled_contrast)),
            np.finfo(float).tiny,
        )
        return bool(np.linalg.norm(projection) <= tolerance)

    def coefficient_estimable(self) -> NDArray:
        """Return all unit-coordinate estimability decisions in one projection."""
        scaled_columns = self.column_scale > 0.0
        result = np.zeros(self.width, dtype=bool)
        null = self.null_basis()
        if null.shape[1] == 0:
            result[scaled_columns] = True
            return result

        equilibrated_null = null[scaled_columns, :] * self.column_scale[scaled_columns, None]
        null_norms = np.linalg.norm(equilibrated_null, axis=0)
        retained_null = null_norms > np.finfo(float).eps
        if not np.any(retained_null):
            result[scaled_columns] = True
            return result
        normalized_null = equilibrated_null[:, retained_null] / null_norms[retained_null]
        result[scaled_columns] = (
            np.linalg.norm(normalized_null, axis=1) <= SHARED_RANK_POLICY.factor_rcond
        )
        return result


@dataclass(frozen=True)
class RankInfo:
    """Compact fitted-subspace metadata in solver coefficient coordinates."""

    policy_version: int
    coordinate_space: Literal["solver"]
    selected_columns: NDArray
    selected_group_names: Sequence[str]
    sum_w: float
    mean_x: NDArray
    intercept_edf: float
    data: RankDecomposition
    augmented: RankDecomposition
    coefficient: RankDecomposition
    feature_edf: NDArray
    group_edf: Mapping[str, float]
    objective_loss: float | None

    @property
    def total_edf(self) -> float:
        return self.intercept_edf + float(np.sum(self.feature_edf))

    def solve(self, rhs: NDArray) -> NDArray:
        rhs = np.asarray(rhs, dtype=float)
        if rhs.shape != self.mean_x.shape:
            raise ValueError("rhs width does not match fitted coefficient space")
        result = np.zeros_like(rhs)
        result[self.selected_columns] = self.augmented.solve(rhs[self.selected_columns])
        return result

    def pseudo_inverse(self) -> NDArray:
        width = len(self.mean_x)
        result = np.zeros((width, width))
        result[np.ix_(self.selected_columns, self.selected_columns)] = (
            self.augmented.pseudo_inverse()
        )
        return result

    def is_estimable(self, contrast: NDArray) -> bool:
        contrast = np.asarray(contrast, dtype=float)
        if contrast.shape != self.mean_x.shape:
            raise ValueError("contrast width does not match fitted coefficient space")
        unselected = np.ones(len(contrast), dtype=bool)
        unselected[self.selected_columns] = False
        tolerance = SHARED_RANK_POLICY.factor_rcond * max(1.0, float(np.linalg.norm(contrast)))
        if np.linalg.norm(contrast[unselected]) > tolerance:
            return False
        return self.data.is_estimable(contrast[self.selected_columns])

    def coefficient_estimable(self) -> NDArray:
        result = np.zeros(len(self.mean_x), dtype=bool)
        result[self.selected_columns] = self.data.coefficient_estimable()
        return result


def _equilibrate_gram(
    matrix: NDArray, *, allow_indefinite: bool = False
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    symmetric = _symmetric_part(values)
    diagonal = np.diag(symmetric)
    scale_reference = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
    if not allow_indefinite and np.any(diagonal < -100.0 * _EPS * scale_reference):
        raise ValueError("matrix has a materially negative diagonal")
    if allow_indefinite:
        row_scale = np.max(np.abs(symmetric), axis=1, initial=0.0)
        diagonal_scale = np.maximum(np.abs(diagonal), _EPS * row_scale)
    else:
        diagonal_scale = np.maximum(diagonal, 0.0)
    active_columns = np.flatnonzero(diagonal_scale > 0.0)
    column_scale = np.zeros(len(diagonal))
    column_scale[active_columns] = np.sqrt(diagonal_scale[active_columns])
    if active_columns.size:
        active_scale = column_scale[active_columns]
        equilibrated = symmetric[np.ix_(active_columns, active_columns)] / np.outer(
            active_scale, active_scale
        )
        equilibrated = 0.5 * (equilibrated + equilibrated.T)
    else:
        equilibrated = np.zeros((0, 0))
    return equilibrated, column_scale, active_columns, symmetric


def _null_basis(
    width: int,
    active_columns: NDArray,
    active_scale: NDArray,
    discarded_vectors: NDArray,
) -> NDArray:
    """Stack the parameter-space null basis: discarded spectral, then structural.

    The **row layout** here is a load-bearing cross-module invariant, not just
    an implementation detail of this function:

    * the discarded-spectral columns are supported only on ``active_columns``,
    * the structural columns are exact unit vectors on the inactive columns,
    * so the two blocks have **disjoint row supports** and split ``null(H)``
      orthogonally.

    ``constrained_qp._null_space_mass`` depends on exactly that.  Changing the
    row supports, or mixing the two kinds of column into shared rows, silently
    breaks that consumer's split.
    """
    pieces: list[NDArray] = []
    if discarded_vectors.shape[1]:
        discarded = np.zeros((width, discarded_vectors.shape[1]))
        discarded[active_columns, :] = discarded_vectors / active_scale[:, None]
        pieces.append(discarded)
    inactive = np.setdiff1d(np.arange(width), active_columns, assume_unique=True)
    if inactive.size:
        inactive_basis = np.zeros((width, inactive.size))
        inactive_basis[inactive, np.arange(inactive.size)] = 1.0
        pieces.append(inactive_basis)
    return np.column_stack(pieces) if pieces else np.zeros((width, 0))


def _earliest_representatives(null_vectors: NDArray, rank: int) -> NDArray | None:
    """Earliest independent representatives, read off a null basis already in hand.

    Index-order greedy selection keeps column ``j`` unless it is a combination
    of the columns before it -- equivalently, unless some null vector has its
    LAST nonzero at ``j``.  Eliminating the null basis from the right-hand end
    pivots on precisely those positions, so the pivots are the rejected columns
    and the complement is the greedy selection, not an approximation of it.

    Deciding the same question by testing each prefix's spectrum costs one
    eigendecomposition per candidate, which is ``O(m**4)`` across the sweep and
    is why a rank-deficient block used to cost hundreds of times a full-rank one
    of the same width.  This is ``O(k**2 m)`` in the NULLITY ``k``, so a block
    that is a few columns short of full rank is nearly free.

    The pivot threshold is relative to each ROW's own norm, not to the matrix
    maximum.  These vectors come out of an eigendecomposition of a singular
    system, so a component that is mathematically zero arrives as noise many
    orders above machine epsilon -- measured at 1e-12 against a 1e-14 absolute
    threshold on an 8-column block, which pivoted on dust and returned a
    rank-deficient selection.  ``sqrt(eps)`` is the floor below which a
    component of a unit-norm null vector carries no information here.

    Returns ``None`` when elimination cannot place all ``k`` pivots.  The caller
    then keeps whatever exact path it already had rather than proceeding on a
    selection this could not certify.
    """
    width, nullity = null_vectors.shape
    if nullity == 0:
        return np.arange(width, dtype=int)
    working = np.array(null_vectors.T, dtype=float)
    if not np.all(np.isfinite(working)):
        return None
    relative = float(np.sqrt(np.finfo(float).eps))

    rejected: list[int] = []
    live = np.ones(nullity, dtype=bool)
    # Row norms only move when a pivot eliminates into them, which happens k
    # times, not once per column -- recomputing per column would put an m**2
    # term back into a routine whose whole point is not having one.
    floors = relative * np.linalg.norm(working, axis=1)
    for column in range(width - 1, -1, -1):
        candidates = np.flatnonzero(live)
        if candidates.size == 0:
            break
        entries = np.abs(working[candidates, column])
        significant = np.flatnonzero(entries > floors[candidates])
        if significant.size == 0:
            continue
        best = int(significant[np.argmax(entries[significant])])
        pivot = int(candidates[best])
        rejected.append(column)
        working[pivot] /= working[pivot, column]
        others = candidates[candidates != pivot]
        if others.size:
            working[others] -= np.outer(working[others, column], working[pivot])
            floors[others] = relative * np.linalg.norm(working[others], axis=1)
        live[pivot] = False

    if len(rejected) != nullity:
        return None
    keep = np.setdiff1d(np.arange(width), np.asarray(rejected, dtype=int))
    return keep if keep.size == rank else None


# A fallback pivot may be this fraction of the largest component available, so
# a single elimination step costs at most a factor ``1 / _PIVOT_THRESHOLD`` in
# ``_selection_amplification``.  See ``_threshold_pivot_representatives`` for why
# it is not 1.
_PIVOT_THRESHOLD = 0.5


def _threshold_pivot_representatives(null_vectors: NDArray, rank: int) -> NDArray | None:
    """Representatives chosen for conditioning rather than for index order.

    Same elimination as ``_earliest_representatives``, with the column order
    freed.  Each step rejects the LATEST column whose largest live component is
    within ``_PIVOT_THRESHOLD`` of the largest component available anywhere,
    instead of the latest column carrying any component at all.  At threshold 1
    this is Gaussian elimination with complete pivoting on the null basis --
    the standard route to the largest-volume ``k x k`` submatrix, and so to the
    smallest ``_selection_amplification``.

    It is not 1, because index order is still worth keeping wherever
    conditioning does not pay for it.  Complete pivoting reorders the WHOLE
    block, including exact aliases elsewhere in it that were never the problem:
    on a design carrying both a 5e-8 near alias at columns (0, 1) and an exact
    duplicate at columns (2, 4), complete pivoting selects ``[1, 3, 4]`` --
    surrendering the earlier member of the exact pair -- where every threshold
    from 0.1 to 0.75 selects ``[0, 2, 3]``, which keeps the convention.  Both
    reach amplification 1.414214 and condition 1.1309, against 2.5634e+07 and
    3.6532e+07 for the earliest selection, so the conditioning is not what is
    being traded.

    Relaxing further does start to cost: over 300 random alias-carrying blocks,
    173 of which failed the earliest rule's certificate, thresholds 0.25, 0.5,
    0.75 and 1.0 each landed under ``_achievable_amplification`` on all 173,
    while 0.1 left one block at 4.04 times it.  0.5 sits in the middle of the
    range that costs nothing measurable and bounds one step's multiplier by 2.

    Cost is the same ``O(k**2 m)`` in the nullity ``k`` as the earliest rule:
    revisiting every remaining column at each of the ``k`` steps is the same
    ``O(k m)`` per step that the elimination itself already pays.
    """
    width, nullity = null_vectors.shape
    if nullity == 0:
        return np.arange(width, dtype=int)
    working = np.array(null_vectors.T, dtype=float)
    if not np.all(np.isfinite(working)):
        return None
    relative = float(np.sqrt(np.finfo(float).eps))

    rejected: list[int] = []
    live = np.ones(nullity, dtype=bool)
    remaining = np.ones(width, dtype=bool)
    floors = relative * np.linalg.norm(working, axis=1)
    for _step in range(nullity):
        candidates = np.flatnonzero(live)
        columns = np.flatnonzero(remaining)
        if candidates.size == 0 or columns.size == 0:
            break
        block = np.abs(working[np.ix_(candidates, columns)])
        # Same per-row noise floor as the earliest rule: a component of a unit
        # null vector below sqrt(eps) of its own row norm is elimination dust,
        # and pivoting on it is exactly the failure this module already guards.
        block[block <= floors[candidates][:, None]] = 0.0
        column_peaks = block.max(axis=0)
        peak = float(column_peaks.max(initial=0.0))
        if peak <= 0.0:
            break
        # Latest qualifying column, so an exact alias -- whose null direction
        # carries equal weight on every column it ties together -- still gives
        # up its last column rather than its first.
        local = int(np.flatnonzero(column_peaks >= _PIVOT_THRESHOLD * peak)[-1])
        chosen = int(columns[local])
        pivot = int(candidates[int(np.argmax(block[:, local]))])
        rejected.append(chosen)
        remaining[chosen] = False
        working[pivot] /= working[pivot, chosen]
        others = candidates[candidates != pivot]
        if others.size:
            working[others] -= np.outer(working[others, chosen], working[pivot])
            floors[others] = relative * np.linalg.norm(working[others], axis=1)
        live[pivot] = False

    if len(rejected) != nullity:
        return None
    keep = np.setdiff1d(np.arange(width), np.asarray(rejected, dtype=int))
    return keep if keep.size == rank else None


def _selection_amplification(null_vectors: NDArray, keep: NDArray) -> float:
    """How much worse than the retained subspace itself a selection is.

    Split the rows of the null basis ``N`` -- orthonormal columns, so
    ``N.T @ N = I`` -- into the REJECTED rows ``N_R`` and the KEPT rows ``N_K``.
    Then ``N_R.T N_R = I - N_K.T N_K`` gives

        (N_R.T N_R)^-1 - I = (N_K N_R^-1).T (N_K N_R^-1),

    so ``1 / sigma_min(N_R)**2 == 1 + ||N_K N_R^-1||_2**2`` identically, and the
    selected block inherits

        sigma_min(X[:, keep]) >= sigma_rank(X) * sigma_min(N_R).

    The returned ``1 / sigma_min(N_R)`` is therefore the factor by which the
    CHOICE OF REPRESENTATIVES multiplies the condition number the retained
    subspace already has.  It is a property of the selection alone, which is
    why it sees what no test against the rank cutoff can: a block may sit
    comfortably above the cutoff that decided the rank -- not deficient by that
    standard, and accepted by Cholesky -- while still being the worst basis
    available for the subspace it spans.

    Verified over 400 random rank-deficient blocks: the identity holds to
    2.37e-13 relative, and the tightest observed
    ``sigma_min(X_keep) / (sigma_rank(X) * sigma_min(N_R))`` was 1.000002, so
    the bound is attained rather than merely true.
    """
    width = null_vectors.shape[0]
    rejected = np.setdiff1d(np.arange(width), keep)
    if rejected.size == 0:
        return 1.0
    spectrum = np.linalg.svd(null_vectors[rejected, :], compute_uv=False)
    smallest = float(np.min(spectrum)) if spectrum.size else 0.0
    return float("inf") if smallest <= 0.0 else 1.0 / smallest


def _achievable_amplification(width: int, nullity: int) -> float:
    """``sqrt(1 + k*(m-k))``, the amplification a rank-revealing choice reaches.

    The largest-volume ``k x k`` submatrix of a null basis with orthonormal
    columns has ``|N_K N_R^-1|`` bounded entrywise by 1 -- otherwise a swap
    would increase the volume -- so its spectral norm is at most
    ``sqrt(k*(m-k))`` and, by the identity in ``_selection_amplification``, its
    amplification is at most ``sqrt(1 + k*(m-k))``.  A selection worse than
    that is worse than one that provably exists, which makes this the natural
    place to stop trusting index order, rather than a tuned constant.
    """
    return float(np.sqrt(1.0 + float(nullity) * float(width - nullity)))


def _conditioned_representatives(null_vectors: NDArray, rank: int) -> NDArray | None:
    """Earliest representatives, unless index order costs more than it may.

    The earliest rule is a labelling convention -- it decides which of a set of
    aliased columns carries the reproducible zero -- and it is chosen blind to
    conditioning.  Where the earliest independent columns happen to be two
    near-duplicates, that convention is paid for in every downstream solve: the
    block is positive definite, Cholesky accepts it, its smallest eigenvalue is
    above the cutoff that decided the rank, and it is still the worst basis for
    its own span.

    So the convention is kept, and certified.  ``_selection_amplification`` is
    exact and costs ``O(k**3)`` in the NULLITY, below the ``O(m**3)``
    eigendecomposition this path has already paid.  Only when it exceeds what a
    rank-revealing selection provably achieves is a second, conditioning-driven
    selection computed, and even then the better-conditioned of the two is
    returned -- so this can only improve the block it hands on.
    """
    earliest = _earliest_representatives(null_vectors, rank)
    if earliest is None:
        return None
    amplification = _selection_amplification(null_vectors, earliest)
    if amplification <= _achievable_amplification(*null_vectors.shape):
        return earliest
    alternative = _threshold_pivot_representatives(null_vectors, rank)
    if alternative is None:
        return earliest
    if _selection_amplification(null_vectors, alternative) < amplification:
        return alternative
    return earliest


def _scaled_subspace_logdet(coordinates: NDArray) -> float:
    """Return ``log(det(coordinates.T @ coordinates))`` across extreme row scales."""
    width = coordinates.shape[1]
    if width == 0:
        return 0.0

    # Ordinary QR/SVD only provides absolute accuracy.  DGEJSV's 'F' mode
    # applies full row and column pivoting so diagonal scaling cannot erase a
    # genuine retained direction.  Ask for the unrestricted singular-value
    # range because rank has already been decided in equilibrated coordinates.
    singular_values, _, _, scaling, _, info = scipy.linalg.lapack.dgejsv(
        np.asfortranarray(coordinates),
        joba=2,  # 'F': full-pivoting, high-relative-accuracy preprocessing
        jobu=3,  # 'N': singular values only
        jobv=3,  # 'N': singular values only
        jobr=0,  # 'N': do not truncate the requested singular-value range
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"high-accuracy retained SVD failed with info={info}")
    if np.any(singular_values <= 0.0) or np.any(scaling[:2] <= 0.0):
        raise ValueError("retained coordinate basis is not full rank")
    log_scale = float(np.log(scaling[0]) - np.log(scaling[1]))
    return 2.0 * (float(np.sum(np.log(singular_values))) + width * log_scale)


def _retained_log_pdet(
    active_scale: NDArray,
    retained_vectors: NDArray,
    discarded_vectors: NDArray,
    retained_values: NDArray,
) -> float:
    """Return the retained pseudo-logdet without forming a coordinate Gram."""
    if retained_values.size == 0:
        return 0.0

    # V (retained) and N (discarded) form an orthogonal basis.  Jacobi's
    # complementary-minor identity gives
    #
    # det(V.T D^2 V) = det(D)^2 det(N.T D^-2 N).
    #
    # Evaluate whichever side has fewer columns; this is both cheaper and more
    # accurate for the common one-alias case.
    if retained_vectors.shape[1] <= discarded_vectors.shape[1]:
        coordinate_logdet = _scaled_subspace_logdet(active_scale[:, None] * retained_vectors)
    else:
        coordinate_logdet = 2.0 * float(np.sum(np.log(active_scale)))
        coordinate_logdet += _scaled_subspace_logdet(discarded_vectors / active_scale[:, None])
    return coordinate_logdet + float(np.sum(np.log(np.abs(retained_values))))


def _decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
    allow_indefinite: bool = False,
    omit_uncertifiable: bool = False,
) -> RankDecomposition | None:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix.

    ``omit_uncertifiable`` is a pure optimization hint, never a semantic one:
    when it is set this may return ``None`` instead of a decomposition that
    :func:`needs_factor_certification` would have rejected anyway.  It is
    permitted to return a decomposition in that case too, so the caller still
    owns the predicate -- see :func:`decompose_gram_if_authoritative`.
    """
    equilibrated, column_scale, active_columns, _ = _equilibrate_gram(
        matrix, allow_indefinite=allow_indefinite
    )
    width = len(column_scale)
    structural_aliases = column_scale == 0.0
    if active_columns.size == 0:
        return RankDecomposition(
            policy_version=policy.version,
            method="empty",
            column_scale=_freeze(column_scale),
            active_columns=_freeze(active_columns, dtype=int),
            rank=0,
            pre_truncation_condition=float("inf"),
            cutoff=0.0,
            rank_truncated=width > 0,
            used_svd_fallback=False,
            resolution_limited=False,
            log_pdet=0.0,
            parameter_null_basis=_freeze(np.eye(width)),
            structural_aliases=_freeze(structural_aliases, dtype=bool),
            retained_values=_freeze(np.array([])),
        )

    if not allow_indefinite:
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            matrix_norm = float(np.linalg.norm(equilibrated, ord=1))
            trtri = scipy.linalg.get_lapack_funcs("trtri", (factor,))
            inverse_factor, inverse_info = trtri(
                factor,
                lower=1,
                unitdiag=0,
                overwrite_c=0,
            )
            if inverse_info != 0:
                raise np.linalg.LinAlgError("triangular inverse failed during rank certification")
            inverse_factor_frobenius = float(np.linalg.norm(inverse_factor, ord="fro"))
            min_eigenvalue_lower_bound = 1.0 / inverse_factor_frobenius**2
            pocon = scipy.linalg.get_lapack_funcs("pocon", (factor,))
            reciprocal_condition, info = pocon(factor, matrix_norm, uplo="L")
            safely_full_rank = (
                np.isfinite(min_eigenvalue_lower_bound)
                and min_eigenvalue_lower_bound
                > policy.certification_band * policy.gram_rcond * matrix_norm
            )
            if safely_full_rank:
                probe = np.arange(1.0, len(active_columns) + 1.0)
                solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
                residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                    np.linalg.norm(probe), 1e-300
                )
                if residual <= residual_tol:
                    log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                        np.sum(np.log(column_scale[active_columns]))
                    )
                    null = _null_basis(
                        width,
                        active_columns,
                        column_scale[active_columns],
                        np.zeros((len(active_columns), 0)),
                    )
                    return RankDecomposition(
                        policy_version=policy.version,
                        method="cholesky",
                        column_scale=_freeze(column_scale),
                        active_columns=_freeze(active_columns, dtype=int),
                        rank=len(active_columns),
                        pre_truncation_condition=float(
                            np.sqrt(1.0 / reciprocal_condition)
                            if info == 0
                            and np.isfinite(reciprocal_condition)
                            and reciprocal_condition > 0.0
                            else np.sqrt(matrix_norm / min_eigenvalue_lower_bound)
                        ),
                        cutoff=policy.gram_rcond * matrix_norm,
                        rank_truncated=len(active_columns) < width,
                        used_svd_fallback=False,
                        resolution_limited=False,
                        log_pdet=log_pdet,
                        cholesky_factor=_freeze(factor),
                        parameter_null_basis=_freeze(null),
                        structural_aliases=_freeze(structural_aliases, dtype=bool),
                    )
        except (np.linalg.LinAlgError, ValueError):
            pass

    eigenvalues, eigenvectors = np.linalg.eigh(equilibrated)
    raw_eigenvalues = eigenvalues
    max_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues), initial=0.0))
    negative_tolerance = 100.0 * _EPS * max(max_abs_eigenvalue, 1.0)
    materially_indefinite = bool(eigenvalues[0] < -negative_tolerance)
    if not allow_indefinite and materially_indefinite:
        raise ValueError(
            "matrix is materially indefinite "
            f"(min equilibrated eigenvalue={eigenvalues[0]:.3e}, "
            f"scale={max_abs_eigenvalue:.3e})"
        )
    psd_semantics = not materially_indefinite
    if psd_semantics:
        eigenvalues = np.maximum(eigenvalues, 0.0)
        max_abs_eigenvalue = max_eigenvalue
    cutoff = policy.gram_rcond * max_abs_eigenvalue
    retained_mask = eigenvalues > cutoff if psd_semantics else np.abs(eigenvalues) > cutoff
    rank = int(np.count_nonzero(retained_mask))
    positive = np.abs(eigenvalues[np.abs(eigenvalues) > 0.0])
    condition = (
        float(np.sqrt(max_abs_eigenvalue / np.min(positive)))
        if positive.size and max_abs_eigenvalue > 0.0
        else float("inf")
    )

    if rank == len(active_columns) and np.all(eigenvalues > 0.0):
        try:
            factor = scipy.linalg.cholesky(equilibrated, lower=True, check_finite=False)
            probe = np.arange(1.0, len(active_columns) + 1.0)
            solved = scipy.linalg.cho_solve((factor, True), probe, check_finite=False)
            residual = np.linalg.norm(equilibrated @ solved - probe) / max(
                np.linalg.norm(probe), 1e-300
            )
            if residual <= residual_tol:
                log_pdet = 2.0 * float(np.sum(np.log(np.diag(factor)))) + 2.0 * float(
                    np.sum(np.log(column_scale[active_columns]))
                )
                null = _null_basis(
                    width,
                    active_columns,
                    column_scale[active_columns],
                    np.zeros((len(active_columns), 0)),
                )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(active_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=rank < width,
                    used_svd_fallback=False,
                    resolution_limited=False,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(factor),
                    parameter_null_basis=_freeze(null),
                    structural_aliases=_freeze(structural_aliases, dtype=bool),
                    retained_values=_freeze(eigenvalues),
                )
        except (np.linalg.LinAlgError, ValueError):
            pass

    # Normal equations cannot distinguish an exact active-column alias from a
    # full-rank factor direction whose squared singular value rounded to zero.
    # Structural zero columns were removed above; every other PSD truncation
    # therefore needs observation-factor certification when one is available.
    #
    # Hoisted above the subspace construction on purpose: it reads only the
    # spectrum, and it is the last field the certification predicate needs.
    resolution_limited = bool(
        (psd_semantics and rank < len(active_columns))
        or np.any((np.abs(raw_eigenvalues) > 0.0) & ~retained_mask)
        or (fallback_factor is not None and decompose_factor(fallback_factor).rank > rank)
    )
    # Everything past this point -- two width-by-rank bases, the null basis,
    # the retained pseudo-determinant, the representative selection and its
    # Cholesky -- exists only to be read off the returned decomposition.  Both
    # returns below are reached with the ``rank``, ``width``,
    # ``pre_truncation_condition`` and ``resolution_limited`` computed above,
    # and neither reports ``qr_svd``, so the predicate settles here exactly as
    # it would on the finished object.  When it says the caller must certify
    # against the observation factor, none of that work can be read back.
    if omit_uncertifiable and _certification_required(
        method="gram_eigh",
        width=width,
        rank=rank,
        pre_truncation_condition=condition,
        resolution_limited=resolution_limited,
        policy=policy,
    ):
        return None
    retained_vectors = eigenvectors[:, retained_mask]
    discarded_vectors = eigenvectors[:, ~retained_mask]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    active_scale = column_scale[active_columns]
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = eigenvalues[retained_mask]
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    if psd_semantics and 0 < rank < len(active_columns):
        # Choose the earliest original-coordinate representative whose
        # principal system has the certified rank. This gives exact aliases a
        # reproducible zero coefficient while estimability still uses the true
        # spectral null space above.  The selection is read off the null basis
        # this decomposition has already computed -- `_earliest_representatives`
        # documents why eliminating it from the right gives the same columns as
        # walking prefixes, without an eigendecomposition per candidate, and
        # `_conditioned_representatives` documents when index order is too
        # expensive a convention to keep.
        selected_local_array = _conditioned_representatives(discarded_vectors, rank)
        if selected_local_array is not None:
            representative_columns = active_columns[selected_local_array]
            representative = equilibrated[np.ix_(selected_local_array, selected_local_array)]
            try:
                representative_factor = scipy.linalg.cholesky(
                    representative, lower=True, check_finite=False
                )
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                return RankDecomposition(
                    policy_version=policy.version,
                    method="pivoted_cholesky",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=False,
                    resolution_limited=resolution_limited,
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
    return RankDecomposition(
        policy_version=policy.version,
        method="gram_eigh",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=False,
        resolution_limited=resolution_limited,
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(structural_aliases, dtype=bool),
        retained_values=_freeze(retained_values),
    )


def decompose_gram(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
    allow_indefinite: bool = False,
) -> RankDecomposition:
    """Equilibrate and decompose a symmetric positive-semidefinite matrix."""
    decomposition = _decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        fallback_factor=fallback_factor,
        allow_indefinite=allow_indefinite,
    )
    if decomposition is None:  # pragma: no cover - omit_uncertifiable defaults off
        raise RuntimeError("gram decomposition omitted its subspace without being asked to")
    return decomposition


def decompose_gram_if_authoritative(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
    fallback_factor: NDArray | None = None,
) -> RankDecomposition | None:
    """The Gram decomposition when it is authoritative, else ``None``.

    ``None`` means exactly what ``needs_factor_certification`` means on the
    eager result: this Gram cannot certify its own retained subspace, and the
    caller must go to the observation factor.  Callers that do nothing else in
    that case should prefer this to ``decompose_gram`` plus the predicate,
    because a Gram that is about to be superseded never builds the retained
    subspace, the null basis, the representative Cholesky or the retained
    pseudo-determinant that only the superseded object could have exposed.

    The predicate below is the authority, so the contract holds whatever the
    hint inside chooses to skip: the eager and deferring paths agree on every
    field that decides it, and a spared decomposition is one no caller in this
    shape could have read.
    """
    decomposition = _decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        fallback_factor=fallback_factor,
        omit_uncertifiable=True,
    )
    if decomposition is None or needs_factor_certification(decomposition, policy=policy):
        return None
    return decomposition


def decompose_symmetric(
    matrix: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    residual_tol: float = 1e-6,
) -> RankDecomposition:
    """Decompose symmetric full-Newton curvature that may be indefinite."""
    return decompose_gram(
        matrix,
        policy=policy,
        residual_tol=residual_tol,
        allow_indefinite=True,
    )


def decompose_factor(
    factor: NDArray,
    *,
    policy: RankPolicy = SHARED_RANK_POLICY,
    retain_factor_solve: bool = False,
) -> RankDecomposition:
    """Decompose a weighted/augmented factor using the factor-space rule."""
    factor = np.asarray(factor, dtype=float)
    if factor.ndim != 2 or not np.all(np.isfinite(factor)):
        raise ValueError("factor must be a finite matrix")
    width = factor.shape[1]
    column_scale = np.linalg.norm(factor, axis=0)
    active_columns = np.flatnonzero(column_scale > 0.0)
    if active_columns.size == 0:
        decomposition = decompose_gram(np.zeros((width, width)), policy=policy)
        if retain_factor_solve:
            decomposition = replace(
                decomposition,
                factor_rhs_left_basis=_freeze(np.zeros((factor.shape[0], 0))),
            )
        return decomposition
    active_scale = column_scale[active_columns]
    equilibrated = factor[:, active_columns] / active_scale
    # A tall observation factor needs only its thin left singular vectors;
    # requesting a full U would allocate O(n²) memory.  A wide factor still
    # needs full right vectors so exact row-rank null directions are retained.
    full_matrices = equilibrated.shape[0] < equilibrated.shape[1]
    left_vectors, singular_values, Vh = np.linalg.svd(
        equilibrated,
        full_matrices=full_matrices,
    )
    cutoff = policy.factor_rcond * singular_values[0]
    retained_mask = singular_values > cutoff
    rank = int(np.count_nonzero(retained_mask))
    retained_vectors = Vh[: len(singular_values), :].T[:, retained_mask]
    discarded_vectors = Vh.T[:, rank:]
    solution_basis = np.zeros((width, rank))
    estimable_basis = np.zeros((width, rank))
    solution_basis[active_columns, :] = retained_vectors / active_scale[:, None]
    estimable_basis[active_columns, :] = retained_vectors * active_scale[:, None]
    null = _null_basis(width, active_columns, active_scale, discarded_vectors)
    retained_values = singular_values[retained_mask] ** 2
    factor_rhs_left_basis = None
    if retain_factor_solve:
        retained_left = left_vectors[:, : len(singular_values)][:, retained_mask]
        factor_rhs_left_basis = retained_left / singular_values[retained_mask]
    log_pdet = (
        2.0 * float(np.sum(np.log(active_scale))) + float(np.sum(np.log(np.abs(retained_values))))
        if rank == width
        else _retained_log_pdet(
            active_scale,
            retained_vectors,
            discarded_vectors,
            retained_values,
        )
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    if 0 < rank < len(active_columns):
        # Same certified representative choice as the Gram path, off the right
        # singular vectors that span this factor's null space.  The Gram is
        # formed only for the selected block, not to search for it.
        selected_local_array = _conditioned_representatives(discarded_vectors, rank)
        if selected_local_array is not None:
            equilibrated_gram = equilibrated.T @ equilibrated
            representative_columns = active_columns[selected_local_array]
            representative = equilibrated_gram[np.ix_(selected_local_array, selected_local_array)]
            try:
                representative_factor = scipy.linalg.cholesky(
                    representative, lower=True, check_finite=False
                )
                representative_basis = np.zeros((width, rank))
                representative_basis[representative_columns, np.arange(rank)] = (
                    1.0 / column_scale[representative_columns]
                )
                representative_aliases = np.ones(width, dtype=bool)
                representative_aliases[representative_columns] = False
                representative_rhs_left_basis = None
                representative_rhs_triangular = None
                if retain_factor_solve:
                    selected_factor = factor[:, representative_columns]
                    representative_rhs_left_basis, representative_rhs_triangular = np.linalg.qr(
                        selected_factor,
                        mode="reduced",
                    )
                return RankDecomposition(
                    policy_version=policy.version,
                    method="qr_svd",
                    column_scale=_freeze(column_scale),
                    active_columns=_freeze(representative_columns, dtype=int),
                    rank=rank,
                    pre_truncation_condition=condition,
                    cutoff=cutoff,
                    rank_truncated=True,
                    used_svd_fallback=True,
                    resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
                    log_pdet=log_pdet,
                    cholesky_factor=_freeze(representative_factor),
                    pivots=_freeze(representative_columns, dtype=int),
                    solution_basis=_freeze(representative_basis),
                    parameter_null_basis=_freeze(null),
                    estimable_functional_basis=_freeze(estimable_basis),
                    structural_aliases=_freeze(representative_aliases, dtype=bool),
                    retained_values=_freeze(retained_values),
                    factor_rhs_left_basis=(
                        None
                        if representative_rhs_left_basis is None
                        else _freeze(representative_rhs_left_basis)
                    ),
                    factor_rhs_triangular=(
                        None
                        if representative_rhs_triangular is None
                        else _freeze(representative_rhs_triangular)
                    ),
                )
            except (np.linalg.LinAlgError, ValueError):
                pass
    return RankDecomposition(
        policy_version=policy.version,
        method="qr_svd",
        column_scale=_freeze(column_scale),
        active_columns=_freeze(active_columns, dtype=int),
        rank=rank,
        pre_truncation_condition=condition,
        cutoff=cutoff,
        rank_truncated=rank < width,
        used_svd_fallback=True,
        resolution_limited=bool(np.any((singular_values > 0.0) & ~retained_mask)),
        log_pdet=log_pdet,
        solution_basis=_freeze(solution_basis),
        parameter_null_basis=_freeze(null),
        estimable_functional_basis=_freeze(estimable_basis),
        structural_aliases=_freeze(column_scale == 0.0, dtype=bool),
        retained_values=_freeze(retained_values),
        factor_rhs_left_basis=(
            None if factor_rhs_left_basis is None else _freeze(factor_rhs_left_basis)
        ),
    )


def selected_group_name_set(result, groups: Sequence, *, penalty=None) -> set[str]:
    """Return explicit solver selection, with a legacy coefficient fallback.

    Legacy results predate explicit rank/selection metadata.  When the fitted
    penalty is available, preserve every group that was not subject to a
    positive nonsmooth penalty; a valid zero estimate is not deselection.
    """
    if getattr(result, "rank_info", None) is not None:
        return set(result.rank_info.selected_group_names)
    if penalty is not None:
        from superglm.penalties.base import penalty_can_zero_groups, penalty_targets_group

        can_zero_groups = penalty_can_zero_groups(penalty)
        return {
            group.name
            for group in groups
            if not can_zero_groups
            or not penalty_targets_group(penalty, group)
            or np.linalg.norm(result.beta[group.sl]) > 1e-12
        }
    return {group.name for group in groups if np.linalg.norm(result.beta[group.sl]) > 1e-12}
