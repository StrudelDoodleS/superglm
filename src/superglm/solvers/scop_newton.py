"""Joint Newton step for shape-constrained (SCOP) terms.

Pya & Wood (2015, Appendix 1) fit a shape-constrained additive model by one
full Newton step on all of its coefficients at once: the ordinary
coefficients ``b`` and each SCOP term's latent coordinates ``theta``, whose
mapped coefficients ``gamma(theta)`` enter the predictor linearly. With the
intercept profiled out by centring, the penalized Hessian in ``(b, theta)`` is

    [[A,        C J               ],
     [J C',     J G J + lambda S + E]]

where ``A`` is the ordinary block's penalized Hessian, ``C = X' W X_s`` and
``G = X_s' W X_s`` are the centred cross and SCOP moments in mapped
coordinates, each a sum of products of centred columns, ``J =
diag(gamma'(theta))``, and ``E = -diag(gamma''(theta) * X_s' W r)`` is the
curvature of the map at the working residual ``r``.

The step is taken by block elimination. The caller has already solved the
ordinary block with ``theta`` held fixed; ``SCOPJointNewton.step`` reduces
the SCOP block to its Schur complement ``J (G - C' A^-1 C) J + lambda S + E``,
takes the Newton step there and corrects the ordinary solve by
back-substitution. Pya & Wood (2015, §3.2) report that Fisher scoring caused
convergence problems for these models, so ``E`` is kept unless the reduced
block is then not positive definite, when that iteration takes the
Gauss-Newton block instead. The Schur block's data part comes from its Grams
when their rounding resolves every direction, and otherwise from a factor of
its rows, where a direction the rows cannot tell from zero gets no curvature
(``_resolved_root``). Directions the block cannot resolve are truncated at
``sqrt(eps)`` of its largest singular value (§3.2).
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from superglm._group_matrix._group_matrix_algebra import (
    _MAX_CROSS_EXPANSION_BYTES,
    _MAX_DISC_DISC_HIST_CELLS,
    _support_support_raw_cross,
)
from superglm._group_matrix._group_matrix_centered import (
    _MAX_PACKED_HIST_CELLS,
    _compact_support,
    _compact_support_rows,
)
from superglm._roundoff import SQRT_EPS as _SQRT_EPS
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    CrossMatrixExecutionPlan,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    MatrixExecutionPlan,
    _disc_disc_2d_hist,
)
from superglm.solvers.centered_system import _dense_split, penalty_factor
from superglm.solvers.mode_score import centred_matvec, dense_centred_rmatvec, dense_columns
from superglm.solvers.rank import (
    _GREY_FLOOR,
    _eigensolver_relative_bar,
    _gamma_up,
    streamed_weighted_factor,
    tsqr_leaf_rows,
)
from superglm.solvers.scop import SCOPSolverReparam
from superglm.solvers.scop_exact_support import build_exact_scop_support
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.solvers.centered_system import CenteredSystem

_NEWTON_CURVATURE_MARGIN = 1e-10
"""Guard band on the ``I + curvature`` positive-definiteness test."""


@njit(cache=True)
def _scop_bin_moments(bin_idx, W, z, eta, n_bins):
    """``bincount(W)`` and ``bincount(W * (z - eta))`` in one pass over the rows.

    For one discretized SCOP group: the working residual is never formed at
    observation level.
    """
    W_agg = np.zeros(n_bins)
    Wr_agg = np.zeros(n_bins)
    for r in range(W.shape[0]):
        b = bin_idx[r]
        w = W[r]
        W_agg[b] += w
        Wr_agg[b] += w * (z[r] - eta[r])
    return W_agg, Wr_agg


@njit(cache=True)
def _joint_weight_tables(codes, offsets, scop_bins, n_scop, W, total):
    """Every group's weight table against the SCOP bins, in one pass over the rows.

    ``table[offsets[g] + codes[r, g] * n_scop + scop_bins[r]] += W[r]``: the
    sums ``_disc_disc_2d_hist`` forms for one pair of groups, here for every
    group at once, each table summing its rows in ascending order.
    """
    table = np.zeros(total)
    for r in range(W.shape[0]):
        w = W[r]
        s = scop_bins[r]
        for g in range(codes.shape[1]):
            table[offsets[g] + codes[r, g] * n_scop + s] += w
    return table


def _warmup_scop() -> None:
    """Compile the SCOP kernels for the operands a shape-constrained fit passes."""
    empty = np.zeros(0, dtype=np.float64)
    _scop_bin_moments(np.zeros(0, dtype=np.int64), empty, empty, empty, 0)
    _joint_weight_tables(
        np.zeros((0, 1), dtype=np.int32),
        np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        1,
        empty,
        1,
    )


@dataclass(frozen=True)
class SCOPGroupSpec:
    """Static definition of one SCOP group, separate from trial state."""

    group_index: int
    group: GroupSlice
    reparam: SCOPSolverReparam
    B_scop: NDArray
    S_scop: NDArray
    bin_idx: NDArray | None


@dataclass
class SCOPNewtonResult:
    """One SCOP group's share of a joint Newton step."""

    beta_new: NDArray
    """Updated latent coordinates (q_eff,)."""

    step_norm: float
    """L2 norm of the latent update."""

    used_fisher_fallback: bool
    """True if the step took the Gauss-Newton block because the full one was not PD."""

    H_penalized: NDArray | None = None
    """(q_eff, q_eff) block of the SCOP Hessian the step used, on its resolved range."""

    discarded_directions: NDArray | None = None
    """This group's columns of the directions the step could not resolve.

    Rows are the below-cutoff right singular vectors of the step's factor,
    restricted to this group's parameters. Every group's result carries the
    same number of rows in the same order, so the full joint direction is
    recovered by concatenating groups row-wise.
    """


@dataclass(frozen=True)
class SCOPJointStep:
    """A joint Newton step: each SCOP group's result and the ordinary block's correction."""

    results: dict[int, SCOPNewtonResult]
    ordinary_correction: NDArray
    """Added to the ordinary solve taken with the latent coordinates held fixed."""

    intercept_shift: float
    """Subtracted from that solve's profiled intercept: ``x_s' J delta_theta``."""


def _append_scop_trace(
    debug_recorder,
    *,
    result: SCOPNewtonResult,
    debug_context: dict[str, object] | None,
    group_name: str,
) -> None:
    """Append a private SCOP step summary when debug tracing is enabled."""
    if debug_recorder is None or getattr(debug_recorder, "enabled_level", 0) < 2:
        return

    payload = dict(debug_context or {})
    payload["group_name"] = group_name
    payload.update(
        {
            "mode": "joint",
            "step_norm": float(result.step_norm),
            "used_fisher_fallback": bool(result.used_fisher_fallback),
            "discarded_directions": int(
                0 if result.discarded_directions is None else result.discarded_directions.shape[0]
            ),
        }
    )
    debug_recorder.append_jsonl("scop", payload)


def _positive_sum_parts(values: NDArray, exponents: NDArray) -> tuple[float, int]:
    """Sum nonnegative contributions in their largest actual exponent unit."""
    nonzero = values != 0
    if not np.any(nonzero):
        return 0.0, 0
    unit = int(np.max(exponents[nonzero]))
    total = math.fsum(np.ldexp(values, exponents - unit).ravel().tolist())
    mantissa, exponent = math.frexp(total)
    return mantissa, exponent + unit


def _positive_quadratic_roundoff(
    vector: NDArray, matrix: NDArray, *weights: float, right: NDArray | None = None
) -> float:
    """Carry range, but no additional significand bits, through an action.

    Ordinary inputs use two native products. Extreme inputs align the actual
    triple-product exponents, since separately normalizing the coordinates and
    matrix could erase complementary large/small terms. fsum avoids a p**2
    reduction allowance; the caller's existing inflation covers the products.
    """
    left, matrix = np.abs(vector), np.abs(matrix)
    right = left if right is None else np.abs(right)
    if any(weight == 0 for weight in weights):
        return 0.0
    ordinary = all(
        np.all((value == 0) | ((value >= 2.0**-128) & (value <= 2.0**128)))
        for value in (left, matrix, right)
    )
    if ordinary:
        mantissa, exponent = math.frexp(float(left @ matrix @ right))
    else:
        lm, le = np.frexp(left)
        mm, me = np.frexp(matrix)
        rm, re = np.frexp(right)
        terms = lm[:, None] * mm * rm[None, :]
        powers = le[:, None] + me + re[None, :]
        mantissa, exponent = _positive_sum_parts(terms, powers)
    for weight in weights:
        part, power = np.frexp(weight)
        mantissa *= part
        exponent += int(power)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return float(np.ldexp(mantissa, exponent))


def _restrict_to_resolved_range(H: NDArray, discarded: NDArray) -> NDArray:
    """Project ``H`` onto the range the step could actually resolve.

    ``H_penalized`` is consumed as *the* curvature of the accepted step -- the
    REML joint Hessian substitutes it for the SCOP diagonal block, and the
    determinant and geometry are taken from that. Reporting the unrestricted
    full Newton Hessian would contradict the step: at the log-space boundary
    the discarded direction carries essentially the whole penalized score (the
    coefficient is running to minus infinity for a vanishing gain, which is
    precisely why the fit cannot terminate without truncation), and that score
    enters the Hessian through ``diag(grad_data)`` as a *negative* diagonal.
    Consumers then see a materially indefinite matrix and refuse it.

    Restricting removes the frozen direction together with its curvature, and
    what is reported is positive semidefinite by construction rather than by
    hope. For the block's own curvature (``block_curvature``) ``P' H P == I +
    curvature`` in the retained basis, so the positive-definiteness test
    already applied to ``I + curvature`` is exactly the statement that the
    restriction is positive definite. A joint step's retained basis comes
    from its Schur block ``J (G~ - C' A^-1 C) J + lambda S (+ E)``, which the
    own block ``J G J + lambda S (+ E)`` dominates (``G - G~ + C' A^-1 C`` is
    positive semidefinite), so the restriction of the own block is at least
    as positive as the one the test certified. A curvature that discarded
    nothing is returned untouched.
    """
    if not discarded.size:
        return H
    projector = np.eye(H.shape[0]) - discarded.T @ discarded
    restricted = projector @ H @ projector
    return 0.5 * (restricted + restricted.T)


def _psd_root(S: NDArray) -> NDArray:
    """Return ``R`` with ``R.T @ R == S``, for symmetric PSD ``S``."""
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (S + S.T))
    return np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T


def _resolved_root(
    gram: NDArray,
    score: NDArray,
    scale: NDArray,
    count: int,
    row_factor: Callable[[], tuple[NDArray, NDArray, int]],
) -> tuple[NDArray, NDArray]:
    """``(R, s)``: a root of the Schur block's data part on the directions it resolves, and ``score`` there.

    ``gram`` is that data part ``G~ - C' A^-1 C``, formed from Grams of
    centred columns. To first order its rounding is the joint Gram's along
    ``(v, -A^-1 C v)``, so ``|dG_ij| <= gamma_count g_i g_j`` with ``g =
    scale``: each SCOP column's weighted norm plus the norms of the ordinary
    columns its regression ``A^-1 C`` draws on (Higham 2002, sec. 3.5, for
    every formed product, ``count`` terms deep). Scaled by ``g`` that error
    is at most ``gamma_count q`` in the 2-norm, and the eigensolver adds ``q
    eps`` of the largest eigenvalue (*LAPACK Users' Guide*, sec. 4.7). When
    every eigenvalue clears ``_GREY_FLOOR`` times their sum, as on the
    freMTPL2 fits, the Gram is the root.

    Otherwise the Gram cannot decide, and not always because of the data: an
    exact alias (an ordinary column equal to a SCOP slope) puts a true zero
    inside the bound, but nearly collinear ordinary columns in a full-rank
    design make ``g`` large enough to put genuine curvature inside it as
    well. The decision is then taken on a factor of the
    rows, ``row_factor()``, which returns ``(F, g, count)`` with ``F' F``
    the same block, each column of ``F`` within ``gamma_count g_j`` of its
    exact value: a factor squares no condition number, so its error in
    ``g``-scaled singular values is ``gamma_count sqrt(q)`` (Weyl), not the
    square root of the Gram's. A direction at or below that is one the rows
    cannot tell from zero: it gets no data curvature, and ``score``, whose
    component there is rounding (``|s_v| <= sigma_v ||r||_W``), loses it,
    so the penalty alone decides the step along it. Every other direction
    keeps its curvature and its score; whether the step can resolve it is
    the factor cut's decision (``_truncated_factor_step``).
    """
    q = gram.shape[0]
    norms = np.where(scale > 0.0, scale, 1.0)
    scaled = gram / norms[:, None] / norms[None, :]
    values, vectors = np.linalg.eigh(0.5 * (scaled + scaled.T))
    largest = float(np.max(np.abs(values), initial=0.0))
    bar = _gamma_up(count) * q + _eigensolver_relative_bar(q) * largest
    if np.all(values > _GREY_FLOOR * bar):
        singular, resolved = np.sqrt(values), np.full(q, True)
    else:
        factor, scale, count = row_factor()
        norms = np.where(scale > 0.0, scale, 1.0)
        _, singular, right = np.linalg.svd(factor / norms[None, :])
        singular = np.concatenate((singular, np.zeros(q - singular.size)))
        vectors = right.T
        resolved = singular > _gamma_up(count) * math.sqrt(q)
    kept = vectors[:, resolved]
    root = singular[resolved][:, None] * kept.T * norms[None, :]
    return root, norms * (kept @ (kept.T @ (score / norms)))


def _truncated_factor_step(
    factor: NDArray,
    second_order: NDArray,
    grad: NDArray,
) -> tuple[NDArray | None, bool, NDArray]:
    """Newton step computed from the augmented factor, rank handled first.

    Returns ``(step, used_fisher, discarded)``; ``step`` is None only for a
    degenerate factor with no usable direction at all. ``discarded`` holds the
    right singular vectors below the cutoff as rows, and is the record of which
    directions this step deliberately did not move in -- downstream mode
    certification has to measure stationarity on the retained subspace, or it
    demands a correction along a direction the data cannot resolve.

    The published algorithm works from a factor rather than the Gram.
    ``factor`` is ``[R J ; sqrt(lambda) rS]``, with ``R`` a root of the SCOP
    block's data Gram in mapped coordinates, so its cross-product is the
    block's Gauss-Newton Hessian; the order of operations is the point:

    1. ``R`` from a QR of the factor, then an SVD of ``R``, discarding
       directions below ``sigma_max * sqrt(eps)`` -- Pya & Wood (2015) §3.2
       substitute a singular value decomposition for the R factor when it is
       rank deficient, judging too-small "relative to the largest singular
       value multiplied by some power (in the range .5 to 1) of the machine
       precision". We take the ``.5`` end of that range.
    2. Only then is the second-order Newton term introduced, expressed in the
       basis where the Gauss-Newton Hessian is the identity.

    Truncating the Gram ``J R' R J`` instead cannot work, and not merely for
    want of a better threshold: it squares the condition number that ``J``
    contributes, so the unidentifiable directions this is meant to discard
    have already been ground into the same rounding noise as small-but-real
    ones. On a measured boundary fit the Gram's spectrum leaves no threshold
    that separates them. ``J`` is applied to the factor's columns here; only
    ``R``, whose conditioning is the data's in mapped coordinates and does
    not involve ``J``, comes from a Gram.

    Handling rank first is also what keeps the second-order term safe. That
    term is what makes the full Newton Hessian indefinite, and applied to an
    unidentifiable direction it amplifies rather than corrects it -- the
    mechanism behind a SCOP coefficient drifting toward its log-space boundary
    while the deviance stands still.
    """
    width = factor.shape[1]
    empty_discarded = np.zeros((0, width), dtype=float)

    upper_triangular = np.linalg.qr(factor, mode="r")
    _, singular_values, right_vectors = np.linalg.svd(upper_triangular)
    if singular_values.size == 0:
        return None, False, empty_discarded

    largest = float(singular_values[0])
    if not np.isfinite(largest) or largest <= 0.0:
        return None, False, empty_discarded

    retained = singular_values >= largest * _SQRT_EPS
    if not np.any(retained):
        return None, False, empty_discarded

    discarded = right_vectors[~retained]

    # ``identifiable`` maps the retained subspace back to parameter space and
    # rescales it, so that ``identifiable @ identifiable.T`` is the
    # pseudo-inverse of the Gauss-Newton Hessian and the Gauss-Newton
    # curvature becomes the identity in these coordinates.
    identifiable = right_vectors[retained].T / singular_values[retained][None, :]

    curvature = identifiable.T @ (second_order[:, None] * identifiable)
    curvature = 0.5 * (curvature + curvature.T)

    if float(np.linalg.eigvalsh(curvature)[0]) <= -1.0 + _NEWTON_CURVATURE_MARGIN:
        # ``I + curvature`` is not positive definite, so neither is the full
        # Newton Hessian. Fisher scoring for this iteration, which in this
        # basis is exactly the rank-truncated Gauss-Newton solve.
        return identifiable @ (identifiable.T @ grad), True, discarded

    identity = np.eye(identifiable.shape[1])
    step = identifiable @ np.linalg.solve(identity + curvature, identifiable.T @ grad)
    return step, False, discarded


def _compute_cross_gram(
    B_i: NDArray,
    bin_i: NDArray | None,
    B_j: NDArray,
    bin_j: NDArray | None,
    W: NDArray,
) -> NDArray:
    """Cross-group weighted Gram ``B_i' diag(W) B_j`` of two SCOP groups.

    Each group is a basis and an optional bin index into it. Two binned
    groups meet in their joint weight table when it holds at most
    ``_MAX_DISC_DISC_HIST_CELLS`` cells, the group-matrix kernels' cap, and
    otherwise in one group's weighted rows added onto the other's support
    (``_support_support_raw_cross``), whose memory does not grow with the
    rows. Otherwise the binned side is scattered to observation level.
    """
    if bin_i is not None and bin_j is not None:
        if B_i.shape[0] * B_j.shape[0] > _MAX_DISC_DISC_HIST_CELLS:
            return _support_support_raw_cross(B_i, bin_i, B_j, bin_j, W)
        W_2d = _disc_disc_2d_hist(bin_i, bin_j, W, B_i.shape[0], B_j.shape[0])
        return B_i.T @ W_2d @ B_j
    rows_i = B_i if bin_i is None else B_i[bin_i]
    rows_j = B_j if bin_j is None else B_j[bin_j]
    return rows_i.T @ (rows_j * W[:, None])


def _centre_rows(values: NDArray, mass: NDArray, sum_w: float) -> tuple[NDArray, NDArray]:
    """``values`` less their ``mass``-weighted mean, and that mean.

    The rows are differenced from the heaviest row before the mean of the
    differences is taken, as ``_anchor_center_support`` centres a support:
    no row meets the mean's rounding at its own magnitude.
    """
    anchor = int(np.argmax(mass))
    centred = values - values[anchor]
    shift = (mass @ centred) / sum_w
    centred -= shift
    return centred, values[anchor] + shift


@dataclass(frozen=True)
class _SCOPMoments:
    """The SCOP block's weighted moments at one iterate, joint over its groups.

    Moments are in mapped coordinates (``gamma``). ``gram`` and ``score``
    are products of columns centred about their weighted means, ``X~_s' W
    X~_s`` and ``X~_s' W r`` for the working residual ``r = z - eta``; the
    block's own, uncentred moments follow from them and ``mean``.
    """

    gram: NDArray
    score: NDArray
    mean: NDArray
    """Each SCOP column's weighted mean."""
    weighted_sum: NDArray
    """``X~_s' W 1``: zero but for rounding."""
    sum_w: float
    centred: tuple[tuple[NDArray, NDArray | None], ...]
    """Each group's centred support and its row index (``None``: the rows themselves)."""
    jacobian: NDArray
    pull: NDArray
    """``lambda S theta``, the penalty's gradient."""
    own_gram: NDArray
    """The block's own Gram ``X_s' W X_s``, uncentred."""
    gauss_newton: NDArray
    """The block's Gauss-Newton Hessian ``J X_s' W X_s J + lambda S``."""
    curvature: NDArray
    """The diagonal of the map's curvature ``E = -gamma''(theta) * X_s' W r``."""

    @property
    def finite(self) -> bool:
        return all(
            bool(np.all(np.isfinite(values)))
            for values in (
                self.gram,
                self.score,
                self.mean,
                self.pull,
                self.gauss_newton,
                self.curvature,
            )
        )


@dataclass(frozen=True)
class _CompactLayout:
    """A discretized design's groups as compact supports, with their codes side by side.

    A categorical's support, the identity with a zero row for its base level
    (``_compact_support``), is never formed: its ``values`` are ``None``, and
    its block of a cross moment is its table's level rows.
    """

    supports: tuple[tuple[NDArray | None, NDArray | None], ...]
    """Each group's ``(values, transform)``: its rows are ``values[codes] @ transform``."""
    codes: NDArray
    """``(n, groups)``: each row's support row in every group."""
    bounds: NDArray
    """Cumulative support rows: group ``g``'s table holds rows ``bounds[g]`` to ``bounds[g + 1]``."""
    starts: NDArray
    """Cumulative group widths: group ``g``'s columns are ``starts[g]`` to ``starts[g + 1]``."""


# Owner: the reduced design, held weakly, so a REML run whose fits share one
# reduced design (``SCOPRunCentring``) lays it out once and the layout dies
# with the design. A layout depends on the design's group matrices alone,
# which never change, so nothing invalidates it. It retains the packed codes,
# 4 bytes per row and group, and the supports the design does not own: a
# discretized spline-by-category level's support with its zero row.
_COMPACT_LAYOUTS: weakref.WeakKeyDictionary[DesignMatrix, _CompactLayout | None] = (
    weakref.WeakKeyDictionary()
)


def _compact_layout(design: DesignMatrix, rows: Sequence[int]) -> _CompactLayout | None:
    """The design's compact layout; ``rows`` are its groups' support rows (``_compact_support_rows``)."""
    if design in _COMPACT_LAYOUTS:
        return _COMPACT_LAYOUTS[design]
    supports = []
    codes = np.empty((design.n, len(rows)), dtype=np.int32)
    for g, gm in enumerate(design.group_matrices):
        if isinstance(gm, CategoricalGroupMatrix):
            supports.append((None, None))
            codes[:, g] = gm.codes
        else:
            values, group_codes, transform = _compact_support(gm)
            supports.append((values, transform))
            codes[:, g] = group_codes
    layout = None
    if not codes.size or (codes.min() >= 0 and np.all(codes.max(axis=0) < rows)):
        widths = [gm.shape[1] for gm in design.group_matrices]
        layout = _CompactLayout(
            supports=tuple(supports),
            codes=codes,
            bounds=np.concatenate(([0], np.cumsum(rows))).astype(np.int64),
            starts=np.concatenate(([0], np.cumsum(widths))).astype(np.int64),
        )
    _COMPACT_LAYOUTS[design] = layout
    return layout


class _JointWeightCross:
    """``X~_r' W X~_s`` from each ordinary group's weight table against the SCOP bins.

    Every group of a discretized design is a compact support, its rows
    ``values[codes] @ transform`` (``_compact_support``), and a binned SCOP
    group's columns are ``basis[bins]``, so group ``g``'s cross block is
    ``transform' values~' H_g basis~`` with ``H_g`` the weight table of
    ``(codes_g, bins)`` and both supports centred about their weighted means
    before any product: the anchor-support route's algebra
    (``_anchor_support_gram_rhs``). The table's row sums are the support's
    weights, so centring costs no pass over the rows. One row pass forms
    every ``H_g``, reading the weights and the SCOP bins once rather than once
    per group.

    Sizes are read from the groups' metadata before anything is allocated
    (``_compact_support_rows``, as ``_compact_supports_fit``): a design with a
    group that has no compact support, tables past the packed-histogram cap,
    or packed codes past the package's 64 MiB byte budget is not built here.
    """

    def __init__(self, layout: _CompactLayout) -> None:
        self.layout = layout

    @classmethod
    def build(
        cls, design: DesignMatrix, columns: Sequence[tuple[NDArray, NDArray | None]]
    ) -> _JointWeightCross | None:
        if any(bins is None for _, bins in columns):
            return None
        rows = [_compact_support_rows(gm) for gm in design.group_matrices]
        widest = max(basis.shape[0] for basis, _ in columns)
        if (
            any(count is None for count in rows)
            or sum(rows) * widest > _MAX_PACKED_HIST_CELLS
            or 4 * design.n * len(rows) > _MAX_CROSS_EXPANSION_BYTES
        ):
            return None
        layout = _compact_layout(design, rows)
        return None if layout is None else cls(layout)

    def cross_moment(
        self, W: NDArray, scop: Sequence[tuple[NDArray, NDArray]], sum_w: float
    ) -> NDArray:
        """The centred cross block against ``scop``, each group's centred support and bins."""
        layout = self.layout
        cross = np.zeros((int(layout.starts[-1]), sum(basis.shape[1] for basis, _ in scop)))
        supports = None
        column = 0
        for basis, bins in scop:
            n_scop = basis.shape[0]
            edges = layout.bounds * n_scop
            table = _joint_weight_tables(layout.codes, edges[:-1], bins, n_scop, W, int(edges[-1]))
            if supports is None:
                supports = [
                    _centred_support(values, table[a:b].reshape(-1, n_scop).sum(axis=1), sum_w)
                    for (values, _), a, b in zip(
                        layout.supports, edges[:-1], edges[1:], strict=True
                    )
                ]
            block = slice(column, column + basis.shape[1])
            for g, ((values, transform), support) in enumerate(
                zip(layout.supports, supports, strict=True)
            ):
                moment = table[edges[g] : edges[g + 1]].reshape(-1, n_scop) @ basis
                if values is None:
                    # a categorical's centred support rows are its levels' indicators less
                    # their weighted frequencies ``support``, then its base level's -support
                    moment = moment[:-1] - np.outer(support, moment.sum(axis=0))
                else:
                    moment = support.T @ moment
                if transform is not None:
                    moment = transform.T @ moment
                cross[layout.starts[g] : layout.starts[g + 1], block] = moment
            column = block.stop
        return cross


def _centred_support(values: NDArray | None, mass: NDArray, sum_w: float) -> NDArray:
    """A compact support centred about its weighted mean; a categorical's level frequencies."""
    if values is None:
        return mass[:-1] / sum_w
    return _centre_rows(np.asarray(values, dtype=np.float64), mass, sum_w)[0]


class _OrdinaryCross:
    """``X~_r' W X~_s``, the centred cross block of the ordinary design and the SCOP columns.

    Every entry is a sum of products of centred columns; no raw moment is
    subtracted, as the reduced system itself never subtracts one outside its
    certificate (``build_centered_system``). The design splits by column
    type, as ``centred_data_score`` does:

    - a ``DenseGroupMatrix`` column, whose type does not bound its offset,
      is centred row by row about the system's exact pair, weighted, and
      summed onto each SCOP group's bins;
    - the other groups take one pass of joint weight tables
      (``_JointWeightCross``) when each has a compact support within the
      caps, and otherwise the per-pair group-matrix kernels against the
      centred SCOP supports, less ``mean_x (1' W X~_s)``, a product of two
      roundings.

    ``moment`` also returns the weighted norm with which each ordinary column
    entered its products: centred on the first two routes, raw on the
    per-pair one, whose rows meet the SCOP columns uncentred.
    """

    def __init__(self, design: DesignMatrix, columns) -> None:
        self.design = design
        dense = dense_columns(design)
        self.bounded_index = np.flatnonzero(~dense)
        bounded = design if not dense.any() else _dense_split(design).bounded
        self.bounded = bounded if bounded.p else None
        self.tables = None if self.bounded is None else _JointWeightCross.build(bounded, columns)

    def _dense_columns(self):
        offset = 0
        for matrix in self.design.group_matrices:
            if type(matrix) is DenseGroupMatrix:
                for local in range(matrix.shape[1]):
                    yield offset + local, matrix.M[:, local]
            offset += matrix.shape[1]

    def moment(self, W: NDArray, moments: _SCOPMoments, system: CenteredSystem):
        """``(cross, norms)``: the centred cross block and each ordinary column's product norm."""
        scop = moments.centred
        cross = np.zeros((self.design.p, moments.gram.shape[0]))
        norms = np.sqrt(np.maximum(np.diag(system.data_gram), 0.0))
        if self.bounded is not None:
            index = self.bounded_index
            if self.tables is not None:
                cross[index] = self.tables.cross_moment(W, scop, moments.sum_w)
            else:
                columns = [
                    DenseGroupMatrix(basis)
                    if bins is None
                    else DiscretizedSCOPGroupMatrix(basis, bins)
                    for basis, bins in scop
                ]
                plan = CrossMatrixExecutionPlan(
                    self.bounded.execution_plan, MatrixExecutionPlan(columns, n=self.design.n)
                )
                mean = system.mean_x[index]
                cross[index] = plan.cross_moment(W) - np.outer(mean, moments.weighted_sum)
                norms[index] = np.sqrt(norms[index] ** 2 + moments.sum_w * mean**2)
        centre, centre_lo = system.centre_pair()
        for k, values in self._dense_columns():
            rows = values - centre[k]
            if centre_lo is not None:
                rows -= centre_lo[k]
            rows *= W
            column = 0
            for basis, bins in scop:
                block = slice(column, column + basis.shape[1])
                summed = rows if bins is None else np.bincount(bins, rows, basis.shape[0])
                cross[k, block] = basis.T @ summed
                column = block.stop
        return cross, norms


def _exact_support(spec: SCOPGroupSpec, supports: dict | None) -> tuple[NDArray, NDArray | None]:
    """A group's columns as a basis and a row index into it.

    A group with no bins is compressed to its distinct rows only when they
    are at most half its rows: its moments then cost a bincount over the
    rows plus at most half the row-level products. An almost-unique
    support saves nothing, and every table it indexes grows with its rows.
    ``supports``, the run's record by group (``SCOPRunCentring``), keeps
    the compressed support or ``None`` for the next fit of the same rows.
    """
    if spec.bin_idx is not None:
        return spec.B_scop, spec.bin_idx
    if supports is not None and spec.group_index in supports:
        compressed = supports[spec.group_index]
    else:
        support = build_exact_scop_support(spec.B_scop)
        compressed = (
            None
            if support is None or 2 * support.n_unique > spec.B_scop.shape[0]
            else (support.B_unique, support.row_to_support)
        )
        if supports is not None:
            supports[spec.group_index] = compressed
    return (spec.B_scop, None) if compressed is None else compressed


def _group_lambda(lambdas: dict[str, float] | float, group: GroupSlice) -> float:
    return float(lambdas.get(group.name, 0.0) if isinstance(lambdas, dict) else lambdas)


class SCOPJointNewton:
    """The joint Newton step of one inner fit's shape-constrained groups.

    Holds what is fixed for the fit: the groups in latent order, their
    penalties at the fit's smoothing parameters, each group's columns on
    its exact support (a basis and a row index into it, the distinct rows
    of a non-discretized group, read from the run's ``supports`` when it
    has them), and the centred cross block between the ordinary (reduced)
    design and those columns (``_OrdinaryCross``).
    """

    def __init__(
        self,
        specs: Sequence[SCOPGroupSpec],
        lambdas: dict[str, float] | float,
        reduced_design: DesignMatrix,
        n: int,
        supports: dict | None = None,
    ) -> None:
        self.specs = tuple(specs)
        self.n = int(n)
        widths = [spec.S_scop.shape[0] for spec in self.specs]
        offsets = np.concatenate(([0], np.cumsum(widths))).astype(int)
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))
        self.width = int(offsets[-1])
        self.penalty = np.zeros((self.width, self.width))
        self.penalty_root = np.zeros((self.width, self.width))
        for spec, block in zip(self.specs, self.slices, strict=True):
            lam = _group_lambda(lambdas, spec.group)
            self.penalty[block, block] = lam * spec.S_scop
            self.penalty_root[block, block] = np.sqrt(max(lam, 0.0)) * _psd_root(spec.S_scop)
        self.columns = tuple(_exact_support(spec, supports) for spec in self.specs)
        self._cross = _OrdinaryCross(reduced_design, self.columns) if reduced_design.p else None

    def predictor(self, thetas: Sequence[NDArray]) -> NDArray:
        """``X_s gamma(theta)`` at observation level, summed over the groups."""
        total = np.zeros(self.n)
        for spec, (basis, bins), theta in zip(self.specs, self.columns, thetas, strict=True):
            values = basis @ spec.reparam.forward(theta)
            total += values if bins is None else values[bins]
        return total

    def _moments(
        self, thetas: Sequence[NDArray], W: NDArray, z: NDArray, eta: NDArray
    ) -> _SCOPMoments:
        q = self.width
        sum_w = float(np.sum(W))
        gram = np.zeros((q, q))
        score = np.empty(q)
        mean = np.empty(q)
        weighted_sum = np.empty(q)
        jacobian = np.empty(q)
        second = np.empty(q)
        centred = []
        for spec, (basis, bins), block, theta in zip(
            self.specs, self.columns, self.slices, thetas, strict=True
        ):
            if bins is None:
                weights, weighted_residual = W, W * (z - eta)
            else:
                weights, weighted_residual = _scop_bin_moments(bins, W, z, eta, basis.shape[0])
            support, mean[block] = _centre_rows(basis, weights, sum_w)
            centred.append((support, bins))
            gram[block, block] = support.T @ (support * weights[:, None])
            score[block] = support.T @ weighted_residual
            weighted_sum[block] = support.T @ weights
            jacobian[block] = spec.reparam.jacobian_diagonal(theta)
            second[block] = spec.reparam.second_derivative_diagonal(theta)
        # every group's weighted residuals, by bin or by row, sum to ``W' r``
        residual_sum = float(np.sum(weighted_residual))
        for a in range(len(self.specs)):
            for b in range(a + 1, len(self.specs)):
                cross = _compute_cross_gram(*centred[a], *centred[b], W)
                gram[self.slices[a], self.slices[b]] = cross
                gram[self.slices[b], self.slices[a]] = cross.T
        pull = self.penalty @ np.concatenate(thetas)
        with np.errstate(over="ignore", invalid="ignore"):
            # X_s = X~_s + 1 mean', so X_s' W X_s and X_s' W r follow exactly
            shift = np.outer(weighted_sum, mean)
            own_gram = gram + shift + shift.T + sum_w * np.outer(mean, mean)
            gauss_newton = jacobian[:, None] * own_gram * jacobian[None, :] + self.penalty
            curvature = -second * (score + mean * residual_sum)
        return _SCOPMoments(
            gram=gram,
            score=score,
            mean=mean,
            weighted_sum=weighted_sum,
            sum_w=sum_w,
            centred=tuple(centred),
            jacobian=jacobian,
            pull=pull,
            own_gram=own_gram,
            gauss_newton=gauss_newton,
            curvature=curvature,
        )

    def _results(
        self,
        thetas: Sequence[NDArray],
        theta_change: NDArray,
        moments: _SCOPMoments,
        used_fisher: bool,
        discarded: NDArray,
    ) -> dict[int, SCOPNewtonResult]:
        """Per-group results carrying the block curvature the step used.

        That curvature is the SCOP block's own Hessian ``J G J + lambda S +
        E`` (without ``E`` after a Gauss-Newton fallback), uncentred and on
        the resolved range: the REML geometry substitutes it for the SCOP
        diagonal block of the joint Hessian it assembles.
        """
        hessian = moments.gauss_newton.copy()
        if not used_fisher:
            hessian[np.diag_indices_from(hessian)] += moments.curvature
        hessian = _restrict_to_resolved_range(hessian, discarded)
        return {
            spec.group_index: SCOPNewtonResult(
                beta_new=theta + theta_change[block],
                step_norm=float(np.linalg.norm(theta_change[block])),
                used_fisher_fallback=used_fisher,
                H_penalized=hessian[block, block].copy(),
                discarded_directions=discarded[:, block].copy(),
            )
            for spec, block, theta in zip(self.specs, self.slices, thetas, strict=True)
        }

    def _no_step(self, thetas: Sequence[NDArray]) -> dict[int, SCOPNewtonResult]:
        """A non-finite SCOP system takes no step and reports the penalty as its curvature."""
        return {
            spec.group_index: SCOPNewtonResult(
                beta_new=np.array(theta, dtype=float, copy=True),
                step_norm=0.0,
                used_fisher_fallback=False,
                H_penalized=self.penalty[block, block].copy(),
                discarded_directions=np.zeros((0, block.stop - block.start)),
            )
            for spec, block, theta in zip(self.specs, self.slices, thetas, strict=True)
        }

    def _solve(
        self, root: NDArray, moments: _SCOPMoments, gradient: NDArray
    ) -> tuple[NDArray, bool, NDArray]:
        """Newton step of a SCOP block whose data part is ``J R' R J``.

        The factor is ``[R J ; sqrt(lambda) rS]`` with ``R = root``:
        ``J`` scales the factor's columns rather than a Gram's, so a latent
        coordinate running to its log-space boundary (``J -> 0``) is resolved
        to the factor's accuracy, not its square. Returns the step with the
        sign of a Newton direction subtracted from ``theta``, whether the
        Gauss-Newton block was used, and the discarded directions. No
        identifiable direction at all gives no step.
        """
        factor = np.vstack([root * moments.jacobian[None, :], self.penalty_root])
        step, used_fisher, discarded = _truncated_factor_step(factor, moments.curvature, gradient)
        if step is None:
            return np.zeros(self.width), True, np.zeros((0, self.width))
        return step, used_fisher, discarded

    def _schur_rows(
        self,
        W: NDArray,
        moments: _SCOPMoments,
        reduced: CenteredSystem,
        solve: Callable[[NDArray], NDArray],
        coupled: NDArray,
    ) -> tuple[NDArray, NDArray, int]:
        """``(F, g, count)``: a factor of the Schur block's data part, from the rows (``_resolved_root``).

        The centred SCOP columns less their regression on the ordinary
        design, ``Z = X~_s - X~_r A^+ C``, over the penalty's rows ``L_r A^+
        C`` (``L_r' L_r = S_r``): ``[sqrt(W) Z; L_r A^+ C]`` has the Gram ``G~
        - C' A^+ C`` that the step's back-substitution takes, ``A^+`` being
        the ordinary block's own ``solve``. That is one block of classical
        Gram-Schmidt against the ordinary design through its seminormal
        equations, so it is projected twice. The first pass takes ``coupled
        = A^+ C`` from the formed cross block, whose rounding ``dC`` leaves
        ``sqrt(dC' A^+ dC)`` in ``Z`` however small ``Z`` is; the second
        regresses what the first left, as Bjorck's corrected seminormal
        equations refine a seminormal solve (*Linear Algebra Appl.* 88, 1987)
        and as reorthogonalized block Gram-Schmidt projects each block twice
        (BCGSI+; Carson, Lund, Ma & Oktay, arXiv 2408.10109). It leaves the
        first pass's error times order ``n u kappa(A)``. The rows are then
        factored by the TSQR (``streamed_weighted_factor``).

        ``g`` bounds each column's rounding: its SCOP column's norm plus
        ``|A^+ C|`` times each ordinary column's, the raw norm on a column the
        product does not centre row by row (``centred_matvec``), with the
        penalty's diagonal. ``count`` is two such products of ``p + 3`` terms
        and the Householder QR's ``m q`` on ``m`` rows (Higham 2002, Thm 19.4).
        """
        q, p = self.width, coupled.shape[0]
        columns = np.empty((self.n, q))
        for (support, bins), block in zip(moments.centred, self.slices, strict=True):
            columns[:, block] = support if bins is None else support[bins]
        scale = np.sqrt(np.maximum(np.diag(moments.gram), 0.0))
        penalty_rows = np.zeros((0, q))
        if self._cross is not None and p:
            design, bounded = self._cross.design, self._cross.bounded
            centre, centre_lo = reduced.centre_pair()

            def regress(change: NDArray) -> None:
                for j, coefficients in enumerate(change.T):
                    columns[:, j] -= centred_matvec(design, coefficients, centre)
                    if centre_lo is not None:
                        columns[:, j] += math.fsum(centre_lo * coefficients)

            regress(coupled)
            weighted = (W[:, None] * columns).T
            residual = np.column_stack(
                [dense_centred_rmatvec(design, values, centre, centre_lo) for values in weighted]
            )
            if bounded is not None:
                index = self._cross.bounded_index
                residual[index] = np.column_stack(
                    [bounded.rmatvec(values) for values in weighted]
                ) - np.outer(reduced.mean_x[index], weighted.sum(axis=1))
            residual -= reduced.penalty @ coupled
            correction = np.column_stack([solve(column) for column in residual.T])
            regress(correction)
            norms = np.sqrt(np.maximum(np.diag(reduced.hessian), 0.0))
            raw = ~dense_columns(design)
            norms[raw] += 2.0 * np.abs(reduced.mean_x[raw]) * math.sqrt(moments.sum_w)
            scale = scale + (np.abs(coupled) + np.abs(correction)).T @ norms
            penalty_rows = penalty_factor(reduced.penalty) @ (coupled + correction)
        leaf = tsqr_leaf_rows(q)
        chunks = (
            (start, min(start + leaf, self.n), columns[start : start + leaf])
            for start in range(0, self.n, leaf)
        )
        factor = np.vstack((streamed_weighted_factor(chunks, W), penalty_rows))
        rows = self.n + penalty_rows.shape[0]
        return factor, scale, rows * q + 2 * (p + 3)

    def step(
        self,
        thetas: Sequence[NDArray],
        *,
        W: NDArray,
        z: NDArray,
        eta: NDArray,
        reduced: CenteredSystem,
        solve: Callable[[NDArray], NDArray],
        ordinary_change: NDArray,
        debug_recorder=None,
        debug_context: dict[str, object] | None = None,
    ) -> SCOPJointStep:
        """One joint Newton step, completing an ordinary solve taken with ``theta`` fixed.

        ``W`` and ``z`` are the working weights and response, ``eta`` the
        committed predictor (unclipped, offset included), so ``z - eta`` is
        the working residual at the current coefficients. ``reduced`` is the
        ordinary block's centred system and ``solve`` applies its
        (pseudo-)inverse; ``ordinary_change`` is that block's step with the
        SCOP coordinates held fixed.

        With ``r`` the working residual, ``delta_b`` that step and ``C =
        X~_r' W X~_s`` the centred cross block, the reduced SCOP gradient is
        ``-J (X~_s' W r - C' delta_b) + lambda S theta``: the residual the
        fixed-``theta`` solve leaves, without forming it. The block's data
        part is taken on the directions the data resolve, from the rows when
        the Grams cannot decide (``_resolved_root``, ``_schur_rows``).
        """
        moments = self._moments(thetas, W, z, eta)
        if not moments.finite:
            return SCOPJointStep(self._no_step(thetas), np.zeros_like(ordinary_change), 0.0)
        data_gram = moments.gram
        data_score = moments.score
        scale = np.sqrt(np.maximum(np.diag(data_gram), 0.0))
        coupled = np.zeros((ordinary_change.shape[0], self.width))
        if self._cross is not None:
            cross, norms = self._cross.moment(W, moments, reduced)
            coupled = np.column_stack([solve(column) for column in cross.T])
            data_gram = data_gram - cross.T @ coupled
            data_score = data_score - cross.T @ ordinary_change
            # the reduced Gram's raw rungs admit a column at kappa^2 <= 2
            # (``_raw_centering_well_scaled``): its rounding is within sqrt(2) of the norm
            centred = np.sqrt(np.maximum(np.diag(reduced.data_gram), 0.0))
            scale = scale + np.abs(coupled).T @ np.maximum(np.sqrt(2.0) * centred, norms)
        root, data_score = _resolved_root(
            data_gram,
            data_score,
            scale,
            2 * self.n + coupled.shape[0] + self.width + 2,
            lambda: self._schur_rows(W, moments, reduced, solve, coupled),
        )
        gradient = -moments.jacobian * data_score + moments.pull
        step, used_fisher, discarded = self._solve(root, moments, gradient)
        mapped_change = -moments.jacobian * step
        results = self._results(thetas, -step, moments, used_fisher, discarded)
        for spec in self.specs:
            _append_scop_trace(
                debug_recorder,
                result=results[spec.group_index],
                debug_context=debug_context,
                group_name=spec.group.name,
            )
        return SCOPJointStep(
            results=results,
            ordinary_correction=-(coupled @ mapped_change),
            intercept_shift=float(moments.mean @ mapped_change),
        )

    def block_curvature(
        self, thetas: Sequence[NDArray], *, W: NDArray, z: NDArray, eta: NDArray
    ) -> dict[int, SCOPNewtonResult]:
        """The SCOP block's own curvature at ``theta``, which does not move.

        The Newton Hessian ``J G J + lambda S + E`` of the SCOP block alone
        (uncentred, the ordinary coefficients held fixed), whether it is
        positive definite on the block's resolved range, and the directions
        that block cannot resolve: what the REML geometry reads for the
        retained model.
        """
        moments = self._moments(thetas, W, z, eta)
        if not moments.finite:
            return self._no_step(thetas)
        root = _psd_root(moments.own_gram)
        _, used_fisher, discarded = self._solve(root, moments, np.zeros(self.width))
        return self._results(thetas, np.zeros(self.width), moments, used_fisher, discarded)
