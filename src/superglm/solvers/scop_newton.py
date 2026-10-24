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
coordinates, ``J = diag(gamma'(theta))``, and ``E = -diag(gamma''(theta) *
X_s' W r)`` is the curvature of the map at the working residual ``r``.

The step is taken by block elimination. The caller has already solved the
ordinary block with ``theta`` held fixed; ``SCOPJointNewton.step`` reduces
the SCOP block to its Schur complement ``J (G - C' A^-1 C) J + lambda S + E``,
takes the Newton step there and corrects the ordinary solve by
back-substitution. Pya & Wood (2015, §3.2) report that Fisher scoring caused
convergence problems for these models, so ``E`` is kept unless the reduced
block is then not positive definite, when that iteration takes the
Gauss-Newton block instead; directions the block cannot resolve are truncated
at ``sqrt(eps)`` of its largest singular value (§3.2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from superglm.group_matrix import (
    CrossMatrixExecutionPlan,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    MatrixExecutionPlan,
    _disc_disc_2d_hist,
)
from superglm.solvers.scop import SCOPSolverReparam
from superglm.solvers.scop_exact_support import build_exact_scop_support
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.solvers.centered_system import CenteredSystem

_SQRT_EPS = float(np.finfo(np.float64).eps) ** 0.5
"""Pya & Wood (2015) §3.2 singular-value cutoff, relative to the largest."""

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


def _warmup_scop() -> None:
    """Compile the SCOP bin moments for the operands a shape-constrained fit passes."""
    empty = np.zeros(0, dtype=np.float64)
    _scop_bin_moments(np.zeros(0, dtype=np.int64), empty, empty, empty, 0)


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

    Restricting removes the frozen direction together with its curvature.
    Because ``P' H_full P == I + curvature`` for the retained basis, the
    positive-definiteness test already applied to ``I + curvature`` is exactly
    the statement that the restriction is positive definite, so what is
    reported here is positive semidefinite by construction rather than by
    hope. A step that discarded nothing is returned untouched.
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

    Each group is a basis and an optional bin index into it; two
    discretized groups meet in a 2D weight histogram, otherwise the binned
    side is scattered to observation level.
    """
    if bin_i is not None and bin_j is not None:
        W_2d = _disc_disc_2d_hist(bin_i, bin_j, W, B_i.shape[0], B_j.shape[0])
        return B_i.T @ W_2d @ B_j
    rows_i = B_i if bin_i is None else B_i[bin_i]
    rows_j = B_j if bin_j is None else B_j[bin_j]
    return rows_i.T @ (rows_j * W[:, None])


@dataclass(frozen=True)
class _SCOPMoments:
    """The SCOP block's weighted moments at one iterate, joint over its groups.

    Moments are in mapped coordinates (``gamma``) and uncentred; ``score``
    is ``X_s' W r`` for the working residual ``r = z - eta``.
    """

    gram: NDArray
    xtw: NDArray
    score: NDArray
    residual_sum: float
    jacobian: NDArray
    pull: NDArray
    """``lambda S theta``, the penalty's gradient."""
    gauss_newton: NDArray
    """The block's Gauss-Newton Hessian ``J G J + lambda S``."""
    curvature: NDArray
    """The diagonal of the map's curvature ``E = -gamma''(theta) * score``."""

    @property
    def finite(self) -> bool:
        return all(
            bool(np.all(np.isfinite(values)))
            for values in (self.gram, self.score, self.pull, self.gauss_newton, self.curvature)
        )


def _exact_support(spec: SCOPGroupSpec) -> tuple[NDArray, NDArray | None]:
    """A group's columns as a basis and a row index into it, compressed where rows repeat."""
    if spec.bin_idx is not None:
        return spec.B_scop, spec.bin_idx
    support = build_exact_scop_support(spec.B_scop)
    if support is None:
        return spec.B_scop, None
    return support.B_unique, support.row_to_support


def _group_lambda(lambdas: dict[str, float] | float, group: GroupSlice) -> float:
    return float(lambdas.get(group.name, 0.0) if isinstance(lambdas, dict) else lambdas)


class SCOPJointNewton:
    """The joint Newton step of one inner fit's shape-constrained groups.

    Holds what is fixed for the fit: the groups in latent order, their
    penalties at the fit's smoothing parameters, each group's columns on
    its exact support (a basis and a row index into it, the distinct rows
    of a non-discretized group), and the cross-moment plan between the
    ordinary (reduced) design and those columns, which uses the
    group-matrix kernels and bins the reduced Gram is built on.
    """

    def __init__(
        self,
        specs: Sequence[SCOPGroupSpec],
        lambdas: dict[str, float] | float,
        reduced_design: DesignMatrix,
        n: int,
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
        self.columns = tuple(_exact_support(spec) for spec in self.specs)
        self._cross: CrossMatrixExecutionPlan | None = None
        if reduced_design.p:
            columns = [
                DenseGroupMatrix(basis) if bins is None else DiscretizedSCOPGroupMatrix(basis, bins)
                for basis, bins in self.columns
            ]
            self._cross = CrossMatrixExecutionPlan(
                reduced_design.execution_plan, MatrixExecutionPlan(columns, n=self.n)
            )

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
        gram = np.zeros((q, q))
        xtw = np.empty(q)
        score = np.empty(q)
        jacobian = np.empty(q)
        second = np.empty(q)
        residual_sum = 0.0
        for spec, (basis, bins), block, theta in zip(
            self.specs, self.columns, self.slices, thetas, strict=True
        ):
            if bins is None:
                weights, weighted_residual = W, W * (z - eta)
            else:
                weights, weighted_residual = _scop_bin_moments(bins, W, z, eta, basis.shape[0])
            gram[block, block] = basis.T @ (basis * weights[:, None])
            xtw[block] = basis.T @ weights
            score[block] = basis.T @ weighted_residual
            residual_sum = float(np.sum(weighted_residual))
            jacobian[block] = spec.reparam.jacobian_diagonal(theta)
            second[block] = spec.reparam.second_derivative_diagonal(theta)
        for a in range(len(self.specs)):
            for b in range(a + 1, len(self.specs)):
                cross = _compute_cross_gram(*self.columns[a], *self.columns[b], W)
                gram[self.slices[a], self.slices[b]] = cross
                gram[self.slices[b], self.slices[a]] = cross.T
        pull = self.penalty @ np.concatenate(thetas)
        with np.errstate(over="ignore", invalid="ignore"):
            gauss_newton = jacobian[:, None] * gram * jacobian[None, :] + self.penalty
            curvature = -second * score
        return _SCOPMoments(gram, xtw, score, residual_sum, jacobian, pull, gauss_newton, curvature)

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
        self, data_gram: NDArray, moments: _SCOPMoments, gradient: NDArray
    ) -> tuple[NDArray, bool, NDArray]:
        """Newton step of a SCOP block whose data part is ``J data_gram J``.

        The factor is ``[R J ; sqrt(lambda) rS]`` with ``R' R = data_gram``:
        ``J`` scales the factor's columns rather than a Gram's, so a latent
        coordinate running to its log-space boundary (``J -> 0``) is resolved
        to the factor's accuracy, not its square. Returns the step with the
        sign of a Newton direction subtracted from ``theta``, whether the
        Gauss-Newton block was used, and the discarded directions. No
        identifiable direction at all gives no step.
        """
        factor = np.vstack([_psd_root(data_gram) * moments.jacobian[None, :], self.penalty_root])
        step, used_fisher, discarded = _truncated_factor_step(factor, moments.curvature, gradient)
        if step is None:
            return np.zeros(self.width), True, np.zeros((0, self.width))
        return step, used_fisher, discarded

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

        With ``r`` the working residual and ``delta_b`` that step, the
        reduced SCOP gradient is ``-J (X_s' W r - C' delta_b) + lambda S
        theta`` in centred moments: the residual the fixed-``theta`` solve
        leaves, without forming it.
        """
        moments = self._moments(thetas, W, z, eta)
        if not moments.finite:
            return SCOPJointStep(self._no_step(thetas), np.zeros_like(ordinary_change), 0.0)
        sum_w = reduced.sum_w
        mean_s = moments.xtw / sum_w
        data_gram = moments.gram - np.outer(moments.xtw, mean_s)
        data_score = moments.score - moments.xtw * (moments.residual_sum / sum_w)
        coupled = np.zeros((ordinary_change.shape[0], self.width))
        if self._cross is not None:
            cross = self._cross.cross_moment(W) - np.outer(reduced.mean_x, moments.xtw)
            coupled = np.column_stack([solve(column) for column in cross.T])
            data_gram = data_gram - cross.T @ coupled
            data_score = data_score - cross.T @ ordinary_change
        data_gram = 0.5 * (data_gram + data_gram.T)
        gradient = -moments.jacobian * data_score + moments.pull
        step, used_fisher, discarded = self._solve(data_gram, moments, gradient)
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
            intercept_shift=float(mean_s @ mapped_change),
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
        gradient = -moments.jacobian * moments.score + moments.pull
        _, used_fisher, discarded = self._solve(moments.gram, moments, gradient)
        return self._results(thetas, np.zeros(self.width), moments, used_fisher, discarded)
