"""Pure convergence decisions shared by exact and discrete REML optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# The Newton engines' active-set freeze bar never drops below this floor,
# whatever reml_tol the caller asked for. Freezing classifies GEOMETRY --
# "is this direction informative at all" -- and an inferentially flat
# direction stays flat however precisely the informative lambdas are
# located. Coupling the bar to 0.1*reml_tol alone meant the tight 1e-9
# default un-froze the null directions the historical 1e-6 default froze at
# exactly this value; unfrozen, they march geometrically toward the lambda
# cap, paying 8-15 extra iterations, publishing platform-dependent lambda
# values, and exhausting the line search at tight tolerances.
FLAT_DIRECTION_FREEZE_FLOOR = 1e-7

# The curvature arm of the freeze decision. score_scale = 1+|objective|
# grows with the row count while log-lambda curvature saturates (measured
# on the flat-lambda stress design: informative |H_ii| 0.25 at 12k rows,
# 0.62 at 1e6, while score_scale went 7.5e4 -> 8.5e6), so judging |H_ii|
# against freeze_tol*score_scale froze an informative direction at 400k
# rows and every direction at 1e6 by iteration 3 -- publishing lambdas a
# factor e^5.6 from the optimum with SEs off by up to 87%. Absolute
# curvature cannot classify either: a null direction mid-march carries
# |H_ii| of the same order as an informative endgame (measured 0.056 vs
# 0.060 at 1e6 rows).
#
# The comparison is per penalty DIMENSION: raw row curvature scales with
# penalty rank (a random effect measured 112 -> 255 -> 391 across
# 300 -> 600 -> 1000 levels, ~0.4 per rank, while an informative low-rank
# spline held ~2.5 raw, ~0.8 per rank), so a raw ratio-to-strongest bar
# froze a real-signal spline beside a 600-level random effect.
# Rank-normalized, null directions measure <= 6e-5 per dimension at their
# freeze states, the tightest informative direction 5.2e-3, and strong
# ones 0.3 .. 0.8 -- commensurate across rank scales. The anchor keeps
# the bar meaningful when every direction is weak (an all-null model has
# no strong direction; without the anchor its last null would chase the
# lambda cap forever): 1e-2 * 0.1 = 1e-3 per dimension splits the fully
# null (<= 6e-5) from the tightest informative (5.2e-3).
FLAT_DIRECTION_CURVATURE_REL = 1e-2
FLAT_DIRECTION_CURVATURE_ANCHOR = 0.1


def _score_scale(objective: float) -> float:
    return max(1.0 + abs(objective), 1.0)


def _freeze_gradient_bar(objective: float, tolerance: float) -> float:
    freeze_tol = max(0.1 * tolerance, FLAT_DIRECTION_FREEZE_FLOOR)
    return freeze_tol * _score_scale(objective)


@dataclass(frozen=True)
class FlatDirectionDecision:
    """The freeze verdict with the per-direction quantities it judged."""

    frozen: NDArray
    row_curvature: NDArray
    penalty_rank: NDArray
    normalized_curvature: NDArray
    curvature_bar: float


def direction_penalty_ranks(penalty_components, penalty_ranks) -> NDArray:
    """Per-direction penalty rank, resolved the way the bootstrap resolves it.

    Computed at the freeze site from the CURRENT penalty context, never
    hoisted: the discrete engine replaces its penalty components at the
    bootstrap swap and again after every accepted update, and a rank
    resolved before those rebuilds normalizes the current Hessian with an
    obsolete value whenever a near-null eigenvalue crosses the rank
    threshold as the solver space changes.
    """
    return np.array(
        [
            max(
                pc.rank
                if pc.rank > 0
                else (penalty_ranks.get(pc.name, 0.0) if penalty_ranks else 0.0),
                1.0,
            )
            for pc in penalty_components
        ],
        dtype=np.float64,
    )


def freeze_flat_directions(
    projected_gradient: NDArray,
    hessian: NDArray,
    penalty_ranks: NDArray,
    estimated_mask: NDArray,
    *,
    objective: float,
    tolerance: float,
) -> FlatDirectionDecision:
    """Active-set freeze: mark directions with negligible gradient and curvature.

    The gradient arm is judged against the objective's scale -- both grow
    with the row count, so the comparison is dimensionally sound. The
    curvature arm judges each direction's ROW of the estimated Hessian
    block, not its diagonal alone: REML Hessians carry cross-terms
    (multi-penalty anisotropy adds them explicitly) and need not be
    positive definite, so a coordinate can hold a small diagonal with
    large off-diagonal curvature -- a [[0, c], [c, 0]] block has zero
    diagonals yet real curvature in the coupled eigenvector. Coupling to
    a FIXED direction is excluded: that lambda never moves, so the cross
    term is not exploitable. Curvature is normalized per penalty
    dimension SYMMETRICALLY -- the matrix is scaled D^{-1/2} H D^{-1/2}
    with D = diag(ranks), so a diagonal reads H_ii / r_i and a shared
    cross-term reads H_ij / sqrt(r_i * r_j), the same value from both
    ends. Row-asymmetric division read a rank-[1000, 1] coupled pair as
    per-rank [5e-4, 0.5]: the high-rank half froze and the reduced solve
    on its orphaned partner saw a zero one-dimensional Hessian where the
    only curvature was joint. The normalized rows are judged against the
    strongest estimated direction, anchored for all-weak models (see the
    constants' calibration): per-dimension curvature is O(1) in both the
    row count and the rank, where an absolute, objective-scaled, or
    raw-relative bar is not. Fixed lambdas are always frozen.
    """
    gradient = np.abs(np.asarray(projected_gradient, dtype=np.float64))
    hess = np.abs(np.asarray(hessian, dtype=np.float64))
    ranks = np.maximum(np.asarray(penalty_ranks, dtype=np.float64), 1.0)
    estimated = np.asarray(estimated_mask, dtype=bool)

    gradient_bar = _freeze_gradient_bar(objective, tolerance)
    if estimated.any():
        row_curvature = hess[:, estimated].max(axis=1)
        scale = np.sqrt(ranks)
        symmetric = hess / np.outer(scale, scale)
        normalized = symmetric[:, estimated].max(axis=1)
        curvature_anchor = float(normalized[estimated].max())
    else:
        row_curvature = np.zeros(len(gradient))
        normalized = row_curvature
        curvature_anchor = 0.0
    curvature_bar = FLAT_DIRECTION_CURVATURE_REL * max(
        curvature_anchor, FLAT_DIRECTION_CURVATURE_ANCHOR
    )
    frozen = ~estimated | ((gradient < gradient_bar) & (normalized < curvature_bar))
    return FlatDirectionDecision(
        frozen=frozen,
        row_curvature=row_curvature,
        penalty_rank=ranks,
        normalized_curvature=normalized,
        curvature_bar=float(curvature_bar),
    )


def mask_frozen_stop_gradient(
    projected_gradient: NDArray,
    previously_frozen: NDArray | None,
    *,
    objective: float,
    tolerance: float,
) -> NDArray:
    """Zero previously-frozen directions that are still flat this iteration.

    The freeze decision needs the Hessian, so the stop criterion for
    iteration k judges against iteration k-1's mask. A direction whose
    gradient has since grown past the freeze bar has re-activated and must
    not be hidden by that stale mask; the gradient arm is available before
    the Hessian, so intersect with it.
    """
    if previously_frozen is None:
        return projected_gradient
    projected = np.asarray(projected_gradient, dtype=np.float64)
    still_flat = np.abs(projected) < _freeze_gradient_bar(objective, tolerance)
    return np.where(np.asarray(previously_frozen, dtype=bool) & still_flat, 0.0, projected)


def trial_counts_as_precision_evidence(converged: bool, objective: float) -> bool:
    """Whether a rejected line-search trial is evidence for the precision exit.

    On the Fisher path an exhausted-PIRLS trial is still scored; its
    objective sits at a non-stationary beta, so an Armijo rejection of it
    proves nothing about the true profile objective. Only a STATIONARY
    trial with a finite objective counts. On the Fisher path the
    step-length flag is the only available evidence of that; under observed
    geometry the caller passes the KKT certificate's verdict instead, which
    is strictly stronger and, unlike the flag, can still be reached when the
    step tolerance sits below the iteration's round-off floor.
    """
    return bool(converged) and bool(np.isfinite(objective))


def classify_dead_feasible_exit(
    active_gradient_norm: float,
    *,
    objective: float,
    tolerance: float,
    evaluated_trial: bool = True,
) -> str:
    """Classify a line search whose every feasible trial was rejected.

    The optimum is resolved when the current active set's gradient is
    below the precision actually asked for: the resolved tolerance, never
    tighter than the achievable-precision floor. Holding a loose-tolerance
    fit to the floor misreported a resolved optimum as line_search_failed.

    The proof requires evidence: at least one trial whose objective was
    actually evaluated and rejected. On observed-geometry paths every
    trial can be skipped (PIRLS non-convergence, geometry failure, an
    uncertified mode) without any objective computed -- a search that
    evaluated nothing has shown nothing, and stays an honest failure.
    """
    if not evaluated_trial:
        return "line_search_failed"
    bar = max(FLAT_DIRECTION_FREEZE_FLOOR, tolerance) * _score_scale(objective)
    if active_gradient_norm < bar:
        return "converged_at_precision"
    return "line_search_failed"


@dataclass(frozen=True)
class REMLCandidateConvergence:
    """Diagnostics for one fully evaluated REML lambda candidate."""

    projected_gradient_norm: float
    score_scale: float
    objective_change: float
    converged: bool


def project_reml_gradient(
    gradient: NDArray,
    rho: NDArray,
    estimated_mask: NDArray,
    *,
    log_lower: float | NDArray,
    log_upper: float | NDArray,
    bound_window: float = 0.01,
) -> NDArray:
    """Project fixed and outward-pointing bound scores to zero."""
    score = np.asarray(gradient, dtype=np.float64)
    log_lambda = np.asarray(rho, dtype=np.float64)
    estimated = np.asarray(estimated_mask, dtype=bool)
    if score.shape != log_lambda.shape or score.shape != estimated.shape:
        raise ValueError("gradient, rho, and estimated_mask must have identical shapes")

    projected = score.copy()
    fixed = ~estimated
    upper_stationary = estimated & (log_lambda >= log_upper - bound_window) & (score < 0.0)
    lower_stationary = estimated & (log_lambda <= log_lower + bound_window) & (score > 0.0)
    projected[fixed | upper_stationary | lower_stationary] = 0.0
    return projected


def evaluate_reml_candidate(
    *,
    iteration: int,
    objective: float,
    previous_objective: float,
    projected_gradient: NDArray,
    tolerance: float,
) -> REMLCandidateConvergence:
    """Evaluate Wood's compound score/objective stopping criterion."""
    projected = np.asarray(projected_gradient, dtype=np.float64)
    gradient_norm = float(np.max(np.abs(projected))) if projected.size else 0.0
    score_scale = _score_scale(objective)
    objective_change = abs(objective - previous_objective) if iteration > 0 else np.inf
    converged = (
        iteration >= 1
        and gradient_norm < tolerance * score_scale
        and objective_change < tolerance * score_scale
    )
    return REMLCandidateConvergence(
        projected_gradient_norm=gradient_norm,
        score_scale=score_scale,
        objective_change=objective_change,
        converged=converged,
    )
