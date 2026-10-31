"""Pure convergence decisions shared by exact and discrete REML optimizers."""

from __future__ import annotations

from collections.abc import Callable
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

# The gradient ceiling of the dead-search decrement arm, in multiples of the
# gradient bar max(1e-7, reml_tol) * (1 + |objective|). The decrement is
# computed from the outer Hessian the step used, which by default
# (w_correction_order=1) is a modified-Newton approximation: if it overstates
# the curvature by a factor K the decrement understates the remaining
# decrease by K, and no bound on K is available. The ceiling makes the arm's
# guarantee independent of K. Along a direction of true curvature h a fit the
# arm grants has true remaining decrease g^2 / (2h) < (10 bar)^2 / (2h): at
# most 10^2 times the most the gradient arm already accepts at that h,
# whatever K is. It also keeps the stall invariant the dead-search exit relies
# on: a granted fit's active gradient is within one decade of the bar, so an
# undetermined stall, orders above it, stays a failure. Measured: #459's
# panel endgames sit at 1.009 to 2.03 times the bar (margin 4.9); every
# genuine dead-search stall in the focused REML suites at 2.8e4 times it or
# more; and at the panel's optimum the order-1 Hessian is within 0.42% of a
# finite-difference Hessian of the exact gradient (order 2 within 2e-6).
DECREMENT_GRADIENT_CEILING = 10.0


# The curvature-to-gradient ratio band of a plateau shoulder
# (``concave_live_directions``).
SHOULDER_CURVATURE_RATIO_LOW = 0.5
SHOULDER_CURVATURE_RATIO_HIGH = 2.0


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
    # Concave-shoulder directions the flat test alone would have held
    # (``concave_live_directions``): left active, and stepped to the step
    # bound by the engines rather than by a flipped-curvature Newton step.
    shoulder: NDArray | None = None


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


def concave_live_directions(
    projected_gradient: NDArray,
    hessian: NDArray,
    estimated_mask: NDArray,
    *,
    objective: float,
    tolerance: float,
    step_bound: float,
) -> NDArray:
    """Estimated directions with negative curvature and a decrease left to take.

    A direction whose diagonal curvature is negative fails the second-order
    necessary condition for a minimum (More & Sorensen 1979): the quadratic
    model decreases along it without bound, so a trust-region step lies on
    the region's boundary.  Over a step of ``step_bound`` the model
    promises ``step_bound * |g_i| + step_bound**2 * |H_ii| / 2``; when that
    reaches the stopping resolution ``tolerance * (1 + |objective|)`` the
    point is not a minimum to the precision asked for, whatever the
    gradient alone says.  A projected gradient of exactly zero is a bound
    holding the direction (``project_reml_gradient``) or exact
    stationarity, and is left to the first-order test.

    Only the shoulder's own signature is read, its sign included.  A flat
    direction sits at the lambda -> infinity end (towards lambda -> 0 an
    identified term's log-penalty slope, -r/2, is not flat), where the
    criterion expands in powers of 1/lambda, ``V_inf + c exp(-j rho) + ...``
    with j >= 1 the leading power (the penalized part of a term shrinks
    like 1/lambda), so ``g_i = -j c exp(-j rho)`` and ``H_ii = -j g_i``.  A
    shoulder approaches the plateau from below (c < 0): ``g_i > 0`` and
    ``H_ii < 0``, and descent leads away from the plateau, into the
    interior.  A direction with ``g_i < 0`` decreases into the plateau, the
    projected-Newton hold applies (Bertsekas 1982), and its true curvature
    is positive: a negative diagonal read there is an error in that entry,
    not a shoulder, at any ratio.  (Measured on a spline-by-factor fit with
    aliased levels: every negative diagonal read on its frozen ``x``
    direction, -7.3e-5 to -1.4e-4, came with g = -2.7e-4 to -3.3e-4, while
    differences of the criterion along ``x`` gave curvature +9.4e-4 against
    slope -9.4e-4, j = 1.  Read inside the band once on native Windows, it
    stepped ``x`` from 7e6 to 1.3e9, where the search ended
    line_search_failed; default OpenBLAS kernels read it there on another
    variant.)  The band brackets j: measured k = 0.66 .. 1.72 on freMTPL2
    fits started at lambda = 1e4.  A negative diagonal hundreds of times the
    gradient is the roughness the rank decisions of a strongly penalized
    system can give the criterion near the lambda cap (measured -0.0025 ..
    -0.016 against a gradient of 2e-5 on a spline-by-factor fit with aliased
    levels, whose steps from there ended line_search_failed on OpenBLAS
    Haswell and Zen kernels), and stays with the first-order hold.  The lower
    edge has no derivation: off the asymptote k need not be the integer j,
    and 0.5 only brackets the measured k from below as 2 does from above.  A
    shoulder with k under 0.5 keeps the first-order hold, so a start on one
    can still stop there.
    """
    signed_gradient = np.asarray(projected_gradient, dtype=np.float64)
    gradient = np.abs(signed_gradient)
    diagonal = np.diagonal(np.asarray(hessian, dtype=np.float64))
    model_decrease = step_bound * gradient + 0.5 * step_bound**2 * np.abs(diagonal)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(signed_gradient > 0.0, -diagonal / signed_gradient, 0.0)
    return (
        np.asarray(estimated_mask, dtype=bool)
        & (signed_gradient > 0.0)
        & (diagonal < 0.0)
        & (ratio >= SHOULDER_CURVATURE_RATIO_LOW)
        & (ratio <= SHOULDER_CURVATURE_RATIO_HIGH)
        & (model_decrease >= tolerance * _score_scale(objective))
    )


def freeze_flat_directions(
    projected_gradient: NDArray,
    hessian: NDArray,
    penalty_ranks: NDArray,
    estimated_mask: NDArray,
    *,
    objective: float,
    tolerance: float,
    step_bound: float | None = None,
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

    Holding a direction is Wood's (2008, sec. 3.4) "working infinity":
    a smoothing parameter on a flat end of the criterion is dropped from
    the Newton step.  That is a projected-Newton hold at a virtual bound,
    and it is valid only where the objective decreases INTO that bound
    (Bertsekas 1982: a coordinate is held only when its gradient pushes it
    against the bound).  On a convex tail -- ``H_ii > 0``, whichever end
    -- descent leads to the end the direction is moving toward.  A
    direction with ``H_ii < 0`` is on a concave shoulder: descent leads
    away from the plateau, back into the interior, and negative curvature
    is itself a descent direction (More & Sorensen 1979), so it is never
    held while its trust-region model still promises a decrease the
    stopping rule can resolve (``concave_live_directions``).  Such a
    direction the flat test alone would have held is reported in
    ``shoulder``; a caller without a trust-region step (``step_bound``
    None) keeps the first-order hold.  (Measured on freMTPL2 as published,
    before the claim recount, with the mains and 10 pairs: a start at
    lambda = 1e4 held VehAge x VehBrand[B4] at 1e4 against an optimum near
    7.8, gradient 1.8e-3 and curvature -1.9e-3, and reported a fit 8.8 worse
    as converged.  On the recounted book with 4 gated pairs the same start
    reported 1.89 worse.)
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
    flat = estimated & (gradient < gradient_bar) & (normalized < curvature_bar)
    shoulder = (
        np.zeros_like(flat)
        if step_bound is None
        else flat
        & concave_live_directions(
            projected_gradient,
            hessian,
            estimated,
            objective=objective,
            tolerance=tolerance,
            step_bound=step_bound,
        )
    )
    return FlatDirectionDecision(
        frozen=~estimated | (flat & ~shoulder),
        row_curvature=row_curvature,
        penalty_rank=ranks,
        normalized_curvature=normalized,
        curvature_bar=float(curvature_bar),
        shoulder=shoulder,
    )


def freeze_record(
    names: list[str],
    projected_gradient: NDArray,
    hessian: NDArray,
    decision: FlatDirectionDecision,
    *,
    score_scale: float,
    estimated_mask: NDArray,
    frozen: NDArray | None = None,
) -> dict:
    """The public record of one freeze decision (``profile["reml_freeze_decision"]``).

    It keeps the per-direction quantities the decision judged, the
    calibration evidence for its bars.  The engines overwrite it at every
    decision, so what survives is the last decision made: on a tolerance
    exit that is iteration k-1's, because the freeze runs after the stop
    criterion that ended iteration k, unless a stop revalidation decided
    later (``revalidate_stop``).  ``frozen`` is the mask the engine steps on
    when it is not the decision's own.
    """
    return {
        "names": list(names),
        "proj_grad": [float(abs(v)) for v in projected_gradient],
        "hess_diag": [float(hessian[i, i]) for i in range(len(names))],
        "row_curvature": [float(v) for v in decision.row_curvature],
        "penalty_rank": [float(v) for v in decision.penalty_rank],
        "normalized_curvature": [float(v) for v in decision.normalized_curvature],
        "curvature_bar": float(decision.curvature_bar),
        "score_scale": float(score_scale),
        "estimated": [bool(v) for v in estimated_mask],
        "frozen": [bool(v) for v in (decision.frozen if frozen is None else frozen)],
        "shoulder": [bool(v) for v in np.asarray(decision.shoulder, dtype=bool)],
        # The shoulder arm reads the sign (``concave_live_directions``).
        "proj_grad_signed": [float(v) for v in projected_gradient],
    }


def fixed_freeze_record(
    names: list[str], penalty_rank: NDArray, *, objective: float, estimated_mask: NDArray
) -> dict:
    """The freeze record of a fit whose every lambda is fixed.

    Fixed lambdas freeze definitionally -- the projection zeroes their
    scores -- so the public record exists for this pre-Newton exit too.  The
    zero derivatives are the projection's definition, not measurements.
    """
    m = len(names)
    return {
        "names": list(names),
        "proj_grad": [0.0] * m,
        "hess_diag": [0.0] * m,
        "row_curvature": [0.0] * m,
        "penalty_rank": [float(v) for v in penalty_rank],
        "normalized_curvature": [0.0] * m,
        "curvature_bar": 0.0,
        "score_scale": _score_scale(objective),
        "estimated": [bool(v) for v in estimated_mask],
        "frozen": [True] * m,
        "shoulder": [False] * m,
        "proj_grad_signed": [0.0] * m,
    }


def revalidate_stop(
    projected_gradient: NDArray,
    hessian: NDArray,
    previously_frozen: NDArray | None,
    estimated_mask: NDArray,
    *,
    objective: float,
    tolerance: float,
    score_scale: float,
    step_bound: float,
    names: list[str],
    iteration: int,
    profile: dict | None,
    concave_live: Callable[..., NDArray],
    freeze: Callable[..., FlatDirectionDecision],
    ranks: Callable[[], NDArray],
) -> tuple[bool, bool]:
    """Whether a stop the first-order test accepted stands, and whether a shoulder refused it.

    The compound stop test is first order.  A point it accepts must also
    pass the second-order necessary condition to the precision asked for
    (``concave_live_directions``): on a concave shoulder of a plateau the
    gradient is small however far the optimum is.  Its stop mask is the
    previous iteration's freeze decision intersected with the current
    gradient arm, which cannot see a curvature-only re-activation: a masked
    direction can hold a gradient below the freeze floor yet above the
    tolerance while its coupled partner's update raised its row curvature
    past the bar.  So when a shoulder refuses, or a masked direction's
    gradient reaches the tolerance, the freeze decision is recomputed
    against the current Hessian, and the stop stands only if that
    reactivates nothing.  The revalidation is then the last freeze decision
    made, and the published record describes it.

    ``concave_live`` and ``freeze`` are the engine's own bindings of
    ``concave_live_directions`` and ``freeze_flat_directions``; ``ranks``
    resolves its penalty ranks from its current penalty context
    (``direction_penalty_ranks``), only when the freeze decision is remade.
    """
    gradient_live = np.abs(projected_gradient) >= tolerance * score_scale
    masked_live = previously_frozen is not None and bool(
        np.any(np.asarray(previously_frozen) & gradient_live)
    )
    live = concave_live(
        projected_gradient,
        hessian,
        estimated_mask,
        objective=objective,
        tolerance=tolerance,
        step_bound=step_bound,
    )
    refused = bool(np.any(live))
    if refused and profile is not None:
        # The score test passed but the second-order check did not: record
        # which shoulders refused the stop (score-test exit only).
        idx = np.flatnonzero(live)
        profile["reml_concave_stop_refusal"] = {
            "iter": int(iteration),
            "names": [names[i] for i in idx],
            "gradient": [float(abs(projected_gradient[i])) for i in idx],
            "curvature": [float(hessian[i, i]) for i in idx],
            "step_bound": float(step_bound),
        }
    if not masked_live and not refused:
        return True, refused
    revalidation = freeze(
        projected_gradient,
        hessian,
        ranks(),
        estimated_mask,
        objective=objective,
        tolerance=tolerance,
        step_bound=step_bound,
    )
    if profile is not None:
        profile["reml_freeze_revalidated"] = True
        profile["reml_freeze_decision"] = freeze_record(
            names,
            projected_gradient,
            hessian,
            revalidation,
            score_scale=score_scale,
            estimated_mask=estimated_mask,
        )
    held = (
        np.asarray(previously_frozen)
        if previously_frozen is not None
        else np.zeros(len(names), dtype=bool)
    )
    reactivated = live | (~np.asarray(revalidation.frozen) & held & gradient_live)
    return not bool(np.any(reactivated)), refused


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


def newton_predicted_decrease(
    gradient: NDArray,
    eigenvalues: NDArray,
    eigenvectors: NDArray,
    *,
    eigenvalue_floor: float,
) -> float | None:
    """Half the squared Newton decrement, ``g' H^-1 g / 2``, or None.

    The decrease the quadratic model predicts from its minimizer, the
    estimate of ``f(x) - p*`` that Newton's method stops on (Boyd &
    Vandenberghe 2004, Convex Optimization, sec. 9.5.1 and Algorithm 9.5).
    It is defined only for a positive definite Hessian, judged on the
    caller's own modified-Newton floor: when every eigenvalue is at or above
    it the modification left ``H`` unchanged, so this is the decrease of the
    step the line search actually tried, before any step cap. Otherwise the
    model has no minimizer and there is nothing to predict: None. Eigenvalues
    that clear the floor (relative ``eps**0.7``) lie far above the
    eigensolver's backward error, of order ``m u ||H||``, so their
    perturbation moves the result by a relative amount far below the
    decisions it informs.
    """
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.size == 0 or not bool(np.all(values >= eigenvalue_floor)):
        return None
    coordinates = np.asarray(eigenvectors, dtype=np.float64).T @ np.asarray(
        gradient, dtype=np.float64
    )
    return 0.5 * float(np.sum(coordinates * coordinates / values))


def classify_dead_feasible_exit(
    active_gradient_norm: float,
    *,
    objective: float,
    tolerance: float,
    evaluated_trial: bool = True,
    predicted_decrease: float | None = None,
) -> str:
    """Classify a line search whose every feasible trial was rejected.

    The optimum is resolved when either of two measures of what is left
    falls below the precision asked for:

    - the current active set's gradient is below the resolved tolerance,
      never tighter than the achievable-precision floor, times the
      objective's scale. Holding a loose-tolerance fit to the floor
      misreported a resolved optimum as line_search_failed.
    - the decrease the active set's Newton model still predicts,
      ``predicted_decrease`` (half the squared Newton decrement, from
      :func:`newton_predicted_decrease`; None when the active Hessian is
      not positive definite or the caller's step cap or trust region bound
      the Newton step), is below ``tolerance * (1 + |objective|)``: the
      resolution at which the compound stop criterion already treats an
      objective change as no change. This is Newton's decrement stopping
      test (Boyd & Vandenberghe 2004, sec. 9.5.1). It catches a gradient
      just above the bar along a strongly curved direction, where the
      decrease left is quadratically small and below the objective's
      evaluation noise, so rounding decides whether a trial is accepted:
      a line search is expected to fail there (Shi, Xie, Byrd & Nocedal
      2022, SIAM J. Optim. 32(1), sec. 4.1; Berahas, Byrd & Nocedal 2019,
      SIAM J. Optim. 29(2)). Measured on #459's panel model: active
      gradient 1.009 to 2.03 times the bar, predicted decrease 1.4e-3 to
      5.8e-3 of the resolution, objective noise at fixed lambda 4.2e-7.
      This arm applies only with the active gradient under
      ``DECREMENT_GRADIENT_CEILING`` (10) times the gradient bar, so an
      error in a modified-Newton Hessian cannot grant a fit whose gradient
      is orders above the bar (see the constant). It does not also need
      the exact Hessian (``w_correction_order=2``): the ceiling's bound
      holds whatever the Hessian's error, and the default order-1 fits are
      the ones #459 measured.

    The proof requires evidence: at least one trial whose objective was
    actually evaluated and rejected. On observed-geometry paths every
    trial can be skipped (PIRLS non-convergence, geometry failure, an
    uncertified mode) without any objective computed -- a search that
    evaluated nothing has shown nothing, and stays an honest failure.
    """
    if not evaluated_trial:
        return "line_search_failed"
    score_scale = _score_scale(objective)
    bar = max(FLAT_DIRECTION_FREEZE_FLOOR, tolerance) * score_scale
    if active_gradient_norm < bar:
        return "converged_at_precision"
    if (
        predicted_decrease is not None
        and active_gradient_norm < DECREMENT_GRADIENT_CEILING * bar
        and predicted_decrease < tolerance * score_scale
    ):
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
