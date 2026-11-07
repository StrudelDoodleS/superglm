"""Pure convergence decisions shared by exact and discrete REML optimizers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import NEWTON_EIGENVALUE_FLOOR

# The freeze bar never drops below this floor, whatever reml_tol: freezing
# classifies geometry (is a direction informative at all), and a flat
# direction stays flat however tightly the informative lambdas are located.
FLAT_DIRECTION_FREEZE_FLOOR = 1e-7

# The curvature arm of the freeze decision, per penalty dimension and
# relative to the strongest estimated direction. score_scale grows with the
# row count while log-lambda curvature saturates, and raw row curvature
# scales with penalty rank, so neither an objective-scaled nor a raw bar
# classifies. The anchor keeps the bar meaningful when every direction is
# weak: 1e-2 * 0.1 = 1e-3 per dimension separates null from informative.
FLAT_DIRECTION_CURVATURE_REL = 1e-2
FLAT_DIRECTION_CURVATURE_ANCHOR = 0.1

# Ceiling of the dead-search decrement arm, in multiples of the gradient bar
# max(1e-7, reml_tol) * (1 + |objective|). The decrement comes from a
# modified-Newton Hessian of unknown error, so the arm is granted only when
# the active gradient is within this factor of the bar: the true remaining
# decrease is then at most 10^2 times what the gradient arm accepts, and an
# undetermined stall, orders above the bar, stays a failure.
DECREMENT_GRADIENT_CEILING = 10.0


# The curvature-to-gradient ratio band of a plateau shoulder
# (``concave_live_directions``).
SHOULDER_CURVATURE_RATIO_LOW = 0.5
SHOULDER_CURVATURE_RATIO_HIGH = 2.0

NEWTON_STEP_STOP_BAR = 1e-3
NEWTON_STEP_STOP_PER_TOLERANCE = 1e6


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

    Negative curvature fails the second-order necessary condition (More &
    Sorensen 1979). Over a step of ``step_bound`` the model promises
    ``step_bound * |g_i| + step_bound**2 * |H_ii| / 2``; when that reaches
    ``tolerance * (1 + |objective|)`` the point is not a minimum to the
    precision asked for. A projected gradient of exactly zero is left to the
    first-order test.

    Only a shoulder's signature counts. Toward lambda -> infinity the
    criterion expands as ``V_inf + c exp(-j rho)``, so ``H_ii = -j g_i``; a
    shoulder has ``g_i > 0`` and ``H_ii < 0``, and ``H_ii / g_i`` must lie in
    the band ``[-2, -0.5]`` bracketing ``j``. A direction with ``g_i < 0``
    decreases into the plateau and keeps the projected-Newton hold
    (Bertsekas 1982), as does a negative diagonal far outside the band, which
    is rank-decision roughness near the lambda cap. A shoulder with ratio
    below 0.5 keeps the first-order hold.
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

    The gradient arm is judged against the objective's scale. The curvature
    arm judges each direction's row of the estimated Hessian block, not its
    diagonal alone (REML Hessians carry cross-terms and need not be PD),
    excluding coupling to fixed directions, normalized symmetrically per
    penalty dimension (``D^{-1/2} H D^{-1/2}``, ``D = diag(ranks)``) and judged
    against the strongest estimated direction with an anchor for all-weak
    models. Fixed lambdas are always frozen.

    Holding a direction is Wood's (2008, sec. 3.4) "working infinity", a
    projected-Newton hold valid only where the objective decreases into the
    bound (Bertsekas 1982). A direction on a concave shoulder is never held
    while its trust-region model promises a resolvable decrease
    (``concave_live_directions``, More & Sorensen 1979); such directions are
    reported in ``shoulder``. Without ``step_bound`` the first-order hold
    stays.
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

    An accepted point must also pass the second-order condition
    (``concave_live_directions``). The stop mask is the previous freeze
    intersected with the current gradient arm, which misses curvature-only
    re-activation, so when a shoulder refuses or a masked direction's gradient
    reaches the tolerance the freeze is recomputed against the current
    Hessian, and the stop stands only if nothing reactivates. That
    revalidation is then the published freeze decision.

    ``concave_live`` and ``freeze`` are the engine's bindings of those
    functions; ``ranks`` resolves penalty ranks only when the freeze is remade.
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


def newton_step_stop_bar(tolerance: float) -> float:
    """The largest Newton step in log lambda a stop accepts at this ``reml_tol``.

    At fixed working weights a step of ``d`` keeps ``H`` within ``e**-d H`` and
    ``e**d H`` in the Loewner order, so it moves a standard error by at most
    ``d / 2`` in log. The default bar therefore bounds that move by 0.05%. A
    caller's tighter tolerance tightens the bar in proportion.
    """
    return min(NEWTON_STEP_STOP_BAR, NEWTON_STEP_STOP_PER_TOLERANCE * float(tolerance))


def newton_step_left(hessian: NDArray, gradient: NDArray) -> float:
    """The largest entry of Newton's step on these directions, on an unmodified Hessian.

    The step is curvature-scaled, as the Newton decrement is (Boyd & Vandenberghe
    2004, sec. 9.5.1), so unlike a gradient bar scaled by the objective it does
    not depend on the objective's size. A Hessian that is not positive definite
    above the modified-Newton floor gives no such step, which reads as infinite.
    """
    if not np.all(np.isfinite(hessian)):
        return math.inf
    values, vectors = np.linalg.eigh(hessian)
    floor = max(float(np.max(np.abs(values))), 1e-12) * NEWTON_EIGENVALUE_FLOOR
    if not np.all(values >= floor):
        return math.inf
    step = vectors @ ((vectors.T @ gradient) / values)
    return float(np.max(np.abs(step)))


def newton_step_same_terms(previous: NDArray | None, current: NDArray) -> bool:
    """Whether the preceding step was read on exactly these free directions."""
    return previous is not None and np.array_equal(previous, current)


def newton_step_settled(step_left: float, taken_step: float, *, bar: float) -> bool:
    """Whether the distance left after a taken step is at most ``bar``.

    A Hessian that omits the second-derivative weight terms contracts linearly,
    at a rate ``r`` the two steps read for free. Steps that keep one sign then
    leave a distance of about ``step_left / (1 - r)``, and a step that did not
    contract says nothing about the distance.
    """
    if step_left >= taken_step:
        return False
    return step_left / (1.0 - step_left / taken_step) <= bar


def newton_step_resolved(hessian: NDArray, gradient: NDArray, *, bar: float) -> bool:
    """Whether Newton's step on these directions is at most ``bar``."""
    return newton_step_left(hessian, gradient) <= bar


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

    Newton's predicted decrease (Boyd & Vandenberghe 2004, sec. 9.5.1),
    defined only when every eigenvalue is at or above the caller's
    modified-Newton floor (relative ``eps**0.7``, far above the eigensolver's
    ``m u ||H||`` backward error), so ``H`` was unmodified. Otherwise None.
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

    The optimum is resolved when either measure of what is left is below the
    precision asked for:

    - the active gradient is below the resolved tolerance (never tighter than
      the achievable-precision floor) times the objective's scale;
    - ``predicted_decrease`` (:func:`newton_predicted_decrease`; None when
      the Hessian is not PD or a step cap bound the step) is below
      ``tolerance * (1 + |objective|)``: Newton's decrement test (Boyd &
      Vandenberghe 2004, sec. 9.5.1). There the decrease left is below the
      objective's evaluation noise and a line search is expected to fail
      (Shi, Xie, Byrd & Nocedal 2022, SIAM J. Optim. 32(1), sec. 4.1; Berahas,
      Byrd & Nocedal 2019, SIAM J. Optim. 29(2)). Only with the active
      gradient under ``DECREMENT_GRADIENT_CEILING`` times the bar, which
      bounds the result whatever the Hessian's error.

    At least one trial must have been evaluated and rejected; a search that
    evaluated nothing stays a failure.
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
