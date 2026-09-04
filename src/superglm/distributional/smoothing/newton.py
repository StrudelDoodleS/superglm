"""Newton endgame primitives for the smoothing loop.

Damped Newton on the exact LAML gradient and Hessian in log lambda: the step
(Wood, Pya and Säfken 2016 §3.1.2 diagonal preconditioning, a growing ridge
until the active Hessian is positive definite, the scalar path's proportional
cap), the damped-BFGS fallback (Nocedal and Wright 2006 §18.3, Powell damping
on the inverse update), the hand-off rule from the Fellner–Schall warm-up, and
the bracketed root search in ``tau = 1/lambda`` beyond the cap (Brent's method,
``scipy.optimize.brentq``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve
from scipy.optimize import brentq

from superglm.distributional.family import DistributionalFamily, FamilyLikelihoodPlan
from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSIteration,
    _assessment_is_numerically_stationary,
)
from superglm.distributional.smoothing.authority import (
    _face_authority_config,
    _fit_endpoint_authority_stationary,
    _fit_fixed_state,
)
from superglm.distributional.smoothing.derivatives import (
    LamlDerivativeError,
    LamlDerivatives,
    LamlDerivativeWorkspace,
    laml_derivatives,
)
from superglm.distributional.smoothing.evidence import _fresh_raw_evidence, _FreshRawEvidence
from superglm.distributional.smoothing.objective import _complete_mapping, _laplace_objective
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.solver.chunks import ChunkSize
from superglm.distributional.solver.solver import _DenseObservedReuseSession
from superglm.distributional.timing import FitPhaseRecorder, measure_phase
from superglm.reml.convergence import (
    evaluate_reml_candidate,
    freeze_flat_directions,
    project_reml_gradient,
)

#: EFS stop reasons after which the endgame takes over under ``outer="efs+newton"``.
HANDOFF_REASONS = frozenset(
    {
        "lambda_change",
        "objective_plateau",
        "practical_plateau",
        "objective_rejected",
        "max_iterations",
        "lambda_cap_unresolved",
    }
)
#: EFS stop reasons the endgame cannot improve on: nothing to estimate, or no
#: converged coefficient fit to differentiate at.
NO_HANDOFF_REASONS = frozenset(
    {"fixed_only", "coefficient_not_converged", "endpoint_revalidation_failed"}
)
INITIAL_RIDGE = 1.0e-8
BRACKET_WIDTH = 1.0e-3
BRACKET_ROOT = 1.0e-10


def _finite_vector(values: NDArray, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector")
    return array


def _finite_matrix(values: NDArray, *, name: str, size: int) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}-by-{size} matrix")
    return 0.5 * (array + array.T)


def _active_mask(active: NDArray, *, size: int) -> NDArray[np.bool_]:
    mask = np.asarray(active, dtype=bool)
    if mask.shape != (size,):
        raise ValueError("active must be a boolean mask with one entry per component")
    return mask


def _validated_cap(max_log_step: float) -> float:
    if (
        isinstance(max_log_step, bool)
        or not isinstance(max_log_step, int | float)
        or not math.isfinite(max_log_step)
        or max_log_step <= 0.0
    ):
        raise ValueError("max_log_step must be a finite positive float")
    return float(max_log_step)


def _capped(step: NDArray[np.float64], max_log_step: float) -> NDArray[np.float64]:
    """Proportional cap on the whole vector, as ``reml/direct.py`` applies it."""
    largest = float(np.max(np.abs(step), initial=0.0))
    if largest > max_log_step:
        step = step * (max_log_step / largest)
    return step


def newton_direction(
    gradient: NDArray,
    hessian: NDArray,
    active: NDArray,
    *,
    max_log_step: float,
    ridge_growth: float = 10.0,
    max_ridge_attempts: int = 20,
) -> tuple[NDArray[np.float64], float]:
    """Return the capped Newton step on the active set and the ridge used.

    The active block is scaled by ``D = diag(|h_ii|^-1/2)`` (WPS16 §3.1.2),
    factored by Cholesky with a ridge ``0, 1e-8, 1e-7, ...`` until positive
    definite, solved, rescaled and capped proportionally; inactive components
    get a zero step.  The ridge is ``0.0`` when the Hessian was already
    positive definite.
    """
    g = _finite_vector(gradient, name="gradient")
    size = g.size
    h = _finite_matrix(hessian, name="hessian", size=size)
    mask = _active_mask(active, size=size)
    cap = _validated_cap(max_log_step)
    if (
        isinstance(ridge_growth, bool)
        or not isinstance(ridge_growth, int | float)
        or not math.isfinite(ridge_growth)
        or ridge_growth <= 1.0
    ):
        raise ValueError("ridge_growth must be a finite float above one")
    if (
        isinstance(max_ridge_attempts, bool)
        or not isinstance(max_ridge_attempts, int)
        or max_ridge_attempts < 0
    ):
        raise ValueError("max_ridge_attempts must be a non-negative integer")
    step = np.zeros(size, dtype=np.float64)
    index = np.flatnonzero(mask)
    if index.size == 0:
        return step, 0.0
    sub_gradient = g[index]
    sub_hessian = h[np.ix_(index, index)]
    scale = 1.0 / np.sqrt(np.maximum(np.abs(np.diag(sub_hessian)), np.finfo(np.float64).tiny))
    scaled = (scale[:, None] * sub_hessian) * scale[None, :]
    identity = np.eye(index.size)
    ridge = 0.0
    factor: NDArray[np.float64] | None = None
    for _attempt in range(max_ridge_attempts + 1):
        try:
            factor = np.linalg.cholesky(scaled + ridge * identity)
        except np.linalg.LinAlgError:
            ridge = INITIAL_RIDGE if ridge == 0.0 else ridge * ridge_growth
            continue
        break
    if factor is None:
        raise np.linalg.LinAlgError("the active Hessian could not be made positive definite")
    solved = cho_solve((factor, True), scale * sub_gradient)
    direction = -(scale * solved)
    if not np.all(np.isfinite(direction)):
        raise np.linalg.LinAlgError("the Newton direction is not finite")
    step[index] = direction
    return _capped(step, cap), float(ridge)


def bfgs_direction(
    gradient: NDArray,
    inverse_hessian: NDArray,
    active: NDArray,
    *,
    max_log_step: float,
) -> NDArray[np.float64]:
    """Capped quasi-Newton step ``-B g`` on the active set from an inverse-Hessian ``B``."""
    g = _finite_vector(gradient, name="gradient")
    size = g.size
    inverse = _finite_matrix(inverse_hessian, name="inverse_hessian", size=size)
    mask = _active_mask(active, size=size)
    cap = _validated_cap(max_log_step)
    step = np.zeros(size, dtype=np.float64)
    index = np.flatnonzero(mask)
    if index.size == 0:
        return step
    step[index] = -(inverse[np.ix_(index, index)] @ g[index])
    return _capped(step, cap)


def bfgs_update(
    inverse_hessian: NDArray,
    delta_rho: NDArray,
    delta_gradient: NDArray,
) -> NDArray[np.float64]:
    """Damped BFGS inverse update (Nocedal and Wright 2006, §18.3).

    With ``B = H^-1`` the current inverse approximation, Powell damping replaces
    the gradient change ``y`` by ``r = theta y + (1 - theta) B s`` where
    ``theta = 1`` when ``s'y >= 0.2 s'Bs`` and ``0.8 s'Bs / (s'Bs - s'y)``
    otherwise, so ``s'r >= 0.2 s'Bs > 0`` and the update stays positive
    definite; the update is skipped when ``s'r <= 1e-8 s'Bs`` after damping
    or when ``s'Bs`` is not positive.
    """
    s = _finite_vector(delta_rho, name="delta_rho")
    y = _finite_vector(delta_gradient, name="delta_gradient")
    size = s.size
    if y.size != size:
        raise ValueError("delta_rho and delta_gradient must have the same length")
    inverse = _finite_matrix(inverse_hessian, name="inverse_hessian", size=size)
    try:
        b_s = np.linalg.solve(inverse, s)
    except np.linalg.LinAlgError:
        return inverse
    s_b_s = float(s @ b_s)
    if not math.isfinite(s_b_s) or s_b_s <= 0.0:
        return inverse
    s_y = float(s @ y)
    theta = 1.0 if s_y >= 0.2 * s_b_s else 0.8 * s_b_s / (s_b_s - s_y)
    r = theta * y + (1.0 - theta) * b_s
    r_s = float(r @ s)
    if not math.isfinite(r_s) or r_s <= 1.0e-8 * s_b_s:
        return inverse
    rho = 1.0 / r_s
    left = np.eye(size) - rho * np.outer(s, r)
    updated = left @ inverse @ left.T + rho * np.outer(s, s)
    updated = 0.5 * (updated + updated.T)
    if not np.all(np.isfinite(updated)):
        return inverse
    return updated


def should_hand_off(
    reason: str | None,
    *,
    max_accepted_step: float,
    iterations: int,
    config: DistributionalEFSConfig,
) -> bool:
    """Whether the EFS warm-up hands over to the Newton endgame now.

    ``reason`` is the EFS stop the loop is about to return with (``None`` at
    the tail of an accepted iteration).  Under ``outer="efs+newton"`` every
    stop in :data:`HANDOFF_REASONS` hands off; the stops in
    :data:`NO_HANDOFF_REASONS` never do; an accepted iteration hands off once
    its largest accepted ``|delta log lambda|`` is at most ``handoff_step`` or
    the warm-up has run ``handoff_iterations`` iterations.
    """
    if not isinstance(config, DistributionalEFSConfig):
        raise TypeError("config must be DistributionalEFSConfig")
    if config.outer != "efs+newton":
        return False
    if reason is not None:
        if reason in NO_HANDOFF_REASONS:
            return False
        if reason in HANDOFF_REASONS:
            return True
        raise ValueError(f"unknown EFS stop reason {reason!r}")
    if (
        isinstance(max_accepted_step, bool)
        or not isinstance(max_accepted_step, int | float)
        or not math.isfinite(max_accepted_step)
        or max_accepted_step < 0.0
    ):
        raise ValueError("max_accepted_step must be a finite non-negative float")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    return max_accepted_step <= config.handoff_step or iterations >= config.handoff_iterations


@dataclass(frozen=True)
class BracketOutcome:
    """The result of the root search beyond the cap in ``u = log(lambda / lambda_cap)``."""

    found: bool
    log_lambda_ratio: float | None
    evaluations: int
    bracket: tuple[float, float] | None

    def __post_init__(self) -> None:
        if not isinstance(self.found, bool):
            raise TypeError("found must be bool")
        if self.found != (self.log_lambda_ratio is not None):
            raise ValueError("a found root must carry its log lambda ratio, and only then")
        if self.log_lambda_ratio is not None and not (
            math.isfinite(self.log_lambda_ratio) and self.log_lambda_ratio >= 0.0
        ):
            raise ValueError("log_lambda_ratio must be finite and non-negative")
        if (
            isinstance(self.evaluations, bool)
            or not isinstance(self.evaluations, int)
            or self.evaluations < 0
        ):
            raise ValueError("evaluations must be a non-negative integer")
        if self.bracket is not None:
            low, high = self.bracket
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low <= high):
                raise ValueError("bracket must be an ordered finite pair in [0, log_span]")
            object.__setattr__(self, "bracket", (float(low), float(high)))


def _finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return float(value)


def bracket_beyond_cap(
    phi_at_cap: float,
    phi_at_endpoint: float,
    evaluate: Callable[[float], float],
    *,
    log_span: float,
    max_evaluations: int = 12,
) -> BracketOutcome:
    """Locate the root of ``phi(u) = dF/dtau`` for ``u = log(lambda/lambda_cap)`` in ``[0, log_span]``.

    A search happens only when the derivative points outward at the cap
    (``phi_at_cap > 0``: ``F`` still falls as ``lambda`` grows) and inward at
    the endpoint (``phi_at_endpoint < 0``: the optimum is finite), so a root
    lies between the cap and infinity.  ``evaluate(u)`` returns ``dF/dtau`` at
    ``lambda_cap * exp(u)``; the cap value is reused, never re-evaluated.
    Brent's method on ``[0, log_span]`` stops when the bracket is narrower
    than 1e-3 or ``phi`` is below 1e-10 in magnitude; a root beyond
    ``log_span`` (``phi`` still positive there) or an exhausted budget is
    reported as not found with the bracket searched.
    """
    cap_value = _finite_scalar(phi_at_cap, name="phi_at_cap")
    endpoint_value = _finite_scalar(phi_at_endpoint, name="phi_at_endpoint")
    span = _finite_scalar(log_span, name="log_span")
    if span <= 0.0:
        raise ValueError("log_span must be positive")
    if (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 2
    ):
        raise ValueError("max_evaluations must be an integer of at least two")
    if not (cap_value > 0.0 > endpoint_value):
        return BracketOutcome(found=False, log_lambda_ratio=None, evaluations=0, bracket=None)
    evaluations = 0
    cache: dict[float, float] = {0.0: cap_value}

    def phi(u: float) -> float:
        nonlocal evaluations
        known = cache.get(float(u))
        if known is not None:
            return known
        evaluations += 1
        value = float(evaluate(float(u)))
        if not math.isfinite(value):
            raise ValueError("the bracket evaluation is not finite")
        cache[float(u)] = value
        return value

    # The far end of the span may not be fittable (a conditioning limit is
    # exactly where a coefficient fit stops being stationary): halve it, at
    # most twice, before giving up on the search.
    far_end = span
    far: float | None = None
    for _attempt in range(3):
        try:
            far = phi(far_end)
        except ValueError:
            far_end *= 0.5
            continue
        break
    if far is None:
        return BracketOutcome(
            found=False, log_lambda_ratio=None, evaluations=evaluations, bracket=(0.0, span)
        )
    if abs(far) < BRACKET_ROOT:
        return BracketOutcome(
            found=True,
            log_lambda_ratio=far_end,
            evaluations=evaluations,
            bracket=(far_end, far_end),
        )
    if far > 0.0:
        return BracketOutcome(
            found=False, log_lambda_ratio=None, evaluations=evaluations, bracket=(0.0, far_end)
        )
    try:
        root, results = brentq(
            phi,
            0.0,
            far_end,
            xtol=BRACKET_WIDTH,
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=max_evaluations,
            full_output=True,
            disp=False,
        )
    except (ValueError, RuntimeError):
        return BracketOutcome(
            found=False, log_lambda_ratio=None, evaluations=evaluations, bracket=(0.0, span)
        )
    if not bool(results.converged):
        return BracketOutcome(
            found=False, log_lambda_ratio=None, evaluations=evaluations, bracket=(0.0, span)
        )
    location = min(max(float(root), 0.0), far_end)
    return BracketOutcome(
        found=True,
        log_lambda_ratio=location,
        evaluations=evaluations,
        bracket=(max(location - BRACKET_WIDTH, 0.0), min(location + BRACKET_WIDTH, far_end)),
    )


# --------------------------------------------------------------------------
# The endgame inside the smoothing loop
# --------------------------------------------------------------------------

EndgameKind = Literal[
    "stationary", "cap_pressure", "gradient_unresolved", "max_iterations", "objective_rejected"
]
ARMIJO_CONSTANT = 1.0e-4
#: Outer-loop derivative and bookkeeping work is booked under the EFS phase;
#: ``PHASE_NAMES`` is the closed telemetry vocabulary for these timings.
_ENDGAME_PHASE = "newton_endgame"


@dataclass(frozen=True)
class EndgameState:
    """The accepted smoothing state the endgame starts from or ends at."""

    lambdas: Mapping[str, float]
    fit: DenseSolverResult
    objective: float
    face: PenaltyFace | None
    evidence: _FreshRawEvidence
    terminal_fit_index: int


@dataclass(frozen=True)
class NewtonEndgameOutcome:
    """How one endgame run ended, with the state and derivatives it ended at."""

    kind: EndgameKind
    state: EndgameState
    derivatives: LamlDerivatives | None
    projected_gradient_norm: float | None
    cap_pressure: tuple[str, ...]
    newton_iterations: int
    bfgs_iterations: int


class _TrialFitter:
    """Warm-started, face-aware trial fits under the EFS loop's own policies."""

    def __init__(
        self,
        family: DistributionalFamily,
        layout: StackedLayout,
        y: NDArray,
        likelihood_plan: FamilyLikelihoodPlan,
        *,
        face: PenaltyFace | None,
        solver_config: DenseSolverConfig,
        chunk_size: ChunkSize | None,
        phase_recorder: FitPhaseRecorder | None,
        reuse_session: _DenseObservedReuseSession,
    ) -> None:
        self.family = family
        self.layout = layout
        self.y = y
        self.likelihood_plan = likelihood_plan
        self.face = face
        self.config = solver_config if face is None else _face_authority_config(solver_config)
        self.chunk_size = chunk_size
        self.phase_recorder = phase_recorder
        self.reuse_session = reuse_session

    @property
    def tolerance(self) -> float:
        return float(self.config.tolerance)

    def fit(
        self,
        lambdas: Mapping[str, float],
        *,
        initial: NDArray,
        reuse_source: DenseSolverResult | None,
    ) -> tuple[DenseSolverResult, float | None]:
        """Fit at ``lambdas``; the objective is ``None`` unless the fit is stationary."""
        if self.face is None:
            result = _fit_fixed_state(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                lambdas=lambdas,
                face=None,
                initial=initial,
                config=self.config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                _reuse_session=self.reuse_session,
                _reuse_source=reuse_source,
            )
        else:
            result = _fit_endpoint_authority_stationary(
                self.family,
                self.layout,
                self.y,
                self.likelihood_plan,
                lambdas=lambdas,
                face=self.face,
                initial=initial,
                config=self.config,
                chunk_size=self.chunk_size,
                phase_recorder=self.phase_recorder,
                _reuse_session=self.reuse_session,
            )
        stationary = result.converged and (
            self.face is None or _assessment_is_numerically_stationary(result, self.tolerance)
        )
        if not stationary:
            return result, None
        try:
            with measure_phase(self.phase_recorder, _ENDGAME_PHASE):
                objective = _laplace_objective(
                    result, layout=self.layout, lambdas=lambdas, face=self.face
                )
        except ValueError:
            return result, None
        return result, float(objective)


def _log_steps(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, float]:
    return {
        name: (
            0.0 if after[name] == before[name] else math.log(after[name]) - math.log(before[name])
        )
        for name in before
    }


def newton_iteration_record(
    *,
    iteration: int,
    state: EndgameState,
    proposed_lambdas: Mapping[str, float],
    lambdas_after: Mapping[str, float],
    objective_after: float,
    evidence: _FreshRawEvidence,
    fit_indices: tuple[int, ...],
    tolerances: tuple[float, ...],
    accepted_fit: DenseSolverResult | None,
    step_source: Literal["newton", "bfgs", "bracket"],
    derivatives: LamlDerivatives,
    projected_gradient_norm: float,
    hessian_certificate: float | None,
    ridge: float | None,
    estimated_names: tuple[str, ...],
) -> DistributionalEFSIteration:
    """One Newton-stage iteration in the smoothing history's own vocabulary."""
    update = evidence.update
    if update is None:
        raise RuntimeError("a Newton iteration requires a fresh EFS update")
    before = dict(state.lambdas)
    proposed = _complete_mapping(before, proposed_lambdas, missing=before)
    after = _complete_mapping(before, lambdas_after, missing=before)
    proposed_steps = _log_steps(before, proposed)
    accepted_steps = _log_steps(before, after)
    accepted = accepted_fit is not None
    objective_before = float(state.objective)
    objective_after = float(objective_after)
    return DistributionalEFSIteration(
        iteration=iteration,
        source_fit_index=state.terminal_fit_index,
        lambdas_before=before,
        proposed_lambdas=proposed,
        lambdas_after=after,
        proposed_log_steps=proposed_steps,
        accepted_log_steps=accepted_steps,
        quadratic_forms=_complete_mapping(before, update.quadratic_forms, missing=0.0),
        trace_terms=_complete_mapping(before, update.trace_terms, missing=0.0),
        objective_before=objective_before,
        objective_after=objective_after,
        objective_relative_change=abs(objective_after - objective_before)
        / (1.0 + abs(objective_before)),
        max_proposed_log_step=max(
            (abs(proposed_steps[name]) for name in estimated_names), default=0.0
        ),
        max_accepted_log_step=max(
            (abs(accepted_steps[name]) for name in estimated_names), default=0.0
        ),
        accepted=accepted,
        acceleration_outcome="disabled",
        acceleration_refusal_reason=None,
        accelerated_fit_index=None,
        backtracks=len(fit_indices) - 1,
        raw_backtracks=len(fit_indices) - 1,
        coefficient_fit_indices=fit_indices,
        accepted_fit_index=fit_indices[-1] if accepted else None,
        coefficient_tolerances=tolerances,
        boundary_nominations=(),
        update_curvature=state.fit.terminal_curvature,
        accepted_curvature=None if accepted_fit is None else accepted_fit.terminal_curvature,
        stage="newton",
        step_source=step_source,
        gradient=dict(zip(derivatives.names, derivatives.gradient.tolist(), strict=True)),
        gradient_certificate=dict(
            zip(derivatives.names, derivatives.gradient_certificate.tolist(), strict=True)
        ),
        hessian_certificate=hessian_certificate,
        projected_gradient_norm=float(projected_gradient_norm),
        newton_ridge=ridge,
    )


def _positive_definite_inverse(hessian: NDArray[np.float64]) -> NDArray[np.float64]:
    """Modified-Newton inverse: negative and tiny eigenvalues floored in magnitude."""
    values, vectors = np.linalg.eigh(0.5 * (hessian + hessian.T))
    floor = max(float(np.max(np.abs(values), initial=0.0)) * 1.0e-8, np.finfo(np.float64).tiny)
    adjusted = np.maximum(np.abs(values), floor)
    inverse = (vectors / adjusted) @ vectors.T
    return 0.5 * (inverse + inverse.T)


def _hessian_from_inverse(inverse_hessian: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """The Hessian a quasi-Newton inverse memory stands for; ``None`` when it cannot be inverted."""
    try:
        hessian = np.linalg.inv(0.5 * (inverse_hessian + inverse_hessian.T))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(hessian)):
        return None
    return 0.5 * (hessian + hessian.T)


def _lambda_from_log(
    name: str,
    value: float,
    *,
    lower_log: float,
    upper_log: float,
    upper_lambda: float,
    config: DistributionalEFSConfig,
) -> float:
    """``exp(value)`` with the bounds landed on exactly."""
    if value <= lower_log:
        return config.minimum_lambda
    if value >= upper_log:
        return upper_lambda
    return float(math.exp(value))


def run_newton_endgame(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    state: EndgameState,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    reuse_session: _DenseObservedReuseSession,
    history: list[DistributionalEFSIteration],
    coefficient_fits: list[DenseSolverResult],
    upper_bounds: Mapping[str, float] | None = None,
    budget: int | None = None,
) -> NewtonEndgameOutcome:
    """Damped Newton on the exact LAML gradient from an accepted smoothing state.

    Each iteration evaluates the gradient at the current fit, projects it
    onto the box, freezes flat directions (WPS16 step 4(b) through
    ``freeze_flat_directions``) and tests Wood's compound criterion, judging
    the curvature arms on the secant-updated memory of the last exact Hessian
    (the Fellner–Schall diagonal before one exists).  Only when it must step
    does it need a Hessian: the memory's quasi-Newton step when the memory
    holds, else one exact Hessian pass reusing the gradient pass's stencils
    (damped BFGS when that Hessian's certificate fails), with Armijo
    backtracking through warm-started fits; a refused quasi-Newton step is
    retried from the same point with a fresh exact Hessian.  ``upper_bounds``
    raises the box for components released beyond the cap.  Iterations and
    trial fits are appended to ``history`` and ``coefficient_fits`` as the
    loop's own are; a convergence check that makes no trial leaves no record.
    """
    config = efs_config
    tolerance = float(config.tolerance)
    lower_log = math.log(config.minimum_lambda)
    released = {} if upper_bounds is None else dict(upper_bounds)
    remaining = config.max_newton_iterations if budget is None else int(budget)
    fitter = _TrialFitter(
        family,
        layout,
        y,
        likelihood_plan,
        face=state.face,
        solver_config=solver_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        reuse_session=reuse_session,
    )
    dense = reuse_session.dense_matrices(layout, phase_recorder=phase_recorder)
    current = state
    previous: LamlDerivatives | None = None
    previous_rho: NDArray[np.float64] | None = None
    previous_gradient: NDArray[np.float64] | None = None
    # Wood's compound criterion judges the objective change of the last accepted
    # step, whoever made it: at the hand-off that is the warm-up's, so a hand-off
    # point already inside the bar ends stationary on its first gradient pass.
    previous_objective = float(state.objective)
    accepted_count = 0
    if (
        history
        and history[-1].accepted
        and dict(history[-1].lambdas_after) == dict(state.lambdas)
        and history[-1].objective_after == float(state.objective)
    ):
        previous_objective = float(history[-1].objective_before)
        accepted_count = 1
    consecutive_provenance_changes = 0
    # The quasi-Newton memory ``B ~ H^-1``, secant-updated with every accepted
    # step.  Seeded from a TRUSTED exact Hessian it is reusable: the passes
    # after that step judge and step with it until a step from it is refused.
    # A fallback memory (an untrusted Hessian's iteration, seeded from the
    # last good Hessian or the identity) keeps updating but is not reused: the
    # next step tries the exact Hessian at its point again, as the fallback
    # was designed per iteration.
    inverse_memory: NDArray[np.float64] | None = None
    memory_reusable = False
    last_good_hessian: NDArray[np.float64] | None = None
    last_exact_positive_definite: bool | None = None
    newton_count = 0
    bfgs_count = 0
    last_step: (
        tuple[EndgameState, NDArray[np.float64], tuple[str, ...], LamlDerivatives, float, str]
        | None
    ) = None
    halved = False

    def finish(
        kind: EndgameKind,
        derivatives: LamlDerivatives | None,
        norm: float | None,
        caps: tuple[str, ...] = (),
    ) -> NewtonEndgameOutcome:
        return NewtonEndgameOutcome(
            kind=kind,
            state=current,
            derivatives=derivatives,
            projected_gradient_norm=norm,
            cap_pressure=caps,
            newton_iterations=newton_count,
            bfgs_iterations=bfgs_count,
        )

    def upper_lambda(name: str) -> float:
        return float(released.get(name, config.maximum_lambda))

    def derivative_pass(want_hessian: bool, workspace: LamlDerivativeWorkspace) -> LamlDerivatives:
        with measure_phase(phase_recorder, _ENDGAME_PHASE):
            return laml_derivatives(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=current.lambdas,
                fit=current.fit,
                dense_matrices=dense,
                step=config.derivative_step,
                want_hessian=want_hessian,
                reuse=workspace,
            )

    while True:
        workspace = LamlDerivativeWorkspace()
        try:
            derivatives = derivative_pass(False, workspace)
        except LamlDerivativeError:
            if last_step is None or halved or remaining <= 0:
                return finish("objective_rejected", previous, None)
            # Halve the step that led here: one warm trial from the previous point.
            halved = True
            source, step_vector, step_names, source_derivatives, source_norm, source_kind = (
                last_step
            )
            source_rho = np.array([math.log(source.lambdas[name]) for name in step_names])
            halved_lambdas = dict(current.lambdas)
            for k, name in enumerate(step_names):
                upper_log = math.log(upper_lambda(name))
                value = float(np.clip(source_rho[k] + 0.5 * step_vector[k], lower_log, upper_log))
                halved_lambdas[name] = _lambda_from_log(
                    name,
                    value,
                    lower_log=lower_log,
                    upper_log=upper_log,
                    upper_lambda=upper_lambda(name),
                    config=config,
                )
            trial_fit, trial_objective = fitter.fit(
                halved_lambdas, initial=current.fit.coefficients, reuse_source=current.fit
            )
            coefficient_fits.append(trial_fit)
            index = len(coefficient_fits) - 1
            newton_count += 1
            remaining -= 1
            accepted = trial_objective is not None and trial_objective <= current.objective + (
                config.objective_tolerance * (1.0 + abs(current.objective))
            )
            evidence = (
                _fresh_raw_evidence(layout, halved_lambdas, trial_fit, config, face=current.face)
                if accepted
                else current.evidence
            )
            history.append(
                newton_iteration_record(
                    iteration=len(history) + 1,
                    state=current,
                    proposed_lambdas=halved_lambdas,
                    lambdas_after=halved_lambdas if accepted else current.lambdas,
                    objective_after=trial_objective if accepted else current.objective,
                    evidence=evidence,
                    fit_indices=(index,),
                    tolerances=(fitter.tolerance,),
                    accepted_fit=trial_fit if accepted else None,
                    step_source=source_kind,  # type: ignore[arg-type]
                    derivatives=source_derivatives,
                    projected_gradient_norm=source_norm,
                    hessian_certificate=None,
                    ridge=None,
                    estimated_names=step_names,
                )
            )
            if not accepted:
                return finish("objective_rejected", previous, None)
            assert trial_objective is not None
            current = EndgameState(
                lambdas=halved_lambdas,
                fit=trial_fit,
                objective=trial_objective,
                face=current.face,
                evidence=evidence,
                terminal_fit_index=index,
            )
            continue

        names = derivatives.names
        count = len(names)
        rho = np.array([math.log(current.lambdas[name]) for name in names])
        upper_logs = np.array([math.log(upper_lambda(name)) for name in names])
        if previous is not None and derivatives.provenance != previous.provenance:
            consecutive_provenance_changes += 1
            inverse_memory = None
            memory_reusable = False
            last_good_hessian = None
            last_exact_positive_definite = None
            previous_gradient = None
            previous_rho = None
            if consecutive_provenance_changes >= 2:
                return finish("gradient_unresolved", derivatives, None)
        else:
            consecutive_provenance_changes = 0
        gradient = derivatives.gradient
        certificate = derivatives.gradient_certificate
        if (
            inverse_memory is not None
            and previous_gradient is not None
            and previous_rho is not None
            and previous_gradient.shape == gradient.shape
        ):
            inverse_memory = bfgs_update(
                inverse_memory, rho - previous_rho, gradient - previous_gradient
            )
        # The curvature the gradient-only pass freezes and judges with: the
        # reusable memory's Hessian after a trusted step, else the penalty-only
        # Hessian (h_kl without the likelihood's third and fourth derivatives),
        # which vanishes at a working infinity as the exact one does.  The
        # Fellner--Schall diagonal does not: it tends to half the penalty rank
        # there, and a direction it never froze was stepped by 1e-9 for ever on
        # the NB2 linear-effect cap fixture.
        estimate: NDArray[np.float64] | None = None
        if inverse_memory is not None and memory_reusable:
            estimate = _hessian_from_inverse(inverse_memory)
        if estimate is None:
            estimate = derivatives.penalty_hessian
        objective = float(current.objective)
        score_scale = 1.0 + abs(objective)
        estimated_mask = np.ones(count, dtype=bool)
        projected = project_reml_gradient(
            gradient,
            rho,
            estimated_mask,
            log_lower=lower_log,
            log_upper=upper_logs,
            bound_window=0.0,
        )
        fs_stepped = certificate >= np.abs(gradient)
        blocking = certificate > tolerance * score_scale
        ranks_by_name = {
            component.name: component.rank for component in current.evidence.components
        }
        penalty_ranks = np.array([ranks_by_name.get(name, 1.0) for name in names])
        freeze = freeze_flat_directions(
            projected,
            estimate,
            penalty_ranks,
            estimated_mask,
            objective=objective,
            tolerance=tolerance,
        )
        frozen = np.asarray(freeze.frozen, dtype=bool)
        stop_gradient = np.where(frozen, 0.0, projected)
        candidate = evaluate_reml_candidate(
            iteration=accepted_count,
            objective=objective,
            previous_objective=previous_objective,
            projected_gradient=stop_gradient,
            tolerance=tolerance,
        )
        norm = float(candidate.projected_gradient_norm)
        active = ~frozen & ~fs_stepped
        active_index = np.flatnonzero(active)
        active_pairs = np.ix_(active_index, active_index)
        if active_index.size:
            # WPS16 4(c) on what is at hand: the estimate's active block, and the
            # last exact Hessian's verdict where one has been formed.
            positive_definite = bool(np.min(np.linalg.eigvalsh(estimate[active_pairs])) > 0.0) and (
                last_exact_positive_definite is not False
            )
        else:
            positive_definite = True

        def converged_outcome(published: LamlDerivatives) -> NewtonEndgameOutcome:
            if bool(np.any(blocking)):
                return finish("gradient_unresolved", published, norm)
            # A component at its upper bound is under cap pressure only when its
            # gradient is MATERIALLY outward.  Within the bar it is KKT-stationary
            # at the box bound (measured: an NB2 linear effect at the cap with
            # g = -2.9e-9 against a bar of 3.5e-3, whose endpoint verdict was
            # "finite" and whose bracketed probe at 1e14 was still outward --
            # the objective is flat to 1e-9 there, and the fit is converged).
            caps = tuple(
                name
                for k, name in enumerate(names)
                if current.lambdas[name] == upper_lambda(name)
                and gradient[k] < -tolerance * score_scale
            )
            if caps:
                return finish("cap_pressure", published, norm, caps)
            return finish("stationary", published, norm)

        # Wood's criterion judges the gradient and the objective change on the
        # objective's scale.  A third arm on the same scale: the objective the
        # remaining Newton step could still win, the Newton decrement
        # 0.5 * g' H^-1 g, must be below the bar too.  A bound on the step's
        # LENGTH in log lambda is the wrong quantity: on a real book the
        # objective is flat to round-off along some directions, Newton steps
        # there are 1e-4 to 1e-2 in log lambda for ever, and a length bar of
        # reml_tol never fires (measured: 17 wasted iterations after Wood's
        # criterion was met at the third Newton step).
        remaining_gain = 0.0
        if candidate.converged and active_index.size:
            try:
                probe_step, _probe_ridge = newton_direction(
                    stop_gradient, estimate, active, max_log_step=config.max_log_step
                )
                remaining_gain = float(-0.5 * stop_gradient @ probe_step)
            except np.linalg.LinAlgError:
                remaining_gain = math.inf
        if candidate.converged and positive_definite and remaining_gain <= tolerance * score_scale:
            return converged_outcome(derivatives)
        if bool(np.any(blocking)):
            # Within its certificate the gradient cannot be told from zero: the
            # endgame can make no certified progress on such a component.
            effective = np.sign(stop_gradient) * np.maximum(
                np.abs(stop_gradient) - certificate, 0.0
            )
            limited = evaluate_reml_candidate(
                iteration=accepted_count,
                objective=objective,
                previous_objective=previous_objective,
                projected_gradient=effective,
                tolerance=tolerance,
            )
            if limited.converged:
                return finish("gradient_unresolved", derivatives, norm)
        if remaining <= 0:
            return finish("max_iterations", derivatives, norm)

        # The step.  A quasi-Newton step from the memory when one holds; else one
        # exact Hessian pass on the gradient pass's stencils.  A refused
        # quasi-Newton step comes back here for the exact Hessian at this point.
        force_exact = False
        while True:
            hessian_pass: LamlDerivatives | None = None
            active_certificate: float | None = None
            ridge: float | None = None
            reused = inverse_memory is not None and memory_reusable and not force_exact
            if reused:
                assert inverse_memory is not None
                step = bfgs_direction(
                    gradient, inverse_memory, active, max_log_step=config.max_log_step
                )
                step_source: Literal["newton", "bfgs"] = "bfgs"
                bfgs_count += 1
            else:
                hessian_trusted = False
                hessian: NDArray[np.float64] | None = None
                try:
                    hessian_pass = derivative_pass(True, workspace)
                except LamlDerivativeError:
                    hessian_pass = None
                if hessian_pass is not None:
                    hessian = hessian_pass.hessian
                    assert hessian is not None and hessian_pass.hessian_certificate is not None
                    if active_index.size:
                        active_block = hessian[active_pairs]
                        active_certificate = float(
                            np.max(hessian_pass.hessian_certificate[active_pairs])
                        )
                        hessian_trusted = (
                            config.hessian_certificate_fraction
                            * float(np.min(np.diag(active_block)))
                            > active_certificate
                        )
                        last_exact_positive_definite = bool(
                            np.min(np.linalg.eigvalsh(active_block)) > 0.0
                        )
                    else:
                        active_certificate = 0.0
                        hessian_trusted = True
                        last_exact_positive_definite = True
                if hessian_trusted:
                    assert hessian is not None
                    try:
                        step, ridge_value = newton_direction(
                            gradient, hessian, active, max_log_step=config.max_log_step
                        )
                    except np.linalg.LinAlgError:
                        return finish("objective_rejected", hessian_pass, norm)
                    ridge = float(ridge_value)
                    last_good_hessian = hessian
                    inverse_memory = _positive_definite_inverse(hessian)
                    memory_reusable = True
                    step_source = "newton"
                else:
                    if inverse_memory is None:
                        # Seed the quasi-Newton memory from the last trusted exact
                        # Hessian; with none yet, the identity (Nocedal and Wright
                        # 2006 §6.1): an untrusted Hessian's inverse can be 1e-12
                        # and freeze the search.
                        inverse_memory = (
                            np.eye(count)
                            if last_good_hessian is None
                            else _positive_definite_inverse(last_good_hessian)
                        )
                    step = bfgs_direction(
                        gradient, inverse_memory, active, max_log_step=config.max_log_step
                    )
                    memory_reusable = False
                    step_source = "bfgs"
                    bfgs_count += 1
            recorded = derivatives if hessian_pass is None else hessian_pass
            update = current.evidence.update
            if update is not None:
                for k in np.flatnonzero(fs_stepped & ~frozen):
                    step[k] = float(update.log_steps[names[k]])
            step[frozen] = 0.0
            if not np.any(step != 0.0):
                # Nothing to move: every direction is frozen or at a bound.  Wood's
                # compound criterion needs one accepted step to judge the objective
                # change, but there is no step to take; a gradient under the bar is
                # the stationary point (an EFS warm-up can hand over a flat problem).
                if norm < tolerance * score_scale:
                    return converged_outcome(recorded)
                return finish("objective_rejected", recorded, norm)

            newton_count += 1
            remaining -= 1
            descent = float(gradient @ step)
            proposed_lambdas = dict(current.lambdas)
            for k, name in enumerate(names):
                value = float(np.clip(rho[k] + step[k], lower_log, upper_logs[k]))
                proposed_lambdas[name] = _lambda_from_log(
                    name,
                    value,
                    lower_log=lower_log,
                    upper_log=upper_logs[k],
                    upper_lambda=upper_lambda(name),
                    config=config,
                )
            fit_indices: list[int] = []
            tolerances: list[float] = []
            accepted_fit: DenseSolverResult | None = None
            accepted_lambdas: dict[str, float] | None = None
            accepted_objective: float | None = None
            ceiling_slack = config.objective_tolerance * score_scale
            for backtrack in range(config.max_backtracks + 1):
                scale = config.backtrack_factor**backtrack
                trial_lambdas = dict(current.lambdas)
                for k, name in enumerate(names):
                    value = float(np.clip(rho[k] + scale * step[k], lower_log, upper_logs[k]))
                    trial_lambdas[name] = _lambda_from_log(
                        name,
                        value,
                        lower_log=lower_log,
                        upper_log=upper_logs[k],
                        upper_lambda=upper_lambda(name),
                        config=config,
                    )
                if all(trial_lambdas[name] == current.lambdas[name] for name in names):
                    break
                trial_fit, trial_objective = fitter.fit(
                    trial_lambdas, initial=current.fit.coefficients, reuse_source=current.fit
                )
                coefficient_fits.append(trial_fit)
                fit_indices.append(len(coefficient_fits) - 1)
                tolerances.append(fitter.tolerance)
                if trial_objective is None:
                    continue
                if trial_objective <= objective + ARMIJO_CONSTANT * scale * descent + ceiling_slack:
                    accepted_fit = trial_fit
                    accepted_lambdas = trial_lambdas
                    accepted_objective = trial_objective
                    break
            if not fit_indices:
                # Every trial collapsed onto the current point: nothing to move.
                newton_count -= 1
                remaining += 1
                if norm < tolerance * score_scale:
                    return converged_outcome(recorded)
                return finish("objective_rejected", recorded, norm)
            accepted = accepted_fit is not None
            evidence = (
                _fresh_raw_evidence(
                    layout, accepted_lambdas, accepted_fit, config, face=current.face
                )
                if accepted and accepted_lambdas is not None and accepted_fit is not None
                else current.evidence
            )
            history.append(
                newton_iteration_record(
                    iteration=len(history) + 1,
                    state=current,
                    proposed_lambdas=proposed_lambdas,
                    lambdas_after=accepted_lambdas
                    if accepted and accepted_lambdas
                    else current.lambdas,
                    objective_after=accepted_objective
                    if accepted and accepted_objective is not None
                    else objective,
                    evidence=evidence,
                    fit_indices=tuple(fit_indices),
                    tolerances=tuple(tolerances),
                    accepted_fit=accepted_fit,
                    step_source=step_source,
                    derivatives=recorded,
                    projected_gradient_norm=norm,
                    hessian_certificate=active_certificate,
                    ridge=ridge,
                    estimated_names=names,
                )
            )
            if accepted:
                break
            if reused and remaining > 0:
                # The memory misjudged this point: form the exact Hessian here
                # and step again before giving up on the line search.
                inverse_memory = None
                force_exact = True
                continue
            # A dead line search with the gradient under the bar is convergence
            # at precision (the scalar path's classify_dead_feasible_exit).
            if norm < tolerance * score_scale:
                return converged_outcome(recorded)
            return finish("objective_rejected", recorded, norm)
        workspace.clear()
        assert accepted_fit is not None and accepted_lambdas is not None
        assert accepted_objective is not None
        previous = recorded
        previous_rho = rho
        previous_gradient = gradient
        previous_objective = objective
        accepted_count += 1
        halved = False
        last_step = (current, step, names, recorded, norm, step_source)
        current = EndgameState(
            lambdas=accepted_lambdas,
            fit=accepted_fit,
            objective=accepted_objective,
            face=current.face,
            evidence=evidence,
            terminal_fit_index=fit_indices[-1],
        )


@dataclass(frozen=True)
class BracketRelease:
    """A refused cap component whose optimum the bracketed search located."""

    name: str
    lambdas: dict[str, float]
    fit: DenseSolverResult
    objective: float
    fits: tuple[DenseSolverResult, ...]
    tolerance: float
    outcome: BracketOutcome


@dataclass(frozen=True)
class BracketAttempt:
    """A bracketed search that made fits but found no root within the span."""

    name: str
    fits: tuple[DenseSolverResult, ...]
    tolerance: float
    outcome: BracketOutcome


def bracket_refused_component(
    family: DistributionalFamily,
    layout: StackedLayout,
    y: NDArray,
    likelihood_plan: FamilyLikelihoodPlan,
    *,
    name: str,
    endpoint_derivative: float,
    pending: NewtonEndgameOutcome,
    state: EndgameState,
    solver_config: DenseSolverConfig,
    efs_config: DistributionalEFSConfig,
    chunk_size: ChunkSize | None,
    phase_recorder: FitPhaseRecorder | None,
    reuse_session: _DenseObservedReuseSession,
) -> BracketRelease | BracketAttempt | None:
    """Search ``(lambda_cap, maximum_lambda_conditioning]`` for the optimum of one refused cap.

    ``phi(u) = dF/dtau = -g_k lambda_k`` at ``lambda_k = lambda_cap exp(u)``,
    every evaluation a warm-started fit plus a gradient pass.  ``None`` when
    the search never starts (no gradient for the component, or the signs do
    not bracket a root).
    """
    derivatives = pending.derivatives
    if derivatives is None or name not in derivatives.names:
        return None
    index = derivatives.names.index(name)
    cap = float(state.lambdas[name])
    phi_at_cap = -float(derivatives.gradient[index]) * cap
    maximum_lambda_conditioning = efs_config.maximum_lambda_conditioning
    assert maximum_lambda_conditioning is not None
    log_span = math.log(maximum_lambda_conditioning) - math.log(cap)
    if log_span <= 0.0:
        return None
    fitter = _TrialFitter(
        family,
        layout,
        y,
        likelihood_plan,
        face=state.face,
        solver_config=solver_config,
        chunk_size=chunk_size,
        phase_recorder=phase_recorder,
        reuse_session=reuse_session,
    )
    dense = reuse_session.dense_matrices(layout, phase_recorder=phase_recorder)
    made: list[tuple[float, DenseSolverResult, dict[str, float], float]] = []

    def evaluate(u: float) -> float:
        lambdas = dict(state.lambdas)
        lambdas[name] = cap * math.exp(u)
        warm = made[-1][1] if made else state.fit
        fit, objective = fitter.fit(lambdas, initial=warm.coefficients, reuse_source=warm)
        if objective is None:
            raise ValueError("bracket fit did not reach a stationary coefficient state")
        made.append((u, fit, lambdas, objective))
        with measure_phase(phase_recorder, _ENDGAME_PHASE):
            gradient = laml_derivatives(
                family,
                layout,
                y,
                likelihood_plan,
                lambdas=lambdas,
                fit=fit,
                dense_matrices=dense,
                step=efs_config.derivative_step,
                want_hessian=False,
            ).gradient
        return -float(gradient[index]) * lambdas[name]

    try:
        outcome = bracket_beyond_cap(phi_at_cap, endpoint_derivative, evaluate, log_span=log_span)
    except LamlDerivativeError:
        outcome = BracketOutcome(
            found=False, log_lambda_ratio=None, evaluations=len(made), bracket=(0.0, log_span)
        )
    fits = tuple(record[1] for record in made)
    if not outcome.found or outcome.log_lambda_ratio is None:
        if not fits:
            return None
        return BracketAttempt(name=name, fits=fits, tolerance=fitter.tolerance, outcome=outcome)
    root = outcome.log_lambda_ratio
    if made and made[-1][0] == root:
        _u, fit, lambdas, objective = made[-1]
    else:
        lambdas = dict(state.lambdas)
        lambdas[name] = cap * math.exp(root)
        warm = made[-1][1] if made else state.fit
        fit, objective_value = fitter.fit(lambdas, initial=warm.coefficients, reuse_source=warm)
        if objective_value is None:
            return BracketAttempt(
                name=name, fits=(*fits, fit), tolerance=fitter.tolerance, outcome=outcome
            )
        objective = objective_value
        fits = (*fits, fit)
    return BracketRelease(
        name=name,
        lambdas=lambdas,
        fit=fit,
        objective=objective,
        fits=fits,
        tolerance=fitter.tolerance,
        outcome=outcome,
    )


__all__ = [
    "ARMIJO_CONSTANT",
    "HANDOFF_REASONS",
    "NO_HANDOFF_REASONS",
    "BracketAttempt",
    "BracketOutcome",
    "BracketRelease",
    "EndgameState",
    "NewtonEndgameOutcome",
    "bfgs_direction",
    "bfgs_update",
    "bracket_beyond_cap",
    "bracket_refused_component",
    "newton_direction",
    "newton_iteration_record",
    "run_newton_endgame",
    "should_hand_off",
]
