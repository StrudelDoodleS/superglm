"""Immutable IRLS snapshots and atomic trial-step selection."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, Poisson, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import Link, SqrtLink, stabilize_eta

_MAX_FLOAT64_HALVING_DEPTH = 1074


@dataclass(frozen=True)
class SolverState:
    """A complete, immutable evaluated coefficient-state snapshot."""

    beta: NDArray
    intercept: float
    eta_unclipped: NDArray
    eta: NDArray
    mu: NDArray
    deviance: float
    penalized_deviance: float | None = None
    state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    lambdas: tuple[tuple[str, object], ...] = ()
    dispersion: float | None = None


# Migration alias retained while direct IRLS and downstream private callers
# move onto the shared state identity contract.
_IRLSState = SolverState


@dataclass(frozen=True)
class _IRLSStepDecision:
    """The accepted fraction of a proposal, or an atomic rejection."""

    alpha: float
    step_halvings: int
    step_rejected: bool
    trials_attempted: int = 1


def _immutable_array(values: NDArray) -> NDArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _freeze_owned_array(values: NDArray) -> NDArray:
    """Freeze an array produced inside state evaluation without copying it."""
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


def _evaluate_irls_state(
    dm: DesignMatrix,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    offset: NDArray,
    beta: NDArray,
    intercept: float,
    *,
    deviance: float | None = None,
    eta_unclipped: NDArray | None = None,
    penalized_deviance: float | None = None,
    state_id: int | None = None,
    evaluation_id: int | None = None,
    state_space: str = "solver",
    basis_id: int | None = None,
    lambdas: tuple[tuple[str, object], ...] = (),
    dispersion: float | None = None,
) -> _IRLSState:
    """Evaluate and freeze all state derived from one coefficient vector."""
    retained_beta = _immutable_array(beta)
    if eta_unclipped is None:
        eta_unclipped = dm.matvec(retained_beta) + intercept + offset
    else:
        eta_unclipped = np.asarray(eta_unclipped, dtype=float)
        if eta_unclipped.shape != (dm.n,):
            raise ValueError(f"eta_unclipped must have shape {(dm.n,)}, got {eta_unclipped.shape}")
    eta = stabilize_eta(eta_unclipped, link)
    mu = clip_mu(link.inverse(eta), family)
    retained_deviance = (
        float(np.sum(weights * family.deviance_unit(y, mu)))
        if deviance is None
        else float(deviance)
    )
    eta_unclipped = _freeze_owned_array(eta_unclipped)
    eta = _freeze_owned_array(eta)
    mu = _freeze_owned_array(mu)
    return _IRLSState(
        beta=retained_beta,
        intercept=float(intercept),
        eta_unclipped=eta_unclipped,
        eta=eta,
        mu=mu,
        deviance=retained_deviance,
        penalized_deviance=(None if penalized_deviance is None else float(penalized_deviance)),
        state_id=state_id,
        evaluation_id=evaluation_id,
        state_space=state_space,
        basis_id=basis_id,
        lambdas=lambdas,
        dispersion=dispersion,
    )


StateInvalid = Callable[[_IRLSState], bool]
MeritDelta = Callable[[_IRLSState, _IRLSState], float]


def _stable_penalized_deviance_delta(
    candidate: _IRLSState,
    committed: _IRLSState,
    penalty_matvec: Callable[[NDArray], NDArray] | NDArray | None = None,
    nonsmooth_penalty: Callable[[NDArray], float] | None = None,
) -> float:
    """Compare penalized deviances without subtracting two large quadratics.

    In an ill-conditioned smooth basis, the two penalty quadratics can each be
    accurately evaluated while their tiny difference loses enough digits to
    reverse the sign of an otherwise safe terminal step.  The polarization
    identity evaluates that difference directly from the coefficient update.

    ``penalty_matvec`` supplies the quadratic penalty ``S`` (a matrix or a
    matvec); pass ``None`` when the fit carries no quadratic penalty. **It
    must apply a symmetric operator.** The polarization identity used here
    evaluates ``(b1 - b0)' S (b1 + b0)``, which equals the difference of the
    two merit quadratics ``b' S b`` only when ``S = S'``; for an asymmetric
    ``S`` the antisymmetric part cancels out of ``b' S b`` but not out of the
    polarized form, so the merit and its delta would disagree. Every in-tree
    penalty is symmetric by construction; the constraint is stated because
    ``S_override`` accepts an arbitrary ``(p, p)`` array on a shape check
    alone.
    ``nonsmooth_penalty`` supplies any non-quadratic penalty term as a
    function of ``beta``, already scaled to match the caller's merit
    convention; its two evaluations enter the same ``math.fsum``.
    """
    terms = [float(candidate.deviance), -float(committed.deviance)]

    if penalty_matvec is not None:
        delta_beta = candidate.beta - committed.beta
        summed_beta = candidate.beta + committed.beta
        penalty_direction = (
            penalty_matvec(summed_beta)
            if callable(penalty_matvec)
            else np.asarray(penalty_matvec, dtype=np.float64) @ summed_beta
        )
        terms.append(
            math.fsum(
                float(delta_value * direction_value)
                for delta_value, direction_value in zip(
                    delta_beta,
                    penalty_direction,
                    strict=True,
                )
            )
        )

    if nonsmooth_penalty is not None:
        terms.append(float(nonsmooth_penalty(candidate.beta)))
        terms.append(-float(nonsmooth_penalty(committed.beta)))

    return math.fsum(terms)


def _state_is_finite(state: _IRLSState) -> bool:
    return bool(
        np.isfinite(state.intercept)
        and np.all(np.isfinite(state.beta))
        and np.all(np.isfinite(state.eta_unclipped))
        and np.all(np.isfinite(state.eta))
        and np.all(np.isfinite(state.mu))
        and np.isfinite(state.deviance)
        and (state.penalized_deviance is None or np.isfinite(state.penalized_deviance))
    )


def _state_merit(state: _IRLSState) -> float:
    """Return the fitted objective used to judge an IRLS trial."""
    if state.penalized_deviance is not None:
        return float(state.penalized_deviance)
    return float(state.deviance)


def _irls_objective_scale(
    *,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
) -> float:
    """Return the natural additive scale for an IRLS objective."""
    if type(family) is not Poisson or type(link) is not SqrtLink:
        return 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        response_mass = float(np.sum(weights * y, dtype=np.float64))
    if not np.isfinite(response_mass):
        return 1.0
    return max(response_mass, np.finfo(np.float64).tiny)


def _irls_objective_relative_change(
    *,
    objective: float,
    previous: float,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
) -> float:
    """Return a convergence ratio with link-appropriate objective units."""
    # Poisson deviance is homogeneous of degree one when y and μ are rescaled
    # together.  A fixed ``+1`` denominator therefore declares convergence
    # prematurely for genuinely tiny sqrt-link means.  Response mass carries
    # the same units and makes this stopping rule scale-equivariant.
    objective_scale = _irls_objective_scale(
        y=y,
        weights=weights,
        family=family,
        link=link,
    )
    return abs(objective - previous) / (abs(previous) + objective_scale)


def _poisson_sqrt_halving_budget(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    y: NDArray,
    weights: NDArray,
    family: Distribution,
    link: Link,
    default: int,
) -> int:
    """Scale a binary backtracking budget for an exact Poisson/sqrt step.

    For a proposal direction ``d_eta``, let

        R = max_i |d_eta_i| / sqrt(y_i)

    over positive-weight, positive-response rows.  ``R`` is dimensionless and
    invariant when the response is rescaled by ``c`` and eta by ``sqrt(c)``.
    Adding ``ceil(log2(R))`` to the ordinary budget gives the search the same
    number of halvings *after* the proposal displacement reaches its natural
    response scale.  This changes only how far backtracking may look; it does
    not alter the Fisher direction, score, or fitted objective.
    """
    if default < 1:
        raise ValueError("default halving budget must be at least 1")
    if type(family) is not Poisson or type(link) is not SqrtLink:
        return default

    y_values = np.asarray(y, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    active = (weight_values > 0.0) & (y_values > 0.0)
    if not np.any(active):
        return default

    eta_step = np.abs(proposal.eta_unclipped[active] - committed.eta_unclipped[active])
    moving = eta_step > 0.0
    if not np.any(moving):
        return default

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log2_ratio = np.log2(eta_step[moving]) - 0.5 * np.log2(y_values[active][moving])
    # A non-finite endpoint direction cannot be recovered by multiplying it by
    # a positive alpha: IEEE ``alpha * inf`` remains infinite.  Retain the
    # ordinary bounded rejection path instead of attempting 1074 futile trials.
    if np.any(np.isposinf(log2_ratio)):
        return default
    finite = log2_ratio[np.isfinite(log2_ratio)]
    if finite.size == 0:
        return default

    extra_depth = max(0, int(np.ceil(float(np.max(finite)))))
    return min(default + extra_depth, _MAX_FLOAT64_HALVING_DEPTH)


def _irls_trial_is_unsafe(
    candidate: _IRLSState,
    committed: _IRLSState,
    invalid_state: StateInvalid | None = None,
    merit_delta: MeritDelta | None = None,
    merit_scale: float = 1.0,
) -> bool:
    """Reject invalid states or a material increase in the fitted objective."""
    if not _state_is_finite(candidate):
        return True
    if invalid_state is not None and invalid_state(candidate):
        return True
    if (candidate.penalized_deviance is None) != (committed.penalized_deviance is None):
        return True

    candidate_merit = _state_merit(candidate)
    committed_merit = _state_merit(committed)
    if not np.isfinite(committed_merit):
        return False
    roundoff = (
        64.0
        * np.finfo(float).eps
        * max(
            merit_scale,
            abs(candidate_merit),
            abs(committed_merit),
        )
    )
    if merit_delta is not None:
        delta = float(merit_delta(candidate, committed))
        return not np.isfinite(delta) or bool(delta > roundoff)
    return bool(candidate_merit > committed_merit + roundoff)


def _select_irls_trial(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    evaluate_state: Callable[[float], _IRLSState],
    invalid_state: StateInvalid | None = None,
    max_halving: int = 20,
    merit_delta: MeritDelta | None = None,
    merit_scale: float = 1.0,
) -> _IRLSStepDecision:
    """Return the largest safe fixed-endpoint trial, or reject atomically."""
    if max_halving < 1:
        raise ValueError("max_halving must be at least 1")
    if not _irls_trial_is_unsafe(
        proposal,
        committed,
        invalid_state,
        merit_delta,
        merit_scale,
    ):
        return _IRLSStepDecision(1.0, 0, False, trials_attempted=1)

    trials_attempted = 1
    for depth in range(1, max_halving + 1):
        alpha = 2.0**-depth
        if alpha == 0.0:
            break
        candidate = evaluate_state(alpha)
        trials_attempted += 1
        if not _irls_trial_is_unsafe(
            candidate,
            committed,
            invalid_state,
            merit_delta,
            merit_scale,
        ):
            return _IRLSStepDecision(alpha, depth, False, trials_attempted=depth + 1)

    return _IRLSStepDecision(0.0, 0, True, trials_attempted=trials_attempted)
