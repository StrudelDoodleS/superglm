"""Immutable IRLS snapshots and atomic trial-step selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from superglm.distributions import Distribution, clip_mu
from superglm.group_matrix import DesignMatrix
from superglm.links import Link, stabilize_eta


@dataclass(frozen=True)
class SolverState:
    """A complete, immutable evaluated coefficient-state snapshot."""

    beta: NDArray
    intercept: float
    eta_unclipped: NDArray
    eta: NDArray
    mu: NDArray
    deviance: float
    state_id: int | None = None
    evaluation_id: int | None = None
    state_space: str = "solver"
    basis_id: int | None = None
    lambdas: tuple[tuple[str, object], ...] = ()
    dispersion: float | None = None
    convergence_value: float | None = None
    termination_reason: str | None = None


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
    state_id: int | None = None,
    evaluation_id: int | None = None,
    state_space: str = "solver",
    basis_id: int | None = None,
    lambdas: tuple[tuple[str, object], ...] = (),
    dispersion: float | None = None,
) -> _IRLSState:
    """Evaluate and freeze all state derived from one coefficient vector."""
    retained_beta = _immutable_array(beta)
    eta_unclipped = dm.matvec(retained_beta) + intercept + offset
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
        state_id=state_id,
        evaluation_id=evaluation_id,
        state_space=state_space,
        basis_id=basis_id,
        lambdas=lambdas,
        dispersion=dispersion,
    )


StateInvalid = Callable[[_IRLSState], bool]


def _state_is_finite(state: _IRLSState) -> bool:
    return bool(
        np.isfinite(state.intercept)
        and np.all(np.isfinite(state.beta))
        and np.all(np.isfinite(state.eta_unclipped))
        and np.all(np.isfinite(state.eta))
        and np.all(np.isfinite(state.mu))
        and np.isfinite(state.deviance)
    )


def _irls_trial_is_unsafe(
    candidate: _IRLSState,
    committed: _IRLSState,
    invalid_state: StateInvalid | None = None,
) -> bool:
    """Apply the centralized legacy catastrophe predicate to a full state."""
    if not _state_is_finite(candidate):
        return True
    if invalid_state is not None and invalid_state(candidate):
        return True

    candidate_deviance = candidate.deviance
    committed_deviance = committed.deviance
    return bool(
        np.isfinite(committed_deviance)
        and (
            candidate_deviance > 2.0 * committed_deviance
            or (committed_deviance >= 0.0 and candidate_deviance < -abs(committed_deviance))
        )
    )


def _select_irls_trial(
    *,
    committed: _IRLSState,
    proposal: _IRLSState,
    evaluate_state: Callable[[float], _IRLSState],
    invalid_state: StateInvalid | None = None,
    max_halving: int = 5,
) -> _IRLSStepDecision:
    """Return the largest safe fixed-endpoint trial, or reject atomically."""
    if max_halving < 1:
        raise ValueError("max_halving must be at least 1")
    if not _irls_trial_is_unsafe(proposal, committed, invalid_state):
        return _IRLSStepDecision(1.0, 0, False, trials_attempted=1)

    for depth in range(1, max_halving + 1):
        alpha = 2.0**-depth
        candidate = evaluate_state(alpha)
        if not _irls_trial_is_unsafe(candidate, committed, invalid_state):
            return _IRLSStepDecision(alpha, depth, False, trials_attempted=depth + 1)

    return _IRLSStepDecision(0.0, 0, True, trials_attempted=max_halving + 1)
