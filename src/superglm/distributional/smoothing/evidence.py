"""Fresh raw evidence for a smoothing state and the pressure it puts on its bounds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import DenseSolverResult, DistributionalEFSConfig
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import (
    _component_states,
    _estimated_names,
    _stable_isolated_gfs_update,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult, wood_fasiolo_update


@dataclass(frozen=True)
class _FreshRawEvidence:
    """One accepted state's freshly rebuilt outer-convergence authority."""

    components: tuple[EFSComponentState, ...]
    estimated_names: tuple[str, ...]
    update: EFSUpdateResult | None
    maximum: float
    working_infinity: tuple[str, ...]
    unresolved_upper_bound: tuple[str, ...]


def _fresh_raw_evidence(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    fit: DenseSolverResult,
    config: DistributionalEFSConfig,
    *,
    face: PenaltyFace | None = None,
) -> _FreshRawEvidence:
    """Rebuild the proposal and normalized stationarity residual at one fit."""

    components = (
        _component_states(layout, lambdas)
        if face is None
        else projected_component_states(layout=layout, lambdas=lambdas, face=face)
    )
    estimated_names = _estimated_names(components)
    if not components:
        return _FreshRawEvidence(
            components=components,
            estimated_names=estimated_names,
            update=None,
            maximum=0.0,
            working_infinity=(),
            unresolved_upper_bound=(),
        )
    inverse = fit.terminal_pseudo_inverse()
    update = wood_fasiolo_update(
        components,
        fit.coefficients,
        inverse,
        inverse_scale=1.0,
        max_log_step=config.max_log_step,
        minimum_lambda=config.minimum_lambda,
        maximum_lambda=config.maximum_lambda,
    )
    update, stable_raw_names = _stable_isolated_gfs_update(
        components,
        fit,
        inverse,
        update,
        config,
    )
    working_infinity = tuple(
        name for name in estimated_names if update.proposal_kinds[name] == "working_infinity"
    )
    capped_names = frozenset(
        name for name in estimated_names if lambdas[name] == config.maximum_lambda
    )
    # Normalized fixed-point stationarity is the finite convergence
    # certificate.  The raw GFS quotient is cancellation-amplified near a
    # saturated penalty, so it may only override that certificate where its
    # direction has operational meaning: working infinity, or a coordinate
    # currently held at the upper cap.  At the cap, a materially negative raw
    # step must move back into the finite interior, while a positive step asks
    # the endpoint assessor to distinguish a finite optimum beyond the cap
    # from λ = ∞.  Roundoff-sized negative drift is compatible with a finite
    # optimum at the cap itself.
    convergence_evidence = [
        abs(float(update.stationarity_log_residuals[name])) for name in estimated_names
    ]
    for name in estimated_names:
        raw_step = float(update.raw_log_steps[name])
        if name in stable_raw_names or update.proposal_kinds[name] == "working_infinity":
            convergence_evidence.append(abs(raw_step))
        elif name in capped_names and (raw_step > 0.0 or raw_step < -config.tolerance):
            convergence_evidence.append(abs(raw_step))
    maximum = max(convergence_evidence, default=0.0)
    unresolved_upper_bound = tuple(
        name
        for name in estimated_names
        if name in capped_names and update.raw_log_steps[name] > 0.0
    )
    return _FreshRawEvidence(
        components=components,
        estimated_names=estimated_names,
        update=update,
        maximum=maximum,
        working_infinity=working_infinity,
        unresolved_upper_bound=unresolved_upper_bound,
    )


def _lower_bound_pressure(
    evidence: _FreshRawEvidence,
    lambdas: Mapping[str, float],
    raw_log_steps: Mapping[str, float],
    config: DistributionalEFSConfig,
) -> tuple[str, ...]:
    """Estimated components pinned at the minimum whose raw step points outward."""
    return tuple(
        name
        for name in evidence.estimated_names
        if lambdas[name] == config.minimum_lambda and raw_log_steps[name] < -config.tolerance
    )


def _saturated_names(
    components: tuple[EFSComponentState, ...],
    update: EFSUpdateResult,
    estimated_names: tuple[str, ...],
    threshold: float,
) -> frozenset[str]:
    """Names saturated at a large lambda AND still being pushed upward.

    ``lambda_j tr(H^-1 S_j) / r_j`` is bounded above by one and approaches it as
    ``lambda_j -> inf``, because the penalty then dominates its own block; an
    interior component settles strictly below one because the data curvature
    keeps a share.  Saturation alone is a statement about the current lambda,
    not about the optimum: a fit started above a finite optimum is saturated
    while walking DOWN toward it.  What distinguishes the infinity drift is its
    direction -- the fixed point keeps proposing lambda increases forever -- so
    a name counts only while its proposed log step is positive.  A threshold of
    one disables the classification.
    """
    if threshold >= 1.0:
        return frozenset()
    estimated = set(estimated_names)
    saturated: set[str] = set()
    for component in components:
        if component.name not in estimated or component.rank <= 0.0:
            continue
        trace = update.trace_terms.get(component.name)
        if trace is None:
            continue
        if update.raw_log_steps.get(component.name, 0.0) <= 0.0:
            continue
        if component.lambda_value * float(trace) / component.rank > threshold:
            saturated.add(component.name)
    return frozenset(saturated)
