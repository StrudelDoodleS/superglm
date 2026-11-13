"""Fresh raw evidence for a smoothing state and the pressure it puts on its bounds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from superglm.distributional.layout import StackedLayout
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverResult
from superglm.distributional.smoothing.face_efs import projected_component_states
from superglm.distributional.smoothing.objective import (
    _component_states,
    _estimated_names,
    _stable_isolated_gfs_update,
)
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.smoothing.penalty_geometry import _PenaltyEvaluationMemo
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
    penalty_memo: _PenaltyEvaluationMemo | None = None,
) -> _FreshRawEvidence:
    """Rebuild the proposal and normalized stationarity residual at one fit."""

    components = (
        _component_states(layout, lambdas)
        if face is None
        else projected_component_states(
            layout=layout,
            lambdas=lambdas,
            face=face,
            evaluation=None if penalty_memo is None else penalty_memo.get(layout, lambdas, face),
        )
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
        units=dict(config.lambda_units),
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
        name for name in estimated_names if lambdas[name] == config.upper(name)
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
        if lambdas[name] == config.lower(name) and raw_log_steps[name] < -config.tolerance
    )
