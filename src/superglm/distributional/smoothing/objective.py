"""The joint Laplace objective and the isolated Wood-Fasiolo update it drives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.layout import StackedLayout
from superglm.distributional.result import DenseSolverResult, DistributionalEFSConfig
from superglm.distributional.smoothing.endpoint_laml import evaluate_endpoint_laplace
from superglm.distributional.smoothing.penalty_face import PenaltyFace
from superglm.distributional.smoothing.proposals import _scaled_proposal
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult
from superglm.reml.penalty_algebra import (
    compute_logdet_s_derivatives,
    compute_logdet_s_plus,
    penalty_component_dense_matrix,
)
from superglm.types import LambdaPolicy


def joint_laplace_objective(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
) -> float:
    """Return the generic joint negative Laplace criterion, up to constants.

    This evaluates the objective used to safeguard EFS fixed-point proposals;
    it does not introduce a Newton/LAML smoothing optimizer and it does not
    profile a scalar dispersion.  A modeled scale remains part of ``result``'s
    observation-level joint likelihood.
    """
    if not isinstance(result, DenseSolverResult):
        raise TypeError("result must be a DenseSolverResult")
    if not isinstance(layout, StackedLayout):
        raise TypeError("layout must be a StackedLayout")
    if not isinstance(lambdas, Mapping):
        raise TypeError("lambdas must be a mapping")
    expected_penalty = layout.penalty_matrix(lambdas)
    if not np.array_equal(result.penalty, expected_penalty):
        raise ValueError("result penalty does not match layout and lambdas")
    penalty_log_pdet = compute_logdet_s_plus(
        dict(lambdas),
        list(layout.penalties),
    )
    assert result.penalized_optimizing_log_likelihood is not None
    objective = -result.penalized_optimizing_log_likelihood + 0.5 * (
        result.terminal_rank.log_pdet - penalty_log_pdet
    )
    if not math.isfinite(objective):
        raise ValueError("joint Laplace objective is non-finite")
    return float(objective)


def _laplace_objective(
    result: DenseSolverResult,
    *,
    layout: StackedLayout,
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
) -> float:
    if face is None:
        return joint_laplace_objective(result, layout=layout, lambdas=lambdas)
    return evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    ).objective


def _penalty_lambdas(
    lambdas: Mapping[str, float],
    face: PenaltyFace | None,
) -> dict[str, float]:
    values = dict(lambdas)
    if face is not None:
        for name in face.component_names:
            values[name] = 0.0
    return values


def initialize_distributional_lambdas(
    layout: StackedLayout,
    supplied: Mapping[str, float] | None,
    config: DistributionalEFSConfig,
) -> dict[str, float]:
    """Resolve qualified initial values while keeping ``LambdaPolicy`` authoritative."""
    values: Mapping[str, float] = {} if supplied is None else supplied
    if not isinstance(values, Mapping):
        raise TypeError("lambdas must be a qualified penalty-name mapping")
    unknown = set(values) - set(layout.penalty_names)
    if unknown:
        raise ValueError(f"unknown penalty lambda names: {sorted(unknown)}")

    resolved: dict[str, float] = {}
    for component in layout.penalties:
        policy = component.lambda_policy
        if policy is not None and policy.mode == "fixed":
            fixed_value = policy.value
            if fixed_value is None:  # Defensive against malformed deserialized policies.
                raise ValueError(f"fixed policy for {component.name!r} has no value")
            value = float(fixed_value)
        else:
            value = float(values.get(component.name, config.initial_lambda))
            if not config.minimum_lambda <= value <= config.maximum_lambda:
                raise ValueError(
                    f"initial lambda for {component.name!r} lies outside configured bounds"
                )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"lambda for {component.name!r} must be finite and non-negative")
        resolved[component.name] = value
    return resolved


def _component_states(
    layout: StackedLayout,
    lambdas: Mapping[str, float],
) -> tuple[EFSComponentState, ...]:
    effective_ranks, _ = compute_logdet_s_derivatives(
        dict(lambdas),
        list(layout.penalties),
    )
    states: list[EFSComponentState] = []
    for component in layout.penalties:
        # ``EFSComponentState`` demands a dense (width, width) block, but a
        # PenaltyComponent stores a compact representation whose expansion
        # depends on ``penalty_kind``: ``identity`` carries no array at all,
        # ``repeated`` a single diagonal block, ``sum_to_zero`` a raw-level
        # block needing the contrast.  Reading ``omega_ssp``/``omega_raw``
        # directly gets three of the four kinds wrong, so delegate to the one
        # expander that owns that algebra rather than restating it here.
        penalty = penalty_component_dense_matrix(component)
        rank = float(effective_ranks[component.name])
        policy = component.lambda_policy
        if policy is None:
            policy = LambdaPolicy.estimate()
        states.append(
            EFSComponentState(
                name=component.name,
                coefficient_slice=component.group_sl,
                penalty=penalty,
                rank=rank,
                lambda_value=lambdas[component.name],
                policy=policy,
            )
        )
    return tuple(states)


def _estimated_names(components: tuple[EFSComponentState, ...]) -> tuple[str, ...]:
    return tuple(component.name for component in components if component.policy.mode == "estimate")


def _slices_overlap(left: slice, right: slice) -> bool:
    return left.start < right.stop and right.start < left.stop


def _stable_isolated_gfs_update(
    components: tuple[EFSComponentState, ...],
    fit: DenseSolverResult,
    inverse: NDArray[np.float64],
    update: EFSUpdateResult,
    config: DistributionalEFSConfig,
) -> tuple[EFSUpdateResult, frozenset[str]]:
    """Replace cancellation-prone GFS quotients where an exact identity applies.

    For one isolated penalty S with range projector P and penalized curvature
    H = A + λS, cyclicity of the trace gives

        rank(S) - λ tr(H⁻¹S) = tr(P H⁻¹ A).

    The left side loses relative accuracy as λ grows because it subtracts two
    quantities tending to ``rank(S)``.  The right side evaluates the same
    positive residual degrees of freedom without that subtraction.  The
    identity is used only for a full-rank fit and a penalty whose coefficient
    slice overlaps no other penalty; shared-penalty blocks retain the generic
    update until they have an equally strong joint identity.
    """
    if fit.coefficient_face is not None or fit.terminal_rank.rank != len(fit.coefficients):
        return update, frozenset()

    data_curvature = fit.terminal_data_curvature
    penalized_curvature = fit.terminal_penalized_curvature
    epsilon = np.finfo(np.float64).eps
    stable_raw_steps: dict[str, float] = {}
    for component in components:
        name = component.name
        if update.proposal_kinds[name] != "gfs":
            continue
        if component.rank <= 0.0:
            continue
        # This is an algebra-selection gate, not an endpoint classifier.  The
        # direct numerator is well behaved until its trace term has absorbed
        # nearly the whole effective rank; changing formulas earlier would
        # perturb ordinary EFS trajectories for no numerical benefit.
        saturation = component.lambda_value * float(update.trace_terms[name]) / component.rank
        if saturation < 1.0 - math.sqrt(epsilon):
            continue
        if any(
            other.name != name
            and _slices_overlap(component.coefficient_slice, other.coefficient_slice)
            for other in components
        ):
            continue

        penalty = component.penalty
        try:
            values, vectors = np.linalg.eigh(penalty)
        except np.linalg.LinAlgError:
            continue
        penalty_scale = float(np.max(np.abs(values), initial=0.0))
        if penalty_scale == 0.0:
            continue
        width = len(values)
        cutoff = 64.0 * max(width, 1) * epsilon * penalty_scale
        active = values > cutoff
        numerical_rank = int(np.count_nonzero(active))
        rank_tolerance = (
            256.0
            * max(width, 1)
            * epsilon
            * max(
                1.0,
                abs(component.rank),
            )
        )
        if numerical_rank == 0 or abs(component.rank - numerical_rank) > rank_tolerance:
            continue
        projector = vectors[:, active] @ vectors[:, active].T
        block = component.coefficient_slice

        # Authenticate the projector identity before using its non-cancelling
        # form.  A truncated inverse or unresolved solve is not silently
        # promoted into convergence authority.
        identity_product = inverse[block, :] @ penalized_curvature[:, block]
        identity_trace = float(np.trace(projector @ identity_product))
        identity_scale = float(np.sum(np.abs(projector) * np.abs(identity_product.T)))
        identity_bound = (
            1024.0
            * max(len(fit.coefficients), 1)
            * epsilon
            * max(
                1.0,
                abs(component.rank),
                identity_scale,
            )
        )
        if abs(identity_trace - component.rank) > identity_bound:
            continue

        residual_product = inverse[block, :] @ data_curvature[:, block]
        numerator = float(np.trace(projector @ residual_product))
        numerator_scale = float(np.sum(np.abs(projector) * np.abs(residual_product.T)))
        numerator_bound = (
            1024.0
            * max(len(fit.coefficients), 1)
            * epsilon
            * max(
                numerator_scale,
                np.finfo(np.float64).tiny,
            )
        )
        quadratic = float(update.quadratic_forms[name])
        if numerator <= numerator_bound or quadratic <= 0.0:
            continue
        raw_step = math.log(numerator) - math.log(quadratic) - math.log(component.lambda_value)
        if math.isfinite(raw_step):
            stable_raw_steps[name] = raw_step

    if not stable_raw_steps:
        return update, frozenset()

    raw_steps = dict(update.raw_log_steps)
    raw_steps.update(stable_raw_steps)
    bounded_steps = {
        name: float(np.clip(raw_steps[name], -config.max_log_step, config.max_log_step))
        for name in raw_steps
    }
    proposed_lambdas, log_steps = _scaled_proposal(
        {component.name: component.lambda_value for component in components},
        bounded_steps,
        tuple(component.name for component in components if component.policy.mode == "estimate"),
        1.0,
        config,
    )
    return (
        replace(
            update,
            lambdas=proposed_lambdas,
            log_steps=log_steps,
            raw_log_steps=raw_steps,
        ),
        frozenset(stable_raw_steps),
    )


def _maximum_step(update: EFSUpdateResult, estimated_names: tuple[str, ...]) -> float:
    return max((abs(update.log_steps[name]) for name in estimated_names), default=0.0)


def _complete_mapping(
    names: Mapping[str, float],
    values: Mapping[str, float],
    *,
    missing: float | Mapping[str, float],
) -> dict[str, float]:
    defaults = missing if isinstance(missing, Mapping) else None
    completed: dict[str, float] = {}
    for name in names:
        if name in values:
            value = values[name]
        elif defaults is not None:
            value = defaults[name]
        else:
            value = cast(float, missing)
        completed[name] = float(value)
    return completed
