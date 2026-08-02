"""Internal setup helpers for the REML fitting path."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from superglm.group_matrix import (
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.types import GroupSlice, LambdaPolicy


def collect_reml_groups(
    groups: list[GroupSlice],
    group_matrices: list[Any],
) -> list[tuple[int, GroupSlice]]:
    """Return REML-eligible penalized SSP groups."""
    reml_groups: list[tuple[int, GroupSlice]] = []
    for i, group in enumerate(groups):
        group_matrix = group_matrices[i]
        if (
            isinstance(
                group_matrix,
                RandomEffectGroupMatrix | FactorSmoothGroupMatrix,
            )
            and group.penalized
        ):
            reml_groups.append((i, group))
            continue
        if (
            isinstance(
                group_matrix,
                SparseSSPGroupMatrix
                | SplineCategoricalGroupMatrix
                | DiscretizedSplineCategoricalGroupMatrix
                | DiscretizedSSPGroupMatrix,
            )
            and group.penalized
            and group_matrix.omega is not None
        ):
            reml_groups.append((i, group))
    return reml_groups


def initialize_component_lambdas(
    reml_penalties: list[Any],
    default_lambda: float | Mapping[str, float],
) -> tuple[dict[str, float], set[str]]:
    """Seed the REML lambda dict from per-component policies."""
    lambdas: dict[str, float] = {}
    estimated_names: set[str] = set()
    for penalty_component in reml_penalties:
        lambda_policy = penalty_component.lambda_policy
        if lambda_policy is not None and lambda_policy.mode == "fixed":
            lambdas[penalty_component.name] = float(lambda_policy.value)
            continue
        if isinstance(default_lambda, Mapping):
            lam = default_lambda.get(
                penalty_component.name,
                default_lambda.get(penalty_component.group_name, 0.1),
            )
        else:
            lam = default_lambda
        lambdas[penalty_component.name] = float(lam)
        estimated_names.add(penalty_component.name)
    return lambdas, estimated_names


def scop_fixed_lambda_value(spec: Any) -> float | None:
    """Return a fixed SCOP lambda value, or None if it should be estimated."""
    lambda_policy = getattr(spec, "_lambda_policy", None)
    if lambda_policy is None:
        return None
    if isinstance(lambda_policy, LambdaPolicy):
        return float(lambda_policy.value) if lambda_policy.mode == "fixed" else None

    unknown = set(lambda_policy) - {"wiggle"}
    if unknown:
        raise ValueError(
            f"lambda_policy contains unknown component names: {unknown}. Valid names: ['wiggle']"
        )

    wiggle_policy = lambda_policy.get("wiggle", LambdaPolicy.estimate())
    return float(wiggle_policy.value) if wiggle_policy.mode == "fixed" else None


def scop_group_spec(groupspecs: dict[str, Any], group: GroupSlice) -> Any | None:
    """Return the feature spec backing a SCOP-constrained group."""
    return groupspecs.get(group.feature_name or group.name)


def inject_fixed_scop_lambdas(
    groups: list[GroupSlice],
    specs: dict[str, Any],
    lambdas: dict[str, float],
) -> bool:
    """Inject fixed lambdas for SCOP-constrained groups and report whether any remain unfixed."""
    any_unfixed_scop = False
    for group in groups:
        if group.monotone_engine != "scop" or not group.penalized:
            continue
        spec = scop_group_spec(specs, group)
        if spec is None:
            any_unfixed_scop = True
            continue
        fixed_value = scop_fixed_lambda_value(spec)
        if fixed_value is None:
            any_unfixed_scop = True
            continue
        lambdas[group.name] = fixed_value
    return any_unfixed_scop


def promote_estimated_scop_lambdas(
    groups: list[GroupSlice],
    specs: dict[str, Any],
    lambdas: dict[str, float],
    estimated_names: set[str],
    default_lambda: float,
) -> None:
    """Add unfixed SCOP-constrained groups to the estimated-lambda set."""
    for group in groups:
        if group.monotone_engine != "scop" or not group.penalized:
            continue
        spec = scop_group_spec(specs, group)
        fixed_value = scop_fixed_lambda_value(spec)
        if fixed_value is not None:
            continue
        estimated_names.add(group.name)
        lambdas[group.name] = default_lambda


def constraint_engine_flags(groups: list[GroupSlice]) -> tuple[bool, bool, bool]:
    """Return whether any, QP, or SCOP fit-time constrained groups are present."""
    has_any = False
    has_qp = False
    has_scop = False
    for group in groups:
        engine = group.monotone_engine
        if engine is None:
            continue
        has_any = True
        has_qp = has_qp or engine == "qp"
        has_scop = has_scop or engine == "scop"
    return has_any, has_qp, has_scop


def strip_qp_constraints(groups: list[GroupSlice]) -> list[tuple[int, Any, Any]]:
    """Temporarily disable QP fit-time constraints for passthrough REML."""
    saved_state: list[tuple[int, Any, Any]] = []
    for group_index, group in enumerate(groups):
        if group.monotone_engine != "qp":
            continue
        saved_state.append((group_index, group.monotone_engine, group.constraints))
        group.monotone_engine = None
        group.constraints = None
    return saved_state


def restore_qp_constraints(model, saved_state: list[tuple[int, Any, Any]]) -> None:
    """Restore QP constraints in the model's current solver coordinates.

    QP passthrough strips constraints before REML changes the SSP
    reparameterization. The saved matrix is therefore composed with the old
    ``R_inv`` and cannot be installed unchanged after the design is rebuilt.
    Rebuild spline constraints from their raw coefficient rows and compose
    them with the current group matrix instead.
    """
    groups = model._groups
    dm = getattr(model, "_dm", None)
    for group_index, monotone_engine, constraints in saved_state:
        group = groups[group_index]
        group.monotone_engine = monotone_engine

        # A successful retain_fit_state=False path has already restored the
        # current constraint matrix before releasing the design. Preserve it
        # rather than falling back to the stale saved coordinates in finally.
        if dm is None:
            if group.constraints is None:
                raise RuntimeError(
                    f"cannot restore QP constraints for group {group.name!r}: "
                    "its fitted design was released before current-coordinate "
                    "constraints were restored"
                )
            continue

        spec = model._specs.get(group.feature_name)
        if spec is None:
            spec = model._interaction_specs.get(group.feature_name)
        if spec is None:
            raise RuntimeError(
                f"cannot restore QP constraints for group {group.name!r}: "
                "its fitted spline specification is unavailable"
            )
        raw_builder = getattr(spec, "_build_monotone_constraints_raw", None)
        if raw_builder is None:
            raise RuntimeError(
                f"cannot restore QP constraints for group {group.name!r}: "
                "its raw constraint geometry is unavailable"
            )
        current_map = getattr(dm.group_matrices[group_index], "R_inv", None)
        if current_map is None:
            raise RuntimeError(
                f"cannot restore QP constraints for group {group.name!r}: "
                "its current solver-coordinate map is unavailable"
            )

        current_constraints = raw_builder().compose(current_map)
        if current_constraints.n_params != group.size:
            raise RuntimeError(
                "restored QP constraint width does not match its current coefficient group"
            )
        group.constraints = current_constraints
