"""Internal setup helpers for the REML fitting path."""

from __future__ import annotations

from typing import Any

from superglm.group_matrix import DiscretizedSSPGroupMatrix, SparseSSPGroupMatrix
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
            isinstance(group_matrix, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix)
            and group.penalized
            and group_matrix.omega is not None
        ):
            reml_groups.append((i, group))
    return reml_groups


def initialize_component_lambdas(
    reml_penalties: list[Any],
    default_lambda: float,
) -> tuple[dict[str, float], set[str]]:
    """Seed the REML lambda dict from per-component policies."""
    lambdas: dict[str, float] = {}
    estimated_names: set[str] = set()
    for penalty_component in reml_penalties:
        lambda_policy = penalty_component.lambda_policy
        if lambda_policy is not None and lambda_policy.mode == "fixed":
            lambdas[penalty_component.name] = float(lambda_policy.value)
            continue
        lambdas[penalty_component.name] = default_lambda
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


def inject_fixed_scop_lambdas(
    groups: list[GroupSlice],
    specs: dict[str, Any],
    lambdas: dict[str, float],
) -> bool:
    """Inject fixed SCOP lambdas and report whether any remain unfixed."""
    any_unfixed_scop = False
    for group in groups:
        if group.monotone_engine != "scop" or not group.penalized:
            continue
        spec = specs.get(group.feature_name)
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
    """Add unfixed SCOP groups to the estimated-lambda set."""
    for group in groups:
        if group.monotone_engine != "scop" or not group.penalized:
            continue
        spec = specs.get(group.feature_name or group.name)
        fixed_value = scop_fixed_lambda_value(spec)
        if fixed_value is not None:
            continue
        estimated_names.add(group.name)
        lambdas[group.name] = default_lambda


def monotone_flags(groups: list[GroupSlice]) -> tuple[bool, bool, bool]:
    """Return whether any, QP, or SCOP monotone groups are present."""
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
    """Temporarily disable QP monotone constraints for passthrough REML."""
    saved_state: list[tuple[int, Any, Any]] = []
    for group_index, group in enumerate(groups):
        if group.monotone_engine != "qp":
            continue
        saved_state.append((group_index, group.monotone_engine, group.constraints))
        group.monotone_engine = None
        group.constraints = None
    return saved_state


def restore_qp_constraints(
    groups: list[GroupSlice],
    saved_state: list[tuple[int, Any, Any]],
) -> None:
    """Restore QP monotone constraints after passthrough REML."""
    for group_index, monotone_engine, constraints in saved_state:
        groups[group_index].monotone_engine = monotone_engine
        groups[group_index].constraints = constraints
