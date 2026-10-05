"""Internal setup helpers for the REML fitting path."""

from __future__ import annotations

import math
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


def _warm_value(name: str, value: Any) -> float:
    """One warm-start smoothing parameter, refused unless finite and positive."""
    try:
        lam = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"lambda2_init[{name!r}] must be a real number, got {value!r}") from exc
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"lambda2_init[{name!r}] must be finite and positive, got {lam!r}")
    return lam


def warm_start_lambdas(
    reml_penalties: list[Any],
    lambda2_init: Any,
    estimated_names: set[str],
) -> dict[str, float]:
    """Estimated components a mapping ``lambda2_init`` starts the REML search from.

    A mapping keyed by component name (or by group name, which then starts every
    component of that group) is a warm start: the engines bootstrap there instead
    of at their own cold seeds. A scalar ``lambda2_init`` is not a warm start --
    it keeps its historical meaning, an anchor the cold bootstrap may move away
    from -- so cold fits are unchanged. Names the mapping does not cover stay
    cold; fixed-policy components are never started from it.
    """
    if not isinstance(lambda2_init, Mapping):
        return {}
    warm: dict[str, float] = {}
    for penalty_component in reml_penalties:
        name = penalty_component.name
        if name not in estimated_names:
            continue
        if name in lambda2_init:
            warm[name] = _warm_value(name, lambda2_init[name])
        elif penalty_component.group_name in lambda2_init:
            group_name = penalty_component.group_name
            warm[name] = _warm_value(group_name, lambda2_init[group_name])
    return warm


def live_reml_lambdas(model: Any) -> dict[str, float]:
    """A fitted model's smoothing parameters, less those it left on a flat plateau.

    The warm start for a refit on similar data. Two kinds of component start
    cold instead, because a search started where they ended cannot reach the
    new data's optimum:

    - one the Newton engines froze as inferentially flat, where the
      criterion's gradient and curvature vanish. Measured: a fold
      warm-started on a tensor margin's plateau converged to an objective
      1.05 worse, interaction EDF 1.00 against 2.10;
    - one a SCOP suppression hold covered at the final iterate
      (``REMLResult.flat_components``): a flat end of the criterion. Measured,
      under the earlier decrease hold on ``tr(H^-1 S_j)``: folds warm-started
      at the first fold's LogDensity lambda kept it exactly, term EDF 0.7 and
      1.2 below the cold folds, whose own optima were 4 and 8 times lower.

    A component that shares its penalty block with others (a tensor product's
    margins) starts cold with all of them when any one is left out: a half-warm
    block, one margin at the cold seed beside another at its fitted value, has
    a penalty whose spread the bootstrap's log-determinant cannot certify.
    Measured on the cleaned freMTPL2 book: every fold after a first fold that
    left the VehAge margin out failed with ``PenaltyNumericalError``.
    """
    reml = model._reml_result
    flat = set(getattr(reml, "flat_components", None) or ())
    decision = (getattr(model, "_reml_profile", None) or {}).get("reml_freeze_decision")
    if isinstance(decision, Mapping):
        names = decision.get("names", ())
        frozen = decision.get("frozen", ())
        flat.update(name for name, is_frozen in zip(names, frozen, strict=False) if is_frozen)
    penalties = getattr(model, "_reml_penalties", None) or getattr(reml, "reml_penalties", None)
    block_of = {pc.name: pc.group_name for pc in penalties or ()}
    cold_blocks = {block_of[name] for name in flat if name in block_of}
    return {
        str(k): float(v)
        for k, v in reml.lambdas.items()
        if k not in flat and block_of.get(k) not in cold_blocks
    }


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


def scop_group_spec(groupspecs: Mapping[Any, Any], group: GroupSlice) -> Any | None:
    """Return the feature spec backing a SCOP-constrained group."""
    return groupspecs.get(group.feature_name)


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
    default_lambda: float | Mapping[str, float],
) -> dict[str, float]:
    """Add unfixed SCOP-constrained groups to the estimated-lambda set.

    Returns the SCOP groups a mapping ``default_lambda`` warm-starts (see
    :func:`warm_start_lambdas`). A mapping without the group's name leaves it
    at the scalar default the SCOP engine seeds cold anyway; the value stored
    is always a float, never the mapping itself.
    """
    warm: dict[str, float] = {}
    for group in groups:
        if group.monotone_engine != "scop" or not group.penalized:
            continue
        spec = scop_group_spec(specs, group)
        fixed_value = scop_fixed_lambda_value(spec)
        if fixed_value is not None:
            continue
        estimated_names.add(group.name)
        if isinstance(default_lambda, Mapping):
            if group.name in default_lambda:
                warm[group.name] = _warm_value(group.name, default_lambda[group.name])
                lambdas[group.name] = warm[group.name]
            else:
                lambdas[group.name] = 0.1
        else:
            lambdas[group.name] = default_lambda
    return warm


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
