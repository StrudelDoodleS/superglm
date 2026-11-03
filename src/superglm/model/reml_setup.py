"""Internal setup helpers for the REML fitting path."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from typing import Any

from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    FactorSmoothGroupMatrix,
    RandomEffectGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
)
from superglm.types import GroupSlice


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
    other_names: Iterable[str] = (),
) -> dict[str, float]:
    """Estimated components a mapping ``lambda2_init`` starts the REML search from.

    A mapping keyed by component or group name is a warm start; a scalar is
    an anchor the cold bootstrap may move away from. Unnamed and fixed-policy
    components stay cold. A penalty block starts warm only whole, since a
    half-warm block's penalty spread can defeat the bootstrap's
    log-determinant; a partly named block starts cold and an unknown key
    (``other_names`` adds the model's other terms) is ignored, each with a
    ``UserWarning``.
    """
    if not isinstance(lambda2_init, Mapping):
        return {}
    known = {name for pc in reml_penalties for name in (pc.name, pc.group_name)}
    known.update(other_names)
    unknown = sorted(str(key) for key in lambda2_init if key not in known)
    if unknown:
        warnings.warn(
            f"lambda2_init names no smoothing parameter of this model: {', '.join(unknown)}. "
            "Those entries are ignored. Use the names in "
            "model.reml_diagnostics()['lambdas'], or a group name.",
            UserWarning,
            stacklevel=5,
        )
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
    blocks: dict[str, list[str]] = {}
    for penalty_component in reml_penalties:
        if penalty_component.name in estimated_names:
            blocks.setdefault(penalty_component.group_name, []).append(penalty_component.name)
    half_warm = sorted(
        block
        for block, names in blocks.items()
        if any(name in warm for name in names) and not all(name in warm for name in names)
    )
    for block in half_warm:
        for name in blocks[block]:
            warm.pop(name, None)
    if half_warm:
        warnings.warn(
            f"lambda2_init names only some smoothing parameters of {', '.join(half_warm)}, "
            "so that term starts its smoothing search from the default start instead. To "
            "start it from lambda2_init, name every one of its components, or the term "
            "itself.",
            UserWarning,
            stacklevel=5,
        )
    return warm


def live_reml_lambdas(model: Any) -> dict[str, float]:
    """A fitted model's smoothing parameters, less those it left on a flat plateau.

    The warm start for a refit on similar data. A search started on a flat
    plateau cannot reach the new data's optimum, so components the Newton
    engines froze as flat, and a SCOP-era result's ``flat_components``, start
    cold. A penalty block (a tensor product's margins) starts cold whole when
    any of its components does: a half-warm block's penalty spread defeats
    the bootstrap's log-determinant.
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


def constraint_engine_flags(groups: list[GroupSlice]) -> tuple[bool, bool]:
    """Return whether any, and any QP, fit-time constrained groups are present."""
    has_any = False
    has_qp = False
    for group in groups:
        engine = group.monotone_engine
        if engine is None:
            continue
        has_any = True
        has_qp = has_qp or engine == "qp"
    return has_any, has_qp


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
    them with the current group matrix instead; a categorical's one-hot
    block has no ``R_inv``, is in raw coordinates and takes the raw rows
    unchanged.
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
        # A categorical's one-hot block has no SSP map: its columns are the raw
        # coefficients, so its raw rows are the current ones.
        group_matrix = dm.group_matrices[group_index]
        current_map = getattr(group_matrix, "R_inv", None)
        if current_map is None and not isinstance(group_matrix, CategoricalGroupMatrix):
            raise RuntimeError(
                f"cannot restore QP constraints for group {group.name!r}: "
                "its current solver-coordinate map is unavailable"
            )
        raw_constraints = raw_builder()
        current_constraints = (
            raw_constraints if current_map is None else raw_constraints.compose(current_map)
        )
        if current_constraints.n_params != group.size:
            raise RuntimeError(
                "restored QP constraint width does not match its current coefficient group"
            )
        group.constraints = current_constraints
