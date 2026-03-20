"""Constructor, prediction, and core helpers for SuperGLM."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.distributions import Distribution, clip_mu
from superglm.dm_builder import (
    add_interaction,
    auto_detect_features,
    build_design_matrix,
    rebuild_design_matrix_with_lambdas,
)
from superglm.group_matrix import DesignMatrix
from superglm.links import Link, stabilize_eta
from superglm.penalties.base import (
    Penalty,
    penalty_has_targets,
    penalty_targets_group,
    validate_penalty_features,
)
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.solvers.pirls import PIRLSResult
from superglm.types import FeatureSpec, FitStats, GroupSlice

logger = logging.getLogger(__name__)

_PENALTY_SHORTCUTS: dict[str, type[Penalty]] = {
    "group_lasso": GroupLasso,
    "group_elastic_net": GroupElasticNet,
    "sparse_group_lasso": SparseGroupLasso,
    "ridge": Ridge,
}


def resolve_penalty(
    penalty: Penalty | str | None,
    lambda1: float | None,
    penalty_features: str | list[str] | None = None,
) -> Penalty:
    """Convert string shorthand / None to a Penalty object."""
    if penalty is None:
        return GroupLasso(lambda1=lambda1, features=penalty_features)
    if isinstance(penalty, str):
        if penalty not in _PENALTY_SHORTCUTS:
            raise ValueError(
                f"Unknown penalty '{penalty}'. "
                f"Use one of {list(_PENALTY_SHORTCUTS)} or pass a Penalty object."
            )
        return _PENALTY_SHORTCUTS[penalty](lambda1=lambda1, features=penalty_features)
    if lambda1 is not None:
        raise ValueError(
            "Cannot set 'selection_penalty' when passing a Penalty object directly. "
            "Set lambda1 on the Penalty object instead."
        )
    if penalty_features is not None:
        raise ValueError(
            "Cannot set 'penalty_features' when passing a Penalty object directly. "
            "Set features on the Penalty object instead."
        )
    return penalty


def resolve_knots(model, spline_cols: list[str]) -> dict[str, int]:
    """Map spline column names to their n_knots values."""
    if not spline_cols:
        return {}
    if isinstance(model._n_knots, int):
        return {col: model._n_knots for col in spline_cols}
    if len(model._n_knots) != len(spline_cols):
        raise ValueError(
            f"n_knots has length {len(model._n_knots)} but splines "
            f"has length {len(spline_cols)}. Must match or pass a single int."
        )
    return dict(zip(spline_cols, model._n_knots))


def resolve_sample_weight_alias(
    exposure: NDArray | None,
    sample_weight: NDArray | None,
    *,
    method_name: str,
) -> NDArray | None:
    """Resolve the public sample_weight alias for exposure/frequency weights."""
    if exposure is not None and sample_weight is not None:
        raise TypeError(
            f"{method_name} received both 'exposure' and 'sample_weight'. "
            "Use only 'sample_weight'; 'exposure' is a backward-compatible alias."
        )
    return sample_weight if sample_weight is not None else exposure


def init_model(
    model,
    family: str | Distribution = "poisson",
    link: str | Link | None = None,
    penalty: Penalty | str | None = None,
    lambda1: float | None = None,
    lambda2: float = 0.1,
    penalty_features: str | list[str] | None = None,
    features: dict[str, FeatureSpec] | None = None,
    splines: list[str] | None = None,
    n_knots: int | list[int] = 10,
    degree: int = 3,
    categorical_base: str = "most_exposed",
    interactions: list[tuple[str, str]] | None = None,
    active_set: bool = False,
    direct_solve: str = "auto",
    discrete: bool = False,
    n_bins: int | dict[str, int] = 256,
):
    """Initialize model state (body of SuperGLM.__init__)."""
    if features is not None and splines is not None:
        raise ValueError(
            "Cannot set both 'features' and 'splines'. "
            "Use 'features' for explicit specs or 'splines' for auto-detect."
        )
    model.family = family
    model.link = link
    model.penalty = resolve_penalty(penalty, lambda1, penalty_features)
    model.lambda2 = lambda2
    model._splines = splines
    model._n_knots = n_knots
    model._degree = degree
    model._categorical_base = categorical_base
    model._active_set = active_set
    if direct_solve not in ("auto", "gram", "qr"):
        raise ValueError(f"direct_solve must be 'auto', 'gram', or 'qr', got {direct_solve!r}")
    model._direct_solve = direct_solve
    model._discrete = discrete
    model._n_bins = n_bins

    model._specs: dict[str, FeatureSpec] = {}
    model._feature_order: list[str] = []
    model._groups: list[GroupSlice] = []
    model._distribution: Distribution | None = None
    model._link: Link | None = None
    model._result: PIRLSResult | None = None
    model._dm: DesignMatrix | None = None
    model._fit_weights: NDArray | None = None
    model._fit_offset: NDArray | None = None
    model._fit_stats: FitStats | None = None
    model._nb_profile_result = None
    model._tweedie_profile_result = None
    model._last_fit_meta: dict[str, Any] | None = None
    model._monotone_repairs: dict = {}

    # Interaction support
    model._interaction_specs: dict[str, Any] = {}
    model._interaction_order: list[str] = []
    model._pending_interactions: list[tuple[str, str]] = interactions or []

    # Register explicit features dict
    if features is not None:
        for name, spec in features.items():
            model._specs[name] = spec
            model._feature_order.append(name)


def clone_without_features(
    model,
    drop: set[str],
    *,
    lambda1: float | None = ...,  # sentinel: ... means "keep current"
    lambda2: float | dict[str, float] | None = ...,
):
    """Create a new SuperGLM with a subset of features removed.

    Copies family, link, penalty type, and solver options. Interactions
    whose parents include a dropped feature are also removed.
    """
    keep_features = {n: s for n, s in model._specs.items() if n not in drop}

    # Filter interactions: drop any whose parent is being dropped
    keep_interactions: list[tuple[str, str]] = []
    # Check resolved interactions (fitted model)
    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        p1, p2 = ispec.parent_names
        if p1 not in drop and p2 not in drop:
            keep_interactions.append((p1, p2))
    # Check pending interactions (unfitted model)
    for p1, p2 in model._pending_interactions:
        if p1 not in drop and p2 not in drop:
            keep_interactions.append((p1, p2))

    # Resolve lambda1
    if lambda1 is ...:
        lam1 = model.penalty.lambda1
    else:
        lam1 = lambda1

    new_penalty = copy.deepcopy(model.penalty)
    new_penalty.lambda1 = lam1

    # Deep-copy specs so the new model doesn't share mutable state
    new_features = {n: copy.deepcopy(s) for n, s in keep_features.items()}

    new_model = type(model)(
        family=model.family,
        link=model.link,
        penalty=new_penalty,
        features=new_features,
        interactions=keep_interactions if keep_interactions else None,
        active_set=model._active_set,
        direct_solve=model._direct_solve,
        discrete=model._discrete,
        n_bins=model._n_bins,
    )

    # Resolve lambda2
    if lambda2 is ...:
        reml_lam = getattr(model, "_reml_lambdas", None)
        if reml_lam is not None:
            # Filter REML lambdas to remaining groups
            new_model.lambda2 = {
                k: v
                for k, v in reml_lam.items()
                if not any(k == d or k.startswith(f"{d}:") for d in drop)
            }
        else:
            new_model.lambda2 = model.lambda2
    elif lambda2 is None:
        new_model.lambda2 = 0.0
    else:
        new_model.lambda2 = lambda2

    return new_model


def auto_detect(model, X: pd.DataFrame, exposure: NDArray | None) -> None:
    """Auto-detect feature types from DataFrame columns."""
    spline_cols = model._splines or []
    knots_map = resolve_knots(model, spline_cols)
    auto_detect_features(
        X,
        exposure,
        spline_cols=spline_cols,
        knots_map=knots_map,
        degree=model._degree,
        categorical_base=model._categorical_base,
        specs=model._specs,
        feature_order=model._feature_order,
    )


def model_add_interaction(model, feat1: str, feat2: str, name: str | None = None, **kwargs) -> None:
    """Register an interaction between two already-registered features."""
    add_interaction(
        feat1,
        feat2,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        interaction_order=model._interaction_order,
        name=name,
        **kwargs,
    )


def model_build_design_matrix(
    model,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray,
    offset: NDArray | None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Build features, groups, design matrix.

    Sets model._dm, model._groups, model._distribution, model._link.
    Returns (y, exposure, offset) as float64 arrays.
    """
    result = build_design_matrix(
        X,
        y,
        exposure,
        offset,
        family=model.family,
        link_spec=model.link,
        specs=model._specs,
        feature_order=model._feature_order,
        interaction_specs=model._interaction_specs,
        interaction_order=model._interaction_order,
        pending_interactions=model._pending_interactions,
        model_discrete=model._discrete,
        n_bins_config=model._n_bins,
        lambda2=model.lambda2,
    )
    model._distribution = result.distribution
    model._link = result.link
    model._groups = result.groups
    validate_penalty_features(model.penalty, result.groups)
    model._dm = result.dm
    return result.y, result.exposure, result.offset


def compute_lambda_max(model, y, weights):
    """Smallest lambda1 at which all groups are zeroed (null model)."""
    from superglm.distributions import initial_mean

    mu_null = initial_mean(y, weights, model._distribution)
    residual = weights * (y - mu_null)
    grad = model._dm.rmatvec(residual)
    n = model._dm.n
    lmax = 0.0
    for g in model._groups:
        if not penalty_targets_group(model.penalty, g):
            continue
        lmax = max(lmax, np.linalg.norm(grad[g.sl]) / g.weight)
    return lmax / n


def model_has_lambda1_targets(model) -> bool:
    """Whether the lambda1 penalty applies to any fitted group."""
    return penalty_has_targets(model.penalty, model._groups)


def rebuild_dm_with_lambdas(model, lambdas: dict[str, float], exposure: NDArray) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas."""
    return rebuild_design_matrix_with_lambdas(
        model._dm, model._groups, lambdas, exposure, model.lambda2
    )


def predict(model, X: pd.DataFrame, offset: NDArray | None = None) -> NDArray:
    """Predict the response mean for new data."""
    blocks = []
    for name in model._feature_order:
        spec = model._specs[name]
        blocks.append(spec.transform(np.asarray(X[name])))

    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        p1, p2 = ispec.parent_names
        blocks.append(ispec.transform(np.asarray(X[p1]), np.asarray(X[p2])))

    eta = np.hstack(blocks) @ model.result.beta + model.result.intercept
    if offset is not None:
        eta = eta + np.asarray(offset, dtype=np.float64)
    eta = stabilize_eta(eta, model._link)
    return clip_mu(model._link.inverse(eta), model._distribution)
