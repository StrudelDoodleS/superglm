"""Design-matrix building helpers.

Contains the private DM construction functions extracted from SuperGLM.model:
R_inv computation, discretization helpers, feature auto-detection, interaction
dispatch, and the main design-matrix builder.

All functions take explicit state (specs, feature_order, lambda2, etc.) rather
than accessing ``self``, making them independently testable and keeping
model.py focused on orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.distributions import Distribution, resolve_distribution
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    GroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.types import FeatureSpec, GroupInfo, GroupSlice

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Pure computation helpers
# ═══════════════════════════════════════════════════════════════════


def _resolve_lambda2(lambda2: float | dict) -> float:
    """Resolve lambda2 to a scalar (dict → 1.0 fallback for initial basis)."""
    if isinstance(lambda2, dict):
        return 1.0
    return lambda2


def compute_R_inv(
    B: sp.spmatrix | NDArray,
    omega: NDArray,
    exposure: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv."""
    lam2 = _resolve_lambda2(lambda2)
    if sp.issparse(B):
        G = np.asarray((B.multiply(exposure[:, None]).T @ B).todense()) / np.sum(exposure)
    else:
        G = (B * exposure[:, None]).T @ B / np.sum(exposure)
    M = G + lam2 * omega + np.eye(omega.shape[0]) * 1e-8
    R = np.linalg.cholesky(M).T
    return np.linalg.inv(R)


def compute_projected_R_inv(
    B: sp.spmatrix | NDArray,
    projection: NDArray,
    penalty_sub: NDArray,
    exposure: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP R_inv within a projected subspace (linear-split range space)."""
    lam2 = _resolve_lambda2(lambda2)
    if sp.issparse(B):
        G_full = np.asarray((B.multiply(exposure[:, None]).T @ B).todense()) / np.sum(exposure)
    else:
        G_full = (B * exposure[:, None]).T @ B / np.sum(exposure)
    G_sub = projection.T @ G_full @ projection
    n_sub = penalty_sub.shape[0]
    M_sub = G_sub + lam2 * penalty_sub + np.eye(n_sub) * 1e-8
    R_sub = np.linalg.cholesky(M_sub).T
    return np.linalg.inv(R_sub)


def should_discretize(spec: FeatureSpec, model_discrete: bool) -> bool:
    """Check if a feature spec should use fit-time discretization."""
    from superglm.features.spline import _SplineBase

    if not isinstance(spec, _SplineBase):
        return False
    if spec.penalty != "ssp":
        return False
    if spec.discrete is not None:
        return spec.discrete
    return model_discrete


def should_discretize_tensor_interaction(
    ispec: Any, specs: dict[str, FeatureSpec], model_discrete: bool
) -> bool:
    """Check if a tensor interaction should use fit-time discretization."""
    from superglm.features.interaction import TensorInteraction

    if not isinstance(ispec, TensorInteraction):
        return False
    p1, p2 = ispec.parent_names
    return should_discretize(specs[p1], model_discrete) and should_discretize(
        specs[p2], model_discrete
    )


def resolve_discrete_n_bins(
    name: str, spec: FeatureSpec, n_bins_config: int | dict[str, int]
) -> int:
    """Resolve the requested bin count for a discretized feature.

    Feature-level ``spec.n_bins`` takes priority. Otherwise the model-level
    ``n_bins_config`` may be a single int or a per-feature dict with a fallback
    of 256 for unspecified features.
    """
    n_bins = getattr(spec, "n_bins", None)
    if n_bins is None:
        if isinstance(n_bins_config, dict):
            n_bins = n_bins_config.get(name, 256)
        else:
            n_bins = n_bins_config

    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError(f"n_bins for feature '{name}' must be >= 1, got {n_bins}")
    return n_bins


# ═══════════════════════════════════════════════════════════════════
# Feature auto-detection and interaction dispatch
# ═══════════════════════════════════════════════════════════════════


def auto_detect_features(
    X: pd.DataFrame,
    exposure: NDArray | None,
    *,
    spline_cols: list[str],
    knots_map: dict[str, int],
    degree: int,
    categorical_base: str,
    standardize_numeric: bool,
    specs: dict[str, FeatureSpec],
    feature_order: list[str],
) -> None:
    """Auto-detect feature types from DataFrame columns.

    Mutates ``specs`` and ``feature_order`` in place.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.spline import BasisSpline

    lines = ["SuperGLM features:"]
    for col in X.columns:
        if col in spline_cols:
            nk = knots_map[col]
            spec = BasisSpline(n_knots=nk, degree=degree, penalty="ssp")
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {col:<20s} → Spline(n_knots={nk}, degree={degree})")
        elif X[col].dtype.kind in ("O", "U") or isinstance(X[col].dtype, pd.CategoricalDtype):
            base = categorical_base
            if base == "most_exposed" and exposure is None:
                base = "first"
            spec = Categorical(base=base)
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {col:<20s} → Categorical(base={base})")
        else:
            spec = Numeric(standardize=standardize_numeric)
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {col:<20s} → Numeric(standardize={standardize_numeric})")
    logger.info("\n".join(lines))


def _spec_kind(spec: FeatureSpec) -> str:
    """Classify a feature spec into one of the four canonical kinds."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    if isinstance(spec, _SplineBase):
        return "spline"
    if isinstance(spec, Polynomial):
        return "polynomial"
    if isinstance(spec, Numeric):
        return "numeric"
    if isinstance(spec, Categorical):
        return "categorical"
    return type(spec).__name__


# ── Interaction factories ─────────────────────────────────────────
# Each returns (iname, ispec) given (feat1, feat2, name, **kwargs).
# For asymmetric pairs the factory receives the canonical orientation
# (e.g. spline first, categorical second).


def _make_spline_categorical(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import SplineCategorical

    return name or f"{f1}:{f2}", SplineCategorical(f1, f2)


def _make_polynomial_categorical(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import PolynomialCategorical

    return name or f"{f1}:{f2}", PolynomialCategorical(f1, f2)


def _make_numeric_categorical(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import NumericCategorical

    return name or f"{f1}:{f2}", NumericCategorical(f1, f2)


def _make_categorical_interaction(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import CategoricalInteraction

    return name or f"{f1}:{f2}", CategoricalInteraction(f1, f2)


def _make_numeric_interaction(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import NumericInteraction

    return name or f"{f1}:{f2}", NumericInteraction(f1, f2)


def _make_polynomial_interaction(
    f1: str, f2: str, *, name: str | None, **kw: Any
) -> tuple[str, Any]:
    from superglm.features.interaction import PolynomialInteraction

    return name or f"{f1}:{f2}", PolynomialInteraction(f1, f2)


def _make_tensor_interaction(f1: str, f2: str, *, name: str | None, **kw: Any) -> tuple[str, Any]:
    from superglm.features.interaction import TensorInteraction

    return name or f"{f1}:{f2}", TensorInteraction(f1, f2, **kw)


_INTERACTION_FACTORIES: dict[tuple[str, str], Any] = {
    ("spline", "categorical"): _make_spline_categorical,
    ("polynomial", "categorical"): _make_polynomial_categorical,
    ("numeric", "categorical"): _make_numeric_categorical,
    ("categorical", "categorical"): _make_categorical_interaction,
    ("numeric", "numeric"): _make_numeric_interaction,
    ("polynomial", "polynomial"): _make_polynomial_interaction,
    ("spline", "spline"): _make_tensor_interaction,
}


def add_interaction(
    feat1: str,
    feat2: str,
    specs: dict[str, FeatureSpec],
    interaction_specs: dict[str, Any],
    interaction_order: list[str],
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """Register an interaction between two already-registered features.

    Mutates ``interaction_specs`` and ``interaction_order`` in place.
    """
    if feat1 not in specs:
        raise ValueError(f"Parent feature not found: {feat1}")
    if feat2 not in specs:
        raise ValueError(f"Parent feature not found: {feat2}")

    kind1 = _spec_kind(specs[feat1])
    kind2 = _spec_kind(specs[feat2])

    factory = _INTERACTION_FACTORIES.get((kind1, kind2))
    if factory is not None:
        iname, ispec = factory(feat1, feat2, name=name, **kwargs)
    else:
        # Try swapped orientation (asymmetric pairs like categorical+spline)
        factory = _INTERACTION_FACTORIES.get((kind2, kind1))
        if factory is not None:
            iname, ispec = factory(feat2, feat1, name=name, **kwargs)
        else:
            raise TypeError(
                f"Cannot create interaction between {kind1} "
                f"and {kind2}. Supported: {', '.join('+'.join(k) for k in _INTERACTION_FACTORIES)}."
            )

    if iname in interaction_specs:
        raise ValueError(f"Interaction already added: {iname}")

    interaction_specs[iname] = ispec
    interaction_order.append(iname)


# ═══════════════════════════════════════════════════════════════════
# Design-matrix builder
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BuildResult:
    """Return value of build_design_matrix."""

    dm: DesignMatrix
    groups: list[GroupSlice]
    distribution: Distribution
    link: Link
    y: NDArray
    exposure: NDArray
    offset: NDArray | None


def build_design_matrix(
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None,
    offset: NDArray | None,
    *,
    family: str | Distribution,
    link_spec: str | Link | None,
    nb_theta: float | str | None,
    tweedie_p: float | None,
    specs: dict[str, FeatureSpec],
    feature_order: list[str],
    interaction_specs: dict[str, Any],
    interaction_order: list[str],
    pending_interactions: list[tuple[str, str]],
    model_discrete: bool,
    n_bins_config: int | dict[str, int],
    lambda2: float | dict,
) -> BuildResult:
    """Build features, groups, and design matrix from specs.

    Returns a BuildResult. Mutates ``interaction_specs``,
    ``interaction_order`` (resolves pending), ``pending_interactions``
    (empties it), and ``specs`` (via set_reparametrisation calls).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    resolved_nb_theta = nb_theta if isinstance(nb_theta, int | float) else None
    distribution = resolve_distribution(family, tweedie_p=tweedie_p, nb_theta=resolved_nb_theta)
    link = resolve_link(link_spec, distribution)

    group_matrices: list[GroupMatrix] = []
    col_offset = 0
    groups: list[GroupSlice] = []

    for name in feature_order:
        spec = specs[name]
        x_col = np.asarray(X[name])

        # Check if this feature should use fit-time discretization
        use_discrete = should_discretize(spec, model_discrete)
        B_unique = None
        bin_idx = None
        exposure_agg = None

        if use_discrete:
            from scipy.interpolate import BSpline as BSpl

            omega, n_cols_penalty, projection_penalty = spec.build_knots_and_penalty(x_col)
            n_bins_feat = resolve_discrete_n_bins(name, spec, n_bins_config)
            bin_centers, bin_idx = _discretize_column(x_col, n_bins_feat)
            bin_centers_clip = np.clip(bin_centers, spec._knots[0], spec._knots[-1])
            B_unique = BSpl.design_matrix(bin_centers_clip, spec._knots, spec.degree).toarray()
            exposure_agg = np.bincount(bin_idx, weights=exposure, minlength=len(bin_centers))

            if getattr(spec, "supports_linear_split", False) and getattr(
                spec, "split_linear", False
            ):
                n_null = 1
                n_range = spec._U_range.shape[1]
                infos = [
                    GroupInfo(
                        columns=None,
                        n_cols=n_null,
                        penalty_matrix=np.eye(n_null),
                        reparametrize=False,
                        penalized=True,
                        subgroup_name="linear",
                        projection=spec._U_null,
                    ),
                    GroupInfo(
                        columns=None,
                        n_cols=n_range,
                        penalty_matrix=spec._omega_range,
                        reparametrize=True,
                        subgroup_name="spline",
                        projection=spec._U_range,
                    ),
                ]
            else:
                infos = [
                    GroupInfo(
                        columns=None,
                        n_cols=n_cols_penalty,
                        penalty_matrix=omega,
                        reparametrize=(spec.penalty == "ssp"),
                        projection=projection_penalty,
                    )
                ]
        else:
            result = spec.build(x_col, exposure=exposure)
            infos = result if isinstance(result, list) else [result]

        # Collect per-subgroup R_inv columns to set combined R_inv on the spec
        r_inv_parts: list[NDArray] = []

        for info in infos:
            if info.projection is not None:
                P = info.projection

                if info.reparametrize and info.penalty_matrix is not None:
                    B_for_rinv = B_unique if use_discrete else info.columns
                    exp_for_rinv = exposure_agg if use_discrete else exposure
                    R_inv_local = compute_projected_R_inv(
                        B_for_rinv, P, info.penalty_matrix, exp_for_rinv, lambda2
                    )
                    R_inv_combined = P @ R_inv_local
                else:
                    R_inv_combined = P

                r_inv_parts.append(R_inv_combined)

                if use_discrete:
                    gm: GroupMatrix = DiscretizedSSPGroupMatrix(B_unique, R_inv_combined, bin_idx)
                    if info.penalty_matrix is not None:
                        gm.omega = P @ info.penalty_matrix @ P.T
                    gm.projection = P
                elif sp.issparse(info.columns):
                    gm = SparseSSPGroupMatrix(info.columns, R_inv_combined)
                    if info.penalty_matrix is not None:
                        gm.omega = P @ info.penalty_matrix @ P.T
                    gm.projection = P
                else:
                    gm = DenseGroupMatrix(info.columns @ R_inv_combined)

            elif info.reparametrize and info.penalty_matrix is not None:
                if use_discrete:
                    R_inv = compute_R_inv(B_unique, info.penalty_matrix, exposure_agg, lambda2)
                    if hasattr(spec, "set_reparametrisation"):
                        spec.set_reparametrisation(R_inv)
                    gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
                    gm.omega = info.penalty_matrix
                else:
                    R_inv = compute_R_inv(info.columns, info.penalty_matrix, exposure, lambda2)
                    if hasattr(spec, "set_reparametrisation"):
                        spec.set_reparametrisation(R_inv)
                    if sp.issparse(info.columns):
                        gm = SparseSSPGroupMatrix(info.columns, R_inv)
                        gm.omega = info.penalty_matrix
                    else:
                        gm = DenseGroupMatrix(info.columns @ R_inv)
            elif sp.issparse(info.columns):
                gm = SparseGroupMatrix(info.columns)
            else:
                gm = DenseGroupMatrix(info.columns)

            group_matrices.append(gm)
            subgroup_suffix = f":{info.subgroup_name}" if info.subgroup_name else ""
            group_name = f"{name}{subgroup_suffix}"
            weight = np.sqrt(info.n_cols)
            groups.append(
                GroupSlice(
                    name=group_name,
                    start=col_offset,
                    end=col_offset + info.n_cols,
                    weight=weight,
                    penalized=info.penalized,
                    feature_name=name,
                    subgroup_type=info.subgroup_name,
                )
            )
            col_offset += info.n_cols

        # Set combined R_inv on spec for transform/reconstruct
        if r_inv_parts and hasattr(spec, "set_reparametrisation"):
            spec.set_reparametrisation(np.hstack(r_inv_parts))

    # ── Interactions ──────────────────────────────────────────
    # Resolve pending interactions from constructor
    for pair in pending_interactions:
        if f"{pair[0]}:{pair[1]}" not in interaction_specs and (
            f"{pair[1]}:{pair[0]}" not in interaction_specs
        ):
            add_interaction(pair[0], pair[1], specs, interaction_specs, interaction_order)
    pending_interactions.clear()

    for iname in interaction_order:
        ispec = interaction_specs[iname]
        p1, p2 = ispec.parent_names
        x1 = np.asarray(X[p1])
        x2 = np.asarray(X[p2])
        use_discrete_tensor = should_discretize_tensor_interaction(ispec, specs, model_discrete)
        B_unique_inter = None
        bin_idx_inter = None
        exposure_agg_inter = None
        if use_discrete_tensor:
            n_bins1 = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            n_bins2 = resolve_discrete_n_bins(p2, specs[p2], n_bins_config)
            result, B_unique_inter, bin_idx_inter = ispec.build_discrete(
                x1,
                x2,
                specs,
                (n_bins1, n_bins2),
                exposure=exposure,
            )
            exposure_agg_inter = np.bincount(
                bin_idx_inter,
                weights=exposure,
                minlength=B_unique_inter.shape[0],
            )
        else:
            result = ispec.build(x1, x2, specs, exposure=exposure)

        if isinstance(result, list):
            has_subgroups = any(
                info.subgroup_name is not None or info.projection is not None for info in result
            )
            if has_subgroups:
                r_inv_parts_i: list[NDArray] = []
                for info in result:
                    if info.projection is not None:
                        P = info.projection
                        if info.reparametrize and info.penalty_matrix is not None:
                            B_for_rinv = B_unique_inter if use_discrete_tensor else info.columns
                            exp_for_rinv = exposure_agg_inter if use_discrete_tensor else exposure
                            R_inv_local = compute_projected_R_inv(
                                B_for_rinv, P, info.penalty_matrix, exp_for_rinv, lambda2
                            )
                            R_inv_combined = P @ R_inv_local
                        else:
                            R_inv_combined = P
                        r_inv_parts_i.append(R_inv_combined)

                        if use_discrete_tensor:
                            gm = DiscretizedSSPGroupMatrix(
                                B_unique_inter, R_inv_combined, bin_idx_inter
                            )
                            if info.penalty_matrix is not None:
                                gm.omega = P @ info.penalty_matrix @ P.T
                            gm.projection = P
                        elif sp.issparse(info.columns):
                            gm = SparseSSPGroupMatrix(info.columns, R_inv_combined)
                            if info.penalty_matrix is not None:
                                gm.omega = P @ info.penalty_matrix @ P.T
                            gm.projection = P
                        else:
                            gm = DenseGroupMatrix(info.columns @ R_inv_combined)
                        n_cols = R_inv_combined.shape[1]
                    elif info.reparametrize and info.penalty_matrix is not None:
                        B_for_rinv = B_unique_inter if use_discrete_tensor else info.columns
                        exp_for_rinv = exposure_agg_inter if use_discrete_tensor else exposure
                        R_inv = compute_R_inv(
                            B_for_rinv, info.penalty_matrix, exp_for_rinv, lambda2
                        )
                        r_inv_parts_i.append(R_inv)
                        n_cols = R_inv.shape[1]
                        if use_discrete_tensor:
                            gm = DiscretizedSSPGroupMatrix(B_unique_inter, R_inv, bin_idx_inter)
                            gm.omega = info.penalty_matrix
                        elif sp.issparse(info.columns):
                            gm = SparseSSPGroupMatrix(info.columns, R_inv)
                            gm.omega = info.penalty_matrix
                        else:
                            gm = DenseGroupMatrix(info.columns @ R_inv)
                    else:
                        n_cols = info.n_cols
                        if use_discrete_tensor:
                            gm = DiscretizedSSPGroupMatrix(
                                B_unique_inter,
                                np.eye(info.n_cols, dtype=np.float64),
                                bin_idx_inter,
                            )
                        elif sp.issparse(info.columns):
                            gm = SparseGroupMatrix(info.columns)
                        else:
                            gm = DenseGroupMatrix(info.columns)

                    group_matrices.append(gm)
                    subgroup_suffix = f":{info.subgroup_name}" if info.subgroup_name else ""
                    groups.append(
                        GroupSlice(
                            name=f"{iname}{subgroup_suffix}",
                            start=col_offset,
                            end=col_offset + n_cols,
                            weight=np.sqrt(n_cols),
                            penalized=info.penalized,
                            feature_name=iname,
                            subgroup_type=info.subgroup_name,
                        )
                    )
                    col_offset += n_cols

                if r_inv_parts_i and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(np.hstack(r_inv_parts_i))
            else:
                # Per-level groups (SplineCategorical, PolynomialCategorical)
                r_inv_dict: dict[str, NDArray] = {}
                for level, info in zip(ispec._non_base, result):
                    if info.reparametrize and info.penalty_matrix is not None:
                        R_inv = compute_R_inv(info.columns, info.penalty_matrix, exposure, lambda2)
                        r_inv_dict[level] = R_inv
                        n_cols = R_inv.shape[1]
                        if sp.issparse(info.columns):
                            gm = SparseSSPGroupMatrix(info.columns, R_inv)
                            gm.omega = info.penalty_matrix
                        else:
                            gm = DenseGroupMatrix(info.columns @ R_inv)
                    else:
                        n_cols = info.n_cols
                        if sp.issparse(info.columns):
                            gm = SparseGroupMatrix(info.columns)
                        else:
                            gm = DenseGroupMatrix(info.columns)

                    group_matrices.append(gm)
                    groups.append(
                        GroupSlice(
                            name=f"{iname}[{level}]",
                            start=col_offset,
                            end=col_offset + n_cols,
                            weight=np.sqrt(n_cols),
                            penalized=True,
                            feature_name=iname,
                        )
                    )
                    col_offset += n_cols

                if r_inv_dict and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(r_inv_dict)
        else:
            # Single group (CategoricalInteraction, NumericCategorical,
            # NumericInteraction, PolynomialInteraction, TensorInteraction)
            info = result
            if info.reparametrize and info.penalty_matrix is not None:
                B_for_rinv = B_unique_inter if use_discrete_tensor else info.columns
                exp_for_rinv = exposure_agg_inter if use_discrete_tensor else exposure
                R_inv = compute_R_inv(B_for_rinv, info.penalty_matrix, exp_for_rinv, lambda2)
                if hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(R_inv)
                n_cols = R_inv.shape[1]
                if use_discrete_tensor:
                    gm = DiscretizedSSPGroupMatrix(B_unique_inter, R_inv, bin_idx_inter)
                    gm.omega = info.penalty_matrix
                elif sp.issparse(info.columns):
                    gm = SparseSSPGroupMatrix(info.columns, R_inv)
                    gm.omega = info.penalty_matrix
                else:
                    gm = DenseGroupMatrix(info.columns @ R_inv)
            else:
                n_cols = info.n_cols
                if use_discrete_tensor:
                    gm = DiscretizedSSPGroupMatrix(
                        B_unique_inter,
                        np.eye(info.n_cols, dtype=np.float64),
                        bin_idx_inter,
                    )
                elif sp.issparse(info.columns):
                    gm = SparseGroupMatrix(info.columns)
                else:
                    gm = DenseGroupMatrix(info.columns)

            group_matrices.append(gm)
            groups.append(
                GroupSlice(
                    name=iname,
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(n_cols),
                    penalized=True,
                    feature_name=iname,
                )
            )
            col_offset += n_cols

    dm = DesignMatrix(group_matrices, n, col_offset)
    return BuildResult(
        dm=dm,
        groups=groups,
        distribution=distribution,
        link=link,
        y=y,
        exposure=exposure,
        offset=offset,
    )


# ═══════════════════════════════════════════════════════════════════
# Design-matrix rebuild with updated lambdas
# ═══════════════════════════════════════════════════════════════════


def rebuild_design_matrix_with_lambdas(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    lambdas: dict[str, float],
    exposure: NDArray,
    lambda2: float | dict,
) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas.

    Only recomputes R_inv for SSP groups whose lambda changed;
    non-SSP groups are reused unchanged.
    """
    new_gms: list[GroupMatrix] = []
    for gm, g in zip(dm.group_matrices, groups):
        if isinstance(gm, SparseSSPGroupMatrix) and g.name in lambdas:
            omega = gm.omega
            if omega is None:
                new_gms.append(gm)
                continue
            lam = lambdas[g.name]
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega @ P
                R_inv_local = compute_projected_R_inv(gm.B, P, omega_proj, exposure, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B, omega, exposure, lam)
            new_gm = SparseSSPGroupMatrix(gm.B, R_inv_new)
            new_gm.omega = omega
            new_gm.projection = gm.projection
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedSSPGroupMatrix) and g.name in lambdas:
            omega = gm.omega
            if omega is None:
                new_gms.append(gm)
                continue
            lam = lambdas[g.name]
            exposure_agg = np.bincount(gm.bin_idx, weights=exposure, minlength=gm.n_bins)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega @ P
                R_inv_local = compute_projected_R_inv(gm.B_unique, P, omega_proj, exposure_agg, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega, exposure_agg, lam)
            new_gm = DiscretizedSSPGroupMatrix(gm.B_unique, R_inv_new, gm.bin_idx)
            new_gm.omega = omega
            new_gm.projection = gm.projection
            new_gms.append(new_gm)
        else:
            new_gms.append(gm)
    return DesignMatrix(new_gms, dm.n, dm.p)
