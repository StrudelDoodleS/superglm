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
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    GroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.types import DiscreteTensorBuildResult, FeatureSpec, GroupInfo, GroupSlice

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
    sample_weight: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv.

    Wood (2011) Section 3.1 / Section 5: absorb penalty into parameterization.
    R = chol(B'WB/n + λΩ + εI)^T, then R_inv = R^{-1} so that the SSP basis
    X_ssp = B @ R_inv has near-identity X'WX regardless of λ.
    """
    lam2 = _resolve_lambda2(lambda2)
    if sp.issparse(B):
        G = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / np.sum(sample_weight)
    else:
        G = (B * sample_weight[:, None]).T @ B / np.sum(sample_weight)
    M = G + lam2 * omega + np.eye(omega.shape[0]) * 1e-8
    R = np.linalg.cholesky(M).T
    return np.linalg.inv(R)


def compute_projected_R_inv(
    B: sp.spmatrix | NDArray,
    projection: NDArray,
    penalty_sub: NDArray,
    sample_weight: NDArray,
    lambda2: float | dict,
) -> NDArray:
    """Compute SSP R_inv within a projected subspace (linear-split range space)."""
    lam2 = _resolve_lambda2(lambda2)
    if sp.issparse(B):
        G_full = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / np.sum(
            sample_weight
        )
    else:
        G_full = (B * sample_weight[:, None]).T @ B / np.sum(sample_weight)
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
    sample_weight: NDArray | None,
    *,
    spline_cols: list[str],
    knots_map: dict[str, int],
    degree: int,
    categorical_base: str,
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
            if base == "most_exposed" and sample_weight is None:
                base = "first"
            spec = Categorical(base=base)
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {col:<20s} → Categorical(base={base})")
        else:
            spec = Numeric()
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {col:<20s} → Numeric()")
    logger.info("\n".join(lines))


def _spec_kind(spec: FeatureSpec) -> str:
    """Classify a feature spec into one of the four canonical kinds."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
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
    if isinstance(spec, OrderedCategorical):
        return "spline" if spec.basis == "spline" else "categorical"
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


def _process_info(
    info: GroupInfo,
    *,
    B_unique: NDArray | None = None,
    bin_idx: NDArray | None = None,
    sample_weight: NDArray,
    exposure_agg: NDArray | None = None,
    lambda2: float | dict,
    tensor_build: DiscreteTensorBuildResult | None = None,
    tensor_id: int = -1,
) -> tuple[GroupMatrix, NDArray | None, int]:
    """Compute R_inv and construct a GroupMatrix from a single GroupInfo.

    Returns ``(group_matrix, r_inv_or_none, n_cols)`` where *r_inv_or_none*
    is the R_inv column block (for collecting into combined R_inv) or None
    if no reparametrization was applied.
    """
    use_discrete = B_unique is not None
    use_tensor = tensor_build is not None
    R_inv: NDArray | None = None
    # R_inv_local: SSP transform in post-identifiability space (projected -> solver).
    # Used to compose constraints that are already in post-identifiability space.
    R_inv_local: NDArray | None = None

    if info.projection is not None:
        P = info.projection
        if info.reparametrize and info.penalty_matrix is not None:
            B_for = B_unique if use_discrete else info.columns
            exp_for = exposure_agg if use_discrete else sample_weight
            R_inv_local = compute_projected_R_inv(B_for, P, info.penalty_matrix, exp_for, lambda2)
            R_inv = P @ R_inv_local
        else:
            R_inv = P
            # No SSP reparametrization — projected space IS solver space
            R_inv_local = None
        n_cols = R_inv.shape[1]
        omega_full = P @ info.penalty_matrix @ P.T if info.penalty_matrix is not None else None
        if use_tensor:
            gm: GroupMatrix = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                R_inv,
                bin_idx,
                tensor_id=tensor_id,
            )
        elif use_discrete:
            gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
        elif sp.issparse(info.columns):
            gm = SparseSSPGroupMatrix(info.columns, R_inv)
        else:
            gm = DenseGroupMatrix(info.columns @ R_inv)
        if omega_full is not None and hasattr(gm, "omega"):
            gm.omega = omega_full
        if hasattr(gm, "projection"):
            gm.projection = P
        if info.penalty_components is not None and hasattr(gm, "omega_components"):
            gm.omega_components = [
                (suffix, P @ omega_j @ P.T) for suffix, omega_j in info.penalty_components
            ]
            gm.component_types = info.component_types
            if info.lambda_policies is not None:
                gm.lambda_policies = info.lambda_policies

    elif info.reparametrize and info.penalty_matrix is not None:
        B_for = B_unique if use_discrete else info.columns
        exp_for = exposure_agg if use_discrete else sample_weight
        R_inv = compute_R_inv(B_for, info.penalty_matrix, exp_for, lambda2)
        # No projection — constraints are in raw space, same as R_inv input
        R_inv_local = R_inv
        n_cols = R_inv.shape[1]
        if use_tensor:
            gm = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                R_inv,
                bin_idx,
                tensor_id=tensor_id,
            )
            gm.omega = info.penalty_matrix
        elif use_discrete:
            gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
            gm.omega = info.penalty_matrix
        elif sp.issparse(info.columns):
            gm = SparseSSPGroupMatrix(info.columns, R_inv)
            gm.omega = info.penalty_matrix
        else:
            gm = DenseGroupMatrix(info.columns @ R_inv)
        if info.penalty_components is not None and hasattr(gm, "omega_components"):
            gm.omega_components = info.penalty_components
            gm.component_types = info.component_types
            if info.lambda_policies is not None:
                gm.lambda_policies = info.lambda_policies

    else:
        n_cols = info.n_cols
        R_inv = None
        if use_tensor:
            gm = DiscretizedTensorGroupMatrix(
                tensor_build.B1_unique,
                tensor_build.B2_unique,
                tensor_build.idx1,
                tensor_build.idx2,
                B_unique,
                np.eye(info.n_cols, dtype=np.float64),
                bin_idx,
                tensor_id=tensor_id,
            )
        elif info.cat_codes is not None:
            gm = CategoricalGroupMatrix(info.cat_codes, info.n_cols)
        elif sp.issparse(info.columns):
            gm = SparseGroupMatrix(info.columns)
        else:
            gm = DenseGroupMatrix(info.columns)

    # ── Compose constraints into solver coordinates ──
    # Constraints from build() are in post-identifiability space (after projection).
    # R_inv_local maps projected -> solver coords (SSP transform only).
    if info.constraints is not None and R_inv_local is not None:
        info.constraints = info.constraints.compose(R_inv_local)
    # raw_to_solver_map from build() is the identifiability projection (raw -> projected).
    # Extend it to the full chain (raw -> solver) by composing with R_inv_local.
    if info.raw_to_solver_map is not None and R_inv_local is not None:
        info.raw_to_solver_map = info.raw_to_solver_map @ R_inv_local

    return gm, R_inv, n_cols


@dataclass
class BuildResult:
    """Return value of build_design_matrix."""

    dm: DesignMatrix
    groups: list[GroupSlice]
    distribution: Distribution
    link: Link
    y: NDArray
    sample_weight: NDArray
    offset: NDArray | None


def build_design_matrix(
    X: pd.DataFrame,
    y: NDArray,
    sample_weight: NDArray | None,
    offset: NDArray | None,
    *,
    family: str | Distribution,
    link_spec: str | Link | None,
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
    sample_weight = (
        np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    )
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    distribution = resolve_distribution(family)
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

        if (
            use_discrete
            and getattr(spec, "monotone", None) is not None
            and getattr(spec, "monotone_mode", "postfit") == "fit"
            and hasattr(spec, "_build_scop_reparameterization")
        ):
            raise NotImplementedError(
                "SCOP monotone fit-time constraints are not yet supported with "
                "discrete=True. Use discrete=False or monotone_mode='postfit'."
            )

        if use_discrete:
            omega, n_cols_penalty, projection_penalty = spec.build_knots_and_penalty(
                x_col, sample_weight
            )
            n_bins_feat = resolve_discrete_n_bins(name, spec, n_bins_config)
            bin_centers, bin_idx = _discretize_column(x_col, n_bins_feat)
            B_unique = spec._raw_basis_matrix(bin_centers)
            exposure_agg = np.bincount(bin_idx, weights=sample_weight, minlength=len(bin_centers))

            # ── QP monotone constraint metadata (coefficient-space, survives discretization) ──
            constraints = None
            monotone_engine = None
            raw_to_solver_map = None
            if (
                getattr(spec, "monotone", None) is not None
                and getattr(spec, "monotone_mode", "postfit") == "fit"
                and hasattr(spec, "_build_monotone_constraints_raw")
            ):
                cs_raw = spec._build_monotone_constraints_raw()
                constraints = (
                    cs_raw.compose(projection_penalty) if projection_penalty is not None else cs_raw
                )
                monotone_engine = "qp"
                raw_to_solver_map = projection_penalty

            if getattr(spec, "select", False):
                n_null = 1
                n_range = spec._U_range.shape[1]
                n_combined = n_null + n_range
                U_combined = np.hstack([spec._U_null, spec._U_range])
                Z = projection_penalty  # constraint projection

                omega_null = np.zeros((n_combined, n_combined))
                omega_null[:n_null, :n_null] = np.eye(n_null)

                components: list[tuple[str, np.ndarray]] = [("null", omega_null)]

                if len(spec._m_orders) > 1:
                    # Multi-m: project each per-order penalty into combined basis
                    U_null_c = (
                        spec._U_null
                        if Z is None
                        else np.linalg.lstsq(Z, spec._U_null, rcond=None)[0]
                    )
                    U_range_c = (
                        spec._U_range
                        if Z is None
                        else np.linalg.lstsq(Z, spec._U_range, rcond=None)[0]
                    )
                    U_combined_c = np.hstack([U_null_c, U_range_c])
                    for order in spec._m_orders:
                        omega_raw_j = spec._build_penalty_for_order(order)
                        _, omega_c_j, _, _ = spec._apply_constraints(None, omega_raw_j)
                        omega_combined_j = U_combined_c.T @ omega_c_j @ U_combined_c
                        components.append((f"d{order}", omega_combined_j))
                else:
                    omega_wiggle = np.zeros((n_combined, n_combined))
                    omega_wiggle[n_null:, n_null:] = spec._omega_range
                    components.append(("wiggle", omega_wiggle))

                penalty_matrix = sum(omega for _, omega in components)
                infos = [
                    GroupInfo(
                        columns=None,
                        n_cols=n_combined,
                        penalty_matrix=penalty_matrix,
                        reparametrize=True,
                        penalized=True,
                        projection=U_combined,
                        penalty_components=components,
                        component_types={"null": "selection"},
                        constraints=constraints,
                        monotone_engine=monotone_engine,
                        raw_to_solver_map=raw_to_solver_map,
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
                        penalty_components=getattr(spec, "_penalty_components", None),
                        constraints=constraints,
                        monotone_engine=monotone_engine,
                        raw_to_solver_map=raw_to_solver_map,
                    )
                ]
        else:
            result = spec.build(x_col, sample_weight=sample_weight)
            infos = result if isinstance(result, list) else [result]

        # Build GroupMatrix + GroupSlice for each subgroup
        r_inv_parts: list[NDArray] = []

        for info in infos:
            gm, r_inv, n_cols = _process_info(
                info,
                B_unique=B_unique,
                bin_idx=bin_idx,
                sample_weight=sample_weight,
                exposure_agg=exposure_agg,
                lambda2=lambda2,
            )
            if r_inv is not None:
                r_inv_parts.append(r_inv)

            group_matrices.append(gm)
            subgroup_suffix = f":{info.subgroup_name}" if info.subgroup_name else ""
            groups.append(
                GroupSlice(
                    name=f"{name}{subgroup_suffix}",
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(n_cols),
                    penalized=info.penalized,
                    feature_name=name,
                    subgroup_type=info.subgroup_name,
                    constraints=info.constraints,
                    monotone_engine=info.monotone_engine,
                    scop_reparameterization=info.scop_reparameterization,
                )
            )
            col_offset += n_cols

        # Set R_inv on spec for transform/reconstruct
        if r_inv_parts and hasattr(spec, "set_reparametrisation"):
            combined = np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
            spec.set_reparametrisation(combined)

    # ── Interactions ──────────────────────────────────────────
    # Resolve pending interactions from constructor
    for pair in pending_interactions:
        if f"{pair[0]}:{pair[1]}" not in interaction_specs and (
            f"{pair[1]}:{pair[0]}" not in interaction_specs
        ):
            add_interaction(pair[0], pair[1], specs, interaction_specs, interaction_order)
    pending_interactions.clear()
    _next_tensor_id = 0

    for iname in interaction_order:
        ispec = interaction_specs[iname]
        p1, p2 = ispec.parent_names
        x1 = np.asarray(X[p1])
        x2 = np.asarray(X[p2])
        use_discrete_tensor = should_discretize_tensor_interaction(ispec, specs, model_discrete)
        B_unique_inter = None
        bin_idx_inter = None
        exposure_agg_inter = None
        tensor_build: DiscreteTensorBuildResult | None = None
        tensor_id = -1
        if use_discrete_tensor:
            n_bins1 = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            n_bins2 = resolve_discrete_n_bins(p2, specs[p2], n_bins_config)
            tensor_build = ispec.build_discrete(
                x1,
                x2,
                specs,
                (n_bins1, n_bins2),
                sample_weight=sample_weight,
            )
            result = tensor_build.infos
            B_unique_inter = tensor_build.B_joint
            bin_idx_inter = tensor_build.pair_idx
            exposure_agg_inter = np.bincount(
                bin_idx_inter,
                weights=sample_weight,
                minlength=B_unique_inter.shape[0],
            )
            tensor_id = _next_tensor_id
            _next_tensor_id += 1
        else:
            result = ispec.build(x1, x2, specs, sample_weight=sample_weight)

        pi_kwargs = dict(
            B_unique=B_unique_inter,
            bin_idx=bin_idx_inter,
            sample_weight=sample_weight,
            exposure_agg=exposure_agg_inter,
            lambda2=lambda2,
            tensor_build=tensor_build,
            tensor_id=tensor_id,
        )

        if isinstance(result, list):
            has_subgroups = any(info.subgroup_name is not None for info in result)
            if has_subgroups:
                r_inv_parts_i: list[NDArray] = []
                for info in result:
                    gm, r_inv, n_cols = _process_info(info, **pi_kwargs)
                    if r_inv is not None:
                        r_inv_parts_i.append(r_inv)

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
                            constraints=info.constraints,
                            monotone_engine=info.monotone_engine,
                        )
                    )
                    col_offset += n_cols

                if r_inv_parts_i and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(np.hstack(r_inv_parts_i))
            else:
                # Per-level groups (SplineCategorical, PolynomialCategorical)
                r_inv_dict: dict[str, NDArray] = {}
                for level, info in zip(ispec._non_base, result):
                    gm, r_inv, n_cols = _process_info(info, **pi_kwargs)
                    if r_inv is not None:
                        r_inv_dict[level] = r_inv

                    group_matrices.append(gm)
                    groups.append(
                        GroupSlice(
                            name=f"{iname}[{level}]",
                            start=col_offset,
                            end=col_offset + n_cols,
                            weight=np.sqrt(n_cols),
                            penalized=True,
                            feature_name=iname,
                            constraints=info.constraints,
                            monotone_engine=info.monotone_engine,
                        )
                    )
                    col_offset += n_cols

                if r_inv_dict and hasattr(ispec, "set_reparametrisation"):
                    ispec.set_reparametrisation(r_inv_dict)
        else:
            # Single group (CategoricalInteraction, NumericCategorical,
            # NumericInteraction, PolynomialInteraction, TensorInteraction)
            gm, r_inv, n_cols = _process_info(result, **pi_kwargs)
            if r_inv is not None and hasattr(ispec, "set_reparametrisation"):
                ispec.set_reparametrisation(r_inv)

            group_matrices.append(gm)
            groups.append(
                GroupSlice(
                    name=iname,
                    start=col_offset,
                    end=col_offset + n_cols,
                    weight=np.sqrt(n_cols),
                    penalized=True,
                    feature_name=iname,
                    constraints=result.constraints,
                    monotone_engine=result.monotone_engine,
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
        sample_weight=sample_weight,
        offset=offset,
    )


# ═══════════════════════════════════════════════════════════════════
# Design-matrix rebuild with updated lambdas
# ═══════════════════════════════════════════════════════════════════


def _resolve_group_lambda(gm, g, lambdas):
    """Resolve the effective lambda and omega for a group, handling multi-penalty.

    Returns (effective_lambda, effective_omega, has_components) where:
    - For single-penalty groups: (lambdas[g.name], gm.omega, False)
    - For multi-penalty groups: (1.0, sum(lam_j * omega_j), True)
      The effective lambda is 1.0 because it's already baked into the omega.
    """
    if gm.omega_components is not None:
        effective_omega = sum(
            lambdas[f"{g.name}:{suffix}"] * omega_j for suffix, omega_j in gm.omega_components
        )
        return 1.0, effective_omega, True
    return lambdas[g.name], gm.omega, False


def _group_has_lambda(gm, g, lambdas):
    """Check whether lambdas dict contains entries for this group."""
    if g.name in lambdas:
        return True
    if gm.omega_components is not None:
        first_suffix = gm.omega_components[0][0]
        return f"{g.name}:{first_suffix}" in lambdas
    return False


def rebuild_design_matrix_with_lambdas(
    dm: DesignMatrix,
    groups: list[GroupSlice],
    lambdas: dict[str, float],
    sample_weight: NDArray,
    lambda2: float | dict,
) -> DesignMatrix:
    """Rebuild design matrix with per-group smoothing lambdas.

    Only recomputes R_inv for SSP groups whose lambda changed;
    non-SSP groups are reused unchanged.
    """
    new_gms: list[GroupMatrix] = []
    for gm, g in zip(dm.group_matrices, groups):
        if isinstance(gm, SparseSSPGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B, P, omega_proj, sample_weight, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B, omega_eff, sample_weight, lam)
            new_gm = SparseSSPGroupMatrix(gm.B, R_inv_new)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedTensorGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            exposure_agg = np.bincount(gm.bin_idx, weights=sample_weight, minlength=gm.n_bins)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_unique, P, omega_proj, exposure_agg, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, exposure_agg, lam)
            new_gm = DiscretizedTensorGroupMatrix(
                gm.B1_unique_t,
                gm.B2_unique_t,
                gm.idx1,
                gm.idx2,
                gm.B_unique,
                R_inv_new,
                gm.bin_idx,
                tensor_id=gm.tensor_id,
            )
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedSSPGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            exposure_agg = np.bincount(gm.bin_idx, weights=sample_weight, minlength=gm.n_bins)
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_unique, P, omega_proj, exposure_agg, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, exposure_agg, lam)
            new_gm = DiscretizedSSPGroupMatrix(gm.B_unique, R_inv_new, gm.bin_idx)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gms.append(new_gm)
        else:
            new_gms.append(gm)
    return DesignMatrix(new_gms, dm.n, dm.p)
