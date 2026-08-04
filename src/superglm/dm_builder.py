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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm._frame import EagerFrame
from superglm._utils import _validate_strict_prior_weights
from superglm.distributions import Distribution, Tweedie, resolve_distribution
from superglm.features.ordered_categorical import resolve_interaction_parent_of
from superglm.group_matrix import (
    CategoricalGroupMatrix,
    DenseGroupMatrix,
    DesignMatrix,
    DiscretizedSCOPGroupMatrix,
    DiscretizedSplineCategoricalGroupMatrix,
    DiscretizedSSPGroupMatrix,
    DiscretizedTensorGroupMatrix,
    FactorSmoothGroupMatrix,
    GroupMatrix,
    RandomEffectGroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
    SplineCategoricalGroupMatrix,
    SupportCompressedSplineCategoricalGroupMatrix,
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.types import DiscreteTensorBuildResult, FeatureSpec, GroupInfo, GroupSlice

logger = logging.getLogger(__name__)


def validate_term_name_namespace(
    specs: dict[str, FeatureSpec],
    interaction_specs: dict[str, Any],
    *,
    additional_interactions: Iterable[str] = (),
) -> None:
    """Require one public name to identify exactly one model term."""
    interaction_names = set(interaction_specs)
    interaction_names.update(additional_interactions)
    collisions = sorted(set(specs).intersection(interaction_names))
    if collisions:
        names = ", ".join(repr(name) for name in collisions)
        raise ValueError(
            f"Term name(s) {names} are registered as both a main feature and an interaction."
        )


def validate_fitted_group_names(groups: list[GroupSlice]) -> None:
    """Reject fitted coefficient groups whose public names would alias."""
    counts: dict[str, int] = {}
    for group in groups:
        counts[group.name] = counts.get(group.name, 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Generated fitted group names must be unique; found {duplicates!r}.")


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
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        G = np.zeros((omega.shape[0], omega.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / weight_sum
    else:
        G = (B * sample_weight[:, None]).T @ B / weight_sum
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
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        G_full = np.zeros((projection.shape[0], projection.shape[0]), dtype=np.float64)
    elif sp.issparse(B):
        G_full = np.asarray((B.multiply(sample_weight[:, None]).T @ B).todense()) / np.sum(
            sample_weight
        )
    else:
        G_full = (B * sample_weight[:, None]).T @ B / weight_sum
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
    from superglm.features.ordered_categorical import OrderedCategorical

    if any(
        isinstance(specs.get(p), OrderedCategorical) for p in getattr(ispec, "parent_names", ())
    ):
        # OC margins live on <= n_levels score points; there is nothing for
        # fit-time discretization to compress, and the fast-discrete predict
        # metadata reads raw columns as float64, which label data cannot be.
        return False
    if not isinstance(ispec, TensorInteraction):
        return False
    p1, p2 = ispec.parent_names
    return should_discretize(specs[p1], model_discrete) and should_discretize(
        specs[p2], model_discrete
    )


def should_discretize_spline_categorical_interaction(
    ispec: Any, specs: dict[str, FeatureSpec], model_discrete: bool
) -> bool:
    """Check if a spline-categorical interaction should use spline support compression."""
    from superglm.features.interaction import SplineCategorical
    from superglm.features.ordered_categorical import OrderedCategorical

    if any(
        isinstance(specs.get(p), OrderedCategorical) for p in getattr(ispec, "parent_names", ())
    ):
        # OC margins live on <= n_levels score points; there is nothing for
        # fit-time discretization to compress, and the fast-discrete predict
        # metadata reads raw columns as float64, which label data cannot be.
        return False
    if not isinstance(ispec, SplineCategorical):
        return False
    spline_name, _cat_name = ispec.parent_names
    return should_discretize(specs[spline_name], model_discrete)


def should_discretize_factor_smooth(ispec: Any, model_discrete: bool) -> bool:
    """Use compact marginal support bins for factor smooths in discrete models."""
    from superglm.features.factor_smooth import FactorSmooth

    return isinstance(ispec, FactorSmooth) and model_discrete


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


def _discretize_spline_column(
    x: NDArray,
    n_bins: int,
    geometry_weight: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """Discretize on the positive-frequency support and map every fit row.

    Zero-frequency rows need a valid bin index because the design retains its
    physical row count, but they must not widen or otherwise change the
    learned support. Mapping those inactive rows to the nearest active support
    point is consistent with clipped spline evaluation and cannot affect the
    weighted fit.
    """
    values = np.asarray(x, dtype=np.float64).ravel()
    if geometry_weight is None:
        return _discretize_column(values, n_bins)

    weights = np.asarray(geometry_weight, dtype=np.float64).ravel()
    active = weights > 0.0
    if np.all(active):
        return _discretize_column(values, n_bins)

    support, active_index = _discretize_column(values[active], n_bins)
    if len(support) == 1:
        full_index = np.zeros(len(values), dtype=np.intp)
    else:
        insertion = np.searchsorted(support, values)
        right = np.clip(insertion, 0, len(support) - 1)
        left = np.clip(insertion - 1, 0, len(support) - 1)
        choose_right = np.abs(values - support[right]) <= np.abs(values - support[left])
        full_index = np.where(choose_right, right, left).astype(np.intp)
    full_index[active] = active_index
    return support, full_index


# ═══════════════════════════════════════════════════════════════════
# Feature auto-detection and interaction dispatch
# ═══════════════════════════════════════════════════════════════════


def auto_detect_features(
    X: EagerFrame,
    sample_weight: NDArray | None,
    *,
    spline_cols: list[str],
    knots_map: dict[str, int],
    degree: int,
    categorical_base: str,
    specs: dict[str, FeatureSpec],
    feature_order: list[str],
) -> None:
    """Auto-detect feature types from native dataframe columns.

    Mutates ``specs`` and ``feature_order`` in place.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.spline import PSpline

    lines = ["SuperGLM features:"]
    for raw_col in X.columns:
        col = cast(str, raw_col)
        kind = X.column_kind(col)
        if col in spline_cols:
            nk = knots_map[col]
            spec = PSpline(n_knots=nk, degree=degree, penalty="ssp")
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Spline(n_knots={nk}, degree={degree})")
        elif kind == "categorical":
            base = categorical_base
            if base == "most_exposed" and sample_weight is None:
                base = "first"
            spec = Categorical(base=base)
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Categorical(base={base})")
        elif kind in ("numeric", "boolean"):
            spec = Numeric()
            specs[col] = spec
            feature_order.append(col)
            lines.append(f"  {str(col):<20s} → Numeric()")
        else:
            raise ValueError(
                f"X column {col!r} has unsupported dtype {X.column_dtype(col)!r} "
                "for automatic feature detection; configure the feature explicitly "
                "or convert the column to numeric, boolean, string, or categorical data"
            )
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


def _build_ssp_group(B_csr, R_inv):
    """Cheapest exact representation of a factored SSP block.

    Compression is lossless deduplication of repeated rows; it never bins and is
    independent of ``discrete=True``.  Declines whenever the measured cost model
    says the current CSR path is cheaper.
    """
    from superglm._group_matrix._group_matrix_discretized import (
        SupportCompressedSSPGroupMatrix,
    )
    from superglm._group_matrix._group_matrix_support import detect_row_support

    detected = detect_row_support(B_csr)
    if detected is None:
        return SparseSSPGroupMatrix(B_csr, R_inv)
    B_unique_rows, row_index = detected
    return SupportCompressedSSPGroupMatrix(B_unique_rows, R_inv, row_index)


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
    from superglm.features.ordered_categorical import OrderedCategorical

    if feat1 not in specs:
        raise ValueError(f"Parent feature not found: {feat1}")
    if feat2 not in specs:
        raise ValueError(f"Parent feature not found: {feat2}")

    # _spec_kind reads a step-mode OrderedCategorical as "categorical", which
    # is right for its MAIN effect but wrong for an interaction parent: the
    # deprecated one-hot geometry has no marginal smooth to cross with, and
    # resolve_interaction_parent refuses it.  Without this the pair registers
    # here and only fails much later, mid design-matrix build, after the
    # caller has already committed a fit.
    for parent in (feat1, feat2):
        spec = specs[parent]
        if isinstance(spec, OrderedCategorical) and (
            spec.basis != "spline" or spec._spline is None
        ):
            raise NotImplementedError(
                f"cannot add the interaction ({feat1!r}, {feat2!r}): {parent!r} is an "
                "OrderedCategorical with basis='step', which is deprecated and cannot "
                "parent an interaction; use basis=Spline(...) for a smoothed ordinal "
                "parent or a Categorical feature for unsmoothed level effects."
            )

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
    validate_term_name_namespace(
        specs,
        interaction_specs,
        additional_interactions=(iname,),
    )

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
    is_spline_cat = info.spline_cat_mask is not None and (
        info.spline_cat_basis is not None
        or (info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None)
    )

    if is_spline_cat:
        mask = np.asarray(info.spline_cat_mask, dtype=bool)
        row_idx = np.flatnonzero(mask)
        if info.spline_cat_basis_unique is not None and info.spline_cat_bin_idx is not None:
            B_support: sp.spmatrix | NDArray = np.asarray(
                info.spline_cat_basis_unique,
                dtype=np.float64,
            )
            bin_idx_level = np.asarray(info.spline_cat_bin_idx, dtype=np.intp)[row_idx]
            level_weight = np.bincount(
                bin_idx_level,
                weights=sample_weight[row_idx],
                minlength=B_support.shape[0],
            )
            B_for: sp.spmatrix | NDArray = B_support
            use_discrete_spline_cat = True
        else:
            B_full = sp.csr_matrix(info.spline_cat_basis)
            B_level = B_full[row_idx]
            B_for = B_level
            level_weight = sample_weight[row_idx]
            use_discrete_spline_cat = False
        if info.projection is not None:
            P = info.projection
            if info.reparametrize and info.penalty_matrix is not None:
                R_inv_local = compute_projected_R_inv(
                    B_for,
                    P,
                    info.penalty_matrix,
                    level_weight,
                    lambda2,
                )
                R_inv = P @ R_inv_local
            else:
                R_inv = P
                R_inv_local = None
            omega_full = P @ info.penalty_matrix @ P.T if info.penalty_matrix is not None else None
        elif info.reparametrize and info.penalty_matrix is not None:
            R_inv = compute_R_inv(B_for, info.penalty_matrix, level_weight, lambda2)
            R_inv_local = R_inv
            omega_full = info.penalty_matrix
        else:
            R_inv = np.eye(info.n_cols, dtype=np.float64)
            R_inv_local = None
            omega_full = info.penalty_matrix

        n_cols = R_inv.shape[1]
        if use_discrete_spline_cat:
            # Same storage, different contract: the lossless support indexes
            # exact distinct rows, so the class must not claim the binned route.
            spline_cat_cls = (
                SupportCompressedSplineCategoricalGroupMatrix
                if info.spline_cat_support_lossless
                else DiscretizedSplineCategoricalGroupMatrix
            )
            gm = spline_cat_cls(
                B_support,
                R_inv,
                info.spline_cat_bin_idx,
                row_idx,
            )
        else:
            gm = SplineCategoricalGroupMatrix(B_full, R_inv, row_idx)
        gm.spline_cat_level = info.spline_cat_level
        gm.spline_cat_feature = info.spline_cat_feature
        if omega_full is not None:
            gm.omega = omega_full
        if info.projection is not None:
            gm.projection = info.projection
        if info.penalty_components is not None:
            if info.projection is not None:
                P = info.projection
                gm.omega_components = [
                    (suffix, P @ omega_j @ P.T) for suffix, omega_j in info.penalty_components
                ]
            else:
                gm.omega_components = info.penalty_components
            gm.component_types = info.component_types
            if info.lambda_policies is not None:
                gm.lambda_policies = info.lambda_policies

    elif info.projection is not None:
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
            gm = _build_ssp_group(info.columns, R_inv)
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
            gm = _build_ssp_group(info.columns, R_inv)
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
            if info.penalty_matrix is not None:
                gm.omega = info.penalty_matrix
            if info.penalty_components is not None:
                gm.omega_components = info.penalty_components
                gm.component_types = info.component_types
                if info.lambda_policies is not None:
                    gm.lambda_policies = info.lambda_policies
        elif use_discrete and info.scop_reparameterization is not None and bin_idx is not None:
            # SCOP discrete: columns holds the bin-level centered design
            gm = DiscretizedSCOPGroupMatrix(info.columns, bin_idx)
        elif info.structured_kind == "factor_smooth":
            factor_basis = (
                info.factor_smooth_basis_unique
                if info.factor_smooth_basis_unique is not None
                else info.factor_smooth_basis
            )
            gm = FactorSmoothGroupMatrix(
                factor_basis,
                info.factor_smooth_codes,
                info.factor_smooth_n_levels,
                natural_map=info.factor_smooth_transform,
                levels=info.factor_smooth_levels,
                repeated_penalty_components=info.repeated_penalty_components,
                factor_basis=info.factor_smooth_factor_basis,
                lambda_policies=info.lambda_policies,
                bin_idx=info.factor_smooth_bin_idx,
            )
        elif info.structured_kind == "random_effect":
            gm = RandomEffectGroupMatrix(
                info.cat_codes,
                info.n_cols,
                lambda_policies=info.lambda_policies,
            )
        elif info.cat_codes is not None:
            gm = CategoricalGroupMatrix(info.cat_codes, info.n_cols)
        elif sp.issparse(info.columns):
            if info.penalty_matrix is not None or info.penalty_components is not None:
                gm = _build_ssp_group(info.columns, np.eye(info.n_cols, dtype=np.float64))
                if info.penalty_matrix is not None:
                    gm.omega = info.penalty_matrix
                if info.penalty_components is not None:
                    gm.omega_components = info.penalty_components
                    gm.component_types = info.component_types
                    if info.lambda_policies is not None:
                        gm.lambda_policies = info.lambda_policies
            else:
                gm = SparseGroupMatrix(info.columns)
        else:
            gm = DenseGroupMatrix(info.columns)

    # ── Compose constraints into solver coordinates ──
    # Constraints from build() are in post-identifiability space (after projection).
    # R_inv_local maps projected -> solver coords (SSP transform only).
    # Main-effect QP constraints are rebuilt once from raw rows and the full
    # R_inv in the feature loop below; this provisional composition preserves
    # the generic GroupInfo contract for other callers.
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
    X: EagerFrame,
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
    distribution = resolve_distribution(family)
    if sample_weight is None:
        sample_weight = np.ones(n, dtype=np.float64)
    elif isinstance(distribution, Tweedie):
        sample_weight = _validate_strict_prior_weights(sample_weight, n)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64)
    link = resolve_link(link_spec, distribution)
    from superglm.features.spline import _SplineBase

    # Non-Tweedie weights are frequency mass for learned spline geometry.
    # Tweedie weights are EDM prior weights, so its geometry stays a function
    # of physical rows only.
    geometry_weight = None if isinstance(distribution, Tweedie) else sample_weight

    group_matrices: list[GroupMatrix] = []
    col_offset = 0
    groups: list[GroupSlice] = []

    for name in feature_order:
        spec = specs[name]
        x_col = X.column_array(name)
        spline_geometry_weight = geometry_weight if isinstance(spec, _SplineBase) else sample_weight

        # Check if this feature should use fit-time discretization
        use_discrete = should_discretize(spec, model_discrete)
        B_unique = None
        bin_idx = None
        exposure_agg = None

        constraint_kind = getattr(spec, "constraint_kind", getattr(spec, "monotone", None))
        constraint_mode = getattr(
            spec, "constraint_mode", getattr(spec, "monotone_mode", "postfit")
        )

        _scop_discrete = (
            use_discrete
            and constraint_kind is not None
            and constraint_mode == "fit"
            and hasattr(spec, "_build_scop_reparameterization")
        )

        if use_discrete:
            omega, n_cols_penalty, projection_penalty = spec.build_knots_and_penalty(
                x_col, spline_geometry_weight
            )
            n_bins_feat = resolve_discrete_n_bins(name, spec, n_bins_config)
            bin_centers, bin_idx = _discretize_spline_column(
                x_col,
                n_bins_feat,
                spline_geometry_weight,
            )
            B_unique = spec._raw_basis_matrix(bin_centers)
            exposure_agg = np.bincount(bin_idx, weights=sample_weight, minlength=len(bin_centers))

            # ── QP monotone constraint metadata (coefficient-space, survives discretization) ──
            constraints = None
            monotone_engine = None
            raw_to_solver_map = None
            if (
                constraint_kind is not None
                and constraint_mode == "fit"
                and hasattr(spec, "_build_monotone_constraints_raw")
            ):
                cs_raw = spec._build_monotone_constraints_raw()
                constraints = (
                    cs_raw.compose(projection_penalty) if projection_penalty is not None else cs_raw
                )
                monotone_engine = "qp"
                raw_to_solver_map = projection_penalty

            if _scop_discrete:
                # ── SCOP monotone + discrete: bin-level centered design ──
                from superglm.solvers.scop import build_scop_reparam, build_scop_solver_reparam

                q_raw = spec._n_basis
                raw_reparam = build_scop_reparam(
                    q_raw,
                    kind=constraint_kind,
                    knots=spec._knots,
                    degree=spec.degree,
                    domain=(spec._lo, spec._hi),
                )
                X_sigma_unique = B_unique @ raw_reparam.Sigma  # (n_bins, K)

                # Centering weighted by bin counts (equivalent to full-data mean)
                n_obs = len(bin_idx)
                bin_counts = np.bincount(bin_idx, minlength=len(bin_centers))
                drop_dim = 1
                X_shape_unique = X_sigma_unique[:, drop_dim:]
                col_means = (X_shape_unique.T @ bin_counts) / n_obs
                X_centered_unique = X_shape_unique - col_means  # (n_bins, q_eff)

                # Store on spec for predict-time transform
                spec._scop_Sigma = raw_reparam.Sigma
                setattr(spec, "_scop_null_dim", drop_dim)
                spec._scop_col_means = col_means

                solver_reparam = build_scop_solver_reparam(
                    q_raw,
                    kind=constraint_kind,
                    knots=spec._knots,
                    degree=spec.degree,
                    domain=(spec._lo, spec._hi),
                )
                S_scop = solver_reparam.penalty_matrix()
                q_eff = X_centered_unique.shape[1]

                infos = [
                    GroupInfo(
                        columns=X_centered_unique,  # bin-level centered design
                        n_cols=q_eff,
                        penalty_matrix=S_scop,
                        reparametrize=False,
                        penalized=True,
                        scop_reparameterization=solver_reparam,
                        monotone_engine="scop",
                    )
                ]

            elif getattr(spec, "select", False):
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
            result = spec.build(x_col, sample_weight=spline_geometry_weight)
            infos = result if isinstance(result, list) else [result]

        # Resolve lambda_policies from the spec onto each GroupInfo.
        # build() does this internally, but the discrete path constructs
        # GroupInfo manually and needs an explicit resolution step.
        #
        # For single-penalty discrete groups, build_penalty_components uses
        # the multi-penalty path (omega_components with "wiggle" suffix) when
        # penalty_components is set. We need to ensure penalty_components is
        # populated for single-penalty terms with lambda_policy, matching
        # what build() does for non-discrete terms.
        if hasattr(spec, "_lambda_policy") and spec._lambda_policy is not None:
            for info in infos:
                if info.lambda_policies is None:
                    info.lambda_policies = spec._resolve_lambda_policies(info)
                # Single-penalty terms need a synthetic penalty_components
                # for the lambda_policy to flow through to build_penalty_components.
                if info.penalty_components is None and info.penalty_matrix is not None:
                    info.penalty_components = [("wiggle", info.penalty_matrix)]
                    info.component_types = {"wiggle": "wiggle"}

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
            qp_info = cast(GroupInfo, info)
            if qp_info.monotone_engine == "qp":
                raw_builder = getattr(spec, "_build_monotone_constraints_raw", None)
                if raw_builder is None:
                    raise RuntimeError(
                        f"cannot build QP constraints for feature {name!r}: "
                        "its raw constraint geometry is unavailable"
                    )
                raw_constraints = raw_builder()
                qp_info.constraints = (
                    raw_constraints if r_inv is None else raw_constraints.compose(r_inv)
                )
                if qp_info.constraints.n_params != n_cols:
                    raise RuntimeError(
                        f"QP constraint width for feature {name!r} does not match "
                        "its solver-coordinate group"
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
        spec1, x1 = resolve_interaction_parent_of(ispec, specs.get(p1), X.column_array(p1))
        spec2, x2 = resolve_interaction_parent_of(ispec, specs.get(p2), X.column_array(p2))
        if spec1 is specs.get(p1) and spec2 is specs.get(p2):
            parent_specs = specs
        else:
            parent_specs = {**specs, p1: spec1, p2: spec2}
        use_discrete_tensor = should_discretize_tensor_interaction(ispec, specs, model_discrete)
        use_discrete_spline_cat = should_discretize_spline_categorical_interaction(
            ispec,
            specs,
            model_discrete,
        )
        use_discrete_factor_smooth = should_discretize_factor_smooth(
            ispec,
            model_discrete,
        )
        B_unique_inter = None
        bin_idx_inter = None
        exposure_agg_inter = None
        tensor_build: DiscreteTensorBuildResult | None = None
        tensor_id = -1
        if use_discrete_factor_smooth:
            n_bins_factor_smooth = resolve_discrete_n_bins(p1, ispec, n_bins_config)
            result = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                n_bins_factor_smooth,
                sample_weight=sample_weight,
            )
        elif use_discrete_tensor:
            n_bins1 = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            n_bins2 = resolve_discrete_n_bins(p2, specs[p2], n_bins_config)
            tensor_build = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
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
        elif use_discrete_spline_cat:
            n_bins_spline = resolve_discrete_n_bins(p1, specs[p1], n_bins_config)
            result = ispec.build_discrete(
                x1,
                x2,
                parent_specs,
                n_bins_spline,
                sample_weight=sample_weight,
            )
        else:
            result = ispec.build(x1, x2, parent_specs, sample_weight=sample_weight)

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
            g_new = GroupSlice(
                name=iname,
                start=col_offset,
                end=col_offset + n_cols,
                weight=np.sqrt(n_cols),
                penalized=True,
                feature_name=iname,
                constraints=result.constraints,
                monotone_engine=result.monotone_engine,
            )
            if hasattr(ispec, "set_reparametrisation") and hasattr(gm, "R_inv"):
                ispec.set_reparametrisation(gm.R_inv)
            elif r_inv is not None and hasattr(ispec, "set_reparametrisation"):
                ispec.set_reparametrisation(r_inv)

            group_matrices.append(gm)
            groups.append(g_new)
            col_offset = g_new.end

    validate_term_name_namespace(specs, interaction_specs)
    validate_fitted_group_names(groups)
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


# Discrete tensor interactions are deliberately *not* constrained against one
# another.  ``ti()`` marginals are already centered, so two terms sharing a
# marginal — ``ti(a, b) + ti(a, c)``, the standard mgcv functional-ANOVA
# pattern — span distinct, jointly identifiable surfaces.  A cross-Gram null
# space is also the wrong instrument for cross-term aliasing: it keeps the
# directions a tensor shares *least* with its neighbours, so on a holey joint
# support (correlated marginals, common in insurance data) it retains that
# tensor's numerically null subspace and discards the whole fitted surface.
# The exact path builds the full span and lets the penalty and shared rank
# policy handle degenerate directions; the discrete path must match it.


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
            new_gm.lambda_policies = gm.lambda_policies
            new_gms.append(new_gm)
        elif isinstance(gm, SplineCategoricalGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            level_weight = sample_weight[gm.row_idx]
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(gm.B_level, P, omega_proj, level_weight, lam)
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_level, omega_eff, level_weight, lam)
            new_gm = SplineCategoricalGroupMatrix(gm.B, R_inv_new, gm.row_idx)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.lambda_policies = gm.lambda_policies
            new_gm.spline_cat_level = gm.spline_cat_level
            new_gm.spline_cat_feature = gm.spline_cat_feature
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedSplineCategoricalGroupMatrix) and _group_has_lambda(
            gm, g, lambdas
        ):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            lam, omega_eff, has_comp = _resolve_group_lambda(gm, g, lambdas)
            level_weight = np.bincount(
                gm.bin_idx_level,
                weights=sample_weight[gm.row_idx],
                minlength=gm.n_bins,
            )
            if gm.projection is not None:
                P = gm.projection
                omega_proj = P.T @ omega_eff @ P
                R_inv_local = compute_projected_R_inv(
                    gm.B_unique,
                    P,
                    omega_proj,
                    level_weight,
                    lam,
                )
                R_inv_new = P @ R_inv_local
            else:
                R_inv_new = compute_R_inv(gm.B_unique, omega_eff, level_weight, lam)
            # type(gm), not the parent: every REML lambda update comes through
            # here, and a lossless block must not be downgraded to a binned one.
            new_gm = type(gm)(
                gm.B_unique,
                R_inv_new,
                gm.bin_idx_level,
                gm.row_idx,
                n_rows=gm.n_rows,
                bin_idx_is_level=True,
            )
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gm.lambda_policies = gm.lambda_policies
            new_gm.spline_cat_level = gm.spline_cat_level
            new_gm.spline_cat_feature = gm.spline_cat_feature
            new_gms.append(new_gm)
        elif isinstance(gm, DiscretizedTensorGroupMatrix) and _group_has_lambda(gm, g, lambdas):
            if gm.omega is None:
                new_gms.append(gm)
                continue
            if gm.omega_components is not None and gm.projection is None:
                # Discrete tensor interactions are emitted in fixed centered
                # tensor coordinates with explicit marginal penalty components.
                # Lambda updates should change S(lambda), not rebuild the
                # packed design/R_inv. This mirrors mgcv bam(discrete=TRUE),
                # where the packed marginal/index representation is fixed
                # while smoothing parameters move.
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
            # type(gm) preserves subclass semantics (e.g. lossless support
            # compression) across the per-lambda rebuild.
            new_gm = type(gm)(gm.B_unique, R_inv_new, gm.bin_idx)
            new_gm.omega = gm.omega
            new_gm.projection = gm.projection
            new_gm.omega_components = gm.omega_components
            new_gm.component_types = gm.component_types
            new_gms.append(new_gm)
        else:
            new_gms.append(gm)
    rebuilt = DesignMatrix(new_gms, dm.n, dm.p)
    rebuilt._centered_pattern_plan = dm._centered_pattern_plan
    return rebuilt
