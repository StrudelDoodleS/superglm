"""SuperGLM: main model class."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from superglm.discretize import DiscretizationResult
    from superglm.metrics import ModelMetrics

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
    _block_xtwx,
    _block_xtwx_signed,
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.penalties.base import Penalty
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    _invert_xtwx_plus_penalty,
    fit_irls_direct,
)
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FeatureSpec, GroupInfo, GroupSlice

logger = logging.getLogger(__name__)

_PENALTY_SHORTCUTS: dict[str, type[Penalty]] = {
    "group_lasso": GroupLasso,
    "group_elastic_net": GroupElasticNet,
    "sparse_group_lasso": SparseGroupLasso,
    "ridge": Ridge,
}


@dataclass
class PathResult:
    """Container for regularization path results."""

    lambda_seq: NDArray  # shape (n_lambda,)
    coef_path: NDArray  # shape (n_lambda, p)
    intercept_path: NDArray  # shape (n_lambda,)
    deviance_path: NDArray  # shape (n_lambda,)
    n_iter_path: NDArray  # shape (n_lambda,) — PIRLS iters per lambda
    converged_path: NDArray  # shape (n_lambda,) — bool
    edf_path: NDArray | None = None  # shape (n_lambda,) — effective df


class SuperGLM:
    def __init__(
        self,
        family: str | Distribution = "poisson",
        link: str | Link | None = None,
        penalty: Penalty | str | None = None,
        lambda1: float | None = None,
        lambda2: float = 0.1,
        tweedie_p: float | None = None,
        # Feature configuration
        nb_theta: float | str | None = None,
        # Feature configuration
        features: dict[str, FeatureSpec] | None = None,
        splines: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        standardize_numeric: bool = True,
        # Interactions
        interactions: list[tuple[str, str]] | None = None,
        # Solver options
        anderson_memory: int = 0,
        active_set: bool = False,
        # Discretization
        discrete: bool = False,
        n_bins: int | dict[str, int] = 256,
    ):
        if features is not None and splines is not None:
            raise ValueError(
                "Cannot set both 'features' and 'splines'. "
                "Use 'features' for explicit specs or 'splines' for auto-detect."
            )
        self.family = family
        self.link = link
        self.penalty = self._resolve_penalty(penalty, lambda1)
        self.lambda2 = lambda2
        self.tweedie_p = tweedie_p
        self.nb_theta = nb_theta
        self._splines = splines
        self._n_knots = n_knots
        self._degree = degree
        self._categorical_base = categorical_base
        self._standardize_numeric = standardize_numeric
        self._anderson_memory = anderson_memory
        self._active_set = active_set
        self._discrete = discrete
        self._n_bins = n_bins

        self._specs: dict[str, FeatureSpec] = {}
        self._feature_order: list[str] = []
        self._groups: list[GroupSlice] = []
        self._distribution: Distribution | None = None
        self._link: Link | None = None
        self._result: PIRLSResult | None = None
        self._dm: DesignMatrix | None = None
        self._fit_weights: NDArray | None = None
        self._fit_offset: NDArray | None = None
        self._nb_profile_result = None  # NBProfileResult, set by estimate_theta()
        self._tweedie_profile_result = None  # TweedieProfileResult, set by estimate_tweedie_p()

        # Interaction support
        self._interaction_specs: dict[str, Any] = {}
        self._interaction_order: list[str] = []
        self._pending_interactions: list[tuple[str, str]] = interactions or []

        # Register explicit features dict
        if features is not None:
            for name, spec in features.items():
                self._specs[name] = spec
                self._feature_order.append(name)

    @staticmethod
    def _resolve_penalty(penalty: Penalty | str | None, lambda1: float | None) -> Penalty:
        """Convert string shorthand / None to a Penalty object."""
        if penalty is None:
            return GroupLasso(lambda1=lambda1)
        if isinstance(penalty, str):
            if penalty not in _PENALTY_SHORTCUTS:
                raise ValueError(
                    f"Unknown penalty '{penalty}'. "
                    f"Use one of {list(_PENALTY_SHORTCUTS)} or pass a Penalty object."
                )
            return _PENALTY_SHORTCUTS[penalty](lambda1=lambda1)
        if lambda1 is not None:
            raise ValueError(
                "Cannot set 'lambda1' when passing a Penalty object directly. "
                "Set lambda1 on the Penalty object instead."
            )
        return penalty

    def _resolve_knots(self, spline_cols: list[str]) -> dict[str, int]:
        """Map spline column names to their n_knots values."""
        if not spline_cols:
            return {}
        if isinstance(self._n_knots, int):
            return {col: self._n_knots for col in spline_cols}
        if len(self._n_knots) != len(spline_cols):
            raise ValueError(
                f"n_knots has length {len(self._n_knots)} but splines "
                f"has length {len(spline_cols)}. Must match or pass a single int."
            )
        return dict(zip(spline_cols, self._n_knots))

    def _clone_without_features(
        self,
        drop: set[str],
        *,
        lambda1: float | None = ...,  # sentinel: ... means "keep current"
        lambda2: float | dict[str, float] | None = ...,
    ) -> SuperGLM:
        """Create a new SuperGLM with a subset of features removed.

        Copies family, link, penalty type, and solver options. Interactions
        whose parents include a dropped feature are also removed.
        """
        keep_features = {n: s for n, s in self._specs.items() if n not in drop}

        # Filter interactions: drop any whose parent is being dropped
        keep_interactions: list[tuple[str, str]] = []
        # Check resolved interactions (fitted model)
        for iname in self._interaction_order:
            ispec = self._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            if p1 not in drop and p2 not in drop:
                keep_interactions.append((p1, p2))
        # Check pending interactions (unfitted model)
        for p1, p2 in self._pending_interactions:
            if p1 not in drop and p2 not in drop:
                keep_interactions.append((p1, p2))

        # Resolve lambda1
        if lambda1 is ...:
            lam1 = self.penalty.lambda1
        else:
            lam1 = lambda1

        # Create penalty of the same type
        new_penalty = type(self.penalty)(lambda1=lam1)

        # Deep-copy specs so the new model doesn't share mutable state
        new_features = {n: copy.deepcopy(s) for n, s in keep_features.items()}

        new_model = SuperGLM(
            family=self.family,
            link=self.link,
            penalty=new_penalty,
            features=new_features,
            interactions=keep_interactions if keep_interactions else None,
            anderson_memory=self._anderson_memory,
            active_set=self._active_set,
            discrete=self._discrete,
            n_bins=self._n_bins,
        )

        # Resolve lambda2
        if lambda2 is ...:
            reml_lam = getattr(self, "_reml_lambdas", None)
            if reml_lam is not None:
                # Filter REML lambdas to remaining groups
                new_model.lambda2 = {
                    k: v
                    for k, v in reml_lam.items()
                    if not any(k == d or k.startswith(f"{d}:") for d in drop)
                }
            else:
                new_model.lambda2 = self.lambda2
        elif lambda2 is None:
            new_model.lambda2 = 0.0
        else:
            new_model.lambda2 = lambda2

        return new_model

    def _auto_detect_features(self, X: pd.DataFrame, exposure: NDArray | None) -> None:
        """Auto-detect feature types from DataFrame columns."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        spline_cols = self._splines or []
        knots_map = self._resolve_knots(spline_cols)

        lines = ["SuperGLM features:"]
        for col in X.columns:
            if col in spline_cols:
                nk = knots_map[col]
                spec = Spline(n_knots=nk, degree=self._degree, penalty="ssp")
                self._specs[col] = spec
                self._feature_order.append(col)
                lines.append(f"  {col:<20s} → Spline(n_knots={nk}, degree={self._degree})")
            elif X[col].dtype.kind in ("O", "U") or isinstance(X[col].dtype, pd.CategoricalDtype):
                base = self._categorical_base
                if base == "most_exposed" and exposure is None:
                    base = "first"
                spec = Categorical(base=base)
                self._specs[col] = spec
                self._feature_order.append(col)
                lines.append(f"  {col:<20s} → Categorical(base={base})")
            else:
                spec = Numeric(standardize=self._standardize_numeric)
                self._specs[col] = spec
                self._feature_order.append(col)
                lines.append(f"  {col:<20s} → Numeric(standardize={self._standardize_numeric})")
        logger.info("\n".join(lines))

    def _add_interaction(self, feat1: str, feat2: str, name: str | None = None, **kwargs) -> None:
        """Register an interaction between two already-registered features."""
        from superglm.features.categorical import Categorical
        from superglm.features.interaction import (
            CategoricalInteraction,
            NumericCategorical,
            NumericInteraction,
            PolynomialCategorical,
            PolynomialInteraction,
            SplineCategorical,
            TensorInteraction,
        )
        from superglm.features.numeric import Numeric
        from superglm.features.polynomial import Polynomial
        from superglm.features.spline import _SplineBase

        if feat1 not in self._specs:
            raise ValueError(f"Parent feature not found: {feat1}")
        if feat2 not in self._specs:
            raise ValueError(f"Parent feature not found: {feat2}")

        spec1 = self._specs[feat1]
        spec2 = self._specs[feat2]

        is_spline1 = isinstance(spec1, _SplineBase)
        is_spline2 = isinstance(spec2, _SplineBase)
        is_poly1 = isinstance(spec1, Polynomial)
        is_poly2 = isinstance(spec2, Polynomial)
        is_num1 = isinstance(spec1, Numeric)
        is_num2 = isinstance(spec2, Numeric)
        is_cat1 = isinstance(spec1, Categorical)
        is_cat2 = isinstance(spec2, Categorical)

        iname: str
        ispec: Any

        # Spline + Categorical → SplineCategorical (swap so spline is first)
        if is_spline1 and is_cat2:
            iname = name or f"{feat1}:{feat2}"
            ispec = SplineCategorical(feat1, feat2)
        elif is_cat1 and is_spline2:
            iname = name or f"{feat2}:{feat1}"
            ispec = SplineCategorical(feat2, feat1)
        # Polynomial + Categorical → PolynomialCategorical
        elif is_poly1 and is_cat2:
            iname = name or f"{feat1}:{feat2}"
            ispec = PolynomialCategorical(feat1, feat2)
        elif is_cat1 and is_poly2:
            iname = name or f"{feat2}:{feat1}"
            ispec = PolynomialCategorical(feat2, feat1)
        # Numeric + Categorical → NumericCategorical
        elif is_num1 and is_cat2:
            iname = name or f"{feat1}:{feat2}"
            ispec = NumericCategorical(feat1, feat2)
        elif is_cat1 and is_num2:
            iname = name or f"{feat2}:{feat1}"
            ispec = NumericCategorical(feat2, feat1)
        # Categorical + Categorical → CategoricalInteraction
        elif is_cat1 and is_cat2:
            iname = name or f"{feat1}:{feat2}"
            ispec = CategoricalInteraction(feat1, feat2)
        # Numeric + Numeric → NumericInteraction
        elif is_num1 and is_num2:
            iname = name or f"{feat1}:{feat2}"
            ispec = NumericInteraction(feat1, feat2)
        # Polynomial + Polynomial → PolynomialInteraction
        elif is_poly1 and is_poly2:
            iname = name or f"{feat1}:{feat2}"
            ispec = PolynomialInteraction(feat1, feat2)
        # Spline + Spline → TensorInteraction
        elif is_spline1 and is_spline2:
            iname = name or f"{feat1}:{feat2}"
            ispec = TensorInteraction(feat1, feat2, **kwargs)
        else:
            raise TypeError(
                f"Cannot create interaction between {type(spec1).__name__} "
                f"and {type(spec2).__name__}. Supported: Spline+Spline, "
                f"Spline+Categorical, Polynomial+Categorical, "
                f"Numeric+Categorical, Categorical+Categorical, "
                f"Numeric+Numeric, Polynomial+Polynomial."
            )

        if iname in self._interaction_specs:
            raise ValueError(f"Interaction already added: {iname}")

        self._interaction_specs[iname] = ispec
        self._interaction_order.append(iname)

    def _build_design_matrix(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray,
        offset: NDArray | None,
    ) -> tuple[NDArray, NDArray, NDArray | None]:
        """Build features, groups, design matrix.

        Sets self._dm, self._groups, self._distribution.
        Returns (y, exposure, offset) as float64 arrays.
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
        resolved_nb_theta = self.nb_theta if isinstance(self.nb_theta, int | float) else None
        self._distribution = resolve_distribution(
            self.family, tweedie_p=self.tweedie_p, nb_theta=resolved_nb_theta
        )
        self._link = resolve_link(self.link, self._distribution)

        group_matrices: list[GroupMatrix] = []
        col_offset = 0
        self._groups = []

        for name in self._feature_order:
            spec = self._specs[name]
            x_col = np.asarray(X[name])

            # Check if this feature should use fit-time discretization
            use_discrete = self._should_discretize(spec)
            B_unique = None
            bin_idx = None
            exposure_agg = None

            if use_discrete:
                from scipy.interpolate import BSpline as BSpl

                # Lightweight path: place knots and build penalty without O(n) basis.
                # Returns constraint-projected penalty for NaturalSpline/CRS.
                omega, n_cols_penalty, projection_penalty = spec.build_knots_and_penalty(x_col)
                n_bins_feat = self._resolve_discrete_n_bins(name, spec)
                bin_centers, bin_idx = _discretize_column(x_col, n_bins_feat)
                bin_centers_clip = np.clip(bin_centers, spec._knots[0], spec._knots[-1])
                B_unique = BSpl.design_matrix(bin_centers_clip, spec._knots, spec.degree).toarray()
                exposure_agg = np.bincount(bin_idx, weights=exposure, minlength=len(bin_centers))

                # Build infos without the full B matrix
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
                    # split_linear=True subgroup: shared B with projected subspace
                    P = info.projection

                    if info.reparametrize and info.penalty_matrix is not None:
                        # Range-space: compute R_inv in projected subspace
                        B_for_rinv = B_unique if use_discrete else info.columns
                        exp_for_rinv = exposure_agg if use_discrete else exposure
                        R_inv_local = self._compute_projected_R_inv(
                            B_for_rinv, P, info.penalty_matrix, exp_for_rinv
                        )
                        R_inv_combined = P @ R_inv_local  # (K, n_range)
                    else:
                        # Null-space: projection is the R_inv equivalent
                        R_inv_combined = P  # (K, n_null)

                    r_inv_parts.append(R_inv_combined)

                    if use_discrete:
                        gm: GroupMatrix = DiscretizedSSPGroupMatrix(
                            B_unique, R_inv_combined, bin_idx
                        )
                        if info.penalty_matrix is not None:
                            gm.omega = P @ info.penalty_matrix @ P.T
                        gm.projection = P
                    elif sp.issparse(info.columns):
                        gm = SparseSSPGroupMatrix(info.columns, R_inv_combined)
                        # Store full B-spline-space penalty for covariance/REML
                        if info.penalty_matrix is not None:
                            gm.omega = P @ info.penalty_matrix @ P.T
                        gm.projection = P
                    else:
                        gm = DenseGroupMatrix(info.columns @ R_inv_combined)

                elif info.reparametrize and info.penalty_matrix is not None:
                    if use_discrete:
                        R_inv = self._compute_R_inv(B_unique, info.penalty_matrix, exposure_agg)
                        if hasattr(spec, "set_reparametrisation"):
                            spec.set_reparametrisation(R_inv)
                        gm = DiscretizedSSPGroupMatrix(B_unique, R_inv, bin_idx)
                        gm.omega = info.penalty_matrix
                    else:
                        R_inv = self._compute_R_inv(
                            info.columns,
                            info.penalty_matrix,
                            exposure,
                        )
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
                self._groups.append(
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
        for pair in self._pending_interactions:
            if f"{pair[0]}:{pair[1]}" not in self._interaction_specs and (
                f"{pair[1]}:{pair[0]}" not in self._interaction_specs
            ):
                self._add_interaction(*pair)
        self._pending_interactions = []

        for iname in self._interaction_order:
            ispec = self._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            x1 = np.asarray(X[p1])
            x2 = np.asarray(X[p2])
            use_discrete_tensor = self._should_discretize_tensor_interaction(ispec)
            B_unique_inter = None
            bin_idx_inter = None
            exposure_agg_inter = None
            if use_discrete_tensor:
                n_bins1 = self._resolve_discrete_n_bins(p1, self._specs[p1])
                n_bins2 = self._resolve_discrete_n_bins(p2, self._specs[p2])
                result, B_unique_inter, bin_idx_inter = ispec.build_discrete(
                    x1,
                    x2,
                    self._specs,
                    (n_bins1, n_bins2),
                    exposure=exposure,
                )
                exposure_agg_inter = np.bincount(
                    bin_idx_inter,
                    weights=exposure,
                    minlength=B_unique_inter.shape[0],
                )
            else:
                result = ispec.build(x1, x2, self._specs, exposure=exposure)

            if isinstance(result, list):
                has_subgroups = any(
                    info.subgroup_name is not None or info.projection is not None for info in result
                )
                if has_subgroups:
                    r_inv_parts: list[NDArray] = []
                    for info in result:
                        if info.projection is not None:
                            P = info.projection
                            if info.reparametrize and info.penalty_matrix is not None:
                                B_for_rinv = B_unique_inter if use_discrete_tensor else info.columns
                                exp_for_rinv = (
                                    exposure_agg_inter if use_discrete_tensor else exposure
                                )
                                R_inv_local = self._compute_projected_R_inv(
                                    B_for_rinv, P, info.penalty_matrix, exp_for_rinv
                                )
                                R_inv_combined = P @ R_inv_local
                            else:
                                R_inv_combined = P
                            r_inv_parts.append(R_inv_combined)

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
                            R_inv = self._compute_R_inv(
                                B_for_rinv, info.penalty_matrix, exp_for_rinv
                            )
                            r_inv_parts.append(R_inv)
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
                        self._groups.append(
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

                    if r_inv_parts and hasattr(ispec, "set_reparametrisation"):
                        ispec.set_reparametrisation(np.hstack(r_inv_parts))
                else:
                    # Per-level groups (SplineCategorical, PolynomialCategorical)
                    r_inv_dict: dict[str, NDArray] = {}
                    for level, info in zip(ispec._non_base, result):
                        if info.reparametrize and info.penalty_matrix is not None:
                            R_inv = self._compute_R_inv(info.columns, info.penalty_matrix, exposure)
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
                        self._groups.append(
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
                    R_inv = self._compute_R_inv(B_for_rinv, info.penalty_matrix, exp_for_rinv)
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
                self._groups.append(
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

        self._dm = DesignMatrix(group_matrices, n, col_offset)
        return y, exposure, offset

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> SuperGLM:
        """Fit the model to data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix with columns matching registered features.
        y : array-like
            Response variable.
        exposure : array-like, optional
            **Frequency weights** (prior weights), typically policy exposure
            in insurance applications. Defaults to 1 for all observations.

            Exposure is a frequency weight: it represents the amount of risk
            observed, not observation precision. A policy with exposure=0.5
            (6 months on risk) contributes half as much information as one with
            exposure=1.0 (12 months). The standard assumption is that the
            expected response scales linearly with exposure:
            ``E[Y_i] = exposure_i * lambda_i``.

            For Poisson and Gamma models this only affects dispersion and
            standard errors. For Negative Binomial and Tweedie, exposure
            enters the profile likelihood for theta/p estimation, so the
            distinction between frequency and variance weights matters.

            Do **not** pass variance weights (e.g. credibility weights,
            inverse-variance weights) via this parameter — those require a
            different variance scaling that is not implemented.

            References: De Jong & Heller (2008) §5.4, §6.1; Ohlsson &
            Johansson (2010) Ch. 2; CAS Monograph No. 5 (Goldburd et al.,
            2016); Renshaw (1994) ASTIN Bulletin 24(2).
        offset : array-like, optional
            Offset added to the linear predictor. For count models with
            exposure, use ``offset=np.log(exposure)`` so that the model
            estimates a rate rather than a raw count.

        Returns
        -------
        SuperGLM
            The fitted model (self).
        """
        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)

        # Auto-estimate NB theta if requested
        if self.nb_theta == "auto":
            from superglm.nb_profile import estimate_nb_theta

            nb_result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset)
            self.nb_theta = nb_result.theta_hat
            self._nb_profile_result = nb_result
            logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = exposure  # store for covariance computation
        self._fit_offset = offset  # store for covariance computation

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max(y, exposure) * 0.1

        # Invalidate cached covariance from previous fit
        self.__dict__.pop("_coef_covariance", None)

        # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed)
        if self.penalty.lambda1 is not None and self.penalty.lambda1 == 0:
            self._result, _ = fit_irls_direct(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                lambda2=self.lambda2,
                offset=offset,
            )
        else:
            self._result = fit_pirls(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                penalty=self.penalty,
                offset=offset,
                anderson_memory=self._anderson_memory,
                active_set=self._active_set,
                lambda2=self.lambda2,
            )

        # Fix phi for known-scale families (Poisson): phi is always 1.0.
        # Solvers compute Pearson phi = dev/(n-edf) generically.
        scale_known = getattr(self._distribution, "scale_known", True)
        if scale_known and self._result.phi != 1.0:
            self._result = PIRLSResult(
                beta=self._result.beta,
                intercept=self._result.intercept,
                n_iter=self._result.n_iter,
                deviance=self._result.deviance,
                converged=self._result.converged,
                phi=1.0,
                effective_df=self._result.effective_df,
            )
        return self

    def fit_path(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        lambda_seq: NDArray | None = None,
    ) -> PathResult:
        """Fit a regularization path from lambda_max down to lambda_min.

        Warm-starts each lambda from the previous solution.
        """
        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = exposure
        self._fit_offset = offset
        self.__dict__.pop("_coef_covariance", None)
        lambda_max = self._compute_lambda_max(y, exposure)

        if lambda_seq is None:
            lambda_seq = np.geomspace(
                lambda_max,
                lambda_max * lambda_ratio,
                n_lambda,
            )
        else:
            lambda_seq = np.asarray(lambda_seq, dtype=np.float64)
            n_lambda = len(lambda_seq)

        p = self._dm.p
        coef_path = np.zeros((n_lambda, p))
        intercept_path = np.zeros(n_lambda)
        deviance_path = np.zeros(n_lambda)
        edf_path = np.zeros(n_lambda)
        n_iter_path = np.zeros(n_lambda, dtype=int)
        converged_path = np.zeros(n_lambda, dtype=bool)

        beta_warm = None
        intercept_warm = None

        for i, lam in enumerate(lambda_seq):
            self.penalty.lambda1 = lam
            result = fit_pirls(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                penalty=self.penalty,
                offset=offset,
                beta_init=beta_warm,
                intercept_init=intercept_warm,
                anderson_memory=self._anderson_memory,
                active_set=self._active_set,
                lambda2=self.lambda2,
            )
            coef_path[i] = result.beta
            intercept_path[i] = result.intercept
            deviance_path[i] = result.deviance
            edf_path[i] = result.effective_df
            n_iter_path[i] = result.n_iter
            converged_path[i] = result.converged
            beta_warm = result.beta
            intercept_warm = result.intercept

        # Set model state to the last (least-regularized) fit
        self._result = result

        return PathResult(
            lambda_seq=lambda_seq,
            coef_path=coef_path,
            intercept_path=intercept_path,
            deviance_path=deviance_path,
            n_iter_path=n_iter_path,
            converged_path=converged_path,
            edf_path=edf_path,
        )

    def fit_cv(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_folds: int = 5,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        lambda_seq: NDArray | None = None,
        rule: str = "1se",
        refit: bool = True,
        random_state: int | None = None,
    ):
        """Select lambda by K-fold cross-validation.

        Builds the design matrix on full data (fixing knots, categories, and
        SSP reparametrisation), then evaluates a lambda path on each fold.
        Selects the best lambda using the 1-SE rule or minimum CV deviance.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Frequency weights (exposure). Defaults to ones.
        offset : array-like, optional
            Offset added to the linear predictor.
        n_folds : int
            Number of CV folds (default 5).
        n_lambda : int
            Number of lambda values (default 50).
        lambda_ratio : float
            Ratio of lambda_min to lambda_max (default 1e-3).
        lambda_seq : array-like, optional
            Explicit lambda sequence (overrides n_lambda and lambda_ratio).
        rule : str
            Selection rule: ``"1se"`` (default) selects the most regularised
            lambda within 1 SE of the minimum; ``"min"`` selects the lambda
            with minimum CV deviance.
        refit : bool
            If True (default), refit on full data with the selected lambda.
            After refitting, ``model.predict()`` works immediately.
        random_state : int, optional
            Random seed for fold assignment.

        Returns
        -------
        CVResult
        """
        from superglm.cv import CVResult, _fit_cv_folds, _select_lambda

        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)

        if self.nb_theta == "auto":
            from superglm.nb_profile import estimate_nb_theta

            nb_result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset)
            self.nb_theta = nb_result.theta_hat
            self._nb_profile_result = nb_result
            logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = exposure
        self._fit_offset = offset
        self.__dict__.pop("_coef_covariance", None)

        # Lambda sequence from full data
        lambda_max = self._compute_lambda_max(y, exposure)
        if lambda_seq is None:
            lambda_seq = np.geomspace(lambda_max, lambda_max * lambda_ratio, n_lambda)
        else:
            lambda_seq = np.asarray(lambda_seq, dtype=np.float64)

        # Create fold indices
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(y))
        fold_indices = [arr for arr in np.array_split(indices, n_folds)]

        # Run CV
        fold_deviance = _fit_cv_folds(
            dm=self._dm,
            y=y,
            exposure=exposure,
            groups=self._groups,
            family=self._distribution,
            link=self._link,
            penalty=self.penalty,
            lambda_seq=lambda_seq,
            fold_indices=fold_indices,
            offset=offset,
            anderson_memory=self._anderson_memory,
            active_set=self._active_set,
        )

        mean_cv = fold_deviance.mean(axis=0)
        se_cv = fold_deviance.std(axis=0) / np.sqrt(n_folds)

        best_lambda, best_lambda_1se, best_idx, idx_1se = _select_lambda(
            lambda_seq, mean_cv, se_cv, rule
        )

        # Refit on full data with selected lambda
        path_result = None
        if refit:
            self.penalty.lambda1 = best_lambda
            self.fit(X, y, exposure=exposure, offset=offset)

        return CVResult(
            lambda_seq=lambda_seq,
            mean_cv_deviance=mean_cv,
            se_cv_deviance=se_cv,
            best_lambda=best_lambda,
            best_lambda_1se=best_lambda_1se,
            best_index=best_idx,
            best_index_1se=idx_1se,
            fold_deviance=fold_deviance,
            path_result=path_result,
        )

    def _should_discretize(self, spec: FeatureSpec) -> bool:
        """Check if a feature spec should use fit-time discretization."""
        from superglm.features.spline import _SplineBase

        if not isinstance(spec, _SplineBase):
            return False
        if spec.penalty != "ssp":
            return False
        if spec.discrete is not None:
            return spec.discrete
        return self._discrete

    def _should_discretize_tensor_interaction(self, ispec: Any) -> bool:
        """Check if a tensor interaction should use fit-time discretization."""
        from superglm.features.interaction import TensorInteraction

        if not isinstance(ispec, TensorInteraction):
            return False
        p1, p2 = ispec.parent_names
        return self._should_discretize(self._specs[p1]) and self._should_discretize(self._specs[p2])

    def _resolve_discrete_n_bins(self, name: str, spec: FeatureSpec) -> int:
        """Resolve the requested bin count for a discretized feature.

        Feature-level ``spec.n_bins`` takes priority. Otherwise the model-level
        ``n_bins`` may be a single int or a per-feature dict with a fallback of
        256 for unspecified features.
        """
        n_bins = getattr(spec, "n_bins", None)
        if n_bins is None:
            if isinstance(self._n_bins, dict):
                n_bins = self._n_bins.get(name, 256)
            else:
                n_bins = self._n_bins

        n_bins = int(n_bins)
        if n_bins < 1:
            raise ValueError(f"n_bins for feature '{name}' must be >= 1, got {n_bins}")
        return n_bins

    def _compute_R_inv(self, B, omega, exposure, lambda2_override=None):
        """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv."""
        lam2 = lambda2_override if lambda2_override is not None else self.lambda2
        if isinstance(lam2, dict):
            lam2 = 1.0  # dict lambda2 resolved per-group at fit time; use 1.0 for initial basis
        if sp.issparse(B):
            # B.multiply(exposure[:, None]) is sparse-safe
            G = np.asarray((B.multiply(exposure[:, None]).T @ B).todense()) / np.sum(exposure)
        else:
            G = (B * exposure[:, None]).T @ B / np.sum(exposure)
        M = G + lam2 * omega + np.eye(omega.shape[0]) * 1e-8
        R = np.linalg.cholesky(M).T
        return np.linalg.inv(R)

    def _compute_projected_R_inv(self, B, projection, penalty_sub, exposure, lambda2_override=None):
        """Compute SSP R_inv within a projected subspace (linear-split range space)."""
        lam2 = lambda2_override if lambda2_override is not None else self.lambda2
        if isinstance(lam2, dict):
            lam2 = 1.0  # dict lambda2 resolved per-group at fit time; use 1.0 for initial basis
        if sp.issparse(B):
            G_full = np.asarray((B.multiply(exposure[:, None]).T @ B).todense()) / np.sum(exposure)
        else:
            G_full = (B * exposure[:, None]).T @ B / np.sum(exposure)
        G_sub = projection.T @ G_full @ projection
        n_sub = penalty_sub.shape[0]
        M_sub = G_sub + lam2 * penalty_sub + np.eye(n_sub) * 1e-8
        R_sub = np.linalg.cholesky(M_sub).T
        return np.linalg.inv(R_sub)

    def _compute_lambda_max(self, y, weights):
        """Smallest lambda1 at which all groups are zeroed (null model).

        At beta=0 the intercept-only model has mu_i = y_bar (weighted).
        The score for group g is X_g' w (y - mu).  The KKT condition
        gives lambda_max = max_g ||score_g|| / (n * weight_g).
        """
        y_safe = np.where(y > 0, y, 0.1)
        mu_null = np.average(y_safe, weights=weights)
        residual = weights * (y - mu_null)
        grad = self._dm.rmatvec(residual)
        n = self._dm.n
        lmax = 0.0
        for g in self._groups:
            if not g.penalized:
                continue
            lmax = max(lmax, np.linalg.norm(grad[g.sl]) / g.weight)
        return lmax / n

    def _rebuild_design_matrix_with_lambdas(
        self, lambdas: dict[str, float], exposure: NDArray
    ) -> DesignMatrix:
        """Rebuild design matrix with per-group smoothing lambdas.

        Only recomputes R_inv for SSP groups whose lambda changed;
        non-SSP groups are reused unchanged.
        """
        new_gms: list[GroupMatrix] = []
        for gm, g in zip(self._dm.group_matrices, self._groups):
            if isinstance(gm, SparseSSPGroupMatrix) and g.name in lambdas:
                omega = gm.omega
                if omega is None:
                    new_gms.append(gm)
                    continue
                lam = lambdas[g.name]
                if gm.projection is not None:
                    P = gm.projection
                    omega_proj = P.T @ omega @ P
                    R_inv_local = self._compute_projected_R_inv(
                        gm.B, P, omega_proj, exposure, lambda2_override=lam
                    )
                    R_inv_new = P @ R_inv_local
                else:
                    R_inv_new = self._compute_R_inv(gm.B, omega, exposure, lambda2_override=lam)
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
                # Aggregate exposure by bin for R_inv computation
                exposure_agg = np.bincount(gm.bin_idx, weights=exposure, minlength=gm.n_bins)
                if gm.projection is not None:
                    P = gm.projection
                    omega_proj = P.T @ omega @ P
                    R_inv_local = self._compute_projected_R_inv(
                        gm.B_unique, P, omega_proj, exposure_agg, lambda2_override=lam
                    )
                    R_inv_new = P @ R_inv_local
                else:
                    R_inv_new = self._compute_R_inv(
                        gm.B_unique, omega, exposure_agg, lambda2_override=lam
                    )
                new_gm = DiscretizedSSPGroupMatrix(gm.B_unique, R_inv_new, gm.bin_idx)
                new_gm.omega = omega
                new_gm.projection = gm.projection
                new_gms.append(new_gm)
            else:
                new_gms.append(gm)
        return DesignMatrix(new_gms, self._dm.n, self._dm.p)

    def _compute_dW_deta(self, mu: NDArray, eta: NDArray, exposure: NDArray) -> NDArray | None:
        """Derivative of IRLS weights w.r.t. the linear predictor.

        W_i = exposure_i · (dμ/dη)² / V(μ)

        dW_i/dη = exposure_i · (dμ/dη / V(μ)) · [2(d²μ/dη²) − (dμ/dη)² V'(μ)/V(μ)]

        For log link: dW/dη = W·(2 − μV'(μ)/V(μ)).
        Poisson/log: dW/dη = W. Gamma/log: dW/dη = 0 identically.

        Returns None if the link or distribution does not provide the
        required second-order methods (deriv2_inverse, variance_derivative),
        which skips the W(ρ) correction for custom objects.
        """
        if not hasattr(self._link, "deriv2_inverse") or not hasattr(
            self._distribution, "variance_derivative"
        ):
            return None
        g1 = self._link.deriv_inverse(eta)  # dμ/dη
        g2 = self._link.deriv2_inverse(eta)  # d²μ/dη²
        V = np.maximum(self._distribution.variance(mu), 1e-10)
        Vp = self._distribution.variance_derivative(mu)
        return exposure * (g1 / V) * (2.0 * g2 - g1**2 * Vp / V)

    def _reml_w_correction(
        self,
        pirls_result: PIRLSResult,
        XtWX_S_inv: NDArray,
        lambdas: dict[str, float],
        reml_groups: list[tuple[int, GroupSlice]],
        penalty_caches: dict | None,
        exposure: NDArray,
        offset_arr: NDArray,
    ) -> tuple[NDArray, dict[int, NDArray]] | None:
        """First-order W(ρ) correction for REML derivatives.

        Computes the contribution from d(X'WX)/dρ_j = X'diag(dW/dρ_j)X
        which the fixed-W Laplace approximation drops.  The gradient
        correction is exact to first order; the Hessian C_j matrices are
        first-order (d²W/dρ² terms are dropped).

        Returns (grad_correction, dH_extra) or None if the correction vanishes
        (e.g. Gamma with log link where dW/dη = 0 identically) or if the
        link/distribution does not provide the required second-order methods.
        """
        eta = np.clip(
            self._dm.matvec(pirls_result.beta) + pirls_result.intercept + offset_arr,
            -20,
            20,
        )
        mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
        dW_deta = self._compute_dW_deta(mu, eta, exposure)

        if dW_deta is None:
            return None  # Custom link/distribution without second-order methods

        if np.max(np.abs(dW_deta)) < 1e-12:
            return None  # No correction (e.g. Gamma/log)

        p = XtWX_S_inv.shape[0]
        m = len(reml_groups)
        grad_correction = np.zeros(m)
        dH_extra: dict[int, NDArray] = {}

        gms = self._dm.group_matrices

        for i, (idx, g) in enumerate(reml_groups):
            if penalty_caches is not None:
                omega_ssp = penalty_caches[g.name].omega_ssp
            else:
                gm = gms[idx]
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            lam = lambdas[g.name]
            beta_g = pirls_result.beta[g.sl]

            # S_j β̂ (p-vector, nonzero only in g.sl block)
            s_beta = np.zeros(p)
            s_beta[g.sl] = lam * (omega_ssp @ beta_g)

            # dβ̂/dρ_j = -H⁻¹ S_j β̂  (IFT)
            dbeta_j = -(XtWX_S_inv @ s_beta)

            # dη/dρ_j = X dβ̂/dρ_j
            deta_j = self._dm.matvec(dbeta_j)

            # a_j = (dW/dη) ⊙ dη_j  — weights change per observation
            a_j = dW_deta * deta_j

            # C_j = X'diag(a_j)X — the dW contribution to dH/dρ_j
            # Uses _block_xtwx_signed which handles negative weights via
            # _gram_any_sign (SSP/Discretized use native gram, Dense/Sparse
            # use explicit W[:, None]*X fallback).
            C_j = _block_xtwx_signed(gms, self._groups, a_j)

            # Gradient correction: ½ tr(H⁻¹ C_j)
            grad_correction[i] = 0.5 * float(np.sum(XtWX_S_inv * C_j))

            dH_extra[i] = C_j

        return grad_correction, dH_extra

    def _reml_laml_objective(
        self,
        y: NDArray,
        result: PIRLSResult,
        lambdas: dict[str, float],
        exposure: NDArray,
        offset_arr: NDArray,
        XtWX: NDArray | None = None,
        penalty_caches: dict | None = None,
    ) -> float:
        """Laplace REML/LAML objective up to additive constants.

        Minimizing this objective over log-lambdas is the direct analogue of
        mgcv's outer REML optimization.  Handles both known-scale families
        (Poisson, NB2 where φ=1) and estimated-scale families (Gamma, Tweedie)
        via φ-profiled REML (Wood 2017, §6.2.2):

            Known scale (φ=1):
                V_R = -ℓ + ½β̂'Sβ̂ + ½log|H| - ½log|S|₊

            Estimated scale (φ̂ profiled):
                V_R = (1/φ̂)[-ℓ + ½β̂'Sβ̂] + ½log|H| - ½log|S|₊
                      + ((n - p_eff)/2)·log(φ̂)

        where φ̂ = (dev + β̂'Sβ̂) / (n - M_p) with M_p = Σ rank(Ω_j).
        """
        eta = np.clip(self._dm.matvec(result.beta) + result.intercept + offset_arr, -20, 20)
        mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
        if XtWX is None:
            V = self._distribution.variance(mu)
            dmu_deta = self._link.deriv_inverse(eta)
            W = exposure * dmu_deta**2 / np.maximum(V, 1e-10)
            XtWX = _block_xtwx(self._dm.group_matrices, self._groups, W)

        p = XtWX.shape[0]
        S = _build_penalty_matrix(self._dm.group_matrices, self._groups, lambdas, p)
        penalty_quad = float(result.beta @ S @ result.beta)

        # log|S|₊: use cached eigenstructure if available, otherwise compute
        if penalty_caches is not None:
            from superglm.reml import cached_logdet_s_plus

            logdet_s = cached_logdet_s_plus(lambdas, penalty_caches)
        else:
            eigvals_s = np.linalg.eigvalsh(S)
            thresh_s = 1e-10 * max(eigvals_s.max(), 1e-12)
            pos_s = eigvals_s[eigvals_s > thresh_s]
            logdet_s = float(np.sum(np.log(pos_s))) if pos_s.size else 0.0

        # log|H| = log|X'WX + S|
        M = XtWX + S
        eigvals_m = np.linalg.eigvalsh(M)
        thresh_m = 1e-10 * max(eigvals_m.max(), 1e-12)
        pos_m = eigvals_m[eigvals_m > thresh_m]
        logdet_m = float(np.sum(np.log(pos_m))) if pos_m.size else 0.0

        # φ-profiled REML for estimated-scale families (Gamma, Tweedie)
        # V = ½log|H| - ½log|S|₊ + ½(n-M_p)·log(D + β̂'Sβ̂)
        # The (D+PQ)/(2φ̂) term collapses to (n-M_p)/2 under profiling (constant).
        scale_known = getattr(self._distribution, "scale_known", True)
        if not scale_known:
            n = len(y)
            if penalty_caches is not None:
                M_p = sum(c.rank for c in penalty_caches.values())
            else:
                M_p = float(len(pos_s))  # fallback: rank of S
            d_plus_pq = max(result.deviance + penalty_quad, 1e-300)
            scale_term = 0.5 * max(n - M_p, 1.0) * np.log(d_plus_pq)
            return float(0.5 * (logdet_m - logdet_s) + scale_term)

        nll = -self._distribution.log_likelihood(y, mu, exposure, phi=1.0)
        return float(nll + 0.5 * (penalty_quad + logdet_m - logdet_s))

    def _reml_direct_gradient(
        self,
        result: PIRLSResult,
        XtWX_S_inv: NDArray,
        lambdas: dict[str, float],
        reml_groups: list[tuple[int, GroupSlice]],
        penalty_ranks: dict[str, float],
        phi_hat: float = 1.0,
    ) -> NDArray:
        """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W).

        This is the partial derivative holding IRLS weights W fixed.  The
        total gradient includes an additional W(ρ) correction term
        ½tr(H⁻¹ C_j) computed by ``_reml_w_correction``.

        For estimated-scale families, the quadratic term β'S_jβ is scaled
        by 1/φ̂ (profiled scale), while the trace term is unaffected since
        H_φ⁻¹ S_j/φ = H⁻¹ S_j (the φ cancels).
        """
        grad = np.zeros(len(reml_groups), dtype=np.float64)
        inv_phi = 1.0 / max(phi_hat, 1e-10)
        for i, (idx, g) in enumerate(reml_groups):
            gm = self._dm.group_matrices[idx]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            beta_g = result.beta[g.sl]
            quad = float(beta_g @ omega_ssp @ beta_g)
            H_inv_jj = XtWX_S_inv[g.sl, g.sl]
            trace_term = float(np.trace(H_inv_jj @ omega_ssp))
            lam = float(lambdas[g.name])
            # For estimated scale: quad scaled by 1/φ̂, trace unscaled
            grad[i] = 0.5 * (lam * (inv_phi * quad + trace_term) - penalty_ranks[g.name])
        return grad

    def _reml_direct_hessian(
        self,
        XtWX_S_inv: NDArray,
        lambdas: dict[str, float],
        reml_groups: list[tuple[int, GroupSlice]],
        gradient: NDArray,
        penalty_ranks: dict[str, float],
        penalty_caches: dict | None = None,
        pirls_result: object | None = None,
        n_obs: int = 0,
        phi_hat: float = 1.0,
        dH_extra: dict[int, NDArray] | None = None,
    ) -> NDArray:
        """Outer Hessian of the REML criterion w.r.t. log-lambdas.

        Differentiates through β̂(ρ) via the IFT (dβ̂/dρ_k = -H⁻¹ S_k β̂)
        and includes the first-order W(ρ) correction via dH_extra when
        provided.

        When ``dH_extra`` is not None, the trace product uses the first-order
        dH_j/dρ_j = S_j + C_j (where C_j = X'diag(dW/dρ_j)X) instead of
        the fixed-W approximation dH_j ≈ S_j.  Second-order d²W/dρ² terms
        are dropped, so this is approximate for families where dW/dη is
        large (e.g. NB2, Tweedie).

        Known scale:
            H_{jk} = δ_{jk}(g_j^∂ + ½r_j) - ½tr(H⁻¹ dH_j H⁻¹ dH_k)
                     - (S_jβ̂)'H⁻¹(S_kβ̂)

        Estimated scale (profiled φ̂):
            H_{jk} = δ_{jk}(g_j^∂ + ½r_j) - ½tr(H⁻¹ dH_j H⁻¹ dH_k)
                     - (1/φ̂)(S_jβ̂)'H⁻¹(S_kβ̂) - ½(n-M_p)q_jq_k/(D+PQ)²

        where g_j^∂ is the partial gradient (at fixed W), passed as
        ``gradient``.  The diagonal uses the partial gradient because the
        d(log|S|)/dρ diagonal contribution only involves dS_j/dρ_j = S_j.

        Parameters
        ----------
        gradient : (m,) array
            The *partial* gradient (at fixed W), not the total gradient.
        pirls_result : PIRLSResult, optional
            Needed for IFT and estimated-scale corrections.
        n_obs : int, optional
            Number of observations. Needed for estimated-scale correction.
        phi_hat : float
            Profiled scale parameter (1.0 for known-scale families).
        dH_extra : dict, optional
            W(ρ) correction matrices {group_index: C_j (p×p)}, from
            ``_reml_w_correction``.  When provided, the trace product
            uses H⁻¹(S_j + C_j) instead of H⁻¹ S_j.
        """
        m = len(reml_groups)
        p = XtWX_S_inv.shape[0]
        hess = np.zeros((m, m))

        # Form full p×p products H⁻¹ dH_j where dH_j = S_j + C_j
        full_HdHj: dict[int, NDArray] = {}
        quad_per_group: list[float] = []  # q_j = β̂'S_j β̂ for scale correction
        s_beta_list: list[NDArray] = []  # S_j β̂ vectors for IFT correction
        for i, (idx, g) in enumerate(reml_groups):
            if penalty_caches is not None:
                omega_ssp = penalty_caches[g.name].omega_ssp
            else:
                gm = self._dm.group_matrices[idx]
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            lam = lambdas[g.name]
            F = np.zeros((p, p))
            F[:, g.sl] = XtWX_S_inv[:, g.sl] @ (lam * omega_ssp)

            # Add W(ρ) correction: H⁻¹ C_j
            if dH_extra is not None and i in dH_extra:
                F = F + XtWX_S_inv @ dH_extra[i]

            full_HdHj[i] = F

            # S_j β̂ vector and quadratic form for IFT / scale corrections
            if pirls_result is not None:
                beta_g = pirls_result.beta[g.sl]
                quad_per_group.append(lam * float(beta_g @ omega_ssp @ beta_g))
                v = np.zeros(p)
                v[g.sl] = lam * (omega_ssp @ beta_g)
                s_beta_list.append(v)
            else:
                quad_per_group.append(0.0)
                s_beta_list.append(np.zeros(p))

        for i in range(m):
            for j in range(i, m):
                # -½ tr(H⁻¹ dH_i · H⁻¹ dH_j)
                h = -0.5 * float(np.sum(full_HdHj[i] * full_HdHj[j].T))
                hess[i, j] = h
                hess[j, i] = h
            # δ_{jj}(g_j^∂ + ½r_j) — uses partial gradient (at fixed W)
            name_i = reml_groups[i][1].name
            hess[i, i] += gradient[i] + 0.5 * penalty_ranks[name_i]

        # IFT correction: -(1/φ̂)(S_jβ̂)'H⁻¹(S_kβ̂) from dβ̂/dρ = -H⁻¹ S_k β̂
        if pirls_result is not None:
            inv_phi = 1.0 / max(phi_hat, 1e-10)
            S_beta = np.column_stack(s_beta_list)  # p × m
            HinvSbeta = XtWX_S_inv @ S_beta  # p × m
            hess -= inv_phi * (S_beta.T @ HinvSbeta)  # m × m

        # Rank-1 correction for estimated-scale families (profiled φ̂)
        scale_known = getattr(self._distribution, "scale_known", True)
        if not scale_known and pirls_result is not None and n_obs > 0:
            M_p = sum(penalty_ranks[g.name] for _, g in reml_groups)
            pq_total = sum(quad_per_group)
            d_plus_pq = max(pirls_result.deviance + pq_total, 1e-300)
            q = np.array(quad_per_group)
            hess -= 0.5 * max(n_obs - M_p, 1.0) * np.outer(q, q) / d_plus_pq**2

        return hess

    def _optimize_direct_reml(
        self,
        y: NDArray,
        exposure: NDArray,
        offset_arr: NDArray,
        reml_groups: list[tuple[int, GroupSlice]],
        penalty_ranks: dict[str, float],
        lambdas: dict[str, float],
        *,
        max_reml_iter: int,
        reml_tol: float,
        verbose: bool,
        penalty_caches: dict | None = None,
        profile: dict | None = None,
    ):
        """Optimize the direct REML objective via damped Newton (Wood 2011).

        Uses gradient and outer Hessian with first-order W(ρ) correction
        (IFT through β̂ + dW/dρ), PD-projected Newton step, step-halving
        line search, and steepest descent fallback.

        The W(ρ) correction accounts for d(X'WX)/dρ_j = X'diag(dW/dρ_j)X
        in the gradient (exact to first order) and Hessian (first-order;
        second-order d²W/dρ² terms are dropped).  For Gamma with log link,
        dW/dη = 0 identically so the correction vanishes.  For Poisson with
        log link, dW/dη = W, giving a material correction.
        """
        from superglm.reml import REMLResult

        scale_known = getattr(self._distribution, "scale_known", True)
        group_names = [g.name for _, g in reml_groups]
        m = len(group_names)
        log_lo, log_hi = np.log(1e-6), np.log(1e6)
        max_newton_step = 5.0  # mgcv maxNstep=5

        lambda_history: list[dict[str, float]] = [lambdas.copy()]
        warm_beta: NDArray | None = None
        warm_intercept: float | None = None
        grad_tol = max(reml_tol, 1e-6)

        best_obj = np.inf
        best_lambdas = lambdas.copy()
        best_pirls = None
        best_grad: NDArray | None = None
        converged = False
        n_iter = 0
        n_warmup = 3  # fixed-point iterations after bootstrap

        import time as _time

        _t_reml_start = _time.perf_counter()
        _t_pirls = 0.0
        _t_objective = 0.0
        _t_gradient = 0.0
        _t_hessian = 0.0
        _t_w_correction = 0.0
        _t_linesearch = 0.0
        _t_fp_update = 0.0
        _n_linesearch_fits = 0

        # === Bootstrap: one FP step from minimal penalty ===
        # Fit PIRLS with λ_min so that β is not suppressed for any group.
        # The resulting FP update gives data-driven initial lambdas that are
        # independent of the user's lambda2_init, ensuring start-invariant
        # convergence.
        boot_lambdas = {name: 1e-4 for name in lambdas}
        _t0 = _time.perf_counter()
        boot_result, boot_inv, boot_xtwx = fit_irls_direct(
            X=self._dm,
            y=y,
            weights=exposure,
            family=self._distribution,
            link=self._link,
            groups=self._groups,
            lambda2=boot_lambdas,
            offset=offset_arr,
            return_xtwx=True,
            profile=profile,
        )
        _t_pirls += _time.perf_counter() - _t0
        warm_beta = boot_result.beta.copy()
        warm_intercept = float(boot_result.intercept)

        # Compute profiled φ̂ for the bootstrap fit
        boot_phi = 1.0
        if not scale_known and penalty_caches is not None:
            p_dim = boot_xtwx.shape[0]
            S_boot = _build_penalty_matrix(
                self._dm.group_matrices, self._groups, boot_lambdas, p_dim
            )
            pq_boot = float(boot_result.beta @ S_boot @ boot_result.beta)
            M_p = sum(c.rank for c in penalty_caches.values())
            boot_phi = max((boot_result.deviance + pq_boot) / max(len(y) - M_p, 1.0), 1e-10)
        boot_inv_phi = 1.0 / max(boot_phi, 1e-10)

        rho = np.zeros(m, dtype=np.float64)
        for i, (idx, g) in enumerate(reml_groups):
            gm = self._dm.group_matrices[idx]
            if penalty_caches is not None:
                omega_ssp = penalty_caches[g.name].omega_ssp
            else:
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            beta_g = boot_result.beta[g.sl]
            quad = float(beta_g @ omega_ssp @ beta_g)
            H_inv_jj = boot_inv[g.sl, g.sl]
            trace_term = float(np.trace(H_inv_jj @ omega_ssp))
            r_j = penalty_ranks[g.name]
            denom = boot_inv_phi * quad + trace_term
            lam_fp = r_j / denom if denom > 1e-12 else 1.0
            rho[i] = np.clip(np.log(max(lam_fp, 1e-6)), log_lo, log_hi)

        rho_prev = rho.copy()

        if verbose:
            boot_lam_str = ", ".join(
                f"{name}={np.exp(rho[i]):.4g}" for i, name in enumerate(group_names)
            )
            print(f"  REML bootstrap: lambdas=[{boot_lam_str}]")

        for outer in range(max_reml_iter):
            n_iter = outer + 1
            rho_clipped = np.clip(rho, log_lo, log_hi)

            # Build candidate lambdas from current rho
            cand_lambdas = lambdas.copy()
            for name, val in zip(group_names, np.exp(rho_clipped), strict=False):
                cand_lambdas[name] = float(np.clip(val, 1e-6, 1e6))

            # === Converge PIRLS for this λ ===
            _t0 = _time.perf_counter()
            pirls_result, XtWX_S_inv, XtWX = fit_irls_direct(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                lambda2=cand_lambdas,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                return_xtwx=True,
                profile=profile,
            )
            _t_pirls += _time.perf_counter() - _t0
            warm_beta = pirls_result.beta.copy()
            warm_intercept = float(pirls_result.intercept)

            # === REML objective ===
            _t0 = _time.perf_counter()
            obj = self._reml_laml_objective(
                y,
                pirls_result,
                cand_lambdas,
                exposure,
                offset_arr,
                XtWX=XtWX,
                penalty_caches=penalty_caches,
            )

            # Compute profiled φ̂ for estimated-scale derivatives
            phi_hat = 1.0
            if not scale_known and penalty_caches is not None:
                p_dim = XtWX.shape[0]
                S_eval = _build_penalty_matrix(
                    self._dm.group_matrices, self._groups, cand_lambdas, p_dim
                )
                pq = float(pirls_result.beta @ S_eval @ pirls_result.beta)
                M_p = sum(c.rank for c in penalty_caches.values())
                phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
            inv_phi = 1.0 / max(phi_hat, 1e-10)
            _t_objective += _time.perf_counter() - _t0

            # === Gradient: partial (fixed W) + W(ρ) correction ===
            _t0 = _time.perf_counter()
            grad_partial = self._reml_direct_gradient(
                pirls_result,
                XtWX_S_inv,
                cand_lambdas,
                reml_groups,
                penalty_ranks,
                phi_hat=phi_hat,
            )
            _t_gradient += _time.perf_counter() - _t0

            # W(ρ) correction: accounts for d(X'WX)/dρ_j = X'diag(dW/dρ_j)X
            # Skip during FP warmup (correction only affects Newton direction)
            # and on discrete path (already approximate, matches mgcv bam behavior)
            _t0 = _time.perf_counter()
            if outer >= n_warmup and not self._discrete:
                w_corr = self._reml_w_correction(
                    pirls_result,
                    XtWX_S_inv,
                    cand_lambdas,
                    reml_groups,
                    penalty_caches,
                    exposure,
                    offset_arr,
                )
            else:
                w_corr = None
            _t_w_correction += _time.perf_counter() - _t0
            if w_corr is not None:
                grad_w_correction, dH_extra = w_corr
                grad = grad_partial + grad_w_correction
            else:
                grad = grad_partial.copy()
                dH_extra = None

            # Track best
            if obj < best_obj:
                best_obj = obj
                best_lambdas = cand_lambdas.copy()
                best_pirls = pirls_result
                best_grad = grad.copy()

            lambda_history.append(cand_lambdas.copy())

            # === Convergence check ===
            # Projected gradient: zero out components at boundary pushing outward
            proj_grad = grad.copy()
            for i in range(m):
                if rho_clipped[i] >= log_hi - 0.01 and grad[i] < 0:
                    proj_grad[i] = 0.0  # at upper bound, wants higher λ
                elif rho_clipped[i] <= log_lo + 0.01 and grad[i] > 0:
                    proj_grad[i] = 0.0  # at lower bound, wants lower λ
            proj_grad_norm = float(np.max(np.abs(proj_grad)))
            rho_change = float(np.max(np.abs(rho_clipped - rho_prev)))

            if verbose:
                phase = "FP" if outer < n_warmup else "Newton"
                lam_str = ", ".join(f"{name}={cand_lambdas[name]:.4g}" for name in group_names)
                print(
                    f"  REML {phase} iter={n_iter}  obj={obj:.4f}  "
                    f"|∇|={proj_grad_norm:.6f}  Δρ={rho_change:.4f}  "
                    f"lambdas=[{lam_str}]"
                )

            rho_prev = rho_clipped.copy()

            # Converge when projected gradient is small
            if outer >= n_warmup and proj_grad_norm < max(reml_tol, 5e-3):
                converged = True
                break

            # === Phase 1: Fixed-point warm-up ===
            if outer < n_warmup:
                _t0 = _time.perf_counter()
                for i, (idx, g) in enumerate(reml_groups):
                    gm = self._dm.group_matrices[idx]
                    if penalty_caches is not None:
                        omega_ssp = penalty_caches[g.name].omega_ssp
                    else:
                        omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
                    beta_g = pirls_result.beta[g.sl]
                    quad = float(beta_g @ omega_ssp @ beta_g)
                    H_inv_jj = XtWX_S_inv[g.sl, g.sl]
                    trace_term = float(np.trace(H_inv_jj @ omega_ssp))
                    r_j = penalty_ranks[g.name]
                    denom = inv_phi * quad + trace_term
                    lam_new = r_j / denom if denom > 1e-12 else cand_lambdas[g.name]
                    rho[i] = np.clip(np.log(max(lam_new, 1e-6)), log_lo, log_hi)
                _t_fp_update += _time.perf_counter() - _t0
                continue

            # === Phase 2: Newton with exact outer Hessian ===
            _t0 = _time.perf_counter()
            hess = self._reml_direct_hessian(
                XtWX_S_inv,
                cand_lambdas,
                reml_groups,
                grad_partial,  # partial gradient for diagonal term
                penalty_ranks,
                penalty_caches=penalty_caches,
                pirls_result=pirls_result,
                n_obs=len(y),
                phi_hat=phi_hat,
                dH_extra=dH_extra,
            )

            # PD-projected Newton step (with SD fallback)
            eigvals_h, eigvecs_h = np.linalg.eigh(hess)
            max_eig = max(eigvals_h.max(), 1e-12)
            eigvals_pd = np.maximum(eigvals_h, 1e-6 * max_eig)

            if eigvals_h.min() < -0.1 * max_eig:
                step_scale = min(1.0, max_newton_step / max(np.linalg.norm(grad), 1e-8))
                delta = -grad * step_scale
            else:
                hess_pd = (eigvecs_h * eigvals_pd) @ eigvecs_h.T
                delta = -np.linalg.solve(hess_pd, grad)
                delta = np.clip(delta, -max_newton_step, max_newton_step)
            _t_hessian += _time.perf_counter() - _t0

            # === Step-halving line search with Armijo condition ===
            _t0 = _time.perf_counter()
            step = 1.0
            armijo_c = 1e-4
            descent = float(grad @ delta)
            accepted = False
            for ls in range(8):  # max 8 halvings
                rho_trial = np.clip(rho_clipped + step * delta, log_lo, log_hi)
                trial_lambdas = lambdas.copy()
                for name, val in zip(group_names, np.exp(rho_trial), strict=False):
                    trial_lambdas[name] = float(np.clip(val, 1e-6, 1e6))

                _n_linesearch_fits += 1
                trial_result, trial_inv, trial_xtwx = fit_irls_direct(
                    X=self._dm,
                    y=y,
                    weights=exposure,
                    family=self._distribution,
                    link=self._link,
                    groups=self._groups,
                    lambda2=trial_lambdas,
                    offset=offset_arr,
                    beta_init=warm_beta,
                    intercept_init=warm_intercept,
                    return_xtwx=True,
                    profile=profile,
                )

                trial_obj = self._reml_laml_objective(
                    y,
                    trial_result,
                    trial_lambdas,
                    exposure,
                    offset_arr,
                    XtWX=trial_xtwx,
                    penalty_caches=penalty_caches,
                )

                # Armijo sufficient decrease
                if trial_obj <= obj + armijo_c * step * descent:
                    rho = rho_trial
                    warm_beta = trial_result.beta.copy()
                    warm_intercept = float(trial_result.intercept)
                    accepted = True
                    break
                step *= 0.5
            _t_linesearch += _time.perf_counter() - _t0

            if not accepted:
                # Line search failed — accept tiny steepest descent step
                rho = np.clip(
                    rho_clipped - 0.1 * grad / max(np.linalg.norm(grad), 1e-8),
                    log_lo,
                    log_hi,
                )

        if best_pirls is None:
            raise RuntimeError("Direct REML Newton did not evaluate any candidates")

        grad_norm = float(np.max(np.abs(best_grad))) if best_grad is not None else np.inf
        converged = converged or grad_norm <= grad_tol

        # Populate profile with outer-loop phase timing
        if profile is not None:
            profile["reml_optimizer_s"] = _time.perf_counter() - _t_reml_start
            profile["reml_pirls_s"] = _t_pirls
            profile["reml_objective_s"] = _t_objective
            profile["reml_gradient_s"] = _t_gradient
            profile["reml_w_correction_s"] = _t_w_correction
            profile["reml_hessian_newton_s"] = _t_hessian
            profile["reml_linesearch_s"] = _t_linesearch
            profile["reml_fp_update_s"] = _t_fp_update
            profile["reml_n_linesearch_fits"] = _n_linesearch_fits
            profile["reml_n_outer_iter"] = n_iter

        return REMLResult(
            lambdas=best_lambdas,
            pirls_result=best_pirls,
            n_reml_iter=n_iter,
            converged=converged,
            lambda_history=lambda_history,
            objective=float(best_obj),
        )

    def _run_reml_once(
        self,
        y: NDArray,
        exposure: NDArray,
        offset_arr: NDArray,
        reml_groups: list[tuple[int, GroupSlice]],
        penalty_ranks: dict[str, float],
        lambdas: dict[str, float],
        *,
        max_reml_iter: int,
        reml_tol: float,
        verbose: bool,
        use_direct: bool,
        penalty_caches: dict | None = None,
    ):
        """Run a single REML fixed-point outer loop from a chosen initial lambda scale."""
        from superglm.metrics import _penalised_xtwx_inv_gram
        from superglm.reml import REMLResult, _map_beta_between_bases

        scale_known = getattr(self._distribution, "scale_known", True)

        if use_direct:
            reml_update_names = [g.name for _, g in reml_groups]
        else:
            reml_update_names = [g.name for _, g in reml_groups if penalty_ranks[g.name] > 1]

        warm_beta = None
        warm_intercept = None
        lambda_history: list[dict[str, float]] = [lambdas.copy()]
        converged = False
        n_reml_iter = 0
        aa_prev_log_x: NDArray | None = None
        aa_prev_log_gx: NDArray | None = None
        cheap_iter = False
        cached_direct_xtwx: NDArray | None = None
        last_pirls_iters = 0
        direct_has_scalar_groups = any(penalty_ranks[g.name] <= 1 for _, g in reml_groups)
        direct_cheap_threshold = 0.01 if direct_has_scalar_groups else 0.2
        bcd_cheap_threshold = 0.01

        for reml_iter in range(max_reml_iter):
            n_reml_iter = reml_iter + 1

            if use_direct and not cheap_iter:
                pirls_result, XtWX_S_inv_full, XtWX_full = fit_irls_direct(
                    X=self._dm,
                    y=y,
                    weights=exposure,
                    family=self._distribution,
                    link=self._link,
                    groups=self._groups,
                    lambda2=lambdas,
                    offset=offset_arr,
                    beta_init=warm_beta,
                    intercept_init=warm_intercept,
                    return_xtwx=True,
                )
                beta = pirls_result.beta
                intercept = pirls_result.intercept
                last_pirls_iters = pirls_result.n_iter
                cached_direct_xtwx = XtWX_full

                eta = np.clip(self._dm.matvec(beta) + intercept + offset_arr, -20, 20)
                mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
                V = self._distribution.variance(mu)
                dmu_deta = self._link.deriv_inverse(eta)
                W = exposure * dmu_deta**2 / np.maximum(V, 1e-10)

                active_groups = list(self._groups)
                XtWX_S_inv = XtWX_S_inv_full
            elif not use_direct and not cheap_iter:
                pirls_result = fit_pirls(
                    X=self._dm,
                    y=y,
                    weights=exposure,
                    family=self._distribution,
                    link=self._link,
                    groups=self._groups,
                    penalty=self.penalty,
                    offset=offset_arr,
                    beta_init=warm_beta,
                    intercept_init=warm_intercept,
                    anderson_memory=self._anderson_memory,
                    active_set=self._active_set,
                    lambda2=lambdas,
                )
                beta = pirls_result.beta
                intercept = pirls_result.intercept
                last_pirls_iters = pirls_result.n_iter

                eta = np.clip(self._dm.matvec(beta) + intercept + offset_arr, -20, 20)
                mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
                V = self._distribution.variance(mu)
                dmu_deta = self._link.deriv_inverse(eta)
                W = exposure * dmu_deta**2 / np.maximum(V, 1e-10)

            if use_direct and cheap_iter:
                if cached_direct_xtwx is None:
                    raise RuntimeError("REML cheap iteration missing cached direct XtWX")
                XtWX_S_inv = _invert_xtwx_plus_penalty(
                    cached_direct_xtwx, self._dm.group_matrices, self._groups, lambdas
                )
                active_groups = list(self._groups)
            elif not use_direct:
                XtWX_S_inv, active_groups = _penalised_xtwx_inv_gram(
                    beta, W, self._dm.group_matrices, self._groups, lambdas
                )

            # Compute profiled φ̂ for estimated-scale fixed-point update
            inv_phi = 1.0
            if not scale_known and penalty_caches is not None:
                p_dim = self._dm.p
                S_fp = _build_penalty_matrix(self._dm.group_matrices, self._groups, lambdas, p_dim)
                pq = float(beta @ S_fp @ beta)
                M_p = sum(c.rank for c in penalty_caches.values())
                phi_hat = max((pirls_result.deviance + pq) / max(len(y) - M_p, 1.0), 1e-10)
                inv_phi = 1.0 / phi_hat

            lambdas_new = lambdas.copy()
            for idx, g in reml_groups:
                if not use_direct and penalty_ranks[g.name] <= 1:
                    continue

                gm = self._dm.group_matrices[idx]
                beta_g = beta[g.sl]
                if np.linalg.norm(beta_g) < 1e-12:
                    continue

                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
                quad = float(beta_g @ omega_ssp @ beta_g)

                ag = next((a for a in active_groups if a.name == g.name), None)
                if ag is None:
                    continue

                H_inv_jj = XtWX_S_inv[ag.sl, ag.sl]
                trace_term = float(np.trace(H_inv_jj @ omega_ssp))

                r_j = penalty_ranks[g.name]
                # For estimated scale: quad scaled by 1/φ̂, trace unscaled
                denom = inv_phi * quad + trace_term
                lam_new = r_j / denom if denom > 1e-12 else lambdas[g.name]
                lambdas_new[g.name] = float(np.clip(lam_new, 1e-6, 1e6))

            if aa_prev_log_x is not None and len(reml_update_names) > 0:
                log_x = np.array([np.log(lambdas[n]) for n in reml_update_names])
                log_gx = np.array([np.log(lambdas_new[n]) for n in reml_update_names])
                f_curr = log_gx - log_x
                f_prev = aa_prev_log_gx - aa_prev_log_x
                df = f_curr - f_prev
                df_sq = float(np.dot(df, df))
                if df_sq > 1e-20:
                    theta = float(-np.dot(f_curr, df) / df_sq)
                    theta = max(-0.5, min(theta, 2.0))
                    log_acc = (1.0 + theta) * log_gx - theta * aa_prev_log_gx
                    for i, name in enumerate(reml_update_names):
                        lambdas_new[name] = float(np.clip(np.exp(log_acc[i]), 1e-6, 1e6))

            if len(reml_update_names) > 0:
                aa_prev_log_x = np.array([np.log(lambdas[n]) for n in reml_update_names])
                aa_prev_log_gx = np.array([np.log(lambdas_new[n]) for n in reml_update_names])

            if use_direct:
                changes = [
                    abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                    for _, g in reml_groups
                    if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
                ]
            else:
                changes = [
                    abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                    for _, g in reml_groups
                    if lambdas[g.name] > 0 and lambdas_new[g.name] > 0 and penalty_ranks[g.name] > 1
                ]
                if not changes:
                    changes = [
                        abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                        for _, g in reml_groups
                        if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
                    ]
            max_change = max(changes) if changes else 0.0

            if verbose:
                lam_str = ", ".join(f"{g.name}={lambdas_new[g.name]:.4g}" for _, g in reml_groups)
                mode = "cheap" if cheap_iter else f"pirls={last_pirls_iters}"
                print(
                    f"  REML iter={n_reml_iter}  max_change={max_change:.6f}  "
                    f"({mode})  lambdas=[{lam_str}]"
                )

            lambda_history.append(lambdas_new.copy())

            if max_change < reml_tol:
                converged = True
                lambdas = lambdas_new
                break

            if use_direct:
                warm_beta = beta
                warm_intercept = intercept
                cheap_iter = max_change <= direct_cheap_threshold
            elif max_change > bcd_cheap_threshold:
                old_gms = self._dm.group_matrices
                self._dm = self._rebuild_design_matrix_with_lambdas(lambdas_new, exposure)
                warm_beta = _map_beta_between_bases(
                    beta, old_gms, self._dm.group_matrices, self._groups
                )
                warm_intercept = intercept
                cheap_iter = False
            else:
                cheap_iter = True

            lambdas = lambdas_new

        if cheap_iter and converged and not use_direct:
            self._dm = self._rebuild_design_matrix_with_lambdas(lambdas, exposure)

        if use_direct:
            final_result, _ = fit_irls_direct(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                lambda2=lambdas,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
            )
        else:
            final_result = fit_pirls(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                penalty=self.penalty,
                offset=offset_arr,
                beta_init=warm_beta,
                intercept_init=warm_intercept,
                anderson_memory=self._anderson_memory,
                active_set=self._active_set,
                lambda2=lambdas,
            )

        # For the BCD path, penalty_caches may be stale after basis rebuild
        # (R_inv changes but caches still hold old omega_ssp). Recompute from
        # scratch for the final objective — this is called only once.
        # NOTE: This is a narrow fix for the final objective only.  If future
        # code reuses penalty_caches after _rebuild_design_matrix_with_lambdas
        # on the BCD path, caches must be rebuilt there too.
        final_caches = penalty_caches if use_direct else None
        return REMLResult(
            lambdas=lambdas,
            pirls_result=final_result,
            n_reml_iter=n_reml_iter,
            converged=converged,
            lambda_history=lambda_history,
            objective=self._reml_laml_objective(
                y,
                final_result,
                lambdas,
                exposure,
                offset_arr,
                penalty_caches=final_caches,
            ),
        )

    def fit_reml(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        max_reml_iter: int = 20,
        reml_tol: float = 1e-4,
        lambda2_init: float | None = None,
        verbose: bool = False,
    ) -> SuperGLM:
        """Fit with REML estimation of per-term smoothing parameters.

        When ``lambda1=0``, the exact/direct path optimizes a Laplace
        approximate REML objective over log-lambdas. When group selection is
        also active, REML falls back to the existing Wood (2011) fixed-point
        outer loop around PIRLS.

        REML coexists with group lasso: REML controls within-group
        smoothness (per-term lambda_j), group lasso controls between-group
        selection (lambda1). They are orthogonal.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Frequency weights.
        offset : array-like, optional
            Offset term.
        max_reml_iter : int
            Maximum REML outer iterations (default 20).
        reml_tol : float
            Convergence tolerance on log-lambda (default 1e-4).
        lambda2_init : float, optional
            Initial per-group lambda. Defaults to ``self.lambda2``.
        verbose : bool
            Print progress.

        Returns
        -------
        SuperGLM
            The fitted model (self).
        """
        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)

        # Auto-estimate NB theta if requested
        if self.nb_theta == "auto":
            from superglm.nb_profile import estimate_nb_theta

            nb_result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset)
            self.nb_theta = nb_result.theta_hat
            self._nb_profile_result = nb_result
            logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

        import time as _time

        _t_total_start = _time.perf_counter()
        _profile: dict = {}

        _t0 = _time.perf_counter()
        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        _profile["dm_build_s"] = _time.perf_counter() - _t0

        self._fit_weights = exposure
        self._fit_offset = offset
        self.__dict__.pop("_coef_covariance", None)

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max(y, exposure) * 0.1

        # Identify REML-eligible groups: penalized SSP groups with stored omega
        reml_groups: list[tuple[int, GroupSlice]] = []
        for i, g in enumerate(self._groups):
            gm = self._dm.group_matrices[i]
            if (
                isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix)
                and g.penalized
                and gm.omega is not None
            ):
                reml_groups.append((i, g))

        if not reml_groups:
            logger.warning("fit_reml: no REML-eligible groups found, falling back to fit()")
            self._result = fit_pirls(
                X=self._dm,
                y=y,
                weights=exposure,
                family=self._distribution,
                link=self._link,
                groups=self._groups,
                penalty=self.penalty,
                offset=offset,
                anderson_memory=self._anderson_memory,
                active_set=self._active_set,
                lambda2=self.lambda2,
            )
            return self

        # Initialize per-group lambdas
        lam_init = lambda2_init if lambda2_init is not None else self.lambda2
        lambdas = {g.name: lam_init for _, g in reml_groups}

        # Build penalty caches (eigenstructure computed once, reused across iterations)
        from superglm.reml import build_penalty_caches

        penalty_caches = build_penalty_caches(self._dm.group_matrices, self._groups, reml_groups)
        penalty_ranks = {name: cache.rank for name, cache in penalty_caches.items()}

        # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed)
        offset_arr = offset if offset is not None else np.zeros(len(y))
        use_direct = self.penalty.lambda1 is not None and self.penalty.lambda1 == 0

        if use_direct:
            best = self._optimize_direct_reml(
                y,
                exposure,
                offset_arr,
                reml_groups,
                penalty_ranks,
                lambdas,
                max_reml_iter=max_reml_iter,
                reml_tol=reml_tol,
                verbose=verbose,
                penalty_caches=penalty_caches,
                profile=_profile,
            )
        else:
            best = self._run_reml_once(
                y,
                exposure,
                offset_arr,
                reml_groups,
                penalty_ranks,
                lambdas,
                max_reml_iter=max_reml_iter,
                reml_tol=reml_tol,
                verbose=verbose,
                use_direct=False,
                penalty_caches=penalty_caches,
            )
        self._result = best.pirls_result
        self._reml_lambdas = best.lambdas
        self._reml_result = best
        lambdas = best.lambdas
        n_reml_iter = best.n_reml_iter
        converged = best.converged

        # Fix phi: known-scale families (Poisson) get phi=1.0;
        # estimated-scale families get REML profiled φ̂ instead of the raw
        # PIRLS phi = dev/(n-edf) which doesn't include the penalty.
        scale_known = getattr(self._distribution, "scale_known", True)
        if scale_known:
            phi_fixed = 1.0
        else:
            p_dim = self._dm.p
            S_final = _build_penalty_matrix(self._dm.group_matrices, self._groups, lambdas, p_dim)
            pq_final = float(best.pirls_result.beta @ S_final @ best.pirls_result.beta)
            M_p = sum(penalty_ranks[g.name] for _, g in reml_groups)
            phi_fixed = max((best.pirls_result.deviance + pq_final) / max(len(y) - M_p, 1.0), 1e-10)

        corrected = PIRLSResult(
            beta=best.pirls_result.beta,
            intercept=best.pirls_result.intercept,
            n_iter=best.pirls_result.n_iter,
            deviance=best.pirls_result.deviance,
            converged=best.pirls_result.converged,
            phi=phi_fixed,
            effective_df=best.pirls_result.effective_df,
        )
        self._result = corrected
        self._reml_result.pirls_result = corrected

        # Update spec R_inv for predict/reconstruct
        for idx, g in reml_groups:
            gm = self._dm.group_matrices[idx]
            spec = self._specs.get(g.feature_name)
            if spec is not None and hasattr(spec, "set_reparametrisation"):
                # Gather R_inv from all groups of this feature
                feature_groups = self._feature_groups(g.feature_name)
                r_inv_parts = []
                for fg in feature_groups:
                    fg_idx = next(i for i, gg in enumerate(self._groups) if gg.name == fg.name)
                    fg_gm = self._dm.group_matrices[fg_idx]
                    if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                        r_inv_parts.append(fg_gm.R_inv)
                if r_inv_parts:
                    spec.set_reparametrisation(
                        np.hstack(r_inv_parts) if len(r_inv_parts) > 1 else r_inv_parts[0]
                    )

        # Same for interaction specs
        for iname in self._interaction_order:
            ispec = self._interaction_specs[iname]
            if not hasattr(ispec, "set_reparametrisation"):
                continue
            feature_groups = self._feature_groups(iname)
            # Check if any group was updated by REML
            updated = any(fg.name in lambdas for fg in feature_groups)
            if not updated:
                continue
            if len(feature_groups) > 1:
                if any(fg.subgroup_type is not None for fg in feature_groups):
                    r_inv_parts = []
                    for fg in feature_groups:
                        fg_idx = next(i for i, gg in enumerate(self._groups) if gg.name == fg.name)
                        fg_gm = self._dm.group_matrices[fg_idx]
                        if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                            r_inv_parts.append(fg_gm.R_inv)
                    if r_inv_parts:
                        ispec.set_reparametrisation(np.hstack(r_inv_parts))
                else:
                    # Per-level (SplineCategorical): gather dict
                    r_inv_dict = {}
                    for fg in feature_groups:
                        fg_idx = next(i for i, gg in enumerate(self._groups) if gg.name == fg.name)
                        fg_gm = self._dm.group_matrices[fg_idx]
                        if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                            level = fg.name.split("[")[1].rstrip("]") if "[" in fg.name else fg.name
                            r_inv_dict[level] = fg_gm.R_inv
                    if r_inv_dict:
                        ispec.set_reparametrisation(r_inv_dict)
            else:
                # Single group (TensorInteraction)
                fg = feature_groups[0]
                fg_idx = next(i for i, gg in enumerate(self._groups) if gg.name == fg.name)
                fg_gm = self._dm.group_matrices[fg_idx]
                if isinstance(fg_gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                    ispec.set_reparametrisation(fg_gm.R_inv)

        _profile["total_s"] = _time.perf_counter() - _t_total_start
        _profile["n_reml_iter"] = n_reml_iter
        _profile["converged"] = converged
        self._reml_profile = _profile

        logger.info(f"REML converged={converged} in {n_reml_iter} iters, lambdas={lambdas}")
        return self

    @property
    def result(self) -> PIRLSResult:
        if self._result is None:
            raise RuntimeError("Not fitted")
        return self._result

    def summary(self) -> dict[str, Any]:
        res = self.result
        out = {}
        for g in self._groups:
            bg = res.beta[g.sl]
            out[g.name] = {
                "active": bool(np.any(bg != 0)),
                "group_norm": float(np.linalg.norm(bg)),
                "n_params": g.size,
            }
        out["_model"] = {
            "intercept": res.intercept,
            "deviance": res.deviance,
            "phi": res.phi,
            "effective_df": res.effective_df,
            "n_iter": res.n_iter,
            "converged": res.converged,
            "lambda1": self.penalty.lambda1,
        }
        return out

    def _feature_groups(self, name: str) -> list[GroupSlice]:
        """Get all groups belonging to a feature (1 normally, 2 for split-linear splines)."""
        return [g for g in self._groups if g.feature_name == name]

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        res = self.result
        groups = self._feature_groups(name)
        beta_combined = np.concatenate([res.beta[g.sl] for g in groups])
        if name in self._specs:
            return self._specs[name].reconstruct(beta_combined)
        if name in self._interaction_specs:
            return self._interaction_specs[name].reconstruct(beta_combined)
        raise KeyError(f"Feature not found: {name}")

    @cached_property
    def _coef_covariance(self) -> tuple[NDArray, list[GroupSlice]]:
        """Phi-scaled Bayesian covariance for active coefficients.

        Returns (Cov_active, active_groups) where:
        - Cov_active: (p_active, p_active) = phi * (X'WX + S)^{-1}
        - active_groups: list of GroupSlice re-indexed to Cov_active columns
        """
        from superglm.metrics import _penalised_xtwx_inv_gram

        res = self.result
        beta = res.beta
        eta = self._dm.matvec(beta) + res.intercept
        if self._fit_offset is not None:
            eta = eta + self._fit_offset
        eta = np.clip(eta, -20, 20)
        mu = self._link.inverse(eta)
        V = self._distribution.variance(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        W = self._fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

        lam2 = getattr(self, "_reml_lambdas", None) or self.lambda2
        XtWX_S_inv, active_groups = _penalised_xtwx_inv_gram(
            beta, W, self._dm.group_matrices, self._groups, lam2
        )
        return res.phi * XtWX_S_inv, active_groups

    def estimate_p(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        **kwargs,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        Thin wrapper around :func:`superglm.tweedie_profile.estimate_tweedie_p`.
        After estimation, sets ``self.tweedie_p`` to the optimised value and
        refits the model.

        Returns
        -------
        TweedieProfileResult
        """
        from superglm.tweedie_profile import estimate_tweedie_p

        result = estimate_tweedie_p(self, X, y, exposure=exposure, offset=offset, **kwargs)
        self.tweedie_p = result.p_hat
        self._tweedie_profile_result = result
        self.fit(X, y, exposure=exposure, offset=offset)
        return result

    def estimate_theta(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        **kwargs,
    ):
        """Estimate NB theta via profile likelihood, refit, and return result.

        Thin wrapper around :func:`superglm.nb_profile.estimate_nb_theta`.
        After estimation, sets ``self.nb_theta`` to the optimised value and
        refits the model.

        Returns
        -------
        NBProfileResult
        """
        from superglm.nb_profile import estimate_nb_theta

        result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset, **kwargs)
        self.nb_theta = result.theta_hat
        self._nb_profile_result = result
        self.fit(X, y, exposure=exposure, offset=offset)
        return result

    def metrics(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model.

        Returns a ModelMetrics object with information criteria, residuals,
        leverage, Cook's distance, etc.
        """
        from superglm.metrics import ModelMetrics

        return ModelMetrics(self, X, y, exposure, offset)

    def drop1(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        test: str = "Chisq",
    ) -> pd.DataFrame:
        """Drop-one deviance analysis for each feature.

        For each feature, refits the model with that feature (and any
        dependent interactions) removed, keeping the same penalty
        configuration. Compares deviances via a chi-squared or F test
        using effective degrees of freedom.

        .. note::

           This is an *approximate* deviance comparison, not a classical
           likelihood ratio test. P-values use effective df (hat matrix
           trace) rather than parametric df and should be treated as
           approximate guides, not exact tests.

           After ``fit_reml()``, reduced models inherit the full model's
           smoothing parameters as fixed values — REML is **not**
           re-run for each reduced model. This is computationally
           practical and follows the spirit of ``mgcv::anova.gam``,
           but means the comparison conditions on the full model's
           smoothing selection.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (same as used for fitting).
        y : array-like
            Response variable.
        exposure : array-like, optional
            Frequency weights.
        offset : array-like, optional
            Offset added to the linear predictor.
        test : {"Chisq", "F"}
            ``"Chisq"`` for known-scale families (Poisson).
            ``"F"`` for estimated-scale families (Gamma, NB2, Tweedie).

        Returns
        -------
        pd.DataFrame
            Rows sorted by p-value with columns: feature, deviance_full,
            deviance_reduced, delta_deviance, delta_df, statistic, p_value.
        """
        from scipy.stats import chi2
        from scipy.stats import f as f_dist

        if self._result is None:
            raise RuntimeError("Model must be fitted before calling drop1().")

        dev_full = self._result.deviance
        edf_full = self._result.effective_df
        n = len(y) if not hasattr(y, "__len__") else len(y)
        phi = self._result.phi

        rows = []
        for name in self._feature_order:
            # Identify dependent interactions
            drop_set = {name}
            for iname in self._interaction_order:
                ispec = self._interaction_specs[iname]
                p1, p2 = ispec.parent_names
                if p1 == name or p2 == name:
                    drop_set.add(iname)

            remaining = [f for f in self._feature_order if f not in drop_set]

            if not remaining:
                # Intercept-only model: compute null deviance directly
                y_arr = np.asarray(y, dtype=np.float64)
                w = (
                    np.ones(n, dtype=np.float64)
                    if exposure is None
                    else np.asarray(exposure, dtype=np.float64)
                )
                if offset is not None:
                    # With offset: solve for intercept b0 such that
                    # mu = link^{-1}(b0 + offset) minimises deviance.
                    # Moment approximation: set b0 so mean(mu) ≈ mean(y).
                    offset_arr = np.asarray(offset, dtype=np.float64)
                    y_mean = np.average(y_arr, weights=w)
                    b0 = self._link.link(max(y_mean, 1e-10)) - np.average(offset_arr, weights=w)
                    null_mu = np.maximum(self._link.inverse(b0 + offset_arr), 1e-10)
                else:
                    null_mu = np.full(n, np.average(y_arr, weights=w))
                dev_reduced = float(np.sum(w * self._distribution.deviance_unit(y_arr, null_mu)))
                edf_reduced = 1.0  # intercept only
            else:
                reduced = self._clone_without_features(drop_set)
                reduced.fit(X, y, exposure=exposure, offset=offset)
                dev_reduced = reduced.result.deviance
                edf_reduced = reduced.result.effective_df
            delta_dev = dev_reduced - dev_full
            delta_df = max(edf_full - edf_reduced, 1e-4)

            if test == "F":
                stat = (delta_dev / delta_df) / phi
                resid_df = max(n - edf_full, 1.0)
                p_value = float(f_dist.sf(stat, delta_df, resid_df))
            else:
                stat = delta_dev
                p_value = float(chi2.sf(stat, delta_df))

            rows.append(
                {
                    "feature": name,
                    "deviance_full": dev_full,
                    "deviance_reduced": dev_reduced,
                    "delta_deviance": delta_dev,
                    "delta_df": delta_df,
                    "statistic": stat,
                    "p_value": p_value,
                }
            )

        df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
        return df

    def refit_unpenalised(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        keep_smoothing: bool = True,
    ) -> SuperGLM:
        """Refit the model with only the active features and no selection penalty.

        After a penalized fit that selected features via group lasso, this
        method refits with ``lambda1=0`` on only the active features, removing
        the shrinkage bias from L1 selection.

        When ``keep_smoothing=True``, the smoothing penalty (lambda2 or
        REML-estimated lambdas) is inherited as fixed values — smoothing
        parameters are **not** re-optimized via REML on the reduced model.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Frequency weights.
        offset : array-like, optional
            Offset added to the linear predictor.
        keep_smoothing : bool
            If True (default), keep the smoothing penalty (lambda2 or
            REML-estimated lambdas). If False, set lambda2=0 for a
            fully unpenalised refit.

        Returns
        -------
        SuperGLM
            A new fitted model with only the active features.
        """
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling refit_unpenalised().")

        beta = self._result.beta

        # Identify inactive features
        inactive = set()
        for name in self._feature_order:
            groups = self._feature_groups(name)
            if all(np.linalg.norm(beta[g.sl]) < 1e-12 for g in groups):
                inactive.add(name)

        # Also drop interactions whose parents are inactive
        for iname in self._interaction_order:
            ispec = self._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            if p1 in inactive or p2 in inactive:
                inactive.add(iname)

        lam2: float | dict[str, float] | None
        if not keep_smoothing:
            lam2 = 0.0
        else:
            lam2 = ...  # sentinel: use original lambda2 / REML lambdas

        new_model = self._clone_without_features(inactive, lambda1=0.0, lambda2=lam2)
        new_model.fit(X, y, exposure=exposure, offset=offset)
        return new_model

    def relativities(self, with_se: bool = False) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features.

        Parameters
        ----------
        with_se : bool
            If True, add an ``se_log_relativity`` column to each DataFrame.
            Uses phi-scaled (quasi-likelihood) covariance.

        Returns dict keyed by feature name. Each DataFrame has columns
        ``relativity`` and ``log_relativity`` plus a type-specific index column:

        - Spline / Polynomial (has ``x``): ``x``, ``relativity``, ``log_relativity``
        - Categorical (has ``levels``): ``level``, ``relativity``, ``log_relativity``
        - Numeric (has ``relativity_per_unit``): ``label``, ``relativity``, ``log_relativity``
        """

        # Pre-compute covariance if SEs requested
        if with_se:
            Cov_active, active_groups = self._coef_covariance

        result: dict[str, pd.DataFrame] = {}
        for name in self._feature_order:
            raw = self.reconstruct_feature(name)
            if "x" in raw:
                # Spline or Polynomial
                df = pd.DataFrame(
                    {
                        "x": raw["x"],
                        "relativity": raw["relativity"],
                        "log_relativity": raw["log_relativity"],
                    }
                )
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups, n_points=len(raw["x"])
                    )
                result[name] = df
            elif "levels" in raw:
                # Categorical
                levels = raw["levels"]
                rels = raw["relativities"]
                log_rels = raw["log_relativities"]
                df = pd.DataFrame(
                    {
                        "level": levels,
                        "relativity": [rels[lv] for lv in levels],
                        "log_relativity": [log_rels[lv] for lv in levels],
                    }
                )
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups
                    )
                result[name] = df
            elif "relativity_per_unit" in raw:
                # Numeric
                rel = raw["relativity_per_unit"]
                df = pd.DataFrame(
                    {
                        "label": ["per_unit"],
                        "relativity": [rel],
                        "log_relativity": [np.log(rel)],
                    }
                )
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups
                    )
                result[name] = df

        # Interaction relativities — dispatch on reconstruct dict keys
        for iname in self._interaction_order:
            raw = self.reconstruct_feature(iname)

            if "per_level" in raw and "x" in raw:
                # SplineCategorical / PolynomialCategorical: per-level curves
                for level in raw["levels"]:
                    level_data = raw["per_level"][level]
                    key = f"{iname}[{level}]"
                    df = pd.DataFrame(
                        {
                            "x": raw["x"],
                            "relativity": level_data["relativity"],
                            "log_relativity": level_data["log_relativity"],
                        }
                    )
                    result[key] = df

            elif "pairs" in raw:
                # CategoricalInteraction: per-pair relativities
                pairs_labels = [f"{l1}:{l2}" for l1, l2 in raw["pairs"]]
                rels = raw["relativities"]
                log_rels = raw["log_relativities"]
                df = pd.DataFrame(
                    {
                        "level": pairs_labels,
                        "relativity": [rels[k] for k in pairs_labels],
                        "log_relativity": [log_rels[k] for k in pairs_labels],
                    }
                )
                result[iname] = df

            elif "relativities_per_unit" in raw:
                # NumericCategorical: per-level slope relativities
                levels = raw["levels"]
                rels = raw["relativities_per_unit"]
                log_rels = raw["log_relativities_per_unit"]
                df = pd.DataFrame(
                    {
                        "level": levels,
                        "relativity_per_unit": [rels[lv] for lv in levels],
                        "log_relativity_per_unit": [log_rels[lv] for lv in levels],
                    }
                )
                result[iname] = df

            elif "relativity_per_unit_unit" in raw:
                # NumericInteraction: single product coefficient
                b_orig = raw["coef_original"]
                df = pd.DataFrame(
                    {
                        "label": ["per_unit_unit"],
                        "relativity": [raw["relativity_per_unit_unit"]],
                        "log_relativity": [b_orig],
                    }
                )
                result[iname] = df

            elif "x1" in raw and "x2" in raw:
                # PolynomialInteraction: 2D surface — store raw dict
                # (doesn't fit 1D DataFrame; use reconstruct_feature() directly)
                pass

        return result

    def _feature_se_from_cov(
        self,
        name: str,
        Cov_active: NDArray,
        active_groups: list[GroupSlice],
        n_points: int = 200,
    ) -> NDArray:
        """Compute feature-level SEs from the precomputed covariance matrix."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import _SplineBase

        beta = self.result.beta
        groups = self._feature_groups(name)
        spec = self._specs.get(name) or self._interaction_specs.get(name)

        # Inactive feature: zeros (all subgroups zeroed)
        beta_combined = np.concatenate([beta[g.sl] for g in groups])
        if np.linalg.norm(beta_combined) < 1e-12:
            if isinstance(spec, _SplineBase):
                return np.zeros(n_points)
            elif isinstance(spec, Categorical):
                return np.zeros(len(spec._levels))
            else:
                return np.zeros(1)

        # Gather covariance blocks from all active subgroups
        active_subs = [ag for ag in active_groups if ag.feature_name == name]
        if not active_subs:
            if isinstance(spec, _SplineBase):
                return np.zeros(n_points)
            elif isinstance(spec, Categorical):
                return np.zeros(len(spec._levels))
            else:
                return np.zeros(1)

        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        Cov_g = Cov_active[np.ix_(indices, indices)]

        if isinstance(spec, _SplineBase):
            from scipy.interpolate import BSpline as BSpl

            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            x_clip = np.clip(x_grid, spec._knots[0], spec._knots[-1])
            B_grid = BSpl.design_matrix(x_clip, spec._knots, spec.degree).toarray()
            M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid

            # For split_linear=True: only use columns for active subgroups
            active_cols = np.concatenate(
                [
                    np.arange(g.start, g.end) - groups[0].start
                    for g in groups
                    if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
                ]
            )
            M = M[:, active_cols]

            Q = M @ Cov_g
            return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

        elif isinstance(spec, Categorical):
            se_nonbase = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            se_all = np.zeros(len(spec._levels))
            for i, lev in enumerate(spec._levels):
                if lev != spec._base_level:
                    idx = spec._non_base.index(lev)
                    se_all[i] = se_nonbase[idx]
            return se_all

        elif isinstance(spec, Numeric):
            se_transformed = np.sqrt(max(Cov_g[0, 0], 0.0))
            if spec.standardize:
                return np.array([se_transformed / spec._std])
            return np.array([se_transformed])

        else:
            return np.sqrt(np.maximum(np.diag(Cov_g), 0.0))

    def simultaneous_bands(
        self,
        feature: str,
        *,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        n_points: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simultaneous confidence bands for a spline feature.

        Uses the Wood (2006) simulation approach: draws from the posterior
        ``MVN(0, Cov_g)``, computes the supremum of the standardised deviation
        across the curve, and returns the ``(1-alpha)`` quantile as the critical
        value for the simultaneous band.

        Parameters
        ----------
        feature : str
            Name of a spline feature.
        alpha : float
            Significance level (default 0.05 for 95% bands).
        n_sim : int
            Number of posterior simulations (default 10,000).
        n_points : int
            Grid size for evaluating the curve.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: ``x``, ``log_relativity``, ``relativity``, ``se``,
            ``ci_lower_pointwise``, ``ci_upper_pointwise``,
            ``ci_lower_simultaneous``, ``ci_upper_simultaneous``.
        """
        from scipy.interpolate import BSpline as BSpl
        from scipy.stats import norm

        from superglm.features.spline import _SplineBase

        if self._result is None:
            raise RuntimeError("Model must be fitted before calling simultaneous_bands().")

        spec = self._specs.get(feature)
        if not isinstance(spec, _SplineBase):
            raise TypeError(
                f"simultaneous_bands() only supports spline features, "
                f"got {type(spec).__name__} for '{feature}'."
            )

        # Get covariance and basis matrix (same logic as _feature_se_from_cov)
        Cov_active, active_groups = self._coef_covariance
        beta = self.result.beta
        groups = self._feature_groups(feature)

        active_subs = [ag for ag in active_groups if ag.feature_name == feature]
        if not active_subs:
            raise ValueError(f"Feature '{feature}' is inactive (all coefficients zeroed).")

        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        Cov_g = Cov_active[np.ix_(indices, indices)]

        # Build basis evaluation matrix
        x_grid = np.linspace(spec._lo, spec._hi, n_points)
        x_clip = np.clip(x_grid, spec._knots[0], spec._knots[-1])
        B_grid = BSpl.design_matrix(x_clip, spec._knots, spec.degree).toarray()
        M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid

        # For split_linear=True: only use columns for active subgroups
        active_cols = np.concatenate(
            [
                np.arange(g.start, g.end) - groups[0].start
                for g in groups
                if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
            ]
        )
        M = M[:, active_cols]

        # Pointwise SEs
        Q = M @ Cov_g
        se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

        # Log-relativity on grid
        beta_g = np.concatenate(
            [
                beta[g.sl]
                for g in groups
                if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
            ]
        )
        log_rel = M @ beta_g

        # Simultaneous critical value via simulation (Wood 2006)
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(Cov_g + 1e-12 * np.eye(Cov_g.shape[0]))
        beta_sim = rng.standard_normal((n_sim, Cov_g.shape[0])) @ L.T
        f_sim = beta_sim @ M.T  # (n_sim, n_points)

        se_safe = np.maximum(se, 1e-20)
        T_sim = np.max(np.abs(f_sim) / se_safe[np.newaxis, :], axis=1)
        c_sim = float(np.quantile(T_sim, 1.0 - alpha))

        z = norm.ppf(1.0 - alpha / 2.0)

        return pd.DataFrame(
            {
                "x": x_grid,
                "log_relativity": log_rel,
                "relativity": np.exp(log_rel),
                "se": se,
                "ci_lower_pointwise": np.exp(log_rel - z * se),
                "ci_upper_pointwise": np.exp(log_rel + z * se),
                "ci_lower_simultaneous": np.exp(log_rel - c_sim * se),
                "ci_upper_simultaneous": np.exp(log_rel + c_sim * se),
            }
        )

    def plot_relativities(
        self,
        X: pd.DataFrame | None = None,
        exposure: NDArray | None = None,
        with_ci: bool = True,
        **kwargs,
    ):
        """Plot relativity curves/bars for all features.

        Parameters
        ----------
        X : DataFrame, optional
            Training data — when provided with *exposure*, overlays the
            exposure distribution on each subplot.
        exposure : array-like, optional
            Exposure weights corresponding to rows of *X*.
        with_ci : bool
            If *True* (default), show 95 % confidence bands / error bars.
        **kwargs
            Forwarded to :func:`superglm.plotting.plot_relativities`.
        """
        from superglm.plotting import plot_relativities

        return plot_relativities(
            self.relativities(with_se=with_ci),
            X=X,
            exposure=exposure,
            with_ci=with_ci,
            **kwargs,
        )

    def plot_interaction(
        self,
        name: str,
        *,
        engine: str = "matplotlib",
        with_ci: bool = True,
        **kwargs,
    ):
        """Plot an interaction surface/effect.

        Parameters
        ----------
        name : str
            Interaction name, e.g. ``"DrivAge:Area"``.
        engine : {"matplotlib", "plotly"}
            Plotting backend.  ``"plotly"`` requires plotly to be installed.
        with_ci : bool
            Show confidence bands where applicable.
        **kwargs
            Forwarded to :func:`superglm.plotting.plot_interaction`.
        """
        from superglm.plotting import plot_interaction

        return plot_interaction(self, name, engine=engine, with_ci=with_ci, **kwargs)

    def discretization_impact(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing spline/polynomial curves.

        See :func:`superglm.discretize.discretization_impact` for full docs.
        """
        from superglm.discretize import discretization_impact

        return discretization_impact(self, X, y, exposure, **kwargs)

    def predict(self, X: pd.DataFrame, offset: NDArray | None = None) -> NDArray:
        blocks = []
        for name in self._feature_order:
            spec = self._specs[name]
            blocks.append(spec.transform(np.asarray(X[name])))

        for iname in self._interaction_order:
            ispec = self._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            blocks.append(ispec.transform(np.asarray(X[p1]), np.asarray(X[p2])))

        eta = np.hstack(blocks) @ self.result.beta + self.result.intercept
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=np.float64)
        return self._link.inverse(eta)
