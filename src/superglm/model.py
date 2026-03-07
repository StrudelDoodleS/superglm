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
    _discretize_column,
)
from superglm.links import Link, resolve_link
from superglm.penalties.base import Penalty
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.solvers.irls_direct import _invert_xtwx_plus_penalty, fit_irls_direct
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
        n_bins: int = 256,
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
                n_bins_feat = spec.n_bins if spec.n_bins is not None else self._n_bins
                bin_centers, bin_idx = _discretize_column(x_col, n_bins_feat)
                bin_centers_clip = np.clip(bin_centers, spec._knots[0], spec._knots[-1])
                B_unique = BSpl.design_matrix(bin_centers_clip, spec._knots, spec.degree).toarray()
                exposure_agg = np.bincount(bin_idx, weights=exposure, minlength=len(bin_centers))

                # Build infos without the full B matrix
                from superglm.features.spline import Spline

                if isinstance(spec, Spline) and spec.select:
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
                    # select=True subgroup: shared B with projected subspace
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
            result = ispec.build(x1, x2, self._specs, exposure=exposure)

            if isinstance(result, list):
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
                    R_inv = self._compute_R_inv(info.columns, info.penalty_matrix, exposure)
                    if hasattr(ispec, "set_reparametrisation"):
                        ispec.set_reparametrisation(R_inv)
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
        """Compute SSP R_inv within a projected subspace (select=True range space)."""
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

        REML adds an outer loop around the existing PIRLS solver:
        fix lambda_j → warm-start PIRLS → compute H^{-1} → update lambda_j
        via the Wood (2011) fixed-point formula. Typically 3-6 outer
        iterations.

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
        from superglm.metrics import _penalised_xtwx_inv_gram
        from superglm.reml import REMLResult, _map_beta_between_bases

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

        # Compute penalty ranks (rank of R_inv.T @ omega @ R_inv)
        penalty_ranks: dict[str, float] = {}
        for idx, g in reml_groups:
            gm = self._dm.group_matrices[idx]
            omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv
            eigvals = np.linalg.eigvalsh(omega_ssp)
            penalty_ranks[g.name] = float(np.sum(eigvals > 1e-8 * max(eigvals.max(), 1e-12)))

        # Direct IRLS when lambda1=0 (no L1 penalty → no BCD needed)
        _use_direct = self.penalty.lambda1 is not None and self.penalty.lambda1 == 0

        # Names of groups eligible for REML lambda updates.
        # Direct solver: estimate ALL lambdas (no BCD aliasing → 1-col groups are fine).
        # BCD solver: skip 1-col groups (weakly identified, causes drift).
        if _use_direct:
            _reml_update_names = [g.name for _, g in reml_groups]
        else:
            _reml_update_names = [g.name for _, g in reml_groups if penalty_ranks[g.name] > 1]

        # REML outer loop
        warm_beta = None
        warm_intercept = None
        lambda_history: list[dict[str, float]] = [lambdas.copy()]
        converged = False
        n_reml_iter = 0
        offset_arr = offset if offset is not None else np.zeros(len(y))
        _aa_prev_log_x: NDArray | None = None  # Anderson(1) state
        _aa_prev_log_gx: NDArray | None = None
        _cheap_iter = False  # True → skip PIRLS/DM rebuild, only update lambdas
        _cached_direct_xtwx: NDArray | None = None
        _last_pirls_iters = 0
        # Direct REML can switch to cheap updates earlier when all updated groups
        # are genuine multi-parameter smooths. Models with 1-column subgroups
        # (notably select=True linear pieces) are more sensitive, so keep the
        # conservative threshold there.
        _direct_has_scalar_groups = any(penalty_ranks[g.name] <= 1 for _, g in reml_groups)
        _direct_cheap_threshold = 0.01 if _direct_has_scalar_groups else 0.2
        _bcd_cheap_threshold = 0.01

        for reml_iter in range(max_reml_iter):
            n_reml_iter = reml_iter + 1

            if _use_direct and not _cheap_iter:
                # Direct IRLS path: solver returns XtWX_S_inv directly
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
                _last_pirls_iters = pirls_result.n_iter
                _cached_direct_xtwx = XtWX_full

                # Compute working weights for potential cheap iterations
                eta = np.clip(self._dm.matvec(beta) + intercept + offset_arr, -20, 20)
                mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
                V = self._distribution.variance(mu)
                dmu_deta = self._link.deriv_inverse(eta)
                W = exposure * dmu_deta**2 / np.maximum(V, 1e-10)

                # Build active_groups mapping (all groups, re-indexed to full beta)
                # For the direct solver, all groups are "active" (no L1 zeroing)
                active_groups = list(self._groups)
                XtWX_S_inv = XtWX_S_inv_full
            elif not _use_direct and not _cheap_iter:
                # 1. Run PIRLS with current lambdas (warm-started)
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
                _last_pirls_iters = pirls_result.n_iter

                # 2. Compute working weights
                eta = np.clip(self._dm.matvec(beta) + intercept + offset_arr, -20, 20)
                mu = np.clip(self._link.inverse(eta), 1e-7, 1e7)
                V = self._distribution.variance(mu)
                dmu_deta = self._link.deriv_inverse(eta)
                W = exposure * dmu_deta**2 / np.maximum(V, 1e-10)
            # else (_cheap_iter): reuse beta, W from previous full iteration

            if _use_direct and _cheap_iter:
                if _cached_direct_xtwx is None:
                    raise RuntimeError("REML cheap iteration missing cached direct XtWX")
                XtWX_S_inv = _invert_xtwx_plus_penalty(
                    _cached_direct_xtwx, self._dm.group_matrices, self._groups, lambdas
                )
                active_groups = list(self._groups)
            elif not _use_direct:
                # Compute H^{-1} = (X'WX + S)^{-1} via gram-based fast path.
                # For cheap iterations: uses old beta/W but new lambdas.
                # For BCD full iterations: separate computation needed.
                XtWX_S_inv, active_groups = _penalised_xtwx_inv_gram(
                    beta, W, self._dm.group_matrices, self._groups, lambdas
                )

            # 4. Fixed-point update per REML-eligible group
            lambdas_new = lambdas.copy()
            for idx, g in reml_groups:
                # BCD path: skip 1-column groups (weakly identified, causes drift).
                # Direct path: estimate all lambdas (no BCD aliasing).
                if not _use_direct and penalty_ranks[g.name] <= 1:
                    continue

                gm = self._dm.group_matrices[idx]
                beta_g = beta[g.sl]

                # Skip zeroed groups
                if np.linalg.norm(beta_g) < 1e-12:
                    continue

                # Omega_ssp = R_inv.T @ omega @ R_inv (unscaled by lambda)
                omega_ssp = gm.R_inv.T @ gm.omega @ gm.R_inv

                # Quadratic form: beta' @ Omega_ssp @ beta
                quad = float(beta_g @ omega_ssp @ beta_g)

                # Trace term: tr(H^{-1}[j,j] @ Omega_ssp)
                ag = None
                for a in active_groups:
                    if a.name == g.name:
                        ag = a
                        break
                if ag is None:
                    continue

                H_inv_jj = XtWX_S_inv[ag.sl, ag.sl]
                trace_term = float(np.trace(H_inv_jj @ omega_ssp))

                # Fixed-point: lambda_j = r_j / (beta'Omega beta + tr(H^{-1} Omega))
                r_j = penalty_ranks[g.name]
                denom = quad + trace_term
                if denom > 1e-12:
                    lam_new = r_j / denom
                else:
                    lam_new = lambdas[g.name]

                # Clamp to reasonable range
                lambdas_new[g.name] = float(np.clip(lam_new, 1e-6, 1e6))

            # 4b. Anderson(1) acceleration on log-lambda scale.
            # Accelerates linearly converging fixed-point iterations.
            if _aa_prev_log_x is not None and len(_reml_update_names) > 0:
                log_x = np.array([np.log(lambdas[n]) for n in _reml_update_names])
                log_gx = np.array([np.log(lambdas_new[n]) for n in _reml_update_names])
                f_curr = log_gx - log_x
                f_prev = _aa_prev_log_gx - _aa_prev_log_x
                df = f_curr - f_prev
                df_sq = float(np.dot(df, df))
                if df_sq > 1e-20:
                    theta = float(-np.dot(f_curr, df) / df_sq)
                    theta = max(-0.5, min(theta, 2.0))  # stability guard
                    log_acc = (1.0 + theta) * log_gx - theta * _aa_prev_log_gx
                    for i, n in enumerate(_reml_update_names):
                        lambdas_new[n] = float(np.clip(np.exp(log_acc[i]), 1e-6, 1e6))

            if len(_reml_update_names) > 0:
                _aa_prev_log_x = np.array([np.log(lambdas[n]) for n in _reml_update_names])
                _aa_prev_log_gx = np.array([np.log(lambdas_new[n]) for n in _reml_update_names])

            # 5. Check convergence.
            # BCD path: only check multi-parameter smooth terms (1-col groups
            #   converge slowly due to weak identification and don't affect fit).
            # Direct path: check all groups (no aliasing → all converge properly).
            if _use_direct:
                changes = [
                    abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                    for _, g in reml_groups
                    if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
                ]
            else:
                changes = [
                    abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                    for idx, g in reml_groups
                    if lambdas[g.name] > 0 and lambdas_new[g.name] > 0 and penalty_ranks[g.name] > 1
                ]
                if not changes:
                    # All groups are 1-column — fall back to full check
                    changes = [
                        abs(np.log(lambdas_new[g.name]) - np.log(lambdas[g.name]))
                        for _, g in reml_groups
                        if lambdas[g.name] > 0 and lambdas_new[g.name] > 0
                    ]
            max_change = max(changes) if changes else 0.0

            if verbose:
                lam_str = ", ".join(f"{g.name}={lambdas_new[g.name]:.4g}" for _, g in reml_groups)
                mode = "cheap" if _cheap_iter else f"pirls={_last_pirls_iters}"
                print(
                    f"  REML iter={n_reml_iter}  max_change={max_change:.6f}  "
                    f"({mode})  lambdas=[{lam_str}]"
                )

            lambda_history.append(lambdas_new.copy())

            if max_change < reml_tol:
                converged = True
                lambdas = lambdas_new
                break

            # 6. Decide: full rebuild or cheap iteration next time.
            # For the direct solver (lambda1=0), the spline basis can stay fixed:
            # lambda_j only rescales the quadratic penalty, and rebuilding the
            # SSP reparametrisation changes coordinates without changing the
            # fitted objective. Keeping the basis fixed avoids an O(nK) pass
            # through every smooth at each outer REML step.
            #
            # For the BCD path (lambda1>0), we still rebuild because the sparse
            # penalty acts on the transformed coefficients and is not invariant
            # to a basis change in the same way.
            if _use_direct:
                warm_beta = beta
                warm_intercept = intercept
                _cheap_iter = max_change <= _direct_cheap_threshold
            elif max_change > _bcd_cheap_threshold:
                old_gms = self._dm.group_matrices
                self._dm = self._rebuild_design_matrix_with_lambdas(lambdas_new, exposure)
                warm_beta = _map_beta_between_bases(
                    beta, old_gms, self._dm.group_matrices, self._groups
                )
                warm_intercept = intercept
                _cheap_iter = False
            else:
                _cheap_iter = True

            lambdas = lambdas_new

        # If the BCD path ended on cheap iterations, rebuild DM with final lambdas.
        # The direct solver keeps a fixed basis throughout REML, so there is
        # nothing to rebuild at convergence.
        if _cheap_iter and converged and not _use_direct:
            self._dm = self._rebuild_design_matrix_with_lambdas(lambdas, exposure)

        # Final fit with converged lambdas
        if _use_direct:
            self._result, _ = fit_irls_direct(
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
            self._result = fit_pirls(
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

        # Store REML results
        self._reml_lambdas = lambdas
        self._reml_result = REMLResult(
            lambdas=lambdas,
            pirls_result=self._result,
            n_reml_iter=n_reml_iter,
            converged=converged,
            lambda_history=lambda_history,
        )

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
        """Get all groups belonging to a feature (1 for normal, 2 for select=True splines)."""
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

            # For select=True: only use columns for active subgroups
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

        # For select=True: only use columns for active subgroups
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
