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
from numpy.typing import NDArray

from superglm.distributions import Distribution
from superglm.dm_builder import (
    add_interaction,
    auto_detect_features,
    build_design_matrix,
    compute_projected_R_inv,
    compute_R_inv,
    rebuild_design_matrix_with_lambdas,
    resolve_discrete_n_bins,
    should_discretize,
    should_discretize_tensor_interaction,
)
from superglm.group_matrix import (
    DesignMatrix,
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.links import Link
from superglm.penalties.base import Penalty
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.reml_optimizer import (
    compute_dW_deta,
    optimize_direct_reml,
    optimize_efs_reml,
    reml_direct_gradient,
    reml_direct_hessian,
    reml_laml_objective,
    reml_w_correction,
    run_reml_once,
)
from superglm.solvers.irls_direct import (
    _build_penalty_matrix,
    fit_irls_direct,
)
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FeatureSpec, GroupSlice

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
        spline_cols = self._splines or []
        knots_map = self._resolve_knots(spline_cols)
        auto_detect_features(
            X,
            exposure,
            spline_cols=spline_cols,
            knots_map=knots_map,
            degree=self._degree,
            categorical_base=self._categorical_base,
            standardize_numeric=self._standardize_numeric,
            specs=self._specs,
            feature_order=self._feature_order,
        )

    def _add_interaction(self, feat1: str, feat2: str, name: str | None = None, **kwargs) -> None:
        """Register an interaction between two already-registered features."""
        add_interaction(
            feat1,
            feat2,
            specs=self._specs,
            interaction_specs=self._interaction_specs,
            interaction_order=self._interaction_order,
            name=name,
            **kwargs,
        )

    def _build_design_matrix(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray,
        offset: NDArray | None,
    ) -> tuple[NDArray, NDArray, NDArray | None]:
        """Build features, groups, design matrix.

        Sets self._dm, self._groups, self._distribution, self._link.
        Returns (y, exposure, offset) as float64 arrays.
        """
        result = build_design_matrix(
            X,
            y,
            exposure,
            offset,
            family=self.family,
            link_spec=self.link,
            nb_theta=self.nb_theta,
            tweedie_p=self.tweedie_p,
            specs=self._specs,
            feature_order=self._feature_order,
            interaction_specs=self._interaction_specs,
            interaction_order=self._interaction_order,
            pending_interactions=self._pending_interactions,
            model_discrete=self._discrete,
            n_bins_config=self._n_bins,
            lambda2=self.lambda2,
        )
        self._distribution = result.distribution
        self._link = result.link
        self._groups = result.groups
        self._dm = result.dm
        return result.y, result.exposure, result.offset

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
        return should_discretize(spec, self._discrete)

    def _should_discretize_tensor_interaction(self, ispec: Any) -> bool:
        """Check if a tensor interaction should use fit-time discretization."""
        return should_discretize_tensor_interaction(ispec, self._specs, self._discrete)

    def _resolve_discrete_n_bins(self, name: str, spec: FeatureSpec) -> int:
        """Resolve the requested bin count for a discretized feature."""
        return resolve_discrete_n_bins(name, spec, self._n_bins)

    def _compute_R_inv(self, B, omega, exposure, lambda2_override=None):
        """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv."""
        lam2 = lambda2_override if lambda2_override is not None else self.lambda2
        return compute_R_inv(B, omega, exposure, lam2)

    def _compute_projected_R_inv(self, B, projection, penalty_sub, exposure, lambda2_override=None):
        """Compute SSP R_inv within a projected subspace (linear-split range space)."""
        lam2 = lambda2_override if lambda2_override is not None else self.lambda2
        return compute_projected_R_inv(B, projection, penalty_sub, exposure, lam2)

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
        """Rebuild design matrix with per-group smoothing lambdas."""
        return rebuild_design_matrix_with_lambdas(
            self._dm, self._groups, lambdas, exposure, self.lambda2
        )

    def _compute_dW_deta(self, mu: NDArray, eta: NDArray, exposure: NDArray) -> NDArray | None:
        """Derivative of IRLS weights w.r.t. the linear predictor."""
        return compute_dW_deta(self._link, self._distribution, mu, eta, exposure)

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
        """First-order W(ρ) correction for REML derivatives."""
        return reml_w_correction(
            self._dm,
            self._link,
            self._groups,
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
            self._distribution,
        )

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
        """Laplace REML/LAML objective up to additive constants."""
        return reml_laml_objective(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
            y,
            result,
            lambdas,
            exposure,
            offset_arr,
            XtWX=XtWX,
            penalty_caches=penalty_caches,
        )

    def _reml_direct_gradient(
        self,
        result: PIRLSResult,
        XtWX_S_inv: NDArray,
        lambdas: dict[str, float],
        reml_groups: list[tuple[int, GroupSlice]],
        penalty_ranks: dict[str, float],
        phi_hat: float = 1.0,
    ) -> NDArray:
        """Partial gradient of the LAML objective w.r.t. log-lambdas (fixed W)."""
        return reml_direct_gradient(
            self._dm.group_matrices,
            result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat=phi_hat,
        )

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
        """Outer Hessian of the REML criterion w.r.t. log-lambdas."""
        return reml_direct_hessian(
            self._dm.group_matrices,
            self._distribution,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            gradient,
            penalty_ranks,
            penalty_caches=penalty_caches,
            pirls_result=pirls_result,
            n_obs=n_obs,
            phi_hat=phi_hat,
            dH_extra=dH_extra,
        )

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
        """Optimize the direct REML objective via damped Newton (Wood 2011)."""
        return optimize_direct_reml(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
            self._discrete,
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
            profile=profile,
            max_analytical_per_w=getattr(self, "_max_analytical_per_w", 30),
        )

    def _optimize_discrete_reml_cached_w(
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
        """Cached-W fREML optimizer for the discrete path."""
        from superglm.reml_optimizer import optimize_discrete_reml_cached_w

        return optimize_discrete_reml_cached_w(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
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
            profile=profile,
            max_analytical_per_w=getattr(self, "_max_analytical_per_w", 30),
        )

    def _optimize_efs_reml(
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
    ):
        """EFS REML optimizer for the BCD path (lambda1 > 0)."""
        result, dm = optimize_efs_reml(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
            self.penalty,
            self._anderson_memory,
            self._active_set,
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
            rebuild_dm=self._rebuild_design_matrix_with_lambdas,
        )
        self._dm = dm
        return result

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
        result, dm = run_reml_once(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
            self.penalty,
            self._anderson_memory,
            self._active_set,
            y,
            exposure,
            offset_arr,
            reml_groups,
            penalty_ranks,
            lambdas,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            verbose=verbose,
            use_direct=use_direct,
            penalty_caches=penalty_caches,
            rebuild_dm=self._rebuild_design_matrix_with_lambdas,
        )
        self._dm = dm
        return result

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
            best = self._optimize_efs_reml(
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
