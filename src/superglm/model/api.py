"""SuperGLM: main model class."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import pandas as pd
from numpy.typing import NDArray

from superglm.distributions import Distribution
from superglm.links import Link
from superglm.penalties.base import Penalty
from superglm.solvers.pirls import PIRLSResult
from superglm.types import FeatureSpec

from . import base, explain_ops, fit_ops, profile_ops, state_ops

if TYPE_CHECKING:
    from superglm.discretize import DiscretizationResult
    from superglm.inference import InteractionInference, TermInference
    from superglm.metrics import ModelMetrics
    from superglm.model.fit_ops import PathResult
    from superglm.types import GroupSlice


class SuperGLM:
    """Penalised generalised linear model with splines, group penalties, and REML.

    Supports Poisson, Gamma, NB2, and Tweedie families with group lasso,
    sparse group lasso, or ridge penalties.  Smoothing parameters can be
    estimated via REML (``fit_reml``) or cross-validation (``fit_cv``).
    """

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
        """
        Parameters
        ----------
        family : str or Distribution
            Response distribution. One of ``"poisson"``, ``"gamma"``,
            ``"tweedie"``, ``"negative_binomial"``, or a Distribution object.
        link : str or Link, optional
            Link function. Defaults to the family's canonical link (log).
        penalty : str or Penalty, optional
            Penalty type. One of ``"group_lasso"``, ``"sparse_group_lasso"``,
            ``"group_elastic_net"``, ``"ridge"``, or a Penalty object.
            Defaults to ``GroupLasso(lambda1=lambda1)``.
        lambda1 : float, optional
            Regularisation strength for the group penalty.  ``None`` (default)
            auto-calibrates to 10% of lambda_max at fit time.  Set to ``0.0``
            for unpenalised / REML-only fits.
        lambda2 : float
            Ridge shrinkage applied within each group (only used by
            ``GroupElasticNet``).
        tweedie_p : float, optional
            Power parameter for ``family="tweedie"``.  Must be in (1, 2).
        nb_theta : float, optional
            Overdispersion parameter for ``family="negative_binomial"``.
        features : dict[str, FeatureSpec], optional
            Explicit feature specifications mapping column names to feature
            objects (``Spline``, ``Categorical``, ``Numeric``, ``Polynomial``).
            Mutually exclusive with *splines*.
        splines : list[str], optional
            Column names to treat as splines in auto-detect mode.  All other
            columns are auto-detected as categorical or numeric.
            Mutually exclusive with *features*.
        n_knots : int or list[int]
            Number of interior knots for auto-detect splines.
        degree : int
            B-spline degree for auto-detect splines.
        categorical_base : str
            Base level strategy for auto-detected categoricals.
        standardize_numeric : bool
            Whether to standardize auto-detected numeric features.
        interactions : list[tuple[str, str]], optional
            Pairs of feature names to interact.  Interaction type is
            auto-detected from the parent feature specs.
        anderson_memory : int
            Anderson acceleration memory for the BCD solver (0 = off).
        active_set : bool
            Use active-set cycling in the BCD solver.
        discrete : bool
            Use discretized basis matrices for large-*n* REML (fREML-style).
        n_bins : int or dict[str, int]
            Number of discretization bins per feature when ``discrete=True``.
        """
        base.init_model(
            self,
            family=family,
            link=link,
            penalty=penalty,
            lambda1=lambda1,
            lambda2=lambda2,
            tweedie_p=tweedie_p,
            nb_theta=nb_theta,
            features=features,
            splines=splines,
            n_knots=n_knots,
            degree=degree,
            categorical_base=categorical_base,
            standardize_numeric=standardize_numeric,
            interactions=interactions,
            anderson_memory=anderson_memory,
            active_set=active_set,
            discrete=discrete,
            n_bins=n_bins,
        )

    # ── Static / class helpers ────────────────────────────────────

    @staticmethod
    def _resolve_penalty(penalty, lambda1):
        return base.resolve_penalty(penalty, lambda1)

    def _resolve_knots(self, spline_cols):
        return base.resolve_knots(self, spline_cols)

    @staticmethod
    def _resolve_sample_weight_alias(exposure, sample_weight, *, method_name):
        return base.resolve_sample_weight_alias(exposure, sample_weight, method_name=method_name)

    @staticmethod
    def _resolve_ci(ci):
        return explain_ops.resolve_ci(ci)

    # ── Core model operations ─────────────────────────────────────

    def _clone_without_features(self, drop, *, lambda1=..., lambda2=...):
        return base.clone_without_features(self, drop, lambda1=lambda1, lambda2=lambda2)

    def _auto_detect_features(self, X, exposure=None):
        return base.auto_detect(self, X, exposure)

    def _add_interaction(self, feat1, feat2, name=None, **kwargs):
        return base.model_add_interaction(self, feat1, feat2, name=name, **kwargs)

    def _build_design_matrix(self, X, y, exposure, offset):
        return base.model_build_design_matrix(self, X, y, exposure, offset)

    def _should_discretize(self, spec):
        return base.model_should_discretize(self, spec)

    def _should_discretize_tensor_interaction(self, ispec):
        return base.model_should_discretize_tensor_interaction(self, ispec)

    def _resolve_discrete_n_bins(self, name, spec):
        return base.model_resolve_discrete_n_bins(self, name, spec)

    def _compute_R_inv(self, B, omega, exposure, lambda2_override=None):
        return base.model_compute_R_inv(self, B, omega, exposure, lambda2_override)

    def _compute_projected_R_inv(self, B, projection, penalty_sub, exposure, lambda2_override=None):
        return base.model_compute_projected_R_inv(
            self, B, projection, penalty_sub, exposure, lambda2_override
        )

    def _compute_lambda_max(self, y, weights):
        return base.compute_lambda_max(self, y, weights)

    def _rebuild_design_matrix_with_lambdas(self, lambdas, exposure):
        return base.rebuild_dm_with_lambdas(self, lambdas, exposure)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
    ) -> SuperGLM:
        """Fit the model to data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix with columns matching registered features.
        y : array-like
            Response variable.
        exposure : array-like, optional
            Backward-compatible alias for ``sample_weight``.
        sample_weight : array-like, optional
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
        return fit_ops.fit(self, X, y, exposure, offset, sample_weight=sample_weight)

    def fit_path(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        lambda_seq: NDArray | None = None,
    ) -> PathResult:
        """Fit a regularization path from lambda_max down to lambda_min.

        Warm-starts each lambda from the previous solution.
        """
        return fit_ops.fit_path(
            self,
            X,
            y,
            exposure,
            offset,
            sample_weight=sample_weight,
            n_lambda=n_lambda,
            lambda_ratio=lambda_ratio,
            lambda_seq=lambda_seq,
        )

    def fit_cv(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        n_folds: int = 5,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        lambda_seq: NDArray | None = None,
        rule: str = "1se",
        refit: bool = True,
        random_state: int | None = None,
    ):
        """Select lambda by K-fold cross-validation."""
        return fit_ops.fit_cv(
            self,
            X,
            y,
            exposure,
            offset,
            sample_weight=sample_weight,
            n_folds=n_folds,
            n_lambda=n_lambda,
            lambda_ratio=lambda_ratio,
            lambda_seq=lambda_seq,
            rule=rule,
            refit=refit,
            random_state=random_state,
        )

    def fit_reml(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
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
            Backward-compatible alias for ``sample_weight``.
        sample_weight : array-like, optional
            Frequency/exposure weights.
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
        return fit_ops.fit_reml(
            self,
            X,
            y,
            exposure,
            offset,
            sample_weight=sample_weight,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            lambda2_init=lambda2_init,
            verbose=verbose,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def result(self) -> PIRLSResult:
        """The fitted PIRLS result (coefficients, deviance, convergence info).

        Raises ``RuntimeError`` if the model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Not fitted")
        return self._result

    @cached_property
    def _coef_covariance(self):
        return state_ops.coef_covariance(self)

    @cached_property
    def _fit_active_info(self):
        return state_ops.fit_active_info(self)

    @cached_property
    def _group_edf(self):
        return state_ops.group_edf(self)

    # ── Diagnostics & summary ─────────────────────────────────────

    def diagnostics(self) -> dict[str, Any]:
        """Per-group diagnostic dict for programmatic / audit access."""
        return state_ops.diagnostics(self)

    def summary(self, alpha: float = 0.05):
        """Rich model summary with coefficient table (statsmodels-style)."""
        return state_ops.summary(self, alpha)

    def _feature_groups(self, name: str) -> list[GroupSlice]:
        """Get all groups belonging to a feature."""
        return state_ops.feature_groups(self, name)

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        """Reconstruct a fitted feature's curve or effect on its original scale."""
        return state_ops.reconstruct_feature(self, name)

    def knot_summary(self) -> dict[str, dict[str, Any]]:
        """Return fitted knot metadata for all spline features."""
        return state_ops.knot_summary(self)

    # ── Inference ─────────────────────────────────────────────────

    def metrics(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model."""
        return explain_ops.metrics(self, X, y, exposure, offset, sample_weight=sample_weight)

    def drop1(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        test: str = "Chisq",
    ) -> pd.DataFrame:
        """Drop-one deviance analysis for each feature."""
        return explain_ops.drop1(
            self, X, y, exposure, offset, sample_weight=sample_weight, test=test
        )

    def refit_unpenalised(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        keep_smoothing: bool = True,
    ) -> SuperGLM:
        """Refit with only active features and no selection penalty."""
        return explain_ops.refit_unpenalised(
            self,
            X,
            y,
            exposure,
            offset,
            sample_weight=sample_weight,
            keep_smoothing=keep_smoothing,
        )

    def relativities(self, with_se: bool = False) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features."""
        return explain_ops.relativities(self, with_se)

    def _feature_se_from_cov(self, name, Cov_active, active_groups, n_points=200):
        return explain_ops.model_feature_se_from_cov(
            self, name, Cov_active, active_groups, n_points
        )

    def simultaneous_bands(
        self,
        feature: str,
        *,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        n_points: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simultaneous confidence bands for a spline feature."""
        return explain_ops.simultaneous_bands(
            self, feature, alpha=alpha, n_sim=n_sim, n_points=n_points, seed=seed
        )

    def term_inference(
        self,
        name: str,
        *,
        with_se: bool = True,
        simultaneous: bool = False,
        n_points: int = 200,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
    ) -> TermInference | InteractionInference:
        """Per-term inference: curve, uncertainty, and metadata in one object."""
        return explain_ops.term_inference(
            self,
            name,
            with_se=with_se,
            simultaneous=simultaneous,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
        )

    # ── Profile estimation ────────────────────────────────────────

    def estimate_p(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        fit_mode: str = "fit",
        **kwargs,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result."""
        return profile_ops.estimate_p(
            self, X, y, exposure, offset, sample_weight=sample_weight, fit_mode=fit_mode, **kwargs
        )

    def estimate_theta(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        **kwargs,
    ):
        """Estimate NB theta via profile likelihood, refit, and return result."""
        return profile_ops.estimate_theta(
            self, X, y, exposure, offset, sample_weight=sample_weight, **kwargs
        )

    # ── Plotting ──────────────────────────────────────────────────

    def plot(
        self,
        terms: str | list[str] | None = None,
        *,
        ci: str | bool | None = "pointwise",
        X: pd.DataFrame | None = None,
        sample_weight: NDArray | None = None,
        show_density: bool = True,
        show_knots: bool = False,
        engine: str = "matplotlib",
        n_points: int = 200,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        **kwargs,
    ):
        """Plot model terms.

        Single entry point for all plotting.  Dispatches based on *terms*:

        - ``None`` — all main effects in a grid.
        - ``"age"`` — one main effect.
        - ``["age", "region"]`` — subset of main effects.
        - ``"age:region"`` — one interaction.

        Parameters
        ----------
        terms : str, list of str, or None
            Which term(s) to plot.  ``None`` plots all main effects.
        ci : {None, False, "pointwise", "simultaneous", "both"}
            Confidence interval style.  ``None`` or ``False`` disables bands.
        X : DataFrame, optional
            Training data for density overlays.
        sample_weight : array-like, optional
            Frequency weights / exposure for density overlays.
        show_density : bool
            Show exposure/observation density (strip for continuous,
            bars for categorical).  Default True.
        show_knots : bool
            Show interior knot ticks (spline terms only).
        engine : {"matplotlib", "plotly"}
            Plotting backend.  ``"plotly"`` is currently supported only
            for single interactions.
        n_points : int
            Grid resolution for spline/polynomial curves.
        figsize : tuple, optional
            Figure size override.
        title, subtitle : str, optional
            Figure-level title and subtitle.
        alpha : float
            Significance level for CIs (default 0.05).
        n_sim : int
            Posterior simulations for simultaneous bands.
        seed : int
            Random seed for simultaneous bands.
        **kwargs
            Forwarded to the underlying renderer (e.g. ``ncols`` for
            grid plots, ``colormap`` for interactions).

        Returns
        -------
        matplotlib.figure.Figure or plotly Figure
        """
        return explain_ops.plot(
            self,
            terms,
            ci=ci,
            X=X,
            sample_weight=sample_weight,
            show_density=show_density,
            show_knots=show_knots,
            engine=engine,
            n_points=n_points,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            **kwargs,
        )

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame, offset: NDArray | None = None) -> NDArray:
        """Predict the response mean for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features with the same columns used during fitting.
        offset : NDArray or None
            Optional offset added to the linear predictor before
            applying the inverse link.

        Returns
        -------
        NDArray
            Predicted mean on the response scale (inverse-link of eta).
        """
        return base.predict(self, X, offset)

    # ── Discretization ────────────────────────────────────────────

    def discretization_impact(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        *,
        sample_weight: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing spline/polynomial curves."""
        return explain_ops.discretization_impact(
            self, X, y, exposure, sample_weight=sample_weight, **kwargs
        )

    # ── REML adapter methods (used by reml_optimizer) ─────────────

    def _compute_dW_deta(self, mu, eta, exposure):
        return fit_ops.model_compute_dW_deta(self, mu, eta, exposure)

    def _reml_w_correction(
        self, pirls_result, XtWX_S_inv, lambdas, reml_groups, penalty_caches, exposure, offset_arr
    ):
        return fit_ops.model_reml_w_correction(
            self,
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            exposure,
            offset_arr,
        )

    def _reml_laml_objective(
        self, y, result, lambdas, exposure, offset_arr, XtWX=None, penalty_caches=None
    ):
        return fit_ops.model_reml_laml_objective(
            self, y, result, lambdas, exposure, offset_arr, XtWX, penalty_caches
        )

    def _reml_direct_gradient(
        self, result, XtWX_S_inv, lambdas, reml_groups, penalty_ranks, phi_hat=1.0
    ):
        return fit_ops.model_reml_direct_gradient(
            self, result, XtWX_S_inv, lambdas, reml_groups, penalty_ranks, phi_hat
        )

    def _reml_direct_hessian(
        self,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        gradient,
        penalty_ranks,
        penalty_caches=None,
        pirls_result=None,
        n_obs=0,
        phi_hat=1.0,
        dH_extra=None,
    ):
        return fit_ops.model_reml_direct_hessian(
            self,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            gradient,
            penalty_ranks,
            penalty_caches,
            pirls_result,
            n_obs,
            phi_hat,
            dH_extra,
        )

    def _optimize_direct_reml(
        self,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        *,
        max_reml_iter,
        reml_tol,
        verbose,
        penalty_caches=None,
        profile=None,
    ):
        return fit_ops.model_optimize_direct_reml(
            self,
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
        )

    def _optimize_discrete_reml_cached_w(
        self,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        *,
        max_reml_iter,
        reml_tol,
        verbose,
        penalty_caches=None,
        profile=None,
    ):
        return fit_ops.model_optimize_discrete_reml_cached_w(
            self,
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
        )

    def _optimize_efs_reml(
        self,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        *,
        max_reml_iter,
        reml_tol,
        verbose,
        penalty_caches=None,
    ):
        return fit_ops.model_optimize_efs_reml(
            self,
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

    def _run_reml_once(
        self,
        y,
        exposure,
        offset_arr,
        reml_groups,
        penalty_ranks,
        lambdas,
        *,
        max_reml_iter,
        reml_tol,
        verbose,
        use_direct,
        penalty_caches=None,
    ):
        return fit_ops.model_run_reml_once(
            self,
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
        )
