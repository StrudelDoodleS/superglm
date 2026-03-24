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

from . import base, explain_ops, fit_ops, monotone_ops, profile_ops, state_ops

if TYPE_CHECKING:
    from superglm.discretize import DiscretizationResult
    from superglm.inference import InteractionInference, TermInference
    from superglm.metrics import ModelMetrics
    from superglm.model.fit_ops import PathResult
    from superglm.types import GroupSlice


class SuperGLM:
    """Penalised generalised linear model with splines, group penalties, and REML.

    Supports Poisson, Gaussian, Gamma, NB2, Tweedie, and Binomial families with group
    lasso, sparse group lasso, or ridge penalties.  Smoothing parameters can
    be estimated via REML (``fit_reml``) or cross-validation (``cross_validate``).
    """

    def __init__(
        self,
        family: str | Distribution = "poisson",
        link: str | Link | None = None,
        penalty: Penalty | str | None = None,
        selection_penalty: float | None = None,
        spline_penalty: float | None = None,
        penalty_features: str | list[str] | None = None,
        # Feature configuration
        features: dict[str, FeatureSpec] | None = None,
        splines: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        # Interactions
        interactions: list[tuple[str, str]] | None = None,
        # Solver options
        active_set: bool = False,
        direct_solve: str = "auto",
        # Discretization
        discrete: bool = False,
        n_bins: int | dict[str, int] = 256,
    ):
        """
        Parameters
        ----------
        family : str or Distribution
            Response distribution. Strings ``"poisson"``, ``"gaussian"``,
            ``"gamma"``, ``"binomial"`` are accepted for parameter-free families.
            For parameterized families use Distribution objects:
            ``Tweedie(p=1.5)``, ``NegativeBinomial(theta=1.0)``, or
            the ``families`` module (e.g. ``families.tweedie(p=1.5)``).
            For ``"binomial"``, y must be in {0, 1} and ``predict()`` returns
            probabilities.
        link : str or Link, optional
            Link function. Defaults to the family's configured default link.
        penalty : str or Penalty, optional
            Penalty type. One of ``"group_lasso"``, ``"sparse_group_lasso"``,
            ``"group_elastic_net"``, ``"ridge"``, or a Penalty object.
            Defaults to ``GroupLasso``.
        selection_penalty : float, optional
            Regularisation strength for the group penalty (feature selection).
            ``None`` (default) auto-calibrates to 10% of lambda_max at fit
            time. Set to ``0.0`` for unpenalised / REML-only fits.
        spline_penalty : float, optional
            Within-group ridge shrinkage for spline smoothing.
            Defaults to 0.1.
        penalty_features : str or list[str], optional
            Restrict the selection penalty to specific feature or group names.
            ``None`` (default) applies to all penalizable groups.
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
        interactions : list[tuple[str, str]], optional
            Pairs of feature names to interact.  Interaction type is
            auto-detected from the parent feature specs.
        active_set : bool
            Use active-set cycling in the BCD solver.
        direct_solve : {"auto", "gram", "qr"}
            Strategy for the direct IRLS solver (lambda1=0).
            ``"auto"`` uses gram-based Cholesky with residual-checked SVD
            fallback, warning after repeated fallbacks.  ``"gram"`` forces
            the gram path without warnings.  ``"qr"`` uses QR on the
            materialised weighted design matrix — backward-stable but
            O(n·p²) per iteration.  Intended for smaller datasets.
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
            lambda1=selection_penalty,
            lambda2=spline_penalty if spline_penalty is not None else 0.1,
            penalty_features=penalty_features,
            features=features,
            splines=splines,
            n_knots=n_knots,
            degree=degree,
            categorical_base=categorical_base,
            interactions=interactions,
            active_set=active_set,
            direct_solve=direct_solve,
            discrete=discrete,
            n_bins=n_bins,
        )

    def __repr__(self) -> str:
        family = type(self._distribution).__name__ if self._distribution else self.family
        fitted = self._result is not None
        if fitted:
            n_params = int(self._result.effective_df)
            dev = self._result.deviance
            return f"SuperGLM(family={family}, fitted=True, {n_params} params, dev={dev:.1f})"
        n_features = len(self._specs) if self._specs else "?"
        return f"SuperGLM(family={family}, fitted=False, {n_features} features)"

    @property
    def features(self) -> dict:
        """Feature specs dict (column name → feature object)."""
        return self._specs

    # ── Static / class helpers ────────────────────────────────────

    @staticmethod
    def _resolve_penalty(penalty, lambda1, penalty_features=None):
        return base.resolve_penalty(penalty, lambda1, penalty_features)

    def _resolve_knots(self, spline_cols):
        return base.resolve_knots(self, spline_cols)

    @staticmethod
    def _resolve_ci(ci):
        return explain_ops.resolve_ci(ci)

    # ── Core model operations ─────────────────────────────────────

    def _clone_without_features(self, drop, *, lambda1=..., lambda2=...):
        return base.clone_without_features(self, drop, lambda1=lambda1, lambda2=lambda2)

    def _auto_detect_features(self, X, sample_weight=None):
        return base.auto_detect(self, X, sample_weight)

    def _add_interaction(self, feat1, feat2, name=None, **kwargs):
        return base.model_add_interaction(self, feat1, feat2, name=name, **kwargs)

    def _build_design_matrix(self, X, y, sample_weight, offset):
        return base.model_build_design_matrix(self, X, y, sample_weight, offset)

    def _compute_lambda_max(self, y, weights):
        return base.compute_lambda_max(self, y, weights)

    def _rebuild_design_matrix_with_lambdas(self, lambdas, sample_weight):
        return base.rebuild_dm_with_lambdas(self, lambdas, sample_weight)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        record_diagnostics: bool = False,
    ) -> SuperGLM:
        """Fit the model to data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix with columns matching registered features.
        y : array-like
            Response variable.
        sample_weight : array-like, optional
            **Frequency weights** (prior weights), typically policy exposure
            in insurance applications. Defaults to 1 for all observations.

            This is a frequency weight: it represents the amount of risk
            observed, not observation precision. A policy with sample_weight=0.5
            (6 months on risk) contributes half as much information as one with
            sample_weight=1.0 (12 months). The standard assumption is that the
            expected response scales linearly with sample_weight:
            ``E[Y_i] = sample_weight_i * lambda_i``.

            For Poisson and Gamma models this only affects dispersion and
            standard errors. For Negative Binomial and Tweedie, sample_weight
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
            sample_weight, use ``offset=np.log(sample_weight)`` so that the model
            estimates a rate rather than a raw count.
        record_diagnostics : bool
            If True, record per-iteration IRLS diagnostics (W range,
            mu/eta range, step halvings, worst-observation indices) on
            ``result.iteration_log``.  Useful for debugging convergence.

        Returns
        -------
        SuperGLM
            The fitted model (self).
        """
        return fit_ops.fit(
            self,
            X,
            y,
            sample_weight,
            offset,
            record_diagnostics=record_diagnostics,
        )

    def fit_path(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
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
            sample_weight,
            offset,
            n_lambda=n_lambda,
            lambda_ratio=lambda_ratio,
            lambda_seq=lambda_seq,
        )

    def fit_reml(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        max_reml_iter: int = 20,
        reml_tol: float = 1e-4,
        lambda2_init: float | None = None,
        verbose: bool = False,
    ) -> SuperGLM:
        """Fit with REML estimation of per-term smoothing parameters.

        When ``selection_penalty=0``, the exact/direct path optimizes a Laplace
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
        sample_weight : array-like, optional
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
        return fit_ops.fit_reml(
            self,
            X,
            y,
            sample_weight,
            offset,
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
    def _fit_inference_info(self):
        return state_ops.fit_inference_info(self)

    @cached_property
    def _group_edf(self):
        return state_ops.group_edf(self)

    # ── Diagnostics & summary ─────────────────────────────────────

    def iteration_diagnostics(self):
        """Return per-iteration IRLS diagnostics as a DataFrame.

        Only available if ``fit(record_diagnostics=True)`` was used.
        Shows W range, mu/eta range, deviance, step halvings, and the
        observation indices with the largest/smallest working weights
        at each iteration.
        """
        import pandas as pd

        log = self.result.iteration_log
        if log is None:
            raise RuntimeError(
                "No iteration diagnostics recorded. Refit with fit(record_diagnostics=True)."
            )
        rows = []
        for d in log:
            rows.append(
                {
                    "iter": d.iteration,
                    "deviance": d.deviance,
                    "W_min": d.w_min,
                    "W_max": d.w_max,
                    "W_ratio": d.w_ratio,
                    "mu_min": d.mu_min,
                    "mu_max": d.mu_max,
                    "eta_min": d.eta_min,
                    "eta_max": d.eta_max,
                    "intercept": d.intercept,
                    "step_halvings": d.step_halvings,
                    "top_W_obs": list(d.top_w_indices),
                    "bottom_W_obs": list(d.bottom_w_indices),
                    "cond_estimate": d.cond_estimate,
                    "used_svd_fallback": d.used_svd_fallback,
                }
            )
        return pd.DataFrame(rows)

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
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model."""
        return explain_ops.metrics(self, X, y, sample_weight, offset)

    def drop1(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        test: str = "Chisq",
    ) -> pd.DataFrame:
        """Drop-one deviance analysis for each feature."""
        return explain_ops.drop1(self, X, y, sample_weight, offset, test=test)

    def refit_unpenalised(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        keep_smoothing: bool = True,
    ) -> SuperGLM:
        """Refit with only active features and no selection penalty."""
        return explain_ops.refit_unpenalised(
            self,
            X,
            y,
            sample_weight,
            offset,
            keep_smoothing=keep_smoothing,
        )

    def relativities(
        self, with_se: bool = False, centering: str = "mean"
    ) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features.

        Parameters
        ----------
        centering : {"native", "mean"}
            ``"native"`` preserves internal centering (SSP for splines,
            base-level for categoricals). ``"mean"`` shifts so the geometric
            mean of relativities = 1 across levels/grid — recommended for
            underwriter-facing output where cross-feature comparability matters.
        """
        return explain_ops.relativities(self, with_se, centering=centering)

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
        centering: str = "mean",
    ) -> TermInference | InteractionInference:
        """Per-term inference: curve, uncertainty, and metadata in one object.

        Parameters
        ----------
        centering : {"native", "mean"}
            ``"native"`` preserves internal centering. ``"mean"`` shifts so
            geometric mean of relativities = 1. Recommended for cross-feature
            comparison.
        """
        return explain_ops.term_inference(
            self,
            name,
            with_se=with_se,
            simultaneous=simultaneous,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
        )

    # ── Profile estimation ────────────────────────────────────────

    def estimate_p(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        fit_mode: str = "fit",
        phi_method: str = "pearson",
        method: str = "brent",
        **kwargs,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        Parameters
        ----------
        fit_mode : {"fit", "reml", "inherit"}
            Fitting regime for each candidate ``p`` evaluation.
        phi_method : {"pearson", "mle"}
            How to profile out Tweedie dispersion ``phi`` at each candidate ``p``.
            ``"pearson"`` uses the weighted Pearson moment estimate, while
            ``"mle"`` runs a nested 1D likelihood optimization in ``phi``.
        method : {"brent", "grid", "grid_refine", "profile_opt"}
            Search strategy. ``"brent"`` (default) uses bounded scalar
            optimisation. ``"grid"`` does exhaustive grid search.
            ``"grid_refine"`` does a coarse grid + local Brent refinement.
            ``"profile_opt"`` uses a general-purpose optimizer on
            logit-transformed p.
        """
        return profile_ops.estimate_p(
            self,
            X,
            y,
            sample_weight,
            offset,
            fit_mode=fit_mode,
            phi_method=phi_method,
            method=method,
            **kwargs,
        )

    def estimate_theta(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        **kwargs,
    ):
        """Estimate NB theta via profile likelihood, refit, and return result."""
        return profile_ops.estimate_theta(self, X, y, sample_weight, offset, **kwargs)

    # ── Plotting ──────────────────────────────────────────────────

    def plot(
        self,
        terms: str | list[str] | None = None,
        *,
        kind: str = "global",
        ci: str | bool | None = "pointwise",
        X: pd.DataFrame | None = None,
        sample_weight: NDArray | None = None,
        show_density: bool = True,
        show_knots: bool = False,
        show_bases: bool = False,
        scale: str = "response",
        ci_style: str = "band",
        engine: str = "matplotlib",
        n_points: int = 200,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        centering: str = "mean",
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
        kind : {"global", "local"}
            ``"global"`` shows model-wide fitted effects (default).
            ``"local"`` is reserved for per-row explanations (not yet
            implemented).
        ci : {None, False, "pointwise", "simultaneous", "both"}
            Confidence interval style.  ``None`` or ``False`` disables bands.
        X : DataFrame, optional
            Training data for density overlays.
        sample_weight : array-like, optional
            Frequency weights / sample_weight for density overlays.
        show_density : bool
            Show sample_weight/observation density (strip for continuous,
            bars for categorical).  Default True.
        show_knots : bool
            Show interior knot ticks (spline terms only).
        show_bases : bool
            Initial visibility for coefficient-weighted spline basis
            contributions in the Plotly explorer.  Only meaningful when
            ``scale="link"``; ignored in response-scale mode and by
            the matplotlib renderer.
        scale : {"response", "link"}
            ``"response"`` (default) shows the fitted effect on the
            inverse-link scale (relativities).  ``"link"`` shows the
            additive link-scale contribution η(x) = Σ β_j B_j(x),
            with optional basis decomposition overlays.  Only used
            by the Plotly renderer.
        ci_style : {"band", "lines"}
            Plotly CI presentation. ``"band"`` (default) draws filled
            confidence bands. ``"lines"`` draws line-only CI bounds with
            no fill.
        engine : {"matplotlib", "plotly"}
            Plotting backend. ``"matplotlib"`` is the chart/export path for
            single terms and grids. ``"plotly"`` is the interactive
            main-effect explorer path, with a response/link scale toggle and
            term selector. For main effects, Plotly requires at least two
            terms (or ``terms=None``); use ``engine="matplotlib"`` for a
            single-term chart. Requires the ``plotly`` optional dependency
            (``pip install superglm[plotting]``).
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
        matplotlib.figure.Figure or plotly.graph_objects.Figure

        Examples
        --------
        >>> fig = model.plot(engine="plotly", X=X_train, sample_weight=w)
        >>> fig.show()                      # interactive main-effect explorer
        >>> fig.write_html("effects.html") # standalone HTML export
        """
        return explain_ops.plot(
            self,
            terms,
            kind=kind,
            ci=ci,
            X=X,
            sample_weight=sample_weight,
            show_density=show_density,
            show_knots=show_knots,
            show_bases=show_bases,
            scale=scale,
            ci_style=ci_style,
            engine=engine,
            n_points=n_points,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
            **kwargs,
        )

    def plot_data(
        self,
        terms: str | list[str] | None = None,
        *,
        kind: str = "global",
        ci: str | bool | None = "pointwise",
        X: pd.DataFrame | None = None,
        sample_weight: NDArray | None = None,
        show_density: bool = True,
        show_knots: bool = False,
        show_bases: bool = False,
        n_points: int = 200,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        centering: str = "mean",
    ) -> dict[str, Any]:
        """Return plain data needed to recreate SuperGLM plots.

        This is the data/export companion to :meth:`plot`. It returns plain
        pandas DataFrames, NumPy arrays, and metadata dictionaries instead of a
        figure object, so users can rebuild charts in matplotlib, plotly, Excel,
        or another reporting system.

        For main effects, the payload includes per-term fitted effects and, when
        requested, density overlays, spline knot positions, and basis
        contributions. For interactions, it includes the reconstructed effect
        data and, for continuous x continuous surfaces, optional density / HDR
        grid data when ``X`` and ``sample_weight`` are supplied.

        Examples
        --------
        >>> payload = model.plot_data("DrivAge", X=X_train, sample_weight=w, show_knots=True)
        >>> curve_df = payload["terms"][0]["effect"]
        >>> knots_df = payload["terms"][0]["knots"]
        """
        return explain_ops.plot_data(
            self,
            terms,
            kind=kind,
            ci=ci,
            X=X,
            sample_weight=sample_weight,
            show_density=show_density,
            show_knots=show_knots,
            show_bases=show_bases,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
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

    # ── Monotone repair ─────────────────────────────────────────

    def apply_monotone_postfit(
        self,
        X: pd.DataFrame,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_grid: int = 500,
    ) -> SuperGLM:
        """Apply post-fit monotone repair to splines with ``monotone`` set.

        Finds all spline features with ``monotone='increasing'`` or
        ``monotone='decreasing'``, applies weighted isotonic regression
        to the fitted curve, and projects back to spline coefficients.

        Idempotent: calling twice does not re-repair already-repaired features.

        Parameters
        ----------
        X : DataFrame
            Training data (used to compute density-based grid weights).
        sample_weight : array-like, optional
            Frequency weights.
        offset : array-like, optional
            Offset term (unused, reserved for deviance computation).
        n_grid : int
            Grid resolution for isotonic regression (default 500).

        Returns
        -------
        SuperGLM
            The model (self), with monotone repairs stored.
        """
        return monotone_ops.apply_monotone_postfit(self, X, sample_weight, offset, n_grid=n_grid)

    # ── Diagnostics ───────────────────────────────────────────────

    def term_importance(
        self,
        X: pd.DataFrame,
        sample_weight: NDArray | None = None,
    ) -> pd.DataFrame:
        """Weighted variance of each term's contribution to eta.

        Returns a DataFrame with columns: ``term``, ``feature``,
        ``subgroup_type``, ``variance_eta``, ``sd_eta``, ``edf``,
        ``lambda``, ``group_norm``.
        """
        return explain_ops.term_importance(self, X, sample_weight)

    def term_drop_diagnostics(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        mode: str = "refit",
        X_val: pd.DataFrame | None = None,
        y_val: NDArray | None = None,
    ) -> pd.DataFrame:
        """Drop-term diagnostics: AIC/BIC deltas or holdout loss deltas.

        Parameters
        ----------
        mode : {"refit", "holdout"}
            ``"refit"`` calls ``drop1()`` and adds delta IC columns.
            ``"holdout"`` zeros each term on a validation set (no refit).
        X_val, y_val : optional
            Validation data for ``mode="holdout"``.
        """
        return explain_ops.term_drop_diagnostics(
            self,
            X,
            y,
            sample_weight,
            offset,
            mode=mode,
            X_val=X_val,
            y_val=y_val,
        )

    def spline_redundancy(
        self,
        X: pd.DataFrame,
        sample_weight: NDArray | None = None,
    ) -> dict:
        """Spline redundancy diagnostics: knot spacing, basis correlation, effective rank."""
        return explain_ops.spline_redundancy(self, X, sample_weight)

    # ── Discretization ────────────────────────────────────────────

    def discretization_impact(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing spline/polynomial curves."""
        return explain_ops.discretization_impact(self, X, y, sample_weight, **kwargs)

    # ── REML adapter methods (used by reml_optimizer) ─────────────

    def _compute_dW_deta(self, mu, eta, sample_weight):
        return fit_ops.model_compute_dW_deta(self, mu, eta, sample_weight)

    def _reml_w_correction(
        self,
        pirls_result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_caches,
        sample_weight,
        offset_arr,
    ):
        return fit_ops.model_reml_w_correction(
            self,
            pirls_result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_caches,
            sample_weight,
            offset_arr,
        )

    def _reml_laml_objective(
        self, y, result, lambdas, sample_weight, offset_arr, XtWX=None, penalty_caches=None
    ):
        return fit_ops.model_reml_laml_objective(
            self, y, result, lambdas, sample_weight, offset_arr, XtWX, penalty_caches
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
        sample_weight,
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
            sample_weight,
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
        sample_weight,
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
            sample_weight,
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
        sample_weight,
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
            sample_weight,
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
        sample_weight,
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
            sample_weight,
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
