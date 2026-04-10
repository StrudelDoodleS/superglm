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

from . import (
    base,
    explain_ops,
    fit_ops,
    monotone_ops,
    plot_ops,
    profile_ops,
    report_ops,
    state_ops,
)

if TYPE_CHECKING:
    from superglm.model.fit_ops import PathResult


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
        # Convergence
        tol: float = 1e-6,
        max_iter: int = 100,
        convergence: str = "deviance",
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
        tol : float
            Convergence tolerance for IRLS / PIRLS.  Default ``1e-6``.
            Can also be set per-call via ``fit(tol=...)`` or
            ``fit_reml(pirls_tol=...)``.  Fit-time values take precedence.
            Larger values (e.g. ``1e-6``) converge faster but may stop
            before near-separated coefficients have stabilised.
        max_iter : int
            Maximum IRLS / PIRLS outer iterations.  Default ``100``.
        convergence : {"deviance", "coefficients"}
            Convergence criterion.  ``"deviance"`` (default) stops when
            relative deviance change drops below *tol* — fast, since
            well-identified coefficients lock in early.
            ``"coefficients"`` (**experimental**) stops when the maximum
            relative coefficient change drops below *tol*.  May not
            converge for near-separated levels where the MLE is at −∞.
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
            tol=tol,
            max_iter=max_iter,
            convergence=convergence,
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
        return plot_ops.resolve_ci(ci)

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
        tol: float | None = None,
        max_iter: int | None = None,
        convergence: str | None = None,
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
        # Resolve fit controls: explicit kwargs > constructor fallback
        resolved_tol = tol if tol is not None else self._tol
        resolved_max_iter = max_iter if max_iter is not None else self._max_iter
        resolved_convergence = convergence if convergence is not None else self._convergence
        if resolved_convergence not in ("deviance", "coefficients"):
            raise ValueError(
                f"convergence must be 'deviance' or 'coefficients', got {resolved_convergence!r}"
            )
        if convergence == "coefficients":
            import warnings

            warnings.warn(
                "convergence='coefficients' is experimental. Near-separated levels "
                "have no finite MLE, so coefficient-based convergence may not "
                "terminate or may produce numerically unstable results. "
                "Use convergence='deviance' (default) for production fits.",
                UserWarning,
                stacklevel=2,
            )

        return fit_ops.fit(
            self,
            X,
            y,
            sample_weight,
            offset,
            tol=resolved_tol,
            max_iter=resolved_max_iter,
            convergence=resolved_convergence,
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
        reml_tol: float = 1e-6,
        pirls_tol: float | None = None,
        max_pirls_iter: int | None = None,
        lambda2_init: float | None = None,
        verbose: bool = False,
        w_correction_order: int = 1,
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
            Convergence tolerance on log-lambda (default 1e-6).
        pirls_tol : float, optional
            Inner PIRLS/IRLS convergence tolerance. Defaults to
            constructor ``tol`` (1e-6). Pass explicitly to override.
        max_pirls_iter : int, optional
            Maximum inner PIRLS iterations per REML step. Defaults to
            constructor ``max_iter`` (100).
        lambda2_init : float, optional
            Initial per-group lambda. Defaults to ``self.lambda2``.
        verbose : bool
            Print progress.
        w_correction_order : int
            Order of the W(rho) implicit-differentiation correction.
            1 = first-order (default, fast). 2 = includes second-order
            d²W/dη² Hessian cross-terms (Wood 2011 Appendix C, computed
            via FD approximation). Only affects the exact REML path.

        Returns
        -------
        SuperGLM
            The fitted model (self).
        """
        # Resolve PIRLS controls: explicit kwargs > constructor fallback
        resolved_pirls_tol = pirls_tol if pirls_tol is not None else self._tol
        resolved_max_pirls_iter = max_pirls_iter if max_pirls_iter is not None else self._max_iter

        return fit_ops.fit_reml(
            self,
            X,
            y,
            sample_weight,
            offset,
            max_reml_iter=max_reml_iter,
            reml_tol=reml_tol,
            pirls_tol=resolved_pirls_tol,
            max_pirls_iter=resolved_max_pirls_iter,
            lambda2_init=lambda2_init,
            verbose=verbose,
            w_correction_order=w_correction_order,
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
        categorical_display: str = "auto",
        engine: str = "matplotlib",
        n_points: int = 200,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
        plotly_style: dict[str, Any] | None = None,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        centering: str = "native",
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
            inverse-link scale (relativities).  With ``centering="native"``,
            this is the exponentiated fitted term contribution under the
            model's identifiability constraint — not a portfolio-average
            relativity. ``"link"`` shows the additive link-scale
            contribution eta(x) = B(x) @ beta, with optional basis
            decomposition overlays.  Only used by the Plotly renderer.
        ci_style : {"band", "lines"}
            Plotly CI presentation. ``"band"`` (default) draws filled
            confidence bands. ``"lines"`` draws line-only CI bounds with
            no fill.
        categorical_display : {"auto", "bars", "markers", "bars+markers"}
            Plotly categorical rendering mode. ``"auto"`` (default) uses
            bars+markers up to 30 levels and markers-only above that.
        engine : {"matplotlib", "plotly"}
            Plotting backend. ``"matplotlib"`` is the chart/export path for
            single terms and grids. ``"plotly"`` is the interactive
            main-effect explorer path, with a response/link scale toggle and
            term selector. For main effects, Plotly requires at least two
            terms (or ``terms=None``); use ``engine="matplotlib"`` for a
            single-term chart. Requires the ``plotly`` optional dependency
            (``pip install superglm[plotting]``).
        centering : {"native", "mean"}
            ``"native"`` (default) returns the canonical fitted term
            contribution under the model's identifiability constraint.
            ``"mean"`` is a reporting convenience that shifts so the
            geometric mean of relativities = 1.
        n_points : int
            Grid resolution for spline/polynomial curves.
        figsize : tuple, optional
            Figure size override.
        title, subtitle : str, optional
            Figure-level title and subtitle.
        plotly_style : dict, optional
            Plotly main-effect explorer style overrides. Supported keys include
            ``line_color``, ``bar_color``, ``density_fill_color``,
            ``density_edge_color``, ``error_bar_color``, ``text_color``, and
            ``text_outline_color``. Ignored by the matplotlib renderer.
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
        return plot_ops.plot(
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
            categorical_display=categorical_display,
            engine=engine,
            n_points=n_points,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
            plotly_style=plotly_style,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
            centering=centering,
            **kwargs,
        )

    def plot_diagnostics(
        self,
        X: pd.DataFrame,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_sim: int = 100,
        figsize: tuple[float, float] | None = None,
        max_points: int = 50_000,
        seed: int = 42,
        residual_type: str = "auto",
    ):
        """GLM/GAM diagnostic figure with simulation-based Q-Q envelope.

        Four panels using quantile residuals (Dunn & Smyth 1996):

        1. Q-Q with simulation envelope
        2. Calibration (exposure-weighted observed vs predicted)
        3. Residuals vs Linear Predictor
        4. Residual distribution (histogram + N(0,1) overlay)

        Parameters
        ----------
        X : pd.DataFrame
            Design matrix.
        y : NDArray
            Response vector.
        sample_weight : NDArray or None
            Optional observation weights (exposure for frequency models).
        offset : NDArray or None
            Optional offset.
        n_sim : int
            Number of simulation replicates for the Q-Q envelope.
        figsize : tuple or None
            Figure size in inches. Defaults to ``(10, 8)``.
        max_points : int
            Threshold for scatter vs hexbin rendering.
        seed : int
            Random seed for quantile residuals, simulation, and
            subsampling.
        residual_type : str
            .. deprecated::
                Ignored. All panels use quantile residuals.

        Returns
        -------
        matplotlib.figure.Figure
            A figure with 4 diagnostic subplots.
        """
        from superglm.plotting.diagnostics import plot_diagnostics

        return plot_diagnostics(
            self,
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            n_sim=n_sim,
            figsize=figsize,
            max_points=max_points,
            seed=seed,
            residual_type=residual_type,
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
        centering: str = "native",
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

        With ``centering="native"`` (default), relativity values are
        the exponentiated fitted term contributions under the model's
        identifiability constraint — not portfolio-average relativities.
        Pass ``centering="mean"`` for a reporting view where the
        geometric mean of relativities = 1.

        Examples
        --------
        >>> payload = model.plot_data("DrivAge", X=X_train, sample_weight=w, show_knots=True)
        >>> curve_df = payload["terms"][0]["effect"]
        >>> knots_df = payload["terms"][0]["knots"]
        """
        return plot_ops.plot_data(
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


SuperGLM._compute_dW_deta = fit_ops.model_compute_dW_deta
SuperGLM._reml_w_correction = fit_ops.model_reml_w_correction
SuperGLM._reml_laml_objective = fit_ops.model_reml_laml_objective
SuperGLM._reml_direct_gradient = fit_ops.model_reml_direct_gradient
SuperGLM._reml_direct_hessian = fit_ops.model_reml_direct_hessian
SuperGLM._optimize_direct_reml = fit_ops.model_optimize_direct_reml
SuperGLM._optimize_discrete_reml_cached_w = fit_ops.model_optimize_discrete_reml_cached_w
SuperGLM._optimize_efs_reml = fit_ops.model_optimize_efs_reml
SuperGLM._run_reml_once = fit_ops.model_run_reml_once

SuperGLM.diagnostics = report_ops.diagnostics
SuperGLM.summary = report_ops.summary
SuperGLM._feature_groups = report_ops.feature_groups
SuperGLM.reconstruct_feature = report_ops.reconstruct_feature
SuperGLM.knot_summary = report_ops.knot_summary

SuperGLM.metrics = explain_ops.metrics
SuperGLM.drop1 = explain_ops.drop1
SuperGLM.refit_unpenalised = explain_ops.refit_unpenalised
SuperGLM.relativities = explain_ops.relativities
SuperGLM._feature_se_from_cov = explain_ops.model_feature_se_from_cov
SuperGLM.simultaneous_bands = explain_ops.simultaneous_bands
SuperGLM.term_inference = explain_ops.term_inference
SuperGLM.term_importance = explain_ops.term_importance
SuperGLM.term_drop_diagnostics = explain_ops.term_drop_diagnostics
SuperGLM.spline_redundancy = explain_ops.spline_redundancy
SuperGLM.discretization_impact = explain_ops.discretization_impact

SuperGLM.estimate_p = profile_ops.estimate_p
SuperGLM.estimate_theta = profile_ops.estimate_theta

SuperGLM.monotonize = monotone_ops.monotonize
SuperGLM.apply_monotone_postfit = monotone_ops.apply_monotone_postfit

# ── REML adapter methods (used by reml_optimizer) ─────────────
