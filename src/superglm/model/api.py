"""SuperGLM: main model class."""

from __future__ import annotations

import copy
import re
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from numpy.typing import NDArray

from superglm._blas_threads import solver_blas_threads
from superglm._frame import FrameLike, as_eager_frame
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
from .fit_state import (
    configured_family,
    configured_lambda2,
    configured_link,
    configured_penalty,
)

_SPLINES_DEPRECATION_MESSAGE = (
    "`splines=` auto-detection is deprecated and will be removed in a future "
    "release; use explicit features such as `features={'age': Spline(...)}`."
)

if TYPE_CHECKING:
    from superglm.diagnostics.discretize import DiscretizationResult
    from superglm.inference.factor_smooths import FactorSmoothResult
    from superglm.inference.metrics import ModelMetrics
    from superglm.inference.random_effects import RandomEffectResult
    from superglm.inference.term import InteractionInference, TermInference
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
        selection_penalty: float | Literal["auto"] | None = None,
        spline_penalty: float | None = None,
        penalty_features: str | list[str] | None = None,
        # Feature configuration
        features: dict[str, FeatureSpec] | None = None,
        splines: list[str] | None = None,
        n_knots: int | list[int] = 10,
        degree: int = 3,
        categorical_base: str = "most_exposed",
        # Interactions
        interactions: list[tuple[str, str] | object] | None = None,
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
        retain_fit_state: bool = True,
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
        selection_penalty : float, {"auto"}, or None, optional
            Regularisation strength for the group penalty (feature selection).
            ``None`` (default) and ``0.0`` disable selection. ``"auto"``
            explicitly requests calibration to 10% of lambda_max at fit time.
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
            Deprecated auto-detection shorthand. Column names in this list are
            treated as splines and all other columns are inferred as categorical
            or numeric. Use explicit ``features={"age": Spline(...)}`` for new
            code. Mutually exclusive with *features*.
        n_knots : int or list[int]
            Number of interior knots for auto-detect splines.
        degree : int
            B-spline degree for auto-detect splines.
        categorical_base : str
            Base level strategy for auto-detected categoricals.
        interactions : list[tuple[str, str] or interaction spec], optional
            Pairs of feature names to interact, or explicit interaction
            specifications such as ``FactorSmooth``. Tuple interaction types
            are auto-detected from their parent feature specs.
        active_set : bool
            Use active-set cycling in the BCD solver.
        direct_solve : {"auto", "gram", "qr", "structured"}
            Strategy for the direct IRLS solver (lambda1=0).
            ``"auto"`` selects compact structured elimination for eligible
            random-effect and factor-smooth terms above the measured crossover,
            otherwise using Gram. A globally unidentifiable SZ system also
            retries on Gram with an explicit recorded reason. ``"gram"`` forces
            the gram path. ``"qr"`` uses QR on the
            materialised weighted design matrix — backward-stable but
            O(n·p²) per iteration.  Intended for smaller datasets.
            ``"structured"`` forces structured elimination for an eligible
            random-effect, FS, or SZ block.
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
        retain_fit_state : bool
            If True (default), keep training-scale fit state such as the fitted
            design matrix for later diagnostics. If False, eagerly computes
            compact inference state after fitting, then releases row-scale
            training caches while preserving prediction, summaries, and term
            confidence intervals.
        """
        if splines is not None:
            import warnings

            warnings.warn(
                _SPLINES_DEPRECATION_MESSAGE,
                FutureWarning,
                stacklevel=2,
            )

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
            retain_fit_state=retain_fit_state,
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

    def clone_unfitted(self) -> SuperGLM:
        """Return an independent unfitted model with the same constructor intent.

        The inherited implementation owns the :class:`SuperGLM` constructor
        contract. Subclasses that add constructor parameters must override this
        method so their additional configuration cannot be silently reset.
        """
        if type(self) is SuperGLM:
            return self._config.materialize(SuperGLM)

        import inspect
        import warnings

        base_parameters = set(inspect.signature(SuperGLM.__init__).parameters) - {"self"}
        subclass_parameters = inspect.signature(type(self).__init__).parameters
        additional_parameters = {
            name
            for name, parameter in subclass_parameters.items()
            if name != "self"
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and name not in base_parameters
        }
        if additional_parameters:
            raise TypeError(
                f"{type(self).__name__} adds constructor configuration "
                f"{sorted(additional_parameters)!r}; override clone_unfitted() "
                "to preserve it explicitly"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="convergence='coefficients' is experimental",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=re.escape(_SPLINES_DEPRECATION_MESSAGE) + r"\Z",
                category=FutureWarning,
            )
            try:
                cloned = type(self)(**self._config.constructor_kwargs())
            except TypeError as exc:
                raise TypeError(
                    f"{type(self).__name__} cannot be reconstructed from the "
                    "SuperGLM constructor contract; override clone_unfitted()"
                ) from exc

        cloned._interaction_specs = {
            name: copy.deepcopy(spec) for name, spec in self._config.interaction_templates
        }
        cloned._interaction_order = list(self._config.interaction_order)
        cloned._pending_interactions = tuple(self._config.interactions)
        cloned._config = type(self._config).capture(cloned)
        return cloned

    @property
    def features(self) -> dict:
        """Defensive fitted/configured feature-spec view."""
        return copy.deepcopy(self._specs)

    @property
    def family(self):
        """Defensive copy of configured family intent."""
        return copy.deepcopy(configured_family(self))

    @family.setter
    def family(self, value) -> None:
        owned = copy.deepcopy(value)
        self._family_config = owned
        if hasattr(self, "_config"):
            self._config = self._config.with_value(family=owned)
            self._config_revision += 1

    @property
    def link(self):
        """Defensive copy of configured link intent."""
        return copy.deepcopy(configured_link(self))

    @link.setter
    def link(self, value) -> None:
        owned = copy.deepcopy(value)
        self._link_config = owned
        if hasattr(self, "_config"):
            self._config = self._config.with_value(link=owned)
            self._config_revision += 1

    @property
    def penalty(self):
        """Defensive copy of the configured penalty template."""
        return copy.deepcopy(configured_penalty(self))

    @penalty.setter
    def penalty(self, value) -> None:
        owned = copy.deepcopy(value)
        self._penalty_config = owned
        if hasattr(self, "_config"):
            self._config = self._config.with_value(penalty=owned)
            self._config_revision += 1

    @property
    def selection_penalty(self) -> float | Literal["auto"] | None:
        """Configured selection-penalty intent."""
        return configured_penalty(self).lambda1

    @selection_penalty.setter
    def selection_penalty(self, value: float | Literal["auto"] | None) -> None:
        penalty = copy.deepcopy(configured_penalty(self))
        penalty.lambda1 = base.normalize_selection_penalty(value)
        self.penalty = penalty

    @property
    def selection_penalty_(self) -> float:
        """Resolved selection penalty from the latest successful fit."""
        if self._fit_state is None:
            raise RuntimeError("Model is not fitted")
        return self._fit_state.selection_penalty

    @property
    def lambda2(self):
        """Defensive copy of configured smoothing penalties."""
        return copy.deepcopy(configured_lambda2(self))

    @lambda2.setter
    def lambda2(self, value) -> None:
        owned = copy.deepcopy(value)
        self._lambda2_config = owned
        if hasattr(self, "_config"):
            self._config = self._config.with_value(lambda2=owned)
            self._config_revision += 1

    @property
    def distribution_(self) -> Distribution:
        """Resolved distribution from the latest successful fit."""
        if self._fit_state is None:
            raise RuntimeError("Model is not fitted")
        return copy.deepcopy(self._fit_state.distribution)

    @property
    def theta_(self) -> float:
        """Resolved NB2 dispersion parameter from the latest successful fit."""
        distribution = self.distribution_
        theta = getattr(distribution, "theta", None)
        if theta is None or theta == "auto":
            raise AttributeError("theta_ is only available after fitting a negative-binomial model")
        return float(theta)

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
        return base.auto_detect(self, as_eager_frame(X), sample_weight)

    def _add_interaction(self, feat1, feat2, name=None, **kwargs):
        return base.model_add_interaction(self, feat1, feat2, name=name, **kwargs)

    def _build_design_matrix(self, X, y, sample_weight, offset):
        return base.model_build_design_matrix(
            self,
            as_eager_frame(X),
            y,
            sample_weight,
            offset,
        )

    def _compute_lambda_max(self, y, weights):
        return base.compute_lambda_max(self, y, weights)

    def _rebuild_design_matrix_with_lambdas(self, lambdas, sample_weight):
        return base.rebuild_dm_with_lambdas(self, lambdas, sample_weight)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(
        self,
        X: FrameLike,
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
        X : pandas or eager Polars DataFrame
            Feature matrix with columns matching registered features. Lazy frames
            must be collected before fitting.
        y : array-like
            Response variable.
        sample_weight : array-like, optional
            Observation weights. Their likelihood interpretation is family-specific.
            Defaults to 1 for all observations.

            For Tweedie, these are finite, strictly positive EDM prior weights:
            ``Y_i ~ Tw_p(mu_i, phi / w_i)``, so
            ``Var(Y_i | x_i) = phi * mu_i**p / w_i``. They are not replication counts;
            zero or non-finite weights are rejected. Non-Tweedie families retain their
            existing weighting behavior.

            Weights affect fitting but do not enter the linear predictor or
            automatically scale the conditional mean. The model mean is
            ``mu_i = g**-1(x_i.T @ beta + offset_i)``.
        offset : array-like, optional
            Offset added to the linear predictor. ``sample_weight`` does not supply an
            offset. To make a raw-count mean scale with exposure, pass
            ``offset=np.log(exposure)``. For a per-exposure response, pass exposure as
            ``sample_weight`` without adding it again to the linear predictor.
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

        with solver_blas_threads():
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
        X: FrameLike,
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
        with solver_blas_threads():
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
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        max_reml_iter: int = 20,
        reml_tol: float = 1e-6,
        pirls_tol: float | None = None,
        max_pirls_iter: int | None = None,
        lambda2_init: float | None = None,
        interaction_mode: str = "full",
        runtime_validation: str | bool = "auto",
        verbose: bool = False,
        w_correction_order: int = 1,
    ) -> SuperGLM:
        """Fit with REML estimation of per-term smoothing parameters.

        ``fit_reml()`` is the smoothness-selection path and does not support a
        selection penalty: configure ``selection_penalty=None`` or ``0.0``.
        It optimizes a Laplace approximate REML objective over per-term
        smoothing parameters. For sparse/group selection, use ``fit()`` or
        ``fit_path()``. To let REML shrink spline null spaces, use
        ``select=True`` on the spline terms instead.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Feature matrix. Lazy frames must be collected before fitting.
        y : array-like
            Response variable.
        sample_weight : array-like, optional
            Observation weights. Their likelihood interpretation is family-specific.
            For Tweedie, these are finite, strictly positive EDM prior weights with
            ``Var(Y_i | x_i) = phi * mu_i**p / w_i``; they are not replication counts.
            Weights affect fitting but do not enter the linear predictor or
            automatically scale the conditional mean. Non-Tweedie families retain their
            existing weighting behavior.
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
        interaction_mode : {"full", "fast_candidate"}
            ``"full"`` runs ordinary REML. ``"fast_candidate"`` caps REML
            outer updates for interaction models and then runs the normal final
            refit, intended for screening candidate interactions before a full
            final model fit.
        runtime_validation : {"auto", "full", "skip"} or bool
            Controls the post-fit public-runtime parity diagnostic.
            ``"auto"`` validates small fits and skips the full training-row
            diagnostic for large fits or fast candidate interaction fits.
            ``"full"`` always validates; ``"skip"`` skips validation while
            still canonicalizing the public prediction state.
        verbose : bool
            Print progress.
        w_correction_order : int
            Order of the W(rho) implicit-differentiation correction.
            1 gives the exact objective and gradient with a modified-Newton
            outer Hessian (default, fast). 2 also includes the available exact
            d²W/dη² Hessian cross-terms from Wood (2011, Appendix C). Only
            affects the exact REML path.

        Returns
        -------
        SuperGLM
            The fitted model (self).
        """
        # Resolve PIRLS controls: explicit kwargs > constructor fallback
        resolved_pirls_tol = pirls_tol if pirls_tol is not None else self._tol
        resolved_max_pirls_iter = max_pirls_iter if max_pirls_iter is not None else self._max_iter

        with solver_blas_threads():
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
                interaction_mode=interaction_mode,
                runtime_validation=runtime_validation,
                verbose=verbose,
                w_correction_order=w_correction_order,
            )

    def screen_interactions(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        *,
        offset: NDArray | None = None,
        candidates: list[tuple[str, str]] | None = None,
        edf0: float | tuple[float, ...] = (2.0, 4.0, 8.0, 16.0),
        max_cells: int = 5_000_000,
        screen_bins: int = 256,
        phi: float | None = None,
    ) -> pd.DataFrame:
        """Rank candidate spline-pair interactions by PSST.

        PSST (Penalized Smooth Score Test) asks, for each candidate pair of
        fitted spline features, how much of the model's leftover working
        signal the pair's actual ``ti()`` tensor smooth could absorb at a
        fixed screening complexity, after profiling out what the pair's own
        main effects already explain.  ``edf0`` is a probe bandwidth; the
        default ladder evaluates each pair at several budgets and ranks by
        the best noise-normalized score ``z`` (the statistic is scaled by
        the fit's Pearson dispersion estimate first, so the noise floor
        stays honest beyond unit-dispersion families), so smooth and
        high-frequency interactions are both visible.  One O(n) cell pass
        per pair; no refits.  ``offset`` and ``sample_weight`` both default
        to the values the model was fitted with (weights only when the
        fit's were non-unit); inheriting requires ``X``/``y`` to be the
        training data — pass both arrays explicitly to screen a holdout,
        subsample, or reordered frame.  The dispersion used is attached as
        ``table.attrs["phi"]`` and can be overridden via ``phi=``.

        The returned frame is sorted by ``z`` descending; rank by ``z`` —
        ``statistic``/``edf0``/``lambda0`` describe each pair's winning
        rung, so ``statistic`` is not comparable across rows.  Pairs whose
        unique-value grid exceeds ``max_cells`` are quantile-binned to
        ``screen_bins`` support points per margin and flagged
        ``approx=True``; pairs within budget are always computed exactly.
        Screening always probes the exact-basis tensor; a pair whose
        confirmatory ``ti()`` refit would discretize LOSSILY (both parents
        resolve to fit-time discretization and at least one parent's
        cardinality exceeds its bin count) is flagged ``approx=True`` to
        make that support-discretization gap — measured at ~3.5% on signal
        pairs, the same class as the quantile fallback — visible in the
        output.  Lossless binning returns the exact support and stays
        ``approx=False``.
        Pairs already fitted as tensor terms are excluded from the sweep.  The
        statistic is a ranking device, not a calibrated p-value: confirm
        the top-ranked pairs by refitting them as ``ti()`` terms.
        """
        from superglm.model.screening_ops import screen_interactions

        return screen_interactions(
            self,
            X,
            y,
            sample_weight,
            offset=offset,
            candidates=candidates,
            edf0=edf0,
            max_cells=max_cells,
            screen_bins=screen_bins,
            phi=phi,
        )

    # ── Properties ────────────────────────────────────────────────

    @property
    def result(self) -> PIRLSResult:
        """The fitted public PIRLS result (canonical coefficients and fit stats).

        Raises ``RuntimeError`` if the model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Not fitted")
        return self._result

    def _solver_pirls_result(self) -> PIRLSResult:
        """Return the private solver-space PIRLS result for internal helpers."""
        if self._solver_result is None:
            raise RuntimeError("Not fitted")
        return self._solver_result

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

    def random_effects(
        self,
        name: str,
        *,
        exposure: NDArray | None = None,
        X: FrameLike | None = None,
        y: NDArray | None = None,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> RandomEffectResult:
        """Return variance-component and per-level credibility diagnostics."""
        from superglm.inference.random_effects import random_effect_result

        return random_effect_result(
            self,
            name,
            exposure=exposure,
            X=X,
            y=y,
            sample_weight=sample_weight,
            offset=offset,
        )

    def factor_smooth(
        self,
        name: str,
        *,
        grid: int | NDArray | None = 100,
        levels: list[object] | tuple[object, ...] | None = None,
        confidence_level: float = 0.95,
    ) -> FactorSmoothResult:
        """Return basis-aware penalties, level diagnostics, and smooth curves."""
        from superglm.inference.factor_smooths import factor_smooth_result

        return factor_smooth_result(
            self,
            name,
            grid=grid,
            levels=levels,
            confidence_level=confidence_level,
        )

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
                    "raw_W_min": d.raw_w_min,
                    "raw_W_max": d.raw_w_max,
                    "raw_W_ratio": d.raw_w_ratio,
                    "mu_min": d.mu_min,
                    "mu_max": d.mu_max,
                    "eta_min": d.eta_min,
                    "eta_max": d.eta_max,
                    "eta_min_unclipped": d.eta_min_unclipped,
                    "eta_max_unclipped": d.eta_max_unclipped,
                    "eta_clipped": d.eta_clipped,
                    "working_mu_min": d.working_mu_min,
                    "working_mu_max": d.working_mu_max,
                    "working_eta_min": d.working_eta_min,
                    "working_eta_max": d.working_eta_max,
                    "working_eta_min_unclipped": d.working_eta_min_unclipped,
                    "working_eta_max_unclipped": d.working_eta_max_unclipped,
                    "working_eta_clipped": d.working_eta_clipped,
                    "intercept": d.intercept,
                    "step_halvings": d.step_halvings,
                    "step_rejected": d.step_rejected,
                    "rank_truncated": d.rank_truncated,
                    "top_W_obs": list(d.top_w_indices),
                    "bottom_W_obs": list(d.bottom_w_indices),
                    "cond_estimate": d.cond_estimate,
                    "used_svd_fallback": d.used_svd_fallback,
                }
            )
        return pd.DataFrame(rows)

    def training_telemetry(self) -> dict[str, Any]:
        """Return dependency-free training telemetry for external tracking.

        The payload contains plain JSON-serializable Python objects. SuperGLM
        does not import or own any experiment-tracking backend; callers can send
        this payload to MLflow, files, logs, or governance systems.
        """
        from superglm.model import telemetry_ops

        return telemetry_ops.training_telemetry(self)

    def reml_diagnostics(self) -> dict[str, Any]:
        """Return dependency-free REML telemetry for external tracking."""
        from superglm.model import telemetry_ops

        return telemetry_ops.reml_diagnostics(self)

    def diagnostics(self) -> dict[str, Any]:
        """Per-group diagnostic dict for programmatic / audit access."""
        return report_ops.diagnostics(self)

    def design_summary(self) -> pd.DataFrame:
        """Describe fitted design storage and static route eligibility.

        The summary does not build an accelerated matrix or prove that an
        eligible kernel executed. Fit and REML traces remain authoritative for
        actual dispatch.
        """
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling design_summary().")
        if self._dm is None:
            raise RuntimeError(
                "retain_fit_state=False discarded the fitted design; refit with "
                "retain_fit_state=True before calling design_summary()."
            )
        from superglm.model.design_summary import build_design_summary

        return build_design_summary(self)

    def summary(
        self,
        alpha: float = 0.05,
        detail: str = "compact",
        level_display: str = "expanded",
    ):
        """Rich model summary with coefficient table (statsmodels-style).

        ``level_display="expanded"`` shows exact original categorical levels.
        Use ``"grouped"`` for one row per fitted group plus a membership legend.
        """
        return report_ops.summary(
            self,
            alpha,
            detail=detail,
            level_display=level_display,
        )

    def _feature_groups(self, name: str) -> list[GroupSlice]:
        """Get all groups belonging to a feature."""
        return report_ops.feature_groups(self, name)

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        """Reconstruct a fitted feature's curve or effect on its original scale."""
        return report_ops.reconstruct_feature(self, name)

    def knot_summary(self) -> dict[str, dict[str, Any]]:
        """Return fitted knot metadata for all spline features."""
        return report_ops.knot_summary(self)

    # ── Inference ─────────────────────────────────────────────────

    def metrics(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model."""
        return explain_ops.metrics(self, X, y, sample_weight, offset)

    def drop1(
        self,
        X: FrameLike,
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
        X: FrameLike,
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
        self, with_se: bool = False, centering: str = "native"
    ) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features.

        Parameters
        ----------
        centering : {"native", "mean"}
            ``"native"`` (default) returns the canonical fitted term
            contribution under the model's identifiability constraint.
            ``"mean"`` is a reporting convenience that shifts so the
            geometric mean of relativities = 1 — useful for cross-feature
            comparison but not the fitted term decomposition.
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
        centering: str = "native",
    ) -> TermInference | InteractionInference:
        """Per-term inference: curve, uncertainty, and metadata in one object.

        Parameters
        ----------
        centering : {"native", "mean"}
            ``"native"`` (default) returns the canonical fitted term
            contribution under the model's identifiability constraint.
            ``"mean"`` is a reporting convenience that shifts so the
            geometric mean of relativities = 1.
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
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        fit_mode: str = "fit",
        phi_method: str = "mle",
        method: str = "auto",
        ci_alpha: float | None = None,
        **kwargs,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Feature matrix. Lazy frames must be collected before fitting.
        y : array-like
            Response variable.
        sample_weight : array-like, optional
            Finite, strictly positive EDM prior weights, not replication or frequency weights.
            The Tweedie variance convention is
            ``Var(Y_i | x_i) = phi * mu_i**p / w_i``.
            Remove zero-weight rows consistently from ``X``, ``y``, ``sample_weight``,
            and ``offset`` before calling this method.
        offset : array-like, optional
            Offset added to the linear predictor.
        fit_mode : {"fit", "reml", "inherit"}
            Fitting regime for each candidate ``p`` evaluation.
        phi_method : {"pearson", "mle"}
            How to profile out Tweedie dispersion ``phi`` at each candidate ``p``.
            ``"mle"`` (default) maximizes the likelihood in ``phi``; the joint
            fast path uses exact derivatives and defensive searches use a nested
            scalar optimization. ``"pearson"`` is an explicit faster plug-in and
            does not support likelihood-ratio confidence intervals.
        method : {"auto", "joint_ml", "brent", "grid", "grid_refine", "profile_opt"}
            Search strategy. ``"auto"`` (default) uses safeguarded exact joint
            ML for ordinary MLE profiles and Brent otherwise. ``"joint_ml"``
            explicitly requests that fast path within its stable ``p`` range and
            falls back defensively otherwise. ``"brent"`` uses bounded scalar
            optimisation. ``"grid"`` does exhaustive grid search.
            ``"grid_refine"`` does a coarse grid + local Brent refinement.
            ``"profile_opt"`` uses a general-purpose optimizer on
            logit-transformed p.
        ci_alpha : float, optional
            Significance level for an explicitly requested likelihood-ratio
            profile confidence interval. For example, ``0.05`` computes a 95%
            interval and caches it for ``model.summary(alpha=0.05)``. The
            default ``None`` performs no confidence-interval evaluations.
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
            ci_alpha=ci_alpha,
            **kwargs,
        )

    def estimate_theta(
        self,
        X: FrameLike,
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
        X: FrameLike | None = None,
        sample_weight: NDArray | None = None,
        show_density: bool = True,
        show_knots: bool = False,
        show_bases: bool = False,
        scale: str = "response",
        ci_style: str = "band",
        categorical_display: str = "auto",
        grouped_level_display: str = "auto",
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
        X : pandas or eager Polars DataFrame, optional
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
        grouped_level_display : {"auto", "expanded", "collapsed"}
            Display option for grouped categorical levels in main-effect plots.
            ``"auto"`` collapses grouped ordered-categorical terms and leaves
            unordered categoricals expanded. This is a plotting-only option;
            scoring, inference tables, and exports remain expanded over the
            original levels.
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
            grouped_level_display=grouped_level_display,
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
        X: FrameLike,
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
        X : pandas or eager Polars DataFrame
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
        X: FrameLike | None = None,
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

    def _predict_eta_exact(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Private exact canonical predictor on the link scale."""
        return base.predict_eta_exact(self, X, offset, random_effects=random_effects)

    def _predict_eta_fast_discrete(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Private fast discrete predictor on the link scale."""
        return base.predict_eta_fast_discrete(
            self,
            X,
            offset,
            random_effects=random_effects,
        )

    def _predict_exact(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Private exact canonical predictor on the response scale."""
        return base.predict_exact(self, X, offset, random_effects=random_effects)

    def _predict_fast_discrete(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Private fast discrete predictor on the response scale."""
        return base.predict_fast_discrete(
            self,
            X,
            offset,
            random_effects=random_effects,
        )

    def predict(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Predict the response mean for new data.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Eager input features with the same columns used during fitting.
        offset : NDArray or None
            Optional offset added to the linear predictor before
            applying the inverse link.
        random_effects : {"conditional", "population"}
            Whether to include fitted random-effect and factor-smooth
            deviations. Population prediction sets all such contributions to
            zero.

        Returns
        -------
        NDArray
            Predicted mean on the response scale (inverse-link of eta).
        """
        return self._predict_exact(X, offset, random_effects=random_effects)

    # ── Monotone repair ─────────────────────────────────────────

    def monotonize(
        self,
        X: FrameLike,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_grid: int = 500,
    ) -> SuperGLM:
        """Repair monotone-annotated spline terms after fitting.

        This is a manual, post-fit repair step. It finds all spline features
        with ``monotone='increasing'`` or ``monotone='decreasing'``, applies
        weighted isotonic regression to the fitted curve, and projects the
        repaired curve back to spline coefficients.

        Idempotent: calling twice does not re-repair already-repaired features.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
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
        return monotone_ops.monotonize(self, X, sample_weight, offset, n_grid=n_grid)

    def apply_shape_postfit(
        self,
        X: FrameLike,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_grid: int = 500,
    ) -> SuperGLM:
        """Repair postfit monotone and curvature-constrained spline terms."""
        from superglm.model import shape_ops

        return shape_ops.apply_shape_postfit(self, X, sample_weight, offset, n_grid=n_grid)

    def apply_monotone_postfit(
        self,
        X: FrameLike,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_grid: int = 500,
    ) -> SuperGLM:
        """Compatibility alias for :meth:`monotonize`."""
        return self.monotonize(X, sample_weight, offset, n_grid=n_grid)

    # ── Diagnostics ───────────────────────────────────────────────

    def term_importance(
        self,
        X: FrameLike,
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
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        mode: str = "refit",
        X_val: FrameLike | None = None,
        y_val: NDArray | None = None,
        sample_weight_val: NDArray | None = None,
        offset_val: NDArray | None = None,
    ) -> pd.DataFrame:
        """Drop-term diagnostics: AIC/BIC deltas or holdout loss deltas.

        Parameters
        ----------
        X, y : training rows and response
            Used for refit mode and as the identity anchor for same-object
            holdout fallback.
        sample_weight, offset : array-like, optional
            Training/refit weights and offset.
        mode : {"refit", "holdout"}
            ``"refit"`` calls ``drop1()`` and adds delta IC columns.
            ``"holdout"`` zeros each term on a validation set (no refit).
        X_val, y_val : optional
            Validation data for ``mode="holdout"``.
        sample_weight_val, offset_val : array-like, optional
            Validation-specific geometry for holdout mode. Training vectors
            are reused only when ``X_val is X`` and ``y_val is y``; separate
            validation objects require ``sample_weight_val`` when
            ``sample_weight`` is supplied and require ``offset_val`` for an
            offset-fitted model.
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
            sample_weight_val=sample_weight_val,
            offset_val=offset_val,
        )

    def spline_redundancy(
        self,
        X: FrameLike,
        sample_weight: NDArray | None = None,
    ) -> dict:
        """Spline redundancy diagnostics: knot spacing, basis correlation, effective rank."""
        return explain_ops.spline_redundancy(self, X, sample_weight)

    # ── Discretization ────────────────────────────────────────────

    def discretization_impact(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing spline/polynomial curves."""
        return explain_ops.discretization_impact(self, X, y, sample_weight, **kwargs)

    def export_rating_tables(
        self,
        file_path,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ):
        """Export deployment rating tables for the fitted model."""
        from superglm.export import export_rating_tables

        return export_rating_tables(self, file_path, X, y, sample_weight=sample_weight, **kwargs)

    def rating_table_payload(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ):
        """Build the renderer-independent deployment rating-table payload."""
        from superglm.export.rating_tables import build_rating_table_payload

        return build_rating_table_payload(self, X, y, sample_weight=sample_weight, **kwargs)

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
        w_correction_order=1,
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
            w_correction_order=w_correction_order,
        )

    def _reml_laml_objective(
        self, y, result, lambdas, sample_weight, offset_arr, XtWX=None, penalty_caches=None
    ):
        return fit_ops.model_reml_laml_objective(
            self, y, result, lambdas, sample_weight, offset_arr, XtWX, penalty_caches
        )

    def _reml_direct_gradient(
        self,
        result,
        XtWX_S_inv,
        lambdas,
        reml_groups,
        penalty_ranks,
        phi_hat=1.0,
        *,
        inverse_phi=None,
    ):
        return fit_ops.model_reml_direct_gradient(
            self,
            result,
            XtWX_S_inv,
            lambdas,
            reml_groups,
            penalty_ranks,
            phi_hat,
            inverse_phi=inverse_phi,
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
        dH2_cross=None,
        *,
        inverse_phi=None,
        d_inverse_phi_d_penalized_deviance=None,
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
            dH2_cross,
            inverse_phi=inverse_phi,
            d_inverse_phi_d_penalized_deviance=(d_inverse_phi_d_penalized_deviance),
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
