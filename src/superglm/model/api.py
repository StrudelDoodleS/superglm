"""SuperGLM: main model class."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
from numpy.typing import NDArray

from superglm._blas_threads import solver_blas_threads
from superglm._fit_gate import fit_record
from superglm._frame import FrameLike, as_eager_frame
from superglm.distributions import Distribution
from superglm.links import Link
from superglm.penalties.base import Penalty
from superglm.solvers.pirls import PIRLSResult
from superglm.terms import TermInput, TermResults, resolve_label
from superglm.types import FeatureSpec

from . import (
    base,
    explain_ops,
    fit_ops,
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
    from superglm.inference._term_types import InteractionInference, TermInference
    from superglm.inference.factor_smooths import FactorSmoothResult
    from superglm.inference.metrics import ModelMetrics
    from superglm.inference.random_effects import RandomEffectResult
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
        features: Mapping[Hashable, FeatureSpec] | None = None,
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
        separation: str = "warn",
        group_pricing: Literal["rank", "spanned"] = "rank",
        weight_semantics: Literal["prior", "frequency"] = "prior",
        # Threads
        n_jobs: int | Literal["auto"] = "auto",
        max_memory: int | str = "auto",
        *,
        terms: Sequence[TermInput] | None = None,
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
        features : mapping[hashable, FeatureSpec], optional
            Explicit feature specifications mapping hashable column labels to feature
            objects (``Spline``, ``Categorical``, ``Numeric``, ``Polynomial``).
            Mutually exclusive with *splines* and *terms*. A list of term
            declarations belongs in *terms*; ``features=`` refuses one with a
            ``TypeError``.
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
            are auto-detected from their parent feature specs. Mutually
            exclusive with *terms*.
        terms : list of str, BoundTerm, BoundInteraction or mapping, optional
            Keyword only. The model's terms as one list, in the form the
            ``SuperLSS`` family helpers such as ``family.location`` take: a
            column name for a numeric linear term; ``s``, ``cat``, ``re`` and
            ``term`` for main effects; ``ti`` and ``interaction`` for
            interactions; and ``{column: spec}`` mappings, each standing for
            one ``term(column, spec)`` per entry, in the mapping's order. For
            example, ``terms=[s("age", k=10), s("value", k=8), cat("region"),
            "density", ti("age", "value")]``. A list gives the same model as
            the equivalent ``features=`` and ``interactions=``, bit for bit,
            with main effects then interactions in declaration order. Cannot
            be combined with *features*, *splines* or *interactions*. A
            non-default *n_knots*, *degree* or *categorical_base* raises
            ``ValueError``; set those options on each term instead, as in
            ``s("age", n_knots=8)``.
        active_set : bool
            Use active-set cycling in the BCD solver.
        direct_solve : {"auto", "gram", "qr", "structured"}
            Strategy for the direct IRLS solver (lambda1=0).
            ``"auto"`` (default) uses compact structured elimination for large
            random-effect and factor-smooth (``basis="fs"`` or ``"sz"``)
            terms and the Gram solver otherwise; the choice reads only the
            terms, their sizes and the grouping columns' nesting, and
            ``result.direct_fallback_reason`` says why Gram was chosen. A fit
            never switches solver: when the structured solver cannot proceed
            the fit stops with an error naming the cause, and ``"gram"`` fits
            the model with the dense solver instead. ``"qr"`` uses QR on the
            materialised weighted design, backward-stable but O(n·p²) per
            iteration, for smaller datasets. ``"structured"`` forces
            structured elimination for an eligible random-effect, FS or SZ
            block, and raises for any other model.
        discrete : bool
            Use discretized basis matrices for large-*n* REML (fREML-style).
        n_bins : int or dict[str, int]
            Number of discretization bins per feature when ``discrete=True``.
        tol : float
            Convergence tolerance for IRLS / PIRLS.  Default ``1e-6``.
            Can also be set per-call via ``fit(tol=...)`` or
            ``fit_reml(pirls_tol=...)``.  Fit-time values take precedence.
            Larger values converge faster but may stop before
            near-separated coefficients have stabilised.
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
        separation : {"warn", "error", "ignore"}
            Check for separated cells: categorical levels or crossed cells
            with exposure whose responses all sit on the response boundary
            (no positive response under a log-link Poisson fit, say), so they
            have no finite maximum-likelihood effect. ``"warn"``
            (default) emits a ``SeparationWarning`` naming the cells and the
            remedies before fitting; ``"error"`` raises ``SeparationError``;
            ``"ignore"`` disables the check. Terms under an active selection
            penalty are exempt. The same mode governs the solver's backstop
            for separation the build scan cannot see, the hold on a
            spline-by-factor level's curve over a boundary-only range, and the
            warning for a ``FactorSmooth(basis="sz")`` level whose line
            separates (warned, never refused; ``select=True`` is the remedy).
            See the how-to *Interactions*, section *Separated cells*.
        group_pricing : {"rank", "spanned"}
            Dimension ``p_g`` at which the selection penalty and the fallback
            df ledger price a group whose spec emits fewer columns than the
            term spans (a categorical interaction with empty or nested
            cells).  ``"rank"`` (default) prices the emitted, identifiable
            width, following the group-lasso literature's derivation of
            ``sqrt(p_g)`` from the df of the group's score statistic.
            ``"spanned"`` prices the width the term spans, so cell pruning is
            a pure reparametrisation of the fit.  The choice moves fitted
            results only for models that combine such an interaction with an
            active group penalty, and moves the reported
            ``effective_df``/``phi``/AIC/BIC of any fit whose df falls back
            to the Breheny-Huang allocation.
        weight_semantics : {"prior", "frequency"}
            What ``sample_weight`` says about a row.  ``"prior"`` (default)
            reads it as an EDM prior weight -- a statement of precision,
            ``Var(Y_i) = phi V(mu_i) / w_i`` -- which is what you have when the
            response is an average: ``incurred / exposure`` weighted by
            exposure, or an average severity weighted by claim count.  This is
            the reading R's ``glm`` and glum give their single weight
            argument, and the one statsmodels calls ``var_weights``.
            ``"frequency"`` reads it as a replication count, so an integer
            weight is exactly equivalent to repeating the row; that is
            statsmodels' ``freq_weights`` and Stata's ``fweight``.

            The two agree only at ``w == 1``, and only the prior reading is a
            likelihood at fractional weights.  At fixed smoothing they share
            ``beta``; ``phi``, Wald standard errors, residual degrees of
            freedom, AIC/BIC, and through REML the smoothing parameters and
            fitted surface all differ, as does learned knot placement. See
            the explanation *Families and Dispersion Estimation*.
        n_jobs : int or "auto"
            The most threads a fit uses for its heaviest steps: forming the
            weighted Gram block by block and checking the rank of the design.
            Refits and inference on the fitted model (``summary``,
            ``metrics``, ``drop1``, ``estimate_p`` and the like) keep it.
            ``"auto"`` (default) uses the ``SUPERGLM_N_JOBS`` environment
            variable when it is set and the number of physical cores
            otherwise; ``1`` runs those steps on the calling thread.  The
            fitted model does not depend on it: every thread count gives the
            same coefficients, bit for bit.
        max_memory : int, str or "auto"
            The working memory those threads may hold at once, in bytes or as
            a size such as ``"4G"``.  A fit starts fewer threads when each
            needs more than this budget allows, so it is the limit to set on
            a shared machine.  ``"auto"`` (default) uses
            ``SUPERGLM_MAX_MEMORY`` when it is set and otherwise a quarter
            of the machine's memory, or of the container's memory limit
            where that is lower.
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
            separation=separation,
            group_pricing=group_pricing,
            weight_semantics=weight_semantics,
            n_jobs=n_jobs,
            max_memory=max_memory,
            terms=terms,
        )

    def _parallel_scope(self):
        """The fit's thread and memory limits (``_parallel.estimator_scope``).

        Every public method that can refit, build a Gram or factor the design
        enters it in its own body, not through a wrapper, whose extra frame
        would shift every warning's ``stacklevel``.
        """
        from superglm._parallel import estimator_scope

        return estimator_scope(
            getattr(self, "_n_jobs", "auto"), getattr(self, "_max_memory", "auto")
        )

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle; a retained solver state naming a retired class is dropped.

        Such a model loads and predicts as saved, and its first inference call
        rebuilds the linear system with the current engine
        (``superglm.model.retired_state``).  A v0.35.0 Tweedie model's REML
        memo is dropped, and so is a summary v0.35.0 cached after
        ``estimate_theta``.  A kindless spline the saved fit built as a SCOP
        P-spline is marked to warn at its next build
        (``mark_kindless_saved_on_scop``).
        """
        self.__dict__.update(state)
        # Metadata of a retired fast discrete predictor; nothing reads it.
        self.__dict__.pop("_fast_prediction_state", None)
        from superglm.model.retired_state import (
            mark_retired_linear_state,
            release_retired_tweedie_state,
            release_v0_35_nb_summaries,
        )

        mark_retired_linear_state(self)
        release_retired_tweedie_state(self)
        release_v0_35_nb_summaries(self)
        from superglm.features._spline_build import mark_kindless_saved_on_scop

        mark_kindless_saved_on_scop(self)

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

    def bind_levels(self, X: FrameLike, sample_weight: NDArray | None = None) -> SuperGLM:
        """Bind categorical level universes from the outermost frame.

        Runs the same pre-pass ``cross_validate`` applies to its own input:
        for every categorical-family feature, resolve the level universe and
        any unresolved ``most_exposed`` base from *X*, and store the result on
        this model's configuration.  Call it with the FULL frame before any
        train/val/test carve; every subsequent fit on any slice then shares one
        universe, and a level living only in the holdout is a pinned known
        level rather than a predict-time error.

        Explicit ``levels=`` on a term always wins, and a frame value outside
        such a declared universe fails here rather than at fit.  Re-calling
        replaces the stored bindings.  Returns ``self`` so construction chains:
        ``model = SuperGLM(...).bind_levels(df)``.
        """
        from superglm.model.binding_ops import resolve_level_bindings, validate_binding_weight

        frame = as_eager_frame(X)
        # The weight decides the pinned base and which levels count as empty
        # for every later fit, so it is validated here as `fit` validates its own.
        sample_weight = validate_binding_weight(sample_weight, len(frame))
        # strict=True: this frame is the caller's stated universe source, so a
        # column that cannot bind is an error now, not a surprise per fold.
        bindings = resolve_level_bindings(self, frame, sample_weight, strict=True)
        stored = tuple(bindings.items()) if bindings else None
        self._level_bindings = stored
        self._config = self._config.with_value(level_bindings=stored)
        self._config_revision += 1
        return self

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
            Observation weights, read under the model's declared
            ``weight_semantics``. Defaults to 1 for all observations.

            Under ``"prior"`` (the default) they are EDM prior weights:
            ``Y_i ~ ED(mu_i, phi / w_i)``, so
            ``Var(Y_i | x_i) = phi * V(mu_i) / w_i``, and estimated dispersion
            uses the count of positive-weight rows minus ``edf``. Under
            ``"frequency"`` they are replication counts: once feature geometry
            is fixed, integer weights are likelihood-equivalent to row
            replication, and estimated dispersion uses ``sum(w) - edf``. A
            Tweedie fit under ``"prior"`` additionally requires finite,
            strictly positive weights, because its compound-Poisson normalizer
            carries ``log w``.

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
        with fit_record(self, "fit", locals()):
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

            with solver_blas_threads(), self._parallel_scope():
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

        Warm-starts each lambda from the previous solution, and leaves the
        model fitted at the last one.  The returned ``coef_path`` and
        ``intercept_path`` are raw coefficients: beside a numeric column far
        from zero, rebuilding predictions from them cancels the column's
        offset (``PathResult``); predict with a fitted model instead.
        """
        with fit_record(self, "fit_path", locals()):
            with solver_blas_threads(), self._parallel_scope():
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
        max_reml_iter: int | None = None,
        reml_tol: float | None = None,
        pirls_tol: float | None = None,
        max_pirls_iter: int | None = None,
        lambda2_init: float | Mapping[str, float] | None = None,
        interaction_mode: str = "full",
        runtime_validation: str | bool | None = None,
        verbose: bool = False,
        w_correction_order: int = 1,
    ) -> SuperGLM:
        """Fit with REML estimation of per-term smoothing parameters.

        It optimizes a Laplace approximate REML objective over per-term
        smoothing parameters, and does not support a selection penalty:
        configure ``selection_penalty=None`` or ``0.0``. For sparse/group
        selection, use ``fit()`` or ``fit_path()``; to let REML shrink spline
        null spaces, use ``select=True`` on the spline terms.

        A Tweedie power within about 0.005 of 1, or further out for repeated
        responses, makes the dispersion profile multimodal; each solve then
        searches it globally, and a power too close to 1 for that search
        raises ``superglm.NearPoissonDispersionError``.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Feature matrix. Lazy frames must be collected before fitting.
        y : array-like
            Response variable.
        sample_weight : array-like, optional
            Observation weights, read under the model's declared
            ``weight_semantics``. Under ``"prior"`` (the default) they state a
            precision, ``Var(Y_i | x_i) = phi * V(mu_i) / w_i``; under
            ``"frequency"`` they are replication counts and integer weights are
            likelihood-equivalent to row replication once feature geometry is
            fixed. The contract sets the likelihood size the REML criterion
            profiles against, and so reaches the selected smoothing parameters
            as well as the dispersion. Weights affect fitting but do not enter
            the linear predictor or automatically scale the conditional mean.
        offset : array-like, optional
            Offset term.
        max_reml_iter : int, optional
            Maximum REML outer iterations. Unset, it is 20. A fit that
            reaches the cap is returned with a ``ConvergenceWarning``,
            ``reml_diagnostics()["converged"]`` reads ``False`` and the
            summary says why. A cold exact fit with at least eight smoothing
            parameters and no ``RandomEffect`` or ``FactorSmooth`` term may
            first run up to this many cached-W updates (a pre-solve);
            ``reml_diagnostics()["n_reml_iter"]`` counts the exact loop and
            ``["n_presolve_iter"]`` the pre-solve.
        reml_tol : float, optional
            Stopping tolerance for the smoothing-parameter optimizer. Unset,
            it is 1e-9 for both Newton engines (exact and discrete); the
            discrete engine floors an explicit value at 1e-12, and
            candidate-grade fits (``interaction_mode="fast_candidate"`` and
            ``estimate_p``'s power-search candidates) use 1e-6. The engines
            stop when the projected log-lambda gradient and the objective
            change both fall below ``reml_tol * (1 + |objective|)`` over the
            active directions, or when Newton's step on those directions is at
            most ``min(1e-3, 1e6 * reml_tol)`` in log lambda on an unmodified
            positive definite Hessian
            (``termination_reason="newton_step_tolerance"``), with the distance
            left, the step over one minus its contraction rate, within that bar.
            The exact engine takes the step and reads the rate from the next
            gradient, or from the preceding uncapped Newton step on the same
            free terms when no trial verifies this one. The discrete engine reads
            it from two successive steps on the same terms at a settled working
            model. A direction whose gradient and relative row
            curvature are both negligible is frozen where it stands, as
            recorded in ``reml_diagnostics()`` under
            ``profile["reml_freeze_decision"]``. A tolerance tighter than the
            line search can resolve ends as
            ``termination_reason="converged_at_precision"`` with
            ``converged=True``; ``line_search_failed`` with
            ``converged=False`` is reserved for undetermined stalls. Looser
            values (``1e-6``) are faster, but leave standard errors less
            determined well before predictions move.
        pirls_tol : float, optional
            Inner PIRLS/IRLS convergence tolerance. Defaults to
            constructor ``tol`` (1e-6). Pass explicitly to override.
        max_pirls_iter : int, optional
            Maximum inner PIRLS iterations per REML step. Defaults to
            constructor ``max_iter`` (100).
        lambda2_init : float or mapping, optional
            Where the smoothing-parameter search starts. A mapping keyed by
            smoothing component (or group) name, such as an earlier fit's
            ``reml_diagnostics()["lambdas"]``, is a warm start: each named
            component begins there, so a refit on similar data (a
            cross-validation fold, a later year) takes fewer outer
            iterations. Values must be finite and positive; unnamed and
            fixed-policy components start as usual. A term whose components
            are only partly named starts cold, and an unknown key is ignored,
            each with a ``UserWarning``. Coefficients are not carried over,
            and where the criterion is flat a different start can stop at a
            different point of the flat region. A float, or leaving it unset,
            keeps the engines' own start.
        interaction_mode : {"full", "fast_candidate"}
            ``"full"`` runs ordinary REML. ``"fast_candidate"`` caps REML
            outer updates for interaction models and then runs the normal final
            refit, intended for screening candidate interactions before a full
            final model fit.
        runtime_validation : optional
            Deprecated and ignored. It controlled a post-fit check that
            re-scored the training rows; the check has been retired, so any
            value only raises a ``FutureWarning``. Remove the argument.
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
        if runtime_validation is not None:
            import warnings

            warnings.warn(
                "fit_reml(runtime_validation=...) has no effect: the post-fit check it "
                "controlled has been retired. Remove the argument; it will be removed "
                "in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        with fit_record(self, "fit_reml", locals()):
            # Resolve PIRLS controls: explicit kwargs > constructor fallback
            resolved_pirls_tol = pirls_tol if pirls_tol is not None else self._tol
            resolved_max_pirls_iter = (
                max_pirls_iter if max_pirls_iter is not None else self._max_iter
            )

            from superglm.solvers.centered_system import reuse_data_factors

            with solver_blas_threads(), self._parallel_scope(), reuse_data_factors():
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
        """Rank candidate pair interactions by PSST.

        PSST (Penalized Smooth Score Test) asks, for each candidate pair of
        fitted features, how much of the model's leftover working signal the
        interaction the pair would actually refit as could absorb at a
        fixed screening complexity, after profiling out what the pair's own
        main effects already explain.  The ``kind`` column names that refit
        target — ``ti`` for spline x spline, ``spline_cat`` for a spline
        crossed with a factor, ``cat_cat`` for two factors, ``numeric_cat``
        for a per-level numeric slope and ``numeric_numeric`` for a product
        of two numerics — and ``z`` normalizes each kind against its own
        noise floor, so the one sorted table ranks them together, though not
        on equal terms: the null maxima differ by kind, so compare like with
        like before spending a refit (see the floors in the screening
        guide).  A spline-mode ``OrderedCategorical`` margin screens as a
        spline on its mapped level scores, so its pairs carry the spline
        kinds.  A kind whose block carries no penalty (``cat_cat``,
        ``numeric_cat``, ``numeric_numeric``) has no bandwidth to scan and is
        evaluated at a single rung: ``edf0`` then reports the block's
        achieved rank with ``lambda0`` 0.  ``edf0`` is a probe bandwidth; the
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

        Fitted mains with no screenable margin — ``Polynomial``,
        ``RandomEffect``, step-mode ``OrderedCategorical`` and any
        ``OrderedCategorical`` carrying ``specials=`` — are excluded from the
        sweep and reported in ``table.attrs["deferred_features"]``, a
        ``{feature: reason}`` mapping that is empty when everything fitted was
        screened.  Naming one of them in ``candidates`` raises with the same
        reason.

        The returned frame is sorted by ``z`` descending; rank by ``z`` —
        ``statistic``/``edf0``/``lambda0`` describe each pair's winning
        rung, so ``statistic`` is not comparable across rows.  Pairs whose
        unique-value grid exceeds ``max_cells`` are quantile-binned to
        ``screen_bins`` support points per margin and flagged
        ``approx=True``; pairs within budget are always computed exactly.
        ``max_cells`` bounds allocation AND time: the probe block's dimension
        ``k`` enters every rung as a ``(k, k)`` decomposition — a pivoted QR
        of the pair's design factor with one SVD beside it — so per-pair time
        grows as ``k^3`` where the allocations grow as ``k^2``.  A probe
        column collinear with the overlap span is routine rather than
        exceptional (one empty ``cat_cat`` cell or one singleton level does
        it) and is settled by a rank cut on that factor.  A cubic-work gate
        caps ``k`` at the FLOOR of ``(1000 * max_cells)^(1/3)`` for an
        unpenalized block and of ``(500 * max_cells)^(1/3)`` for a penalized
        one, whose ladder can bisect: 1709 and 1357 at the default.  Take the
        floor — at ``max_cells`` 1e4 the penalized root is 170.998 and the cap
        is 170, not 171.  That cap is NOT a constant — ``max_cells`` sets it
        and nothing else does.  Penalized ceilings: 1357 at 5e6, 793 at 1e6,
        368 at 1e5, 292 at 5e4, 170 at 1e4, with an allocation gate
        ``k <= 2 * sqrt(max_cells)`` binding instead below ``max_cells`` ~3906
        (the crossover is ~15625 for an unpenalized block, so a wide
        ``cat_cat`` reaches the allocation gate at a ``max_cells`` four times
        larger).  Family, link, ``phi``, penalty order, basis kind, prior
        weights and the ``edf0`` ladder move none of it; only the block's own
        width does, and for ``spline_cat`` that width is ``k_spline`` times
        the factor's CONTRAST count — the levels that carry a column, not the
        declared universe, since a level with no effective training rows is
        pinned to base and contributes none.

        Which kernel scores a pair follows from that.  The dense path gets
        first refusal and keeps every pair it can score exactly.  A
        ``spline_cat`` pair it cannot is RETRIED through the arrow kernel —
        linear in the level count, with no block-dimension ceiling — rather
        than refused, at any of three exits.  First, the block over the cap
        above: at the default a library-default ``Spline()`` margin (a ``cr``
        of 10 knots, 11 probe columns) crosses it at 124 contrasts, so from
        ``L = 125`` UNPINNED levels (``ps(10)`` from 106, ``ps(15)`` from 77,
        ``ps(20)`` from 61) — a declared ``levels=`` universe can be
        far larger and still route dense, since only the unpinned levels
        widen the block.  Second, the pair's dense support intermediate over
        budget where the arrow kernel's transposed one still fits, which
        fires with the dense block far BELOW the cap (measured at
        ``k = 231``: a 22-level factor against a ``ps(8)`` margin with 46,119
        distinct values).  Third, the exhaustion of the binning fallback.
        Only ``spline_cat`` has that retry: ``ti`` and ``cat_cat`` are
        gridded and dense-only, so for them the cap IS the refusal.  The two
        numeric kinds are neither — they enter through the z-moment kernels
        rather than the gridded one, and ``numeric_numeric`` contracts to a
        five-column design whatever the supports, so it never meets the cap at
        all.  Every kind is reduced to a triangular FACTOR of its own weighted
        design rather than to a curvature matrix.

        A NaN row is not by itself a routing signal.  A ``spline_cat`` pair
        reaches one either because the arrow gates or its ladder refused
        after a handoff, or because the ladder that ran — on EITHER path —
        resolved no direction at any rung, which is reported as no test
        rather than as an infinite ``z``.

        A categorical margin never bins — its support is the fitted level
        set — and a numeric margin never grids at all: it enters its probe
        linearly, so a ``numeric_cat``/``numeric_numeric`` row that carries a
        statistic is exact.  Those kinds have no binning fallback, so their
        only degradation is refusal: a ``numeric_cat`` pair whose factor is
        too wide for the pair's blocks to fit ``max_cells`` — or to be solved
        inside it, which binds first and admits 1710 UNPINNED levels (1709
        contrasts) at the default, counted the same way as the
        ``spline_cat`` threshold above — gets a NaN row, and raising
        ``max_cells`` computes it exactly.  Screening always probes exact
        bases, so a pair whose confirmatory refit would discretize LOSSILY is
        flagged ``approx=True`` to make that support-discretization gap, the
        same class as the quantile fallback, visible in the output.  The flag
        applies the refit's own gate, which differs by kind: a ``ti`` refit
        bins its marginals only when BOTH parents resolve to fit-time
        discretization, a ``spline_cat`` refit bins its spline margin whenever
        that ONE parent does, and the
        ``cat_cat``/``numeric_cat``/``numeric_numeric`` refits bin nothing.
        Lossless binning returns the exact support and stays ``approx=False``.
        Pairs already fitted as an interaction of any class are excluded from
        the sweep.  The statistic is a ranking device, not a calibrated
        p-value: confirm the top-ranked pairs by refitting them as the
        interaction their ``kind`` names.
        """
        with self._parallel_scope():
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
        with self._parallel_scope():
            return state_ops.coef_covariance(self)

    @cached_property
    def _fit_active_info(self):
        with self._parallel_scope():
            return state_ops.fit_active_info(self)

    @cached_property
    def _fit_inference_info(self):
        with self._parallel_scope():
            return state_ops.fit_inference_info(self)

    @cached_property
    def _group_edf(self):
        with self._parallel_scope():
            return state_ops.group_edf(self)

    # ── Diagnostics & summary ─────────────────────────────────────

    def random_effects(
        self,
        name: Hashable,
        *,
        exposure: NDArray | None = None,
        X: FrameLike | None = None,
        y: NDArray | None = None,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> RandomEffectResult:
        """Return variance-component and per-level credibility diagnostics."""
        with self._parallel_scope():
            from superglm.inference.random_effects import random_effect_result

            return random_effect_result(
                self,
                self._term_label(name),
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
        with self._parallel_scope():
            from superglm.inference.factor_smooths import factor_smooth_result

            return factor_smooth_result(
                self,
                cast(str, self._term_label(name)),
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
        with self._parallel_scope():
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

        Coefficient rows use fitted-family covariance: known-scale families
        retain ``phi=1`` and estimated-scale families use fitted ``phi``.
        Pearson quasi-likelihood diagnostics are available separately from
        :meth:`metrics` through ``coefficient_se``, ``intercept_se``, and
        ``feature_se``. The dict-like ``standard_errors`` payload labels its
        coefficient-SE scale explicitly as ``"fitted-family"``.

        The deviance, log-likelihood, information criteria and Pearson
        statistic reported here are the fit's own, computed when it was
        fitted on its fitting design.  On a ``discrete=True`` fit that design
        is the binned one, so these differ slightly from
        :meth:`metrics` on the same data, which evaluates the model's
        predictions on the rows passed.
        """
        with self._parallel_scope():
            return report_ops.summary(
                self,
                alpha,
                detail=detail,
                level_display=level_display,
            )

    def _term_label(self, name: Hashable) -> Any:
        """The term ``name`` names, by its label or by its label's text."""
        model = cast(Any, self)
        return resolve_label(name, [*model._specs, *model._interaction_specs])

    def _feature_groups(self, name: Hashable) -> list[GroupSlice]:
        """Get all groups belonging to a feature."""
        return report_ops.feature_groups(self, cast(Any, name))

    def reconstruct_feature(self, name: Hashable) -> dict[str, Any]:
        """Reconstruct a fitted feature's curve or effect on its original scale."""
        with self._parallel_scope():
            return report_ops.reconstruct_feature(self, self._term_label(name))

    def knot_summary(self) -> dict[str, dict[str, Any]]:
        """Return fitted knot metadata for all spline features."""
        return TermResults(report_ops.knot_summary(self))

    # ── Inference ─────────────────────────────────────────────────

    def metrics(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model.

        The deviance, likelihoods, Pearson statistic and ``residuals()`` are
        computed from the model's predictions on ``X``, the values ``predict``
        returns.  They do not depend on object identity: equal copies of the
        training data and a saved and reloaded model give the same values as
        the objects the model was fitted on.  Leverage-based diagnostics do
        depend on it.  On the fit's own objects, ``leverage``, the standardized
        residuals and Cook's distance read the fit's own geometry; see the
        :class:`ModelMetrics` notes.

        :meth:`summary` instead reports the fit's own statistics from its
        fitting design.  On a ``discrete=True`` fit that design is the binned
        one, so the two differ slightly on the training data.
        """
        with self._parallel_scope():
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
        """Drop-one deviance analysis for each feature.

        At exactly zero fitted dispersion, a zero deviance change has statistic
        zero and p-value one; a nonzero deviance change is undefined and raises.
        """
        with self._parallel_scope():
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
        """Refit with only active features and no selection penalty.

        The refit keeps the model's declared ``separation`` mode. A level whose
        separation the selection penalty exempted is not exempt without it, so
        under ``separation="error"`` the refit raises :class:`SeparationError`
        for that level, and under ``"warn"`` it fits with a
        :class:`SeparationWarning`.
        """
        with self._parallel_scope():
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
            comparison but not the fitted term decomposition.  It is a change
            of identifiability constraint, so ``se_log_relativity`` changes
            with it: it becomes the error of the centered contrast.
        """
        with self._parallel_scope():
            return TermResults(explain_ops.relativities(self, with_se, centering=centering))

    def simultaneous_bands(
        self,
        feature: Hashable,
        *,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        n_points: int = 200,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Simultaneous confidence bands for a spline feature."""
        with self._parallel_scope():
            return explain_ops.simultaneous_bands(
                self,
                self._term_label(feature),
                alpha=alpha,
                n_sim=n_sim,
                n_points=n_points,
                seed=seed,
            )

    def term_inference(
        self,
        name: Hashable,
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
            geometric mean of relativities = 1.  It is a change of
            identifiability constraint, so the standard errors and intervals
            change with it: they become those of the centered contrast, and
            the level the fit pinned stops carrying a zero-width interval.
        """
        with self._parallel_scope():
            return explain_ops.term_inference(
                self,
                self._term_label(name),
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
        search_fit_mode: str | None = None,
        p_bounds: tuple[float, float] = (1.05, 1.95),
        xatol: float = 1e-3,
        ci_alpha: float | None = None,
        max_reml_iter: int | None = None,
        progress_callback: Callable[..., None] | None = None,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        At each candidate ``p`` the mean is refitted and ``phi`` is its
        maximum-likelihood value at that mean; ``p`` minimises the resulting
        profile negative log-likelihood by bounded Brent search (Dunn & Smyth
        2005), and the published fit is refitted at the selected ``p``. The
        search is local: if the profile has more than one interior minimum
        in ``p_bounds``, it can return one that is not the global minimum,
        with no warning. Only an estimate on a bound is flagged.

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
            Fitting regime for the published final fit, and by default for each
            candidate ``p`` evaluation too.
        search_fit_mode : {"fit", "reml", "inherit"}, optional
            Fitting regime for the candidate ``p`` evaluations alone, when it
            should differ from the regime the final fit is published under. The
            default ``None`` searches under ``fit_mode``.

            ``fit_mode="reml", search_fit_mode="fit"`` selects ``p`` under
            ordinary ML and then publishes one REML fit at the selected ``p``,
            the cost of one smoothing-parameter search rather than one per
            candidate. Selecting ``p`` under a different objective than the
            one that publishes it is an approximation, so check it on your
            own data. Either way the published refit runs at the publication
            tolerance and re-profiles dispersion against its own mean. A
            coupled REML search warns when its optimum is pinned against the
            certifiable-region boundary; a decoupled refit may fail at the selected ``p`` with a
            ``superglm.PublicationModeError`` naming the ways out, and its
            likelihood-ratio intervals describe the ``search_fit_mode``
            profile.
        p_bounds : tuple of float
            Search interval for ``p``, strictly inside ``(1, 2)``. An estimate on
            a bound is warned about and recorded in ``result.warnings``: a
            maximum as ``p -> 1`` can be an artefact of rounded responses. Near
            ``p = 1`` the dispersion profile is multimodal and its global search
            grows like ``1 / (p - 1)``; a power whose search would exceed its
            bound (``superglm.NearPoissonDispersionError``) is skipped as
            infeasible and recorded.
        xatol : float
            Absolute resolution of the Brent search in ``p``.
        ci_alpha : float, optional
            Significance level for an explicitly requested likelihood-ratio
            profile confidence interval. For example, ``0.05`` computes a 95%
            interval and caches it for ``model.summary(alpha=0.05)``. The
            default ``None`` performs no confidence-interval evaluations. If
            the searched winner's fit, or a fit the interval evaluates, did not
            converge, the interval is still computed, and the cause is warned
            and recorded in ``result.warnings``.
        max_reml_iter : int, optional
            Outer-iteration budget for the REML *publication* refit alone;
            candidate search fits keep their own budget. Requires
            ``fit_mode="reml"`` -- a pure-ML publication has no REML
            iteration to budget and refuses the parameter. The default
            ``None`` resolves as ``fit_reml``'s does, to 20.
        progress_callback : callable, optional
            Called as ``progress_callback(phase, payload)``: ``"profiling"``
            with ``{"profile_trace": [row]}`` for each feasible search
            candidate, then ``"best_found"`` and ``"final_refit"`` with
            ``{"profile_estimate": ...}``.
        """
        with fit_record(self, "estimate_p", locals()):
            with self._parallel_scope():
                return profile_ops.estimate_p(
                    self,
                    X,
                    y,
                    sample_weight,
                    offset,
                    fit_mode=fit_mode,
                    search_fit_mode=search_fit_mode,
                    p_bounds=p_bounds,
                    xatol=xatol,
                    ci_alpha=ci_alpha,
                    max_reml_iter=max_reml_iter,
                    progress_callback=progress_callback,
                )

    def estimate_theta(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        fit_mode: str = "fit",
        theta_bounds: tuple[float, float] = (1e-8, 1e8),
        xatol: float = 1e-2,
        ci_alpha: float | None = None,
        progress_callback: Callable[..., None] | None = None,
    ):
        """Estimate NB2 theta via profile likelihood, refit, and return result.

        The mean fit alternates with the root of the closed-form profile score
        in theta at that mean (Venables & Ripley 2002, ch. 7.4; Lawless 1987)
        until theta settles, and the published fit is refitted at the estimate.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Feature matrix. Lazy frames must be collected before fitting.
        y : array-like
            Count response.
        sample_weight : array-like, optional
            Observation weights, read under the model's ``weight_semantics``.
        offset : array-like, optional
            Offset added to the linear predictor.
        fit_mode : {"fit", "reml", "inherit"}
            Fitting regime for the published final fit. The alternation itself
            uses ordinary fits.
        theta_bounds : tuple of float
            Search range for theta. An estimate on a bound warns with
            ``NBThetaBoundWarning`` and reports ``converged=False``.
        xatol : float
            The alternation stops once theta moves by at most this fraction of
            itself between successive mean fits.
        ci_alpha : float, optional
            Compute the ``1 - ci_alpha`` likelihood-ratio interval at the
            published mean before returning. It is inverted from that mean's
            own profile optimum; where theta_hat is not it (an alternation or
            joint refinement that stopped short, or theta_hat outside that
            interval), a caution is warned, recorded, and shown in the
            interval's status.
        progress_callback : callable, optional
            Called as ``progress_callback(phase, payload)``: ``"profiling"``
            with ``{"profile_trace": [row]}`` for each alternation step, then
            ``"best_found"`` and ``"final_refit"`` with ``{"profile_estimate": ...}``.
        """
        with fit_record(self, "estimate_theta", locals()):
            with self._parallel_scope():
                return profile_ops.estimate_theta(
                    self,
                    X,
                    y,
                    sample_weight,
                    offset,
                    fit_mode=fit_mode,
                    theta_bounds=theta_bounds,
                    xatol=xatol,
                    ci_alpha=ci_alpha,
                    progress_callback=progress_callback,
                )

    # ── Plotting ──────────────────────────────────────────────────

    def plot(
        self,
        terms: Hashable | Sequence[Hashable] | None = cast(Any, plot_ops.TERMS_UNSET),
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
        terms : hashable label, list of labels, or None
            Which term(s) to plot. ``None`` plots all main effects unless the
            fitted model has a feature whose exact column label is ``None``.
        kind : {"global", "local"}
            ``"global"`` shows model-wide fitted effects (default).
            ``"local"`` is reserved for per-row explanations (not yet
            implemented).
        ci : {None, False, "pointwise", "simultaneous", "both"}
            Confidence interval style.  ``None`` or ``False`` disables bands.
        X : pandas or eager Polars DataFrame, optional
            Training data for density overlays.
        sample_weight : array-like, optional
            Observation weights for density overlays. When reusing fitting
            weights, they keep the model's declared ``weight_semantics``:
            replication counts under ``"frequency"``, precisions under
            ``"prior"``.
        show_density : bool
            Show sample-weighted observation density (strip for continuous,
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
            geometric mean of relativities = 1.  It is a change of
            identifiability constraint, so the standard errors and intervals
            change with it: they become those of the centered contrast, and
            the level the fit pinned stops carrying a zero-width interval.
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
        with self._parallel_scope():
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
        2. Calibration (sample-weighted observed vs predicted)
        3. Residuals vs Linear Predictor
        4. Residual distribution (histogram + N(0,1) overlay)

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Design matrix.
        y : NDArray
            Response vector.
        sample_weight : NDArray or None
            Observation weights, read under the model's declared
            ``weight_semantics``: replication counts under ``"frequency"``,
            precisions under ``"prior"``. The quantile residuals below take
            the row's own marginal distribution from that contract, so a
            prior weight enters the reference distribution and a replication
            count does not.
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
        with self._parallel_scope():
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
        terms: Hashable | Sequence[Hashable] | None = cast(Any, plot_ops.TERMS_UNSET),
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

        A ``FactorSmooth`` term exports like a ``SplineCategorical``, with
        ``plot_kind="varying_coefficient"``: ``effect`` has one row per level
        and grid value of the smoothed variable (``n_points`` values), with
        columns named by the variable and the group, ``log_relativity``,
        ``relativity``, ``is_base_level`` (always ``False``: every level has a
        curve) and ``se_log_relativity``, the curve's posterior standard error.
        These are the curves :meth:`factor_smooth` reports: each level's own
        curve for ``basis="fs"``, its deviation from the population curve for
        ``basis="sz"``.

        With ``centering="native"`` (default), relativity values are
        the exponentiated fitted term contributions under the model's
        identifiability constraint — not portfolio-average relativities.
        Pass ``centering="mean"`` for a reporting view where the
        geometric mean of relativities = 1.

        A grouped ``OrderedCategorical`` is always exported on the **expanded**
        axis, one row per original level, where :meth:`plot` defaults to the
        collapsed view. For a merged group the two payload pieces then live
        on different axes: ``effect["x_position"]`` holds the declared level
        values, while ``smooth_curve["x"]`` spans the fitted axis, on which a
        merged group sits at its members' mean position. Join the two on the
        level table, not on x.

        Examples
        --------
        >>> payload = model.plot_data("DrivAge", X=X_train, sample_weight=w, show_knots=True)
        >>> curve_df = payload["terms"][0]["effect"]
        >>> knots_df = payload["terms"][0]["knots"]
        """
        with self._parallel_scope():
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
        fitted: bool = False,
    ) -> NDArray:
        """Private exact canonical predictor on the link scale (``fitted``: ``base._predict_eta``)."""
        return base.predict_eta_exact(
            self, X, offset, random_effects=random_effects, fitted=fitted, warn=not fitted
        )

    def _predict_eta_raw_exact(
        self,
        X: FrameLike,
        offset: NDArray | None = None,
        *,
        random_effects: str = "conditional",
    ) -> NDArray:
        """Private unstabilized exact predictor for raw link-scale APIs."""
        return base.predict_eta_raw_exact(
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
            zero, except that a ``basis="sz"`` FactorSmooth with levels its
            data identify only in part, or levels whose line separates the
            response, contributes its population curve's offset (see "SZ
            levels the data cannot identify" in the interactions guide).

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
        """Repair post-fit shape-constrained spline terms after fitting.

        This is a manual, post-fit repair step. It finds all spline features
        configured with ``Constraint.postfit.*``, applies the corresponding
        weighted shape projection to the fitted curve, and projects the
        repaired curve back to spline coefficients.

        Idempotent: calling twice does not re-repair already-repaired features.

        Parameters
        ----------
        X : pandas or eager Polars DataFrame
            Training data (used to compute density-based grid weights).
        sample_weight : array-like, optional
            Fitting weights used to weight the post-fit projection, read
            under the model's declared ``weight_semantics``: replication counts
            under ``"frequency"``, precisions under ``"prior"``.
        offset : array-like, optional
            Offset term (unused, reserved for deviance computation).
        n_grid : int
            Grid resolution for the shape repair (default 500).

        Returns
        -------
        SuperGLM
            The model (self), with post-fit shape repairs stored.
        """
        with self._parallel_scope():
            from superglm.model import shape_ops

            return shape_ops.apply_shape_postfit(self, X, sample_weight, offset, n_grid=n_grid)

    def apply_shape_postfit(
        self,
        X: FrameLike,
        sample_weight: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_grid: int = 500,
    ) -> SuperGLM:
        """Repair postfit monotone and curvature-constrained spline terms."""
        with self._parallel_scope():
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
        """Compatibility alias for :meth:`superglm.SuperGLM.monotonize`."""
        with self._parallel_scope():
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
        with self._parallel_scope():
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
        with self._parallel_scope():
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
        with self._parallel_scope():
            return explain_ops.spline_redundancy(self, X, sample_weight)

    # ── Discretization ────────────────────────────────────────────

    def discretization_impact(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing this fit's smooth terms.

        Spline and polynomial main effects into bins, and a
        continuous-by-continuous interaction onto the grid its rating-table
        block is sampled on."""
        with self._parallel_scope():
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
        with self._parallel_scope():
            from superglm.export import export_rating_tables

            return export_rating_tables(
                self, file_path, X, y, sample_weight=sample_weight, **kwargs
            )

    def rating_table_payload(
        self,
        X: FrameLike,
        y: NDArray,
        sample_weight: NDArray | None = None,
        **kwargs,
    ):
        """Build the renderer-independent deployment rating-table payload."""
        with self._parallel_scope():
            from superglm.export.rating_tables import build_rating_table_payload

            return build_rating_table_payload(self, X, y, sample_weight=sample_weight, **kwargs)


SuperGLM.__module__ = "superglm.model"
