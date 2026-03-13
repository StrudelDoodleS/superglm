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
from superglm.inference import (
    InteractionInference,
    TermInference,
    compute_coef_covariance,
    feature_se_from_cov,
)
from superglm.inference import drop1 as _drop1
from superglm.inference import refit_unpenalised as _refit_unpenalised
from superglm.inference import relativities as _relativities
from superglm.inference import simultaneous_bands as _simultaneous_bands
from superglm.inference import term_inference as _term_inference
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
from superglm.types import FeatureSpec, FitStats, GroupSlice

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


def _compute_fit_stats(
    y: NDArray,
    mu: NDArray,
    weights: NDArray,
    offset: NDArray | None,
    distribution: Distribution,
    link: Link,
    phi: float,
) -> FitStats:
    """Compute scalar fit statistics from training arrays.

    Called once at end of fit()/fit_reml() while y/mu are still in scope,
    so that summary() never needs to cache the full training arrays.
    """
    ll = distribution.log_likelihood(y, mu, weights, phi)

    # Null model (intercept-only MLE), offset-aware
    y_bar = float(np.average(y, weights=weights))
    y_bar = max(y_bar, 1e-10)

    if offset is None or np.all(offset == 0):
        null_mu = np.full(len(y), y_bar)
    else:
        b0 = link.link(y_bar) - float(np.average(offset, weights=weights))
        for _ in range(25):
            eta_null = b0 + offset
            mu_null = link.inverse(np.clip(eta_null, -20, 20))
            dmu = link.deriv_inverse(np.clip(eta_null, -20, 20))
            V = distribution.variance(mu_null)
            score = np.sum(weights * (y - mu_null) * dmu / V)
            info = np.sum(weights * dmu**2 / V)
            step = score / max(info, 1e-10)
            b0 += step
            if abs(step) < 1e-8:
                break
        eta_null = b0 + offset
        null_mu = np.maximum(link.inverse(np.clip(eta_null, -20, 20)), 1e-10)

    null_ll = distribution.log_likelihood(y, null_mu, weights, phi)
    null_dev = float(np.sum(weights * distribution.deviance_unit(y, null_mu)))
    dev = float(np.sum(weights * distribution.deviance_unit(y, mu)))
    expl_dev = 1.0 - dev / null_dev if null_dev > 0 else 0.0

    V = distribution.variance(mu)
    pearson = float(np.sum(weights * (y - mu) ** 2 / V))

    return FitStats(
        log_likelihood=ll,
        null_log_likelihood=null_ll,
        null_deviance=null_dev,
        explained_deviance=expl_dev,
        pearson_chi2=pearson,
        n_obs=len(y),
    )


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
        self._fit_stats: FitStats | None = None
        self._nb_profile_result = None  # NBProfileResult, set by estimate_theta()
        self._tweedie_profile_result = None  # TweedieProfileResult, set by estimate_tweedie_p()
        self._last_fit_meta: dict[str, Any] | None = None

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

    @staticmethod
    def _resolve_sample_weight_alias(
        exposure: NDArray | None,
        sample_weight: NDArray | None,
        *,
        method_name: str,
    ) -> NDArray | None:
        """Resolve the public sample_weight alias for exposure/frequency weights."""
        if exposure is not None and sample_weight is not None:
            raise TypeError(
                f"{method_name} received both 'exposure' and 'sample_weight'. "
                "Use only 'sample_weight'; 'exposure' is a backward-compatible alias."
            )
        return sample_weight if sample_weight is not None else exposure

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
        exposure = self._resolve_sample_weight_alias(exposure, sample_weight, method_name="fit()")
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
        self._fit_weights = np.array(exposure)  # copy: caller mutation must not affect summary
        self._fit_offset = np.array(offset) if offset is not None else None
        exposure = self._fit_weights  # use our copy downstream
        offset = self._fit_offset

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max(y, exposure) * 0.1

        # Invalidate cached properties from previous fit
        self.__dict__.pop("_coef_covariance", None)
        self.__dict__.pop("_fit_active_info", None)

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

        eta = self._dm.matvec(self._result.beta) + self._result.intercept
        if offset is not None:
            eta = eta + offset
        mu = self._link.inverse(eta)

        self._fit_stats = _compute_fit_stats(
            y, mu, exposure, offset, self._distribution, self._link, self._result.phi
        )

        self._last_fit_meta = {"method": "fit", "discrete": self._discrete}
        return self

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
        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="fit_path()"
        )
        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = np.array(exposure)
        self._fit_offset = np.array(offset) if offset is not None else None
        exposure = self._fit_weights
        offset = self._fit_offset
        self.__dict__.pop("_coef_covariance", None)
        self.__dict__.pop("_fit_active_info", None)
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
        sample_weight: NDArray | None = None,
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

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="fit_cv()"
        )

        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)

        if self.nb_theta == "auto":
            from superglm.nb_profile import estimate_nb_theta

            nb_result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset)
            self.nb_theta = nb_result.theta_hat
            self._nb_profile_result = nb_result
            logger.info(f"NB theta estimated: {nb_result.theta_hat:.4f}")

        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = np.array(exposure)
        self._fit_offset = np.array(offset) if offset is not None else None
        exposure = self._fit_weights
        offset = self._fit_offset
        self.__dict__.pop("_coef_covariance", None)
        self.__dict__.pop("_fit_active_info", None)

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
            split_linear_snap=getattr(self, "_split_linear_snap", True),
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
        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="fit_reml()"
        )
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

        self._fit_weights = np.array(exposure)
        self._fit_offset = np.array(offset) if offset is not None else None
        exposure = self._fit_weights
        offset = self._fit_offset
        self.__dict__.pop("_coef_covariance", None)
        self.__dict__.pop("_fit_active_info", None)

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

        eta = self._dm.matvec(self._result.beta) + self._result.intercept
        if offset is not None:
            eta = eta + offset
        mu = self._link.inverse(eta)

        self._fit_stats = _compute_fit_stats(
            y, mu, exposure, offset, self._distribution, self._link, self._result.phi
        )

        self._last_fit_meta = {"method": "fit_reml", "discrete": self._discrete}

        logger.info(f"REML converged={converged} in {n_reml_iter} iters, lambdas={lambdas}")
        return self

    @property
    def result(self) -> PIRLSResult:
        """The fitted PIRLS result (coefficients, deviance, convergence info).

        Raises ``RuntimeError`` if the model has not been fitted.
        """
        if self._result is None:
            raise RuntimeError("Not fitted")
        return self._result

    def diagnostics(self) -> dict[str, Any]:
        """Per-group diagnostic dict for programmatic / audit access.

        Returns a dict keyed by group name with ``active``, ``group_norm``,
        ``n_params`` (plus spline metadata when applicable) and a ``_model``
        entry with scalar fit statistics.
        """
        from superglm.features.spline import _SplineBase
        from superglm.inference import spline_group_enrichment

        res = self.result
        group_edf = self._group_edf
        reml_lam = getattr(self, "_reml_lambdas", None)

        out = {}
        for g in self._groups:
            bg = res.beta[g.sl]
            entry: dict[str, Any] = {
                "active": bool(np.any(bg != 0)),
                "group_norm": float(np.linalg.norm(bg)),
                "n_params": g.size,
            }
            spec = self._specs.get(g.feature_name)
            if isinstance(spec, _SplineBase):
                entry.update(
                    spline_group_enrichment(g.name, spec, group_edf, reml_lam, self.lambda2)
                )
            out[g.name] = entry
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

    def summary(self, alpha: float = 0.05):
        """Rich model summary with coefficient table (statsmodels-style).

        Uses cached fit statistics so no ``X`` / ``y`` arguments are needed.
        For diagnostics on a different sample, use ``model.metrics(X, y).summary()``.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals (default 0.05 → 95% CI).

        Returns
        -------
        ModelSummary
            Object with ``__str__`` (ASCII), ``_repr_html_`` (HTML/Jupyter),
            and dict-like access for backward compatibility.
        """
        import re

        from superglm.metrics import build_coef_rows
        from superglm.summary import ModelSummary

        if self._fit_stats is None:
            raise RuntimeError("No fit stats — call fit() or fit_reml() first.")

        fs = self._fit_stats
        res = self.result
        edf = res.effective_df
        n = fs.n_obs
        ll = fs.log_likelihood

        # Derived ICs
        aic = -2 * ll + 2 * edf
        bic = -2 * ll + np.log(n) * edf
        denom = n - edf - 1.0
        aicc = aic + 2 * edf * (edf + 1) / denom if denom > 0 else np.inf
        n_active = sum(1 for g in self._groups if np.linalg.norm(res.beta[g.sl]) > 1e-12)
        p_total = len(self._groups)
        from scipy.special import gammaln

        ebic = bic + 2 * 0.5 * (
            gammaln(p_total + 1) - gammaln(n_active + 1) - gammaln(p_total - n_active + 1)
        )

        data = {
            "information_criteria": {
                "log_likelihood": ll,
                "null_log_likelihood": fs.null_log_likelihood,
                "aic": aic,
                "bic": bic,
                "aicc": aicc,
                "ebic": ebic,
            },
            "deviance": {
                "deviance": res.deviance,
                "null_deviance": fs.null_deviance,
                "explained_deviance": fs.explained_deviance,
            },
            "fit": {
                "phi": res.phi,
                "effective_df": edf,
                "pearson_chi2": fs.pearson_chi2,
                "n_obs": n,
                "n_active_groups": n_active,
            },
        }

        # Model info (same structure as ModelMetrics.summary())
        penalty = self.penalty
        link_name = type(self._link).__name__
        if link_name.endswith("Link"):
            link_name = link_name[:-4]
        penalty_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", type(penalty).__name__)

        # Append "+ SL" if any split_linear splines are present
        has_select = any(g.subgroup_type is not None for g in self._groups)
        penalty_abbrevs: dict[str, str] = {}  # abbrev -> full name
        if has_select:
            short = {"Group Lasso": "GL", "Sparse Group Lasso": "SGL", "Group Elastic Net": "GEN"}
            abbrev = short.get(penalty_name)
            if abbrev is not None:
                penalty_abbrevs[abbrev] = penalty_name
                penalty_name = abbrev
            penalty_abbrevs["SL"] = "split-linear double penalty (Wood, 2011)"
            penalty_name += " + SL"

        # Build method string from fit metadata
        meta = self._last_fit_meta or {}
        method_parts = []
        if meta.get("method") == "fit_reml":
            method_parts.append("REML")
        else:
            method_parts.append("ML")
        if meta.get("discrete"):
            method_parts.append("discrete")
        method_str = ", ".join(method_parts)

        lam1 = penalty.lambda1
        model_info = {
            "family": {"NegativeBinomial": "Neg. Binomial"}.get(
                type(self._distribution).__name__, type(self._distribution).__name__
            ),
            "link": link_name,
            "penalty": penalty_name,
            "penalty_abbrevs": penalty_abbrevs,
            "method": method_str,
            "n_obs": n,
            "effective_df": edf,
            "lambda1": lam1,
            "phi": res.phi,
            "deviance": res.deviance,
            "log_likelihood": ll,
            "aic": aic,
            "converged": res.converged,
            "n_iter": res.n_iter,
        }

        # NB theta profile info
        nb_pr = getattr(self, "_nb_profile_result", None)
        if nb_pr is not None:
            ci = nb_pr.ci(alpha=alpha)
            model_info["nb_theta"] = nb_pr.theta_hat
            model_info["nb_theta_ci"] = ci
            model_info["nb_theta_method"] = "Profile (exact)"

        # Tweedie p profile info
        tw_pr = getattr(self, "_tweedie_profile_result", None)
        if tw_pr is not None:
            ci = tw_pr.ci(alpha=alpha)
            model_info["tweedie_p"] = tw_pr.p_hat
            model_info["tweedie_p_ci"] = ci
            model_info["tweedie_phi"] = tw_pr.phi_hat
            model_info["tweedie_p_method"] = "Profile (exact)"

        # Coef rows from shared builder
        X_a, W, XtWX_inv, active_groups = self._fit_active_info
        known_scale = getattr(self._distribution, "scale_known", True)
        coef_rows = build_coef_rows(
            groups=self._groups,
            specs=self._specs,
            interaction_specs=self._interaction_specs,
            result=res,
            X_a=X_a,
            W=W,
            XtWX_inv=XtWX_inv,
            active_groups=active_groups,
            known_scale=known_scale,
            group_edf_map=self._group_edf,
            reml_lambdas=getattr(self, "_reml_lambdas", None),
            lambda2=self.lambda2,
            n_obs=n,
            alpha=alpha,
        )

        # Standard errors for backward compat dict access
        phi = res.phi
        se_dict: dict[str, NDArray] = {}
        se_raw_dict: dict[str, NDArray] = {}
        beta = res.beta
        for g in self._groups:
            if np.linalg.norm(beta[g.sl]) < 1e-12:
                se_dict[g.name] = np.zeros(g.size)
                se_raw_dict[g.name] = np.zeros(g.size)
            else:
                ag = next((a for a in active_groups if a.name == g.name), None)
                if ag is None:
                    se_dict[g.name] = np.zeros(g.size)
                    se_raw_dict[g.name] = np.zeros(g.size)
                else:
                    var_diag = np.diag(XtWX_inv[ag.sl, ag.sl])
                    se_raw_dict[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
                    se_dict[g.name] = np.sqrt(np.maximum(phi * var_diag, 0.0))

        data["standard_errors"] = {
            "coefficient_se": se_dict,
            "coefficient_se_raw": se_raw_dict,
        }

        return ModelSummary(data, model_info, coef_rows, alpha=alpha)

    def _feature_groups(self, name: str) -> list[GroupSlice]:
        """Get all groups belonging to a feature (1 normally, 2 for split-linear splines)."""
        return [g for g in self._groups if g.feature_name == name]

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        """Reconstruct a fitted feature's curve or effect on its original scale.

        Parameters
        ----------
        name : str
            Feature or interaction name.

        Returns
        -------
        dict[str, Any]
            Feature-type-specific reconstruction (e.g. grid values,
            relativity curve, knot positions).
        """
        res = self.result
        groups = self._feature_groups(name)
        beta_combined = np.concatenate([res.beta[g.sl] for g in groups])
        if name in self._specs:
            return self._specs[name].reconstruct(beta_combined)
        if name in self._interaction_specs:
            return self._interaction_specs[name].reconstruct(beta_combined)
        raise KeyError(f"Feature not found: {name}")

    def knot_summary(self) -> dict[str, dict[str, Any]]:
        """Return fitted knot metadata for all spline features.

        Returns a dict keyed by feature name, each containing:

        * ``kind`` — spline class name (e.g. ``"BasisSpline"``,
          ``"CardinalCRSpline"``).
        * ``knot_strategy`` — the strategy actually used: ``"uniform"``,
          ``"quantile"``, ``"quantile_rows"``,
          ``"quantile_tempered"``, or ``"explicit"``.  If a quantile
          strategy fell back to uniform (too few distinct knots), this
          reports ``"uniform"``.
        * ``interior_knots`` — 1-D array of interior knot positions.
        * ``boundary`` — ``(lo, hi)`` tuple.
        * ``n_basis`` — number of raw basis functions (before
          identifiability / SSP).
        * ``knot_alpha`` — tempering exponent (only present when
          ``knot_strategy`` is ``"quantile_tempered"``).

        To fully freeze placement on a refit with different data, pass
        both ``knots`` and ``boundary``::

            info = model.knot_summary()["DrivAge"]
            Spline(knots=info["interior_knots"],
                   boundary=info["boundary"])
        """
        from superglm.features.spline import _SplineBase

        out: dict[str, dict[str, Any]] = {}
        for name, spec in self._specs.items():
            if not isinstance(spec, _SplineBase):
                continue
            entry: dict[str, Any] = {
                "kind": type(spec).__name__,
                "knot_strategy": spec._knot_strategy_actual,
                "interior_knots": spec.fitted_knots,
                "boundary": spec.fitted_boundary,
                "n_basis": spec._n_basis,
            }
            if spec._knot_strategy_actual == "quantile_tempered":
                entry["knot_alpha"] = spec.knot_alpha
            out[name] = entry
        return out

    @cached_property
    def _coef_covariance(self) -> tuple[NDArray, list[GroupSlice]]:
        """Phi-scaled Bayesian covariance for active coefficients."""
        lam2 = getattr(self, "_reml_lambdas", None) or self.lambda2
        return compute_coef_covariance(
            self._dm,
            self._distribution,
            self._link,
            self._groups,
            self.result,
            self._fit_weights,
            self._fit_offset,
            lam2,
        )

    @cached_property
    def _fit_active_info(self) -> tuple[NDArray, NDArray, NDArray, list[GroupSlice]]:
        """Active design columns, weights, and (X'WX+S)^{-1} from fit state.

        Same computation as ``ModelMetrics._active_info`` but recomputes mu
        from beta so no cached ``_train_mu`` is needed.
        """
        from superglm.metrics import _penalised_xtwx_inv

        eta = self._dm.matvec(self.result.beta) + self.result.intercept
        if self._fit_offset is not None:
            eta = eta + self._fit_offset
        mu = self._link.inverse(np.clip(eta, -20, 20))
        V = self._distribution.variance(mu)
        dmu_deta = self._link.deriv_inverse(np.clip(eta, -20, 20))
        W = self._fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

        lam2 = getattr(self, "_reml_lambdas", None) or self.lambda2
        X_a, XtWX_inv, active_groups, _ = _penalised_xtwx_inv(
            self.result.beta, W, self._dm.group_matrices, self._groups, lam2
        )
        return X_a, W, XtWX_inv, active_groups

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
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        Thin wrapper around :func:`superglm.tweedie_profile.estimate_tweedie_p`.
        After estimation, sets ``self.tweedie_p`` to the optimised value and
        refits the model using the same fitting regime.

        Parameters
        ----------
        fit_mode : {"fit", "reml", "inherit"}
            Fitting regime for each candidate p evaluation:

            - ``"fit"``: use ``fit()`` / ``fit_pirls`` (current default)
            - ``"reml"``: use ``fit_reml()`` for each candidate p
            - ``"inherit"``: use the last fitting method (from ``_last_fit_meta``)

        Returns
        -------
        TweedieProfileResult
        """
        from superglm.tweedie_profile import estimate_tweedie_p

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="estimate_p()"
        )

        # Resolve to internal method name: "fit" or "fit_reml"
        _VALID_FIT_MODES = {"fit", "reml", "inherit"}
        if fit_mode not in _VALID_FIT_MODES:
            raise ValueError(
                f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(_VALID_FIT_MODES)}"
            )
        if fit_mode == "reml":
            resolved_mode = "fit_reml"
        elif fit_mode == "inherit":
            if self._last_fit_meta is not None:
                resolved_mode = self._last_fit_meta["method"]
            else:
                resolved_mode = "fit"
        else:
            resolved_mode = "fit"

        result = estimate_tweedie_p(
            self, X, y, exposure=exposure, offset=offset, fit_mode=resolved_mode, **kwargs
        )
        self.tweedie_p = result.p_hat
        self._tweedie_profile_result = result

        # Refit with the same regime used for profiling
        if resolved_mode == "fit_reml":
            self.fit_reml(X, y, exposure=exposure, offset=offset)
        else:
            self.fit(X, y, exposure=exposure, offset=offset)
        return result

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
        """Estimate NB theta via profile likelihood, refit, and return result.

        Thin wrapper around :func:`superglm.nb_profile.estimate_nb_theta`.
        After estimation, sets ``self.nb_theta`` to the optimised value and
        refits the model.

        Returns
        -------
        NBProfileResult
        """
        from superglm.nb_profile import estimate_nb_theta

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="estimate_theta()"
        )

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
        *,
        sample_weight: NDArray | None = None,
    ) -> ModelMetrics:
        """Compute comprehensive diagnostics for the fitted model.

        Returns a ModelMetrics object with information criteria, residuals,
        leverage, Cook's distance, etc.
        """
        from superglm.metrics import ModelMetrics

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="metrics()"
        )

        return ModelMetrics(self, X, y, exposure, offset)

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
        exposure = self._resolve_sample_weight_alias(exposure, sample_weight, method_name="drop1()")
        return _drop1(self, X, y, exposure=exposure, offset=offset, test=test)

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
        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="refit_unpenalised()"
        )
        return _refit_unpenalised(
            self,
            X,
            y,
            exposure=exposure,
            offset=offset,
            keep_smoothing=keep_smoothing,
        )

    def relativities(self, with_se: bool = False) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features."""
        return _relativities(
            self._feature_order,
            self._interaction_order,
            self._specs,
            self._interaction_specs,
            self._groups,
            self.result,
            with_se=with_se,
            covariance_fn=(lambda: self._coef_covariance) if with_se else None,
        )

    def _feature_se_from_cov(
        self,
        name: str,
        Cov_active: NDArray,
        active_groups: list[GroupSlice],
        n_points: int = 200,
    ) -> NDArray:
        """Compute feature-level SEs from the precomputed covariance matrix."""
        return feature_se_from_cov(
            name,
            Cov_active,
            active_groups,
            self.result,
            self._groups,
            self._specs,
            self._interaction_specs,
            n_points=n_points,
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
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling simultaneous_bands().")
        return _simultaneous_bands(
            feature,
            result=self.result,
            groups=self._groups,
            specs=self._specs,
            covariance_fn=lambda: self._coef_covariance,
            alpha=alpha,
            n_sim=n_sim,
            n_points=n_points,
            seed=seed,
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
        """Per-term inference: curve, uncertainty, and metadata in one object.

        Unifies ``reconstruct_feature()``, ``relativities(with_se=True)``,
        and ``simultaneous_bands()`` into a single coherent result.

        Parameters
        ----------
        name : str
            Feature or interaction name.
        with_se : bool
            Compute standard errors and pointwise CIs (default True).
        simultaneous : bool
            Compute simultaneous confidence bands (spline only).
        n_points : int
            Grid size for spline/polynomial curves.
        alpha : float
            Significance level for CIs (default 0.05 → 95%).
        n_sim : int
            Number of simulations for simultaneous bands.
        seed : int
            Random seed for simultaneous bands.

        Returns
        -------
        TermInference or InteractionInference
        """
        if self._result is None:
            raise RuntimeError("Model must be fitted before calling term_inference().")
        return _term_inference(
            name,
            result=self.result,
            groups=self._groups,
            specs=self._specs,
            interaction_specs=self._interaction_specs,
            covariance_fn=lambda: self._coef_covariance,
            reml_lambdas=getattr(self, "_reml_lambdas", None),
            lambda2=self.lambda2,
            group_edf=self._group_edf,
            with_se=with_se,
            simultaneous=simultaneous,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
        )

    @cached_property
    def _group_edf(self) -> dict[str, float] | None:
        """Per-group effective degrees of freedom via F = (X'WX+S)^{-1} X'WX."""
        from superglm.metrics import _penalised_xtwx_inv

        if self._dm is None or self._result is None:
            return None

        beta = self._result.beta
        eta = self._dm.matvec(beta) + self._result.intercept
        if self._fit_offset is not None:
            eta = eta + self._fit_offset
        eta = np.clip(eta, -20, 20)
        mu = self._link.inverse(eta)
        V = self._distribution.variance(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        W = self._fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

        lam2 = getattr(self, "_reml_lambdas", None) or self.lambda2
        X_a, XtWX_S_inv, active_groups, _ = _penalised_xtwx_inv(
            beta, W, self._dm.group_matrices, self._groups, lam2
        )

        if X_a.shape[1] == 0:
            return {}

        XtWX = X_a.T @ (X_a * W[:, None])
        F = XtWX_S_inv @ XtWX
        edf_vec = np.diag(F)

        out: dict[str, float] = {}
        for ag in active_groups:
            out[ag.name] = float(np.sum(edf_vec[ag.sl]))
        return out

    def plot_relativity(
        self,
        name: str,
        *,
        X: pd.DataFrame | None = None,
        exposure: NDArray | None = None,
        sample_weight: NDArray | None = None,
        with_ci: bool = True,
        interval: str | None = "pointwise",
        show_exposure: bool = True,
        show_knots: bool = False,
        n_points: int = 200,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        subtitle: str | None = None,
    ):
        """Plot a single term's relativity curve or bar chart.

        This is the primary single-term plotting entry point.

        Parameters
        ----------
        name : str
            Feature name (must be in the fitted model).
        X : DataFrame, optional
            Training data for exposure overlays.
        exposure : array-like, optional
            Exposure / frequency weights.
        with_ci : bool
            When *False*, forces ``interval=None`` (no bands).
        interval : {"pointwise", "simultaneous", "both", None}
            Band style.  For categoricals/numerics, simultaneous/both
            fall back to pointwise.
        show_exposure : bool
            Show exposure distribution (density strip for continuous,
            vertical bars for categorical).
        show_knots : bool
            Show interior knot ticks (spline only).
        n_points : int
            Grid resolution for spline/polynomial curves.
        alpha : float
            Significance level for confidence intervals.
        n_sim : int
            Number of posterior simulations for simultaneous bands.
        seed : int
            Random seed for simultaneous band simulation.
        figsize : tuple, optional
            Figure size override.
        title, subtitle : str, optional
            Figure-level title and subtitle.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from superglm.plotting import plot_term

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="plot_relativity()"
        )

        if not with_ci:
            interval = None

        need_sim = interval in ("simultaneous", "both")

        ti = self.term_inference(
            name,
            with_se=(interval is not None),
            simultaneous=need_sim,
            n_points=n_points,
            alpha=alpha,
            n_sim=n_sim,
            seed=seed,
        )

        return plot_term(
            ti,
            X=X,
            exposure=exposure,
            interval=interval,
            show_exposure=show_exposure,
            show_knots=show_knots,
            figsize=figsize,
            title=title,
            subtitle=subtitle,
        )

    def plot_relativities(
        self,
        X: pd.DataFrame | None = None,
        exposure: NDArray | None = None,
        with_ci: bool = True,
        *,
        sample_weight: NDArray | None = None,
        interval: str | None = "pointwise",
        show_exposure: bool = True,
        show_knots: bool = False,
        n_points: int = 200,
        alpha: float = 0.05,
        n_sim: int = 10_000,
        seed: int = 42,
        title: str | None = None,
        subtitle: str | None = None,
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
            When *False*, forces ``interval=None`` (no bands).
        interval : {"pointwise", "simultaneous", "both", None}
            ``"pointwise"``: orange CI band.
            ``"simultaneous"``: blue simultaneous band.
            ``"both"``: nested (simultaneous outside, pointwise inside).
            ``None``: no bands.
        show_exposure : bool
            Show exposure density strip below spline panels (default *True*).
        show_knots : bool
            Show interior knot ticks on x-axis (default *False*).
        n_points : int
            Grid resolution for spline/polynomial curves.
        alpha : float
            Significance level for confidence intervals.
        n_sim : int
            Number of posterior simulations for simultaneous bands.
        seed : int
            Random seed for simultaneous band simulation.
        title, subtitle : str, optional
            Figure-level title and subtitle.
        **kwargs
            Forwarded to :func:`superglm.plotting.plot_relativities`.
        """
        from superglm.plotting import plot_relativities

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="plot_relativities()"
        )

        if not with_ci:
            interval = None

        need_sim = interval in ("simultaneous", "both")

        terms = []
        for name in self._feature_order:
            ti = self.term_inference(
                name,
                with_se=(interval is not None),
                simultaneous=need_sim,
                n_points=n_points,
                alpha=alpha,
                n_sim=n_sim,
                seed=seed,
            )
            terms.append(ti)

        return plot_relativities(
            terms,
            X=X,
            exposure=exposure,
            interval=interval,
            show_exposure=show_exposure,
            show_knots=show_knots,
            title=title,
            subtitle=subtitle,
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
        *,
        sample_weight: NDArray | None = None,
        **kwargs,
    ) -> DiscretizationResult:
        """Analyse the impact of discretizing spline/polynomial curves.

        See :func:`superglm.discretize.discretization_impact` for full docs.
        """
        from superglm.discretize import discretization_impact

        exposure = self._resolve_sample_weight_alias(
            exposure, sample_weight, method_name="discretization_impact()"
        )

        return discretization_impact(self, X, y, exposure, **kwargs)

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
