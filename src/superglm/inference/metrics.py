"""Comprehensive GLM diagnostics: information criteria, residuals, influence."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from superglm.distributions import weighted_log_likelihood
from superglm.inference._metrics_design import (
    EvaluationDesign,
    MetricsDesign,
    factor_from_gram,
    iter_dense_chunks,
    quadratic_form_diagonal,
    weighted_moments,
)
from superglm.inference.coef_tables import build_basis_detail, build_coef_rows  # noqa: F401
from superglm.inference.covariance import (  # noqa: F401
    _active_penalty_matrix,
    _second_diff_penalty,
    covariance_selected_block,
    covariance_selected_diagonal,
    covariance_slope_view,
)
from superglm.inference.summary import ModelSummary, _CoefRow
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.model.state_ops import (
    _public_augmented_covariance,
    _rank_active_state,
    _rank_augmented_covariance,
    _solver_space_working_weights,
)
from superglm.penalties.base import selection_shrunk_group_names
from superglm.profiling._reporting import (
    cached_tweedie_profile_ci,
    tweedie_profile_method_label,
)
from superglm.solvers.centered_system import penalty_factor
from superglm.solvers.rank import (
    decompose_factor,
    decompose_gram,
    decompose_gram_if_authoritative,
    diagonal_of_square,
    selected_group_name_set,
    streamed_weighted_factor,
)
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.model import SuperGLM


def _active_feature_columns(
    feature_name: str,
    groups: list[GroupSlice],
    active_subs: list[GroupSlice],
) -> NDArray:
    return np.concatenate(
        [
            np.arange(group.start, group.end) - groups[0].start
            for group in groups
            if any(
                active_group.feature_name == feature_name and active_group.name == group.name
                for active_group in active_subs
            )
        ]
    )


def _selected_group_state(
    result,
    groups: list[GroupSlice],
    *,
    penalty=None,
) -> tuple[NDArray, list[GroupSlice]]:
    """Return selected original columns and compact active-group slices."""
    selected_names = selected_group_name_set(result, groups, penalty=penalty)
    selected_columns: list[int] = []
    active_groups: list[GroupSlice] = []
    column = 0
    for group in groups:
        if group.name not in selected_names:
            continue
        selected_columns.extend(range(group.start, group.end))
        active_groups.append(
            GroupSlice(
                name=group.name,
                start=column,
                end=column + group.size,
                weight=group.weight,
                penalty_dim=group.penalty_dim,
                penalized=group.penalized,
                feature_name=group.feature_name,
                subgroup_type=group.subgroup_type,
                constraints=group.constraints,
                monotone_engine=group.monotone_engine,
                scop_reparameterization=group.scop_reparameterization,
            )
        )
        column += group.size
    selected_columns_array = np.asarray(selected_columns, dtype=np.intp)
    rank_info = getattr(result, "rank_info", None)
    if rank_info is not None and not np.array_equal(
        selected_columns_array,
        np.asarray(rank_info.selected_columns, dtype=np.intp),
    ):
        raise ValueError("rank metadata selected columns do not match active groups")
    return selected_columns_array, active_groups


def _grouped_active_design(model, active_groups: list[GroupSlice]):
    """Return the fitted grouped design restricted to compact active groups."""
    from superglm.group_matrix import DesignMatrix

    active_names = {group.name for group in active_groups}
    matrices = [
        matrix
        for matrix, group in zip(model._dm.group_matrices, model._groups, strict=True)
        if group.name in active_names
    ]
    return DesignMatrix(matrices, n=model._dm.n, p=sum(group.size for group in active_groups))


def _profiled_augmented_covariance(
    data_gram: NDArray,
    penalty: NDArray,
    xtw1: NDArray,
    sum_w: float,
    *,
    profile_rank=None,
) -> NDArray:
    """Invert after profiling the intercept, then map back to raw coordinates."""
    profiled_inverse = (
        profile_rank.pseudo_inverse()
        if profile_rank is not None
        else decompose_gram(data_gram + penalty).pseudo_inverse()
    )
    mean_x = xtw1 / sum_w
    intercept_cross = -(profiled_inverse @ mean_x)
    augmented = np.empty((len(xtw1) + 1, len(xtw1) + 1), dtype=np.float64)
    augmented[0, 0] = 1.0 / sum_w + float(mean_x @ profiled_inverse @ mean_x)
    augmented[0, 1:] = intercept_cross
    augmented[1:, 0] = intercept_cross
    augmented[1:, 1:] = profiled_inverse
    return augmented


def _certified_data_rank(
    design: MetricsDesign,
    W: NDArray,
    data_gram: NDArray,
    xtw1: NDArray,
):
    """Certify ambiguous centered data geometry with observation rows."""
    decomposition = decompose_gram_if_authoritative(data_gram)
    if decomposition is not None:
        return decomposition
    factor = streamed_weighted_factor(
        iter_dense_chunks(design),
        W,
        center=xtw1 / float(np.sum(W)),
    )
    return decompose_factor(factor)


def _certified_coefficient_rank(
    design: MetricsDesign,
    W: NDArray,
    raw_gram: NDArray,
    penalty: NDArray,
):
    """Certify ambiguous raw penalized geometry with observation rows."""
    decomposition = decompose_gram_if_authoritative(raw_gram + penalty)
    if decomposition is not None:
        return decomposition
    factor = streamed_weighted_factor(iter_dense_chunks(design), W)
    smooth_factor = penalty_factor(penalty)
    if smooth_factor.shape[0]:
        factor = np.vstack((factor, smooth_factor))
    return decompose_factor(factor)


def _certified_profile_rank(
    design: MetricsDesign,
    W: NDArray,
    data_gram: NDArray,
    xtw1: NDArray,
    penalty: NDArray,
    data_rank,
):
    """Return the certified centered Hessian decomposition for covariance."""
    if not np.any(penalty):
        return data_rank
    decomposition = decompose_gram_if_authoritative(data_gram + penalty)
    if decomposition is not None:
        return decomposition
    factor = streamed_weighted_factor(
        iter_dense_chunks(design),
        W,
        center=xtw1 / float(np.sum(W)),
    )
    smooth_factor = penalty_factor(penalty)
    if smooth_factor.shape[0]:
        factor = np.vstack((factor, smooth_factor))
    return decompose_factor(factor)


def _coefficient_estimability(
    data_rank,
    groups: list[GroupSlice],
    active_groups: list[GroupSlice],
    width: int,
) -> NDArray:
    """Map active-data rank estimability back to full fitted coordinates."""
    estimable = np.zeros(width, dtype=bool)
    original_by_name = {group.name: group for group in groups}
    active_estimable = data_rank.coefficient_estimable()
    for active_group in active_groups:
        original = original_by_name[active_group.name]
        estimable[original.sl] = active_estimable[active_group.sl]
    return estimable


def _requires_wood_inference(model, active_groups: list[GroupSlice]) -> bool:
    """Whether active summary rows contain a smooth term requiring Wood's test."""
    from superglm.features.interaction import SplineCategorical, TensorInteraction
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.spline import _SplineBase

    active_names = {group.name for group in active_groups}
    for group in model._groups:
        if group.name not in active_names:
            continue
        spec = model._specs.get(group.feature_name) or model._interaction_specs.get(
            group.feature_name
        )
        if isinstance(spec, _SplineBase) and group.subgroup_type != "linear":
            return True
        # A Piecewise/Polynomial inner basis is an unpenalized parametric
        # block: its test is a plain Wald chi-square, never Wood's smooth
        # test, so it must not trigger the smooth-inference machinery.
        if isinstance(spec, OrderedCategorical) and spec.basis_kind == "spline":
            return True
        if isinstance(spec, SplineCategorical | TensorInteraction):
            return True
    return False


class ModelMetrics:
    """Post-fit diagnostics for a SuperGLM model.

    Parameters
    ----------
    model : SuperGLM
        A fitted model.
    X : pandas or eager Polars DataFrame
        Feature matrix used for fitting (or evaluation).
    y : array-like
        Response variable.
    sample_weight : array-like, optional
        Observation weights / sample_weight.
    offset : array-like, optional
        Offset term.
    """

    def __init__(
        self,
        model: SuperGLM,
        X=None,
        y=None,
        sample_weight=None,
        offset=None,
        *,
        _fit_data_matches: bool | None = None,
        _mu: NDArray | None = None,
        _null_mu: NDArray | None = None,
        _fit_stats=None,
    ):
        self._model = model
        self._family = model._distribution
        self._link = model._link
        self._groups = model._groups
        self._dm = model._dm
        self._result = model.result
        fit_X = getattr(model, "_fit_X_ref", None)
        self._X = fit_X if X is None else X
        same_fit_object = X is None or X is fit_X
        if _fit_data_matches is None:
            fit_data_guard = getattr(model, "_fit_data_guard", None)
            if fit_data_guard is None:
                _fit_data_matches = same_fit_object
            else:
                _fit_data_matches = bool(
                    same_fit_object
                    and fit_data_guard.matches(
                        self._X,
                        y,
                        sample_weight,
                        offset,
                        fit_weights=getattr(model, "_fit_weights", None),
                        fit_offset=getattr(model, "_fit_offset", None),
                    )
                )
        self._uses_fit_rows = bool(same_fit_object and _fit_data_matches)

        from superglm.solvers.dispersion import model_weight_semantics

        self._weight_semantics = model_weight_semantics(model)
        self._y = np.asarray(y, dtype=np.float64)
        n = len(self._y)
        if sample_weight is None:
            self._weights = np.ones(n)
        else:
            from superglm.distributions import Tweedie

            if isinstance(self._family, Tweedie) and self._weight_semantics == "prior":
                from superglm._utils import _validate_strict_prior_weights

                self._weights = _validate_strict_prior_weights(sample_weight, n)
            else:
                from superglm.model.input_validation import _finite_vector

                self._weights = _finite_vector("sample_weight", sample_weight, n)
                if np.any(self._weights < 0.0):
                    raise ValueError("sample_weight must be nonnegative")
                if not np.any(self._weights > 0.0):
                    raise ValueError("sample_weight must not be all zero")
            # Evaluation takes its OWN rows and weights, so the fit-time
            # lattice check does not cover it: a clean fitted model can be
            # scored on holdout rows whose `w * y` is not integral, and both
            # the reported log-likelihood and the randomized quantile
            # residuals would then be silently fabricated -- the residual
            # rounds the impossible count onto a neighbouring lattice point
            # and inverts the CDF there. Re-check at this boundary.
        self._offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
        fit_offset = getattr(model, "_fit_offset", None)
        fit_offset_array = np.zeros(n) if fit_offset is None else np.asarray(fit_offset)
        self._uses_fit_design = bool(
            self._uses_fit_rows
            and self._offset.shape == fit_offset_array.shape
            and np.array_equal(self._offset, fit_offset_array)
        )
        fit_weights = getattr(model, "_fit_weights", None)
        released_geometry_guard = getattr(model, "_fit_geometry_guard", None)
        if self._dm is None and released_geometry_guard is not None:
            self._fit_geometry_matches = released_geometry_guard.matches(
                self._X,
                self._weights,
                self._offset,
            )
        else:
            self._fit_geometry_matches = bool(
                self._uses_fit_design
                and fit_weights is not None
                and np.shape(fit_weights) == np.shape(self._weights)
                and np.array_equal(np.asarray(fit_weights), self._weights)
            )
        self._uses_compact_fit_inference = bool(
            (
                self._dm is None
                and "_fit_inference_info" in model.__dict__
                and self._fit_geometry_matches
            )
            or (
                getattr(model, "_linear_system_state", None) is not None
                and self._fit_geometry_matches
            )
        )

        if _mu is not None:
            self._mu = _mu
        else:
            self._mu = model.predict(self._X, offset=offset)

        # Evaluation takes its OWN rows and weights, so the fit-time check does
        # not cover it: a clean fitted model can be scored on holdout rows whose
        # `w * y` is not integral, and both the reported log-likelihood and the
        # randomized quantile residuals would then be silently fabricated -- the
        # residual rounds the impossible count onto a neighbouring lattice point
        # and inverts the CDF there.
        #
        # Not inside a `sample_weight is not None` branch: synthesized all-ones
        # weights still define a counting lattice, so a fractional response on
        # an unweighted holdout is just as off it.
        #
        # After predict(), not before: predict is what validates the evaluation
        # frame and the offset, and this warning describes a likelihood about to
        # be computed. Ahead of that validation it fires on input that will never
        # reach one, and under `-W error` it surfaces instead of the frame or
        # offset error the caller needs to see.
        from superglm.model.input_validation import check_weight_contract

        check_weight_contract(
            self._y,
            self._weights,
            self._family,
            self._weight_semantics,
        )
        if _null_mu is not None:
            self.__dict__["_null_mu"] = _null_mu
        if _fit_stats is not None:
            self.__dict__["log_likelihood"] = _fit_stats.log_likelihood
            self.__dict__["null_log_likelihood"] = _fit_stats.null_log_likelihood
            self.__dict__["null_deviance"] = _fit_stats.null_deviance
            self.__dict__["explained_deviance"] = _fit_stats.explained_deviance
            self.__dict__["pearson_chi2"] = _fit_stats.pearson_chi2

    def _build_S_from_penalties(self, lam2) -> NDArray | None:
        """Build full penalty matrix from model._reml_penalties if available."""
        penalties = getattr(self._model, "_reml_penalties", None)
        if penalties is None:
            return None
        from superglm.reml.penalty_algebra import build_penalty_matrix

        return build_penalty_matrix(
            self._dm.group_matrices,
            self._groups,
            lam2,
            self._dm.p,
            reml_penalties=penalties,
        )

    @cached_property
    def _fit_working_weights(self) -> NDArray:
        """Working weights represented by the retained fitted-rank state."""
        return _solver_space_working_weights(self._model)

    def _working_weights_match_fit(self, working_weights: NDArray) -> bool:
        """Whether fitted rank/covariance factors are valid for this evaluation."""
        rank_info = getattr(self._result, "rank_info", None)
        return bool(
            rank_info is not None
            and self._fit_geometry_matches
            and np.shape(getattr(self._model, "_fit_weights", None)) == np.shape(working_weights)
        )

    @cached_property
    def _working_eta_mu(self) -> tuple[NDArray, NDArray]:
        """Unclipped-link eta and guarded mu for this diagnostic evaluation."""
        from superglm.distributions import clip_mu
        from superglm.links import stabilize_eta

        if self._uses_fit_design:
            solver = self._model._solver_pirls_result()
            eta = self._dm.matvec(solver.beta) + solver.intercept + self._offset
            eta = stabilize_eta(eta, self._link)
        else:
            from superglm.model import base

            eta = base.predict_eta_exact(self._model, self._X, offset=self._offset)
        mu = clip_mu(self._link.inverse(eta), self._family)
        return eta, mu

    # ── Scalar properties ─────────────────────────────────────────

    @property
    def n_obs(self) -> int:
        return len(self._y)

    @property
    def effective_df(self) -> float:
        return self._result.effective_df

    @property
    def phi(self) -> float:
        return self._result.phi

    @cached_property
    def deviance(self) -> float:
        """Weighted residual deviance on this object's evaluation rows."""
        return float(np.sum(self._weights * self._family.deviance_unit(self._y, self._mu)))

    @cached_property
    def log_likelihood(self) -> float:
        return weighted_log_likelihood(
            self._family,
            self._y,
            self._mu,
            self._weights,
            self.phi,
            weight_semantics=self._weight_semantics,
        )

    @cached_property
    def _null_mu(self) -> NDArray:
        """Authoritative offset-aware intercept-only null prediction."""
        from superglm.model.fit_ops import _compute_null_mu

        return _compute_null_mu(
            self._y,
            self._weights,
            self._offset,
            self._family,
            self._link,
            weight_semantics=self._weight_semantics,
        )

    @cached_property
    def null_log_likelihood(self) -> float:
        """Log-likelihood at the intercept-only (null) model."""
        return weighted_log_likelihood(
            self._family,
            self._y,
            self._null_mu,
            self._weights,
            self.phi,
            weight_semantics=self._weight_semantics,
        )

    @cached_property
    def null_deviance(self) -> float:
        return float(np.sum(self._weights * self._family.deviance_unit(self._y, self._null_mu)))

    @cached_property
    def explained_deviance(self) -> float:
        """1 - deviance / null_deviance. Analogous to R-squared."""
        from superglm._utils import _explained_deviance

        return _explained_deviance(
            self.deviance,
            self.null_deviance,
            self._y,
            self._null_mu,
            self._weights,
        )

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.effective_df

    @cached_property
    def _likelihood_size(self) -> float:
        """Family-specific likelihood size used by finite-sample criteria."""
        from superglm.solvers.dispersion import dispersion_likelihood_size

        return dispersion_likelihood_size(
            self._weights,
            weight_semantics=self._weight_semantics,
        )

    @property
    def bic(self) -> float:
        return -2.0 * self.log_likelihood + np.log(self._likelihood_size) * self.effective_df

    @property
    def aicc(self) -> float:
        edf = self.effective_df
        n = self._likelihood_size
        denom = n - edf - 1.0
        if denom <= 0:
            return np.inf
        return self.aic + 2.0 * edf * (edf + 1.0) / denom

    def ebic(self, gamma: float = 0.5) -> float:
        """Extended BIC (Chen & Chen 2008)."""
        p_total = len(self._groups)
        n_active = self.n_active_groups
        return self.bic + 2.0 * gamma * (
            gammaln(p_total + 1) - gammaln(n_active + 1) - gammaln(p_total - n_active + 1)
        )

    @cached_property
    def pearson_chi2(self) -> float:
        V = self._family.variance(self._mu)
        return float(np.sum(self._weights * (self._y - self._mu) ** 2 / V))

    @cached_property
    def n_active_groups(self) -> int:
        return len(
            selected_group_name_set(
                self._result,
                self._groups,
                penalty=fitted_penalty(self._model),
            )
        )

    @cached_property
    def eta(self) -> NDArray:
        """Linear predictor (link-scale fitted values)."""
        return self._working_eta_mu[0]

    # ── Residuals ─────────────────────────────────────────────────

    def residuals(self, kind: str = "deviance", *, seed: int | None = 42) -> NDArray:
        """Compute residuals of the specified type.

        Deviance and Pearson residuals use a compressed-row aggregate
        representation: a weighted row is multiplied by ``sqrt(sample_weight)``
        so its squared residual equals the sum of the squared residuals from
        literal replicated rows. Response, working, and quantile residuals
        retain per-row semantics and are not multiplied by the weight. Under
        ``weight_semantics="prior"`` the weight is part of the row's own
        distribution and therefore stays in every diagnostic that depends on
        it.
        Influence diagnostics combine the aggregate residual with compressed
        leverage and therefore describe deletion of the whole weighted cell,
        not deletion of one literal expanded copy.

        Parameters
        ----------
        kind : str
            One of "deviance", "pearson", "response", "working", "quantile".
        seed : int or None
            Random seed for quantile residuals (discrete families only).
            Default 42 for reproducibility. Ignored for non-quantile types.
        """
        y, mu, w = self._y, self._mu, self._weights
        family = self._family

        if kind == "deviance":
            d = family.deviance_unit(y, mu)
            return np.sign(y - mu) * np.sqrt(w * d)

        if kind == "pearson":
            V = family.variance(mu)
            return np.sqrt(w) * (y - mu) / np.sqrt(V)

        if kind == "response":
            return y - mu

        if kind == "working":
            dmu_deta = self._link.deriv_inverse(self.eta)
            return (y - mu) / dmu_deta

        if kind == "quantile":
            return self._quantile_residuals(seed=seed)

        raise ValueError(
            f"Unknown residual type '{kind}'. "
            "Use 'deviance', 'pearson', 'response', 'working', or 'quantile'."
        )

    def _quantile_residuals(self, seed: int | None = 42) -> NDArray:
        """Randomized quantile residuals (Dunn & Smyth 1996).

        A replication weight does not change an observation's family CDF:
        integer weights represent repeated copies of that same response. A
        prior weight does change the per-row distribution -- through
        ``phi / w``, or the equivalent scaling of the family's own parameters
        -- and therefore stays in the CDF construction.

        For discrete families (Poisson, NB2, Binomial), uses jittered
        uniform on the CDF interval [F(y-1), F(y)]. For continuous
        families (Gamma, Gaussian), uses the CDF directly.

        Parameters
        ----------
        seed : int or None
            Random seed for the jitter in discrete families. Default 42
            for reproducibility. Pass None for non-deterministic.
        """
        from scipy.stats import gamma as gamma_dist
        from scipy.stats import nbinom, norm, poisson

        from superglm.distributions import (
            Binomial,
            Gamma,
            Gaussian,
            NegativeBinomial,
            Poisson,
            Tweedie,
        )

        y, mu, w = self._y, self._mu, self._weights
        rng = np.random.default_rng(seed)
        # A randomized quantile residual is a probability-integral transform, so
        # it needs the row's own marginal distribution.  A replicated row has the
        # unit-weight marginal -- copies of a draw are draws -- so the frequency
        # contract evaluates every CDF below at w = 1.  A prior weight is part of
        # the row's distribution: it scales the Gaussian's variance, the Gamma's
        # shape, the Poisson's and negative binomial's lattice, and the
        # binomial's trial count, and the transform is wrong without it.
        prior = self._weight_semantics == "prior"
        if isinstance(self._family, Binomial):
            # Binomial is deliberately CONTRACT-INVARIANT here, and it is the
            # one family that has to be.
            #
            # A randomized quantile residual is only a probability-integral
            # transform if the masses it splits sum to one. The prior
            # construction `w Y ~ Binomial(w, mu)` lives on {0, 1/w, ..., 1},
            # but `validate_response` pins y to {0, 1} -- so the reported
            # likelihood's two reachable masses are (1-mu)**w and mu**w, which
            # sum to one ONLY at w == 1 (0.378 at mu=0.71, w=3, measured).
            # There is therefore no {0,1}-supported distribution whose masses
            # are the reported likelihood's, and no PIT of it exists to build.
            #
            # Splitting [0, 1] at 1 - mu instead keeps the transform exactly
            # uniform, which is the property the Q-Q envelope is a reference
            # band for. What it costs is that the residual inverts the
            # unit-weight marginal rather than the reported likelihood -- said
            # here rather than left for a reader to discover, and already
            # warned about at fit time by PriorWeightLatticeWarning.
            a = np.where(y == 0, 0.0, 1.0 - mu)
            b = np.where(y == 0, 1.0 - mu, 1.0)
            u = rng.uniform(a, b)
        elif isinstance(self._family, Poisson):
            count = np.round(w * y) if prior else np.round(y)
            lam = w * mu if prior else mu
            a = poisson.cdf(count - 1, lam)
            b = poisson.cdf(count, lam)
            u = rng.uniform(a, b)
        elif isinstance(self._family, NegativeBinomial):
            theta = self._family.theta
            p_nb = theta / (mu + theta)
            count = np.round(w * y) if prior else np.round(y)
            n_param = w * theta if prior else theta
            a = nbinom.cdf(count - 1, n=n_param, p=p_nb)
            b = nbinom.cdf(count, n=n_param, p=p_nb)
            u = rng.uniform(a, b)
        elif isinstance(self._family, Gamma):
            shape = (w if prior else 1.0) / self.phi
            scale = mu * self.phi / (w if prior else 1.0)
            u = gamma_dist.cdf(y, a=shape, scale=scale)
        elif isinstance(self._family, Gaussian):
            variance = self.phi / w if prior else np.full_like(y, self.phi)
            u = norm.cdf(y, loc=mu, scale=np.sqrt(variance))
        elif isinstance(self._family, Tweedie):
            # Tweedie p in (1,2): compound Poisson-Gamma.
            # With weights: lambda and scale both depend on w.
            p_tw = self._family.p
            phi = self.phi

            # Weight-adjusted Poisson rate and compound Gamma parameters.  A
            # replicated row's marginal is the unit-weight one, so the
            # frequency contract evaluates the compound distribution at w = 1.
            w_eff = w if prior else np.ones_like(w)
            lam = w_eff * np.power(mu, 2 - p_tw) / ((2 - p_tw) * phi)
            p_zero = np.exp(-lam)
            alpha_tw = (2 - p_tw) / (p_tw - 1)  # Gamma shape per claim
            scale_tw = phi * (p_tw - 1) * np.power(mu, p_tw - 1) / w_eff

            u = np.empty_like(y)

            # y = 0: jitter in [0, P(Y=0)]
            zero_mask = y == 0
            if np.any(zero_mask):
                u[zero_mask] = rng.uniform(0.0, p_zero[zero_mask])

            # y > 0: F(y) = P(Y=0) + sum_k P(N=k) * Gamma_CDF(y; k*alpha, scale)
            pos_mask = ~zero_mask
            if np.any(pos_mask):
                y_p = y[pos_mask]
                lam_p = lam[pos_mask]
                p_zero_p = p_zero[pos_mask]
                alpha_p = alpha_tw  # scalar
                scale_p = scale_tw[pos_mask]

                # Truncate Poisson sum where tail prob < 1e-12
                lam_max = float(np.max(lam_p))
                k_max = max(int(lam_max + 6 * np.sqrt(max(lam_max, 1))) + 1, 5)

                cdf_vals = p_zero_p.copy()
                for k in range(1, k_max + 1):
                    pk = poisson.pmf(k, lam_p)
                    gk = gamma_dist.cdf(y_p, a=k * alpha_p, scale=scale_p)
                    cdf_vals += pk * gk

                cdf_vals = np.clip(cdf_vals, p_zero_p + 1e-10, 1.0 - 1e-10)
                u[pos_mask] = cdf_vals
        else:
            raise NotImplementedError(
                f"Quantile residuals not implemented for {type(self._family).__name__}."
            )

        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        return norm.ppf(u)

    # ── Influence diagnostics (lazy) ──────────────────────────────

    @cached_property
    def _active_info(
        self,
    ) -> tuple[MetricsDesign, NDArray, NDArray, NDArray, list[GroupSlice]]:
        """Shared computation for leverage and SEs.

        Returns (X_a, W, XtWX_inv, XtWX_inv_aug, active_groups) where:
        - X_a: grouped/chunked active design (legacy fits may return a dense array)
        - W: (n,) working weights
        - XtWX_inv: (p_active, p_active) = (X'WX + S)^{-1}, unscaled by phi
        - XtWX_inv_aug: (p_active+1, p_active+1) augmented inverse incl. intercept
        - active_groups: list of GroupSlice for active groups (re-indexed to X_a columns)
        """
        if self._dm is None and not self._uses_compact_fit_inference:
            raise RuntimeError(
                "retain_fit_state=False discarded the fitted design and the requested "
                "inference geometry does not match the fit geometry; refit with "
                "retain_fit_state=True for diagnostics on alternate rows, weights, or offsets"
            )
        beta = self._result.beta
        rank_info = getattr(self._result, "rank_info", None)
        if self._fit_geometry_matches and self._dm is not None:
            W = self._fit_working_weights
        else:
            eta, mu = self._working_eta_mu
            V = self._family.variance(mu)
            dmu_deta = self._link.deriv_inverse(eta)
            from superglm.distributions import _VARIANCE_FLOOR

            W = self._weights * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

        uses_fitted_rank = self._working_weights_match_fit(W)
        scop_inference = getattr(self._result, "scop_inference", None)
        if scop_inference is not None or self._uses_compact_fit_inference:
            fit_inference = self._model._fit_inference_info
            inverse = fit_inference["XtWX_inv"]
            augmented = fit_inference["XtWX_inv_aug"]
            active_groups = fit_inference["active_groups"]
            if self._uses_fit_design:
                fit_X_a, _, _, _, _ = self._model._fit_active_info
                X_a = fit_X_a
            else:
                rank_info = getattr(self._result, "rank_info", None)
                if rank_info is not None:
                    selected_columns = np.asarray(rank_info.selected_columns, dtype=np.intp)
                    self.__dict__["_coefficient_estimable"] = rank_info.coefficient_estimable()
                else:
                    selected_columns, _ = _selected_group_state(
                        self._result,
                        self._groups,
                        penalty=fitted_penalty(self._model),
                    )
                    self.__dict__["_coefficient_estimable"] = np.ones(len(beta), dtype=bool)
                X_a = EvaluationDesign(self._model, self._X, selected_columns)
            compact_estimable = fit_inference.get("coefficient_estimable")
            if compact_estimable is not None:
                self.__dict__["_coefficient_estimable"] = compact_estimable
            elif "_coefficient_estimable" not in self.__dict__:
                if rank_info is not None:
                    self.__dict__["_coefficient_estimable"] = rank_info.coefficient_estimable()
                else:
                    self.__dict__["_coefficient_estimable"] = np.ones(len(beta), dtype=bool)
            return X_a, W, inverse, augmented, active_groups

        if not self._uses_fit_design:
            selected_columns, active_groups = _selected_group_state(
                self._result,
                self._groups,
                penalty=fitted_penalty(self._model),
            )
            X_a = EvaluationDesign(self._model, self._X, selected_columns)
            lam2 = fitted_lambda2(self._model)
            S_active = _active_penalty_matrix(
                self._model._dm.group_matrices,
                self._groups,
                active_groups,
                lam2,
                reml_penalties=getattr(self._model, "_reml_penalties", None),
            )
            XtWX, XtW1, centered_data_gram = weighted_moments(X_a, W)
            self.__dict__["_active_centered_data_gram"] = centered_data_gram
            data_rank = _certified_data_rank(X_a, W, centered_data_gram, XtW1)
            self.__dict__["_coefficient_estimable"] = _coefficient_estimability(
                data_rank,
                self._groups,
                active_groups,
                len(beta),
            )
            inverse = _certified_coefficient_rank(X_a, W, XtWX, S_active).pseudo_inverse()
            profile_rank = _certified_profile_rank(
                X_a,
                W,
                centered_data_gram,
                XtW1,
                S_active,
                data_rank,
            )

            augmented = _profiled_augmented_covariance(
                centered_data_gram,
                S_active,
                XtW1,
                float(np.sum(W)),
                profile_rank=profile_rank,
            )
            return X_a, W, inverse, augmented, active_groups

        if uses_fitted_rank:
            X_a, active_groups = _rank_active_state(self._model, rank_info)
            inverse = rank_info.coefficient.pseudo_inverse()
            augmented = _rank_augmented_covariance(self._model, rank_info, active_groups)
            self.__dict__["_coefficient_estimable"] = rank_info.coefficient_estimable()
            return X_a, W, inverse, augmented, active_groups

        _, active_groups = _selected_group_state(
            self._result,
            self._groups,
            penalty=fitted_penalty(self._model),
        )
        X_a = _grouped_active_design(self._model, active_groups)
        lam2 = fitted_lambda2(self._model)
        S_active = _active_penalty_matrix(
            self._dm.group_matrices,
            self._groups,
            active_groups,
            lam2,
            reml_penalties=getattr(self._model, "_reml_penalties", None),
        )
        XtWX, XtW1, centered_data_gram = weighted_moments(X_a, W)
        inverse = _certified_coefficient_rank(X_a, W, XtWX, S_active).pseudo_inverse()
        self.__dict__["_active_centered_data_gram"] = centered_data_gram
        data_rank = _certified_data_rank(X_a, W, centered_data_gram, XtW1)
        self.__dict__["_coefficient_estimable"] = _coefficient_estimability(
            data_rank,
            self._groups,
            active_groups,
            len(beta),
        )
        profile_rank = _certified_profile_rank(
            X_a,
            W,
            centered_data_gram,
            XtW1,
            S_active,
            data_rank,
        )
        augmented = _profiled_augmented_covariance(
            centered_data_gram,
            S_active,
            XtW1,
            float(np.sum(W)),
            profile_rank=profile_rank,
        )
        augmented = _public_augmented_covariance(self._model, augmented, active_groups)
        return X_a, W, inverse, augmented, active_groups

    @cached_property
    def _active_design_moments(self) -> tuple[NDArray, NDArray, NDArray]:
        """Raw and centered moments shared by R and influence calculations."""
        X_a, W, _, _, _ = self._active_info
        return weighted_moments(X_a, W)

    @cached_property
    def _active_centered_data_gram(self) -> NDArray:
        """Centered Gram retained without pinning a duplicate raw p-by-p matrix."""
        X_a, W, _, _, _ = self._active_info
        return weighted_moments(X_a, W)[2]

    @cached_property
    def _active_R_factor(self) -> NDArray:
        """Upper-triangular factor used by Wood-style smooth tests.

        The smooth-term test operates on the relevant columns of the
        weighted, intercept-profiled design factor rather than the raw
        ``n x p_g`` design block. For active design ``X_a`` and working
        weights ``W``, ``R.T @ R`` is the centered data Gram after profiling
        the intercept. The Wood test therefore operates on columns of this
        factor, not on the raw design or an augmented ``[X; sqrt(S)]`` system.
        """
        X_a, W, _, _, _ = self._active_info
        if X_a.shape[1] == 0:
            return np.empty((0, 0))

        if (
            getattr(self._result, "scop_inference", None) is not None
            or self._uses_compact_fit_inference
        ):
            return self._model._fit_inference_info["R_a"]

        if self._working_weights_match_fit(W):
            return self._model._fit_inference_info["R_a"]

        data_gram = self._active_centered_data_gram
        return factor_from_gram(data_gram)

    @cached_property
    def _influence_edf(self) -> tuple[NDArray, NDArray]:
        """Per-coefficient edf and edf1 from influence matrix F.

        edf = diag(F) where F = (X'WX + S)^{-1} X'WX
        edf1 = 2*edf - diag(F @ F)  (Wood's alternative EDF)
        """
        X_a, W, _, _, _ = self._active_info

        if X_a.shape[1] == 0:
            return np.array([]), np.array([])

        if (
            getattr(self._result, "scop_inference", None) is not None
            or self._uses_compact_fit_inference
        ):
            inference = self._model._fit_inference_info
            return inference["edf"], inference["edf1"]

        if self._working_weights_match_fit(W):
            inference = self._model._fit_inference_info
            return inference["edf"], inference["edf1"]

        data_gram = self._active_centered_data_gram
        profile_inverse = self._active_info[3][1:, 1:]
        F = profile_inverse @ data_gram
        edf = np.diag(F)
        edf1 = 2.0 * edf - diagonal_of_square(F)
        return edf, edf1

    @property
    def _known_scale(self) -> bool:
        """Whether the fitted family defines rather than estimates dispersion."""
        return bool(getattr(self._family, "scale_known", False))

    @cached_property
    def _hat_diag(self) -> NDArray:
        """Hat matrix diagonal h_i via active-column inversion."""
        X_a, W, XtWX_inv, _, _ = self._active_info

        if X_a.shape[1] == 0:
            return np.zeros(self.n_obs)

        # h_i = W_i * x_i' XtWX_inv x_i = W * rowsum((X_a @ XtWX_inv) * X_a)
        if hasattr(X_a, "row_subset") or isinstance(X_a, EvaluationDesign):
            h = W * quadratic_form_diagonal(X_a, XtWX_inv)
        else:
            Q = X_a @ XtWX_inv
            h = W * np.sum(Q * X_a, axis=1)
        return np.clip(h, 0.0, 1.0)

    @property
    def leverage(self) -> NDArray:
        """Compressed-row hat diagonal under the supplied evaluation weights.

        This is the influence of one weighted cell as represented in the
        compressed design, not the leverage of each literal expanded copy.
        ``sum(h)`` is approximately effective_df - 1 (excluding the intercept).
        """
        return self._hat_diag

    @cached_property
    def cooks_distance(self) -> NDArray:
        """Compressed-row aggregate Cook's distance.

        Both the Pearson residual contribution and leverage correspond to
        deletion of the supplied weighted cell, rather than deletion of one
        literal expanded copy.
        """
        h = self._hat_diag
        r_p = self.residuals("pearson")
        p = self.effective_df
        phi = self.phi
        denom = (1.0 - h) ** 2 * p * phi
        denom = np.where(denom > 0, denom, np.inf)
        return r_p**2 * h / denom

    @cached_property
    def std_deviance_residuals(self) -> NDArray:
        """Standardized compressed-row deviance residuals.

        Equal to ``r_dev / sqrt(phi * (1 - h))`` using aggregate weighted
        residual and compressed-row leverage, so it represents deletion of the
        whole weighted cell rather than one literal expanded copy.
        """
        h = self._hat_diag
        r = self.residuals("deviance")
        scale = np.sqrt(self.phi * np.maximum(1.0 - h, 1e-10))
        return r / scale

    @cached_property
    def std_pearson_residuals(self) -> NDArray:
        """Standardized compressed-row Pearson residuals.

        Equal to ``r_pear / sqrt(phi * (1 - h))`` using aggregate weighted
        residual and compressed-row leverage, so it represents deletion of the
        whole weighted cell rather than one literal expanded copy.
        """
        h = self._hat_diag
        r = self.residuals("pearson")
        scale = np.sqrt(self.phi * np.maximum(1.0 - h, 1e-10))
        return r / scale

    # ── Coefficient standard errors ──────────────────────────────

    @cached_property
    def _current_coefficient_estimable(self) -> NDArray:
        """Coefficient estimability under this diagnostic design and weights."""
        _ = self._active_info
        cached = self.__dict__.get("_coefficient_estimable")
        if cached is not None:
            return cached
        rank_info = getattr(self._result, "rank_info", None)
        if rank_info is not None:
            return rank_info.coefficient_estimable()
        return np.ones(len(self._result.beta), dtype=bool)

    @cached_property
    def _coefficient_dispersion(self) -> float:
        """Dispersion used by the explicitly quasi-likelihood SE accessor."""
        if not self._known_scale:
            return max(float(self.phi), 0.0)
        from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

        residual_df = pearson_residual_degrees_of_freedom(
            self._weights,
            self.effective_df,
            weight_semantics=self._weight_semantics,
        )
        return max(float(self.pearson_chi2) / residual_df, 0.0)

    @cached_property
    def coefficient_se(self) -> dict[str, NDArray]:
        """Per-group quasi-likelihood coefficient standard errors.

        For known-scale families such as Poisson and Binomial, the covariance
        is scaled by the Pearson dispersion estimate
        ``pearson_chi2 / residual_df``, where ``residual_df`` follows the
        family's sample-weight semantics. For estimated-scale families such
        as Gaussian, Gamma, and Tweedie, it uses the fitted dispersion
        ``phi``. Use :attr:`coefficient_se_raw` for the unscaled,
        exact-family covariance.

        This accessor is an explicit quasi-likelihood diagnostic. Rendered
        coefficient tables retain the fitted-family scale (and hence
        ``phi=1`` for known-scale families) unless a quasi-summary option is
        requested by a future API.

        Inactive groups get all-zero SEs.

        Note: These are conditional-on-the-selected-model SEs from the
        penalized estimate. They do not account for model selection
        uncertainty (same convention as glmnet / mgcv).
        """
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        dispersion = self._coefficient_dispersion
        selected_names = selected_group_name_set(
            self._result,
            self._groups,
            penalty=fitted_penalty(self._model),
        )

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if g.name not in selected_names:
                result[g.name] = np.zeros(g.size)
            else:
                # Find corresponding active group
                ag = next(ag for ag in active_groups if ag.name == g.name)
                augmented_indices = np.arange(
                    1 + ag.start,
                    1 + ag.end,
                    dtype=np.intp,
                )
                var_diag = dispersion * covariance_selected_diagonal(
                    XtWX_inv_aug,
                    augmented_indices,
                )
                values = np.sqrt(np.maximum(var_diag, 0.0))
                values[~self._current_coefficient_estimable[g.sl]] = np.nan
                result[g.name] = values
        return result

    @cached_property
    def coefficient_se_raw(self) -> dict[str, NDArray]:
        """Per-group coefficient standard errors assuming phi=1.

        For known-scale families these are the exact-family standard errors,
        without the Pearson quasi-likelihood correction applied by
        :attr:`coefficient_se`. For estimated-scale families these omit the
        fitted dispersion and therefore differ from :attr:`coefficient_se`
        whenever ``phi != 1``.

        Inactive groups get all-zero SEs.
        """
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        selected_names = selected_group_name_set(
            self._result,
            self._groups,
            penalty=fitted_penalty(self._model),
        )

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if g.name not in selected_names:
                result[g.name] = np.zeros(g.size)
            else:
                ag = next(ag for ag in active_groups if ag.name == g.name)
                augmented_indices = np.arange(
                    1 + ag.start,
                    1 + ag.end,
                    dtype=np.intp,
                )
                var_diag = covariance_selected_diagonal(
                    XtWX_inv_aug,
                    augmented_indices,
                )
                values = np.sqrt(np.maximum(var_diag, 0.0))
                values[~self._current_coefficient_estimable[g.sl]] = np.nan
                result[g.name] = values
        return result

    @cached_property
    def intercept_se(self) -> float:
        """Quasi-likelihood standard error of the intercept.

        Computed from the [0,0] element of the augmented Fisher information
        inverse, which accounts for covariance between the intercept and
        all other coefficients. Uses the same dispersion convention as
        :attr:`coefficient_se`, including Pearson dispersion for known-scale
        families. Rendered coefficient tables retain fitted-family scale.
        """
        _, _, _, XtWX_inv_aug, _ = self._active_info
        icpt_var = float(XtWX_inv_aug[0, 0])
        if icpt_var <= 0:
            return 0.0
        return float(np.sqrt(self._coefficient_dispersion * icpt_var))

    @cached_property
    def intercept_se_raw(self) -> float:
        """Exact-family intercept SE for known scale, otherwise assuming phi=1."""
        _, _, _, XtWX_inv_aug, _ = self._active_info
        icpt_var = float(XtWX_inv_aug[0, 0])
        if icpt_var <= 0:
            return 0.0
        return float(np.sqrt(icpt_var))

    def _feature_se_impl(
        self,
        name: str,
        n_points: int = 200,
        *,
        phi_scale: bool = True,
    ) -> dict[str, Any]:
        """SE of the log-relativity curve/levels for a feature.

        Propagates the covariance of the fitted coefficients through the
        feature's design matrix to produce SEs on the interpretable scale.

        For splines: returns ``{x, se_log_relativity}`` on a grid.
        For categoricals: returns ``{levels, se_log_relativity}`` per level.
        For numerics: returns ``{se_coef}``.

        Uses the same quasi-likelihood dispersion as
        :attr:`coefficient_se` when ``phi_scale=True``.

        Parameters
        ----------
        name : str
            Feature name (e.g. "DrivAge"). For select=True splines with multiple
            subgroups, all subgroups are gathered automatically.
        """
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.ordered_categorical import OrderedCategorical
        from superglm.features.piecewise import Piecewise
        from superglm.features.spline import _SplineBase
        from superglm.inference._term_covariance import feature_se_from_cov

        groups = self._model._feature_groups(name)
        spec = self._model._specs[name]

        # Inactive feature: return zeros (all subgroups zeroed)
        selected_names = selected_group_name_set(
            self._result,
            self._groups,
            penalty=fitted_penalty(self._model),
        )
        if isinstance(spec, OrderedCategorical) and spec.basis == "spline":
            _, _, _, XtWX_inv_aug, active_groups = self._active_info
            phi = self._coefficient_dispersion if phi_scale else 1.0
            se = feature_se_from_cov(
                name,
                covariance_slope_view(XtWX_inv_aug, scale=phi),
                active_groups,
                self._result,
                self._groups,
                self._model._specs,
                self._model._interaction_specs,
                n_points=n_points,
            )
            return {
                "levels": spec._ordered_levels,
                "base_level": spec._base_level,
                "se_log_relativity": se,
            }

        if not any(group.name in selected_names for group in groups):
            if isinstance(spec, _SplineBase):
                x_grid = np.linspace(spec._lo, spec._hi, n_points)
                return {"x": x_grid, "se_log_relativity": np.zeros(n_points)}
            elif isinstance(spec, Categorical):
                return {
                    "levels": spec._non_base,
                    "base_level": spec._base_level,
                    "se_log_relativity": np.zeros(len(spec._non_base)),
                }
            elif isinstance(spec, Piecewise):
                return {
                    "x": spec._knots.copy(),
                    "se_log_relativity": np.zeros(spec._knots.size),
                }
            else:
                return {"se_coef": 0.0}

        # Gather covariance from all active subgroups (use augmented inverse)
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        phi = self._coefficient_dispersion if phi_scale else 1.0
        active_subs = [ag for ag in active_groups if ag.feature_name == name]
        if not active_subs:
            if isinstance(spec, _SplineBase):
                x_grid = np.linspace(spec._lo, spec._hi, n_points)
                return {"x": x_grid, "se_log_relativity": np.zeros(n_points)}
            elif isinstance(spec, Categorical):
                return {
                    "levels": spec._non_base,
                    "base_level": spec._base_level,
                    "se_log_relativity": np.zeros(len(spec._non_base)),
                }
            elif isinstance(spec, Piecewise):
                return {
                    "x": spec._knots.copy(),
                    "se_log_relativity": np.zeros(spec._knots.size),
                }
            else:
                return {"se_coef": 0.0}

        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        aug_indices = indices + 1  # offset by 1 for intercept row/col
        Cov_g = phi * covariance_selected_block(XtWX_inv_aug, aug_indices)

        if isinstance(spec, _SplineBase):
            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            M = np.asarray(spec.transform(x_grid), dtype=np.float64)
            active_cols = _active_feature_columns(name, groups, active_subs)
            M = M[:, active_cols]

            Q = M @ Cov_g
            se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))
            return {"x": x_grid, "se_log_relativity": se}

        elif isinstance(spec, Categorical):
            se = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            return {
                "levels": spec._non_base,
                "base_level": spec._base_level,
                "se_log_relativity": se,
            }

        elif isinstance(spec, Piecewise):
            # One SE per KNOT with a hard zero at the base -- the same shape,
            # order and labelling as `term_inference(...).se_log_relativity` and
            # the workbook's knot rows.  The unlabelled `{"se": diag}` fallback
            # below returns the J+1 per-coefficient values with no knot vector
            # beside them, so a caller zipping them against the term's knots
            # pairs every knot from the base onward with the wrong SE.
            M = np.asarray(spec._raw_basis_matrix(spec._knots), dtype=np.float64)[
                :, spec._non_base_indices
            ]
            Q = M @ Cov_g
            se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))
            return {"x": spec._knots.copy(), "se_log_relativity": se}

        elif isinstance(spec, Numeric):
            return {"se_coef": float(np.sqrt(max(Cov_g[0, 0], 0.0)))}

        else:
            se = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            return {"se": se}

    def feature_se(self, name: str, n_points: int = 200) -> dict[str, Any]:
        """Quasi-likelihood SE of a feature curve, levels, or coefficient.

        Uses the same dispersion convention as :attr:`coefficient_se`,
        including Pearson dispersion for known-scale families. Rendered
        coefficient tables retain fitted-family scale.

        Refuses for a term carrying a post-fit shape repair: the band would be
        propagated from the unconstrained fit's covariance around constrained
        coefficients, which is the quantity ``summary()`` withholds for that
        term. A per-feature accessor can say so exactly, so it does -- unlike
        :attr:`coefficient_se`, which is one dict over every group and already
        spends the all-zeros array on "not selected".
        """
        from superglm.model.explain_ops import _shape_repaired

        if _shape_repaired(self._model, name):
            raise RuntimeError(
                f"Feature {name!r} carries a post-fit shape repair, so its standard errors "
                "would be propagated from the unconstrained fit's covariance around "
                "constrained coefficients -- the quantity summary() withholds for this term. "
                "Fit the constraint (Constraint.fit.*) if you need a band."
            )
        return self._feature_se_impl(name, n_points=n_points, phi_scale=True)

    # ── Summary ───────────────────────────────────────────────────

    @staticmethod
    def _penalty_name(penalty: Any) -> str:
        """Human-readable penalty name from class name."""
        name = type(penalty).__name__
        # CamelCase -> spaced: GroupLasso -> Group Lasso
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)

    def _build_coef_rows(self, alpha: float = 0.05) -> list[_CoefRow]:
        """Build coefficient table rows for the summary."""
        X_a, W, XtWX_inv, XtWX_inv_aug, active_groups = self._active_info
        R_a = None
        edf = None
        edf1 = None
        if _requires_wood_inference(self._model, active_groups):
            R_a = self._active_R_factor
            edf, edf1 = self._influence_edf
        uses_fitted_rank = self._working_weights_match_fit(W)
        selected_names = {group.name for group in active_groups}
        centered_data_gram = self.__dict__.get("_active_centered_data_gram")
        precomputed_moments = (
            (np.empty((0, 0)), np.empty(0), centered_data_gram)
            if centered_data_gram is not None
            else None
        )
        from superglm.model.report_ops import (
            _withhold_repaired_term_inference,
            shape_repaired_feature_names,
        )

        repaired = shape_repaired_feature_names(self._model)
        rows = build_coef_rows(
            groups=self._groups,
            specs=self._model._specs,
            interaction_specs=self._model._interaction_specs,
            result=self._result,
            X_a=X_a,
            W=W,
            XtWX_inv=XtWX_inv,
            XtWX_inv_aug=XtWX_inv_aug,
            active_groups=active_groups,
            known_scale=self._known_scale,
            group_edf_map=(dict(self._result.rank_info.group_edf) if uses_fitted_rank else None),
            reml_lambdas=getattr(self._model, "_reml_lambdas", None),
            lambda2=fitted_lambda2(self._model),
            n_obs=self.n_obs,
            alpha=alpha,
            monotone_repairs=repaired,
            # A shape repair republishes coefficients from a constrained
            # estimator; the marker and the withholding that go with it belong
            # on every summary surface, not only ``model.summary()``.
            precomputed_R_a=R_a,
            precomputed_edf=edf,
            precomputed_edf1=edf1,
            precomputed_design_moments=precomputed_moments,
            coefficient_estimable_override=self._current_coefficient_estimable,
            selected_group_names=selected_names,
            group_matrices=(
                self._dm.group_matrices if self._uses_fit_design and self._dm is not None else None
            ),
            sample_weights=self._weights,
            weight_semantics=self._weight_semantics,
            selection_shrunk_group_names=selection_shrunk_group_names(
                fitted_penalty(self._model), self._groups
            ),
        )
        _withhold_repaired_term_inference(rows, repaired)
        return rows

    def summary(
        self,
        alpha: float = 0.05,
        detail: str = "compact",
        level_display: str = "expanded",
    ) -> ModelSummary:
        """Formatted model summary with coefficient table.

        The rendered coefficient table uses fitted-family covariance:
        known-scale families retain ``phi=1`` and estimated-scale families
        use their fitted ``phi``. It deliberately does not reuse the explicit
        Pearson quasi-likelihood accessors :attr:`coefficient_se`,
        :attr:`intercept_se`, or :attr:`feature_se`.

        The dict-like ``standard_errors`` payload retains the historical
        ``coefficient_se`` key for the explicit quasi-likelihood accessor and
        labels its scale separately from the rendered fitted-family table.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals (default 0.05 → 95% CI).
        detail : str
            Level of detail for spline terms. ``"compact"`` (default) shows
            one row per spline group. ``"full"`` adds per-coefficient
            detail rows (ASCII: printed inline; HTML: pre-expanded
            ``<details>`` disclosure). Default ``"compact"`` still shows
            closed disclosures in HTML.
        level_display : str
            Categorical level presentation. ``"expanded"`` (default) shows
            exact original levels; ``"grouped"`` shows one row per fitted
            group with a membership legend.

        Returns
        -------
        ModelSummary
            Object with ``__str__`` (ASCII), ``_repr_html_`` (HTML),
            and dict-like access for backward compatibility.
        """
        from superglm.inference.summary_levels import (
            build_level_universes,
            build_summary_level_display,
            validate_level_display,
        )

        level_display = validate_level_display(level_display)
        data = {
            # Same record as `model.summary()`: the two payloads are read as one
            # surface and are tested for agreement.
            "level_universes": build_level_universes(
                self._model._specs, self._model._interaction_specs
            ),
            "information_criteria": {
                "log_likelihood": self.log_likelihood,
                "null_log_likelihood": self.null_log_likelihood,
                "aic": self.aic,
                "bic": self.bic,
                "aicc": self.aicc,
                "ebic": self.ebic(),
            },
            "deviance": {
                "deviance": self.deviance,
                "null_deviance": self.null_deviance,
                "explained_deviance": self.explained_deviance,
            },
            "fit": {
                "phi": self.phi,
                "effective_df": self.effective_df,
                "pearson_chi2": self.pearson_chi2,
                "n_obs": self.n_obs,
                "n_active_groups": self.n_active_groups,
            },
            "standard_errors": {
                "coefficient_se": self.coefficient_se,
                "coefficient_se_raw": self.coefficient_se_raw,
                "coefficient_se_scale": "quasi-likelihood",
                "rendered_coefficient_se_scale": "fitted-family",
            },
            "diagnostic_contract": {
                "deviance_pearson_residuals": "compressed-row aggregate",
                "response_working_quantile_residuals": "per-row",
                "leverage_standardized_residuals_cooks_distance": (
                    "compressed weighted-cell deletion"
                ),
            },
        }

        penalty = fitted_penalty(self._model)
        link_name = type(self._link).__name__
        if link_name.endswith("Link"):
            link_name = link_name[:-4]

        model_info = {
            "family": type(self._family).__name__,
            "link": link_name,
            "penalty": self._penalty_name(penalty),
            "n_obs": self.n_obs,
            "effective_df": self.effective_df,
            "lambda1": penalty.lambda1,
            "phi": self.phi,
            "deviance": self.deviance,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
            "ebic": self.ebic(),
            "converged": self._result.converged,
            "n_iter": self._result.n_iter,
        }

        # NB theta profile info
        nb_pr = getattr(self._model, "_nb_profile_result", None)
        if nb_pr is not None:
            ci = nb_pr.ci(alpha=alpha)
            model_info["nb_theta"] = nb_pr.theta_hat
            model_info["nb_theta_ci"] = ci
            model_info["nb_theta_method"] = "Profile (exact)"

        # Tweedie p profile info
        tw_pr = getattr(self._model, "_tweedie_profile_result", None)
        if tw_pr is not None:
            ci, ci_status = cached_tweedie_profile_ci(tw_pr, alpha)
            model_info["tweedie_p"] = tw_pr.p_hat
            model_info["tweedie_p_ci"] = ci
            model_info["tweedie_p_ci_status"] = ci_status
            model_info["tweedie_phi"] = tw_pr.phi_hat
            model_info["tweedie_p_method"] = tweedie_profile_method_label(tw_pr)

        from superglm.model.report_ops import (
            _drop_repaired_basis_detail,
            shape_repaired_feature_names,
        )

        coef_rows = self._build_coef_rows(alpha=alpha)

        X_a, W, XtWX_inv, XtWX_inv_aug, active_groups = self._active_info
        specs = self._model._specs
        basis_detail = build_basis_detail(
            groups=self._groups,
            specs=specs,
            interaction_specs=self._model._interaction_specs,
            result=self._result,
            XtWX_inv_aug=XtWX_inv_aug,
            active_groups=active_groups,
            known_scale=self._known_scale,
            alpha=alpha,
            coefficient_estimable_override=self._current_coefficient_estimable,
            selected_group_names={group.name for group in active_groups},
        )
        # The other half of the withholding, and it does not come with the
        # rows: `build_basis_detail` is an independent pass, so `summary()`
        # here would otherwise print a term row reading "inference withheld"
        # above a per-coefficient disclosure of exactly the withheld
        # quantities -- which the HTML renderer emits unconditionally, so it
        # reaches a notebook at the default `detail="compact"`.
        _drop_repaired_basis_detail(
            basis_detail, self._groups, shape_repaired_feature_names(self._model)
        )

        level_presentation = build_summary_level_display(
            coef_rows,
            specs=specs,
            groups=self._groups,
            level_display=level_display,
        )
        return ModelSummary(
            data,
            model_info,
            coef_rows,
            alpha=alpha,
            detail=detail,
            basis_detail=basis_detail,
            level_presentation=level_presentation,
        )
