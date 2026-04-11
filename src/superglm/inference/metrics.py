"""Comprehensive GLM diagnostics: information criteria, residuals, influence."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from superglm.inference.coef_tables import build_basis_detail, build_coef_rows  # noqa: F401
from superglm.inference.covariance import (  # noqa: F401
    _penalised_xtwx_inv,
    _penalised_xtwx_inv_gram,
    _second_diff_penalty,
)
from superglm.inference.summary import ModelSummary, _CoefRow
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.model import SuperGLM


class ModelMetrics:
    """Post-fit diagnostics for a SuperGLM model.

    Parameters
    ----------
    model : SuperGLM
        A fitted model.
    X : DataFrame
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
        _mu: NDArray | None = None,
    ):
        self._model = model
        self._family = model._distribution
        self._link = model._link
        self._groups = model._groups
        self._dm = model._dm
        self._result = model.result

        self._y = np.asarray(y, dtype=np.float64)
        n = len(self._y)
        self._weights = (
            np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        )
        self._offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)

        if _mu is not None:
            self._mu = _mu
        else:
            self._mu = model.predict(X, offset=offset)

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

    @property
    def deviance(self) -> float:
        return self._result.deviance

    @cached_property
    def log_likelihood(self) -> float:
        return self._family.log_likelihood(self._y, self._mu, self._weights, self.phi)

    @cached_property
    def _null_mu(self) -> NDArray:
        """Null model prediction: intercept-only MLE, offset-aware.

        Without offset: mu = weighted mean of y (exact for canonical links).
        With offset: solves for b0 via Newton so that sum(w*(y-mu))=0
        where mu_i = link^{-1}(b0 + offset_i).
        """
        from superglm.distributions import Binomial, Gaussian, clip_mu
        from superglm.links import stabilize_eta

        y_bar = float(np.average(self._y, weights=self._weights))
        if isinstance(self._family, Binomial):
            y_bar = np.clip(y_bar, 1e-3, 1 - 1e-3)
        elif isinstance(self._family, Gaussian):
            y_bar = float(y_bar)
        else:
            y_bar = max(y_bar, 1e-10)

        if np.all(self._offset == 0):
            return np.full(self.n_obs, y_bar)

        # Newton iterations for intercept-only with offset
        b0 = float(self._link.link(np.atleast_1d(y_bar))[0]) - np.average(
            self._offset, weights=self._weights
        )
        for _ in range(25):
            eta = stabilize_eta(b0 + self._offset, self._link)
            mu = clip_mu(self._link.inverse(eta), self._family)
            dmu = self._link.deriv_inverse(eta)
            score = np.sum(self._weights * (self._y - mu) * dmu / self._family.variance(mu))
            info = np.sum(self._weights * dmu**2 / self._family.variance(mu))
            step = score / max(info, 1e-10)
            b0 += step
            if abs(step) < 1e-8:
                break

        eta = stabilize_eta(b0 + self._offset, self._link)
        return clip_mu(self._link.inverse(eta), self._family)

    @cached_property
    def null_log_likelihood(self) -> float:
        """Log-likelihood at the intercept-only (null) model."""
        return self._family.log_likelihood(self._y, self._null_mu, self._weights, self.phi)

    @cached_property
    def null_deviance(self) -> float:
        return float(np.sum(self._weights * self._family.deviance_unit(self._y, self._null_mu)))

    @cached_property
    def explained_deviance(self) -> float:
        """1 - deviance / null_deviance. Analogous to R-squared."""
        return 1.0 - self.deviance / self.null_deviance

    @property
    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.effective_df

    @property
    def bic(self) -> float:
        return -2.0 * self.log_likelihood + np.log(self.n_obs) * self.effective_df

    @property
    def aicc(self) -> float:
        edf = self.effective_df
        n = self.n_obs
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
        beta = self._result.beta
        return sum(1 for g in self._groups if np.linalg.norm(beta[g.sl]) > 1e-12)

    @cached_property
    def eta(self) -> NDArray:
        """Linear predictor (link-scale fitted values)."""
        return self._link.link(self._mu)

    # ── Residuals ─────────────────────────────────────────────────

    def residuals(self, kind: str = "deviance", *, seed: int | None = 42) -> NDArray:
        """Compute residuals of the specified type.

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
            eta = self._link.link(mu)
            dmu_deta = self._link.deriv_inverse(eta)
            return (y - mu) / dmu_deta

        if kind == "quantile":
            return self._quantile_residuals(seed=seed)

        raise ValueError(
            f"Unknown residual type '{kind}'. "
            "Use 'deviance', 'pearson', 'response', 'working', or 'quantile'."
        )

    def _quantile_residuals(self, seed: int | None = 42) -> NDArray:
        """Randomized quantile residuals (Dunn & Smyth 1996).

        Weight-aware: for rate-encoded data (e.g. Poisson frequency with
        exposure weights), the CDF is computed on the count scale
        (y*w ~ Poisson(mu*w)) so that residuals correctly reflect the
        precision of each observation.

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

        if isinstance(self._family, Binomial):
            # Bernoulli: w is case/frequency weight, not trials.
            # CDF is the same regardless of weight.
            a = np.where(y == 0, 0.0, 1.0 - mu)
            b = np.where(y == 0, 1.0 - mu, 1.0)
            u = rng.uniform(a, b)
        elif isinstance(self._family, Poisson):
            # Rate encoding: count = y * w ~ Poisson(mu * w).
            # CDF on the count scale, then jitter.
            count = np.round(y * w)
            lam = mu * w
            a = poisson.cdf(count - 1, lam)
            b = poisson.cdf(count, lam)
            u = rng.uniform(a, b)
        elif isinstance(self._family, NegativeBinomial):
            theta = self._family.theta
            p_nb = theta / (mu + theta)
            # NB2: count = y * w ~ NB(theta, p_nb) with mean mu * w.
            # For weighted NB2, adjust n and p to match mean = mu * w:
            # E[Y] = n*(1-p)/p = mu*w => n = theta*w, p = theta/(mu+theta)
            # But theta*w may not be integer; use scipy which handles float n.
            count = np.round(y * w)
            n_param = theta * w
            a = nbinom.cdf(count - 1, n=n_param, p=p_nb)
            b = nbinom.cdf(count, n=n_param, p=p_nb)
            u = rng.uniform(a, b)
        elif isinstance(self._family, Gamma):
            # Gamma: effective shape = w/phi, scale = mu*phi/w
            # E[Y] = mu, Var[Y] = mu^2 * phi / w
            shape = w / self.phi
            scale = mu * self.phi / w
            u = gamma_dist.cdf(y, a=shape, scale=scale)
        elif isinstance(self._family, Gaussian):
            # Effective variance = phi / w
            u = norm.cdf(y, loc=mu, scale=np.sqrt(self.phi / w))
        elif isinstance(self._family, Tweedie):
            # Tweedie p in (1,2): compound Poisson-Gamma.
            # With weights: lambda and scale both depend on w.
            p_tw = self._family.p
            phi = self.phi

            # Weight-adjusted Poisson rate and compound Gamma parameters
            lam = w * np.power(mu, 2 - p_tw) / ((2 - p_tw) * phi)
            p_zero = np.exp(-lam)
            alpha_tw = (2 - p_tw) / (p_tw - 1)  # Gamma shape per claim
            scale_tw = phi * (p_tw - 1) * np.power(mu, p_tw - 1) / w

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
    def _active_info(self) -> tuple[NDArray, NDArray, NDArray, NDArray, list[GroupSlice]]:
        """Shared computation for leverage and SEs.

        Returns (X_a, W, XtWX_inv, XtWX_inv_aug, active_groups) where:
        - X_a: (n, p_active) active design columns
        - W: (n,) working weights
        - XtWX_inv: (p_active, p_active) = (X'WX + S)^{-1}, unscaled by phi
        - XtWX_inv_aug: (p_active+1, p_active+1) augmented inverse incl. intercept
        - active_groups: list of GroupSlice for active groups (re-indexed to X_a columns)
        """
        beta = self._result.beta
        mu = self._mu
        V = self._family.variance(mu)
        eta = self._link.link(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        W = self._weights * dmu_deta**2 / V

        lam2 = getattr(self._model, "_reml_lambdas", None) or self._model.lambda2
        S_full = self._build_S_from_penalties(lam2)
        X_a, XtWX_inv, XtWX_inv_aug, active_groups, _ = _penalised_xtwx_inv(
            beta, W, self._dm.group_matrices, self._groups, lam2, S_override=S_full
        )
        return X_a, W, XtWX_inv, XtWX_inv_aug, active_groups

    @cached_property
    def _active_R_factor(self) -> NDArray:
        """Upper-triangular factor used by Wood-style smooth tests.

        The smooth-term test operates on the relevant columns of the
        weighted design QR factor rather than the raw ``n x p_g`` design
        block. For a fitted active design ``X_a`` with working weights
        ``W``, the factor ``R`` satisfies
        ``R.T @ R = X_a.T @ diag(W) @ X_a``. The Wood
        test should therefore operate on columns of this weighted QR factor,
        not on the raw design and not on an augmented ``[X; sqrt(S)]`` system.
        """
        X_a, W, _, _, active_groups = self._active_info
        if X_a.shape[1] == 0:
            return np.empty((0, 0))

        _, R = np.linalg.qr(X_a * np.sqrt(W)[:, None], mode="reduced")
        return R

    @cached_property
    def _influence_edf(self) -> tuple[NDArray, NDArray]:
        """Per-coefficient edf and edf1 from influence matrix F.

        edf = diag(F) where F = (X'WX + S)^{-1} X'WX
        edf1 = 2*edf - diag(F @ F)  (Wood's alternative EDF)
        """
        X_a, W, XtWX_inv, _, _ = self._active_info

        if X_a.shape[1] == 0:
            return np.array([]), np.array([])

        XtWX = X_a.T @ (X_a * W[:, None])
        F = XtWX_inv @ XtWX
        edf = np.diag(F)
        edf1 = 2.0 * edf - np.sum(F * F, axis=1)
        return edf, edf1

    @property
    def _known_scale(self) -> bool:
        """Poisson has known scale (phi=1 for test purposes)."""
        from superglm.distributions import Poisson

        return isinstance(self._family, Poisson)

    @cached_property
    def _hat_diag(self) -> NDArray:
        """Hat matrix diagonal h_i via active-column inversion."""
        X_a, W, XtWX_inv, _, _ = self._active_info

        if X_a.shape[1] == 0:
            return np.zeros(self.n_obs)

        # h_i = W_i * x_i' XtWX_inv x_i = W * rowsum((X_a @ XtWX_inv) * X_a)
        Q = X_a @ XtWX_inv
        h = W * np.sum(Q * X_a, axis=1)
        return np.clip(h, 0.0, 1.0)

    @property
    def leverage(self) -> NDArray:
        """Hat matrix diagonal. sum(h) approx effective_df - 1 (excludes intercept)."""
        return self._hat_diag

    @cached_property
    def cooks_distance(self) -> NDArray:
        """Cook's distance for each observation."""
        h = self._hat_diag
        r_p = self.residuals("pearson")
        p = self.effective_df
        phi = self.phi
        denom = (1.0 - h) ** 2 * p * phi
        denom = np.where(denom > 0, denom, np.inf)
        return r_p**2 * h / denom

    @cached_property
    def std_deviance_residuals(self) -> NDArray:
        """Standardized deviance residuals: r_dev / sqrt(phi * (1 - h))."""
        h = self._hat_diag
        r = self.residuals("deviance")
        scale = np.sqrt(self.phi * np.maximum(1.0 - h, 1e-10))
        return r / scale

    @cached_property
    def std_pearson_residuals(self) -> NDArray:
        """Standardized Pearson residuals: r_pear / sqrt(phi * (1 - h))."""
        h = self._hat_diag
        r = self.residuals("pearson")
        scale = np.sqrt(self.phi * np.maximum(1.0 - h, 1e-10))
        return r / scale

    # ── Coefficient standard errors ──────────────────────────────

    @cached_property
    def coefficient_se(self) -> dict[str, NDArray]:
        """Per-group coefficient standard errors (phi-scaled).

        Uses estimated phi (quasi-likelihood correction). For Poisson,
        this gives quasi-Poisson SEs. For Gamma/Tweedie, phi is always
        estimated so this is the standard choice.

        Inactive groups get all-zero SEs.

        Note: These are conditional-on-the-selected-model SEs from the
        penalized estimate. They do not account for model selection
        uncertainty (same convention as glmnet / mgcv).
        """
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        phi = self.phi
        beta = self._result.beta

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if np.linalg.norm(beta[g.sl]) < 1e-12:
                result[g.name] = np.zeros(g.size)
            else:
                # Find corresponding active group
                ag = next(ag for ag in active_groups if ag.name == g.name)
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                var_diag = phi * np.diag(XtWX_inv_aug[aug_sl, aug_sl])
                result[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
        return result

    @cached_property
    def coefficient_se_raw(self) -> dict[str, NDArray]:
        """Per-group coefficient standard errors assuming phi=1.

        For Poisson: these assume the Poisson variance is exactly correct
        (no overdispersion). For Gamma/Tweedie: these differ from
        coefficient_se since phi != 1.

        Inactive groups get all-zero SEs.
        """
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        beta = self._result.beta

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if np.linalg.norm(beta[g.sl]) < 1e-12:
                result[g.name] = np.zeros(g.size)
            else:
                ag = next(ag for ag in active_groups if ag.name == g.name)
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                var_diag = np.diag(XtWX_inv_aug[aug_sl, aug_sl])
                result[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
        return result

    @cached_property
    def intercept_se(self) -> float:
        """Standard error of the intercept (phi-scaled).

        Computed from the [0,0] element of the augmented Fisher information
        inverse, which accounts for covariance between the intercept and
        all other coefficients.
        """
        _, _, _, XtWX_inv_aug, _ = self._active_info
        icpt_var = float(XtWX_inv_aug[0, 0])
        if icpt_var <= 0:
            return 0.0
        return float(np.sqrt(max(self.phi, 0.0) * icpt_var))

    @cached_property
    def intercept_se_raw(self) -> float:
        """Standard error of the intercept assuming phi=1."""
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

        Uses phi-scaled covariance (quasi-likelihood) when ``phi_scale=True``.

        Parameters
        ----------
        name : str
            Feature name (e.g. "DrivAge"). For select=True splines with multiple
            subgroups, all subgroups are gathered automatically.
        """
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import _SplineBase

        beta = self._result.beta
        groups = self._model._feature_groups(name)
        spec = self._model._specs[name]

        # Inactive feature: return zeros (all subgroups zeroed)
        beta_combined = np.concatenate([beta[g.sl] for g in groups])
        if np.linalg.norm(beta_combined) < 1e-12:
            if isinstance(spec, _SplineBase):
                x_grid = np.linspace(spec._lo, spec._hi, n_points)
                return {"x": x_grid, "se_log_relativity": np.zeros(n_points)}
            elif isinstance(spec, Categorical):
                return {
                    "levels": spec._non_base,
                    "base_level": spec._base_level,
                    "se_log_relativity": np.zeros(len(spec._non_base)),
                }
            else:
                return {"se_coef": 0.0}

        # Gather covariance from all active subgroups (use augmented inverse)
        _, _, _, XtWX_inv_aug, active_groups = self._active_info
        phi = self.phi if phi_scale else 1.0
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
            else:
                return {"se_coef": 0.0}

        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        aug_indices = indices + 1  # offset by 1 for intercept row/col
        Cov_g = phi * XtWX_inv_aug[np.ix_(aug_indices, aug_indices)]

        if isinstance(spec, _SplineBase):
            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            B_grid = spec._raw_basis_matrix(x_grid)

            if spec._R_inv is not None:
                M = B_grid @ spec._R_inv
            else:
                M = B_grid

            # Only use columns for active subgroups
            active_cols = np.concatenate(
                [
                    np.arange(g.start, g.end) - groups[0].start
                    for g in groups
                    if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
                ]
            )
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

        elif isinstance(spec, Numeric):
            return {"se_coef": float(np.sqrt(max(Cov_g[0, 0], 0.0)))}

        else:
            se = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            return {"se": se}

    def feature_se(self, name: str, n_points: int = 200) -> dict[str, Any]:
        """SE of the log-relativity curve/levels for a feature."""
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
        return build_coef_rows(
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
            # Pass None so build_coef_rows computes EDF from this
            # ModelMetrics instance's own active info (which may use
            # different weights/data than the fit).
            group_edf_map=None,
            reml_lambdas=getattr(self._model, "_reml_lambdas", None),
            lambda2=self._model.lambda2,
            n_obs=self.n_obs,
            alpha=alpha,
            group_matrices=self._dm.group_matrices if self._dm is not None else None,
            sample_weights=self._weights,
        )

    def summary(self, alpha: float = 0.05, detail: str = "compact") -> ModelSummary:
        """Formatted model summary with coefficient table.

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

        Returns
        -------
        ModelSummary
            Object with ``__str__`` (ASCII), ``_repr_html_`` (HTML),
            and dict-like access for backward compatibility.
        """
        data = {
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
            },
        }

        penalty = self._model.penalty
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
            ci = tw_pr.ci(alpha=alpha)
            model_info["tweedie_p"] = tw_pr.p_hat
            model_info["tweedie_p_ci"] = ci
            model_info["tweedie_phi"] = tw_pr.phi_hat
            model_info["tweedie_p_method"] = f"Profile ({tw_pr.method}, phi={tw_pr.phi_method})"

        coef_rows = self._build_coef_rows(alpha=alpha)

        X_a, W, XtWX_inv, XtWX_inv_aug, active_groups = self._active_info
        basis_detail = build_basis_detail(
            groups=self._groups,
            specs=self._model._specs,
            interaction_specs=self._model._interaction_specs,
            result=self._result,
            XtWX_inv_aug=XtWX_inv_aug,
            active_groups=active_groups,
            known_scale=self._known_scale,
            alpha=alpha,
        )

        return ModelSummary(
            data, model_info, coef_rows, alpha=alpha, detail=detail, basis_detail=basis_detail
        )
