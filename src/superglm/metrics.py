"""Comprehensive GLM diagnostics: information criteria, residuals, influence."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from superglm.group_matrix import SparseSSPGroupMatrix
from superglm.summary import ModelSummary, _CoefRow, _compute_coef_stats
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.model import SuperGLM


def _second_diff_penalty(p: int) -> NDArray:
    """Second-difference penalty matrix D2'D2 for p basis functions."""
    D2 = np.diff(np.eye(p), n=2, axis=0)
    return D2.T @ D2


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
    exposure : array-like, optional
        Observation weights / exposure.
    offset : array-like, optional
        Offset term.
    """

    def __init__(
        self,
        model: SuperGLM,
        X,
        y,
        exposure=None,
        offset=None,
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
            np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
        )
        self._offset = (
            np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)
        )

        # Recompute mu from fitted model
        self._mu = model.predict(X, offset=offset)

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
        """Null model prediction: weighted mean of y."""
        y_safe = np.where(self._y > 0, self._y, 0.1)
        mu_null = np.average(y_safe, weights=self._weights)
        return np.full(self.n_obs, mu_null)

    @cached_property
    def null_log_likelihood(self) -> float:
        """Log-likelihood at the intercept-only (null) model."""
        return self._family.log_likelihood(
            self._y, self._null_mu, self._weights, self.phi
        )

    @cached_property
    def null_deviance(self) -> float:
        return float(
            np.sum(self._weights * self._family.deviance_unit(self._y, self._null_mu))
        )

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
        return (
            self.bic
            + 2.0 * gamma * (
                gammaln(p_total + 1)
                - gammaln(n_active + 1)
                - gammaln(p_total - n_active + 1)
            )
        )

    @cached_property
    def pearson_chi2(self) -> float:
        V = self._family.variance(self._mu)
        return float(np.sum(self._weights * (self._y - self._mu) ** 2 / V))

    @cached_property
    def n_active_groups(self) -> int:
        beta = self._result.beta
        return sum(
            1 for g in self._groups if np.linalg.norm(beta[g.sl]) > 1e-12
        )

    # ── Residuals ─────────────────────────────────────────────────

    def residuals(self, kind: str = "deviance") -> NDArray:
        """Compute residuals of the specified type.

        Parameters
        ----------
        kind : str
            One of "deviance", "pearson", "response", "working", "quantile".
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
            return self._quantile_residuals()

        raise ValueError(
            f"Unknown residual type '{kind}'. "
            "Use 'deviance', 'pearson', 'response', 'working', or 'quantile'."
        )

    def _quantile_residuals(self) -> NDArray:
        """Randomized quantile residuals (Dunn & Smyth 1996)."""
        from scipy.stats import nbinom, norm, poisson

        from superglm.distributions import NegativeBinomial, Poisson

        y, mu = self._y, self._mu
        rng = np.random.default_rng(42)

        if isinstance(self._family, Poisson):
            a = poisson.cdf(y - 1, mu)
            b = poisson.cdf(y, mu)
        elif isinstance(self._family, NegativeBinomial):
            theta = self._family.theta
            p_nb = theta / (mu + theta)
            a = nbinom.cdf(y - 1, n=theta, p=p_nb)
            b = nbinom.cdf(y, n=theta, p=p_nb)
        else:
            raise NotImplementedError(
                "Quantile residuals currently only implemented for Poisson "
                "and Negative Binomial."
            )

        u = rng.uniform(a, b)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        return norm.ppf(u)

    # ── Influence diagnostics (lazy) ──────────────────────────────

    @cached_property
    def _active_info(self) -> tuple[NDArray, NDArray, NDArray, list[GroupSlice]]:
        """Shared computation for leverage and SEs.

        Returns (X_a, W, XtWX_inv, active_groups) where:
        - X_a: (n, p_active) active design columns
        - W: (n,) working weights
        - XtWX_inv: (p_active, p_active) inverse Fisher information (unscaled by phi)
        - active_groups: list of GroupSlice for active groups (re-indexed to X_a columns)
        """
        beta = self._result.beta
        mu = self._mu
        V = self._family.variance(mu)
        eta = self._link.link(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        W = self._weights * dmu_deta**2 / V

        # Collect active group matrices and build re-indexed group slices
        active_cols = []
        active_groups = []
        active_gms = []
        col = 0
        for gm, g in zip(self._dm.group_matrices, self._groups):
            if np.linalg.norm(beta[g.sl]) > 1e-12:
                arr = gm.toarray()
                active_cols.append(arr)
                active_gms.append(gm)
                p_g = arr.shape[1]
                active_groups.append(
                    GroupSlice(name=g.name, start=col, end=col + p_g, weight=g.weight)
                )
                col += p_g

        if not active_cols:
            return np.empty((self.n_obs, 0)), W, np.empty((0, 0)), []

        X_a = np.hstack(active_cols)
        p_a = X_a.shape[1]

        # Wood (2006) Bayesian covariance via augmented QR.
        # Forming X'WX explicitly squares the condition number, producing
        # float64 noise eigenvalues at ~1e-11 for large n. Instead, QR-
        # decompose the augmented system [sqrt(W)*X; sqrt(S)] to get R
        # such that R'R = X'WX + S, preserving the original conditioning.
        # Then (X'WX + S)^{-1} = R^{-1} R^{-T}.

        # Build sqrt(S) factor: L such that L'L = S (block-diagonal)
        lambda2 = self._model.lambda2
        S_rows = np.zeros((p_a, p_a))
        for gm_orig, ag in zip(active_gms, active_groups):
            if isinstance(gm_orig, SparseSSPGroupMatrix):
                R_inv = gm_orig.R_inv
                p_b = R_inv.shape[0]
                omega = _second_diff_penalty(p_b)
                S_g = lambda2 * R_inv.T @ omega @ R_inv
                # Eigendecomposition for stable square root (S_g is PSD)
                # L = diag(sqrt(d)) @ V.T  →  L'L = V diag(d) V' = S_g
                eigvals_g, eigvecs_g = np.linalg.eigh(S_g)
                eigvals_g = np.maximum(eigvals_g, 0.0)
                L_g = np.sqrt(eigvals_g)[:, None] * eigvecs_g.T
                S_rows[ag.sl, ag.sl] = L_g

        # Augmented matrix: [sqrt(W) * X_a; sqrt(S)]
        A = np.vstack([X_a * np.sqrt(W)[:, None], S_rows])
        _, R = np.linalg.qr(A, mode='reduced')

        # Truncated SVD of R: threshold near-zero singular values
        # that represent genuinely unidentifiable directions (cross-group
        # collinearity). QR gives accurate singular values — no noise
        # from condition-number squaring.
        U_R, s_R, Vh_R = np.linalg.svd(R, full_matrices=False)
        threshold = 1e-6 * s_R[0]
        inv_s2 = np.where(s_R > threshold, 1.0 / s_R**2, 0.0)
        XtWX_inv = (Vh_R.T * inv_s2[None, :]) @ Vh_R

        return X_a, W, XtWX_inv, active_groups

    @cached_property
    def _hat_diag(self) -> NDArray:
        """Hat matrix diagonal h_i via active-column inversion."""
        X_a, W, XtWX_inv, _ = self._active_info

        if X_a.shape[1] == 0:
            return np.zeros(self.n_obs)

        # h_i = W_i * x_i' XtWX_inv x_i = W * rowsum((X_a @ XtWX_inv) * X_a)
        Q = X_a @ XtWX_inv
        h = W * np.sum(Q * X_a, axis=1)
        return np.clip(h, 0.0, 1.0)

    @property
    def leverage(self) -> NDArray:
        """Hat matrix diagonal. sum(h) approx effective_df."""
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

        Note: These are "naive" SEs from the penalized estimate — they do
        not account for model selection (same convention as glmnet).
        """
        _, _, XtWX_inv, active_groups = self._active_info
        phi = self.phi
        beta = self._result.beta

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if np.linalg.norm(beta[g.sl]) < 1e-12:
                result[g.name] = np.zeros(g.size)
            else:
                # Find corresponding active group
                ag = next(ag for ag in active_groups if ag.name == g.name)
                var_diag = phi * np.diag(XtWX_inv[ag.sl, ag.sl])
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
        _, _, XtWX_inv, active_groups = self._active_info
        beta = self._result.beta

        result: dict[str, NDArray] = {}
        for g in self._groups:
            if np.linalg.norm(beta[g.sl]) < 1e-12:
                result[g.name] = np.zeros(g.size)
            else:
                ag = next(ag for ag in active_groups if ag.name == g.name)
                var_diag = np.diag(XtWX_inv[ag.sl, ag.sl])
                result[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
        return result

    @cached_property
    def intercept_se(self) -> float:
        """Standard error of the intercept (phi-scaled, conditional).

        Computed as sqrt(phi / sum(W)) where W are the GLM working weights.
        This is the conditional SE given other coefficients, consistent with
        the BCD solver's closed-form intercept update.
        """
        _, W, _, _ = self._active_info
        w_sum = float(np.sum(W))
        if w_sum <= 0:
            return 0.0
        return float(np.sqrt(self.phi / w_sum))

    def feature_se(self, name: str, n_points: int = 200) -> dict[str, Any]:
        """SE of the log-relativity curve/levels for a feature.

        Propagates the covariance of the fitted coefficients through the
        feature's design matrix to produce SEs on the interpretable scale.

        For splines: returns ``{x, se_log_relativity}`` on a grid.
        For categoricals: returns ``{levels, se_log_relativity}`` per level.
        For numerics: returns ``{se_coef}`` (unstandardized).

        Uses phi-scaled covariance (quasi-likelihood).
        """
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        beta = self._result.beta
        g = next(g for g in self._groups if g.name == name)

        # Inactive group: return zeros
        if np.linalg.norm(beta[g.sl]) < 1e-12:
            spec = self._model._specs[name]
            if isinstance(spec, Spline):
                x_grid = np.linspace(spec._lo, spec._hi, n_points)
                return {"x": x_grid, "se_log_relativity": np.zeros(n_points)}
            elif isinstance(spec, Categorical):
                return {
                    "levels": spec._levels,
                    "se_log_relativity": np.zeros(len(spec._non_base)),
                }
            else:
                return {"se_coef": 0.0}

        # Get active-group covariance block
        _, _, XtWX_inv, active_groups = self._active_info
        phi = self.phi
        ag = next(ag for ag in active_groups if ag.name == name)
        Cov_g = phi * XtWX_inv[ag.sl, ag.sl]

        spec = self._model._specs[name]

        if isinstance(spec, Spline):
            # SE(grid_i) = sqrt(diag(B_grid @ R_inv @ Cov_g @ R_inv.T @ B_grid.T))
            from scipy.interpolate import BSpline as BSpl

            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            x_clip = np.clip(x_grid, spec._knots[0], spec._knots[-1])
            B_grid = BSpl.design_matrix(x_clip, spec._knots, spec.degree).toarray()

            if spec._R_inv is not None:
                # Solver covariance is in alpha (reparametrized) space.
                # Transform: Cov_orig = R_inv @ Cov_alpha @ R_inv.T
                # Then: SE = sqrt(diag(B_grid @ Cov_orig @ B_grid.T))
                # = sqrt(rowsum((B_grid @ R_inv @ L) ** 2)) where Cov_g = L @ L.T
                M = B_grid @ spec._R_inv
            else:
                M = B_grid

            # SE = sqrt(diag(M @ Cov_g @ M.T)) = sqrt(rowsum((M @ Cov_g) * M))
            Q = M @ Cov_g
            se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))
            return {"x": x_grid, "se_log_relativity": se}

        elif isinstance(spec, Categorical):
            # Each level's SE is just sqrt(Cov_g[i, i])
            se = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            return {"levels": spec._levels, "se_log_relativity": se}

        elif isinstance(spec, Numeric):
            # Single coefficient
            se_transformed = np.sqrt(max(Cov_g[0, 0], 0.0))
            if spec.standardize:
                se_original = se_transformed / spec._std
            else:
                se_original = se_transformed
            return {"se_coef": float(se_original)}

        else:
            # Generic fallback: just return diagonal SEs
            se = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            return {"se": se}

    # ── Summary ───────────────────────────────────────────────────

    @staticmethod
    def _penalty_name(penalty: Any) -> str:
        """Human-readable penalty name from class name."""
        name = type(penalty).__name__
        # CamelCase -> spaced: GroupLasso -> Group Lasso
        return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)

    def _build_coef_rows(self, alpha: float = 0.05) -> list[_CoefRow]:
        """Build coefficient table rows for the summary."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        beta = self._result.beta
        se_dict = self.coefficient_se
        rows: list[_CoefRow] = []

        # Intercept row
        intercept = self._result.intercept
        icpt_se = self.intercept_se
        z, p, ci_lo, ci_hi = _compute_coef_stats(intercept, icpt_se, alpha)
        rows.append(_CoefRow(
            name="Intercept",
            coef=intercept,
            se=icpt_se,
            z=z,
            p=p,
            ci_low=ci_lo,
            ci_high=ci_hi,
        ))

        # Feature rows
        for g in self._groups:
            spec = self._model._specs[g.name]
            b_g = beta[g.sl]
            se_g = se_dict[g.name]
            active = np.linalg.norm(b_g) > 1e-12

            if isinstance(spec, Spline):
                if active:
                    # Group-level Wald chi² test: β' Cov⁻¹ β ~ chi²(p_g)
                    # Individual basis SEs are meaningless due to collinearity
                    from scipy.stats import chi2 as chi2_dist

                    _, _, XtWX_inv, active_groups = self._active_info
                    ag = next(a for a in active_groups if a.name == g.name)
                    Cov_g = self.phi * XtWX_inv[ag.sl, ag.sl]
                    try:
                        wald_chi2 = float(b_g @ np.linalg.solve(Cov_g, b_g))
                        wald_p = float(1.0 - chi2_dist.cdf(wald_chi2, df=g.size))
                    except np.linalg.LinAlgError:
                        wald_chi2 = float("nan")
                        wald_p = float("nan")

                    # Curve-level SE range (the interpretable uncertainty)
                    fse = self.feature_se(g.name)
                    se_curve = fse["se_log_relativity"]
                    curve_se_min = float(np.min(se_curve))
                    curve_se_max = float(np.max(se_curve))

                    rows.append(_CoefRow(
                        name=g.name,
                        group=g.name,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=wald_chi2,
                        wald_p=wald_p,
                        curve_se_min=curve_se_min,
                        curve_se_max=curve_se_max,
                    ))
                else:
                    rows.append(_CoefRow(
                        name=g.name,
                        group=g.name,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                    ))

            elif isinstance(spec, Categorical):
                for i, level in enumerate(spec._non_base):
                    coef_val = float(b_g[i])
                    se_val = float(se_g[i])
                    z, p, ci_lo, ci_hi = _compute_coef_stats(
                        coef_val, se_val, alpha
                    )
                    rows.append(_CoefRow(
                        name=f"{g.name}[{level}]",
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    ))

            elif isinstance(spec, Numeric):
                # Show original-scale coefficient and SE
                coef_internal = float(b_g[0])
                se_internal = float(se_g[0])
                if spec.standardize:
                    coef_display = coef_internal / spec._std
                    se_display = se_internal / spec._std
                else:
                    coef_display = coef_internal
                    se_display = se_internal
                z, p, ci_lo, ci_hi = _compute_coef_stats(
                    coef_display, se_display, alpha
                )
                rows.append(_CoefRow(
                    name=g.name,
                    group=g.name,
                    coef=coef_display,
                    se=se_display,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                ))

            else:
                # Generic: show first coefficient
                coef_val = float(b_g[0])
                se_val = float(se_g[0]) if len(se_g) > 0 else 0.0
                z, p, ci_lo, ci_hi = _compute_coef_stats(
                    coef_val, se_val, alpha
                )
                rows.append(_CoefRow(
                    name=g.name,
                    group=g.name,
                    coef=coef_val,
                    se=se_val,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                ))

        return rows

    def summary(self, alpha: float = 0.05) -> ModelSummary:
        """Formatted model summary with coefficient table.

        Parameters
        ----------
        alpha : float
            Significance level for confidence intervals (default 0.05 → 95% CI).

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
            "converged": self._result.converged,
            "n_iter": self._result.n_iter,
        }

        coef_rows = self._build_coef_rows(alpha=alpha)

        return ModelSummary(data, model_info, coef_rows, alpha=alpha)
