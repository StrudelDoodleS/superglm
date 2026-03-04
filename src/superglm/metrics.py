"""Comprehensive GLM diagnostics: information criteria, residuals, influence."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

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
            return (y - mu) / mu

        if kind == "quantile":
            return self._quantile_residuals()

        raise ValueError(
            f"Unknown residual type '{kind}'. "
            "Use 'deviance', 'pearson', 'response', 'working', or 'quantile'."
        )

    def _quantile_residuals(self) -> NDArray:
        """Randomized quantile residuals (Dunn & Smyth 1996). Poisson only."""
        from scipy.stats import norm, poisson

        from superglm.distributions import Poisson

        if not isinstance(self._family, Poisson):
            raise NotImplementedError(
                "Quantile residuals currently only implemented for Poisson."
            )

        y, mu = self._y, self._mu
        rng = np.random.default_rng(42)

        # For y_i ~ Poisson(mu_i), CDF is discrete:
        # F(y) = P(Y <= y), F(y-1) = P(Y <= y-1)
        # Randomized: u ~ Uniform(F(y-1), F(y))
        a = poisson.cdf(y - 1, mu)
        b = poisson.cdf(y, mu)
        u = rng.uniform(a, b)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        return norm.ppf(u)

    # ── Influence diagnostics (lazy) ──────────────────────────────

    @cached_property
    def _hat_diag(self) -> NDArray:
        """Hat matrix diagonal h_i via active-column inversion."""
        beta = self._result.beta
        mu = self._mu
        V = self._family.variance(mu)
        W = self._weights * mu**2 / V  # PIRLS working weights for log link

        # Collect active group matrices
        active_cols = []
        for gm, g in zip(self._dm.group_matrices, self._groups):
            if np.linalg.norm(beta[g.sl]) > 1e-12:
                active_cols.append(gm.toarray())

        if not active_cols:
            return np.zeros(self.n_obs)

        X_a = np.hstack(active_cols)
        # XtWX = X_a' diag(W) X_a
        XtWX = X_a.T @ (X_a * W[:, None])
        XtWX[np.diag_indices_from(XtWX)] += 1e-8  # regularise
        XtWX_inv = np.linalg.inv(XtWX)

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

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """All scalar metrics grouped logically."""
        return {
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
        }
