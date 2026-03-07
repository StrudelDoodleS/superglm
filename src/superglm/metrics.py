"""Comprehensive GLM diagnostics: information criteria, residuals, influence."""

from __future__ import annotations

import re
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln

from superglm.group_matrix import (
    DiscretizedSSPGroupMatrix,
    SparseSSPGroupMatrix,
    _block_xtwx,
)
from superglm.summary import ModelSummary, _CoefRow, _compute_coef_stats
from superglm.types import GroupSlice

if TYPE_CHECKING:
    from superglm.model import SuperGLM


def _second_diff_penalty(p: int) -> NDArray:
    """Second-difference penalty matrix D2'D2 for p basis functions."""
    D2 = np.diff(np.eye(p), n=2, axis=0)
    return D2.T @ D2


def _penalised_xtwx_inv(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
) -> tuple[NDArray, NDArray, list[GroupSlice], list]:
    """Compute (X'WX + S)^{-1} via augmented QR + truncated SVD.

    Shared by ``model._coef_covariance`` and ``ModelMetrics._active_info``.

    Parameters
    ----------
    beta : (p,) array — full coefficient vector.
    W : (n,) array — working weights (dmu_deta² / V, exposure-scaled).
    group_matrices : list of GroupMatrix — design matrices per group.
    groups : list of GroupSlice — slices into beta for each group.
    lambda2 : float or dict[str, float]
        Smoothing penalty weight. A scalar applies to all groups; a dict
        maps group names to per-group lambdas (REML).

    Returns
    -------
    X_a : (n, p_active) dense active design matrix.
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    active_groups : list of GroupSlice re-indexed to X_a columns.
    active_gms : list of GroupMatrix for active groups.
    """
    active_cols: list[NDArray] = []
    active_groups_out: list[GroupSlice] = []
    active_gms: list = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if np.linalg.norm(beta[g.sl]) > 1e-12:
            arr = gm.toarray()
            active_cols.append(arr)
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = arr.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                )
            )
            col += p_g

    if not active_cols:
        n = len(W)
        return np.empty((n, 0)), np.empty((0, 0)), [], []

    X_a = np.hstack(active_cols)
    p_a = X_a.shape[1]

    # Build sqrt(S) factor: L such that L'L = S (block-diagonal penalty)
    # Unpenalized groups (e.g. select=True null-space) get no penalty contribution.
    S_rows = np.zeros((p_a, p_a))
    for gm_orig, ag, gname in zip(active_gms, active_groups_out, active_group_names):
        if isinstance(gm_orig, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix) and ag.penalized:
            # Resolve per-group lambda
            if isinstance(lambda2, dict):
                lam_g = lambda2.get(gname, 0.0)
            else:
                lam_g = lambda2

            # Use stored omega (correct for CRS, TensorInteraction, etc.)
            # Fall back to second-difference penalty for backward compatibility.
            R_inv = gm_orig.R_inv
            omega = gm_orig.omega
            if omega is None:
                p_b = R_inv.shape[0]
                omega = _second_diff_penalty(p_b)

            S_g = lam_g * R_inv.T @ omega @ R_inv
            eigvals_g, eigvecs_g = np.linalg.eigh(S_g)
            eigvals_g = np.maximum(eigvals_g, 0.0)
            L_g = np.sqrt(eigvals_g)[:, None] * eigvecs_g.T
            S_rows[ag.sl, ag.sl] = L_g

    # Augmented QR: [sqrt(W)*X; sqrt(S)] → R'R = X'WX + S
    A = np.vstack([X_a * np.sqrt(W)[:, None], S_rows])
    _, R = np.linalg.qr(A, mode="reduced")

    # Truncated SVD for numerical stability
    _, s_R, Vh_R = np.linalg.svd(R, full_matrices=False)
    threshold = 1e-6 * s_R[0]
    inv_s2 = np.where(s_R > threshold, 1.0 / s_R**2, 0.0)
    XtWX_S_inv = (Vh_R.T * inv_s2[None, :]) @ Vh_R

    return X_a, XtWX_S_inv, active_groups_out, active_gms


def _penalised_xtwx_inv_gram(
    beta: NDArray,
    W: NDArray,
    group_matrices: list,
    groups: list[GroupSlice],
    lambda2: float | dict[str, float],
) -> tuple[NDArray, list[GroupSlice]]:
    """Fast (X'WX + S)^{-1} via per-group gram matrices.

    Same result as ``_penalised_xtwx_inv`` but avoids forming the dense
    (n, p) matrix. Computes X'WX block-by-block using the group gram
    kernels, then inverts the (p_active, p_active) system directly.
    Cost is O(n · sum(p_g²)) + O(p³) instead of O(n · p²).

    Does NOT return X_a (not needed for REML). For leverage/hat matrix
    diagnostics, use ``_penalised_xtwx_inv`` instead.

    Returns
    -------
    XtWX_S_inv : (p_active, p_active) inverse of (X'WX + S).
    active_groups : list of GroupSlice re-indexed to active columns.
    """
    # Identify active groups
    active_gms: list = []
    active_groups_out: list[GroupSlice] = []
    active_group_names: list[str] = []
    col = 0
    for gm, g in zip(group_matrices, groups):
        if np.linalg.norm(beta[g.sl]) > 1e-12:
            active_gms.append(gm)
            active_group_names.append(g.name)
            p_g = gm.shape[1]
            active_groups_out.append(
                GroupSlice(
                    name=g.name,
                    start=col,
                    end=col + p_g,
                    weight=g.weight,
                    penalized=g.penalized,
                    feature_name=g.feature_name,
                    subgroup_type=g.subgroup_type,
                )
            )
            col += p_g

    p_a = col
    if p_a == 0:
        return np.empty((0, 0)), []

    # Build X'WX block-by-block: gram for diagonal, cross_gram for off-diagonal.
    # For discretized groups this is O(n_bins) per block instead of O(n·p²).
    XtWX = _block_xtwx(active_gms, active_groups_out, W)

    # Add penalty S (same logic as _penalised_xtwx_inv)
    S = np.zeros((p_a, p_a))
    for gm_orig, ag, gname in zip(active_gms, active_groups_out, active_group_names):
        if isinstance(gm_orig, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix) and ag.penalized:
            if isinstance(lambda2, dict):
                lam_g = lambda2.get(gname, 0.0)
            else:
                lam_g = lambda2

            R_inv = gm_orig.R_inv
            omega = gm_orig.omega
            if omega is None:
                p_b = R_inv.shape[0]
                omega = _second_diff_penalty(p_b)
            S[ag.sl, ag.sl] = lam_g * R_inv.T @ omega @ R_inv

    # Invert (X'WX + S) via truncated SVD
    M = XtWX + S
    eigvals, eigvecs = np.linalg.eigh(M)
    threshold = 1e-6 * max(eigvals.max(), 1e-12)
    inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
    XtWX_S_inv = (eigvecs * inv_eigvals[None, :]) @ eigvecs.T

    return XtWX_S_inv, active_groups_out


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
        self._weights = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
        self._offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=np.float64)

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
        """Null model prediction: intercept-only MLE, offset-aware.

        Without offset: mu = weighted mean of y (exact for canonical links).
        With offset: solves for b0 via Newton so that sum(w*(y-mu))=0
        where mu_i = link^{-1}(b0 + offset_i).
        """
        y_bar = np.average(self._y, weights=self._weights)
        y_bar = max(y_bar, 1e-10)  # guard for log link

        if np.all(self._offset == 0):
            return np.full(self.n_obs, y_bar)

        # Newton iterations for intercept-only with offset
        b0 = self._link.link(y_bar) - np.average(self._offset, weights=self._weights)
        for _ in range(25):
            eta = b0 + self._offset
            mu = self._link.inverse(np.clip(eta, -20, 20))
            dmu = self._link.deriv_inverse(np.clip(eta, -20, 20))
            score = np.sum(self._weights * (self._y - mu) * dmu / self._family.variance(mu))
            info = np.sum(self._weights * dmu**2 / self._family.variance(mu))
            step = score / max(info, 1e-10)
            b0 += step
            if abs(step) < 1e-8:
                break

        eta = b0 + self._offset
        return np.maximum(self._link.inverse(np.clip(eta, -20, 20)), 1e-10)

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

        For discrete families (Poisson, NB2), uses jittered uniform on the
        CDF interval [F(y-1), F(y)]. For continuous families (Gamma), uses
        the CDF directly (no randomization needed).

        Parameters
        ----------
        seed : int or None
            Random seed for the jitter in discrete families. Default 42
            for reproducibility. Pass None for non-deterministic.
        """
        from scipy.stats import gamma as gamma_dist
        from scipy.stats import nbinom, norm, poisson

        from superglm.distributions import Gamma, NegativeBinomial, Poisson, Tweedie

        y, mu = self._y, self._mu
        rng = np.random.default_rng(seed)

        if isinstance(self._family, Poisson):
            a = poisson.cdf(y - 1, mu)
            b = poisson.cdf(y, mu)
            u = rng.uniform(a, b)
        elif isinstance(self._family, NegativeBinomial):
            theta = self._family.theta
            p_nb = theta / (mu + theta)
            a = nbinom.cdf(y - 1, n=theta, p=p_nb)
            b = nbinom.cdf(y, n=theta, p=p_nb)
            u = rng.uniform(a, b)
        elif isinstance(self._family, Gamma):
            # Gamma is continuous: shape k = 1/phi, scale = mu*phi
            shape = 1.0 / self.phi
            scale = mu * self.phi
            u = gamma_dist.cdf(y, a=shape, scale=scale)
        elif isinstance(self._family, Tweedie):
            # Tweedie p in (1,2): compound Poisson-Gamma.
            # Y = sum_{j=1}^N X_j where N ~ Pois(lam), X_j ~ Gamma(alpha, scale).
            # CDF via compound Poisson sum — fully vectorized per Poisson term.
            p_tw = self._family.p
            phi = self.phi

            # Poisson rate and compound Gamma parameters
            lam = np.power(mu, 2 - p_tw) / ((2 - p_tw) * phi)
            p_zero = np.exp(-lam)
            alpha_tw = (2 - p_tw) / (p_tw - 1)  # Gamma shape per claim
            scale_tw = phi * (p_tw - 1) * np.power(mu, p_tw - 1)  # Gamma scale

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
    def _active_info(self) -> tuple[NDArray, NDArray, NDArray, list[GroupSlice]]:
        """Shared computation for leverage and SEs.

        Returns (X_a, W, XtWX_inv, active_groups) where:
        - X_a: (n, p_active) active design columns
        - W: (n,) working weights
        - XtWX_inv: (p_active, p_active) = (X'WX + S)^{-1}, unscaled by phi
        - active_groups: list of GroupSlice for active groups (re-indexed to X_a columns)
        """
        beta = self._result.beta
        mu = self._mu
        V = self._family.variance(mu)
        eta = self._link.link(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        W = self._weights * dmu_deta**2 / V

        lam2 = getattr(self._model, "_reml_lambdas", None) or self._model.lambda2
        X_a, XtWX_inv, active_groups, _ = _penalised_xtwx_inv(
            beta, W, self._dm.group_matrices, self._groups, lam2
        )
        return X_a, W, XtWX_inv, active_groups

    @cached_property
    def _active_R_factor(self) -> NDArray:
        """Upper-triangular factor used by mgcv-style smooth tests.

        ``mgcv::summary.gam`` feeds ``testStat`` the relevant columns of the
        model's ``R`` factor rather than the raw ``n x p_g`` design block. The
        distinction matters once smoothing is active: using the raw design block
        can materially overstate smooth significance for weak/noisy terms.
        """
        beta = self._result.beta
        X_a, W, _, active_groups = self._active_info
        p_a = X_a.shape[1]
        if p_a == 0:
            return np.empty((0, 0))

        lam2 = getattr(self._model, "_reml_lambdas", None) or self._model.lambda2
        active_gms = [
            gm
            for gm, g in zip(self._dm.group_matrices, self._groups)
            if np.linalg.norm(beta[g.sl]) > 1e-12
        ]

        # sqrt(S) factor: blockwise eigendecomposition keeps semidefinite
        # spline penalties numerically stable inside the augmented QR.
        S_rows = np.zeros((p_a, p_a))
        for gm, ag in zip(active_gms, active_groups):
            if not isinstance(gm, SparseSSPGroupMatrix | DiscretizedSSPGroupMatrix):
                continue
            if not ag.penalized or gm.omega is None:
                continue

            if isinstance(lam2, dict):
                lam_g = lam2.get(ag.name, 0.0)
            else:
                lam_g = lam2

            S_g = lam_g * gm.R_inv.T @ gm.omega @ gm.R_inv
            eigvals_g, eigvecs_g = np.linalg.eigh(S_g)
            eigvals_g = np.maximum(eigvals_g, 0.0)
            L_g = np.sqrt(eigvals_g)[:, None] * eigvecs_g.T
            S_rows[ag.sl, ag.sl] = L_g

        A = np.vstack([X_a * np.sqrt(W)[:, None], S_rows])
        _, R = np.linalg.qr(A, mode="reduced")
        return R

    @cached_property
    def _influence_edf(self) -> tuple[NDArray, NDArray]:
        """Per-coefficient edf and edf1 from influence matrix F.

        edf = diag(F) where F = (X'WX + S)^{-1} X'WX
        edf1 = 2*edf - diag(F @ F)  (Wood's alternative EDF)
        """
        X_a, W, XtWX_inv, _ = self._active_info

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
        X_a, W, XtWX_inv, _ = self._active_info

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
        return float(np.sqrt(max(self.phi, 0.0) / w_sum))

    @cached_property
    def intercept_se_raw(self) -> float:
        """Standard error of the intercept assuming phi=1."""
        _, W, _, _ = self._active_info
        w_sum = float(np.sum(W))
        if w_sum <= 0:
            return 0.0
        return float(np.sqrt(1.0 / w_sum))

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
        For numerics: returns ``{se_coef}`` (unstandardized).

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

        # Gather covariance from all active subgroups
        _, _, XtWX_inv, active_groups = self._active_info
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
        Cov_g = phi * XtWX_inv[np.ix_(indices, indices)]

        if isinstance(spec, _SplineBase):
            from scipy.interpolate import BSpline as BSpl

            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            x_clip = np.clip(x_grid, spec._knots[0], spec._knots[-1])
            B_grid = BSpl.design_matrix(x_clip, spec._knots, spec.degree).toarray()

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
            se_transformed = np.sqrt(max(Cov_g[0, 0], 0.0))
            if spec.standardize:
                se_original = se_transformed / spec._std
            else:
                se_original = se_transformed
            return {"se_coef": float(se_original)}

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
        from superglm.features.categorical import Categorical
        from superglm.features.interaction import (
            CategoricalInteraction,
            NumericCategorical,
            NumericInteraction,
            PolynomialCategorical,
            PolynomialInteraction,
            SplineCategorical,
        )
        from superglm.features.numeric import Numeric
        from superglm.features.spline import _SplineBase

        beta = self._result.beta
        se_dict = self.coefficient_se_raw if self._known_scale else self.coefficient_se
        rows: list[_CoefRow] = []

        # Intercept row
        intercept = self._result.intercept
        icpt_se = self.intercept_se_raw if self._known_scale else self.intercept_se
        z, p, ci_lo, ci_hi = _compute_coef_stats(intercept, icpt_se, alpha)
        rows.append(
            _CoefRow(
                name="Intercept",
                coef=intercept,
                se=icpt_se,
                z=z,
                p=p,
                ci_low=ci_lo,
                ci_high=ci_hi,
            )
        )

        # Feature rows
        for g in self._groups:
            spec = self._model._specs.get(g.feature_name) or self._model._interaction_specs.get(
                g.feature_name
            )
            b_g = beta[g.sl]
            se_g = se_dict[g.name]
            active = np.linalg.norm(b_g) > 1e-12

            if isinstance(spec, _SplineBase):
                is_linear_subgroup = g.subgroup_type == "linear"
                if active:
                    stat = float("nan")
                    p_val = float("nan")
                    ref_df = float(g.size)
                    curve_se_min = float("nan")
                    curve_se_max = float("nan")

                    X_a, _, XtWX_inv, active_groups = self._active_info
                    R_a = self._active_R_factor
                    ag = next(a for a in active_groups if a.name == g.name)
                    if self._known_scale:
                        V_b_j = XtWX_inv[ag.sl, ag.sl]
                    else:
                        V_b_j = self.phi * XtWX_inv[ag.sl, ag.sl]

                    if is_linear_subgroup:
                        # Wald chi-squared for linear subgroup
                        from scipy.stats import chi2 as chi2_dist

                        try:
                            stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                            ref_df = float(g.size)
                            p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                        except np.linalg.LinAlgError:
                            pass

                        # Curve SE range for the full feature
                        fse = self._feature_se_impl(
                            g.feature_name,
                            phi_scale=not self._known_scale,
                        )
                        se_curve = fse["se_log_relativity"]
                        curve_se_min = float(np.min(se_curve))
                        curve_se_max = float(np.max(se_curve))
                    else:
                        # Wood (2013) Bayesian test for smooth terms
                        from superglm.wood_pvalue import wood_test_smooth

                        edf, edf1 = self._influence_edf
                        edf1_j = float(np.sum(edf1[ag.sl]))
                        X_j = R_a[:, ag.sl]
                        res_df = -1.0 if self._known_scale else float(self.n_obs - np.sum(edf))

                        try:
                            stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                        except Exception:
                            pass

                        # Curve-level SE range (using feature_name for full feature)
                        fse = self._feature_se_impl(
                            g.feature_name,
                            phi_scale=not self._known_scale,
                        )
                        se_curve = fse["se_log_relativity"]
                        curve_se_min = float(np.min(se_curve))
                        curve_se_max = float(np.max(se_curve))

                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=True,
                            group_norm=float(np.linalg.norm(b_g)),
                            wald_chi2=stat,
                            wald_p=p_val,
                            ref_df=ref_df,
                            curve_se_min=curve_se_min,
                            curve_se_max=curve_se_max,
                            subgroup_type=g.subgroup_type,
                        )
                    )
                else:
                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=False,
                            group_norm=0.0,
                            subgroup_type=g.subgroup_type,
                        )
                    )

            elif isinstance(spec, Categorical):
                for i, level in enumerate(spec._non_base):
                    coef_val = float(b_g[i])
                    se_val = float(se_g[i])
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.name}[{level}]",
                            group=g.name,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                        )
                    )

            elif isinstance(spec, SplineCategorical):
                # Interaction spline group — Wood test (has penalty/SSP)
                if active:
                    stat = float("nan")
                    p_val = float("nan")
                    ref_df = float(g.size)

                    X_a, _, XtWX_inv, active_groups = self._active_info
                    R_a = self._active_R_factor
                    ag = next(a for a in active_groups if a.name == g.name)
                    if self._known_scale:
                        V_b_j = XtWX_inv[ag.sl, ag.sl]
                    else:
                        V_b_j = self.phi * XtWX_inv[ag.sl, ag.sl]

                    from superglm.wood_pvalue import wood_test_smooth

                    edf, edf1 = self._influence_edf
                    edf1_j = float(np.sum(edf1[ag.sl]))
                    X_j = R_a[:, ag.sl]
                    res_df = -1.0 if self._known_scale else float(self.n_obs - np.sum(edf))

                    try:
                        stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                    except Exception:
                        pass

                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=True,
                            group_norm=float(np.linalg.norm(b_g)),
                            wald_chi2=stat,
                            wald_p=p_val,
                            ref_df=ref_df,
                        )
                    )
                else:
                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=False,
                            group_norm=0.0,
                        )
                    )

            elif isinstance(spec, PolynomialCategorical):
                # Per-level polynomial group — Wald chi2 (no penalty/SSP)
                if active:
                    stat = float("nan")
                    p_val = float("nan")
                    ref_df = float(g.size)

                    _, _, XtWX_inv, active_groups = self._active_info
                    ag = next(a for a in active_groups if a.name == g.name)
                    if self._known_scale:
                        V_b_j = XtWX_inv[ag.sl, ag.sl]
                    else:
                        V_b_j = self.phi * XtWX_inv[ag.sl, ag.sl]

                    from scipy.stats import chi2 as chi2_dist

                    try:
                        stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                        p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                    except np.linalg.LinAlgError:
                        pass

                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=True,
                            group_norm=float(np.linalg.norm(b_g)),
                            wald_chi2=stat,
                            wald_p=p_val,
                            ref_df=ref_df,
                        )
                    )
                else:
                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=False,
                            group_norm=0.0,
                        )
                    )

            elif isinstance(spec, CategoricalInteraction):
                # Per-pair coefficient rows
                for i, (lev1, lev2) in enumerate(spec._pairs):
                    coef_val = float(b_g[i])
                    se_val = float(se_g[i])
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.name}[{lev1}:{lev2}]",
                            group=g.name,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                        )
                    )

            elif isinstance(spec, NumericCategorical):
                # Per-level slope coefficient rows
                for i, level in enumerate(spec._non_base):
                    coef_val = float(b_g[i])
                    se_val = float(se_g[i])
                    if spec._standardize:
                        coef_val = coef_val / spec._std
                        se_val = se_val / spec._std
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.name}[{level}]",
                            group=g.name,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                        )
                    )

            elif isinstance(spec, NumericInteraction | PolynomialInteraction):
                # Single or multi-column group — show group norm + Wald
                if active and g.size <= 4:
                    # Few enough coefficients to show individually
                    for i in range(g.size):
                        coef_val = float(b_g[i])
                        se_val = float(se_g[i])
                        z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                        rows.append(
                            _CoefRow(
                                name=f"{g.name}[{i}]" if g.size > 1 else g.name,
                                group=g.name,
                                coef=coef_val,
                                se=se_val,
                                z=z,
                                p=p,
                                ci_low=ci_lo,
                                ci_high=ci_hi,
                            )
                        )
                elif active:
                    # Too many columns — show as group summary
                    stat = float("nan")
                    p_val = float("nan")
                    ref_df = float(g.size)
                    _, _, XtWX_inv, active_groups = self._active_info
                    ag = next(a for a in active_groups if a.name == g.name)
                    if self._known_scale:
                        V_b_j = XtWX_inv[ag.sl, ag.sl]
                    else:
                        V_b_j = self.phi * XtWX_inv[ag.sl, ag.sl]
                    from scipy.stats import chi2 as chi2_dist

                    try:
                        stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                        p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                    except np.linalg.LinAlgError:
                        pass
                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=True,
                            group_norm=float(np.linalg.norm(b_g)),
                            wald_chi2=stat,
                            wald_p=p_val,
                            ref_df=ref_df,
                        )
                    )
                else:
                    rows.append(
                        _CoefRow(
                            name=g.name,
                            group=g.feature_name,
                            is_spline=True,
                            n_params=g.size,
                            active=False,
                            group_norm=0.0,
                        )
                    )

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
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_display, se_display, alpha)
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.name,
                        coef=coef_display,
                        se=se_display,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

            else:
                # Generic: show first coefficient
                coef_val = float(b_g[0])
                se_val = float(se_g[0]) if len(se_g) > 0 else 0.0
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

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
            model_info["tweedie_p_method"] = "Profile (exact)"

        coef_rows = self._build_coef_rows(alpha=alpha)

        return ModelSummary(data, model_info, coef_rows, alpha=alpha)
