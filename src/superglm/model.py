"""SuperGLM: main model class."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.distributions import Distribution, resolve_distribution
from superglm.links import Link, LogLink, resolve_link
from superglm.group_matrix import (
    DenseGroupMatrix,
    DesignMatrix,
    GroupMatrix,
    SparseGroupMatrix,
    SparseSSPGroupMatrix,
)
from superglm.penalties.base import Penalty
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.penalties.sparse_group_lasso import SparseGroupLasso
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FeatureSpec, GroupInfo, GroupSlice

logger = logging.getLogger(__name__)

_PENALTY_SHORTCUTS: dict[str, type[Penalty]] = {
    "group_lasso": GroupLasso,
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


class SuperGLM:
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
    ):
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

        self._specs: dict[str, FeatureSpec] = {}
        self._feature_order: list[str] = []
        self._groups: list[GroupSlice] = []
        self._distribution: Distribution | None = None
        self._link: Link | None = None
        self._result: PIRLSResult | None = None
        self._dm: DesignMatrix | None = None
        self._fit_weights: NDArray | None = None

        # Register explicit features dict
        if features is not None:
            for name, spec in features.items():
                self.add_feature(name, spec)

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

    def _auto_detect_features(self, X: pd.DataFrame, exposure: NDArray | None) -> None:
        """Auto-detect feature types from DataFrame columns."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        spline_cols = self._splines or []
        knots_map = self._resolve_knots(spline_cols)

        lines = ["SuperGLM features:"]
        for col in X.columns:
            if col in spline_cols:
                nk = knots_map[col]
                self.add_feature(col, Spline(n_knots=nk, degree=self._degree, penalty="ssp"))
                lines.append(f"  {col:<20s} → Spline(n_knots={nk}, degree={self._degree})")
            elif X[col].dtype.kind in ("O", "U") or isinstance(X[col].dtype, pd.CategoricalDtype):
                base = self._categorical_base
                if base == "most_exposed" and exposure is None:
                    base = "first"
                self.add_feature(col, Categorical(base=base))
                lines.append(f"  {col:<20s} → Categorical(base={base})")
            else:
                self.add_feature(col, Numeric(standardize=self._standardize_numeric))
                lines.append(f"  {col:<20s} → Numeric(standardize={self._standardize_numeric})")
        logger.info("\n".join(lines))

    def add_feature(self, name: str, spec: FeatureSpec) -> SuperGLM:
        if name in self._specs:
            raise ValueError(f"Feature already added: {name}")
        self._specs[name] = spec
        self._feature_order.append(name)
        return self

    def _build_design_matrix(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray,
        offset: NDArray | None,
    ) -> tuple[NDArray, NDArray, NDArray | None]:
        """Build features, groups, design matrix.

        Sets self._dm, self._groups, self._distribution.
        Returns (y, exposure, offset) as float64 arrays.
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
        resolved_nb_theta = self.nb_theta if isinstance(self.nb_theta, (int, float)) else None
        self._distribution = resolve_distribution(
            self.family, tweedie_p=self.tweedie_p, nb_theta=resolved_nb_theta
        )
        self._link = resolve_link(self.link, self._distribution)

        group_matrices: list[GroupMatrix] = []
        col_offset = 0
        self._groups = []

        for name in self._feature_order:
            spec = self._specs[name]
            x_col = np.asarray(X[name])
            info: GroupInfo = spec.build(x_col, exposure=exposure)

            if info.reparametrize and info.penalty_matrix is not None:
                R_inv = self._compute_R_inv(
                    info.columns,
                    info.penalty_matrix,
                    exposure,
                )
                if hasattr(spec, "set_reparametrisation"):
                    spec.set_reparametrisation(R_inv)
                # Factored sparse SSP: store B_sparse + R_inv separately
                if sp.issparse(info.columns):
                    gm: GroupMatrix = SparseSSPGroupMatrix(info.columns, R_inv)
                else:
                    gm = DenseGroupMatrix(info.columns @ R_inv)
            elif sp.issparse(info.columns):
                gm = SparseGroupMatrix(info.columns)
            else:
                gm = DenseGroupMatrix(info.columns)

            group_matrices.append(gm)
            weight = np.sqrt(info.n_cols)
            self._groups.append(
                GroupSlice(name=name, start=col_offset, end=col_offset + info.n_cols, weight=weight)
            )
            col_offset += info.n_cols

        self._dm = DesignMatrix(group_matrices, n, col_offset)
        return y, exposure, offset

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> SuperGLM:
        """Fit the model to data.

        Parameters
        ----------
        X : DataFrame
            Feature matrix with columns matching registered features.
        y : array-like
            Response variable.
        exposure : array-like, optional
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
        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)

        # Auto-estimate NB theta if requested
        if self.nb_theta == "auto":
            from superglm.nb_profile import estimate_nb_theta

            result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset)
            self.nb_theta = result.theta_hat
            logger.info(f"NB theta estimated: {result.theta_hat:.4f}")

        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = exposure  # store for covariance computation

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max(y, exposure) * 0.1

        # Invalidate cached covariance from previous fit
        self.__dict__.pop("_coef_covariance", None)

        self._result = fit_pirls(
            X=self._dm,
            y=y,
            weights=exposure,
            family=self._distribution,
            link=self._link,
            groups=self._groups,
            penalty=self.penalty,
            offset=offset,
        )
        return self

    def fit_path(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        *,
        n_lambda: int = 50,
        lambda_ratio: float = 1e-3,
        lambda_seq: NDArray | None = None,
    ) -> PathResult:
        """Fit a regularization path from lambda_max down to lambda_min.

        Warm-starts each lambda from the previous solution.
        """
        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)
        self._fit_weights = exposure
        self.__dict__.pop("_coef_covariance", None)
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
            )
            coef_path[i] = result.beta
            intercept_path[i] = result.intercept
            deviance_path[i] = result.deviance
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
        )

    def _compute_R_inv(self, B, omega, exposure):
        """Compute SSP reparametrisation matrix R_inv without forming B @ R_inv."""
        if sp.issparse(B):
            # B.multiply(exposure[:, None]) is sparse-safe
            G = np.asarray((B.multiply(exposure[:, None]).T @ B).todense()) / np.sum(exposure)
        else:
            G = (B * exposure[:, None]).T @ B / np.sum(exposure)
        M = G + self.lambda2 * omega + np.eye(omega.shape[0]) * 1e-8
        R = np.linalg.cholesky(M).T
        return np.linalg.inv(R)

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
            lmax = max(lmax, np.linalg.norm(grad[g.sl]) / g.weight)
        return lmax / n

    @property
    def result(self) -> PIRLSResult:
        if self._result is None:
            raise RuntimeError("Not fitted")
        return self._result

    def summary(self) -> dict[str, Any]:
        res = self.result
        out = {}
        for g in self._groups:
            bg = res.beta[g.sl]
            out[g.name] = {
                "active": bool(np.any(bg != 0)),
                "group_norm": float(np.linalg.norm(bg)),
                "n_params": g.size,
            }
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

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        res = self.result
        g = next(g for g in self._groups if g.name == name)
        return self._specs[name].reconstruct(res.beta[g.sl])

    @cached_property
    def _coef_covariance(self) -> tuple[NDArray, list[GroupSlice]]:
        """Phi-scaled covariance matrix for active coefficients.

        Returns (Cov_active, active_groups) where:
        - Cov_active: (p_active, p_active) = phi * (X'WX)^{-1}
        - active_groups: list of GroupSlice re-indexed to Cov_active columns
        """
        res = self.result
        beta = res.beta
        eta = np.clip(self._dm.matvec(beta) + res.intercept, -20, 20)
        mu = self._link.inverse(eta)
        V = self._distribution.variance(mu)
        dmu_deta = self._link.deriv_inverse(eta)
        weights = self._fit_weights
        W = weights * dmu_deta**2 / np.maximum(V, 1e-10)

        # Build active design and compute XtWX_inv
        active_cols = []
        active_groups = []
        col = 0
        for gm, g in zip(self._dm.group_matrices, self._groups):
            if np.linalg.norm(beta[g.sl]) > 1e-12:
                arr = gm.toarray()
                active_cols.append(arr)
                p_g = arr.shape[1]
                active_groups.append(
                    GroupSlice(name=g.name, start=col, end=col + p_g, weight=g.weight)
                )
                col += p_g

        if not active_cols:
            return np.empty((0, 0)), []

        X_a = np.hstack(active_cols)
        XtWX = X_a.T @ (X_a * W[:, None])
        XtWX[np.diag_indices_from(XtWX)] += 1e-8
        XtWX_inv = np.linalg.inv(XtWX)

        return res.phi * XtWX_inv, active_groups

    def estimate_p(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
        **kwargs,
    ):
        """Estimate Tweedie p via profile likelihood, refit, and return result.

        Thin wrapper around :func:`superglm.tweedie_profile.estimate_tweedie_p`.
        After estimation, sets ``self.tweedie_p`` to the optimised value and
        refits the model.

        Returns
        -------
        TweedieProfileResult
        """
        from superglm.tweedie_profile import estimate_tweedie_p

        result = estimate_tweedie_p(self, X, y, exposure=exposure, offset=offset, **kwargs)
        self.tweedie_p = result.p_hat
        self.fit(X, y, exposure=exposure, offset=offset)
        return result

    def estimate_theta(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
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

        result = estimate_nb_theta(self, X, y, exposure=exposure, offset=offset, **kwargs)
        self.nb_theta = result.theta_hat
        self.fit(X, y, exposure=exposure, offset=offset)
        return result

    def metrics(self, X: pd.DataFrame, y, exposure=None, offset=None):
        """Compute comprehensive diagnostics for the fitted model.

        Returns a ModelMetrics object with information criteria, residuals,
        leverage, Cook's distance, etc.
        """
        from superglm.metrics import ModelMetrics

        return ModelMetrics(self, X, y, exposure, offset)

    def relativities(self, with_se: bool = False) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features.

        Parameters
        ----------
        with_se : bool
            If True, add an ``se_log_relativity`` column to each DataFrame.
            Uses phi-scaled (quasi-likelihood) covariance.

        Returns dict keyed by feature name. Each DataFrame has columns
        ``relativity`` and ``log_relativity`` plus a type-specific index column:

        - Spline / Polynomial (has ``x``): ``x``, ``relativity``, ``log_relativity``
        - Categorical (has ``levels``): ``level``, ``relativity``, ``log_relativity``
        - Numeric (has ``relativity_per_unit``): ``label``, ``relativity``, ``log_relativity``
        """
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        # Pre-compute covariance if SEs requested
        if with_se:
            Cov_active, active_groups = self._coef_covariance

        result: dict[str, pd.DataFrame] = {}
        for name in self._feature_order:
            raw = self.reconstruct_feature(name)
            if "x" in raw:
                # Spline or Polynomial
                df = pd.DataFrame({
                    "x": raw["x"],
                    "relativity": raw["relativity"],
                    "log_relativity": raw["log_relativity"],
                })
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups, n_points=len(raw["x"])
                    )
                result[name] = df
            elif "levels" in raw:
                # Categorical
                levels = raw["levels"]
                rels = raw["relativities"]
                log_rels = raw["log_relativities"]
                df = pd.DataFrame({
                    "level": levels,
                    "relativity": [rels[lv] for lv in levels],
                    "log_relativity": [log_rels[lv] for lv in levels],
                })
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups
                    )
                result[name] = df
            elif "relativity_per_unit" in raw:
                # Numeric
                rel = raw["relativity_per_unit"]
                df = pd.DataFrame({
                    "label": ["per_unit"],
                    "relativity": [rel],
                    "log_relativity": [np.log(rel)],
                })
                if with_se:
                    df["se_log_relativity"] = self._feature_se_from_cov(
                        name, Cov_active, active_groups
                    )
                result[name] = df
        return result

    def _feature_se_from_cov(
        self,
        name: str,
        Cov_active: NDArray,
        active_groups: list[GroupSlice],
        n_points: int = 200,
    ) -> NDArray:
        """Compute feature-level SEs from the precomputed covariance matrix."""
        from superglm.features.categorical import Categorical
        from superglm.features.numeric import Numeric
        from superglm.features.spline import Spline

        beta = self.result.beta
        g = next(g for g in self._groups if g.name == name)
        spec = self._specs[name]

        # Inactive group: zeros
        if np.linalg.norm(beta[g.sl]) < 1e-12:
            if isinstance(spec, Spline):
                return np.zeros(n_points)
            elif isinstance(spec, Categorical):
                return np.zeros(len(spec._levels))
            else:
                return np.zeros(1)

        ag = next(ag for ag in active_groups if ag.name == name)
        Cov_g = Cov_active[ag.sl, ag.sl]

        if isinstance(spec, Spline):
            from scipy.interpolate import BSpline as BSpl

            x_grid = np.linspace(spec._lo, spec._hi, n_points)
            x_clip = np.clip(x_grid, spec._knots[0], spec._knots[-1])
            B_grid = BSpl.design_matrix(x_clip, spec._knots, spec.degree).toarray()
            M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid
            Q = M @ Cov_g
            return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

        elif isinstance(spec, Categorical):
            # Base level gets SE=0, non-base levels from diagonal
            se_nonbase = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
            # Build full SE array aligned with spec._levels
            se_all = np.zeros(len(spec._levels))
            for i, lev in enumerate(spec._levels):
                if lev != spec._base_level:
                    idx = spec._non_base.index(lev)
                    se_all[i] = se_nonbase[idx]
            return se_all

        elif isinstance(spec, Numeric):
            se_transformed = np.sqrt(max(Cov_g[0, 0], 0.0))
            if spec.standardize:
                return np.array([se_transformed / spec._std])
            return np.array([se_transformed])

        else:
            return np.sqrt(np.maximum(np.diag(Cov_g), 0.0))

    def plot_relativities(
        self,
        X: pd.DataFrame | None = None,
        exposure: NDArray | None = None,
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
        **kwargs
            Forwarded to :func:`superglm.plotting.plot_relativities`.
        """
        from superglm.plotting import plot_relativities

        return plot_relativities(self.relativities(), X=X, exposure=exposure, **kwargs)

    def discretization_impact(self, X, y, exposure=None, **kwargs):
        """Analyse the impact of discretizing spline/polynomial curves.

        See :func:`superglm.discretize.discretization_impact` for full docs.
        """
        from superglm.discretize import discretization_impact

        return discretization_impact(self, X, y, exposure, **kwargs)

    def predict(self, X: pd.DataFrame, offset: NDArray | None = None) -> NDArray:
        blocks = []
        for name in self._feature_order:
            spec = self._specs[name]
            blocks.append(spec.transform(np.asarray(X[name])))
        eta = np.hstack(blocks) @ self.result.beta + self.result.intercept
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=np.float64)
        return self._link.inverse(eta)
