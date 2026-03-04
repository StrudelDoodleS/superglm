"""SuperGLM: main model class."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from superglm.distributions import Distribution, resolve_distribution
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
        penalty: Penalty | str | None = None,
        lambda1: float | None = None,
        lambda2: float = 0.1,
        tweedie_p: float | None = None,
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
        self.penalty = self._resolve_penalty(penalty, lambda1)
        self.lambda2 = lambda2
        self.tweedie_p = tweedie_p
        self._splines = splines
        self._n_knots = n_knots
        self._degree = degree
        self._categorical_base = categorical_base
        self._standardize_numeric = standardize_numeric

        self._specs: dict[str, FeatureSpec] = {}
        self._feature_order: list[str] = []
        self._groups: list[GroupSlice] = []
        self._distribution: Distribution | None = None
        self._result: PIRLSResult | None = None
        self._dm: DesignMatrix | None = None

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
        self._distribution = resolve_distribution(self.family, tweedie_p=self.tweedie_p)

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
        if self._splines is not None and not self._specs:
            self._auto_detect_features(X, exposure)
        y, exposure, offset = self._build_design_matrix(X, y, exposure, offset)

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max(y, exposure) * 0.1

        self._result = fit_pirls(
            X=self._dm,
            y=y,
            weights=exposure,
            family=self._distribution,
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

    def metrics(self, X: pd.DataFrame, y, exposure=None, offset=None):
        """Compute comprehensive diagnostics for the fitted model.

        Returns a ModelMetrics object with information criteria, residuals,
        leverage, Cook's distance, etc.
        """
        from superglm.metrics import ModelMetrics

        return ModelMetrics(self, X, y, exposure, offset)

    def relativities(self) -> dict[str, pd.DataFrame]:
        """Extract plot-ready relativity DataFrames for all features.

        Returns dict keyed by feature name. Each DataFrame has columns
        ``relativity`` and ``log_relativity`` plus a type-specific index column:

        - Spline / Polynomial (has ``x``): ``x``, ``relativity``, ``log_relativity``
        - Categorical (has ``levels``): ``level``, ``relativity``, ``log_relativity``
        - Numeric (has ``relativity_per_unit``): ``label``, ``relativity``, ``log_relativity``
        """
        result: dict[str, pd.DataFrame] = {}
        for name in self._feature_order:
            raw = self.reconstruct_feature(name)
            if "x" in raw:
                # Spline or Polynomial
                result[name] = pd.DataFrame({
                    "x": raw["x"],
                    "relativity": raw["relativity"],
                    "log_relativity": raw["log_relativity"],
                })
            elif "levels" in raw:
                # Categorical
                levels = raw["levels"]
                rels = raw["relativities"]
                log_rels = raw["log_relativities"]
                result[name] = pd.DataFrame({
                    "level": levels,
                    "relativity": [rels[lv] for lv in levels],
                    "log_relativity": [log_rels[lv] for lv in levels],
                })
            elif "relativity_per_unit" in raw:
                # Numeric
                rel = raw["relativity_per_unit"]
                result[name] = pd.DataFrame({
                    "label": ["per_unit"],
                    "relativity": [rel],
                    "log_relativity": [np.log(rel)],
                })
        return result

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
        return np.exp(eta)
