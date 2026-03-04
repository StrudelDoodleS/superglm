"""SuperGLM: main model class."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm.distributions import Distribution, resolve_distribution
from superglm.penalties.base import Penalty
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.pirls import PIRLSResult, fit_pirls
from superglm.types import FeatureSpec, GroupInfo, GroupSlice


class SuperGLM:
    def __init__(
        self,
        family: str | Distribution = "poisson",
        link: str = "log",
        penalty: Penalty | None = None,
        lambda2: float = 0.1,
        tweedie_p: float | None = None,
    ):
        self.family = family
        self.link = link
        self.penalty = penalty if penalty is not None else GroupLasso()
        self.lambda2 = lambda2
        self.tweedie_p = tweedie_p

        self._specs: dict[str, FeatureSpec] = {}
        self._feature_order: list[str] = []
        self._groups: list[GroupSlice] = []
        self._distribution: Distribution | None = None
        self._result: PIRLSResult | None = None
        self._X_built: NDArray | None = None

    def add_feature(self, name: str, spec: FeatureSpec) -> SuperGLM:
        if name in self._specs:
            raise ValueError(f"Feature already added: {name}")
        self._specs[name] = spec
        self._feature_order.append(name)
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: NDArray,
        exposure: NDArray | None = None,
        offset: NDArray | None = None,
    ) -> SuperGLM:
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=np.float64)
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)
        self._distribution = resolve_distribution(self.family, tweedie_p=self.tweedie_p)

        blocks = []
        col_offset = 0
        self._groups = []

        for name in self._feature_order:
            spec = self._specs[name]
            x_col = np.asarray(X[name])
            info: GroupInfo = spec.build(x_col, exposure=exposure)

            if info.reparametrize and info.penalty_matrix is not None:
                columns, R_inv = self._reparametrise_ssp(
                    info.columns, info.penalty_matrix, exposure
                )
                if hasattr(spec, "set_reparametrisation"):
                    spec.set_reparametrisation(R_inv)
            else:
                columns = info.columns

            blocks.append(columns)
            weight = np.sqrt(info.n_cols)
            self._groups.append(
                GroupSlice(name=name, start=col_offset,
                           end=col_offset + info.n_cols, weight=weight)
            )
            col_offset += info.n_cols

        self._X_built = np.hstack(blocks)

        # Auto-calibrate lambda1 if not set
        if self.penalty.lambda1 is None:
            self.penalty.lambda1 = self._compute_lambda_max() * 0.1

        self._result = fit_pirls(
            X=self._X_built, y=y, weights=exposure,
            family=self._distribution, groups=self._groups,
            penalty=self.penalty, offset=offset,
        )
        return self

    def _reparametrise_ssp(self, B, omega, exposure):
        G = (B * exposure[:, None]).T @ B / len(exposure)
        M = G + self.lambda2 * omega + np.eye(omega.shape[0]) * 1e-8
        R = np.linalg.cholesky(M).T
        R_inv = np.linalg.inv(R)
        return B @ R_inv, R_inv

    def _compute_lambda_max(self):
        grad = self._X_built.T @ np.ones(self._X_built.shape[0])
        lmax = 0.0
        for g in self._groups:
            lmax = max(lmax, np.linalg.norm(grad[g.sl]) / g.weight)
        return lmax / self._X_built.shape[0]

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
            "intercept": res.intercept, "deviance": res.deviance,
            "phi": res.phi, "effective_df": res.effective_df,
            "n_iter": res.n_iter, "converged": res.converged,
            "lambda1": self.penalty.lambda1,
        }
        return out

    def reconstruct_feature(self, name: str) -> dict[str, Any]:
        res = self.result
        g = next(g for g in self._groups if g.name == name)
        return self._specs[name].reconstruct(res.beta[g.sl])

    def predict(self, X: pd.DataFrame, offset: NDArray | None = None) -> NDArray:
        blocks = []
        for name in self._feature_order:
            spec = self._specs[name]
            info = spec.build(np.asarray(X[name]))
            if info.reparametrize and hasattr(spec, "_R_inv") and spec._R_inv is not None:
                blocks.append(info.columns @ spec._R_inv)
            else:
                blocks.append(info.columns)
        eta = np.hstack(blocks) @ self.result.beta + self.result.intercept
        if offset is not None:
            eta = eta + np.asarray(offset, dtype=np.float64)
        return np.exp(eta)
