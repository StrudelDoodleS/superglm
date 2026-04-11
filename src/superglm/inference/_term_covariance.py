"""Internal covariance, SE, and simultaneous-band helpers for term inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from superglm.distributions import _VARIANCE_FLOOR
from superglm.inference._term_helpers import _spline_se
from superglm.inference._term_types import _safe_exp

if TYPE_CHECKING:
    from superglm.distributions import Distribution
    from superglm.group_matrix import DesignMatrix
    from superglm.links import Link
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def compute_coef_covariance(
    dm: DesignMatrix,
    distribution: Distribution,
    link: Link,
    groups: list[GroupSlice],
    result: PIRLSResult,
    fit_weights: NDArray,
    fit_offset: NDArray | None,
    lambda2: float | dict[str, float],
    S_override: NDArray | None = None,
) -> tuple[NDArray, list[GroupSlice]]:
    """Phi-scaled Bayesian covariance for active coefficients."""
    from superglm.inference.covariance import _penalised_xtwx_inv_gram
    from superglm.links import stabilize_eta

    beta = result.beta
    eta = dm.matvec(beta) + result.intercept
    if fit_offset is not None:
        eta = eta + fit_offset
    from superglm.distributions import clip_mu

    eta = stabilize_eta(eta, link)
    mu = clip_mu(link.inverse(eta), distribution)
    V = distribution.variance(mu)
    dmu_deta = link.deriv_inverse(eta)
    W = fit_weights * dmu_deta**2 / np.maximum(V, _VARIANCE_FLOOR)

    XtWX_S_inv, XtWX_S_inv_aug, active_groups, _, _ = _penalised_xtwx_inv_gram(
        beta, W, dm.group_matrices, groups, lambda2, S_override=S_override
    )
    cov_features = result.phi * XtWX_S_inv_aug[1:, 1:]
    return cov_features, active_groups


def feature_se_from_cov(
    name: str,
    Cov_active: NDArray,
    active_groups: list[GroupSlice],
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    n_points: int = 200,
) -> NDArray:
    """Compute feature-level SEs from a precomputed covariance matrix."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == name]
    spec = specs.get(name) or interaction_specs.get(name)

    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            level_values = np.array([spec._level_to_value[lev] for lev in spec._ordered_levels])
            return _spline_se(
                spec._spline,
                name,
                beta,
                feature_groups,
                active_groups,
                Cov_active,
                x_eval=level_values,
            )
        beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
        if np.linalg.norm(beta_combined) < 1e-12:
            return np.zeros(len(spec._ordered_levels))
        active_subs = [ag for ag in active_groups if ag.feature_name == name]
        if not active_subs:
            return np.zeros(len(spec._ordered_levels))
        indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
        Cov_g = Cov_active[np.ix_(indices, indices)]
        if spec._R_inv is not None:
            Cov_orig = spec._R_inv @ Cov_g @ spec._R_inv.T
        else:
            Cov_orig = Cov_g
        se_nonbase = np.sqrt(np.maximum(np.diag(Cov_orig), 0.0))
        se_all = np.zeros(len(spec._ordered_levels))
        for i, lev in enumerate(spec._ordered_levels):
            if lev != spec._base_level:
                idx = spec._non_base.index(lev)
                se_all[i] = se_nonbase[idx]
        return se_all

    beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
    if np.linalg.norm(beta_combined) < 1e-12:
        if isinstance(spec, _SplineBase | Polynomial):
            return np.zeros(n_points)
        if isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        return np.zeros(1)

    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        if isinstance(spec, _SplineBase | Polynomial):
            return np.zeros(n_points)
        if isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        return np.zeros(1)

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]

    if isinstance(spec, _SplineBase):
        return _spline_se(
            spec,
            name,
            beta,
            feature_groups,
            active_groups,
            Cov_active,
            n_points,
        )

    if isinstance(spec, Polynomial):
        x_grid = np.linspace(spec._lo, spec._hi, n_points)
        M = spec.transform(x_grid)
        Q = M @ Cov_g
        return cast(NDArray, np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0)))

    if isinstance(spec, Categorical):
        se_nonbase = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
        se_all = np.zeros(len(spec._levels))
        for i, lev in enumerate(spec._levels):
            if lev != spec._base_level:
                idx = spec._non_base.index(lev)
                se_all[i] = se_nonbase[idx]
        return se_all

    if isinstance(spec, Numeric):
        return np.array([np.sqrt(max(Cov_g[0, 0], 0.0))])

    return cast(NDArray, np.sqrt(np.maximum(np.diag(Cov_g), 0.0)))


def simultaneous_bands(
    feature: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    covariance_fn,
    alpha: float = 0.05,
    n_sim: int = 10_000,
    n_points: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Simultaneous confidence bands for a spline feature."""
    from scipy.stats import norm

    from superglm.features.spline import _SplineBase

    spec = specs.get(feature)
    if not isinstance(spec, _SplineBase):
        raise TypeError(
            f"simultaneous_bands() only supports spline features, "
            f"got {type(spec).__name__} for '{feature}'."
        )

    Cov_active, active_groups = covariance_fn()
    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == feature]

    active_subs = [ag for ag in active_groups if ag.feature_name == feature]
    if not active_subs:
        raise ValueError(f"Feature '{feature}' is inactive (all coefficients zeroed).")

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]

    x_grid = np.linspace(spec._lo, spec._hi, n_points)
    B_grid = spec._raw_basis_matrix(x_grid)
    M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid

    active_cols = np.concatenate(
        [
            np.arange(g.start, g.end) - feature_groups[0].start
            for g in feature_groups
            if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
        ]
    )
    M = M[:, active_cols]

    Q = M @ Cov_g
    se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

    beta_g = np.concatenate(
        [
            beta[g.sl]
            for g in feature_groups
            if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
        ]
    )
    log_rel = M @ beta_g

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Cov_g + 1e-12 * np.eye(Cov_g.shape[0]))
    beta_sim = rng.standard_normal((n_sim, Cov_g.shape[0])) @ L.T
    f_sim = beta_sim @ M.T

    se_safe = np.maximum(se, 1e-20)
    T_sim = np.max(np.abs(f_sim) / se_safe[np.newaxis, :], axis=1)
    c_sim = float(np.quantile(T_sim, 1.0 - alpha))

    z = norm.ppf(1.0 - alpha / 2.0)

    return pd.DataFrame(
        {
            "x": x_grid,
            "log_relativity": log_rel,
            "relativity": _safe_exp(log_rel),
            "se": se,
            "ci_lower_pointwise": _safe_exp(log_rel - z * se),
            "ci_upper_pointwise": _safe_exp(log_rel + z * se),
            "ci_lower_simultaneous": _safe_exp(log_rel - c_sim * se),
            "ci_upper_simultaneous": _safe_exp(log_rel + c_sim * se),
        }
    )


__all__ = [
    "compute_coef_covariance",
    "feature_se_from_cov",
    "simultaneous_bands",
]
