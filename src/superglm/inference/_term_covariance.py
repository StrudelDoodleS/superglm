"""Internal covariance, SE, and simultaneous-band helpers for term inference."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
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


def _active_subgroup_columns(
    feature_name: Hashable,
    feature_groups: list[GroupSlice],
    active_subs: list[GroupSlice],
) -> NDArray:
    return np.concatenate(
        [
            np.arange(group.start, group.end) - feature_groups[0].start
            for group in feature_groups
            if any(
                active_group.feature_name == feature_name and active_group.name == group.name
                for active_group in active_subs
            )
        ]
    )


def feature_se_from_cov(
    name: Hashable,
    Cov_active: NDArray,
    active_groups: list[GroupSlice],
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: Mapping[Any, Any],
    interaction_specs: Mapping[Any, Any],
    n_points: int = 200,
) -> NDArray:
    """Compute feature-level SEs from a precomputed covariance matrix."""
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase

    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == name]
    spec = specs.get(name) or interaction_specs.get(name)

    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            return _spline_se(
                spec,
                name,
                beta,
                feature_groups,
                active_groups,
                Cov_active,
                x_eval=np.array(spec._ordered_levels, dtype=object),
                reference_x=np.array([spec._base_level], dtype=object),
            )
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

    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        if isinstance(spec, _SplineBase | Polynomial):
            return np.zeros(n_points)
        if isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        if isinstance(spec, RandomEffect):
            return np.zeros(len(spec._levels))
        if isinstance(spec, Piecewise):
            # One SE per KNOT, base included -- the same length as the term's
            # log_relativity vector.  The zeros(1) fallback below would leave a
            # dropped piecewise term's CI arrays a different length from its
            # values, which every downstream renderer zips against.
            return np.zeros(spec._knots.size)
        return np.zeros(1)

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])

    if isinstance(spec, RandomEffect):
        from superglm.inference.covariance import covariance_selected_diagonal

        variance = covariance_selected_diagonal(Cov_active, indices)
        return cast(NDArray, np.sqrt(np.maximum(variance, 0.0)))

    Cov_g = Cov_active[np.ix_(indices, indices)]

    if isinstance(spec, _SplineBase):
        return _spline_se(
            spec,
            name,
            beta,
            feature_groups,
            active_groups,
            Cov_active,
            n_points=n_points,
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

    if isinstance(spec, Piecewise):
        # ``_raw_basis_matrix`` at the knots is the identity, so restricting it
        # to the retained columns is the identity padded with a zero row at the
        # base knot.  Going through the basis rather than indexing the diagonal
        # keeps this branch a plain quadratic form: the base knot's SE is 0
        # because its contrast against itself is, not because it was special-cased.
        M = spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices]
        Q = M @ Cov_g
        return cast(NDArray, np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0)))

    if isinstance(spec, Numeric):
        return np.array([np.sqrt(max(Cov_g[0, 0], 0.0))])

    return cast(NDArray, np.sqrt(np.maximum(np.diag(Cov_g), 0.0)))


def piecewise_knot_covariance(
    name: str,
    Cov_active: NDArray,
    active_groups: list[GroupSlice],
    specs: dict[str, Any],
) -> NDArray | None:
    """Covariance of a Piecewise term's per-knot log relativities.

    ``(J+2, J+2)`` with the base row/column identically zero, built through
    the same raw-basis-at-the-knots map as ``feature_se_from_cov``'s SE
    branch, so ``sqrt(diag(V))`` is that SE vector.  Carried on
    ``TermInference.knot_covariance`` because the variance BETWEEN knots is
    the quadratic form of both adjacent hats with their covariance --
    ``var f(x) = h1^2 V11 + 2 h1 h2 V12 + h2^2 V22`` -- which pointwise knot
    SEs cannot reproduce.  Returns ``None`` when no subgroup is active.
    """
    spec = specs[name]
    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        return None
    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]
    M = spec._raw_basis_matrix(spec._knots)[:, spec._non_base_indices]
    return cast(NDArray, M @ Cov_g @ M.T)


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
    M = np.asarray(spec.transform(x_grid), dtype=np.float64)
    active_cols = _active_subgroup_columns(feature, feature_groups, active_subs)
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
    "piecewise_knot_covariance",
    "simultaneous_bands",
]
