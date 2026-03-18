"""Inference helpers: covariance, SEs, confidence bands, drop1, relativities.

Extracted from model.py to keep the main class focused on fit/predict
orchestration. All functions take explicit state parameters; thin wrappers
on SuperGLM delegate here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from superglm.distributions import Distribution
    from superglm.group_matrix import DesignMatrix
    from superglm.links import Link
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


# ── Term Inference Result ────────────────────────────────────────


@dataclass(frozen=True)
class SplineMetadata:
    """Knot and basis metadata for a spline term."""

    kind: str  # e.g. "BasisSpline", "NaturalSpline", "CubicRegressionSpline"
    knot_strategy: str  # "uniform", "quantile", "quantile_tempered", "explicit"
    interior_knots: NDArray
    boundary: tuple[float, float]
    n_basis: int
    degree: int
    extrapolation: str  # "clip", "extend", "error"
    knot_alpha: float | None = None  # only for "quantile_tempered"


@dataclass(frozen=True)
class TermInference:
    """Per-term inference result.

    Holds the fitted curve (or levels/slope), uncertainty measures, and
    metadata for a single model term.  Returned by
    ``SuperGLM.term_inference()``.
    """

    # Identity
    name: str
    kind: str  # "spline", "categorical", "numeric", "polynomial"
    active: bool

    # Curve / levels / slope
    x: NDArray | None = None  # grid for spline/polynomial, None otherwise
    levels: list[str] | None = None  # for categorical
    log_relativity: NDArray | None = None
    relativity: NDArray | None = None

    # Uncertainty (pointwise)
    se_log_relativity: NDArray | None = None
    ci_lower: NDArray | None = None  # pointwise lower
    ci_upper: NDArray | None = None  # pointwise upper

    # Uncertainty (simultaneous) — only when simultaneous=True
    ci_lower_simultaneous: NDArray | None = None
    ci_upper_simultaneous: NDArray | None = None
    critical_value_simultaneous: float | None = None

    # Centering
    absorbs_intercept: bool = True
    centering_mode: str = "training_mean_zero_unweighted"

    # Smoothness / penalty
    edf: float | None = None
    smoothing_lambda: float | dict[str, float] | None = None

    # Spline-specific metadata
    spline: SplineMetadata | None = None

    # Monotonicity
    monotone: str | None = None  # "increasing", "decreasing", or None
    monotone_repaired: bool = False

    # CI alpha used
    alpha: float = 0.05

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a tidy DataFrame for plotting or export."""
        if self.kind in ("spline", "polynomial"):
            d: dict[str, Any] = {
                "x": self.x,
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            if self.ci_lower_simultaneous is not None:
                d["ci_lower_simultaneous"] = self.ci_lower_simultaneous
                d["ci_upper_simultaneous"] = self.ci_upper_simultaneous
            return pd.DataFrame(d)

        elif self.kind == "categorical":
            d = {
                "level": self.levels,
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            return pd.DataFrame(d)

        else:
            # numeric
            d = {
                "label": ["per_unit"],
                "log_relativity": self.log_relativity,
                "relativity": self.relativity,
            }
            if self.se_log_relativity is not None:
                d["se_log_relativity"] = self.se_log_relativity
            if self.ci_lower is not None:
                d["ci_lower"] = self.ci_lower
                d["ci_upper"] = self.ci_upper
            return pd.DataFrame(d)


@dataclass(frozen=True)
class InteractionInference:
    """Per-interaction inference result (lighter than TermInference)."""

    name: str
    kind: str  # "spline_categorical", "categorical", "numeric_categorical", etc.
    active: bool

    # For spline×categorical: per-level curves
    x: NDArray | None = None
    levels: list[str] | None = None
    per_level: dict[str, dict[str, NDArray]] | None = None

    # For categorical×categorical: per-pair
    pairs: list[tuple[str, str]] | None = None
    log_relativity: NDArray | dict[str, float] | None = None
    relativity: NDArray | dict[str, float] | None = None

    # For numeric×categorical: per-level slopes
    relativities_per_unit: dict[str, float] | None = None
    log_relativities_per_unit: dict[str, float] | None = None

    # For numeric×numeric: single product coefficient
    relativity_per_unit_unit: float | None = None
    coef: float | None = None


# ── Covariance ────────────────────────────────────────────────────


def compute_coef_covariance(
    dm: DesignMatrix,
    distribution: Distribution,
    link: Link,
    groups: list[GroupSlice],
    result: PIRLSResult,
    fit_weights: NDArray,
    fit_offset: NDArray | None,
    lambda2: float | dict[str, float],
) -> tuple[NDArray, list[GroupSlice]]:
    """Phi-scaled Bayesian covariance for active coefficients.

    Returns (Cov_active, active_groups) where:
    - Cov_active: (p_active, p_active) = phi * (X'WX + S)^{-1}
    - active_groups: list of GroupSlice re-indexed to Cov_active columns
    """
    from superglm.links import stabilize_eta
    from superglm.metrics import _penalised_xtwx_inv_gram

    beta = result.beta
    eta = dm.matvec(beta) + result.intercept
    if fit_offset is not None:
        eta = eta + fit_offset
    from superglm.distributions import clip_mu

    eta = stabilize_eta(eta, link)
    mu = clip_mu(link.inverse(eta), distribution)
    V = distribution.variance(mu)
    dmu_deta = link.deriv_inverse(eta)
    W = fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

    XtWX_S_inv, active_groups = _penalised_xtwx_inv_gram(
        beta, W, dm.group_matrices, groups, lambda2
    )
    return result.phi * XtWX_S_inv, active_groups


# ── Feature SEs ───────────────────────────────────────────────────


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

    # OrderedCategorical: delegate to spline or categorical logic
    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            # Delegate to internal spline's SE computation
            inner = spec._spline
            beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
            if np.linalg.norm(beta_combined) < 1e-12:
                return np.zeros(n_points)
            active_subs = [ag for ag in active_groups if ag.feature_name == name]
            if not active_subs:
                return np.zeros(n_points)
            indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
            Cov_g = Cov_active[np.ix_(indices, indices)]
            x_grid = np.linspace(inner._lo, inner._hi, n_points)
            B_grid = inner._raw_basis_matrix(x_grid)
            M = B_grid @ inner._R_inv if inner._R_inv is not None else B_grid
            active_cols = np.concatenate(
                [
                    np.arange(g.start, g.end) - feature_groups[0].start
                    for g in feature_groups
                    if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
                ]
            )
            M = M[:, active_cols]
            Q = M @ Cov_g
            return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))
        else:
            # Step mode: unwind reparametrisation for SEs
            beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
            if np.linalg.norm(beta_combined) < 1e-12:
                return np.zeros(len(spec._ordered_levels))
            active_subs = [ag for ag in active_groups if ag.feature_name == name]
            if not active_subs:
                return np.zeros(len(spec._ordered_levels))
            indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
            Cov_g = Cov_active[np.ix_(indices, indices)]
            # Cov_g is in reparametrised space; map to original via R_inv
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

    # Inactive feature: zeros (all subgroups zeroed)
    beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
    if np.linalg.norm(beta_combined) < 1e-12:
        if isinstance(spec, _SplineBase):
            return np.zeros(n_points)
        elif isinstance(spec, Polynomial):
            return np.zeros(n_points)
        elif isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        else:
            return np.zeros(1)

    # Gather covariance blocks from all active subgroups
    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        if isinstance(spec, _SplineBase):
            return np.zeros(n_points)
        elif isinstance(spec, Polynomial):
            return np.zeros(n_points)
        elif isinstance(spec, Categorical):
            return np.zeros(len(spec._levels))
        else:
            return np.zeros(1)

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]

    if isinstance(spec, _SplineBase):
        x_grid = np.linspace(spec._lo, spec._hi, n_points)
        B_grid = spec._raw_basis_matrix(x_grid)
        M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid

        # For select=True: only use columns for active subgroups
        active_cols = np.concatenate(
            [
                np.arange(g.start, g.end) - feature_groups[0].start
                for g in feature_groups
                if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
            ]
        )
        M = M[:, active_cols]

        Q = M @ Cov_g
        return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

    elif isinstance(spec, Polynomial):
        x_grid = np.linspace(spec._lo, spec._hi, n_points)
        M = spec.transform(x_grid)
        Q = M @ Cov_g
        return np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

    elif isinstance(spec, Categorical):
        se_nonbase = np.sqrt(np.maximum(np.diag(Cov_g), 0.0))
        se_all = np.zeros(len(spec._levels))
        for i, lev in enumerate(spec._levels):
            if lev != spec._base_level:
                idx = spec._non_base.index(lev)
                se_all[i] = se_nonbase[idx]
        return se_all

    elif isinstance(spec, Numeric):
        return np.array([np.sqrt(max(Cov_g[0, 0], 0.0))])

    else:
        return np.sqrt(np.maximum(np.diag(Cov_g), 0.0))


# ── Simultaneous Bands ────────────────────────────────────────────


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
    """Simultaneous confidence bands for a spline feature.

    Uses the Wood (2006) simulation approach: draws from the posterior
    MVN(0, Cov_g), computes the supremum of the standardised deviation
    across the curve, and returns the (1-alpha) quantile as the critical
    value for the simultaneous band.

    Parameters
    ----------
    feature : str
        Name of a spline feature.
    result : PIRLSResult
        Fitted model result.
    groups : list[GroupSlice]
        Group definitions from the fitted model.
    specs : dict
        Feature specs dict.
    covariance_fn : callable
        Zero-arg callable returning ``(Cov_active, active_groups)``.
    alpha : float
        Significance level (default 0.05 for 95% bands).
    n_sim : int
        Number of posterior simulations.
    n_points : int
        Grid size for evaluating the curve.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: x, log_relativity, relativity, se,
        ci_lower_pointwise, ci_upper_pointwise,
        ci_lower_simultaneous, ci_upper_simultaneous.
    """
    from scipy.stats import norm

    from superglm.features.spline import _SplineBase

    spec = specs.get(feature)
    if not isinstance(spec, _SplineBase):
        raise TypeError(
            f"simultaneous_bands() only supports spline features, "
            f"got {type(spec).__name__} for '{feature}'."
        )

    # Get covariance
    Cov_active, active_groups = covariance_fn()
    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == feature]

    active_subs = [ag for ag in active_groups if ag.feature_name == feature]
    if not active_subs:
        raise ValueError(f"Feature '{feature}' is inactive (all coefficients zeroed).")

    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]

    # Build basis evaluation matrix
    x_grid = np.linspace(spec._lo, spec._hi, n_points)
    B_grid = spec._raw_basis_matrix(x_grid)
    M = B_grid @ spec._R_inv if spec._R_inv is not None else B_grid

    # For select=True: only use columns for active subgroups
    active_cols = np.concatenate(
        [
            np.arange(g.start, g.end) - feature_groups[0].start
            for g in feature_groups
            if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
        ]
    )
    M = M[:, active_cols]

    # Pointwise SEs
    Q = M @ Cov_g
    se = np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0))

    # Log-relativity on grid
    beta_g = np.concatenate(
        [
            beta[g.sl]
            for g in feature_groups
            if any(ag.feature_name == feature and ag.name == g.name for ag in active_subs)
        ]
    )
    log_rel = M @ beta_g

    # Simultaneous critical value via simulation (Wood 2006)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Cov_g + 1e-12 * np.eye(Cov_g.shape[0]))
    beta_sim = rng.standard_normal((n_sim, Cov_g.shape[0])) @ L.T
    f_sim = beta_sim @ M.T  # (n_sim, n_points)

    se_safe = np.maximum(se, 1e-20)
    T_sim = np.max(np.abs(f_sim) / se_safe[np.newaxis, :], axis=1)
    c_sim = float(np.quantile(T_sim, 1.0 - alpha))

    z = norm.ppf(1.0 - alpha / 2.0)

    return pd.DataFrame(
        {
            "x": x_grid,
            "log_relativity": log_rel,
            "relativity": np.exp(log_rel),
            "se": se,
            "ci_lower_pointwise": np.exp(log_rel - z * se),
            "ci_upper_pointwise": np.exp(log_rel + z * se),
            "ci_lower_simultaneous": np.exp(log_rel - c_sim * se),
            "ci_upper_simultaneous": np.exp(log_rel + c_sim * se),
        }
    )


# ── Relativities ──────────────────────────────────────────────────


def relativities(
    feature_order: list[str],
    interaction_order: list[str],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    groups: list[GroupSlice],
    result: PIRLSResult,
    *,
    with_se: bool = False,
    covariance_fn=None,
) -> dict[str, pd.DataFrame]:
    """Extract plot-ready relativity DataFrames for all features.

    Parameters
    ----------
    feature_order : list[str]
        Ordered feature names.
    interaction_order : list[str]
        Ordered interaction names.
    specs : dict
        Feature specs.
    interaction_specs : dict
        Interaction specs.
    groups : list[GroupSlice]
        Group definitions.
    result : PIRLSResult
        Fitted model result.
    with_se : bool
        If True, add ``se_log_relativity`` column. Requires *covariance_fn*.
    covariance_fn : callable, optional
        Zero-arg callable returning ``(Cov_active, active_groups)``.

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    if with_se:
        Cov_active, active_groups = covariance_fn()

    def _feature_groups(name: str) -> list[GroupSlice]:
        return [g for g in groups if g.feature_name == name]

    def _reconstruct(name: str) -> dict[str, Any]:
        fgroups = _feature_groups(name)
        beta_combined = np.concatenate([result.beta[g.sl] for g in fgroups])
        if name in specs:
            return specs[name].reconstruct(beta_combined)
        if name in interaction_specs:
            return interaction_specs[name].reconstruct(beta_combined)
        raise KeyError(f"Feature not found: {name}")

    out: dict[str, pd.DataFrame] = {}
    for name in feature_order:
        raw = _reconstruct(name)
        if "x" in raw:
            # Spline or Polynomial
            df = pd.DataFrame(
                {
                    "x": raw["x"],
                    "relativity": raw["relativity"],
                    "log_relativity": raw["log_relativity"],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                    n_points=len(raw["x"]),
                )
            out[name] = df
        elif "levels" in raw:
            # Categorical
            levels = raw["levels"]
            rels = raw["relativities"]
            log_rels = raw["log_relativities"]
            df = pd.DataFrame(
                {
                    "level": levels,
                    "relativity": [rels[lv] for lv in levels],
                    "log_relativity": [log_rels[lv] for lv in levels],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
            out[name] = df
        elif "relativity_per_unit" in raw:
            # Numeric
            rel = raw["relativity_per_unit"]
            df = pd.DataFrame(
                {
                    "label": ["per_unit"],
                    "relativity": [rel],
                    "log_relativity": [np.log(rel)],
                }
            )
            if with_se:
                df["se_log_relativity"] = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
            out[name] = df

    # Interaction relativities — dispatch on reconstruct dict keys
    for iname in interaction_order:
        raw = _reconstruct(iname)

        if "per_level" in raw and "x" in raw:
            # SplineCategorical / PolynomialCategorical: per-level curves
            for level in raw["levels"]:
                level_data = raw["per_level"][level]
                key = f"{iname}[{level}]"
                df = pd.DataFrame(
                    {
                        "x": raw["x"],
                        "relativity": level_data["relativity"],
                        "log_relativity": level_data["log_relativity"],
                    }
                )
                out[key] = df

        elif "pairs" in raw:
            # CategoricalInteraction: per-pair relativities
            pairs_labels = [f"{l1}:{l2}" for l1, l2 in raw["pairs"]]
            rels = raw["relativities"]
            log_rels = raw["log_relativities"]
            df = pd.DataFrame(
                {
                    "level": pairs_labels,
                    "relativity": [rels[k] for k in pairs_labels],
                    "log_relativity": [log_rels[k] for k in pairs_labels],
                }
            )
            out[iname] = df

        elif "relativities_per_unit" in raw:
            # NumericCategorical: per-level slope relativities
            levels = raw["levels"]
            rels = raw["relativities_per_unit"]
            log_rels = raw["log_relativities_per_unit"]
            df = pd.DataFrame(
                {
                    "level": levels,
                    "relativity_per_unit": [rels[lv] for lv in levels],
                    "log_relativity_per_unit": [log_rels[lv] for lv in levels],
                }
            )
            out[iname] = df

        elif "relativity_per_unit_unit" in raw:
            # NumericInteraction: single product coefficient
            b_orig = raw["coef"]
            df = pd.DataFrame(
                {
                    "label": ["per_unit_unit"],
                    "relativity": [raw["relativity_per_unit_unit"]],
                    "log_relativity": [b_orig],
                }
            )
            out[iname] = df

        elif "x1" in raw and "x2" in raw:
            # PolynomialInteraction: 2D surface — store raw dict
            # (doesn't fit 1D DataFrame; use reconstruct_feature() directly)
            pass

    return out


# ── Drop-one Analysis ─────────────────────────────────────────────


def drop1(
    model,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    test: str = "Chisq",
) -> pd.DataFrame:
    """Drop-one deviance analysis for each feature.

    For each feature, refits the model with that feature (and any
    dependent interactions) removed, keeping the same penalty
    configuration. Compares deviances via a chi-squared or F test
    using effective degrees of freedom.

    .. note::

       This is an *approximate* deviance comparison, not a classical
       likelihood ratio test. P-values use effective df (hat matrix
       trace) rather than parametric df and should be treated as
       approximate guides, not exact tests.

       After ``fit_reml()``, reduced models inherit the full model's
       smoothing parameters as fixed values — REML is **not**
       re-run for each reduced model. This is computationally
       practical and follows the spirit of ``mgcv::anova.gam``,
       but means the comparison conditions on the full model's
       smoothing selection.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : DataFrame
        Feature matrix (same as used for fitting).
    y : array-like
        Response variable.
    exposure : array-like, optional
        Frequency weights.
    offset : array-like, optional
        Offset added to the linear predictor.
    test : {"Chisq", "F"}
        ``"Chisq"`` for known-scale families (Poisson).
        ``"F"`` for estimated-scale families (Gamma, NB2, Tweedie).

    Returns
    -------
    pd.DataFrame
        Rows sorted by p-value with columns: feature, deviance_full,
        deviance_reduced, delta_deviance, delta_df, statistic, p_value.
    """
    from scipy.stats import chi2
    from scipy.stats import f as f_dist

    if model._result is None:
        raise RuntimeError("Model must be fitted before calling drop1().")

    dev_full = model._result.deviance
    edf_full = model._result.effective_df
    n = len(y) if not hasattr(y, "__len__") else len(y)
    phi = model._result.phi

    rows = []
    for name in model._feature_order:
        # Identify dependent interactions
        drop_set = {name}
        for iname in model._interaction_order:
            ispec = model._interaction_specs[iname]
            p1, p2 = ispec.parent_names
            if p1 == name or p2 == name:
                drop_set.add(iname)

        remaining = [f for f in model._feature_order if f not in drop_set]

        if not remaining:
            # Intercept-only model: compute null deviance directly
            from superglm.distributions import Binomial, Gaussian, clip_mu
            from superglm.links import stabilize_eta

            y_arr = np.asarray(y, dtype=np.float64)
            w = (
                np.ones(n, dtype=np.float64)
                if exposure is None
                else np.asarray(exposure, dtype=np.float64)
            )
            y_mean = float(np.average(y_arr, weights=w))
            if isinstance(model._distribution, Binomial):
                y_mean = np.clip(y_mean, 1e-3, 1 - 1e-3)
            elif isinstance(model._distribution, Gaussian):
                y_mean = float(y_mean)
            else:
                y_mean = max(y_mean, 1e-10)

            if offset is not None:
                offset_arr = np.asarray(offset, dtype=np.float64)
                b0 = float(model._link.link(np.atleast_1d(y_mean))[0]) - np.average(
                    offset_arr, weights=w
                )
                eta0 = stabilize_eta(b0 + offset_arr, model._link)
                null_mu = clip_mu(model._link.inverse(eta0), model._distribution)
            else:
                null_mu = np.full(n, y_mean)
            dev_reduced = float(np.sum(w * model._distribution.deviance_unit(y_arr, null_mu)))
            edf_reduced = 1.0  # intercept only
        else:
            reduced = model._clone_without_features(drop_set)
            reduced.fit(X, y, exposure=exposure, offset=offset)
            dev_reduced = reduced.result.deviance
            edf_reduced = reduced.result.effective_df
        delta_dev = dev_reduced - dev_full
        delta_df = max(edf_full - edf_reduced, 1e-4)

        if test == "F":
            stat = (delta_dev / delta_df) / phi
            resid_df = max(n - edf_full, 1.0)
            p_value = float(f_dist.sf(stat, delta_df, resid_df))
        else:
            stat = delta_dev
            p_value = float(chi2.sf(stat, delta_df))

        rows.append(
            {
                "feature": name,
                "deviance_full": dev_full,
                "deviance_reduced": dev_reduced,
                "delta_deviance": delta_dev,
                "delta_df": delta_df,
                "statistic": stat,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ── Refit Unpenalised ─────────────────────────────────────────────


def refit_unpenalised(
    model,
    X: pd.DataFrame,
    y: NDArray,
    exposure: NDArray | None = None,
    offset: NDArray | None = None,
    *,
    keep_smoothing: bool = True,
):
    """Refit the model with only the active features and no selection penalty.

    After a penalized fit that selected features via group lasso, this
    refits with ``lambda1=0`` on only the active features, removing
    the shrinkage bias from L1 selection.

    When ``keep_smoothing=True``, the smoothing penalty (lambda2 or
    REML-estimated lambdas) is inherited as fixed values — smoothing
    parameters are **not** re-optimized via REML on the reduced model.

    Parameters
    ----------
    model : SuperGLM
        A fitted SuperGLM model.
    X : DataFrame
        Feature matrix.
    y : array-like
        Response variable.
    exposure : array-like, optional
        Frequency weights.
    offset : array-like, optional
        Offset added to the linear predictor.
    keep_smoothing : bool
        If True (default), keep the smoothing penalty (lambda2 or
        REML-estimated lambdas). If False, set lambda2=0 for a
        fully unpenalised refit.

    Returns
    -------
    SuperGLM
        A new fitted model with only the active features.
    """
    if model._result is None:
        raise RuntimeError("Model must be fitted before calling refit_unpenalised().")

    beta = model._result.beta

    # Identify inactive features
    inactive = set()
    for name in model._feature_order:
        fgroups = [g for g in model._groups if g.feature_name == name]
        if all(np.linalg.norm(beta[g.sl]) < 1e-12 for g in fgroups):
            inactive.add(name)

    # Also drop interactions whose parents are inactive
    for iname in model._interaction_order:
        ispec = model._interaction_specs[iname]
        p1, p2 = ispec.parent_names
        if p1 in inactive or p2 in inactive:
            inactive.add(iname)

    lam2: float | dict[str, float] | None
    if not keep_smoothing:
        lam2 = 0.0
    else:
        lam2 = ...  # sentinel: use original lambda2 / REML lambdas

    new_model = model._clone_without_features(inactive, lambda1=0.0, lambda2=lam2)
    new_model.fit(X, y, exposure=exposure, offset=offset)
    return new_model


# ── Term Inference ───────────────────────────────────────────────


def term_inference(
    name: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    covariance_fn,
    reml_lambdas: dict[str, float] | None,
    lambda2: float,
    group_edf: dict[str, float] | None = None,
    with_se: bool = True,
    simultaneous: bool = False,
    n_points: int = 200,
    alpha: float = 0.05,
    n_sim: int = 10_000,
    seed: int = 42,
) -> TermInference | InteractionInference:
    """Build a per-term inference object.

    Parameters
    ----------
    name : str
        Feature or interaction name.
    result : PIRLSResult
        Fitted model result.
    groups : list[GroupSlice]
        Group definitions from the fitted model.
    specs, interaction_specs : dict
        Feature and interaction specs.
    covariance_fn : callable
        Zero-arg callable returning ``(Cov_active, active_groups)``.
    reml_lambdas : dict or None
        REML-estimated per-group lambdas (from model._reml_lambdas).
    lambda2 : float
        Global smoothing penalty.
    group_edf : dict[str, float] or None
        Per-group effective degrees of freedom (keyed by group name).
    with_se : bool
        Compute standard errors and pointwise CIs.
    simultaneous : bool
        Compute simultaneous bands (spline only, requires with_se).
    n_points : int
        Grid size for spline/polynomial curves.
    alpha : float
        Significance level for CIs.
    n_sim : int
        Number of simulations for simultaneous bands.
    seed : int
        Random seed for simultaneous bands.

    Returns
    -------
    TermInference or InteractionInference
    """
    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase

    beta = result.beta
    feature_groups = [g for g in groups if g.feature_name == name]

    # ── Ambiguity check ───────────────────────────────────────────
    if name in specs and name in interaction_specs:
        raise ValueError(
            f"Ambiguous name {name!r}: exists as both a main effect "
            f"and an interaction. Use the feature or interaction spec "
            f"directly to disambiguate."
        )

    # ── Interaction dispatch ─────────────────────────────────────
    if name in interaction_specs:
        return _interaction_inference(
            name,
            result=result,
            groups=groups,
            interaction_specs=interaction_specs,
        )

    spec = specs.get(name)
    if spec is None:
        raise KeyError(f"Feature not found: {name}")

    # Check active
    beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
    active = bool(np.linalg.norm(beta_combined) > 1e-12)

    # Covariance (lazy, only if needed)
    Cov_active = active_groups_cov = None
    if with_se and active:
        Cov_active, active_groups_cov = covariance_fn()

    # Per-group edf
    edf = _compute_term_edf(name, feature_groups, group_edf)

    # Per-group lambda
    lam = _resolve_term_lambda(name, feature_groups, reml_lambdas, lambda2)

    z_alpha = float(__import__("scipy").stats.norm.ppf(1.0 - alpha / 2.0))

    # ── OrderedCategorical ────────────────────────────────────────
    if isinstance(spec, OrderedCategorical):
        if spec.basis == "spline":
            # Spline mode: delegate to internal spline for curve
            inner = spec._spline
            raw = spec.reconstruct(beta_combined)
            x_grid = raw["x"]
            log_rel = raw["log_relativity"]
            rel = raw["relativity"]

            se = ci_lo = ci_hi = None
            if with_se and active and Cov_active is not None:
                se = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups_cov,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                    n_points=n_points,
                )
                ci_lo = np.exp(log_rel - z_alpha * se)
                ci_hi = np.exp(log_rel + z_alpha * se)

            return TermInference(
                name=name,
                kind="spline",
                active=active,
                x=x_grid,
                log_relativity=log_rel,
                relativity=rel,
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                absorbs_intercept=inner.absorbs_intercept,
                edf=edf,
                smoothing_lambda=lam,
                spline=_build_spline_metadata(inner),
                alpha=alpha,
            )
        else:
            # Step mode: categorical-style output
            raw = spec.reconstruct(beta_combined)
            levels = raw["levels"]
            log_rels = np.array([raw["log_relativities"][lv] for lv in levels])
            rels = np.array([raw["relativities"][lv] for lv in levels])

            se = ci_lo = ci_hi = None
            if with_se and active and Cov_active is not None:
                se = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups_cov,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
                ci_lo = np.exp(log_rels - z_alpha * se)
                ci_hi = np.exp(log_rels + z_alpha * se)

            return TermInference(
                name=name,
                kind="categorical",
                active=active,
                levels=levels,
                log_relativity=log_rels,
                relativity=rels,
                se_log_relativity=se,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                absorbs_intercept=False,
                centering_mode="base_level",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            )

    # ── Spline ───────────────────────────────────────────────────
    if isinstance(spec, _SplineBase):
        raw = spec.reconstruct(beta_combined, n_points=n_points)
        x_grid = raw["x"]
        log_rel = raw["log_relativity"]
        rel = raw["relativity"]

        se = ci_lo = ci_hi = None
        ci_lo_sim = ci_hi_sim = c_sim = None

        if with_se and active and Cov_active is not None:
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
                n_points=n_points,
            )
            ci_lo = np.exp(log_rel - z_alpha * se)
            ci_hi = np.exp(log_rel + z_alpha * se)

            if simultaneous:
                bands = simultaneous_bands(
                    name,
                    result=result,
                    groups=groups,
                    specs=specs,
                    covariance_fn=covariance_fn,
                    alpha=alpha,
                    n_sim=n_sim,
                    n_points=n_points,
                    seed=seed,
                )
                ci_lo_sim = bands["ci_lower_simultaneous"].values
                ci_hi_sim = bands["ci_upper_simultaneous"].values
                # Back out the critical value: ci_upper_sim = exp(log_rel + c*se)
                safe_se = np.maximum(se, 1e-20)
                c_vals = (np.log(ci_hi_sim) - log_rel) / safe_se
                c_sim = float(np.median(c_vals[safe_se > 1e-15]))

        spline_meta = _build_spline_metadata(spec)

        return TermInference(
            name=name,
            kind="spline",
            active=active,
            x=x_grid,
            log_relativity=log_rel,
            relativity=rel,
            se_log_relativity=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            ci_lower_simultaneous=ci_lo_sim,
            ci_upper_simultaneous=ci_hi_sim,
            critical_value_simultaneous=c_sim,
            absorbs_intercept=spec.absorbs_intercept,
            edf=edf,
            smoothing_lambda=lam,
            spline=spline_meta,
            monotone=getattr(spec, "monotone", None),
            monotone_repaired=False,  # caller can override if repairs exist
            alpha=alpha,
        )

    # ── Categorical ──────────────────────────────────────────────
    elif isinstance(spec, Categorical):
        raw = spec.reconstruct(beta_combined)
        levels = raw["levels"]
        log_rels = np.array([raw["log_relativities"][lv] for lv in levels])
        rels = np.array([raw["relativities"][lv] for lv in levels])

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            )
            ci_lo = np.exp(log_rels - z_alpha * se)
            ci_hi = np.exp(log_rels + z_alpha * se)

        return TermInference(
            name=name,
            kind="categorical",
            active=active,
            levels=levels,
            log_relativity=log_rels,
            relativity=rels,
            se_log_relativity=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            absorbs_intercept=False,
            centering_mode="base_level",
            edf=edf,
            smoothing_lambda=lam,
            alpha=alpha,
        )

    # ── Polynomial ───────────────────────────────────────────────
    elif isinstance(spec, Polynomial):
        raw = spec.reconstruct(beta_combined)
        x_grid = raw["x"]
        log_rel = raw["log_relativity"]
        rel = raw["relativity"]

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
                n_points=n_points,
            )
            ci_lo = np.exp(log_rel - z_alpha * se)
            ci_hi = np.exp(log_rel + z_alpha * se)

        return TermInference(
            name=name,
            kind="polynomial",
            active=active,
            x=x_grid,
            log_relativity=log_rel,
            relativity=rel,
            se_log_relativity=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            absorbs_intercept=True,
            edf=edf,
            smoothing_lambda=lam,
            alpha=alpha,
        )

    # ── Numeric ──────────────────────────────────────────────────
    elif isinstance(spec, Numeric):
        raw = spec.reconstruct(beta_combined)
        log_rel = np.array([np.log(raw["relativity_per_unit"])])
        rel = np.array([raw["relativity_per_unit"]])

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            )
            ci_lo = np.exp(log_rel - z_alpha * se)
            ci_hi = np.exp(log_rel + z_alpha * se)

        return TermInference(
            name=name,
            kind="numeric",
            active=active,
            log_relativity=log_rel,
            relativity=rel,
            se_log_relativity=se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            absorbs_intercept=False,
            centering_mode="none",
            edf=edf,
            smoothing_lambda=lam,
            alpha=alpha,
        )

    else:
        raise TypeError(f"Unknown feature type: {type(spec).__name__}")


def _build_spline_metadata(spec) -> SplineMetadata:
    """Extract spline knot/basis metadata from a fitted spline spec."""
    knot_alpha = None
    if getattr(spec, "_knot_strategy_actual", None) == "quantile_tempered":
        knot_alpha = spec.knot_alpha

    return SplineMetadata(
        kind=type(spec).__name__,
        knot_strategy=spec._knot_strategy_actual,
        interior_knots=spec.fitted_knots,
        boundary=spec.fitted_boundary,
        n_basis=spec._n_basis,
        degree=spec.degree,
        extrapolation=spec.extrapolation,
        knot_alpha=knot_alpha,
    )


def _compute_term_edf(
    name: str,
    feature_groups: list[GroupSlice],
    group_edf: dict[str, float] | None,
) -> float | None:
    """Sum per-group edf for a feature term."""
    if group_edf is None:
        return None
    total = 0.0
    for g in feature_groups:
        if g.name in group_edf:
            total += group_edf[g.name]
    return total


def _resolve_term_lambda(
    name: str,
    feature_groups: list[GroupSlice],
    reml_lambdas: dict[str, float] | None,
    lambda2: float,
) -> float | dict[str, float] | None:
    """Resolve the smoothing lambda for a term."""
    if reml_lambdas is not None:
        group_lams = {}
        for g in feature_groups:
            if g.name in reml_lambdas:
                group_lams[g.name] = reml_lambdas[g.name]
        if len(group_lams) == 1:
            return next(iter(group_lams.values()))
        if group_lams:
            return group_lams
    return lambda2


def spline_group_enrichment(
    group_name: str,
    spec,
    group_edf: dict[str, float] | None,
    reml_lambdas: dict[str, float] | None,
    lambda2: float | dict | None,
) -> dict[str, Any]:
    """Return spline metadata dict for a single group.

    Shared by ``model.diagnostics()`` and ``metrics._build_coef_rows()``
    so both surfaces emit identical spline metadata.

    Returns
    -------
    dict with keys: edf, smoothing_lambda, spline_kind, knot_strategy, boundary.
    """
    edf = group_edf.get(group_name) if group_edf else None
    if reml_lambdas and group_name in reml_lambdas:
        lam: float | None = reml_lambdas[group_name]
    else:
        lam = float(lambda2) if isinstance(lambda2, int | float) else None
    return {
        "edf": edf,
        "smoothing_lambda": lam,
        "spline_kind": type(spec).__name__,
        "knot_strategy": getattr(spec, "_knot_strategy_actual", None),
        "boundary": getattr(spec, "fitted_boundary", None),
    }


def _interaction_inference(
    name: str,
    *,
    result: PIRLSResult,
    groups: list[GroupSlice],
    interaction_specs: dict[str, Any],
) -> InteractionInference:
    """Build an InteractionInference from a fitted interaction."""
    ispec = interaction_specs[name]
    feature_groups = [g for g in groups if g.feature_name == name]
    beta_combined = np.concatenate([result.beta[g.sl] for g in feature_groups])
    active = bool(np.linalg.norm(beta_combined) > 1e-12)

    raw = ispec.reconstruct(beta_combined)

    # SplineCategorical / PolynomialCategorical
    if "per_level" in raw and "x" in raw:
        return InteractionInference(
            name=name,
            kind="spline_categorical",
            active=active,
            x=raw["x"],
            levels=raw["levels"],
            per_level=raw["per_level"],
        )

    # CategoricalInteraction
    if "pairs" in raw:
        return InteractionInference(
            name=name,
            kind="categorical",
            active=active,
            pairs=raw["pairs"],
            log_relativity=raw["log_relativities"],
            relativity=raw["relativities"],
        )

    # NumericCategorical
    if "relativities_per_unit" in raw:
        return InteractionInference(
            name=name,
            kind="numeric_categorical",
            active=active,
            levels=raw["levels"],
            relativities_per_unit=raw["relativities_per_unit"],
            log_relativities_per_unit=raw["log_relativities_per_unit"],
        )

    # NumericInteraction
    if "relativity_per_unit_unit" in raw:
        return InteractionInference(
            name=name,
            kind="numeric",
            active=active,
            relativity_per_unit_unit=raw["relativity_per_unit_unit"],
            coef=raw["coef"],
        )

    # Fallback (e.g. PolynomialInteraction 2D surface)
    return InteractionInference(
        name=name,
        kind="surface",
        active=active,
    )
