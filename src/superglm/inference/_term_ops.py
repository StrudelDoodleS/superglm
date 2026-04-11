"""Operational term-inference assembly entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm.inference._term_covariance import feature_se_from_cov, simultaneous_bands
from superglm.inference._term_helpers import (
    _VALID_CENTERING,
    _build_spline_metadata,
    _compute_term_edf,
    _expand_grouped_term,
    _recenter_term,
    _resolve_term_lambda,
    _spline_se,
)
from superglm.inference._term_interactions import _interaction_inference
from superglm.inference._term_types import (
    InteractionInference,
    SmoothCurve,
    TermInference,
    _safe_exp,
)

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _maybe_array(value: NDArray | float | None) -> NDArray | None:
    """Normalize optional scalar-or-array results to ndarray for typed dataclasses."""
    if value is None:
        return None
    return cast(NDArray, np.asarray(value))


# ── Main Entry Point ──────────────────────────────────────────────


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
    centering: str = "native",
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
    centering : {"native", "mean"}
        ``"native"`` (default) returns the canonical fitted term
        contribution under the model's identifiability constraint.
        ``"mean"`` is a reporting convenience that shifts so the
        geometric mean of relativities = 1.

    Returns
    -------
    TermInference or InteractionInference
    """
    if centering not in _VALID_CENTERING:
        raise ValueError(f"centering must be one of {_VALID_CENTERING}, got {centering!r}")

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
            # Spline mode: primary output is categorical (K levels with SEs),
            # plus a smooth_curve for plotting the fitted spline.
            inner = spec._spline
            raw = spec.reconstruct(beta_combined)
            levels = raw["levels"]
            level_log_rels = np.array([raw["level_log_relativities"][lv] for lv in levels])
            level_rels = np.array([raw["level_relativities"][lv] for lv in levels])

            # Per-level SEs (at K category positions)
            se = ci_lo = ci_hi = None
            curve = None
            if with_se and active and Cov_active is not None:
                assert active_groups_cov is not None
                se = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups_cov,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
                ci_lo = _safe_exp(level_log_rels - z_alpha * se)
                ci_hi = _safe_exp(level_log_rels + z_alpha * se)

                # Continuous curve for plotting
                level_x = np.array([raw["level_values"][lv] for lv in levels])
                assert active_groups_cov is not None
                curve_se = _spline_se(
                    inner,
                    name,
                    result.beta,
                    feature_groups,
                    active_groups_cov,
                    Cov_active,
                    n_points=n_points,
                )
                curve = SmoothCurve(
                    x=raw["x"],
                    log_relativity=raw["log_relativity"],
                    relativity=raw["relativity"],
                    level_x=level_x,
                    se_log_relativity=curve_se,
                    ci_lower=_maybe_array(_safe_exp(raw["log_relativity"] - z_alpha * curve_se)),
                    ci_upper=_maybe_array(_safe_exp(raw["log_relativity"] + z_alpha * curve_se)),
                )
            elif active:
                # No SEs requested but still provide the curve shape
                level_x = np.array([raw["level_values"][lv] for lv in levels])
                curve = SmoothCurve(
                    x=raw["x"],
                    log_relativity=raw["log_relativity"],
                    relativity=raw["relativity"],
                    level_x=level_x,
                )

            spline_meta = _build_spline_metadata(inner) if inner is not None else None
            # OrderedCategorical: base level already shifted to 0/1 — skip recentering
            ti_result = TermInference(
                name=name,
                kind="categorical",
                active=active,
                levels=levels,
                log_relativity=level_log_rels,
                relativity=level_rels,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="base_level",
                edf=edf,
                smoothing_lambda=lam,
                smooth_curve=curve,
                spline=spline_meta,
                alpha=alpha,
            )
            if spec._grouping is not None:
                ti_result = _expand_grouped_term(
                    ti_result, spec._grouping, spec._original_level_to_value
                )
            return ti_result
        else:
            # Step mode: categorical-style output
            raw = spec.reconstruct(beta_combined)
            levels = raw["levels"]
            log_rels = np.array([raw["log_relativities"][lv] for lv in levels])
            rels = np.array([raw["relativities"][lv] for lv in levels])

            se = ci_lo = ci_hi = None
            if with_se and active and Cov_active is not None:
                assert active_groups_cov is not None
                se = feature_se_from_cov(
                    name,
                    Cov_active,
                    active_groups_cov,
                    result,
                    groups,
                    specs,
                    interaction_specs,
                )
                ci_lo = _safe_exp(log_rels - z_alpha * se)
                ci_hi = _safe_exp(log_rels + z_alpha * se)

            # OrderedCategorical: base level already at 0/1 — skip recentering
            ti_result = TermInference(
                name=name,
                kind="categorical",
                active=active,
                levels=levels,
                log_relativity=log_rels,
                relativity=rels,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="base_level",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            )
            if spec._grouping is not None:
                ti_result = _expand_grouped_term(
                    ti_result, spec._grouping, spec._original_level_to_value
                )
            return ti_result

    # ── Spline ───────────────────────────────────────────────────
    if isinstance(spec, _SplineBase):
        raw = spec.reconstruct(beta_combined, n_points=n_points)
        x_grid = raw["x"]
        log_rel = raw["log_relativity"]
        rel = raw["relativity"]

        se = ci_lo = ci_hi = None
        ci_lo_sim = ci_hi_sim = c_sim = None

        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
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
            ci_lo = _safe_exp(log_rel - z_alpha * se)
            ci_hi = _safe_exp(log_rel + z_alpha * se)

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

        return _recenter_term(
            TermInference(
                name=name,
                kind="spline",
                active=active,
                x=x_grid,
                log_relativity=log_rel,
                relativity=rel,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
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
            ),
            centering,
        )

    # ── Categorical ──────────────────────────────────────────────
    elif isinstance(spec, Categorical):
        raw = spec.reconstruct(beta_combined)
        levels = raw["levels"]
        log_rels = np.array([raw["log_relativities"][lv] for lv in levels])
        rels = np.array([raw["relativities"][lv] for lv in levels])

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            )
            ci_lo = _safe_exp(log_rels - z_alpha * se)
            ci_hi = _safe_exp(log_rels + z_alpha * se)

        ti_result = _recenter_term(
            TermInference(
                name=name,
                kind="categorical",
                active=active,
                levels=levels,
                log_relativity=log_rels,
                relativity=rels,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="base_level",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            ),
            centering,
        )
        if spec._grouping is not None:
            ti_result = _expand_grouped_term(ti_result, spec._grouping)
        return ti_result

    # ── Polynomial ───────────────────────────────────────────────
    elif isinstance(spec, Polynomial):
        raw = spec.reconstruct(beta_combined)
        x_grid = raw["x"]
        log_rel = raw["log_relativity"]
        rel = raw["relativity"]

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
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
            ci_lo = _safe_exp(log_rel - z_alpha * se)
            ci_hi = _safe_exp(log_rel + z_alpha * se)

        return _recenter_term(
            TermInference(
                name=name,
                kind="polynomial",
                active=active,
                x=x_grid,
                log_relativity=log_rel,
                relativity=rel,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=True,
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            ),
            centering,
        )

    # ── Numeric ──────────────────────────────────────────────────
    elif isinstance(spec, Numeric):
        raw = spec.reconstruct(beta_combined)
        log_rel = np.array([np.log(raw["relativity_per_unit"])])
        rel = np.array([raw["relativity_per_unit"]])

        se = ci_lo = ci_hi = None
        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
            se = feature_se_from_cov(
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            )
            ci_lo = _safe_exp(log_rel - z_alpha * se)
            ci_hi = _safe_exp(log_rel + z_alpha * se)

        return _recenter_term(
            TermInference(
                name=name,
                kind="numeric",
                active=active,
                log_relativity=log_rel,
                relativity=rel,
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                absorbs_intercept=False,
                centering_mode="none",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            ),
            centering,
        )

    else:
        raise TypeError(f"Unknown feature type: {type(spec).__name__}")


__all__ = [
    "term_inference",
]
