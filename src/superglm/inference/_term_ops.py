"""Operational term-inference assembly entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm.inference._term_covariance import (
    feature_se_from_cov,
    piecewise_knot_covariance,
    simultaneous_bands,
)
from superglm.inference._term_helpers import (
    _VALID_CENTERING,
    _build_spline_metadata,
    _compute_term_edf,
    _expand_grouped_term,
    _recenter_term,
    _resolve_term_lambda,
    _spline_se,
    spline_groups,
)
from superglm.inference._term_interactions import _interaction_inference
from superglm.inference._term_types import (
    InteractionInference,
    SmoothCurve,
    TermInference,
    _safe_exp,
)
from superglm.solvers.rank import selected_group_name_set

if TYPE_CHECKING:
    from superglm.solvers.pirls import PIRLSResult
    from superglm.types import GroupSlice


def _maybe_array(value: NDArray | float | None) -> NDArray | None:
    """Normalize optional scalar-or-array results to ndarray for typed dataclasses."""
    if value is None:
        return None
    return cast(NDArray, np.asarray(value))


def _centered_se(
    centering: str,
    name: str,
    Cov_active,
    active_groups_cov,
    result: PIRLSResult,
    groups: list[GroupSlice],
    specs: dict[str, Any],
    interaction_specs: dict[str, Any],
    *,
    n_points: int = 200,
) -> NDArray | None:
    """The errors of the mean-centered report, or ``None`` when not centering.

    ``centering="mean"`` reports the estimable contrast ``C b`` rather than
    ``b``, so its errors come from ``C V C'`` and cannot be recovered from the
    against-base errors the native report carries.  Computed here, beside the
    covariance, and handed to ``_recenter_term``, which has no access to it.
    """
    if centering != "mean" or Cov_active is None:
        return None
    return feature_se_from_cov(
        name,
        Cov_active,
        active_groups_cov,
        result,
        groups,
        specs,
        interaction_specs,
        n_points=n_points,
        center=True,
    )


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
        geometric mean of relativities = 1.  The shift is a function of the
        same fitted coefficients, so the errors and intervals are propagated
        through the centering contrast rather than translated with the
        values -- see ``_recenter_term``.

    Returns
    -------
    TermInference or InteractionInference
    """
    if centering not in _VALID_CENTERING:
        raise ValueError(f"centering must be one of {_VALID_CENTERING}, got {centering!r}")

    from superglm.features.categorical import Categorical
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
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

    # Check active. Deliberately FEATURE-wide, unlike the smooth-row `active` in
    # coef_tables and the export: this one gates the level table, and a free
    # special is genuinely still fitted when the spline block is dropped. The
    # smooth's own survival is `smooth_active` below.
    beta_combined = np.concatenate([beta[g.sl] for g in feature_groups])
    selected_names = selected_group_name_set(result, groups)
    active = any(group.name in selected_names for group in feature_groups)
    smooth_active = any(group.name in selected_names for group in spline_groups(feature_groups))

    # Covariance (lazy, only if needed)
    Cov_active = active_groups_cov = None
    if with_se and active:
        Cov_active, active_groups_cov = covariance_fn()

    # Per-group edf, over the SMOOTH's blocks. coef_tables and report_ops both
    # scope this quantity the same way, and TermInference.edf is the copy that
    # reaches the editor context bar and the plot data -- so summing the free
    # block in here as well made one fitted term report two different edfs
    # depending on which surface you read it from.
    edf = _compute_term_edf(name, spline_groups(feature_groups), group_edf)

    # Per-group lambda
    lam = _resolve_term_lambda(name, feature_groups, reml_lambdas, lambda2)

    z_alpha = float(__import__("scipy").stats.norm.ppf(1.0 - alpha / 2.0))

    # ── RandomEffect ─────────────────────────────────────────────
    if isinstance(spec, RandomEffect):
        raw = spec.reconstruct(beta_combined)
        levels = raw["levels"]
        log_rels = np.asarray([raw["effects"][level] for level in levels])
        rels = np.exp(log_rels)
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
        return _recenter_term(
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
                centering_mode="population_zero",
                edf=edf,
                smoothing_lambda=lam,
                alpha=alpha,
            ),
            centering,
            se_centered=_centered_se(
                centering,
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            ),
        )

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

            # Specials are free levels with no position on the spline axis: they
            # stay out of level_x and are flagged for the renderers instead.
            #
            # Match on the labels `reconstruct` itself exported, NOT on
            # `spec._specials`: the latter is string-coerced at construction, so
            # for `order=[1, ..., 9], specials=[9]` it holds "9" while `levels`
            # holds 9. `9 in {"9"}` is False, every level then reads
            # not-special, and `raw["level_values"][9]` -- keyed on the smooth
            # levels alone -- raises KeyError three lines further down.
            special_labels = set(raw.get("special_levels") or ())
            level_is_special = (
                np.array([lv in special_labels for lv in levels], dtype=bool)
                if special_labels
                else None
            )
            smooth_levels = [lv for lv in levels if lv not in special_labels]

            # Per-level SEs (at K category positions)
            se = ci_lo = ci_hi = None
            curve = None
            # `active` gates the LEVEL table (a free special is still fitted when
            # the curve is gone); `smooth_active` gates the CURVE. Without the
            # second, dropping the spline block while the unpenalized special
            # survives left every spline group filtered out of the SE call, so
            # `_spline_se` returned zeros and a SmoothCurve was emitted anyway --
            # a plot then renders an active-looking smooth, with zero-width bands,
            # for a block that selection removed.
            # `active` gates the LEVEL table (a free special is still fitted when
            # the curve is gone); `smooth_active` gates the CURVE. Without the
            # second, dropping the spline block while the unpenalized special
            # survives leaves every spline group filtered out of the SE call, so
            # `_spline_se` returns zeros and a SmoothCurve is emitted anyway -- a
            # plot then renders an active-looking smooth, with zero-width bands,
            # for a block that selection removed.
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

                # Continuous curve for plotting (ordered levels only), emitted
                # only if the SPLINE block survived selection. Otherwise every
                # spline group is filtered out of the SE call below, _spline_se
                # returns zeros, and the curve renders as an active-looking smooth
                # with zero-width bands for a block that was dropped.
                if smooth_active:
                    level_x = np.array([raw["level_values"][lv] for lv in smooth_levels])
                    assert active_groups_cov is not None
                    # The curve is a statement about the spline block alone; its SE
                    # must be too.  feature_se_from_cov's level SEs are unaffected —
                    # that path transforms through the OC spec at full p+s width.
                    smooth_feature_groups = spline_groups(feature_groups)
                    smooth_active_cov = [
                        ag
                        for ag in active_groups_cov
                        if not (ag.feature_name == name and ag.subgroup_type == "special")
                    ]
                    curve_se = _spline_se(
                        inner,
                        name,
                        result.beta,
                        smooth_feature_groups,
                        smooth_active_cov,
                        Cov_active,
                        x_eval=raw["x"],
                        reference_x=np.array(
                            [raw["level_values"][spec._base_level]],
                            dtype=np.float64,
                        ),
                    )
                    curve = SmoothCurve(
                        x=raw["x"],
                        log_relativity=raw["log_relativity"],
                        relativity=raw["relativity"],
                        level_x=level_x,
                        se_log_relativity=curve_se,
                        ci_lower=_maybe_array(
                            _safe_exp(raw["log_relativity"] - z_alpha * curve_se)
                        ),
                        ci_upper=_maybe_array(
                            _safe_exp(raw["log_relativity"] + z_alpha * curve_se)
                        ),
                    )
            elif smooth_active:
                # No SEs requested but still provide the curve shape
                level_x = np.array([raw["level_values"][lv] for lv in smooth_levels])
                curve = SmoothCurve(
                    x=raw["x"],
                    log_relativity=raw["log_relativity"],
                    relativity=raw["relativity"],
                    level_x=level_x,
                )

            # Spline-shaped metadata exists only for a spline inner basis; a
            # Piecewise/Polynomial inner has no knot strategy or B-spline
            # degree to report, and the editor treats a None here as
            # "no spline metadata", which is exactly true.
            from superglm.features.spline import _SplineBase

            spline_meta = _build_spline_metadata(inner) if isinstance(inner, _SplineBase) else None
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
                level_is_special=level_is_special,
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
            # `active` gates the LEVEL table (a free special is still fitted when
            # the curve is gone); `smooth_active` gates the CURVE. Without the
            # second, dropping the spline block while the unpenalized special
            # survives left every spline group filtered out of the SE call, so
            # `_spline_se` returned zeros and a SmoothCurve was emitted anyway --
            # a plot then renders an active-looking smooth, with zero-width bands,
            # for a block that selection removed.
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
                    center=centering == "mean",
                )
                ci_lo_sim = bands["ci_lower_simultaneous"].values
                ci_hi_sim = bands["ci_upper_simultaneous"].values
                # Back out the critical value: ci_upper_sim = exp(log_rel + c*se).
                # Under centering the band was simulated through the centered
                # map, so it is backed out against the scale it was built on --
                # the outer `se` and `log_rel` are still the against-base pair.
                band_se = bands["se"].to_numpy() if centering == "mean" else se
                band_log_rel = (
                    bands["log_relativity"].to_numpy() if centering == "mean" else log_rel
                )
                safe_se = np.maximum(band_se, 1e-20)
                c_vals = (np.log(ci_hi_sim) - band_log_rel) / safe_se
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
            se_centered=_centered_se(
                centering,
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
                n_points=n_points,
            ),
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
            se_centered=_centered_se(
                centering,
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
            ),
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
            se_centered=_centered_se(
                centering,
                name,
                Cov_active,
                active_groups_cov,
                result,
                groups,
                specs,
                interaction_specs,
                n_points=n_points,
            ),
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

    # ── Piecewise ────────────────────────────────────────────────
    elif isinstance(spec, Piecewise):
        raw = spec.reconstruct(beta_combined)
        knots = raw["knots"]
        log_rel = raw["log_relativity"]

        se = ci_lo = ci_hi = None
        knot_cov = None
        if with_se and active and Cov_active is not None:
            assert active_groups_cov is not None
            # The full per-knot covariance, not just its diagonal: plotting
            # needs the off-diagonal terms to evaluate the band between knots
            # exactly.  sqrt(diag) IS the per-knot SE (same basis map as
            # feature_se_from_cov's Piecewise branch), so the two stay one
            # computation rather than two that can drift.  It is also why this
            # branch passes no `se_centered`: it publishes the covariance of
            # its own reported vector, so `_recenter_term` transforms that and
            # reads the centered errors off the diagonal.
            knot_cov = piecewise_knot_covariance(name, Cov_active, active_groups_cov, specs)
            se = (
                np.sqrt(np.maximum(np.diag(knot_cov), 0.0))
                if knot_cov is not None
                else np.zeros(knots.size)
            )
            ci_lo = _safe_exp(log_rel - z_alpha * se)
            ci_hi = _safe_exp(log_rel + z_alpha * se)

        return _recenter_term(
            TermInference(
                name=name,
                kind="piecewise",
                active=active,
                # x is the knot vector itself, not a display grid: a piecewise
                # term is exactly determined by its values at the knots, so the
                # curve, the editor handles and the workbook rows are the same
                # J+2 points rather than three resamplings of one function.
                x=knots,
                log_relativity=log_rel,
                relativity=_maybe_array(_safe_exp(log_rel)),
                se_log_relativity=se,
                ci_lower=_maybe_array(ci_lo),
                ci_upper=_maybe_array(ci_hi),
                knot_covariance=knot_cov,
                absorbs_intercept=False,
                centering_mode="base_knot",
                # Report the MEASURED edf, not the nominal J+1.  The design
                # states edf is fixed at J+1; that holds only while the group
                # carries no shrinkage, and GroupInfo.penalized is True here, so
                # a selection_penalty does shrink this block.  Measuring keeps
                # the claim true where it applies and honest where it does not;
                # the J+1 fallback is for callers that asked for no edf at all.
                edf=edf if edf is not None else float(spec._non_base_indices.size),
                # No smoothing lambda, never the global lambda2 fallback: the
                # group carries no penalty matrix, so lambda2 contributes no
                # smoothing penalty to this block, and reporting it would leak
                # a nonexistent smoothing parameter into TermInference, editor
                # metadata and plot payloads.  Selection shrinkage is a
                # different object and is not a smoothing lambda either.
                smoothing_lambda=None,
                alpha=alpha,
            ),
            centering,
        )

    else:
        raise TypeError(f"Unknown feature type: {type(spec).__name__}")


__all__ = [
    "term_inference",
]
