"""Internal term-inference helpers shared by operational entry points."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm.inference._term_types import (
    SplineMetadata,
    TermInference,
    _safe_exp,
)
from superglm.types import GroupSlice

_VALID_CENTERING = ("native", "mean")


def _maybe_array(value: NDArray | float | None) -> NDArray | None:
    """Normalize optional scalar-or-array results to ndarray for typed dataclasses."""
    if value is None:
        return None
    return cast(NDArray, np.asarray(value))


def mean_centered_variance(variance: NDArray, cross: NDArray, total: float) -> NDArray:
    """``diag(C V C')`` for the mean-centering contrast ``C = I - 11'/L``.

    Stated in the moments ``variance = diag(V)``, ``cross = Vp`` and ``total = p'Vp`` with
    ``p = 1/L``, so ``V`` is never formed: the result is ``V_ii - 2(Vp)_i + p'Vp``. Clamped at
    zero, since cancellation leaves the reference level at ``-1e-17`` when its row of ``V``
    is zero.
    """
    return cast(NDArray, np.maximum(variance - 2.0 * cross + total, 0.0))


def mean_centered_covariance(V: NDArray) -> NDArray:
    """``C V C'`` for the same ``C = I - 11'/L``, formed in ``O(L^2)``.

    ``(1/L)11'V`` has every row equal to ``V``'s column means and
    ``(1/L^2)11'V11'`` is its grand mean, so the two matrix products collapse
    to a rank-two update.  Its diagonal is :func:`mean_centered_variance` of
    ``V``'s diagonal, column means and grand mean -- one identity, written
    twice only because this caller holds ``V`` and the others do not.
    """
    values = np.asarray(V, dtype=np.float64)
    column_means = values.mean(axis=0)
    grand_mean = float(column_means.mean())
    return cast(
        NDArray,
        values - column_means[None, :] - column_means[:, None] + grand_mean,
    )


def _recenter_term(
    ti: TermInference,
    centering: str,
    *,
    se_centered: NDArray | None = None,
) -> TermInference:
    """Apply mean centering to a TermInference if requested.

    Shifts log-relativities so the relativities' geometric mean is 1 (single-value numeric
    terms are skipped). The centred vector is the contrast ``C b``, ``C = I - 11'/L``, with
    covariance ``C V C'``; each row of ``C`` sums to zero, so the errors do not depend on the
    dropped level (Firth, *JSS* 5(4), 2000, sec. 5.1). ``se_centered`` is that propagated error
    from a caller holding the covariance; a term publishing its own vector's covariance
    (``knot_covariance``) needs none. The shift is recorded on ``centering_shift`` (0.0 on every
    early return) for callers, such as the rating-table export, that must put it back.
    """
    if centering == "native" or ti.log_relativity is None:
        return ti
    log_rel = np.asarray(ti.log_relativity, dtype=float)
    if log_rel.size <= 1:
        return ti  # numeric: single value, skip
    shift = float(np.mean(log_rel))
    factor = _safe_exp(-shift)
    new_log_rel = log_rel - shift
    new_rel = cast(NDArray, np.asarray(_safe_exp(new_log_rel)))

    # The reported vector's own covariance, where the term carries one: the band between knots
    # is a quadratic form in it, so it must be centered too to pass through the knot errors.
    new_knot_cov = ti.knot_covariance
    if new_knot_cov is not None:
        new_knot_cov = mean_centered_covariance(new_knot_cov)
        if se_centered is None:
            se_centered = np.sqrt(np.maximum(np.diag(new_knot_cov), 0.0))

    new_se = ti.se_log_relativity
    new_ci_lo = _maybe_array(ti.ci_lower * factor if ti.ci_lower is not None else None)
    new_ci_hi = _maybe_array(ti.ci_upper * factor if ti.ci_upper is not None else None)
    new_ci_lo_sim = _maybe_array(
        ti.ci_lower_simultaneous * factor if ti.ci_lower_simultaneous is not None else None
    )
    new_ci_hi_sim = _maybe_array(
        ti.ci_upper_simultaneous * factor if ti.ci_upper_simultaneous is not None else None
    )
    if se_centered is not None:
        from scipy.stats import norm

        new_se = np.asarray(se_centered, dtype=np.float64)
        z_alpha = float(norm.ppf(1.0 - ti.alpha / 2.0))
        new_ci_lo = _maybe_array(_safe_exp(new_log_rel - z_alpha * new_se))
        new_ci_hi = _maybe_array(_safe_exp(new_log_rel + z_alpha * new_se))
        if ti.critical_value_simultaneous is not None:
            c_sim = float(ti.critical_value_simultaneous)
            new_ci_lo_sim = _maybe_array(_safe_exp(new_log_rel - c_sim * new_se))
            new_ci_hi_sim = _maybe_array(_safe_exp(new_log_rel + c_sim * new_se))

    # Re-center smooth_curve if present
    new_curve = ti.smooth_curve
    if new_curve is not None:
        # ``replace``, not a hand-listed constructor, so a field added to ``SmoothCurve`` later
        # survives centering, which only shifts.
        new_curve = replace(
            new_curve,
            log_relativity=np.asarray(new_curve.log_relativity, dtype=float) - shift,
            relativity=np.asarray(new_curve.relativity, dtype=float) * factor,
            ci_lower=_maybe_array(
                new_curve.ci_lower * factor if new_curve.ci_lower is not None else None
            ),
            ci_upper=_maybe_array(
                new_curve.ci_upper * factor if new_curve.ci_upper is not None else None
            ),
        )

    return replace(
        ti,
        log_relativity=new_log_rel,
        relativity=new_rel,
        se_log_relativity=new_se,
        ci_lower=new_ci_lo,
        ci_upper=new_ci_hi,
        ci_lower_simultaneous=new_ci_lo_sim,
        ci_upper_simultaneous=new_ci_hi_sim,
        knot_covariance=new_knot_cov,
        smooth_curve=new_curve,
        centering_mode="mean",
        centering_shift=shift,
    )


# ── Feature SEs ───────────────────────────────────────────────────


def _active_spline_columns(name: Hashable, feature_groups: list, active_subs: list) -> NDArray:
    """Columns of a spline's basis that belong to its active subgroups.

    For ``select=True``: only the columns of active subgroups enter a quadratic
    form against the active covariance.
    """
    return np.concatenate(
        [
            np.arange(g.start, g.end) - feature_groups[0].start
            for g in feature_groups
            if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
        ]
    )


def _spline_se(
    spline_spec,
    name: Hashable,
    feature_groups: list,
    active_groups: list,
    Cov_active: NDArray,
    n_points: int = 200,
    x_eval: NDArray | None = None,
    reference_x: NDArray | None = None,
    center: bool = False,
) -> NDArray:
    """Shared public-runtime SE computation for spline-style terms.

    Parameters
    ----------
    x_eval : array, optional
        Evaluate SEs at these x positions instead of a linspace grid; ``n_points`` is ignored.
    reference_x : array, optional
        Propagate uncertainty for ``f(x_eval) - f(reference_x)``.
    center : bool
        Propagate uncertainty for ``f(x) - mean_x f(x)`` (``centering="mean"``): subtract the
        map's own column means, as ``reference_x`` does against one reference row.
    """
    n_out = len(x_eval) if x_eval is not None else n_points
    active_subs = [ag for ag in active_groups if ag.feature_name == name]
    if not active_subs:
        return np.zeros(n_out)
    indices = np.concatenate([np.arange(ag.start, ag.end) for ag in active_subs])
    Cov_g = Cov_active[np.ix_(indices, indices)]
    x_grid = (
        x_eval if x_eval is not None else np.linspace(spline_spec._lo, spline_spec._hi, n_points)
    )
    M = np.asarray(spline_spec.transform(x_grid), dtype=np.float64)
    active_cols = _active_spline_columns(name, feature_groups, active_subs)
    M = M[:, active_cols]
    if reference_x is not None:
        M_ref = np.asarray(spline_spec.transform(reference_x), dtype=np.float64)[:, active_cols]
        if M_ref.shape[0] == 1:
            M = M - M_ref
        elif M_ref.shape[0] == M.shape[0]:
            M = M - M_ref
        else:
            raise ValueError("reference_x must produce one row or match x_eval row count.")
    if center:
        M = M - M.mean(axis=0)
    Q = M @ Cov_g
    return cast(NDArray, np.sqrt(np.maximum(np.sum(Q * M, axis=1), 0.0)))


def _build_spline_metadata(spec) -> SplineMetadata:
    """Extract spline knot/basis metadata from a fitted spline spec."""
    from superglm.features._spline_penalties import difference_penalty_kind

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
        difference_penalty=difference_penalty_kind(spec),
    )


def _expand_grouped_term(
    ti: TermInference, grouping, original_level_values: dict[str, float] | None = None
) -> TermInference:
    """Expand a grouped TermInference back to all original levels.

    Each original level gets its group's relativity, SE and CI (uncorrected for multiple
    comparison). The display curve (``x``, ``log_relativity``, ``relativity``,
    ``se_log_relativity``, ``ci_lower``, ``ci_upper``) is never rebuilt, since a curve through
    the markers is not linear in beta and has no band (Nychka, *JASA* 83, 1988; Marra and Wood,
    *Scand. J. Statist.* 39(1), 2012); only ``level_x`` moves, so an end merge leaves the curve
    short of the outer markers. ``group_axis_position`` and the collapse average the same
    members in the same order (``grouping.group_to_originals[label]``), so ``level_x``
    round-trips bitwise; other per-level quantities round-trip only to one rounding.
    """
    if ti.levels is None:
        raise ValueError("Grouped term expansion requires categorical levels.")
    grouped_levels = list(ti.levels)
    # Join on text. A LevelGrouping is string-keyed by construction, while a
    # term's reported levels carry the declaration's own types -- an ordered
    # categorical declared `order=[1, 2, 9]` reports ints. Matching those
    # directly raises KeyError on the first level whose spellings differ.
    group_idx = {str(lev): i for i, lev in enumerate(grouped_levels)}

    expanded_levels = grouping.all_original_levels
    indices = [group_idx[str(grouping.original_to_group[lev])] for lev in expanded_levels]

    log_rel = np.asarray(ti.log_relativity)[indices]
    rel = np.asarray(ti.relativity)[indices]

    se = ti.se_log_relativity
    ci_lo = ti.ci_lower
    ci_hi = ti.ci_upper
    if se is not None:
        se = np.asarray(se)[indices]
    if ci_lo is not None:
        ci_lo = np.asarray(ci_lo)[indices]
    if ci_hi is not None:
        ci_hi = np.asarray(ci_hi)[indices]

    # Expand smooth_curve: give each original level its own MARKER x-position
    # and leave every other field of the curve exactly as fitted.  level_x
    # covers the SMOOTHED levels only, so specials are held out here and keep
    # their detached marker rows.
    expanded_special = (
        None
        if ti.level_is_special is None
        else np.asarray(ti.level_is_special, dtype=bool)[indices]
    )
    curve = ti.smooth_curve
    if curve is not None and curve.level_x is not None:
        smooth_mask = (
            np.ones(len(expanded_levels), dtype=bool)
            if expanded_special is None
            else ~expanded_special
        )
        smooth_levels = [lev for lev, keep in zip(expanded_levels, smooth_mask) if keep]

        if original_level_values is not None:
            expanded_level_x = np.array([original_level_values[lev] for lev in smooth_levels])
        else:
            # The only place left that invents a position (an even spread between the grouped
            # extremes). Unreachable in-package, where both OC call sites pass
            # ``spec._original_level_to_value``; it covers direct calls and old pickles.
            grouped_lx = np.asarray(curve.level_x)
            n_expanded = len(smooth_levels)
            expanded_level_x = (
                np.linspace(float(grouped_lx.min()), float(grouped_lx.max()), n_expanded)
                if n_expanded > 1
                else grouped_lx[np.asarray(indices, dtype=np.intp)[smooth_mask]]
            )

        # The curve is not rebuilt: only the markers move. ``replace`` keeps any field added to
        # ``SmoothCurve`` later. Interpolating a PCHIP through the expanded markers is avoided
        # because its slope rule (Fritsch and Butland, *SIAM J. Sci. Stat. Comput.* 5(2), 1984)
        # sets the slope to zero at sign changes and flat secants, which misdraws a merged group,
        # and PCHIP is C1 so a ``Piecewise`` term's C0 corner is undrawable.
        curve = replace(curve, level_x=expanded_level_x)

    # dataclasses.replace, not a hand-listed rebuild: every field this function
    # does not touch — including level_is_special's siblings — survives by
    # construction rather than by remembering to list it.
    return replace(
        ti,
        levels=expanded_levels,
        log_relativity=log_rel,
        relativity=rel,
        se_log_relativity=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        smooth_curve=curve,
        level_is_special=expanded_special,
    )


def spline_groups(feature_groups: Sequence[GroupSlice]) -> list[GroupSlice]:
    """The blocks of a feature that make up its smooth, dropping any free-level block.

    An OrderedCategorical with ``specials=`` owns two GroupSlices under one feature name: the
    penalized spline block and an unpenalized indicator block. Everything describing the smooth
    (edf, Wald test, ref_df, n_params, group_norm, curve and band, survival of selection) is
    about the spline block alone. A caller that wants the whole feature (the level table) should
    use ``feature_groups`` directly. ``subgroup_type`` is read as a plain attribute, so a rename
    raises instead of silently turning the filter into a no-op. Returns a fresh list.
    """
    return [g for g in feature_groups if g.subgroup_type != "special"]


def ordered_level_fit(
    level: object,
    *,
    special_labels: set | None,
    pinned_labels: set,
    main_fit: str,
) -> str | None:
    """The ``Fit`` word for one OrderedCategorical level row.

    Lives beside ``spline_groups`` for the same reason: the smooth/free split
    was re-derived at both level-row builders (``coef_tables`` and the
    editor-stale path in ``report_ops``), and they drifted. A pinned special is
    a third state -- declared, carrying no rows, and therefore no coefficient --
    and reporting it as ``"free"`` claims a fitted level effect that was never
    estimated.

    ``special_labels`` is ``None`` for a term with no ``specials=``, which is
    what suppresses the column entirely.
    """
    if special_labels is None:
        return None
    if level in pinned_labels:
        return "pinned"
    return "free" if level in special_labels else main_fit


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


def _resolve_group_lambda(
    group_name: str,
    reml_lambdas: dict[str, float] | None,
    lambda2: float | dict | None,
) -> float | None:
    """Look up REML lambda for a group, handling multi-penalty component keys.

    For single-penalty groups, returns ``reml_lambdas[group_name]`` directly.
    For multi-penalty groups (e.g. select=True, multi-m), the keys are
    ``"group:suffix"``; returns the geometric mean of all component lambdas
    as the representative smoothing level.
    """
    if reml_lambdas:
        if group_name in reml_lambdas:
            return reml_lambdas[group_name]
        comp_keys = [k for k in reml_lambdas if k.startswith(f"{group_name}:")]
        if comp_keys:
            import numpy as np

            vals = [reml_lambdas[k] for k in comp_keys]
            return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-300)))))
    if isinstance(lambda2, int | float):
        return float(lambda2)
    return None


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
    dict with keys: edf, smoothing_lambda, spline_kind, knot_strategy, boundary,
    difference_penalty.
    """
    from superglm.features._spline_penalties import difference_penalty_kind

    edf = group_edf.get(group_name) if group_edf else None
    lam = _resolve_group_lambda(group_name, reml_lambdas, lambda2)
    return {
        "edf": edf,
        "smoothing_lambda": lam,
        "spline_kind": type(spec).__name__,
        "knot_strategy": getattr(spec, "_knot_strategy_actual", None),
        "boundary": getattr(spec, "fitted_boundary", None),
        "difference_penalty": difference_penalty_kind(spec),
    }


__all__ = [
    "_VALID_CENTERING",
    "_build_spline_metadata",
    "_compute_term_edf",
    "_expand_grouped_term",
    "_recenter_term",
    "_resolve_group_lambda",
    "_resolve_term_lambda",
    "_spline_se",
    "mean_centered_covariance",
    "mean_centered_variance",
    "spline_group_enrichment",
]
