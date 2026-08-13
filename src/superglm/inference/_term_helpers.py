"""Internal term-inference helpers shared by operational entry points."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from superglm.inference._term_types import (
    SmoothCurve,
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

    ``variance`` is ``diag(V)``, ``cross`` is ``Vp`` and ``total`` is
    ``p'Vp``, with ``p = 1/L`` the weights the shift averages over; expanding
    ``(I - 1p')V(I - p1')`` leaves ``V_ii - 2(Vp)_i + p'Vp``.  Stated in those
    three moments because the reports that need it -- a categorical level
    table, a random effect -- read one coefficient per row, so ``V`` is never
    formed and ``C V C'`` would cost a dense ``L x L`` that ``Vp`` does not.

    Cancellation leaves the fit's reference level at ``-1e-17`` rather than
    exactly zero when its whole row of ``V`` is zero, so the result is clamped
    before any caller takes a square root.
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

    Shifts log-relativities so geometric mean of relativities = 1.  Numeric
    terms (single value) are skipped since centering is meaningless.

    The shift is the mean of the SAME fitted coefficients the term reports,
    not a known constant, so the centered vector is the linear contrast
    ``C b`` with ``C = I - 11'/L`` and its covariance is ``C V C'`` -- the
    variance of an estimable function under a change of identifiability
    constraint (SAS/STAT, *The Four Types of Estimable Functions*; the linear
    case of the delta method Stata's ``nlcom`` documents as ``GVG'``).
    Translating the interval endpoints by ``exp(-shift)`` and leaving the
    errors untouched is valid only for a known shift, and it left the level
    the fit happened to pin printing a point estimate away from unity with an
    exactly zero standard error and a zero-width interval -- the reference's
    certainty, on a report whose whole premise is that no level is the
    reference.

    ``se_centered`` is that propagated error, supplied by the caller holding
    the coefficient covariance.  A term that publishes the covariance of its
    own reported vector -- ``knot_covariance`` on a piecewise term -- needs no
    separate argument: transforming that matrix puts the errors on its
    diagonal.  Every row of ``C`` sums to zero, so each centered effect is an
    estimable contrast and its error does not depend on which level the fit
    dropped (Firth, "Quasi-variances in Xlisp-Stat and on the web", *JSS*
    5(4), 2000, §5.1).  The errors are redistributed, not inflated: some
    levels come back narrower, so nothing here may assume they widen.

    The constant that was removed is recorded on ``centering_shift`` -- it is
    not recoverable from the returned values, and anything that has to put it
    back (the rating-table export folds it into the exported base relativity)
    needs the constant this function actually subtracted rather than a
    re-derivation from the shifted output.  Every early return leaves the field
    at its ``0.0`` default, which is the truth for a term this function did not
    touch.
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

    # The reported vector's own covariance, where the term carries one.  The
    # band BETWEEN knots is a quadratic form in this matrix, so leaving it in
    # against-base coordinates would draw a band contradicting the very knot
    # errors it is required to pass through.
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
        new_curve = SmoothCurve(
            x=new_curve.x,
            log_relativity=np.asarray(new_curve.log_relativity, dtype=float) - shift,
            relativity=np.asarray(new_curve.relativity, dtype=float) * factor,
            level_x=new_curve.level_x,
            se_log_relativity=new_curve.se_log_relativity,
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


def _spline_se(
    spline_spec,
    name: Hashable,
    beta: NDArray,
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
        Evaluate SEs at these specific x positions instead of a linspace grid.
        When provided, ``n_points`` is ignored.
    reference_x : array, optional
        When provided, propagate uncertainty for ``f(x_eval) - f(reference_x)``.
    center : bool
        Propagate uncertainty for ``f(x) - mean_x f(x)`` instead, which is what
        ``centering="mean"`` reports.  ``C M = M - 1 (1'M/L)`` for
        ``C = I - 11'/L``, so subtracting the map's own column means is the
        whole of it -- the same route ``reference_x`` takes, against the grid
        mean rather than one reference row.
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
    # For select=True: only use columns for active subgroups
    active_cols = np.concatenate(
        [
            np.arange(g.start, g.end) - feature_groups[0].start
            for g in feature_groups
            if any(ag.feature_name == name and ag.name == g.name for ag in active_subs)
        ]
    )
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


def _expand_grouped_term(
    ti: TermInference, grouping, original_level_values: dict[str, float] | None = None
) -> TermInference:
    """Expand a grouped TermInference back to all original levels.

    Each original level gets the relativity/SE/CI of its group.
    For OrderedCategorical smooth_curve, level_x positions use the original
    level numeric values so each level gets its own x-position on the plot.
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

    # Expand smooth_curve: give each original level its own x-position and,
    # WHERE THE COLLAPSE MOVED THE DISPLAY AXIS, rebuild the display curve via
    # PCHIP interpolation through the expanded (level_x, relativity) pairs so it
    # passes through every marker.  level_x covers the SMOOTHED levels only, so
    # specials are held out of the rebuild and keep their detached marker rows.
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
        smooth_log_rel = log_rel[smooth_mask]

        if original_level_values is not None:
            expanded_level_x = np.array([original_level_values[lev] for lev in smooth_levels])
        else:
            grouped_lx = np.asarray(curve.level_x)
            n_expanded = len(smooth_levels)
            expanded_level_x = (
                np.linspace(float(grouped_lx.min()), float(grouped_lx.max()), n_expanded)
                if n_expanded > 1
                else grouped_lx[np.asarray(indices, dtype=np.intp)[smooth_mask]]
            )

        # Rebuild display curve through expanded level positions
        # Deduplicate x-positions (grouped levels share the same relativity
        # but may have different x — keep first occurrence for interpolation)
        seen_x = {}
        for xi, yi in zip(expanded_level_x, smooth_log_rel):
            if xi not in seen_x:
                seen_x[xi] = yi
        uniq_x = np.array(sorted(seen_x.keys()))
        uniq_log_y = np.array([seen_x[x] for x in uniq_x])

        # A collapse that leaves the display axis WHERE IT WAS has nothing to
        # reconcile, and re-interpolating there replaces the fitted curve with a
        # smooth read of its own samples.  A Piecewise or Polynomial inner basis
        # lives on level POSITIONS 0..L-1 and every original level sits AT its
        # group's position, so the expanded positions are the fitted curve's own
        # ``level_x`` exactly -- the rebuild then draws a different function
        # through points that already lie on the fitted one.  Measured on a
        # hosted ``Piecewise`` with one stated break, one grouped pair and a
        # V-shaped signal: the PCHIP rendered the C0 corner 6.3x too shallow
        # (slope jump +0.0743 against the fitted +0.4703) and ran 0.0393 in
        # log-relativity -- 4.0% in relativity -- away from the fitted shape one
        # band before the break.  PCHIP is C1 by construction, so the corner is
        # not merely rounded, it cannot be drawn at all.  That contradicts the
        # term's headline feature at the very break it states, so the fitted
        # curve is kept and only the markers move.  The test is the geometry
        # itself, not the basis name: any expansion that genuinely re-spreads the
        # levels (a spline's ``order=`` linspace or ``values=`` spacing, where
        # the grouped curve sits on group MEANS) fails it and rebuilds.
        #
        # Exact equality is deliberate, and it is exact BY CONSTRUCTION on both
        # sides rather than approximately equal by luck: a position axis is
        # ``float(position)`` for an integer position, and a singleton group's
        # mean is its one element returned unchanged.  Anything that starts
        # generating positions arithmetically -- normalising them through a
        # linspace, say -- should fall back to the rebuild rather than compare
        # equal to within a tolerance, which is what this spelling gets.
        axis_unmoved = np.array_equal(
            np.asarray(curve.level_x, dtype=np.float64), uniq_x.astype(np.float64)
        )

        if axis_unmoved:
            # Only the markers move.  The bands ride along untouched because the
            # grid they were computed on is the grid still being drawn.
            curve = replace(curve, level_x=expanded_level_x)
        elif len(uniq_x) >= 2:
            from scipy.interpolate import PchipInterpolator

            pchip = PchipInterpolator(uniq_x, uniq_log_y)
            new_x = np.linspace(float(uniq_x[0]), float(uniq_x[-1]), 200)
            new_log_rel = pchip(new_x)

            # A band is a statement about the curve it is drawn around, so it
            # cannot outlive a change to that curve's VALUES.  The rebuilt curve
            # is a PCHIP through the markers; the fitted curve's
            # ``se_log_relativity`` / ``ci_*`` were computed from the design at
            # the fitted x, and PCHIP's slope limiter is not a linear functional
            # of beta, so there is no band for the drawn line to inherit.
            #
            # The former test was whether the two GRIDS lined up, which is a
            # weaker question -- it asks whether the arrays zip, not whether they
            # describe the same function -- and for the common ``order=`` spline
            # both grids ARE the same 200-point linspace, so the fitted bands
            # were exported around interpolated values.  Measured on a grouped
            # ``Spline(kind="cr", n_knots=4)`` merging one interior pair: the
            # drawn curve ran up to 0.0477 in log-relativity (4.65% in
            # relativity) from the fitted one against a band half-width of at
            # most 0.00302, i.e. 68.8 standard errors, and 61 of the 200 drawn
            # points fell strictly outside their own exported band.  The band
            # centre was the FITTED curve to 1.1e-16 -- a ribbon around a line
            # nobody drew.  Merging a mid-range pair gave 0.0558 (5.74%), 59.2
            # standard errors and 60 of 200.
            #
            # Per-level SEs -- the rated quantities -- are expanded above and
            # stay; it is only the display band that goes.
            curve = SmoothCurve(
                x=new_x,
                log_relativity=new_log_rel,
                relativity=np.exp(new_log_rel),
                level_x=expanded_level_x,
                se_log_relativity=None,
                ci_lower=None,
                ci_upper=None,
            )
        else:
            # Fewer than two distinct smoothed positions: there is nothing to
            # interpolate through, so the curve's own values are kept unchanged
            # and its bands still describe exactly the line being drawn.
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

    An OrderedCategorical with ``specials=`` owns two GroupSlices under one
    feature name: the penalized spline block and an unpenalized indicator block.
    Everything that describes THE SMOOTH -- edf, the Wald test, ref_df, n_params,
    group_norm, the curve and its SE band, and whether the smooth survived
    selection -- is a statement about the spline block alone.

    This lives in one place because the filter was previously re-derived at each
    call site, and three separate review passes each found a different site that
    had been missed. A caller that wants the whole feature (the level table, where
    a free special keeps a real standard error even when the curve is dropped)
    should use ``feature_groups`` directly and say why.

    The attribute is read directly rather than through a defaulting
    ``getattr(g, "subgroup_type", None)``. That form fails OPEN: rename the
    field and ``None != "special"`` holds for every group, so the filter becomes
    a silent no-op and every smoothed level goes back to inheriting the
    special block's activity -- the exact regression review found twice at the
    export site, restored by a rename with no test failure. A plain attribute
    read turns that rename into an AttributeError instead.

    Takes a ``Sequence`` so tuple-holding callers (the export payload keeps its
    source groups as a tuple) can pass theirs without a round trip; the return
    is always a fresh list, so callers needing a tuple wrap the result.
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
    dict with keys: edf, smoothing_lambda, spline_kind, knot_strategy, boundary.
    """
    edf = group_edf.get(group_name) if group_edf else None
    lam = _resolve_group_lambda(group_name, reml_lambdas, lambda2)
    return {
        "edf": edf,
        "smoothing_lambda": lam,
        "spline_kind": type(spec).__name__,
        "knot_strategy": getattr(spec, "_knot_strategy_actual", None),
        "boundary": getattr(spec, "fitted_boundary", None),
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
