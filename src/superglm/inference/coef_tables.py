"""Coefficient table and basis detail builders for model summaries."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.inference._metrics_design import MetricsDesign, factor_from_gram, weighted_moments
from superglm.inference.covariance import (
    covariance_quadratic_form,
    covariance_selected_block,
    covariance_selected_diagonal,
    covariance_slope_view,
)
from superglm.inference.summary import _BasisDetailRow, _CoefRow, _compute_coef_stats
from superglm.solvers.rank import diagonal_of_square, selected_group_name_set
from superglm.types import GroupSlice


def build_coef_rows(
    *,
    groups: list[GroupSlice],
    specs: dict,
    interaction_specs: dict,
    result: Any,
    X_a: NDArray | MetricsDesign,
    W: NDArray,
    XtWX_inv: NDArray,
    XtWX_inv_aug: NDArray,
    active_groups: list[GroupSlice],
    known_scale: bool,
    group_edf_map: dict | None,
    reml_lambdas: dict | None,
    lambda2: float | dict,
    n_obs: int,
    alpha: float = 0.05,
    monotone_repairs: dict | None = None,
    # Precomputed inference quantities (avoids recomputing QR/EDF)
    precomputed_R_a: NDArray | None = None,
    precomputed_edf: NDArray | None = None,
    precomputed_edf1: NDArray | None = None,
    precomputed_design_moments: tuple[NDArray, NDArray, NDArray] | None = None,
    coefficient_estimable_override: NDArray | None = None,
    selected_group_names: set[str] | None = None,
    group_matrices: list | None = None,
    sample_weights: NDArray | None = None,
    distribution: Any | None = None,
) -> list[_CoefRow]:
    """Build coefficient table rows for summary output.

    Standalone function that can be called from ``ModelMetrics._build_coef_rows``
    or from ``SuperGLM.summary()`` without a ``ModelMetrics`` instance.

    Parameters
    ----------
    XtWX_inv : (p_active, p_active) inverse used for EDF computation.
    XtWX_inv_aug : (p_active+1, p_active+1) augmented inverse including
        intercept row/column, used for SE computation.
    """
    from superglm.features.categorical import Categorical
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.interaction import (
        CategoricalInteraction,
        NumericCategorical,
        NumericInteraction,
        PolynomialCategorical,
        PolynomialInteraction,
        SplineCategorical,
        TensorInteraction,
    )
    from superglm.features.numeric import Numeric
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.features.piecewise import Piecewise
    from superglm.features.polynomial import Polynomial
    from superglm.features.random_effect import RandomEffect
    from superglm.features.spline import _SplineBase
    from superglm.group_matrix import CategoricalGroupMatrix
    from superglm.inference._ordered_reference import (
        ordered_reference_beta_contrast,
        ordered_reference_intercept,
    )
    from superglm.inference._term_covariance import feature_se_from_cov
    from superglm.inference._term_helpers import (
        _resolve_group_lambda,
        spline_group_enrichment,
        spline_groups,
    )

    beta = result.beta
    selected_names = (
        selected_group_name_set(result, groups)
        if selected_group_names is None
        else set(selected_group_names)
    )
    coefficient_estimable = (
        np.asarray(coefficient_estimable_override, dtype=bool)
        if coefficient_estimable_override is not None
        else (
            result.rank_info.coefficient_estimable()
            if getattr(result, "rank_info", None) is not None
            else np.ones(len(beta), dtype=bool)
        )
    )

    # ── Per-level diagnostics for categorical features ────────────
    # Compute observation count and exposure share per non-base level.
    _level_diag: dict[str, dict[int, tuple[int, float]]] = {}
    if group_matrices is not None and sample_weights is not None:
        total_weight = float(np.sum(sample_weights))
        for gm, g in zip(group_matrices, groups):
            if isinstance(gm, CategoricalGroupMatrix):
                K = gm.n_levels
                n_per = np.bincount(gm.codes, minlength=K + 1)[:K]
                exp_per = np.bincount(gm.codes, weights=sample_weights, minlength=K + 1)[:K]
                exp_share = exp_per / max(total_weight, 1e-300)
                _level_diag[g.name] = {i: (int(n_per[i]), float(exp_share[i])) for i in range(K)}
    phi = result.phi

    def _wood_residual_df(edf: NDArray) -> float:
        if known_scale:
            return -1.0
        effective_df = 1.0 + float(np.sum(edf))
        if distribution is not None and sample_weights is not None:
            from superglm.solvers.dispersion import pearson_residual_degrees_of_freedom

            return pearson_residual_degrees_of_freedom(
                distribution,
                sample_weights,
                effective_df,
            )
        return max(float(n_obs) - effective_df, 1.0)

    # Compute per-group SEs from augmented inverse (accounts for intercept).
    # The augmented inverse has intercept at row/col 0; feature blocks start at 1.
    se_dict: dict[str, NDArray] = {}
    for g in groups:
        if g.name not in selected_names:
            se_dict[g.name] = np.zeros(g.size)
        else:
            ag = next((a for a in active_groups if a.name == g.name), None)
            if ag is None:
                se_dict[g.name] = np.zeros(g.size)
            else:
                scale = 1.0 if known_scale else phi
                augmented_indices = np.arange(
                    1 + ag.start,
                    1 + ag.end,
                    dtype=np.intp,
                )
                var_diag = scale * covariance_selected_diagonal(
                    XtWX_inv_aug,
                    augmented_indices,
                )
                se_dict[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
        if g.name in selected_names:
            se_dict[g.name] = se_dict[g.name].astype(float, copy=True)
            se_dict[g.name][~coefficient_estimable[g.sl]] = np.nan

    # Ordered spline effects are displayed relative to their chosen base levels.
    # Apply the same affine transformation to the intercept and its covariance.
    feature_order = list(specs)
    intercept = ordered_reference_intercept(
        result.intercept,
        beta,
        feature_order,
        specs,
        groups,
    )
    full_reference_contrast = ordered_reference_beta_contrast(
        len(beta),
        feature_order,
        specs,
        groups,
    )
    augmented_reference_contrast = np.zeros(XtWX_inv_aug.shape[0], dtype=np.float64)
    augmented_reference_contrast[0] = 1.0
    original_groups = {group.name: group for group in groups}
    for active_group in active_groups:
        original_group = original_groups[active_group.name]
        augmented_reference_contrast[1 + active_group.start : 1 + active_group.end] = (
            full_reference_contrast[original_group.sl]
        )
    icpt_var = covariance_quadratic_form(
        XtWX_inv_aug,
        augmented_reference_contrast,
    )
    scale = 1.0 if known_scale else max(phi, 0.0)
    icpt_se = float(np.sqrt(max(scale * icpt_var, 0.0)))

    rows: list[_CoefRow] = []

    # Intercept row
    z, p, ci_lo, ci_hi = _compute_coef_stats(intercept, icpt_se, alpha)
    rows.append(
        _CoefRow(
            name="Intercept",
            coef=intercept,
            se=icpt_se,
            z=z,
            p=p,
            ci_low=ci_lo,
            ci_high=ci_hi,
        )
    )

    # Lazily computed R factor and influence edf (only needed for smooth tests).
    # When precomputed values are provided, use them directly.
    _R_factor = precomputed_R_a
    _influence_edf = None
    _design_moments = precomputed_design_moments
    if precomputed_edf is not None and precomputed_edf1 is not None:
        _influence_edf = (precomputed_edf, precomputed_edf1)

    def _get_design_moments():
        nonlocal _design_moments
        if _design_moments is None:
            _design_moments = weighted_moments(X_a, W)
        return _design_moments

    def _get_R_factor():
        nonlocal _R_factor
        if _R_factor is None:
            if X_a.shape[1] == 0:
                _R_factor = np.empty((0, 0))
            else:
                _R_factor = factor_from_gram(_get_design_moments()[2])
        return _R_factor

    def _get_influence_edf():
        nonlocal _influence_edf
        if _influence_edf is None:
            if X_a.shape[1] == 0:
                _influence_edf = (np.array([]), np.array([]))
            else:
                data_gram = _get_design_moments()[2]
                F = XtWX_inv_aug[1:, 1:] @ data_gram
                edf = np.diag(F)
                edf1 = 2.0 * edf - diagonal_of_square(F)
                _influence_edf = (edf, edf1)
        return _influence_edf

    # Per-group EDF map: use precomputed group_edf_map when provided.
    _group_edf_cache: dict[str, float] | None = group_edf_map

    def _get_group_edf_map() -> dict[str, float]:
        nonlocal _group_edf_cache
        if _group_edf_cache is None:
            edf, _ = _get_influence_edf()
            _group_edf_cache = {}
            for ag in active_groups:
                _group_edf_cache[ag.name] = float(np.sum(edf[ag.sl]))
        return _group_edf_cache

    def _curve_se_range(feature_name):
        """Compute curve SE min/max for a spline feature."""
        scale = phi if not known_scale else 1.0
        # Use the feature block of the augmented inverse for correct marginal SEs
        Cov_active = covariance_slope_view(XtWX_inv_aug, scale=scale)
        se_curve = feature_se_from_cov(
            feature_name, Cov_active, active_groups, result, groups, specs, interaction_specs
        )
        # One SE per level, and for a specials term that includes the FREE levels,
        # whose uncertainty is not a point on the curve. Every other statistic on
        # the ordered-spline row describes the spline block; _term_ops already
        # filters the special block out of the curve's own SE band. Levels are
        # ordered smooth-then-special (the build() contract), so drop the tail.
        spec = specs.get(feature_name)
        n_special = len(getattr(spec, "_specials", ()) or ())
        if n_special and len(se_curve) > n_special:
            se_curve = np.asarray(se_curve)[:-n_special]
        return float(np.min(se_curve)), float(np.max(se_curve))

    def _augmented_group_block(active_group: GroupSlice) -> NDArray:
        augmented_indices = np.arange(
            1 + active_group.start,
            1 + active_group.end,
            dtype=np.intp,
        )
        covariance = covariance_selected_block(
            XtWX_inv_aug,
            augmented_indices,
        )
        return covariance if known_scale else phi * covariance

    def _spline_enrichment(g_name, spec):
        d = spline_group_enrichment(g_name, spec, _get_group_edf_map(), reml_lambdas, lambda2)
        return (
            d["edf"],
            d["smoothing_lambda"],
            d["spline_kind"],
            d["knot_strategy"],
            d["boundary"],
        )

    def _structured_lambdas(group_name: str) -> tuple[tuple[str, float], ...]:
        source = reml_lambdas if reml_lambdas is not None else lambda2
        if isinstance(source, dict):
            exact = (("lambda", float(source[group_name])),) if group_name in source else ()
            prefix = f"{group_name}:"
            components = tuple(
                (name[len(prefix) :], float(value))
                for name, value in source.items()
                if name.startswith(prefix)
            )
            return exact + components
        return (("lambda", float(source)),)

    # Monotone repair info
    _mono_repairs = monotone_repairs or {}
    handled_ordered_features: set[Hashable] = set()

    # Feature rows
    for g in groups:
        spec = specs.get(g.feature_name) or interaction_specs.get(g.feature_name)
        feature_label = str(g.feature_name)
        b_g = beta[g.sl]
        se_g = se_dict[g.name]
        active = g.name in selected_names

        if isinstance(spec, RandomEffect | FactorSmooth):
            edf = _get_group_edf_map().get(g.name, 0.0) if active else 0.0
            lambdas = _structured_lambdas(g.name)
            if isinstance(spec, RandomEffect):
                structured_kind = "random_effect"
            else:
                structured_kind = f"factor_smooth_{spec.basis}"
            rows.append(
                _CoefRow(
                    name=g.name,
                    group=feature_label,
                    structured_kind=structured_kind,
                    n_levels=len(spec._levels),
                    n_params=g.size,
                    active=active,
                    group_norm=float(np.linalg.norm(b_g)) if active else 0.0,
                    edf=edf,
                    smoothing_lambda=(lambdas[0][1] if len(lambdas) == 1 else None),
                    smoothing_lambdas=lambdas,
                )
            )
            continue

        if isinstance(spec, OrderedCategorical):
            if g.feature_name in handled_ordered_features:
                continue
            handled_ordered_features.add(g.feature_name)

            feature_groups = [fg for fg in groups if fg.feature_name == g.feature_name]
            # A specials term owns a second, unpenalized GroupSlice under the
            # same feature_name.  ``reconstruct`` needs the full-width vector,
            # but every statistic reported on the smooth row — edf, the Wood
            # test, ref_df, n_params — is a statement about the spline block.
            smooth_groups = spline_groups(feature_groups)
            beta_combined = np.concatenate([beta[fg.sl] for fg in feature_groups])
            # Two notions of "active" for a specials term.  ``feature_active``
            # asks whether *any* block survived selection and gates the level
            # table, because a free special keeps its own real standard error
            # even when the curve is dropped.  ``smooth_active`` asks about the
            # spline block alone and gates everything printed on the smooth
            # row: an unpenalized specials block is always selected, so testing
            # the whole feature there would make the row permanently "active"
            # while advertising zero edf and a NaN Wald test.
            feature_active = any(fg.name in selected_names for fg in feature_groups)
            smooth_active = any(fg.name in selected_names for fg in smooth_groups)
            feature_edf = (
                sum(_get_group_edf_map().get(fg.name, 0.0) for fg in smooth_groups)
                if smooth_active
                else 0.0
            )

            scale = 1.0 if known_scale else phi
            Cov_active = covariance_slope_view(XtWX_inv_aug, scale=scale)
            se_levels = feature_se_from_cov(
                g.feature_name,
                Cov_active,
                active_groups,
                result,
                groups,
                specs,
                interaction_specs,
            )
            raw = spec.reconstruct(beta_combined)

            if spec.basis == "spline":
                active_pairs = []
                for feature_group in smooth_groups:
                    active_group = next(
                        (ag for ag in active_groups if ag.name == feature_group.name),
                        None,
                    )
                    if active_group is not None:
                        active_pairs.append((feature_group, active_group))

                stat = float("nan")
                p_val = float("nan")
                ref_df = float(sum(fg.size for fg in smooth_groups))
                curve_se_min = float("nan")
                curve_se_max = float("nan")
                beta_active = (
                    np.concatenate([beta[fg.sl] for fg, _ in active_pairs])
                    if active_pairs
                    else np.empty(0, dtype=float)
                )

                if active_pairs:
                    from superglm.stats.wood_pvalue import wood_test_smooth

                    active_indices = np.concatenate(
                        [np.arange(ag.start, ag.end) for _, ag in active_pairs]
                    )
                    augmented_indices = active_indices + 1
                    V_b_j = scale * covariance_selected_block(
                        XtWX_inv_aug,
                        augmented_indices,
                    )
                    R_a = _get_R_factor()
                    edf, edf1 = _get_influence_edf()
                    edf1_j = float(np.sum(edf1[active_indices]))
                    X_j = R_a[:, active_indices]
                    res_df = _wood_residual_df(edf)

                    try:
                        stat, p_val, ref_df = wood_test_smooth(
                            beta_active,
                            X_j,
                            V_b_j,
                            edf1_j,
                            res_df,
                        )
                    except np.linalg.LinAlgError:
                        pass
                    curve_se_min, curve_se_max = _curve_se_range(g.feature_name)

                _, s_lam, s_kind, s_knot_strat, s_bnd = _spline_enrichment(
                    smooth_groups[0].name,
                    spec._spline,
                )
                rows.append(
                    _CoefRow(
                        name=feature_label,
                        group=feature_label,
                        is_spline=True,
                        n_params=sum(fg.size for fg in smooth_groups),
                        active=smooth_active,
                        group_norm=float(np.linalg.norm(beta_active)),
                        wald_chi2=stat if smooth_active else None,
                        wald_p=p_val if smooth_active else None,
                        ref_df=ref_df if smooth_active else None,
                        curve_se_min=curve_se_min,
                        curve_se_max=curve_se_max,
                        subgroup_type="ordered_spline",
                        edf=feature_edf,
                        smoothing_lambda=s_lam,
                        spline_kind=s_kind,
                        knot_strategy=s_knot_strat,
                        boundary=s_bnd,
                        monotone=getattr(spec._spline, "monotone", None),
                        monotone_engine=g.monotone_engine,
                        monotone_repaired=g.feature_name in _mono_repairs,
                    )
                )

                levels = raw["levels"]
                # `raw["special_levels"]`, not `spec._specials`: the latter is
                # string-coerced, so for `specials=[9]` the free level is 9 in
                # `levels` and "9" here, `9 in {"9"}` is False, and the Fit
                # column silently reports the free level as "smooth".
                special_labels = set(raw.get("special_levels") or ()) if spec.has_specials else None
                for i, level in enumerate(levels):
                    coef_val = float(raw["level_log_relativities"][level])
                    se_val: float | None = (
                        float(se_levels[i]) if feature_active and i < len(se_levels) else None
                    )
                    level_ci_lo: float | None
                    level_ci_hi: float | None
                    if se_val is not None and np.isfinite(se_val) and se_val > 0.0:
                        _, _, level_ci_lo, level_ci_hi = _compute_coef_stats(
                            coef_val, se_val, alpha
                        )
                    elif se_val is not None and np.isfinite(se_val) and level == spec._base_level:
                        level_ci_lo = level_ci_hi = coef_val
                    else:
                        se_val = None
                        level_ci_lo = level_ci_hi = None
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=feature_label,
                            coef=coef_val,
                            se=se_val,
                            ci_low=level_ci_lo,
                            ci_high=level_ci_hi,
                            level_fit=(
                                None
                                if special_labels is None
                                else ("free" if level in special_labels else "smooth")
                            ),
                        )
                    )
            else:
                row_idx = 0
                for i, level in enumerate(raw["levels"]):
                    if level == spec._base_level:
                        continue
                    coef_val = float(raw["log_relativities"][level])
                    se_val = float(se_levels[i]) if i < len(se_levels) else 0.0
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=feature_label,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                            edf=feature_edf if row_idx == 0 else None,
                        )
                    )
                    row_idx += 1
            continue

        if isinstance(spec, _SplineBase):
            is_linear_subgroup = g.subgroup_type == "linear"
            _mono_dir = getattr(spec, "monotone", None)
            _mono_engine = g.monotone_engine
            _mono_repaired = g.feature_name in _mono_repairs
            if active:
                stat = float("nan")
                p_val = float("nan")
                ref_df = float(g.size)
                curve_se_min = float("nan")
                curve_se_max = float("nan")

                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)

                if is_linear_subgroup:
                    from scipy.stats import chi2 as chi2_dist

                    try:
                        stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                        ref_df = float(g.size)
                        p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                    except np.linalg.LinAlgError:
                        pass

                    curve_se_min, curve_se_max = _curve_se_range(g.feature_name)
                else:
                    from superglm.stats.wood_pvalue import wood_test_smooth

                    R_a = _get_R_factor()
                    edf, edf1 = _get_influence_edf()
                    edf1_j = float(np.sum(edf1[ag.sl]))
                    X_j = R_a[:, ag.sl]
                    res_df = _wood_residual_df(edf)

                    try:
                        stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                    except Exception:
                        pass

                    curve_se_min, curve_se_max = _curve_se_range(g.feature_name)

                s_edf, s_lam, s_kind, s_knot_strat, s_bnd = _spline_enrichment(g.name, spec)
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=stat,
                        wald_p=p_val,
                        ref_df=ref_df,
                        curve_se_min=curve_se_min,
                        curve_se_max=curve_se_max,
                        subgroup_type=g.subgroup_type,
                        edf=s_edf,
                        smoothing_lambda=s_lam,
                        spline_kind=s_kind,
                        knot_strategy=s_knot_strat,
                        boundary=s_bnd,
                        monotone=_mono_dir,
                        monotone_engine=_mono_engine,
                        monotone_repaired=_mono_repaired,
                    )
                )
            else:
                s_edf, s_lam, s_kind, s_knot_strat, s_bnd = _spline_enrichment(g.name, spec)
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                        subgroup_type=g.subgroup_type,
                        edf=0.0,
                        smoothing_lambda=s_lam,
                        spline_kind=s_kind,
                        knot_strategy=s_knot_strat,
                        boundary=s_bnd,
                        monotone=_mono_dir,
                        monotone_engine=_mono_engine,
                        monotone_repaired=_mono_repaired,
                    )
                )

        elif isinstance(spec, Categorical):
            gedf = _get_group_edf_map()
            cat_edf = gedf.get(g.name, 0.0) if active else 0.0
            diag = _level_diag.get(g.name, {})
            for i, level in enumerate(spec._non_base):
                coef_val = float(b_g[i])
                se_val = float(se_g[i])
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                n_obs_i, exp_share_i = diag.get(i, (None, None))
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{level}]",
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                        edf=cat_edf if i == 0 else None,
                        level_n_obs=n_obs_i,
                        level_exposure_share=exp_share_i,
                    )
                )

        elif isinstance(spec, SplineCategorical):
            if active:
                stat = float("nan")
                p_val = float("nan")
                ref_df = float(g.size)

                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)

                from superglm.stats.wood_pvalue import wood_test_smooth

                R_a = _get_R_factor()
                edf, edf1 = _get_influence_edf()
                edf1_j = float(np.sum(edf1[ag.sl]))
                X_j = R_a[:, ag.sl]
                res_df = _wood_residual_df(edf)

                try:
                    stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                except Exception:
                    pass

                _edf_map = _get_group_edf_map()
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=stat,
                        wald_p=p_val,
                        ref_df=ref_df,
                        edf=_edf_map.get(g.name) if _edf_map else None,
                        smoothing_lambda=_resolve_group_lambda(g.name, reml_lambdas, lambda2),
                    )
                )
            else:
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                    )
                )

        elif isinstance(spec, Polynomial):
            # Rows are labelled by the stated power, not by column position:
            # with powers={1, 2, 4} the third row is [P4].
            powers = spec.powers
            if powers == tuple(range(1, spec.degree + 1)):
                poly_group = f"{g.name} P({spec.degree})"
            else:
                poly_group = f"{g.name} P({','.join(str(p) for p in powers)})"
            for i in range(g.size):
                coef_val = float(b_g[i])
                se_val = float(se_g[i]) if len(se_g) > i else 0.0
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[P{powers[i]}]",
                        group=poly_group,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

        elif isinstance(spec, PolynomialCategorical):
            if active:
                stat = float("nan")
                p_val = float("nan")
                ref_df = float(g.size)

                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)

                from scipy.stats import chi2 as chi2_dist

                try:
                    stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                    p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                except np.linalg.LinAlgError:
                    pass

                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=stat,
                        wald_p=p_val,
                        ref_df=ref_df,
                    )
                )
            else:
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                    )
                )

        elif isinstance(spec, CategoricalInteraction):
            for i, (lev1, lev2) in enumerate(spec._pairs):
                coef_val = float(b_g[i])
                se_val = float(se_g[i])
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{lev1}:{lev2}]",
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

        elif isinstance(spec, NumericCategorical):
            for i, level in enumerate(spec._non_base):
                coef_val = float(b_g[i])
                se_val = float(se_g[i])
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{level}]",
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

        elif isinstance(spec, NumericInteraction | PolynomialInteraction):
            if active and g.size <= 4:
                for i in range(g.size):
                    coef_val = float(b_g[i])
                    se_val = float(se_g[i])
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.name}[{i}]" if g.size > 1 else g.name,
                            group=g.name,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                        )
                    )
            elif active:
                stat = float("nan")
                p_val = float("nan")
                ref_df = float(g.size)
                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)
                from scipy.stats import chi2 as chi2_dist

                try:
                    stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                    p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                except np.linalg.LinAlgError:
                    pass
                _edf_map = _get_group_edf_map()
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=stat,
                        wald_p=p_val,
                        ref_df=ref_df,
                        edf=_edf_map.get(g.name) if _edf_map else None,
                        smoothing_lambda=_resolve_group_lambda(g.name, reml_lambdas, lambda2),
                    )
                )
            else:
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                    )
                )

        elif isinstance(spec, TensorInteraction):
            _edf_map = _get_group_edf_map()
            ti_edf = _edf_map.get(g.name) if _edf_map else None
            ti_lam = _resolve_group_lambda(g.name, reml_lambdas, lambda2)
            if active:
                stat = float("nan")
                p_val = float("nan")
                ref_df = float(g.size)

                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)

                from superglm.stats.wood_pvalue import wood_test_smooth

                R_a = _get_R_factor()
                edf, edf1 = _get_influence_edf()
                edf1_j = float(np.sum(edf1[ag.sl]))
                X_j = R_a[:, ag.sl]
                res_df = _wood_residual_df(edf)

                try:
                    stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                except Exception:
                    pass

                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=True,
                        group_norm=float(np.linalg.norm(b_g)),
                        wald_chi2=stat,
                        wald_p=p_val,
                        ref_df=ref_df,
                        edf=ti_edf,
                        smoothing_lambda=ti_lam,
                    )
                )
            else:
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=feature_label,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                        edf=0.0,
                        smoothing_lambda=ti_lam,
                    )
                )

        elif isinstance(spec, Numeric):
            gedf = _get_group_edf_map()
            num_edf = gedf.get(g.name, 0.0) if active else 0.0
            coef_display = float(b_g[0])
            se_display = float(se_g[0])
            z, p, ci_lo, ci_hi = _compute_coef_stats(coef_display, se_display, alpha)
            rows.append(
                _CoefRow(
                    name=g.name,
                    group=g.name,
                    coef=coef_display,
                    se=se_display,
                    z=z,
                    edf=num_edf,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                    estimable=bool(coefficient_estimable[g.start]),
                )
            )

        elif isinstance(spec, Piecewise):
            # Per-knot Wald rows, their CIs, and edf = J+1 are valid only when
            # ALL FOUR of these hold:
            #   1. the term is unpenalized -- the slope penalty the design defers
            #      is deferred precisely because it forfeits the fixed-df
            #      contract (an L1 slope-change penalty on this model class is
            #      k=1 trend filtering, whose df is E[#knots] + k + 1, i.e.
            #      data-dependent);
            #   2. the group carries no selection shrinkage -- GroupInfo.penalized
            #      is True for a Piecewise group, so a group-lasso
            #      selection_penalty shrinks this block and breaks both the Wald
            #      rows and the fixed edf;
            #   3. the breakpoints are FIXED INPUTS, not selected on the response
            #      from the same data.  When a breakpoint is data-chosen the
            #      statistic converges to a supremum of a nonstandard Gaussian
            #      process and nominal Wald calibration fails, even though df is
            #      still nominally J+1.  breaks=int quantile placement is
            #      materially milder: it looks only at x, never at y.  Reading
            #      kinks off a fitted Spline and refitting them here on the same
            #      data is the real offender;
            #   4. the term is unconstrained -- under an active monotone
            #      constraint the effective df becomes the size of the active
            #      face, which is data-dependent.
            # Withdraw the per-coefficient p-values if any of these stops holding.
            #
            # Condition 3 is the load-bearing one and it is NOT "because the df
            # is fixed and known": df stays nominally J+1 under response-selected
            # breaks while the p-values go wrong.  Validity rests on the
            # breakpoints being inputs.
            knots = spec._knots
            pw_edf = _get_group_edf_map().get(g.name, 0.0) if active else 0.0
            for i, knot_index in enumerate(spec._non_base_indices):
                coef_val = float(b_g[i])
                se_val = float(se_g[i]) if len(se_g) > i else 0.0
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{float(knots[knot_index]):.10g}]",
                        group=g.name,
                        coef=coef_val,
                        se=se_val,
                        z=z,
                        p=p,
                        ci_low=ci_lo,
                        ci_high=ci_hi,
                    )
                )

            # edf is carried by the whole-term row below and by nothing else.
            # Categorical puts it on its first level row because a categorical
            # term has no term-level row to put it on; this one does, and
            # reporting it twice would both read as two numbers about one term
            # and land the same degrees of freedom in the summary's parametric
            # and smooth buckets at once.  Ordered-spline rows do it this way.
            #
            # Whole-term test: a J+1 df chi-square on all coefficients jointly,
            # testing that the term is flat.  Deliberately not a Wood smooth
            # test -- there is no smoothing parameter to have been estimated.
            stat = float("nan")
            p_val = float("nan")
            ref_df = float(g.size)
            if active:
                from scipy.stats import chi2 as chi2_dist

                ag = next(a for a in active_groups if a.name == g.name)
                V_b_j = _augmented_group_block(ag)
                try:
                    stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                    p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                except np.linalg.LinAlgError:
                    pass
            rows.append(
                _CoefRow(
                    name=g.name,
                    group=feature_label,
                    is_spline=True,
                    n_params=g.size,
                    active=active,
                    group_norm=float(np.linalg.norm(b_g)) if active else 0.0,
                    wald_chi2=stat if active else None,
                    wald_p=p_val if active else None,
                    ref_df=ref_df if active else None,
                    # Names the row for what it is in both renderers, which
                    # otherwise label every group-test row "spline".
                    subgroup_type="piecewise",
                    edf=pw_edf,
                )
            )

        else:
            coef_val = float(b_g[0])
            se_val = float(se_g[0]) if len(se_g) > 0 else 0.0
            z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
            rows.append(
                _CoefRow(
                    name=g.name,
                    group=g.name,
                    coef=coef_val,
                    se=se_val,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                    estimable=bool(coefficient_estimable[g.start]),
                )
            )

    # Every coefficient-style branch consumes the same masked SE arrays, but
    # not every feature-specific row constructor sets ``estimable`` directly.
    # Keep the public flag consistent with the rank mask for categoricals,
    # polynomials, and interaction coefficient rows as well as numerics.
    for row in rows:
        if row.se is not None and np.isnan(row.se):
            row.estimable = False

    # ── Quasi-separation detection ──────────────────────────────
    # Primary: data-driven — flag categorical levels with too few obs.
    # Fallback: SE-based — for non-categorical features or when
    # per-level diagnostics are unavailable.
    for r in rows:
        if r.is_spline or r.name == "Intercept":
            continue
        # Data-driven: insufficient observations or exposure
        if r.level_n_obs is not None and r.level_n_obs < 20:
            r.quasi_separated = True
        elif r.level_exposure_share is not None and r.level_exposure_share < 0.0005:
            r.quasi_separated = True

    # SE-based fallback for rows without per-level diagnostics
    parametric_ses = [
        r.se
        for r in rows
        if r.se is not None
        and r.se > 0
        and r.p is not None
        and not r.is_spline
        and r.name != "Intercept"
    ]
    if parametric_ses:
        median_se = float(np.median(parametric_ses))
        sep_threshold = max(median_se * 50, 10.0)
        for r in rows:
            if r.quasi_separated or r.is_spline or r.name == "Intercept":
                continue
            if r.p is None:
                continue
            if r.level_n_obs is not None:
                continue  # already handled by data-driven check
            if r.se is not None and r.se > sep_threshold:
                r.quasi_separated = True

    return rows


def build_basis_detail(
    groups,
    specs,
    interaction_specs,
    result,
    XtWX_inv_aug,
    active_groups,
    known_scale,
    alpha=0.05,
    coefficient_estimable_override=None,
    selected_group_names=None,
):
    """Build per-coefficient detail for active 1-D spline groups.

    Uses the same known_scale-aware covariance path as ``build_coef_rows``
    so that SE/z/p/CI values are consistent with the main summary.
    """
    from superglm.features.spline import _SplineBase

    beta = result.beta
    phi = result.phi
    selected_names = (
        selected_group_name_set(result, groups)
        if selected_group_names is None
        else set(selected_group_names)
    )
    coefficient_estimable = (
        np.asarray(coefficient_estimable_override, dtype=bool)
        if coefficient_estimable_override is not None
        else (
            result.rank_info.coefficient_estimable()
            if getattr(result, "rank_info", None) is not None
            else np.ones(len(beta), dtype=bool)
        )
    )
    detail: dict[str, list] = {}

    for g in groups:
        # V1: skip interactions
        if g.feature_name in interaction_specs:
            continue
        spec = specs.get(g.feature_name)
        if not isinstance(spec, _SplineBase):
            continue
        b_g = beta[g.sl]
        if g.name not in selected_names:
            continue

        ag = next((a for a in active_groups if a.name == g.name), None)
        if ag is None:
            continue

        scale = 1.0 if known_scale else phi
        augmented_indices = np.arange(
            1 + ag.start,
            1 + ag.end,
            dtype=np.intp,
        )
        var_diag = scale * covariance_selected_diagonal(
            XtWX_inv_aug,
            augmented_indices,
        )
        se_arr = np.sqrt(np.maximum(var_diag, 0.0))
        se_arr[~coefficient_estimable[g.sl]] = np.nan

        rows = []
        for i in range(g.size):
            coef_val = float(b_g[i])
            se_val = float(se_arr[i])
            z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
            rows.append(
                _BasisDetailRow(
                    parent_name=g.name,
                    basis_index=i,
                    coef=coef_val,
                    se=se_val,
                    z=z,
                    p=p,
                    ci_low=ci_lo,
                    ci_high=ci_hi,
                )
            )
        detail[g.name] = rows

    return detail
