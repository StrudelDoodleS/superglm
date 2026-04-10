"""Coefficient table and basis detail builders for model summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.inference.summary import _BasisDetailRow, _CoefRow, _compute_coef_stats
from superglm.types import GroupSlice


def build_coef_rows(
    *,
    groups: list[GroupSlice],
    specs: dict,
    interaction_specs: dict,
    result: Any,
    X_a: NDArray,
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
    group_matrices: list | None = None,
    sample_weights: NDArray | None = None,
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
    from superglm.features.polynomial import Polynomial
    from superglm.features.spline import _SplineBase
    from superglm.group_matrix import CategoricalGroupMatrix
    from superglm.inference._term_covariance import feature_se_from_cov
    from superglm.inference._term_helpers import (
        _resolve_group_lambda,
        spline_group_enrichment,
    )

    beta = result.beta

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

    # Compute per-group SEs from augmented inverse (accounts for intercept).
    # The augmented inverse has intercept at row/col 0; feature blocks start at 1.
    se_dict: dict[str, NDArray] = {}
    for g in groups:
        if np.linalg.norm(beta[g.sl]) < 1e-12:
            se_dict[g.name] = np.zeros(g.size)
        else:
            ag = next((a for a in active_groups if a.name == g.name), None)
            if ag is None:
                se_dict[g.name] = np.zeros(g.size)
            else:
                scale = 1.0 if known_scale else phi
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                var_diag = scale * np.diag(XtWX_inv_aug[aug_sl, aug_sl])
                se_dict[g.name] = np.sqrt(np.maximum(var_diag, 0.0))

    # Intercept SE from augmented inverse [0, 0] element
    icpt_var = float(XtWX_inv_aug[0, 0])
    if icpt_var > 0:
        icpt_se = (
            float(np.sqrt(icpt_var)) if known_scale else float(np.sqrt(max(phi, 0.0) * icpt_var))
        )
    else:
        icpt_se = 0.0

    rows: list[_CoefRow] = []

    # Intercept row
    intercept = result.intercept
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
    if precomputed_edf is not None and precomputed_edf1 is not None:
        _influence_edf = (precomputed_edf, precomputed_edf1)

    def _get_R_factor():
        nonlocal _R_factor
        if _R_factor is None:
            if X_a.shape[1] == 0:
                _R_factor = np.empty((0, 0))
            else:
                _, _R_factor = np.linalg.qr(X_a * np.sqrt(W)[:, None], mode="reduced")
        return _R_factor

    def _get_influence_edf():
        nonlocal _influence_edf
        if _influence_edf is None:
            if X_a.shape[1] == 0:
                _influence_edf = (np.array([]), np.array([]))
            else:
                XtWX = X_a.T @ (X_a * W[:, None])
                F = XtWX_inv @ XtWX
                edf = np.diag(F)
                edf1 = 2.0 * edf - np.sum(F * F, axis=1)
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
        Cov_active = scale * XtWX_inv_aug[1:, 1:]
        se_curve = feature_se_from_cov(
            feature_name, Cov_active, active_groups, result, groups, specs, interaction_specs
        )
        return float(np.min(se_curve)), float(np.max(se_curve))

    def _spline_enrichment(g_name, spec):
        d = spline_group_enrichment(g_name, spec, _get_group_edf_map(), reml_lambdas, lambda2)
        return (
            d["edf"],
            d["smoothing_lambda"],
            d["spline_kind"],
            d["knot_strategy"],
            d["boundary"],
        )

    # Monotone repair info
    _mono_repairs = monotone_repairs or {}
    handled_ordered_features: set[str] = set()

    # Feature rows
    for g in groups:
        spec = specs.get(g.feature_name) or interaction_specs.get(g.feature_name)
        b_g = beta[g.sl]
        se_g = se_dict[g.name]
        active = np.linalg.norm(b_g) > 1e-12

        if isinstance(spec, OrderedCategorical):
            if g.feature_name in handled_ordered_features:
                continue
            handled_ordered_features.add(g.feature_name)

            feature_groups = [fg for fg in groups if fg.feature_name == g.feature_name]
            beta_combined = np.concatenate([beta[fg.sl] for fg in feature_groups])
            feature_active = bool(np.linalg.norm(beta_combined) > 1e-12)
            feature_edf = (
                sum(_get_group_edf_map().get(fg.name, 0.0) for fg in feature_groups)
                if feature_active
                else 0.0
            )

            scale = 1.0 if known_scale else phi
            Cov_active = scale * XtWX_inv_aug[1:, 1:]
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
                levels = raw["levels"]
                for i, level in enumerate(levels):
                    coef_val = float(raw["level_log_relativities"][level])
                    se_val = float(se_levels[i]) if i < len(se_levels) else 0.0
                    z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=g.feature_name,
                            coef=coef_val,
                            se=se_val,
                            z=z,
                            p=p,
                            ci_low=ci_lo,
                            ci_high=ci_hi,
                            edf=feature_edf if i == 0 else None,
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
                            group=g.feature_name,
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
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                V_b_j = (
                    XtWX_inv_aug[aug_sl, aug_sl]
                    if known_scale
                    else phi * XtWX_inv_aug[aug_sl, aug_sl]
                )

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
                    res_df = -1.0 if known_scale else float(n_obs - np.sum(edf))

                    try:
                        stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                    except Exception:
                        pass

                    curve_se_min, curve_se_max = _curve_se_range(g.feature_name)

                s_edf, s_lam, s_kind, s_knot_strat, s_bnd = _spline_enrichment(g.name, spec)
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.feature_name,
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
                        group=g.feature_name,
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
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                V_b_j = (
                    XtWX_inv_aug[aug_sl, aug_sl]
                    if known_scale
                    else phi * XtWX_inv_aug[aug_sl, aug_sl]
                )

                from superglm.stats.wood_pvalue import wood_test_smooth

                R_a = _get_R_factor()
                edf, edf1 = _get_influence_edf()
                edf1_j = float(np.sum(edf1[ag.sl]))
                X_j = R_a[:, ag.sl]
                res_df = -1.0 if known_scale else float(n_obs - np.sum(edf))

                try:
                    stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                except Exception:
                    pass

                _edf_map = _get_group_edf_map()
                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.feature_name,
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
                        group=g.feature_name,
                        is_spline=True,
                        n_params=g.size,
                        active=False,
                        group_norm=0.0,
                    )
                )

        elif isinstance(spec, Polynomial):
            poly_group = f"{g.name} P({spec.degree})"
            for i in range(g.size):
                coef_val = float(b_g[i])
                se_val = float(se_g[i]) if len(se_g) > i else 0.0
                z, p, ci_lo, ci_hi = _compute_coef_stats(coef_val, se_val, alpha)
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[P{i + 1}]",
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
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                V_b_j = (
                    XtWX_inv_aug[aug_sl, aug_sl]
                    if known_scale
                    else phi * XtWX_inv_aug[aug_sl, aug_sl]
                )

                from scipy.stats import chi2 as chi2_dist

                try:
                    stat = float(b_g @ np.linalg.solve(V_b_j, b_g))
                    p_val = 1.0 - chi2_dist.cdf(stat, ref_df)
                except np.linalg.LinAlgError:
                    pass

                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.feature_name,
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
                        group=g.feature_name,
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
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                V_b_j = (
                    XtWX_inv_aug[aug_sl, aug_sl]
                    if known_scale
                    else phi * XtWX_inv_aug[aug_sl, aug_sl]
                )
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
                        group=g.feature_name,
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
                        group=g.feature_name,
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
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                V_b_j = (
                    XtWX_inv_aug[aug_sl, aug_sl]
                    if known_scale
                    else phi * XtWX_inv_aug[aug_sl, aug_sl]
                )

                from superglm.stats.wood_pvalue import wood_test_smooth

                R_a = _get_R_factor()
                edf, edf1 = _get_influence_edf()
                edf1_j = float(np.sum(edf1[ag.sl]))
                X_j = R_a[:, ag.sl]
                res_df = -1.0 if known_scale else float(n_obs - np.sum(edf))

                try:
                    stat, p_val, ref_df = wood_test_smooth(b_g, X_j, V_b_j, edf1_j, res_df)
                except Exception:
                    pass

                rows.append(
                    _CoefRow(
                        name=g.name,
                        group=g.feature_name,
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
                        group=g.feature_name,
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
                )
            )

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
        if r.se is not None and r.se > 0 and not r.is_spline and r.name != "Intercept"
    ]
    if parametric_ses:
        median_se = float(np.median(parametric_ses))
        sep_threshold = max(median_se * 50, 10.0)
        for r in rows:
            if r.quasi_separated or r.is_spline or r.name == "Intercept":
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
):
    """Build per-coefficient detail for active 1-D spline groups.

    Uses the same known_scale-aware covariance path as ``build_coef_rows``
    so that SE/z/p/CI values are consistent with the main summary.
    """
    from superglm.features.spline import _SplineBase

    beta = result.beta
    phi = result.phi
    detail: dict[str, list] = {}

    for g in groups:
        # V1: skip interactions
        if g.feature_name in interaction_specs:
            continue
        spec = specs.get(g.feature_name)
        if not isinstance(spec, _SplineBase):
            continue
        b_g = beta[g.sl]
        if np.linalg.norm(b_g) < 1e-12:
            continue

        ag = next((a for a in active_groups if a.name == g.name), None)
        if ag is None:
            continue

        scale = 1.0 if known_scale else phi
        aug_sl = slice(1 + ag.start, 1 + ag.end)
        var_diag = scale * np.diag(XtWX_inv_aug[aug_sl, aug_sl])
        se_arr = np.sqrt(np.maximum(var_diag, 0.0))

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
