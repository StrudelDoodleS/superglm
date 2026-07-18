"""Model report and reconstruction helpers."""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np

from superglm.inference._term_helpers import spline_group_enrichment
from superglm.inference.summary import _CoefRow
from superglm.model.fit_state import fitted_lambda2, fitted_penalty
from superglm.profiling._reporting import (
    cached_tweedie_profile_ci,
    tweedie_profile_method_label,
    tweedie_profile_report_identity,
)
from superglm.solvers.rank import selected_group_name_set
from superglm.types import GroupSlice


def diagnostics(model) -> dict[str, Any]:
    """Per-group diagnostic dict for programmatic / audit access."""
    from superglm.features.spline import _SplineBase

    res = model.result
    group_edf = model._group_edf
    reml_lam = getattr(model, "_reml_lambdas", None)
    penalty = fitted_penalty(model)
    lambda2 = fitted_lambda2(model)
    selected_names = selected_group_name_set(res, model._groups, penalty=penalty)

    out = {}
    for g in model._groups:
        bg = res.beta[g.sl]
        entry: dict[str, Any] = {
            "active": g.name in selected_names,
            "group_norm": float(np.linalg.norm(bg)),
            "n_params": g.size,
        }
        spec = model._specs.get(g.feature_name)
        if isinstance(spec, _SplineBase):
            entry.update(spline_group_enrichment(g.name, spec, group_edf, reml_lam, lambda2))
        out[g.name] = entry
    out["_model"] = {
        "intercept": res.intercept,
        "deviance": res.deviance,
        "phi": res.phi,
        "effective_df": res.effective_df,
        "n_iter": (
            model._reml_result.n_reml_iter
            if getattr(model, "_reml_result", None) is not None
            else res.n_iter
        ),
        "converged": (
            model._reml_result.converged
            if getattr(model, "_reml_result", None) is not None
            else res.converged
        ),
        "lambda1": penalty.lambda1,
    }
    return out


def summary(model, alpha: float = 0.05, detail: str = "compact"):
    """Rich model summary with coefficient table (statsmodels-style)."""
    from scipy.special import gammaln

    from superglm.inference.coef_tables import build_basis_detail, build_coef_rows
    from superglm.inference.summary import ModelSummary

    cache = model._summary_cache
    if cache is None:
        cache = {}
        model._summary_cache = cache
    tw_pr = getattr(model, "_tweedie_profile_result", None)
    tweedie_identity = None if tw_pr is None else tweedie_profile_report_identity(tw_pr, alpha)
    key = (float(alpha), detail, tweedie_identity)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if model._fit_stats is None:
        raise RuntimeError("No fit stats — call fit() or fit_reml() first.")

    fs = model._fit_stats
    res = model.result
    edf = res.effective_df
    n = fs.n_obs
    ll = fs.log_likelihood

    aic = -2 * ll + 2 * edf
    bic = -2 * ll + np.log(n) * edf
    denom = n - edf - 1.0
    aicc = aic + 2 * edf * (edf + 1) / denom if denom > 0 else np.inf
    selected_names = selected_group_name_set(res, model._groups, penalty=fitted_penalty(model))
    n_active = len(selected_names)
    p_total = len(model._groups)

    ebic = bic + 2 * 0.5 * (
        gammaln(p_total + 1) - gammaln(n_active + 1) - gammaln(p_total - n_active + 1)
    )

    data = {
        "information_criteria": {
            "log_likelihood": ll,
            "null_log_likelihood": fs.null_log_likelihood,
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "ebic": ebic,
        },
        "deviance": {
            "deviance": res.deviance,
            "null_deviance": fs.null_deviance,
            "explained_deviance": fs.explained_deviance,
        },
        "fit": {
            "phi": res.phi,
            "effective_df": edf,
            "pearson_chi2": fs.pearson_chi2,
            "n_obs": n,
            "n_active_groups": n_active,
        },
    }

    penalty = fitted_penalty(model)
    link_name = type(model._link).__name__
    if link_name.endswith("Link"):
        link_name = link_name[:-4]
    lam1 = penalty.lambda1
    if lam1 is not None and lam1 == 0:
        penalty_name = "None"
    else:
        penalty_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", type(penalty).__name__)

    has_select = any(
        getattr(model._specs.get(g.feature_name), "select", False) for g in model._groups
    )
    penalty_abbrevs: dict[str, str] = {}
    if has_select:
        short = {"Group Lasso": "GL", "Sparse Group Lasso": "SGL", "Group Elastic Net": "GEN"}
        abbrev = short.get(penalty_name)
        if abbrev is not None:
            penalty_abbrevs[abbrev] = penalty_name
            penalty_name = abbrev
        penalty_abbrevs["SEL"] = "double penalty selection (Wood, 2011)"
        penalty_name += " + SEL"

    meta = model._last_fit_meta or {}
    method_parts = ["REML" if meta.get("method") == "fit_reml" else "ML"]
    if meta.get("discrete"):
        method_parts.append("discrete")
    method_str = ", ".join(method_parts)

    model_info = {
        "family": {"NegativeBinomial": "Neg. Binomial"}.get(
            type(model._distribution).__name__, type(model._distribution).__name__
        ),
        "link": link_name,
        "penalty": penalty_name,
        "penalty_abbrevs": penalty_abbrevs,
        "method": method_str,
        "n_obs": n,
        "effective_df": edf,
        "lambda1": lam1,
        "phi": res.phi,
        "pearson_chi2": fs.pearson_chi2,
        "deviance": res.deviance,
        "log_likelihood": ll,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "ebic": ebic,
        "converged": (
            model._reml_result.converged
            if getattr(model, "_reml_result", None) is not None
            else res.converged
        ),
        "n_iter": (
            model._reml_result.n_reml_iter
            if getattr(model, "_reml_result", None) is not None
            else res.n_iter
        ),
    }
    editor_meta = getattr(model, "_editor_edits", None)
    if getattr(model, "_editor_inference_stale", False):
        model_info["editor_inference_stale"] = True
        model_info["editor_edited_terms"] = list((editor_meta or {}).get("terms", []))
        model_info["editor_message"] = (editor_meta or {}).get(
            "message",
            "Manual editor edits were applied; fitted inference is reference-only.",
        )

    offset_meta = getattr(model, "_editor_offset", None)
    if offset_meta is not None:
        model_info["editor_offset_terms"] = list(offset_meta.get("terms", []))
        model_info["editor_offset_message"] = offset_meta.get(
            "message",
            "Manual editor terms are fixed as an offset.",
        )

    nb_pr = getattr(model, "_nb_profile_result", None)
    if nb_pr is not None:
        ci = nb_pr.ci(alpha=alpha)
        model_info["nb_theta"] = nb_pr.theta_hat
        model_info["nb_theta_ci"] = ci
        model_info["nb_theta_method"] = "Profile (exact)"
        model_info["nb_profile_nll"] = nb_pr.nll

    if tw_pr is not None:
        ci, ci_status = cached_tweedie_profile_ci(tw_pr, alpha)
        model_info["tweedie_p"] = tw_pr.p_hat
        model_info["tweedie_p_ci"] = ci
        model_info["tweedie_p_ci_status"] = ci_status
        model_info["tweedie_phi"] = tw_pr.phi_hat
        model_info["tweedie_p_method"] = tweedie_profile_method_label(tw_pr)
        if hasattr(tw_pr, "nll"):
            model_info["tweedie_profile_nll"] = tw_pr.nll

    editor_inference_stale = bool(model_info.get("editor_inference_stale", False))
    if editor_inference_stale:
        coef_rows = _build_editor_stale_coef_rows(model)
        _suppress_editor_inference(coef_rows)
        data["standard_errors"] = {"inference_stale": True}
        summary_obj = ModelSummary(data, model_info, coef_rows, alpha=alpha, detail=detail)
        cache[key] = summary_obj
        return summary_obj

    inf = model._fit_inference_info
    XtWX_inv = inf["XtWX_inv"]
    XtWX_inv_aug = inf["XtWX_inv_aug"]
    active_groups = inf["active_groups"]
    known_scale = getattr(model._distribution, "scale_known", True)
    coef_rows = build_coef_rows(
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        result=res,
        X_a=np.empty((0, 0)),
        W=inf["W"],
        XtWX_inv=XtWX_inv,
        XtWX_inv_aug=XtWX_inv_aug,
        active_groups=active_groups,
        known_scale=known_scale,
        group_edf_map=inf["group_edf_map"],
        reml_lambdas=getattr(model, "_reml_lambdas", None),
        lambda2=fitted_lambda2(model),
        n_obs=n,
        alpha=alpha,
        monotone_repairs=getattr(model, "_monotone_repairs", None),
        precomputed_R_a=inf["R_a"],
        precomputed_edf=inf["edf"],
        precomputed_edf1=inf["edf1"],
        coefficient_estimable_override=inf.get("coefficient_estimable"),
        selected_group_names=selected_names,
        group_matrices=model._dm.group_matrices if model._dm is not None else None,
        sample_weights=model._fit_weights,
    )
    phi = res.phi
    se_dict: dict[str, np.ndarray] = {}
    se_raw_dict: dict[str, np.ndarray] = {}
    for g in model._groups:
        if g.name not in selected_names:
            se_dict[g.name] = np.zeros(g.size)
            se_raw_dict[g.name] = np.zeros(g.size)
        else:
            ag = next((a for a in active_groups if a.name == g.name), None)
            if ag is None:
                se_dict[g.name] = np.zeros(g.size)
                se_raw_dict[g.name] = np.zeros(g.size)
            else:
                aug_sl = slice(1 + ag.start, 1 + ag.end)
                var_diag = np.diag(XtWX_inv_aug[aug_sl, aug_sl])
                se_raw_dict[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
                se_dict[g.name] = np.sqrt(np.maximum(phi * var_diag, 0.0))

    data["standard_errors"] = {
        "coefficient_se": se_dict,
        "coefficient_se_raw": se_raw_dict,
    }
    basis_detail = build_basis_detail(
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        result=res,
        XtWX_inv_aug=XtWX_inv_aug,
        active_groups=active_groups,
        known_scale=known_scale,
        alpha=alpha,
        coefficient_estimable_override=inf.get("coefficient_estimable"),
        selected_group_names=selected_names,
    )

    summary_obj = ModelSummary(
        data, model_info, coef_rows, alpha=alpha, detail=detail, basis_detail=basis_detail
    )
    cache[key] = summary_obj
    return summary_obj


def feature_groups(model, name: str) -> list[GroupSlice]:
    """Get all groups belonging to a feature."""
    return [g for g in model._groups if g.feature_name == name]


def _build_editor_stale_coef_rows(model) -> list[_CoefRow]:
    from superglm.features.interaction import PolynomialCategorical, SplineCategorical
    from superglm.features.ordered_categorical import OrderedCategorical
    from superglm.inference._ordered_reference import ordered_reference_intercept

    intercept = ordered_reference_intercept(
        model.result.intercept,
        model.result.beta,
        model._feature_order,
        model._specs,
        model._groups,
    )
    rows = [_CoefRow(name="Intercept", coef=intercept)]
    group_edf = getattr(model, "_group_edf", None)
    reml_lambdas = getattr(model, "_reml_lambdas", None)
    selected_names = selected_group_name_set(
        model.result,
        model._groups,
        penalty=fitted_penalty(model),
    )
    handled_ordered_features: set[str] = set()

    for g in model._groups:
        beta_g = np.asarray(model.result.beta[g.sl], dtype=float)
        norm = float(np.linalg.norm(beta_g))
        active = g.name in selected_names
        spec = model._specs.get(g.feature_name) or model._interaction_specs.get(g.feature_name)
        edf = group_edf.get(g.name) if group_edf else None

        if isinstance(spec, OrderedCategorical):
            if g.feature_name in handled_ordered_features:
                continue
            handled_ordered_features.add(g.feature_name)

            feature_groups = [fg for fg in model._groups if fg.feature_name == g.feature_name]
            beta_combined = np.concatenate([model.result.beta[fg.sl] for fg in feature_groups])
            feature_edf = (
                sum(group_edf.get(fg.name, 0.0) for fg in feature_groups) if group_edf else None
            )
            raw = spec.reconstruct(beta_combined)
            if spec.basis == "spline":
                metadata = spline_group_enrichment(
                    feature_groups[0].name,
                    spec._spline,
                    group_edf,
                    reml_lambdas,
                    fitted_lambda2(model),
                )
                metadata["edf"] = feature_edf
                rows.append(
                    _CoefRow(
                        name=g.feature_name,
                        group=g.feature_name,
                        is_spline=True,
                        n_params=len(beta_combined),
                        active=any(fg.name in selected_names for fg in feature_groups),
                        group_norm=float(np.linalg.norm(beta_combined)),
                        subgroup_type="ordered_spline",
                        **metadata,
                    )
                )
                for level in raw["levels"]:
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=g.feature_name,
                            coef=float(raw["level_log_relativities"][level]),
                        )
                    )
            else:
                row_idx = 0
                for level in raw["levels"]:
                    if level == spec._base_level:
                        continue
                    rows.append(
                        _CoefRow(
                            name=f"{g.feature_name}[{level}]",
                            group=g.feature_name,
                            coef=float(raw["log_relativities"][level]),
                            edf=feature_edf if row_idx == 0 else None,
                        )
                    )
                    row_idx += 1
            continue

        if isinstance(spec, SplineCategorical | PolynomialCategorical):
            metadata = spline_group_enrichment(
                g.name,
                spec,
                group_edf,
                reml_lambdas,
                fitted_lambda2(model),
            )
            rows.append(
                _CoefRow(
                    name=g.name,
                    group=g.feature_name,
                    is_spline=True,
                    n_params=g.size,
                    active=active,
                    group_norm=norm,
                    subgroup_type=g.subgroup_type,
                    **metadata,
                )
            )
            continue

        non_base = getattr(spec, "_non_base", None)
        if non_base is not None and len(non_base) >= g.size:
            for i, level in enumerate(non_base[: g.size]):
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{level}]",
                        group=g.name,
                        coef=float(beta_g[i]),
                        edf=edf if i == 0 else None,
                    )
                )
            continue

        pairs = getattr(spec, "_pairs", None)
        if pairs is not None and len(pairs) >= g.size:
            for i, (lev1, lev2) in enumerate(pairs[: g.size]):
                rows.append(
                    _CoefRow(
                        name=f"{g.name}[{lev1}:{lev2}]",
                        group=g.name,
                        coef=float(beta_g[i]),
                        edf=edf if i == 0 else None,
                    )
                )
            continue

        if g.size == 1:
            rows.append(_CoefRow(name=g.name, group=g.name, coef=float(beta_g[0]), edf=edf))
            continue

        metadata = (
            spline_group_enrichment(
                g.name,
                spec,
                group_edf,
                reml_lambdas,
                fitted_lambda2(model),
            )
            if spec is not None
            else {}
        )
        rows.append(
            _CoefRow(
                name=g.name,
                group=g.feature_name,
                is_spline=True,
                n_params=g.size,
                active=active,
                group_norm=norm,
                subgroup_type=g.subgroup_type,
                **metadata,
            )
        )

    return rows


def _suppress_editor_inference(coef_rows) -> None:
    for row in coef_rows:
        row.se = None
        row.z = None
        row.p = None
        row.ci_low = None
        row.ci_high = None
        row.wald_chi2 = None
        row.wald_p = None
        row.ref_df = None
        row.curve_se_min = None
        row.curve_se_max = None


def reconstruct_feature(model, name: str) -> dict[str, Any]:
    """Reconstruct a fitted feature's curve or effect on its original scale."""
    res = model.result
    groups = feature_groups(model, name)
    beta_combined = np.concatenate([res.beta[g.sl] for g in groups])
    in_main = name in model._specs
    in_inter = name in model._interaction_specs
    if in_main and in_inter:
        raise ValueError(
            f"Ambiguous name {name!r}: exists as both a main effect "
            f"and an interaction. Use the feature or interaction spec "
            f"directly to disambiguate."
        )
    if in_main:
        return cast(dict[str, Any], model._specs[name].reconstruct(beta_combined))
    if in_inter:
        return cast(dict[str, Any], model._interaction_specs[name].reconstruct(beta_combined))
    raise KeyError(f"Feature not found: {name}")


def knot_summary(model) -> dict[str, dict[str, Any]]:
    """Return fitted knot metadata for all spline features."""
    from superglm.features.spline import _SplineBase

    out: dict[str, dict[str, Any]] = {}
    for name, spec in model._specs.items():
        if not isinstance(spec, _SplineBase):
            continue
        entry: dict[str, Any] = {
            "kind": type(spec).__name__,
            "knot_strategy": spec._knot_strategy_actual,
            "interior_knots": spec.fitted_knots,
            "boundary": spec.fitted_boundary,
            "n_basis": spec._n_basis,
        }
        if spec._knot_strategy_actual == "quantile_tempered":
            entry["knot_alpha"] = spec.knot_alpha
        out[name] = entry
    return out
