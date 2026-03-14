"""Summary, diagnostics, reconstruction, and covariance computations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from superglm.inference import compute_coef_covariance
from superglm.types import GroupSlice


def diagnostics(model) -> dict[str, Any]:
    """Per-group diagnostic dict for programmatic / audit access."""
    from superglm.features.spline import _SplineBase
    from superglm.inference import spline_group_enrichment

    res = model.result
    group_edf = model._group_edf
    reml_lam = getattr(model, "_reml_lambdas", None)

    out = {}
    for g in model._groups:
        bg = res.beta[g.sl]
        entry: dict[str, Any] = {
            "active": bool(np.any(bg != 0)),
            "group_norm": float(np.linalg.norm(bg)),
            "n_params": g.size,
        }
        spec = model._specs.get(g.feature_name)
        if isinstance(spec, _SplineBase):
            entry.update(spline_group_enrichment(g.name, spec, group_edf, reml_lam, model.lambda2))
        out[g.name] = entry
    out["_model"] = {
        "intercept": res.intercept,
        "deviance": res.deviance,
        "phi": res.phi,
        "effective_df": res.effective_df,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "lambda1": model.penalty.lambda1,
    }
    return out


def summary(model, alpha: float = 0.05):
    """Rich model summary with coefficient table (statsmodels-style)."""
    from scipy.special import gammaln

    from superglm.metrics import build_coef_rows
    from superglm.summary import ModelSummary

    if model._fit_stats is None:
        raise RuntimeError("No fit stats — call fit() or fit_reml() first.")

    fs = model._fit_stats
    res = model.result
    edf = res.effective_df
    n = fs.n_obs
    ll = fs.log_likelihood

    # Derived ICs
    aic = -2 * ll + 2 * edf
    bic = -2 * ll + np.log(n) * edf
    denom = n - edf - 1.0
    aicc = aic + 2 * edf * (edf + 1) / denom if denom > 0 else np.inf
    n_active = sum(1 for g in model._groups if np.linalg.norm(res.beta[g.sl]) > 1e-12)
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

    # Model info (same structure as ModelMetrics.summary())
    penalty = model.penalty
    link_name = type(model._link).__name__
    if link_name.endswith("Link"):
        link_name = link_name[:-4]
    penalty_name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", type(penalty).__name__)

    # Append "+ SEL" if any select=True splines are present
    has_select = any(g.subgroup_type is not None for g in model._groups)
    penalty_abbrevs: dict[str, str] = {}  # abbrev -> full name
    if has_select:
        short = {"Group Lasso": "GL", "Sparse Group Lasso": "SGL", "Group Elastic Net": "GEN"}
        abbrev = short.get(penalty_name)
        if abbrev is not None:
            penalty_abbrevs[abbrev] = penalty_name
            penalty_name = abbrev
        penalty_abbrevs["SEL"] = "double penalty selection (Wood, 2011)"
        penalty_name += " + SEL"

    # Build method string from fit metadata
    meta = model._last_fit_meta or {}
    method_parts = []
    if meta.get("method") == "fit_reml":
        method_parts.append("REML")
    else:
        method_parts.append("ML")
    if meta.get("discrete"):
        method_parts.append("discrete")
    method_str = ", ".join(method_parts)

    lam1 = penalty.lambda1
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
        "deviance": res.deviance,
        "log_likelihood": ll,
        "aic": aic,
        "converged": res.converged,
        "n_iter": res.n_iter,
    }

    # NB theta profile info
    nb_pr = getattr(model, "_nb_profile_result", None)
    if nb_pr is not None:
        ci = nb_pr.ci(alpha=alpha)
        model_info["nb_theta"] = nb_pr.theta_hat
        model_info["nb_theta_ci"] = ci
        model_info["nb_theta_method"] = "Profile (exact)"

    # Tweedie p profile info
    tw_pr = getattr(model, "_tweedie_profile_result", None)
    if tw_pr is not None:
        ci = tw_pr.ci(alpha=alpha)
        model_info["tweedie_p"] = tw_pr.p_hat
        model_info["tweedie_p_ci"] = ci
        model_info["tweedie_phi"] = tw_pr.phi_hat
        model_info["tweedie_p_method"] = "Profile (exact)"

    # Coef rows from shared builder
    X_a, W, XtWX_inv, active_groups = model._fit_active_info
    known_scale = getattr(model._distribution, "scale_known", True)
    coef_rows = build_coef_rows(
        groups=model._groups,
        specs=model._specs,
        interaction_specs=model._interaction_specs,
        result=res,
        X_a=X_a,
        W=W,
        XtWX_inv=XtWX_inv,
        active_groups=active_groups,
        known_scale=known_scale,
        group_edf_map=model._group_edf,
        reml_lambdas=getattr(model, "_reml_lambdas", None),
        lambda2=model.lambda2,
        n_obs=n,
        alpha=alpha,
    )

    # Standard errors for backward compat dict access
    phi = res.phi
    se_dict: dict[str, NDArray] = {}
    se_raw_dict: dict[str, NDArray] = {}
    beta = res.beta
    for g in model._groups:
        if np.linalg.norm(beta[g.sl]) < 1e-12:
            se_dict[g.name] = np.zeros(g.size)
            se_raw_dict[g.name] = np.zeros(g.size)
        else:
            ag = next((a for a in active_groups if a.name == g.name), None)
            if ag is None:
                se_dict[g.name] = np.zeros(g.size)
                se_raw_dict[g.name] = np.zeros(g.size)
            else:
                var_diag = np.diag(XtWX_inv[ag.sl, ag.sl])
                se_raw_dict[g.name] = np.sqrt(np.maximum(var_diag, 0.0))
                se_dict[g.name] = np.sqrt(np.maximum(phi * var_diag, 0.0))

    data["standard_errors"] = {
        "coefficient_se": se_dict,
        "coefficient_se_raw": se_raw_dict,
    }

    return ModelSummary(data, model_info, coef_rows, alpha=alpha)


def feature_groups(model, name: str) -> list[GroupSlice]:
    """Get all groups belonging to a feature."""
    return [g for g in model._groups if g.feature_name == name]


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
        return model._specs[name].reconstruct(beta_combined)
    if in_inter:
        return model._interaction_specs[name].reconstruct(beta_combined)
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


def coef_covariance(model):
    """Phi-scaled Bayesian covariance for active coefficients."""
    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    return compute_coef_covariance(
        model._dm,
        model._distribution,
        model._link,
        model._groups,
        model.result,
        model._fit_weights,
        model._fit_offset,
        lam2,
    )


def fit_active_info(model):
    """Active design columns, weights, and (X'WX+S)^{-1} from fit state."""
    from superglm.distributions import clip_mu
    from superglm.metrics import _penalised_xtwx_inv

    eta = model._dm.matvec(model.result.beta) + model.result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    mu = clip_mu(model._link.inverse(np.clip(eta, -20, 20)), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(np.clip(eta, -20, 20))
    W = model._fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    X_a, XtWX_inv, active_groups, _ = _penalised_xtwx_inv(
        model.result.beta, W, model._dm.group_matrices, model._groups, lam2
    )
    return X_a, W, XtWX_inv, active_groups


def group_edf(model) -> dict[str, float] | None:
    """Per-group effective degrees of freedom via F = (X'WX+S)^{-1} X'WX."""
    from superglm.distributions import clip_mu
    from superglm.metrics import _penalised_xtwx_inv

    if model._dm is None or model._result is None:
        return None

    beta = model._result.beta
    eta = model._dm.matvec(beta) + model._result.intercept
    if model._fit_offset is not None:
        eta = eta + model._fit_offset
    eta = np.clip(eta, -20, 20)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    V = model._distribution.variance(mu)
    dmu_deta = model._link.deriv_inverse(eta)
    W = model._fit_weights * dmu_deta**2 / np.maximum(V, 1e-10)

    lam2 = getattr(model, "_reml_lambdas", None) or model.lambda2
    X_a, XtWX_S_inv, active_groups, _ = _penalised_xtwx_inv(
        beta, W, model._dm.group_matrices, model._groups, lam2
    )

    if X_a.shape[1] == 0:
        return {}

    XtWX = X_a.T @ (X_a * W[:, None])
    F = XtWX_S_inv @ XtWX
    edf_vec = np.diag(F)

    out: dict[str, float] = {}
    for ag in active_groups:
        out[ag.name] = float(np.sum(edf_vec[ag.sl]))
    return out
