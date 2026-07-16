"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import logging

from superglm.distributions import NegativeBinomial, Tweedie
from superglm.profiling._reporting import cached_tweedie_profile_ci

logger = logging.getLogger(__name__)


def estimate_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    phi_method="mle",
    method="brent",
    progress_callback=None,
    **kwargs,
):
    """Estimate Tweedie p via profile likelihood, refit, and return result."""
    from superglm.profiling.tweedie import estimate_tweedie_p

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)

    result = estimate_tweedie_p(
        model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        fit_mode=resolved_mode,
        phi_method=phi_method,
        method=method,
        **kwargs,
    )
    if progress_callback is not None:
        progress_callback("best_found", {"profile_estimate": _tweedie_estimate_payload(result)})
    model.family = Tweedie(p=result.p_hat)

    # Refit with the same regime used for profiling (clears stale profile results)
    if progress_callback is not None:
        progress_callback("final_refit", {"profile_estimate": _tweedie_estimate_payload(result)})
    if resolved_mode == "fit_reml":
        model.fit_reml(X, y, sample_weight=sample_weight, offset=offset)
    else:
        model.fit(X, y, sample_weight=sample_weight, offset=offset)

    # Set after refit so the clear in fit() doesn't wipe it
    model._tweedie_profile_result = result

    # Use the profiler's phi so summary LL/AIC/BIC are consistent with
    # the profile NLL (both evaluate the density at the same dispersion).
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import _compute_fit_stats

    model._result.phi = result.phi_hat
    # Recompute fit stats (LL, AIC inputs) at the profiler's phi
    eta = model._dm.matvec(model._result.beta) + model._result.intercept
    offset_arr = model._fit_offset
    if offset_arr is not None:
        eta = eta + offset_arr
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), model._distribution)
    weights = model._fit_weights
    model._fit_stats = _compute_fit_stats(
        y, mu, weights, offset_arr, model._distribution, model._link, result.phi_hat
    )

    return result


def estimate_theta(model, X, y, sample_weight=None, offset=None, *, fit_mode="fit", **kwargs):
    """Estimate NB theta via profile likelihood, refit, and return result."""
    from superglm.profiling.nb import estimate_nb_theta

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)

    progress_callback = kwargs.pop("progress_callback", None)
    result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset, **kwargs)
    if progress_callback is not None:
        progress_callback("best_found", {"profile_estimate": _theta_estimate_payload(result)})
    model.family = NegativeBinomial(theta=result.theta_hat)
    if progress_callback is not None:
        progress_callback("final_refit", {"profile_estimate": _theta_estimate_payload(result)})
    if resolved_mode == "fit_reml":
        model.fit_reml(X, y, sample_weight=sample_weight, offset=offset)
    else:
        model.fit(X, y, sample_weight=sample_weight, offset=offset)
    model._nb_profile_result = result  # after refit so fit()'s clear doesn't wipe it
    return result


def _resolve_profile_fit_mode(model, fit_mode: str) -> str:
    """Resolve public profile fit mode to an internal final-refit method."""
    valid_fit_modes = {"fit", "reml", "inherit"}
    if fit_mode not in valid_fit_modes:
        raise ValueError(
            f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(valid_fit_modes)}"
        )
    if fit_mode == "reml":
        return "fit_reml"
    if fit_mode == "inherit":
        meta = getattr(model, "_last_fit_meta", None)
        if meta is not None and meta.get("method") == "fit_reml":
            return "fit_reml"
    return "fit"


def _tweedie_estimate_payload(result):
    ci, ci_status = cached_tweedie_profile_ci(result, 0.05)
    ci_low, ci_high = (None, None) if ci is None else ci
    return {
        "parameter": "p",
        "label": "p_hat",
        "value": getattr(result, "p_hat", None),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_status": ci_status,
        "objective": getattr(result, "nll", None),
        "objective_label": "loss",
        "lower_is_better": True,
    }


def _theta_estimate_payload(result):
    return {
        "parameter": "theta",
        "label": "theta_hat",
        "value": getattr(result, "theta_hat", None),
        "ci_low": _cached_ci(result)[0],
        "ci_high": _cached_ci(result)[1],
        "objective": getattr(result, "nll", None),
        "objective_label": "loss",
        "lower_is_better": True,
    }


def _cached_ci(result):
    cache = getattr(result, "_ci_cache", None)
    if isinstance(cache, dict) and 0.05 in cache:
        return cache[0.05]
    return (None, None)
