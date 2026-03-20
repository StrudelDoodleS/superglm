"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import logging

from superglm.distributions import NegativeBinomial, Tweedie

logger = logging.getLogger(__name__)


def estimate_p(
    model,
    X,
    y,
    exposure=None,
    offset=None,
    *,
    sample_weight=None,
    fit_mode="fit",
    phi_method="pearson",
    **kwargs,
):
    """Estimate Tweedie p via profile likelihood, refit, and return result."""
    from superglm.model.base import resolve_sample_weight_alias
    from superglm.tweedie_profile import estimate_tweedie_p

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="estimate_p()")

    # Resolve to internal method name: "fit" or "fit_reml"
    _VALID_FIT_MODES = {"fit", "reml", "inherit"}
    if fit_mode not in _VALID_FIT_MODES:
        raise ValueError(
            f"fit_mode={fit_mode!r} is not valid, expected one of {sorted(_VALID_FIT_MODES)}"
        )
    if fit_mode == "reml":
        resolved_mode = "fit_reml"
    elif fit_mode == "inherit":
        if model._last_fit_meta is not None:
            resolved_mode = model._last_fit_meta["method"]
        else:
            resolved_mode = "fit"
    else:
        resolved_mode = "fit"

    result = estimate_tweedie_p(
        model,
        X,
        y,
        exposure=exposure,
        offset=offset,
        fit_mode=resolved_mode,
        phi_method=phi_method,
        **kwargs,
    )
    model.family = Tweedie(p=result.p_hat)
    model._tweedie_profile_result = result

    # Refit with the same regime used for profiling
    if resolved_mode == "fit_reml":
        model.fit_reml(X, y, exposure=exposure, offset=offset)
    else:
        model.fit(X, y, exposure=exposure, offset=offset)

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


def estimate_theta(model, X, y, exposure=None, offset=None, *, sample_weight=None, **kwargs):
    """Estimate NB theta via profile likelihood, refit, and return result."""
    from superglm.model.base import resolve_sample_weight_alias
    from superglm.nb_profile import estimate_nb_theta

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="estimate_theta()")

    result = estimate_nb_theta(model, X, y, exposure=exposure, offset=offset, **kwargs)
    model.family = NegativeBinomial(theta=result.theta_hat)
    model._nb_profile_result = result
    model.fit(X, y, exposure=exposure, offset=offset)
    return result
