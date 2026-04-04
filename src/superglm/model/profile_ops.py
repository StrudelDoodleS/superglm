"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import logging

from superglm.distributions import NegativeBinomial, Tweedie

logger = logging.getLogger(__name__)


def estimate_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    phi_method="pearson",
    method="brent",
    **kwargs,
):
    """Estimate Tweedie p via profile likelihood, refit, and return result."""
    from superglm.profiling.tweedie import estimate_tweedie_p

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
        sample_weight=sample_weight,
        offset=offset,
        fit_mode=resolved_mode,
        phi_method=phi_method,
        method=method,
        **kwargs,
    )
    model.family = Tweedie(p=result.p_hat)

    # Refit with the same regime used for profiling (clears stale profile results)
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

    # Eagerly compute the default CI so summary() doesn't trigger
    # expensive profile refits on first access.  The REML CI objective
    # mutates model.family during Brent evaluations, so save and restore
    # the full model state around the CI computation.
    if result._objective is not None:
        saved_family = model.family
        saved_result = model._result
        saved_fit_stats = model._fit_stats
        saved_profile = model._tweedie_profile_result

        result.ci(alpha=0.05)

        # Restore model state (CI refits leave model at last Brent eval)
        model.family = saved_family
        model._result = saved_result
        model._fit_stats = saved_fit_stats
        model._tweedie_profile_result = saved_profile

    return result


def estimate_theta(model, X, y, sample_weight=None, offset=None, **kwargs):
    """Estimate NB theta via profile likelihood, refit, and return result."""
    from superglm.profiling.nb import estimate_nb_theta

    result = estimate_nb_theta(model, X, y, sample_weight=sample_weight, offset=offset, **kwargs)
    model.family = NegativeBinomial(theta=result.theta_hat)
    model.fit(X, y, sample_weight=sample_weight, offset=offset)
    model._nb_profile_result = result  # after refit so fit()'s clear doesn't wipe it
    return result
