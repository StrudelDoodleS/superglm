"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import logging

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
    model.tweedie_p = result.p_hat
    model._tweedie_profile_result = result

    # Refit with the same regime used for profiling
    if resolved_mode == "fit_reml":
        model.fit_reml(X, y, exposure=exposure, offset=offset)
    else:
        model.fit(X, y, exposure=exposure, offset=offset)
    return result


def estimate_theta(model, X, y, exposure=None, offset=None, *, sample_weight=None, **kwargs):
    """Estimate NB theta via profile likelihood, refit, and return result."""
    from superglm.model.base import resolve_sample_weight_alias
    from superglm.nb_profile import estimate_nb_theta

    exposure = resolve_sample_weight_alias(exposure, sample_weight, method_name="estimate_theta()")

    result = estimate_nb_theta(model, X, y, exposure=exposure, offset=offset, **kwargs)
    model.nb_theta = result.theta_hat
    model._nb_profile_result = result
    model.fit(X, y, exposure=exposure, offset=offset)
    return result
