"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import copy
import logging
from dataclasses import fields, is_dataclass, replace

import numpy as np

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
    method="auto",
    progress_callback=None,
    **kwargs,
):
    """Estimate Tweedie p and atomically publish one profiled final fit."""
    from superglm.model import fit_ops
    from superglm.model.fit_state import (
        ModelConfigPublication,
        _install_fit_state,
        capture_fit_state,
    )
    from superglm.model.fit_workspace import FitWorkspace
    from superglm.profiling.tweedie import estimate_tweedie_p

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)

    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = fit_ops._validate_entrypoint_input(
        model,
        X,
        y,
        sample_weight,
        offset,
    )
    validated_inputs = (X, y, sample_weight, offset)

    # Profiling is attempt-local too: lazy CI closures may retain this model,
    # but can never retain or mutate the caller's installed fitted revision.
    profile_workspace = FitWorkspace.start(
        model,
        mode="estimate_p_profile",
        validated_inputs=validated_inputs,
    )
    result = estimate_tweedie_p(
        profile_workspace.model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        fit_mode=resolved_mode,
        phi_method=phi_method,
        method=method,
        **kwargs,
    )
    del profile_workspace
    if progress_callback is not None:
        progress_callback("best_found", {"profile_estimate": _tweedie_estimate_payload(result)})

    if progress_callback is not None:
        progress_callback("final_refit", {"profile_estimate": _tweedie_estimate_payload(result)})

    selected_family = Tweedie(p=result.p_hat)
    selected_config = model._config.with_value(family=selected_family)
    final_workspace = FitWorkspace.start(
        model,
        mode=resolved_mode,
        validated_inputs=validated_inputs,
        config_overrides={
            "family": selected_family,
            # Synchronization needs fitted rows even when the durable public
            # state is compact. Release them only after phi and fit statistics
            # have been revised on this private candidate.
            "retain_fit_state": True,
        },
    )
    debug_recorder = None
    if resolved_mode == "fit_reml":
        debug_recorder = fit_ops._fit_reml_in_workspace(
            final_workspace.model,
            X,
            y,
            sample_weight,
            offset,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
            pirls_tol=final_workspace.model._tol,
            max_pirls_iter=final_workspace.model._max_iter,
        )
    else:
        fit_ops._fit_in_workspace(
            final_workspace.model,
            X,
            y,
            sample_weight,
            offset,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
        )

    final_model = final_workspace.model
    _synchronize_tweedie_profile_refit(final_model, y, result)
    if not model._retain_fit_state:
        final_model._retain_fit_state = False
        fit_ops._maybe_release_fit_state(final_model)
    # Install last on the private candidate too: any state carrying this
    # result has already been synchronized and, if requested, compacted.
    installed_result = _installed_tweedie_profile_copy(result)
    final_model._tweedie_profile_result = installed_result

    candidate = capture_fit_state(
        final_workspace,
        model,
        revision=model._fit_revision + 1,
        config_publication=replace(
            ModelConfigPublication.capture(model),
            config=selected_config,
            revision=model._config_revision + 1,
            family=final_model._family_config,
        ),
    )
    _install_fit_state(model, candidate)
    if resolved_mode == "fit_reml":
        fit_ops._record_reml_terminal_best_effort(model, debug_recorder)

    return result


_TWEEDIE_PROFILE_SHARED_RUNTIME_FIELDS = frozenset(
    {
        "_objective",
        "_evaluation_count",
        "_evaluation_record",
    }
)


def _installed_tweedie_profile_copy(result):
    """Detach published estimates while retaining lazy-CI runtime caches.

    The public result remains usable for later lazy likelihood-ratio inference.
    Its objective/evaluation registry is shared, but estimate-dependent CI
    caches are independently owned: mutating a returned estimate and then
    calling ``ci()`` must not poison the installed model's connected profile
    component. Core estimates and reporting containers are independent too.
    """
    installed = copy.copy(result)
    field_names = (
        (field.name for field in fields(result)) if is_dataclass(result) else iter(vars(result))
    )
    for field_name in field_names:
        if field_name in _TWEEDIE_PROFILE_SHARED_RUNTIME_FIELDS:
            continue
        setattr(installed, field_name, copy.deepcopy(getattr(result, field_name)))
    return installed


def _replace_dataclass_preserving_dynamic_attributes(instance, **changes):
    """Replace dataclass fields without dropping solver-added attributes."""
    replacement = replace(instance, **changes)
    declared_names = {field.name for field in fields(instance)}
    for name, value in vars(instance).items():
        if name not in declared_names:
            setattr(replacement, name, value)
    return replacement


def _replace_pirls_phi(result, phi):
    """Return a phi-adjusted PIRLS result with all runtime metadata intact."""
    return _replace_dataclass_preserving_dynamic_attributes(result, phi=float(phi))


def _synchronize_tweedie_profile_refit(model, y, profile_result) -> None:
    """Atomically synchronize a retained final refit to the profiled dispersion."""
    from superglm.distributions import clip_mu
    from superglm.links import stabilize_eta
    from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu

    distribution = model._distribution
    if not isinstance(distribution, Tweedie) or distribution.p != profile_result.p_hat:
        raise RuntimeError("Final Tweedie refit does not match the profiled power parameter")
    if model._dm is None or model._fit_weights is None:
        raise RuntimeError("Final Tweedie refit state was released before synchronization")

    public_result = model.result
    solver_result = model._solver_pirls_result()
    weights = model._fit_weights
    offset_arr = model._fit_offset
    y_arr = np.asarray(y, dtype=np.float64)

    eta = model._dm.matvec(solver_result.beta) + solver_result.intercept
    if offset_arr is not None:
        eta = eta + offset_arr
    eta = stabilize_eta(eta, model._link)
    mu = clip_mu(model._link.inverse(eta), distribution)
    null_mu = _compute_null_mu(y_arr, weights, offset_arr, distribution, model._link)
    fit_stats = _compute_fit_stats(
        y_arr,
        mu,
        weights,
        offset_arr,
        distribution,
        model._link,
        profile_result.phi_hat,
        null_mu=null_mu,
    )

    replacement_public = _replace_pirls_phi(public_result, profile_result.phi_hat)
    replacement_solver = _replace_pirls_phi(solver_result, profile_result.phi_hat)
    reml_result = getattr(model, "_reml_result", None)
    replacement_reml = (
        None
        if reml_result is None
        else _replace_dataclass_preserving_dynamic_attributes(
            reml_result, pirls_result=replacement_solver
        )
    )

    model._result = replacement_public
    model._solver_result = replacement_solver
    if reml_result is not None:
        model._reml_result = replacement_reml
    model._fit_mu = mu
    model._fit_null_mu = null_mu
    model._fit_stats = fit_stats

    for cache_name in (
        "_coef_covariance",
        "_fit_active_info",
        "_fit_inference_info",
        "_group_edf",
    ):
        model.__dict__.pop(cache_name, None)
    model._fit_metrics_cache = None
    model._fit_metrics_cache_signature = None
    model._summary_cache = None


def estimate_theta(model, X, y, sample_weight=None, offset=None, *, fit_mode="fit", **kwargs):
    """Estimate NB theta and atomically publish one profiled final fit."""
    from superglm.model import fit_ops
    from superglm.model.fit_state import (
        ModelConfigPublication,
        _install_fit_state,
        capture_fit_state,
    )
    from superglm.model.fit_workspace import FitWorkspace
    from superglm.profiling.nb import estimate_nb_theta

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)
    progress_callback = kwargs.pop("progress_callback", None)

    X_ref = X
    y_ref = y
    sample_weight_ref = sample_weight
    offset_ref = offset
    X, y, sample_weight, offset = fit_ops._validate_entrypoint_input(
        model,
        X,
        y,
        sample_weight,
        offset,
    )
    validated_inputs = (X, y, sample_weight, offset)

    profile_workspace = FitWorkspace.start(
        model,
        mode="estimate_theta_profile",
        validated_inputs=validated_inputs,
    )
    result = estimate_nb_theta(
        profile_workspace.model,
        X,
        y,
        sample_weight=sample_weight,
        offset=offset,
        **kwargs,
    )
    # The profile result retains only the vectors needed for reporting/CI.
    # Release its design workspace before allocating the final-fit design.
    del profile_workspace
    if progress_callback is not None:
        progress_callback("best_found", {"profile_estimate": _theta_estimate_payload(result)})
    if progress_callback is not None:
        progress_callback("final_refit", {"profile_estimate": _theta_estimate_payload(result)})

    selected_family = NegativeBinomial(theta=result.theta_hat)
    selected_config = model._config.with_value(family=selected_family)
    final_workspace = FitWorkspace.start(
        model,
        mode=resolved_mode,
        validated_inputs=validated_inputs,
        config_overrides={
            "family": selected_family,
            # Profile publication must synchronize against the final refit even
            # when the public model requests compact fitted state.  Row-scale
            # buffers are released again before the atomic install below.
            "retain_fit_state": True,
        },
    )
    debug_recorder = None
    if resolved_mode == "fit_reml":
        debug_recorder = fit_ops._fit_reml_in_workspace(
            final_workspace.model,
            X,
            y,
            sample_weight,
            offset,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
            pirls_tol=final_workspace.model._tol,
            max_pirls_iter=final_workspace.model._max_iter,
        )
    else:
        fit_ops._fit_in_workspace(
            final_workspace.model,
            X,
            y,
            sample_weight,
            offset,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
        )

    final_model = final_workspace.model
    installed_result = result._published_with_data(
        y,
        final_model._fit_mu,
        final_model._fit_weights,
    )
    final_model._nb_profile_result = installed_result
    if not model._retain_fit_state:
        final_model._retain_fit_state = False
        fit_ops._maybe_release_fit_state(final_model)
    # Allocate the distinct public handle before the no-fail dictionary swap.
    # A future custom result implementation may make this operation fallible;
    # such a failure must preserve the previously installed model revision.
    public_result = installed_result._detached_public_copy()
    candidate = capture_fit_state(
        final_workspace,
        model,
        revision=model._fit_revision + 1,
        config_publication=replace(
            ModelConfigPublication.capture(model),
            config=selected_config,
            revision=model._config_revision + 1,
            family=final_model._family_config,
        ),
    )
    _install_fit_state(model, candidate)
    if resolved_mode == "fit_reml":
        fit_ops._record_reml_terminal_best_effort(model, debug_recorder)
    return public_result


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
