"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import copy
import logging
from dataclasses import fields, is_dataclass, replace

import numpy as np

from superglm.distributions import NegativeBinomial, Tweedie
from superglm.profiling._reporting import cached_tweedie_profile_ci
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError

logger = logging.getLogger(__name__)


class PublicationModeError(RuntimeError):
    """The publication refit could not certify a penalized coefficient mode.

    Subclasses ``RuntimeError`` so pre-existing broad handlers keep working;
    the dedicated type lets callers route this recoverable certifiability
    condition without string matching. The certification failure that caused
    it is chained as ``__cause__``.
    """


def _publication_mode_failure(exc, *, parameter, value, decoupled) -> PublicationModeError:
    """Turn a mode-certification failure at the publish refit into guidance.

    The search either never evaluated REML at the selected point (a decoupled
    p search, and every theta search -- alternating ML fits) or evaluated it
    under trial smoothing parameters that differ from the final ones, so
    publication is where this can first surface. Left untranslated, the
    caller gets an internal certification message with no mention of which
    point failed or what to do about it.
    """
    score = getattr(exc, "relative_max", float("inf"))
    achieved = (
        f"relative mode score {score:.3e} against a bar of {exc.tolerance:.3e}"
        if np.isfinite(score)
        else "PIRLS found no converged penalized mode"
    )
    region = (
        "  The certifiable region is a property of this data; its boundary moves "
        "with the realisation and cannot be widened by solver settings.\n"
    )
    if parameter == "theta":
        how = (
            f"The theta search selected theta={value:.6g} through alternating ML fits "
            "without evaluating REML certifiability, and the REML publication refit "
            f"cannot certify a penalized mode there ({achieved})."
        )
        options = (
            "  Options: fit_mode='fit' publishes the ML fit at the selected theta; "
            "or restrict theta_bounds away from the failing region."
        )
    else:
        if decoupled:
            how = (
                f"The ML search selected {parameter}={value:.6g} without evaluating REML "
                "certifiability, and the REML publication refit cannot certify a "
                f"penalized mode there ({achieved})."
            )
        else:
            how = (
                f"The search certified {parameter}={value:.6g}, but the publication refit "
                f"-- which runs at the final smoothing parameters -- cannot ({achieved})."
            )
        options = (
            "  Options: fit_mode='reml' searches only certifiable points (it may warn "
            "that the optimum is boundary-censored); fit_mode='fit' publishes the ML "
            f"fit at the selected {parameter}; or restrict p_bounds away from the "
            "failing region."
        )
    return PublicationModeError(f"{how}\n{region}{options}")


def estimate_p(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    search_fit_mode=None,
    phi_method="mle",
    method="auto",
    ci_alpha=None,
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
    from superglm.profiling.tweedie import _validate_profile_ci_alpha, estimate_tweedie_p

    resolved_mode = _resolve_profile_fit_mode(model, fit_mode)
    resolved_search_mode = (
        resolved_mode
        if search_fit_mode is None
        else _resolve_profile_fit_mode(model, search_fit_mode, parameter="search_fit_mode")
    )
    decoupled = resolved_search_mode != resolved_mode
    _validate_profile_selection_mode(model, resolved_mode)
    _validate_profile_selection_mode(model, resolved_search_mode)
    resolved_ci_alpha = None if ci_alpha is None else _validate_profile_ci_alpha(ci_alpha)
    if resolved_ci_alpha is not None and phi_method == "pearson":
        raise RuntimeError(
            "Tweedie likelihood-ratio profile CI requires exact MLE dispersion "
            "profiling (phi_method='mle'); use bootstrap/sandwich inference for "
            "Pearson plug-in profiles."
        )
    # A decoupled run is entitled to an eager interval: `result.ci` inverts the
    # searched objective around its own recorded value at ``p_hat``
    # (``search_nll``), and the eager call below runs only after publication
    # has recorded it. Refusing here while the returned object hands out the
    # same interval lazily would be a contradiction, not a safeguard.

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
        fit_mode=resolved_search_mode,
        phi_method=phi_method,
        method=method,
        **kwargs,
    )
    result.fit_mode = resolved_mode
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
        try:
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
                durable_retain_fit_state=bool(model._retain_fit_state),
            )
        except ObservedModeNotCertifiedError as exc:
            raise _publication_mode_failure(
                exc, parameter="p", value=float(result.p_hat), decoupled=decoupled
            ) from exc
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
    _synchronize_tweedie_profile_refit(
        final_model,
        y,
        result,
        # Both modes publish a refit whose settings differ from the candidate
        # fits -- decoupled by regime, coupled by the publication tolerance --
        # so the published dispersion is profiled against the published mean
        # in both. search_nll keeps the searched curve for the CI and plots.
        reprofile_phi=True,
        phi_method=phi_method,
        # The canonical public mean: on discretized models the internal
        # design's matvec is a binned approximation of it.
        public_mu=final_model.predict(X, offset=offset),
    )
    if not model._retain_fit_state:
        final_model._retain_fit_state = False
        fit_ops._maybe_release_fit_state(final_model)
    if resolved_ci_alpha is not None:
        result.ci(alpha=resolved_ci_alpha)
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
    ci_cache = getattr(installed, "_ci_cache", None)
    if ci_cache is not None:
        owned_ci_cache = {}
        details_cache = getattr(installed, "_ci_details_cache", None)
        for alpha, interval in ci_cache.items():
            owned_interval = (float(interval[0]), float(interval[1]))
            owned_ci_cache[alpha] = owned_interval
            details = None if details_cache is None else details_cache.get(alpha)
            if details_cache is not None and details is not None and is_dataclass(details):
                details_cache[alpha] = replace(details, interval=owned_interval)
        installed._ci_cache = owned_ci_cache
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


def _reprofile_published_dispersion(model, y_arr, weights, mu, profile_result, phi_method) -> None:
    """Re-profile dispersion against the published fit.

    The search profiled phi at its own candidate fits' fitted means. The
    publication refit never reproduces those fits: a decoupled run publishes
    under a different regime entirely, and a coupled run publishes at the
    tight publication tolerance while candidates ran at the search bar.
    Carrying the search's phi across would hand back a dispersion estimated
    from coefficients the caller never receives. Re-profile at the published
    mean instead, and move the objective with it so the reported likelihood
    still refers to the reported estimates.
    """
    from superglm.profiling.tweedie import _profile_phi_detailed

    edf = float(getattr(model.result, "effective_df", 0.0) or 0.0)
    phi_result = _profile_phi_detailed(
        y_arr,
        mu,
        float(profile_result.p_hat),
        weights=weights,
        df_resid=max(float(len(y_arr)) - edf, 1.0),
        phi_method=phi_method,
        phi_start=float(profile_result.phi_hat),
    )
    # Keep the searched objective's value at `p_hat` before moving `nll` onto
    # the published fit's curve. The CI, the profile plot and the deviance
    # curve all measure searched values against it; without it they would
    # subtract this published number and report a likelihood ratio that is
    # negative at the search's own optimum.
    if profile_result.search_nll is None:
        profile_result.search_nll = float(profile_result.nll)
        # Stash the searched winner's certification flags with it: the CI
        # inverts the searched curve, so its guard must judge these after
        # the overwrite below repoints the live flags at the re-profile.
        profile_result.search_objective_finite = bool(profile_result.objective_finite)
        profile_result.search_phi_converged = bool(profile_result.phi_converged)
    profile_result.phi_hat = float(phi_result.phi)
    profile_result.nll = float(phi_result.nll)
    profile_result.objective_finite = bool(phi_result.objective_finite)
    profile_result.phi_converged = bool(phi_result.converged)
    profile_result.phi_optimizer = str(phi_result.optimizer)
    profile_result.phi_score = phi_result.score
    profile_result.phi_used_fallback = bool(phi_result.used_fallback)
    profile_result.phi_fallback_reason = phi_result.fallback_reason
    profile_result.phi_branch_switch_detected = bool(phi_result.branch_switch_detected)
    profile_result.phi_message = str(phi_result.message)
    # The published dispersion is this re-profile, so the whole derived
    # story must be the published story: the boundary label, the density
    # classification, the phi warnings and the aggregate convergence flag
    # all describe the re-profile from here on -- a boundary hit, a
    # fallback, a saddlepoint evaluation or a non-convergent profile here
    # cannot hide behind the search's clean record, and the search's
    # troubles cannot outlive the dispersion they described.
    from superglm.profiling.tweedie import (
        _build_density_messages,
        _classify_density_diagnostics,
        _phi_boundary_label,
        _warning_describes_winner_phi,
    )

    profile_result.phi_boundary = _phi_boundary_label(phi_result)
    density = _classify_density_diagnostics(float(profile_result.p_hat), phi_result.diagnostics)
    profile_result.saddlepoint_fraction = float(density.fraction)
    profile_result.n_saddlepoint = int(density.n_saddlepoint)
    profile_result.n_positive = int(density.n_positive)
    profile_result.density_method = density.method
    profile_result.density_exact = density.exact
    profile_result.density_warning_severity = density.severity
    profile_result.near_power_boundary = bool(density.near_power_boundary)

    kept = [w for w in profile_result.warnings if not _warning_describes_winner_phi(w)]
    kept.extend(_build_density_messages(float(profile_result.p_hat), density))
    if not phi_result.converged or phi_result.used_fallback:
        detail = "did not converge" if not phi_result.converged else "used a fallback"
        kept.append(f"published dispersion re-profile {detail}: {phi_result.message}")
    if profile_result.phi_boundary:
        kept.append(
            f"Published dispersion estimate is at the {profile_result.phi_boundary} "
            "dispersion boundary."
        )
    profile_result.warnings = kept
    # `converged` aggregates the same components `_finalize_profile_record`
    # combined, with the phi flags now the published re-profile's: the
    # aggregate and `phi_converged` cannot disagree on the result callers
    # actually receive.
    profile_result.converged = bool(
        phi_result.objective_finite
        and profile_result.outer_converged
        and profile_result.fit_converged
        and phi_result.converged
    )
    profile_result.phi_n_evaluations += phi_result.n_evaluations
    profile_result.phi_n_score_evaluations += phi_result.n_score_evaluations
    profile_result.phi_n_value_only_evaluations += phi_result.n_value_only_evaluations
    profile_result.phi_n_fallback_evaluations += phi_result.n_fallback_evaluations


def _synchronize_tweedie_profile_refit(
    model,
    y,
    profile_result,
    *,
    reprofile_phi: bool = False,
    phi_method: str = "mle",
    public_mu=None,
) -> None:
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
    if reprofile_phi:
        # On a discretized model the internal design's matvec is a binned
        # approximation of the mean callers get from predict(); the published
        # dispersion must be profiled at the public mean, so the caller passes
        # it in. The internal mu remains the fallback for direct invocations.
        reprofile_mu = mu if public_mu is None else np.asarray(public_mu, dtype=np.float64)
        _reprofile_published_dispersion(
            model, y_arr, weights, reprofile_mu, profile_result, phi_method
        )
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
    _validate_profile_selection_mode(model, resolved_mode)
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
        try:
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
                durable_retain_fit_state=bool(model._retain_fit_state),
            )
        except ObservedModeNotCertifiedError as exc:
            raise _publication_mode_failure(
                exc, parameter="theta", value=float(result.theta_hat), decoupled=False
            ) from exc
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


def _resolve_profile_fit_mode(model, fit_mode: str, *, parameter: str = "fit_mode") -> str:
    """Resolve public profile fit mode to an internal final-refit method."""
    valid_fit_modes = {"fit", "reml", "inherit"}
    if fit_mode not in valid_fit_modes:
        raise ValueError(
            f"{parameter}={fit_mode!r} is not valid, expected one of {sorted(valid_fit_modes)}"
        )
    if fit_mode == "reml":
        return "fit_reml"
    if fit_mode == "inherit":
        meta = getattr(model, "_last_fit_meta", None)
        if meta is not None and meta.get("method") == "fit_reml":
            return "fit_reml"
    return "fit"


def _validate_profile_selection_mode(model, resolved_mode: str) -> None:
    """Fail REML profile requests before allocating a profile workspace."""
    if resolved_mode != "fit_reml":
        return
    from superglm.model.base import validate_selection_penalty_for_reml
    from superglm.model.fit_state import configured_penalty

    validate_selection_penalty_for_reml(configured_penalty(model))


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
