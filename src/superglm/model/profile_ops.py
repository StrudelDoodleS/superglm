"""Profile estimation for NB theta and Tweedie p."""

from __future__ import annotations

import copy
import logging
import operator
from dataclasses import fields, replace
from functools import partial

import numpy as np

from superglm.distributions import NegativeBinomial, Tweedie
from superglm.model.fit_state import configured_family
from superglm.model.input_validation import THETA_ESTIMATED
from superglm.profiling._reporting import cached_tweedie_profile_ci
from superglm.profiling.tweedie import profile_phi_at
from superglm.reml.observed_geometry import ObservedModeNotCertifiedError
from superglm.solvers.dispersion import FREQUENCY_WEIGHTS, model_weight_semantics

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
        # A raise site that reached a mode and failed at a later stage names
        # that stage; every site that does mean "PIRLS found no mode" leaves it
        # unset and keeps this wording verbatim.
        else (getattr(exc, "infeasible_detail", None) or "PIRLS found no converged penalized mode")
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
        if decoupled:
            options = (
                "  Options: fit_mode='reml' searches only certifiable points (it may "
                "warn that the optimum is boundary-censored); fit_mode='fit' publishes "
                f"the ML fit at the selected {parameter}; or restrict p_bounds away "
                "from the failing region."
            )
        else:
            # A coupled caller already passed fit_mode='reml'; the search
            # certified this point and the publication refit still cannot, so
            # recommending the coupled search again would recommend the
            # configuration that just failed.
            options = (
                f"  Options: fit_mode='fit' publishes the ML fit at the selected "
                f"{parameter}; or restrict p_bounds away from the failing region."
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
    p_bounds=(1.05, 1.95),
    xatol=1e-3,
    ci_alpha=None,
    max_reml_iter=None,
    progress_callback=None,
):
    """Estimate Tweedie p and atomically publish one profiled final fit."""
    from superglm.model import fit_ops

    publish_mode, search_mode, reml_budget = _resolve_power_request(
        model,
        fit_mode=fit_mode,
        search_fit_mode=search_fit_mode,
        p_bounds=p_bounds,
        xatol=xatol,
        ci_alpha=ci_alpha,
        max_reml_iter=max_reml_iter,
    )
    report = progress_callback or _ignore_progress
    references = {"X_ref": X, "y_ref": y, "sample_weight_ref": sample_weight, "offset_ref": offset}
    validated = fit_ops._validate_entrypoint_input(model, X, y, sample_weight, offset)
    X, y, sample_weight, offset = validated
    _refuse_replication_weights(model, sample_weight)
    result = _search_power_privately(
        model, validated, fit_mode=search_mode, p_bounds=p_bounds, xatol=xatol, report=report
    )
    result.fit_mode = publish_mode
    if ci_alpha is not None:
        result.interval(ci_alpha)
    estimate = {"profile_estimate": _tweedie_estimate_payload(result)}
    report("best_found", estimate)
    report("final_refit", estimate)
    _publish_profiled_family(
        model,
        validated,
        references,
        fit_mode=publish_mode,
        family=Tweedie(p=result.p_hat),
        parameter="p",
        value=result.p_hat,
        synchronize=partial(_install_tweedie_profile, X=X, y=y, offset=offset, result=result),
        decoupled=search_mode != publish_mode,
        max_reml_iter=reml_budget,
    )
    return result


def _search_power_privately(model, validated, *, fit_mode, p_bounds, xatol, report):
    """The power search on an attempt-local copy of the model.

    The result's lazy interval keeps refitting this private model, never the
    caller's installed fitted revision.
    """
    from superglm.model.fit_workspace import FitWorkspace
    from superglm.profiling.tweedie import search_power

    profile_workspace = FitWorkspace.start(
        model, mode="estimate_p_profile", validated_inputs=validated
    )
    return search_power(
        profile_workspace.model,
        *validated,
        fit_mode=fit_mode,
        p_bounds=p_bounds,
        xatol=xatol,
        on_evaluation=lambda row: report("profiling", {"profile_trace": [row]}),
    )


def _ignore_progress(phase, payload=None) -> None:
    """The progress sink when the caller passed none."""


def _require_positive_tolerance(xatol) -> None:
    """Refuse a search tolerance before any candidate fit runs.

    An infinite one lets the bounded search stop at once and publish its best
    endpoint; NaN or a negative one fails only after expensive candidate fits.
    """
    if not (np.isfinite(xatol) and xatol > 0.0):
        raise ValueError(f"xatol must be finite and strictly positive, got {xatol!r}")


def _resolve_power_request(
    model, *, fit_mode, search_fit_mode, p_bounds, xatol, ci_alpha, max_reml_iter
):
    """Validate an estimate_p call before any data is read.

    Returns the publication fit mode, the search fit mode and the REML
    publication budget.
    """
    family = configured_family(model)
    if not isinstance(family, Tweedie):
        raise ValueError(
            f"estimate_p requires a Tweedie family, got {family!r}. "
            "Use families.tweedie(p=...) to create one."
        )
    # The compound Poisson-gamma density exists only strictly inside (1, 2).
    if not 1.0 < p_bounds[0] < p_bounds[1] < 2.0:
        raise ValueError(
            f"p_bounds must be increasing and strictly inside (1, 2), got {p_bounds!r}"
        )
    _require_positive_tolerance(xatol)
    # The interval checks its level too; this one fails before the search runs.
    if ci_alpha is not None and not 0.0 < ci_alpha < 1.0:
        raise ValueError(f"ci_alpha must be strictly between 0 and 1, got {ci_alpha!r}")
    publish_mode = _resolve_profile_fit_mode(model, fit_mode)
    search_mode = (
        publish_mode
        if search_fit_mode is None
        else _resolve_profile_fit_mode(model, search_fit_mode, parameter="search_fit_mode")
    )
    _validate_profile_selection_mode(model, publish_mode)
    _validate_profile_selection_mode(model, search_mode)
    return (
        publish_mode,
        search_mode,
        _publication_reml_budget(max_reml_iter, publish_mode, fit_mode),
    )


def _publication_reml_budget(max_reml_iter, publish_mode: str, fit_mode) -> int:
    """Outer-iteration budget of the REML publication refit; candidate fits keep their own."""
    if max_reml_iter is None:
        return 20
    # Refused under an ML publication: a mode-scoped parameter that silently
    # no-ops is how inert knobs are born.
    if publish_mode != "fit_reml":
        raise ValueError(
            "max_reml_iter controls the REML publication refit and requires "
            f"fit_mode='reml'; this call publishes with fit_mode={fit_mode!r}, "
            "which has no REML iteration to budget"
        )
    # An iteration count is a non-boolean integer (the integer-index protocol):
    # coercing first let 1.9, True and "5" through as budgets. np.bool_ is not
    # a Python bool but implements __index__, so it is named explicitly.
    message = f"max_reml_iter must be an integer iteration count, got {max_reml_iter!r}"
    if isinstance(max_reml_iter, (bool, np.bool_)):
        raise ValueError(message)
    try:
        budget = operator.index(max_reml_iter)
    except TypeError:
        raise ValueError(message) from None
    if budget < 1:
        raise ValueError(
            "max_reml_iter must be >= 1: the publication refit needs at least "
            f"one REML iteration, got {max_reml_iter!r}"
        )
    return budget


def _refuse_replication_weights(model, weights) -> None:
    """The power profile is the prior-weight EDM likelihood.

    Read as replication counts, the weights would need the unit-weight density
    counted w times: a different objective, not a rescaled one. The two
    contracts coincide only at unit weights, so those are admitted.
    """
    if model_weight_semantics(model) == FREQUENCY_WEIGHTS and np.any(weights != 1.0):
        raise ValueError(
            "estimate_p profiles the Tweedie power against the EDM "
            "prior-weight likelihood, so it cannot honour "
            'weight_semantics="frequency" with non-unit weights. Fit with '
            'weight_semantics="prior", or expand the rows the replication '
            "counts stand for and profile with unit weights."
        )


def _publish_profiled_family(
    model,
    validated,
    references,
    *,
    fit_mode,
    family,
    parameter,
    value,
    synchronize,
    decoupled=False,
    max_reml_iter=20,
):
    """Refit at the selected parameter on a private candidate, then install it atomically.

    ``synchronize(final_model)`` restates the refit around the profile estimate
    while the refit still holds its fitted rows; the durable state is compacted
    only afterwards, and the model changes in the single install at the end.
    Returns what ``synchronize`` returns, allocated before that install.
    """
    from superglm.model.fit_workspace import FitWorkspace

    final_workspace = FitWorkspace.start(
        model,
        mode=fit_mode,
        validated_inputs=validated,
        config_overrides={"family": family, "retain_fit_state": True},
    )
    try:
        debug_recorder = _refit_selected(
            final_workspace.model,
            validated,
            references,
            fit_mode,
            max_reml_iter,
            model._retain_fit_state,
        )
    except ObservedModeNotCertifiedError as exc:
        raise _publication_mode_failure(
            exc, parameter=parameter, value=float(value), decoupled=decoupled
        ) from exc
    published = synchronize(final_workspace.model)
    _install_refit(model, final_workspace, family)
    if fit_mode == "fit_reml":
        from superglm.model import fit_ops

        fit_ops._record_reml_terminal_best_effort(model, debug_recorder)
    return published


def _install_refit(model, final_workspace, family) -> None:
    """Compact the synchronized refit as the model asks, then swap it in as one revision."""
    from superglm.model import fit_ops
    from superglm.model.fit_state import (
        ModelConfigPublication,
        _install_fit_state,
        capture_fit_state,
    )

    final_model = final_workspace.model
    if not model._retain_fit_state:
        final_model._retain_fit_state = False
        fit_ops._maybe_release_fit_state(final_model)
    candidate = capture_fit_state(
        final_workspace,
        model,
        revision=model._fit_revision + 1,
        config_publication=replace(
            ModelConfigPublication.capture(model),
            config=model._config.with_value(family=family),
            revision=model._config_revision + 1,
            family=final_model._family_config,
        ),
    )
    _install_fit_state(model, candidate)


def _refit_selected(final_model, validated, references, fit_mode, max_reml_iter, retain):
    """The publication fit: REML at the tight publication default, or an ordinary fit."""
    from superglm.model import fit_ops

    if fit_mode != "fit_reml":
        fit_ops._fit_in_workspace(final_model, *validated, **references)
        return None
    return fit_ops._fit_reml_in_workspace(
        final_model,
        *validated,
        **references,
        max_reml_iter=max_reml_iter,
        pirls_tol=final_model._tol,
        max_pirls_iter=final_model._max_iter,
        durable_retain_fit_state=bool(retain),
    )


def _install_tweedie_profile(final_model, *, X, y, offset, result) -> None:
    """Restate the refit at the profiled dispersion and attach the result.

    The installed copy shares the searched objective but owns its interval
    cache and warnings: an interval computed through the returned result must
    not reach the model's summary.
    """
    # The canonical public mean: on a discretized model the internal design's
    # matvec is a binned approximation of it.
    _synchronize_tweedie_profile_refit(
        final_model, y, final_model.predict(X, offset=offset), result
    )
    # The estimate is converged only if the fit it is published with is.
    reml = getattr(final_model, "_reml_result", None)
    result.converged = bool(
        result.converged and final_model.result.converged and (reml is None or reml.converged)
    )
    final_model._tweedie_profile_result = _detached_copy(result)


def _detached_copy(result):
    """A copy that owns its interval cache and warnings, so later intervals stay apart."""
    detached = copy.copy(result)
    detached._ci_cache = dict(result._ci_cache)
    detached.warnings = list(result.warnings)
    return detached


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


def _reprofile_published_dispersion(result, y_arr, weights, mu) -> None:
    """Profile phi at the PUBLISHED mean; the searched curve keeps search_nll for the CI.

    The search profiled phi at its candidates' means, which the publication
    refit does not reproduce: a decoupled run publishes under another regime,
    a coupled one at the tight publication bar. Carrying the search's phi
    across would report a dispersion of coefficients the caller never receives.
    """
    solved = profile_phi_at(y_arr, mu, weights, result.p_hat)
    result.phi_hat = solved.phi
    result.nll = solved.criterion / float(len(y_arr))


def _synchronize_tweedie_profile_refit(model, y, public_mu, profile_result) -> None:
    """Atomically synchronize a retained final refit to the profiled dispersion."""
    from superglm.model.fit_ops import _compute_fit_stats, _compute_null_mu

    distribution = model._distribution
    weights = model._fit_weights
    offset_arr = model._fit_offset
    _reprofile_published_dispersion(profile_result, y, weights, public_mu)
    null_mu = _compute_null_mu(
        y,
        weights,
        offset_arr,
        distribution,
        model._link,
        weight_semantics=model_weight_semantics(model),
    )
    # One published fit, one mean: the summary statistics describe the same
    # mean the published dispersion was profiled at. Splitting them would
    # publish a hybrid -- public-mean phi inside binned-mean likelihood,
    # Pearson chi-square and explained deviance.
    fit_stats = _compute_fit_stats(
        y,
        public_mu,
        weights,
        offset_arr,
        distribution,
        model._link,
        profile_result.phi_hat,
        null_mu=null_mu,
        weight_semantics=model_weight_semantics(model),
    )

    replacement_public = _replace_pirls_phi(model.result, profile_result.phi_hat)
    replacement_solver = _replace_pirls_phi(model._solver_pirls_result(), profile_result.phi_hat)
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
    model._fit_mu = public_mu
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


def estimate_theta(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    fit_mode="fit",
    theta_bounds=(1e-8, 1e8),
    xatol=1e-2,
    ci_alpha=None,
    progress_callback=None,
):
    """Estimate NB theta and atomically publish one profiled final fit."""
    from superglm.model import fit_ops

    publish_mode = _resolve_theta_request(
        model, fit_mode=fit_mode, theta_bounds=theta_bounds, xatol=xatol, ci_alpha=ci_alpha
    )
    report = progress_callback or _ignore_progress
    references = {"X_ref": X, "y_ref": y, "sample_weight_ref": sample_weight, "offset_ref": offset}
    # The family holds a numeric theta, but this call re-estimates it and refits.
    validated = fit_ops._validate_entrypoint_input(
        model, X, y, sample_weight, offset, theta_role=THETA_ESTIMATED
    )
    result = _search_theta_privately(
        model, validated, theta_bounds=theta_bounds, xatol=xatol, report=report
    )
    estimate = {"profile_estimate": _theta_estimate_payload(result)}
    report("best_found", estimate)
    report("final_refit", estimate)
    return _publish_profiled_family(
        model,
        validated,
        references,
        fit_mode=publish_mode,
        family=NegativeBinomial(theta=result.theta_hat),
        parameter="theta",
        value=result.theta_hat,
        synchronize=partial(_install_nb_profile, y=validated[1], result=result, ci_alpha=ci_alpha),
    )


def _resolve_theta_request(model, *, fit_mode, theta_bounds, xatol, ci_alpha):
    """Validate an estimate_theta call before any data is read; returns the publication mode."""
    family = configured_family(model)
    if not isinstance(family, NegativeBinomial):
        raise ValueError(
            f"estimate_theta requires a NegativeBinomial family, got {family!r}. "
            "Use families.nb2(theta=...) to create one."
        )
    lower, upper = theta_bounds
    if not 0.0 < lower < upper < np.inf:
        raise ValueError(f"theta_bounds must satisfy 0 < lower < upper < inf, got {theta_bounds!r}")
    _require_positive_tolerance(xatol)
    if ci_alpha is not None and not 0.0 < ci_alpha < 1.0:
        raise ValueError(f"ci_alpha must be strictly between 0 and 1, got {ci_alpha!r}")
    publish_mode = _resolve_profile_fit_mode(model, fit_mode)
    _validate_profile_selection_mode(model, publish_mode)
    return publish_mode


def _search_theta_privately(model, validated, *, theta_bounds, xatol, report):
    """The theta alternation on an attempt-local copy of the model.

    Returning drops the copy's design before the publication refit builds its own.
    """
    from superglm.model.fit_workspace import FitWorkspace
    from superglm.profiling.nb import estimate_nb_theta

    profile_workspace = FitWorkspace.start(
        model, mode="estimate_theta_profile", validated_inputs=validated
    )
    return estimate_nb_theta(
        profile_workspace.model,
        *validated,
        theta_bounds=theta_bounds,
        xatol=xatol,
        on_evaluation=lambda row: report("profiling", {"profile_trace": [row]}),
    )


def _install_nb_profile(final_model, *, y, result, ci_alpha):
    """Restate the estimate at the published mean and attach it; the caller gets its own copy.

    The NLL and interval are properties of the mean they are measured at, and
    the alternation's last mean is not the published one.
    """
    published = result._at_mean(y, final_model._fit_mu, final_model._fit_weights)
    if ci_alpha is not None:
        published.interval(ci_alpha)
    final_model._nb_profile_result = _detached_copy(published)
    return published


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
    """Fail doomed profile requests before allocating a profile workspace.

    Each regime preflights its own final-fit restrictions. The ordinary-fit
    side matters most for reverse coupling (search REML, publish ML): the
    features the publication fit rejects (RandomEffect, FactorSmooth,
    lambda_policy) would otherwise be discovered only after the entire
    multi-candidate search has run to a guaranteed failure.
    """
    if resolved_mode != "fit_reml":
        from superglm.model import fit_ops

        fit_ops._reject_random_effect_selection_fit(model, "fit")
        fit_ops._reject_lambda_policy_fit(model, "fit")
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
        "value": result.p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_status": ci_status,
        "objective": result.nll,
        "objective_label": "loss",
        "lower_is_better": True,
    }


def _theta_estimate_payload(result):
    # Reported before publication: the interval is measured at the published
    # mean, so none exists yet.
    return {
        "parameter": "theta",
        "label": "theta_hat",
        "value": result.theta_hat,
        "ci_low": None,
        "ci_high": None,
        "objective": result.nll,
        "objective_label": "loss",
        "lower_is_better": True,
    }
