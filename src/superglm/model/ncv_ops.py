"""``SuperGLM.fit_ncv``: smoothing chosen by held-out prediction, published as ``fit`` publishes.

A fit runs on a workspace copy of the model, so a refusal or a failure leaves an earlier fit
as it was.  Its steps, in order:

1. **Refusals, before any solve.**  The controls, the model's settings and the hold-out
   arguments are checked first (:func:`fit_ncv`, :func:`_refuse_unsupported_model`,
   ``neighbourhoods.build_neighbourhoods``); the terms once they are known, and the
   neighbourhoods against the factor levels and a forced solver route against the terms
   once the design is built (:func:`_refuse_unsupported_terms`,
   :func:`_refuse_constrained_terms`, :func:`_refuse_ineligible_route`,
   :func:`_refuse_removed_levels`).
2. **The search** over the estimated smoothing parameters, by the criterion of
   :mod:`superglm.ncv.engine` (``search.search_smoothing``).  Rows that stop identifying the
   model once a neighbourhood is removed are refused there, after its first two coefficient
   fits and before any step.
3. **Publication** (:func:`_publish`).  The design is rebuilt in the basis the chosen
   smoothing gives and fitted once more there, on the solver route the model is configured
   with; that fit is published by the steps ``fit`` publishes with (fit statistics, both
   scales, the public coefficients).  What is read of the model afterwards is what ``fit``
   gives at that smoothing.
4. **The held-out results** (:func:`_evaluate_published`): one evaluation of the criterion on
   the published fit's own geometry, with no further solve, gives the criterion and one
   entry per scored (neighbourhood, row) pair (:class:`HeldOut`).

Convergence is disclosed and never refused.  A search that did not meet its test, a final
coefficient fit that did not, a search that found no smoothing at which the criterion could
be computed (published as a coefficient fit at the smoothing it started from) and a smoothing
parameter on a bound of the search each return a fitted model and one warning
(:func:`warn_published_ncv`).

The solver's weight unit (``fit``'s exact rescaling of prior weights below ``2**-512``) is
not applied: every fit here asks the solver for its geometry, which it returns in the
weights' own units, as ``fit_reml`` does.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from superglm._lambda_box import LambdaBox
from superglm._warn import warn_caller
from superglm.diagnostics.convergence import (
    initial_precision_advisory,
    warn_ncv_bounds,
    warn_ncv_nonconvergence,
)
from superglm.diagnostics.separation import refuses_on_publication
from superglm.distributions import resolve_distribution
from superglm.links import resolve_link
from superglm.model import fit_method, runtime_canonicalize
from superglm.model.fit_ops import (
    _auto_detect_specs_if_needed,
    _clear_fit_inference_caches,
    _clear_profile_results,
    _clear_reml_state,
    _maybe_release_fit_state,
    _publish_fit_statistics,
    _store_fit_arrays,
    _validate_entrypoint_input,
    _validate_pirls_iteration_limit,
    _validate_rule_fit_conflicts,
    _warn_unresolved_rows,
)
from superglm.model.fit_state import (
    _install_fit_state,
    capture_fit_state,
    configured_family,
    configured_lambda2,
    configured_link,
    configured_penalty,
)
from superglm.model.fit_workspace import FitWorkspace
from superglm.model.reml_setup import collect_reml_groups, initialize_component_lambdas
from superglm.model.reml_state import update_reml_r_inv
from superglm.ncv.engine import (
    DEFAULT_NCV_TOL,
    LeaveOutRankError,
    ModeFit,
    build_problem,
    evaluate_fit,
    ncv_mode_bar,
)
from superglm.ncv.neighbourhoods import Neighbourhoods, build_neighbourhoods, unidentified_levels
from superglm.ncv.result import NCVResult
from superglm.ncv.rows import row_loss
from superglm.ncv.search import DEFAULT_MAX_NCV_ITER, SearchOutcome, search_smoothing
from superglm.reml.penalty_algebra import build_penalty_context
from superglm.reml.result import _map_beta_between_bases, _map_centred_state_between_bases
from superglm.solvers._structured.selection import resolve_structured_backend
from superglm.solvers.dispersion import dispersion_likelihood_size, model_weight_semantics
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_warm_start

CONVERGED_STOPS = ("gradient", "no_smoothing_parameters")
"""The stops of a search that met its test (``result.STOP_REASONS``)."""


@dataclass(frozen=True)
class HeldOut:
    """The held-out results of a published ``fit_ncv`` fit.

    The arrays hold one entry per scored (neighbourhood, row) pair, in neighbourhood order; a
    row scored by several neighbourhoods has an entry for each.  ``heldout`` is the mean
    predicted for the row by the fit without its neighbourhood, ``fitted`` the published
    fit's, ``deviance`` the weighted unit deviance at ``heldout`` (zero on a row of zero
    weight).  ``labels`` names each neighbourhood of a ``groups`` fit by its group.
    ``by_neighbourhood`` is the per-neighbourhood table of a ``groups`` or ``cv`` fit.

    A model fitted with ``retain_fit_state=False`` keeps ``by_neighbourhood`` only: the
    arrays, which are as long as the scored rows, are ``None``.
    """

    form: str
    n_rows: int
    row: NDArray[np.intp] | None
    neighbourhood: NDArray[np.intp] | None
    heldout: NDArray[np.float64] | None
    fitted: NDArray[np.float64] | None
    deviance: NDArray[np.float64] | None
    weight: NDArray[np.float64] | None
    actual: NDArray[np.float64] | None
    labels: NDArray | None
    by_neighbourhood: pd.DataFrame | None


# ── the entry point ───────────────────────────────────────────────────────────


def fit_ncv(
    model,
    X,
    y,
    sample_weight=None,
    offset=None,
    *,
    groups=None,
    cv=None,
    neighbours=None,
    max_ncv_iter=None,
    ncv_tol=None,
    pirls_tol=None,
    max_pirls_iter=None,
    lambda2_init=None,
):
    """Fit by neighbourhood cross-validation in a private workspace and publish on success.

    A control left as None takes its default: the search's own for ``ncv_tol`` and
    ``max_ncv_iter``, the model's constructor ``tol`` and ``max_iter`` for the coefficient fits.
    """
    tol = DEFAULT_NCV_TOL if ncv_tol is None else ncv_tol
    max_iter = DEFAULT_MAX_NCV_ITER if max_ncv_iter is None else max_ncv_iter
    pirls_tol = model._tol if pirls_tol is None else pirls_tol
    max_pirls_iter = model._max_iter if max_pirls_iter is None else max_pirls_iter
    _validate_controls(tol, max_iter, max_pirls_iter, lambda2_init)
    _refuse_unsupported_model(model)
    X_ref, y_ref, sample_weight_ref, offset_ref = X, y, sample_weight, offset
    X, y, sample_weight, offset = _validate_entrypoint_input(model, X, y, sample_weight, offset)
    nb = build_neighbourhoods(
        len(y), sample_weight > 0.0, frame=X_ref, y=y, groups=groups, cv=cv, neighbours=neighbours
    )
    workspace = FitWorkspace.start(
        model, mode="fit_ncv", validated_inputs=(X, y, sample_weight, offset)
    )
    published = not getattr(model, "_discarded_fit", False)
    with initial_precision_advisory(workspace.model, enabled=published) as precision:
        _fit_ncv_in_workspace(
            workspace.model,
            X,
            y,
            sample_weight,
            offset,
            nb=nb,
            X_ref=X_ref,
            y_ref=y_ref,
            sample_weight_ref=sample_weight_ref,
            offset_ref=offset_ref,
            tol=float(tol),
            max_iter=int(max_iter),
            pirls_tol=float(pirls_tol),
            max_pirls_iter=int(max_pirls_iter),
            lambda2_init=lambda2_init,
            published=published,
        )
    if precision:
        fit_method.annotate(workspace.model, initial_precision=precision)
    candidate = capture_fit_state(workspace, model, revision=model._fit_revision + 1)
    _install_fit_state(model, candidate)
    warn_published_ncv(model)
    return model


def _validate_controls(tol, max_iter, max_pirls_iter, lambda2_init) -> None:
    """Refuse a control the search cannot run with, naming the argument."""
    if not (isinstance(tol, Real) and math.isfinite(tol) and tol > 0.0):
        raise ValueError(
            f"ncv_tol must be a positive number, got {tol!r}; the default is {DEFAULT_NCV_TOL:g}."
        )
    if not (isinstance(max_iter, Integral) and max_iter >= 1):
        raise ValueError(
            f"max_ncv_iter must be a whole number of at least 1, got {max_iter!r}; "
            f"the default is {DEFAULT_MAX_NCV_ITER}."
        )
    _validate_pirls_iteration_limit(max_pirls_iter)
    if lambda2_init is None:
        return
    if not isinstance(lambda2_init, Mapping):
        raise TypeError(
            "lambda2_init must map smoothing components or terms to the smoothing parameter "
            f"each starts from, got {type(lambda2_init).__name__}; pass a mapping such as an "
            "earlier fit's ncv_diagnostics()['lambdas'], or leave it out."
        )
    for name, value in lambda2_init.items():
        if not (isinstance(value, Real) and math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"lambda2_init[{name!r}] must be a positive number, got {value!r}; give the "
                "smoothing parameter that term starts from, or leave it out."
            )


# ── refusals ──────────────────────────────────────────────────────────────────


def _refuse_unsupported_model(model) -> None:
    """Refuse the model settings ``fit_ncv`` does not support; reads no data."""
    from superglm.model.base import normalize_selection_penalty

    if model._discrete:
        raise ValueError(
            "fit_ncv does not support discrete=True, which fits on binned covariates; "
            "construct the model with discrete=False, or choose the smoothing of the binned "
            "fit with fit_reml."
        )
    distribution = resolve_distribution(configured_family(model))
    link = resolve_link(configured_link(model), distribution)
    try:
        row_loss(distribution, link)
    except ValueError:
        raise ValueError(
            "fit_ncv supports a Gaussian family with an identity link and a Poisson family "
            f"with a log link; this model has {type(distribution).__name__} with "
            f"{type(link).__name__}. Choose its smoothing with fit_reml."
        ) from None
    intent = normalize_selection_penalty(configured_penalty(model).lambda1)
    if intent == "auto" or (intent is not None and intent > 0.0):
        raise ValueError(
            f"fit_ncv does not support a selection penalty (selection_penalty={intent!r}); "
            "set selection_penalty to None or 0, or use fit or fit_path for sparse "
            "selection. To let the search shrink a spline's unpenalised part, declare the "
            "spline with select=True."
        )


def _refuse_unsupported_terms(model) -> None:
    """Refuse the term types whose penalties the criterion does not cover."""
    from superglm.features.factor_smooth import FactorSmooth
    from superglm.features.random_effect import RandomEffect

    for kind, specs in ((RandomEffect, model._specs), (FactorSmooth, model._interaction_specs)):
        names = [repr(name) for name, spec in specs.items() if isinstance(spec, kind)]
        if names:
            raise ValueError(
                f"fit_ncv does not support {kind.__name__} terms ({', '.join(names)}); "
                "choose this model's smoothing with fit_reml."
            )


def _refuse_constrained_terms(model) -> None:
    """Refuse a design whose terms carry fit-time constraints (monotone, convex, capped)."""
    names = list(
        dict.fromkeys(
            repr(group.feature_name)
            for group in model._groups
            if group.monotone_engine is not None or group.constraints is not None
        )
    )
    if names:
        raise ValueError(
            "fit_ncv does not support monotone or otherwise constrained terms "
            f"({', '.join(names)}); choose this model's smoothing with fit_reml, or drop the "
            "fit-time constraint from the term."
        )


def _refuse_ineligible_route(model) -> None:
    """Refuse a ``direct_solve`` route that ``fit`` refuses for this design, with its reason.

    ``fit`` decides a forced structured route from the terms alone, so the decision is made
    here on the built design, before any solve.  A route ``fit`` accepts is the published
    fit's route.
    """
    if model._direct_solve != "structured":
        return
    try:
        resolve_structured_backend(
            list(model._dm.group_matrices),
            model._groups,
            direct_solve=model._direct_solve,
            coefficient_width=model._dm.p,
        )
    except ValueError as error:
        raise ValueError(
            f"{str(error).rstrip('.')}. Construct the model with direct_solve='auto', 'gram' "
            "or 'qr'."
        ) from None


def _refuse_removed_levels(model, nb: Neighbourhoods, active, penalties) -> None:
    """Refuse neighbourhoods that remove every weighted row of an unpenalised factor level."""
    penalised = {component.group_index for component in penalties}
    conflicts = unidentified_levels(nb, model._dm, model._groups, active, penalised)
    if not conflicts:
        return
    first = conflicts[0]
    term = model._groups[first.group_index].feature_name
    spec = model._specs.get(term, model._interaction_specs.get(term))
    message = first.message(nb, _level_label(spec, first.level, first.reference))
    if len(conflicts) > 1:
        message += (
            f" {len(conflicts) - 1} more levels of this model are removed whole in the same way."
        )
    raise ValueError(message)


def _level_label(spec, code: int, reference: bool) -> object:
    """The label a user gave the level with design code ``code``; the code where none is kept."""
    if reference:
        return getattr(spec, "_base_level", code)
    labels = getattr(spec, "_non_base", None)
    return labels[code] if labels is not None and code < len(labels) else code


# ── the fit ───────────────────────────────────────────────────────────────────


def _fit_ncv_in_workspace(
    model,
    X,
    y,
    sample_weight,
    offset,
    *,
    nb: Neighbourhoods,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    tol: float,
    max_iter: int,
    pirls_tol: float,
    max_pirls_iter: int,
    lambda2_init,
    published: bool,
) -> None:
    """Build the design, search the smoothing and publish the fit on private model state."""
    from superglm.model.base import model_build_design_matrix, resolve_selection_penalty_for_reml

    resolve_selection_penalty_for_reml(configured_penalty(model))
    _clear_profile_results(model)
    _clear_reml_state(model)
    _auto_detect_specs_if_needed(model, X, sample_weight_ref)
    _refuse_unsupported_terms(model)
    _validate_rule_fit_conflicts(model)
    declared = configured_lambda2(model)
    y, sample_weight, offset = model_build_design_matrix(model, X, y, sample_weight, offset)
    sample_weight, offset = _store_fit_arrays(model, sample_weight, offset, y)
    _clear_fit_inference_caches(model)
    _refuse_constrained_terms(model)
    _refuse_ineligible_route(model)

    reml_groups: list[tuple[int, Any]] = collect_reml_groups(
        model._groups, model._dm.group_matrices
    )
    penalties, _, _ = build_penalty_context(model._dm.group_matrices, reml_groups)
    _refuse_removed_levels(model, nb, sample_weight > 0.0, penalties)
    lambdas, estimated = initialize_component_lambdas(penalties, declared)
    fixed = {name: value for name, value in lambdas.items() if name not in estimated}
    bar = ncv_mode_bar(tol)

    def problem_of(components):
        """The engine's problem on the model's current design: built anew, it holds no cache."""
        return build_problem(
            dm=model._dm,
            groups=model._groups,
            penalties=components,
            distribution=model._distribution,
            link=model._link,
            y=y,
            weights=sample_weight,
            offset=offset,
            weight_semantics=model_weight_semantics(model),
            fixed_lambdas=fixed,
            mode_bar=bar,
            pirls_tol=pirls_tol,
            max_pirls_iter=max_pirls_iter,
        )

    # The search's problem lives for this call only, and its cross-product cache with it.
    outcome = search_smoothing(
        problem_of(penalties),
        nb,
        configured={name: value for name, value in lambdas.items() if name in estimated},
        start=_start_map(lambda2_init, penalties, model._groups),
        tol=tol,
        max_iter=max_iter,
    )
    terminal, penalties = _publish(
        model,
        outcome,
        reml_groups=reml_groups,
        penalties=penalties,
        y=y,
        sample_weight=sample_weight,
        offset=offset,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
        bar=bar,
        pirls_tol=pirls_tol,
        max_pirls_iter=max_pirls_iter,
        published=published,
    )
    result = _evaluate_published(model, outcome.result, problem_of(penalties), nb, terminal)
    result.pirls_result = model._solver_result
    result.reml_penalties = penalties
    model._reml_result = result
    model._reml_profile = None
    fit_method.record(model, "fit_ncv")
    _maybe_release_fit_state(model)


def _start_map(lambda2_init, penalties, groups) -> dict[str, float]:
    """``lambda2_init`` by penalty component: its own entry, else its term's.

    A name that is no component or term of the model is ignored with a warning.  The search
    reads the estimated components only, so a fixed component keeps its value.
    """
    if lambda2_init is None:
        return {}
    known = {name for component in penalties for name in (component.name, component.group_name)}
    known.update(group.name for group in groups)
    unknown = sorted(str(name) for name in lambda2_init if name not in known)
    if unknown:
        warn_caller(
            f"lambda2_init names no smoothing parameter of this model: {', '.join(unknown)}. "
            "Those entries are ignored. Use the names in model.ncv_diagnostics()['lambdas'], "
            "or a term's name."
        )
    start = {}
    for component in penalties:
        name = component.name if component.name in lambda2_init else component.group_name
        if name in lambda2_init:
            start[component.name] = lambda2_init[name]
    return start


def _publish(
    model,
    outcome: SearchOutcome,
    *,
    reml_groups,
    penalties,
    y,
    sample_weight,
    offset,
    X_ref,
    y_ref,
    sample_weight_ref,
    offset_ref,
    bar: float,
    pirls_tol: float,
    max_pirls_iter: int,
    published: bool,
) -> tuple[ModeFit, list]:
    """Fit the model at the search's smoothing and publish that fit as ``fit`` publishes one.

    The design is rebuilt in the basis the smoothing gives, as a ``fit`` declared with that
    smoothing builds it, and solved on the model's configured route from the search's last
    certified coefficients.  Returns the solve as the engine reads one, with the geometry it
    was made in, and the penalty components in the rebuilt basis.
    """
    from superglm.model.base import rebuild_dm_with_lambdas

    lambdas = outcome.result.lambdas
    searched = model._dm.group_matrices
    design = rebuild_dm_with_lambdas(model, lambdas, sample_weight)
    model._dm = design
    penalties, _, _ = build_penalty_context(
        list(design.group_matrices), reml_groups, _reuse_raw_from=penalties
    )
    cache: dict[str, Any] = {}
    solved = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=sample_weight,
        family=model._distribution,
        link=model._link,
        groups=model._groups,
        lambda2=lambdas,
        offset=offset,
        max_iter=max_pirls_iter,
        tol=pirls_tol,
        convergence="mode_score",
        _mode_bar=bar,
        cache_out=cache,
        direct_solve=model._direct_solve,
        reml_penalties=penalties,
        _retain_reml_decomposition=True,
        _published=True,
        separation="error" if published and refuses_on_publication(model) else "record",
        weight_semantics=model_weight_semantics(model),
        **_terminal_start(outcome.fit, searched, model),
    )
    terminal = ModeFit.from_solve(lambdas, solved[0], solved[1], cache)

    # The decomposition serves the held-out evaluation; a published result carries none.
    result = replace(solved[0], reml_slope_decomposition=None)
    if getattr(model._distribution, "scale_known", True) and result.phi != 1.0:
        result = replace(result, phi=1.0)
    model._result = result
    model._reml_lambdas = lambdas
    model._reml_penalties = penalties
    _publish_fit_statistics(
        model,
        y,
        sample_weight,
        offset,
        X_ref=X_ref,
        y_ref=y_ref,
        sample_weight_ref=sample_weight_ref,
        offset_ref=offset_ref,
    )
    update_reml_r_inv(model, reml_groups, lambdas)
    runtime_canonicalize.canonicalize_fitted_model(model)
    _warn_unresolved_rows(model)
    return terminal, penalties


def _terminal_start(fit: ModeFit | None, searched, model) -> dict[str, Any]:
    """The search's last certified coefficients, read in the rebuilt basis; nothing without one."""
    if fit is None:
        return {}
    rebuilt = model._dm.group_matrices
    beta = fit.result.beta
    return {
        "beta_init": _map_beta_between_bases(beta, searched, rebuilt, model._groups),
        "intercept_init": float(fit.result.intercept),
        "_centred_init": _map_centred_state_between_bases(
            centred_warm_start(fit.result), beta, searched, rebuilt, model._groups
        ),
    }


def _evaluate_published(
    model, searched: NCVResult, problem, nb: Neighbourhoods, terminal: ModeFit
) -> NCVResult:
    """The search's result completed at the published fit: the criterion and the held-out pairs.

    A search that found a criterion is evaluated once more, on the published fit's own
    geometry and with no solve.  Where that gives none (a final fit short of its
    certificate, as a rule), the search's own criterion stays and the result holds no
    held-out table, with the reason in ``heldout_failure``.  The criterion's mean is over
    the likelihood size of the scored pairs, a row scored twice counted twice.
    """
    certified = bool(terminal.result.converged)
    result = replace(
        searched,
        converged=bool(searched.converged) and certified,
        terminal_refit_termination=None if certified else str(terminal.result.termination_reason),
        neighbourhoods=_neighbourhood_summary(nb),
    )
    if searched.objective is None:
        return result
    size = dispersion_likelihood_size(
        problem.weights[nb.scored_rows()], weight_semantics=problem.weight_semantics
    )
    result.criterion_mean = float(searched.objective) / size
    try:
        evaluation = evaluate_fit(problem, nb, terminal, gradient=False, heldout=True)
    except LeaveOutRankError:
        result.heldout_failure = "leave_out_refused"
        return result
    pairs = evaluation.heldout
    if pairs is None:
        result.heldout_failure = evaluation.infeasible_reason or "non_finite"
        return result
    result.objective = float(evaluation.value)
    result.criterion_mean = float(evaluation.value) / size
    result.heldout = _heldout(model, nb, pairs, problem.weights[pairs.row], problem.y[pairs.row])
    return result


def _neighbourhood_summary(nb: Neighbourhoods) -> dict[str, Any]:
    """The neighbourhoods' form, count and sizes, and how many rows they score."""
    sizes = nb.sizes()
    return {
        "form": nb.form,
        "splitter": nb.source,
        "count": int(nb.m),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
        "pairs_scored": int(len(nb.scored_rows())),
        "rows_scored": nb.rows_scored,
        "rows_scored_more_than_once": nb.rows_scored_more_than_once,
    }


# ── held-out results ──────────────────────────────────────────────────────────


def _heldout(model, nb: Neighbourhoods, pairs, weights, actual) -> HeldOut:
    """The record of ``pairs``; a model that releases its fit state keeps the table only."""
    fitted = np.asarray(model._fit_mu, dtype=np.float64)[pairs.row]
    table = None
    if nb.form in ("groups", "cv"):
        table = _group_table(
            pairs.neighbourhood,
            nb.m,
            pd.Index(nb.labels, name="group" if nb.form == "groups" else "split"),
            weights,
            actual,
            pairs.mean,
            pairs.deviance,
        )
    kept: dict[str, NDArray | None] = dict.fromkeys(
        ("row", "neighbourhood", "heldout", "fitted", "deviance", "weight", "actual", "labels")
    )
    if getattr(model, "_retain_fit_state", True):
        kept.update(
            row=pairs.row,
            neighbourhood=pairs.neighbourhood,
            heldout=pairs.mean,
            fitted=fitted,
            deviance=pairs.deviance,
            weight=weights,
            actual=actual,
            labels=nb.labels if nb.form == "groups" else None,
        )
    frozen = {name: None if array is None else _frozen(array) for name, array in kept.items()}
    return HeldOut(form=nb.form, n_rows=nb.n_rows, by_neighbourhood=table, **frozen)


def _frozen(array: NDArray) -> NDArray:
    """A read-only copy that owns its data."""
    copy = np.array(array)
    copy.setflags(write=False)
    return copy


def _group_table(codes, count, index, weights, actual, heldout, deviance) -> pd.DataFrame:
    """Per group code: scored pairs, weight, actual and held-out expected totals, their ratio, deviance.

    A row of zero weight counts as a pair and adds to no total.  The ratio is missing where
    the expected total is zero or negative.
    """
    on = weights > 0.0
    observed = np.bincount(codes[on], weights=weights[on] * actual[on], minlength=count)
    expected = np.bincount(codes[on], weights=weights[on] * heldout[on], minlength=count)
    ratio = np.full(count, np.nan)
    positive = expected > 0.0
    ratio[positive] = observed[positive] / expected[positive]
    return pd.DataFrame(
        {
            "rows": np.bincount(codes, minlength=count),
            "weight": np.bincount(codes[on], weights=weights[on], minlength=count),
            "actual": observed,
            "heldout_expected": expected,
            "actual_over_expected": ratio,
            "heldout_deviance": np.bincount(codes[on], weights=deviance[on], minlength=count),
        },
        index=index,
    )


def _ncv_result(model, reader: str) -> NCVResult:
    result = getattr(model, "_reml_result", None)
    if not isinstance(result, NCVResult):
        raise RuntimeError(
            f"{reader} reads a fit made by fit_ncv, and this model holds none; "
            "fit it with fit_ncv first."
        )
    return result


def _held(model, reader: str) -> HeldOut:
    result = _ncv_result(model, reader)
    if result.termination_reason == "coefficients_revised":
        raise RuntimeError(
            f"{reader} has no held-out results to read: the coefficients were revised after "
            "fitting, and the held-out results described the fitted ones; refit with fit_ncv "
            "to compute them for this model."
        )
    if result.heldout is None:
        raise RuntimeError(
            f"{reader} has no held-out results to read: the criterion could not be computed "
            "at this fit's published coefficients; ncv_diagnostics() gives the stop of the "
            "search, of the final coefficient fit, and 'heldout_failure'."
        )
    return result.heldout


def ncv_heldout(model) -> pd.DataFrame:
    """One row per scored (neighbourhood, row) pair of the model's ``fit_ncv`` fit."""
    held = _held(model, "ncv_heldout()")
    if held.row is None:
        raise RuntimeError(
            "ncv_heldout() needs the fit's rows, which retain_fit_state=False released; "
            "refit with retain_fit_state=True, or read the per-group table with "
            "ncv_heldout_groups()."
        )
    labels = held.neighbourhood if held.labels is None else held.labels[held.neighbourhood]
    # A frame made from a dict of arrays copies them, so the table shares nothing with the fit.
    return pd.DataFrame(
        {
            "neighbourhood": held.neighbourhood,
            "label": labels,
            "row": held.row,
            "weight": held.weight,
            "actual": held.actual,
            "heldout": held.heldout,
            "fitted": held.fitted,
            "heldout_deviance": held.deviance,
        }
    )


def ncv_heldout_groups(model, by=None) -> pd.DataFrame:
    """The held-out results of the model's ``fit_ncv`` fit summed per group."""
    held = _held(model, "ncv_heldout_groups()")
    if by is None:
        if held.by_neighbourhood is None:
            raise ValueError(
                "ncv_heldout_groups() has no groups to sum over: this fit held out single "
                "rows or rows with their neighbours. Pass by=, one label per row of X."
            )
        return held.by_neighbourhood.copy()
    if held.row is None:
        raise RuntimeError(
            "ncv_heldout_groups(by=...) needs the fit's rows, which retain_fit_state=False "
            "released; refit with retain_fit_state=True, or leave by out to read the table "
            "of the fit's own groups."
        )
    values = np.asarray(by)
    if values.shape != (held.n_rows,):
        raise ValueError(
            f"by must hold one label per row of X: it has shape {values.shape} for "
            f"{held.n_rows} rows."
        )
    codes, labels = pd.factorize(values[held.row], sort=True)
    if np.any(codes < 0):
        raise ValueError("by has missing labels on scored rows; give every row a label.")
    return _group_table(
        codes,
        len(labels),
        pd.Index(labels, name=getattr(by, "name", None) or "group"),
        held.weight,
        held.actual,
        held.heldout,
        held.deviance,
    )


# ── diagnostics and disclosure ────────────────────────────────────────────────


def ncv_diagnostics(model) -> dict[str, Any]:
    """How the model's ``fit_ncv`` fit went, as plain values that share nothing with the model."""
    result = _ncv_result(model, "ncv_diagnostics()")
    box = result.lambda_box
    floor, ceiling = (box.floor, box.ceiling) if isinstance(box, LambdaBox) else (None, None)
    search_converged = result.termination_reason in CONVERGED_STOPS
    # Revised coefficients are not the final fit's, whatever that fit's stop was.
    coefficients_converged = (
        result.terminal_refit_termination is None
        and result.termination_reason != "coefficients_revised"
    )
    return {
        "method": "fit_ncv",
        "criterion": None if result.objective is None else float(result.objective),
        "criterion_mean": result.criterion_mean,
        "neighbourhoods": dict(result.neighbourhoods),
        "n_iter": int(result.n_reml_iter),
        "n_evaluations": int(result.n_evaluations),
        "infeasible_trials": int(result.infeasible_trials),
        "infeasible_reasons": dict(result.infeasible_reasons),
        "termination_reason": result.termination_reason,
        "converged": bool(result.converged),
        "search_converged": search_converged,
        "coefficients_converged": coefficients_converged,
        "coefficient_termination": result.terminal_refit_termination,
        "heldout_available": result.heldout is not None,
        "heldout_failure": result.heldout_failure,
        "lambdas": dict(result.lambdas),
        "start_lambdas": dict(result.start_lambdas),
        "gradient": None if result.gradient is None else np.asarray(result.gradient).tolist(),
        "at_bound": [tuple(hit) for hit in result.at_bound],
        "bound_pressure": [tuple(hit) for hit in result.bound_pressure],
        "lambda_floor": floor,
        "lambda_ceiling": ceiling,
        "lambda_history": [dict(step) for step in result.lambda_history],
        "criterion_history": list(result.objective_history or ()),
        "work": dict(result.work),
    }


def warn_published_ncv(model) -> None:
    """Disclose, once each, an unconverged ``fit_ncv`` fit and smoothing held on a bound."""
    if getattr(model, "_suppress_convergence_warning", False):
        return
    result = model._reml_result
    warn_ncv_nonconvergence(result)
    warn_ncv_bounds(result)
