"""Saved models whose structured solver state was retired (one-engine design §3.12, decision 13).

``SuperGLM.__setstate__`` drops a retained linear system that names a retired
class (``solvers._structured.retired``) and marks the model.  Predictions,
coefficients and relativities never read that state.  The first inference call
reads it through ``retained_linear_state``, which rebuilds it with the current
engine at the saved coefficients and smoothing parameters -- one pass at the
saved working weights and one factorization (a random effect's nested chain, a
lone level a chain of one; the fs leaf factor; or the balance tree of an sz
term) -- with a one-time notice.  A model that did not retain its fit state
cannot be rebuilt, and the call says so and asks for a refit.
"""

from __future__ import annotations

import warnings

import numpy as np

_RETIRED_FLAG = "_retired_linear_state"


def mark_retired_linear_state(model) -> None:
    """``SuperGLM.__setstate__``'s part: drop a retired state and remember that it was there."""
    from superglm.solvers._structured.retired import holds_retired_state

    if holds_retired_state(model.__dict__.get("_linear_system_state")):
        model.__dict__["_linear_system_state"] = None
        model.__dict__[_RETIRED_FLAG] = True
        # the inference caches derived from it hold the retired factors too
        for name in ("_coef_covariance", "_fit_active_info", "_fit_inference_info", "_group_edf"):
            model.__dict__.pop(name, None)
        for name in ("_fit_metrics_cache", "_fit_metrics_cache_signature", "_summary_cache"):
            if name in model.__dict__:
                model.__dict__[name] = None


def release_retired_tweedie_state(model) -> None:
    """``SuperGLM.__setstate__``'s part for the v0.35.0 Tweedie classes #422 retired.

    A REML fit kept its saturated-density memo on the REML result; it is a
    cache that only the fit's own terminal evaluations read, so it is dropped.
    Predictions, coefficients and standard errors never read it.  (A saved
    ``estimate_p`` or ``estimate_theta`` result restates itself on load, in
    its own ``__setstate__``.)
    """
    from superglm.solvers._structured.retired import RetiredStructuredState

    reml = model.__dict__.get("_reml_result")
    if isinstance(getattr(reml, "tweedie_scale_data", None), RetiredStructuredState):
        reml.tweedie_scale_data = None


def release_v0_35_nb_summaries(model) -> None:
    """``SuperGLM.__setstate__``'s part for summaries v0.35.0 cached after ``estimate_theta``.

    v0.35.0's summary reported the theta interval as a bare ``(lower, upper)``;
    the current one also says whether it is censored (``nb_theta_ci_status``),
    which only the restated ``NBProfileResult`` knows.  Such a cached summary
    is dropped, and the next ``summary()`` computes it from the restated result.
    """
    cache = model.__dict__.get("_summary_cache")
    infos = [getattr(summary, "_info", {}) for summary in (cache or {}).values()]
    if any("nb_theta" in info and "nb_theta_ci_status" not in info for info in infos):
        model.__dict__["_summary_cache"] = None


def retained_linear_state(model):
    """The model's retained structured linear system, rebuilt once if it was retired."""
    state = getattr(model, "_linear_system_state", None)
    if state is None and model.__dict__.get(_RETIRED_FLAG, False):
        state = _rebuild(model)
        model.__dict__["_linear_system_state"] = state
        model.__dict__[_RETIRED_FLAG] = False
    return state


def _rebuild(model):
    from superglm.model.reml_finalize import _build_structured_linear_system_state
    from superglm.model.state_ops import _solver_space_working_weights
    from superglm.solvers._structured.retired import REBUILT_NOTICE
    from superglm.solvers.structured import (
        FactorSmoothLeafFactor,
        NestedSchurFactor,
        ProfiledFactorSmoothLeafFactor,
        ProfiledNestedSchurFactor,
        ProfiledSumToZeroTreeFactor,
        SumToZeroTreeFactor,
        build_augmented_structured_factor,
        build_penalized_structured_operator,
        build_structured_system,
        cancelled_column_row_norms,
        get_structured_layout,
        resolve_structured_backend,
    )

    dm = getattr(model, "_dm", None)
    if dm is None or getattr(model, "_fit_weights", None) is None:
        raise RuntimeError(
            "This model was saved by an earlier superglm build whose structured solver state "
            "is retired, and retain_fit_state=False discarded the fitted design it would be "
            "rebuilt from; refit with retain_fit_state=True to compute its standard errors, "
            "leverage and summaries."
        )
    groups = model._groups
    matrices = list(dm.group_matrices)
    lambdas = model._reml_lambdas
    # the one structural decision a fit makes (a saved structured state had one)
    decision = resolve_structured_backend(
        matrices,
        groups,
        direct_solve="structured",
        coefficient_width=dm.p,
        lambda2=lambdas,
        nesting_cache=getattr(dm, "_structured_layout_cache", None),
    )
    index = decision.group_index
    if index is None:  # pragma: no cover - a structured state had a structured term
        raise RuntimeError("A retired structured state needs a structured term.")
    layout = get_structured_layout(
        dm, groups, dominant_group_index=index, chain_group_indices=decision.chain_group_indices
    )
    weights = _solver_space_working_weights(model)
    system = build_structured_system(
        matrices,
        groups,
        weights,
        np.zeros_like(weights),
        dominant_group_index=index,
        layout=layout,
        prior_weights=model._fit_weights,
    )
    penalized = build_penalized_structured_operator(
        system, matrices, groups, lambdas, reml_penalties=model._reml_penalties
    )
    factor, _ = build_augmented_structured_factor(system, penalized)
    xtw = np.empty(system.operator.shape[0])
    xtw[system.operator.small_indices] = system.xtw_small
    xtw[system.operator.structured_indices] = system.xtw_structured
    cache = {
        "structured_system": system,
        "penalized_operator": penalized,
    }
    if isinstance(factor, NestedSchurFactor):
        profiled = ProfiledNestedSchurFactor(
            augmented_factor=factor, sum_w=system.sum_w, xtw=xtw, data_operator=system.operator
        )
    elif isinstance(factor, SumToZeroTreeFactor):
        profiled = ProfiledSumToZeroTreeFactor(augmented_factor=factor, sum_w=system.sum_w, xtw=xtw)
    elif isinstance(factor, FactorSmoothLeafFactor):
        profiled = ProfiledFactorSmoothLeafFactor(
            augmented_factor=factor, sum_w=system.sum_w, xtw=xtw
        )
        # the retained estimability rule reads them (as the fit that built the state)
        cache["structured_row_column_norm"] = cancelled_column_row_norms(
            system.centred_data_operator, dm, weights
        )
    else:  # pragma: no cover - structured dispatch invariant
        raise TypeError(f"Unsupported structured factor {type(factor).__name__}.")
    reporting = getattr(model, "_reporting_support_state", None)
    support = {} if reporting is None else dict(reporting.support_totals)
    state = _build_structured_linear_system_state(
        factor=profiled, data_operator=system.operator, cache=cache, support_totals=support
    )
    warnings.warn(REBUILT_NOTICE, UserWarning, stacklevel=4)
    return state
