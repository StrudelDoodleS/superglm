"""Saved models whose structured solver state was retired.

``SuperGLM.__setstate__`` drops a retained linear system that names a retired
class (``solvers._structured.retired``) and marks the model; predictions,
coefficients and relativities never read it.  The first inference call
rebuilds it through ``retained_linear_state`` with the current engine at the
saved coefficients and smoothing parameters (one pass, one factorization),
with a one-time notice.  A model that did not retain its fit state asks for a
refit.
"""

from __future__ import annotations

import dataclasses
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
    """``SuperGLM.__setstate__``'s part for the retired v0.35.0 Tweedie classes.

    A REML fit's saturated-density memo is read only by the fit's own terminal
    evaluations, so it is dropped. A saved ``estimate_p`` or
    ``estimate_theta`` result restates itself in its own ``__setstate__``.
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


def release_saved_mask_summaries(model) -> None:
    """``SuperGLM.__setstate__``'s part for a model saved before fits recorded which standard errors they show.

    Its record keeps the blanks it was saved with (``RankInfo.standard_error_mask``).  A summary
    cached beside it is dropped: the next ``summary()`` is formed from the same inference, and
    says a refit shows the blank ones (``state_ops.saved_standard_error_note``).
    """
    record = model.__dict__.get("_linear_system_state")
    if record is None:
        # the solver-space result first, as the readers take it (``state_ops._fit_record``)
        result = model.__dict__.get("_solver_result")
        if result is None:
            result = model.__dict__.get("_result")
        record = getattr(result, "rank_info", None)
    if getattr(record, "standard_error_mask", None) == "data":
        model.__dict__["_summary_cache"] = None


def release_capped_separation_records(model) -> None:
    """``SuperGLM.__setstate__``'s part for a model saved before a fit-time cap exempted its levels.

    A capped ``Categorical``'s relativities are bounded wherever the fit enforced the cap
    (every fit but ``fit_path``), so a level an older build recorded separated has a
    finite optimum there: its record is cleared, and with it the flags and notices it
    drove.
    """
    if (model.__dict__.get("_last_fit_meta") or {}).get("method") == "fit_path":
        return
    from superglm.model.separated_levels import record_separated_levels

    for spec in (model.__dict__.get("_specs") or {}).values():
        if getattr(spec, "_constraint_rules", None) and getattr(
            spec, "_separated_levels_at_fit", ()
        ):
            record_separated_levels(spec)


def release_separated_level_summaries(model) -> None:
    """``SuperGLM.__setstate__``'s part for a summary cached beside a separated level or cell.

    A summary cached before its readers flagged such a level holds the level's Wald
    statistics; it is dropped, and the next ``summary()`` is formed from the same fit.
    """
    specs = [
        *(model.__dict__.get("_specs") or {}).values(),
        *(model.__dict__.get("_interaction_specs") or {}).values(),
    ]
    if any(
        getattr(spec, "_separated_levels_at_fit", ())
        or getattr(spec, "_separated_cells_at_fit", ())
        for spec in specs
    ):
        if model.__dict__.get("_summary_cache"):
            model.__dict__["_summary_cache"] = None


def release_undetermined_row_summaries(model) -> None:
    """Clear a cached summary on unpickle when the saved estimability mask has blanked entries.

    A cached summary saved with undetermined rows (such as an ordered term's
    slope change, curvature, level, or base-level intercept, or a blanked
    coefficient entry) is dropped whenever ``coefficient_estimable`` has any
    ``False`` entry so the next ``summary()`` call rebuilds it from the fit.
    """
    if not model.__dict__.get("_summary_cache"):
        return
    mask = (model.__dict__.get("_fit_inference_info") or {}).get("coefficient_estimable")
    if mask is not None and not np.all(mask):
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
    from superglm.solvers._structured.assembly import (
        build_augmented_structured_factor,
        build_penalized_structured_operator,
    )
    from superglm.solvers._structured.balance_tree import (
        ProfiledSumToZeroTreeFactor,
        SumToZeroTreeFactor,
    )
    from superglm.solvers._structured.block_leaves import (
        FactorSmoothLeafFactor,
        ProfiledFactorSmoothLeafFactor,
    )
    from superglm.solvers._structured.geometry import (
        cancelled_column_row_norms,
    )
    from superglm.solvers._structured.layout import (
        get_structured_layout,
    )
    from superglm.solvers._structured.moments import (
        build_structured_system,
    )
    from superglm.solvers._structured.nested import (
        NestedSchurFactor,
        ProfiledNestedSchurFactor,
    )
    from superglm.solvers._structured.retired import REBUILT_NOTICE
    from superglm.solvers._structured.selection import (
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
    # a model saved before fits recorded which standard errors they show keeps the ones it had
    return None if state is None else dataclasses.replace(state, standard_error_mask="data")
