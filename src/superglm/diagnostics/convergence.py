"""Disclosure for fits that stop before their convergence test passes.

A fit that cannot meet its convergence test is returned, not refused, and it
says so: a ``ConvergenceWarning`` when it finishes, and a statement in the
summary and diagnostics of the model it publishes.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from math import frexp, inf
from typing import Any

import numpy as np

from superglm._roundoff import TINY, UNIT_ROUNDOFF
from superglm._warn import warn_caller
from superglm.links import is_multiplicative


class FitPrecisionWarning(UserWarning):
    """Some rows of a fit's first working system lie beyond what binary64 carries.

    Issued once by a published ``fit``, ``fit_reml`` or profiled refit, or
    before the error of one that raises (a fit refused as separated says
    nothing), when a row of positive sample weight has a subnormal weight or
    initial working information, information that rounds to zero, a weight or
    information at least 2**49 (about 5.6e14) times another row's, or, under a
    log link, an initial mean that underflows or overflows before clipping.
    Rows of weight zero contribute nothing to the fit and are not read. A
    spread compares all rows, so the two it names may inform separate
    coefficients and lose nothing. The
    check only reports: the fit is the same with or without it. Check the
    sample_weight units and the offsets on the rows it names;
    ``model.diagnostics()['_model']['initial_precision']``
    holds the counts, example rows and values.
    """


_PRECISION_BLOCK_SIZE = 8192
# Binary64 round-to-nearest breakpoints in eta for exp(eta): half the least
# subnormal, the midpoint below the least normal, and the overflow midpoint.
_LOG_MEAN_ZERO = -745.1332191019412
_LOG_MEAN_NORMAL = -708.3964185322641
_LOG_MEAN_LIMIT = 709.782712893384
# Within a factor of sixteen of 1/u, small contributions can be comparable to
# rounding in a shared sum. A row spread alone does not determine rank.
_SPREAD_BITS = frexp(1.0 / (16.0 * UNIT_ROUNDOFF))[1] - 1

_PRECISION_MESSAGES = {
    "subnormal_sample_weight": "Subnormal sample weight",
    "sample_weight_spread": "Large sample-weight spread",
    "zero_information": "Zero initial {source} information on positive-weight rows",
    "subnormal_information": "Subnormal initial {source} information",
    "information_spread": "Large initial {source} information spread",
    "log_mean_zero": "Initial log-link mean rounds to zero before clipping",
    "log_mean_subnormal": "Subnormal initial log-link mean before clipping",
    "log_mean_overflow": "Initial log-link mean overflows before clipping",
}

_CHECK_WEIGHTS = "Check the sample_weight units."
_CHECK_WEIGHTS_AND_OFFSETS = "Check the sample_weight units and offsets on these rows."

# The fit attempt owns this scope. Match its group list by identity so that
# intercept-only solves inside setup cannot consume the advisory.
_INITIAL_PRECISION: ContextVar[tuple[Any, dict] | None] = ContextVar(
    "initial_fit_precision", default=None
)


@contextmanager
def initial_precision_advisory(model, *, enabled: bool):
    """Collect one fit's entry findings without retaining the working rows."""
    findings: dict = {}
    token = _INITIAL_PRECISION.set((model, findings) if enabled else None)
    try:
        yield findings
    except Exception as exc:
        from superglm.diagnostics.separation import SeparationError

        # A fit that raises is never published, so it states its findings
        # here; a fit refused as separated says nothing beyond its refusal.
        if enabled and findings and not isinstance(exc, SeparationError):
            warn_initial_precision(findings)
        raise
    finally:
        _INITIAL_PRECISION.reset(token)


def inspect_initial_precision(groups, rows, eta_unclipped, link) -> None:
    """Inspect the first working-row carrier of the active scalar fit."""
    capture = _INITIAL_PRECISION.get()
    if capture is None or groups is not capture[0]._groups:
        return
    model, findings = capture
    _INITIAL_PRECISION.set(None)
    findings.update(_scan_initial_precision(model._fit_weights, rows.weights, eta_unclipped, link))
    for kind, finding in findings.items():
        if kind.endswith("information") or kind == "information_spread":
            finding["curvature_source"] = rows.curvature_source


def warn_initial_precision(findings: dict) -> None:
    """Disclose one fit's findings: at publication, or as a fit that raises leaves."""
    parts = []
    checks = {}
    for kind, finding in findings.items():
        reason = _PRECISION_MESSAGES[kind]
        if "curvature_source" in finding:
            source = "Fisher" if finding["curvature_source"] == "fisher" else "observed"
            reason = reason.format(source=source)
            checks[_CHECK_WEIGHTS_AND_OFFSETS] = None
        elif kind.startswith("log_mean_"):
            checks["Check the offsets on these rows."] = None
        else:
            checks[_CHECK_WEIGHTS] = None
        positions = ", ".join(str(row) for row in finding["rows"])
        count = finding.get("count", len(finding["rows"]))
        examples = " e.g." if count > len(finding["rows"]) else ""
        parts.append(f"{reason} at row positions{examples} {positions}.")
    if _CHECK_WEIGHTS_AND_OFFSETS in checks:
        checks.pop(_CHECK_WEIGHTS, None)
    warn_caller(
        "SuperGLM binary64 precision advisory, with zero-based row positions. "
        + " ".join([*parts, *checks])
        + " See model.diagnostics()['_model']['initial_precision'] for details.",
        FitPrecisionWarning,
    )


def _scan_initial_precision(weights, information, eta, link) -> dict:
    """One sweep in fixed-size blocks. Keep counts and up to three row examples.

    A row of weight zero is exact in binary64 and contributes nothing, so it
    is not an event: padding a fit with such rows changes no finding. Nor is
    information that is zero analytically, where the inverse link's derivative
    has an isolated zero (a square-root link at eta = 0), not rounding.
    """
    findings: dict = {}
    extrema = [[(inf, -1), (0.0, -1)] for _ in range(2)]
    log_link = is_multiplicative(link)

    def record(kind, mask, start):
        count = int(np.count_nonzero(mask))
        if count:
            finding = findings.setdefault(kind, {"count": 0, "rows": ()})
            finding["count"] += count
            remaining = 3 - len(finding["rows"])
            if remaining:
                finding["rows"] += tuple(int(i) + start for i in np.flatnonzero(mask)[:remaining])

    for start in range(0, weights.size, _PRECISION_BLOCK_SIZE):
        block = slice(start, start + _PRECISION_BLOCK_SIZE)
        weight, info = weights[block], information[block]
        active = weight > 0.0
        record("subnormal_sample_weight", active & (weight < TINY), start)
        zero_information = active & (info == 0.0)
        if zero_information.any():
            zero_information[zero_information] = ~_isolated_zero_slope(
                link, eta[block][zero_information]
            )
        record("zero_information", zero_information, start)
        finite_positive_info = active & (info > 0.0) & np.isfinite(info)
        record("subnormal_information", finite_positive_info & (info < TINY), start)
        for values, positive, pair in zip((weight, info), (active, finite_positive_info), extrema):
            low = int(np.argmin(np.where(positive, values, inf)))
            high = int(np.argmax(np.where(positive, values, 0.0)))
            if positive[low] and values[low] < pair[0][0]:
                pair[0] = (float(values[low]), start + low)
            if positive[high] and values[high] > pair[1][0]:
                pair[1] = (float(values[high]), start + high)
        if log_link:
            raw_eta = eta[block]
            record("log_mean_zero", active & (raw_eta <= _LOG_MEAN_ZERO), start)
            record(
                "log_mean_subnormal",
                active & (raw_eta > _LOG_MEAN_ZERO) & (raw_eta < _LOG_MEAN_NORMAL),
                start,
            )
            record("log_mean_overflow", active & (raw_eta > _LOG_MEAN_LIMIT), start)

    for kind, ((low, low_row), (high, high_row)) in zip(
        ("sample_weight_spread", "information_spread"), extrema
    ):
        lo_mantissa, lo_exponent = frexp(low)
        hi_mantissa, hi_exponent = frexp(high)
        spread = hi_exponent - lo_exponent - (hi_mantissa < lo_mantissa)
        if low_row >= 0 and spread >= _SPREAD_BITS:
            findings[kind] = {"rows": (low_row, high_row), "values": (low, high)}
    return findings


def _isolated_zero_slope(link, eta):
    """Where d(mu)/d(eta) is zero at eta but not at either neighbouring float.

    Such a zero is the derivative's own, not an underflow: one that rounded
    to zero is still zero a float away.
    """
    with np.errstate(all="ignore"):
        slopes = [np.abs(link.deriv_inverse(np.nextafter(eta, toward))) for toward in (-inf, inf)]
        return (np.abs(link.deriv_inverse(eta)) == 0.0) & (slopes[0] > 0.0) & (slopes[1] > 0.0)


class ConvergenceWarning(UserWarning):
    """A fit stopped before its convergence test passed, or fell back on curvature.

    The fitted model is still returned. ``SuperGLM.fit`` warns when its
    coefficient fit stops unconverged. ``SuperGLM.fit_reml`` warns when
    smoothing-parameter selection or its final coefficient fit stops
    unconverged, as do ``estimate_p`` and ``estimate_theta`` with
    ``fit_mode="reml"`` for the fit they publish; the summary's ``Converged``
    row and ``reml_diagnostics()["converged"]`` then read ``False``.

    ``SuperGLM.fit_ncv`` warns when its smoothing search or its final
    coefficient fit stops unconverged, when the search finds no smoothing at
    which its criterion can be computed, and when it ends with a smoothing
    parameter on a limit of its range; ``ncv_diagnostics()`` holds each verdict.

    ``SuperLSS.fit_reml`` warns when its coefficient or smoothing loop stops
    unconverged; ``result_.converged`` then reads ``False`` and the summary's
    ``note`` column says so (``SuperLSS.fit``, with fixed smoothing, reports
    it there without a warning). ``SuperLSS`` also warns, and records the
    message in ``result_.warnings``, on a fit that may itself have converged
    when:

    - the final curvature is projected to the nearest positive semidefinite
      matrix, or falls back to expected Fisher information for
      ``NegativeBinomialLS`` (other families' Fisher curvature fallbacks do
      not warn);
    - the null comparison model did not converge;
    - an inner fit stops with ``no_ascent_direction``; or
    - a ``NegativeBinomialLS`` fit has rows at the Poisson size limit and its
      final curvature is singular in a size direction (the message gives the
      row count).

    The warning message names the reason and what to change. Silence it with
    ``warnings.filterwarnings`` once the reason is understood.
    """


# Why a smoothing search stopped short of its convergence test, for each
# termination reason that means it did.
_REML_REASON_TEXT = {
    "max_reml_iter": "it reached the max_reml_iter limit",
    "line_search_failed": "no smoothing step improved the REML score",
    "line_search_stalled": "no smoothing step improved the REML score",
}

# Termination reasons of a smoothing search that met its convergence test. A
# fit stopped on one of these is unconverged only through a later stage (the
# final coefficient refit's certificate), and the statement names that stage,
# never the search.
_REML_CONVERGED_REASONS = frozenset(
    {
        "lambda_tolerance",
        "objective_plateau",
        "score_objective_tolerance",
        "newton_step_tolerance",
        "active_set_stationary",
        "converged_at_precision",
        "fixed_lambdas",
    }
)


def _iterations(count: int) -> str:
    return f"{count} iteration{'' if count == 1 else 's'}"


def reml_nonconvergence_message(reml_result: Any) -> str | None:
    """The statement a SuperGLM REML fit that did not converge publishes, or None.

    Two stages can fail, and the statement names each one that did: the
    smoothing search stopping short of its test (a cap or a stalled step,
    named by its reason), and the final coefficient fit at the selected
    smoothing parameters missing its convergence certificate. A search reason
    that means it converged is never named as the cause. A model whose
    coefficients were revised after fitting (``"coefficients_revised"``)
    publishes nothing here: the editor's own note discloses that.
    """
    if reml_result is None or bool(getattr(reml_result, "converged", True)):
        return None
    reason = getattr(reml_result, "termination_reason", None)
    if reason == "coefficients_revised":
        return None
    n_iter = int(getattr(reml_result, "n_reml_iter", 0) or 0)
    refit_reason = getattr(reml_result, "terminal_refit_termination", None)
    refit = (
        "the final coefficient fit at the selected smoothing parameters did not meet its "
        "convergence test"
        + (f" (it stopped with termination_reason={refit_reason!r})" if refit_reason else "")
    )
    if refit_reason == "constraint_kkt_incomplete":
        refit = (
            "the final coefficient fit at the selected smoothing parameters keeps every "
            "shape constraint, yet could not confirm that its coefficients are the "
            "constrained optimum: the inner quadratic program did not finish its optimality "
            f"(KKT) check (termination_reason={refit_reason!r})"
        )
    # The final refit stopped on the mean-space boundary at the published
    # smoothing parameters, which says where the iterate stopped, not whether
    # an interior optimum exists; Wald inference is unreliable at a boundary.
    # Smoothing parameters the search tried and left (the exact engine's
    # restorations) say nothing of the ones it selected, so they are reported
    # beside the advice, never instead of it.
    boundary = refit_reason == "mean_space_boundary"
    restorations = int(getattr(reml_result, "mean_space_restorations", 0) or 0)
    if boundary:
        refit = (
            "the final coefficient fit at the selected smoothing parameters stopped on the "
            "boundary of the family's mean space (a binomial fitted probability of 0 or 1) "
            f"before its coefficients converged (termination_reason={refit_reason!r})"
        )
    if reason in _REML_CONVERGED_REASONS:
        message = (
            f"fit_reml did not converge: smoothing-parameter selection stopped after "
            f"{_iterations(n_iter)} (termination_reason={reason!r}), but {refit}. The model "
            "is returned with the coefficients, effective degrees of freedom and standard "
            "errors of that fit's last step."
        )
    else:
        why = _REML_REASON_TEXT.get(str(reason), f"termination_reason={reason!r}")
        if reason == "bootstrap_uncertified":
            why = _bootstrap_reason(reml_result)
        message = (
            f"fit_reml did not converge: smoothing-parameter selection stopped after "
            f"{_iterations(n_iter)} because {why}"
            + (f", and {refit}" if refit_reason is not None else "")
            + ". The model is returned, but its smoothing parameters, effective degrees of "
            "freedom (how flexible each term is) and standard errors are those of the last "
            "step it reached, not of a REML optimum."
        )
    if getattr(reml_result, "scop_newton_fallback", None) == "line_search_at_cap":
        message += (
            " Its last Newton step on the smoothing parameters improved nothing, and no "
            "iteration was left to restart the search with smaller steps."
        )
    if restorations and not boundary:
        message += (
            f" At {restorations} smoothing parameter{'s' if restorations != 1 else ''} the "
            "search tried, the penalized maximum held fitted means on the boundary of the "
            "family's mean space (a fitted probability of 1 under the binomial log link); "
            "the search raised the penalties there until the maximum was interior, and "
            "selects only among smoothing parameters that have one."
        )
    if boundary:
        message += (
            " Standard errors and intervals (Wald inference) are unreliable for the terms "
            "whose fitted probabilities reach the boundary. Coarser levels, more data in "
            "those cells, or a link whose means stay inside (0, 1), such as logit, keep "
            "the fit off it."
        )
    elif reason == "max_reml_iter":
        message += (
            " Refit with a larger max_reml_iter"
            + (" and a larger max_pirls_iter" if refit_reason == "max_iter" else "")
            + ", passing lambda2_init=model.reml_diagnostics()['lambdas'] to continue "
            "from this fit."
        )
    elif reason == "bootstrap_uncertified":
        states = getattr(reml_result, "scop_states", None) or {}
        names = sorted({str(state.get("group_name")) for state in states.values()})
        terms = f" ({', '.join(repr(name) for name in names)})" if names else ""
        message += (
            f" The inner fit of the shape-constrained terms{terms} did not settle, so the "
            "smoothing parameters are the "
            + ("data-scaled " if _bootstrap_retried(reml_result) else "")
            + "starting ones in model.reml_diagnostics()['lambdas']. "
            "Refit with lambda2_init naming larger starting values for the smooth terms"
            + (", or with a larger max_pirls_iter" if refit_reason == "max_iter" else "")
            + ", or with a smaller basis (k) for the shape-constrained terms and the terms "
            "they interact with."
        )
    elif refit_reason == "max_iter":
        message += " Refit with a larger max_pirls_iter to let the final fit finish."
    else:
        message += " model.reml_diagnostics() holds the iteration history."
    return message


def _bootstrap_retried(reml_result: Any) -> bool:
    """Whether a SCOP bootstrap retried at data-scaled starts: its ``lambda_history`` holds both."""
    return len(getattr(reml_result, "lambda_history", None) or ()) > 1


def _bootstrap_reason(reml_result: Any) -> str:
    """Why a SCOP bootstrap published no certified mode, naming only the starts that ran."""
    if _bootstrap_retried(reml_result):
        return (
            "no coefficient fit at its starting smoothing parameters, nor at the starting "
            "values scaled to the data's curvature that it retried at, reached a mode that "
            "passes the convergence check"
        )
    return (
        "no coefficient fit at its starting smoothing parameters reached a mode that passes "
        "the convergence check (no retry ran: the starting values scaled to the data's "
        "curvature were the same)"
    )


# Why a neighbourhood cross-validation search stopped short of its first-order test, for each
# stop that means it did (``ncv.result.STOP_REASONS``), and what to do about it.
_NCV_STOP_TEXT = {
    "max_iter": (
        "it reached the max_ncv_iter limit",
        "Refit with a larger max_ncv_iter, passing "
        "lambda2_init=model.ncv_diagnostics()['lambdas'] to continue from this fit.",
    ),
    "line_search": (
        "no further step lowered the held-out deviance",
        "The held-out deviance is flat or uneven around this smoothing: refit with "
        "lambda2_init=model.ncv_diagnostics()['lambdas'] to search again from here, or "
        "with a larger ncv_tol if the smoothing is good enough.",
    ),
    "criterion_settled": (
        "its steps stopped lowering the held-out deviance before its slopes met ncv_tol",
        "The held-out deviance is nearly flat around this smoothing: refit with "
        "lambda2_init=model.ncv_diagnostics()['lambdas'] to search again from here, or "
        "with a larger ncv_tol if the smoothing is good enough.",
    ),
}

# What to do about each reason a trial of the search gave no criterion
# (``ncv.result.TRIAL_FAILURES``).
_NCV_TRIAL_ADVICE = {
    "uncertified_mode": "Refit with a larger max_pirls_iter.",
    "clipped_prediction": (
        "Check the responses and offsets of the held-out rows for extreme values."
    ),
    "non_finite": "Check the responses, weights and offsets for extreme values.",
    "leave_out_rank": (
        "Choose neighbourhoods that leave rows of every term in the fit, or simplify the terms."
    ),
    "rank_deficient_fit": "Drop a term that repeats what another term already carries.",
    "kept_rows_limit": (
        "Neighbourhoods need fits of their own at every smoothing when two terms carry the "
        "same information (a numeric term beside a spline of the same variable, or a spline "
        "of a column that does not vary): drop one of the two, or choose this model's "
        "smoothing with fit_reml."
    ),
    "leave_out_refused": (
        "Choose neighbourhoods that divide the rows of every factor level and term."
    ),
}


def _ncv_trial_reasons(result: Any) -> tuple[str, str]:
    """Why a search's trials gave no criterion, a sentence a reason, and what to do about each."""
    from superglm.ncv.result import TRIAL_FAILURES

    counted = getattr(result, "infeasible_reasons", None) or {}
    reasons = " ".join(
        f"At {count} of them, {TRIAL_FAILURES.get(reason, str(reason))}."
        for reason, count in counted.items()
    )
    advice = " ".join(
        _NCV_TRIAL_ADVICE[reason] for reason in counted if reason in _NCV_TRIAL_ADVICE
    )
    return reasons, advice


def ncv_nonconvergence_message(result: Any) -> str | None:
    """The statement a ``fit_ncv`` fit that did not converge publishes, or None.

    Three things can fall short, and the statement names each that did: the smoothing search
    finding no smoothing at which its criterion could be computed (the model is then a
    coefficient fit at the smoothing the search started from, and the reasons its trials
    recorded are given in words); the search stopping short of its first-order test; and the
    final coefficient fit at the returned smoothing missing its convergence certificate.  A
    model whose coefficients were revised after fitting publishes nothing here.
    """
    if result is None or bool(getattr(result, "converged", True)):
        return None
    reason = getattr(result, "termination_reason", None)
    if reason == "coefficients_revised":
        return None
    refit_reason = getattr(result, "terminal_refit_termination", None)
    refit = (
        "the final coefficient fit at that smoothing did not meet its convergence test "
        f"(it stopped with termination_reason={refit_reason!r})"
    )
    more_iterations = (
        " Refit with a larger max_pirls_iter to let the final fit finish."
        if refit_reason == "max_iter"
        else ""
    )
    if reason == "no_finite_criterion":
        tried = int(getattr(result, "n_evaluations", 0) or 0)
        reasons, advice = _ncv_trial_reasons(result)
        return (
            "fit_ncv chose no smoothing: its held-out deviance could not be computed at the "
            f"{tried} smoothing setting{'' if tried == 1 else 's'} it tried."
            + (f" {reasons}" if reasons else "")
            + " The model is returned as a coefficient fit at the smoothing the search "
            "started from (model.ncv_diagnostics()['lambdas']), which held-out prediction "
            "did not choose"
            + (f", and {refit}" if refit_reason is not None else "")
            + "."
            + (f" {advice}" if advice else "")
        )
    steps = int(getattr(result, "n_reml_iter", 0) or 0)
    after = f"after {steps} step{'' if steps == 1 else 's'}"
    if reason in _NCV_STOP_TEXT:
        why, advice = _NCV_STOP_TEXT[reason]
        return (
            f"fit_ncv did not converge: its smoothing search stopped {after} because {why}"
            + (f", and {refit}" if refit_reason is not None else "")
            + ". The model is returned at the lowest held-out deviance the search reached, "
            "with the coefficients, effective degrees of freedom (how flexible each term is) "
            "and standard errors of the fit there; that smoothing is not established as the "
            f"one that predicts held-out rows best. {advice}" + more_iterations
        )
    return (
        f"fit_ncv did not converge: its smoothing search met its test {after}, but {refit}. "
        "The model is returned with the coefficients, effective degrees of freedom and "
        "standard errors of that fit's last step, and without held-out results."
        + (
            more_iterations
            or " The stop is not an iteration cap, so more iterations will not help: check "
            "the responses, weights and offsets for extreme values, or simplify the terms."
        )
    )


def warn_ncv_nonconvergence(result: Any) -> str | None:
    """Emit the ``fit_ncv`` non-convergence statement as a ConvergenceWarning; return it."""
    message = ncv_nonconvergence_message(result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def ncv_bound_message(result: Any) -> str | None:
    """The terms a ``fit_ncv`` search left on a limit of its smoothing range, or None.

    Every estimated component on a bound is named, whether or not the criterion still fell
    there: a spline at the largest smoothing is a straight line, which the reader should know.
    Those the bound held back (an outward slope of at least the stopping bar) are named again.
    """
    from superglm._lambda_box import LambdaBox, bound_warning_message

    box = getattr(result, "lambda_box", None)
    if not isinstance(box, LambdaBox):
        return None
    lambdas = result.lambdas
    message = bound_warning_message(
        [
            (
                name,
                side,
                float(lambdas[name]),
                box.floor if side == "floor" else box.ceiling,
                box.unit,
            )
            for name, side in getattr(result, "at_bound", ()) or ()
            if name in lambdas
        ],
        lead="fit_ncv's smoothing search ended at a limit of its range on ",
    )
    held = [repr(name) for name, _ in getattr(result, "bound_pressure", ()) or ()]
    if message is None or not held:
        return message
    return (
        f"{message} The held-out deviance was still falling at the limit for "
        f"{', '.join(held)}, so the limit, not the data, set that smoothing."
    )


def warn_ncv_bounds(result: Any) -> str | None:
    """Emit ``ncv_bound_message`` as a ConvergenceWarning; return it."""
    message = ncv_bound_message(result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def search_nonconvergence_message(criterion: str, result: Any) -> str | None:
    """The non-convergence statement of a SuperGLM smoothing search by ``criterion``, or None.

    ``criterion`` is the search's (``fit_method.search_criterion``): ``"reml"`` gives
    ``fit_reml``'s statement, ``"ncv"`` ``fit_ncv``'s.
    """
    return _SEARCH_NONCONVERGENCE[criterion](result)


_SEARCH_NONCONVERGENCE = {
    "reml": reml_nonconvergence_message,
    "ncv": ncv_nonconvergence_message,
}


def lss_nonconvergence_reason(
    fitted_result: Any, smoothing_reason: str | None, coefficient_reason: str | None = None
) -> str | None:
    """Why a SuperLSS fit did not converge, in a few words, or None when it did."""
    if fitted_result is None or bool(getattr(fitted_result, "converged", True)):
        return None
    if not bool(getattr(fitted_result, "coefficient_converged", True)):
        return "the coefficient fit did not converge" + (
            f" (reason {coefficient_reason!r})" if coefficient_reason else ""
        )
    return f"smoothing selection stopped with reason {smoothing_reason!r}"


def lss_nonconvergence_message(
    fitted_result: Any, smoothing_reason: str | None, coefficient_reason: str | None = None
) -> str | None:
    """The statement a SuperLSS fit that did not converge publishes, or None.

    The budget that ran out is named only when the stop was that budget.
    """
    why = lss_nonconvergence_reason(fitted_result, smoothing_reason, coefficient_reason)
    if why is None:
        return None
    coefficient = not bool(getattr(fitted_result, "coefficient_converged", True))
    reason = coefficient_reason if coefficient else smoothing_reason
    budget = (
        ("max_inner_iter" if coefficient else "max_reml_iter")
        if reason == "max_iterations"
        else None
    )
    return (
        f"SuperLSS fit did not converge: {why}. The model is returned, but its "
        "coefficients, smoothing parameters, effective degrees of freedom (how flexible "
        "each term is) and standard errors are those of the last step it reached; the "
        "effective degrees of freedom of a term can then be negative. model.diagnose() "
        "explains the stop. Refit with "
        + (f"a larger {budget}, or with " if budget else "")
        + "other starting smoothing parameters (initial_lambda or lambdas), or simplify the "
        "terms it names, before reading the summary." + lss_ill_conditioned_note(fitted_result)
    )


def lss_ill_conditioned_note(fitted_result: Any) -> str:
    """The cause, when an unconverged coefficient fit's curvature resolves under half of float64's digits.

    ``condition_estimate`` is the equilibrated curvature's on the factor
    scale, so the curvature's own is its square. Past ``u**-0.5`` the
    curvature resolves fewer than half of float64's digits, and the stop
    certificate, which carries ``||H^-1||``, can miss the fit's tolerance at
    a point the iteration has reached.
    """
    if bool(getattr(fitted_result, "coefficient_converged", True)):
        return ""
    telemetry = getattr(fitted_result, "curvature_telemetry", None)
    factor = getattr(telemetry, "condition_estimate", None)
    if factor is None or not factor**2 > 2.0**26.5:
        return ""
    return (
        f" The coefficient curvature's condition number is about {factor**2:.0e} (a measure "
        "of how close to singular it is). The usual "
        "causes are a numeric column far from zero relative to its spread, beside an intercept "
        "(subtract its mean before fitting), and columns that are nearly linear combinations "
        "of one another (drop or combine them)."
    )


def lss_semidefinite_curvature_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its final curvature had to be projected."""
    return (
        f"SuperLSS {family_name} fit: the observed curvature at the final coefficients still "
        "curved the wrong way in at least one direction (it was indefinite) after a tighter "
        "retry, and no usable expected information was available, "
        "so the standard errors, effective degrees of freedom and smoothing updates use the "
        "nearest positive semidefinite approximation of the observed curvature. This "
        "usually means the coefficient fit stopped short of a maximum; "
        "model.diagnose() explains the stop. Refit with a larger max_inner_iter, larger "
        "starting smoothing parameters (initial_lambda), or simpler terms before reading the "
        "summary."
    )


def lss_fisher_curvature_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its final curvature fell back to Fisher information."""
    return (
        f"SuperLSS {family_name} fit: the observed curvature at the final coefficients still "
        "curved the wrong way in at least one direction (it was indefinite) after a tighter "
        "retry, so the standard errors, effective degrees of freedom "
        "and smoothing updates use the expected (Fisher) information instead. Curvature like "
        "that means the fit is not at a proper maximum of the observed "
        "likelihood: it stopped short of one, or the maximum lies on a boundary such as a "
        "size at its Poisson limit; model.diagnose() explains the stop. Refit with a larger "
        "max_inner_iter, larger starting smoothing parameters (initial_lambda), or simpler "
        "terms before reading the summary."
    )


def lss_no_ascent_direction_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its coefficient solver found no ascent direction."""
    return (
        f"SuperLSS {family_name} fit: the coefficient solver found no ascent direction (no step "
        "it tried, at any damping of the curvature, improved the penalized likelihood), "
        "so it stopped unconverged at the last coefficients it reached, with reason "
        "'no_ascent_direction'. model.diagnose() reports the stop. Refit with "
        "coefficient_curvature='fisher' where the family supplies expected information, larger "
        "starting smoothing parameters (initial_lambda), or simpler terms."
    )


def lss_null_model_nonconvergence_message(family_name: str, reason: str) -> str:
    """The disclosure a SuperLSS fit makes when its intercept-only null fit did not converge."""
    return (
        f"SuperLSS {family_name} fit: the intercept-only null model did not converge "
        f"(reason {reason!r}). The fitted model is unaffected, but result_.null_objective, "
        "the null model kept with the fit and anything compared with it are based on the "
        "last step the unconverged null fit reached. Refit with a larger max_inner_iter to let "
        "the null fit converge; model.diagnose() reports the stop."
    )


def null_model_nonconvergence_message(offset_span: float) -> str:
    """The disclosure a SuperGLM fit makes when its offset-aware null fit did not converge."""
    return (
        "The intercept-only null model (the offset kept, no terms) used its step limit "
        "before it converged, so the null deviance, null log-likelihood and explained "
        "deviance in summary() and metrics() are those of its last step. The fitted model, "
        f"its predictions and standard errors are unaffected. The offset spans {offset_span:.4g} "
        "on the scale of the linear predictor; check it for extreme values."
    )


def warn_null_model_nonconvergence(offset_span: float) -> str:
    """Emit ``null_model_nonconvergence_message`` as a ConvergenceWarning; return it."""
    message = null_model_nonconvergence_message(offset_span)
    warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def warn_null_model_unavailable(link_name: str, mean_response: float) -> str:
    """Warn that the offset-aware null model has no fit, so its figures are NaN."""
    message = (
        f"The intercept-only null model (the offset kept, no terms) has no fit: the {link_name} "
        f"has no linear predictor for the weighted mean response {mean_response:.6g}, so the "
        "null deviance, null log-likelihood and explained deviance in summary() and metrics() "
        "are NaN. The fitted model, its predictions and standard errors are unaffected. "
        "The null model needs a mean response the link can take (a positive one under a log "
        "link)."
    )
    warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def lss_corrected_covariance_substitute_message(source: str) -> str:
    """What a corrected covariance means when the final curvature fell back."""
    curvature = (
        "expected (Fisher) information"
        if source == "fisher"
        else "nearest positive semidefinite approximation of the observed curvature"
    )
    return (
        "SuperLSS corrected covariance: the standard errors include extra uncertainty from "
        "having chosen the smoothing parameters. The observed curvature at the final "
        f"coefficients was materially indefinite, so the fit used the {curvature} instead, and "
        "the correction was worked out from it too, as a first-order correction for the "
        "substitute curvature. It is therefore approximate; model.diagnose() explains the "
        "fallback."
    )


# The Fellner-Schall stops are the ones the endgame can improve on, on a family with
# continuous curvature; an outer='efs+newton' fit already ran it.
_NONSTATIONARY_DEFAULT = "model.diagnose() explains the stop."
_NONSTATIONARY_REMEDY = dict.fromkeys(
    ("lambda_change", "objective_plateau", "practical_plateau"),
    "Refit with outer='efs+newton' to stop at an optimum; model.diagnose() explains the stop.",
)


def lss_corrected_covariance_nonstationary_message(reason: str, *, newton_can_help: bool) -> str:
    """What a corrected covariance means when smoothing did not stop at an optimum.

    ``newton_can_help`` says whether refitting with the Newton endgame is advice
    worth giving: the fit stopped on Fellner-Schall on a family whose curvature is
    continuous.
    """
    remedy = _NONSTATIONARY_REMEDY.get(reason) if newton_can_help else None
    return (
        f"SuperLSS corrected covariance: smoothing stopped with reason {reason!r}, before "
        "reaching an optimum, so the correction for smoothing-parameter uncertainty is "
        "approximate. The correction of Wood, Pya and Saefken (2016, section 4) is derived at "
        "an optimum; here it is a first-order correction about the fitted smoothing "
        f"parameters. {remedy or _NONSTATIONARY_DEFAULT}"
    )


def lss_corrected_covariance_unavailable_message(cause: str) -> str:
    """The disclosure when no smoothing-parameter correction can be formed for a fit."""
    return (
        f"SuperLSS corrected covariance is unavailable for this fit: {cause}. Returned the "
        "conditional covariance at the fitted smoothing parameters, as covariance='fixed' "
        "gives, so it omits smoothing-parameter uncertainty and intervals built on it may be "
        "narrower than corrected ones would be."
    )


def _clip_held_message(fit_result: Any) -> str:
    """The statement for a fit stopped by the solver's clip on rows held against their outcome."""
    rows = int(getattr(fit_result, "clip_held_rows", 0) or 0)
    return (
        f"The fit holds {rows} row{'' if rows == 1 else 's'} at the solver's clip against "
        "the response: an event at a fitted probability below 1e-7, or a non-event above "
        "1 - 1e-7. The solver read such a row at the clip, so the clip, not the likelihood, "
        "decided where the fit stopped: it is reported as not converged and its "
        "coefficients and standard errors may be far from the maximum-likelihood estimate "
        "(termination_reason='clip_held'). A logit, probit or cloglog link without a "
        "selection penalty reads such rows exactly; otherwise check those rows' offsets and "
        "responses."
    )


def coefficient_nonconvergence_message(fit_result: Any) -> str | None:
    """The statement for an unconverged coefficient fit with no smoothing selection."""
    if fit_result is None or bool(getattr(fit_result, "converged", True)):
        return None
    reason = getattr(fit_result, "termination_reason", None)
    if reason == "coefficients_revised":
        return None
    if reason == "constraint_kkt_incomplete":
        return (
            "The coefficient fit satisfies every shape constraint. Its inner quadratic "
            "program could not confirm optimality (KKT conditions), so the coefficients, "
            "predictions and standard errors may be inaccurate "
            "(termination_reason='constraint_kkt_incomplete')."
        )
    if reason == "mean_space_boundary":
        return (
            f"The fit holds {fit_result.mean_space_boundary_rows} row(s) at a fitted "
            "probability of 0 or 1, the boundary of the model's parameter space. Its maximum "
            "lies on that boundary rather than inside it, so the fit is reported as not "
            "converged and its standard errors there are not reliable. A logit link keeps "
            "every probability inside (0, 1); a categorical level whose responses are all 0 "
            "or all 1 can also be merged."
        )
    if reason == "clip_held":
        return _clip_held_message(fit_result)
    n_iter = getattr(fit_result, "n_iter", None)
    return (
        "SuperGLM coefficient fit did not converge"
        + (f" after {_iterations(n_iter)}" if n_iter is not None else "")
        + (f" (termination_reason={reason!r})" if reason else "")
        + ". The model is returned, but its coefficients and standard errors may be "
        "inaccurate. "
        + (
            "Refit with a larger max_iter (max_pirls_iter under fit_reml)."
            if reason == "max_iter"
            else "The stop is not an iteration cap, so more iterations will not help: check "
            "the responses, weights and offsets for extreme values, or simplify the terms."
        )
    )


# A warning names the first frame outside the package, however the fit was reached.
_PACKAGE_PREFIX = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def warn_coefficient_nonconvergence(fit_result: Any) -> str | None:
    """Emit the coefficient non-convergence statement as a ConvergenceWarning; return it."""
    message = coefficient_nonconvergence_message(fit_result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def warn_reml_nonconvergence(reml_result: Any) -> str | None:
    """Emit the REML non-convergence statement as a ConvergenceWarning; return it."""
    message = reml_nonconvergence_message(reml_result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message


def reml_bound_message(reml_result: Any) -> str | None:
    """The terms whose smoothing a search bound held back, or None."""
    from superglm._lambda_box import LambdaBox, bound_warning_message

    box = getattr(reml_result, "lambda_box", None)
    if not isinstance(box, LambdaBox):
        return None
    lambdas = reml_result.lambdas
    return bound_warning_message(
        [
            (
                name,
                side,
                float(lambdas[name]),
                box.floor if side == "floor" else box.ceiling,
                box.unit,
            )
            for name, side in getattr(reml_result, "bound_pressure", ()) or ()
            if name in lambdas
        ]
    )


def warn_reml_bounds(reml_result: Any) -> str | None:
    """Emit ``reml_bound_message`` as a ConvergenceWarning; return it."""
    message = reml_bound_message(reml_result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, skip_file_prefixes=(_PACKAGE_PREFIX,))
    return message
