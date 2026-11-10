"""Disclosure for fits that stop before their convergence test passes.

A fit that cannot meet its convergence test is returned, not refused, and it
says so: a ``ConvergenceWarning`` when it finishes, and a statement in the
summary and diagnostics of the model it publishes.
"""

from __future__ import annotations

import os
import warnings
from typing import Any


class ConvergenceWarning(UserWarning):
    """A fit stopped before its convergence test passed.

    The fitted model is still returned. ``SuperGLM.fit_reml`` warns when
    smoothing-parameter selection stops unconverged, or its final coefficient
    fit does, and so do ``estimate_p`` and ``estimate_theta`` with
    ``fit_mode="reml"`` for the fit they publish; the summary's ``Converged``
    row and ``reml_diagnostics()["converged"]`` then read ``False``.
    ``SuperLSS.fit_reml`` warns when its coefficient or smoothing loop stops
    unconverged; ``result_.converged`` then reads ``False`` and the summary's
    ``note`` column says so (``SuperLSS.fit``, with fixed smoothing, reports
    it there without a warning). ``SuperLSS`` also warns, and keeps the text
    in ``result_.warnings``, on a fit that may itself have converged: when
    its final curvature is the nearest semidefinite matrix, or is Fisher
    information for ``NegativeBinomialLS`` (other families' Fisher fallback
    is silent, as in 0.40), when its null model did not converge, when
    an inner fit stops with ``no_ascent_direction``, and when a
    ``NegativeBinomialLS`` fit has rows whose size is at the Poisson limit
    and its final curvature is singular in a size direction, whatever the
    cause (the message gives the row count). The message names the
    reason and what to change. Silence it with ``warnings.filterwarnings``
    once the reason is understood.
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
    # The penalized maximum on the mean-space boundary at the published
    # smoothing parameters: the model has no interior optimum there, which
    # no iteration budget changes.  Smoothing parameters the search tried and
    # left (the exact engine's restorations) say nothing of the ones it
    # selected, so they are reported beside the advice, never instead of it.
    boundary = refit_reason == "mean_space_boundary"
    restorations = int(getattr(reml_result, "mean_space_restorations", 0) or 0)
    if boundary:
        refit = (
            "the final coefficient fit at the selected smoothing parameters holds fitted "
            "means on the boundary of the family's mean space "
            f"(termination_reason={refit_reason!r})"
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
            " The penalized maximum lies on the boundary of the family's mean space (a "
            "binomial fitted probability of 0 or 1) at the selected smoothing "
            "parameters: the model has no interior optimum there, so more iterations will "
            "not make it converge. A link whose means stay inside (0, 1), such as logit, "
            "has one."
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


def coefficient_nonconvergence_message(fit_result: Any) -> str | None:
    """The statement for an unconverged coefficient fit with no smoothing selection."""
    if fit_result is None or bool(getattr(fit_result, "converged", True)):
        return None
    reason = getattr(fit_result, "termination_reason", None)
    if reason == "coefficients_revised":
        return None
    if reason == "mean_space_boundary":
        return (
            f"The fit holds {fit_result.mean_space_boundary_rows} row(s) at a fitted "
            "probability of 0 or 1, the boundary of the model's parameter space. Its maximum "
            "lies on that boundary rather than inside it, so the fit is reported as not "
            "converged and its standard errors there are not reliable. A logit link keeps "
            "every probability inside (0, 1); a categorical level whose responses are all 0 "
            "or all 1 can also be merged."
        )
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
