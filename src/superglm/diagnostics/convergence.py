"""Disclosure for fits that stop before their convergence test passes.

A fit that cannot meet its convergence test is returned, not refused, and it
says so: a ``ConvergenceWarning`` when it finishes, and a statement in the
summary and diagnostics of the model it publishes.
"""

from __future__ import annotations

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
    it there without a warning). The message names the reason and what to
    change. Silence it with ``warnings.filterwarnings`` once the reason is
    understood.
    """


# Why a smoothing search stopped short of its convergence test, for each
# termination reason that means it did.
_REML_REASON_TEXT = {
    "max_reml_iter": "it reached the max_reml_iter limit",
    "line_search_failed": "no smoothing step improved the REML objective",
    "line_search_stalled": "no smoothing step improved the REML objective",
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
        "active_set_stationary",
        "converged_at_precision",
        "fixed_lambdas",
    }
)


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
            "shape constraint, yet could not certify that its coefficients are the "
            "constrained optimum: the inner quadratic program did not complete its KKT "
            f"certificate (termination_reason={refit_reason!r})"
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
            f"{n_iter} iterations (termination_reason={reason!r}), but {refit}. The model "
            "is returned with the coefficients, effective degrees of freedom and standard "
            "errors of that fit's last iterate."
        )
    else:
        why = _REML_REASON_TEXT.get(str(reason), f"termination_reason={reason!r}")
        if reason == "bootstrap_uncertified":
            why = _bootstrap_reason(reml_result)
        message = (
            f"fit_reml did not converge: smoothing-parameter selection stopped after "
            f"{n_iter} iterations because {why}"
            + (f", and {refit}" if refit_reason is not None else "")
            + ". The model is returned, but its smoothing parameters, effective degrees of "
            "freedom and standard errors are those of the last iterate, not of a REML "
            "optimum."
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
            "fitted probability of 1 under the binomial log link) at the selected smoothing "
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
            "passes the convergence certificate"
        )
    return (
        "no coefficient fit at its starting smoothing parameters reached a mode that passes "
        "the convergence certificate (no retry ran: the starting values scaled to the data's "
        "curvature were the same)"
    )


def lss_nonconvergence_reason(fitted_result: Any, smoothing_reason: str | None) -> str | None:
    """Why a SuperLSS fit did not converge, in a few words, or None when it did."""
    if fitted_result is None or bool(getattr(fitted_result, "converged", True)):
        return None
    if not bool(getattr(fitted_result, "coefficient_converged", True)):
        return "the coefficient fit did not converge"
    return f"smoothing selection stopped with reason {smoothing_reason!r}"


def lss_nonconvergence_message(fitted_result: Any, smoothing_reason: str | None) -> str | None:
    """The statement a SuperLSS fit that did not converge publishes, or None."""
    why = lss_nonconvergence_reason(fitted_result, smoothing_reason)
    if why is None:
        return None
    return (
        f"SuperLSS fit did not converge: {why}. The model is returned, but its "
        "coefficients, smoothing parameters, effective degrees of freedom and standard "
        "errors are those of the last accepted iterate; an effective degree of freedom "
        "can then be negative. model.diagnose() explains the stop. Refit with other "
        "starting smoothing parameters (initial_lambda or lambdas), or simplify the "
        "terms it names, before reading the summary."
    )


def lss_semidefinite_curvature_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its final curvature had to be projected."""
    return (
        f"SuperLSS {family_name} fit: the observed curvature at the final coefficients stayed "
        "indefinite after a tighter retry, and no usable expected information was available, "
        "so the standard errors, effective degrees of freedom and smoothing updates use the "
        "nearest positive semidefinite approximation of the observed curvature. Indefinite "
        "curvature usually means the coefficient fit stopped short of a maximum; "
        "model.diagnose() explains the stop. Refit with a larger max_inner_iter, larger "
        "starting smoothing parameters (initial_lambda), or simpler terms before reading the "
        "summary."
    )


def lss_fisher_curvature_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its final curvature fell back to Fisher information."""
    return (
        f"SuperLSS {family_name} fit: the observed curvature at the final coefficients stayed "
        "indefinite after a tighter retry, so the standard errors, effective degrees of freedom "
        "and smoothing updates use the expected (Fisher) information instead. Indefinite "
        "observed curvature means the fit is not at a proper maximum of the observed "
        "likelihood: it stopped short of one, or the maximum lies on a boundary such as a "
        "size at its Poisson limit; model.diagnose() explains the stop. Refit with a larger "
        "max_inner_iter, larger starting smoothing parameters (initial_lambda), or simpler "
        "terms before reading the summary."
    )


def lss_no_ascent_direction_message(family_name: str) -> str:
    """The disclosure a SuperLSS fit makes when its coefficient solver found no ascent direction."""
    return (
        f"SuperLSS {family_name} fit: the coefficient solver found no ascent direction (no "
        "damping of the curvature gave a step certified to increase the penalized likelihood), "
        "so it stopped unconverged at its last accepted coefficients with reason "
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
        "unconverged null's last accepted iterate. Refit with a larger max_inner_iter to let "
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
        "SuperLSS corrected covariance: the observed curvature at the final coefficients was "
        f"materially indefinite, so the fit used the {curvature} instead, and the correction "
        "uses it too. V is that curvature's inverse, J = d beta / d rho is the implicit "
        "derivative of the coefficients under it rather than under the observed Hessian, and "
        "V_rho inverts the Hessian in log smoothing parameters of the Laplace objective formed "
        "with it, replayed from the retained rows. V + J V_rho J' is a first-order correction "
        "for the substitute curvature; model.diagnose() explains the fallback."
    )


def coefficient_nonconvergence_message(fit_result: Any) -> str | None:
    """The statement for an unconverged coefficient fit with no smoothing selection."""
    if fit_result is None or bool(getattr(fit_result, "converged", True)):
        return None
    reason = getattr(fit_result, "termination_reason", None)
    n_iter = getattr(fit_result, "n_iter", None)
    return (
        "SuperGLM coefficient fit did not converge"
        + (f" after {n_iter} iterations" if n_iter is not None else "")
        + (f" (termination_reason={reason!r})" if reason else "")
        + ". This model has no smoothing selection to run, so fit_reml fitted the "
        "coefficients directly. The model is returned, but its coefficients and "
        "standard errors may be inaccurate. Refit with a larger max_pirls_iter."
    )


def warn_coefficient_nonconvergence(fit_result: Any, *, stacklevel: int = 2) -> str | None:
    """Emit the coefficient non-convergence statement as a ConvergenceWarning; return it."""
    message = coefficient_nonconvergence_message(fit_result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, stacklevel=stacklevel + 1)
    return message


def warn_reml_nonconvergence(reml_result: Any, *, stacklevel: int = 2) -> str | None:
    """Emit the REML non-convergence statement as a ConvergenceWarning; return it."""
    message = reml_nonconvergence_message(reml_result)
    if message is not None:
        warnings.warn(message, ConvergenceWarning, stacklevel=stacklevel + 1)
    return message
