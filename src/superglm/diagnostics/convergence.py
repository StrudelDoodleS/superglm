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
    if reason in _REML_CONVERGED_REASONS:
        message = (
            f"fit_reml did not converge: smoothing-parameter selection stopped after "
            f"{n_iter} iterations (termination_reason={reason!r}), but {refit}. The model "
            "is returned, but its coefficients, effective degrees of freedom and standard "
            "errors are those of that fit's last iterate."
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
    if reason == "max_reml_iter":
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
