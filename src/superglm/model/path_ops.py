"""Helpers for regularization path fitting."""

from __future__ import annotations

import numpy as np

from superglm.solvers.pirls import fit_pirls


def resolve_lambda_sequence(lambda_max, *, n_lambda=50, lambda_ratio=1e-3, lambda_seq=None):
    """Resolve the lambda path sequence."""
    if lambda_seq is None:
        return np.geomspace(lambda_max, lambda_max * lambda_ratio, n_lambda)
    return np.asarray(lambda_seq, dtype=np.float64)


def run_lambda_path(
    model,
    *,
    y,
    sample_weight,
    offset,
    lambda_seq,
):
    """Run the PIRLS warm-start path and return arrays plus the final result."""
    n_lambda = len(lambda_seq)
    p = model._dm.p
    coef_path = np.zeros((n_lambda, p))
    intercept_path = np.zeros(n_lambda)
    deviance_path = np.zeros(n_lambda)
    edf_path = np.zeros(n_lambda)
    n_iter_path = np.zeros(n_lambda, dtype=int)
    converged_path = np.zeros(n_lambda, dtype=bool)

    beta_warm = None
    intercept_warm = None
    result = None

    for i, lam in enumerate(lambda_seq):
        model.penalty.lambda1 = lam
        result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=model.penalty,
            offset=offset,
            beta_init=beta_warm,
            intercept_init=intercept_warm,
            active_set=model._active_set,
            lambda2=model.lambda2,
        )
        coef_path[i] = result.beta
        intercept_path[i] = result.intercept
        deviance_path[i] = result.deviance
        edf_path[i] = result.effective_df
        n_iter_path[i] = result.n_iter
        converged_path[i] = result.converged
        beta_warm = result.beta
        intercept_warm = result.intercept

    return {
        "coef_path": coef_path,
        "intercept_path": intercept_path,
        "deviance_path": deviance_path,
        "edf_path": edf_path,
        "n_iter_path": n_iter_path,
        "converged_path": converged_path,
        "result": result,
    }
