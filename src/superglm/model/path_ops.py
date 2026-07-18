"""Helpers for regularization path fitting."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from superglm.model.fit_state import configured_lambda2, configured_penalty
from superglm.solvers.pirls import fit_pirls


def validate_lambda_path_controls(*, n_lambda=50, lambda_ratio=1e-3, lambda_seq=None):
    """Validate path controls and take ownership of a caller-supplied sequence."""
    if lambda_seq is not None:
        sequence = np.array(lambda_seq, dtype=np.float64, copy=True)
        if sequence.ndim != 1:
            raise ValueError("lambda_seq must be one-dimensional")
        if sequence.size == 0:
            raise ValueError("lambda_seq must be non-empty")
        if not np.all(np.isfinite(sequence)):
            raise ValueError("lambda_seq values must be finite")
        if np.any(sequence < 0.0):
            raise ValueError("lambda_seq values must be non-negative")
        if np.any(np.diff(sequence) > 0.0):
            raise ValueError("lambda_seq must be non-increasing")
        return sequence

    if isinstance(n_lambda, bool) or not isinstance(n_lambda, Integral) or n_lambda < 1:
        raise ValueError("n_lambda must be a positive integer")
    if not np.isscalar(lambda_ratio):
        raise ValueError("lambda_ratio must be a finite scalar in (0, 1]")
    ratio = float(lambda_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError("lambda_ratio must be finite and in (0, 1]")
    return None


def resolve_lambda_sequence(lambda_max, *, n_lambda=50, lambda_ratio=1e-3, lambda_seq=None):
    """Resolve a validated, model-owned lambda path sequence."""
    sequence = validate_lambda_path_controls(
        n_lambda=n_lambda,
        lambda_ratio=lambda_ratio,
        lambda_seq=lambda_seq,
    )
    if sequence is not None:
        return sequence
    lambda_max = float(lambda_max)
    if not np.isfinite(lambda_max) or lambda_max < 0.0:
        raise ValueError("lambda_max must be finite and non-negative")
    if lambda_max == 0.0:
        return np.zeros(int(n_lambda), dtype=np.float64)
    return np.geomspace(lambda_max, lambda_max * float(lambda_ratio), int(n_lambda))


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
    penalty = configured_penalty(model)
    lambda2 = configured_lambda2(model)

    for i, lam in enumerate(lambda_seq):
        penalty.lambda1 = lam
        result = fit_pirls(
            X=model._dm,
            y=y,
            weights=sample_weight,
            family=model._distribution,
            link=model._link,
            groups=model._groups,
            penalty=penalty,
            offset=offset,
            beta_init=beta_warm,
            intercept_init=intercept_warm,
            active_set=model._active_set,
            lambda2=lambda2,
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
