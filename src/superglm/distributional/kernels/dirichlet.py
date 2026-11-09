"""Float64 Dirichlet density and signed natural jets for row-first compositions.

The response is the mathematical normalization of the supplied positive row,
not a repeatedly rounded division. Close log ratios use compensated cross
products. Gamma normalizers and scaled digamma residuals remove cancellation
at high concentration. No parameter-dependent cache or shape wall is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from superglm.distributional.kernels._common import readonly, validated_derivative_order
from superglm.distributional.kernels._dirichlet_numba import (
    _FAST_MAX,
    _FAST_MIN,
    _curvature_batch,
    _curvature_python,
    _evaluate_batch,
    _row_state,
    _scaled_product,
)

_U = 2.0**-53
NORMALIZATION_POLICY = "positive-shares/compensated-normalization-8gamma/v1"


def _sum(values):
    return math.fsum(map(float, values))


def prepare_log_shares(response):
    """Validate the simplex and prepare normalized logs once per bound plan.

    For K positive inputs the allowed sum error is 8 gamma_(K+1) times
    the row magnitude, with unit roundoff u=2^-53. Retain the supplied bytes.
    The low sum part matters when one component rounds to one.
    """
    x = np.asarray(response, dtype=np.float64)
    if x.ndim != 2 or not len(x) or x.shape[1] < 2:
        raise ValueError("Dirichlet response must have shape (n, K), K >= 2")
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Dirichlet shares must be finite and strictly positive")
    k = x.shape[1]
    allowance = 8 * ((k + 1) * _U / (1 - (k + 1) * _U))
    logs = np.empty_like(x)
    for i, row in enumerate(x):
        try:
            total = _sum(row)
            tail = _sum([*row, -total])
        except OverflowError as exc:
            raise ValueError("Dirichlet shares must sum to one") from exc
        if abs(total - 1.0) > allowance * total:
            raise ValueError("Dirichlet shares must sum to one within 8 gamma_(K+1)")
        logtotal = math.log(total) + math.log1p(tail / total)
        logs[i] = [math.log(float(v)) - logtotal for v in row]
    return logs


@dataclass(frozen=True)
class DirichletRows:
    optimizing_log_likelihood: np.ndarray
    parameter_independent_carrier: np.ndarray
    score: np.ndarray | None
    hessian_packed: np.ndarray | None
    valid: np.ndarray
    failure_reasons: tuple[str | None, ...]

    @property
    def reported_log_likelihood(self):
        return self.optimizing_log_likelihood

    @property
    def derivative_order(self):
        return 0 if self.score is None else (1 if self.hessian_packed is None else 2)


def evaluate_dirichlet_rows(response, theta, multiplier, *, derivative_order=2, log_shares=None):
    """Primitive core result; the family adapter supplies NaturalLikelihoodEvaluation.

    Invalid requested numerical channels have finite zero placeholders and a
    false row-validity flag. The scaled log-link reconstruction is an admission
    check, not a replacement Hessian or a rank decision. Order zero uses exactly
    the same normalized value as the derivative orders.
    """
    order = validated_derivative_order(derivative_order)
    x, alpha, mass = (np.asarray(v, dtype=np.float64) for v in (response, theta, multiplier))
    if x.ndim != 2 or alpha.shape != x.shape or mass.shape != (len(x),):
        raise ValueError("Dirichlet response, parameters and row multiplier must agree")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("row multiplier must be finite and positive")
    logs = prepare_log_shares(x) if log_shares is None else np.asarray(log_shares)
    if logs.shape != x.shape or not np.isfinite(logs).all():
        raise ValueError("prepared log shares must match the response")
    n, k = x.shape
    pairs = tuple(combinations_with_replacement(range(k), 2))
    values, score_work, hessian_work, valid = _evaluate_batch(x, alpha, mass, order, logs)
    score = score_work if order else None
    hessian = hessian_work if order == 2 else None
    failures = [None] * n
    # Exceptional rows keep the original numerical exception/failure message.
    # Ordinary batches perform no Python per-row numerical work.
    for i in np.flatnonzero(~valid):
        try:
            a = alpha[i]
            if np.any(a <= 0) or not np.isfinite(a).all():
                raise ValueError("alphas must be finite and positive")
            with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
                value, scaled_s, scaled_h, *_ = _row_state(x[i], logs[i], a)
                value *= mass[i]
                if not math.isfinite(value):
                    raise ValueError("nonrepresentable normalized log likelihood")
                if score is not None:
                    natural_s = (scaled_s / a) * mass[i]
                    target = scaled_s * mass[i]
                    reconstructed = natural_s * a
                    if not np.isfinite(natural_s).all() or np.any(
                        np.abs(reconstructed - target) > 128 * _U * np.abs(target)
                    ):
                        raise ValueError("nonrepresentable natural score / scaled log-link score")
                if hessian is not None:
                    natural_h = np.array(
                        [
                            _scaled_product(scaled_h[j, column], mass[i], 1 / a[j], 1 / a[column])
                            for j, column in pairs
                        ]
                    )
                    target_h = np.array([scaled_h[j, column] * mass[i] for j, column in pairs])
                    reconstruction = np.array(
                        [
                            _scaled_product(h, a[j], a[column])
                            for h, (j, column) in zip(natural_h, pairs, strict=True)
                        ]
                    )
                    if not np.isfinite(natural_h).all() or np.any(
                        np.abs(reconstruction - target_h) > 256 * _U * np.abs(target_h)
                    ):
                        raise ValueError(
                            "nonrepresentable natural Hessian / scaled log-link curvature"
                        )
                values[i] = value
                if score is not None:
                    score[i] = natural_s
                if hessian is not None:
                    hessian[i] = natural_h
            valid[i] = True
        except (ValueError, OverflowError, FloatingPointError, ZeroDivisionError) as exc:
            failures[i] = str(exc)
    return DirichletRows(
        readonly(values),
        readonly(np.zeros(n)),
        None if score is None else readonly(score),
        None if hessian is None else readonly(hessian),
        readonly(valid, dtype=bool),
        tuple(failures),
    )


def normalized_mean(theta):
    alpha = np.asarray(theta, dtype=np.float64)
    out = np.empty_like(alpha)
    for i, row in enumerate(alpha):
        try:
            total = _sum(row)
        except OverflowError as exc:
            raise ValueError("Dirichlet concentration sum is outside float64 range") from exc
        out[i] = row / total
    if np.any(out <= 0) or not np.isfinite(out).all():
        raise ValueError("Dirichlet positive mean share is outside float64 range")
    return readonly(out)


def dirichlet_curvature_direction(
    response, theta, multiplier, link_derivatives, first, second=None, *, log_shares
):
    """Observed predictor-curvature directions, using separable links to order four."""
    double = second is not None
    second = np.zeros_like(first) if second is None else second
    derivatives = tuple(link_derivatives)
    in_range = (
        np.all(theta >= _FAST_MIN)
        and np.all(theta <= _FAST_MAX)
        and np.all(multiplier <= _FAST_MAX)
        and all(np.all(np.abs(v) <= _FAST_MAX) for v in (*link_derivatives, first, second))
    )
    result = (
        _curvature_batch(
            response, theta, multiplier, derivatives, first, second, double, log_shares
        )
        if in_range
        else None
    )
    if result is None or not np.isfinite(result).all():
        # Retain the Python floating-point exceptions and refusal semantics.
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            result = _curvature_python(
                response, theta, multiplier, derivatives, first, second, double, log_shares
            )
        if not np.isfinite(result).all():
            raise ValueError("Dirichlet optional curvature direction is outside float64 range")
    return readonly(result)
