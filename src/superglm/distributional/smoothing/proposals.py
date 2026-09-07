"""Smoothing-parameter proposals: bound snapping, trust scaling and acceleration."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from superglm.distributional.result import DenseSolverResult, DistributionalEFSConfig


def _ordered_log_lambdas(
    lambdas: Mapping[str, float],
    names: tuple[str, ...],
) -> NDArray[np.float64]:
    return np.array([math.log(lambdas[name]) for name in names], dtype=np.float64)


def _ordered_steps(
    steps: Mapping[str, float],
    names: tuple[str, ...],
) -> NDArray[np.float64]:
    return np.array([steps[name] for name in names], dtype=np.float64)


def _named_steps(
    names: tuple[str, ...],
    values: NDArray,
) -> dict[str, float]:
    return dict(zip(names, (float(value) for value in values), strict=True))


def _acceleration_provenance(
    estimated_names: tuple[str, ...],
    fit: DenseSolverResult,
) -> tuple[object, ...]:
    rank = fit.terminal_rank if fit.coefficient_face is None else fit.terminal_reduced_rank
    if rank is None:
        raise RuntimeError("a coefficient-face fit requires reduced-rank provenance")
    return (
        estimated_names,
        fit.family_likelihood_plan_identifier,
        fit.execution_backend_identifier,
        fit.terminal_curvature.requested_source,
        fit.terminal_curvature.actual_source,
        rank.policy_version,
        rank.method,
        rank.rank,
        tuple(int(index) for index in rank.active_columns),
    )


def _snap_to_bounds(value: float, config: DistributionalEFSConfig) -> float:
    """Return the exact configured bound when ``value`` is within round-off of it.

    Bound detection elsewhere is an exact float comparison against
    ``config.maximum_lambda`` / ``config.minimum_lambda``; a proposal that
    lands one ulp inside the bound would otherwise be invisible to it.
    """
    eps = np.finfo(np.float64).eps
    if value >= config.maximum_lambda * (1.0 - 8.0 * eps):
        return float(config.maximum_lambda)
    if value <= config.minimum_lambda * (1.0 + 8.0 * eps):
        return float(config.minimum_lambda)
    return float(value)


def _scaled_proposal(
    current: Mapping[str, float],
    steps: Mapping[str, float],
    estimated_names: tuple[str, ...],
    scale: float,
    config: DistributionalEFSConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    estimated = set(estimated_names)
    lambdas: dict[str, float] = {}
    log_steps: dict[str, float] = {}
    for name, old_value in current.items():
        if name not in estimated:
            lambdas[name] = old_value
            log_steps[name] = 0.0
            continue
        step = float(scale * steps[name])
        proposed = old_value * math.exp(step)
        proposed = _snap_to_bounds(
            float(np.clip(proposed, config.minimum_lambda, config.maximum_lambda)),
            config,
        )
        lambdas[name] = proposed
        log_steps[name] = math.log(proposed) - math.log(old_value)
    return lambdas, log_steps


def _accelerated_proposal(
    current: Mapping[str, float],
    estimated_names: tuple[str, ...],
    log_lambdas: NDArray,
    log_steps: NDArray,
    config: DistributionalEFSConfig,
) -> tuple[dict[str, float], dict[str, float]] | None:
    named_log_lambdas = _named_steps(estimated_names, log_lambdas)
    named_log_steps = _named_steps(estimated_names, log_steps)
    minimum_log_lambda = math.log(config.minimum_lambda)
    maximum_log_lambda = math.log(config.maximum_lambda)
    lambdas: dict[str, float] = {}
    steps: dict[str, float] = {}
    for name, old_value in current.items():
        if name not in named_log_lambdas:
            lambdas[name] = old_value
            steps[name] = 0.0
            continue
        log_value = named_log_lambdas[name]
        if log_value == minimum_log_lambda:
            proposed = config.minimum_lambda
        elif log_value == maximum_log_lambda:
            proposed = config.maximum_lambda
        elif not minimum_log_lambda < log_value < maximum_log_lambda:
            return None
        else:
            proposed = math.exp(log_value)
            if (
                not math.isfinite(proposed)
                or proposed < config.minimum_lambda
                or proposed > config.maximum_lambda
            ):
                return None
            proposed = _snap_to_bounds(proposed, config)
        lambdas[name] = proposed
        steps[name] = named_log_steps[name]
    return lambdas, steps
