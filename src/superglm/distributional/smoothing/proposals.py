"""Smoothing-parameter proposals: bound snapping, trust scaling and acceleration."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from superglm._roundoff import EPS, UNIT_ROUNDOFF
from superglm.distributional.results.iteration import DistributionalEFSConfig
from superglm.distributional.results.solver import DenseSolverResult
from superglm.reml.convergence import flat_tail_certified
from superglm.reml.efs_update import EFSComponentState, EFSUpdateResult

# The largest step another component may propose in the round of an endpoint trial.
# The certificate holds the other smoothing parameters fixed, so the trial waits for
# them to settle.
ENDPOINT_GATE_LOG_STEP = 1.0


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


def _snap_to_bounds(value: float, config: DistributionalEFSConfig, name: str) -> float:
    """Return component ``name``'s exact bound when ``value`` is within round-off of it.

    Bound detection elsewhere is an exact float comparison against
    ``config.upper(name)`` / ``config.lower(name)``; a proposal that lands one
    ulp inside the bound would otherwise be invisible to it.
    """
    eps = np.finfo(np.float64).eps
    upper, lower = config.upper(name), config.lower(name)
    if value >= upper * (1.0 - 8.0 * eps):
        return float(upper)
    if value <= lower * (1.0 + 8.0 * eps):
        return float(lower)
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
        lower, upper = config.lower(name), config.upper(name)
        log_old = math.log(old_value)
        # A step that reaches a bound is assigned the bound. Its distance is formed as
        # the returned log step is, so a bound-limited step compares equal to it;
        # ``old * exp(step)`` rounds through a log and an exp and can land further from
        # the bound than the snap reaches, where the exact comparisons that detect a
        # bound miss it.
        if step >= math.log(upper) - log_old:
            proposed = upper
        elif step <= math.log(lower) - log_old:
            proposed = lower
        else:
            proposed = _snap_to_bounds(
                float(np.clip(old_value * math.exp(step), lower, upper)), config, name
            )
        lambdas[name] = proposed
        log_steps[name] = math.log(proposed) - log_old
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
    lambdas: dict[str, float] = {}
    steps: dict[str, float] = {}
    for name, old_value in current.items():
        if name not in named_log_lambdas:
            lambdas[name] = old_value
            steps[name] = 0.0
            continue
        log_value = named_log_lambdas[name]
        lower, upper = config.lower(name), config.upper(name)
        minimum_log_lambda, maximum_log_lambda = math.log(lower), math.log(upper)
        if log_value == minimum_log_lambda:
            proposed = lower
        elif log_value == maximum_log_lambda:
            proposed = upper
        elif not minimum_log_lambda < log_value < maximum_log_lambda:
            return None
        else:
            proposed = math.exp(log_value)
            if not math.isfinite(proposed) or proposed < lower or proposed > upper:
                return None
            proposed = _snap_to_bounds(proposed, config, name)
        lambdas[name] = proposed
        steps[name] = named_log_steps[name]
    return lambdas, steps


def flat_tail_names(
    components: tuple[EFSComponentState, ...],
    blocks: Sequence[tuple[str, slice]],
    update: EFSUpdateResult,
    inverse: NDArray[np.float64],
    config: DistributionalEFSConfig,
) -> tuple[str, ...]:
    """Estimated penalties whose working problem falls all the way to their upper bound.

    A component qualifies below its bound, alone on its coefficient block
    (``blocks`` lists every penalty's, those on an exact face included), and
    certified by ``flat_tail_certified`` with E = rank - lambda tr(H^-1 S) and
    B = lambda beta'S beta. The certificate's premise is nonnegative data
    curvature on the block, which an observed Hessian need not have. With D
    that curvature profiled onto the block, ``(H^-1)_bb = (D + lambda S)^-1``
    and ``lambda S^(1/2) (H^-1)_bb S^(1/2)`` has eigenvalues lambda/(d + lambda)
    on S's range: none above one says every d >= 0. An eigenvalue above one by
    theta moves the certificate's bound on E by a term of order theta^2, so
    theta = sqrt(u) keeps that term at rounding level while standing far above
    the rounding in the eigenvalue itself.
    """
    names: list[str] = []
    for component in components:
        name, lam, block = component.name, component.lambda_value, component.coefficient_slice
        if component.policy.mode != "estimate" or lam >= config.upper(name):
            continue
        if any(
            other != name and block.start < taken.stop and taken.start < block.stop
            for other, taken in blocks
        ):
            continue
        edf = component.rank - lam * update.trace_terms[name]
        if not flat_tail_certified(edf, lam * update.quadratic_forms[name])[0]:
            continue
        values, vectors = np.linalg.eigh(component.penalty)
        retained = values > 64.0 * len(values) * EPS * values[-1]
        root = vectors[:, retained] * np.sqrt(values[retained])
        spectrum = np.linalg.eigvalsh(lam * (root.T @ inverse[block, block] @ root))
        if np.max(spectrum, initial=0.0) <= 1.0 + math.sqrt(UNIT_ROUNDOFF):
            names.append(name)
    return tuple(names)


def flat_tail_fall(component: EFSComponentState, update: EFSUpdateResult) -> float:
    """The most a flat tail's working problem falls on the way to its upper bound.

    With E and B as ``flat_tail_names`` forms them: in S's eigen-directions E
    is a sum of e_i = d_i / (lambda + d_i) and B a sum of B_i >= 0, and the
    working criterion stands (-sum log(1 - e_i) - sum B_i / (1 - e_i)) / 2 above
    its value at zero variance. That is at most (-log(1 - E) - B) / 2, since
    -log(1 - x) is superadditive and no 1 - e_i exceeds one, and the
    certificate keeps it positive.
    """
    name, lam = component.name, component.lambda_value
    edf = component.rank - lam * update.trace_terms[name]
    return 0.5 * (-math.log1p(-edf) - lam * update.quadratic_forms[name])


def resolved_flat_tails(
    flat_tail: Mapping[str, float],
    latched: Collection[str],
    resolution: float,
    arriving: Collection[str] = (),
) -> tuple[str, ...]:
    """The untried flat tails, where a trial of them is worth its face assessment.

    ``flat_tail`` gives each component's largest fall on the way to its bound
    (``flat_tail_fall``). One trial takes them all, so they are judged as one:
    a set worth its trial takes its small members along, which would otherwise
    reach their bounds later, one face at a time. A fall within ``resolution``
    is no reason for a trial, whose bounds each cost a face assessment: mgcv's
    Fellner--Schall iteration likewise counts a REML change under its
    ``efs.tol`` as negligible (the ``gam.control`` help page). Nor is it one to
    hold a tail back that the step itself puts on its bound (``arriving``): the
    assessment is then owed, and the trial holds that bound until the others stop.
    """
    untried = {name: fall for name, fall in flat_tail.items() if name not in latched}
    owed = not untried.keys().isdisjoint(arriving)
    return tuple(untried) if owed or sum(untried.values()) > resolution else ()


def endpoint_trial_names(
    flat_tail: Collection[str],
    log_steps: Mapping[str, float],
    estimated_names: tuple[str, ...],
    latched: Collection[str],
) -> tuple[str, ...]:
    """The flat-tail components to try at their upper bounds beside the step ``log_steps``.

    A component tried before is ``latched`` to ordinary steps. No trial is made
    while another estimated component proposes a step beyond
    ``ENDPOINT_GATE_LOG_STEP``.
    """
    names = tuple(name for name in flat_tail if name not in latched)
    others = (name for name in estimated_names if name not in names)
    if any(abs(log_steps[name]) > ENDPOINT_GATE_LOG_STEP for name in others):
        return ()
    return names


def _endpoint_proposal(
    current: Mapping[str, float],
    steps: Mapping[str, float],
    estimated_names: tuple[str, ...],
    endpoint_names: tuple[str, ...],
    config: DistributionalEFSConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    """The plain proposal with ``endpoint_names`` at their own upper bounds."""
    lambdas, log_steps = _scaled_proposal(current, steps, estimated_names, 1.0, config)
    for name in endpoint_names:
        lambdas[name] = config.upper(name)
        log_steps[name] = math.log(lambdas[name]) - math.log(current[name])
    return lambdas, log_steps
