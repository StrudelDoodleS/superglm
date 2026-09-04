"""Warm-started Richardson difference of the negative LAML in one log lambda.

Refit at ``rho_k +- h`` from the base fit's coefficients with a tight inner
tolerance, take the central difference of ``joint_laplace_objective`` on a step
ladder, and report the order-4 Richardson estimate of the finest pair with the
difference between the two finest central differences as its resolution.
A probe whose rank method, rank or curvature source differs from the base
fit is refused: the objective is not a smooth function of lambda across such
a seam and the difference would be noise.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from superglm.distributional.smoothing.objective import joint_laplace_objective
from superglm.distributional.solver.solver import fit_dense_fixed_lambda


@dataclass(frozen=True)
class OracleGradient:
    value: float
    resolution: float


def probe_fit(family, layout, y, plan, lambdas, base_fit, config):
    """Warm-started tight refit at ``lambdas`` with the base fit's provenance."""
    penalty = layout.penalty_matrix(lambdas)
    fit = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        plan,
        penalty,
        initial=base_fit.coefficients,
        config=replace(config, tolerance=1e-11, max_iterations=500),
    )
    if not fit.converged:
        raise RuntimeError("oracle probe did not converge")
    same = (
        fit.terminal_rank.method == base_fit.terminal_rank.method
        and fit.terminal_rank.rank == base_fit.terminal_rank.rank
        and fit.terminal_curvature.actual_source == base_fit.terminal_curvature.actual_source
    )
    if not same:
        raise RuntimeError("oracle probe changed provenance; the objective is not smooth here")
    return fit


def _objective_at(family, layout, y, plan, lambdas, base_fit, config) -> float:
    fit = probe_fit(family, layout, y, plan, lambdas, base_fit, config)
    return joint_laplace_objective(fit, layout=layout, lambdas=lambdas)


def oracle_gradient(
    family,
    layout,
    y,
    plan,
    lambdas: Mapping[str, float],
    base_fit,
    config,
    name: str,
    *,
    steps=(2e-2, 1e-2, 5e-3),
) -> OracleGradient:
    def central(h: float) -> float:
        up = dict(lambdas)
        up[name] = lambdas[name] * math.exp(h)
        down = dict(lambdas)
        down[name] = lambdas[name] * math.exp(-h)
        return (
            _objective_at(family, layout, y, plan, up, base_fit, config)
            - _objective_at(family, layout, y, plan, down, base_fit, config)
        ) / (2.0 * h)

    g = [central(h) for h in steps]
    richardson = (4.0 * g[-1] - g[-2]) / 3.0
    resolution = abs(g[-1] - g[-2])
    return OracleGradient(value=float(richardson), resolution=float(resolution))
