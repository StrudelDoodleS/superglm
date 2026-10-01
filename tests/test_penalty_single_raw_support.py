"""A single dense penalty is selected once per fit, on its raw basis.

Discrete REML and EFS rebuild each smooth's SSP map ``C = R_inv`` whenever
lambda moves. Because log pdet(C.T Omega C) = log pdet(Omega) +
log det(U.T C C.T U) for an orthonormal range basis U of Omega, the raw support
and its unit-weight summary are lambda-free and transfer between REML steps;
only the certified coordinate volume follows each map.
"""

import math
from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Spline, SuperGLM
from superglm.reml import multi_penalty as kernel
from superglm.reml import penalty_algebra as algebra
from superglm.reml import penalty_support as support_module
from superglm.reml.objective import reml_laml_objective
from superglm.solvers.dispersion import model_weight_semantics


def _data():
    rng = np.random.default_rng(11)
    n = 3000
    x1, x2, x3 = rng.uniform(size=(3, n))
    level = rng.choice(np.array(["a", "b", "c"]), size=n)
    y = rng.poisson(np.exp(-0.5 + np.sin(2 * np.pi * x1) + 0.5 * x2**2 + 0.3 * (level == "b")))
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "level": level}), y


def _fit(discrete):
    frame, y = _data()
    features = {
        "x1": Spline(kind="ps", k=10),
        "x2": Spline(kind="ps", k=10),
        "x3": Spline(kind="ps", k=8),
        "level": Categorical(),
    }
    return SuperGLM(family="poisson", discrete=discrete, features=features).fit_reml(frame, y)


def _solver_space(component):
    """The same component on its solver-space support, as before the raw path."""
    copied = replace(component)
    algebra._attach_context_geometry([copied])
    return copied


@pytest.mark.parametrize("discrete", [True, False])
def test_fit_builds_each_penalty_support_once(monkeypatch, discrete):
    counts = {"support": 0, "unit": 0}
    build_support = support_module._penalty_support
    summarize = kernel._evaluate_penalty_summary

    def counted_support(*args, **kwargs):
        counts["support"] += 1
        return build_support(*args, **kwargs)

    def counted_summary(support, values, *args, **kwargs):
        counts["unit"] += bool(np.all(np.asarray(values) == 1.0))
        return summarize(support, values, *args, **kwargs)

    monkeypatch.setattr(support_module, "_penalty_support", counted_support)
    monkeypatch.setattr(kernel, "_evaluate_penalty_summary", counted_summary)
    model = _fit(discrete)
    groups = {component.group_name for component in model._reml_penalties}
    assert len(groups) == 3
    # Every REML step rebuilds the SSP maps; the raw selection must not follow.
    assert counts["support"] <= len(groups)
    assert counts["unit"] <= len(groups)
    assert all(
        algebra._context_geometry([component]).coordinate_map is not None
        for component in model._reml_penalties
    )


def test_raw_support_logdet_matches_the_solver_space_support():
    model = _fit(discrete=True)
    for component in model._reml_penalties:
        weights = {component.name: model._reml_lambdas[component.name]}
        geometry = algebra._context_geometry([component])
        assert geometry.coordinate_map is not None and geometry.volume[0] != 0.0
        raw = algebra._compute_penalty_logdet_evaluation(weights, [component])
        reference = _solver_space(component)
        assert algebra._context_geometry([reference]).coordinate_map is None
        solver = algebra._compute_penalty_logdet_evaluation(weights, [reference])
        assert raw.rank == solver.rank == component.rank
        assert raw.gradient == solver.gradient
        assert raw.hessian == solver.hessian
        # Both evaluations certify their own representative of the same
        # penalty, so they agree to within the sum of their certificates.
        assert abs(raw.logdet - solver.logdet) <= raw.logdet_error + solver.logdet_error


def _decimal_log_pdet(root, basis, coordinate_map):
    """ln det(Y Y.T) for Y = root P C, with P the exact projector onto range(basis)."""
    cast = [[[Decimal.from_float(float(v)) for v in row] for row in m] for m in (root, basis)]
    root, basis = cast
    mapping = [[Decimal.from_float(float(v)) for v in row] for row in coordinate_map]
    width, rank = len(basis), len(basis[0])
    gram = [[sum(row[i] * row[j] for row in basis) for j in range(rank)] for i in range(rank)]
    # Solve gram @ coefficients = basis.T by Gauss-Jordan, then P = basis @ coefficients.
    augmented = [gram[i] + [basis[k][i] for k in range(width)] for i in range(rank)]
    for column in range(rank):
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(rank):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [
                    a - factor * b for a, b in zip(augmented[row], augmented[column], strict=True)
                ]
    coefficients = [row[rank:] for row in augmented]
    projector = [
        [sum(basis[i][a] * coefficients[a][j] for a in range(rank)) for j in range(width)]
        for i in range(width)
    ]
    projected = [
        [sum(row[k] * projector[k][j] for k in range(width)) for j in range(width)] for row in root
    ]
    mapped = [
        [sum(row[k] * mapping[k][j] for k in range(width)) for j in range(len(mapping[0]))]
        for row in projected
    ]
    work = [[sum(a * b for a, b in zip(u, v, strict=True)) for v in mapped] for u in mapped]
    determinant = Decimal(1)
    for column in range(len(work)):
        pivot = work[column][column]
        determinant *= pivot
        for row in range(column + 1, len(work)):
            factor = work[row][column] / pivot
            for j in range(column + 1, len(work)):
                work[row][j] -= factor * work[column][j]
    return determinant.ln()


def test_projected_map_logdet_is_certified_against_exact_arithmetic():
    # A centred spline's map drops one raw coordinate: C is 7 x 6, not square.
    omega = np.diff(np.eye(7), n=2, axis=0)
    omega = omega.T @ omega
    coordinate_map = np.random.default_rng(5).normal(size=(7, 6))
    group = SimpleNamespace(name="s", sl=slice(0, 6), size=6)
    matrix = SimpleNamespace(R_inv=coordinate_map, omega=omega)
    components, _, _ = algebra.build_penalty_context([matrix], [(0, group)])
    geometry = algebra._context_geometry(components)
    assert geometry.coordinate_map.shape == (7, 6) and geometry.support.rank == 5
    evaluation = algebra._compute_penalty_logdet_evaluation({"s": 3.0}, components)
    assert evaluation.rank == 5
    with localcontext() as context:
        context.prec = 100
        root = geometry.support.component_roots[0]
        expected = _decimal_log_pdet(root, geometry.support.Q_plus, coordinate_map)
        expected += 5 * Decimal(3).ln()
        error = abs(Decimal.from_float(evaluation.logdet) - expected)
        assert error <= Decimal.from_float(float(evaluation.logdet_error))


def test_reml_objective_moves_only_within_the_logdet_certificates():
    """At one fitted state, the two supports change the objective only through log|S|+.

    Ranks and log-lambda derivatives are equal, so a REML step can differ only
    where it compares objective values; this bounds that difference.
    """
    model = _fit(discrete=False)
    _, y = _data()
    penalties = list(model._reml_penalties)
    solver_space = [_solver_space(component) for component in penalties]
    lambdas = model._reml_lambdas
    evaluations, logdets = [], []
    for reml_penalties in (penalties, solver_space):
        evaluations.append(
            reml_laml_objective(
                model._dm,
                model._distribution,
                model._link,
                model._groups,
                y,
                model.result,
                lambdas,
                np.ones(len(y)),
                np.zeros(len(y)),
                log_det_H=model.result.log_det_H,
                hessian_rank=model.result.reml_hessian_rank,
                reml_penalties=reml_penalties,
                return_evaluation=True,
                weight_semantics=model_weight_semantics(model),
            )
        )
        logdets.append(algebra._compute_penalty_logdet_evaluation(lambdas, reml_penalties))
    raw, solver = evaluations
    assert raw.penalty_quad == solver.penalty_quad
    assert raw.penalty_nullity == solver.penalty_nullity
    assert logdets[0].rank == logdets[1].rank
    assert logdets[0].gradient == logdets[1].gradient
    assert logdets[0].hessian == logdets[1].hessian
    logdet_bound = logdets[0].logdet_error + logdets[1].logdet_error
    assert abs(logdets[0].logdet - logdets[1].logdet) <= logdet_bound
    # value = nll + 0.5 * ((pq + log|H|) - log|S|+), with nll, pq and log|H|
    # shared. Each rounded operation errs by at most u|result| (Higham 2002,
    # section 2.2), and halving is exact.
    unit = np.finfo(float).eps / 2
    shared = abs(raw.penalty_quad + model.result.log_det_H)
    values = [raw.value, solver.value]
    bound = (
        0.5 * logdet_bound
        + 0.5 * unit * math.fsum(shared + abs(item.logdet) for item in logdets)
        + unit * math.fsum(map(abs, values)) / (1 - unit)
    )
    assert abs(values[0] - values[1]) <= bound
