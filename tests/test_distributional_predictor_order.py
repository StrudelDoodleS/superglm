"""One predictor addition order across admitted fitting and publication paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional._global_moment_policy import automatic_global_moment_budget
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.model import _predict_one_eta
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import chunks, fit_dense_fixed_lambda
from superglm.distributional.solver._global_moments import build_global_moment_plan
from superglm.group_matrix import CategoricalGroupMatrix

from ._distributional_weights import resolved_prior
from .test_distributional_automatic_panels import _predictor


@pytest.fixture(scope="module")
def admitted_problem():
    # Real rows cross the unmodified automatic threshold; paired responses
    # keep the complete-fit comparison well conditioned and inexpensive.
    small = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 37),
            "s": np.cos(np.arange(37) * 0.7),
            "g": np.resize(list("abc"), 37),
        }
    )
    family = GaussianLS()
    compiled = compile_predictors(
        as_eager_frame(small),
        resolved_prior(np.ones(37)),
        family.parameters,
        (_predictor("location", "mixed"), _predictor("scale", "intercept")),
        model_discrete=True,
        n_bins_config=9,
    )
    layout = build_stacked_layout(compiled)
    indices = np.tile(np.repeat(np.arange(37, dtype=np.intp), 2), 3543)
    n = len(indices)
    layout = replace(
        layout,
        predictors=tuple(
            replace(
                state,
                design=state.design.row_subset(indices),
                offset=np.full(n, 0.125 if state.parameter_index == 0 else 0.0),
            )
            for state in layout.predictors
        ),
    )
    y = 0.125 + np.tile([-1.0, 1.0], n // 2)
    plan = family.bind_likelihood(y, resolved_prior(np.ones(n)), COMPLETE_OBSERVATION)
    assert automatic_global_moment_budget(family, plan, layout) == 64 << 20
    return SimpleNamespace(
        family=family,
        layout=layout,
        y=y,
        plan=plan,
        compiled=compiled,
        probe=as_eager_frame(small.iloc[indices[:7]].reset_index(drop=True)),
    )


@pytest.mark.parametrize("route", ["value", "terminal", "public"])
def test_cancellation_uses_the_same_admitted_state_for_every_route(admitted_problem, route):
    problem = admitted_problem
    layout = problem.layout
    beta = np.zeros(layout.n_coefficients)
    beta[0] = 2.0**53
    beta[1] = 1.0
    column = 1
    for group in layout.predictors[0].design.group_matrices:
        if type(group) is CategoricalGroupMatrix:
            beta[column : column + group.shape[1]] = -(2.0**53)
        column += group.shape[1]
    stream = chunks.iter_likelihood_chunks(
        problem.family,
        layout,
        problem.y,
        problem.plan,
        beta,
        chunk_size=7,
        curvature_source="observed",
        _range_geometry=True,
    )
    try:
        chunk = next(stream)
        built = build_global_moment_plan(layout, byte_budget=64 << 20, chunk_size=7)
        assert built.plan is not None, built.reason
        plan = built.plan
        try:
            plan.reset(coefficients=beta, penalty=np.zeros((len(beta), len(beta))))
            plan.add_row_range(chunk.plans, 0, 7, chunk.score_eta, chunk.curvature_packed)
            assert plan.stats["rows"] == 7
            assert plan.stats["batched_moment_calls"] == 1
        finally:
            plan.close()
        if route == "value":
            actual = chunks._predictor_values(layout, beta, chunk.rows, include_offsets=True)
        elif route == "terminal":
            actual, _ = chunks.materialize_terminal_predictions(layout, beta, chunk_size=8192)
            actual = actual[:7]
        else:
            actual = np.column_stack(
                [
                    _predict_one_eta(
                        problem.probe,
                        predictor,
                        layout,
                        SimpleNamespace(coefficients=beta),
                        layout.predictors[predictor.parameter_index].offset[:7],
                    )
                    for predictor in problem.compiled
                ]
            )
        np.testing.assert_array_equal(actual, chunk.eta)
        grouped, _ = chunks._predictor_chunk(layout, beta, chunk.rows, include_offsets=True)
        np.testing.assert_array_equal(actual, grouped)
        theta = chunks._theta_chunk(layout, actual)
        np.testing.assert_array_equal(theta, chunk.theta)
        natural = problem.family.evaluate_natural(
            problem.y[:7], theta, problem.plan.take(chunk.rows.indices), derivative_order=0
        )
        np.testing.assert_array_equal(
            natural.optimizing_log_likelihood, chunk.optimizing_log_likelihood
        )
        # Agreement of computational states is distinct from forward accuracy
        # under cancellation. Compare the ordered sum to the exact stored row
        # model with the standard absolute-operand accumulation bound.
        contributions = []
        column = 1
        for group in layout.predictors[0].design.group_matrices:
            width = group.shape[1]
            contributions.append(
                group.row_subset(np.arange(7)).matvec(beta[column : column + width])
            )
            column += width
        operands = np.vstack(
            [
                np.full(7, beta[0]),
                *contributions,
                layout.predictors[0].offset[:7],
            ]
        ).astype(np.longdouble)
        exact = operands.sum(axis=0, dtype=np.longdouble)
        operations = len(operands) + 1
        epsilon = np.finfo(np.float64).eps
        bound = operations * epsilon / (1 - operations * epsilon) * np.abs(operands).sum(axis=0)
        assert np.all(np.abs(actual[:, 0] - exact) <= bound)
    finally:
        stream.close()


def test_complete_admitted_fit_matches_grouped_stationarity(admitted_problem, monkeypatch):
    problem = admitted_problem
    q = problem.layout.n_coefficients
    initial = np.zeros(q)
    initial[0] = 0.01
    initial[-1] = 0.05
    penalty = np.eye(q)
    finished = []
    original = chunks.build_global_moment_plan

    def tracked(*args, **kwargs):
        built = original(*args, **kwargs)
        assert built.plan is not None, built.reason
        finish = built.plan.finish

        def record():
            result = finish()
            finished.append(built.plan.stats["rows"])
            return result

        built.plan.finish = record
        return built

    def fit():
        return fit_dense_fixed_lambda(
            problem.family,
            problem.layout,
            problem.y,
            problem.plan,
            penalty,
            initial=initial,
            chunk_size=8192,
            config=DenseSolverConfig(tolerance=1e-7, max_iterations=12),
        )

    monkeypatch.setattr(chunks, "build_global_moment_plan", tracked)
    actual = fit()
    assert finished and set(finished) == {len(problem.y)}
    monkeypatch.setattr(chunks, "automatic_global_moment_budget", lambda *args: None)
    monkeypatch.setattr(chunks, "automatic_small_group_panel_budget", lambda *args: None)
    grouped = fit()
    assert actual.converged and grouped.converged
    gap = min(
        np.linalg.eigvalsh(result.terminal_penalized_curvature)[0] for result in (actual, grouped)
    )
    assert gap > 0
    epsilon = np.finfo(np.float64).eps
    arithmetic = 128 * len(problem.y) * q * epsilon
    bound = 4 * sum(np.linalg.norm(result.terminal_score) for result in (actual, grouped)) / gap
    bound += arithmetic / (1 - arithmetic) * (1 + np.linalg.norm(initial))
    assert np.linalg.norm(actual.coefficients - grouped.coefficients) <= bound
    # These 74 rows contain every repeated design row. Propagate coefficient
    # error through their actual predictor Jacobian without materializing N×q.
    row_norm = 0.0
    for state in problem.layout.predictors:
        pieces = [np.ones((74, 1))] if state.intercept_index is not None else []
        pieces.extend(
            group.row_subset(np.arange(74)).toarray() for group in state.design.group_matrices
        )
        row_norm = max(row_norm, float(np.max(np.linalg.norm(np.column_stack(pieces), axis=1))))
    np.testing.assert_allclose(actual.eta, grouped.eta, rtol=0, atol=row_norm * bound)
