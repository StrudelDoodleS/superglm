"""One predictor addition order across chunked fitting and publication paths."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.model import _predict_one_eta
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import chunks
from superglm.features import Categorical, Numeric, Spline
from superglm.group_matrix import CategoricalGroupMatrix

from ._distributional_weights import resolved_prior
from ._exact_reference import exact_sum


def _predictor(name, mode):
    features = {}
    if mode in ("mixed", "numeric", "numeric_spline"):
        features["x"] = Numeric()
    if mode in ("mixed", "support", "support_category", "numeric_spline"):
        features["s"] = Spline(kind="cr", k=4)
    if mode in ("mixed", "category", "support_category"):
        features["g"] = Categorical()
    # Different intercept choices give rectangular curvature blocks without
    # relying on tiny, roundoff-sensitive raw spline values for dispatch.
    return Predictor(
        name,
        features,
        interactions=(("s", "g"),) if mode == "mixed" else (),
        intercept=not (name == "scale" and mode == "mixed"),
    )


@pytest.fixture(scope="module")
def mixed_problem():
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
    indices = np.tile(np.repeat(np.arange(37, dtype=np.intp), 2), 3)
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
    return SimpleNamespace(
        family=family,
        layout=layout,
        y=y,
        plan=plan,
        compiled=compiled,
        probe=as_eager_frame(small.iloc[indices[:7]].reset_index(drop=True)),
    )


@pytest.mark.parametrize("route", ["value", "terminal", "public"])
def test_cancellation_uses_the_same_chunk_state_for_every_route(mixed_problem, route):
    problem = mixed_problem
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
    )
    try:
        chunk = next(stream)
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
        )
        exact = exact_sum(operands)
        operations = len(operands) + 1
        epsilon = np.finfo(np.float64).eps
        bound = operations * epsilon / (1 - operations * epsilon) * np.abs(operands).sum(axis=0)
        assert np.all(np.abs(actual[:, 0] - exact) <= bound)
    finally:
        stream.close()
