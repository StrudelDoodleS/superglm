"""Value-only chunk work and numerical correctness are separate contracts."""

from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.chunks as chunking
from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.features import Numeric, Spline
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior


def _problem(*, discrete=True, scale_intercept=True):
    n = 23
    x = np.linspace(-0.9, 1.1, n)
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": np.sin(2.3 * x)}))
    family = GaussianLS(scale_floor=0.02)
    weights = resolved_prior(np.linspace(0.6, 1.8, n))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "location",
                    {"x": Numeric(), "z": Spline(kind="cr", n_knots=4, discrete=discrete)},
                ),
                Predictor("scale", {"x": Numeric()}, intercept=scale_intercept),
            ),
            offsets={"location": 0.03 * x**2, "scale": -0.1 + 0.02 * x},
            model_discrete=discrete,
            n_bins_config=7,
        )
    )
    coefficients = np.linspace(-0.2, 0.3, layout.n_coefficients)
    response = 0.4 + np.cos(x)
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    return family, layout, response, plan, coefficients


def _evaluate(operation, problem, chunk_size):
    family, layout, response, plan, coefficients = problem
    if operation == "likelihood":
        return chunking.evaluate_chunked_log_likelihood(
            family, layout, response, plan, coefficients, chunk_size=chunk_size
        )
    if operation == "change":
        return chunking.maximum_chunked_predictor_change(
            layout, coefficients, chunk_size=chunk_size
        )
    return chunking.materialize_terminal_predictions(layout, coefficients, chunk_size=chunk_size)


@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal"])
def test_value_only_passes_do_not_prepare_geometry(monkeypatch, operation):
    """Reintroducing chunk DesignMatrix/geometry construction breaks this witness."""
    problem = _problem()
    design_subsets = 0
    execution_plans = 0
    original_subset = DesignMatrix.row_subset
    original_plan = chunking.PredictorExecutionPlan

    def counted_subset(self, indices):
        nonlocal design_subsets
        design_subsets += 1
        return original_subset(self, indices)

    def counted_plan(*args, **kwargs):
        nonlocal execution_plans
        execution_plans += 1
        return original_plan(*args, **kwargs)

    monkeypatch.setattr(DesignMatrix, "row_subset", counted_subset)
    monkeypatch.setattr(chunking, "PredictorExecutionPlan", counted_plan)
    _evaluate(operation, problem, 7)
    assert (design_subsets, execution_plans) == (0, 0)


@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal"])
def test_value_only_group_subsets_are_bounded_and_not_retained(monkeypatch, operation):
    problem = _problem()
    references = []
    largest_subset = 0
    most_live = 0
    group_types = {
        type(group) for state in problem[1].predictors for group in state.design.group_matrices
    }

    def tracked_subset(original):
        def subset(self, indices):
            nonlocal largest_subset, most_live
            result = original(self, indices)
            largest_subset = max(largest_subset, result.shape[0])
            storage = result.bin_idx if hasattr(result, "bin_idx") else result.M
            references.append(weakref.ref(storage))
            most_live = max(most_live, sum(ref() is not None for ref in references))
            return result

        return subset

    for group_type in group_types:
        monkeypatch.setattr(group_type, "row_subset", tracked_subset(group_type.row_subset))
    _evaluate(operation, problem, 7)
    assert largest_subset == 7
    assert most_live <= 2
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("chunk_size", [1, 7, 23, 31])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("scale_intercept", [False, True])
def test_chunk_values_match_stored_design_with_offsets(chunk_size, discrete, scale_intercept):
    problem = _problem(discrete=discrete, scale_intercept=scale_intercept)
    family, layout, response, plan, coefficients = problem
    expected_change = np.empty((len(response), len(layout.predictors)))
    expected_eta = np.empty_like(expected_change)
    expected_theta = np.empty_like(expected_change)
    for state in layout.predictors:
        local = coefficients[state.coefficient_slice]
        intercept = state.intercept_index is not None
        # The stored design includes the actual discretised training basis.
        values = state.design.toarray() @ local[int(intercept) :]
        if intercept:
            values += local[0]
        expected_change[:, state.parameter_index] = values
        expected_eta[:, state.parameter_index] = values + state.offset
        expected_theta[:, state.parameter_index] = state.link.inverse(values + state.offset)
    eta, theta = _evaluate("terminal", problem, chunk_size)
    # Dot products and chunk reductions have dimension-scaled roundoff bounds.
    tolerance = 16 * max(len(response), layout.n_coefficients) * np.finfo(float).eps
    np.testing.assert_allclose(eta, expected_eta, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(theta, expected_theta, rtol=tolerance, atol=tolerance)
    assert not eta.flags.writeable
    assert not theta.flags.writeable
    np.testing.assert_allclose(
        _evaluate("change", problem, chunk_size),
        np.max(np.abs(expected_change)),
        rtol=tolerance,
        atol=tolerance,
    )
    expected = family.evaluate_natural(response, expected_theta, plan, derivative_order=0)
    actual = _evaluate("likelihood", problem, chunk_size)
    np.testing.assert_allclose(
        actual.optimizing_log_likelihood,
        np.sum(expected.optimizing_log_likelihood),
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.parameter_independent_carrier,
        np.sum(expected.parameter_independent_carrier),
        rtol=tolerance,
        atol=tolerance,
    )
