"""Pass-local support work, separate from prediction correctness."""

import linecache
import sys
import weakref
from dataclasses import replace

import numpy as np
import pytest

import superglm.distributional.solver.chunks as chunking
from superglm.distributional.solver.chunks import iter_likelihood_chunks
from superglm.group_matrix import DesignMatrix, DiscretizedSplineCategoricalGroupMatrix

from .test_distributional_chunk_execution import _evaluate, _problem


def _support_products(operation, problem, size):
    """Count support-factor product statements without replacing array algebra."""
    count = 0

    def trace(frame, event, arg):
        nonlocal count
        if event == "line" and "/src/superglm/" in frame.f_code.co_filename:
            line = linecache.getline(frame.f_code.co_filename, frame.f_lineno)
            count += ("B_unique @" in line or "R_inv @" in line) and not line.lstrip().startswith(
                "#"
            )
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        if operation == "geometry":
            list(iter_likelihood_chunks(*problem, chunk_size=size, curvature_source="observed"))
        else:
            _evaluate(operation, problem, size)
    finally:
        sys.settrace(previous)
    return count


def _support_problem(category):
    problem = list(_problem())
    if category:
        layout = problem[1]
        state = layout.predictors[0]
        groups = list(state.design.group_matrices)
        group = groups[1]
        rows = np.arange(0, state.design.n, 2, dtype=np.intp)
        groups[1] = DiscretizedSplineCategoricalGroupMatrix(
            group.B_unique,
            group.R_inv,
            group.bin_idx[rows],
            rows,
            n_rows=state.design.n,
            bin_idx_is_level=True,
        )
        state = replace(state, design=DesignMatrix(groups, state.design.n, state.design.p))
        problem[1] = replace(layout, predictors=(state, layout.predictors[1]))
    return tuple(problem)


@pytest.mark.parametrize("category", [False, True])
@pytest.mark.parametrize("operation", ["likelihood", "change", "terminal", "geometry"])
def test_support_products_are_once_per_pass(operation, category):
    problem = _support_problem(category)
    full = _support_products(operation, problem, 23)
    chunked = _support_products(operation, problem, 7)
    assert full > 0
    assert chunked == full


@pytest.mark.parametrize("category", [False, True])
@pytest.mark.parametrize("mutation", ["basis", "transform", "coefficients", "bins", "rows"])
def test_generator_reads_live_sources_after_yield(category, mutation):
    problem = _support_problem(category)
    _, layout, _, _, coefficients = problem
    group = layout.predictors[0].design.group_matrices[1]
    iterator = iter_likelihood_chunks(*problem, chunk_size=7, curvature_source="observed")
    next(iterator)
    if mutation == "basis":
        group.B_unique[:] *= 0.5
    elif mutation == "transform":
        group.R_inv[:] *= 0.75
    elif mutation == "coefficients":
        coefficients[:] *= 0.6
    elif mutation == "bins":
        bins = group.bin_idx_level if category else group.bin_idx
        bins[:] = (bins + 1) % len(group.B_unique)
    elif category:
        group.row_idx = (group.row_idx + 1) % len(problem[2])
    for chunk in iterator:
        expected, _ = chunking._predictor_chunk(
            layout, coefficients, chunk.rows, include_offsets=True
        )
        np.testing.assert_array_equal(chunk.eta, expected)


def test_support_association_and_geometry_addition_order():
    problem = _support_problem(False)
    _, layout, _, _, coefficients = problem
    state = layout.predictors[0]
    numeric, group = state.design.group_matrices
    coefficients[:] = 0
    coefficients[state.coefficient_slice.start] = 1e16
    coefficients[state.coefficient_slice.start + 1] = -1e16
    coefficients[state.coefficient_slice.start + 2 : state.coefficient_slice.start + 4] = [1, -1]
    numeric.M = np.ones_like(numeric.M)
    group.B_unique[:] = 0
    group.B_unique[:, :2] = [1e16, 1]
    group.R_inv[:] = 0
    group.R_inv[:2, :2] = [[1, 1], [1, 0]]
    state.offset.setflags(write=True)
    state.offset[:] = 0
    for chunk in iter_likelihood_chunks(*problem, chunk_size=7, curvature_source="observed"):
        # Slopes sum to -1e16 before adding the intercept. Reassociation to
        # (B @ R) @ beta or intercept-first accumulation loses this witness.
        np.testing.assert_array_equal(chunk.eta[:, 0], 0)
    eta, _ = chunking.materialize_terminal_predictions(layout, coefficients, chunk_size=7)
    np.testing.assert_array_equal(eta[:, 0], 1)


def test_pass_workspace_is_released_on_close_and_not_shared(monkeypatch):
    original = chunking._SupportPredictions
    references = []

    def tracked():
        workspace = original()
        references.append(weakref.ref(workspace))
        return workspace

    monkeypatch.setattr(chunking, "_SupportPredictions", tracked)
    problem = _problem()
    iterator = iter_likelihood_chunks(*problem, chunk_size=7, curvature_source="observed")
    next(iterator)
    assert references[0]() is not None
    iterator.close()
    assert references[0]() is None
    _evaluate("terminal", problem, 7)
    assert len(references) == 2
    assert all(ref() is None for ref in references)


@pytest.mark.parametrize("category", [False, True])
def test_budget_refusal_preserves_results(monkeypatch, category):
    problem = _support_problem(category)
    expected = _evaluate("terminal", problem, 7)
    monkeypatch.setattr(chunking, "_SUPPORT_PREDICTION_BYTES", 0)
    actual = _evaluate("terminal", problem, 7)
    np.testing.assert_array_equal(actual, expected)
    assert _support_products("terminal", problem, 7) > _support_products("terminal", problem, 23)


def test_custom_design_geometry_keeps_its_matvec():
    class CustomDesign(DesignMatrix):
        def row_subset(self, indices):
            child = super().row_subset(indices)
            return CustomDesign(child.group_matrices, child.n, child.p)

        def matvec(self, beta):
            return super().matvec(beta) + 0.25

    problem = list(_problem())
    layout = problem[1]
    state = layout.predictors[0]
    custom = CustomDesign(state.design.group_matrices, state.design.n, state.design.p)
    problem[1] = replace(layout, predictors=(replace(state, design=custom), layout.predictors[1]))
    for chunk in iter_likelihood_chunks(*problem, chunk_size=7, curvature_source="observed"):
        expected, _ = chunking._predictor_chunk(
            problem[1], problem[-1], chunk.rows, include_offsets=True
        )
        np.testing.assert_array_equal(chunk.eta, expected)
