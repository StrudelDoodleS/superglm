"""Binned derivative-pass kernels equal the dense row chunks they replace, reading no dense rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import GaussianLS, TweedieLSS
from superglm.distributional._row_design import BoundedPredictorMatrix
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.smoothing.derivatives import _cross_all, _leverage_blocks
from superglm.distributional.solver import binned_passes as bp
from superglm.distributional.solver._ordinary_packing import predictor_panels
from superglm.distributional.solver.assembly import dense_predictor_matrices
from superglm.features import Categorical, Numeric, RandomEffect, Spline, SplineCategorical
from superglm.features.interaction import TensorInteraction
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior

U = 2.0**-53
N = 640


def _layout(k_parameters: int, discrete: bool = True):
    rng = np.random.default_rng(8810 + k_parameters)
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": rng.uniform(-1, 1, N),
                "z": rng.uniform(-1, 1, N),
                "d": rng.normal(size=N),
                "g": rng.choice(list("abc"), N),
                "h": rng.choice(list("pqrs"), N),
                "r": rng.choice([f"l{i}" for i in range(23)], N),
            }
        )
    )
    family = GaussianLS() if k_parameters == 2 else TweedieLSS()
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(
            names[0],
            {
                "x": Spline(kind="cr", n_knots=6),
                "z": Spline(kind="cr", n_knots=5),
                "g": Categorical(base="a"),
                "d": Numeric(),
                "r": RandomEffect(),
            },
            interaction_specs={"x:z": TensorInteraction("x", "z", n_knots=(4, 4))},
        ),
        Predictor(
            names[1],
            {"x": Spline(kind="cr", n_knots=5), "h": Categorical(base="p"), "d": Numeric()},
            interaction_specs={"x:h": SplineCategorical("x", "h")},
        ),
    ]
    if k_parameters == 3:
        predictors.append(Predictor(names[2], {"z": Spline(kind="cr", n_knots=4)}))
    compiled = compile_predictors(
        frame,
        resolved_prior(np.ones(N)),
        family.parameters,
        predictors,
        model_discrete=discrete,
        n_bins_config=24,
    )
    layout = build_stacked_layout(compiled)
    matrices = dense_predictor_matrices(layout)
    return layout, matrices, tuple(state.coefficient_slice for state in layout.predictors), rng


def _excess(got, reference, magnitude, width):
    """How far ``got`` lies outside ``gamma``-style bounds ``8 P u`` times the magnitude."""
    return float(np.max(np.abs(got - reference) - 8 * width * U * magnitude, initial=-np.inf))


@pytest.fixture(params=[2, 3], ids=["K2", "K3"])
def problem(request):
    return _layout(request.param)


@pytest.fixture
def split_blocks(monkeypatch):
    """Blocks of 97 and 37 rows: every bin's rows split across blocks."""
    monkeypatch.setattr(bp, "_LEVERAGE_ROWS", 37)
    monkeypatch.setattr(bp, "contraction_block_rows", lambda n, row_bytes, chunk: 97)


def test_fixture_covers_every_group_kind_and_packs_panels(problem):
    layout = problem[0]
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert kinds == {
        "DiscretizedSSPGroupMatrix",
        "DiscretizedTensorGroupMatrix",
        "DiscretizedSplineCategoricalGroupMatrix",
        "CategoricalGroupMatrix",
        "RandomEffectGroupMatrix",
        "DenseGroupMatrix",
    }
    assert bp.binned_layout(layout)
    # Several blocks pack each predictor's intercept, dense and small categorical
    # groups into an OrdinaryPanelGroupMatrix.
    assert sum(panel is not None for panel in predictor_panels(layout)) >= 2


@pytest.mark.parametrize("cell_cap", [None, 0], ids=["tables", "rows"])
def test_leverage_blocks_equal_the_dense_rows(problem, split_blocks, monkeypatch, cell_cap):
    layout, matrices, slices, rng = problem
    if cell_cap is not None:
        monkeypatch.setattr(bp, "_CELL_CAP", cell_cap)
    width = layout.n_coefficients
    factor = rng.normal(size=(width, width))
    inverse = factor @ factor.T / width
    reference = _leverage_blocks(matrices, slices, inverse)
    magnitude = _leverage_blocks(tuple(map(np.abs, matrices)), slices, np.abs(inverse))
    got = bp.binned_leverage_blocks(layout, inverse)
    assert got.shape == reference.shape
    assert _excess(got, reference, magnitude, width) <= 0.0


@pytest.mark.parametrize("block", [97, N], ids=["blocks", "one-block"])
def test_cross_grams_equal_the_dense_rows_with_one_producer_call_per_block(
    problem, monkeypatch, block
):
    monkeypatch.setattr(bp, "contraction_block_rows", lambda n, row_bytes, chunk: block)
    layout, matrices, slices, rng = problem
    k_parameters = len(matrices)
    width = layout.n_coefficients
    packed = rng.normal(size=(N, 4, k_parameters * (k_parameters + 1) // 2))
    calls = []

    def packed_rows(start, stop):
        calls.append((start, stop))
        return packed[start:stop]

    got = bp.binned_cross_grams(layout, packed_rows, width)
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    assert _excess(got, reference, magnitude, N) <= 0.0
    np.testing.assert_array_equal(got, np.transpose(got, (0, 2, 1)))
    assert calls == [(start, min(start + block, N)) for start in range(0, N, block)]


@pytest.mark.parametrize("rows", [(0, N), (13, 13), (31, 500)])
def test_design_dot_rows_equal_the_dense_rows(problem, rows):
    layout, matrices, _slices, rng = problem
    start, stop = rows
    for q, matrix in enumerate(matrices):
        D = rng.normal(size=(matrix.shape[1], 7))
        got = bp.design_dot_rows(layout, q, D, start, stop)
        magnitude = np.abs(matrix[start:stop]) @ np.abs(D)
        assert got.shape == (stop - start, 7)
        assert _excess(got, matrix[start:stop] @ D, magnitude, matrix.shape[1]) <= 0.0


def test_exact_route_layouts_have_no_kernel():
    layout = _layout(2, discrete=False)[0]
    assert not bp.binned_layout(layout)
    assert bp.binned_leverage_blocks(layout, np.eye(layout.n_coefficients)) is None
    assert bp.binned_cross_grams(layout, pytest.fail, layout.n_coefficients) is None
    assert bp.design_dot_rows(layout, 0, np.zeros((1, 1)), 0, 1) is None


def test_no_dense_design_rows_are_formed(problem, split_blocks, monkeypatch):
    layout, matrices, _slices, rng = problem
    width = layout.n_coefficients

    def forbidden(*args, **kwargs):
        pytest.fail("a binned kernel formed dense design rows")

    for state in layout.predictors:
        for group in state.design.group_matrices:
            if type(group).__name__ != "DenseGroupMatrix":
                monkeypatch.setattr(type(group), "toarray", forbidden)
    monkeypatch.setattr(DesignMatrix, "toarray", forbidden)
    monkeypatch.setattr(BoundedPredictorMatrix, "__getitem__", forbidden)
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 2, k_parameters * (k_parameters + 1) // 2))
    assert bp.binned_leverage_blocks(layout, np.eye(width)) is not None
    assert bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width) is not None
    assert bp.design_dot_rows(layout, 1, np.ones((matrices[1].shape[1], 2)), 5, 90) is not None


# Mutation checks: each kernel's comparison fails against a deliberately broken variant.


def _level_rows_read_bin_zero(original):
    def codes(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "DiscretizedSplineCategoricalGroupMatrix":
            result = np.where(result == term.group.n_bins, 0, result)
        return result

    return codes


def test_mutation_leverage_off_level_rows_reading_a_bin_fail(problem, split_blocks, monkeypatch):
    layout, matrices, slices, rng = problem
    width = layout.n_coefficients
    inverse = np.eye(width) + 0.01
    reference = _leverage_blocks(matrices, slices, inverse)
    magnitude = _leverage_blocks(tuple(map(np.abs, matrices)), slices, np.abs(inverse))
    monkeypatch.setattr(bp, "_codes", _level_rows_read_bin_zero(bp._codes))
    got = bp.binned_leverage_blocks(layout, inverse)
    assert _excess(got, reference, magnitude, width) > 0.0


def test_mutation_cross_grams_reusing_the_first_block_plans_fail(
    problem, split_blocks, monkeypatch
):
    layout, matrices, slices, rng = problem
    width = layout.n_coefficients
    k_parameters = len(matrices)
    packed = rng.normal(size=(N, 2, k_parameters * (k_parameters + 1) // 2))
    reference = _cross_all(matrices, slices, packed, width)
    magnitude = _cross_all(tuple(map(np.abs, matrices)), slices, np.abs(packed), width)
    original = bp.block_plans
    first = {}

    def stale(layout, panels, start, stop):
        plans = first.setdefault("plans", original(layout, panels, start, stop))
        return plans if stop - start == 97 else original(layout, panels, start, stop)

    monkeypatch.setattr(bp, "block_plans", stale)
    got = bp.binned_cross_grams(layout, lambda a, b: packed[a:b], width)
    assert _excess(got, reference, magnitude, N) > 0.0


def test_mutation_design_dot_rows_reading_the_base_level_fails(problem, monkeypatch):
    layout, matrices, _slices, rng = problem
    original = bp._left

    def base_reads_level_zero(term, M):
        result = original(term, M)
        if term.one_hot:
            result[-1] = M[0]
        return result

    monkeypatch.setattr(bp, "_left", base_reads_level_zero)
    D = rng.normal(size=(matrices[0].shape[1], 3))
    got = bp.design_dot_rows(layout, 0, D, 0, N)
    magnitude = np.abs(matrices[0]) @ np.abs(D)
    assert _excess(got, matrices[0] @ D, magnitude, matrices[0].shape[1]) > 0.0
