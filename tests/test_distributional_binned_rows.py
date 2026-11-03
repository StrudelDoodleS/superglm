"""A binned layout's rows gathered from its bin tables equal its dense design rows, reading no dense rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import GaussianLS, TweedieLSS
from superglm.distributional._row_design import BoundedPredictorMatrix
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import binned_rows as br
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


def _within(got, reference, width):
    """got within 8 p u of the dense rows, normwise: the tables' B R against the rows'."""
    return bool(
        np.max(np.abs(got - reference), initial=0.0)
        <= 8 * width * U * np.max(np.abs(reference), initial=0.0)
    )


@pytest.fixture(params=[2, 3], ids=["K2", "K3"])
def problem(request):
    return _layout(request.param)


def test_fixture_covers_every_group_kind(problem):
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
    assert br.binned_layout(layout)


@pytest.mark.parametrize("rows", [(0, N), (13, 13), (31, 500)])
def test_rows_equal_the_dense_design_rows(problem, rows):
    layout, matrices, _slices, _rng = problem
    reader = br.binned_rows(layout)
    start, stop = rows
    for q, matrix in enumerate(matrices):
        got = reader.rows(q, start, stop)
        assert got.shape == (stop - start, matrix.shape[1]) and got.flags.c_contiguous
        assert _within(got, matrix[start:stop], matrix.shape[1])


def _compressed_layout():
    """An exact-mode layout on integer covariates, which compresses each spline to its supports."""
    rng = np.random.default_rng(3)
    n = 1500
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "a": rng.integers(18, 60, n).astype(float),
                "b": rng.integers(0, 12, n).astype(float),
                "d": rng.normal(size=n),
                "g": rng.choice(list("abc"), n),
                "h": rng.choice(list("pqrs"), n),
            }
        )
    )
    family = GaussianLS()
    names = [parameter.name for parameter in family.parameters]
    predictors = [
        Predictor(
            names[0],
            {
                "a": Spline(kind="cr", n_knots=6),
                "g": Categorical(base="a"),
                "h": Categorical(base="p"),
                "d": Numeric(),
            },
            interaction_specs={"a:h": SplineCategorical("a", "h")},
        ),
        Predictor(names[1], {"b": Spline(kind="cr", n_knots=5), "h": Categorical(base="p")}),
    ]
    compiled = compile_predictors(
        frame, resolved_prior(np.ones(n)), family.parameters, predictors, model_discrete=False
    )
    layout = build_stacked_layout(compiled)
    matrices = dense_predictor_matrices(layout)
    return layout, matrices, tuple(state.coefficient_slice for state in layout.predictors), rng


def test_support_compressed_layouts_equal_the_dense_rows():
    """Exact fits on few-valued covariates keep support-compressed splines and levels,
    which binned_layout admits: their gathered rows match the dense rows."""
    layout, matrices, _slices, _rng = _compressed_layout()
    kinds = {type(g).__name__ for state in layout.predictors for g in state.design.group_matrices}
    assert {
        "SupportCompressedSSPGroupMatrix",
        "SupportCompressedSplineCategoricalGroupMatrix",
    } <= kinds
    reader = br.binned_rows(layout)
    for q, matrix in enumerate(matrices):
        assert _within(reader.rows(q, 31, 900), matrix[31:900], matrix.shape[1])


def test_exact_route_layouts_have_no_reader():
    layout = _layout(2, discrete=False)[0]
    assert not br.binned_layout(layout)
    assert br.binned_rows(layout) is None
    with pytest.raises(ValueError, match="binned layout"):
        br.BinnedRows(layout)


def test_rows_outside_the_design_are_refused(problem):
    reader = br.binned_rows(problem[0])
    for start, stop in ((-1, 3), (5, 4), (0, N + 1)):
        with pytest.raises(ValueError, match="inside the design"):
            reader.rows(0, start, stop)


def test_tables_are_formed_once_and_no_dense_rows_are_read(problem, monkeypatch):
    """The reader forms each predictor's tables at construction; its row reads form no
    dense design rows and re-form no table."""
    layout, matrices, _slices, _rng = problem
    built = []
    original = br._terms
    monkeypatch.setattr(br, "_terms", lambda state: built.append(state) or original(state))

    def forbidden(*args, **kwargs):
        pytest.fail("the reader formed dense design rows")

    for state in layout.predictors:
        for group in state.design.group_matrices:
            if type(group).__name__ != "DenseGroupMatrix":
                monkeypatch.setattr(type(group), "toarray", forbidden)
    monkeypatch.setattr(DesignMatrix, "toarray", forbidden)
    monkeypatch.setattr(BoundedPredictorMatrix, "__getitem__", forbidden)
    reader = br.binned_rows(layout)
    for q in range(len(matrices)):
        for start in range(0, N, 97):
            reader.rows(q, start, min(start + 97, N))
    assert len(built) == len(layout.predictors)


def test_rows_repeat_their_bits(problem):
    layout = problem[0]
    first, second = br.binned_rows(layout), br.binned_rows(layout)
    for q in range(len(layout.predictors)):
        np.testing.assert_array_equal(first.rows(q, 0, N), first.rows(q, 0, N))
        np.testing.assert_array_equal(first.rows(q, 7, 300), second.rows(q, 7, 300))


# Mutation checks: the comparison fails against a deliberately broken reader.


def test_mutation_off_level_rows_reading_a_bin_fail(problem, monkeypatch):
    layout, matrices, _slices, _rng = problem
    original = br._codes

    def level_rows_read_bin_zero(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "DiscretizedSplineCategoricalGroupMatrix":
            result = np.where(result == term.group.n_bins, 0, result)
        return result

    monkeypatch.setattr(br, "_codes", level_rows_read_bin_zero)
    reader = br.binned_rows(layout)
    assert not all(_within(reader.rows(q, 0, N), m, m.shape[1]) for q, m in enumerate(matrices))


def test_mutation_base_level_reading_the_first_column_fails(problem, monkeypatch):
    layout, matrices, _slices, _rng = problem
    original = br._codes

    def base_reads_level_zero(term, start, stop):
        result = original(term, start, stop)
        if type(term.group).__name__ == "CategoricalGroupMatrix":
            width = term.columns.stop - term.columns.start
            result = np.where(result >= width, 0, result)
        return result

    monkeypatch.setattr(br, "_codes", base_reads_level_zero)
    reader = br.binned_rows(layout)
    assert not _within(reader.rows(0, 0, N), matrices[0], matrices[0].shape[1])
