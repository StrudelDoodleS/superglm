"""Constructed rank outputs isolate the wall classifier from eigensolver round-off (#560)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import Predictor
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.fit_state import _wall_disclosure
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.features import Numeric
from superglm.solvers.rank import decompose_gram
from tests._distributional_weights import resolved_prior


def _wall_event(kind):
    """Real designs and clipped link values; tests supply the dropped direction."""
    x = np.linspace(0.5, 2.0, 80)
    frame = as_eager_frame(pd.DataFrame({"x": x, "x2": x + 1.0e-10}))
    family = GeneralizedParetoLSS()
    predictors = (
        (Predictor("scale", {"x": Numeric()}), Predictor("shape", {}))
        if kind == "scale_alias"
        else (
            Predictor("scale", {}),
            Predictor("shape", {"x": Numeric(), "x2": Numeric()}),
        )
    )
    weights = resolved_prior(np.ones(len(x)))
    layout = build_stacked_layout(compile_predictors(frame, weights, family.parameters, predictors))
    initial = np.zeros(layout.n_coefficients)
    initial[layout.predictor("shape").intercept_index] = 40.0
    u = (np.arange(len(x)) + 0.5) / len(x)
    y = np.expm1(-0.3 * np.log(u)) / 0.3
    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        family.bind_likelihood(y, weights, COMPLETE_OBSERVATION),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        initial=initial,
    )
    assert np.all(result.theta[:, 1] == np.nextafter(1.0, 0.0))
    return family, layout, result


@pytest.fixture(scope="module")
def scale_alias_event():
    return _wall_event("scale_alias")


@pytest.fixture(scope="module")
def shape_alias_event():
    return _wall_event("shape_alias")


def _with_dropped_direction(result, direction, column_scale):
    """Force the rank output, including tiny components no LAPACK driver must produce.

    The projector has exactly this one-dimensional kernel in exact arithmetic.
    Replace its computed null vector with the specified direction, so the wall
    classification is tested independently of the eigensolver's rounding.
    A zero score removes step pressure, isolating dropped-direction attribution.
    """
    direction = np.asarray(direction, dtype=np.float64)
    unit = direction / np.linalg.norm(direction)
    gram = np.eye(len(direction)) - np.outer(unit, unit)
    rank = replace(
        decompose_gram(gram),
        parameter_null_basis=direction[:, None],
        column_scale=np.asarray(column_scale, dtype=np.float64),
    )
    return replace(
        result,
        terminal_data_curvature=gram,
        terminal_penalized_curvature=gram,
        terminal_rank=rank,
        terminal_score=np.zeros(len(direction)),
    )


@pytest.mark.parametrize(
    "direction, scales, unreliable",
    [
        ((1.0, -1.0, 1.0e-12), (1.0, 1.0, 1.0), False),
        ((0.0, 0.0, 1.0), (1.0, 1.0, 1.0), True),
        # Shape share is measured after equilibration, not in raw coefficients.
        ((1.0, -1.0, 1.0e-12), (1.0e-12, 1.0e-12, 1.0), True),
        # A column with no curvature still counts at face value.
        ((0.0, 0.0, 1.0), (1.0, 1.0, 0.0), True),
    ],
    ids=["scale_alias", "shape_direction", "equilibrated_share", "zero_curvature_column"],
)
def test_only_a_substantial_equilibrated_shape_direction_makes_clipped_uncertainty_unreliable(
    scale_alias_event, direction, scales, unreliable
):
    family, layout, result = scale_alias_event
    forced = _with_dropped_direction(result, direction, scales)
    message = _wall_disclosure(family, forced, layout)
    assert "80 of 80 rows" in message and "float64 can represent" in message
    assert "the data put the shape" not in message
    assert ("not reliable" in message) is unreliable


@pytest.mark.parametrize(
    "direction, unreliable",
    [((0.0, 0.0, 1.0, -1.0), False), ((0.0, 0.0, 1.0, 1.0), True)],
    ids=["cancelling_design", "shape_movement"],
)
def test_only_a_direction_the_shape_design_moves_makes_clipped_uncertainty_unreliable(
    shape_alias_event, direction, unreliable
):
    family, layout, result = shape_alias_event
    forced = _with_dropped_direction(result, direction, np.ones(4))
    message = _wall_disclosure(family, forced, layout)
    assert "80 of 80 rows" in message and "float64 can represent" in message
    assert "the data put the shape" not in message
    assert ("not reliable" in message) is unreliable
