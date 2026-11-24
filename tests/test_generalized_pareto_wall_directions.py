"""Constructed rank outputs isolate the wall classifier from eigensolver round-off (#560)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional import Predictor
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.fit_state import _wall_disclosure
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.results.solver import DenseSolverConfig, DenseSolverResult
from superglm.distributional.telemetry import CurvatureTelemetry
from superglm.distributional.walls import predictor_values
from superglm.features import Numeric
from superglm.solvers.rank import decompose_gram
from tests._distributional_weights import resolved_prior


def _terminal_result(layout, coefficients, gram, score, rank):
    """Deliver a prescribed terminal score and rank with parameters from the real link."""
    eta = np.column_stack(
        [
            predictor_values(state, coefficients[state.coefficient_slice]) + state.offset
            for state in layout.predictors
        ]
    )
    theta = np.column_stack(
        [state.link.inverse(eta[:, state.parameter_index]) for state in layout.predictors]
    )
    return DenseSolverResult(
        config=DenseSolverConfig(),
        family_likelihood_plan_identifier="test-plan:v1",
        resolved_chunk_size=None,
        execution_backend_identifier="distributional-dense-v1",
        coefficients=coefficients,
        eta=eta,
        theta=theta,
        penalty=np.zeros_like(gram),
        initial_penalized_log_likelihood=0.0,
        log_likelihood=0.0,
        penalty_value=0.0,
        penalized_log_likelihood=0.0,
        terminal_score=score,
        score_relative=float(np.linalg.norm(score, ord=np.inf)),
        objective_relative_change=0.0,
        step_relative=0.0,
        converged=True,
        convergence_reason="score",
        iterations=0,
        history=(),
        backtracking_steps=0,
        terminal_data_curvature=gram,
        terminal_penalized_curvature=gram,
        terminal_rank=rank,
        terminal_curvature=CurvatureTelemetry(
            requested_source="observed",
            actual_source="observed",
            reason=None,
            minimum_eigenvalue=0.0 if rank.rank < len(gram) else float(np.min(np.diag(gram))),
            rank=rank.rank,
            condition_estimate=None,
            fallback_count=0,
        ),
    )


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
    coefficients = np.zeros(layout.n_coefficients)
    coefficients[layout.predictor("shape").intercept_index] = 40.0
    return family, layout, coefficients


@pytest.fixture(scope="module")
def scale_alias_event():
    return _wall_event("scale_alias")


@pytest.fixture(scope="module")
def shape_alias_event():
    return _wall_event("shape_alias")


def _with_dropped_direction(layout, coefficients, direction, column_scale):
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
    return _terminal_result(layout, coefficients, gram, np.zeros(len(direction)), rank)


@pytest.mark.pins_behaviour
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
    """Pin attribution of an uncertainty caveat to the equilibrated shape direction."""
    family, layout, coefficients = scale_alias_event
    forced = _with_dropped_direction(layout, coefficients, direction, scales)
    message = _wall_disclosure(family, forced, layout)
    assert "80 of 80 rows" in message and "float64 can represent" in message
    assert "the data put the shape" not in message
    assert ("not reliable" in message) is unreliable


@pytest.mark.pins_behaviour
@pytest.mark.parametrize(
    "direction, unreliable",
    [((0.0, 0.0, 1.0, -1.0), False), ((0.0, 0.0, 1.0, 1.0), True)],
    ids=["cancelling_design", "shape_movement"],
)
def test_only_a_direction_the_shape_design_moves_makes_clipped_uncertainty_unreliable(
    shape_alias_event, direction, unreliable
):
    """Pin how a near-alias affects uncertainty on clipped rows."""
    family, layout, coefficients = shape_alias_event
    forced = _with_dropped_direction(layout, coefficients, direction, np.ones(4))
    message = _wall_disclosure(family, forced, layout)
    assert "80 of 80 rows" in message and "float64 can represent" in message
    assert "the data put the shape" not in message
    assert ("not reliable" in message) is unreliable


@pytest.mark.pins_behaviour
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("event", ["zero-slope", "outward-step", "stationary-clip", "interior"])
def test_wall_disclosure_uses_the_terminal_shape_and_uncertainty(side, event):
    """Pin wall row counts and uncertainty caveats from prescribed terminal fits."""
    family = GeneralizedParetoLSS(shape_lower=0.5)
    group = np.repeat([0.0, 1.0], 4)
    frame = as_eager_frame(pd.DataFrame({"group": group}))
    predictors = (Predictor("scale", {}), Predictor("shape", {"group": Numeric()}))
    layout = build_stacked_layout(
        compile_predictors(
            frame, resolved_prior(np.ones(len(group))), family.parameters, predictors
        )
    )
    upper = side == "upper"
    sign = 1.0 if upper else -1.0
    magnitudes = {
        "zero-slope": 1000.0,
        "outward-step": 36.0,
        "stationary-clip": 36.0,
        "interior": 32.0,
    }
    eta = sign * magnitudes[event]
    coefficients = np.array([0.0, 0.0, eta])
    slope = float(layout.predictor("shape").link.deriv_inverse(np.array([eta]))[0])
    assert (slope == 0.0) is (event == "zero-slope")
    # The wall group loses its own direction at zero slope. In the interior
    # control a dropped direction alone cannot cause a wall disclosure.
    group_curvature = 0.0 if event == "interior" else slope**2
    gram = np.diag([1.0, 1.0, group_curvature])
    score = np.array([0.0, 0.0, sign * group_curvature if event == "outward-step" else 0.0])
    result = _terminal_result(layout, coefficients, gram, score, decompose_gram(gram))
    assert result.terminal_rank.rank == (2 if group_curvature == 0.0 else 3)
    assert np.all(result.theta[group == 0.0, 1] == 0.75)
    wall, other = (1.0, 0.5) if upper else (0.5, 1.0)
    last = np.nextafter(wall, other)
    assert np.all((result.theta[group == 1.0, 1] == last) == (event != "interior"))

    message = _wall_disclosure(family, result, layout)
    if event == "interior":
        assert message is None
    else:
        assert "4 of 8 rows" in message and "float64 can represent" in message
        assert f"{side} wall {wall:g}" in message and "the data put the shape" not in message
        assert ("not reliable" in message) is (event != "stationary-clip")
        if upper:
            assert "mean and expected shortfall" in message and "set by that rounding" in message
