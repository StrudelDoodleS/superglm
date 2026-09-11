"""Observed stationarity geometry is handed to the same accepted endpoint."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.solver as solver
from superglm._frame import as_eager_frame
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.features import Numeric

from ._distributional_weights import resolved_prior
from .test_distributional_endpoint_laml import _UnitGaussian


def _problem(scale=1.0):
    family = _UnitGaussian()
    response = np.array([1.0, 1.0, np.nextafter(1.0, np.inf)])
    weights = resolved_prior(np.ones(3))
    frame = as_eager_frame(pd.DataFrame({"x": np.full(3, scale)}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"x": Numeric()}, intercept=False),),
        )
    )
    return family, layout, response, family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)


def _record_geometry(monkeypatch, alter=None):
    records = []
    original = solver._measured_geometry

    def measure(context, state, source, recorder):
        geometry = original(context, state, source, recorder)
        if alter is not None:
            geometry = alter(source, geometry)
        records.append((state, source, geometry))
        return geometry

    monkeypatch.setattr(solver, "_measured_geometry", measure)
    return records


@pytest.mark.parametrize("chunk_size", [None, 1], ids=["dense", "chunked"])
@pytest.mark.parametrize("distinct_step", [False, True], ids=["noop", "accepted_step"])
@pytest.mark.parametrize("certified", [False, True], ids=["refused", "certified"])
def test_certificate_geometry_reaches_initial_terminal(
    monkeypatch, chunk_size, distinct_step, certified
):
    scale = 2.0**20 if distinct_step else 1.0
    problem = _problem(scale)
    initial = np.array([np.nextafter(1.0 / scale, 0.0) if distinct_step else 1.0])
    config = DenseSolverConfig(max_iterations=1, tolerance=1.0e-20, coefficient_curvature="fisher")
    records = _record_geometry(monkeypatch)
    if not certified:
        monkeypatch.setattr(solver, "_newton_decrement_is_certified", lambda **kwargs: False)
    result = solver.fit_dense_fixed_lambda(
        *problem, np.zeros((1, 1)), initial=initial, config=config, chunk_size=chunk_size
    )
    observed = [record for record in records if record[1] == "observed"]
    assert len(observed) == 1, "certificate geometry must not be remeasured at publication"
    np.testing.assert_array_equal(result.coefficients, observed[0][0].coefficients)
    np.testing.assert_array_equal(result.terminal_score, observed[0][2].score_penalized)
    assert result.config.coefficient_curvature == "fisher"
    assert result.terminal_curvature.actual_source == "observed"
    assert result.terminal_rank.rank == 1
    assert result.converged is certified
    assert result.convergence_reason == (
        "objective_and_step"
        if certified
        else "max_iterations"
        if distinct_step
        else "line_search_failed"
    )
    assert result.iterations == len(result.history) == int(distinct_step)
    # The represented-data optimum is within one rounding unit of one.
    assert abs(scale * result.coefficients[0] - 1.0) <= np.finfo(float).eps


@pytest.mark.parametrize("chunk_size", [None, 1], ids=["dense", "chunked"])
def test_new_accepted_state_invalidates_refused_certificate_geometry(monkeypatch, chunk_size):
    problem = _problem(2.0**10)
    config = DenseSolverConfig(
        max_iterations=2,
        tolerance=1.0e-4,
        max_predictor_step=2.0**-20,
        coefficient_curvature="fisher",
    )
    records = _record_geometry(monkeypatch)
    original = solver._solve_coefficient_direction
    directions = []

    def solve(*args):
        direction = original(*args)
        directions.append(direction)
        # The second valid step still advances the accepted state, but its
        # reported residual disqualifies a new objective/step certificate.
        return (
            replace(direction, residual=2 * config.residual_tolerance)
            if len(directions) == 2
            else direction
        )

    monkeypatch.setattr(solver, "_solve_coefficient_direction", solve)
    result = solver.fit_dense_fixed_lambda(
        *problem, np.zeros((1, 1)), initial=np.zeros(1), config=config, chunk_size=chunk_size
    )
    observed = [record for record in records if record[1] == "observed"]
    assert len(observed) == 2
    assert observed[0][0] is not observed[1][0]
    assert observed[0][0].coefficients[0] < observed[1][0].coefficients[0]
    np.testing.assert_array_equal(result.coefficients, observed[1][0].coefficients)
    np.testing.assert_array_equal(result.terminal_score, observed[1][2].score_penalized)
    assert result.iterations == len(result.history) == 2
    assert not result.converged
    assert result.convergence_reason == "max_iterations"


@pytest.mark.parametrize("chunk_size", [None, 1], ids=["dense", "chunked"])
@pytest.mark.parametrize("retry_recovers", [False, True], ids=["fisher_fallback", "observed_retry"])
def test_certificate_geometry_reaches_retry_terminal(monkeypatch, chunk_size, retry_recovers):
    problem = _problem(2.0**10)

    def alter(source, geometry):
        if source == "observed" and not retry_recovers:
            # Drive the real terminal retry/fallback policy with material
            # indefiniteness. Fisher solve geometry must remain untouched.
            negative = -geometry.penalized_curvature
            negative.setflags(write=False)
            return replace(geometry, data_curvature=negative, penalized_curvature=negative)
        return geometry

    records = _record_geometry(monkeypatch, alter)
    if retry_recovers:
        original = solver.resolve_curvature
        decisions = []

        def force_first_retry(source, matrix, **kwargs):
            # Exercise a successful retry handoff with unchanged, valid
            # geometry. Only the first policy input is forced indefinite.
            decisions.append(True)
            return original(source, -matrix if len(decisions) == 1 else matrix, **kwargs)

        monkeypatch.setattr(solver, "resolve_curvature", force_first_retry)
    result = solver.fit_dense_fixed_lambda(
        *problem,
        np.zeros((1, 1)),
        initial=np.array([2.0**-10]),
        config=DenseSolverConfig(
            max_iterations=1,
            terminal_retry_iterations=1,
            tolerance=1.0e-20,
            coefficient_curvature="fisher",
        ),
        chunk_size=chunk_size,
    )
    observed = [record for record in records if record[1] == "observed"]
    assert len(observed) == 2, "each run's measured certificate must reach its terminal decision"
    assert result.terminal_curvature.actual_source == ("observed" if retry_recovers else "fisher")
    assert result.converged is retry_recovers
    assert result.convergence_reason == (
        "objective_and_step" if retry_recovers else "line_search_failed"
    )
    assert result.iterations == 0
    assert result.history == ()
    assert result.terminal_rank.rank == 1
    np.testing.assert_array_equal(result.terminal_score, observed[-1][2].score_penalized)
    fisher = [record for record in records if record[1] == "fisher"]
    assert all(record[2].penalized_curvature[0, 0] > 0 for record in fisher)
    if not retry_recovers:
        assert result.terminal_curvature.reason == "material_indefiniteness_after_retry"
        np.testing.assert_array_equal(
            result.terminal_penalized_curvature, fisher[-1][2].penalized_curvature
        )
