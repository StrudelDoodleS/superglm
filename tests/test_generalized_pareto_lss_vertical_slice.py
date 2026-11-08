"""Public fits, recovery and cross-checks for ``GeneralizedParetoLSS``."""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

from superglm import Spline
from superglm._frame import as_eager_frame
from superglm.diagnostics.convergence import ConvergenceWarning
from superglm.distributional import Predictor
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import generalized_pareto as gp
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric, RandomEffect
from tests._distributional_weights import resolved_prior
from tests._generalized_pareto_lss_oracles import fit_scale_regression, profile_shape
from tests.bound_predictor_fixtures import model_from_templates, with_offset_declarations

_TRUE_SHAPE = 0.3


def _simulate(n, seed, *, shape=_TRUE_SHAPE):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    log_scale = 0.4 + 0.7 * np.sin(np.pi * x)
    scale = np.exp(log_scale)
    excess = scale * np.expm1(-shape * np.log(rng.random(n))) / shape
    return pd.DataFrame({"x": x}), excess, log_scale


def _predictors(k=6):
    return (Predictor("scale", {"x": Spline(kind="cr", k=k)}), Predictor("shape", {}))


def test_the_scale_surface_and_the_shape_are_recovered_under_reml():
    frame, y, log_scale = _simulate(12_000, 20260902)
    model = model_from_templates(family=GeneralizedParetoLSS(), predictors=_predictors()).fit_reml(
        frame, y, method="efs"
    )
    fitted = model.predict_parameters(frame)
    residual = np.log(fitted["scale"].to_numpy()) - log_scale
    assert math.sqrt(float(np.mean((residual - residual.mean()) ** 2))) <= 0.12
    assert (
        abs(float(residual.mean())) <= 0.10
    )  # the level, which the shape is partly confounded with
    assert abs(float(fitted["shape"].iloc[0]) - _TRUE_SHAPE) <= 0.06
    assert np.allclose(
        model.predict(frame),
        fitted["scale"].to_numpy() / (1.0 - fitted["shape"].to_numpy()),
        rtol=0,
        atol=1e-12,
    )


def test_the_predicted_quantile_round_trips_and_covers_the_sample():
    frame, y, _ = _simulate(12_000, 5)
    # _predictors() carries a penalised spline, so the fixed-lambda path needs the
    # lambda named; how the smoothing is chosen is not what this test is about.
    model = model_from_templates(family=GeneralizedParetoLSS(), predictors=_predictors()).fit(
        frame, y, lambdas={"scale:x#wiggle": 1.0}
    )
    p90 = model.predict_quantile(frame, 0.9)
    assert p90.shape == (len(frame),) and not p90.flags.writeable
    assert np.allclose(model.predict_cdf(frame, p90), 0.9, rtol=0, atol=1e-12)
    assert float(np.mean(y <= p90)) == pytest.approx(0.9, abs=0.02)
    per_row = model.predict_quantile(frame, np.linspace(0.05, 0.95, len(frame)))
    assert per_row.shape == (len(frame),)


def test_the_unpenalised_fit_matches_an_independent_scipy_maximum_likelihood_fit():
    frame, y, _ = _simulate(4_000, 11)
    x = frame["x"].to_numpy()
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    fitted = model.predict_parameters(frame)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, shape, log_likelihood = fit_scale_regression(y, design)
    ours = np.polyfit(x, np.log(fitted["scale"].to_numpy()), 1)
    assert abs(float(ours[1]) - float(coefficients[0])) <= 1e-4 * (1 + abs(float(coefficients[0])))
    assert abs(float(ours[0]) - float(coefficients[1])) <= 1e-4 * (1 + abs(float(coefficients[1])))
    assert abs(float(fitted["shape"].iloc[0]) - shape) <= 1e-4
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours_log_likelihood = float(
        np.sum(
            family.evaluate_natural(
                y, fitted.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    assert abs(ours_log_likelihood - log_likelihood) <= 1e-8 * (1 + abs(log_likelihood))
    # the profile at the truth is below the maximum, and the maximum is at our shape
    grid = np.array([0.15, shape, 0.55])
    profile = profile_shape(y, design, grid)
    assert profile[1] >= profile[0] and profile[1] >= profile[2]


def test_fisher_curvature_request_is_honoured():
    frame, y, _ = _simulate(2_000, 3)
    model = model_from_templates(
        family=GeneralizedParetoLSS(), predictors=_predictors(), coefficient_curvature="fisher"
    ).fit(frame, y, lambdas={"scale:x#wiggle": 1.0})
    assert model.coefficient_curvature == "fisher"
    assert np.all(np.isfinite(model.predict(frame)))


def test_narrow_walls_keep_an_interior_shape_inside_them():
    frame, y, _ = _simulate(6_000, 8, shape=0.3)
    model = model_from_templates(
        family=GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.5),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    shape = float(model.predict_parameters(frame)["shape"].iloc[0])
    assert 0.05 < shape < 0.5
    assert abs(shape - 0.3) <= 0.08


def test_data_beyond_the_upper_wall_presses_against_it_or_stops():
    """The wall is a hard constraint, so the honest outcome is a pressed fit or a named failure."""
    frame, y, _ = _simulate(3_000, 8, shape=0.7)
    try:
        model = model_from_templates(
            family=GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.35),
            predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
        ).fit(frame, y)
    except Exception as failure:  # noqa: BLE001 - the named failure is the assertion
        message = str(failure).lower()
        assert any(
            word in message for word in ("wall", "shape", "converge", "curvature", "null model")
        ), message
        return
    shape = float(model.predict_parameters(frame)["shape"].iloc[0])
    assert 0.05 < shape < 0.35
    assert shape > 0.3, "data generated above the upper wall must press against it"


def _wall_messages(model):
    return [message for message in model.result_.warnings if "tail shape at the" in message]


def _constant_shape_fit(family, frame, y):
    return model_from_templates(
        family=family, predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {}))
    ).fit(frame, y)


def test_a_shape_pressed_past_the_upper_wall_of_one_is_disclosed_as_having_no_finite_mean():
    # a true shape of 1.5 puts the maximum-likelihood shape far beyond the wall at one
    frame, y, _ = _simulate(3_000, 8, shape=1.5)
    with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
        model = _constant_shape_fit(GeneralizedParetoLSS(), frame, y)
    (message,) = _wall_messages(model)
    assert "3000 of 3000 rows" in message and "infinite" in message
    assert "solver's tolerance" in message
    assert np.all(np.isfinite(model.predict_quantile(frame, 0.99)))


@pytest.mark.parametrize("inner_tol", [1.0e-4, 1.0e-5])
def test_the_wall_disclosure_does_not_depend_on_where_a_loose_tolerance_stops_the_fit(inner_tol):
    # a loose tolerance stops a pressed fit up to about 6e-4 short of the wall, not 1e-9
    frame, y, _ = _simulate(3_000, 8, shape=1.5)
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    )
    with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
        model.fit(frame, y, inner_tol=inner_tol)
    assert len(_wall_messages(model)) == 1


def test_a_shape_pressed_below_the_lower_wall_is_disclosed_with_a_finite_mean():
    # a bounded tail, true shape -0.3, puts the maximum-likelihood shape below zero
    frame, y, _ = _simulate(3_000, 8, shape=-0.3)
    with pytest.warns(ConvergenceWarning, match="at the lower wall 0"):
        model = _constant_shape_fit(GeneralizedParetoLSS(), frame, y)
    (message,) = _wall_messages(model)
    assert "negative shapes" in message
    assert np.all(np.isfinite(model.predict(frame)))


def test_a_shape_pressed_past_a_narrow_upper_wall_names_its_remedy_and_keeps_the_mean():
    frame, y, _ = _simulate(3_000, 8, shape=0.9)
    family = GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.35)
    with pytest.warns(ConvergenceWarning, match="at the upper wall 0.35"):
        model = _constant_shape_fit(family, frame, y)
    (message,) = _wall_messages(model)
    assert "shape_upper" in message
    assert np.all(np.isfinite(model.predict(frame)))


def test_a_shape_that_varies_towards_a_wall_without_reaching_it_is_not_disclosed():
    # an interior fit whose shape runs from about 0.007 to 0.994 across x
    x = np.linspace(-1.0, 1.0, 500)
    shape = expit(4.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(3).random(500))) / shape
    frame = pd.DataFrame({"x": x})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    ).fit(frame, y)
    assert not _wall_messages(model)
    assert np.all(np.isfinite(model.predict(frame)))
    # extrapolated to x = 4 the shape is within 1e-8 of one, and its mean stays the law's own
    beyond = pd.DataFrame({"x": [4.0]})
    shape_beyond = model.predict_parameters(beyond)["shape"].to_numpy()
    assert 1.0 - shape_beyond[0] < 1.0e-8
    assert np.isfinite(model.predict(beyond)[0])


@pytest.mark.parametrize("origin, unit", [(0.0, 1.0), (1.7e9, 3.0e7)], ids=["unit", "timestamp"])
def test_a_linear_shape_reaching_near_both_walls_at_an_interior_optimum_is_not_disclosed(
    origin, unit
):
    # the fitted shape comes within 1e-4 of both walls, but no Newton step moves it on; the
    # same covariate in timestamp-sized units gives the same fit
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(10.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    frame = pd.DataFrame({"x": origin + unit * x})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    ).fit(frame, y, inner_tol=1.0e-10)
    fitted = model.predict_parameters(frame)["shape"].to_numpy()
    assert fitted.min() < 1.0e-4 and 1.0 - fitted.max() < 1.0e-4
    assert not _wall_messages(model)


def test_rows_no_shape_coefficient_moves_do_not_hide_a_pressed_group():
    # group 1 is pressed past the upper wall; group 2's shape is fixed near it by its offset
    heavy = np.expm1(-1.5 * np.log(np.random.default_rng(8).random(1_000))) / 1.5
    y = np.concatenate([heavy, np.ones(1_000)])
    frame = pd.DataFrame({"g": np.repeat([1.0, 0.0], 1_000)})
    offset = np.repeat([0.0, np.log(0.99995 / 0.00005)], 1_000)
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(
            Predictor("scale", {"g": Numeric()}),
            Predictor("shape", {"g": Numeric()}, intercept=False),
        ),
    )
    with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
        with_offset_declarations(model, {"shape": offset}).fit(frame, y, inner_tol=1.0e-10)
    (message,) = _wall_messages(model)
    assert "1000 of 2000 rows" in message


@pytest.mark.parametrize("light", [0.0, 0.7])
def test_a_pressed_group_beside_an_interior_group_is_disclosed(light):
    # only the heavy group runs off, whatever code the interior group carries
    rng = np.random.default_rng(8)
    heavy = np.expm1(-1.5 * np.log(rng.random(1_000))) / 1.5
    interior = np.expm1(-0.3 * np.log(rng.random(1_000))) / 0.3
    frame = pd.DataFrame({"g": np.repeat([1.0, light], 1_000)})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"g": Numeric()}), Predictor("shape", {"g": Numeric()})),
    )
    with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
        model.fit(frame, np.concatenate([heavy, interior]), inner_tol=1.0e-10)
    (message,) = _wall_messages(model)
    assert "1000 of 2000 rows" in message


def test_row_varying_offsets_at_an_interior_optimum_are_not_disclosed():
    # the two rows' shape scores cancel through their link slopes, not in the natural scale
    y = np.tile([0.36839766102977684, 4.428502209460899], 100)
    offset = np.tile(np.log([1.0e-5 / (1.0 - 1.0e-5), 0.99995 / 0.00005]), 100)
    model = with_offset_declarations(
        model_from_templates(
            family=GeneralizedParetoLSS(),
            predictors=(Predictor("scale", {}), Predictor("shape", {})),
        ),
        {"shape": offset},
    ).fit(pd.DataFrame({"x": np.zeros(200)}), y, inner_tol=1.0e-12)
    assert not _wall_messages(model)


def test_a_diverging_linear_shape_is_disclosed_at_both_walls():
    # the slope runs off towards both walls while rows near x = 0 are still on their way
    x = np.linspace(-1.0, 1.0, 1_000)
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    )
    with pytest.warns(ConvergenceWarning, match="tail shape at the"):
        model.fit(pd.DataFrame({"x": x}), np.where(x < 0.0, 1.0, 100.0), inner_tol=1.0e-5)
    (message,) = _wall_messages(model)
    assert "lower wall 0" in message and "upper wall 1" in message


def test_a_shape_held_on_a_reml_face_by_its_offset_is_not_disclosed():
    # REML removes the group effect on an exact face, so only the offset sets the shape near one
    y = np.expm1(-1.5 * np.log(np.random.default_rng(8).random(1_000))) / 1.5
    model = with_offset_declarations(
        model_from_templates(
            family=GeneralizedParetoLSS(),
            predictors=(
                Predictor("scale", {}),
                Predictor("shape", {"group": RandomEffect()}, intercept=False),
            ),
        ),
        {"shape": np.full(1_000, np.log(0.99995 / 0.00005))},
    ).fit_reml(
        pd.DataFrame({"group": np.repeat(["a", "b"], 500)}),
        y,
        lambdas={"shape:group#wiggle": 1.0e10},
    )
    assert model.exact_face_components_ == ("shape:group#wiggle",)
    assert not _wall_messages(model)


@pytest.mark.parametrize("shift", [0.0, 300.0])
def test_rows_clipped_at_the_last_float_below_one_get_the_float64_statement(shift):
    # a pressed slope carries rows at g = 10 ten times as far as rows at g = 1, so they reach the
    # last float below one while the others are still pressed; an offset shift the intercept
    # absorbs is the same fit and must say the same
    rng = np.random.default_rng(8)
    interior = np.expm1(-0.3 * np.log(rng.random(1_000))) / 0.3
    heavy = np.expm1(-1.5 * np.log(rng.random(2_000))) / 1.5
    g = np.repeat([0.0, 1.0, 10.0], 1_000)
    frame = pd.DataFrame({"g": g})
    offsets = {"shape": np.full(3_000, shift)}
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"g": Numeric()})),
    )
    with pytest.warns(ConvergenceWarning, match="float64 can represent"):
        with_offset_declarations(model, offsets).fit(frame, np.concatenate([interior, heavy]))
    (message,) = _wall_messages(model)
    assert "1000 of 3000 rows have a tail shape at the upper wall 1: the data put" in message
    assert "1000 of 3000 rows have a tail shape at the last value below the upper wall 1" in message
    shape = model.predict_parameters(frame, offsets=offsets)["shape"].to_numpy()
    assert np.all(shape[g == 10.0] == np.nextafter(1.0, 0.0))


def test_rows_an_offset_clips_get_the_float64_statement_and_no_data_claim():
    # the offset clips the second group, whose data cannot move it there
    y = np.expm1(-0.7 * np.log(np.random.default_rng(8).random(2_000))) / 0.7
    frame = pd.DataFrame({"g": np.repeat([1.0, 0.0], 1_000)})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(
            Predictor("scale", {"g": Numeric()}),
            Predictor("shape", {"g": Numeric()}, intercept=False),
        ),
    )
    with pytest.warns(ConvergenceWarning, match="float64 can represent"):
        with_offset_declarations(model, {"shape": np.repeat([0.0, 40.0], 1_000)}).fit(frame, y)
    (message,) = _wall_messages(model)
    assert "1000 of 2000 rows" in message and "the data put the shape" not in message
    assert "not reliable" not in message


def test_an_interior_shape_clipped_by_its_own_predictor_gets_only_the_float64_statement():
    # a finite optimum whose steep linear shape reaches the last float below one at the edge
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(40.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    ).fit(pd.DataFrame({"x": x}), y, inner_tol=1.0e-12)
    (message,) = _wall_messages(model)
    assert "float64 can represent" in message and "the data put the shape" not in message
    assert "not reliable" not in message


@pytest.mark.parametrize("inner_tol", [1.0e-4, 1.0e-5])
def test_an_interior_fit_stopped_at_a_loose_tolerance_is_not_disclosed(inner_tol):
    # the stop leaves a Newton step of about 3 near the walls, which a few more iterations remove
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(20.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    ).fit(pd.DataFrame({"x": x}), y, inner_tol=inner_tol)
    assert not _wall_messages(model)


def test_an_interior_fit_stopped_by_the_iteration_cap_does_not_blame_the_data():
    # the fit above needs 12 iterations; at 8 its step still carries rows towards the wall
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(20.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric()})),
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(pd.DataFrame({"x": x}), y, max_inner_iter=8)
    assert not model.result_.coefficient_converged
    (message,) = _wall_messages(model)
    assert "has not converged" in message and "the data put the shape" not in message


def test_a_wall_hook_without_settled_still_discloses_an_unfinished_fit():
    class OwnHook(GeneralizedParetoLSS):
        def wall_disclosure(self, pressed, clipped, n_rows, *, unresolved=False):
            return "own: " + super().wall_disclosure(
                pressed, clipped, n_rows, unresolved=unresolved
            )

    y = np.expm1(-1.5 * np.log(np.random.default_rng(8).random(600))) / 1.5
    model = model_from_templates(
        family=OwnHook(), predictors=(Predictor("scale", {}), Predictor("shape", {}))
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(pd.DataFrame({"x": np.zeros(600)}), y, max_inner_iter=8)
    assert not model.result_.coefficient_converged
    (message,) = _wall_messages(model)
    assert message.startswith("own: ")


@pytest.mark.parametrize("cap", [8, 12])
def test_a_pressed_fit_stopped_by_the_iteration_cap_keeps_its_wall_disclosure(cap):
    rng = np.random.default_rng(8)
    x = rng.uniform(-1.0, 1.0, 3_000)
    y = np.exp(0.4 + 0.7 * np.sin(np.pi * x)) * np.expm1(-1.5 * np.log(rng.random(3_000))) / 1.5
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(pd.DataFrame({"x": x}), y, max_inner_iter=cap)
    (message,) = _wall_messages(model)
    assert "3000 of 3000 rows have a tail shape at the upper wall 1" in message
    assert "infinite" in message and "not reliable" in message


def test_a_refit_stopped_at_a_loose_tolerance_is_not_disclosed(monkeypatch):
    # the interior fit above beside a near scale alias, whose refit on the retained
    # coefficients is forced to start where a fresh fit starts: it stops at 1e-4 with a step
    # still carrying rows towards the wall, and is continued as the first run is. Mutation
    # check: without that continuation, 1899 of 5000 rows are disclosed at the upper wall.
    import superglm.distributional.solver.solver as solver

    initial = solver._initial_coefficients

    def from_scratch(context, state, rank, retained, phase_recorder):
        coefficients = np.array(initial(context))
        coefficients[retained.dropped] = 0.0
        return solver._evaluate_state(context, coefficients, phase_recorder=phase_recorder)

    monkeypatch.setattr(solver, "_representative_state", from_scratch)
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(20.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    z = np.random.default_rng(6).uniform(size=5_000)
    frame = pd.DataFrame(
        {"x": x, "z": z, "z2": z + 1.0e-8 * np.random.default_rng(5).standard_normal(5_000)}
    )
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(
            Predictor("scale", {"z": Numeric(), "z2": Numeric()}),
            Predictor("shape", {"x": Numeric()}),
        ),
    ).fit(frame, y, inner_tol=1.0e-4)
    assert model.result_.rank < len(model.result_.coefficients)
    assert not _wall_messages(model)


@pytest.mark.parametrize("inner_tol", [1.0e-4, 1.0e-5])
def test_a_shape_whose_optimum_lies_just_beyond_the_wall_is_disclosed(inner_tol):
    # the unrestricted shape is about 1.0000067, so a loose stop leaves a step of only about 0.49
    u = (np.arange(2_000) + 0.5) / 2_000
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {})),
    )
    with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
        model.fit(
            pd.DataFrame({"x": np.zeros(2_000)}), (u / (1 - u)) ** 1.00035, inner_tol=inner_tol
        )
    assert len(_wall_messages(model)) == 1


@pytest.mark.parametrize("heavy, wall", [(1.5, "upper wall 1"), (-0.3, "lower wall 0")])
def test_a_pressed_group_whose_direction_the_rank_decision_drops_is_disclosed(heavy, wall):
    # under REML the pressed group, coded 1 beside an interior group coded 0.7, runs so close to
    # the wall that the terminal rank decision drops its direction
    rng = np.random.default_rng(8)
    pressed = np.expm1(-heavy * np.log(rng.random(1_000))) / heavy
    interior = np.expm1(-0.3 * np.log(rng.random(1_000))) / 0.3
    frame = pd.DataFrame({"g": np.repeat([1.0, 0.7], 1_000), "x": rng.uniform(size=2_000)})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(
            Predictor("scale", {"x": Spline(kind="cr", k=6), "g": Numeric()}),
            Predictor("shape", {"g": Numeric()}),
        ),
    )
    with pytest.warns(ConvergenceWarning, match=wall):
        model.fit_reml(frame, np.concatenate([pressed, interior]))
    assert model.result_.rank < len(model.result_.coefficients)
    (message,) = _wall_messages(model)
    assert "1000 of 2000 rows" in message and "the data put the shape" in message


def test_a_shape_at_the_last_float_beside_a_scale_alias_keeps_its_caveat():
    # the coefficient fit drops z2 - 2 z, which the design aliases, and the shape at the last
    # float below one, which only the link flattens: zero is no optimum for the second, so
    # neither is refitted. Both are forced: the alias is exact, and at the last float the
    # shape's equilibrated curvature is about 1e-2 of the rank cut, where a shape the wall step
    # stops short of it sits at the cut. Mutation check: refitting a span that holds an alias,
    # and keeping the refit, loses the caveat.
    rng = np.random.default_rng(11)
    y = stats.genpareto(3.0).rvs(2_000, random_state=rng)
    frame = pd.DataFrame({"x": rng.uniform(size=2_000), "z": rng.uniform(size=2_000)})
    frame["z2"] = 2.0 * frame["z"]
    scale = {"x": Spline(kind="cr", k=6), "z": Numeric(), "z2": Numeric()}
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", scale), Predictor("shape", {})),
    )
    with pytest.warns(ConvergenceWarning, match="float64 can represent"):
        model.fit_reml(frame, y, inner_tol=1.0e-12)
    assert model.result_.coefficient_converged
    assert model.result_.rank == len(model.result_.coefficients) - 2
    assert np.all(model.predict_parameters(frame)["shape"].to_numpy() == np.nextafter(1.0, 0.0))
    (message,) = _wall_messages(model)
    assert "not reliable" in message


@pytest.mark.parametrize("discrete", [False, True], ids=["dense", "chunked"])
def test_a_group_code_far_from_zero_fits_the_shapes_of_the_centred_code(discrete):
    # coded 1000 + {1, 0.7}, the groups' shape slope is collinear with the intercept only in
    # uncentred coordinates: the rank drops it with the pressed group's wall, and a refit
    # without it puts both groups at one shape, 0.886, which the data refute
    rng = np.random.default_rng(8)
    pressed = np.expm1(-1.5 * np.log(rng.random(1_000))) / 1.5
    interior = np.expm1(-0.3 * np.log(rng.random(1_000))) / 0.3
    y = np.concatenate([pressed, interior])

    def fitted(shift):
        # 1001 - 1000 and 1000.7 - 1000 are exact (Sterbenz), so both fits read the same data
        frame = pd.DataFrame({"g": np.repeat([1001.0, 1000.7], 1_000) - shift})
        model = model_from_templates(
            family=GeneralizedParetoLSS(),
            predictors=(Predictor("scale", {"g": Numeric()}), Predictor("shape", {"g": Numeric()})),
            discrete=discrete,
        )
        with pytest.warns(ConvergenceWarning, match="at the upper wall 1"):
            model.fit(frame, y, inner_tol=1.0e-7)
        return model, model.predict_parameters(frame.iloc[[0, 1_000]])["shape"].to_numpy()

    (offset, shapes), (centred, reference) = fitted(0.0), fitted(1000.0)
    assert offset.result_.coefficient_converged
    (message,) = _wall_messages(offset)
    assert "1000 of 2000 rows" in message
    # The score stop bounds each coefficient's score by tol (1 + |f|); a group's score is
    # that times the code's leverage on the intercept, (1001 + 1) / 0.3, and moves the
    # interior shape by the score over n I(xi) dxi/deta, I = 2 / ((1 + xi)(1 + 2 xi)) the
    # shape's Fisher information per row and xi (1 - xi) the bounded logit's slope.
    xi = reference[1]
    fisher = 2.0 / ((1.0 + xi) * (1.0 + 2.0 * xi))
    stop = 1.0e-7 * (1.0 + abs(centred.result_.log_likelihood)) * 1002.0 / 0.3
    assert shapes[1] == pytest.approx(xi, abs=stop / (1_000 * fisher * xi * (1.0 - xi)))
    assert shapes[0] == pytest.approx(reference[0], abs=1.0e-6)  # both pressed onto the wall


def test_near_collinear_shape_covariates_at_an_interior_optimum_are_not_disclosed():
    # the rank decision drops the direction in which the two columns nearly cancel; the design,
    # not the link, loses its curvature there
    x = np.linspace(-1.0, 1.0, 5_000)
    shape = expit(10.0 * x)
    y = np.expm1(-shape * np.log(np.random.default_rng(4).random(5_000))) / shape
    frame = pd.DataFrame(
        {"x": x, "x2": x + 1.0e-10 * np.random.default_rng(5).standard_normal(5_000)}
    )
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {}), Predictor("shape", {"x": Numeric(), "x2": Numeric()})),
    ).fit(frame, y)
    assert model.result_.rank < len(model.result_.coefficients)
    assert not _wall_messages(model)


def test_a_zero_derivative_clip_with_a_dropped_shape_direction_keeps_the_standard_error_caveat():
    from superglm.distributional.fit_state import _wall_disclosure

    # Start at the event: g=0 is interior and g=1 has eta=40+logit(0.3). The
    # coefficients are supplied because the default least-squares start absorbs
    # even this row-varying offset in the group coefficient.
    u = (np.arange(200) + 0.5) / 200
    y = np.tile(np.expm1(-0.3 * np.log(u)) / 0.3, 2)
    g = np.repeat([0.0, 1.0], 200)
    family = GeneralizedParetoLSS()
    weights = resolved_prior(np.ones(len(y)))
    layout = build_stacked_layout(
        compile_predictors(
            as_eager_frame(pd.DataFrame({"g": g})),
            weights,
            family.parameters,
            (Predictor("scale", {}), Predictor("shape", {"g": Numeric()})),
            offsets={"shape": 40.0 * g},
        )
    )
    initial = np.array([0.0, np.log(0.3 / 0.7), 0.0])
    link = layout.predictor("shape").link
    initial_eta = initial[1] + 40.0 * g
    assert np.all(link.inverse(initial_eta[g == 1.0]) == np.nextafter(1.0, 0.0))
    assert np.all(link.deriv_inverse(initial_eta[g == 1.0]) == 0.0)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        family.bind_likelihood(y, weights, COMPLETE_OBSERVATION),
        np.zeros((3, 3)),
        initial=initial,
    )
    assert result.converged
    assert result.terminal_rank.rank == 2 < len(result.coefficients)
    assert np.all(result.theta[g == 1.0, 1] == np.nextafter(1.0, 0.0))
    assert np.all((result.theta[g == 0.0, 1] > 0.1) & (result.theta[g == 0.0, 1] < 0.9))
    message = _wall_disclosure(family, result, layout)
    assert "200 of 400 rows" in message
    assert "float64 can represent" in message and "not reliable" in message


def test_a_pressed_row_clipped_below_a_narrow_wall_keeps_the_standard_error_caveat():
    # below an upper wall of 0.35 the link reaches its last float while its derivative is
    # still positive, so no direction is dropped; the clip is forced on the pressed fit
    from superglm.distributional.fit_state import _wall_disclosure

    frame, y, _ = _simulate(3_000, 8, shape=0.9)
    family = GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.35)
    with pytest.warns(ConvergenceWarning, match="at the upper wall 0.35"):
        model = _constant_shape_fit(family, frame, y)
    state = model._require_fitted().fit_state
    theta = state.solver_result.theta.copy()
    theta[:, 1] = np.nextafter(0.35, 0.0)
    clipped = dataclasses.replace(state.solver_result, theta=theta)
    message = _wall_disclosure(family, clipped, state.layout)
    assert "float64 can represent" in message and "not reliable" in message


def test_a_scale_alias_beside_an_interior_shape_near_a_wall_is_not_disclosed():
    # the rank decision drops the direction in which two scale columns nearly cancel; it lies
    # in the shape coefficients only on round-off
    rng = np.random.default_rng(101)
    x = rng.uniform(size=4_000)
    y = stats.genpareto(0.008).rvs(4_000, random_state=rng) * np.exp(0.5 * x)
    frame = pd.DataFrame({"x": x, "x2": x + 1.0e-10 * rng.standard_normal(4_000)})
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric(), "x2": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    assert model.result_.rank < len(model.result_.coefficients)
    assert float(model.predict_parameters(frame.iloc[:1])["shape"].iloc[0]) < 1.0e-2
    assert not _wall_messages(model)


def test_a_pressed_fit_without_positive_shape_information_is_still_disclosed():
    # near-constant excesses: the likelihood falls across the whole positive shape range,
    # and the observed shape information at the wall is negative
    y = np.random.default_rng(0).uniform(0.9, 1.1, 100)
    frame = pd.DataFrame({"x": np.zeros(100)})
    with pytest.warns(ConvergenceWarning, match="at the lower wall 0"):
        model = model_from_templates(
            family=GeneralizedParetoLSS(),
            predictors=(Predictor("scale", {}), Predictor("shape", {})),
        ).fit(frame, y)
    assert len(_wall_messages(model)) == 1


def test_an_interior_shape_records_no_wall_disclosure():
    frame, y, _ = _simulate(3_000, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _constant_shape_fit(GeneralizedParetoLSS(), frame, y)
    assert not _wall_messages(model)
    assert not any("tail shape at the" in str(warning.message) for warning in caught)


def test_the_density_agrees_with_scipy_genpareto_at_the_fitted_parameters():
    frame, y, _ = _simulate(1_000, 13)
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    theta = model.predict_parameters(frame).to_numpy()
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours = family.evaluate_natural(y, theta, plan, derivative_order=0).reported_log_likelihood
    reference = stats.genpareto.logpdf(y, c=theta[:, 1], scale=theta[:, 0])
    assert np.allclose(ours, reference, rtol=0, atol=1e-12)


def test_the_threshold_splice_recipe_composes_a_tail_probability():
    """The documented recipe: body below u, exceedance rate, GPD on the excesses above."""
    rng = np.random.default_rng(19)
    n = 8_000
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    # a genuinely Pareto-tailed book: the generalized Pareto is threshold-stable, so the
    # excesses over any threshold keep the same shape and the tail fit is well posed
    scale = np.exp(0.2 + 0.3 * frame["x"].to_numpy())
    losses = scale * np.expm1(-0.3 * np.log(rng.random(n))) / 0.3
    threshold = float(np.quantile(losses, 0.9))
    above = losses > threshold
    excess = losses[above] - threshold
    tail = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame.loc[above], excess)
    exceedance_rate = float(np.mean(above))
    survival = exceedance_rate * (1.0 - tail.predict_cdf(frame.loc[above], excess))
    assert np.all(survival >= 0.0) and np.all(survival <= exceedance_rate)
    assert float(np.mean(survival)) == pytest.approx(exceedance_rate / 2.0, abs=0.02)


def test_the_kernel_at_zero_shape_is_the_exponential_and_the_family_cannot_reach_it():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    rows = gp.scale_rows(y, np.full(4, 2.0), np.zeros(4), np.ones(4), derivative_order=0)
    # measured: the difference is exactly zero, so 1e-15 is a version-robustness margin only
    assert np.allclose(
        rows.optimizing_log_likelihood, stats.expon(scale=2.0).logpdf(y), rtol=0, atol=1e-15
    )
    with pytest.raises(ValueError):
        GeneralizedParetoLSS().cdf(y, np.tile([2.0, 0.0], (4, 1)))


import json  # noqa: E402
import subprocess  # noqa: E402
from functools import cache  # noqa: E402

from tests._r_harness import ROOT, r_environment, require_r_harness  # noqa: E402

_GP_PROBE = (
    "suppressMessages(library(gamlss)); suppressMessages(library(jsonlite)); "
    "cat(if (exists('GP')) 'ok' else 'missing')"
)


@cache
def _gamlss_gp_available() -> bool:
    completed = subprocess.run(
        ["Rscript", "-e", _GP_PROBE],
        cwd=ROOT,
        env=r_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip().endswith("ok")


_GP_SCRIPT = """
suppressMessages(library(gamlss)); suppressMessages(library(jsonlite))
d <- read.csv(file("stdin"))
fit <- gamlss(y ~ x, sigma.formula = ~ 1, family = GP, data = d,
              control = gamlss.control(c.crit = 1e-9, n.cyc = 400, trace = FALSE))
cat(toJSON(list(mu = as.numeric(coef(fit, what = "mu")),
                sigma = as.numeric(coef(fit, what = "sigma")),
                loglik = as.numeric(logLik(fit))), digits = 16))
"""


def test_the_fit_matches_gamlss_gp_at_a_parametric_specification():
    """gamlss GP is the Lomax form of the GPD: mu = psi/xi, sigma = 1/xi, both on log links.

    The mapping is pinned here, not assumed.  The log-likelihood check is
    mapping-free: if it agrees and the parameter checks do not, the mapping
    constant is what changed and must be re-derived from the gamlss.dist manual,
    never absorbed into a looser tolerance.
    """
    require_r_harness()
    if not _gamlss_gp_available():
        pytest.skip("requires R with gamlss (providing GP) and jsonlite")
    frame, y, _ = _simulate(1_500, 12, shape=0.4)
    payload = pd.DataFrame({"x": frame["x"], "y": y}).to_csv(index=False)
    completed = subprocess.run(
        ["Rscript", "-e", _GP_SCRIPT],
        input=payload,
        cwd=ROOT,
        env=r_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    reference = json.loads(completed.stdout)
    # jsonlite boxes every scalar in a length-one array unless auto_unbox is set
    reference_log_likelihood = float(np.ravel(reference["loglik"])[0])
    model = model_from_templates(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    theta = model.predict_parameters(frame)
    family = GeneralizedParetoLSS()
    weights = resolve_likelihood_weights(
        np.ones(len(y)), n_observations=len(y), contract=WeightContract(semantics="frequency")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    ours_log_likelihood = float(
        np.sum(
            family.evaluate_natural(
                y, theta.to_numpy(), plan, derivative_order=0
            ).reported_log_likelihood
        )
    )
    assert abs(ours_log_likelihood - reference_log_likelihood) <= 1e-8 * (
        1 + abs(reference_log_likelihood)
    )

    shape = float(theta["shape"].iloc[0])
    log_scale = np.log(theta["scale"].to_numpy())
    slope, intercept = np.polyfit(frame["x"].to_numpy(), log_scale, 1)
    # log mu = log psi - log xi with a constant xi, so the slope is shared and the intercept shifts
    assert abs(float(slope) - reference["mu"][1]) <= 1e-5 * (1 + abs(reference["mu"][1]))
    assert abs(float(intercept) - (reference["mu"][0] + math.log(shape))) <= 1e-5 * (
        1 + abs(intercept)
    )
    assert abs(shape - 1.0 / math.exp(reference["sigma"][0])) <= 1e-5 * (1 + shape)
