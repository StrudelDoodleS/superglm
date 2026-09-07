"""Adapter-contract tests for the two two-piece families."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families._links import BoundedLogitLink
from superglm.distributional.families.two_piece import TwoPieceLogNormalLSS, TwoPieceNormalLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import two_piece as tp
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric
from superglm.links import IdentityLink, LogLink


def _weights(values, semantics):
    values = np.asarray(values, dtype=float)
    return resolve_likelihood_weights(
        values, n_observations=len(values), contract=WeightContract(semantics=semantics)
    )


def _bind(family, y, values, semantics):
    return family.bind_likelihood(y, _weights(values, semantics), COMPLETE_OBSERVATION)


class _ExplodingFeature(Numeric):
    def build(self, x, sample_weight=None):
        raise AssertionError("predictor geometry must not be built before the weight refusal")


def test_parameters_links_and_config_follow_the_parametrisation():
    mean_form = TwoPieceLogNormalLSS()
    assert [p.name for p in mean_form.parameters] == ["mean", "scale", "skew"]
    assert [p.role for p in mean_form.parameters] == ["mean", "scale", "shape"]
    assert isinstance(mean_form.parameters[0].default_link, LogLink)
    skew_spec = mean_form.parameters[2]
    assert isinstance(skew_spec.default_link, BoundedLogitLink)
    assert (skew_spec.support.lower, skew_spec.support.upper) == (-0.9, 0.9)
    inside = skew_spec.default_link.inverse(np.array([-20.0, 0.0, 20.0]))
    assert np.all(skew_spec.support.contains(inside))
    location_form = TwoPieceLogNormalLSS(parametrisation="location")
    assert [p.name for p in location_form.parameters] == ["location", "scale", "skew"]
    assert isinstance(location_form.parameters[0].default_link, IdentityLink)
    assert mean_form.to_config() == {
        "type": "TwoPieceLogNormalLSS",
        "parametrisation": "mean",
        "scale_floor": 0.01,
        "skew_bound": 0.9,
    }
    real_line = TwoPieceNormalLSS()
    assert [p.name for p in real_line.parameters] == ["location", "scale", "skew"]
    assert real_line.to_config() == {
        "type": "TwoPieceNormalLSS",
        "scale_floor": 0.01,
        "skew_bound": 0.9,
    }
    for bad in (
        {"parametrisation": "median"},
        {"scale_floor": -1.0},
        {"skew_bound": 1.0},
        {"skew_bound": 0.0},
    ):
        with pytest.raises(ValueError):
            TwoPieceLogNormalLSS(**bad)
    with pytest.raises(ValueError):
        TwoPieceNormalLSS(skew_bound=1.5)


def test_the_log_normal_variant_refuses_a_nonpositive_response():
    family = TwoPieceLogNormalLSS()
    with pytest.raises(ValueError, match="strictly positive"):
        _bind(family, np.array([1.0, 0.0]), np.ones(2), "frequency")
    real_line = TwoPieceNormalLSS()
    plan = _bind(real_line, np.array([-3.0, 0.0, 2.0]), np.ones(3), "frequency")
    assert plan.exact_response.tolist() == [-3.0, 0.0, 2.0]


@pytest.mark.parametrize("factory", [TwoPieceLogNormalLSS, TwoPieceNormalLSS])
def test_nonunit_prior_weights_are_refused_and_unit_prior_bridges_to_frequency(factory):
    family = factory()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        _bind(family, y, [1.0, 2.0, 1.0, 1.0], "prior")
    plan = _bind(family, y, np.ones(4), "prior")
    assert plan.row_law == "unit-prior-explicitly-equals-frequency/v1"
    frequency = _bind(family, y, np.ones(4), "frequency")
    assert frequency.row_law == "two-piece-epsilon-skew-literal-replication/v1"
    theta = family.initialize(y, plan).theta
    a = family.evaluate_natural(y, theta, plan)
    b = family.evaluate_natural(y, theta, frequency)
    assert np.array_equal(a.reported_log_likelihood, b.reported_log_likelihood)
    assert np.array_equal(a.hessian_packed, b.hessian_packed)


@pytest.mark.parametrize(
    ("factory", "response", "row"),
    [
        # the mean-form log-normal arm, and the real-line arm on a signed
        # response with a location-form theta -- the only arm whose
        # parameter-independent carrier is scaled by the multiplier
        (TwoPieceLogNormalLSS, [0.7, 1.9, 4.2], [2.0, 0.8, 0.4]),
        (TwoPieceNormalLSS, [-1.3, 0.4, 2.6], [0.5, 0.9, -0.3]),
    ],
)
def test_integer_frequency_rows_equal_literal_replication(factory, response, row):
    family = factory()
    y = np.array(response)
    counts = np.array([1.0, 3.0, 2.0])
    plan = _bind(family, y, counts, "frequency")
    theta = np.tile(row, (3, 1))
    weighted = family.evaluate_natural(y, theta, plan)
    replicated_y = np.repeat(y, counts.astype(int))
    replicated_plan = _bind(family, replicated_y, np.ones(len(replicated_y)), "frequency")
    unit = family.evaluate_natural(
        replicated_y, np.tile(row, (len(replicated_y), 1)), replicated_plan
    )
    for index, (weighted_row, count) in enumerate(
        zip(weighted.reported_log_likelihood, counts, strict=True)
    ):
        start = int(np.sum(counts[:index]))
        assert weighted_row == pytest.approx(
            float(np.sum(unit.reported_log_likelihood[start : start + int(count)])), abs=1e-12
        )


def test_a_narrow_skew_bound_clamps_the_start_inside_its_own_walls():
    """``skew_bound`` reaches the initialiser, so the clamp and the link's walls agree."""
    rng = np.random.default_rng(404)
    n = 800
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    # exponential-tailed log severity: its log-scale sample skewness is 1.314,
    # past the reach of any |eps| <= 0.5 two-piece law (0.6949; the standard
    # skewness saturates below 1 even at eps = 0.9), so the start is clamped
    y = np.exp(rng.exponential(1.0, n) + 0.5 * frame["x"].to_numpy())
    reach = float(tp.standard_skewness(np.array([0.5 - 1e-6]))[0])
    assert float(pd.Series(np.log(y)).skew()) > reach
    with pytest.warns(tp.TwoPieceInitializationWarning, match="outside the two-piece range"):
        model = SuperLSS(
            family=TwoPieceLogNormalLSS(skew_bound=0.5),
            predictors=(
                Predictor("mean", {"x": Numeric()}),
                Predictor("scale", {}),
                Predictor("skew", {}),
            ),
        ).fit(frame, y)
    skew = model.predict_parameters(frame)["skew"].to_numpy()
    assert np.all(np.abs(skew) < 0.5)
    assert np.all(np.isfinite(model.predict(frame)))


def test_expected_information_under_frequency_equals_the_replicated_rows_summed():
    """coefficient_curvature="fisher" reads this; the multiplier law must hold here too."""
    family = TwoPieceLogNormalLSS()
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    plan = _bind(family, y, counts, "frequency")
    theta = np.tile([2.0, 0.8, 0.4], (3, 1))
    weighted = family.expected_information_natural(theta, plan)
    replicated_y = np.repeat(y, counts.astype(int))
    replicated_plan = _bind(family, replicated_y, np.ones(len(replicated_y)), "frequency")
    unit = family.expected_information_natural(
        np.tile([2.0, 0.8, 0.4], (len(replicated_y), 1)), replicated_plan
    )
    for index, count in enumerate(counts):
        start = int(np.sum(counts[:index]))
        assert np.allclose(
            weighted[index], unit[start : start + int(count)].sum(axis=0), rtol=0, atol=0
        )


def test_exactly_the_requested_derivative_order_is_returned_and_the_plan_is_typed():
    family = TwoPieceLogNormalLSS()
    y = np.array([0.5, 1.5, 2.5])
    plan = _bind(family, y, np.ones(3), "frequency")
    theta = np.tile([1.5, 0.8, -0.3], (3, 1))
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        family.evaluate_natural(y, theta, plan, derivative_order=True)
    other = TwoPieceLogNormalLSS(parametrisation="location")
    with pytest.raises(UnsupportedLikelihoodContractError):
        other.evaluate_natural(y, theta, plan)
    with pytest.raises(UnsupportedLikelihoodContractError):
        TwoPieceNormalLSS().evaluate_natural(y, theta, plan)


def test_plan_take_preserves_root_identity_and_changes_the_identifier():
    family = TwoPieceLogNormalLSS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    plan = _bind(family, y, np.ones(4), "frequency")
    child = plan.take(np.array([3, 1], dtype=np.intp))
    assert child.weights.root_digest == plan.weights.root_digest
    assert child.plan_identifier != plan.plan_identifier
    assert child.exact_response.tolist() == [4.0, 1.5]
    assert not child.parameter_independent_carrier.flags.writeable
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.validate_likelihood_plan(np.array([0.5, 1.5, 2.5, 9.0]), plan)


def test_evaluate_natural_matches_the_kernel_in_both_forms_and_both_families():
    y = np.array([0.5, 1.5])
    mean_form = TwoPieceLogNormalLSS()
    theta = np.array([[2.0, 0.8, 0.4], [3.0, 1.2, -0.5]])
    evaluated = mean_form.evaluate_natural(y, theta, _bind(mean_form, y, np.ones(2), "frequency"))
    kernel = tp.mean_rows(y, theta[:, 0], theta[:, 1], theta[:, 2], np.ones(2), derivative_order=2)
    assert np.array_equal(evaluated.optimizing_log_likelihood, kernel.optimizing_log_likelihood)
    assert np.array_equal(evaluated.hessian_packed, kernel.hessian_packed)
    location_form = TwoPieceLogNormalLSS(parametrisation="location")
    located = location_form.evaluate_natural(
        y, theta, _bind(location_form, y, np.ones(2), "frequency")
    )
    log_kernel = tp.location_rows(
        np.log(y), theta[:, 0], theta[:, 1], theta[:, 2], np.ones(2), derivative_order=2
    )
    assert np.array_equal(located.score, log_kernel.score)
    real_line = TwoPieceNormalLSS()
    t = np.array([-1.0, 2.0])
    on_line = real_line.evaluate_natural(t, theta, _bind(real_line, t, np.ones(2), "frequency"))
    line_kernel = tp.location_rows(
        t, theta[:, 0], theta[:, 1], theta[:, 2], np.ones(2), derivative_order=2
    )
    assert np.array_equal(on_line.score, line_kernel.score)
    assert np.allclose(
        on_line.reported_log_likelihood,
        line_kernel.optimizing_log_likelihood - 0.5 * math.log(2.0 * math.pi),
        rtol=0,
        atol=1e-15,
    )


def test_default_prediction_cdf_quantile_and_information_in_every_form():
    y = np.array([0.5, 1.5, 6.0])
    theta_mean = np.tile([2.0, 0.8, 0.4], (3, 1))
    mean_form = TwoPieceLogNormalLSS()
    plan = _bind(mean_form, y, np.ones(3), "frequency")
    assert np.array_equal(mean_form.default_prediction(theta_mean), theta_mean[:, 0])
    location = tp.location_of_mean(theta_mean[:, 0], theta_mean[:, 1], theta_mean[:, 2])
    theta_location = np.column_stack((location, theta_mean[:, 1], theta_mean[:, 2]))
    location_form = TwoPieceLogNormalLSS(parametrisation="location")
    assert np.allclose(location_form.default_prediction(theta_location), 2.0, rtol=0, atol=1e-13)
    p = np.array([0.1, 0.5, 0.9])
    quantiles = mean_form.quantile(p, theta_mean)
    assert np.all(quantiles > 0.0)
    assert np.allclose(mean_form.cdf(quantiles, theta_mean), p, rtol=0, atol=1e-12)
    assert np.allclose(location_form.quantile(p, theta_location), quantiles, rtol=1e-13, atol=0)
    information = mean_form.expected_information_natural(theta_mean, plan)
    assert np.array_equal(
        information,
        tp.mean_expected_information(
            theta_mean[:, 0], theta_mean[:, 1], theta_mean[:, 2], np.ones(3)
        ),
    )
    real_line = TwoPieceNormalLSS()
    assert np.allclose(
        real_line.default_prediction(theta_location),
        tp.real_line_mean(theta_location[:, 0], theta_location[:, 1], theta_location[:, 2]),
        rtol=0,
        atol=0,
    )
    t = real_line.quantile(p, theta_location)
    assert np.allclose(real_line.cdf(t, theta_location), p, rtol=0, atol=1e-12)


def test_refusal_precedes_geometry_initialization_and_evaluation():
    family = TwoPieceLogNormalLSS()
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        fit_dense_distributional(
            frame,
            np.array([0.4, 1.1, 0.9, 2.3, 1.7, 3.1]),
            family=family,
            predictors=(
                Predictor("mean", {"x": _ExplodingFeature()}),
                Predictor("scale", {}),
                Predictor("skew", {}),
            ),
            sample_weight=np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25]),
            weight_contract=WeightContract(semantics="prior"),
        )


@pytest.mark.parametrize(
    ("family", "config"),
    [
        (
            TwoPieceLogNormalLSS(),
            {
                "type": "TwoPieceLogNormalLSS",
                "parametrisation": "mean",
                "scale_floor": 0.01,
                "skew_bound": 0.9,
            },
        ),
        (
            TwoPieceNormalLSS(scale_floor=0.05, skew_bound=0.75),
            {"type": "TwoPieceNormalLSS", "scale_floor": 0.05, "skew_bound": 0.75},
        ),
    ],
)
def test_artifact_round_trip_preserves_predictions_and_config(family, config):
    rng = np.random.default_rng(9)
    n = 300
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, n)})
    variate = tp.two_piece_quantile(
        rng.uniform(size=n), 0.3 + 0.5 * frame["x"].to_numpy(), np.full(n, 0.6), np.full(n, 0.35)
    )
    y = np.exp(variate) if isinstance(family, TwoPieceLogNormalLSS) else variate
    first = "mean" if family.to_config()["type"] == "TwoPieceLogNormalLSS" else "location"
    model = SuperLSS(
        family=family,
        predictors=(
            Predictor(first, {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("skew", {}),
        ),
    ).fit(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame), model.predict(frame))
    assert np.array_equal(
        restored.predict_quantile(frame, 0.95), model.predict_quantile(frame, 0.95)
    )
    assert restored.family.to_config() == config
