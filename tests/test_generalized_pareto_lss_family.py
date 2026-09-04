"""Adapter-contract tests for ``GeneralizedParetoLSS``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import SuperLSS
from superglm.distributional import DistributionFunctionFamily, GammaLS, Predictor
from superglm.distributional.families._links import BoundedLogitLink
from superglm.distributional.families.generalized_pareto import GeneralizedParetoLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import generalized_pareto as gp
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric


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


def test_parameters_links_supports_and_config():
    family = GeneralizedParetoLSS()
    assert [p.name for p in family.parameters] == ["scale", "shape"]
    assert [p.role for p in family.parameters] == ["scale", "shape"]
    assert family.parameters[0].support.lower == 0.0
    assert family.parameters[1].support.lower == 0.0 and family.parameters[1].support.upper == 1.0
    assert isinstance(family.parameters[1].default_link, BoundedLogitLink)
    assert family.parameters[1].default_link == BoundedLogitLink(0.0, 1.0)
    assert all(p.curvature == "observed" for p in family.parameters)
    assert family.default_prediction_name == "conditional_mean"
    assert family.to_config() == {
        "type": "GeneralizedParetoLSS",
        "shape_lower": 0.0,
        "shape_upper": 1.0,
    }
    capabilities = family.capabilities
    assert capabilities.expected_information and capabilities.cdf and capabilities.quantile
    assert capabilities.response_mean and capabilities.max_derivative_order == 2
    assert isinstance(family, DistributionFunctionFamily)


def test_a_negative_lower_wall_is_refused_and_names_the_support_slot_follow_up():
    with pytest.raises(ValueError) as failure:
        GeneralizedParetoLSS(shape_lower=-0.5)
    message = str(failure.value)
    assert "response-dependent support" in message
    assert "shape_lower" in message


def test_the_walls_must_be_ordered_inside_the_unit_interval():
    for bad in (
        {"shape_upper": 1.5},
        {"shape_lower": 0.6, "shape_upper": 0.4},
        {"shape_lower": 0.5, "shape_upper": 0.5},
        {"shape_lower": float("nan")},
        {"shape_upper": float("inf")},
    ):
        with pytest.raises(ValueError, match="shape"):
            GeneralizedParetoLSS(**bad)
    narrow = GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.4)
    assert narrow.parameters[1].default_link == BoundedLogitLink(0.05, 0.4)
    assert narrow.to_config()["shape_upper"] == 0.4


def test_nonunit_prior_weights_are_refused_and_unit_prior_bridges_to_frequency():
    family = GeneralizedParetoLSS()
    y = np.array([0.0, 1.5, 2.5, 4.0])
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        _bind(family, y, [1.0, 2.0, 1.0, 1.0], "prior")
    plan = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "prior")
    assert plan.row_law == "unit-prior-explicitly-equals-frequency/v1"
    frequency = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "frequency")
    assert frequency.row_law == "gpd-excess-literal-replication/v1"
    theta = family.initialize(y, plan).theta
    a = family.evaluate_natural(y, theta, plan)
    b = family.evaluate_natural(y, theta, frequency)
    assert np.array_equal(a.reported_log_likelihood, b.reported_log_likelihood)
    assert np.array_equal(a.hessian_packed, b.hessian_packed)


def test_integer_frequency_rows_equal_literal_replication():
    family = GeneralizedParetoLSS()
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    plan = _bind(family, y, counts, "frequency")
    theta = np.tile([2.0, 0.4], (3, 1))
    weighted = family.evaluate_natural(y, theta, plan)
    replicated_y = np.repeat(y, counts.astype(int))
    replicated_plan = _bind(family, replicated_y, np.ones(len(replicated_y)), "frequency")
    unit = family.evaluate_natural(
        replicated_y, np.tile([2.0, 0.4], (len(replicated_y), 1)), replicated_plan
    )
    for index, (weighted_row, count) in enumerate(
        zip(weighted.reported_log_likelihood, counts, strict=True)
    ):
        start = int(np.sum(counts[:index]))
        assert np.isclose(
            weighted_row,
            np.sum(unit.reported_log_likelihood[start : start + int(count)]),
            rtol=0,
            atol=1e-12,
        )


def test_the_response_must_be_a_non_negative_excess():
    family = GeneralizedParetoLSS()
    # validated_float_response raises a plain ValueError, not the contract error
    for bad in (np.array([1.0, -1e-12]), np.array([1.0, np.nan]), np.array([])):
        with pytest.raises(ValueError):
            _bind(family, bad, np.ones(max(len(bad), 1)), "frequency")
    plan = _bind(family, np.array([0.0, 1.0]), np.ones(2), "frequency")
    assert plan.exact_response.tolist() == [0.0, 1.0]  # a zero excess is a legitimate row


def test_exactly_the_requested_derivative_order_is_returned():
    family = GeneralizedParetoLSS()
    y = np.array([0.5, 1.5, 2.5])
    plan = _bind(family, y, np.ones(3), "frequency")
    theta = np.tile([1.5, 0.3], (3, 1))
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        family.evaluate_natural(y, theta, plan, derivative_order=True)


def test_plan_take_preserves_root_identity_and_changes_the_identifier():
    family = GeneralizedParetoLSS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    plan = _bind(family, y, np.ones(4), "frequency")
    child = plan.take(np.array([3, 1], dtype=np.intp))
    assert child.weights.root_digest == plan.weights.root_digest
    assert child.plan_identifier != plan.plan_identifier
    assert child.exact_response.tolist() == [4.0, 1.5]
    assert not child.parameter_independent_carrier.flags.writeable
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.validate_likelihood_plan(np.array([0.5, 1.5, 2.5, 9.0]), plan)
    other = GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.4)
    with pytest.raises(UnsupportedLikelihoodContractError, match="another configuration"):
        other.validate_likelihood_plan(y, plan)


def test_the_carrier_is_zero_because_the_optimising_likelihood_is_the_whole_density():
    family = GeneralizedParetoLSS()
    y = np.array([0.5, 1.5])
    plan = _bind(family, y, np.array([1.0, 3.0]), "frequency")
    assert np.array_equal(plan.parameter_independent_carrier, np.zeros(2))
    theta = np.tile([2.0, 0.4], (2, 1))
    evaluated = family.evaluate_natural(y, theta, plan)
    scipy_rows = stats.genpareto.logpdf(y, c=0.4, scale=2.0) * np.array([1.0, 3.0])
    assert np.allclose(evaluated.reported_log_likelihood, scipy_rows, rtol=0, atol=1e-13)


def test_evaluate_natural_matches_the_kernel_and_expected_information_is_the_closed_form():
    family = GeneralizedParetoLSS()
    y = np.array([0.5, 1.5, 6.0])
    plan = _bind(family, y, np.ones(3), "frequency")
    theta = np.tile([2.0, 0.4], (3, 1))
    evaluated = family.evaluate_natural(y, theta, plan)
    kernel = gp.scale_rows(y, theta[:, 0], theta[:, 1], np.ones(3), derivative_order=2)
    assert np.array_equal(evaluated.optimizing_log_likelihood, kernel.optimizing_log_likelihood)
    assert np.array_equal(evaluated.score, kernel.score)
    assert np.array_equal(evaluated.hessian_packed, kernel.hessian_packed)
    assert evaluated.valid.tolist() == [True, True, True]
    information = family.expected_information_natural(theta, plan)
    assert np.array_equal(
        information, gp.expected_information(theta[:, 0], theta[:, 1], np.ones(3))
    )


def test_default_prediction_cdf_and_quantile():
    family = GeneralizedParetoLSS()
    theta = np.array([[2.0, 0.4], [1.0, 0.1], [3.0, 0.75]])
    assert np.allclose(
        family.default_prediction(theta), theta[:, 0] / (1.0 - theta[:, 1]), rtol=0, atol=1e-15
    )
    p = np.array([0.1, 0.5, 0.9])
    quantiles = family.quantile(p, theta)
    assert np.allclose(family.cdf(quantiles, theta), p, rtol=0, atol=1e-13)
    scalar = family.quantile(0.9, theta)
    assert scalar.shape == (3,)
    assert np.allclose(
        scalar, stats.genpareto.ppf(0.9, c=theta[:, 1], scale=theta[:, 0]), rtol=1e-12, atol=0
    )
    assert not family.cdf(np.array([1.0, 1.0, 1.0]), theta).flags.writeable


def test_the_shape_walls_are_enforced_on_the_parameter_matrix():
    family = GeneralizedParetoLSS()
    plan = _bind(family, np.array([1.0]), np.ones(1), "frequency")
    for bad_shape in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            family.evaluate_natural(np.array([1.0]), np.array([[2.0, bad_shape]]), plan)
    with pytest.raises(ValueError):
        family.evaluate_natural(np.array([1.0]), np.array([[0.0, 0.4]]), plan)


def test_the_exponential_limit_agrees_with_gamma_at_unit_coefficient_of_variation():
    y = np.array([0.4, 1.3, 2.8, 7.5])
    weights = _weights(np.ones(4), "frequency")
    gamma = GammaLS()
    pareto = GeneralizedParetoLSS()
    gamma_rows = gamma.evaluate_natural(
        y, np.tile([2.0, 1.0], (4, 1)), gamma.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    pareto_rows = pareto.evaluate_natural(
        y, np.tile([2.0, 1.0e-12], (4, 1)), pareto.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    )
    assert np.allclose(
        gamma_rows.reported_log_likelihood, pareto_rows.reported_log_likelihood, rtol=0, atol=1e-9
    )
    assert np.allclose(gamma_rows.score[:, 0], pareto_rows.score[:, 0], rtol=0, atol=1e-9)


def test_refusal_precedes_geometry_initialization_and_evaluation():
    family = GeneralizedParetoLSS()
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        fit_dense_distributional(
            frame,
            np.array([0.4, 1.1, 0.9, 2.3, 1.7, 3.1]),
            family=family,
            predictors=(Predictor("scale", {"x": _ExplodingFeature()}), Predictor("shape", {})),
            sample_weight=np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25]),
            weight_contract=WeightContract(semantics="prior"),
        )


def test_the_family_is_exported_from_both_packages():
    from superglm import distributional
    from superglm.distributional import families

    assert families.GeneralizedParetoLSS is GeneralizedParetoLSS
    assert distributional.GeneralizedParetoLSS is GeneralizedParetoLSS
    assert (
        "GeneralizedParetoLSS" in families.__all__
        and "GeneralizedParetoLSS" in distributional.__all__
    )


def test_a_public_fit_reaches_the_family_through_the_facade():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 400)})
    scale = np.exp(0.2 + 0.5 * frame["x"].to_numpy())
    y = scale * np.expm1(-0.3 * np.log(rng.random(400))) / 0.3
    model = SuperLSS(
        family=GeneralizedParetoLSS(),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    fitted = model.predict_parameters(frame)
    assert 0.0 < float(fitted["shape"].iloc[0]) < 1.0
    assert np.all(fitted["scale"].to_numpy() > 0.0)
    assert np.allclose(
        model.predict(frame),
        fitted["scale"].to_numpy() / (1.0 - fitted["shape"].to_numpy()),
        rtol=0,
        atol=1e-12,
    )


def test_artifact_round_trip_preserves_predictions_and_config():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 500)})
    scale = np.exp(0.3 + 0.4 * frame["x"].to_numpy())
    y = scale * np.expm1(-0.35 * np.log(rng.random(500))) / 0.35
    model = SuperLSS(
        family=GeneralizedParetoLSS(shape_lower=0.0, shape_upper=1.0),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame), model.predict(frame))
    assert np.array_equal(
        restored.predict_quantile(frame, 0.95), model.predict_quantile(frame, 0.95)
    )
    assert np.array_equal(restored.predict_cdf(frame, 1.0), model.predict_cdf(frame, 1.0))
    assert restored.family.to_config() == {
        "type": "GeneralizedParetoLSS",
        "shape_lower": 0.0,
        "shape_upper": 1.0,
    }


def test_narrow_walls_round_trip_and_a_tampered_config_is_refused():
    from superglm.distributional.serialization import (
        DistributionalSerializationError,
        _validate_generalized_pareto_config,
    )

    rng = np.random.default_rng(21)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 300)})
    y = np.expm1(-0.2 * np.log(rng.random(300))) / 0.2
    model = SuperLSS(
        family=GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.5),
        predictors=(Predictor("scale", {"x": Numeric()}), Predictor("shape", {})),
    ).fit(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert restored.family == GeneralizedParetoLSS(shape_lower=0.05, shape_upper=0.5)
    for bad in (
        {"type": "GeneralizedParetoLSS", "shape_lower": 0.0},
        {"type": "GeneralizedParetoLSS", "shape_lower": 0.0, "shape_upper": 1.0, "extra": 1},
        {"type": "GeneralizedParetoLSS", "shape_lower": -0.5, "shape_upper": 1.0},
        {"type": "GeneralizedParetoLSS", "shape_lower": 0.5, "shape_upper": 0.5},
        {"type": "GeneralizedParetoLSS", "shape_lower": 0.0, "shape_upper": True},
    ):
        with pytest.raises(DistributionalSerializationError):
            _validate_generalized_pareto_config(bad)
