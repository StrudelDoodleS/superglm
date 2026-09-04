"""Adapter-contract tests for ``LogNormalLS``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm.distributional import Predictor
from superglm.distributional.families.log_normal import LogNormalLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import log_normal as ln
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


def test_parameters_follow_the_parametrisation():
    mean_form = LogNormalLS()
    assert [p.name for p in mean_form.parameters] == ["mean", "scale"]
    assert [p.role for p in mean_form.parameters] == ["mean", "scale"]
    assert mean_form.parameters[0].support.lower == 0.0
    assert mean_form.parameters[1].support.lower == 0.01
    assert [p.curvature for p in mean_form.parameters] == ["observed", "observed"]
    location_form = LogNormalLS(parametrisation="location")
    assert [p.name for p in location_form.parameters] == ["location", "scale"]
    assert location_form.parameters[0].support.lower is None
    assert mean_form.to_config() == {
        "type": "LogNormalLS",
        "parametrisation": "mean",
        "scale_floor": 0.01,
    }
    assert mean_form.default_prediction_name == "conditional_mean"
    assert mean_form.capabilities.cdf and mean_form.capabilities.quantile
    with pytest.raises(ValueError):
        LogNormalLS(parametrisation="median")
    with pytest.raises(ValueError):
        LogNormalLS(scale_floor=-1.0)


def test_response_must_be_strictly_positive():
    family = LogNormalLS()
    with pytest.raises(ValueError, match="strictly positive"):
        _bind(family, np.array([1.0, 0.0, 2.0]), np.ones(3), "frequency")
    with pytest.raises(ValueError, match="strictly positive"):
        _bind(family, np.array([1.0, -2.0]), np.ones(2), "frequency")


def test_nonunit_prior_weights_are_refused_and_unit_prior_bridges_to_frequency():
    family = LogNormalLS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        _bind(family, y, [1.0, 2.0, 1.0, 1.0], "prior")
    plan = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "prior")
    assert plan.row_law == "unit-prior-explicitly-equals-frequency/v1"
    frequency = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "frequency")
    assert frequency.row_law == "log-normal-literal-replication/v1"
    theta = family.initialize(y, plan).theta
    a = family.evaluate_natural(y, theta, plan)
    b = family.evaluate_natural(y, theta, frequency)
    assert np.array_equal(a.reported_log_likelihood, b.reported_log_likelihood)
    assert np.array_equal(a.hessian_packed, b.hessian_packed)


def test_integer_frequency_rows_equal_literal_replication():
    family = LogNormalLS()
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    plan = _bind(family, y, counts, "frequency")
    weighted = family.evaluate_natural(y, np.tile([2.0, 0.8], (3, 1)), plan)
    replicated_y = np.repeat(y, counts.astype(int))
    replicated_plan = _bind(family, replicated_y, np.ones(len(replicated_y)), "frequency")
    unit = family.evaluate_natural(
        replicated_y, np.tile([2.0, 0.8], (len(replicated_y), 1)), replicated_plan
    )
    for index, count in enumerate(counts.astype(int)):
        start = int(counts[:index].sum())
        assert np.isclose(
            weighted.reported_log_likelihood[index],
            np.sum(unit.reported_log_likelihood[start : start + count]),
            rtol=0,
            atol=1e-12,
        )


def test_reported_log_likelihood_is_the_scipy_density():
    family = LogNormalLS(parametrisation="location")
    y = np.array([0.4, 1.3, 2.8, 7.5])
    plan = _bind(family, y, np.ones(4), "frequency")
    theta = np.tile([0.3, 0.8], (4, 1))
    evaluated = family.evaluate_natural(y, theta, plan, derivative_order=0)
    reference = stats.lognorm(s=0.8, scale=math.exp(0.3)).logpdf(y)
    assert np.allclose(evaluated.reported_log_likelihood, reference, rtol=1e-14, atol=0)


def test_exactly_the_requested_derivative_order_is_returned():
    family = LogNormalLS()
    y = np.array([0.5, 1.5, 2.5])
    plan = _bind(family, y, np.ones(3), "frequency")
    theta = np.tile([1.5, 0.8], (3, 1))
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        family.evaluate_natural(y, theta, plan, derivative_order=True)


def test_plan_take_preserves_root_identity_and_changes_the_identifier():
    family = LogNormalLS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    plan = _bind(family, y, np.ones(4), "frequency")
    child = plan.take(np.array([3, 1], dtype=np.intp))
    assert child.weights.root_digest == plan.weights.root_digest
    assert child.plan_identifier != plan.plan_identifier
    assert child.exact_response.tolist() == [4.0, 1.5]
    assert not child.parameter_independent_carrier.flags.writeable
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.validate_likelihood_plan(np.array([0.5, 1.5, 2.5, 9.0]), plan)
    with pytest.raises(UnsupportedLikelihoodContractError):
        LogNormalLS(parametrisation="location").validate_likelihood_plan(y, plan)


def test_evaluate_natural_matches_the_kernel_in_both_forms():
    y = np.array([0.5, 1.5, 6.0])
    for family, theta, rows in (
        (LogNormalLS(), np.tile([2.0, 0.8], (3, 1)), ln.mean_rows),
        (LogNormalLS(parametrisation="location"), np.tile([0.3, 0.8], (3, 1)), ln.location_rows),
    ):
        plan = _bind(family, y, np.ones(3), "frequency")
        evaluated = family.evaluate_natural(y, theta, plan)
        kernel = rows(y, theta[:, 0], theta[:, 1], np.ones(3), derivative_order=2)
        assert np.array_equal(evaluated.optimizing_log_likelihood, kernel.optimizing_log_likelihood)
        assert np.array_equal(evaluated.score, kernel.score)
        assert np.array_equal(evaluated.hessian_packed, kernel.hessian_packed)
        assert evaluated.valid is None or bool(np.all(evaluated.valid))


def test_default_prediction_cdf_quantile_and_information_in_both_forms():
    y = np.array([0.5, 1.5, 6.0])
    theta_mean = np.tile([2.0, 0.8], (3, 1))
    mean_form = LogNormalLS()
    plan = _bind(mean_form, y, np.ones(3), "frequency")
    assert np.array_equal(mean_form.default_prediction(theta_mean), theta_mean[:, 0])
    location = ln.location_of_mean(theta_mean[:, 0], theta_mean[:, 1])
    theta_location = np.column_stack((location, theta_mean[:, 1]))
    location_form = LogNormalLS(parametrisation="location")
    assert np.allclose(location_form.default_prediction(theta_location), 2.0, rtol=0, atol=1e-14)
    p = np.array([0.1, 0.5, 0.9])
    quantile = mean_form.quantile(p, theta_mean)
    assert np.allclose(mean_form.cdf(quantile, theta_mean), p, rtol=0, atol=1e-12)
    assert np.allclose(location_form.quantile(p, theta_location), quantile, rtol=1e-13, atol=0)
    information = mean_form.expected_information_natural(theta_mean, plan)
    assert information.shape == (3, 3) and np.all(np.isfinite(information))
    assert np.array_equal(
        information,
        ln.mean_expected_information(theta_mean[:, 0], theta_mean[:, 1], np.ones(3)),
    )
    # the plan carries the family configuration, so the location form needs its own
    location_plan = _bind(location_form, y, np.ones(3), "frequency")
    assert np.array_equal(
        location_form.expected_information_natural(theta_location, location_plan),
        ln.location_expected_information(theta_location[:, 1], np.ones(3)),
    )


def test_refusal_precedes_geometry_initialization_and_evaluation():
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        fit_dense_distributional(
            pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)}),
            np.array([0.4, 1.1, 0.9, 2.3, 1.7, 3.1]),
            family=LogNormalLS(),
            predictors=(
                Predictor("mean", {"x": _ExplodingFeature()}),
                Predictor("scale", {}),
            ),
            sample_weight=np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25]),
            weight_contract=WeightContract(semantics="prior"),
        )


from superglm import SuperLSS  # noqa: E402


def test_artifact_round_trip_preserves_predictions_and_config():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 150)})
    y = np.exp(0.3 + 0.4 * frame["x"].to_numpy() + 0.7 * rng.standard_normal(150))
    model = SuperLSS(
        family=LogNormalLS(),
        predictors=(Predictor("mean", {"x": Numeric()}), Predictor("scale", {})),
    ).fit(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame), model.predict(frame))
    assert np.array_equal(
        restored.predict_quantile(frame, 0.95), model.predict_quantile(frame, 0.95)
    )
    assert np.allclose(restored.predict_cdf(frame, 1.0), model.predict_cdf(frame, 1.0))
    assert restored.family.to_config() == {
        "type": "LogNormalLS",
        "parametrisation": "mean",
        "scale_floor": 0.01,
    }
