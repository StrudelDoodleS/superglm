"""Adapter-contract tests for ``GeneralizedGammaLSS``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families.generalized_gamma import GeneralizedGammaLSS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.kernels import generalized_gamma as gg
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
    mean_form = GeneralizedGammaLSS()
    assert [p.name for p in mean_form.parameters] == ["mean", "scale", "shape"]
    assert [p.role for p in mean_form.parameters] == ["mean", "scale", "shape"]
    assert mean_form.parameters[0].support.lower == 0.0
    assert mean_form.parameters[2].support.lower is None
    location_form = GeneralizedGammaLSS(parametrisation="location")
    assert [p.name for p in location_form.parameters] == ["location", "scale", "shape"]
    assert location_form.parameters[0].support.lower is None
    assert mean_form.to_config() == {
        "type": "GeneralizedGammaLSS",
        "parametrisation": "mean",
        "scale_floor": 0.01,
    }
    with pytest.raises(ValueError):
        GeneralizedGammaLSS(parametrisation="median")
    with pytest.raises(ValueError):
        GeneralizedGammaLSS(scale_floor=-1.0)


def test_nonunit_prior_weights_are_refused_and_unit_prior_bridges_to_frequency():
    family = GeneralizedGammaLSS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        _bind(family, y, [1.0, 2.0, 1.0, 1.0], "prior")
    plan = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "prior")
    assert plan.row_law == "unit-prior-explicitly-equals-frequency/v1"
    frequency = _bind(family, y, [1.0, 1.0, 1.0, 1.0], "frequency")
    assert frequency.row_law == "gg-prentice-literal-replication/v1"
    theta = family.initialize(y, plan).theta
    a = family.evaluate_natural(y, theta, plan)
    b = family.evaluate_natural(y, theta, frequency)
    assert np.array_equal(a.reported_log_likelihood, b.reported_log_likelihood)
    assert np.array_equal(a.hessian_packed, b.hessian_packed)


def test_integer_frequency_rows_equal_literal_replication():
    family = GeneralizedGammaLSS()
    y = np.array([0.7, 1.9, 4.2])
    counts = np.array([1.0, 3.0, 2.0])
    plan = _bind(family, y, counts, "frequency")
    theta = np.tile([2.0, 0.8, 0.4], (3, 1))
    weighted = family.evaluate_natural(y, theta, plan)
    replicated_y = np.repeat(y, counts.astype(int))
    replicated_plan = _bind(family, replicated_y, np.ones(len(replicated_y)), "frequency")
    unit = family.evaluate_natural(
        replicated_y, np.tile([2.0, 0.8, 0.4], (len(replicated_y), 1)), replicated_plan
    )
    for weighted_row, count, index in zip(
        weighted.reported_log_likelihood, counts, range(3), strict=True
    ):
        start = int(np.sum(counts[:index]))
        assert np.isclose(
            weighted_row,
            np.sum(unit.reported_log_likelihood[start : start + int(count)]),
            rtol=0,
            atol=1e-12,
        )


def test_exactly_the_requested_derivative_order_is_returned():
    family = GeneralizedGammaLSS()
    y = np.array([0.5, 1.5, 2.5])
    plan = _bind(family, y, np.ones(3), "frequency")
    theta = np.tile([1.5, 0.8, -0.3], (3, 1))
    for order, has_score, has_hessian in ((0, False, False), (1, True, False), (2, True, True)):
        evaluated = family.evaluate_natural(y, theta, plan, derivative_order=order)
        assert (evaluated.score is not None) is has_score
        assert (evaluated.hessian_packed is not None) is has_hessian
    with pytest.raises(ValueError):
        family.evaluate_natural(y, theta, plan, derivative_order=True)


def test_plan_take_preserves_root_identity_and_changes_the_identifier():
    family = GeneralizedGammaLSS()
    y = np.array([0.5, 1.5, 2.5, 4.0])
    plan = _bind(family, y, np.ones(4), "frequency")
    child = plan.take(np.array([3, 1], dtype=np.intp))
    assert child.weights.root_digest == plan.weights.root_digest
    assert child.plan_identifier != plan.plan_identifier
    assert child.exact_response.tolist() == [4.0, 1.5]
    assert not child.parameter_independent_carrier.flags.writeable
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.validate_likelihood_plan(np.array([0.5, 1.5, 2.5, 9.0]), plan)


def test_evaluate_natural_matches_the_kernel_and_flags_infinite_mean_rows():
    family = GeneralizedGammaLSS()
    y = np.array([0.5, 1.5])
    plan = _bind(family, y, np.ones(2), "frequency")
    theta = np.array([[2.0, 0.8, 0.4], [2.0, 2.5, -0.5]])
    evaluated = family.evaluate_natural(y, theta, plan)
    kernel = gg.mean_rows(y, theta[:, 0], theta[:, 1], theta[:, 2], np.ones(2), derivative_order=2)
    assert np.array_equal(evaluated.optimizing_log_likelihood, kernel.optimizing_log_likelihood)
    assert evaluated.valid.tolist() == [True, False]
    assert np.all(np.isfinite(evaluated.reported_log_likelihood))


def test_default_prediction_cdf_quantile_and_information_in_both_forms():
    y = np.array([0.5, 1.5, 6.0])
    theta_mean = np.array([[2.0, 0.8, 0.4]] * 3)
    mean_form = GeneralizedGammaLSS()
    plan = _bind(mean_form, y, np.ones(3), "frequency")
    assert np.array_equal(mean_form.default_prediction(theta_mean), theta_mean[:, 0])
    location = gg.location_of_mean(theta_mean[:, 0], theta_mean[:, 1], theta_mean[:, 2])
    theta_location = np.column_stack((location, theta_mean[:, 1], theta_mean[:, 2]))
    location_form = GeneralizedGammaLSS(parametrisation="location")
    assert np.allclose(location_form.default_prediction(theta_location), 2.0, rtol=0, atol=1e-14)
    p = np.array([0.1, 0.5, 0.9])
    q_mean = mean_form.quantile(p, theta_mean)
    assert np.allclose(mean_form.cdf(q_mean, theta_mean), p, rtol=0, atol=1e-12)
    assert np.allclose(location_form.quantile(p, theta_location), q_mean, rtol=1e-13, atol=0)
    information = mean_form.expected_information_natural(theta_mean, plan)
    assert information.shape == (3, 6) and np.all(np.isfinite(information))
    assert np.array_equal(
        information,
        gg.mean_expected_information(
            theta_mean[:, 0], theta_mean[:, 1], theta_mean[:, 2], np.ones(3)
        ),
    )


def test_refusal_precedes_geometry_initialization_and_evaluation():
    family = GeneralizedGammaLSS()
    frame = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 6)})
    with pytest.raises(UnsupportedLikelihoodContractError, match="non-unit prior"):
        fit_dense_distributional(
            frame,
            np.array([0.4, 1.1, 0.9, 2.3, 1.7, 3.1]),
            family=family,
            predictors=(
                Predictor("mean", {"x": _ExplodingFeature()}),
                Predictor("scale", {}),
                Predictor("shape", {}),
            ),
            sample_weight=np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25]),
            weight_contract=WeightContract(semantics="prior"),
        )


def test_diagnosis_names_the_infinite_mean_boundary_for_the_mean_form():
    family = GeneralizedGammaLSS()
    diagnosis = family.diagnose_repeated_curvature_failure(
        np.array([1.0, 2.0, 3.0]), _weights(np.ones(3), "frequency")
    )
    assert isinstance(diagnosis, Exception)
    assert "infinite-mean" in str(diagnosis) and "location" in str(diagnosis)
    location_form = GeneralizedGammaLSS(parametrisation="location")
    assert (
        location_form.diagnose_repeated_curvature_failure(
            np.array([1.0, 2.0]), _weights(np.ones(2), "frequency")
        )
        is None
    )


def test_artifact_round_trip_preserves_predictions_and_config():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 150)})
    k = 1.0 / 0.5**2
    y = np.exp(0.3 + 0.4 * frame["x"].to_numpy() + 0.7 * np.log(rng.gamma(k, 1.0, 150) / k) / 0.5)
    model = SuperLSS(
        family=GeneralizedGammaLSS(),
        predictors=(
            Predictor("mean", {"x": Numeric()}),
            Predictor("scale", {}),
            Predictor("shape", {}),
        ),
    ).fit(frame, y)
    restored = SuperLSS.from_bytes(model.to_bytes())
    assert np.array_equal(restored.predict(frame), model.predict(frame))
    assert np.array_equal(
        restored.predict_quantile(frame, 0.95), model.predict_quantile(frame, 0.95)
    )
    assert restored.family.to_config() == {
        "type": "GeneralizedGammaLSS",
        "parametrisation": "mean",
        "scale_floor": 0.01,
    }
