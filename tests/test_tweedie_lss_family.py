from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

import superglm.distributional.families.tweedie as tweedie_module
from superglm.distributional.families.tweedie import (
    BoundedPowerLink,
    TweedieLikelihoodPlan,
    TweedieLSS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DistributionalFamily,
    ExpectedInformationFamily,
    FamilyLikelihoodPlan,
    validate_family,
)
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    WeightSemantics,
    resolve_likelihood_weights,
)
from superglm.links import LogLink
from tests._distributional_family_kernels import tweedie as tweedie_kernel

TweedieNumericalRefusal = tweedie_kernel.TweedieNumericalRefusal
evaluate_tweedie_rows = tweedie_kernel.evaluate_tweedie_rows

_EPS = np.finfo(np.float64).eps


def _tolerance(*, dimension: int, scale: float = 1.0) -> dict[str, float]:
    envelope = 32.0 * max(1, dimension) * _EPS
    return {"rtol": envelope, "atol": envelope * max(1.0, abs(scale))}


def _resolved(weights: np.ndarray, semantics: WeightSemantics) -> ResolvedLikelihoodWeights:
    return resolve_likelihood_weights(
        weights,
        n_observations=len(weights),
        contract=WeightContract(semantics=semantics),
    )


def _plan(
    family: TweedieLSS,
    response: np.ndarray,
    weights: np.ndarray,
    semantics: WeightSemantics,
) -> TweedieLikelihoodPlan:
    return family.bind_likelihood(
        response,
        _resolved(weights, semantics),
        COMPLETE_OBSERVATION,
    )


def _evaluate_one(weight: float, semantics: WeightSemantics):
    family = TweedieLSS()
    response = np.array([1.0])
    resolved = resolve_likelihood_weights(
        np.array([weight]),
        n_observations=1,
        contract=WeightContract(semantics=semantics),
    )
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    return family.evaluate_natural(
        response,
        np.array([[0.8, 0.7, 1.5]]),
        plan,
        derivative_order=2,
    )


def test_family_metadata_is_an_observed_three_parameter_contract() -> None:
    """Kills parameter reordering, widened support, and invented Fisher capability."""
    family = TweedieLSS()
    parameters = validate_family(family)

    assert isinstance(family, DistributionalFamily)
    assert tuple(parameter.name for parameter in parameters) == (
        "mean",
        "dispersion",
        "power",
    )
    assert tuple(parameter.role for parameter in parameters) == (
        "mean",
        "dispersion",
        "power",
    )
    assert tuple(parameter.curvature for parameter in parameters) == (
        "observed",
        "observed",
        "observed",
    )
    assert isinstance(parameters[0].default_link, LogLink)
    assert isinstance(parameters[1].default_link, LogLink)
    assert isinstance(parameters[2].default_link, BoundedPowerLink)
    assert parameters[0].support.lower == 0.0
    assert parameters[1].support.lower == 0.0
    assert parameters[2].support.lower == family.power_lower
    assert parameters[2].support.upper == family.power_upper
    assert not parameters[0].support.lower_inclusive
    assert not parameters[1].support.lower_inclusive
    assert not parameters[2].support.lower_inclusive
    assert not parameters[2].support.upper_inclusive
    assert family.capabilities.max_derivative_order == 2
    assert not family.capabilities.expected_information
    assert family.capabilities.response_mean
    assert not isinstance(family, ExpectedInformationFamily)


def test_family_configuration_is_small_and_json_safe() -> None:
    family = TweedieLSS(power_lower=1.1, power_upper=1.9)

    assert family.to_config() == {
        "type": "TweedieLSS",
        "power_lower": 1.1,
        "power_upper": 1.9,
    }
    assert json.loads(json.dumps(family.to_config())) == family.to_config()


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (1.0, 1.9),
        (1.1, 2.0),
        (1.5, 1.5),
        (1.7, 1.2),
        (math.nan, 1.9),
        (1.1, math.inf),
    ],
)
def test_power_link_requires_finite_interior_ordered_walls(lower: float, upper: float) -> None:
    """Kills accepting a link whose range is not an open subset of (1, 2)."""
    with pytest.raises(ValueError, match="lower|upper|walls"):
        BoundedPowerLink(lower=lower, upper=upper)


def test_bounded_power_link_value_and_derivatives_match_closed_forms() -> None:
    """Kills missing width factors and first/second inverse derivative swaps."""
    link = BoundedPowerLink(lower=1.05, upper=1.95)
    eta = np.array([math.log(3.0)])
    width = 0.9
    probability = 0.75
    expected_power = 1.05 + width * probability
    expected_first = width * probability * (1.0 - probability)
    expected_second = expected_first * (1.0 - 2.0 * probability)

    np.testing.assert_allclose(
        link.inverse(eta),
        [expected_power],
        **_tolerance(dimension=1, scale=expected_power),
    )
    np.testing.assert_allclose(
        link.link(np.array([expected_power])),
        eta,
        **_tolerance(dimension=4, scale=float(eta[0])),
    )
    np.testing.assert_allclose(
        link.deriv_inverse(eta),
        [expected_first],
        **_tolerance(dimension=4, scale=expected_first),
    )
    np.testing.assert_allclose(
        link.deriv2_inverse(eta),
        [expected_second],
        **_tolerance(dimension=6, scale=expected_second),
    )
    np.testing.assert_allclose(
        link.deriv(np.array([expected_power])),
        [1.0 / expected_first],
        **_tolerance(dimension=6, scale=1.0 / expected_first),
    )


@pytest.mark.parametrize("method_name", ["link", "deriv"])
@pytest.mark.parametrize("power", [1.04, 1.05, 1.95, 1.96, math.nan, math.inf])
def test_bounded_power_link_refuses_values_at_or_outside_the_walls(
    method_name: str,
    power: float,
) -> None:
    """Kills clipping or accepting a forward-link value on a configured wall."""
    link = BoundedPowerLink(lower=1.05, upper=1.95)

    with pytest.raises(ValueError, match="strictly between|walls"):
        getattr(link, method_name)(np.array([power]))


def test_plan_is_minimal_frozen_and_deterministically_identified() -> None:
    """Kills response/carrier authority in the plan and unstable plan identifiers."""
    family = TweedieLSS()
    response = np.array([0.0, 1.0, 2.0])
    resolved = _resolved(np.array([1.0, 2.0, 3.0]), "prior")
    first = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    second = family.bind_likelihood(response.copy(), resolved, COMPLETE_OBSERVATION)

    assert isinstance(first, FamilyLikelihoodPlan)
    assert tuple(field.name for field in fields(first)) == ("weights",)
    assert first.weights is resolved
    assert first.plan_identifier == second.plan_identifier
    assert first.plan_identifier.startswith("TweedieLSS/v1:")
    with pytest.raises(FrozenInstanceError):
        first.weights = _resolved(np.ones(3), "prior")  # type: ignore[misc]


def test_plan_take_preserves_concrete_type_and_ordered_weight_identity() -> None:
    """Kills untyped children, reordered children, and identifiers inherited from roots."""
    family = TweedieLSS()
    response = np.array([0.0, 1.0, 2.0, 3.0])
    root = _plan(family, response, np.array([1.0, 2.0, 3.0, 4.0]), "frequency")
    indices = np.array([3, 1], dtype=np.intp)
    child = root.take(indices)

    assert type(child) is type(root) is TweedieLikelihoodPlan
    np.testing.assert_array_equal(child.weights.values, [4.0, 2.0])
    np.testing.assert_array_equal(child.weights.root_take_map, indices)
    assert child.weights.provenance is root.weights.provenance
    assert child.plan_identifier == TweedieLikelihoodPlan(child.weights).plan_identifier
    assert child.plan_identifier != root.plan_identifier


def test_tweedie_plan_subclass_is_validated_structurally() -> None:
    class TweediePlanSubclass(TweedieLikelihoodPlan):
        pass

    family = TweedieLSS()
    response = np.array([0.0, 2.0])
    theta = np.array([[1.2, 0.8, 1.4], [2.1, 1.3, 1.7]])
    base = _plan(family, response, np.ones(2), "prior")
    plan = TweediePlanSubclass(weights=base.weights)

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, theta, plan)

    np.testing.assert_equal(initialized.theta, family.initialize(response, base).theta)
    np.testing.assert_equal(
        evaluated.reported_log_likelihood,
        family.evaluate_natural(response, theta, base).reported_log_likelihood,
    )
    malformed = TweediePlanSubclass(weights=base.weights.take(np.array([0], dtype=np.intp)))
    with pytest.raises(UnsupportedLikelihoodContractError, match="rows"):
        family.initialize(response, malformed)


def test_binding_refuses_row_mismatch_and_noncomplete_observation() -> None:
    """Kills positional-weight drift and acceptance of an unsupported observation law."""
    family = TweedieLSS()
    response = np.array([0.0, 1.0])
    weights = _resolved(np.ones(3), "prior")

    with pytest.raises(UnsupportedLikelihoodContractError, match="rows"):
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    with pytest.raises(UnsupportedLikelihoodContractError, match="complete"):
        family.bind_likelihood(
            response,
            _resolved(np.ones(2), "prior"),
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "response",
    [
        np.array([-np.finfo(np.float64).tiny, 1.0]),
        np.array([0.0, math.nan]),
        np.array([0.0, math.inf]),
    ],
)
def test_response_support_is_exactly_finite_nonnegative(response: np.ndarray) -> None:
    """Kills accepting values outside the mixed law's [0, infinity) support."""
    family = TweedieLSS()

    with pytest.raises(ValueError, match="non-negative|y"):
        family.bind_likelihood(
            response,
            _resolved(np.ones(len(response)), "prior"),
            COMPLETE_OBSERVATION,
        )


def test_zero_responses_bind_but_an_all_zero_initializer_refuses() -> None:
    """Kills confusing valid zero atoms with existence of a finite interior initializer."""
    family = TweedieLSS()
    response = np.zeros(3)
    plan = _plan(family, response, np.array([1.0, 2.0, 3.0]), "prior")

    with pytest.raises(ValueError, match="all-zero"):
        family.initialize(response, plan)


@pytest.mark.parametrize(
    ("semantics", "expected_dispersion"),
    [
        ("prior", (52.0 / 9.0) / ((7.0 / 3.0) ** 1.5)),
        ("frequency", (26.0 / 9.0) / ((7.0 / 3.0) ** 1.5)),
    ],
)
def test_initializer_uses_the_declared_prior_or_frequency_denominator(
    semantics: WeightSemantics,
    expected_dispersion: float,
) -> None:
    """Kills using weight mass for prior rows or physical rows for frequency rows."""
    family = TweedieLSS()
    response = np.array([0.0, 1.0, 4.0])
    weights = np.array([1.0, 2.0, 3.0])
    initialized = family.initialize(response, _plan(family, response, weights, semantics))

    initialized.validate_shape(n_observations=3, k_parameters=3)
    np.testing.assert_allclose(
        initialized.theta[:, 0],
        7.0 / 3.0,
        **_tolerance(dimension=len(response), scale=7.0 / 3.0),
    )
    np.testing.assert_allclose(
        initialized.theta[:, 1],
        expected_dispersion,
        **_tolerance(dimension=3 * len(response), scale=expected_dispersion),
    )
    np.testing.assert_array_equal(initialized.theta[:, 2], np.full(3, 1.5))
    assert not initialized.theta.flags.writeable


@pytest.mark.parametrize("power", [1.05, 1.95])
def test_family_refuses_theta_rounded_to_either_configured_wall(power: float) -> None:
    """Kills clipping a rounded bounded-logit inverse back into the executable range."""
    family = TweedieLSS()
    response = np.array([1.0])
    plan = _plan(family, response, np.ones(1), "prior")

    with pytest.raises(ValueError, match="strictly between|power"):
        family.evaluate_natural(
            response,
            np.array([[1.0, 1.0, power]]),
            plan,
            derivative_order=0,
        )


def test_evaluation_refuses_response_plan_row_mismatch() -> None:
    """Kills evaluating a certified positional carrier against another row count."""
    family = TweedieLSS()
    root_response = np.array([0.0, 1.0])
    plan = _plan(family, root_response, np.ones(2), "prior")

    with pytest.raises(UnsupportedLikelihoodContractError, match="rows"):
        family.evaluate_natural(
            np.array([1.0]),
            np.array([[1.0, 1.0, 1.5]]),
            plan,
        )


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_evaluation_returns_the_exact_requested_derivative_order(
    semantics: WeightSemantics,
) -> None:
    """Kills dummy derivative arrays and evaluation above the requested order."""
    family = TweedieLSS()
    response = np.array([0.0, 1.25])
    theta = np.array([[0.8, 0.7, 1.5], [1.1, 0.6, 1.7]])
    plan = _plan(family, response, np.array([2.0, 3.0]), semantics)

    value_only = family.evaluate_natural(response, theta, plan, derivative_order=0)
    score_only = family.evaluate_natural(response, theta, plan, derivative_order=1)
    full = family.evaluate_natural(response, theta, plan, derivative_order=2)

    assert (value_only.derivative_order, score_only.derivative_order, full.derivative_order) == (
        0,
        1,
        2,
    )
    assert value_only.score is None
    assert value_only.hessian_packed is None
    assert score_only.score is not None
    assert score_only.hessian_packed is None
    assert full.score is not None
    assert full.hessian_packed is not None
    np.testing.assert_array_equal(value_only.reported_log_likelihood, full.reported_log_likelihood)
    np.testing.assert_array_equal(score_only.reported_log_likelihood, full.reported_log_likelihood)
    np.testing.assert_array_equal(score_only.score, full.score)


def test_evaluation_arrays_are_owned_and_immutable() -> None:
    """Kills returning writable kernel views or a mutable synthesized carrier."""
    family = TweedieLSS()
    response = np.array([0.0, 1.25])
    theta = np.array([[0.8, 0.7, 1.5], [1.1, 0.6, 1.7]])
    result = family.evaluate_natural(
        response,
        theta,
        _plan(family, response, np.ones(2), "prior"),
        derivative_order=2,
    )

    arrays = (
        result.optimizing_log_likelihood,
        result.parameter_independent_carrier,
        result.reported_log_likelihood,
        result.score,
        result.hessian_packed,
        result.valid,
    )
    assert all(array is not None and not array.flags.writeable for array in arrays)


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_adapter_preserves_complete_kernel_rows_and_uses_a_zero_carrier(
    semantics: WeightSemantics,
) -> None:
    """Kills a second family-level weight action and any parameter-independent carrier."""
    family = TweedieLSS()
    response = np.array([0.0, 1.3])
    theta = np.array([[0.8, 0.7, 1.5], [1.2, 0.4, 1.7]])
    weights = np.array([3.0, 2.0])
    plan = _plan(family, response, weights, semantics)
    expected = evaluate_tweedie_rows(
        response,
        theta[:, 0],
        theta[:, 1],
        theta[:, 2],
        weights,
        semantics,
        derivative_order=2,
    )

    actual = family.evaluate_natural(response, theta, plan, derivative_order=2)

    np.testing.assert_array_equal(actual.optimizing_log_likelihood, expected.log_likelihood)
    np.testing.assert_array_equal(actual.reported_log_likelihood, expected.log_likelihood)
    np.testing.assert_array_equal(actual.parameter_independent_carrier, np.zeros(2))
    np.testing.assert_array_equal(actual.score, expected.score)
    np.testing.assert_array_equal(actual.hessian_packed, expected.hessian_packed)
    np.testing.assert_array_equal(actual.valid, expected.valid)


def test_prior_weight_is_dispersion_action_not_replication() -> None:
    prior = _evaluate_one(4.0, "prior")
    frequency = _evaluate_one(4.0, "frequency")
    assert prior.reported_log_likelihood[0] != frequency.reported_log_likelihood[0]
    assert prior.score[0, 1] != frequency.score[0, 1]


def test_frequency_equals_literal_replication() -> None:
    weighted = _evaluate_one(4.0, "frequency")
    unit = _evaluate_one(1.0, "frequency")
    np.testing.assert_array_equal(
        weighted.reported_log_likelihood,
        4.0 * unit.reported_log_likelihood,
    )
    np.testing.assert_array_equal(weighted.score, 4.0 * unit.score)
    np.testing.assert_array_equal(weighted.hessian_packed, 4.0 * unit.hessian_packed)


def test_frequency_row_matches_four_physical_unit_rows() -> None:
    """Kills an implementation that only resembles replication in one derivative channel."""
    family = TweedieLSS()
    response = np.ones(4)
    theta = np.repeat(np.array([[0.8, 0.7, 1.5]]), 4, axis=0)
    replicated = family.evaluate_natural(
        response,
        theta,
        _plan(family, response, np.ones(4), "frequency"),
        derivative_order=2,
    )
    compressed = _evaluate_one(4.0, "frequency")

    for compressed_rows, physical_rows in (
        (compressed.reported_log_likelihood, replicated.reported_log_likelihood),
        (compressed.score, replicated.score),
        (compressed.hessian_packed, replicated.hessian_packed),
    ):
        assert compressed_rows is not None
        assert physical_rows is not None
        np.testing.assert_allclose(
            compressed_rows[0],
            np.sum(physical_rows, axis=0),
            **_tolerance(
                dimension=len(physical_rows), scale=float(np.max(np.abs(compressed_rows)))
            ),
        )


def test_unit_prior_and_frequency_rows_are_identical() -> None:
    """Kills a semantic branch that changes the unit-weight Tweedie law."""
    prior = _evaluate_one(1.0, "prior")
    frequency = _evaluate_one(1.0, "frequency")

    for prior_rows, frequency_rows in (
        (prior.reported_log_likelihood, frequency.reported_log_likelihood),
        (prior.parameter_independent_carrier, frequency.parameter_independent_carrier),
        (prior.score, frequency.score),
        (prior.hessian_packed, frequency.hessian_packed),
    ):
        np.testing.assert_array_equal(prior_rows, frequency_rows)


def test_kernel_numerical_refusal_is_translated_to_trial_value_error() -> None:
    """Kills leaking the kernel's private typed refusal through the family boundary."""
    family = TweedieLSS()
    response = np.array([1.0])
    theta = np.array([[1.0e308, 1.0e-308, 1.5]])

    with pytest.raises(ValueError, match="Tweedie.*refus") as caught:
        family.evaluate_natural(
            response,
            theta,
            _plan(family, response, np.ones(1), "prior"),
            derivative_order=2,
        )
    assert not isinstance(caught.value, TweedieNumericalRefusal)
    assert isinstance(caught.value.__cause__, TweedieNumericalRefusal)


def test_non_kernel_programming_error_is_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills broad exception swallowing that makes programming defects trial-rejectable."""
    family = TweedieLSS()
    response = np.array([1.0])
    sentinel = TypeError("programming defect")

    def fail_with_programming_error(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise sentinel

    monkeypatch.setattr(tweedie_module, "evaluate_tweedie_rows", fail_with_programming_error)
    with pytest.raises(TypeError) as caught:
        family.evaluate_natural(
            response,
            np.array([[0.8, 0.7, 1.5]]),
            _plan(family, response, np.ones(1), "prior"),
            derivative_order=2,
        )
    assert caught.value is sentinel


def test_default_prediction_is_an_owned_conditional_mean() -> None:
    family = TweedieLSS()
    theta = np.array([[0.8, 0.7, 1.5], [1.1, 0.6, 1.7]])

    prediction = family.default_prediction(theta)
    theta[:, 0] = -100.0

    assert family.default_prediction_name == "conditional_mean"
    np.testing.assert_array_equal(prediction, [0.8, 1.1])
    assert not prediction.flags.writeable
