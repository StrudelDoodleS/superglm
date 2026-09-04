"""Behavioral count-lattice, weight, and initializer tests for NB2 LSS."""

from __future__ import annotations

import math
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy import special

import superglm.distributional.families.negative_binomial as nb_module
from superglm import SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.families.negative_binomial import (
    NegativeBinomialInitializationError,
    NegativeBinomialLikelihoodPlan,
    NegativeBinomialLS,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    LikelihoodPlanValidatingFamily,
    validate_family,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    WeightSemantics,
    resolve_likelihood_weights,
)
from superglm.features import Categorical
from superglm.links import LogLink
from tests._negative_binomial_lss_oracles import NEGATIVE_BINOMIAL_LSS_CASES


def _plan(
    response: np.ndarray,
    weights: np.ndarray,
    semantics: WeightSemantics,
) -> NegativeBinomialLikelihoodPlan:
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics),
    )
    return NegativeBinomialLS().bind_likelihood(
        response,
        resolved,
        COMPLETE_OBSERVATION,
    )


def _assert_enclosed(
    actual: np.ndarray,
    expected: tuple[float, ...],
    rtol: tuple[float, ...],
    atol: tuple[float, ...],
) -> None:
    actual_values = np.asarray(actual)
    expected_values = np.asarray(expected)
    error = np.abs(actual_values - expected_values)
    bound = np.asarray(atol) + np.asarray(rtol) * np.abs(expected_values)
    assert np.all(error <= bound), f"error {error!r} exceeds bound {bound!r}"


def test_family_declares_the_public_nb2_mean_theta_contract() -> None:
    family = NegativeBinomialLS()
    parameters = validate_family(family)

    assert tuple(parameter.name for parameter in parameters) == ("mean", "theta")
    assert tuple(parameter.role for parameter in parameters) == ("mean", "size")
    assert tuple(parameter.curvature for parameter in parameters) == ("observed", "observed")
    assert all(type(parameter.default_link) is LogLink for parameter in parameters)
    assert family.default_prediction_name == "conditional_mean"
    assert family.capabilities.max_derivative_order == 2
    assert family.capabilities.expected_information is False
    assert family.to_config() == {
        "type": "NegativeBinomialLS",
        "parameterization": "nb2_mean_theta",
    }


@pytest.mark.parametrize(
    ("response", "exposure", "count"),
    [
        (0.0, 0.1, 0.0),
        (1.0 / 1.5, 1.5, 1.0),
        (7.0 / (10.0 / 3.0), 10.0 / 3.0, 7.0),
        (174.64208857839304, 18.649570825179428, 3257.0),
        (99999.0 / math.nextafter(1.0, 2.0), math.nextafter(1.0, 2.0), 99999.0),
    ],
)
def test_prior_accepts_unique_binary64_count_over_exposure_encodings(
    response: float,
    exposure: float,
    count: float,
) -> None:
    plan = _plan(np.array([response]), np.array([exposure]), "prior")

    np.testing.assert_array_equal(plan.exact_count, [count])


@pytest.mark.parametrize(
    ("semantics", "response", "weight"),
    [
        ("frequency", 2.5, 1.0),
        ("prior", 7.25 / 1.3, 1.3),
        ("prior", (1.0 + math.ldexp(1.0, -51)) / 1.5, 1.5),
        ("prior", 1.0, 1.0e-20),
    ],
)
def test_noncanonical_count_lattice_encodings_are_refused(
    semantics: WeightSemantics,
    response: float,
    weight: float,
) -> None:
    with pytest.raises(
        UnsupportedLikelihoodContractError,
        match="count|integer|lattice",
    ):
        _plan(np.array([response]), np.array([weight]), semantics)


def test_integer_response_is_checked_before_lossy_float64_coercion() -> None:
    with pytest.raises(ValueError, match="lossless|float64|exact"):
        _plan(np.array([2**53 + 1], dtype=np.int64), np.ones(1), "frequency")


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_likelihood_plan_owns_response_count_and_factorial_carrier(
    semantics: WeightSemantics,
) -> None:
    if semantics == "prior":
        count = np.array([0.0, 3.0, 5.0])
        weights = np.array([0.5, 1.5, 2.5])
        response = count / weights
        expected_carrier = -special.gammaln(count + 1.0)
    else:
        count = np.array([0.0, 2.0, 5.0])
        weights = np.array([1.0, 3.0, 2.0])
        response = count.copy()
        expected_carrier = -weights * special.gammaln(count + 1.0)
    original_response = response.copy()
    plan = _plan(response, weights, semantics)

    response[:] = 99.0
    weights[:] = 99.0

    np.testing.assert_array_equal(plan.exact_response, original_response)
    np.testing.assert_array_equal(plan.exact_count, count)
    np.testing.assert_allclose(
        plan.parameter_independent_carrier,
        expected_carrier,
        rtol=0.0,
        atol=4.0 * np.finfo(np.float64).eps,
    )
    for values in (
        plan.exact_response,
        plan.exact_count,
        plan.parameter_independent_carrier,
    ):
        assert not values.flags.writeable


def test_plan_shares_the_resolved_root_and_reconstructs_its_identifier() -> None:
    family = NegativeBinomialLS()
    response = np.array([0.0, 2.0, 5.0])
    resolved = resolve_likelihood_weights(
        np.ones(3),
        n_observations=3,
        contract=WeightContract("prior"),
    )
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    reconstructed = _plan(response.copy(), np.ones(3), "prior")
    frequency = _plan(response.copy(), np.ones(3), "frequency")

    assert plan.weights is resolved
    assert reconstructed.plan_identifier == plan.plan_identifier
    assert frequency.plan_identifier != plan.plan_identifier


@pytest.mark.parametrize(
    ("semantics", "response", "weights", "expected_identifier"),
    [
        (
            "prior",
            [0.0, 1.0, 4.0],
            [0.5, 2.0, 1.5],
            "NegativeBinomialLS/v1:"
            "428e6a7ac447988e6bde6a086a4413e9784974147ecede37c2c7bf4b1ab60460",
        ),
        (
            "frequency",
            [0.0, 2.0, 5.0],
            [1.0, 3.0, 2.0],
            "NegativeBinomialLS/v1:"
            "4912347aefe21b494b0946d629061e2d64999388d992ce968f89d374a1f3259e",
        ),
    ],
)
def test_plan_identifier_preserves_v1_artifact_encoding(
    semantics: WeightSemantics,
    response: list[float],
    weights: list[float],
    expected_identifier: str,
) -> None:
    plan = _plan(np.array(response), np.array(weights), semantics)

    assert plan.plan_identifier == expected_identifier


def test_nb2_plan_subclass_is_validated_structurally() -> None:
    class NegativeBinomialPlanSubclass(NegativeBinomialLikelihoodPlan):
        pass

    family = NegativeBinomialLS()
    response = np.array([0.0, 2.0])
    parameters = np.array([[1.2, 0.8], [2.1, 1.3]])
    base = _plan(response, np.ones(2), "prior")
    plan = NegativeBinomialPlanSubclass(**vars(base))

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, parameters, plan)

    np.testing.assert_equal(initialized.theta, family.initialize(response, base).theta)
    np.testing.assert_equal(
        evaluated.reported_log_likelihood,
        family.evaluate_natural(response, parameters, base).reported_log_likelihood,
    )
    malformed = NegativeBinomialPlanSubclass(
        **(vars(base) | {"weights": base.weights.take(np.array([0], dtype=np.intp))})
    )
    with pytest.raises(UnsupportedLikelihoodContractError, match="rows"):
        family.validate_likelihood_plan(response, malformed)


def test_nb2_response_carrier_plan_uses_public_one_shot_protocol() -> None:
    family = NegativeBinomialLS()
    response = np.array([0.0, 2.0])
    plan = _plan(response, np.ones(2), "prior")

    assert isinstance(family, LikelihoodPlanValidatingFamily)
    canonical = family.validate_likelihood_plan(response, plan)

    assert canonical is plan.exact_response
    assert canonical.dtype == np.float64
    assert not canonical.flags.writeable


def test_unit_prior_and_frequency_weights_agree() -> None:
    family = NegativeBinomialLS()
    response = np.array([0.0, 2.0, 7.0])
    parameters = np.array([[2.5, 0.35], [0.8, 0.65], [2.0, 1.0e8]])
    prior = family.evaluate_natural(
        response,
        parameters,
        _plan(response, np.ones(3), "prior"),
    )
    frequency = family.evaluate_natural(
        response,
        parameters,
        _plan(response, np.ones(3), "frequency"),
    )

    for prior_values, frequency_values in (
        (prior.reported_log_likelihood, frequency.reported_log_likelihood),
        (prior.score, frequency.score),
        (prior.hessian_packed, frequency.hessian_packed),
    ):
        np.testing.assert_allclose(prior_values, frequency_values, rtol=2.0e-15, atol=2.0e-15)


def test_frequency_weight_is_literal_row_replication() -> None:
    family = NegativeBinomialLS()
    response = np.array([2.0])
    parameters = np.array([[0.8, 0.65]])
    unit = family.evaluate_natural(
        response,
        parameters,
        _plan(response, np.ones(1), "frequency"),
    )
    replicated = family.evaluate_natural(
        response,
        parameters,
        _plan(response, np.array([3.0]), "frequency"),
    )

    for replicated_values, unit_values in (
        (replicated.reported_log_likelihood, unit.reported_log_likelihood),
        (replicated.score, unit.score),
        (replicated.hessian_packed, unit.hessian_packed),
    ):
        np.testing.assert_allclose(
            replicated_values,
            3.0 * unit_values,
            rtol=2.0e-15,
            atol=2.0e-15,
        )


def test_nonunit_prior_family_adapter_matches_the_frozen_complete_row_oracle() -> None:
    family = NegativeBinomialLS()
    case = next(
        item for item in NEGATIVE_BINOMIAL_LSS_CASES if item.id == "fractional-prior-exposure"
    )
    exposure = case.weight
    response = np.array([case.count / exposure])
    parameters = np.array([[case.mean, case.theta]])
    evaluated = family.evaluate_natural(
        response,
        parameters,
        _plan(response, np.array([exposure]), "prior"),
    )

    assert evaluated.optimizing_log_likelihood[0] == pytest.approx(
        case.optimizing_log_likelihood,
        abs=case.value_atol,
    )
    assert evaluated.parameter_independent_carrier[0] == pytest.approx(
        case.factorial_carrier,
        abs=case.value_atol,
    )
    assert evaluated.reported_log_likelihood[0] == pytest.approx(
        case.full_log_likelihood,
        abs=case.value_atol,
    )
    assert evaluated.score is not None
    assert evaluated.hessian_packed is not None
    _assert_enclosed(
        evaluated.score[0],
        case.natural_score,
        case.score_rtol,
        case.score_atol,
    )
    _assert_enclosed(
        evaluated.hessian_packed[0],
        case.natural_hessian_packed,
        case.hessian_rtol,
        case.hessian_atol,
    )


def test_fractional_prior_is_not_uniformly_scaled_frequency_likelihood() -> None:
    family = NegativeBinomialLS()
    case = next(
        item for item in NEGATIVE_BINOMIAL_LSS_CASES if item.id == "fractional-prior-exposure"
    )
    exposure = case.weight
    response = np.array([case.count / exposure])
    unit = family.evaluate_natural(
        response,
        np.array([[case.mean, case.theta]]),
        _plan(response, np.ones(1), "frequency"),
    )
    assert unit.score is not None and unit.hessian_packed is not None
    assert not np.allclose(
        exposure * unit.score[0],
        case.natural_score,
        rtol=1.0e-8,
        atol=1.0e-8,
    )
    assert not np.allclose(
        exposure * unit.hessian_packed[0],
        case.natural_hessian_packed,
        rtol=1.0e-8,
        atol=1.0e-8,
    )


@pytest.mark.parametrize(
    ("semantics", "response", "weights", "expected_mean", "expected_theta"),
    [
        (
            "frequency",
            np.array([0.0, 1.0, 4.0, 8.0]),
            np.array([1.0, 2.0, 3.0, 1.0]),
            22.0 / 7.0,
            69.14285714285714 / 22.85714285714286,
        ),
        (
            "prior",
            np.array([0.0, 1.0, 4.0, 6.0]),
            np.array([0.5, 1.0, 2.0, 2.5]),
            4.0,
            64.0 / 11.0,
        ),
    ],
)
def test_initializer_uses_the_weight_specific_nb2_moment_equation(
    semantics: WeightSemantics,
    response: np.ndarray,
    weights: np.ndarray,
    expected_mean: float,
    expected_theta: float,
) -> None:
    initialized = NegativeBinomialLS().initialize(
        response,
        _plan(response, weights, semantics),
    )

    np.testing.assert_allclose(initialized.theta[:, 0], expected_mean, rtol=3.0e-15)
    np.testing.assert_allclose(initialized.theta[:, 1], expected_theta, rtol=3.0e-15)


def test_poisson_like_sample_gets_a_finite_interior_initializer() -> None:
    family = NegativeBinomialLS()
    response = np.ones(3)
    plan = _plan(response, np.array([1.0, 2.0, 3.0]), "frequency")

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, initialized.theta, plan)

    assert np.all(np.isfinite(initialized.theta))
    assert np.all(initialized.theta > 0.0)
    assert np.all(np.isfinite(evaluated.reported_log_likelihood))


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_subunit_mean_poisson_like_initializer_is_strictly_inside_kernel_ratio(
    semantics: WeightSemantics,
) -> None:
    family = NegativeBinomialLS()
    response = np.array([0.0, 0.0, 1.0])
    plan = _plan(response, np.ones(3), semantics)

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, initialized.theta, plan)

    mean_theta_ratio = initialized.theta[:, 0] / initialized.theta[:, 1]
    assert np.all(mean_theta_ratio > math.sqrt(np.finfo(np.float64).eps))
    assert np.all(np.isfinite(initialized.theta))
    assert np.all(np.isfinite(evaluated.reported_log_likelihood))


def test_prior_boundary_diagnosis_passes_the_certified_nonroundtrip_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = np.array([7.0 / 0.3])
    weights = resolve_likelihood_weights(
        np.array([0.3]),
        n_observations=1,
        contract=WeightContract("prior"),
    )
    evidence = Mock(return_value=True)
    monkeypatch.setattr(
        nb_module,
        "has_resolved_poisson_boundary",
        evidence,
        raising=False,
    )

    diagnosed = NegativeBinomialLS().diagnose_repeated_curvature_failure(
        response,
        weights,
    )

    assert type(diagnosed) is nb_module.NegativeBinomialPoissonBoundaryError
    evidence.assert_called_once()
    passed_response, exact_count, passed_weights, semantics = evidence.call_args.args
    reconstructed = passed_weights * passed_response
    np.testing.assert_array_equal(exact_count, [7.0])
    assert reconstructed[0] == np.nextafter(7.0, np.inf)
    assert not np.array_equal(reconstructed, exact_count)
    assert semantics == "prior"


def test_all_zero_sample_has_a_typed_initializer_refusal() -> None:
    response = np.zeros(3)

    with pytest.raises(NegativeBinomialInitializationError, match="all-zero|mean"):
        NegativeBinomialLS().initialize(
            response,
            _plan(response, np.ones(3), "frequency"),
        )


@pytest.mark.slow
def test_book_shaped_nb2_fit_is_accepted_and_matches_the_aggregated_fit() -> None:
    """2,000,000 synthetic rows with mean count 0.6 fit through the public API; the same data
    aggregated to (cell, count) with frequency weights gives the same log-likelihood to 1e-9."""
    rng = np.random.default_rng(3)
    n = 2_000_000
    cell = rng.choice(np.array(["a", "b", "c"]), size=n)
    mean = np.select([cell == "a", cell == "b"], [0.45, 0.6], 0.75)
    theta = np.select([cell == "a", cell == "b"], [0.7, 0.8], 1.0)
    counts = rng.negative_binomial(theta, theta / (mean + theta)).astype(np.float64)
    assert counts.sum() > 1_000_000
    frame = pd.DataFrame({"cell": cell})

    def predictors() -> tuple[Predictor, Predictor]:
        return (
            Predictor("mean", {"cell": Categorical()}),
            Predictor("theta", {"cell": Categorical()}),
        )

    per_row = SuperLSS(family=NegativeBinomialLS(), predictors=predictors()).fit(
        frame, counts, lambdas={}, inner_tol=1.0e-9
    )
    aggregated = (
        pd.DataFrame({"cell": cell, "count": counts})
        .groupby(["cell", "count"], as_index=False)
        .size()
    )
    assert len(aggregated) < 100
    compact = SuperLSS(
        family=NegativeBinomialLS(),
        predictors=predictors(),
        weight_semantics="frequency",
    ).fit(
        aggregated[["cell"]],
        aggregated["count"].to_numpy(dtype=np.float64),
        sample_weight=aggregated["size"].to_numpy(dtype=np.float64),
        lambdas={},
        inner_tol=1.0e-9,
    )
    per_row_total = per_row._require_fitted().fit_state.solver_result.log_likelihood
    compact_total = compact._require_fitted().fit_state.solver_result.log_likelihood
    assert per_row._require_fitted().fit_state.solver_result.converged
    assert compact._require_fitted().fit_state.solver_result.converged
    assert abs(per_row_total - compact_total) <= 1.0e-9 * abs(per_row_total)
