from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from superglm.distributional.derivatives import transform_natural_derivatives
from superglm.distributional.families.gaussian import (
    GaussianLikelihoodPlan,
    GaussianLS,
    LowerBoundedLogLink,
)
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    ExpectedInformationFamily,
    validate_family,
)
from superglm.distributional.weights import (
    LikelihoodWeightError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink
from tests._distributional_family_kernels import gaussian as gaussian_kernel


def _plan(
    family: GaussianLS,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    semantics: str,
):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(y),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    return family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)


@pytest.mark.parametrize("scale_floor", [-0.1, np.nan, np.inf, -np.inf, True])
def test_gaussian_ls_rejects_invalid_scale_floor(scale_floor: float) -> None:
    with pytest.raises(ValueError, match="scale_floor.*finite.*non-negative"):
        GaussianLS(scale_floor=scale_floor)


def test_gaussian_ls_configuration_and_parameter_metadata_are_serializable() -> None:
    family = GaussianLS(scale_floor=0.025)
    parameters = validate_family(family)

    assert family.to_config() == {"type": "GaussianLS", "scale_floor": 0.025}
    assert json.loads(json.dumps(family.to_config())) == family.to_config()
    assert tuple(parameter.name for parameter in parameters) == ("location", "scale")
    assert tuple(parameter.role for parameter in parameters) == ("location", "scale")
    assert tuple(parameter.curvature for parameter in parameters) == ("fisher", "fisher")
    assert isinstance(parameters[0].default_link, IdentityLink)
    assert parameters[0].support.contains(np.array([-np.inf, -2.0, 0.0, np.inf])).tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert parameters[1].support.lower == 0.025
    assert parameters[1].support.lower_inclusive is False
    assert isinstance(parameters[1].default_link, LowerBoundedLogLink)
    assert family.default_prediction_name == "conditional_mean"
    assert family.capabilities.max_derivative_order == 2
    assert family.capabilities.expected_information is True
    assert family.capabilities.response_mean is True
    assert family.capabilities.cdf is True
    assert family.capabilities.quantile is True
    assert isinstance(family, ExpectedInformationFamily)

    with pytest.raises(FrozenInstanceError):
        family.scale_floor = 1.0  # ty: ignore[invalid-assignment]


def test_lower_bounded_log_link_round_trip_and_inverse_derivatives() -> None:
    floor = 0.01
    link = LowerBoundedLogLink(floor)
    eta = np.array([-30.0, -7.0, -1.0, 0.0, 3.0])
    expected_increment = np.exp(eta)

    sigma = link.inverse(eta)

    np.testing.assert_allclose(sigma, floor + expected_increment, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(link.link(sigma), eta, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(link.deriv_inverse(eta), expected_increment)
    np.testing.assert_allclose(link.deriv2_inverse(eta), expected_increment)
    np.testing.assert_allclose(link.deriv3_inverse(eta), expected_increment)
    np.testing.assert_allclose(link.deriv(sigma), 1.0 / expected_increment, rtol=2e-6)
    with pytest.raises(ValueError, match="strictly above.*floor"):
        link.link(np.array([floor]))


def test_prior_initialization_uses_physical_row_normalizer_mass() -> None:
    family = GaussianLS(scale_floor=0.01)
    y = np.array([-1.0, 0.5, 2.0, 5.0])
    weights = np.array([0.5, 1.0, 2.0, 3.0])
    expected_mu = np.average(y, weights=weights)
    expected_sigma = np.sqrt(np.dot(weights, (y - expected_mu) ** 2) / len(y))

    initialized = family.initialize(y, _plan(family, y, weights, semantics="prior"))

    initialized.validate_shape(n_observations=4, k_parameters=2)
    np.testing.assert_allclose(initialized.theta[:, 0], expected_mu)
    np.testing.assert_allclose(initialized.theta[:, 1], expected_sigma)
    assert not initialized.theta.flags.writeable


def test_frequency_initialization_uses_literal_replication_mass() -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([-1.0, 0.5, 2.0, 5.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    expected_mu = np.average(y, weights=weights)
    expected_sigma = np.sqrt(np.dot(weights, (y - expected_mu) ** 2) / int(np.sum(weights)))

    initialized = family.initialize(
        y,
        _plan(family, y, weights, semantics="frequency"),
    )

    np.testing.assert_allclose(initialized.theta[:, 0], expected_mu)
    np.testing.assert_allclose(initialized.theta[:, 1], expected_sigma)


@pytest.mark.parametrize(
    ("semantics", "weights", "child_indices"),
    [
        ("prior", np.array([0.5, 1.5, 2.0, 3.0]), np.array([3, 1], dtype=np.intp)),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0]), np.array([3, 1], dtype=np.intp)),
    ],
)
def test_child_initialization_uses_selected_likelihood_mass(
    semantics: str,
    weights: np.ndarray,
    child_indices: np.ndarray,
) -> None:
    family = GaussianLS(scale_floor=0.0)
    root_y = np.array([-1.0, 0.5, 2.0, 5.0])
    root = _plan(family, root_y, weights, semantics=semantics)
    child = root.take(child_indices)
    y = root_y[child_indices]
    selected_weights = weights[child_indices]
    expected_mu = np.average(y, weights=selected_weights)
    denominator = len(y) if semantics == "prior" else sum(int(w) for w in selected_weights)
    expected_sigma = np.sqrt(np.dot(selected_weights, (y - expected_mu) ** 2) / denominator)

    initialized = family.initialize(y, child)

    assert child.weights.provenance is root.weights.provenance
    assert child.weights.physical_count == len(root_y)
    np.testing.assert_allclose(initialized.theta[:, 0], expected_mu)
    np.testing.assert_allclose(initialized.theta[:, 1], expected_sigma)


def test_constant_response_initialization_stays_strictly_above_scale_floor() -> None:
    family = GaussianLS(scale_floor=0.2)
    y = np.ones(5)

    initialized = family.initialize(y, _plan(family, y, np.ones(5), semantics="prior"))

    assert np.all(initialized.theta[:, 1] > family.scale_floor)
    assert np.all(np.isfinite(initialized.theta))


def test_gaussian_primitive_kernel_repeats_and_matches_every_public_entry_point() -> None:
    family = GaussianLS(scale_floor=0.0)
    response = np.array([-0.5, 2.0])
    theta = np.array([[0.0, 0.7], [1.5, 1.2]])
    plan = _plan(family, response, np.ones(2), semantics="prior")
    expected_initial = family.initialize(response, plan).theta
    expected_evaluation = family.evaluate_natural(response, theta, plan)
    expected_information = family.expected_information_natural(theta, plan)

    for _ in range(2):
        actual_initial = gaussian_kernel.initialize_gaussian(
            response,
            plan.weights.values,
            "prior",
            family.scale_floor,
        )
        actual_evaluation = gaussian_kernel.evaluate_gaussian_rows(
            response,
            theta[:, 0],
            theta[:, 1],
            plan.weights.values,
            "prior",
            derivative_order=2,
        )
        actual_information = gaussian_kernel.gaussian_expected_information(
            theta[:, 1],
            plan.weights.values,
            "prior",
        )

        np.testing.assert_equal(actual_initial, expected_initial)
        np.testing.assert_equal(
            actual_evaluation.optimizing_log_likelihood,
            expected_evaluation.optimizing_log_likelihood,
        )
        np.testing.assert_equal(actual_evaluation.score, expected_evaluation.score)
        np.testing.assert_equal(
            actual_evaluation.hessian_packed,
            expected_evaluation.hessian_packed,
        )
        np.testing.assert_equal(actual_evaluation.valid, expected_evaluation.valid)
        np.testing.assert_equal(actual_information, expected_information)


def _call_gaussian_primitive_entry_point(entry_point: str, vector: object) -> None:
    location = np.array([0.4, 1.6], dtype=np.float64)
    scale = np.array([0.7, 1.2], dtype=np.float64)
    weights = np.ones(2, dtype=np.float64)
    if entry_point == "initialize":
        gaussian_kernel.initialize_gaussian(vector, weights, "prior", 0.0)
    elif entry_point == "evaluate":
        gaussian_kernel.evaluate_gaussian_rows(
            vector,
            location,
            scale,
            weights,
            "prior",
            derivative_order=2,
        )
    elif entry_point == "information":
        gaussian_kernel.gaussian_expected_information(vector, weights, "prior")
    else:
        gaussian_kernel.gaussian_predictor_curvature_directional(
            vector,
            np.column_stack((location, np.log(scale))),
            np.ones((2, 2), dtype=np.float64),
            weights,
            "prior",
            scale_floor=0.0,
        )


@pytest.mark.parametrize(
    ("mutation", "vector"),
    [
        ("non-array", [0.5, 1.5]),
        ("string", np.array(["0.5", "1.5"])),
        ("integer", np.array([1, 2], dtype=np.int64)),
        ("float32", np.array([0.5, 1.5], dtype=np.float32)),
        ("boolean", np.array([True, True])),
        ("non-vector", np.array([[0.5, 1.5]], dtype=np.float64)),
        ("nonfinite", np.array([0.5, np.nan], dtype=np.float64)),
    ],
)
@pytest.mark.parametrize("entry_point", ["initialize", "evaluate", "information", "directional"])
def test_gaussian_primitive_entry_points_refuse_vector_coercion(
    entry_point: str,
    mutation: str,
    vector: object,
) -> None:
    """Kills coercion of non-literal float64 vectors at every primitive boundary."""

    del mutation
    with pytest.raises(ValueError, match="float64"):
        _call_gaussian_primitive_entry_point(entry_point, vector)


def test_gaussian_kernel_evaluation_owns_and_freezes_every_array_field() -> None:
    """Kills writable or caller-aliased arrays in the Gaussian kernel record."""

    sources = {
        "optimizing_log_likelihood": np.array([-0.5, -1.5], dtype=np.float64),
        "score": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "hessian_packed": np.array([[-1.0, 0.25, -2.0], [-3.0, 0.5, -4.0]], dtype=np.float64),
        "valid": np.ones(2, dtype=bool),
    }
    expected = {name: values.copy() for name, values in sources.items()}
    evaluation = gaussian_kernel.GaussianKernelEvaluation(**sources)

    for values in sources.values():
        values[...] = 0

    for name, expected_values in expected.items():
        actual = getattr(evaluation, name)
        np.testing.assert_array_equal(actual, expected_values)
        assert actual.flags.owndata
        assert not actual.flags.writeable
        with pytest.raises(ValueError):
            actual.flat[0] = 0


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("optimizing_log_likelihood", np.ones((2, 1), dtype=np.float64)),
        ("optimizing_log_likelihood", np.array([0.0, np.nan], dtype=np.float64)),
        ("score", np.ones((2, 1), dtype=np.float64)),
        ("score", np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64)),
        ("hessian_packed", np.ones((2, 2), dtype=np.float64)),
        (
            "hessian_packed",
            np.array([[0.0, 0.0, np.inf], [0.0, 0.0, 0.0]], dtype=np.float64),
        ),
        ("valid", np.ones((2, 1), dtype=bool)),
        ("valid", np.ones(2, dtype=np.float64)),
    ],
)
def test_gaussian_kernel_evaluation_rejects_malformed_array_fields(
    field: str,
    malformed: np.ndarray,
) -> None:
    """Kills Gaussian kernel records that do not certify result shape and finiteness."""

    fields = {
        "optimizing_log_likelihood": np.zeros(2, dtype=np.float64),
        "score": np.zeros((2, 2), dtype=np.float64),
        "hessian_packed": np.zeros((2, 3), dtype=np.float64),
        "valid": np.ones(2, dtype=bool),
    }
    fields[field] = malformed

    with pytest.raises(ValueError, match=field):
        gaussian_kernel.GaussianKernelEvaluation(**fields)


def test_gaussian_plan_subclass_is_validated_structurally() -> None:
    class GaussianPlanSubclass(GaussianLikelihoodPlan):
        pass

    family = GaussianLS(scale_floor=0.0)
    response = np.array([-0.5, 2.0])
    theta = np.array([[0.0, 0.7], [1.5, 1.2]])
    base = _plan(family, response, np.ones(2), semantics="prior")
    plan = GaussianPlanSubclass(
        weights=base.weights,
        row_law=base.row_law,
        invariant=base.invariant,
        family_config=base.family_config,
        observation=base.observation,
    )

    initialized = family.initialize(response, plan)
    evaluated = family.evaluate_natural(response, theta, plan)

    np.testing.assert_equal(initialized.theta, family.initialize(response, base).theta)
    np.testing.assert_equal(
        evaluated.reported_log_likelihood,
        family.evaluate_natural(response, theta, base).reported_log_likelihood,
    )


def test_prior_gaussian_rows_match_literal_variance_sigma2_over_weight_oracle() -> None:
    family = GaussianLS(scale_floor=0.01)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    mu = np.array([-0.8, 0.5, 1.7, 2.0])
    sigma = np.array([0.3, 0.35, 1.2, 4.5])
    theta = np.column_stack((mu, sigma))
    weights = np.array([0.25, 0.8, 2.5, 4.0])
    residual = y - mu

    plan = _plan(family, y, weights, semantics="prior")
    assert isinstance(plan, GaussianLikelihoodPlan)
    assert plan.row_law == "normal-variance-sigma2-over-w/v1"
    assert plan.invariant == "conditional-location"
    actual = family.evaluate_natural(y, theta, plan)
    expected_optimizing = (
        -np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * weights * residual**2 / sigma**2
    )
    expected_carrier = 0.5 * np.log(weights)
    expected_score = np.column_stack(
        (
            weights * residual / sigma**2,
            -1.0 / sigma + weights * residual**2 / sigma**3,
        )
    )
    expected_hessian = np.column_stack(
        (
            -weights / sigma**2,
            -2.0 * weights * residual / sigma**3,
            1.0 / sigma**2 - 3.0 * weights * residual**2 / sigma**4,
        )
    )
    tolerance = 32.0 * len(y) * np.finfo(np.float64).eps

    np.testing.assert_allclose(
        actual.optimizing_log_likelihood,
        expected_optimizing,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.parameter_independent_carrier,
        expected_carrier,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        actual.reported_log_likelihood,
        expected_optimizing + expected_carrier,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(actual.score, expected_score, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual.hessian_packed,
        expected_hessian,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_array_equal(actual.valid, np.ones(len(y), dtype=bool))


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [
        ("prior", np.array([0.25, 0.8, 2.5, 4.0])),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_gaussian_primitive_rows_match_independent_oracle(
    semantics: str,
    weights: np.ndarray,
) -> None:
    response = np.array([-1.2, 0.1, 2.4, 8.0])
    location = np.array([-0.8, 0.5, 1.7, 2.0])
    scale = np.array([0.3, 0.35, 1.2, 4.5])
    residual = response - location
    base_normalizer = -np.log(scale) - 0.5 * np.log(2.0 * np.pi)
    if semantics == "prior":
        expected_optimizing = base_normalizer - 0.5 * weights * residual**2 / scale**2
        expected_score = np.column_stack(
            (
                weights * residual / scale**2,
                -1.0 / scale + weights * residual**2 / scale**3,
            )
        )
        expected_hessian = np.column_stack(
            (
                -weights / scale**2,
                -2.0 * weights * residual / scale**3,
                1.0 / scale**2 - 3.0 * weights * residual**2 / scale**4,
            )
        )
    else:
        expected_optimizing = weights * (base_normalizer - 0.5 * residual**2 / scale**2)
        expected_score = weights[:, None] * np.column_stack(
            (residual / scale**2, -1.0 / scale + residual**2 / scale**3)
        )
        expected_hessian = weights[:, None] * np.column_stack(
            (
                -1.0 / scale**2,
                -2.0 * residual / scale**3,
                1.0 / scale**2 - 3.0 * residual**2 / scale**4,
            )
        )

    actual = gaussian_kernel.evaluate_gaussian_rows(
        response,
        location,
        scale,
        weights,
        semantics,  # type: ignore[arg-type]
        derivative_order=2,
    )
    tolerance = 32.0 * len(response) * np.finfo(np.float64).eps

    np.testing.assert_allclose(
        actual.optimizing_log_likelihood,
        expected_optimizing,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_allclose(actual.score, expected_score, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        actual.hessian_packed,
        expected_hessian,
        rtol=tolerance,
        atol=tolerance,
    )
    np.testing.assert_array_equal(actual.valid, np.ones(len(response), dtype=bool))


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_gaussian_returns_the_exact_requested_derivative_order(semantics: str) -> None:
    """Kills ignored orders and dummy unrequested Gaussian derivative arrays."""

    family = GaussianLS(scale_floor=0.01)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    theta = np.column_stack((np.array([-0.8, 0.5, 1.7, 2.0]), np.array([0.3, 0.35, 1.2, 4.5])))
    weights = np.array([0.25, 0.8, 2.5, 4.0])
    if semantics == "frequency":
        weights = np.array([1.0, 2.0, 3.0, 4.0])
    plan = _plan(family, y, weights, semantics=semantics)
    full = family.evaluate_natural(y, theta, plan, derivative_order=2)
    score_only = family.evaluate_natural(y, theta, plan, derivative_order=1)
    value_only = family.evaluate_natural(y, theta, plan, derivative_order=0)

    assert (value_only.derivative_order, score_only.derivative_order, full.derivative_order) == (
        0,
        1,
        2,
    )
    assert value_only.score is None and value_only.hessian_packed is None
    assert score_only.score is not None and score_only.hessian_packed is None
    np.testing.assert_array_equal(
        value_only.optimizing_log_likelihood, full.optimizing_log_likelihood
    )
    np.testing.assert_array_equal(
        value_only.parameter_independent_carrier, full.parameter_independent_carrier
    )
    np.testing.assert_array_equal(
        score_only.optimizing_log_likelihood, full.optimizing_log_likelihood
    )
    np.testing.assert_array_equal(
        score_only.parameter_independent_carrier, full.parameter_independent_carrier
    )
    np.testing.assert_array_equal(score_only.score, full.score)


def test_gaussian_lower_orders_skip_unrequested_derivative_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills nominal lower orders that still execute higher-order arithmetic."""

    family = GaussianLS(scale_floor=0.01)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    theta = np.column_stack((np.array([-0.8, 0.5, 1.7, 2.0]), np.array([0.3, 0.35, 1.2, 4.5])))
    plan = _plan(family, y, np.ones(len(y)), semantics="prior")

    def explode(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unrequested Gaussian derivative helper executed")

    monkeypatch.setattr(gaussian_kernel.np, "column_stack", explode)
    result = family.evaluate_natural(y, theta, plan, derivative_order=0)
    assert result.derivative_order == 0


def test_gaussian_order_one_skips_hessian_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills order one that computes and discards the Gaussian Hessian."""

    family = GaussianLS(scale_floor=0.01)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    theta = np.column_stack((np.array([-0.8, 0.5, 1.7, 2.0]), np.array([0.3, 0.35, 1.2, 4.5])))
    plan = _plan(family, y, np.ones(len(y)), semantics="prior")

    original = np.column_stack
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gaussian_kernel.np, "column_stack", counted)
    result = family.evaluate_natural(y, theta, plan, derivative_order=1)
    assert result.derivative_order == 1
    assert calls == 1


def test_prior_scale_score_mutation_point_is_not_frequency_score() -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([1.0])
    theta = np.array([[0.0, 1.0]])
    prior = family.evaluate_natural(y, theta, _plan(family, y, np.array([4.0]), semantics="prior"))
    frequency = family.evaluate_natural(
        y,
        theta,
        _plan(family, y, np.array([4.0]), semantics="frequency"),
    )

    assert prior.score[0, 1] == 3.0
    assert frequency.score[0, 1] == 0.0


def test_prior_natural_expected_information_uses_physical_scale_mass() -> None:
    family = GaussianLS(scale_floor=0.01)
    sigma = np.array([0.011, 0.2, 1.5, 7.0])
    theta = np.column_stack((np.array([-2.0, 0.0, 1.0, 3.0]), sigma))
    weights = np.array([0.2, 0.75, 2.0, 5.0])

    information = family.expected_information_natural(
        theta,
        _plan(family, np.zeros(len(theta)), weights, semantics="prior"),
    )
    expected = np.column_stack(
        (
            weights / sigma**2,
            np.zeros(len(sigma)),
            2.0 / sigma**2,
        )
    )

    np.testing.assert_allclose(information, expected)
    assert not information.flags.writeable


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [
        ("prior", np.array([0.25, 0.8, 2.5, 4.0])),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_gaussian_predictor_curvature_direction_matches_independent_secants(
    semantics: str,
    weights: np.ndarray,
) -> None:
    """The endpoint capability differentiates observed predictor curvature."""

    family = GaussianLS(scale_floor=0.2)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    location = np.array([-0.8, 0.5, 1.7, 2.0])
    scale = np.array([0.3, 0.55, 1.2, 4.5])
    eta = np.column_stack((location, np.log(scale - family.scale_floor)))
    direction = np.array(
        [
            [0.25, -0.4],
            [-0.3, 0.15],
            [0.5, 0.2],
            [-0.1, -0.35],
        ],
        dtype=np.float64,
    )
    links = (IdentityLink(), LowerBoundedLogLink(family.scale_floor))
    plan = _plan(family, y, weights, semantics=semantics)

    actual = family.predictor_curvature_directional_derivative(
        y,
        eta,
        direction,
        links,
        plan,
    )

    def curvature(step: float) -> np.ndarray:
        candidate_eta = eta + step * direction
        candidate_theta = np.column_stack(
            (
                candidate_eta[:, 0],
                family.scale_floor + np.exp(candidate_eta[:, 1]),
            )
        )
        natural = family.evaluate_natural(y, candidate_theta, plan, derivative_order=2)
        return transform_natural_derivatives(
            natural,
            candidate_eta,
            links,
        ).curvature_packed

    errors = []
    for step in (2.0e-4, 1.0e-4, 5.0e-5):
        secant = (curvature(step) - curvature(-step)) / (2.0 * step)
        errors.append(float(np.max(np.abs(actual - secant))))
    assert errors[2] < errors[1] < errors[0]
    scale_bound = max(float(np.max(np.abs(actual))), 1.0)
    assert errors[-1] <= 2.0e-7 * scale_bound
    assert not actual.flags.writeable


@pytest.mark.parametrize(
    ("semantics", "expected_scale_channel"),
    [
        ("prior", -74.0 / 27.0),
        ("frequency", -80.0 / 27.0),
    ],
)
def test_gaussian_predictor_curvature_direction_preserves_cross_and_scale_channels(
    semantics: str,
    expected_scale_channel: float,
) -> None:
    """Kills dropping the cross path or giving prior scale mass frequency semantics."""

    family = GaussianLS(scale_floor=1.0)
    y = np.array([3.0])
    eta = np.array([[0.0, np.log(2.0)]])
    direction = np.ones((1, 2), dtype=np.float64)
    plan = _plan(family, y, np.array([4.0]), semantics=semantics)

    actual = family.predictor_curvature_directional_derivative(
        y,
        eta,
        direction,
        (IdentityLink(), LowerBoundedLogLink(family.scale_floor)),
        plan,
    )

    np.testing.assert_allclose(
        actual,
        np.array([[-16.0 / 27.0, -64.0 / 27.0, expected_scale_channel]]),
        rtol=32.0 * np.finfo(np.float64).eps,
        atol=32.0 * np.finfo(np.float64).eps,
    )


def test_frequency_gaussian_rows_preserve_literal_replication_contribution() -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    mu = np.array([-0.8, 0.5, 1.7, 2.0])
    sigma = np.array([0.3, 0.35, 1.2, 4.5])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    residual = y - mu
    theta = np.column_stack((mu, sigma))
    plan = _plan(family, y, weights, semantics="frequency")
    assert isinstance(plan, GaussianLikelihoodPlan)
    assert plan.row_law == "normal-literal-replication/v1"
    assert plan.invariant == "literal-row-replication"

    actual = family.evaluate_natural(y, theta, plan)
    information = family.expected_information_natural(theta, plan)
    base = -np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * residual**2 / sigma**2
    score = weights[:, None] * np.column_stack(
        (residual / sigma**2, -1.0 / sigma + residual**2 / sigma**3)
    )
    hessian = weights[:, None] * np.column_stack(
        (
            -1.0 / sigma**2,
            -2.0 * residual / sigma**3,
            1.0 / sigma**2 - 3.0 * residual**2 / sigma**4,
        )
    )
    expected_information = np.column_stack(
        (weights / sigma**2, np.zeros(len(y)), 2.0 * weights / sigma**2)
    )
    tolerance = 32.0 * len(y) * np.finfo(np.float64).eps

    np.testing.assert_allclose(
        actual.optimizing_log_likelihood, weights * base, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_array_equal(actual.parameter_independent_carrier, np.zeros(len(y)))
    np.testing.assert_allclose(
        actual.reported_log_likelihood, weights * base, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(actual.score, score, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual.hessian_packed, hessian, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(information, expected_information, rtol=tolerance, atol=tolerance)


def test_unit_prior_and_frequency_channels_and_initialization_agree() -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    theta = np.column_stack((np.array([-0.8, 0.5, 1.7, 2.0]), np.array([0.3, 0.35, 1.2, 4.5])))
    weights = np.ones(len(y))
    prior_plan = _plan(family, y, weights, semantics="prior")
    frequency_plan = _plan(family, y, weights, semantics="frequency")

    prior = family.evaluate_natural(y, theta, prior_plan)
    frequency = family.evaluate_natural(y, theta, frequency_plan)
    prior_information = family.expected_information_natural(theta, prior_plan)
    frequency_information = family.expected_information_natural(theta, frequency_plan)
    prior_initial = family.initialize(y, prior_plan)
    frequency_initial = family.initialize(y, frequency_plan)
    tolerance = 64.0 * len(y) * np.finfo(np.float64).eps

    for left, right in (
        (prior.optimizing_log_likelihood, frequency.optimizing_log_likelihood),
        (prior.reported_log_likelihood, frequency.reported_log_likelihood),
        (prior.score, frequency.score),
        (prior.hessian_packed, frequency.hessian_packed),
        (prior_information, frequency_information),
        (prior_initial.theta, frequency_initial.theta),
    ):
        np.testing.assert_allclose(left, right, rtol=tolerance, atol=tolerance)


def test_prior_plan_children_derive_the_exact_ordered_carrier() -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    weights = resolve_likelihood_weights(
        np.array([0.25, 0.8, 2.5, 4.0]),
        n_observations=len(y),
        contract=WeightContract(semantics="prior"),
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    theta = np.column_stack((np.zeros(len(y)), np.ones(len(y))))

    family.evaluate_natural(y, theta, plan)
    family.evaluate_natural(y, theta, plan)
    child_indices = np.array([3, 1], dtype=np.intp)
    child = plan.take(child_indices)
    family.evaluate_natural(y[child_indices], theta[child_indices], child)

    np.testing.assert_array_equal(
        child.parameter_independent_carrier,
        plan.parameter_independent_carrier[child_indices],
    )
    assert not plan.parameter_independent_carrier.flags.writeable
    assert not child.parameter_independent_carrier.flags.writeable


@pytest.mark.parametrize(
    ("semantics", "weight_values"),
    [
        ("prior", np.array([0.25, 0.8, 2.5, 4.0])),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_root_and_child_carriers_cannot_be_made_writeable(
    semantics: str,
    weight_values: np.ndarray,
) -> None:
    family = GaussianLS(scale_floor=0.0)
    y = np.array([-1.2, 0.1, 2.4, 8.0])
    root = _plan(family, y, weight_values, semantics=semantics)
    child = root.take(np.array([3, 1], dtype=np.intp))

    for plan in (root, child):
        original_carrier = plan.parameter_independent_carrier.copy()
        original_identifier = plan.plan_identifier
        with pytest.raises(ValueError):
            plan.parameter_independent_carrier.setflags(write=True)
        with pytest.raises(ValueError):
            plan.parameter_independent_carrier[0] += 1.0
        np.testing.assert_array_equal(plan.parameter_independent_carrier, original_carrier)
        assert plan.plan_identifier == original_identifier

    if semantics == "frequency":
        np.testing.assert_array_equal(
            root.parameter_independent_carrier,
            np.zeros(len(root.weights.values)),
        )


def test_default_prediction_is_an_owned_conditional_mean() -> None:
    family = GaussianLS()
    theta = np.array([[1.0, 0.5], [-2.0, 1.4]])

    prediction = family.default_prediction(theta)
    theta[:, 0] = 99.0

    np.testing.assert_array_equal(prediction, np.array([1.0, -2.0]))
    assert not prediction.flags.writeable


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("bad_theta_shape", "theta.*shape"),
        ("scale_at_floor", "scale.*support"),
        ("nonfinite_y", "finite"),
        ("bad_derivative_order", "derivative_order"),
    ],
)
def test_family_rejects_invalid_likelihood_and_information_inputs(
    operation: str, message: str
) -> None:
    family = GaussianLS(scale_floor=0.01)
    y = np.array([0.0, 1.0])
    theta = np.array([[0.0, 0.4], [1.0, 0.8]])
    weights = np.ones(2)
    plan = _plan(family, y, weights, semantics="prior")

    with pytest.raises(ValueError, match=message):
        if operation == "bad_theta_shape":
            family.evaluate_natural(y, theta[:, :1], plan)
        elif operation == "scale_at_floor":
            bad = theta.copy()
            bad[0, 1] = family.scale_floor
            family.evaluate_natural(y, bad, plan)
        elif operation == "nonfinite_y":
            family.evaluate_natural(np.array([0.0, np.nan]), theta, plan)
        else:
            family.evaluate_natural(y, theta, plan, derivative_order=3)


def test_gaussian_refuses_raw_weight_arrays_below_binding() -> None:
    family = GaussianLS(scale_floor=0.01)
    y = np.array([0.0, 1.0])
    theta = np.array([[0.0, 0.4], [1.0, 0.8]])

    with pytest.raises(LikelihoodWeightError, match="likelihood"):
        family.evaluate_natural(y, theta, np.ones(2))
    with pytest.raises(LikelihoodWeightError, match="likelihood"):
        family.initialize(y, np.ones(2))
    with pytest.raises(LikelihoodWeightError, match="likelihood"):
        family.expected_information_natural(theta, np.ones(2))


def test_gaussian_binding_refuses_response_weight_row_mismatch() -> None:
    family = GaussianLS()
    weights = resolve_likelihood_weights(
        np.ones(2),
        n_observations=2,
        contract=WeightContract(semantics="prior"),
    )

    with pytest.raises(LikelihoodWeightError, match="rows"):
        family.bind_likelihood(np.ones(3), weights, COMPLETE_OBSERVATION)
