from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM, SuperLSS
from superglm._frame import as_eager_frame
from superglm.distributional import GammaLS, GaussianLS, Predictor
from superglm.distributional.derivatives import (
    transform_natural_derivatives,
    transform_natural_information,
)
from superglm.distributional.families.negative_binomial import NegativeBinomialLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.model import fit_dense_distributional, refit_dense_distributional
from superglm.distributional.prediction_design import build_joint_prediction_design
from superglm.distributional.predictor import compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.weights import (
    LikelihoodWeightError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Categorical, Numeric

from ._gaussian_lss_oracles import (
    assert_gaussian_fit_parity,
    certify_gaussian_result,
    gamma,
    gaussian_row_oracle,
    integer_scale_design_fixture,
    local_root_certificate,
)

_SOLVER_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))


def test_negative_binomial_lss_keeps_the_public_prior_weight_default() -> None:
    """Kills changing the SuperLSS default to frequency for the NB2 family."""
    model = SuperLSS(
        family=NegativeBinomialLS(),
        predictors=(Predictor("mean", {}), Predictor("theta", {})),
    )

    assert model.weight_semantics == "prior"


def _boundary_model(*, weight_semantics: str = "prior") -> SuperLSS:
    return SuperLSS(
        family=GaussianLS(scale_floor=0.01),
        predictors=(
            Predictor("location", {"x": Numeric(), "g": Categorical()}),
            Predictor("scale", {}),
        ),
        weight_semantics=weight_semantics,
    )


def _boundary_fixture() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
]:
    frame = pd.DataFrame(
        {
            "x": [-2.0, -1.5, -1.0, -0.4, np.nan, 0.3, 0.9, 1.4, 2.0],
            "g": ["a", "b", "a", "b", None, "a", "b", "a", "b"],
        }
    )
    response = np.array([-1.4, -0.2, -0.5, 0.4, np.nan, 1.0, 1.4, 2.0, 2.6])
    weights = np.array([0.5, 1.0, 1.5, 0.8, 0.0, 2.0, 0.7, 1.2, 1.8])
    offsets = {
        "location": np.array([-0.1, 0.0, 0.05, -0.03, np.nan, 0.02, -0.04, 0.1, 0.08]),
        "scale": np.array([0.0, 0.02, -0.01, 0.01, np.nan, 0.0, -0.02, 0.01, 0.0]),
    }
    return frame, response, weights, offsets


def test_zero_weight_row_is_removed_before_response_covariate_level_and_offset_validation():
    """A dropped row cannot influence any response or learned predictor state."""
    frame, response, weights, offsets = _boundary_fixture()
    retained = weights > 0.0

    fitted = _boundary_model().fit(
        frame,
        response,
        sample_weight=weights,
        offsets=offsets,
        inner_tol=1.0e-9,
    )
    reference = _boundary_model().fit(
        frame.loc[retained].reset_index(drop=True),
        response[retained],
        sample_weight=weights[retained],
        offsets={name: values[retained] for name, values in offsets.items()},
        inner_tol=1.0e-9,
    )

    scoring_frame = frame.loc[retained].reset_index(drop=True)
    np.testing.assert_allclose(
        fitted.predict_parameters(scoring_frame),
        reference.predict_parameters(scoring_frame),
        rtol=0.0,
        atol=2.0e-12,
    )
    fitted_state = fitted._require_fitted().fit_state
    assert fitted_state.null_model.n_observations == int(np.sum(retained))
    assert fitted_state.retained_rows is not None
    np.testing.assert_array_equal(fitted_state.retained_rows.response, response[retained])
    learned_category = fitted_state.compiled_predictors[0].compiled.specs["g"]
    assert learned_category._levels == ["a", "b"]


def test_all_zero_likelihood_weights_refuse_with_the_typed_contract_error():
    frame, response, _, _ = _boundary_fixture()

    with pytest.raises(LikelihoodWeightError, match="retain at least one row"):
        _boundary_model().fit(frame, response, sample_weight=np.zeros(len(frame)))


def test_fractional_frequency_weights_refuse_without_rounding():
    frame, response, _, _ = _boundary_fixture()
    weights = np.ones(len(frame))
    weights[3] = 1.5

    with pytest.raises(LikelihoodWeightError, match="exact non-negative integers"):
        _boundary_model(weight_semantics="frequency").fit(
            frame,
            response,
            sample_weight=weights,
        )


def _categorical_geometry(
    frame: pd.DataFrame,
    weights: np.ndarray,
    *,
    semantics: str,
    intercept: bool,
    scoring_frame: pd.DataFrame | None = None,
):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(frame),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved,
        GaussianLS().parameters,
        (
            Predictor("location", {"g": Categorical()}, intercept=intercept),
            Predictor("scale", {}),
        ),
    )
    layout = build_stacked_layout(compiled)
    score_frame = frame if scoring_frame is None else scoring_frame
    design = build_joint_prediction_design(score_frame, compiled, layout).local["location"]
    spec = compiled[0].compiled.specs["g"]
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(design.shape) * max(1.0, float(singular_values[0]))
    rank = int(np.sum(singular_values > tolerance))
    projector = design @ np.linalg.pinv(design, rcond=tolerance / singular_values[0])
    return spec, design, rank, projector


@pytest.mark.parametrize("intercept", [True, False], ids=["intercept", "no-intercept"])
def test_prior_categorical_reporting_uses_physical_rows(
    intercept: bool,
) -> None:
    """Kills routing prior precision mass into ``base='most_exposed'``."""

    frame = pd.DataFrame({"g": ["a", "a", "b"]})
    prior_weights = np.array([0.1, 0.1, 10.0])
    weighted = _categorical_geometry(
        frame,
        prior_weights,
        semantics="prior",
        intercept=intercept,
    )
    physical = _categorical_geometry(
        frame,
        np.ones(len(frame)),
        semantics="prior",
        intercept=intercept,
    )
    expected_design = (
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        if intercept
        else np.array([[0.0], [0.0], [1.0]])
    )
    expected_projector = (
        np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        if intercept
        else np.diag([0.0, 0.0, 1.0])
    )

    for spec, design, rank, projector in (weighted, physical):
        assert spec._base_level == "a"
        assert spec._non_base == ["b"]
        np.testing.assert_array_equal(design, expected_design)
        assert rank == expected_design.shape[1]
        np.testing.assert_allclose(projector, expected_projector, rtol=0.0, atol=8e-16)


@pytest.mark.parametrize("intercept", [True, False], ids=["intercept", "no-intercept"])
def test_frequency_categorical_reporting_matches_literal_expansion(
    intercept: bool,
) -> None:
    """Replication mass must choose the same base, columns, rank, and subspace."""

    frame = pd.DataFrame({"g": ["a", "b", "b"]})
    counts = np.array([3, 1, 1], dtype=np.float64)
    repeated = np.repeat(np.arange(len(frame)), counts.astype(np.intp))
    expanded_frame = frame.iloc[repeated].reset_index(drop=True)
    compact = _categorical_geometry(
        frame,
        counts,
        semantics="frequency",
        intercept=intercept,
    )
    expanded = _categorical_geometry(
        expanded_frame,
        np.ones(len(expanded_frame)),
        semantics="frequency",
        intercept=intercept,
        scoring_frame=frame,
    )
    expected_design = (
        np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        if intercept
        else np.array([[0.0], [1.0], [1.0]])
    )
    expected_projector = (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
        if intercept
        else np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])
    )

    for spec, design, rank, projector in (compact, expanded):
        assert spec._base_level == "a"
        assert spec._non_base == ["b"]
        np.testing.assert_array_equal(design, expected_design)
        assert rank == expected_design.shape[1]
        np.testing.assert_allclose(projector, expected_projector, rtol=0.0, atol=8e-16)


def test_scalar_categorical_preserves_historical_likelihood_weight_reporting() -> None:
    """Scalar SuperGLM deliberately keeps sample mass for ordinary categoricals."""

    frame = pd.DataFrame({"g": ["a", "a", "b"]})
    model = SuperGLM(family="poisson", features={"g": Categorical()})
    model.fit(
        frame,
        np.array([1.0, 2.0, 3.0]),
        sample_weight=np.array([0.1, 0.1, 10.0]),
    )

    assert model._specs["g"]._base_level == "b"
    assert model._specs["g"]._non_base == ["a"]


def test_common_scale_continuous_multiplier_identity_is_algebra_only() -> None:
    """The power identity cannot weaken the exact-integer frequency contract."""

    response = np.array([-1.3, -0.2, 0.6, 1.1, 2.4, 4.0])
    weights = np.array([0.25, 0.75, 1.0, 1.5, 2.0, 0.5])
    assert float(np.sum(weights, dtype=np.float64)) == len(response)
    mu = np.full(len(response), 0.4)
    sigma = np.full(len(response), 1.7)
    prior = gaussian_row_oracle(
        response,
        mu,
        sigma,
        weights,
        semantics="prior",
    )
    residual = response - mu
    ordinary = -np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * residual**2 / sigma**2
    continuous_power = weights * ordinary
    scale = max(
        1.0,
        float(np.sum(np.abs(prior.optimizing_log_likelihood), dtype=np.float64)),
        float(np.sum(np.abs(continuous_power), dtype=np.float64)),
    )
    sum_bound = gamma(64 * len(response)) * scale
    np.testing.assert_allclose(
        np.sum(prior.optimizing_log_likelihood, dtype=np.float64),
        np.sum(continuous_power, dtype=np.float64),
        rtol=0.0,
        atol=sum_bound,
    )
    np.testing.assert_allclose(
        np.sum(prior.reported_log_likelihood, dtype=np.float64)
        - np.sum(continuous_power, dtype=np.float64),
        0.5 * np.sum(np.log(weights), dtype=np.float64),
        rtol=0.0,
        atol=sum_bound,
    )

    frame = pd.DataFrame({"row": np.arange(len(response), dtype=np.float64)})
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.0),
        predictors=(Predictor("location", {}), Predictor("scale", {})),
        weight_semantics="frequency",
    )
    with pytest.raises(LikelihoodWeightError, match="exact non-negative integers"):
        model.fit(frame, response, sample_weight=weights)


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [
        ("prior", np.array([0.25, 0.8, 2.5, 4.0])),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_gaussian_row_values_scores_hessians_and_fisher_match_literal_laws(
    semantics: str,
    weights: np.ndarray,
) -> None:
    """Kills weighted prior scale Fisher, a missing carrier, or the wrong score branch."""

    response = np.array([-1.2, 0.1, 2.4, 8.0])
    location = np.array([-0.8, 0.5, 1.7, 2.0])
    scale = np.array([0.3, 0.35, 1.2, 4.5])
    theta = np.column_stack((location, scale))
    family = GaussianLS(scale_floor=0.0)
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    actual = family.evaluate_natural(response, theta, plan, derivative_order=2)
    actual_fisher = family.expected_information_natural(theta, plan)
    expected = gaussian_row_oracle(
        response,
        location,
        scale,
        weights,
        semantics=semantics,  # type: ignore[arg-type]
    )
    scale_bound = max(
        1.0,
        float(np.max(np.abs(expected.optimizing_log_likelihood), initial=0.0)),
        float(np.max(np.abs(expected.natural_score), initial=0.0)),
        float(np.max(np.abs(expected.natural_hessian_packed), initial=0.0)),
        float(np.max(np.abs(expected.natural_fisher_packed), initial=0.0)),
    )
    bound = gamma(64) * scale_bound
    np.testing.assert_allclose(
        actual.optimizing_log_likelihood,
        expected.optimizing_log_likelihood,
        rtol=0.0,
        atol=bound,
    )
    np.testing.assert_allclose(
        actual.parameter_independent_carrier,
        expected.parameter_independent_carrier,
        rtol=0.0,
        atol=bound,
    )
    np.testing.assert_allclose(
        actual.reported_log_likelihood,
        expected.reported_log_likelihood,
        rtol=0.0,
        atol=bound,
    )
    np.testing.assert_allclose(actual.score, expected.natural_score, rtol=0.0, atol=bound)
    np.testing.assert_allclose(
        actual.hessian_packed,
        expected.natural_hessian_packed,
        rtol=0.0,
        atol=bound,
    )
    np.testing.assert_allclose(
        actual_fisher,
        expected.natural_fisher_packed,
        rtol=0.0,
        atol=bound,
    )

    other_semantics = "frequency" if semantics == "prior" else "prior"
    other_weights = np.array([1.0, 2.0, 3.0, 4.0]) if other_semantics == "frequency" else weights
    if np.array_equal(other_weights, weights):
        other = gaussian_row_oracle(
            response,
            location,
            scale,
            other_weights,
            semantics=other_semantics,  # type: ignore[arg-type]
        )
        np.testing.assert_allclose(
            expected.observed_link_curvature_packed,
            other.observed_link_curvature_packed,
            rtol=0.0,
            atol=bound,
        )


@pytest.mark.parametrize(
    ("semantics", "weights"),
    [
        ("prior", np.array([0.25, 0.8, 2.5, 4.0])),
        ("frequency", np.array([1.0, 2.0, 3.0, 4.0])),
    ],
)
def test_gaussian_varied_rows_match_independent_link_score_observed_and_fisher(
    semantics: str,
    weights: np.ndarray,
) -> None:
    """Kills omitting an inverse-link derivative from predictor Fisher."""

    response = np.array([-1.2, 0.1, 2.4, 8.0])
    location = np.array([-0.8, 0.5, 1.7, 2.0])
    scale_floor = 0.17
    scale_increment = np.array([0.21, 0.53, 1.03, 4.11])
    scale = scale_floor + scale_increment
    eta = np.column_stack((location, np.log(scale_increment)))
    theta = np.column_stack((location, scale))
    family = GaussianLS(scale_floor=scale_floor)
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    plan = family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)
    natural = family.evaluate_natural(response, theta, plan, derivative_order=2)
    natural_fisher = family.expected_information_natural(theta, plan)
    links = tuple(parameter.default_link for parameter in family.parameters)
    transformed = transform_natural_derivatives(natural, eta, links)
    transformed_fisher = transform_natural_information(natural_fisher, eta, links)
    expected = gaussian_row_oracle(
        response,
        location,
        scale,
        weights,
        semantics=semantics,  # type: ignore[arg-type]
        scale_floor=scale_floor,
    )
    component_scale = max(
        1.0,
        float(np.max(np.abs(expected.link_score), initial=0.0)),
        float(
            np.max(
                np.abs(expected.observed_link_curvature_packed),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(expected.fisher_link_curvature_packed),
                initial=0.0,
            )
        ),
    )
    component_bound = gamma(96) * component_scale
    np.testing.assert_allclose(
        transformed.score_eta,
        expected.link_score,
        rtol=0.0,
        atol=component_bound,
    )
    np.testing.assert_allclose(
        transformed.curvature_packed,
        expected.observed_link_curvature_packed,
        rtol=0.0,
        atol=component_bound,
    )
    np.testing.assert_allclose(
        transformed_fisher,
        expected.fisher_link_curvature_packed,
        rtol=0.0,
        atol=component_bound,
    )


def test_prior_scale_score_mutation_point_is_three_not_frequency_zero() -> None:
    response = np.array([1.0])
    location = np.array([0.0])
    scale = np.array([1.0])
    weights = np.array([4.0])
    prior = gaussian_row_oracle(
        response,
        location,
        scale,
        weights,
        semantics="prior",
    )
    frequency = gaussian_row_oracle(
        response,
        location,
        scale,
        weights,
        semantics="frequency",
    )
    assert prior.natural_score[0, 1] == 3.0
    assert frequency.natural_score[0, 1] == 0.0
    np.testing.assert_array_equal(
        prior.observed_link_curvature_packed,
        frequency.observed_link_curvature_packed,
    )


def _no_scale_intercept_fit(
    response: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
    *,
    semantics: str,
):
    frame = pd.DataFrame({"z": z})
    model = fit_dense_distributional(
        frame,
        response,
        family=GaussianLS(scale_floor=0.0),
        predictors=(
            Predictor("location", {}),
            Predictor("scale", {"z": Numeric()}, intercept=False),
        ),
        weight_contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
        sample_weight=weights,
        lambdas={},
        config=DenseSolverConfig(
            tolerance=_SOLVER_TOLERANCE,
            coefficient_curvature="observed",
        ),
        initial=np.array([0.35, -0.2]),
    )
    retained = model.fit_state.retained_rows
    assert retained is not None
    certificate = certify_gaussian_result(
        model.layout,
        model.result,
        retained.response,
        retained.likelihood_weights.values,
        semantics=model.fit_state.weight_contract.semantics,
        covariance=model.covariance,
        total_edf=model.inference.total_edf,
        inference_rank=model.inference.rank,
        prediction_parameters=model.predict_parameters(frame),
        default_prediction=model.predict(frame),
    )
    return model, certificate


def _common_integer_law_assertions(
    response: np.ndarray,
    counts: np.ndarray,
    z: np.ndarray,
    *,
    expected_projection: float,
) -> None:
    beta = np.array([0.35, -0.2])
    mu = np.full(len(response), beta[0])
    sigma = np.exp(z * beta[1])
    prior = gaussian_row_oracle(response, mu, sigma, counts, semantics="prior")
    frequency = gaussian_row_oracle(response, mu, sigma, counts, semantics="frequency")
    count_excess = float(np.sum(counts - 1.0, dtype=np.float64))
    expected_optimizing_difference = (
        0.5 * count_excess * np.log(2.0 * np.pi) + beta[1] * expected_projection
    )
    likelihood_scale = max(
        1.0,
        float(np.sum(np.abs(prior.optimizing_log_likelihood), dtype=np.float64)),
        float(np.sum(np.abs(frequency.optimizing_log_likelihood), dtype=np.float64)),
    )
    likelihood_bound = gamma(64 * len(response)) * likelihood_scale
    np.testing.assert_allclose(
        np.sum(prior.optimizing_log_likelihood, dtype=np.float64)
        - np.sum(frequency.optimizing_log_likelihood, dtype=np.float64),
        expected_optimizing_difference,
        rtol=0.0,
        atol=likelihood_bound,
    )
    np.testing.assert_allclose(
        np.sum(prior.reported_log_likelihood, dtype=np.float64)
        - np.sum(frequency.reported_log_likelihood, dtype=np.float64),
        expected_optimizing_difference + 0.5 * np.sum(np.log(counts), dtype=np.float64),
        rtol=0.0,
        atol=likelihood_bound,
    )
    score_prior = np.array(
        [
            np.sum(prior.link_score[:, 0], dtype=np.float64),
            np.dot(z, prior.link_score[:, 1]),
        ]
    )
    score_frequency = np.array(
        [
            np.sum(frequency.link_score[:, 0], dtype=np.float64),
            np.dot(z, frequency.link_score[:, 1]),
        ]
    )
    score_scale = max(1.0, float(np.linalg.norm(score_prior, ord=np.inf)))
    score_bound = gamma(64 * len(response)) * score_scale
    np.testing.assert_allclose(
        score_prior - score_frequency,
        np.array([0.0, expected_projection]),
        rtol=0.0,
        atol=score_bound,
    )
    location_design = np.ones((len(response), 1))
    scale_design = z[:, None]
    prior_curvature = np.block(
        [
            [
                location_design.T
                @ (prior.observed_link_curvature_packed[:, 0, None] * location_design),
                location_design.T
                @ (prior.observed_link_curvature_packed[:, 1, None] * scale_design),
            ],
            [
                scale_design.T
                @ (prior.observed_link_curvature_packed[:, 1, None] * location_design),
                scale_design.T @ (prior.observed_link_curvature_packed[:, 2, None] * scale_design),
            ],
        ]
    )
    frequency_curvature = np.block(
        [
            [
                location_design.T
                @ (frequency.observed_link_curvature_packed[:, 0, None] * location_design),
                location_design.T
                @ (frequency.observed_link_curvature_packed[:, 1, None] * scale_design),
            ],
            [
                scale_design.T
                @ (frequency.observed_link_curvature_packed[:, 1, None] * location_design),
                scale_design.T
                @ (frequency.observed_link_curvature_packed[:, 2, None] * scale_design),
            ],
        ]
    )
    curvature_scale = max(1.0, float(np.linalg.norm(prior_curvature, ord=np.inf)))
    curvature_bound = gamma(96 * len(response)) * curvature_scale
    np.testing.assert_allclose(
        prior_curvature,
        frequency_curvature,
        rtol=0.0,
        atol=curvature_bound,
    )


def test_integer_frequency_no_intercept_equal_projection_has_equivalent_modes() -> None:
    """Kills the claim that legal prior and frequency fits can never agree."""

    response, counts, z_equal, _ = integer_scale_design_fixture()
    projection = float(np.dot(z_equal, counts - 1.0))
    assert projection == 0.0
    _common_integer_law_assertions(
        response,
        counts,
        z_equal,
        expected_projection=projection,
    )
    prior, prior_certificate = _no_scale_intercept_fit(
        response,
        counts,
        z_equal,
        semantics="prior",
    )
    frequency, frequency_certificate = _no_scale_intercept_fit(
        response,
        counts,
        z_equal,
        semantics="frequency",
    )
    frame = pd.DataFrame({"z": z_equal})
    count_excess = float(np.sum(counts - 1.0, dtype=np.float64))
    expected_optimizing_difference = 0.5 * count_excess * np.log(2.0 * np.pi)
    expected_carrier_difference = 0.5 * float(np.sum(np.log(counts), dtype=np.float64))
    expected_reported_difference = expected_optimizing_difference + expected_carrier_difference
    assert_gaussian_fit_parity(
        prior.result,
        frequency.result,
        prior_certificate,
        frequency_certificate,
        expected_optimizing_difference=expected_optimizing_difference,
        expected_carrier_difference=expected_carrier_difference,
        expected_reported_difference=expected_reported_difference,
        left_eta=prior.predict_eta(frame),
        right_eta=frequency.predict_eta(frame),
        left_prediction=prior.predict_parameters(frame),
        right_prediction=frequency.predict_parameters(frame),
    )
    assert prior.fit_state.weight_contract != frequency.fit_state.weight_contract


def test_integer_frequency_no_intercept_nonzero_projection_separates_modes() -> None:
    """Kills the blanket claim that prior and frequency modes always agree."""

    response, counts, _, z_separate = integer_scale_design_fixture()
    projection = float(np.dot(z_separate, counts - 1.0))
    assert projection == 6.0
    _common_integer_law_assertions(
        response,
        counts,
        z_separate,
        expected_projection=projection,
    )
    prior, prior_certificate = _no_scale_intercept_fit(
        response,
        counts,
        z_separate,
        semantics="prior",
    )
    frequency, frequency_certificate = _no_scale_intercept_fit(
        response,
        counts,
        z_separate,
        semantics="frequency",
    )
    prior_local = local_root_certificate(
        prior_certificate.oracle,
        prior.result.coefficients[None, :],
    )
    frequency_local = local_root_certificate(
        frequency_certificate.oracle,
        frequency.result.coefficients[None, :],
    )
    center_gap = abs(prior_local.center[1] - frequency_local.center[1])
    rounding = gamma(64 * len(response) + 32 * len(prior_local.center)) * max(
        1.0,
        abs(prior_local.center[1]),
        abs(frequency_local.center[1]),
    )
    assert center_gap > prior_local.center_error + frequency_local.center_error + rounding


@pytest.mark.parametrize("retain_rows", [True, False], ids=["retained", "compact"])
def test_gamma_refit_rebinds_response_and_rebuilds_omitted_weight_root(
    retain_rows: bool,
) -> None:
    """Changing y rebinds Gamma authority; omitting weights creates a new root."""

    x = np.linspace(-1.0, 1.0, 48)
    frame = pd.DataFrame({"x": x, "z": np.cos(np.pi * x)})
    mean = np.exp(0.5 + 0.3 * frame["x"])
    scale = np.exp(-0.7 + 0.15 * frame["z"])
    rng = np.random.default_rng(77)
    response = rng.gamma(shape=1.0 / scale**2, scale=mean * scale**2)
    weights = 0.5 + np.arange(len(x)) % 5 / 2.0
    model = (
        SuperLSS(
            family=GammaLS(),
            predictors=(Predictor("mean", {"x": Numeric()}), Predictor("scale", {"z": Numeric()})),
        )
        .fit(
            frame,
            response,
            sample_weight=weights,
            inner_tol=float(np.sqrt(np.finfo(np.float64).eps)),
            retain_rows=retain_rows,
        )
        ._require_fitted()
    )
    training = model.fit_state

    changed_response = response * np.exp(np.linspace(-0.03, 0.04, len(response)))
    refit_dense_distributional(
        model,
        frame,
        changed_response,
        sample_weight=weights,
        retain_rows=retain_rows,
    )
    changed = model.fit_state
    assert (changed.revision, changed.result.coefficient_converged, changed.weight_contract) == (
        training.revision + 1,
        True,
        WeightContract("prior"),
    )
    assert changed.weight_provenance.root_digest == training.weight_provenance.root_digest
    assert changed.family_likelihood_plan_identifier != training.family_likelihood_plan_identifier
    assert changed.null_model.family_likelihood_plan_identifier == (
        changed.family_likelihood_plan_identifier
    )
    if retain_rows:
        np.testing.assert_array_equal(changed.retained_rows.response, changed_response)
        np.testing.assert_array_equal(changed.retained_rows.likelihood_weights.values, weights)
    else:
        assert changed.retained_rows is None
        assert (changed.weight_provenance.min_weight, changed.weight_provenance.max_weight) == (
            float(np.min(weights)),
            float(np.max(weights)),
        )

    changed_root = changed.weight_provenance.root_digest
    changed_plan = changed.family_likelihood_plan_identifier
    all_one_response = changed_response * np.exp(np.linspace(0.02, -0.01, len(changed_response)))
    refit_dense_distributional(
        model,
        frame,
        all_one_response,
        retain_rows=retain_rows,
    )
    all_one = model.fit_state
    assert (all_one.revision, all_one.result.coefficient_converged, all_one.weight_contract) == (
        changed.revision + 1,
        True,
        WeightContract("prior"),
    )
    assert all_one.weight_provenance.all_unit is True
    assert all_one.weight_provenance.root_digest != changed_root
    assert all_one.family_likelihood_plan_identifier != changed_plan
    assert all_one.weight_provenance.weight_sum == len(all_one_response)
    if retain_rows:
        np.testing.assert_array_equal(all_one.retained_rows.response, all_one_response)
        np.testing.assert_array_equal(all_one.retained_rows.likelihood_weights.values, 1.0)
    else:
        assert all_one.retained_rows is None
