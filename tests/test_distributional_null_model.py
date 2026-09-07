from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.null_model as null_module
from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS, LowerBoundedLogLink
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.inference import compute_joint_inference
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.null_model import NullModelFitError, fit_joint_null_model
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import DenseSolverError
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric
from superglm.links import LogLink

from ._distributional_weights import resolved_prior
from ._gaussian_lss_oracles import (
    assert_gaussian_fit_parity,
    certify_gaussian_result,
    coefficient_oracle,
    gamma,
    intercept_fixture,
    oracle_bounds,
)

_SEMANTIC_SOLVER_TOLERANCE = float(np.sqrt(np.finfo(np.float64).eps))


def _source_layout(
    frame: pd.DataFrame,
    family: GaussianLS,
    weights: np.ndarray,
    *,
    offsets: dict[str, np.ndarray] | None = None,
    location_log_link: bool = False,
) -> StackedLayout:
    predictors = (
        Predictor(
            "location",
            {"x": Numeric()},
            link=LogLink() if location_log_link else None,
        ),
        Predictor("scale", {"z": Numeric()}),
    )
    compiled = compile_predictors(
        as_eager_frame(frame),
        resolved_prior(weights),
        family.parameters,
        predictors,
        offsets=offsets,
    )
    return build_stacked_layout(compiled)


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(8127)
    x = np.linspace(-1.0, 1.0, 80)
    z = np.cos(np.linspace(0.0, 2.0 * np.pi, len(x)))
    response = 1.8 + 0.45 * x + rng.normal(scale=0.35 + 0.08 * (z + 1.0))
    weights = np.linspace(0.4, 2.3, len(x))
    weights[-8:] = 7.0
    return pd.DataFrame({"x": x, "z": z}), response, weights


def _likelihood_plan(
    family: GaussianLS,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    semantics: str,
):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    return family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)


def test_weighted_and_unweighted_null_fits_use_the_shared_joint_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, weights = _fixture()
    family = GaussianLS(scale_floor=0.07)
    layout = _source_layout(frame, family, np.ones(len(frame)))
    calls: list[tuple[StackedLayout, np.ndarray, np.ndarray]] = []
    original = null_module.fit_dense_fixed_lambda

    def solver_spy(family_arg, layout_arg, y_arg, plan_arg, penalty_arg, **kwargs):
        calls.append(
            (
                layout_arg,
                np.array(plan_arg.weights.values, copy=True),
                np.array(penalty_arg, copy=True),
            )
        )
        return original(family_arg, layout_arg, y_arg, plan_arg, penalty_arg, **kwargs)

    monkeypatch.setattr(null_module, "fit_dense_fixed_lambda", solver_spy)
    unweighted = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(
            family,
            response,
            np.ones(len(response)),
            semantics="prior",
        ),
    )
    weighted = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(family, response, weights, semantics="prior"),
    )

    assert len(calls) == 2
    for null_layout, used_weights, penalty in calls:
        assert null_layout.n_coefficients == len(family.parameters)
        assert null_layout.term_slices == {}
        assert null_layout.penalties == ()
        assert all(state.design.p == 0 and state.groups == () for state in null_layout.predictors)
        np.testing.assert_array_equal(penalty, np.zeros((2, 2)))
        assert np.all(used_weights > 0.0)
    assert unweighted.result.converged is True
    assert weighted.result.converged is True
    assert unweighted.result.penalty_value == 0.0
    assert weighted.result.penalty_value == 0.0
    assert weighted.objective == pytest.approx(weighted.result.objective)
    assert weighted.family_config == {"type": "GaussianLS", "scale_floor": 0.07}
    assert not np.allclose(unweighted.result.coefficients, weighted.result.coefficients)


def test_null_fit_preserves_parameter_specific_links_floor_and_nonzero_offsets() -> None:
    frame, response, weights = _fixture()
    response = response - response.min() + 0.5
    family = GaussianLS(scale_floor=0.2)
    offsets = {
        "location": np.linspace(-0.25, 0.2, len(frame)),
        "scale": 0.08 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(frame))),
    }
    layout = _source_layout(
        frame,
        family,
        weights,
        offsets=offsets,
        location_log_link=True,
    )

    null = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(family, response, weights, semantics="prior"),
        config=DenseSolverConfig(tolerance=1.0e-9),
    )

    assert isinstance(null.parameter_links["location"], LogLink)
    assert isinstance(null.parameter_links["scale"], LowerBoundedLogLink)
    assert null.parameter_links["scale"].floor == 0.2
    np.testing.assert_allclose(
        null.result.eta[:, 0],
        null.result.coefficients[0] + offsets["location"],
    )
    np.testing.assert_allclose(
        null.result.eta[:, 1],
        null.result.coefficients[1] + offsets["scale"],
    )
    np.testing.assert_array_equal(null.offsets["location"], offsets["location"])
    np.testing.assert_array_equal(null.offsets["scale"], offsets["scale"])
    assert np.all(null.result.theta[:, 0] > 0.0)
    assert np.all(null.result.theta[:, 1] > family.scale_floor)
    assert null.layout.coefficient_names == (
        "location:(intercept)",
        "scale:(intercept)",
    )


def test_null_state_records_the_declared_weight_contract_not_a_constant() -> None:
    """The stamp must follow what the caller declared, or it certifies nothing."""
    frame, response, weights = _fixture()
    family = GaussianLS(scale_floor=0.03)
    layout = _source_layout(frame, family, weights)
    frequency_weights = np.resize(np.array([1.0, 2.0, 3.0, 4.0]), len(response))

    declared = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(
            family,
            response,
            frequency_weights,
            semantics="frequency",
        ),
    )
    defaulted = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(family, response, weights, semantics="prior"),
    )

    assert declared.weight_semantics == "frequency"
    assert defaulted.weight_semantics == "prior"
    with pytest.raises(UnsupportedLikelihoodContractError, match="unsupported"):
        _likelihood_plan(
            family,
            response,
            weights,
            semantics="frequency_case",
        )


def test_null_state_retains_complete_auditable_fit_metadata() -> None:
    frame, response, weights = _fixture()
    family = GaussianLS(scale_floor=0.03)
    layout = _source_layout(frame, family, weights)
    null = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=_likelihood_plan(family, response, weights, semantics="prior"),
    )

    assert null.family is family
    assert null.parameter_names == ("location", "scale")
    assert null.weight_semantics == "prior"
    assert null.n_observations == len(frame)
    assert null.weight_sum == pytest.approx(float(np.sum(weights)))
    assert null.converged is null.result.converged is True
    assert null.convergence_reason == null.result.convergence_reason
    assert null.curvature_telemetry is null.result.terminal_curvature
    assert null.result.penalized_optimizing_log_likelihood is not None
    assert null.objective == pytest.approx(-null.result.penalized_optimizing_log_likelihood)
    with pytest.raises(ValueError):
        null.sample_weight[0] = -1.0
    with pytest.raises(ValueError):
        null.offsets["location"][0] = 1.0
    with pytest.raises(TypeError):
        null.family_config["scale_floor"] = 1.0


def test_null_fit_propagates_solver_errors_and_rejects_nonconvergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, response, weights = _fixture()
    family = GaussianLS()
    layout = _source_layout(frame, family, weights)
    likelihood_plan = _likelihood_plan(family, response, weights, semantics="prior")
    successful = fit_joint_null_model(
        family,
        layout,
        response,
        likelihood_plan=likelihood_plan,
    )

    failed = replace(
        successful.result,
        converged=False,
        convergence_reason="max_iterations",
    )
    monkeypatch.setattr(null_module, "fit_dense_fixed_lambda", lambda *args, **kwargs: failed)
    with pytest.raises(NullModelFitError, match="did not converge") as captured:
        fit_joint_null_model(
            family,
            layout,
            response,
            likelihood_plan=likelihood_plan,
        )
    assert captured.value.result is failed

    def raise_solver_error(*args, **kwargs):
        raise DenseSolverError("null sentinel")

    monkeypatch.setattr(null_module, "fit_dense_fixed_lambda", raise_solver_error)
    with pytest.raises(DenseSolverError, match="null sentinel"):
        fit_joint_null_model(
            family,
            layout,
            response,
            likelihood_plan=likelihood_plan,
        )


def _literal_null_fit(
    response: np.ndarray,
    weights: np.ndarray,
    *,
    semantics: str,
):
    n_rows = len(response)
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1.0, 1.0, n_rows),
            "z": np.cos(np.linspace(0.0, np.pi, n_rows)),
        }
    )
    family = GaussianLS(scale_floor=0.0)
    source = _source_layout(frame, family, np.ones(n_rows))
    plan = _likelihood_plan(
        family,
        response,
        weights,
        semantics=semantics,
    )
    null = fit_joint_null_model(
        family,
        source,
        response,
        likelihood_plan=plan,
        config=DenseSolverConfig(tolerance=_SEMANTIC_SOLVER_TOLERANCE),
    )
    result = null.result
    inference = compute_joint_inference(null.layout, result)
    certificate = certify_gaussian_result(
        null.layout,
        result,
        response,
        plan.weights.values,
        semantics=plan.weights.provenance.contract.semantics,
        covariance=inference.covariance,
        total_edf=inference.total_edf,
        inference_rank=inference.rank,
    )
    assert result.family_likelihood_plan_identifier == plan.plan_identifier
    np.testing.assert_allclose(
        null.objective,
        certificate.oracle.objective,
        rtol=0.0,
        atol=certificate.bounds.likelihood_sum,
    )
    return null, certificate


@pytest.mark.parametrize("semantics", ["prior", "frequency"])
def test_joint_null_prior_and_frequency_match_literal_intercept_oracles(
    semantics: str,
) -> None:
    """Joint-null routing is certified directly, never inferred from main-fit parity."""

    response, prior_weights = intercept_fixture()
    weights = prior_weights if semantics == "prior" else np.array([1.0, 2.0, 3.0, 1.0, 2.0, 4.0])
    null, certificate = _literal_null_fit(
        response,
        weights,
        semantics=semantics,
    )
    accepted_root = null.likelihood_plan.weights
    accepted_response = response[accepted_root.input_positions]
    accepted_weights = accepted_root.values
    total_mass = (
        len(accepted_response)
        if semantics == "prior"
        else accepted_root.provenance.likelihood_count
    )
    expected_mu = float(
        np.dot(accepted_weights, accepted_response) / np.sum(accepted_weights, dtype=np.float64)
    )
    residual = accepted_response - expected_mu
    expected_sigma = float(np.sqrt(np.dot(accepted_weights, residual * residual) / total_mass))
    expected_beta = np.array([expected_mu, np.log(expected_sigma)])
    expected = coefficient_oracle(
        accepted_response,
        accepted_weights,
        semantics=semantics,  # type: ignore[arg-type]
        location_design=np.ones((len(accepted_response), 1)),
        scale_design=np.ones((len(accepted_response), 1)),
        coefficients=expected_beta,
        penalty=np.zeros((2, 2)),
    )
    expected_bounds = oracle_bounds(expected)
    closed_form_bound = gamma(128 * len(accepted_response)) * max(
        1.0,
        float(np.sum(accepted_weights, dtype=np.float64)),
        float(np.dot(accepted_weights, residual * residual)),
        float(np.linalg.norm(expected_beta, ord=np.inf)),
        1.0 / expected_sigma,
    )
    np.testing.assert_allclose(
        null.result.coefficients,
        expected_beta,
        rtol=0.0,
        atol=float(certificate.local_root.candidate_errors[0]) + closed_form_bound,
    )
    likelihood_bound = (
        certificate.bounds.likelihood_sum
        + expected_bounds.likelihood_sum
        + abs(certificate.oracle.reported_log_likelihood - expected.reported_log_likelihood)
    )
    np.testing.assert_allclose(
        null.result.log_likelihood,
        expected.reported_log_likelihood,
        rtol=0.0,
        atol=likelihood_bound,
    )

    if semantics == "frequency":
        take = np.repeat(np.arange(len(response)), weights.astype(np.intp))
        expanded, expanded_certificate = _literal_null_fit(
            response[take],
            np.ones(len(take), dtype=np.float64),
            semantics="frequency",
        )
        first_expanded = np.flatnonzero(np.r_[True, np.diff(take) != 0])
        assert_gaussian_fit_parity(
            null.result,
            expanded.result,
            certificate,
            expanded_certificate,
            probe=np.array([0.75, 1.25]),
            left_eta=null.result.eta,
            right_eta=expanded.result.eta[first_expanded],
            left_prediction=null.result.theta,
            right_prediction=expanded.result.theta[first_expanded],
        )
