from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.efs as efs_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.evidence as smoothing_evidence
import superglm.distributional.smoothing.faces as smoothing_faces
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.smoothing.objective as smoothing_objective
import superglm.distributional.smoothing.proposals as smoothing_proposals
from superglm import Spline, SuperLSS
from superglm.distributional import GammaLS, Predictor
from superglm.distributional.inference import compute_joint_inference
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.penalty_face import PenaltyFace, build_penalty_face
from superglm.distributional.result import (
    DenseSolverResult,
    DistributionalEFSConfig,
    JointEndpointDirectionEvidence,
)
from superglm.reml.penalty_algebra import penalty_component_dense_matrix

_FACE_COMPONENTS = ("mean:x#wiggle", "mean:z#wiggle")
_FINITE_COMPONENT = "mean:w#wiggle"


@dataclass(frozen=True)
class _GammaJointFaceCase:
    frame: pd.DataFrame
    response: np.ndarray
    weights: np.ndarray
    model: SuperLSS


def _gamma_joint_face_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_values = np.linspace(-1.0, 1.0, 6)
    z_values = np.linspace(-1.0, 1.0, 6)
    w_values = np.linspace(-1.0, 1.0, 12)
    residual_factors = np.array([0.62, 0.84, 1.16, 1.38])
    w, x, z, residual = np.meshgrid(
        w_values,
        x_values,
        z_values,
        residual_factors,
        indexing="ij",
    )
    w = w.ravel()
    x = x.ravel()
    z = z.ravel()
    residual = residual.ravel()
    mean = np.exp(
        0.4
        + 0.55 * np.sin(np.pi * w)
        + 0.15 * w
        + 0.03 * np.sin(np.pi * x)
        + 0.024 * np.sin(np.pi * z)
    )
    frame = pd.DataFrame({"x": x, "z": z, "w": w})
    weights = 0.65 + 0.7 * (w + 1.0) / 2.0
    return frame, mean * residual, weights


def _fit_public_gamma_joint_face(
    frame: pd.DataFrame,
    response: np.ndarray,
    weights: np.ndarray,
) -> SuperLSS:
    maximum_lambda = DistributionalEFSConfig().maximum_lambda
    # The scenario (a cap fit that is not stationary to solver tolerance) arises under
    # Fisher scoring; observed Newton converges the cap fit to round-off.
    return SuperLSS(
        family=GammaLS(),
        coefficient_curvature="fisher",
        predictors=(
            Predictor(
                "mean",
                {
                    "x": Spline(kind="cr", n_knots=5),
                    "z": Spline(kind="cr", n_knots=5),
                    "w": Spline(kind="cr", n_knots=5),
                },
            ),
            Predictor("scale", {}),
        ),
    ).fit_reml(
        frame,
        response,
        sample_weight=weights,
        lambdas={
            _FACE_COMPONENTS[0]: maximum_lambda,
            _FACE_COMPONENTS[1]: maximum_lambda,
            _FINITE_COMPONENT: 0.5,
        },
        max_reml_iter=120,
        reml_tol=1.0e-3,
        max_inner_iter=150,
        inner_tol=1.0e-9,
        outer="efs",
    )


@pytest.fixture(scope="module")
def gamma_joint_face_case() -> _GammaJointFaceCase:
    frame, response, weights = _gamma_joint_face_data()
    return _GammaJointFaceCase(
        frame=frame,
        response=response,
        weights=weights,
        model=_fit_public_gamma_joint_face(frame, response, weights),
    )


def _retained_kkt_relative(result: DenseSolverResult) -> float:
    score = result.terminal_score
    if result.coefficient_face is not None:
        score = result.coefficient_face.null_basis.T @ score
    objective = result.penalized_optimizing_log_likelihood
    assert objective is not None
    return float(np.linalg.norm(score, ord=np.inf) / (1.0 + abs(objective)))


def _scaled_solve_roundoff(result: DenseSolverResult) -> float:
    face = result.coefficient_face
    decomposition = result.terminal_rank if face is None else result.terminal_reduced_rank
    assert decomposition is not None
    retained_width = result.coefficients.size if face is None else face.reduced_width
    assert decomposition.rank == retained_width
    assert not decomposition.rank_truncated
    condition = max(1.0, float(decomposition.pre_truncation_condition))
    epsilon = float(np.finfo(result.coefficients.dtype).eps)
    operation_error = 64.0 * retained_width * epsilon
    assert operation_error < 1.0
    return float(
        np.nextafter(
            operation_error / (1.0 - operation_error) * condition,
            math.inf,
        )
    )


def _penalty_action_roundoff(
    penalty: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    width = coefficients.size
    epsilon = float(np.finfo(coefficients.dtype).eps)
    operation_error = 64.0 * width * epsilon
    assert operation_error < 1.0
    scale = max(
        1.0,
        float(np.linalg.norm(np.abs(penalty) @ np.abs(coefficients), ord=2)),
    )
    return float(np.nextafter(operation_error / (1.0 - operation_error) * scale, math.inf))


def _assert_exact_face_residual(face: PenaltyFace, coefficients: np.ndarray) -> None:
    residual = float(np.linalg.norm(face.constraint_matrix @ coefficients, ord=2))
    coefficient_scale = max(1.0, float(np.linalg.norm(coefficients, ord=2)))
    assert residual <= face.null_residual_bound * coefficient_scale


def _maximum_absolute_error(actual: np.ndarray, expected: np.ndarray) -> float:
    assert actual.shape == expected.shape
    assert actual.size > 0
    return float(np.max(np.abs(actual - expected)))


def _consumer_roundoff_bound(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    dimension: int,
) -> float:
    assert actual.shape == expected.shape
    epsilon = float(np.finfo(np.result_type(actual.dtype, expected.dtype)).eps)
    operation_error = 128.0 * max(1, dimension) * epsilon
    assert operation_error < 1.0
    scale = max(
        1.0,
        float(np.linalg.norm(actual.ravel(), ord=np.inf)),
        float(np.linalg.norm(expected.ravel(), ord=np.inf)),
    )
    return float(
        np.nextafter(
            operation_error / (1.0 - operation_error) * scale,
            math.inf,
        )
    )


def _assert_consumer_matches(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    dimension: int,
) -> None:
    error = _maximum_absolute_error(actual, expected)
    assert error <= _consumer_roundoff_bound(actual, expected, dimension=dimension)


def _parameters_from_eta(
    fitted: DenseDistributionalModel,
    eta: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        tuple(
            np.asarray(state.link.inverse(eta[:, index]), dtype=np.float64)
            for index, state in enumerate(fitted.layout.predictors)
        )
    )


def test_public_gamma_reml_accepts_the_strict_joint_infinity_face(
    gamma_joint_face_case: _GammaJointFaceCase,
) -> None:
    """Kills scalar activation of only the first of two capped mean smooths."""
    case = gamma_joint_face_case
    fitted = case.model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None
    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert case.model.exact_face_components_ == _FACE_COMPONENTS
    assert case.model.result_.exact_face_components == _FACE_COMPONENTS
    assert fitted.fit_state.exact_face_components == _FACE_COMPONENTS

    finite_lambda = case.model.smoothing_parameters_[_FINITE_COMPONENT]
    assert np.isfinite(finite_lambda)
    assert 0.0 < finite_lambda < smoothing.config.maximum_lambda
    assert _FINITE_COMPONENT not in case.model.exact_face_components_

    activations = tuple(item for item in smoothing.history if item.activated_face_components)
    revalidations = tuple(item for item in smoothing.history if item.revalidated_face_components)
    assert len(activations) == 1
    assert len(revalidations) == 1
    activation = activations[0]
    revalidation = revalidations[0]
    assert activation.activated_face_components == _FACE_COMPONENTS
    assert revalidation.revalidated_face_components == _FACE_COMPONENTS
    assert len(activation.coefficient_fit_indices) == 1
    assert len(revalidation.coefficient_fit_indices) == 1
    assert activation.accepted_fit_index is not None
    assert revalidation.accepted_fit_index is not None
    assert revalidation.iteration > activation.iteration
    assert revalidation.accepted_fit_index != activation.accepted_fit_index
    assert revalidation is smoothing.history[-1]
    assert smoothing.terminal_fit_index == revalidation.accepted_fit_index
    assert smoothing.terminal_fit is not smoothing.coefficient_fits[activation.accepted_fit_index]
    assert tuple(
        index for item in smoothing.history for index in item.coefficient_fit_indices
    ) == tuple(range(1, len(smoothing.coefficient_fits)))

    evidence = revalidation.endpoint_direction_evidence
    assert isinstance(evidence, JointEndpointDirectionEvidence)
    assert evidence.component_names == _FACE_COMPONENTS
    assert evidence.fit_indices == revalidation.coefficient_fit_indices
    assert dict(smoothing.terminal_endpoint_directions) == dict(evidence.component_directions)
    epsilon = float(np.finfo(np.float64).eps)
    for name, direction in evidence.component_directions:
        assert name in _FACE_COMPONENTS
        scalar_roundoff = 64.0 * epsilon * max(1.0, abs(direction.analytic_derivative))
        assert direction.decision == "endpoint"
        assert direction.lower_bound > max(direction.numerical_error, scalar_roundoff)

    terminal = smoothing.terminal_fit
    face = terminal.coefficient_face
    assert face is not None
    assert face.component_names == _FACE_COMPONENTS
    assert terminal.config.coefficient_curvature == "observed"
    assert terminal.config.tolerance <= 1.0e-12
    assert terminal.config.newton_decrement_tolerance is None
    assert terminal.terminal_curvature.actual_source == "observed"
    assert terminal.terminal_curvature.fallback_count == 0
    terminal_kkt = _retained_kkt_relative(terminal)
    assert terminal_kkt <= min(terminal.config.tolerance, _scaled_solve_roundoff(terminal))
    _assert_exact_face_residual(face, terminal.coefficients)

    assert fitted.fit_state.solver_result is terminal
    assert fitted.result is terminal
    np.testing.assert_array_equal(fitted.coefficients, terminal.coefficients)
    np.testing.assert_array_equal(fitted.fitted_result.coefficients, terminal.coefficients)
    np.testing.assert_array_equal(case.model.result_.coefficients, terminal.coefficients)

    dimension = fitted.layout.n_coefficients
    expected_parameters = _parameters_from_eta(fitted, terminal.eta)
    expected_predictions = np.asarray(
        fitted.family.default_prediction(expected_parameters),
        dtype=np.float64,
    )
    public_eta = case.model.predict_link(case.frame).to_numpy()
    public_parameters = case.model.predict_parameters(case.frame).to_numpy()
    public_predictions = case.model.predict(case.frame)
    _assert_consumer_matches(public_eta, terminal.eta, dimension=dimension)
    _assert_consumer_matches(
        public_parameters,
        expected_parameters,
        dimension=dimension,
    )
    _assert_consumer_matches(
        public_predictions,
        expected_predictions,
        dimension=dimension,
    )

    terminal_inference = fitted.fit_state.inference
    terminal_covariance = terminal.terminal_pseudo_inverse()
    assert fitted.inference is terminal_inference
    _assert_consumer_matches(
        terminal_inference.covariance,
        terminal_covariance,
        dimension=dimension,
    )
    np.testing.assert_array_equal(
        fitted.fitted_result.covariance,
        terminal_inference.covariance,
    )
    np.testing.assert_array_equal(
        case.model.result_.covariance,
        terminal_inference.covariance,
    )
    np.testing.assert_array_equal(case.model.covariance_, terminal_inference.covariance)

    components = {component.name: component for component in fitted.layout.penalties}
    first = components[_FACE_COMPONENTS[0]]
    second = components[_FACE_COMPONENTS[1]]
    finite_component = components[_FINITE_COMPONENT]
    assert first.group_sl.stop <= second.group_sl.start
    individual_rank = sum(
        build_penalty_face(fitted.layout, (name,)).constraint_rank for name in _FACE_COMPONENTS
    )
    assert face.constraint_rank == individual_rank
    finite_penalty = finite_lambda * penalty_component_dense_matrix(finite_component)
    finite_coefficients = terminal.coefficients[finite_component.group_sl]
    finite_penalty_mass = float(np.linalg.norm(finite_penalty @ finite_coefficients, ord=2))
    assert finite_penalty_mass > _penalty_action_roundoff(
        finite_penalty,
        finite_coefficients,
    )

    assert np.all(np.isfinite(public_predictions))
    assert np.all(public_predictions > 0.0)
    assert np.all(np.isfinite(public_parameters))
    assert np.all(public_parameters > 0.0)
    assert np.all(np.isfinite(terminal_covariance))
    covariance_scale = max(1.0, float(np.linalg.norm(terminal_covariance, ord=2)))
    covariance_bound = 256.0 * face.width * epsilon * covariance_scale
    assert np.linalg.norm(face.constraint_basis.T @ terminal_covariance, ord=2) <= covariance_bound


def test_scalar_first_counterfactual_refuses_the_finite_companion(
    gamma_joint_face_case: _GammaJointFaceCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills a scalar-first fallback hidden behind a successful joint fit."""

    def disable_joint_activation(*args: object, **kwargs: object):
        del args, kwargs
        return None, None

    monkeypatch.setattr(smoothing_loop, "_try_joint_exact_face", disable_joint_activation)
    monkeypatch.setattr(efs_module, "_try_joint_exact_face", disable_joint_activation)
    case = gamma_joint_face_case
    scalar_first = _fit_public_gamma_joint_face(case.frame, case.response, case.weights)
    fitted = scalar_first._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert scalar_first.exact_face_components_ == ()
    assert smoothing.unresolved_upper_bound == _FACE_COMPONENTS

    refusal = smoothing.history[-1]
    first_name, companion_name = _FACE_COMPONENTS
    assert refusal.refused_face_components == (first_name,)
    assert refusal.endpoint_assessment_failure_reason == "cap_not_stationary"
    assert len(refusal.coefficient_fit_indices) == 1
    cap_fit = smoothing.coefficient_fits[refusal.coefficient_fit_indices[0]]
    assert cap_fit.coefficient_face is None
    cap_kkt = _retained_kkt_relative(cap_fit)
    assert cap_kkt > max(cap_fit.config.tolerance, _scaled_solve_roundoff(cap_fit))

    components = {component.name: component for component in fitted.layout.penalties}
    companion = components[companion_name]
    local_penalty = penalty_component_dense_matrix(companion)
    local_coefficients = cap_fit.coefficients[companion.group_sl]
    weighted_penalty = refusal.lambdas_before[companion_name] * local_penalty
    penalty_mass = float(np.linalg.norm(weighted_penalty @ local_coefficients, ord=2))
    assert penalty_mass > _penalty_action_roundoff(weighted_penalty, local_coefficients)

    scalar_face = build_penalty_face(fitted.layout, (first_name,))
    companion_face = build_penalty_face(fitted.layout, (companion_name,))
    joint_face = build_penalty_face(fitted.layout, _FACE_COMPONENTS)
    assert scalar_face.component_names == (first_name,)
    assert companion_name not in scalar_face.component_names
    assert joint_face.constraint_rank == (
        scalar_face.constraint_rank + companion_face.constraint_rank
    )

    accepted = gamma_joint_face_case.model._require_fitted().smoothing
    assert accepted is not None
    joint_kkt = _retained_kkt_relative(accepted.terminal_fit)
    assert joint_kkt <= min(
        accepted.terminal_fit.config.tolerance,
        _scaled_solve_roundoff(accepted.terminal_fit),
    )
    accepted_face = accepted.terminal_fit.coefficient_face
    assert accepted_face is not None
    assert accepted_face.component_names == _FACE_COMPONENTS
    _assert_exact_face_residual(accepted_face, accepted.terminal_fit.coefficients)


def test_public_consumers_cannot_use_the_provisional_activation_fit(
    gamma_joint_face_case: _GammaJointFaceCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills provisional result/inference consumers with correct terminal history."""
    case = gamma_joint_face_case
    fitted = case.model._require_fitted()
    smoothing = fitted.smoothing
    assert smoothing is not None
    activation = next(item for item in smoothing.history if item.activated_face_components)
    assert activation.accepted_fit_index is not None
    activation_fit = smoothing.coefficient_fits[activation.accepted_fit_index]
    activation_inference = compute_joint_inference(fitted.layout, activation_fit)

    result_property = DenseDistributionalModel.result
    inference_property = DenseDistributionalModel.inference
    covariance_property = DenseDistributionalModel.covariance

    def routed_result(candidate: DenseDistributionalModel) -> DenseSolverResult:
        if candidate is fitted:
            return activation_fit
        return result_property.__get__(candidate, DenseDistributionalModel)

    def routed_inference(candidate: DenseDistributionalModel):
        if candidate is fitted:
            return activation_inference
        return inference_property.__get__(candidate, DenseDistributionalModel)

    def routed_covariance(candidate: DenseDistributionalModel) -> np.ndarray:
        if candidate is fitted:
            return candidate.inference.covariance
        return covariance_property.__get__(candidate, DenseDistributionalModel)

    monkeypatch.setattr(DenseDistributionalModel, "result", property(routed_result))
    monkeypatch.setattr(DenseDistributionalModel, "inference", property(routed_inference))
    monkeypatch.setattr(DenseDistributionalModel, "covariance", property(routed_covariance))

    dimension = fitted.layout.n_coefficients
    activation_parameters = _parameters_from_eta(fitted, activation_fit.eta)
    activation_predictions = np.asarray(
        fitted.family.default_prediction(activation_parameters),
        dtype=np.float64,
    )
    terminal = smoothing.terminal_fit
    terminal_parameters = _parameters_from_eta(fitted, terminal.eta)
    terminal_predictions = np.asarray(
        fitted.family.default_prediction(terminal_parameters),
        dtype=np.float64,
    )
    terminal_covariance = fitted.fit_state.inference.covariance

    routed_eta = case.model.predict_link(case.frame).to_numpy()
    routed_parameters = case.model.predict_parameters(case.frame).to_numpy()
    routed_predictions = case.model.predict(case.frame)
    routed_covariance = case.model.covariance_
    assert fitted.inference is activation_inference
    _assert_consumer_matches(routed_eta, activation_fit.eta, dimension=dimension)
    _assert_consumer_matches(
        routed_parameters,
        activation_parameters,
        dimension=dimension,
    )
    _assert_consumer_matches(
        routed_predictions,
        activation_predictions,
        dimension=dimension,
    )
    np.testing.assert_array_equal(routed_covariance, activation_inference.covariance)
    assert _maximum_absolute_error(routed_eta, terminal.eta) > _consumer_roundoff_bound(
        routed_eta,
        terminal.eta,
        dimension=dimension,
    )
    assert _maximum_absolute_error(
        routed_parameters,
        terminal_parameters,
    ) > _consumer_roundoff_bound(
        routed_parameters,
        terminal_parameters,
        dimension=dimension,
    )
    assert _maximum_absolute_error(
        routed_predictions,
        terminal_predictions,
    ) > _consumer_roundoff_bound(
        routed_predictions,
        terminal_predictions,
        dimension=dimension,
    )
    assert _maximum_absolute_error(
        routed_covariance,
        terminal_covariance,
    ) > _consumer_roundoff_bound(
        routed_covariance,
        terminal_covariance,
        dimension=dimension,
    )

    with pytest.raises(AssertionError):
        test_public_gamma_reml_accepts_the_strict_joint_infinity_face(case)


def test_shared_efs_joint_face_path_has_no_gamma_ls_dependency() -> None:
    """Kills routing the accepted fixture through a family-name special case."""
    modules = (
        smoothing_objective,
        smoothing_evidence,
        smoothing_authority,
        smoothing_faces,
        smoothing_proposals,
        smoothing_loop,
    )
    tree = ast.Module(
        body=[
            node
            for module in modules
            for node in ast.parse(
                Path(module.__file__).read_text(encoding="utf-8"), filename=module.__file__
            ).body
        ],
        type_ignores=[],
    )
    family_specific_nodes: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
            if any(name.endswith(".families.gamma") for name in names):
                family_specific_nodes.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            imported = tuple(alias.name for alias in node.names)
            if (node.module or "").endswith(".families.gamma") or "GammaLS" in imported:
                family_specific_nodes.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == "GammaLS":
            family_specific_nodes.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "GammaLS":
            family_specific_nodes.append(node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "GammaLS" in node.value
        ):
            family_specific_nodes.append(node.lineno)

    assert family_specific_nodes == []
