from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace

import numpy as np
import pandas as pd
import pytest
import scipy.linalg

import superglm.distributional.efs as efs_module
import superglm.distributional.result as result_module
import superglm.distributional.smoothing.authority as smoothing_authority
import superglm.distributional.smoothing.endpoint_laml as endpoint_laml_module
import superglm.distributional.smoothing.faces as smoothing_faces
import superglm.distributional.smoothing.loop as smoothing_loop
import superglm.distributional.solver.solver as solver_module
from superglm._frame import as_eager_frame
from superglm.distributional.efs import fit_distributional_efs
from superglm.distributional.endpoint_laml import (
    EndpointDirectionEvidence,
    EndpointLaplaceDerivative,
    EndpointLaplaceError,
    EndpointLaplaceEvaluation,
    ProjectedPenaltyLogDet,
    evaluate_endpoint_laplace,
    evaluate_endpoint_laplace_derivative,
    projected_finite_penalty_logdet,
)
from superglm.distributional.face_efs import (
    _bounded_effective_rank,
    projected_component_states,
)
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    InitialParameterState,
    NaturalLikelihoodEvaluation,
    ObservationContract,
    ParameterSpec,
    ParameterSupport,
)
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.penalty_face import PenaltyFace, build_penalty_face
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import (
    DenseSolverConfig,
    DenseSolverResult,
    DistributionalEFSConfig,
    DistributionalEFSResult,
    JointEndpointDirectionEvidence,
)
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import ResolvedLikelihoodWeights
from superglm.features import Numeric, RandomEffect, Spline
from superglm.links import IdentityLink
from superglm.reml.multi_penalty import SimilarityTransformResult
from superglm.solvers.rank import decompose_gram
from superglm.types import PenaltyComponent

from ._distributional_weights import resolved_prior


@dataclass(frozen=True)
class _UnitGaussianPlan:
    weights: ResolvedLikelihoodWeights

    @property
    def plan_identifier(self) -> str:
        return f"unit-gaussian:v1:{self.weights.digest}"

    def take(self, indices: np.ndarray) -> _UnitGaussianPlan:
        return _UnitGaussianPlan(self.weights.take(indices))


class _UnitGaussian:
    """One-parameter unit-variance Gaussian used only as an analytic oracle."""

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return (
            ParameterSpec(
                name="mean",
                default_link=IdentityLink(),
                role="location",
                support=ParameterSupport(),
                curvature="fisher",
            ),
        )

    @property
    def default_prediction_name(self) -> str:
        return "mean"

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> _UnitGaussianPlan:
        if observation != COMPLETE_OBSERVATION:
            raise ValueError("unit Gaussian requires complete observations")
        if np.asarray(y).shape != weights.values.shape:
            raise ValueError("response and weights must have the same shape")
        return _UnitGaussianPlan(weights)

    def initialize(
        self,
        y: np.ndarray,
        plan: _UnitGaussianPlan,
    ) -> InitialParameterState:
        del plan
        return InitialParameterState(theta=np.zeros((len(y), 1), dtype=np.float64))

    def evaluate_natural(
        self,
        y: np.ndarray,
        theta: np.ndarray,
        plan: _UnitGaussianPlan,
        *,
        derivative_order: int = 2,
    ) -> NaturalLikelihoodEvaluation:
        residual = np.asarray(y, dtype=np.float64) - np.asarray(theta, dtype=np.float64)[:, 0]
        weights = plan.weights.values
        score = None if derivative_order < 1 else (weights * residual)[:, None]
        hessian = None if derivative_order < 2 else (-weights)[:, None]
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=-0.5 * weights * residual**2,
            parameter_independent_carrier=np.full(len(residual), 0.375, dtype=np.float64),
            score=score,
            hessian_packed=hessian,
        )

    def expected_information_natural(
        self,
        theta: np.ndarray,
        plan: _UnitGaussianPlan,
    ) -> np.ndarray:
        if len(theta) != len(plan.weights.values):
            raise ValueError("theta and weights must have the same row count")
        return plan.weights.values[:, None]

    def predictor_curvature_directional_derivative(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        eta_direction: np.ndarray,
        links,
        plan: _UnitGaussianPlan,
    ) -> np.ndarray:
        del eta_direction, links
        if np.asarray(eta).shape != (len(y), 1) or len(plan.weights.values) != len(y):
            raise ValueError("invalid unit-Gaussian directional state")
        return np.zeros((len(y), 1), dtype=np.float64)

    def default_prediction(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=np.float64)[:, 0]


class _ObservedOnlyGaussian:
    """Local family-neutrality fixture that deliberately exposes no Fisher API."""

    def __init__(self) -> None:
        self.base = GaussianLS()

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _UnitGaussianWithoutCurvatureDirection:
    """One-parameter fixture that deliberately omits the analytic exact-face calculus."""

    def __init__(self) -> None:
        self.base = _UnitGaussian()

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def expected_information_natural(self, theta, plan):
        return self.base.expected_information_natural(theta, plan)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


def _scalar_endpoint_fit(
    mean: float,
) -> tuple[StackedLayout, PenaltyFace, dict[str, float], DenseSolverResult]:
    family = _UnitGaussian()
    response = np.array([mean], dtype=np.float64)
    weights = resolved_prior(np.ones(1, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    assert layout.n_coefficients == 1
    assert len(layout.penalties) == 1
    component = layout.penalties[0]
    assert component.penalty_kind == "identity"
    face = build_penalty_face(layout, (component.name,))
    lambdas = {component.name: 0.0}
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),
        layout.penalty_matrix(lambdas),
        coefficient_face=face,
    )
    return layout, face, lambdas, result


@pytest.mark.parametrize("mean", [0.5, 1.0, 2.0])
def test_scalar_endpoint_derivative_matches_closed_form(mean: float) -> None:
    """The analytic face derivative distinguishes infinity from a finite optimum."""

    family = _UnitGaussian()
    response = np.array([mean], dtype=np.float64)
    weights = resolved_prior(np.ones(1, dtype=np.float64))
    layout, face, lambdas, result = _scalar_endpoint_fit(mean)
    derivative = evaluate_endpoint_laplace_derivative(
        family,
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),
        lambdas=lambdas,
        component_name=face.component_names[0],
        finite_face=None,
        endpoint_face=face,
        endpoint_fit=result,
    )

    assert isinstance(derivative, EndpointLaplaceDerivative)
    assert derivative.authority_identifier == "analytic-observed-curvature-direction/v1"
    np.testing.assert_allclose(derivative.derivative, 0.5 * (1.0 - mean**2), atol=2e-14)
    assert derivative.decision == (
        "endpoint" if mean < 1.0 else "finite" if mean > 1.0 else "unresolved"
    )


@pytest.mark.parametrize(
    ("weight", "zero_work"),
    [(1.0e-308, True), (1.0e-20, False), (1.0e-10, False)],
    ids=["zero-work", "zero-error-scale", "subnormal-error-scale"],
)
def test_derivative_error_arithmetic_has_no_absolute_unit_floor(
    weight: float,
    zero_work: bool,
) -> None:
    """Zero or underflowed error work remains homogeneously unresolved."""
    family = _UnitGaussian()
    response = np.zeros(1, dtype=np.float64)
    weights = resolved_prior(np.array([weight], dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component = layout.penalties[0]
    penalty_scale = 1.0e300
    scaled = np.array([[penalty_scale]], dtype=np.float64)
    component = replace(
        component,
        omega_raw=scaled,
        omega_ssp=scaled,
        eigvals_omega=np.array([penalty_scale], dtype=np.float64),
        log_det_omega_plus=float(np.log(penalty_scale)),
        penalty_kind="dense",
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    lambdas = {component.name: 0.0}
    likelihood_plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    endpoint_fit = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        config=DenseSolverConfig(coefficient_curvature="observed"),
        coefficient_face=face,
    )

    derivative = evaluate_endpoint_laplace_derivative(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        lambdas=lambdas,
        component_name=component.name,
        finite_face=None,
        endpoint_face=face,
        endpoint_fit=endpoint_fit,
    )

    work_magnitude = (
        abs(derivative.profile_score_term)
        + abs(derivative.curvature_schur_term)
        + abs(derivative.curvature_drift_term)
    )
    if zero_work:
        assert derivative.derivative == 0.0
        assert work_magnitude == 0.0
        assert derivative.numerical_error == 0.0
        assert derivative.lower_bound == 0.0
        assert derivative.upper_bound == 0.0
    else:
        assert derivative.derivative > 0.0
        assert work_magnitude > 0.0
        assert abs(derivative.derivative) <= derivative.numerical_error <= work_magnitude
        assert derivative.lower_bound <= 0.0 <= derivative.upper_bound
    assert derivative.decision == "unresolved"


def _scalar_efs_fit(
    mean: float,
    *,
    maximum_lambda: float = 10.0,
    penalty_scale: float = 1.0,
    starting_lambda: float | None = None,
    family=None,
    max_iterations: int = 8,
    practical_convergence: bool = False,
    practical_parameter_tolerance: float = 1.0e-3,
    plateau_tolerance: float = 1.0e-7,
):
    family = _UnitGaussian() if family is None else family
    response = np.array([mean], dtype=np.float64)
    weights = resolved_prior(np.ones(1, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component = layout.penalties[0]
    if penalty_scale != 1.0:
        scaled = np.array([[penalty_scale]], dtype=np.float64)
        component = replace(
            component,
            omega_raw=scaled,
            omega_ssp=scaled,
            eigvals_omega=np.array([penalty_scale], dtype=np.float64),
            log_det_omega_plus=float(np.log(penalty_scale)),
            penalty_kind="dense",
        )
        layout = replace(layout, penalties=(component,))
    result = fit_distributional_efs(
        family,  # type: ignore[arg-type]
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),  # type: ignore[arg-type]
        lambdas={component.name: (maximum_lambda if starting_lambda is None else starting_lambda)},
        solver_config=DenseSolverConfig(max_iterations=50, tolerance=1.0e-12),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=max_iterations,
            tolerance=1.0e-8,
            initial_lambda=min(0.1, maximum_lambda),
            maximum_lambda=maximum_lambda,
            practical_convergence=practical_convergence,
            practical_parameter_tolerance=practical_parameter_tolerance,
            plateau_tolerance=plateau_tolerance,
        ),
    )
    return component.name, result


def test_practical_plateau_stops_an_immaterial_tail_without_certifying_infinity() -> None:
    component_name, smoothing = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=1.0e10,
        starting_lambda=0.1,
        max_iterations=300,
        practical_convergence=True,
        practical_parameter_tolerance=1.0e-3,
    )

    assert smoothing.converged is True
    assert smoothing.convergence_reason == "practical_plateau"
    assert smoothing.iterations < smoothing.config.max_iterations
    assert smoothing.terminal_raw_max_log_step > smoothing.config.tolerance
    assert smoothing.matched_certified is False
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.lambdas[component_name] < smoothing.config.maximum_lambda
    with pytest.raises(RuntimeError, match="strict smoothing convergence"):
        smoothing.assert_matched_certified()
    with pytest.raises(ValueError, match="fitted parameters"):
        replace(
            smoothing,
            config=replace(smoothing.config, practical_parameter_tolerance=0.0),
        )
    with pytest.raises(ValueError, match="objective_relative_change"):
        replace(
            smoothing.history[-1],
            objective_relative_change=2.0 * smoothing.config.plateau_tolerance,
        )
    with pytest.raises(ValueError, match="strict stationarity"):
        replace(
            smoothing,
            terminal_raw_max_log_step=smoothing.config.tolerance,
        )


def test_practical_plateau_requires_stable_fitted_parameters() -> None:
    _component_name, smoothing = _scalar_efs_fit(
        2.0,
        maximum_lambda=1.0e10,
        starting_lambda=100.0,
        max_iterations=3,
        practical_convergence=True,
        practical_parameter_tolerance=1.0e-12,
        plateau_tolerance=1.0,
    )

    assert smoothing.convergence_reason != "practical_plateau"


def test_practical_plateau_never_fires_under_unresolved_upper_pressure() -> None:
    component_name, smoothing = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=1002.5,
        starting_lambda=1000.0,
        max_iterations=10,
        practical_convergence=True,
        practical_parameter_tolerance=1.0e-3,
        plateau_tolerance=1.0e-7,
    )

    assert smoothing.convergence_reason != "practical_plateau"
    assert not (smoothing.converged and smoothing.unresolved_upper_bound)
    assert smoothing.terminal_raw_max_log_step > smoothing.config.tolerance
    assert smoothing.matched_certified is False
    # The cap machinery owns the outcome: the practical run terminates exactly
    # where the strict run does, with the same named pressure.
    _strict_name, strict = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=1002.5,
        starting_lambda=1000.0,
        max_iterations=10,
        practical_convergence=False,
        practical_parameter_tolerance=1.0e-3,
        plateau_tolerance=1.0e-7,
    )
    assert smoothing.convergence_reason == strict.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == strict.unresolved_upper_bound == (component_name,)


def test_practical_plateau_recomputes_objective_change_from_fitted_objectives() -> None:
    _component_name, smoothing = _scalar_efs_fit(
        2.0,
        maximum_lambda=1.0e10,
        starting_lambda=100.0,
        max_iterations=3,
        practical_convergence=False,
        practical_parameter_tolerance=1.0,
        plateau_tolerance=0.0,
    )

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"
    assert all(item.objective_relative_change > 0.0 for item in smoothing.history)
    with pytest.raises(ValueError, match="objective_relative_change"):
        forged_history = tuple(
            replace(item, objective_relative_change=0.0) for item in smoothing.history
        )
        replace(
            smoothing,
            config=replace(smoothing.config, practical_convergence=True),
            converged=True,
            convergence_reason="practical_plateau",
            history=forged_history,
        )


def test_practical_plateau_rejects_a_forged_exact_face() -> None:
    _names, smoothing = _two_face_efs_fit()
    terminal = smoothing.terminal_fit
    fits = list(smoothing.coefficient_fits)
    history = list(smoothing.history)
    zeros = dict.fromkeys(smoothing.lambdas, 0.0)
    for _ in range(smoothing.config.plateau_iterations):
        source_index = len(fits) - 1
        fits.append(terminal)
        accepted_index = len(fits) - 1
        history.append(
            replace(
                history[-1],
                iteration=len(history) + 1,
                source_fit_index=source_index,
                lambdas_before=smoothing.lambdas,
                proposed_lambdas=smoothing.lambdas,
                lambdas_after=smoothing.lambdas,
                proposed_log_steps=zeros,
                accepted_log_steps=zeros,
                objective_before=smoothing.objective,
                objective_after=smoothing.objective,
                objective_relative_change=0.0,
                max_proposed_log_step=0.0,
                max_accepted_log_step=0.0,
                accepted=True,
                acceleration_outcome="disabled",
                acceleration_refusal_reason=None,
                accelerated_fit_index=None,
                backtracks=0,
                raw_backtracks=0,
                coefficient_fit_indices=(accepted_index,),
                accepted_fit_index=accepted_index,
                coefficient_tolerances=(terminal.config.tolerance,),
                boundary_nominations=(),
                update_curvature=terminal.terminal_curvature,
                accepted_curvature=terminal.terminal_curvature,
                activated_face_components=(),
                deactivated_face_components=(),
                revalidated_face_components=(),
                refused_face_components=(),
                endpoint_direction_evidence=None,
                endpoint_assessment_failure_reason=None,
                joint_rollback_penalty_fingerprint=None,
            )
        )

    with pytest.raises(ValueError, match="exact coefficient face"):
        replace(
            smoothing,
            config=replace(smoothing.config, practical_convergence=True),
            convergence_reason="practical_plateau",
            iterations=len(history),
            history=tuple(history),
            coefficient_fits=tuple(fits),
            terminal_fit_index=len(fits) - 1,
            terminal_raw_max_log_step=2.0 * smoothing.config.tolerance,
        )


def _gamma_isolated_cap_face_problem():
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
    response = mean * residual
    weights = resolved_prior(0.65 + 0.7 * (w + 1.0) / 2.0)
    family = GammaLS()
    frame = as_eager_frame(pd.DataFrame({"x": x, "w": w}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "mean",
                    {
                        "x": Spline(kind="cr", n_knots=5),
                        "w": Spline(kind="cr", n_knots=5),
                    },
                ),
                Predictor("scale", {}),
            ),
        )
    )
    capped_name, finite_name = layout.penalty_names
    efs_config = DistributionalEFSConfig(outer="efs", max_iterations=120, tolerance=1.0e-3)
    solver_config = DenseSolverConfig(max_iterations=150, tolerance=1.0e-9)
    lambdas = {
        capped_name: efs_config.maximum_lambda,
        finite_name: 0.5,
    }
    likelihood_plan = family.bind_likelihood(
        response,
        weights,
        COMPLETE_OBSERVATION,
    )
    smoothing = fit_distributional_efs(
        family,
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        solver_config=solver_config,
        efs_config=efs_config,
    )
    return (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        capped_name,
        finite_name,
        solver_config,
        smoothing,
    )


def _zero_step_strict_armijo_face_problem() -> tuple[
    _UnitGaussian,
    StackedLayout,
    np.ndarray,
    _UnitGaussianPlan,
    dict[str, float],
    PenaltyFace,
    np.ndarray,
    DenseSolverConfig,
    DenseSolverResult,
]:
    family = _UnitGaussian()
    response = np.array([1.0, 3.0], dtype=np.float64)
    weights = resolved_prior(np.ones(2, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["left", "right"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}),),
        )
    )
    component_name = layout.penalty_names[0]
    face = build_penalty_face(layout, (component_name,))
    lambdas = {component_name: 1.0}
    likelihood_plan = family.bind_likelihood(
        response,
        weights,
        COMPLETE_OBSERVATION,
    )
    authority_config = efs_module._face_authority_config(
        DenseSolverConfig(max_iterations=10, tolerance=1.0e-9)
    )
    # On this exact quadratic, gain(alpha) / (alpha * score.T @ step) is
    # 1 - alpha / 2.  The only two trials, alpha = 1 and 1/2, therefore
    # cannot meet c = 0.9, while the full Newton correction is the optimum.
    strict_line_search_config = replace(
        authority_config,
        max_backtracks=1,
        armijo_constant=0.9,
    )
    initial = face.project(np.zeros(layout.n_coefficients, dtype=np.float64))
    penalty = layout.penalty_matrix({component_name: 0.0})

    source = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,
        penalty,
        initial=initial,
        config=strict_line_search_config,
        coefficient_face=face,
    )
    return (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        strict_line_search_config,
        source,
    )


def _resolution_limited_face_problem(
    coefficient_scale: float = 1.0,
) -> tuple[
    _UnitGaussian,
    StackedLayout,
    np.ndarray,
    _UnitGaussianPlan,
    dict[str, float],
    PenaltyFace,
    np.ndarray,
    DenseSolverConfig,
]:
    family = _UnitGaussian()
    response = np.array([0.0, np.nextafter(2.0, np.inf)], dtype=np.float64)
    weights = resolved_prior(np.ones(2, dtype=np.float64))
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.full(2, coefficient_scale, dtype=np.float64),
                "effect": ["left", "right"],
            }
        )
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "mean",
                    {"x": Numeric(), "effect": RandomEffect()},
                    intercept=False,
                ),
            ),
        )
    )
    component_name = layout.penalty_names[0]
    face = build_penalty_face(layout, (component_name,))
    lambdas = {component_name: 0.0}
    likelihood_plan = family.bind_likelihood(
        response,
        weights,
        COMPLETE_OBSERVATION,
    )
    initial = face.project(np.array([1.0 / coefficient_scale, 0.0, 0.0], dtype=np.float64))
    config = DenseSolverConfig(
        max_iterations=8,
        tolerance=1.0e-20,
        coefficient_curvature="observed",
    )
    return family, layout, response, likelihood_plan, lambdas, face, initial, config


def test_ordinary_noop_preserves_composite_stop_without_publishing_an_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills evaluating/appending a no-op or changing the ordinary composite stop."""
    family = _UnitGaussian()
    response = np.array([1.0, 1.0, np.nextafter(1.0, np.inf)], dtype=np.float64)
    weights = resolved_prior(np.ones(response.size, dtype=np.float64))
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.ones(response.size, dtype=np.float64),
                "effect": ["left", "middle", "right"],
            }
        )
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "mean",
                    {"x": Numeric(), "effect": RandomEffect()},
                    intercept=False,
                ),
            ),
        )
    )
    component_name = layout.penalty_names[0]
    face = build_penalty_face(layout, (component_name,))
    lambdas = {component_name: 0.0}
    likelihood_plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    initial = face.project(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    config = DenseSolverConfig(
        max_iterations=8,
        tolerance=1.0e-20,
        coefficient_curvature="observed",
    )
    evaluated_coefficients: list[np.ndarray] = []
    real_evaluate_state = solver_module._evaluate_state

    def record_evaluation(*args: object, **kwargs: object):
        coefficients = args[1] if len(args) > 1 else kwargs["coefficients"]
        evaluated_coefficients.append(np.array(coefficients, copy=True))
        return real_evaluate_state(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(solver_module, "_evaluate_state", record_evaluation)
    result = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=face,
    )

    assert result.converged is True
    assert result.convergence_reason == "objective_and_step"
    assert result.iterations == 0
    assert result.history == ()
    assert result.objective_relative_change == 0.0
    assert result.step_relative == 0.0
    assert result.score_relative > config.tolerance
    np.testing.assert_array_equal(
        face.reduce_vector(result.terminal_score),
        np.array([np.spacing(1.0)]),
    )
    assert len(evaluated_coefficients) == 1
    np.testing.assert_array_equal(evaluated_coefficients[0], initial)


def test_private_score_only_solver_reports_resolution_limited_stationarity_honestly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills relabelling objective equality as progress or dropping the raw score."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        config,
    ) = _resolution_limited_face_problem()
    evaluated: list[tuple[np.ndarray, float | None]] = []
    real_evaluate_state = solver_module._evaluate_state

    def record_evaluation(*args: object, **kwargs: object):
        coefficients = args[1] if len(args) > 1 else kwargs["coefficients"]
        state = real_evaluate_state(*args, **kwargs)  # type: ignore[arg-type]
        evaluated.append(
            (
                np.array(coefficients, copy=True),
                None if state is None else state.penalized_optimizing_log_likelihood,
            )
        )
        return state

    monkeypatch.setattr(solver_module, "_evaluate_state", record_evaluation)
    result = solver_module._fit_dense_fixed_lambda_score_only(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=face,
    )

    retained_score = face.reduce_vector(result.terminal_score)
    retained_correction = face.reduce_vector(result.solve_terminal(result.terminal_score))
    predicted_gain = 0.5 * float(retained_score @ retained_correction)
    objective = result.penalized_optimizing_log_likelihood
    assert objective is not None
    objective_ulp = float(np.nextafter(objective, np.inf) - objective)

    assert result.converged is True
    assert result.convergence_reason == "resolution_limited_stationarity"
    assert result.iterations == 0
    assert result.history == ()
    np.testing.assert_array_equal(retained_score, np.array([np.spacing(2.0)]))
    assert efs_module._endpoint_retained_kkt_relative(result) > config.tolerance
    assert 0.0 < predicted_gain < objective_ulp
    assert len(evaluated) >= 2
    assert all(not np.array_equal(coefficients, initial) for coefficients, _ in evaluated[1:])
    assert all(trial_objective == objective for _, trial_objective in evaluated[1:])


@pytest.mark.parametrize(
    "mutation",
    [
        "no_face",
        "fisher",
        "terminal_fallback",
        "rank_loss",
        "levenberg_shift",
        "predictor_cap",
        "no_identical_candidate",
    ],
)
def test_private_resolution_limited_stationarity_refuses_each_missing_gate(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Kills omitting any live face, solve, cap, decrement, or trial-evidence gate."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        config,
    ) = _resolution_limited_face_problem()
    selected_face: PenaltyFace | None = face
    original_decompositions = []

    if mutation == "no_face":
        selected_face = None
    elif mutation == "fisher":
        config = replace(config, coefficient_curvature="fisher")
    elif mutation == "terminal_fallback":
        real_resolve_curvature = solver_module.resolve_curvature

        def inject_terminal_fallback(*args: object, **kwargs: object):
            decision = real_resolve_curvature(*args, **kwargs)  # type: ignore[arg-type]
            return replace(
                decision,
                telemetry=replace(
                    decision.telemetry,
                    fallback_count=1,
                    reason="test fallback mutation",
                ),
            )

        monkeypatch.setattr(solver_module, "resolve_curvature", inject_terminal_fallback)
    elif mutation in {"rank_loss", "levenberg_shift"}:
        real_solve_direction = solver_module._solve_coefficient_direction

        class _RankOnlyMutation:
            def __init__(self, decomposition) -> None:
                self._decomposition = decomposition
                self.rank = decomposition.rank - 1

            def solve(self, right_hand_side: np.ndarray) -> np.ndarray:
                return self._decomposition.solve(right_hand_side)

            def __getattr__(self, name: str):
                return getattr(self._decomposition, name)

        def mutate_direction(*args: object, **kwargs: object):
            direction = real_solve_direction(*args, **kwargs)  # type: ignore[arg-type]
            if mutation == "rank_loss":
                original_decompositions.append(direction.decomposition)
                return replace(
                    direction,
                    decomposition=_RankOnlyMutation(direction.decomposition),
                )
            return replace(direction, levenberg_shift=np.finfo(np.float64).eps)

        monkeypatch.setattr(solver_module, "_solve_coefficient_direction", mutate_direction)
    elif mutation == "predictor_cap":
        real_cap = solver_module._cap_predictor_step

        def shorten_step(*args: object, **kwargs: object):
            step, _scale = real_cap(*args, **kwargs)  # type: ignore[arg-type]
            return 0.5 * step, 0.5

        monkeypatch.setattr(solver_module, "_cap_predictor_step", shorten_step)
    else:
        config = replace(config, max_backtracks=1)

    result = solver_module._fit_dense_fixed_lambda_score_only(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=selected_face,
    )

    assert result.convergence_reason != "resolution_limited_stationarity"
    if mutation != "no_face":
        assert result.converged is False
        assert result.convergence_reason == "line_search_failed"
    if mutation == "rank_loss":
        assert len(original_decompositions) == 1
        retained_score = face.reduce_vector(result.terminal_score)
        retained_correction = original_decompositions[0].solve(retained_score)
        assert result_module._resolution_limited_decrement_is_within_objective_ulp(
            retained_score,
            retained_correction,
            result.penalized_optimizing_log_likelihood,
        )


def test_resolution_limited_decrement_encloses_subnormal_dot_underflow() -> None:
    """Kills a relative-only dot error bound and duplicate live/result predicates."""
    tiny = np.nextafter(0.0, np.inf)
    correction = np.full(32, np.ldexp(1.0, -537), dtype=np.float64)
    score = np.full(32, np.ldexp(1.0, -539), dtype=np.float64)
    score[:2] *= 3.0
    objective = np.ldexp(1.0, -1020)
    diagonal = np.full(32, 0.25, dtype=np.float64)
    diagonal[:2] = 0.75

    # Exactly, the dot is 9*tiny and the predicted gain is 4.5*tiny,
    # while the upward objective ULP is only 4*tiny.
    np.testing.assert_array_equal(diagonal * correction, score)
    assert np.all(diagonal > 0.0)
    assert np.nextafter(objective, np.inf) - objective == 4.0 * tiny
    assert not result_module._resolution_limited_decrement_is_within_objective_ulp(
        score,
        correction,
        objective,
    )
    assert (
        solver_module._resolution_limited_decrement_is_within_objective_ulp
        is result_module._resolution_limited_decrement_is_within_objective_ulp
    )

    objective_ulp = np.spacing(1.0)
    immediately_below = 2.0 * objective_ulp
    for _ in range(6):
        immediately_below = np.nextafter(immediately_below, 0.0)
    nearest_refused = np.nextafter(immediately_below, np.inf)
    assert 0.5 * immediately_below < objective_ulp
    assert result_module._resolution_limited_decrement_is_within_objective_ulp(
        np.array([immediately_below]),
        np.array([1.0]),
        1.0,
    )
    assert not result_module._resolution_limited_decrement_is_within_objective_ulp(
        np.array([nearest_refused]),
        np.array([1.0]),
        1.0,
    )


def test_private_resolution_limited_stationarity_refuses_a_rejected_distinct_improvement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills omitting distinct-trial improvement after a later identical candidate."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        config,
    ) = _resolution_limited_face_problem()
    config = replace(config, max_predictor_step=1.0e6, max_backtracks=200)
    real_solve_direction = solver_module._solve_coefficient_direction
    real_evaluate_state = solver_module._evaluate_state
    distinct_trials: list[np.ndarray] = []
    current_objective = -1.0000000000000004
    improved_objective = float(np.nextafter(current_objective, np.inf))

    def enlarge_applied_direction(*args: object, **kwargs: object):
        direction = real_solve_direction(*args, **kwargs)  # type: ignore[arg-type]
        return replace(direction, step=np.ldexp(direction.step, 67))

    def reject_first_representable_improvement(*args: object, **kwargs: object):
        coefficients = np.asarray(args[1] if len(args) > 1 else kwargs["coefficients"])
        state = real_evaluate_state(*args, **kwargs)  # type: ignore[arg-type]
        if state is None or np.array_equal(coefficients, initial):
            return state
        distinct_trials.append(np.array(coefficients, copy=True))
        objective = improved_objective if len(distinct_trials) == 1 else current_objective
        return replace(
            state,
            optimizing_log_likelihood=objective,
            log_likelihood=objective + state.parameter_independent_carrier,
            penalized_optimizing_log_likelihood=objective,
            penalized_log_likelihood=objective + state.parameter_independent_carrier,
        )

    monkeypatch.setattr(solver_module, "_solve_coefficient_direction", enlarge_applied_direction)
    monkeypatch.setattr(solver_module, "_evaluate_state", reject_first_representable_improvement)
    result = solver_module._fit_dense_fixed_lambda_score_only(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=face,
    )

    retained_score = face.reduce_vector(result.terminal_score)
    retained_correction = face.reduce_vector(result.solve_terminal(result.terminal_score))
    assert len(distinct_trials) > 1
    assert result.history == ()
    assert result.converged is False
    assert result.convergence_reason == "line_search_failed"
    assert result_module._resolution_limited_decrement_is_within_objective_ulp(
        retained_score,
        retained_correction,
        result.penalized_optimizing_log_likelihood,
    )


def _resolution_limited_solver_result() -> DenseSolverResult:
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        config,
    ) = _resolution_limited_face_problem()
    result = solver_module._fit_dense_fixed_lambda_score_only(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=face,
    )
    assert result.convergence_reason == "resolution_limited_stationarity"
    return result


def test_endpoint_authority_routes_an_ordinary_plateau_through_private_resolution() -> None:
    """Kills bypassing private policy for an above-tolerance ordinary plateau."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        config,
    ) = _resolution_limited_face_problem(3.0e6)
    response = np.array([0.0, 2.0], dtype=np.float64)
    for _ in range(19):
        response[1] = np.nextafter(response[1], np.inf)
    likelihood_plan = family.bind_likelihood(
        response,
        likelihood_plan.weights,
        COMPLETE_OBSERVATION,
    )
    ordinary = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=config,
        coefficient_face=face,
    )
    result = efs_module._fit_endpoint_authority_stationary(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        lambdas=lambdas,
        face=face,
        initial=initial,
        config=config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert ordinary.converged is True
    assert ordinary.convergence_reason == "objective_and_step"
    assert efs_module._endpoint_retained_kkt_relative(ordinary) > config.tolerance
    assert result.converged is True
    assert result.convergence_reason == "resolution_limited_stationarity"
    assert result.coefficient_face is face
    assert result.history == ()
    assert efs_module._endpoint_retained_kkt_relative(result) > config.tolerance
    assert efs_module._endpoint_retained_kkt_relative(result) < (
        efs_module._endpoint_retained_kkt_relative(ordinary)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "zero_score",
        "ordinary_score_relabelling",
        "no_face",
        "fisher",
        "fallback",
        "rank_loss",
        "corrupted_curvature",
        "corrupted_score",
        "decrement_above_ulp",
    ],
)
def test_resolution_limited_result_reconstruction_rejects_forged_authority(
    mutation: str,
) -> None:
    """Kills trusting the reason string instead of rebuilding its Newton decrement."""
    result = _resolution_limited_solver_result()

    with pytest.raises(ValueError, match="resolution|score convergence"):
        if mutation == "zero_score":
            replace(result, terminal_score=np.zeros_like(result.terminal_score))
        elif mutation == "ordinary_score_relabelling":
            replace(result, convergence_reason="score")
        elif mutation == "no_face":
            replace(result, coefficient_face=None, terminal_reduced_rank=None)
        elif mutation == "fisher":
            replace(
                result,
                config=replace(result.config, coefficient_curvature="fisher"),
            )
        elif mutation == "fallback":
            replace(
                result,
                terminal_curvature=replace(
                    result.terminal_curvature,
                    fallback_count=1,
                    reason="test fallback mutation",
                ),
            )
        elif mutation == "rank_loss":
            assert result.terminal_reduced_rank is not None
            replace(
                result,
                terminal_reduced_rank=replace(
                    result.terminal_reduced_rank,
                    rank=result.terminal_reduced_rank.rank - 1,
                ),
            )
        elif mutation == "corrupted_curvature":
            corrupted = np.ldexp(result.terminal_data_curvature, -100)
            replace(
                result,
                terminal_data_curvature=corrupted,
                terminal_penalized_curvature=corrupted + result.penalty,
            )
        elif mutation == "corrupted_score":
            replace(result, terminal_score=np.ldexp(result.terminal_score, 30))
        else:
            replace(result, terminal_score=np.ldexp(result.terminal_score, 26))


def test_resolution_limited_newton_decrement_is_invariant_to_coefficient_rescaling() -> None:
    """Kills comparing the invariant predicted gain with a coefficient ULP."""
    gains: list[float] = []
    coefficient_ulp_mutant_verdicts: list[bool] = []

    for coefficient_scale in (1.0, 1.0e16):
        (
            family,
            layout,
            response,
            likelihood_plan,
            lambdas,
            face,
            initial,
            config,
        ) = _resolution_limited_face_problem(coefficient_scale)
        result = solver_module._fit_dense_fixed_lambda_score_only(
            family,  # type: ignore[arg-type]
            layout,
            response,
            likelihood_plan,  # type: ignore[arg-type]
            layout.penalty_matrix(lambdas),
            initial=initial,
            config=config,
            coefficient_face=face,
        )
        retained_score = face.reduce_vector(result.terminal_score)
        retained_correction = face.reduce_vector(result.solve_terminal(result.terminal_score))
        predicted_gain = 0.5 * float(retained_score @ retained_correction)
        coefficient_ulp = float(np.nextafter(abs(initial[0]), np.inf) - abs(initial[0]))

        assert result.convergence_reason == "resolution_limited_stationarity"
        gains.append(predicted_gain)
        coefficient_ulp_mutant_verdicts.append(predicted_gain <= coefficient_ulp)

    np.testing.assert_allclose(gains[1], gains[0], rtol=8.0 * np.finfo(float).eps, atol=0.0)
    assert coefficient_ulp_mutant_verdicts == [True, False]


def test_private_score_only_solver_continues_past_ordinary_objective_step_stop() -> None:
    """Kills ignoring the policy or omitting its accepted-state score check."""
    family = _UnitGaussian()
    response = np.array([5.0], dtype=np.float64)
    weights = resolved_prior(np.ones(1, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"unused": [0.0]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {}),),
        )
    )
    likelihood_plan = family.bind_likelihood(
        response,
        weights,
        COMPLETE_OBSERVATION,
    )
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    initial = np.array([1.0], dtype=np.float64)
    config = DenseSolverConfig(
        max_iterations=16,
        tolerance=1.0 / 5.0,
        coefficient_curvature="observed",
        max_predictor_step=1.0 / 4.0,
    )

    ordinary = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        penalty,
        initial=initial,
        config=config,
    )
    score_only_entry = getattr(
        solver_module,
        "_fit_dense_fixed_lambda_score_only",
        fit_dense_fixed_lambda,
    )
    score_only = score_only_entry(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        penalty,
        initial=initial,
        config=config,
    )

    epsilon = np.finfo(np.float64).eps
    assert ordinary.config is config
    assert ordinary.converged is True
    assert ordinary.convergence_reason == "objective_and_step"
    assert ordinary.iterations == 1
    np.testing.assert_array_equal(ordinary.coefficients, np.array([5.0 / 4.0]))
    assert ordinary.history[0].objective_relative_change == pytest.approx(
        31.0 / 288.0,
        rel=0.0,
        abs=16.0 * epsilon,
    )
    assert ordinary.history[0].step_relative == pytest.approx(
        1.0 / 9.0,
        rel=0.0,
        abs=16.0 * epsilon,
    )
    assert ordinary.score_relative == pytest.approx(
        120.0 / 257.0,
        rel=0.0,
        abs=16.0 * epsilon,
    )
    assert ordinary.score_relative > config.tolerance

    assert score_only.config is config
    assert score_only.converged is True
    assert score_only.convergence_reason == "score"
    assert score_only.iterations == config.max_iterations == 16
    np.testing.assert_array_equal(score_only.coefficients, response)
    np.testing.assert_array_equal(score_only.terminal_score, np.zeros(1, dtype=np.float64))
    assert score_only.score_relative == 0.0


def test_endpoint_authority_polishes_a_zero_step_strict_armijo_failure() -> None:
    """Kills returning a certified zero-step line-search source unchanged."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        strict_line_search_config,
        source,
    ) = _zero_step_strict_armijo_face_problem()
    source_kkt = efs_module._endpoint_retained_kkt_relative(source)
    source_rank = efs_module._endpoint_retained_rank(source)
    coefficient_scale = max(1.0, float(np.linalg.norm(source.coefficients, ord=2)))
    face_residual = float(np.linalg.norm(face.constraint_matrix @ source.coefficients, ord=2))

    assert source.converged is False
    assert source.convergence_reason == "line_search_failed"
    assert source.iterations == 0
    assert source_kkt > strict_line_search_config.tolerance
    assert source.config == strict_line_search_config
    assert source.coefficient_face is face
    assert face_residual <= face.null_residual_bound * coefficient_scale
    assert source_rank is not None
    assert source_rank.rank == face.reduced_width
    assert source_rank.rank_truncated is False
    assert source_rank.used_svd_fallback is False
    assert source.terminal_curvature.requested_source == "observed"
    assert source.terminal_curvature.actual_source == "observed"
    assert source.terminal_curvature.fallback_count == 0

    terminal_correction = np.asarray(source.solve_terminal(source.terminal_score))
    retained_score = efs_module._endpoint_retained_score(source)
    retained_curvature = efs_module._endpoint_retained_curvature(source)
    retained_correction = face.reduce_vector(terminal_correction)
    relative_residual = float(
        np.linalg.norm(retained_curvature @ retained_correction - retained_score, ord=2)
        / max(1.0, float(np.linalg.norm(retained_score, ord=2)))
    )
    epsilon = float(np.finfo(source.coefficients.dtype).eps)
    candidate = face.project(source.coefficients + terminal_correction)
    assert relative_residual <= strict_line_search_config.residual_tolerance
    assert efs_module._endpoint_positive_dot(
        retained_score,
        retained_correction,
        epsilon=epsilon,
    )
    assert not np.array_equal(candidate, source.coefficients)
    source_objective = source.penalized_optimizing_log_likelihood
    assert source_objective == -5.0
    directional_derivative = float(source.terminal_score @ terminal_correction)
    rejection_bound = efs_module._endpoint_objective_accumulation_bound(source, source)
    assert rejection_bound is not None
    for alpha, trial_objective in ((1.0, -1.0), (0.5, -2.0)):
        required_objective = (
            source_objective
            + strict_line_search_config.armijo_constant * alpha * directional_derivative
        )
        assert required_objective - trial_objective > rejection_bound

    polished = efs_module._fit_endpoint_authority_stationary(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=initial,
        config=strict_line_search_config,
        chunk_size=None,
        phase_recorder=None,
    )
    polished_kkt = efs_module._endpoint_retained_kkt_relative(polished)
    polished_rank = efs_module._endpoint_retained_rank(polished)

    assert source.converged is False
    assert source.convergence_reason == "line_search_failed"
    assert source.iterations == 0
    assert polished.converged is True
    assert polished.convergence_reason == "score"
    assert polished.iterations == 0
    assert polished_kkt <= strict_line_search_config.tolerance
    assert polished_kkt < source_kkt
    assert not np.array_equal(polished.coefficients, source.coefficients)
    assert polished.config == source.config
    assert polished.coefficient_face is source.coefficient_face is face
    assert np.array_equal(polished.penalty, source.penalty)
    assert (
        polished.family_likelihood_plan_identifier
        == source.family_likelihood_plan_identifier
        == likelihood_plan.plan_identifier
    )
    assert polished.resolved_chunk_size == source.resolved_chunk_size
    assert polished.execution_backend_identifier == source.execution_backend_identifier
    assert polished_rank is not None
    assert polished_rank.rank == source_rank.rank == face.reduced_width
    assert efs_module._endpoint_retained_rank_provenance(
        polished
    ) == efs_module._endpoint_retained_rank_provenance(source)
    assert (
        polished.terminal_curvature.requested_source,
        polished.terminal_curvature.actual_source,
        polished.terminal_curvature.fallback_count,
    ) == (
        source.terminal_curvature.requested_source,
        source.terminal_curvature.actual_source,
        source.terminal_curvature.fallback_count,
    )

    coefficient_bound = efs_module._endpoint_candidate_refit_bound(
        candidate,
        polished,
        tolerance=strict_line_search_config.tolerance,
    )
    coefficient_movement = float(np.max(np.abs(polished.coefficients - candidate), initial=0.0))
    assert coefficient_bound is not None
    assert coefficient_movement <= coefficient_bound
    objective_bound = efs_module._endpoint_objective_accumulation_bound(source, polished)
    polished_objective = polished.penalized_optimizing_log_likelihood
    assert objective_bound is not None
    assert source_objective is not None and polished_objective is not None
    assert polished_objective >= source_objective - objective_bound


def test_endpoint_authority_score_only_dispatch_has_exact_failure_metadata_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills broadening, suppressing, or moving the private score-only refit."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        strict_line_search_config,
        _source,
    ) = _zero_step_strict_armijo_face_problem()
    real_fit_fixed_state = efs_module._fit_fixed_state
    source_overrides = {
        "eligible": {},
        "wrong_reason": {"convergence_reason": "max_iterations"},
        "nonzero_iterations": {"iterations": 1},
        "already_converged": {
            "converged": True,
            "convergence_reason": "objective_and_step",
        },
    }
    modes_by_source: dict[str, tuple[bool, ...]] = {}
    returned_source_by_case: dict[str, bool] = {}
    source_metadata_by_case: dict[str, tuple[bool, str, int]] = {}

    for case, overrides in source_overrides.items():
        modes: list[bool] = []
        sources: list[DenseSolverResult] = []

        def record_mode(
            *args: object,
            score_only: bool = False,
            **kwargs: object,
        ) -> DenseSolverResult:
            modes.append(score_only)
            if score_only:
                kwargs["score_only"] = True
            result = real_fit_fixed_state(*args, **kwargs)  # type: ignore[arg-type]
            if len(modes) == 1:
                result = replace(result, **overrides)
                sources.append(result)
            return result

        monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", record_mode)
        monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", record_mode)
        monkeypatch.setattr(efs_module, "_fit_fixed_state", record_mode)
        result = efs_module._fit_endpoint_authority_stationary(
            family,  # type: ignore[arg-type]
            layout,
            response,
            likelihood_plan,
            lambdas=lambdas,
            face=face,
            initial=initial,
            config=strict_line_search_config,
            chunk_size=None,
            phase_recorder=None,
        )
        source = sources[0]
        modes_by_source[case] = tuple(modes)
        returned_source_by_case[case] = result is source
        source_metadata_by_case[case] = (
            source.converged,
            source.convergence_reason,
            source.iterations,
        )

        if result is not source:
            source_kkt = efs_module._endpoint_retained_kkt_relative(source)
            result_kkt = efs_module._endpoint_retained_kkt_relative(result)
            assert result.converged is True
            assert result_kkt <= strict_line_search_config.tolerance
            assert result_kkt < source_kkt
            assert efs_module._endpoint_polish_provenance_matches(
                source,
                result,
                config=strict_line_search_config,
                face=face,
            )

    assert modes_by_source == {
        "eligible": (False, True),
        "wrong_reason": (False,),
        "nonzero_iterations": (False,),
        "already_converged": (False, True),
    }
    assert returned_source_by_case == {
        "eligible": False,
        "wrong_reason": True,
        "nonzero_iterations": True,
        "already_converged": False,
    }
    assert source_metadata_by_case == {
        "eligible": (False, "line_search_failed", 0),
        "wrong_reason": (False, "max_iterations", 0),
        "nonzero_iterations": (False, "line_search_failed", 1),
        "already_converged": (True, "objective_and_step", 0),
    }


@pytest.mark.parametrize(
    ("failure_reason", "iterations"),
    [("max_iterations", 0), ("line_search_failed", 1)],
    ids=("wrong_reason", "accepted_iteration"),
)
def test_endpoint_authority_polish_requires_exact_zero_step_failure_metadata(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: str,
    iterations: int,
) -> None:
    """Kills broadening either nonconverged-source eligibility condition."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        strict_line_search_config,
        source,
    ) = _zero_step_strict_armijo_face_problem()
    ineligible_source = replace(
        source,
        convergence_reason=failure_reason,
        iterations=iterations,
    )
    fit_calls = 0

    def return_only_source(*args: object, **kwargs: object) -> DenseSolverResult:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        if fit_calls > 1:
            pytest.fail("ineligible source reached the terminal polish refit")
        return ineligible_source

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", return_only_source)
    monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", return_only_source)
    monkeypatch.setattr(efs_module, "_fit_fixed_state", return_only_source)
    result = efs_module._fit_endpoint_authority_stationary(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=initial,
        config=strict_line_search_config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert ineligible_source.converged is False
    assert (
        efs_module._endpoint_retained_kkt_relative(ineligible_source)
        > strict_line_search_config.tolerance
    )
    assert result is ineligible_source
    assert fit_calls == 1


def test_endpoint_authority_polish_skips_a_stationary_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills entering terminal polish after the real retained KKT already passes."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        _initial,
        strict_line_search_config,
        source,
    ) = _zero_step_strict_armijo_face_problem()
    candidate = face.project(source.coefficients + source.solve_terminal(source.terminal_score))
    stationary = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(efs_module._penalty_lambdas(lambdas, face)),
        initial=candidate,
        config=strict_line_search_config,
        coefficient_face=face,
    )
    fit_calls = 0

    def return_stationary(*args: object, **kwargs: object) -> DenseSolverResult:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return stationary

    real_solve_terminal = DenseSolverResult.solve_terminal

    def forbid_stationary_solve(
        result: DenseSolverResult,
        rhs: np.ndarray,
    ) -> np.ndarray:
        if result is stationary:
            pytest.fail("stationary source reached the terminal correction solve")
        return real_solve_terminal(result, rhs)

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", return_stationary)
    monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", return_stationary)
    monkeypatch.setattr(efs_module, "_fit_fixed_state", return_stationary)
    monkeypatch.setattr(DenseSolverResult, "solve_terminal", forbid_stationary_solve)
    result = efs_module._fit_endpoint_authority_stationary(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=candidate,
        config=strict_line_search_config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert stationary.converged is True
    assert efs_module._endpoint_retained_kkt_relative(stationary) <= (
        strict_line_search_config.tolerance
    )
    assert result is stationary
    assert fit_calls == 1


def test_endpoint_authority_polish_refuses_a_nonconverged_fresh_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills accepting a stationary candidate whose fresh refit did not converge."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        face,
        initial,
        strict_line_search_config,
        source,
    ) = _zero_step_strict_armijo_face_problem()
    correction = source.solve_terminal(source.terminal_score)
    candidate = face.project(source.coefficients + correction)
    real_polished = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(efs_module._penalty_lambdas(lambdas, face)),
        initial=candidate,
        config=strict_line_search_config,
        coefficient_face=face,
    )
    refused_polished = replace(
        real_polished,
        converged=False,
        convergence_reason="max_iterations",
    )
    outputs = iter((source, refused_polished))
    fit_calls = 0

    def return_results(*args: object, **kwargs: object) -> DenseSolverResult:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return next(outputs)

    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", return_results)
    monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", return_results)
    monkeypatch.setattr(efs_module, "_fit_fixed_state", return_results)
    result = efs_module._fit_endpoint_authority_stationary(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=initial,
        config=strict_line_search_config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert real_polished.converged is True
    assert refused_polished.converged is False
    assert efs_module._endpoint_retained_kkt_relative(refused_polished) <= (
        strict_line_search_config.tolerance
    )
    assert result is source
    assert fit_calls == 2


def test_nonstationary_sole_cap_can_continue_to_a_strict_exact_face() -> None:
    """Kills returning cap_not_stationary before assessing a sound exact face."""
    (
        family,
        layout,
        response,
        likelihood_plan,
        lambdas,
        capped_name,
        finite_name,
        solver_config,
        smoothing,
    ) = _gamma_isolated_cap_face_problem()
    assessment = smoothing.history[0]
    cap_fit = smoothing.coefficient_fits[assessment.coefficient_fit_indices[0]]
    cap_kkt = efs_module._endpoint_retained_kkt_relative(cap_fit)
    assert cap_fit.converged is True
    assert cap_fit.convergence_reason == "objective_and_step"
    assert cap_kkt > cap_fit.config.tolerance
    assert lambdas[finite_name] < smoothing.config.maximum_lambda

    face = build_penalty_face(layout, (capped_name,))
    authority_config = efs_module._face_authority_config(solver_config)
    endpoint_fit = efs_module._fit_endpoint_authority_stationary(
        family,
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        face=face,
        initial=cap_fit.coefficients,
        config=authority_config,
        chunk_size=None,
        phase_recorder=None,
    )
    endpoint_rank = endpoint_fit.terminal_reduced_rank
    assert endpoint_fit.converged is True
    assert efs_module._endpoint_retained_kkt_relative(endpoint_fit) <= authority_config.tolerance
    assert endpoint_rank is not None
    assert endpoint_rank.rank == face.reduced_width
    assert endpoint_fit.terminal_curvature.requested_source == "observed"
    assert endpoint_fit.terminal_curvature.actual_source == "observed"
    assert endpoint_fit.terminal_curvature.fallback_count == 0
    assert efs_module._endpoint_shared_provenance(endpoint_fit) == (
        efs_module._endpoint_shared_provenance(cap_fit)
    )
    endpoint_objective = evaluate_endpoint_laplace(
        endpoint_fit,
        layout=layout,
        lambdas=lambdas,
        face=face,
    ).objective
    objective_ceiling = assessment.objective_before + smoothing.config.objective_tolerance * (
        1.0 + abs(assessment.objective_before)
    )
    assert endpoint_objective <= objective_ceiling
    direction = evaluate_endpoint_laplace_derivative(
        family,
        layout,
        response,
        likelihood_plan,
        lambdas=lambdas,
        component_name=capped_name,
        finite_face=None,
        endpoint_face=face,
        endpoint_fit=endpoint_fit,
    )
    assert direction.decision == "endpoint"
    assert direction.lower_bound > 0.0

    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.terminal_fit.coefficient_face.component_names == (capped_name,)
    face_events = tuple(
        item
        for item in smoothing.history
        if item.activated_face_components or item.revalidated_face_components
    )
    assert tuple(item.activated_face_components for item in face_events) == (
        (capped_name,),
        (),
    )
    assert tuple(item.revalidated_face_components for item in face_events) == (
        (),
        (capped_name,),
    )
    for item in face_events:
        cap_index, endpoint_index = item.coefficient_fit_indices
        recorded_cap = smoothing.coefficient_fits[cap_index]
        recorded_endpoint = smoothing.coefficient_fits[endpoint_index]
        recorded_rank = recorded_endpoint.terminal_reduced_rank
        assert recorded_cap.converged is True
        assert (
            efs_module._endpoint_retained_kkt_relative(recorded_cap)
            > item.coefficient_tolerances[0]
        )
        assert recorded_endpoint.converged is True
        assert (
            efs_module._endpoint_retained_kkt_relative(recorded_endpoint)
            <= item.coefficient_tolerances[1]
        )
        assert recorded_rank is not None
        assert recorded_rank.rank == face.reduced_width
        assert recorded_endpoint.terminal_curvature.actual_source == "observed"
        assert recorded_endpoint.terminal_curvature.fallback_count == 0
        assert item.endpoint_direction_evidence is not None
        assert item.endpoint_direction_evidence.lower_bound > 0.0
        assert item.accepted_fit_index == endpoint_index
        objective_ceiling = item.objective_before + smoothing.config.objective_tolerance * (
            1.0 + abs(item.objective_before)
        )
        assert item.objective_after <= objective_ceiling
    assert tuple(
        index for item in smoothing.history for index in item.coefficient_fit_indices
    ) == tuple(range(1, len(smoothing.coefficient_fits)))
    assert smoothing.matched_certified is False


def test_direct_face_result_rejects_a_forged_capped_companion() -> None:
    """Kills authenticating scalar direct-face authority with a second capped name."""
    *_, finite_name, _solver_config, smoothing = _gamma_isolated_cap_face_problem()
    maximum_lambda = smoothing.config.maximum_lambda

    def forge_companion(values):
        return {
            name: maximum_lambda if name == finite_name else value for name, value in values.items()
        }

    forged_history = tuple(
        replace(
            item,
            lambdas_before=forge_companion(item.lambdas_before),
            proposed_lambdas=forge_companion(item.proposed_lambdas),
            lambdas_after=forge_companion(item.lambdas_after),
        )
        for item in smoothing.history
    )
    with pytest.raises(ValueError, match="one sole capped component"):
        replace(
            smoothing,
            initial_lambdas=forge_companion(smoothing.initial_lambdas),
            lambdas=forge_companion(smoothing.lambdas),
            history=forged_history,
        )


def _unindexed_endpoint_direction(
    *,
    endpoint_objective: float = -12.5,
    decision: str = "endpoint",
) -> EndpointDirectionEvidence:
    derivative = {"endpoint": 2.0, "finite": -2.0, "unresolved": 0.0}[decision]
    numerical_error = 0.25
    return EndpointDirectionEvidence(
        authority_identifier="analytic-observed-curvature-direction/v1",
        decision=decision,  # type: ignore[arg-type]
        endpoint_objective=endpoint_objective,
        analytic_derivative=derivative,
        profile_score_term=2.0 * derivative,
        curvature_schur_term=0.0,
        curvature_drift_term=0.0,
        numerical_error=numerical_error,
        lower_bound=derivative - numerical_error,
        upper_bound=derivative + numerical_error,
    )


def test_joint_endpoint_direction_evidence_normalizes_an_unindexed_receipt() -> None:
    first = _unindexed_endpoint_direction()
    second = _unindexed_endpoint_direction()

    joint = JointEndpointDirectionEvidence(
        authority_identifier="joint-analytic-observed-curvature-direction/v1",
        component_directions=(("mean:first#0", first), ("mean:second#0", second)),
    )

    assert joint.component_names == ("mean:first#0", "mean:second#0")
    assert joint.endpoint_objective == -12.5
    assert joint.fit_indices == ()
    assert joint.decision == "endpoint"
    assert joint._derived_authority_matches()
    with pytest.raises(FrozenInstanceError):
        joint.endpoint_fit_index = 3  # type: ignore[misc]


def test_joint_endpoint_direction_evidence_normalizes_an_ordered_list() -> None:
    first = _unindexed_endpoint_direction()
    second = _unindexed_endpoint_direction()

    joint = JointEndpointDirectionEvidence(
        authority_identifier="joint-analytic-observed-curvature-direction/v1",
        component_directions=[  # type: ignore[arg-type]
            ("mean:first#0", first),
            ("mean:second#0", second),
        ],
    )

    assert joint.component_directions == (
        ("mean:first#0", first),
        ("mean:second#0", second),
    )


@pytest.mark.parametrize("collection_kind", ["set", "frozenset", "generator"])
def test_joint_endpoint_direction_evidence_requires_an_ordered_component_sequence(
    collection_kind: str,
) -> None:
    pairs = (
        ("mean:first#0", _unindexed_endpoint_direction()),
        ("mean:second#0", _unindexed_endpoint_direction()),
    )
    if collection_kind == "set":
        component_directions = set(pairs)
    elif collection_kind == "frozenset":
        component_directions = frozenset(pairs)
    else:
        component_directions = iter(pairs)

    with pytest.raises(TypeError, match="ordered sequence"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=component_directions,  # type: ignore[arg-type]
        )


def test_joint_endpoint_direction_evidence_rejects_incomplete_or_ambiguous_directions() -> None:
    first = _unindexed_endpoint_direction()
    second = _unindexed_endpoint_direction()

    with pytest.raises(ValueError, match="at least two"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=(("mean:first#0", first),),
        )
    with pytest.raises(ValueError, match="unique"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=(("mean:first#0", first), ("mean:first#0", second)),
        )
    with pytest.raises(ValueError, match="unindexed"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=(
                (
                    "mean:first#0",
                    replace(
                        first,
                        fit_indices=(2, 3),
                        coefficient_tolerance=1.0e-12,
                    ),
                ),
                ("mean:second#0", second),
            ),
        )
    with pytest.raises(ValueError, match="same endpoint objective"):
        JointEndpointDirectionEvidence(
            authority_identifier="joint-analytic-observed-curvature-direction/v1",
            component_directions=(
                ("mean:first#0", first),
                (
                    "mean:second#0",
                    replace(
                        second,
                        endpoint_objective=np.nextafter(
                            second.endpoint_objective,
                            np.inf,
                        ),
                    ),
                ),
            ),
        )


def test_joint_endpoint_direction_evidence_requires_an_index_and_tolerance_together() -> None:
    joint = JointEndpointDirectionEvidence(
        authority_identifier="joint-analytic-observed-curvature-direction/v1",
        component_directions=(
            ("mean:first#0", _unindexed_endpoint_direction()),
            ("mean:second#0", _unindexed_endpoint_direction()),
        ),
    )

    with pytest.raises(ValueError, match="requires a coefficient tolerance"):
        replace(joint, endpoint_fit_index=3)
    with pytest.raises(ValueError, match="cannot carry a tolerance"):
        replace(joint, coefficient_tolerance=1.0e-12)


def _joint_endpoint_iteration(
    *,
    event: str,
    decisions: tuple[str, ...] = ("endpoint", "endpoint"),
    receipt_component_indices: tuple[int, ...] | None = None,
    indexed: bool = True,
    receipt_fit_index: int | None = None,
    receipt_tolerance: float = 1.0e-12,
    event_fit_indices: tuple[int, ...] | None = None,
    event_tolerances: tuple[float, ...] | None = None,
    scalar_evidence: bool = False,
    failure_reason: str | None = None,
):
    first_name, smoothing = _scalar_efs_fit(0.5)
    source = smoothing.history[0]
    assert source.accepted_fit_index is not None
    component_names = tuple(
        [first_name] + [f"mean:joint-{index}#wiggle" for index in range(1, len(decisions))]
    )

    mapping_updates: dict[str, dict[str, float]] = {}
    for field_name in (
        "lambdas_before",
        "proposed_lambdas",
        "lambdas_after",
        "proposed_log_steps",
        "accepted_log_steps",
        "quadratic_forms",
        "trace_terms",
    ):
        source_mapping = getattr(source, field_name)
        first_value = source_mapping[first_name]
        mapping_updates[field_name] = {name: first_value for name in component_names}

    direction = None
    if failure_reason is None:
        if scalar_evidence:
            direction = source.endpoint_direction_evidence
        else:
            named_components = (
                component_names
                if receipt_component_indices is None
                else tuple(component_names[index] for index in receipt_component_indices)
            )
            named_directions = tuple(
                (
                    name,
                    _unindexed_endpoint_direction(
                        endpoint_objective=source.objective_after,
                        decision=decision,
                    ),
                )
                for name, decision in zip(
                    named_components,
                    decisions[: len(named_components)],
                    strict=True,
                )
            )
            endpoint_fit_index = (
                source.accepted_fit_index if receipt_fit_index is None else receipt_fit_index
            )
            direction = JointEndpointDirectionEvidence(
                authority_identifier="joint-analytic-observed-curvature-direction/v1",
                component_directions=named_directions,
                endpoint_fit_index=endpoint_fit_index if indexed else None,
                coefficient_tolerance=receipt_tolerance if indexed else None,
            )

    event_fields = {
        "activated_face_components": (),
        "deactivated_face_components": (),
        "revalidated_face_components": (),
        "refused_face_components": (),
    }
    event_fields[f"{event}_face_components"] = component_names
    accepted = event != "refused"
    coefficient_fit_indices = (
        (source.accepted_fit_index,) if event_fit_indices is None else event_fit_indices
    )
    coefficient_tolerances = (receipt_tolerance,) if event_tolerances is None else event_tolerances
    iteration = replace(
        source,
        **mapping_updates,
        **event_fields,
        accepted=accepted,
        accepted_fit_index=source.accepted_fit_index if accepted else None,
        accepted_curvature=source.accepted_curvature if accepted else None,
        coefficient_fit_indices=coefficient_fit_indices,
        coefficient_tolerances=coefficient_tolerances,
        endpoint_direction_evidence=direction,
        endpoint_assessment_failure_reason=failure_reason,
    )
    return component_names, iteration


@pytest.mark.parametrize("event", ["activated", "revalidated"])
def test_joint_endpoint_direction_evidence_iteration_accepts_one_atomic_event(
    event: str,
) -> None:
    component_names, iteration = _joint_endpoint_iteration(event=event)

    assert getattr(iteration, f"{event}_face_components") == component_names
    assert isinstance(
        iteration.endpoint_direction_evidence,
        JointEndpointDirectionEvidence,
    )
    assert iteration.endpoint_direction_evidence.component_names == component_names
    assert iteration.endpoint_direction_evidence.fit_indices == (iteration.accepted_fit_index,)


@pytest.mark.parametrize("event", ["activated", "revalidated"])
def test_joint_endpoint_direction_evidence_iteration_rejects_scalar_or_mismatched_receipts(
    event: str,
) -> None:
    with pytest.raises(TypeError, match="joint endpoint direction evidence"):
        _joint_endpoint_iteration(event=event, scalar_evidence=True)
    with pytest.raises(ValueError, match="exactly match"):
        _joint_endpoint_iteration(
            event=event,
            receipt_component_indices=(1, 0),
        )


def test_joint_endpoint_direction_evidence_iteration_rejects_partial_component_coverage() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        _joint_endpoint_iteration(
            event="activated",
            decisions=("endpoint", "endpoint", "endpoint"),
            receipt_component_indices=(0, 1),
        )


@pytest.mark.parametrize("event", ["activated", "revalidated"])
def test_joint_endpoint_direction_evidence_iteration_rejects_nonendpoint_acceptance(
    event: str,
) -> None:
    with pytest.raises(ValueError, match="resolved endpoint evidence"):
        _joint_endpoint_iteration(
            event=event,
            decisions=("endpoint", "finite"),
        )


@pytest.mark.parametrize("decision", ["finite", "unresolved"])
def test_joint_endpoint_direction_evidence_iteration_accepts_a_resolved_joint_refusal(
    decision: str,
) -> None:
    component_names, iteration = _joint_endpoint_iteration(
        event="refused",
        decisions=("endpoint", decision),
    )

    assert iteration.refused_face_components == component_names
    assert isinstance(
        iteration.endpoint_direction_evidence,
        JointEndpointDirectionEvidence,
    )
    assert iteration.endpoint_direction_evidence.decision == decision


def test_joint_endpoint_direction_evidence_iteration_rejects_an_all_endpoint_refusal() -> None:
    with pytest.raises(ValueError, match="at least one non-endpoint"):
        _joint_endpoint_iteration(event="refused")


@pytest.mark.parametrize(
    "failure_reason",
    [
        "joint_endpoint_not_converged",
        "joint_endpoint_not_stationary",
        "joint_analytic_unavailable",
        "joint_objective_rejected",
    ],
)
def test_joint_endpoint_direction_evidence_iteration_accepts_a_typed_joint_failure(
    failure_reason: str,
) -> None:
    component_names, iteration = _joint_endpoint_iteration(
        event="refused",
        failure_reason=failure_reason,
    )

    assert iteration.refused_face_components == component_names
    assert iteration.endpoint_direction_evidence is None
    assert iteration.endpoint_assessment_failure_reason == failure_reason


def test_joint_endpoint_direction_evidence_iteration_rejects_a_scalar_failure_reason() -> None:
    with pytest.raises(ValueError, match="joint failure reason"):
        _joint_endpoint_iteration(
            event="refused",
            failure_reason="endpoint_not_converged",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"indexed": False}, "indexed strict endpoint fit"),
        ({"receipt_fit_index": 1}, "indexed strict endpoint fit"),
        (
            {
                "event_fit_indices": (1, 2),
                "event_tolerances": (1.0e-12, 1.0e-12),
            },
            "indexed strict endpoint fit",
        ),
        (
            {
                "receipt_tolerance": 1.0e-11,
                "event_tolerances": (1.0e-11,),
            },
            "strict coefficient tolerance",
        ),
    ],
)
def test_joint_endpoint_direction_evidence_iteration_requires_one_strict_indexed_fit(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _joint_endpoint_iteration(event="activated", **overrides)


def test_efs_certifies_an_exact_face_by_finite_differences_without_directional_curvature() -> None:
    """A family without the analytic protocol certifies through finite differences.

    The endpoint derivative differences the family's own order-two curvature,
    labels the evidence with the finite-difference authority, and reports the
    exact face without claiming ``matched_certified``.
    """
    component_name, smoothing = _scalar_efs_fit(
        0.5,
        family=_UnitGaussianWithoutCurvatureDirection(),
    )
    _analytic_name, analytic_smoothing = _scalar_efs_fit(0.5)

    assert smoothing.converged is True
    assert smoothing.convergence_reason == "lambda_change"
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.terminal_fit.coefficient_face.component_names == (component_name,)
    evidence = smoothing.terminal_endpoint_directions[component_name]
    assert evidence.authority_identifier == "finite-difference-curvature-direction/v1"
    assert evidence.decision == "endpoint"
    analytic_evidence = analytic_smoothing.terminal_endpoint_directions[component_name]
    assert analytic_evidence.authority_identifier == "analytic-observed-curvature-direction/v1"
    np.testing.assert_allclose(
        evidence.analytic_derivative,
        analytic_evidence.analytic_derivative,
        atol=2e-14,
    )
    assert all(
        item.endpoint_assessment_failure_reason is None and not item.refused_face_components
        for item in smoothing.history
    )
    assert smoothing.matched_certified is False
    with pytest.raises(RuntimeError, match="not certified"):
        smoothing.assert_matched_certified()


def test_observed_newton_decrement_stops_at_a_resolved_quadratic_gap() -> None:
    """Kills requiring a raw score after the remaining objective gap is certified."""
    family = _UnitGaussian()
    response = np.array([1.0])
    weights = resolved_prior(np.ones(1))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component_name = layout.penalty_names[0]
    lambda_value = 1.0e8
    optimum = 1.0 / (1.0 + lambda_value)
    initial = np.array([optimum + 1.0e-8])

    result = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),  # type: ignore[arg-type]
        layout.penalty_matrix({component_name: lambda_value}),
        initial=initial,
        config=DenseSolverConfig(
            max_iterations=10,
            tolerance=1.0e-8,
            newton_decrement_tolerance=1.0e-8,
            coefficient_curvature="observed",
        ),
    )

    assert result.converged is True
    assert result.convergence_reason == "newton_decrement"
    assert result.iterations == 0
    np.testing.assert_array_equal(result.coefficients, initial)


def test_newton_decrement_result_revalidates_its_terminal_certificate() -> None:
    """Kills publishing the reason after its score or curvature provenance changed."""
    family = _UnitGaussian()
    response = np.array([1.0])
    weights = resolved_prior(np.ones(1))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component_name = layout.penalty_names[0]
    lambda_value = 1.0e8
    optimum = 1.0 / (1.0 + lambda_value)
    result = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),  # type: ignore[arg-type]
        layout.penalty_matrix({component_name: lambda_value}),
        initial=np.array([optimum + 1.0e-8]),
        config=DenseSolverConfig(
            max_iterations=10,
            tolerance=1.0e-8,
            newton_decrement_tolerance=1.0e-8,
            coefficient_curvature="observed",
        ),
    )
    assert result.convergence_reason == "newton_decrement"

    with pytest.raises(ValueError, match="Newton decrement certificate"):
        replace(result, terminal_score=1.0e8 * result.terminal_score)

    fallback = replace(
        result.terminal_curvature,
        actual_source="fisher",
        reason="test mutation",
        fallback_count=1,
    )
    with pytest.raises(ValueError, match="observed terminal curvature"):
        replace(result, terminal_curvature=fallback)


def test_endpoint_authority_refits_after_a_loose_newton_decrement_stop() -> None:
    """A parent decrement-only stop cannot supply endpoint direction authority."""
    family = _UnitGaussian()
    response = np.array([0.5], dtype=np.float64)
    weights = resolved_prior(np.ones(1, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["only"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component_name = layout.penalty_names[0]
    endpoint_face = build_penalty_face(layout, (component_name,))
    lambdas = {component_name: 10.0}
    initial = np.array([0.25], dtype=np.float64)
    parent_config = DenseSolverConfig(
        max_iterations=10,
        tolerance=1.0e-4,
        newton_decrement_tolerance=1.0,
        coefficient_curvature="observed",
    )
    likelihood_plan = family.bind_likelihood(
        response,
        weights,
        COMPLETE_OBSERVATION,
    )

    loose_fit = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        layout.penalty_matrix(lambdas),
        initial=initial,
        config=parent_config,
    )
    assert loose_fit.convergence_reason == "newton_decrement"
    assert loose_fit.iterations == 0
    np.testing.assert_array_equal(loose_fit.coefficients, initial)

    attempt = efs_module._check_face_direction(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        lambdas=lambdas,
        component_name=component_name,
        finite_face=None,
        endpoint_face=endpoint_face,
        initial=loose_fit.coefficients,
        solver_config=parent_config,
        chunk_size=None,
        phase_recorder=None,
    )

    assert attempt.check is not None
    assert attempt.check.direction.decision == "endpoint"
    assert all(
        fit.config.coefficient_curvature == "observed"
        and fit.config.tolerance == 1.0e-12
        and fit.config.newton_decrement_tolerance is None
        and fit.convergence_reason != "newton_decrement"
        for fit in attempt.assessment_fits
    )
    np.testing.assert_allclose(
        attempt.assessment_fits[0].coefficients,
        np.array([0.5 / 11.0]),
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )


def _dense_component(
    name: str,
    group_name: str,
    group_index: int,
    group_sl: slice,
    matrix: np.ndarray,
    *,
    rank: float,
) -> PenaltyComponent:
    eigenvalues = np.linalg.eigvalsh(matrix)
    return PenaltyComponent(
        name=name,
        group_name=group_name,
        group_index=group_index,
        group_sl=group_sl,
        omega_raw=matrix,
        omega_ssp=matrix,
        rank=rank,
        eigvals_omega=eigenvalues[eigenvalues > 0.0],
    )


def _independent_penalty_efs_fit(
    monkeypatch: pytest.MonkeyPatch,
    response: tuple[float, ...],
    *,
    maximum_lambda: float = 10.0,
    max_iterations: int = 8,
    starting_lambdas: tuple[float, ...] | None = None,
    overlap: bool = False,
):
    family = _UnitGaussian()
    penalty_labels = ("first", "second", "third")[: len(response)]
    column_labels = (*penalty_labels, "retained")
    response_array = np.asarray((*response, 0.0), dtype=np.float64)
    weights = resolved_prior(np.ones(len(response_array), dtype=np.float64))
    identity = np.eye(len(response_array), dtype=np.float64)
    frame = as_eager_frame(
        pd.DataFrame({name: identity[:, index] for index, name in enumerate(column_labels)})
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "mean",
                    {name: Numeric() for name in column_labels},
                    intercept=False,
                ),
            ),
        )
    )
    components = tuple(
        _dense_component(
            f"{group_name}#identity",
            group_name,
            group_index,
            (
                layout.term_slices["mean:first"]
                if overlap and group_name == "mean:second"
                else layout.term_slices[group_name]
            ),
            np.ones((1, 1), dtype=np.float64),
            rank=1.0,
        )
        for group_index, group_name in enumerate(tuple(f"mean:{label}" for label in penalty_labels))
    )
    layout = replace(layout, penalties=components)
    initial_values = (
        (maximum_lambda,) * len(layout.penalty_names)
        if starting_lambdas is None
        else starting_lambdas
    )
    authority_faces: dict[int, tuple[str, ...]] = {}
    real_fit = efs_module._fit_endpoint_authority_stationary

    def record_authority_face(*args: object, **kwargs: object) -> DenseSolverResult:
        fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        face = kwargs["face"]
        assert face is None or isinstance(face, PenaltyFace)
        authority_faces[id(fit)] = () if face is None else face.component_names
        return fit

    monkeypatch.setattr(
        smoothing_faces, "_fit_endpoint_authority_stationary", record_authority_face
    )
    monkeypatch.setattr(smoothing_loop, "_fit_endpoint_authority_stationary", record_authority_face)
    monkeypatch.setattr(efs_module, "_fit_endpoint_authority_stationary", record_authority_face)
    smoothing = fit_distributional_efs(
        family,  # type: ignore[arg-type]
        layout,
        response_array,
        family.bind_likelihood(  # type: ignore[arg-type]
            response_array,
            weights,
            COMPLETE_OBSERVATION,
        ),
        lambdas=dict(zip(layout.penalty_names, initial_values, strict=True)),
        solver_config=DenseSolverConfig(max_iterations=50, tolerance=1.0e-12),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=max_iterations,
            tolerance=1.0e-8,
            initial_lambda=min(0.1, maximum_lambda),
            maximum_lambda=maximum_lambda,
        ),
    )
    return layout, smoothing, authority_faces


def test_efs_activates_two_independent_cap_components_as_one_joint_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
    )

    transition = next(item for item in smoothing.history if item.activated_face_components)
    assert transition.activated_face_components == layout.penalty_names
    assert len(transition.coefficient_fit_indices) == 1
    assert transition.accepted_fit_index == transition.coefficient_fit_indices[0]
    assert isinstance(transition.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    assert tuple(
        authority_faces[id(smoothing.coefficient_fits[index])]
        for index in transition.coefficient_fit_indices
    ) == (layout.penalty_names,)


def test_efs_revalidates_a_joint_face_from_one_fresh_common_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
    )

    activation = next(item for item in smoothing.history if item.activated_face_components)
    revalidations = tuple(item for item in smoothing.history if item.revalidated_face_components)
    assessment_shape = tuple(
        (
            item.revalidated_face_components,
            tuple(
                authority_faces[id(smoothing.coefficient_fits[index])]
                for index in item.coefficient_fit_indices
            ),
        )
        for item in revalidations
    )
    assert assessment_shape == ((layout.penalty_names, (layout.penalty_names,)),)

    revalidation = revalidations[0]
    assert smoothing.converged is True
    assert revalidation.accepted_fit_index == revalidation.coefficient_fit_indices[0]
    assert revalidation.accepted_fit_index != activation.accepted_fit_index
    assert smoothing.terminal_fit_index == revalidation.accepted_fit_index
    common_fit = smoothing.coefficient_fits[revalidation.accepted_fit_index]
    assert common_fit is not smoothing.coefficient_fits[activation.accepted_fit_index]
    assert tuple(
        fit_id
        for fit_id, face_names in authority_faces.items()
        if face_names == layout.penalty_names
    ) == (
        id(smoothing.coefficient_fits[activation.accepted_fit_index]),
        id(common_fit),
    )
    assert common_fit.config.tolerance == 1.0e-12
    assert common_fit.config.coefficient_curvature == "observed"
    assert common_fit.config.newton_decrement_tolerance is None
    assert common_fit.terminal_curvature.actual_source == "observed"
    assert common_fit.terminal_curvature.fallback_count == 0
    direction = revalidation.endpoint_direction_evidence
    assert isinstance(direction, JointEndpointDirectionEvidence)
    assert direction.fit_indices == revalidation.coefficient_fit_indices
    assert all(
        component_direction.decision == "endpoint" and component_direction.lower_bound > 0.0
        for _name, component_direction in direction.component_directions
    )
    assert dict(smoothing.terminal_endpoint_directions) == dict(direction.component_directions)
    assert all(not item.fit_indices for item in smoothing.terminal_endpoint_directions.values())
    assert smoothing.matched_certified is False
    with pytest.raises(RuntimeError, match="exact coefficient face"):
        smoothing.assert_matched_certified()


def test_joint_terminal_revalidation_consumes_one_outer_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
        max_iterations=2,
    )

    assert smoothing.converged is True
    assert len(smoothing.history) == 2
    assert len(smoothing.history[-1].revalidated_face_components) == 2


def test_efs_rolls_back_the_complete_joint_face_when_the_second_recheck_is_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction_calls = 0
    real_resolve = efs_module.resolve_endpoint_direction

    def make_second_terminal_direction_finite(*args: object, **kwargs: object):
        nonlocal direction_calls
        direction_calls += 1
        direction = real_resolve(*args, **kwargs)  # type: ignore[arg-type]
        if direction_calls != 5:
            return direction
        derivative = -(direction.numerical_error + 1.0)
        return replace(
            direction,
            decision="finite",
            analytic_derivative=derivative,
            profile_score_term=2.0 * derivative,
            curvature_schur_term=0.0,
            curvature_drift_term=0.0,
            lower_bound=derivative - direction.numerical_error,
            upper_bound=derivative + direction.numerical_error,
        )

    monkeypatch.setattr(
        smoothing_faces, "resolve_endpoint_direction", make_second_terminal_direction_finite
    )
    monkeypatch.setattr(
        efs_module, "resolve_endpoint_direction", make_second_terminal_direction_finite
    )
    layout, smoothing, authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5, 0.5),
        starting_lambdas=(10.0, 0.1, 0.1),
    )

    assert direction_calls == 6
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.terminal_endpoint_directions == {}
    assert not any(item.revalidated_face_components for item in smoothing.history)
    retraction = smoothing.history[-1]
    assert retraction.deactivated_face_components == layout.penalty_names
    assert retraction.joint_rollback_penalty_fingerprint is not None
    assert len(retraction.coefficient_fit_indices) == 2
    assert retraction.accepted_fit_index == retraction.coefficient_fit_indices[-1]
    assert tuple(
        authority_faces[id(smoothing.coefficient_fits[index])]
        for index in retraction.coefficient_fit_indices
    ) == (layout.penalty_names, ())
    direction = retraction.endpoint_direction_evidence
    assert isinstance(direction, JointEndpointDirectionEvidence)
    assert direction.decision == "finite"
    assert direction.fit_indices == retraction.coefficient_fit_indices[:1]
    assert direction.component_directions[0][1].decision == "endpoint"
    assert direction.component_directions[1][1].decision == "finite"
    assert direction.component_directions[2][1].decision == "endpoint"
    assert smoothing.terminal_fit_index == retraction.accepted_fit_index
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_fit.converged is True
    assert tuple(
        index for item in smoothing.history for index in item.coefficient_fit_indices
    ) == tuple(range(1, len(smoothing.coefficient_fits)))


@pytest.mark.parametrize("forgery", ["missing", "swapped", "stale_objective"])
def test_joint_terminal_result_rejects_forged_scalar_direction_mappings(
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    _layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
    )
    directions = dict(smoothing.terminal_endpoint_directions)
    first_name, second_name = tuple(directions)

    if forgery == "missing":
        forged = {first_name: directions[first_name]}
    elif forgery == "swapped":
        forged = {
            second_name: directions[second_name],
            first_name: directions[first_name],
        }
    else:
        forged = {
            first_name: replace(
                directions[first_name],
                endpoint_objective=directions[first_name].endpoint_objective + 1.0,
            ),
            second_name: directions[second_name],
        }

    with pytest.raises(ValueError, match="terminal endpoint directions"):
        replace(smoothing, terminal_endpoint_directions=forged)


def test_joint_terminal_result_rejects_partial_common_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5, 0.5),
        starting_lambdas=(10.0, 0.1, 0.1),
    )
    revalidation_index, revalidation = next(
        (index, item)
        for index, item in enumerate(smoothing.history)
        if item.revalidated_face_components
    )
    direction = revalidation.endpoint_direction_evidence
    assert isinstance(direction, JointEndpointDirectionEvidence)
    partial_names = layout.penalty_names[:2]
    partial_direction = replace(
        direction,
        component_directions=direction.component_directions[:2],
    )
    partial_revalidation = replace(
        revalidation,
        revalidated_face_components=partial_names,
        endpoint_direction_evidence=partial_direction,
    )
    forged_history = list(smoothing.history)
    forged_history[revalidation_index] = partial_revalidation

    with pytest.raises(ValueError, match="incomplete component coverage"):
        replace(smoothing, history=tuple(forged_history))


def test_efs_joint_activation_adds_the_whole_cohort_to_an_active_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5, 0.5),
        starting_lambdas=(10.0, 0.1, 0.1),
    )

    activations = tuple(item for item in smoothing.history if item.activated_face_components)
    assert activations[0].activated_face_components == (layout.penalty_names[0],)
    joint = activations[1]
    assert joint.activated_face_components == layout.penalty_names[1:]
    assert len(joint.coefficient_fit_indices) == 1
    assert isinstance(joint.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    source_face = smoothing.coefficient_fits[joint.source_fit_index].coefficient_face
    accepted_face = smoothing.coefficient_fits[joint.accepted_fit_index].coefficient_face
    assert source_face is not None
    assert source_face.component_names == (layout.penalty_names[0],)
    assert accepted_face is not None
    assert accepted_face.component_names == layout.penalty_names


def test_efs_refuses_a_joint_face_when_one_cap_direction_is_weakly_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 1.01),
        maximum_lambda=100.0,
    )

    assert not any(item.activated_face_components for item in smoothing.history)
    assert smoothing.unresolved_upper_bound == (layout.penalty_names[0],)
    refusal = smoothing.history[-1]
    assert refusal.refused_face_components == layout.penalty_names
    assert len(refusal.coefficient_fit_indices) == 1
    assert isinstance(refusal.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    assert refusal.endpoint_direction_evidence.endpoint_objective < refusal.objective_before
    assert refusal.endpoint_direction_evidence.decision == "finite"
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_refuses_a_joint_face_when_one_cap_direction_is_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 1.0),
    )

    assert not any(item.activated_face_components for item in smoothing.history)
    refusal = smoothing.history[-1]
    assert refusal.refused_face_components == layout.penalty_names
    assert len(refusal.coefficient_fit_indices) == 1
    assert isinstance(refusal.endpoint_direction_evidence, JointEndpointDirectionEvidence)
    assert refusal.endpoint_direction_evidence.decision == "unresolved"
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_refuses_overlapping_joint_geometry_before_an_authority_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, smoothing, authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
        overlap=True,
    )

    assert not any(item.activated_face_components for item in smoothing.history)
    assert layout.penalty_names not in authority_faces.values()
    assert smoothing.history == ()
    assert len(smoothing.coefficient_fits) == 1
    assert smoothing.terminal_fit.coefficient_face is None


@pytest.mark.parametrize(
    ("failure_mode", "failure_reason"),
    [
        ("not_converged", "joint_endpoint_not_converged"),
        ("not_stationary", "joint_endpoint_not_stationary"),
        ("analytic_unavailable", "joint_analytic_unavailable"),
        ("objective_rejected", "joint_objective_rejected"),
    ],
)
def test_efs_records_typed_joint_failures_with_their_one_fitted_face(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    failure_reason: str,
) -> None:
    response = (0.5, 2.0) if failure_mode == "objective_rejected" else (0.5, 0.5)
    if failure_mode in {"not_converged", "not_stationary"}:
        real_fit = efs_module._fit_endpoint_authority_stationary

        def fail_joint_fit(*args: object, **kwargs: object) -> DenseSolverResult:
            fit = real_fit(*args, **kwargs)  # type: ignore[arg-type]
            face = kwargs["face"]
            if not isinstance(face, PenaltyFace) or len(face.component_names) != 2:
                return fit
            if failure_mode == "not_converged":
                return replace(
                    fit,
                    converged=False,
                    convergence_reason="max_iterations",
                )
            terminal_score = np.array(fit.terminal_score, copy=True)
            terminal_score[-1] = 1.0e-4
            return replace(
                fit,
                terminal_score=terminal_score,
                convergence_reason="objective_and_step",
            )

        monkeypatch.setattr(smoothing_faces, "_fit_endpoint_authority_stationary", fail_joint_fit)
        monkeypatch.setattr(smoothing_loop, "_fit_endpoint_authority_stationary", fail_joint_fit)
        monkeypatch.setattr(efs_module, "_fit_endpoint_authority_stationary", fail_joint_fit)
    elif failure_mode == "analytic_unavailable":

        def unavailable_direction(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise EndpointLaplaceError("test mutation")

        monkeypatch.setattr(
            smoothing_faces, "evaluate_endpoint_laplace_derivative", unavailable_direction
        )
        monkeypatch.setattr(
            efs_module, "evaluate_endpoint_laplace_derivative", unavailable_direction
        )

    layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        response,
    )

    refusal = smoothing.history[-1]
    assert refusal.refused_face_components == layout.penalty_names
    assert refusal.endpoint_direction_evidence is None
    assert refusal.endpoint_assessment_failure_reason == failure_reason
    assert len(refusal.coefficient_fit_indices) == 1
    joint_fit = smoothing.coefficient_fits[refusal.coefficient_fit_indices[0]]
    assert joint_fit.coefficient_face is not None
    assert joint_fit.coefficient_face.component_names == layout.penalty_names
    assert tuple(
        index for item in smoothing.history for index in item.coefficient_fit_indices
    ) == tuple(range(1, len(smoothing.coefficient_fits)))


def test_joint_face_result_rejects_forged_fit_authority_and_chronology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _layout, smoothing, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
    )
    activation = next(item for item in smoothing.history if item.activated_face_components)
    assert activation.accepted_fit_index is not None
    accepted_index = activation.accepted_fit_index
    endpoint_fit = smoothing.coefficient_fits[accepted_index]

    wrong_face_fits = list(smoothing.coefficient_fits)
    wrong_face_fits[accepted_index] = smoothing.coefficient_fits[activation.source_fit_index]
    with pytest.raises(
        ValueError,
        match="accepted coefficient face|joint endpoint assessment",
    ):
        replace(smoothing, coefficient_fits=tuple(wrong_face_fits))

    shifted_fits = list(smoothing.coefficient_fits)
    objective_shift = 1.0e-4
    assert endpoint_fit.optimizing_log_likelihood is not None
    assert endpoint_fit.penalized_optimizing_log_likelihood is not None
    shifted_fits[accepted_index] = replace(
        endpoint_fit,
        optimizing_log_likelihood=endpoint_fit.optimizing_log_likelihood + objective_shift,
        log_likelihood=endpoint_fit.log_likelihood + objective_shift,
        penalized_optimizing_log_likelihood=(
            endpoint_fit.penalized_optimizing_log_likelihood + objective_shift
        ),
        penalized_log_likelihood=endpoint_fit.penalized_log_likelihood + objective_shift,
    )
    with pytest.raises(ValueError, match="share the fitted objective"):
        replace(smoothing, coefficient_fits=tuple(shifted_fits))

    moving_score_fits = list(smoothing.coefficient_fits)
    terminal_score = np.array(endpoint_fit.terminal_score, copy=True)
    terminal_score[-1] = 1.0e-4
    moving_score_fits[accepted_index] = replace(
        endpoint_fit,
        terminal_score=terminal_score,
        convergence_reason="objective_and_step",
    )
    with pytest.raises(ValueError, match="stationary common fit"):
        replace(smoothing, coefficient_fits=tuple(moving_score_fits))

    fallback_fits = list(smoothing.coefficient_fits)
    fallback_fits[accepted_index] = replace(
        endpoint_fit,
        terminal_curvature=replace(
            endpoint_fit.terminal_curvature,
            actual_source="fisher",
            reason="test mutation",
            fallback_count=1,
        ),
    )
    with pytest.raises(ValueError, match="unfallbacked observed-curvature authority"):
        replace(smoothing, coefficient_fits=tuple(fallback_fits))

    later_joint_index = smoothing.history[1].coefficient_fit_indices[-1]
    direction = activation.endpoint_direction_evidence
    assert isinstance(direction, JointEndpointDirectionEvidence)
    forged_activation = replace(
        activation,
        coefficient_fit_indices=(later_joint_index,),
        accepted_fit_index=later_joint_index,
        accepted_curvature=smoothing.coefficient_fits[later_joint_index].terminal_curvature,
        endpoint_direction_evidence=replace(
            direction,
            endpoint_fit_index=later_joint_index,
        ),
    )
    with pytest.raises(ValueError, match="chronology"):
        replace(
            smoothing,
            history=(forged_activation, *smoothing.history[1:]),
        )


def _projected_penalty_problem() -> tuple[
    StackedLayout,
    PenaltyFace,
    dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    n = 6
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, n),
                "group": ["a", "b", "c", "a", "b", "c"],
            }
        )
    )
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "group": RandomEffect()}),
                Predictor("scale", {}),
            ),
        )
    )
    face_slice = layout.term_slices["location:x"]
    retained_slice = layout.term_slices["location:group"]
    face_component = PenaltyComponent(
        name="location:x#identity",
        group_name="location:x",
        group_index=0,
        group_sl=face_slice,
        omega_raw=None,
        omega_ssp=None,
        rank=1.0,
        eigvals_omega=np.ones(1),
        penalty_kind="identity",
    )
    first_matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    second_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    first_component = _dense_component(
        "location:group#first",
        "location:group",
        1,
        retained_slice,
        first_matrix,
        rank=2.0,
    )
    second_component = _dense_component(
        "location:group#second",
        "location:group",
        1,
        retained_slice,
        second_matrix,
        rank=2.0,
    )
    layout = replace(
        layout,
        penalties=(face_component, first_component, second_component),
    )
    face = build_penalty_face(layout, (face_component.name,))
    lambdas = {
        face_component.name: 13.0,
        first_component.name: 1.75,
        second_component.name: 0.6,
    }
    return layout, face, lambdas, first_matrix, second_matrix


def _independent_projected_penalty_problem() -> tuple[
    StackedLayout,
    PenaltyFace,
    dict[str, float],
]:
    n = 6
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "face": np.linspace(-1.0, 1.0, n),
                "x": np.linspace(0.0, 2.0, n),
                "z": np.linspace(2.0, 0.0, n),
            }
        )
    )
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor(
                    "location",
                    {"face": Numeric(), "x": Numeric(), "z": Numeric()},
                ),
                Predictor("scale", {}),
            ),
        )
    )
    face_component = _dense_component(
        "location:face#identity",
        "location:face",
        0,
        layout.term_slices["location:face"],
        np.ones((1, 1)),
        rank=1.0,
    )
    x_component = _dense_component(
        "location:x#identity",
        "location:x",
        1,
        layout.term_slices["location:x"],
        np.ones((1, 1)),
        rank=1.0,
    )
    z_component = _dense_component(
        "location:z#identity",
        "location:z",
        2,
        layout.term_slices["location:z"],
        np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(face_component, x_component, z_component))
    face = build_penalty_face(layout, (face_component.name,))
    return (
        layout,
        face,
        {
            face_component.name: 7.0,
            x_component.name: 1.0,
            z_component.name: 1.0e-12,
        },
    )


def test_face_component_states_match_independent_reduced_penalty_geometry() -> None:
    """Kills using full-space structural ranks after restricting coefficients."""
    layout, face, lambdas, first_matrix, second_matrix = _projected_penalty_problem()

    states = projected_component_states(layout=layout, lambdas=lambdas, face=face)

    assert tuple(state.name for state in states) == (
        "location:group#first",
        "location:group#second",
    )
    retained_slice = layout.term_slices["location:group"]
    full_first = np.zeros((layout.n_coefficients, layout.n_coefficients))
    full_second = np.zeros_like(full_first)
    full_first[retained_slice, retained_slice] = first_matrix
    full_second[retained_slice, retained_slice] = second_matrix
    reduced = lambdas[states[0].name] * face.reduce_matrix(full_first) + lambdas[
        states[1].name
    ] * face.reduce_matrix(full_second)
    reduced_inverse = np.linalg.pinv(reduced, hermitian=True)
    expected = (
        lambdas[states[0].name] * np.trace(reduced_inverse @ face.reduce_matrix(full_first)),
        lambdas[states[1].name] * np.trace(reduced_inverse @ face.reduce_matrix(full_second)),
    )
    np.testing.assert_allclose(
        [state.rank for state in states],
        expected,
        rtol=128.0 * np.finfo(np.float64).eps,
        atol=128.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_array_equal(states[0].penalty, first_matrix)
    np.testing.assert_array_equal(states[1].penalty, second_matrix)


def test_face_component_rank_admits_only_roundoff_above_its_block_width() -> None:
    """Kills rejecting a trace rank one ULP above its exact upper bound."""
    rank = _bounded_effective_rank(
        np.nextafter(3.0, np.inf),
        width=3,
        problem_width=50,
    )

    assert rank == 3.0


def test_face_component_rank_refuses_material_block_overflow() -> None:
    """Kills hiding incorrect projected penalty geometry behind clipping."""
    with pytest.raises(
        ValueError,
        match="projected effective rank lies outside its coefficient block",
    ):
        _bounded_effective_rank(3.0 + 1.0e-8, width=3, problem_width=50)


def test_face_component_states_resolve_independent_group_traces_separately() -> None:
    """Kills computing component traces from a cross-group global cutoff."""
    layout, face, lambdas = _independent_projected_penalty_problem()

    states = projected_component_states(layout=layout, lambdas=lambdas, face=face)

    assert tuple(state.name for state in states) == (
        "location:x#identity",
        "location:z#identity",
    )
    np.testing.assert_allclose(
        [state.rank for state in states],
        [1.0, 1.0],
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )


def _axis_aligned_projected_face(
    layout: StackedLayout,
    face: PenaltyFace,
    retained: PenaltyComponent,
) -> tuple[PenaltyFace, np.ndarray]:
    constrained = np.zeros(layout.n_coefficients, dtype=bool)
    face_components = {
        component.name: component
        for component in layout.penalties
        if component.name in face.component_names
    }
    for name in face.component_names:
        component = face_components[name]
        constrained[component.group_sl] = True
    constrained_indices = np.flatnonzero(constrained)
    free_indices = np.flatnonzero(~constrained)
    assert len(constrained_indices) == face.constraint_rank
    coordinate_basis = np.eye(layout.n_coefficients, dtype=np.float64)
    aligned_face = replace(
        face,
        null_basis=coordinate_basis[:, free_indices],
        constraint_basis=coordinate_basis[:, constrained_indices],
    )
    retained_full_indices = np.arange(
        retained.group_sl.start,
        retained.group_sl.stop,
    )
    retained_reduced_indices = np.flatnonzero(
        np.isin(free_indices, retained_full_indices),
    )
    assert len(retained_reduced_indices) == retained.group_sl.stop - retained.group_sl.start
    return aligned_face, retained_reduced_indices


def _gaussian_endpoint_problem() -> tuple[
    GaussianLS,
    StackedLayout,
    PenaltyFace,
    dict[str, float],
    np.ndarray,
    ResolvedLikelihoodWeights,
    np.ndarray,
    np.ndarray,
]:
    n = 18
    x = np.linspace(-1.5, 1.5, n)
    groups = np.array(["a", "b", "c"] * 6)
    group_effects = {"a": -0.45, "b": 0.3, "c": 0.15}
    response = np.array(
        [
            1.4 + group_effects[group] + 0.18 * np.sin(0.7 * index) + 0.04 * np.cos(1.3 * index)
            for index, group in enumerate(groups)
        ],
        dtype=np.float64,
    )
    weights = resolved_prior(np.linspace(0.8, 1.3, n))
    frame = as_eager_frame(pd.DataFrame({"x": x, "group": groups}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "group": RandomEffect()}),
                Predictor("scale", {}),
            ),
        )
    )
    face_slice = layout.term_slices["location:x"]
    retained_slice = layout.term_slices["location:group"]
    face_component = PenaltyComponent(
        name="location:x#identity",
        group_name="location:x",
        group_index=0,
        group_sl=face_slice,
        omega_raw=None,
        omega_ssp=None,
        rank=1.0,
        eigvals_omega=np.ones(1),
        penalty_kind="identity",
    )
    first_matrix = np.diag([2.0, 1.0, 0.5])
    second_matrix = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    first_component = _dense_component(
        "location:group#first",
        "location:group",
        1,
        retained_slice,
        first_matrix,
        rank=3.0,
    )
    second_component = _dense_component(
        "location:group#second",
        "location:group",
        1,
        retained_slice,
        second_matrix,
        rank=2.0,
    )
    layout = replace(
        layout,
        penalties=(face_component, first_component, second_component),
    )
    face = build_penalty_face(layout, (face_component.name,))
    lambdas = {
        face_component.name: 0.0,
        first_component.name: 1.4,
        second_component.name: 0.35,
    }
    return (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        first_matrix,
        second_matrix,
    )


def _fit_gaussian_endpoint(
    family: object,
    layout: StackedLayout,
    face: PenaltyFace,
    lambdas: dict[str, float],
    response: np.ndarray,
    weights: ResolvedLikelihoodWeights,
    *,
    coefficient_curvature: str = "fisher",
) -> DenseSolverResult:
    return fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),  # type: ignore[attr-defined]
        layout.penalty_matrix(lambdas),
        config=DenseSolverConfig(
            tolerance=1.0e-10,
            max_iterations=150,
            coefficient_curvature=coefficient_curvature,  # type: ignore[arg-type]
        ),
        coefficient_face=face,
    )


def test_chunked_and_dense_coefficient_face_fits_agree() -> None:
    family, layout, face, lambdas, response, weights, _, _ = _gaussian_endpoint_problem()
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    config = DenseSolverConfig(tolerance=1.0e-10, max_iterations=150)
    dense = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        layout.penalty_matrix(lambdas),
        config=config,
        coefficient_face=face,
    )
    chunked = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        layout.penalty_matrix(lambdas),
        config=config,
        chunk_size=5,
        coefficient_face=face,
    )

    assert dense.converged is True
    assert chunked.converged is True
    np.testing.assert_allclose(chunked.coefficients, dense.coefficients, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        chunked.terminal_pseudo_inverse(),
        dense.terminal_pseudo_inverse(),
        rtol=1e-10,
        atol=1e-11,
    )
    assert np.linalg.norm(face.constraint_matrix @ chunked.coefficients) <= (
        face.null_residual_bound
    )
    chunked_objective = evaluate_endpoint_laplace(
        chunked,
        layout=layout,
        lambdas=lambdas,
        face=face,
    ).objective
    dense_objective = evaluate_endpoint_laplace(
        dense,
        layout=layout,
        lambdas=lambdas,
        face=face,
    ).objective
    objective_scale = 1.0 + max(abs(chunked_objective), abs(dense_objective))
    assert abs(chunked_objective - dense_objective) <= (2.0 * config.tolerance * objective_scale)


@pytest.mark.parametrize("mean", [0.5, 1.0, 1.01, 3.0])
def test_scalar_endpoint_matches_closed_form_without_deciding_the_boundary(
    mean: float,
) -> None:
    layout, face, lambdas, result = _scalar_endpoint_fit(mean)

    evaluation = evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert evaluation.objective == pytest.approx(mean**2 / 2.0, rel=0.0, abs=1e-15)
    assert evaluation.face_component_names == face.component_names
    assert evaluation.finite_component_names == ()
    assert evaluation.reduced_width == 0
    assert evaluation.hessian_rank == 0
    assert evaluation.penalty_rank == 0
    assert evaluation.hessian_log_pdet == 0.0
    assert evaluation.penalty_log_pdet == 0.0
    published_fields = {field.name for field in fields(evaluation)}
    assert not published_fields.intersection(
        {"converged", "convergence_reason", "boundary", "boundary_decision"}
    )


def test_efs_rechecks_a_true_infinity_optimum_before_reporting_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions: list[EndpointDirectionEvidence] = []
    real_resolve = efs_module.resolve_endpoint_direction

    def capture_direction(*args: object, **kwargs: object) -> EndpointDirectionEvidence:
        evidence = real_resolve(*args, **kwargs)  # type: ignore[arg-type]
        decisions.append(evidence)
        return evidence

    monkeypatch.setattr(smoothing_faces, "resolve_endpoint_direction", capture_direction)
    monkeypatch.setattr(efs_module, "resolve_endpoint_direction", capture_direction)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert len(decisions) == 2
    assert all(evidence.decision == "endpoint" for evidence in decisions)
    assert smoothing.converged is True
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.terminal_fit.coefficient_face.component_names == (component_name,)
    np.testing.assert_array_equal(smoothing.terminal_fit.coefficients, np.zeros(1))
    assert smoothing.history[0].activated_face_components == (component_name,)
    assert (
        replace(
            smoothing.history[0].endpoint_direction_evidence,
            fit_indices=(),
            coefficient_tolerance=None,
        )
        == decisions[0]
    )
    assert smoothing.history[0].boundary_nominations == ()
    assert smoothing.history[1].revalidated_face_components == (component_name,)
    assert (
        replace(
            smoothing.history[1].endpoint_direction_evidence,
            fit_indices=(),
            coefficient_tolerance=None,
        )
        == decisions[1]
    )
    assert smoothing.history[0].coefficient_fit_indices == (
        smoothing.history[0].endpoint_direction_evidence.fit_indices
    )
    assert smoothing.history[1].coefficient_fit_indices == (
        smoothing.history[1].endpoint_direction_evidence.fit_indices
    )
    assert tuple(
        index for item in smoothing.history for index in item.coefficient_fit_indices
    ) == tuple(range(1, len(smoothing.coefficient_fits)))
    assert all(
        smoothing.coefficient_fits[index].config.coefficient_curvature == "observed"
        for item in smoothing.history
        for index in item.coefficient_fit_indices
    )
    assert dict(smoothing.terminal_endpoint_directions) == {
        component_name: smoothing.history[1].endpoint_direction_evidence
    }
    assert smoothing.matched_certified is False
    with pytest.raises(RuntimeError, match="numerically supported but not certified"):
        smoothing.assert_matched_certified()


def test_terminal_face_recheck_preserves_the_canonical_endpoint_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finite-cap comparison must not perturb the face state being certified."""
    calls: list[tuple[np.ndarray, np.ndarray | None]] = []
    real_check = efs_module._check_face_direction

    def record_endpoint_start(*args: object, **kwargs: object):
        initial = np.array(kwargs["initial"], copy=True)
        endpoint_initial_value = kwargs.get("endpoint_initial")
        endpoint_initial = (
            None if endpoint_initial_value is None else np.array(endpoint_initial_value, copy=True)
        )
        calls.append((initial, endpoint_initial))
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", record_endpoint_start)
    monkeypatch.setattr(efs_module, "_check_face_direction", record_endpoint_start)
    _component_name, smoothing = _scalar_efs_fit(0.5)

    assert smoothing.converged is True
    assert len(calls) == 2
    assert calls[0][1] is None
    assert calls[1][1] is not None
    np.testing.assert_array_equal(calls[1][1], calls[1][0])
    revalidation = smoothing.history[-1]
    assert revalidation.revalidated_face_components
    assert revalidation.endpoint_direction_evidence is not None
    endpoint_index = revalidation.endpoint_direction_evidence.fit_indices[1]
    np.testing.assert_array_equal(
        smoothing.coefficient_fits[endpoint_index].coefficients,
        smoothing.coefficient_fits[revalidation.source_fit_index].coefficients,
    )
    with pytest.raises(ValueError, match="failure reason"):
        replace(
            revalidation,
            endpoint_assessment_failure_reason="analytic_unavailable",
        )


def test_terminal_face_recheck_refuses_a_moving_endpoint_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = efs_module._check_face_direction

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            endpoint_initial = np.array(kwargs["endpoint_initial"], copy=True)
            endpoint_initial[0] += 1.0e-4
            kwargs["endpoint_initial"] = endpoint_initial
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    monkeypatch.setattr(efs_module, "_check_face_direction", perturb_revalidation_start)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert calls == 2
    assert smoothing.converged is False
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.history[-1].deactivated_face_components == (component_name,)
    assert smoothing.history[-1].endpoint_direction_evidence is None


def test_stationary_cap_recheck_refuses_tiny_endpoint_state_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills applying the direct-route refit envelope to an ordinary cap."""
    calls = 0
    perturbed_starts: list[np.ndarray] = []
    real_check = efs_module._check_face_direction

    def perturb_revalidation_start(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            endpoint_initial = np.array(kwargs["endpoint_initial"], copy=True)
            endpoint_initial[0] += 1.0e-14
            perturbed_starts.append(endpoint_initial)
            kwargs["endpoint_initial"] = endpoint_initial
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", perturb_revalidation_start)
    monkeypatch.setattr(efs_module, "_check_face_direction", perturb_revalidation_start)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert calls == 2
    assert len(perturbed_starts) == 1
    terminal_event = smoothing.history[-1]
    endpoint_index = terminal_event.coefficient_fit_indices[1]
    endpoint_fit = smoothing.coefficient_fits[endpoint_index]
    movement_bound = efs_module._endpoint_candidate_refit_bound(
        perturbed_starts[0],
        endpoint_fit,
        tolerance=terminal_event.coefficient_tolerances[1],
    )
    movement = float(np.max(np.abs(endpoint_fit.coefficients - perturbed_starts[0]), initial=0.0))
    assert movement_bound is not None
    assert 0.0 < movement <= movement_bound

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.terminal_fit.coefficient_face is None
    assert terminal_event.deactivated_face_components == (component_name,)
    assert terminal_event.endpoint_direction_evidence is None


def test_endpoint_assessment_on_two_noise_effects_returns_a_valid_result() -> None:
    """Kills probing so close to the face that fitted-term roundoff aborts EFS."""
    n_observations = 36
    family = GaussianLS()
    response = np.random.default_rng(0).normal(1.0, 0.2, n_observations)
    weights = resolved_prior(np.ones(n_observations))
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "first": np.tile(["a", "b", "c"], 12),
                "second": np.repeat(["u", "v", "w"], 12),
            }
        )
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "location",
                    {"first": RandomEffect(), "second": RandomEffect()},
                    intercept=False,
                ),
                Predictor("scale", {}),
            ),
        )
    )
    smoothing = fit_distributional_efs(
        family,
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),
        lambdas={name: 10.0 for name in layout.penalty_names},
        solver_config=DenseSolverConfig(max_iterations=150, tolerance=1.0e-10),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=15,
            tolerance=1.0e-8,
            maximum_lambda=10.0,
        ),
    )

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "max_iterations"
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.terminal_fit.coefficient_face.component_names == (layout.penalty_names[0],)
    assert smoothing.matched_certified is False


def test_efs_cannot_certify_an_ambiguous_finite_optimum_at_the_default_cap() -> None:
    """Kills treating cancellation in a capped raw update as stationarity."""
    maximum_lambda = DistributionalEFSConfig(outer="efs").maximum_lambda
    component_name, smoothing = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=maximum_lambda,
    )

    assert smoothing.converged is False
    assert smoothing.matched_certified is False
    assert smoothing.lambdas[component_name] <= maximum_lambda
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_cannot_certify_a_finite_optimum_far_below_the_default_cap() -> None:
    """Kills using the saturated fixed-point residual instead of the raw step."""
    maximum_lambda = DistributionalEFSConfig(outer="efs").maximum_lambda
    component_name, smoothing = _scalar_efs_fit(2.0, maximum_lambda=maximum_lambda)

    if smoothing.matched_certified:
        assert smoothing.lambdas[component_name] == pytest.approx(1.0 / 3.0, rel=1.0e-6)
    else:
        assert smoothing.converged is False
    assert not (smoothing.matched_certified and smoothing.lambdas[component_name] == maximum_lambda)
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_accepts_a_finite_optimum_at_the_upper_cap() -> None:
    """Kills classifying roundoff-sized downward drift as endpoint pressure."""
    maximum_lambda = 10.0
    finite_optimum_mean = np.sqrt(1.0 + 1.0 / maximum_lambda)
    component_name, smoothing = _scalar_efs_fit(
        finite_optimum_mean,
        maximum_lambda=maximum_lambda,
    )

    assert smoothing.converged is True
    assert smoothing.matched_certified is True
    assert smoothing.lambdas[component_name] == maximum_lambda
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is None


def test_amplified_gfs_step_cannot_veto_finite_stationarity() -> None:
    """Kills using a cancellation-amplified GFS quotient away from the cap."""
    finite_optimum = 1.0e9
    finite_optimum_mean = np.sqrt(1.0 + 1.0 / finite_optimum)
    component_name, smoothing = _scalar_efs_fit(
        finite_optimum_mean,
        maximum_lambda=1.0e10,
        starting_lambda=finite_optimum,
    )

    assert smoothing.converged is True
    assert smoothing.matched_certified is True
    assert smoothing.lambdas[component_name] == finite_optimum
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_refuses_a_weak_finite_optimum_far_beyond_a_small_trigger_cap() -> None:
    component_name, smoothing = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=0.01,
    )

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == (component_name,)
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_endpoint_directions == {}


def test_efs_endpoint_decision_is_invariant_to_penalty_units() -> None:
    """The analytic direction must transform consistently when S is rescaled."""
    component_name, smoothing = _scalar_efs_fit(
        1.000000000001,
        maximum_lambda=1.0e8,
        penalty_scale=1.0e-8,
    )

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == (component_name,)
    assert smoothing.terminal_fit.coefficient_face is None


def test_analytic_endpoint_direction_follows_the_penalty_scale_orbit() -> None:
    """Penalty units may rescale τ derivatives but cannot change their decision."""
    receipts: list[tuple[float, EndpointDirectionEvidence, np.ndarray, np.ndarray]] = []
    for penalty_scale in (1.0e-12, 1.0e-8, 1.0, 1.0e8, 1.0e12):
        component_name, smoothing = _scalar_efs_fit(
            0.5,
            maximum_lambda=1.0e10 / penalty_scale,
            penalty_scale=penalty_scale,
        )
        assert smoothing.converged is True
        evidence = smoothing.terminal_endpoint_directions[component_name]
        assert smoothing.terminal_fit.coefficient_face is not None
        receipts.append(
            (
                penalty_scale,
                evidence,
                smoothing.terminal_fit.eta,
                smoothing.terminal_fit.theta,
            )
        )

    physical_derivatives = np.array(
        [
            penalty_scale * evidence.analytic_derivative
            for penalty_scale, evidence, _eta, _theta in receipts
        ]
    )
    np.testing.assert_allclose(
        physical_derivatives,
        np.full(len(receipts), physical_derivatives[0]),
        rtol=512.0 * np.finfo(np.float64).eps,
        atol=512.0 * np.finfo(np.float64).eps,
    )
    physical_intervals = np.array(
        [
            (penalty_scale * evidence.lower_bound, penalty_scale * evidence.upper_bound)
            for penalty_scale, evidence, _eta, _theta in receipts
        ]
    )
    assert np.max(physical_intervals[:, 0]) <= np.min(physical_intervals[:, 1])
    baseline_evidence = receipts[0][1]
    baseline_eta = receipts[0][2]
    baseline_theta = receipts[0][3]
    for _, evidence, eta, theta in receipts:
        np.testing.assert_allclose(
            evidence.endpoint_objective, baseline_evidence.endpoint_objective
        )
        np.testing.assert_array_equal(eta, baseline_eta)
        np.testing.assert_array_equal(theta, baseline_theta)
        assert evidence.decision == "endpoint"


def test_analytic_endpoint_direction_resolves_anisotropic_penalty_geometry() -> None:
    family = _UnitGaussian()
    response = np.array([1.01, 1.0], dtype=np.float64)
    weights = resolved_prior(np.ones(2, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"effect": ["first", "second"]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"effect": RandomEffect()}, intercept=False),),
        )
    )
    component = layout.penalties[0]
    local_penalty = np.diag([1.0e-8, 1.0])
    component = replace(
        component,
        omega_raw=local_penalty,
        omega_ssp=local_penalty,
        eigvals_omega=np.array([1.0e-8, 1.0]),
        log_det_omega_plus=float(np.log(1.0e-8)),
        penalty_kind="dense",
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    attempt = efs_module._check_face_direction(
        family,  # type: ignore[arg-type]
        layout,
        response,
        plan,  # type: ignore[arg-type]
        lambdas={component.name: 1.0},
        component_name=component.name,
        finite_face=None,
        endpoint_face=face,
        initial=np.zeros(2),
        solver_config=DenseSolverConfig(max_iterations=50, tolerance=1.0e-12),
        chunk_size=None,
        phase_recorder=None,
    )

    check = attempt.check
    assert check is not None
    assert check.direction.authority_identifier == ("analytic-observed-curvature-direction/v1")
    assert check.direction.analytic_derivative == 0.5 * (
        check.direction.profile_score_term
        + check.direction.curvature_schur_term
        + check.direction.curvature_drift_term
    )
    assert check.direction.decision == "finite"
    assert check.direction.lower_bound < -1.0e5
    assert check.direction.upper_bound < 0.0


def _two_face_efs_fit():
    family = _UnitGaussian()
    response = np.zeros(4, dtype=np.float64)
    weights = resolved_prior(np.ones(4, dtype=np.float64))
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "first": ["a", "a", "b", "b"],
                "second": ["c", "d", "c", "d"],
            }
        )
    )
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (
                Predictor(
                    "mean",
                    {"first": RandomEffect(), "second": RandomEffect()},
                    intercept=False,
                ),
            ),
        )
    )
    first_name, second_name = layout.penalty_names

    smoothing = fit_distributional_efs(
        family,  # type: ignore[arg-type]
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),  # type: ignore[arg-type]
        lambdas={first_name: 0.1, second_name: 10.0},
        solver_config=DenseSolverConfig(max_iterations=50, tolerance=1.0e-12),
        efs_config=DistributionalEFSConfig(
            outer="efs",
            max_iterations=8,
            tolerance=1.0e-8,
            initial_lambda=0.1,
            maximum_lambda=10.0,
        ),
    )
    return (first_name, second_name), smoothing


def _resolution_limited_cap_revalidation_result(
    *,
    cap_objective: float,
) -> tuple[DistributionalEFSResult, int, int, int]:
    """Forge only the finite-cap state on an authenticated scalar revalidation."""
    (first_name, _second_name), smoothing = _two_face_efs_fit()
    activation = next(
        item for item in smoothing.history if item.activated_face_components == (first_name,)
    )
    cap_source_index, endpoint_source_index = activation.coefficient_fit_indices
    cap_source = smoothing.coefficient_fits[cap_source_index]
    endpoint_source = smoothing.coefficient_fits[endpoint_source_index]
    cap_face = cap_source.coefficient_face
    assert cap_face is not None
    tolerance = activation.coefficient_tolerances[0]
    retained_score = np.zeros(cap_face.reduced_width, dtype=np.float64)
    retained_score[0] = 2.0 * tolerance * (1.0 + abs(cap_objective))
    optimizing = cap_objective + cap_source.penalty_value
    reported = optimizing + cap_source.parameter_independent_carrier
    cap = replace(
        cap_source,
        optimizing_log_likelihood=optimizing,
        log_likelihood=reported,
        penalized_optimizing_log_likelihood=cap_objective,
        penalized_log_likelihood=reported - cap_source.penalty_value,
        terminal_score=cap_face.lift_vector(retained_score),
        score_relative=2.0 * tolerance,
        convergence_reason="resolution_limited_stationarity",
    )

    prefix_history = smoothing.history[: activation.iteration]
    prefix_fits = smoothing.coefficient_fits[: endpoint_source_index + 1]
    cap_index = len(prefix_fits)
    endpoint_index = cap_index + 1
    direction = replace(
        activation.endpoint_direction_evidence,
        fit_indices=(cap_index, endpoint_index),
    )
    revalidation = replace(
        activation,
        iteration=len(prefix_history) + 1,
        source_fit_index=endpoint_source_index,
        accepted_fit_index=endpoint_index,
        coefficient_fit_indices=(cap_index, endpoint_index),
        activated_face_components=(),
        revalidated_face_components=(first_name,),
        objective_before=activation.objective_after,
        objective_relative_change=(
            abs(activation.objective_after - activation.objective_after)
            / (1.0 + abs(activation.objective_after))
        ),
        endpoint_direction_evidence=direction,
    )
    authenticated = replace(
        smoothing,
        objective=revalidation.objective_after,
        converged=False,
        convergence_reason="max_iterations",
        iterations=revalidation.iteration,
        history=(*prefix_history, revalidation),
        coefficient_fits=(*prefix_fits, cap, endpoint_source),
        terminal_fit_index=endpoint_index,
        terminal_endpoint_directions={},
    )
    return authenticated, cap_index, endpoint_index, endpoint_source_index


def test_resolution_limited_stationary_cap_rejects_revalidation_movement_forgery() -> None:
    """Kills using raw cap KKT, rather than numerical stationarity, for movement."""
    smoothing, cap_index, endpoint_index, source_index = (
        _resolution_limited_cap_revalidation_result(cap_objective=-1.0)
    )
    revalidation = smoothing.history[-1]
    cap = smoothing.coefficient_fits[cap_index]
    endpoint = smoothing.coefficient_fits[endpoint_index]
    source = smoothing.coefficient_fits[source_index]
    tolerance = revalidation.coefficient_tolerances[0]
    assert cap.convergence_reason == "resolution_limited_stationarity"
    assert efs_module._endpoint_retained_kkt_relative(cap) > tolerance
    assert result_module._assessment_is_numerically_stationary(cap, tolerance)

    moved_coefficients = np.array(endpoint.coefficients, copy=True)
    moved_coefficients[0] = 1.0e-14
    moved_endpoint = replace(endpoint, coefficients=moved_coefficients)
    movement_bound = result_module._assessment_coefficient_refit_bound(
        source.coefficients,
        moved_endpoint.coefficients,
        tolerance=tolerance,
    )
    movement = float(np.max(np.abs(moved_endpoint.coefficients - source.coefficients), initial=0.0))
    assert movement_bound is not None
    assert 0.0 < movement <= movement_bound
    forged_fits = list(smoothing.coefficient_fits)
    forged_fits[endpoint_index] = moved_endpoint

    with pytest.raises(ValueError, match="canonical endpoint state"):
        replace(smoothing, coefficient_fits=tuple(forged_fits))


def test_resolution_limited_stationary_cap_enforces_cap_objective_ceiling() -> None:
    """Kills skipping the finite-cap objective ceiling when raw KKT is above tolerance."""
    smoothing, cap_index, _endpoint_index, _source_index = (
        _resolution_limited_cap_revalidation_result(cap_objective=-1.0)
    )
    revalidation = smoothing.history[-1]
    cap = smoothing.coefficient_fits[cap_index]
    tolerance = revalidation.coefficient_tolerances[0]
    direction = revalidation.endpoint_direction_evidence
    assert direction is not None
    assert cap.convergence_reason == "resolution_limited_stationarity"
    assert efs_module._endpoint_retained_kkt_relative(cap) > tolerance
    assert result_module._assessment_is_numerically_stationary(cap, tolerance)
    assert direction.endpoint_objective <= revalidation.objective_before

    cap_face = cap.coefficient_face
    assert cap_face is not None
    forged_objective = 1.0
    retained_score = np.zeros(cap_face.reduced_width, dtype=np.float64)
    retained_score[0] = 2.0 * tolerance * (1.0 + abs(forged_objective))
    optimizing = forged_objective + cap.penalty_value
    reported = optimizing + cap.parameter_independent_carrier
    forged_cap = replace(
        cap,
        optimizing_log_likelihood=optimizing,
        log_likelihood=reported,
        penalized_optimizing_log_likelihood=forged_objective,
        penalized_log_likelihood=reported - cap.penalty_value,
        terminal_score=cap_face.lift_vector(retained_score),
        score_relative=2.0 * tolerance,
    )
    forged_fits = list(smoothing.coefficient_fits)
    forged_fits[cap_index] = forged_cap

    with pytest.raises(ValueError, match="tightened finite reference"):
        replace(smoothing, coefficient_fits=tuple(forged_fits))


def test_stationary_scalar_face_refuses_multi_cap_runtime_and_result_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills partially accepting one stationary component from a capped cohort."""
    names, scalar_smoothing = _two_face_efs_fit()
    activation = next(item for item in scalar_smoothing.history if item.activated_face_components)
    assessed_name = activation.activated_face_components[0]
    companion_name = next(name for name in names if name != assessed_name)
    cap_fit = scalar_smoothing.coefficient_fits[activation.coefficient_fit_indices[0]]
    assert (
        efs_module._endpoint_retained_kkt_relative(cap_fit) <= activation.coefficient_tolerances[0]
    )
    maximum_lambda = scalar_smoothing.config.maximum_lambda

    def forge_companion(values):
        return {
            name: maximum_lambda if name == companion_name else value
            for name, value in values.items()
        }

    forged_history = tuple(
        replace(
            item,
            lambdas_before=forge_companion(item.lambdas_before),
            proposed_lambdas=forge_companion(item.proposed_lambdas),
            lambdas_after=forge_companion(item.lambdas_after),
        )
        for item in scalar_smoothing.history
    )
    try:
        replace(
            scalar_smoothing,
            initial_lambdas=forge_companion(scalar_smoothing.initial_lambdas),
            lambdas=forge_companion(scalar_smoothing.lambdas),
            history=forged_history,
        )
    except ValueError as error:
        forgery_rejected = "sole capped component" in str(error)
    else:
        forgery_rejected = False

    def disable_joint_activation(*args: object, **kwargs: object):
        del args, kwargs
        return None, None

    monkeypatch.setattr(smoothing_loop, "_try_joint_exact_face", disable_joint_activation)
    monkeypatch.setattr(efs_module, "_try_joint_exact_face", disable_joint_activation)
    layout, runtime_result, _authority_faces = _independent_penalty_efs_fit(
        monkeypatch,
        (0.5, 0.5),
    )
    runtime_event = runtime_result.history[-1]
    runtime_cap_fit = runtime_result.coefficient_fits[runtime_event.coefficient_fit_indices[0]]
    assert (
        efs_module._endpoint_retained_kkt_relative(runtime_cap_fit)
        <= runtime_event.coefficient_tolerances[0]
    )
    runtime_refused = bool(
        not runtime_result.converged
        and runtime_result.convergence_reason == "lambda_cap_unresolved"
        and runtime_result.terminal_fit.coefficient_face is None
        and not any(item.activated_face_components for item in runtime_result.history)
        and runtime_event.refused_face_components == (layout.penalty_names[0],)
    )

    assert (runtime_refused, forgery_rejected) == (True, True)


def _joint_terminal_analytic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[tuple[str, str], DistributionalEFSResult]:
    calls = 0
    real_derivative = efs_module.evaluate_endpoint_laplace_derivative

    def fail_second_terminal_direction(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise EndpointLaplaceError("test mutation")
        return real_derivative(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        smoothing_faces, "evaluate_endpoint_laplace_derivative", fail_second_terminal_direction
    )
    monkeypatch.setattr(
        efs_module, "evaluate_endpoint_laplace_derivative", fail_second_terminal_direction
    )
    names, smoothing = _two_face_efs_fit()
    assert calls == 4
    return names, smoothing


def test_efs_keeps_face_names_in_layout_order_when_the_second_component_activates_first() -> None:
    """Kills using activation chronology as the canonical multi-face order."""
    (first_name, second_name), smoothing = _two_face_efs_fit()

    assert tuple(
        item.activated_face_components
        for item in smoothing.history
        if item.activated_face_components
    ) == ((second_name,), (first_name,))
    assert smoothing.terminal_fit.coefficient_face is not None
    assert smoothing.terminal_fit.coefficient_face.component_names == (first_name, second_name)


def test_terminal_failure_rolls_back_the_whole_multi_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills retaining a face certified only conditionally on another failed face."""
    (first_name, second_name), smoothing = _joint_terminal_analytic_failure(monkeypatch)

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.unresolved_upper_bound == (first_name, second_name)
    retraction = smoothing.history[-1]
    assert retraction.deactivated_face_components == (first_name, second_name)
    assert retraction.endpoint_direction_evidence is None
    assert retraction.endpoint_assessment_failure_reason == "joint_analytic_unavailable"
    assert len(retraction.coefficient_fit_indices) == 2
    common_fit, rollback_fit = (
        smoothing.coefficient_fits[index] for index in retraction.coefficient_fit_indices
    )
    assert common_fit.coefficient_face is not None
    assert common_fit.coefficient_face.component_names == (first_name, second_name)
    assert rollback_fit.coefficient_face is None


def test_joint_terminal_result_rejects_deleting_an_executed_common_assessment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills treating fitted analytic failure and preflight failure as one shape."""
    _names, smoothing = _joint_terminal_analytic_failure(monkeypatch)
    retraction = smoothing.history[-1]
    common_index, rollback_index = retraction.coefficient_fit_indices
    rollback_fit = smoothing.coefficient_fits[rollback_index]

    with pytest.raises(ValueError, match="common assessment|assessment fits"):
        forged_retraction = replace(
            retraction,
            coefficient_fit_indices=(common_index,),
            accepted_fit_index=common_index,
            coefficient_tolerances=(retraction.coefficient_tolerances[-1],),
        )
        forged_history = (*smoothing.history[:-1], forged_retraction)
        forged_fits = (*smoothing.coefficient_fits[:common_index], rollback_fit)
        replace(
            smoothing,
            history=forged_history,
            coefficient_fits=forged_fits,
            terminal_fit_index=common_index,
        )


def test_joint_terminal_result_recomputes_the_finite_rollback_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills trusting the event and terminal result's shared stale objective."""
    _names, smoothing = _joint_terminal_analytic_failure(monkeypatch)
    retraction = smoothing.history[-1]
    forged_objective = retraction.objective_after + 1.0
    forged_retraction = replace(
        retraction,
        objective_after=forged_objective,
        objective_relative_change=(
            abs(forged_objective - retraction.objective_before)
            / (1.0 + abs(retraction.objective_before))
        ),
    )

    with pytest.raises(ValueError, match="rollback objective"):
        replace(
            smoothing,
            history=(*smoothing.history[:-1], forged_retraction),
            objective=smoothing.objective + 1.0,
        )


def test_joint_terminal_result_binds_finite_rollback_penalty_magnitude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills accepting an earlier finite fit whose penalty spans the right subspace."""
    _names, smoothing = _joint_terminal_analytic_failure(monkeypatch)
    retraction = smoothing.history[-1]
    _common_index, rollback_index = retraction.coefficient_fit_indices
    rollback_fit = smoothing.coefficient_fits[rollback_index]
    assert retraction.joint_rollback_penalty_fingerprint is not None
    _earlier_index, earlier_fit = next(
        (index, fit)
        for index, fit in enumerate(smoothing.coefficient_fits[: retraction.source_fit_index])
        if fit.coefficient_face is None
        and fit.config == rollback_fit.config
        and not np.array_equal(fit.penalty, rollback_fit.penalty)
    )
    assert not np.array_equal(earlier_fit.penalty, rollback_fit.penalty)
    earlier_penalty_rank = decompose_gram(earlier_fit.penalty)
    assert earlier_fit.penalized_optimizing_log_likelihood is not None
    objective = -earlier_fit.penalized_optimizing_log_likelihood + 0.5 * (
        earlier_fit.terminal_rank.log_pdet - earlier_penalty_rank.log_pdet
    )
    relative_change = abs(objective - retraction.objective_before) / (
        1.0 + abs(retraction.objective_before)
    )
    forged_retraction = replace(
        retraction,
        objective_after=objective,
        objective_relative_change=relative_change,
        accepted_curvature=earlier_fit.terminal_curvature,
    )
    forged_fits = list(smoothing.coefficient_fits)
    forged_fits[rollback_index] = earlier_fit

    with pytest.raises(ValueError, match="terminal penalty|penalty fingerprint"):
        replace(
            smoothing,
            history=(*smoothing.history[:-1], forged_retraction),
            coefficient_fits=tuple(forged_fits),
            objective=objective,
        )


@pytest.mark.parametrize(
    "forgery",
    ["fallback", "loose_tolerance", "provenance", "penalty_geometry"],
)
def test_joint_terminal_result_authenticates_finite_rollback_authority(
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    """Kills accepting a finite rollback that is not the recorded strict authority fit."""
    _names, smoothing = _joint_terminal_analytic_failure(monkeypatch)
    retraction = smoothing.history[-1]
    rollback_index = retraction.coefficient_fit_indices[-1]
    rollback_fit = smoothing.coefficient_fits[rollback_index]
    forged_retraction = retraction
    if forgery == "fallback":
        forged_fit = replace(
            rollback_fit,
            terminal_curvature=replace(
                rollback_fit.terminal_curvature,
                actual_source="fisher",
                reason="test mutation",
                fallback_count=1,
            ),
        )
    elif forgery == "loose_tolerance":
        loose_tolerance = 1.0e-6
        forged_fit = replace(
            rollback_fit,
            config=replace(rollback_fit.config, tolerance=loose_tolerance),
        )
        forged_retraction = replace(
            retraction,
            coefficient_tolerances=(
                *retraction.coefficient_tolerances[:-1],
                loose_tolerance,
            ),
        )
    elif forgery == "provenance":
        forged_fit = replace(
            rollback_fit,
            terminal_rank=replace(
                rollback_fit.terminal_rank,
                policy_version=rollback_fit.terminal_rank.policy_version + 1,
            ),
        )
    else:
        zero_penalty = np.zeros_like(rollback_fit.penalty)
        forged_fit = replace(
            rollback_fit,
            penalty=zero_penalty,
            terminal_penalized_curvature=rollback_fit.terminal_data_curvature,
        )
    forged_fits = list(smoothing.coefficient_fits)
    forged_fits[rollback_index] = forged_fit

    with pytest.raises(ValueError, match="rollback"):
        replace(
            smoothing,
            history=(*smoothing.history[:-1], forged_retraction),
            coefficient_fits=tuple(forged_fits),
        )


def test_exact_face_result_rejects_missing_stale_or_unrecorded_terminal_authority() -> None:
    component_name, smoothing = _scalar_efs_fit(0.5)
    evidence = smoothing.terminal_endpoint_directions[component_name]

    with pytest.raises(ValueError, match="fresh terminal endpoint directions"):
        replace(smoothing, terminal_endpoint_directions={})
    with pytest.raises(ValueError, match="terminal endpoint directions"):
        replace(
            smoothing,
            terminal_endpoint_directions={
                component_name: replace(
                    evidence,
                    endpoint_objective=np.nextafter(evidence.endpoint_objective, np.inf),
                )
            },
        )
    with pytest.raises(ValueError, match="revalidate an inactive component"):
        replace(
            smoothing,
            history=(
                replace(
                    smoothing.history[0],
                    activated_face_components=(),
                    revalidated_face_components=(component_name,),
                ),
                *smoothing.history[1:],
            ),
        )
    forged_fits = list(smoothing.coefficient_fits)
    activation = smoothing.history[0]
    assert activation.accepted_fit_index is not None
    forged_fits[activation.accepted_fit_index] = forged_fits[activation.coefficient_fit_indices[0]]
    with pytest.raises(ValueError, match="accepted coefficient face"):
        replace(smoothing, coefficient_fits=tuple(forged_fits))

    nonconverged_fits = list(smoothing.coefficient_fits)
    endpoint_index = smoothing.history[0].endpoint_direction_evidence.fit_indices[1]
    nonconverged_fits[endpoint_index] = replace(
        nonconverged_fits[endpoint_index],
        converged=False,
        convergence_reason="max_iterations",
    )
    with pytest.raises(ValueError, match="endpoint assessment fits must converge"):
        replace(smoothing, coefficient_fits=tuple(nonconverged_fits))

    fisher_fits = list(smoothing.coefficient_fits)
    evidence_indices = {
        index
        for item in smoothing.history
        if item.endpoint_direction_evidence is not None
        for index in item.endpoint_direction_evidence.fit_indices
    }
    for index in evidence_indices:
        fit = fisher_fits[index]
        fisher_fits[index] = replace(
            fit,
            terminal_curvature=replace(
                fit.terminal_curvature,
                requested_source="fisher",
                actual_source="fisher",
                reason=None,
                fallback_count=0,
            ),
        )
    with pytest.raises(ValueError, match="observed-curvature authority"):
        replace(smoothing, coefficient_fits=tuple(fisher_fits))

    terminal_fit = smoothing.terminal_fit
    failed_cap = replace(
        terminal_fit,
        converged=False,
        convergence_reason="max_iterations",
    )
    failed_cap_index = len(smoothing.coefficient_fits)
    appended_refusal = replace(
        smoothing.history[-1],
        iteration=smoothing.iterations + 1,
        source_fit_index=smoothing.terminal_fit_index,
        accepted=False,
        accepted_fit_index=None,
        accepted_curvature=None,
        coefficient_fit_indices=(failed_cap_index,),
        coefficient_tolerances=(failed_cap.config.tolerance,),
        activated_face_components=(),
        revalidated_face_components=(),
        refused_face_components=(component_name,),
        endpoint_direction_evidence=None,
        endpoint_assessment_failure_reason="cap_not_converged",
    )
    with pytest.raises(ValueError, match="terminal lambda-cap failure"):
        replace(
            smoothing,
            history=(*smoothing.history, appended_refusal),
            coefficient_fits=(*smoothing.coefficient_fits, failed_cap),
            iterations=smoothing.iterations + 1,
        )

    final_evidence = smoothing.terminal_endpoint_directions[component_name]
    activation_evidence = smoothing.history[0].endpoint_direction_evidence
    with pytest.raises(ValueError, match="final face revalidation"):
        replace(
            smoothing,
            terminal_endpoint_directions={
                component_name: replace(
                    final_evidence,
                    fit_indices=activation_evidence.fit_indices,
                    coefficient_tolerance=activation_evidence.coefficient_tolerance,
                )
            },
        )

    final_iteration = smoothing.history[-1]
    assert final_iteration.endpoint_direction_evidence is final_evidence
    cap_index, endpoint_index = final_evidence.fit_indices
    duplicated_endpoint_fits = list(smoothing.coefficient_fits)
    duplicated_endpoint_fits[endpoint_index] = duplicated_endpoint_fits[cap_index]
    with pytest.raises(
        ValueError,
        match="accepted coefficient face|endpoint assessment",
    ):
        replace(smoothing, coefficient_fits=tuple(duplicated_endpoint_fits))

    with pytest.raises(ValueError, match="cap and endpoint"):
        replace(final_evidence, fit_indices=(cap_index,))


def test_exact_face_result_rejects_a_cap_forged_under_a_large_common_core() -> None:
    """Kills letting absolute fit scale hide a materially better finite cap."""
    component_name, smoothing = _scalar_efs_fit(0.5)
    final_evidence = smoothing.terminal_endpoint_directions[component_name]
    cap_index = final_evidence.fit_indices[0]
    assessment_indices = {
        index for item in smoothing.history for index in item.coefficient_fit_indices
    }
    common_shift = -1.0e15

    def shifted_fit(
        fit: DenseSolverResult,
        *,
        terminal_perturbation: float = 0.0,
    ) -> DenseSolverResult:
        assert fit.optimizing_log_likelihood is not None
        assert fit.initial_penalized_optimizing_log_likelihood is not None
        optimizing = fit.optimizing_log_likelihood + common_shift + terminal_perturbation
        initial_optimizing = fit.initial_penalized_optimizing_log_likelihood + common_shift
        reported = optimizing + fit.parameter_independent_carrier
        penalized_optimizing = optimizing - fit.penalty_value
        return replace(
            fit,
            optimizing_log_likelihood=optimizing,
            log_likelihood=reported,
            penalized_optimizing_log_likelihood=penalized_optimizing,
            penalized_log_likelihood=reported - fit.penalty_value,
            initial_penalized_optimizing_log_likelihood=initial_optimizing,
            initial_penalized_log_likelihood=(
                initial_optimizing + fit.parameter_independent_carrier
            ),
        )

    forged_fits = list(smoothing.coefficient_fits)
    for index in assessment_indices:
        forged_fits[index] = shifted_fit(
            forged_fits[index],
            terminal_perturbation=100.0 if index == cap_index else 0.0,
        )

    with pytest.raises(ValueError, match="endpoint assessment"):
        replace(smoothing, coefficient_fits=tuple(forged_fits))


def test_exact_face_result_rejects_a_terminal_cap_that_beats_the_endpoint() -> None:
    """Kills accepting an endpoint that loses to its tightened finite reference."""
    component_name, smoothing = _scalar_efs_fit(0.5)
    evidence = smoothing.terminal_endpoint_directions[component_name]
    cap_index = evidence.fit_indices[0]
    cap_fit = smoothing.coefficient_fits[cap_index]
    assert cap_fit.optimizing_log_likelihood is not None
    optimizing = cap_fit.optimizing_log_likelihood + 100.0
    reported = optimizing + cap_fit.parameter_independent_carrier
    forged_cap = replace(
        cap_fit,
        optimizing_log_likelihood=optimizing,
        log_likelihood=reported,
        penalized_optimizing_log_likelihood=optimizing - cap_fit.penalty_value,
        penalized_log_likelihood=reported - cap_fit.penalty_value,
    )
    forged_fits = list(smoothing.coefficient_fits)
    forged_fits[cap_index] = forged_cap

    with pytest.raises(ValueError, match="endpoint assessment"):
        replace(smoothing, coefficient_fits=tuple(forged_fits))


def test_dense_result_rejects_coefficients_outside_its_exact_face() -> None:
    """Kills publishing unconstrained coefficients under certified face geometry."""
    _, smoothing = _scalar_efs_fit(0.5)
    terminal = smoothing.terminal_fit

    with pytest.raises(ValueError, match="coefficient face"):
        replace(terminal, coefficients=np.ones_like(terminal.coefficients))


def test_efs_retracts_a_face_when_terminal_endpoint_assessment_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_check = efs_module._check_face_direction

    def make_terminal_recheck_unavailable(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return real_check(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", make_terminal_recheck_unavailable)
    monkeypatch.setattr(efs_module, "_check_face_direction", make_terminal_recheck_unavailable)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert calls == 2
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.unresolved_upper_bound == (component_name,)
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_endpoint_directions == {}
    assert smoothing.history[0].activated_face_components == (component_name,)
    assert smoothing.history[-1].deactivated_face_components == (component_name,)
    assert smoothing.history[-1].endpoint_direction_evidence is None
    assert smoothing.matched_certified is False


@pytest.mark.parametrize(
    "terminal_failure",
    ["negative", "unresolved", "unavailable", "objective_failed"],
)
def test_terminal_face_revalidation_failure_stays_nonconverged_without_upper_pressure(
    monkeypatch: pytest.MonkeyPatch,
    terminal_failure: str,
) -> None:
    """Kills recovering from a failed terminal face certificate after rollback."""
    check_calls = 0
    saw_exact_face = False
    real_check = efs_module._check_face_direction
    real_fresh = efs_module._fresh_raw_evidence

    def fail_terminal_recheck(*args: object, **kwargs: object):
        nonlocal check_calls
        check_calls += 1
        attempt = real_check(*args, **kwargs)  # type: ignore[arg-type]
        if check_calls != 2:
            return attempt
        if terminal_failure == "unavailable":
            return None
        assert attempt.check is not None
        direction = attempt.check.direction
        if terminal_failure == "negative":
            derivative = -(direction.numerical_error + 1.0)
            direction = replace(
                direction,
                decision="finite",
                analytic_derivative=derivative,
                profile_score_term=2.0 * derivative,
                curvature_schur_term=0.0,
                curvature_drift_term=0.0,
                lower_bound=derivative - direction.numerical_error,
                upper_bound=derivative + direction.numerical_error,
            )
            return replace(attempt, check=replace(attempt.check, direction=direction))
        if terminal_failure == "unresolved":
            direction = replace(
                direction,
                decision="unresolved",
                analytic_derivative=0.0,
                profile_score_term=0.0,
                curvature_schur_term=0.0,
                curvature_drift_term=0.0,
                lower_bound=-direction.numerical_error,
                upper_bound=direction.numerical_error,
            )
            return replace(attempt, check=replace(attempt.check, direction=direction))
        assert terminal_failure == "objective_failed"
        return replace(
            attempt,
            check=replace(
                attempt.check,
                cap_objective=attempt.check.endpoint_objective - 1.0,
            ),
        )

    def clear_pressure_after_rollback(*args: object, **kwargs: object):
        nonlocal saw_exact_face
        evidence = real_fresh(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs.get("face") is not None:
            saw_exact_face = True
        elif saw_exact_face:
            return replace(
                evidence,
                maximum=0.0,
                working_infinity=(),
                unresolved_upper_bound=(),
            )
        return evidence

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", fail_terminal_recheck)
    monkeypatch.setattr(efs_module, "_check_face_direction", fail_terminal_recheck)
    monkeypatch.setattr(smoothing_loop, "_fresh_raw_evidence", clear_pressure_after_rollback)
    monkeypatch.setattr(efs_module, "_fresh_raw_evidence", clear_pressure_after_rollback)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert check_calls == 2
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.unresolved_upper_bound == ()
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_fit.converged is True
    assert smoothing.terminal_fit.config.tolerance == 1.0e-12
    assert smoothing.terminal_fit.config.coefficient_curvature == "observed"
    assert smoothing.terminal_fit.config.newton_decrement_tolerance is None
    assert np.all(np.isfinite(smoothing.terminal_fit.coefficients))
    assert smoothing.terminal_endpoint_directions == {}
    assert smoothing.history[-1].deactivated_face_components == (component_name,)
    assert smoothing.matched_certified is False
    with pytest.raises(RuntimeError, match="terminal exact-face revalidation"):
        smoothing.assert_matched_certified()
    with pytest.raises(ValueError, match="terminal exact-face deactivation"):
        replace(smoothing, converged=True, convergence_reason="lambda_change")


def test_endpoint_revalidation_failed_reason_requires_terminal_deactivation() -> None:
    _, smoothing = _scalar_efs_fit(0.5)
    assert smoothing.converged is True

    with pytest.raises(ValueError, match="requires a terminal exact-face deactivation"):
        replace(
            smoothing,
            converged=False,
            convergence_reason="endpoint_revalidation_failed",
        )


def test_efs_retracts_a_face_when_the_fresh_cap_has_a_better_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills revalidating from derivative sign without the fresh cap comparison."""
    calls = 0
    real_check = efs_module._check_face_direction

    def make_terminal_cap_better(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        attempt = real_check(*args, **kwargs)  # type: ignore[arg-type]
        if calls == 2 and attempt.check is not None:
            return replace(
                attempt,
                check=replace(
                    attempt.check,
                    cap_objective=attempt.check.endpoint_objective - 1.0,
                ),
            )
        return attempt

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", make_terminal_cap_better)
    monkeypatch.setattr(efs_module, "_check_face_direction", make_terminal_cap_better)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert calls == 2
    assert smoothing.converged is False
    assert smoothing.matched_certified is False
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.history[-1].deactivated_face_components == (component_name,)


def test_efs_retracts_a_face_when_terminal_direction_resolves_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A two-fit finite direction must coexist with the separate rollback fit."""
    calls = 0
    real_check = efs_module._check_face_direction

    def make_terminal_direction_finite(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        attempt = real_check(*args, **kwargs)  # type: ignore[arg-type]
        if calls != 2 or attempt.check is None:
            return attempt
        direction = attempt.check.direction
        derivative = -1.0
        error = direction.numerical_error
        finite = replace(
            direction,
            decision="finite",
            analytic_derivative=derivative,
            profile_score_term=2.0 * derivative,
            curvature_schur_term=0.0,
            curvature_drift_term=0.0,
            lower_bound=derivative - error,
            upper_bound=derivative + error,
        )
        return replace(attempt, check=replace(attempt.check, direction=finite))

    monkeypatch.setattr(smoothing_faces, "_check_face_direction", make_terminal_direction_finite)
    monkeypatch.setattr(efs_module, "_check_face_direction", make_terminal_direction_finite)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert calls == 2
    retraction = smoothing.history[-1]
    assert retraction.deactivated_face_components == (component_name,)
    assert retraction.endpoint_direction_evidence is not None
    assert retraction.endpoint_direction_evidence.decision == "finite"
    assert len(retraction.coefficient_fit_indices) == 3
    assert (
        retraction.endpoint_direction_evidence.fit_indices
        == (retraction.coefficient_fit_indices[:2])
    )
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_retracts_a_face_when_terminal_assessment_provenance_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance_calls = 0
    fit_calls = 0
    real_provenance = efs_module._endpoint_shared_provenance
    real_fit = efs_module._fit_fixed_state

    def change_terminal_assessment_provenance(result: DenseSolverResult) -> tuple[object, ...]:
        nonlocal provenance_calls
        provenance_calls += 1
        provenance = real_provenance(result)
        if provenance_calls == 4:
            return (*provenance, "changed-at-terminal-recheck")
        return provenance

    def count_fit(*args: object, **kwargs: object) -> DenseSolverResult:
        nonlocal fit_calls
        fit_calls += 1
        return real_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        smoothing_faces, "_endpoint_shared_provenance", change_terminal_assessment_provenance
    )
    monkeypatch.setattr(
        efs_module, "_endpoint_shared_provenance", change_terminal_assessment_provenance
    )
    monkeypatch.setattr(smoothing_authority, "_fit_fixed_state", count_fit)
    monkeypatch.setattr(smoothing_loop, "_fit_fixed_state", count_fit)
    monkeypatch.setattr(efs_module, "_fit_fixed_state", count_fit)
    component_name, smoothing = _scalar_efs_fit(0.5)

    assert provenance_calls == 4
    assert smoothing.converged is False
    assert smoothing.convergence_reason == "endpoint_revalidation_failed"
    assert smoothing.unresolved_upper_bound == (component_name,)
    assert smoothing.terminal_fit.coefficient_face is None
    assert smoothing.terminal_endpoint_directions == {}
    assert smoothing.history[-1].deactivated_face_components == (component_name,)
    assert smoothing.history[-1].endpoint_direction_evidence is None
    assert smoothing.matched_certified is False
    assert len(smoothing.coefficient_fits) == fit_calls + 1


@pytest.mark.parametrize("mean", [1.0, 1.01])
def test_efs_refuses_an_unresolved_or_finite_endpoint_beyond_the_cap(mean: float) -> None:
    component_name, smoothing = _scalar_efs_fit(mean)

    assert smoothing.converged is False
    assert smoothing.convergence_reason == "lambda_cap_unresolved"
    assert smoothing.unresolved_upper_bound == (component_name,)
    assert smoothing.terminal_fit.coefficient_face is None


def test_efs_retains_every_fit_from_a_rejected_endpoint_assessment() -> None:
    """Kills dropping a completed cap/endpoint assessment on refusal."""
    component_name, smoothing = _scalar_efs_fit(1.01)

    assert len(smoothing.history) == 1
    refusal = smoothing.history[0]
    assert refusal.accepted is False
    assert refusal.coefficient_fit_indices == tuple(range(1, len(smoothing.coefficient_fits)))
    assert len(refusal.coefficient_fit_indices) == 2
    assert refusal.refused_face_components == (component_name,)
    assert refusal.endpoint_direction_evidence is not None
    assert refusal.endpoint_assessment_failure_reason is None
    assert refusal.endpoint_direction_evidence.fit_indices == refusal.coefficient_fit_indices
    with pytest.raises(ValueError, match="failure reason|direction evidence"):
        replace(
            smoothing,
            history=(replace(refusal, endpoint_direction_evidence=None),),
        )


def test_endpoint_matches_an_independent_reduced_model_objective() -> None:
    response = np.array([1.3, -0.4, 2.2, 0.7, -1.1, 1.8, 0.2], dtype=np.float64)
    face_x = np.array([-1.5, -0.8, -0.2, 0.4, 0.9, 1.3, 1.8], dtype=np.float64)
    retained_x = np.array([0.2, 1.1, -0.7, 1.8, -1.4, 0.6, -0.3], dtype=np.float64)
    weight_values = np.array([0.7, 1.2, 0.9, 1.6, 1.1, 0.8, 1.4], dtype=np.float64)
    x = np.column_stack((np.ones(len(response)), face_x, retained_x))
    q = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    xq = x @ q
    retained_lambda = 1.6
    expected_penalty = np.diag([0.0, 0.0, retained_lambda])
    expected_reduced_penalty = q.T @ expected_penalty @ q
    expected_data_hessian = x.T @ (weight_values[:, None] * x)
    expected_penalized_hessian = expected_data_hessian + expected_penalty
    expected_reduced_hessian = xq.T @ (weight_values[:, None] * xq)
    expected_reduced_hessian += expected_reduced_penalty
    expected_gamma = np.linalg.solve(
        expected_reduced_hessian,
        xq.T @ (weight_values * response),
    )
    expected_beta = q @ expected_gamma
    expected_eta = x @ expected_beta
    residual = response - expected_eta
    expected_optimizing = -0.5 * float(np.dot(weight_values, residual**2))
    expected_penalty_value = 0.5 * float(expected_beta @ expected_penalty @ expected_beta)
    expected_penalized_optimizing = expected_optimizing - expected_penalty_value
    expected_carrier = 0.375 * len(response)
    assert expected_carrier != 0.0
    expected_reported = expected_optimizing + expected_carrier
    expected_penalized_reported = expected_reported - expected_penalty_value
    h_eigenvalues = np.linalg.eigvalsh(expected_reduced_hessian)
    h_cutoff = (
        len(expected_gamma)
        * np.finfo(np.float64).eps
        * float(np.linalg.norm(expected_reduced_hessian, ord=2))
    )
    retained_h = h_eigenvalues[h_eigenvalues > h_cutoff]
    expected_hessian_log_pdet = float(np.sum(np.log(retained_h)))
    s_eigenvalues = np.linalg.eigvalsh(expected_reduced_penalty)
    s_cutoff = (
        len(expected_gamma)
        * np.finfo(np.float64).eps
        * float(np.linalg.norm(expected_reduced_penalty, ord=2))
    )
    retained_s = s_eigenvalues[s_eigenvalues > s_cutoff]
    expected_penalty_log_pdet = float(np.sum(np.log(retained_s)))
    expected_objective = -expected_penalized_optimizing + 0.5 * (
        expected_hessian_log_pdet - expected_penalty_log_pdet
    )

    family = _UnitGaussian()
    weights = resolved_prior(weight_values)
    frame = as_eager_frame(pd.DataFrame({"face_x": face_x, "retained_x": retained_x}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"face_x": Numeric(), "retained_x": Numeric()}),),
        )
    )
    assert layout.coefficient_names == (
        "mean:(intercept)",
        "mean:face_x",
        "mean:retained_x",
    )
    face_component = PenaltyComponent(
        name="mean:face_x#identity",
        group_name="mean:face_x",
        group_index=0,
        group_sl=slice(1, 2),
        omega_raw=None,
        omega_ssp=None,
        rank=1.0,
        eigvals_omega=np.ones(1),
        penalty_kind="identity",
    )
    retained_component = PenaltyComponent(
        name="mean:retained_x#identity",
        group_name="mean:retained_x",
        group_index=1,
        group_sl=slice(2, 3),
        omega_raw=None,
        omega_ssp=None,
        rank=1.0,
        eigvals_omega=np.ones(1),
        penalty_kind="identity",
    )
    layout = replace(layout, penalties=(face_component, retained_component))
    face = build_penalty_face(layout, (face_component.name,))
    lambdas = {face_component.name: 37.0, retained_component.name: retained_lambda}
    config = DenseSolverConfig(tolerance=1.0e-12, max_iterations=80)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        family.bind_likelihood(response, weights, COMPLETE_OBSERVATION),
        expected_penalty,
        config=config,
        coefficient_face=face,
    )
    evaluation = evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    eps = np.finfo(np.float64).eps
    condition = float(np.linalg.cond(expected_reduced_hessian))
    coefficient_tolerance = (
        512.0
        * max(len(response), layout.n_coefficients)
        * eps
        * max(condition, 1.0)
        * max(float(np.linalg.norm(expected_beta)), 1.0)
        + 64.0 * config.tolerance
    )
    matrix_tolerance = (
        512.0
        * max(len(response), layout.n_coefficients)
        * eps
        * max(float(np.linalg.norm(expected_penalized_hessian, ord=2)), 1.0)
    )
    scalar_tolerance = coefficient_tolerance * max(abs(expected_objective), 1.0)
    np.testing.assert_allclose(
        result.coefficients, expected_beta, rtol=0.0, atol=coefficient_tolerance
    )
    np.testing.assert_allclose(result.eta[:, 0], expected_eta, rtol=0.0, atol=coefficient_tolerance)
    np.testing.assert_allclose(result.penalty, expected_penalty, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.terminal_data_curvature,
        expected_data_hessian,
        rtol=0.0,
        atol=matrix_tolerance,
    )
    np.testing.assert_allclose(
        result.terminal_penalized_curvature,
        expected_penalized_hessian,
        rtol=0.0,
        atol=matrix_tolerance,
    )
    assert result.optimizing_log_likelihood == pytest.approx(
        expected_optimizing, rel=0.0, abs=scalar_tolerance
    )
    assert result.parameter_independent_carrier == expected_carrier
    assert result.log_likelihood == pytest.approx(expected_reported, rel=0.0, abs=scalar_tolerance)
    assert result.penalty_value == pytest.approx(
        expected_penalty_value, rel=0.0, abs=scalar_tolerance
    )
    assert result.penalized_optimizing_log_likelihood == pytest.approx(
        expected_penalized_optimizing, rel=0.0, abs=scalar_tolerance
    )
    assert result.penalized_log_likelihood == pytest.approx(
        expected_penalized_reported, rel=0.0, abs=scalar_tolerance
    )
    assert result.terminal_reduced_rank is not None
    assert result.terminal_reduced_rank.rank == len(retained_h)
    assert result.terminal_reduced_rank.log_pdet == pytest.approx(
        expected_hessian_log_pdet, rel=0.0, abs=scalar_tolerance
    )
    assert evaluation.face_component_names == (face_component.name,)
    assert evaluation.finite_component_names == (retained_component.name,)
    assert evaluation.reduced_width == 2
    assert evaluation.hessian_rank == len(retained_h)
    assert evaluation.penalty_rank == len(retained_s)
    assert evaluation.hessian_log_pdet == pytest.approx(
        expected_hessian_log_pdet, rel=0.0, abs=scalar_tolerance
    )
    assert evaluation.penalty_log_pdet == pytest.approx(
        expected_penalty_log_pdet, rel=0.0, abs=scalar_tolerance
    )
    assert evaluation.objective == pytest.approx(expected_objective, rel=0.0, abs=scalar_tolerance)


def test_endpoint_refuses_result_from_a_different_face_instance() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    copied_face = replace(face)

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="supplied coefficient face",
    ):
        evaluate_endpoint_laplace(
            result,
            layout=layout,
            lambdas=lambdas,
            face=copied_face,
        )


def test_endpoint_refuses_penalty_from_stale_retained_lambdas() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    stale = dict(lambdas)
    stale["location:group#first"] = 2.0 * stale["location:group#first"]

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="endpoint lambdas",
    ):
        evaluate_endpoint_laplace(
            result,
            layout=layout,
            lambdas=stale,
            face=face,
        )


def test_endpoint_refuses_forged_reduced_terminal_decomposition() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    stored = result.terminal_reduced_rank
    assert stored is not None
    assert stored.cholesky_factor is not None
    wrong_inverse = replace(
        stored,
        cholesky_factor=np.sqrt(2.0) * stored.cholesky_factor,
    )
    for forged_rank in (
        decompose_gram(2.0 * np.eye(face.reduced_width)),
        replace(stored, policy_version=stored.policy_version + 1),
        replace(stored, resolution_limited=not stored.resolution_limited),
        wrong_inverse,
    ):
        forged = replace(result, terminal_reduced_rank=forged_rank)
        with pytest.raises(
            endpoint_laml_module.EndpointLaplaceError,
            match="reduced terminal decomposition",
        ):
            evaluate_endpoint_laplace(
                forged,
                layout=layout,
                lambdas=lambdas,
                face=face,
            )


def test_endpoint_contains_out_of_range_active_column_as_provenance_refusal() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    stored = result.terminal_reduced_rank
    assert stored is not None
    assert stored.cholesky_factor is not None
    active_columns = np.array(stored.active_columns, copy=True)
    active_columns[0] = stored.width + 3
    forged_rank = replace(stored, active_columns=active_columns)
    forged = replace(result, terminal_reduced_rank=forged_rank)

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="reduced terminal decomposition",
    ) as error:
        evaluate_endpoint_laplace(
            forged,
            layout=layout,
            lambdas=lambdas,
            face=face,
        )

    assert isinstance(error.value.__cause__, IndexError)


@pytest.mark.parametrize(
    ("field_name", "malformed_kind", "cause_type"),
    [
        pytest.param("log_pdet", "numeric-string", TypeError, id="log-numeric-string"),
        pytest.param("log_pdet", "text-string", TypeError, id="log-text-string"),
        pytest.param("log_pdet", "bool", TypeError, id="log-bool"),
        pytest.param("log_pdet", "nan", ValueError, id="log-nan"),
        pytest.param("log_pdet", "inf", ValueError, id="log-inf"),
        pytest.param("rank", "numeric-string", TypeError, id="rank-numeric-string"),
        pytest.param("rank", "bool", TypeError, id="rank-bool"),
        pytest.param("rank", "nan", TypeError, id="rank-nan"),
    ],
)
def test_endpoint_refuses_malformed_stored_rank_and_log_scalars(
    field_name: str,
    malformed_kind: str,
    cause_type: type[Exception],
) -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    stored = result.terminal_reduced_rank
    assert stored is not None
    malformed = {
        "numeric-string": str(getattr(stored, field_name)),
        "text-string": "not-a-number",
        "bool": True,
        "nan": np.nan,
        "inf": np.inf,
    }[malformed_kind]
    forged_rank = replace(stored, **{field_name: malformed})
    forged = replace(result, terminal_reduced_rank=forged_rank)

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="reduced terminal decomposition",
    ) as error:
        evaluate_endpoint_laplace(
            forged,
            layout=layout,
            lambdas=lambdas,
            face=face,
        )

    assert isinstance(error.value.__cause__, cause_type)


def test_endpoint_objective_uses_the_validated_stored_terminal_log_pdet() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    stored = result.terminal_reduced_rank
    assert stored is not None
    perturbation = (
        8.0 * max(stored.width, 1) * np.finfo(np.float64).eps * max(abs(stored.log_pdet), 1.0)
    )
    perturbed_rank = replace(stored, log_pdet=stored.log_pdet + perturbation)
    perturbed_result = replace(result, terminal_reduced_rank=perturbed_rank)

    evaluation = evaluate_endpoint_laplace(
        perturbed_result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert evaluation.hessian_log_pdet == perturbed_rank.log_pdet
    assert result.penalized_optimizing_log_likelihood is not None
    expected = -result.penalized_optimizing_log_likelihood + 0.5 * (
        perturbed_rank.log_pdet - evaluation.penalty_log_pdet
    )
    assert evaluation.objective == expected


@pytest.mark.parametrize("mean", [1.01, 3.0])
def test_endpoint_is_invariant_to_face_lambda_values(mean: float) -> None:
    layout, face, lambdas, result = _scalar_endpoint_fit(mean)
    face_name = face.component_names[0]
    low = evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas={**lambdas, face_name: 1.0e-12},
        face=face,
    )
    high = evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas={**lambdas, face_name: 1.0e120},
        face=face,
    )

    assert low == high
    assert high.objective == pytest.approx(mean**2 / 2.0, rel=0.0, abs=1e-15)
    assert not {field.name for field in fields(high)}.intersection(
        {"converged", "convergence_reason", "boundary", "boundary_decision"}
    )


def test_endpoint_is_invariant_to_a_refitted_rotated_face_basis() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    assert face.reduced_width >= 2
    rotation = scipy.linalg.block_diag(
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.eye(face.reduced_width - 2),
    )
    rotated_face = replace(face, null_basis=face.null_basis @ rotation)
    original_result = _fit_gaussian_endpoint(
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
    )
    rotated_result = _fit_gaussian_endpoint(
        family,
        layout,
        rotated_face,
        lambdas,
        response,
        weights,
    )

    original = evaluate_endpoint_laplace(
        original_result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )
    rotated = evaluate_endpoint_laplace(
        rotated_result,
        layout=layout,
        lambdas=lambdas,
        face=rotated_face,
    )

    eps = np.finfo(np.float64).eps
    condition = max(
        original_result.terminal_reduced_rank.pre_truncation_condition,  # type: ignore[union-attr]
        rotated_result.terminal_reduced_rank.pre_truncation_condition,  # type: ignore[union-attr]
        1.0,
    )
    tolerance = (
        2048.0
        * max(layout.n_coefficients, face.reduced_width, 1)
        * eps
        * condition
        * max(abs(original.objective), 1.0)
    )
    np.testing.assert_allclose(
        rotated_result.theta,
        original_result.theta,
        rtol=0.0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        rotated.objective,
        original.objective,
        rtol=0.0,
        atol=tolerance,
    )


def test_endpoint_supports_an_observed_only_family() -> None:
    (
        _base,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    family = _ObservedOnlyGaussian()
    result = _fit_gaussian_endpoint(
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        coefficient_curvature="observed",
    )

    evaluation = evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert result.terminal_curvature.actual_source == "observed"
    assert np.isfinite(evaluation.objective)
    assert evaluation.hessian_rank == result.terminal_reduced_rank.rank  # type: ignore[union-attr]


def test_endpoint_does_not_mutate_fit_or_smoothing_state() -> None:
    (
        family,
        layout,
        face,
        lambdas,
        response,
        weights,
        _first_matrix,
        _second_matrix,
    ) = _gaussian_endpoint_problem()
    result = _fit_gaussian_endpoint(family, layout, face, lambdas, response, weights)
    smoothing_snapshot = dict(lambdas)
    coefficients_snapshot = np.array(result.coefficients, copy=True)
    penalty_snapshot = np.array(result.penalty, copy=True)
    curvature_snapshot = np.array(result.terminal_penalized_curvature, copy=True)
    stored_rank = result.terminal_reduced_rank

    evaluate_endpoint_laplace(
        result,
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert lambdas == smoothing_snapshot
    np.testing.assert_array_equal(result.coefficients, coefficients_snapshot)
    np.testing.assert_array_equal(result.penalty, penalty_snapshot)
    np.testing.assert_array_equal(result.terminal_penalized_curvature, curvature_snapshot)
    assert result.terminal_reduced_rank is stored_rank


def test_endpoint_value_is_immutable_and_refuses_malformed_values() -> None:
    evaluation = EndpointLaplaceEvaluation(
        objective=np.float64(2.0),
        face_component_names=["face"],  # type: ignore[arg-type]
        finite_component_names=["finite"],  # type: ignore[arg-type]
        reduced_width=np.int64(2),  # type: ignore[arg-type]
        hessian_rank=np.int64(2),  # type: ignore[arg-type]
        penalty_rank=np.int64(1),  # type: ignore[arg-type]
        hessian_log_pdet=np.float64(3.0),
        penalty_log_pdet=np.float64(1.0),
    )

    assert evaluation.face_component_names == ("face",)
    assert evaluation.finite_component_names == ("finite",)
    assert isinstance(evaluation.objective, float)
    with pytest.raises(FrozenInstanceError):
        evaluation.objective = 1.0  # type: ignore[misc]
    malformed = (
        {"objective": np.inf},
        {"face_component_names": "face"},
        {"face_component_names": ("face", "face")},
        {"finite_component_names": ("face",)},
        {"reduced_width": -1},
        {"hessian_rank": 3},
        {"penalty_rank": 3},
        {"hessian_rank": 0, "hessian_log_pdet": 1.0},
        {"penalty_rank": 0, "penalty_log_pdet": 1.0},
    )
    baseline = {
        "objective": 2.0,
        "face_component_names": ("face",),
        "finite_component_names": ("finite",),
        "reduced_width": 2,
        "hessian_rank": 2,
        "penalty_rank": 1,
        "hessian_log_pdet": 3.0,
        "penalty_log_pdet": 1.0,
    }
    for changes in malformed:
        with pytest.raises((TypeError, ValueError)):
            EndpointLaplaceEvaluation(**{**baseline, **changes})


def test_endpoint_refuses_non_solver_results() -> None:
    layout, face, lambdas, _result = _scalar_endpoint_fit(0.5)

    with pytest.raises(TypeError, match="DenseSolverResult"):
        evaluate_endpoint_laplace(  # type: ignore[arg-type]
            object(),
            layout=layout,
            lambdas=lambdas,
            face=face,
        )


def test_projected_finite_penalty_logdet_matches_direct_reduced_sum() -> None:
    layout, face, lambdas, first_matrix, second_matrix = _projected_penalty_problem()
    expected_full_finite_penalty = np.zeros(
        (layout.n_coefficients, layout.n_coefficients),
        dtype=np.float64,
    )
    retained_slice = layout.term_slices["location:group"]
    expected_full_finite_penalty[retained_slice, retained_slice] = (
        1.75 * first_matrix + 0.6 * second_matrix
    )
    q = face.null_basis
    expected_reduced = q.T @ expected_full_finite_penalty @ q
    expected_eigenvalues = np.linalg.eigvalsh(expected_reduced)
    cutoff = (
        max(expected_reduced.shape[0], 1)
        * np.finfo(np.float64).eps
        * max(
            float(np.max(np.abs(expected_eigenvalues), initial=0.0)),
            1.0,
        )
    )
    retained = expected_eigenvalues[expected_eigenvalues > cutoff]
    expected_log_pdet = float(np.sum(np.log(retained)))
    production_rank_resolution = np.finfo(np.float64).eps ** (2.0 / 3.0) * max(
        float(np.max(expected_eigenvalues, initial=0.0)),
        1e-12,
    )
    assert retained[0] >= 1e6 * max(cutoff, production_rank_resolution)

    result = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert result.component_names == (
        "location:group#first",
        "location:group#second",
    )
    assert result.rank == len(retained)
    condition = float(retained[-1] / retained[0])
    tolerance = (
        512.0
        * max(expected_reduced.shape[0], 1)
        * np.finfo(np.float64).eps
        * max(condition, 1.0)
        * max(abs(expected_log_pdet), 1.0)
    )
    np.testing.assert_allclose(
        result.log_pdet,
        expected_log_pdet,
        rtol=0.0,
        atol=tolerance,
    )


def test_projected_penalty_logdet_resolves_independent_groups_separately() -> None:
    """Kills erasing a valid small independent block with a global cutoff."""
    layout, face, lambdas = _independent_projected_penalty_problem()

    result = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    expected_log_pdet = -12.0 * np.log(10.0)
    assert result.component_names == (
        "location:x#identity",
        "location:z#identity",
    )
    assert result.rank == 2
    assert result.log_pdet == pytest.approx(
        expected_log_pdet,
        rel=0.0,
        abs=128.0 * np.finfo(np.float64).eps * abs(expected_log_pdet),
    )


@pytest.mark.parametrize(
    "operation",
    [projected_finite_penalty_logdet, projected_component_states],
    ids=["logdet", "component-traces"],
)
def test_projected_penalty_apis_refuse_overlapping_independent_group_metadata(
    operation,
) -> None:
    """Kills trusting distinct group names whose coefficient blocks overlap."""
    layout, face, lambdas = _independent_projected_penalty_problem()
    face_component, x_component, z_component = layout.penalties
    overlapping_z = replace(z_component, group_sl=x_component.group_sl)
    malformed_layout = replace(
        layout,
        penalties=(face_component, x_component, overlapping_z),
    )

    with pytest.raises(endpoint_laml_module.EndpointLaplaceError, match="coefficient blocks"):
        operation(layout=malformed_layout, lambdas=lambdas, face=face)


def test_projected_penalty_refuses_nonzero_logdet_for_a_zero_rank_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills allowing another group to mask an invalid rank/logdet pair."""
    layout, face, lambdas = _independent_projected_penalty_problem()
    real_logdet = endpoint_laml_module.similarity_transform_logdet
    calls = 0

    def malformed_second_group(
        penalty_matrices: list[np.ndarray],
        finite_lambdas: np.ndarray,
    ) -> SimilarityTransformResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_logdet(penalty_matrices, finite_lambdas)
        width = penalty_matrices[0].shape[0]
        zeros = np.zeros((width, width), dtype=np.float64)
        return SimilarityTransformResult(
            logdet_s_plus=1.0,
            S_pinv_plus=zeros,
            Q_plus=np.zeros((width, 0), dtype=np.float64),
            Q_zero=np.eye(width, dtype=np.float64),
            E_sqrt=zeros,
            rank=0,
        )

    monkeypatch.setattr(
        endpoint_laml_module,
        "similarity_transform_logdet",
        malformed_second_group,
    )

    with pytest.raises(ValueError, match="zero rank"):
        projected_finite_penalty_logdet(layout=layout, lambdas=lambdas, face=face)


def test_projected_penalty_refuses_materially_indefinite_retained_component() -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    indefinite = _dense_component(
        retained.name,
        retained.group_name,
        retained.group_index,
        retained.group_sl,
        np.diag([2.0, -1.0, 0.0]),
        rank=1.0,
    )
    indefinite_layout = replace(layout, penalties=(face_component, indefinite))

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="positive semidefinite",
    ) as error:
        projected_finite_penalty_logdet(
            layout=indefinite_layout,
            lambdas={face_component.name: 13.0, indefinite.name: 1.0},
            face=face,
        )

    assert isinstance(error.value.__cause__, ValueError)


def test_projected_penalty_accepts_unresolved_negative_roundoff_under_shared_policy() -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    unresolved_negative = -16.0 * np.finfo(np.float64).eps
    roundoff_matrix = np.diag([2.0, unresolved_negative, 0.0])
    roundoff_component = _dense_component(
        retained.name,
        retained.group_name,
        retained.group_index,
        retained.group_sl,
        roundoff_matrix,
        rank=1.0,
    )
    roundoff_layout = replace(
        layout,
        penalties=(face_component, roundoff_component),
    )
    aligned_face, _retained_reduced_indices = _axis_aligned_projected_face(
        roundoff_layout,
        face,
        roundoff_component,
    )
    full = np.zeros(
        (layout.n_coefficients, layout.n_coefficients),
        dtype=np.float64,
    )
    full[retained.group_sl, retained.group_sl] = roundoff_matrix
    reduced = aligned_face.null_basis.T @ full @ aligned_face.null_basis
    shared_decomposition = decompose_gram(reduced)

    result = projected_finite_penalty_logdet(
        layout=roundoff_layout,
        lambdas={face_component.name: 13.0, roundoff_component.name: 1.0},
        face=aligned_face,
    )

    assert unresolved_negative < 0.0
    assert shared_decomposition.rank == 1
    assert result.rank == shared_decomposition.rank
    expected_log_pdet = np.log(2.0)
    tolerance = (
        512.0
        * max(aligned_face.reduced_width, 1)
        * np.finfo(np.float64).eps
        * max(abs(expected_log_pdet), 1.0)
    )
    assert result.log_pdet == pytest.approx(expected_log_pdet, rel=0.0, abs=tolerance)


def test_projected_penalty_symmetrizes_finite_large_projection_without_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    large_value = 0.75 * np.finfo(np.float64).max
    large_matrix = np.diag([large_value, 1.0, 0.0])
    large_component = replace(
        retained,
        omega_raw=large_matrix,
        omega_ssp=large_matrix,
    )
    large_layout = replace(layout, penalties=(face_component, large_component))
    aligned_face, retained_reduced_indices = _axis_aligned_projected_face(
        large_layout,
        face,
        large_component,
    )
    captured: list[np.ndarray] | None = None
    width = aligned_face.reduced_width
    q_plus = np.eye(width, dtype=np.float64)[:, :1]
    q_zero = np.eye(width, dtype=np.float64)[:, 1:]

    def capture_logdet(
        penalty_matrices: list[np.ndarray],
        _finite_lambdas: np.ndarray,
    ) -> SimilarityTransformResult:
        nonlocal captured
        captured = [np.array(matrix, copy=True) for matrix in penalty_matrices]
        return SimilarityTransformResult(
            logdet_s_plus=0.0,
            S_pinv_plus=q_plus @ q_plus.T,
            Q_plus=q_plus,
            Q_zero=q_zero,
            E_sqrt=q_plus @ q_plus.T,
            rank=1,
        )

    monkeypatch.setattr(endpoint_laml_module, "similarity_transform_logdet", capture_logdet)

    result = projected_finite_penalty_logdet(
        layout=large_layout,
        lambdas={face_component.name: 13.0, large_component.name: 0.0},
        face=aligned_face,
    )

    assert result.rank == 1
    assert captured is not None
    assert len(captured) == 1
    assert np.all(np.isfinite(captured[0]))
    assert large_value > np.finfo(np.float64).max / 2.0
    first_retained = retained_reduced_indices[0]
    assert captured[0][first_retained, first_retained] == large_value


def test_projected_penalty_refuses_materially_asymmetric_retained_component() -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    asymmetric_matrix = np.diag([2.0, 1.0, 0.5])
    asymmetric_matrix[0, 1] = 0.25
    asymmetric = _dense_component(
        retained.name,
        retained.group_name,
        retained.group_index,
        retained.group_sl,
        asymmetric_matrix,
        rank=3.0,
    )
    asymmetric_layout = replace(layout, penalties=(face_component, asymmetric))

    with pytest.raises(endpoint_laml_module.EndpointLaplaceError, match="symmetric"):
        projected_finite_penalty_logdet(
            layout=asymmetric_layout,
            lambdas={face_component.name: 13.0, asymmetric.name: 1.0},
            face=face,
        )


@pytest.mark.parametrize(
    "nonfinite", [pytest.param(np.nan, id="nan"), pytest.param(np.inf, id="inf")]
)
def test_projected_penalty_contains_nonfinite_local_retained_component(
    nonfinite: float,
) -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    malformed_matrix = np.asarray(retained.omega_ssp, dtype=np.float64).copy()
    malformed_matrix[0, 0] = nonfinite
    malformed = replace(
        retained,
        omega_raw=malformed_matrix,
        omega_ssp=malformed_matrix,
    )
    malformed_layout = replace(layout, penalties=(face_component, malformed))

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="retained penalty .* contains non-finite local values",
    ) as error:
        projected_finite_penalty_logdet(
            layout=malformed_layout,
            lambdas={face_component.name: 13.0, malformed.name: 1.0},
            face=face,
        )

    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "array must not contain infs or NaNs"


def test_projected_penalty_contains_finite_input_projection_overflow() -> None:
    layout, face, _lambdas, _, _ = _projected_penalty_problem()
    face_component = next(
        component for component in layout.penalties if component.name in face.component_names
    )
    retained = next(
        component for component in layout.penalties if component.name not in face.component_names
    )
    aligned_face, retained_reduced_indices = _axis_aligned_projected_face(
        layout,
        face,
        retained,
    )
    rotated_indices = retained_reduced_indices[:2]
    assert len(rotated_indices) == 2
    rotation = np.eye(aligned_face.reduced_width, dtype=np.float64)
    cosine = 1.0 / np.sqrt(2.0)
    rotation[np.ix_(rotated_indices, rotated_indices)] = np.array(
        [[cosine, -cosine], [cosine, cosine]],
        dtype=np.float64,
    )
    rotated_face = replace(
        aligned_face,
        null_basis=aligned_face.null_basis @ rotation,
    )
    finite_matrix = np.zeros((3, 3), dtype=np.float64)
    finite_matrix[:2, :2] = 0.75 * np.finfo(np.float64).max
    overflowing = replace(
        retained,
        omega_raw=finite_matrix,
        omega_ssp=finite_matrix,
    )
    overflowing_layout = replace(layout, penalties=(face_component, overflowing))

    assert np.all(np.isfinite(finite_matrix))
    full = np.zeros(
        (layout.n_coefficients, layout.n_coefficients),
        dtype=np.float64,
    )
    full[overflowing.group_sl, overflowing.group_sl] = finite_matrix
    with np.errstate(invalid="ignore", over="ignore"):
        direct_projection = rotated_face.null_basis.T @ full @ rotated_face.null_basis
    assert not np.all(np.isfinite(direct_projection))

    with pytest.raises(
        endpoint_laml_module.EndpointLaplaceError,
        match="projection of retained penalty .* produced non-finite values",
    ) as error:
        projected_finite_penalty_logdet(
            layout=overflowing_layout,
            lambdas={face_component.name: 13.0, overflowing.name: 1.0},
            face=rotated_face,
        )

    assert isinstance(error.value.__cause__, FloatingPointError)
    assert str(error.value.__cause__) == "retained penalty projection is non-finite"


@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        pytest.param("materialization", TypeError, id="materialization-type-error"),
        pytest.param("norm", RuntimeError, id="norm-runtime-error"),
        pytest.param("decomposition", ValueError, id="decomposition-value-error"),
        pytest.param("decomposition", RuntimeError, id="decomposition-runtime-error"),
    ],
)
def test_projected_penalty_does_not_reclassify_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[Exception],
) -> None:
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    sentinel = error_type(f"{stage} programming sentinel")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise sentinel

    if stage == "materialization":
        monkeypatch.setattr(endpoint_laml_module, "penalty_component_dense_matrix", fail)
    elif stage == "norm":
        monkeypatch.setattr(endpoint_laml_module, "_spectral_norm", fail)
    else:
        monkeypatch.setattr(endpoint_laml_module, "decompose_gram", fail)

    with pytest.raises(error_type, match="programming sentinel") as error:
        projected_finite_penalty_logdet(
            layout=layout,
            lambdas=lambdas,
            face=face,
        )

    assert error.value is sentinel


def test_projected_penalty_projects_each_component_before_logdet_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, face, lambdas, first_matrix, second_matrix = _projected_penalty_problem()
    captured_matrices: list[np.ndarray] | None = None
    captured_lambdas: np.ndarray | None = None
    real_logdet = endpoint_laml_module.similarity_transform_logdet

    def capture_logdet(
        penalty_matrices: list[np.ndarray],
        finite_lambdas: np.ndarray,
    ) -> SimilarityTransformResult:
        nonlocal captured_matrices, captured_lambdas
        captured_matrices = [np.array(matrix, copy=True) for matrix in penalty_matrices]
        captured_lambdas = np.array(finite_lambdas, copy=True)
        return real_logdet(penalty_matrices, finite_lambdas)

    monkeypatch.setattr(endpoint_laml_module, "similarity_transform_logdet", capture_logdet)

    result = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )

    assert result.component_names == (
        "location:group#first",
        "location:group#second",
    )
    assert captured_matrices is not None
    assert captured_lambdas is not None
    reduced_shape = (face.reduced_width, face.reduced_width)
    assert [matrix.shape for matrix in captured_matrices] == [
        reduced_shape,
        reduced_shape,
    ]
    retained_slice = layout.term_slices["location:group"]
    first_full = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    first_full[retained_slice, retained_slice] = first_matrix
    second_full = np.zeros((layout.n_coefficients, layout.n_coefficients), dtype=np.float64)
    second_full[retained_slice, retained_slice] = second_matrix
    q = face.null_basis
    expected_projected = (
        q.T @ first_full @ q,
        q.T @ second_full @ q,
    )
    eps = np.finfo(np.float64).eps
    for actual, expected in zip(captured_matrices, expected_projected, strict=True):
        tolerance = (
            256.0
            * max(layout.n_coefficients, face.reduced_width, 1)
            * eps
            * max(float(np.linalg.norm(expected, ord=2)), 1.0)
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)
    np.testing.assert_array_equal(captured_lambdas, np.array([1.75, 0.6]))


def test_projected_penalty_is_invariant_to_face_lambda_values() -> None:
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    face_name = face.component_names[0]
    zero_face_lambda = {**lambdas, face_name: 0.0}
    large_face_lambda = {**lambdas, face_name: 1e150}

    zero_result = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=zero_face_lambda,
        face=face,
    )
    large_result = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=large_face_lambda,
        face=face,
    )

    assert zero_result == large_result


def test_projected_penalty_handles_no_retained_components() -> None:
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    face_only_layout = replace(
        layout,
        penalties=tuple(
            component for component in layout.penalties if component.name in face.component_names
        ),
    )
    face_only_lambdas = {name: lambdas[name] for name in face.component_names}

    result = projected_finite_penalty_logdet(
        layout=face_only_layout,
        lambdas=face_only_lambdas,
        face=face,
    )

    assert result == ProjectedPenaltyLogDet(
        component_names=(),
        rank=0,
        log_pdet=0.0,
    )


def test_projected_penalty_handles_zero_reduced_width() -> None:
    layout, _, _, _, _ = _projected_penalty_problem()
    components = tuple(
        PenaltyComponent(
            name=f"{coefficient_name}#face-{index}",
            group_name=coefficient_name,
            group_index=index,
            group_sl=slice(index, index + 1),
            omega_raw=None,
            omega_ssp=None,
            rank=1.0,
            eigvals_omega=np.ones(1),
            penalty_kind="identity",
        )
        for index, coefficient_name in enumerate(layout.coefficient_names)
    )
    constrained_layout = replace(layout, penalties=components)
    full_face = build_penalty_face(constrained_layout, constrained_layout.penalty_names)

    result = projected_finite_penalty_logdet(
        layout=constrained_layout,
        lambdas={
            name: float(index + 1) for index, name in enumerate(constrained_layout.penalty_names)
        },
        face=full_face,
    )

    assert full_face.reduced_width == 0
    assert result == ProjectedPenaltyLogDet(
        component_names=(),
        rank=0,
        log_pdet=0.0,
    )


def test_projected_penalty_is_invariant_to_orthonormal_face_basis_rotation() -> None:
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    assert face.reduced_width >= 2
    rotation = scipy.linalg.block_diag(
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.eye(face.reduced_width - 2),
    )
    rotated_q = face.null_basis @ rotation
    rotated_face = replace(face, null_basis=rotated_q)

    original = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=lambdas,
        face=face,
    )
    rotated = projected_finite_penalty_logdet(
        layout=layout,
        lambdas=lambdas,
        face=rotated_face,
    )

    assert rotated.rank == original.rank
    np.testing.assert_allclose(
        rotated.log_pdet,
        original.log_pdet,
        rtol=0.0,
        atol=512.0 * face.reduced_width * np.finfo(np.float64).eps,
    )


def test_projected_penalty_refuses_incomplete_unknown_or_invalid_lambdas() -> None:
    layout, face, lambdas, _, _ = _projected_penalty_problem()
    retained_name = next(name for name in layout.penalty_names if name not in face.component_names)
    incomplete = dict(lambdas)
    incomplete.pop(retained_name)
    unknown = {**lambdas, "location:group#unknown": 1.0}

    with pytest.raises(ValueError, match="missing"):
        projected_finite_penalty_logdet(
            layout=layout,
            lambdas=incomplete,
            face=face,
        )
    with pytest.raises(ValueError, match="unknown"):
        projected_finite_penalty_logdet(
            layout=layout,
            lambdas=unknown,
            face=face,
        )
    for invalid in (np.nan, np.inf, -1.0, True, "1.0"):
        with pytest.raises((TypeError, ValueError), match="finite|nonnegative"):
            projected_finite_penalty_logdet(
                layout=layout,
                lambdas={**lambdas, retained_name: invalid},
                face=face,
            )


def test_projected_penalty_result_is_immutable_and_refuses_malformed_values() -> None:
    result = ProjectedPenaltyLogDet(
        component_names=["location:group#first"],  # ty: ignore[invalid-argument-type]
        rank=np.int64(1),  # ty: ignore[invalid-argument-type]
        log_pdet=np.float64(2.0),
    )

    assert result.component_names == ("location:group#first",)
    assert isinstance(result.rank, int)
    assert isinstance(result.log_pdet, float)
    with pytest.raises(FrozenInstanceError):
        result.rank = 0  # ty: ignore[invalid-assignment]
    with pytest.raises((TypeError, ValueError)):
        ProjectedPenaltyLogDet(component_names=("duplicate", "duplicate"), rank=1, log_pdet=0.0)
    with pytest.raises((TypeError, ValueError)):
        ProjectedPenaltyLogDet(component_names=(), rank=1, log_pdet=0.0)
    with pytest.raises((TypeError, ValueError)):
        ProjectedPenaltyLogDet(component_names=("retained",), rank=-1, log_pdet=0.0)
    with pytest.raises((TypeError, ValueError)):
        ProjectedPenaltyLogDet(component_names=("retained",), rank=0, log_pdet=1.0)
    with pytest.raises((TypeError, ValueError)):
        ProjectedPenaltyLogDet(component_names=("retained",), rank=1, log_pdet=np.nan)
