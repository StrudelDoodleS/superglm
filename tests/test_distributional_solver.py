from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, dataclass, replace

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.solver.assembly as assembly_module
import superglm.distributional.solver.solver as solver_module
from superglm._frame import as_eager_frame
from superglm.distributional.curvature import CurvaturePolicyError
from superglm.distributional.efs import joint_laplace_objective
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    FamilyLikelihoodPlan,
    NaturalLikelihoodEvaluation,
    ObservationContract,
)
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.solver import DenseSolverConfig, fit_dense_fixed_lambda
from superglm.distributional.weights import (
    ResolvedLikelihoodWeights,
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.features import Numeric
from superglm.group_matrix import DesignMatrix

from ._distributional_weights import resolved_prior


def _intercept_layout(
    n: int,
    *,
    location_offset: np.ndarray | None = None,
    scale_offset: np.ndarray | None = None,
):
    frame = as_eager_frame(pd.DataFrame({"row": np.arange(n, dtype=float)}))
    family = GaussianLS()
    builds = compile_predictors(
        frame,
        resolved_prior(np.ones(n)),
        family.parameters,
        (
            Predictor("location", {}),
            Predictor("scale", {}),
        ),
        offsets={
            "location": np.zeros(n) if location_offset is None else location_offset,
            "scale": np.zeros(n) if scale_offset is None else scale_offset,
        },
    )
    return family, build_stacked_layout(builds)


def _plan(family, y: np.ndarray, weights: np.ndarray, *, semantics: str = "prior"):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(y),
        contract=WeightContract(semantics=semantics),  # type: ignore[arg-type]
    )
    return family.bind_likelihood(y, resolved, COMPLETE_OBSERVATION)


@dataclass(frozen=True)
class _CarrierPlan:
    base: FamilyLikelihoodPlan
    carrier_per_row: float

    @property
    def weights(self) -> ResolvedLikelihoodWeights:
        return self.base.weights

    @property
    def plan_identifier(self) -> str:
        return f"carrier:{self.carrier_per_row}:{self.base.plan_identifier}"

    def take(self, indices: np.ndarray) -> _CarrierPlan:
        return _CarrierPlan(self.base.take(indices), self.carrier_per_row)


class _ConstantCarrierGaussian:
    def __init__(self, carrier_per_row: float) -> None:
        self.base = GaussianLS()
        self.carrier_per_row = carrier_per_row

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(
        self,
        y: np.ndarray,
        weights: ResolvedLikelihoodWeights,
        observation: ObservationContract,
    ) -> _CarrierPlan:
        return _CarrierPlan(
            self.base.bind_likelihood(y, weights, observation),
            self.carrier_per_row,
        )

    def initialize(self, y, plan):
        assert isinstance(plan, _CarrierPlan)
        return self.base.initialize(y, plan.base)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        assert isinstance(plan, _CarrierPlan)
        evaluation = self.base.evaluate_natural(
            y,
            theta,
            plan.base,
            derivative_order=derivative_order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluation.optimizing_log_likelihood,
            parameter_independent_carrier=(
                evaluation.parameter_independent_carrier + plan.carrier_per_row
            ),
            score=evaluation.score,
            hessian_packed=evaluation.hessian_packed,
            valid=evaluation.valid,
        )

    def expected_information_natural(self, theta, plan):
        assert isinstance(plan, _CarrierPlan)
        return self.base.expected_information_natural(theta, plan.base)

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _ObservedOnlyGaussian:
    """Delegate a valid order-two likelihood without exposing Fisher information."""

    def __init__(self, base: GaussianLS) -> None:
        self.base = base
        self.initializations = 0
        self.evaluations = 0

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def default_prediction_name(self):
        return self.base.default_prediction_name

    def bind_likelihood(self, y, weights, observation):
        return self.base.bind_likelihood(y, weights, observation)

    def initialize(self, y, plan):
        self.initializations += 1
        return self.base.initialize(y, plan)

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        self.evaluations += 1
        return self.base.evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )

    def default_prediction(self, theta):
        return self.base.default_prediction(theta)


class _ExpectedInformationSpyGaussian(_ObservedOnlyGaussian):
    """Expose Fisher information while recording whether the solver requests it."""

    def __init__(self, base: GaussianLS) -> None:
        super().__init__(base)
        self.expected_information_calls = 0

    def expected_information_natural(self, theta, plan):
        self.expected_information_calls += 1
        return self.base.expected_information_natural(theta, plan)


class _TypedRefusalGaussian(_ConstantCarrierGaussian):
    def __init__(self, refuse_on: int) -> None:
        super().__init__(0.0)
        self.refuse_on = refuse_on
        self.evaluations = 0

    def evaluate_natural(self, y, theta, plan, *, derivative_order=2):
        self.evaluations += 1
        if self.evaluations == self.refuse_on:
            raise UnsupportedLikelihoodContractError("typed likelihood refusal sentinel")
        return super().evaluate_natural(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )


def test_intercept_only_solver_recovers_weighted_gaussian_mle_exactly() -> None:
    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    weights = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 4.0])
    family, layout = _intercept_layout(len(y))
    expected_mu = np.average(y, weights=weights)
    expected_sigma = np.sqrt(np.dot(weights, (y - expected_mu) ** 2) / len(y))
    expected_eta_scale = np.log(expected_sigma - family.scale_floor)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, weights),
        np.zeros((2, 2)),
    )

    assert result.converged is True
    np.testing.assert_allclose(result.coefficients, [expected_mu, expected_eta_scale], atol=2e-9)
    np.testing.assert_allclose(result.theta[:, 0], expected_mu, atol=2e-9)
    np.testing.assert_allclose(result.theta[:, 1], expected_sigma, atol=2e-9)
    assert result.score_relative <= result.config.tolerance
    assert result.terminal_curvature.requested_source == "observed"
    assert result.terminal_curvature.actual_source == "observed"
    result.terminal_curvature.assert_no_fallback()


def test_dense_fit_reuses_one_materialization_and_bypasses_grouped_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills grouped dispatch or predictor rematerialization inside dense iterations."""

    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    weights = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 4.0])
    family, layout = _intercept_layout(len(y))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    original_materializer = solver_module.dense_predictor_matrices
    materializations = 0

    def materialize_once(candidate_layout):
        nonlocal materializations
        materializations += 1
        return original_materializer(candidate_layout)

    def forbid_rematerialization(*_args, **_kwargs):
        raise AssertionError("dense geometry rematerialized predictor matrices")

    def forbid_grouped_assembly(*_args, **_kwargs):
        raise AssertionError("dense fit entered grouped assembly")

    monkeypatch.setattr(solver_module, "dense_predictor_matrices", materialize_once)
    monkeypatch.setattr(
        assembly_module,
        "dense_predictor_matrices",
        forbid_rematerialization,
    )
    monkeypatch.setattr(
        assembly_module,
        "assemble_grouped_geometry",
        forbid_grouped_assembly,
    )

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, weights),
        penalty,
    )

    assert result.converged is True
    assert materializations == 1


def test_fisher_request_without_expected_information_refuses_before_family_evaluation() -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    base_family, layout = _intercept_layout(len(response))
    family = _ObservedOnlyGaussian(base_family)

    with pytest.raises(ValueError, match=r"Fisher.*expected information"):
        fit_dense_fixed_lambda(
            family,
            layout,
            response,
            _plan(family, response, np.ones(len(response))),
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
            config=DenseSolverConfig(coefficient_curvature="fisher"),
        )

    assert family.initializations == 0
    assert family.evaluations == 0


def test_observed_dense_solver_accepts_order_two_family_without_fisher() -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    base_family, layout = _intercept_layout(len(response))
    family = _ObservedOnlyGaussian(base_family)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, np.ones(len(response))),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        config=DenseSolverConfig(coefficient_curvature="observed"),
    )

    assert result.converged is True
    assert result.terminal_curvature.requested_source == "observed"
    assert result.terminal_curvature.actual_source == "observed"
    result.terminal_curvature.assert_no_fallback()


def test_observed_only_chunked_solver_still_requires_expected_information() -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    base_family, layout = _intercept_layout(len(response))
    family = _ObservedOnlyGaussian(base_family)

    with pytest.raises(ValueError, match=r"chunked.*expected information"):
        fit_dense_fixed_lambda(
            family,
            layout,
            response,
            _plan(family, response, np.ones(len(response))),
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
            config=DenseSolverConfig(coefficient_curvature="observed"),
            chunk_size=2,
        )

    assert family.initializations == 0
    assert family.evaluations == 0


def test_observed_dense_state_never_evaluates_available_expected_information() -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    base_family, layout = _intercept_layout(len(response))
    family = _ExpectedInformationSpyGaussian(base_family)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, np.ones(len(response))),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        config=DenseSolverConfig(coefficient_curvature="observed"),
    )

    assert result.converged is True
    assert result.terminal_curvature.requested_source == "observed"
    assert result.terminal_curvature.actual_source == "observed"
    assert result.terminal_curvature.fallback_count == 0
    assert family.expected_information_calls == 0


def test_fit_scoped_observed_reuse_skips_redundant_initial_row_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    family, layout = _intercept_layout(len(response))
    plan = _plan(family, response, np.ones(len(response)))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    config = DenseSolverConfig(coefficient_curvature="observed")
    session = solver_module._DenseObservedReuseSession()
    first = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        config=config,
        _reuse_session=session,
    )
    original_evaluate = GaussianLS.evaluate_natural
    evaluations = 0

    def counted_evaluate(self, *args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted_evaluate)
    second = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=first.coefficients,
        config=config,
        _reuse_session=session,
        _reuse_source=first,
    )

    assert evaluations == 0
    assert second.iterations == 0
    np.testing.assert_array_equal(second.terminal_score, first.terminal_score)
    np.testing.assert_array_equal(second.terminal_data_curvature, first.terminal_data_curvature)


def test_fit_scoped_observed_reuse_refuses_a_copied_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    family, layout = _intercept_layout(len(response))
    plan = _plan(family, response, np.ones(len(response)))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    config = DenseSolverConfig(coefficient_curvature="observed")
    session = solver_module._DenseObservedReuseSession()
    first = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        config=config,
        _reuse_session=session,
    )
    copied = replace(first)
    original_evaluate = GaussianLS.evaluate_natural
    evaluations = 0

    def counted_evaluate(self, *args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted_evaluate)
    second = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=copied.coefficients,
        config=config,
        _reuse_session=session,
        _reuse_source=copied,
    )

    assert evaluations > 0
    assert second.converged


def test_fit_scoped_observed_reuse_matches_an_ordinary_repenalized_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = np.array([-0.4, 0.2, 1.1, 2.0])
    family, layout = _intercept_layout(len(response))
    plan = _plan(family, response, np.ones(len(response)))
    zero_penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    next_penalty = np.diag([0.2, 0.3])
    config = DenseSolverConfig(coefficient_curvature="observed")
    session = solver_module._DenseObservedReuseSession()
    first = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        zero_penalty,
        config=config,
        _reuse_session=session,
    )
    original_evaluate = GaussianLS.evaluate_natural
    evaluations = 0

    def counted_evaluate(self, *args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original_evaluate(self, *args, **kwargs)

    monkeypatch.setattr(GaussianLS, "evaluate_natural", counted_evaluate)
    reused = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        next_penalty,
        initial=first.coefficients,
        config=config,
        _reuse_session=session,
        _reuse_source=first,
    )
    reused_evaluations = evaluations
    evaluations = 0
    ordinary = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        next_penalty,
        initial=first.coefficients,
        config=config,
    )

    assert reused_evaluations + 1 == evaluations
    np.testing.assert_array_equal(reused.coefficients, ordinary.coefficients)
    np.testing.assert_array_equal(reused.terminal_score, ordinary.terminal_score)
    np.testing.assert_array_equal(
        reused.terminal_penalized_curvature,
        ordinary.terminal_penalized_curvature,
    )
    assert reused.penalized_log_likelihood == ordinary.penalized_log_likelihood


def test_observed_only_terminal_policy_assesses_penalized_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_family, layout = _intercept_layout(4)
    family = _ObservedOnlyGaussian(base_family)
    scale = base_family.scale_floor + 1.0
    response = np.array([-scale, -scale, scale, scale])
    observed_data_curvature = np.array([[1.0, 2.0], [2.0, 1.0]])
    penalty = 2.0 * np.eye(layout.n_coefficients)
    original_geometry = solver_module._geometry

    def force_indefinite_observed_data(context, state, source):
        geometry = original_geometry(context, state, source)
        if source != "observed":
            return geometry
        return replace(
            geometry,
            data_curvature=observed_data_curvature,
            penalized_curvature=observed_data_curvature + geometry.penalty,
        )

    monkeypatch.setattr(solver_module, "_geometry", force_indefinite_observed_data)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, np.ones(len(response))),
        penalty,
        initial=np.zeros(layout.n_coefficients),
        config=DenseSolverConfig(coefficient_curvature="observed"),
    )

    np.testing.assert_array_equal(result.terminal_data_curvature, observed_data_curvature)
    np.testing.assert_array_equal(
        result.terminal_penalized_curvature,
        observed_data_curvature + penalty,
    )
    assert result.terminal_rank.rank == layout.n_coefficients
    terminal = result.terminal_penalized_curvature
    width = terminal.shape[0]
    condition = float(np.linalg.cond(terminal))
    roundoff = 64.0 * width * np.finfo(terminal.dtype).eps * condition

    probe = np.array([1.0, -2.0], dtype=terminal.dtype)
    solution = result.terminal_rank.solve(probe)
    solve_scale = max(
        1.0,
        float(np.linalg.norm(terminal, ord=np.inf)) * float(np.linalg.norm(solution, ord=np.inf)),
        float(np.linalg.norm(probe, ord=np.inf)),
    )
    solve_residual = float(np.linalg.norm(terminal @ solution - probe, ord=np.inf))
    assert solve_residual <= roundoff * solve_scale

    covariance = result.terminal_rank.pseudo_inverse()
    inverse_scale = max(
        1.0,
        float(np.linalg.norm(terminal, ord=np.inf)) * float(np.linalg.norm(covariance, ord=np.inf)),
    )
    inverse_residual = float(np.linalg.norm(terminal @ covariance - np.eye(width), ord=np.inf))
    assert inverse_residual <= roundoff * inverse_scale

    determinant_sign, expected_log_determinant = np.linalg.slogdet(terminal)
    assert determinant_sign == 1.0
    log_determinant_scale = max(1.0, abs(float(expected_log_determinant)))
    assert abs(result.terminal_rank.log_pdet - expected_log_determinant) <= (
        roundoff * log_determinant_scale
    )
    assert result.terminal_curvature.requested_source == "observed"
    assert result.terminal_curvature.actual_source == "observed"
    assert result.terminal_curvature.minimum_eigenvalue == pytest.approx(1.0)
    assert result.terminal_curvature.rank == layout.n_coefficients
    result.terminal_curvature.assert_no_fallback()


def test_observed_only_terminal_policy_refuses_repeated_material_indefiniteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_family, layout = _intercept_layout(4)
    family = _ObservedOnlyGaussian(base_family)
    scale = base_family.scale_floor + 1.0
    response = np.array([-scale, -scale, scale, scale])
    observed_data_curvature = np.array([[1.0, 2.0], [2.0, 1.0]])
    original_geometry = solver_module._geometry
    observed_geometry_calls = 0

    def force_indefinite_observed_data(context, state, source):
        nonlocal observed_geometry_calls
        geometry = original_geometry(context, state, source)
        if source != "observed":
            return geometry
        observed_geometry_calls += 1
        return replace(
            geometry,
            data_curvature=observed_data_curvature,
            penalized_curvature=observed_data_curvature + geometry.penalty,
        )

    monkeypatch.setattr(solver_module, "_geometry", force_indefinite_observed_data)
    with pytest.raises(CurvaturePolicyError, match=r"Fisher.*required"):
        fit_dense_fixed_lambda(
            family,
            layout,
            response,
            _plan(family, response, np.ones(len(response))),
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
            initial=np.zeros(layout.n_coefficients),
            config=DenseSolverConfig(coefficient_curvature="observed"),
        )

    assert observed_geometry_calls == 2


def test_solver_result_plan_and_execution_fields_are_required() -> None:
    parameters = inspect.signature(solver_module.DenseSolverResult).parameters

    for name in (
        "family_likelihood_plan_identifier",
        "resolved_chunk_size",
        "execution_backend_identifier",
    ):
        assert parameters[name].default is inspect.Parameter.empty


def test_solver_stamps_root_plan_and_actual_dense_or_chunked_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    weights = np.ones(len(y))
    family, layout = _intercept_layout(len(y))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))

    dense_plan = _plan(family, y, weights)

    def forbid_chunked(*args, **kwargs):
        raise AssertionError("dense execution reached the chunked likelihood route")

    with monkeypatch.context() as patch:
        patch.setattr(
            solver_module.chunking,
            "evaluate_chunked_log_likelihood",
            forbid_chunked,
        )
        dense = fit_dense_fixed_lambda(family, layout, y, dense_plan, penalty)

    assert dense.family_likelihood_plan_identifier == dense_plan.plan_identifier
    assert dense.resolved_chunk_size is None
    assert dense.execution_backend_identifier == "distributional-dense-v1"

    chunked_plan = _plan(family, y, weights)
    chunk_calls = 0
    original_chunked = solver_module.chunking.evaluate_chunked_log_likelihood

    def count_chunked(*args, **kwargs):
        nonlocal chunk_calls
        chunk_calls += 1
        return original_chunked(*args, **kwargs)

    def forbid_dense(*args, **kwargs):
        raise AssertionError("chunked execution reached the dense predictor route")

    with monkeypatch.context() as patch:
        patch.setattr(
            solver_module.chunking,
            "evaluate_chunked_log_likelihood",
            count_chunked,
        )
        patch.setattr(solver_module, "_evaluate_predictors_from_matrices", forbid_dense)
        chunked = fit_dense_fixed_lambda(
            family,
            layout,
            y,
            chunked_plan,
            penalty,
            chunk_size=2,
        )

    assert chunk_calls > 0
    assert chunked.family_likelihood_plan_identifier == chunked_plan.plan_identifier
    assert (
        chunked.family_likelihood_plan_identifier
        != chunked_plan.take(np.array([0, 1], dtype=np.intp)).plan_identifier
    )
    assert chunked.resolved_chunk_size == 2
    assert chunked.execution_backend_identifier == "distributional-chunked-v1"


@pytest.mark.parametrize(
    ("positions", "forge_root_digest"),
    [([0, 1], False), ([1, 3], True)],
    ids=["ordinary-child", "noncanonical-take-map"],
)
def test_top_level_solver_refuses_positional_child_likelihood_plans(
    positions: list[int],
    forge_root_digest: bool,
) -> None:
    root_response = np.array([-1.2, -0.3, 0.8, 1.7])
    family, layout = _intercept_layout(len(positions))
    root_weights = resolve_likelihood_weights(
        np.array([0.5, 0.9, 1.4, 2.1]),
        n_observations=len(root_response),
        contract=WeightContract("prior"),
    )
    take = np.array(positions, dtype=np.intp)
    root_plan = family.bind_likelihood(
        root_response,
        root_weights,
        COMPLETE_OBSERVATION,
    )
    child_plan = root_plan.take(take)
    if forge_root_digest:
        child_plan = replace(
            child_plan,
            weights=replace(
                child_plan.weights,
                digest=child_plan.weights.provenance.root_digest,
            ),
        )

    with pytest.raises(UnsupportedLikelihoodContractError, match="complete|row subset|root"):
        fit_dense_fixed_lambda(
            family,
            layout,
            root_response[take],
            child_plan,
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
        )


@pytest.mark.parametrize("chunk_size", [None, 2], ids=["dense", "chunked"])
@pytest.mark.parametrize(
    "carrier_per_row",
    [4.0, -4.0],
    ids=["positive-carrier", "negative-carrier"],
)
def test_parameter_independent_carrier_cannot_change_solver_decisions(
    chunk_size: int | None,
    carrier_per_row: float,
) -> None:
    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    weights = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    _, layout = _intercept_layout(len(y))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    config = DenseSolverConfig(max_iterations=80, tolerance=1.0e-10)
    initial = np.array([4.0, -1.0])

    reference_family = _ConstantCarrierGaussian(0.0)
    shifted_family = _ConstantCarrierGaussian(carrier_per_row)
    reference = fit_dense_fixed_lambda(
        reference_family,
        layout,
        y,
        _plan(reference_family, y, weights),
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
    )
    shifted = fit_dense_fixed_lambda(
        shifted_family,
        layout,
        y,
        _plan(shifted_family, y, weights),
        penalty,
        initial=initial,
        config=config,
        chunk_size=chunk_size,
    )

    np.testing.assert_array_equal(shifted.coefficients, reference.coefficients)
    assert shifted.history == reference.history
    assert shifted.convergence_reason == reference.convergence_reason
    np.testing.assert_array_equal(
        shifted.terminal_data_curvature,
        reference.terminal_data_curvature,
    )
    assert shifted.objective == reference.objective
    assert shifted.optimizing_log_likelihood == reference.optimizing_log_likelihood
    assert (
        shifted.penalized_optimizing_log_likelihood == reference.penalized_optimizing_log_likelihood
    )
    assert joint_laplace_objective(shifted, layout=layout, lambdas={}) == (
        joint_laplace_objective(reference, layout=layout, lambdas={})
    )
    carrier_sum = carrier_per_row * len(y)
    tolerance = 8.0 * len(y) * np.finfo(np.float64).eps * carrier_sum
    assert shifted.log_likelihood - reference.log_likelihood == pytest.approx(
        carrier_sum,
        abs=abs(tolerance),
    )
    assert shifted.parameter_independent_carrier == pytest.approx(
        reference.parameter_independent_carrier + carrier_sum,
        abs=abs(tolerance),
    )
    assert shifted.penalized_log_likelihood - reference.penalized_log_likelihood == (
        pytest.approx(carrier_sum, abs=abs(tolerance))
    )
    assert shifted.initial_penalized_optimizing_log_likelihood == (
        reference.initial_penalized_optimizing_log_likelihood
    )
    assert shifted.initial_penalized_log_likelihood - (
        reference.initial_penalized_log_likelihood
    ) == pytest.approx(carrier_sum, abs=abs(tolerance))
    assert (
        shifted.penalized_log_likelihood - shifted.initial_penalized_log_likelihood
    ) == pytest.approx(
        shifted.penalized_optimizing_log_likelihood
        - shifted.initial_penalized_optimizing_log_likelihood,
        abs=abs(tolerance),
    )

    initial_sigma = shifted_family.base.scale_floor + np.exp(initial[1])
    residual = y - initial[0]
    expected_initial_optimizing = float(
        np.sum(
            -np.log(initial_sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * weights * residual * residual / initial_sigma**2,
            dtype=np.float64,
        )
    )
    expected_base_carrier = 0.5 * float(np.sum(np.log(weights), dtype=np.float64))
    assert shifted.initial_penalized_optimizing_log_likelihood == pytest.approx(
        expected_initial_optimizing,
        rel=0.0,
        abs=4e-13,
    )
    assert shifted.initial_penalized_log_likelihood == pytest.approx(
        expected_initial_optimizing + expected_base_carrier + carrier_sum,
        rel=0.0,
        abs=4e-13,
    )


def test_solver_refuses_raw_weights_instead_of_rebinding_likelihood() -> None:
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    family, layout = _intercept_layout(len(y))

    with pytest.raises(
        UnsupportedLikelihoodContractError,
        match="family.bind_likelihood|FamilyLikelihoodPlan",
    ):
        fit_dense_fixed_lambda(
            family,
            layout,
            y,
            np.ones(len(y)),
            np.zeros((2, 2)),
        )


@pytest.mark.parametrize(
    ("chunk_size", "refuse_on"),
    [(None, 2), (6, 3)],
    ids=["dense-trial", "chunked-trial"],
)
def test_typed_likelihood_refusal_propagates_from_trial_evaluation(
    chunk_size: int | None,
    refuse_on: int,
) -> None:
    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    family = _TypedRefusalGaussian(refuse_on)
    _, layout = _intercept_layout(len(y))

    with pytest.raises(
        UnsupportedLikelihoodContractError,
        match="typed likelihood refusal sentinel",
    ):
        fit_dense_fixed_lambda(
            family,
            layout,
            y,
            _plan(family, y, np.ones(len(y))),
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
            initial=np.array([4.0, -1.0]),
            chunk_size=chunk_size,
        )


def test_dense_solver_refuses_wrong_order_before_invalid_trial_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kills treating a malformed family result as an ordinary invalid trial."""

    y = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    family = _ConstantCarrierGaussian(0.0)
    _, layout = _intercept_layout(len(y))
    original_evaluate = family.evaluate_natural

    def wrong_order(y, theta, plan, *, derivative_order=2):
        evaluation = original_evaluate(
            y,
            theta,
            plan,
            derivative_order=derivative_order,
        )
        return NaturalLikelihoodEvaluation(
            optimizing_log_likelihood=evaluation.optimizing_log_likelihood,
            parameter_independent_carrier=evaluation.parameter_independent_carrier,
            score=None,
            hessian_packed=None,
            valid=np.zeros(len(y), dtype=bool),
        )

    monkeypatch.setattr(family, "evaluate_natural", wrong_order)
    with pytest.raises(UnsupportedLikelihoodContractError, match="exact derivative order 2"):
        fit_dense_fixed_lambda(
            family,
            layout,
            y,
            _plan(family, y, np.ones(len(y))),
            np.zeros((layout.n_coefficients, layout.n_coefficients)),
        )


def test_offsets_and_case_weights_use_the_same_weighted_likelihood_target() -> None:
    n = 8
    location_offset = np.linspace(-0.8, 0.6, n)
    residual = np.array([-1.5, -0.7, -0.2, 0.1, 0.4, 0.8, 1.1, 1.8])
    weights = np.array([0.5, 0.7, 1.0, 1.3, 1.8, 2.1, 2.7, 3.2])
    true_intercept = 2.4
    y = true_intercept + location_offset + residual
    family, layout = _intercept_layout(n, location_offset=location_offset)
    expected_mu_intercept = np.average(y - location_offset, weights=weights)
    expected_sigma = np.sqrt(
        np.dot(weights, (y - location_offset - expected_mu_intercept) ** 2) / len(y)
    )

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, weights),
        np.zeros((2, 2)),
    )

    np.testing.assert_allclose(result.coefficients[0], expected_mu_intercept, atol=2e-8)
    np.testing.assert_allclose(result.theta[:, 1], expected_sigma, atol=2e-8)
    np.testing.assert_allclose(result.eta[:, 0], expected_mu_intercept + location_offset, atol=2e-8)


def test_solver_improves_objective_from_poor_state_with_observed_shift_and_backtracking() -> None:
    y = np.array([-2.0, -1.0, -0.2, 0.4, 1.1, 2.3])
    family, layout = _intercept_layout(len(y))
    initial = np.array([20.0, 0.0])
    config = DenseSolverConfig(
        coefficient_curvature="observed",
        max_iterations=80,
        tolerance=1e-9,
    )

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        np.zeros((2, 2)),
        initial=initial,
        config=config,
    )

    assert result.penalized_log_likelihood > result.initial_penalized_log_likelihood
    assert all(
        later.objective_after >= earlier.objective_after - 1e-12
        for earlier, later in zip(result.history, result.history[1:], strict=False)
    )
    assert any(item.levenberg_shift > 0.0 for item in result.history)
    assert any(item.step_scale < 1.0 for item in result.history)
    assert result.backtracking_steps >= 1
    assert max(item.solve_residual for item in result.history) <= config.residual_tolerance
    np.testing.assert_array_equal(initial, np.array([20.0, 0.0]))


def test_nonfinite_trial_is_rejected_without_mutating_accepted_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = np.array([-1.5, -0.4, 0.2, 0.9, 2.0])
    family, layout = _intercept_layout(len(y))
    initial = np.array([5.0, 1.0])
    initial_before = initial.copy()
    original = solver_module._evaluate_state
    calls = 0

    def fail_first_trial(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "_evaluate_state", fail_first_trial)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        np.zeros((2, 2)),
        initial=initial,
    )

    assert result.backtracking_steps >= 1
    assert result.penalized_log_likelihood > result.initial_penalized_log_likelihood
    np.testing.assert_array_equal(initial, initial_before)
    assert np.all(np.isfinite(result.coefficients))


def test_dense_solver_materializes_each_immutable_design_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n = 48
    x = np.linspace(-1.0, 1.0, n)
    z = np.cos(np.linspace(0.0, 2.0 * np.pi, n))
    frame = as_eager_frame(pd.DataFrame({"x": x, "z": z}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor("location", {"x": Numeric()}),
                Predictor("scale", {"z": Numeric()}),
            ),
        )
    )
    response = (
        0.4
        + 0.8 * x
        + np.random.default_rng(2077).normal(
            scale=0.25 + 0.08 * (z + 1.0),
            size=n,
        )
    )
    original_toarray = DesignMatrix.toarray
    materialized: list[int] = []

    def counted_toarray(design: DesignMatrix) -> np.ndarray:
        materialized.append(id(design))
        return original_toarray(design)

    monkeypatch.setattr(DesignMatrix, "toarray", counted_toarray)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, np.ones(n)),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        initial=np.zeros(layout.n_coefficients),
        config=DenseSolverConfig(max_iterations=80, tolerance=1.0e-9),
    )

    assert result.converged is True
    assert result.iterations > 1
    assert sorted(materialized) == sorted(id(state.design) for state in layout.predictors)


def test_fixed_lambda_reuses_each_accepted_geometry_across_iterations_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y = np.array([-2.5, -1.0, -0.2, 0.4, 1.6, 3.2])
    family, layout = _intercept_layout(len(y))
    original_geometry = solver_module._geometry
    geometry_sources: list[str] = []

    def counted_geometry(context, state, source):
        geometry_sources.append(source)
        return original_geometry(context, state, source)

    monkeypatch.setattr(solver_module, "_geometry", counted_geometry)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        initial=np.array([4.0, -1.5]),
        config=DenseSolverConfig(
            max_iterations=80, tolerance=1.0e-9, coefficient_curvature="fisher"
        ),
    )

    assert result.converged is True
    assert result.iterations > 1
    assert result.terminal_curvature.actual_source == "observed"
    result.terminal_curvature.assert_no_fallback()
    assert geometry_sources.count("fisher") == result.iterations + 1
    assert geometry_sources.count("observed") == 1


def test_tiny_positive_direction_uses_a_scale_relative_ascent_check() -> None:
    score = np.array([1.0e-8, -0.5e-8])
    curvature = np.eye(2)

    direction = solver_module._solve_direction(curvature, score, DenseSolverConfig())

    assert float(score @ direction.step) > 0.0
    assert direction.levenberg_shift == 0.0
    assert direction.residual <= DenseSolverConfig().residual_tolerance


def test_well_conditioned_fisher_direction_uses_verified_spd_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.array([[2.0, 0.25], [0.25, 1.5]])
    score = np.array([0.75, -0.2])

    monkeypatch.setattr(
        solver_module,
        "decompose_gram",
        lambda *args, **kwargs: pytest.fail("verified Fisher solve must avoid rank fallback"),
    )
    direction = solver_module._solve_direction(
        matrix,
        score,
        DenseSolverConfig(coefficient_curvature="fisher"),
    )

    np.testing.assert_allclose(matrix @ direction.step, score, rtol=1e-13, atol=1e-13)
    assert direction.decomposition.method == "cholesky"
    assert direction.levenberg_shift == 0.0
    assert direction.residual <= DenseSolverConfig().residual_tolerance


@pytest.mark.parametrize(
    ("curvature", "matrix"),
    [
        ("observed", np.array([[2.0, 0.25], [0.25, 1.5]])),
        ("fisher", np.array([[1.0, 1.0 - 1.0e-10], [1.0 - 1.0e-10, 1.0]])),
    ],
)
def test_uncertified_direction_uses_shared_rank_fallback(
    monkeypatch: pytest.MonkeyPatch,
    curvature: str,
    matrix: np.ndarray,
) -> None:
    original = solver_module.decompose_gram
    calls = 0

    def counted_decomposition(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_module, "decompose_gram", counted_decomposition)
    direction = solver_module._solve_direction(
        matrix,
        np.array([0.5, 1.0e-12]),
        DenseSolverConfig(coefficient_curvature=curvature),
    )

    assert calls == 1
    assert direction.residual <= DenseSolverConfig().residual_tolerance


def test_scale_floor_stress_remains_strictly_inside_support() -> None:
    floor = 0.01
    y = 3.0 + np.array([-0.013, -0.009, -0.004, 0.003, 0.008, 0.015])
    family, layout = _intercept_layout(len(y))
    assert family.scale_floor == floor

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        np.zeros((2, 2)),
        config=DenseSolverConfig(max_iterations=200, tolerance=1e-9),
    )

    assert result.converged is True
    assert np.all(result.theta[:, 1] > floor)
    assert np.all(np.isfinite(result.theta))


def test_near_rank_deficient_design_uses_shared_rank_policy() -> None:
    n = 30
    x = np.linspace(-1.0, 1.0, n)
    frame = as_eager_frame(pd.DataFrame({"x": x, "x_alias": x * (1.0 + 1e-13)}))
    y = 0.7 + 1.2 * x + np.random.default_rng(91).normal(scale=0.4, size=n)
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "x_alias": Numeric()}),
                Predictor("scale", {}),
            ),
        )
    )

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(n)),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
    )

    assert result.converged is True
    assert result.terminal_rank.rank < layout.n_coefficients
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.theta[:, 0], 0.7 + 1.2 * x, atol=0.35)


def test_iteration_exhaustion_is_published_as_not_converged() -> None:
    n = 25
    x = np.linspace(-1.0, 1.0, n)
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
        )
    )
    y = 0.5 + 2.0 * x + np.sin(4.0 * x)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(n)),
        np.zeros((3, 3)),
        initial=np.zeros(3),
        config=DenseSolverConfig(max_iterations=1, tolerance=1e-14),
    )

    assert result.converged is False
    assert result.convergence_reason == "max_iterations"
    assert result.iterations == 1
    assert result.terminal_curvature.actual_source == "observed"
    result.terminal_curvature.assert_no_fallback()


@pytest.mark.parametrize("coefficient_curvature", ["fisher", "observed"])
def test_material_terminal_curvature_retries_then_records_fisher_fallback(
    coefficient_curvature: str,
) -> None:
    n = 25
    x = np.linspace(-1.0, 1.0, n)
    frame = as_eager_frame(pd.DataFrame({"x": x}))
    family = _ExpectedInformationSpyGaussian(GaussianLS())
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (Predictor("location", {"x": Numeric()}), Predictor("scale", {})),
        )
    )
    y = 0.5 + 2.0 * x + np.sin(4.0 * x)

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(n)),
        np.zeros((3, 3)),
        initial=np.array([8.0, -4.0, 2.0]),
        config=DenseSolverConfig(
            coefficient_curvature=coefficient_curvature,  # type: ignore[arg-type]
            max_iterations=1,
            terminal_retry_iterations=1,
            tolerance=1e-14,
        ),
    )

    assert result.iterations == 2
    assert result.terminal_curvature.actual_source == "fisher"
    assert result.terminal_curvature.reason == "material_indefiniteness_after_retry"
    assert result.terminal_curvature.fallback_count == 1
    if coefficient_curvature == "observed":
        assert family.expected_information_calls == 1
    else:
        assert family.expected_information_calls > 1


def test_solver_result_is_defensively_immutable() -> None:
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    family, layout = _intercept_layout(len(y))
    penalty = np.zeros((2, 2))

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        penalty,
    )
    penalty[:] = 99.0

    assert not result.coefficients.flags.writeable
    assert not result.eta.flags.writeable
    assert not result.theta.flags.writeable
    assert not result.terminal_penalized_curvature.flags.writeable
    np.testing.assert_array_equal(result.penalty, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        result.coefficients[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.converged = False  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "config",
    [
        DenseSolverConfig(max_iterations=1),
        DenseSolverConfig(tolerance=1e-8),
    ],
)
def test_solver_configuration_is_retained(config: DenseSolverConfig) -> None:
    y = np.array([-1.0, 0.0, 1.0, 2.0])
    family, layout = _intercept_layout(len(y))

    result = fit_dense_fixed_lambda(
        family,
        layout,
        y,
        _plan(family, y, np.ones(len(y))),
        np.zeros((2, 2)),
        config=config,
    )

    assert result.config == config
