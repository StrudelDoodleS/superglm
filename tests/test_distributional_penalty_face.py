from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from superglm._frame import as_eager_frame
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.inference import compute_joint_inference
from superglm.distributional.layout import StackedLayout, build_stacked_layout
from superglm.distributional.penalty_face import PenaltyFaceError, build_penalty_face
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights
from superglm.features import Numeric, RandomEffect, Spline
from superglm.reml.penalty_algebra import penalty_component_dense_matrix
from superglm.solvers.rank import decompose_gram
from superglm.types import PenaltyComponent

from ._distributional_weights import resolved_prior


class _ObservedOnlyGaussian:
    """Exercise the same dense observed-only contract required by Tweedie."""

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


def _numeric_layout(n: int = 24) -> StackedLayout:
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.linspace(-1.0, 1.0, n),
                "z": np.cos(np.linspace(0.0, 2.0 * np.pi, n)),
            }
        )
    )
    family = GaussianLS()
    return build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor("location", {"x": Numeric(), "z": Numeric()}),
                Predictor("scale", {}),
            ),
        )
    )


def _plan(family: GaussianLS, response: np.ndarray, weights: np.ndarray):
    resolved = resolve_likelihood_weights(
        weights,
        n_observations=len(response),
        contract=WeightContract("prior"),
    )
    return family.bind_likelihood(response, resolved, COMPLETE_OBSERVATION)


def _random_effect_problem():
    groups = np.array(["a", "b", "c", "a", "b", "c"] * 4)
    group_effect = {"a": -0.6, "b": 0.4, "c": 0.2}
    baseline = np.array([group_effect[value] for value in groups])
    response = 1.25 + baseline + np.linspace(-0.35, 0.35, len(groups))
    weights = np.linspace(0.6, 1.8, len(groups))
    frame = as_eager_frame(pd.DataFrame({"group": groups}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(weights),
            family.parameters,
            (
                Predictor("location", {"group": RandomEffect()}),
                Predictor("scale", {}),
            ),
        )
    )
    component = next(item for item in layout.penalties if item.penalty_kind == "identity")
    face = build_penalty_face(layout, (component.name,))
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    return family, layout, component, face, response, weights, penalty


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


def test_face_is_the_certified_global_null_of_disjoint_components() -> None:
    layout = _numeric_layout()
    x_component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        layout.term_slices["location:x"],
        np.ones((1, 1)),
        rank=1.0,
    )
    z_component = _dense_component(
        "location:z#identity",
        "location:z",
        1,
        layout.term_slices["location:z"],
        np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(x_component, z_component))

    face = build_penalty_face(
        layout,
        ("location:x#identity", "location:z#identity"),
    )

    width = layout.n_coefficients
    assert face.constraint_rank == 2
    assert face.null_basis.shape == (width, width - 2)
    assert face.component_names == (
        "location:x#identity",
        "location:z#identity",
    )
    eps = np.finfo(np.float64).eps
    matrix_norm = float(np.linalg.norm(face.constraint_matrix, ord=2))
    np.testing.assert_allclose(
        face.null_basis.T @ face.null_basis,
        np.eye(width - 2),
        rtol=0.0,
        atol=256.0 * width * eps,
    )
    assert np.linalg.norm(face.constraint_matrix @ face.null_basis, ord=2) <= (
        256.0 * width * eps * matrix_norm
    )
    constrained = np.array(
        [layout.term_slices["location:x"].start, layout.term_slices["location:z"].start]
    )
    np.testing.assert_allclose(face.null_basis[constrained], 0.0, rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError):
        face.null_basis[0, 0] = 4.0
    with pytest.raises(FrozenInstanceError):
        face.constraint_rank = 1  # ty: ignore[invalid-assignment]
    forged_basis = np.array(face.null_basis, copy=True)
    forged_basis[:, 0] = face.constraint_basis[:, 0]
    with pytest.raises(PenaltyFaceError, match="null basis"):
        replace(face, null_basis=forged_basis)
    swapped_constraint = np.array(face.constraint_basis, copy=True)
    swapped_constraint[:, 0] = face.null_basis[:, 0]
    with pytest.raises((PenaltyFaceError, TypeError, ValueError), match="residual|init=False"):
        replace(
            face,
            null_basis=forged_basis,
            constraint_basis=swapped_constraint,
            null_residual_bound=2.0,
        )


def test_face_layout_validation_accepts_one_ulp_normalization_drift() -> None:
    """Cross-platform LAPACK roundoff cannot invalidate certified geometry."""
    _family, layout, _component, face, _response, _weights, _penalty = _random_effect_problem()
    perturbed = np.array(face.constraint_matrix, copy=True)
    diagonal = np.flatnonzero(np.diag(perturbed) != 0.0)
    assert len(diagonal) > 0
    index = int(diagonal[0])
    perturbed[index, index] = np.nextafter(perturbed[index, index], np.inf)

    portable = replace(face, constraint_matrix=perturbed)
    portable.validate_layout(layout)


def test_face_of_independent_blocks_is_invariant_to_each_penalty_scale() -> None:
    """Kills resolving independent selected blocks against one global scale."""
    layout = _numeric_layout()
    x_component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        layout.term_slices["location:x"],
        np.ones((1, 1)),
        rank=1.0,
    )
    z_component = _dense_component(
        "location:z#identity",
        "location:z",
        1,
        layout.term_slices["location:z"],
        np.ones((1, 1)),
        rank=1.0,
    )
    names = (x_component.name, z_component.name)
    unit_layout = replace(layout, penalties=(x_component, z_component))
    unit_face = build_penalty_face(unit_layout, names)
    small_matrix = np.full((1, 1), 1.0e-14)
    scaled_z = replace(
        z_component,
        omega_raw=small_matrix,
        omega_ssp=small_matrix,
        eigvals_omega=small_matrix.ravel(),
    )
    scaled_layout = replace(layout, penalties=(x_component, scaled_z))

    scaled_face = build_penalty_face(scaled_layout, names)

    expected_projector = np.eye(layout.n_coefficients)
    expected_projector[layout.term_slices["location:x"], layout.term_slices["location:x"]] = 0.0
    expected_projector[layout.term_slices["location:z"], layout.term_slices["location:z"]] = 0.0
    assert scaled_face.constraint_rank == 2
    np.testing.assert_allclose(
        scaled_face.projector,
        expected_projector,
        rtol=0.0,
        atol=128.0 * layout.n_coefficients * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        scaled_face.projector,
        unit_face.projector,
        rtol=0.0,
        atol=128.0 * layout.n_coefficients * np.finfo(np.float64).eps,
    )


@pytest.mark.parametrize("penalty_scale", [1.0e-308, 1.0e308])
def test_face_geometry_is_invariant_at_finite_scale_extremes(
    penalty_scale: float,
) -> None:
    """Kills overflow or underflow before a positive penalty is normalized."""
    layout = _numeric_layout()
    target = layout.term_slices["location:x"]
    unit_component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        target,
        np.ones((1, 1)),
        rank=1.0,
    )
    scaled_matrix = np.full((1, 1), penalty_scale)
    scaled_component = replace(
        unit_component,
        omega_raw=scaled_matrix,
        omega_ssp=scaled_matrix,
        eigvals_omega=scaled_matrix.ravel(),
    )
    unit_face = build_penalty_face(
        replace(layout, penalties=(unit_component,)), (unit_component.name,)
    )
    scaled_face = build_penalty_face(
        replace(layout, penalties=(scaled_component,)),
        (scaled_component.name,),
    )

    np.testing.assert_allclose(
        scaled_face.projector,
        unit_face.projector,
        rtol=0.0,
        atol=128.0 * layout.n_coefficients * np.finfo(np.float64).eps,
    )


def test_face_expands_a_real_cr_wiggle_and_preserves_its_null_direction() -> None:
    n = 36
    frame = as_eager_frame(pd.DataFrame({"x": np.linspace(-2.0, 3.0, n)}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor("location", {"x": Spline(kind="cr", n_knots=7)}),
                Predictor("scale", {}),
            ),
        )
    )
    component = next(item for item in layout.penalties if item.name.endswith("#wiggle"))
    block = penalty_component_dense_matrix(component)

    face = build_penalty_face(layout, (component.name,))

    block_width = component.group_sl.stop - component.group_sl.start
    declared_rank = int(component.rank)
    assert face.null_basis.shape[1] == layout.n_coefficients - declared_rank
    local_projection = face.projector[component.group_sl, component.group_sl]
    local_eigenvalues = np.linalg.eigvalsh(local_projection)
    eps = np.finfo(np.float64).eps
    assert np.count_nonzero(local_eigenvalues > 1.0 - 512.0 * block_width * eps) == (
        block_width - declared_rank
    )
    assert np.linalg.norm(block @ local_projection, ord=2) <= (
        512.0 * block_width * eps * np.linalg.norm(block, ord=2)
    )


def test_face_refuses_components_whose_coefficient_support_overlaps() -> None:
    layout = _numeric_layout()
    target = layout.term_slices["location:x"]
    first = _dense_component(
        "location:x#first",
        "location:x",
        0,
        target,
        np.ones((1, 1)),
        rank=1.0,
    )
    second = _dense_component(
        "location:x#second",
        "location:x",
        0,
        target,
        2.0 * np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(first, second))

    with pytest.raises(PenaltyFaceError, match="overlap"):
        build_penalty_face(layout, (first.name, second.name))


def test_face_refuses_one_component_from_a_shared_selection_block() -> None:
    n = 30
    frame = as_eager_frame(pd.DataFrame({"x": np.linspace(-1.0, 1.0, n)}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(n)),
            family.parameters,
            (
                Predictor(
                    "location",
                    {"x": Spline(kind="cr", n_knots=6, select=True)},
                ),
                Predictor("scale", {}),
            ),
        )
    )
    shared = tuple(
        component for component in layout.penalties if component.group_name == "location:x"
    )
    assert len(shared) == 2

    for component in shared:
        with pytest.raises(PenaltyFaceError, match="shared|overlap"):
            build_penalty_face(layout, (component.name,))


def test_face_refuses_a_rank_claim_beneath_numerical_resolution() -> None:
    layout = _numeric_layout()
    target = slice(
        layout.term_slices["location:x"].start,
        layout.term_slices["location:z"].stop,
    )
    unresolved = _dense_component(
        "location:x#unresolved",
        "location:x",
        0,
        target,
        np.diag([1.0, np.finfo(np.float64).eps]),
        rank=2.0,
    )
    layout = replace(layout, penalties=(unresolved,))

    with pytest.raises(PenaltyFaceError, match="rank.*resolve|resolve.*rank"):
        build_penalty_face(layout, (unresolved.name,))


def test_identity_penalty_face_removes_its_complete_random_effect_block() -> None:
    groups = np.array(["a", "b", "c", "a", "b", "c"] * 4)
    frame = as_eager_frame(pd.DataFrame({"group": groups}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(np.ones(len(groups))),
            family.parameters,
            (
                Predictor("location", {"group": RandomEffect()}),
                Predictor("scale", {}),
            ),
        )
    )
    component = next(item for item in layout.penalties if item.penalty_kind == "identity")

    face = build_penalty_face(layout, (component.name,))

    np.testing.assert_allclose(
        face.projector[component.group_sl, component.group_sl],
        0.0,
        rtol=0.0,
        atol=128.0 * layout.n_coefficients * np.finfo(np.float64).eps,
    )


def test_lifted_rank_preserves_a_certified_coordinate_scaled_inverse() -> None:
    layout = _numeric_layout()
    target = layout.term_slices["location:x"]
    component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        target,
        np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    assert face.reduced_width == 3
    reduced_curvature = np.diag([1.0, 1.0e14, 3.0])
    reduced = decompose_gram(reduced_curvature)
    assert reduced.rank == face.reduced_width

    lifted = face.lift_rank_decomposition(reduced)

    expected = face.null_basis @ np.diag([1.0, 1.0e-14, 1.0 / 3.0]) @ face.null_basis.T
    np.testing.assert_allclose(
        lifted.pseudo_inverse(),
        expected,
        rtol=64.0 * np.finfo(np.float64).eps,
        atol=64.0 * np.finfo(np.float64).eps,
    )


def test_lifted_rank_preserves_rank_deficient_estimable_functionals() -> None:
    layout = _numeric_layout()
    target = layout.term_slices["location:x"]
    component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        target,
        np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    reduced = decompose_gram(np.ones((face.reduced_width, face.reduced_width)))
    assert reduced.method == "pivoted_cholesky"
    assert reduced.estimable_functional_basis is not None

    lifted = face.lift_rank_decomposition(reduced)

    expected = face.null_basis @ reduced.estimable_functional_basis
    assert lifted.estimable_functional_basis is not None
    np.testing.assert_allclose(
        lifted.estimable_functional_basis,
        expected,
        rtol=0.0,
        atol=64.0 * face.width * np.finfo(np.float64).eps,
    )
    assert all(
        lifted.is_estimable(lifted.estimable_functional_basis[:, index])
        for index in range(lifted.rank)
    )
    assert not np.allclose(
        lifted.estimable_functional_basis,
        lifted.solution_basis,
    )


def test_lifted_rank_preserves_structural_zero_estimability() -> None:
    layout = _numeric_layout()
    target = layout.term_slices["location:x"]
    component = _dense_component(
        "location:x#identity",
        "location:x",
        0,
        target,
        np.ones((1, 1)),
        rank=1.0,
    )
    layout = replace(layout, penalties=(component,))
    face = build_penalty_face(layout, (component.name,))
    reduced = decompose_gram(np.diag([2.0, 0.0, 3.0]))
    assert reduced.method == "cholesky"
    assert reduced.rank < reduced.width
    assert reduced.estimable_functional_basis is None

    lifted = face.lift_rank_decomposition(reduced)

    expected = face.null_basis[:, reduced.active_columns]
    assert lifted.estimable_functional_basis is not None
    np.testing.assert_allclose(
        lifted.estimable_functional_basis,
        expected,
        rtol=0.0,
        atol=64.0 * face.width * np.finfo(np.float64).eps,
    )
    assert all(
        lifted.is_estimable(lifted.estimable_functional_basis[:, index])
        for index in range(lifted.rank)
    )
    inactive = np.setdiff1d(np.arange(reduced.width), reduced.active_columns)
    assert inactive.size == 1
    assert not lifted.is_estimable(face.null_basis[:, inactive[0]])


def test_no_face_keyword_preserves_the_ordinary_solver_path_exactly() -> None:
    response = np.array([-2.0, -0.5, 0.2, 1.4, 3.0, 7.0])
    weights = np.array([0.3, 0.7, 1.0, 1.5, 2.0, 4.0])
    family = GaussianLS()
    frame = as_eager_frame(pd.DataFrame({"row": np.arange(len(response))}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(weights),
            family.parameters,
            (Predictor("location", {}), Predictor("scale", {})),
        )
    )
    penalty = np.zeros((layout.n_coefficients, layout.n_coefficients))
    initial = np.array([0.0, 0.0])

    ordinary = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=initial,
    )
    explicit_none = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=initial,
        coefficient_face=None,
    )

    for name in (
        "coefficients",
        "eta",
        "theta",
        "terminal_score",
        "terminal_data_curvature",
        "terminal_penalized_curvature",
    ):
        np.testing.assert_array_equal(getattr(explicit_none, name), getattr(ordinary, name))
    assert explicit_none.history == ordinary.history
    assert explicit_none.penalized_optimizing_log_likelihood == (
        ordinary.penalized_optimizing_log_likelihood
    )
    assert explicit_none.terminal_rank.log_pdet == ordinary.terminal_rank.log_pdet
    np.testing.assert_array_equal(
        explicit_none.terminal_pseudo_inverse(),
        ordinary.terminal_rank.pseudo_inverse(),
    )


@pytest.mark.parametrize("coefficient_curvature", ["fisher", "observed"])
def test_face_fit_matches_independent_constrained_gaussian_mle(
    coefficient_curvature: str,
) -> None:
    family, layout, component, face, response, weights, penalty = _random_effect_problem()
    config = DenseSolverConfig(
        coefficient_curvature=coefficient_curvature,  # type: ignore[arg-type]
        tolerance=1.0e-10,
        max_iterations=120,
    )

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=np.linspace(-3.0, 4.0, layout.n_coefficients),
        config=config,
        coefficient_face=face,
    )

    expected_location = float(np.average(response, weights=weights))
    expected_scale = float(
        np.sqrt(np.dot(weights, (response - expected_location) ** 2) / len(response))
    )
    expected_scale_coefficient = float(np.log(expected_scale - family.scale_floor))
    scale_intercept = layout.predictor("scale").intercept_index
    assert scale_intercept is not None
    assert result.converged is True
    np.testing.assert_allclose(
        result.coefficients[component.group_sl],
        0.0,
        rtol=0.0,
        atol=32.0 * config.tolerance,
    )
    np.testing.assert_allclose(
        result.coefficients[[0, scale_intercept]],
        [expected_location, expected_scale_coefficient],
        rtol=0.0,
        atol=32.0 * config.tolerance,
    )
    np.testing.assert_allclose(
        result.theta[:, 0],
        expected_location,
        rtol=0.0,
        atol=32.0 * config.tolerance,
    )
    np.testing.assert_allclose(
        result.theta[:, 1],
        expected_scale,
        rtol=0.0,
        atol=64.0 * config.tolerance,
    )
    assert result.coefficient_face is not None
    assert result.terminal_reduced_rank is not None
    assert result.terminal_reduced_rank.width == face.reduced_width
    projected_score = face.reduce_vector(result.terminal_score)
    assert np.linalg.norm(projected_score, ord=np.inf) <= (
        config.tolerance * (1.0 + abs(float(result.penalized_optimizing_log_likelihood)))
    )


def test_face_terminal_inverse_is_lifted_from_the_reduced_curvature() -> None:
    family, layout, _component, face, response, weights, penalty = _random_effect_problem()
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        config=DenseSolverConfig(tolerance=1.0e-10),
        coefficient_face=face,
    )

    covariance = result.terminal_pseudo_inverse()
    assert result.terminal_reduced_rank is not None
    rhs = np.linspace(-2.0, 3.0, layout.n_coefficients)
    expected_solve = face.lift_vector(result.terminal_reduced_rank.solve(face.reduce_vector(rhs)))
    np.testing.assert_allclose(
        result.solve_terminal(rhs),
        expected_solve,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        result.terminal_rank.solve(rhs),
        expected_solve,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        result.terminal_rank.pseudo_inverse(),
        covariance,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    curvature = result.terminal_penalized_curvature
    basis = face.null_basis
    reduced_curvature = basis.T @ curvature @ basis
    condition = float(np.linalg.cond(reduced_curvature))
    eps = np.finfo(np.float64).eps
    bound = 1024.0 * layout.n_coefficients * eps * max(condition, 1.0)
    np.testing.assert_allclose(
        basis.T @ (curvature @ covariance - np.eye(layout.n_coefficients)) @ basis,
        0.0,
        rtol=0.0,
        atol=bound,
    )
    projector = basis @ basis.T
    range_scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
    assert np.linalg.norm((np.eye(layout.n_coefficients) - projector) @ covariance, ord=2) <= (
        bound * range_scale
    )
    inference = compute_joint_inference(layout, result)
    np.testing.assert_allclose(
        inference.covariance,
        covariance,
        rtol=0.0,
        atol=64.0 * eps,
    )
    assert inference.rank == result.terminal_reduced_rank.rank


def test_face_execution_remains_generic_for_an_observed_only_family() -> None:
    _base, layout, component, face, response, weights, penalty = _random_effect_problem()
    family = _ObservedOnlyGaussian()

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),  # type: ignore[arg-type]
        penalty,
        config=DenseSolverConfig(coefficient_curvature="observed", tolerance=1.0e-10),
        coefficient_face=face,
    )

    assert result.converged is True
    assert result.terminal_curvature.actual_source == "observed"
    result.terminal_curvature.assert_no_fallback()
    np.testing.assert_allclose(
        result.coefficients[component.group_sl],
        0.0,
        rtol=0.0,
        atol=32.0 * result.config.tolerance,
    )
    projected_score = face.reduce_vector(result.terminal_score)
    assert np.linalg.norm(projected_score, ord=np.inf) <= (
        result.config.tolerance * (1.0 + abs(float(result.penalized_optimizing_log_likelihood)))
    )


def test_face_can_remove_the_entire_coefficient_space() -> None:
    groups = np.array(["a", "b", "c", "a", "b", "c"] * 3)
    response = np.linspace(-0.8, 0.9, len(groups))
    weights = np.ones(len(groups))
    frame = as_eager_frame(pd.DataFrame({"group": groups}))
    family = GaussianLS()
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            resolved_prior(weights),
            family.parameters,
            (
                Predictor("location", {"group": RandomEffect()}, intercept=False),
                Predictor("scale", {"group": RandomEffect()}, intercept=False),
            ),
        )
    )
    components = tuple(
        component for component in layout.penalties if component.penalty_kind == "identity"
    )
    assert len(components) == 2
    face = build_penalty_face(layout, tuple(component.name for component in components))
    assert face.reduced_width == 0

    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
        coefficient_face=face,
    )

    assert result.converged is True
    assert result.convergence_reason == "score"
    assert result.iterations == 0
    assert result.terminal_rank.width == layout.n_coefficients
    assert result.terminal_rank.rank == 0
    np.testing.assert_array_equal(result.coefficients, np.zeros(layout.n_coefficients))
    np.testing.assert_array_equal(
        result.terminal_pseudo_inverse(),
        np.zeros((layout.n_coefficients, layout.n_coefficients)),
    )


def test_face_fit_surfaces_are_invariant_to_alternate_full_coordinate_starts() -> None:
    family, layout, component, face, response, weights, penalty = _random_effect_problem()
    config = DenseSolverConfig(tolerance=1.0e-10)
    first_start = np.zeros(layout.n_coefficients)
    second_start = np.linspace(-8.0, 9.0, layout.n_coefficients)
    second_start[component.group_sl] = np.array([1.0e5, -2.0e5, 3.0e5])

    first = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=first_start,
        config=config,
        coefficient_face=face,
    )
    second = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        initial=second_start,
        config=config,
        coefficient_face=face,
    )

    scale = max(1.0, float(np.linalg.norm(first.eta, ord=np.inf)))
    np.testing.assert_allclose(
        second.eta,
        first.eta,
        rtol=0.0,
        atol=64.0 * config.tolerance * scale,
    )
    np.testing.assert_allclose(
        second.theta,
        first.theta,
        rtol=0.0,
        atol=128.0 * config.tolerance * scale,
    )


def test_chunked_random_effect_face_matches_dense_execution() -> None:
    family, layout, _component, face, response, weights, penalty = _random_effect_problem()
    config = DenseSolverConfig(tolerance=1.0e-10, max_iterations=120)
    dense = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        coefficient_face=face,
        config=config,
    )
    chunked = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        _plan(family, response, weights),
        penalty,
        coefficient_face=face,
        chunk_size=6,
        config=config,
    )

    assert chunked.execution_backend_identifier == "distributional-chunked-v1"
    assert chunked.converged is dense.converged is True
    fit_scale = max(
        1.0,
        float(np.linalg.norm(dense.coefficients, ord=np.inf)),
        float(np.linalg.norm(dense.eta, ord=np.inf)),
        float(np.linalg.norm(dense.theta, ord=np.inf)),
    )
    fit_bound = 4.0 * config.tolerance * fit_scale
    np.testing.assert_allclose(chunked.coefficients, dense.coefficients, rtol=0.0, atol=fit_bound)
    np.testing.assert_allclose(chunked.eta, dense.eta, rtol=0.0, atol=fit_bound)
    np.testing.assert_allclose(chunked.theta, dense.theta, rtol=0.0, atol=fit_bound)
    dense_covariance = dense.terminal_pseudo_inverse()
    reduced_curvature = face.reduce_matrix(dense.terminal_penalized_curvature)
    covariance_bound = (
        1024.0
        * face.width
        * np.finfo(np.float64).eps
        * max(1.0, float(np.linalg.cond(reduced_curvature)))
        * max(1.0, float(np.linalg.norm(dense_covariance, ord=2)))
    )
    np.testing.assert_allclose(
        chunked.terminal_pseudo_inverse(),
        dense_covariance,
        rtol=0.0,
        atol=covariance_bound,
    )
    assert np.linalg.norm(face.constraint_matrix @ chunked.coefficients, ord=2) <= (
        face.null_residual_bound
    )
