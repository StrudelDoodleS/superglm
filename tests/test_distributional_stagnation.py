from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import superglm.distributional.results.solver as results_solver
import superglm.distributional.solver.solver as solver_module
from superglm._frame import as_eager_frame
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.layout import build_stacked_layout
from superglm.distributional.penalty_face import build_penalty_face
from superglm.distributional.predictor import Predictor, compile_predictors
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.results.solver import _newton_decrement_is_certified
from superglm.distributional.solver import fit_dense_fixed_lambda
from superglm.distributional.solver.curvature import CurvaturePolicyState
from superglm.features import Numeric, RandomEffect

from ._distributional_weights import resolved_prior
from .test_distributional_endpoint_laml import _UnitGaussian


def _no_op_problem():
    family = _UnitGaussian()
    response = np.array([1.0, 1.0, np.nextafter(1.0, np.inf)])
    weights = resolved_prior(np.ones(response.size))
    frame = as_eager_frame(
        pd.DataFrame(
            {
                "x": np.ones(response.size),
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
    face = build_penalty_face(layout, (layout.penalty_names[0],))
    plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    initial = face.project(np.array([1.0, 0.0, 0.0, 0.0]))
    config = DenseSolverConfig(max_iterations=2, tolerance=1.0e-20)
    penalty = layout.penalty_matrix({layout.penalty_names[0]: 0.0})
    return family, response, layout, face, plan, initial, config, penalty


def _fit_no_op(monkeypatch: pytest.MonkeyPatch, *, shifted: bool = False):
    (
        family,
        response,
        layout,
        face,
        plan,
        initial,
        config,
        penalty,
    ) = _no_op_problem()
    if shifted:
        original = solver_module._solve_coefficient_direction

        def shifted_direction(*args, **kwargs):
            direction = original(*args, **kwargs)
            return replace(direction, levenberg_shift=1.0)

        monkeypatch.setattr(solver_module, "_solve_coefficient_direction", shifted_direction)
    return fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=initial,
        config=config,
        coefficient_face=face,
    )


def test_identical_candidate_with_shifted_direction_is_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_no_op(monkeypatch, shifted=True)

    assert result.converged is False
    assert result.convergence_reason == "line_search_failed"
    assert result.score_relative > result.config.tolerance


def test_accepted_tiny_step_with_shifted_direction_is_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        family,
        response,
        layout,
        face,
        plan,
        initial,
        config,
        penalty,
    ) = _no_op_problem()
    initial = face.project(np.zeros(layout.n_coefficients))
    config = replace(config, tolerance=1.0e-10)
    original = solver_module._solve_coefficient_direction

    def tiny_shifted_direction(*args, **kwargs):
        direction = original(*args, **kwargs)
        step = direction.step * 1.0e-12
        return replace(direction, step=step, levenberg_shift=1.0)

    monkeypatch.setattr(solver_module, "_solve_coefficient_direction", tiny_shifted_direction)
    result = fit_dense_fixed_lambda(
        family,
        layout,
        response,
        plan,
        penalty,
        initial=initial,
        config=config,
        coefficient_face=face,
    )

    assert result.converged is False
    assert result.convergence_reason == "max_iterations"
    assert result.iterations == config.max_iterations
    assert result.score_relative > result.config.tolerance


def test_unshifted_resolution_limited_identical_step_keeps_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_no_op(monkeypatch)

    assert result.converged is True
    assert result.convergence_reason == "objective_and_step"
    assert result.score_relative > result.config.tolerance


def test_objective_and_step_result_rejects_corrupted_retained_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_no_op(monkeypatch)
    corrupted_score = result.terminal_score * 1.0e12

    with pytest.raises(ValueError, match="objective-and-step"):
        replace(result, terminal_score=corrupted_score)


def test_ill_conditioned_analytic_decrement_uses_a_conservative_enclosure() -> None:
    """A cancellation-sensitive quadratic cannot certify below its exact gap."""
    off_diagonal = 1.0 - 1.0e-6
    score_component = 1.0e-5
    curvature = np.array(
        [[1.0, off_diagonal], [off_diagonal, 1.0]],
        dtype=np.float64,
    )
    score = np.array([score_component, -score_component], dtype=np.float64)
    # The score is an eigenvector of this SPD matrix.  Using the represented
    # float entries as exact inputs, the decrement is 2 s² / (1 - rho).
    exact_decrement = (
        Fraction(2)
        * Fraction.from_float(score_component) ** 2
        / (Fraction(1) - Fraction.from_float(off_diagonal))
    )
    condition = float(np.linalg.cond(curvature))
    epsilon = float(np.finfo(np.float64).eps)
    enclosure_gap = float(
        np.nextafter(
            256.0 * curvature.shape[0] * epsilon * max(1.0, condition) * float(exact_decrement),
            np.inf,
        )
    )
    below_exact = float(exact_decrement) - enclosure_gap
    above_enclosure = float(exact_decrement) + 2.0 * enclosure_gap
    config = DenseSolverConfig(residual_tolerance=1.0e-7)

    assert float(exact_decrement) > below_exact
    assert not _newton_decrement_is_certified(
        config=config,
        score=score,
        penalized_curvature=curvature,
        penalized_objective=0.0,
        face=None,
        tolerance=below_exact,
    )
    assert _newton_decrement_is_certified(
        config=config,
        score=score,
        penalized_curvature=curvature,
        penalized_objective=0.0,
        face=None,
        tolerance=above_enclosure,
    )


def test_newton_certificate_accounts_for_a_permitted_approximate_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A residual-policy solve error cannot be hidden by a raw decrement midpoint."""
    score = np.ones(2, dtype=np.float64)
    curvature = np.eye(2, dtype=np.float64)
    delta = float(np.finfo(np.float64).eps ** 0.25)
    approximate_decrement = 2.0 * (1.0 - delta)
    exact_decrement = 2.0
    tolerance = 0.5 * (approximate_decrement + exact_decrement)
    config = DenseSolverConfig(residual_tolerance=1.0e-2)
    real_decompose = results_solver.decompose_gram

    class ApproximateDecomposition:
        def __init__(self, decomposition: object) -> None:
            self._decomposition = decomposition

        def __getattr__(self, name: str) -> object:
            return getattr(self._decomposition, name)

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            return (1.0 - delta) * self._decomposition.solve(rhs)  # type: ignore[attr-defined]

        def pseudo_inverse(self) -> np.ndarray:
            return self._decomposition.pseudo_inverse()  # type: ignore[attr-defined]

    def approximate_decompose(*args: object, **kwargs: object) -> ApproximateDecomposition:
        decomposition = real_decompose(*args, **kwargs)  # type: ignore[arg-type]
        assert decomposition is not None
        return ApproximateDecomposition(decomposition)

    monkeypatch.setattr(results_solver, "decompose_gram", approximate_decompose)
    assert approximate_decrement < tolerance < exact_decrement
    residual = np.linalg.norm((1.0 - delta) * score - score)
    assert residual / np.linalg.norm(score) < config.residual_tolerance
    legacy_operation_error = (
        2.0
        * score.size
        * np.finfo(np.float64).eps
        / (1.0 - 2.0 * score.size * np.finfo(np.float64).eps)
    )
    legacy_rounding = float(
        np.nextafter(
            16.0
            * legacy_operation_error
            * max(2.0 * (1.0 - delta), approximate_decrement, tolerance),
            np.inf,
        )
    )
    # The old dot-product-only certificate would accept this mutation.
    assert approximate_decrement + legacy_rounding < tolerance
    assert not _newton_decrement_is_certified(
        config=config,
        score=score,
        penalized_curvature=curvature,
        penalized_objective=0.0,
        face=None,
        tolerance=tolerance,
    )


@pytest.mark.parametrize("scale", (1.0, 1.0e6, 1.0e12))
def test_newton_certificate_is_invariant_to_feature_units(scale: float) -> None:
    """Changing one feature's units must not change a stable decrement stop."""
    curvature = np.diag(np.array([scale**2, 1.0], dtype=np.float64))
    score = np.array([scale * 1.0e-5, 1.0e-5], dtype=np.float64)
    result = _newton_decrement_is_certified(
        config=DenseSolverConfig(residual_tolerance=1.0e-7),
        score=score,
        penalized_curvature=curvature,
        penalized_objective=0.0,
        face=None,
        tolerance=1.0e-8,
    )

    assert result is True


def test_public_solver_keeps_a_resolved_near_collinear_newton_stop() -> None:
    """A resolved, ill-conditioned SPD fit remains a valid decrement stop."""
    rho = 1.0 - 1.0e-9
    design = np.array(
        [[1.0, rho], [0.0, np.sqrt(1.0 - rho**2)]],
        dtype=np.float64,
    )
    target = np.array([1.0e-5, 1.0e-5], dtype=np.float64)
    response = design @ target
    family = _UnitGaussian()
    weights = resolved_prior(np.ones(response.size, dtype=np.float64))
    frame = as_eager_frame(pd.DataFrame({"x1": design[:, 0], "x2": design[:, 1]}))
    layout = build_stacked_layout(
        compile_predictors(
            frame,
            weights,
            family.parameters,
            (Predictor("mean", {"x1": Numeric(), "x2": Numeric()}, intercept=False),),
        )
    )
    likelihood_plan = family.bind_likelihood(response, weights, COMPLETE_OBSERVATION)
    result = fit_dense_fixed_lambda(
        family,  # type: ignore[arg-type]
        layout,
        response,
        likelihood_plan,  # type: ignore[arg-type]
        np.zeros((2, 2), dtype=np.float64),
        initial=np.zeros(2, dtype=np.float64),
        config=DenseSolverConfig(
            max_iterations=4,
            tolerance=1.0e-8,
            newton_decrement_tolerance=1.0e-8,
            coefficient_curvature="observed",
        ),
    )

    assert result.converged is True
    assert result.convergence_reason == "newton_decrement"
    assert result.iterations == 0
    assert result.terminal_rank.rank == 2
    assert result.terminal_curvature.fallback_count == 0
    assert result.score_relative > result.config.tolerance


def test_retry_failure_replaces_the_prior_convergence_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = solver_module._run_iterations
    run_calls = 0

    def forced_retry_verdict(*args, **kwargs):
        nonlocal run_calls
        run_calls += 1
        run = original_run(*args, **kwargs)
        if run_calls == 1:
            return replace(run, converged=True, reason="objective_and_step")
        return replace(run, converged=False, reason="line_search_failed")

    monkeypatch.setattr(solver_module, "_run_iterations", forced_retry_verdict)
    original_resolve = solver_module.resolve_curvature
    resolve_calls = 0

    def force_one_terminal_retry(*args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        decision = original_resolve(*args, **kwargs)
        if resolve_calls == 1:
            return replace(
                decision,
                matrix=None,
                decomposition=None,
                retry_required=True,
                state=CurvaturePolicyState(retry_attempted=True),
            )
        return decision

    monkeypatch.setattr(solver_module, "resolve_curvature", force_one_terminal_retry)
    result = _fit_no_op(monkeypatch)

    assert run_calls == 2
    assert resolve_calls == 2
    assert result.converged is False
    assert result.convergence_reason == "line_search_failed"
