"""Final-state feasibility regressions for hard fit-time constraints."""

import logging
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import BSplineSmooth, Constraint, SuperGLM


def _one_coefficient_line_search_problem(*, constrained: bool):
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.types import GroupSlice, LinearConstraintSet

    x = np.array([-1.0, 1.0, -2.0, 2.0])[:, None]
    design = DesignMatrix([DenseGroupMatrix(x)], n=len(x), p=1)
    constraint_set = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1)) if constrained else None
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=constraint_set,
            monotone_engine="qp" if constrained else None,
        )
    ]
    return design, -x[:, 0], np.ones(len(x)), groups


def _healthy_increasing_problem() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0.0, 1.0, 200))
    y = 2.0 * x + rng.normal(0.0, 0.2, len(x))
    return pd.DataFrame({"x": x}), y


def _increasing_model() -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": BSplineSmooth(
                n_knots=8,
                constraint=Constraint.fit.increasing,
            )
        },
    )


@pytest.mark.parametrize("fit_method", ["fit", "fit_reml"])
def test_healthy_constrained_fit_finishes_with_feasible_coefficients_and_predictions(
    fit_method: str,
) -> None:
    """The terminal check accepts ordinary feasible fit and REML results."""
    frame, y = _healthy_increasing_problem()
    model = _increasing_model()

    getattr(model, fit_method)(frame, y)

    assert model.result.converged
    assert model.result.termination_reason == "converged"
    group = model._groups[0]
    assert group.constraints is not None
    coefficient_slack = group.constraints.A @ model.result.beta[group.sl] - group.constraints.b
    assert np.min(coefficient_slack) >= -1e-10

    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 401)})
    prediction_steps = np.diff(model.predict(grid))
    assert np.min(prediction_steps) >= -1e-10
    if fit_method == "fit_reml":
        assert model._reml_result is not None
        assert model._reml_result.converged


def test_qp_reml_restores_constraints_in_terminal_ssp_coordinates() -> None:
    """Passthrough constraints must follow the lambda-dependent terminal basis."""
    from superglm.constraints import shape_constraint_certificate

    x = np.linspace(0.0, 1.0, 300)
    frame = pd.DataFrame({"x": x})
    y = (x - 0.35) ** 2
    model = SuperGLM(
        family="gaussian",
        features={
            "x": BSplineSmooth(
                n_knots=8,
                constraint=Constraint.fit.convex,
            )
        },
    ).fit_reml(frame, y)

    group = model._groups[0]
    group_matrix = model._dm.group_matrices[0]
    spec = model._specs["x"]
    assert group.constraints is not None
    terminal_constraints = spec._build_monotone_constraints_raw().compose(group_matrix.R_inv)
    np.testing.assert_allclose(
        group.constraints.A,
        terminal_constraints.A,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(group.constraints.b, terminal_constraints.b)

    coefficient_slack = (
        terminal_constraints.A @ model.result.beta[group.sl] - terminal_constraints.b
    )
    assert np.min(coefficient_slack) >= -1e-10
    certificate = shape_constraint_certificate(
        spec,
        model.result.beta[group.sl],
        "convex",
    )
    assert certificate.minimum_scaled_slack >= -1e-10


def test_qp_reml_repairs_an_infeasible_unconstrained_mode() -> None:
    """The terminal QP refit must replace, not publish, the REML surrogate mode."""
    x = np.linspace(0.0, 1.0, 300)
    frame = pd.DataFrame({"x": x})
    y = 1.0 - 2.0 * x + 0.02 * np.sin(12.0 * x)
    model = _increasing_model().fit_reml(frame, y)

    assert model.result.converged
    assert model.result.termination_reason == "converged"
    group = model._groups[0]
    assert group.constraints is not None
    coefficient_slack = group.constraints.A @ model.result.beta[group.sl] - group.constraints.b
    assert np.min(coefficient_slack) >= -1e-10

    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 401)})
    prediction_steps = np.diff(model.predict(grid))
    assert np.min(prediction_steps) >= -1e-10
    assert np.ptp(model.predict(grid)) <= 1e-10


@pytest.mark.parametrize(
    ("missing_state", "message"),
    [
        pytest.param(
            "spec",
            "its fitted spline specification is unavailable",
            id="missing-specification",
        ),
        pytest.param(
            "raw",
            "its raw constraint geometry is unavailable",
            id="missing-raw-geometry",
        ),
        pytest.param(
            "map",
            "its current solver-coordinate map is unavailable",
            id="missing-coordinate-map",
        ),
    ],
)
def test_qp_constraint_restore_refuses_stale_solver_coordinates(
    missing_state: str,
    message: str,
) -> None:
    """A live design must never receive the saved pre-REML constraint matrix."""
    from superglm.model.base import model_build_design_matrix
    from superglm.model.reml_setup import restore_qp_constraints, strip_qp_constraints

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    model_build_design_matrix(model, frame, y, np.ones_like(y), None)
    saved_state = strip_qp_constraints(model._groups)

    assert len(saved_state) == 1
    group_index = saved_state[0][0]
    group = model._groups[group_index]
    assert group.constraints is None

    if missing_state == "spec":
        model._specs.pop(group.feature_name)
        model._interaction_specs.pop(group.feature_name, None)
    elif missing_state == "raw":
        model._specs[group.feature_name] = object()
    else:
        group_matrices = list(model._dm.group_matrices)
        group_matrices[group_index] = object()
        model._dm.group_matrices = tuple(group_matrices)

    with pytest.raises(RuntimeError, match=message):
        restore_qp_constraints(model, saved_state)

    assert group.monotone_engine == "qp"
    assert group.constraints is None


def test_qp_constraint_restore_refuses_saved_coordinates_after_design_release() -> None:
    """Released fit state must already carry its current-coordinate constraint."""
    from superglm.model.base import model_build_design_matrix
    from superglm.model.reml_setup import restore_qp_constraints, strip_qp_constraints

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    model_build_design_matrix(model, frame, y, np.ones_like(y), None)
    saved_state = strip_qp_constraints(model._groups)
    group = model._groups[saved_state[0][0]]

    model._dm = None
    with pytest.raises(
        RuntimeError,
        match="fitted design was released before current-coordinate constraints were restored",
    ):
        restore_qp_constraints(model, saved_state)

    assert group.monotone_engine == "qp"
    assert group.constraints is None


def test_qp_passthrough_finally_preserves_restored_constraints_after_state_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finally path keeps current constraints after retain_fit_state=False."""
    import superglm.model.fit_ops as fit_ops

    frame, y = _healthy_increasing_problem()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        retain_fit_state=False,
        features={
            "x": BSplineSmooth(
                n_knots=8,
                constraint=Constraint.fit.increasing,
            )
        },
    )
    real_restore = fit_ops.restore_qp_constraints
    finally_calls: list[tuple[bool, bool, bool]] = []

    def record_finally_restore(work_model, saved_state):
        group = work_model._groups[saved_state[0][0]]
        current_constraints = group.constraints
        assert current_constraints is not None
        assert current_constraints is not saved_state[0][2]
        dm_was_released = work_model._dm is None

        real_restore(work_model, saved_state)

        finally_calls.append(
            (
                dm_was_released,
                group.constraints is current_constraints,
                group.monotone_engine == "qp",
            )
        )

    monkeypatch.setattr(fit_ops, "restore_qp_constraints", record_finally_restore)
    model.fit_reml(frame, y, max_reml_iter=8, runtime_validation="skip")

    assert finally_calls == [(True, True, True)]
    assert model._dm is None
    assert model.result.converged
    assert model.result.termination_reason == "converged"
    assert model._groups[0].constraints is not None

    grid = pd.DataFrame({"x": np.linspace(0.0, 1.0, 401)})
    assert np.min(np.diff(model.predict(grid))) >= -1e-10


@pytest.mark.parametrize(
    "centered",
    [
        pytest.param(False, id="non-centered"),
        pytest.param(True, id="centered-control"),
    ],
)
def test_constrained_profiled_intercept_matches_full_augmented_reference(
    centered: bool,
) -> None:
    """The constrained slope QP uses the intercept Schur-complement Hessian."""
    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import GroupSlice, LinearConstraintSet

    x = np.array([1.0, 2.0, 4.0, 7.0])
    if centered:
        x = x - np.mean(x)
    X = x[:, None]
    y = 3.0 + 2.0 * x
    weights = np.ones(len(x))
    penalty = np.array([[3.0]])
    design = DesignMatrix([DenseGroupMatrix(X)], n=len(x), p=1)
    nonnegative = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))

    def solve(*, constrained: bool):
        group = GroupSlice(
            "x",
            0,
            1,
            constraints=nonnegative if constrained else None,
            monotone_engine="qp" if constrained else None,
        )
        return fit_irls_direct(
            design,
            y,
            weights,
            Gaussian(),
            IdentityLink(),
            [group],
            lambda2=0.0,
            S_override=penalty,
            beta_init=np.zeros(1),
            intercept_init=float(np.mean(y)),
            max_iter=1,
        )[0]

    ordinary = solve(constrained=False)
    constrained = solve(constrained=True)

    augmented_design = np.column_stack([np.ones(len(x)), X])
    augmented_hessian = augmented_design.T @ augmented_design
    augmented_hessian[1:, 1:] += penalty
    reference = np.linalg.solve(augmented_hessian, augmented_design.T @ y)
    assert reference[1] > 0.0  # the beta >= 0 row is deliberately inactive

    for result in (ordinary, constrained):
        np.testing.assert_allclose(
            np.concatenate([[result.intercept], result.beta]),
            reference,
            rtol=1e-12,
            atol=1e-12,
        )

    np.testing.assert_allclose(constrained.beta, ordinary.beta, rtol=1e-12, atol=1e-12)
    assert constrained.intercept == pytest.approx(ordinary.intercept, abs=1e-12)
    assert constrained.deviance == pytest.approx(ordinary.deviance, abs=1e-12)
    ordinary_merit = ordinary.deviance + float(ordinary.beta @ penalty @ ordinary.beta)
    constrained_merit = constrained.deviance + float(constrained.beta @ penalty @ constrained.beta)
    assert constrained_merit == pytest.approx(ordinary_merit, abs=1e-12)

    if not centered:
        # Liveness: pairing the profiled RHS with the unprofiled slope block
        # gives a materially different answer on this fixture.
        profiled_rhs = float((X.T @ y).item() - np.sum(y) * np.sum(X) / len(x))
        omitted_schur_beta = profiled_rhs / float((X.T @ X + penalty).item())
        assert abs(omitted_schur_beta - reference[1]) > 1.0


@pytest.mark.parametrize(
    "centered",
    [
        pytest.param(False, id="non-centered"),
        pytest.param(True, id="centered-control"),
    ],
)
def test_active_constrained_profiled_intercept_matches_full_augmented_kkt(
    centered: bool,
) -> None:
    """An active slope constraint agrees with the full intercept+slope KKT system."""
    from superglm.distributions import Gaussian
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.types import GroupSlice, LinearConstraintSet

    raw_design = np.column_stack(
        [
            np.arange(6.0),
            np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ]
    )
    y = 1.0 + 2.0 * raw_design[:, 0] - raw_design[:, 1]
    X = raw_design - np.mean(raw_design, axis=0) if centered else raw_design
    weights = np.ones(len(y))
    design = DesignMatrix([DenseGroupMatrix(X)], n=len(y), p=2)
    constraint_matrix = np.array([[-1.0, 1.0]])
    constraint_set = LinearConstraintSet(A=constraint_matrix, b=np.zeros(1))
    group = GroupSlice(
        "x",
        0,
        2,
        constraints=constraint_set,
        monotone_engine="qp",
    )

    result, _ = fit_irls_direct(
        design,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        [group],
        lambda2=0.0,
        beta_init=np.zeros(2),
        intercept_init=float(np.mean(y)),
        max_iter=1,
    )

    augmented_design = np.column_stack([np.ones(len(y)), X])
    full_hessian = augmented_design.T @ augmented_design
    full_constraint = np.column_stack([np.zeros(1), constraint_matrix])
    full_kkt = np.block(
        [
            [full_hessian, -full_constraint.T],
            [full_constraint, np.zeros((1, 1))],
        ]
    )
    reference = np.linalg.solve(
        full_kkt,
        np.concatenate([augmented_design.T @ y, np.zeros(1)]),
    )

    np.testing.assert_allclose(
        np.concatenate([[result.intercept], result.beta]),
        reference[:3],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(constraint_matrix @ result.beta, [0.0], atol=1e-12)
    assert reference[-1] > 0.0

    if centered:
        np.testing.assert_allclose(np.sum(X, axis=0), 0.0, atol=1e-12)
    else:
        # Liveness: the former unprofiled slope block, paired with the
        # profiled RHS, solves to a materially different active-boundary mode.
        profiled_rhs = X.T @ y - X.T @ np.ones(len(y)) * np.mean(y)
        old_kkt = np.block(
            [
                [X.T @ X, -constraint_matrix.T],
                [constraint_matrix, np.zeros((1, 1))],
            ]
        )
        old_reference = np.linalg.solve(
            old_kkt,
            np.concatenate([profiled_rhs, np.zeros(1)]),
        )
        assert np.linalg.norm(old_reference[:2] - reference[1:3]) > 1.0


def test_line_searched_infeasible_final_state_is_not_converged(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tiny accepted step cannot certify an infeasible warm-start state."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm.model.base import model_build_design_matrix
    from superglm.solvers.irls_state import _IRLSStepDecision

    x = np.linspace(0.0, 1.0, 80)
    frame = pd.DataFrame({"x": x})
    y = 0.5 + 2.0 * x + 0.02 * np.sin(8.0 * x)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": BSplineSmooth(
                n_knots=6,
                constraint=Constraint.fit.increasing,
            )
        },
    )
    y_fit, weights, offset = model_build_design_matrix(
        model,
        frame,
        y,
        np.ones_like(y),
        None,
    )
    group = model._groups[0]
    assert group.constraints is not None

    infeasible_beta = np.zeros(model._dm.p)
    infeasible_beta[group.sl] = -10.0 * group.constraints.A[0]
    initial_slack = group.constraints.A @ infeasible_beta[group.sl] - group.constraints.b
    assert np.min(initial_slack) < -1.0

    accepted: dict[str, np.ndarray | float] = {}

    def accept_tiny_line_search_step(**kwargs):
        alpha = 2.0**-19
        trial = kwargs["evaluate_state"](alpha)
        accepted["alpha"] = alpha
        accepted["proposal"] = kwargs["proposal"].beta.copy()
        accepted["trial"] = trial.beta.copy()
        return _IRLSStepDecision(
            alpha=alpha,
            step_halvings=19,
            step_rejected=False,
            trials_attempted=20,
        )

    monkeypatch.setattr(irls_direct, "_select_irls_trial", accept_tiny_line_search_step)
    with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
        result, _ = irls_direct.fit_irls_direct(
            model._dm,
            y_fit,
            weights,
            model._distribution,
            model._link,
            model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            beta_init=infeasible_beta,
            intercept_init=0.0,
            max_iter=1,
            tol=1e-6,
            record_diagnostics=True,
        )

    proposal = np.asarray(accepted["proposal"])
    proposal_slack = group.constraints.A @ proposal[group.sl] - group.constraints.b
    assert np.min(proposal_slack) >= -1e-10
    np.testing.assert_array_equal(result.beta, accepted["trial"])

    final_slack = group.constraints.A @ result.beta[group.sl] - group.constraints.b
    assert np.min(final_slack) < -1.0
    grid_basis = model._specs["x"].transform(np.linspace(0.0, 1.0, 401))
    retained_term = grid_basis @ result.beta[group.sl]
    assert np.min(np.diff(retained_term)) < -0.01
    assert not result.converged
    assert result.termination_reason == "constraint_infeasible"
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == accepted["alpha"]
    primal_warnings = [
        record
        for record in caplog.records
        if "retained coefficient mode violates hard constraints" in record.getMessage()
    ]
    assert len(primal_warnings) == 1
    assert primal_warnings[0].levelno == logging.WARNING


def test_rejected_infeasible_state_has_one_terminal_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostics and the public result must agree after an atomic rejection."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm._fit_trace import MemoryTraceSink, TraceRun
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult
    from superglm.solvers.irls_state import _IRLSStepDecision

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)

    def infeasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([-2.0]), converged=False)

    def reject_every_trial(**_kwargs):
        return _IRLSStepDecision(
            alpha=0.0,
            step_halvings=20,
            step_rejected=True,
            trials_attempted=20,
        )

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", infeasible_qp_proposal)
    monkeypatch.setattr(irls_direct, "_select_irls_trial", reject_every_trial)
    trace_sink = MemoryTraceSink()
    result, _ = irls_direct.fit_irls_direct(
        design,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        beta_init=np.array([-1.0]),
        intercept_init=0.0,
        max_iter=2,
        record_diagnostics=True,
        trace_run=TraceRun("infeasible-rejection", sink=trace_sink),
    )

    assert not result.converged
    assert result.termination_reason == "constraint_infeasible"
    assert result.iteration_log is not None
    assert len(result.iteration_log) == 1
    assert result.iteration_log[-1].step_rejected
    assert result.iteration_log[-1].termination_reason == result.termination_reason
    decisions = [event for event in trace_sink.events if event.event_kind == "step_decision"]
    assert decisions[-1].payload["step_rejected"]
    assert decisions[-1].payload["termination_reason"] == result.termination_reason


def test_qp_reml_refuses_infeasible_terminal_state_before_objective_or_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constrained terminal state gates REML objective and public install."""
    from superglm.model import reml_finalize

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    terminal_objective_calls: list[None] = []

    def force_infeasible_terminal(
        work_model,
        *,
        qp_saved_state,
        pirls_result,
        **_kwargs,
    ):
        reml_finalize.restore_qp_group_state(work_model, qp_saved_state)
        group = next(group for group in work_model._groups if group.constraints is not None)
        infeasible_beta = pirls_result.beta.copy()
        infeasible_beta[group.sl] = -10.0 * group.constraints.A[0]
        slack = group.constraints.A @ infeasible_beta[group.sl] - group.constraints.b
        assert np.min(slack) < -1.0
        return replace(
            pirls_result,
            beta=infeasible_beta,
            converged=False,
            termination_reason="constraint_infeasible",
        )

    def unexpected_terminal_objective(*_args, **_kwargs):
        terminal_objective_calls.append(None)
        raise AssertionError("terminal REML objective ran before the constraint gate")

    monkeypatch.setattr(
        reml_finalize,
        "maybe_qp_passthrough_refit",
        force_infeasible_terminal,
    )
    monkeypatch.setattr(
        reml_finalize,
        "reml_laml_objective",
        unexpected_terminal_objective,
    )

    with pytest.raises(
        RuntimeError,
        match="terminal constrained REML refit ended at an infeasible coefficient mode",
    ):
        model.fit_reml(frame, y, max_reml_iter=8, runtime_validation="skip")

    assert terminal_objective_calls == []
    assert model._result is None
    assert getattr(model, "_reml_result", None) is None


def test_fixed_lambda_qp_reml_refuses_infeasible_result_before_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-lambda route gates infeasibility before installing fit state."""
    import superglm.model.reml_execute as reml_execute
    from superglm import LambdaPolicy

    frame, y = _healthy_increasing_problem()
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x": BSplineSmooth(
                n_knots=8,
                constraint=Constraint.fit.increasing,
                lambda_policy=LambdaPolicy.fixed(1.0),
            )
        },
    )
    real_fit = reml_execute.fit_irls_direct
    fixed_route_calls: list[dict[str, str]] = []

    def force_infeasible_fixed_result(*args, **kwargs):
        result, inverse = real_fit(*args, **kwargs)
        group = next(group for group in kwargs["groups"] if group.constraints is not None)
        infeasible_beta = result.beta.copy()
        infeasible_beta[group.sl] = -10.0 * group.constraints.A[0]
        slack = group.constraints.A @ infeasible_beta[group.sl] - group.constraints.b
        assert np.min(slack) < -1.0
        fixed_route_calls.append(kwargs["debug_context"])
        return (
            replace(
                result,
                beta=infeasible_beta,
                converged=False,
                termination_reason="constraint_infeasible",
            ),
            inverse,
        )

    monkeypatch.setattr(
        reml_execute,
        "fit_irls_direct",
        force_infeasible_fixed_result,
    )
    with pytest.raises(
        RuntimeError,
        match="fixed-lambda constrained REML fit ended at an infeasible coefficient mode",
    ):
        model.fit_reml(frame, y, runtime_validation="skip")

    assert fixed_route_calls == [{"phase": "fixed_constraint"}]
    assert model._result is None
    assert model._solver_result is None
    assert getattr(model, "_reml_result", None) is None
    assert getattr(model, "_reml_lambdas", None) is None
    assert model._fit_stats is None


def test_infeasible_committed_state_prefers_finite_feasible_proposal_over_merit() -> None:
    """Primal feasibility outranks the merit increase from repairing a warm start."""
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import fit_irls_direct

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    result, _ = fit_irls_direct(
        design,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        beta_init=np.array([-1.0]),
        intercept_init=0.0,
        max_iter=1,
        record_diagnostics=True,
    )

    # beta=-1 is an exact but infeasible fit with deviance zero. The
    # constrained QP proposal beta=0 is feasible but has worse merit.
    np.testing.assert_allclose(result.beta, [0.0], atol=1e-12)
    assert result.deviance > 0.0
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == 1.0
    assert not result.iteration_log[0].step_rejected
    assert result.termination_reason != "constraint_infeasible"


def test_feasible_committed_state_never_accepts_infeasible_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merit-improving infeasible proposal is truncated at the feasible boundary."""
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers import irls_direct
    from superglm.solvers.constrained_qp import QPResult

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)

    def infeasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([-1.0]), converged=True)

    monkeypatch.setattr(
        irls_direct,
        "solve_constrained_qp",
        infeasible_qp_proposal,
    )
    result, _ = irls_direct.fit_irls_direct(
        design,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        beta_init=np.array([1.0]),
        intercept_init=0.0,
        max_iter=1,
        record_diagnostics=True,
    )

    # The full beta=-1 proposal has the best merit but violates beta >= 0.
    # The half step lands exactly on the feasible boundary.
    np.testing.assert_allclose(result.beta, [0.0], atol=1e-12)
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == 0.5
    assert result.iteration_log[0].step_halvings == 1
    assert result.beta[0] >= -1e-12
    assert result.termination_reason != "constraint_infeasible"


def test_unconstrained_line_search_still_accepts_the_full_merit_improving_proposal() -> None:
    """The feasibility rule is inert when no hard constraints are present."""
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import fit_irls_direct

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=False)
    result, _ = fit_irls_direct(
        design,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        beta_init=np.array([1.0]),
        intercept_init=0.0,
        max_iter=1,
        record_diagnostics=True,
    )

    np.testing.assert_allclose(result.beta, [-1.0], atol=1e-12)
    assert result.deviance == pytest.approx(0.0, abs=1e-20)
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == 1.0
    assert result.iteration_log[0].step_halvings == 0
