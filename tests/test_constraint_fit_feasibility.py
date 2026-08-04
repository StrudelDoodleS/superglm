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


@pytest.mark.parametrize("cold_converged", [False, True])
def test_failed_warm_qp_retries_cold_without_weakening_certificate(
    monkeypatch: pytest.MonkeyPatch,
    cold_converged: bool,
) -> None:
    """A warm active set is only an optimization hint for the same QP."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm.solvers.constrained_qp import QPResult

    warm_result = QPResult(
        beta=np.array([0.25]),
        active_set=[0],
        converged=False,
    )
    cold_result = QPResult(
        beta=np.array([0.5]),
        active_set=[],
        converged=cold_converged,
    )
    active_sets: list[list[int] | None] = []

    def fake_solve(*_args, active_set_init=None, **_kwargs):
        active_sets.append(active_set_init)
        return warm_result if active_set_init is not None else cold_result

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", fake_solve)
    result = irls_direct._solve_constrained_qp_with_cold_retry(
        np.eye(1),
        np.ones(1),
        np.ones((1, 1)),
        np.zeros(1),
        [0],
    )

    assert active_sets == [[0], None]
    assert result is (cold_result if cold_converged else warm_result)


@pytest.mark.parametrize(
    ("active_set_init", "converged"),
    [
        pytest.param(None, False, id="cold-start"),
        pytest.param([], False, id="empty-warm-set"),
        pytest.param([0], True, id="certified-warm-result"),
    ],
)
def test_qp_cold_retry_is_absent_from_the_normal_solve_path(
    monkeypatch: pytest.MonkeyPatch,
    active_set_init: list[int] | None,
    converged: bool,
) -> None:
    """Cold retry adds no work unless a nonempty warm active set fails."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm.solvers.constrained_qp import QPResult

    result = QPResult(
        beta=np.array([0.25]),
        active_set=[] if active_set_init is None else active_set_init,
        converged=converged,
    )
    active_sets: list[list[int] | None] = []

    def fake_solve(*_args, active_set_init=None, **_kwargs):
        active_sets.append(active_set_init)
        return result

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", fake_solve)
    observed = irls_direct._solve_constrained_qp_with_cold_retry(
        np.eye(1),
        np.ones(1),
        np.ones((1, 1)),
        np.zeros(1),
        active_set_init,
    )

    assert active_sets == [active_set_init]
    assert observed is result


def test_primal_feasibility_cannot_replace_the_inner_qp_kkt_certificate(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite feasible inner iterate is not thereby a converged QP solution."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    calls: list[None] = []

    def feasible_but_uncertified_qp(*_args, **_kwargs):
        calls.append(None)
        return QPResult(beta=np.zeros(1), converged=False)

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", feasible_but_uncertified_qp)
    with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
        result, _ = irls_direct.fit_irls_direct(
            design,
            y,
            weights,
            Gaussian(),
            IdentityLink(),
            groups,
            lambda2=0.0,
            max_iter=3,
            tol=1e-10,
            convergence="coefficients",
        )

    # The outer coefficient criterion is already stationary at beta=0, so
    # current code stops after one solve and reports success unless the inner
    # certificate is authoritative.
    assert len(calls) == 3
    np.testing.assert_array_equal(result.beta, np.zeros(1))
    constraint = groups[0].constraints
    assert constraint is not None
    np.testing.assert_array_equal(constraint.A @ result.beta, constraint.b)
    assert not result.converged
    assert result.termination_reason == "constraint_kkt_incomplete"
    terminal_warnings = [
        record
        for record in caplog.records
        if "no complete constrained-QP KKT certificate" in record.getMessage()
    ]
    assert len(terminal_warnings) == 1
    assert terminal_warnings[0].levelno == logging.WARNING


def test_rejected_poisson_proposal_cannot_reuse_the_previous_working_kkt_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior certificate does not describe newly rebuilt Poisson weights."""
    import superglm.solvers.irls_direct as irls_direct
    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import LogLink
    from superglm.solvers.irls_state import _IRLSStepDecision
    from superglm.types import GroupSlice, LinearConstraintSet

    x = np.arange(-2.0, 3.0)[:, None]
    y = np.array([0.0, 0.0, 1.0, 3.0, 10.0])
    weights = np.ones(len(y))
    design = DesignMatrix([DenseGroupMatrix(x)], n=len(y), p=1)
    constraint_set = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))
    groups = [
        GroupSlice(
            "x",
            0,
            1,
            constraints=constraint_set,
            monotone_engine="qp",
        )
    ]
    real_solve_constrained_qp = irls_direct.solve_constrained_qp
    qp_systems: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
    decision_calls = [0]

    def record_real_qp(H, g, A, b, **kwargs):
        qp_result = real_solve_constrained_qp(H, g, A, b, **kwargs)
        qp_systems.append((H.copy(), g.copy(), qp_result.beta.copy(), qp_result.converged))
        return qp_result

    def accept_then_reject(**_kwargs):
        decision_calls[0] += 1
        accepted = decision_calls[0] == 1
        return _IRLSStepDecision(
            alpha=1.0 if accepted else 0.0,
            step_halvings=0 if accepted else 20,
            step_rejected=not accepted,
            trials_attempted=1 if accepted else 20,
        )

    monkeypatch.setattr(irls_direct, "solve_constrained_qp", record_real_qp)
    monkeypatch.setattr(irls_direct, "_select_irls_trial", accept_then_reject)
    result, _ = irls_direct.fit_irls_direct(
        design,
        y,
        weights,
        Poisson(),
        LogLink(),
        groups,
        lambda2=0.0,
        max_iter=3,
        tol=1e-10,
        convergence="coefficients",
    )

    assert len(qp_systems) == 2
    first_H, first_g, first_qp_beta, first_qp_converged = qp_systems[0]
    second_H, second_g, second_qp_beta, second_qp_converged = qp_systems[1]
    assert first_qp_converged
    assert second_qp_converged
    np.testing.assert_allclose(first_H, [[28.0]], rtol=1e-14)
    np.testing.assert_allclose(first_g, [23.0], rtol=1e-14)
    np.testing.assert_allclose(first_qp_beta, [23.0 / 28.0], rtol=1e-14)
    np.testing.assert_allclose(result.beta, first_qp_beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second_qp_beta, [1.01570625], rtol=1e-8)
    retained_slack = constraint_set.A @ result.beta - constraint_set.b
    assert retained_slack[0] > 0.0
    # The retained coefficient is strictly interior, so its constraint
    # multiplier must be zero.  This nonzero current gradient therefore rules
    # out stationarity for the second working problem.
    current_gradient = second_H @ result.beta - second_g
    np.testing.assert_allclose(current_gradient, [-4.82010409], rtol=1e-8)
    assert not result.converged
    assert result.termination_reason == "constraint_kkt_incomplete"


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


@pytest.mark.parametrize(
    ("failure_reason", "error_pattern"),
    [
        pytest.param(
            "constraint_infeasible",
            "terminal constrained REML refit ended at an infeasible coefficient mode",
            id="primal-infeasible",
        ),
        pytest.param(
            "constraint_kkt_incomplete",
            "terminal constrained REML refit ended without a complete inner-QP KKT certificate",
            id="kkt-incomplete",
        ),
    ],
)
def test_qp_reml_refuses_uncertified_terminal_state_before_objective_or_publication(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: str,
    error_pattern: str,
) -> None:
    """The constrained terminal state gates REML objective and public install."""
    from superglm.model import reml_finalize

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    terminal_objective_calls: list[None] = []

    def force_uncertified_terminal(
        work_model,
        *,
        qp_saved_state,
        pirls_result,
        **_kwargs,
    ):
        reml_finalize.restore_qp_group_state(work_model, qp_saved_state)
        terminal_beta = pirls_result.beta.copy()
        if failure_reason == "constraint_infeasible":
            group = next(group for group in work_model._groups if group.constraints is not None)
            terminal_beta[group.sl] = -10.0 * group.constraints.A[0]
            slack = group.constraints.A @ terminal_beta[group.sl] - group.constraints.b
            assert np.min(slack) < -1.0
        return replace(
            pirls_result,
            beta=terminal_beta,
            converged=False,
            termination_reason=failure_reason,
        )

    def unexpected_terminal_objective(*_args, **_kwargs):
        terminal_objective_calls.append(None)
        raise AssertionError("terminal REML objective ran before the constraint gate")

    monkeypatch.setattr(
        reml_finalize,
        "maybe_qp_passthrough_refit",
        force_uncertified_terminal,
    )
    monkeypatch.setattr(
        reml_finalize,
        "reml_laml_objective",
        unexpected_terminal_objective,
    )

    with pytest.raises(
        RuntimeError,
        match=error_pattern,
    ):
        model.fit_reml(frame, y, max_reml_iter=8, runtime_validation="skip")

    assert terminal_objective_calls == []
    assert model._result is None
    assert getattr(model, "_reml_result", None) is None


@pytest.mark.parametrize(
    ("failure_reason", "error_pattern"),
    [
        pytest.param(
            "constraint_infeasible",
            "fixed-lambda constrained REML fit ended at an infeasible coefficient mode",
            id="primal-infeasible",
        ),
        pytest.param(
            "constraint_kkt_incomplete",
            "fixed-lambda constrained REML fit ended without a complete inner-QP KKT certificate",
            id="kkt-incomplete",
        ),
    ],
)
def test_fixed_lambda_qp_reml_refuses_uncertified_result_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    failure_reason: str,
    error_pattern: str,
) -> None:
    """The fixed-lambda route gates an uncertified result before installing fit state."""
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

    def force_uncertified_fixed_result(*args, **kwargs):
        result, inverse = real_fit(*args, **kwargs)
        terminal_beta = result.beta.copy()
        if failure_reason == "constraint_infeasible":
            group = next(group for group in kwargs["groups"] if group.constraints is not None)
            terminal_beta[group.sl] = -10.0 * group.constraints.A[0]
            slack = group.constraints.A @ terminal_beta[group.sl] - group.constraints.b
            assert np.min(slack) < -1.0
        fixed_route_calls.append(kwargs["debug_context"])
        return (
            replace(
                result,
                beta=terminal_beta,
                converged=False,
                termination_reason=failure_reason,
            ),
            inverse,
        )

    monkeypatch.setattr(
        reml_execute,
        "fit_irls_direct",
        force_uncertified_fixed_result,
    )
    with pytest.raises(
        RuntimeError,
        match=error_pattern,
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
    # The full inner solve was marked converged, but the retained half-step is
    # not that QP solution and therefore does not inherit its KKT certificate.
    assert not result.converged
    assert result.termination_reason == "constraint_kkt_incomplete"


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
