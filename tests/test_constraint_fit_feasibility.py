"""Final-state feasibility regressions for hard fit-time constraints."""

import logging
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from superglm import (
    BSplineSmooth,
    Categorical,
    Constraint,
    OrderedCategorical,
    RandomEffect,
    Spline,
    SuperGLM,
    Tweedie,
)


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


@pytest.mark.parametrize(
    ("cold_converged", "safeguarded_converged"),
    [(False, False), (False, True), (True, False)],
)
def test_failed_warm_qp_retries_cold_without_weakening_certificate(
    monkeypatch: pytest.MonkeyPatch,
    cold_converged: bool,
    safeguarded_converged: bool,
) -> None:
    """A warm active set is only an optimization hint for the same QP.

    The degenerate-vertex attempt (issue #472) runs only after both the warm
    and the cold solve fail. It replaces their result only when it is
    certified.
    """
    import superglm.solvers.irls_direct._qp as irls_direct_qp
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
    safeguarded_result = QPResult(
        beta=np.array([0.75]),
        active_set=[],
        converged=safeguarded_converged,
    )
    calls: list[tuple[list[int] | None, bool]] = []

    def fake_solve(*_args, active_set_init=None, _degeneracy_safeguards=False, **_kwargs):
        calls.append((active_set_init, _degeneracy_safeguards))
        if _degeneracy_safeguards:
            return safeguarded_result
        return warm_result if active_set_init is not None else cold_result

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", fake_solve)
    result = irls_direct_qp._solve_constrained_qp_with_cold_retry(
        np.eye(1),
        np.ones(1),
        np.ones((1, 1)),
        np.zeros(1),
        [0],
    )

    if cold_converged:
        assert calls == [([0], False), (None, False)]
        assert result is cold_result
    else:
        assert calls == [([0], False), (None, False), (None, True)]
        assert result is (safeguarded_result if safeguarded_converged else warm_result)


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
    """Cold retry adds no work unless a nonempty warm active set fails.

    A failed cold start goes straight to the degenerate-vertex attempt,
    and a certified solve makes no further call.
    """
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.solvers.constrained_qp import QPResult

    result = QPResult(
        beta=np.array([0.25]),
        active_set=[] if active_set_init is None else active_set_init,
        converged=converged,
    )
    safeguarded_result = QPResult(beta=np.array([0.5]), converged=False)
    active_sets: list[list[int] | None] = []
    safeguarded_calls: list[list[int] | None] = []

    def fake_solve(*_args, active_set_init=None, _degeneracy_safeguards=False, **_kwargs):
        if _degeneracy_safeguards:
            safeguarded_calls.append(active_set_init)
            return safeguarded_result
        active_sets.append(active_set_init)
        return result

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", fake_solve)
    observed = irls_direct_qp._solve_constrained_qp_with_cold_retry(
        np.eye(1),
        np.ones(1),
        np.ones((1, 1)),
        np.zeros(1),
        active_set_init,
    )

    assert active_sets == [active_set_init]
    assert safeguarded_calls == ([] if converged else [None])
    assert observed is result


def test_primal_feasibility_cannot_replace_the_inner_qp_kkt_certificate(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite feasible inner iterate is not thereby a converged QP solution.

    The coefficient criterion passes on every iteration and the KKT veto holds
    each one open, so every reason recorded reads the vetoed verdict: none is
    ``converged``.  Mutation check: a reason chain that read the verdict from
    before the veto labelled the first two iterations converged.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    calls: list[None] = []

    def feasible_but_uncertified_qp(*_args, _degeneracy_safeguards=False, **_kwargs):
        if not _degeneracy_safeguards:
            calls.append(None)
        return QPResult(beta=np.zeros(1), converged=False)

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", feasible_but_uncertified_qp)
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
            record_diagnostics=True,
            weight_semantics="frequency",
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
    log = result.iteration_log
    assert log is not None
    assert all(row.convergence_value < row.convergence_tolerance for row in log)
    assert [row.termination_reason for row in log] == [
        "continue",
        "continue",
        "constraint_kkt_incomplete",
    ]


@pytest.mark.parametrize("separation", ["warn", "error"])
def test_a_strict_separation_stop_raises_before_the_constraint_refusal(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    separation: str,
) -> None:
    """The separation backstop runs before the QP contract's final refusal.

    The inner QP never certifies, so the fit spends its ten iterations at an
    unchanged deviance, and the working-weight ratio is forced past the
    separation bar: one fit carries the backstop's exhausted-budget signature
    and a state without a KKT certificate.  Under ``"warn"`` it reports both.
    Under ``"error"`` the backstop raises first, and the refusal of a state the
    fit never returns is not reported.  Mutation check: the final refusal run
    before the backstop logged that refusal and then raised.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.diagnostics.separation import SeparationError, SeparationWarning
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    monkeypatch.setattr(
        irls_direct_qp,
        "solve_constrained_qp",
        lambda *_args, **_kwargs: QPResult(beta=np.zeros(1), converged=False),
    )
    monkeypatch.setattr(irls_direct, "_separation_weight_ratio", lambda *_args, **_kwargs: 1e13)

    def fit():
        return irls_direct.fit_irls_direct(
            design,
            y,
            weights,
            Gaussian(),
            IdentityLink(),
            groups,
            lambda2=0.0,
            max_iter=10,
            tol=1e-10,
            convergence="coefficients",
            separation=separation,
            weight_semantics="frequency",
        )

    with caplog.at_level(logging.WARNING, logger="superglm.solvers.irls_direct"):
        if separation == "warn":
            with pytest.warns(SeparationWarning, match="exhausted its iteration budget"):
                result, _ = fit()
            assert result.termination_reason == "constraint_kkt_incomplete"
        else:
            with pytest.raises(SeparationError, match="exhausted its iteration budget"):
                fit()
    refusals = [
        record
        for record in caplog.records
        if "no complete constrained-QP KKT certificate" in record.getMessage()
    ]
    assert len(refusals) == (1 if separation == "warn" else 0)


def test_a_failed_degeneracy_retry_is_latched_off_for_the_rest_of_the_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QP that keeps failing pays for the #472 retry once per fit.

    The retry can cost a full active-set solve, and non-convergence usually
    persists for the rest of a fit. Mutation: without the latch in
    ``fit_irls_direct``, the failed retry runs again on every IRLS iteration,
    three times here instead of once.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    ordinary: list[None] = []
    safeguarded: list[None] = []

    def uncertified_qp(*_args, _degeneracy_safeguards=False, **_kwargs):
        (safeguarded if _degeneracy_safeguards else ordinary).append(None)
        return QPResult(beta=np.zeros(1), converged=False)

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", uncertified_qp)
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
        weight_semantics="frequency",
    )

    assert len(ordinary) == 3
    assert len(safeguarded) == 1
    assert result.termination_reason == "constraint_kkt_incomplete"


def test_a_certified_solve_rearms_the_degeneracy_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary solve that certifies re-arms the #472 retry for the rest of the fit (#477 review).

    The QP fails, certifies, then fails again.  The first failure latches the
    retry off; the certified solve between shows the failure was transient,
    so the third iterate's failure gets the retry again: two retries, not
    one.  Mutation: a latch that never re-arms runs it once.
    """
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult

    # y = +x: the optimum beta = 1 is interior, so the stand-in QP's steps
    # 0.3, 0.6, 0.9 each lower the deviance and are accepted.
    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    y = -y
    ordinary: list[None] = []
    safeguarded: list[None] = []

    def qp(*_args, _degeneracy_safeguards=False, **_kwargs):
        if _degeneracy_safeguards:
            safeguarded.append(None)
            return QPResult(beta=np.full(1, 0.3 * len(ordinary)), converged=False)
        ordinary.append(None)
        return QPResult(beta=np.full(1, 0.3 * len(ordinary)), converged=len(ordinary) == 2)

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", qp)
    irls_direct.fit_irls_direct(
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
        weight_semantics="frequency",
    )

    assert len(ordinary) == 3
    assert len(safeguarded) == 2


def test_rejected_poisson_proposal_cannot_reuse_the_previous_working_kkt_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior certificate does not describe newly rebuilt Poisson weights."""
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    import superglm.solvers.irls_direct._search as irls_direct_search
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
    real_solve_constrained_qp = irls_direct_qp.solve_constrained_qp
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

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", record_real_qp)
    monkeypatch.setattr(irls_direct_search, "_select_irls_trial", accept_then_reject)
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
        weight_semantics="frequency",
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
            weight_semantics="frequency",
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
        weight_semantics="frequency",
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
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._search as irls_direct_search
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

    monkeypatch.setattr(irls_direct_search, "_select_irls_trial", accept_tiny_line_search_step)
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
            weight_semantics="frequency",
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
    import superglm.solvers.irls_direct._fit as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    import superglm.solvers.irls_direct._search as irls_direct_search
    from superglm._fit_trace import MemoryTraceSink, TraceRun
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult
    from superglm.solvers.irls_state import _IRLSStepDecision
    from superglm.types import LinearConstraintSet

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    # beta >= 0.5 has no feasible zero point, so phase one cannot restart it.
    groups = [
        replace(groups[0], constraints=LinearConstraintSet(A=np.ones((1, 1)), b=np.array([0.5])))
    ]

    def infeasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([-2.0]), converged=False)

    def reject_every_trial(**_kwargs):
        return _IRLSStepDecision(
            alpha=0.0,
            step_halvings=20,
            step_rejected=True,
            trials_attempted=20,
        )

    monkeypatch.setattr(irls_direct_qp, "solve_constrained_qp", infeasible_qp_proposal)
    monkeypatch.setattr(irls_direct_search, "_select_irls_trial", reject_every_trial)
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
        weight_semantics="frequency",
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


def test_an_infeasible_retained_qp_state_is_never_kkt_certified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retained state outside ``A beta >= b`` carries the inner QP's KKT certificate.

    The stop rule's one constraint veto is that certificate's
    (``_IRLSDirectConvergence._terminate``): it stands for primal feasibility
    too, because the inner QP certifies a solve only where its own feasibility
    test passes at the contract's tolerance on the point it returns, and the
    line search hands the certificate only to the full proposal it solved
    (``_IRLSDirectQPStep.retained_converged``).  Every verdict below reads the
    retained state's feasibility and certificate: healthy fits, a warm start
    outside the set repaired by its first proposal, and tiny accepted steps
    from it that keep the retained state outside for every iteration.
    Mutation: handing the certificate to any accepted step, not only the full
    one, certifies those infeasible retained states.
    """
    import inspect

    import superglm.solvers.irls_direct._convergence as irls_convergence
    import superglm.solvers.irls_direct._search as irls_direct_search
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import solve_constrained_qp
    from superglm.solvers.irls_direct import fit_irls_direct
    from superglm.solvers.irls_direct._qp import _QP_FEASIBILITY_TOL
    from superglm.solvers.irls_state import _IRLSStepDecision

    # the contract tests feasibility at the inner solver's own tolerance
    assert _QP_FEASIBILITY_TOL == inspect.signature(solve_constrained_qp).parameters["tol"].default

    verdicts: list[tuple[bool, bool]] = []
    terminate = irls_convergence._IRLSDirectConvergence._terminate

    def recorded(self, fit, it):
        verdicts.append((fit.qp.feasible(fit.beta), fit.qp.retained_converged))
        terminate(self, fit, it)

    monkeypatch.setattr(irls_convergence._IRLSDirectConvergence, "_terminate", recorded)
    frame, y = _healthy_increasing_problem()
    _increasing_model().fit(frame, y)
    design, response, weights, groups = _one_coefficient_line_search_problem(constrained=True)
    outside = {
        "beta_init": np.array([-1.0]),
        "intercept_init": 0.0,
        "max_iter": 3,
        "weight_semantics": "frequency",
    }
    fit_irls_direct(design, response, weights, Gaussian(), IdentityLink(), groups, 0.0, **outside)

    def accept_tiny_step(**kwargs):
        alpha = 2.0**-19
        kwargs["evaluate_state"](alpha)
        return _IRLSStepDecision(
            alpha=alpha, step_halvings=19, step_rejected=False, trials_attempted=20
        )

    monkeypatch.setattr(irls_direct_search, "_select_irls_trial", accept_tiny_step)
    fit_irls_direct(design, response, weights, Gaussian(), IdentityLink(), groups, 0.0, **outside)

    assert any(certified for _, certified in verdicts)
    assert any(not feasible for feasible, _ in verdicts)
    assert not any(certified and not feasible for feasible, certified in verdicts)


def _force_terminal_qp_refit(monkeypatch: pytest.MonkeyPatch, failure_reason: str) -> None:
    """Make the QP passthrough's terminal refit end with *failure_reason*."""
    from superglm.model import reml_finalize

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

    monkeypatch.setattr(reml_finalize, "maybe_qp_passthrough_refit", force_uncertified_terminal)


def test_qp_reml_refuses_an_infeasible_terminal_state_before_objective_or_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal mode outside the constraints gates REML objective and public install."""
    from superglm.model import reml_finalize

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    terminal_objective_calls: list[None] = []

    def unexpected_terminal_objective(*_args, **_kwargs):
        terminal_objective_calls.append(None)
        raise AssertionError("terminal REML objective ran before the constraint gate")

    _force_terminal_qp_refit(monkeypatch, "constraint_infeasible")
    monkeypatch.setattr(reml_finalize, "reml_laml_objective", unexpected_terminal_objective)

    with pytest.raises(
        RuntimeError,
        match="terminal constrained REML refit ended at an infeasible coefficient mode",
    ):
        model.fit_reml(frame, y, max_reml_iter=8, runtime_validation="skip")

    assert terminal_objective_calls == []
    assert model._result is None
    assert getattr(model, "_reml_result", None) is None


def test_qp_reml_discloses_a_feasible_terminal_state_without_a_kkt_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A feasible terminal mode short of its inner-QP certificate is published, not refused.

    It keeps every constraint and is only short of certified optimality, so
    it is published as not converged and the warning names that stage (owner
    decision 3).  Mutation: the refusal it replaces raised
    ``ObservedModeNotConvergedError`` and published nothing.
    """
    from superglm.diagnostics.convergence import ConvergenceWarning

    frame, y = _healthy_increasing_problem()
    model = _increasing_model()
    _force_terminal_qp_refit(monkeypatch, "constraint_kkt_incomplete")

    with pytest.warns(ConvergenceWarning, match="could not certify that its coefficients"):
        model.fit_reml(frame, y, max_reml_iter=8, runtime_validation="skip")

    assert model._reml_result is not None and not model._reml_result.converged
    assert model._reml_result.terminal_refit_termination == "constraint_kkt_incomplete"
    assert np.all(np.isfinite(model.predict(frame)))


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
        weight_semantics="frequency",
    )

    # beta=-1 is an exact but infeasible fit with deviance zero. The
    # constrained QP proposal beta=0 is feasible but has worse merit.
    np.testing.assert_allclose(result.beta, [0.0], atol=1e-12)
    assert result.deviance > 0.0
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == 1.0
    assert not result.iteration_log[0].step_rejected
    assert result.termination_reason != "constraint_infeasible"


def test_infeasible_start_without_a_feasible_qp_proposal_restarts_at_the_zero_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QP at an infeasible start that returns no feasible proposal leads to phase one.

    Shape rows are homogeneous, so the violated block's zero is feasible; the
    fit continues from it instead of ending outside the constraints.
    """
    import superglm.solvers.irls_direct as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.model.base import model_build_design_matrix
    from superglm.solvers.constrained_qp import QPResult

    x = np.linspace(0.0, 1.0, 80)
    y = 0.5 + 2.0 * x + 0.02 * np.sin(8.0 * x)
    model = _increasing_model()
    y_fit, weights, offset = model_build_design_matrix(
        model, pd.DataFrame({"x": x}), y, np.ones_like(y), None
    )
    group = model._groups[0]
    assert group.constraints is not None
    infeasible_beta = np.zeros(model._dm.p)
    infeasible_beta[group.sl] = -10.0 * group.constraints.A[0]
    real_qp = irls_direct_qp._solve_constrained_qp_with_cold_retry
    calls: list[int] = []

    def no_feasible_first_proposal(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return QPResult(beta=infeasible_beta.copy(), converged=False)
        return real_qp(*args, **kwargs)

    monkeypatch.setattr(
        irls_direct_qp, "_solve_constrained_qp_with_cold_retry", no_feasible_first_proposal
    )

    def fit(max_iter: int):
        calls.clear()
        return irls_direct.fit_irls_direct(
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
            max_iter=max_iter,
            tol=1e-8,
            weight_semantics="frequency",
        )[0]

    first = fit(max_iter=1)
    assert first.termination_reason == "constraint_kkt_incomplete"
    np.testing.assert_array_equal(first.beta[group.sl], 0.0)

    result = fit(max_iter=50)
    assert result.converged
    slack = group.constraints.A @ result.beta[group.sl] - group.constraints.b
    assert np.min(slack) >= -1e-10 * np.max(
        np.abs(group.constraints.A) @ np.abs(result.beta[group.sl])
    )


def test_feasible_committed_state_never_accepts_infeasible_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A merit-improving infeasible proposal is truncated at the feasible boundary."""
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult
    from superglm.solvers.irls_direct import _fit as irls_direct

    design, y, weights, groups = _one_coefficient_line_search_problem(constrained=True)

    def infeasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([-1.0]), converged=True)

    monkeypatch.setattr(
        irls_direct_qp,
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
        weight_semantics="frequency",
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
        weight_semantics="frequency",
    )

    np.testing.assert_allclose(result.beta, [-1.0], atol=1e-12)
    assert result.deviance == pytest.approx(0.0, abs=1e-20)
    assert result.iteration_log is not None
    assert result.iteration_log[0].accepted_alpha == 1.0
    assert result.iteration_log[0].step_halvings == 0


def _issue_472_capped_reml_case() -> tuple[SuperGLM, pd.DataFrame, np.ndarray, np.ndarray]:
    """The #471 sweep case ``flat:1.2:600:0.5:20000:1:0-4:1e-6``, seed 2 (issue #472).

    Tweedie data whose only ordered-factor effect is a small trend in ``a``,
    with a dip at ``a6``. ``b`` and ``c`` have none, so their monotone fits
    sit flat at zero. Each is a QP-monotone ordered-factor cr spline, beside
    a 600-level Zipf random effect.
    """
    rng = np.random.default_rng(2)
    n, levels = 20_000, 600
    exposure = rng.uniform(0.2, 1.0, n)
    a = rng.integers(0, 12, n)
    b = rng.integers(0, 8, n)
    c = rng.integers(0, 6, n)
    state = rng.integers(0, 10, n)
    make = np.minimum(rng.zipf(1.2, n) - 1, levels - 1)
    effect = rng.normal(0.0, 0.5, levels)
    eta = -2.0 + 0.03 * a - 0.2 * (a == 6) + 0.1 * np.sin(state) + effect[make]
    mu = np.exp(eta) * exposure
    power, phi = 1.6, 2.0
    counts = rng.poisson(mu ** (2 - power) / (phi * (2 - power)))
    shape = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu ** (power - 1)
    y = np.array([rng.gamma(shape, s, k).sum() for k, s in zip(counts, scale, strict=True)])
    frame = pd.DataFrame(
        {
            "a": np.char.add("a", a.astype(str)),
            "b": np.char.add("b", b.astype(str)),
            "c": np.char.add("c", c.astype(str)),
            "state": np.char.add("s", state.astype(str)),
            "make": np.char.add("m", make.astype(str)),
        }
    )
    increasing = Constraint.fit.increasing
    model = SuperGLM(
        family=Tweedie(p=1.6),
        selection_penalty=0.0,
        features={
            "a": OrderedCategorical(
                values={f"a{i}": float(i) for i in range(12)},
                basis=Spline(kind="cr", k=6, constraint=increasing),
            ),
            "b": OrderedCategorical(
                values={f"b{i}": float(i) for i in range(8)},
                basis=Spline(kind="cr", k=5, constraint=increasing),
            ),
            "c": OrderedCategorical(
                values={f"c{i}": float(i) for i in range(6)},
                basis=Spline(kind="cr", k=4, constraint=Constraint.fit.decreasing),
            ),
            "state": Categorical(base="most_exposed"),
            "make": RandomEffect(),
        },
    )
    return model, frame, y / exposure, exposure


@pytest.mark.slow
def test_capped_reml_with_a_flat_monotone_term_at_zero_returns_a_certified_mode() -> None:
    """A flat monotone term at a degenerate vertex no longer refuses the fit (issue #472).

    Its terminal QP has ``b = 0``, a well-conditioned ``H``, and one term
    flat at zero with more active rows than coefficients. Mutation: 0.37.1
    raised ``ObservedModeNotConvergedError`` ("terminal constrained REML
    refit ended at an infeasible coefficient mode"). Without the per-block
    zero face (``_zero_block_faces``) it still does.
    """
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.solvers.constrained_qp import _is_feasible

    model, frame, y, exposure = _issue_472_capped_reml_case()
    with pytest.warns(ConvergenceWarning, match="smoothing-parameter selection stopped"):
        model.fit_reml(frame, y, sample_weight=exposure, max_reml_iter=1, lambda2_init=1e-6)

    assert model._reml_result.terminal_refit_termination is None
    assert model.result.termination_reason == "converged"
    constrained = [group for group in model._groups if group.constraints is not None]
    assert len(constrained) == 3
    for group in constrained:
        A, b = group.constraints.A, group.constraints.b
        assert _is_feasible(A, model.result.beta[group.sl], b, 1e-12)


def _restricted_link_book(link: str, seed: int, n: int = 400) -> tuple[pd.DataFrame, np.ndarray]:
    """A log-scale trend increasing in ``x1``, a wave in ``x2`` and a six-level factor.

    Under the inverse link an increasing mean is a decreasing linear
    predictor, so an ``increasing`` term in ``x1`` is held flat, at a
    smoothing parameter near 1e7. Under the identity link, Poisson rows
    without events pull an unconstrained fit's means through zero.
    """
    rng = np.random.default_rng(seed)
    x1, x2 = rng.uniform(0.0, 10.0, n), rng.uniform(-2.0, 2.0, n)
    levels = np.array(list("abcdef"))
    c1 = rng.choice(levels, n)
    effect = dict(zip(levels, [0.0, 0.2, -0.15, 0.3, -0.1, 0.25], strict=True))
    eta = 0.06 * x1 + 0.3 * np.sin(1.3 * x2) + pd.Series(c1).map(effect).to_numpy()
    eta -= eta.mean()
    if link == "inverse":
        y = rng.gamma(3.0, np.exp(0.5 * eta + 1.0) / 3.0)
    else:
        y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x1": x1, "x2": x2, "c1": c1}), y


def _restricted_link_model(family: str, link: str, kind: str) -> SuperGLM:
    from superglm import PSpline

    increasing = Constraint.fit.increasing
    term = (
        Spline(kind="bs", k=8, constraint=increasing)
        if kind == "bs"
        else PSpline(n_knots=8, constraint=increasing)
    )
    return SuperGLM(
        family=family,
        link=link,
        features={"x1": term, "x2": Spline(kind="cr", k=8), "c1": Categorical()},
    )


def _assert_shape_certified(model: SuperGLM, name: str) -> None:
    from superglm.constraints import shape_constraint_is_roundoff_feasible

    spec = model._specs[name]
    group = next(group for group in model._groups if group.feature_name == name)
    assert group.monotone_engine == "qp"
    beta = model.result.beta[group.start : group.end]
    assert shape_constraint_is_roundoff_feasible(spec, beta, spec.constraint_kind)


@pytest.mark.parametrize("kind", ["bs", "ps"])
def test_an_increasing_term_held_flat_by_an_inverse_link_returns_a_certified_mode(
    kind: str,
) -> None:
    """Gamma/inverse: the passthrough's terminal QP at a flat term certifies.

    The term sits at its zero face, or on a ray of it near 1e-6, in
    coordinates scaled by its smoothing parameter; its QP certifies only on
    the retry, whose multipliers are fitted in the certificate's units.
    Mutation: fitted unweighted, as before, the REML fit ends unconverged.
    """
    frame, y = _restricted_link_book("inverse", seed=0)
    model = _restricted_link_model("gamma", "inverse", kind)
    model.fit_reml(frame, y)

    assert model.reml_diagnostics()["converged"]
    assert model._reml_result.terminal_refit_termination is None
    _assert_shape_certified(model, "x1")


@pytest.mark.parametrize("kind", ["bs", "ps"])
@pytest.mark.parametrize(("seed", "certain"), [(1, True), (7, False)])
def test_an_identity_link_poisson_shape_fit_keeps_every_mean_positive(
    kind: str, seed: int, certain: bool
) -> None:
    """Poisson/identity: every state holds ``mu = eta > 0``, so the QP's weights exist.

    ``clip_mu`` floored a crossed row at ``mu = 1e-50``, whose Fisher weight
    ``1 / mu`` is ``1e50``; the terminal QP's ``H`` then read as rank 6 of 19
    with its gradient in the null space. Mutation: without the identity
    link's domain both seeds raise that ``ValueError``. A search that stops
    short is disclosed, never refused. Every published mean is positive up
    to the rounding of forming ``eta``.
    """
    from superglm._roundoff import gamma
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.solvers.mode_score import linear_predictor

    frame, y = _restricted_link_book("identity", seed=seed)
    model = _restricted_link_model("poisson", "identity", kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(frame, y)

    converged = model.reml_diagnostics()["converged"]
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    assert converged if certain else (converged or warned)
    result = model.result
    reach = abs(result.intercept) + np.abs(model._dm.toarray()) @ np.abs(result.beta)
    if result.centred_intercept is not None:
        reach += abs(result.centred_intercept) + np.abs(result.state_center) @ np.abs(result.beta)
    eta = linear_predictor(model._dm, result, None)
    assert np.all(eta >= -gamma(4) * reach), float(np.min(eta))
    _assert_shape_certified(model, "x1")


def test_a_feasible_proposal_outside_the_link_domain_has_its_intercept_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """From an infeasible start, a feasible QP proposal at ``mu <= 0`` is raised into the domain.

    The start ``beta = -1`` breaks ``beta >= 0``; the QP returns ``beta = 10``,
    whose rows near ``x = -1`` have ``eta < 0``, and every trial is rejected.
    The proposal is accepted for its feasibility, with its intercept raised,
    as a start's is, until the smallest ``eta`` is ``g(mean y)``; the slope
    is the QP's. Mutation: accepted as it stands, ``min eta`` is about -9.4.
    """
    import superglm.solvers.irls_direct as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    import superglm.solvers.irls_direct._search as irls_direct_search
    from superglm.distributions import Poisson
    from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
    from superglm.links import IdentityLink
    from superglm.solvers.constrained_qp import QPResult
    from superglm.solvers.irls_state import _IRLSStepDecision
    from superglm.solvers.mode_score import linear_predictor
    from superglm.types import GroupSlice, LinearConstraintSet

    x = np.linspace(-1.0, 1.0, 41)
    y = np.random.default_rng(0).poisson(1.5, x.size).astype(float)
    design = DesignMatrix([DenseGroupMatrix(x[:, None])], n=x.size, p=1)
    constraint = LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1))
    groups = [GroupSlice("x", 0, 1, constraints=constraint, monotone_engine="qp")]

    def feasible_qp_proposal(*_args, **_kwargs):
        return QPResult(beta=np.array([10.0]), converged=True)

    def reject_every_trial(**_kwargs):
        return _IRLSStepDecision(
            alpha=0.0, step_halvings=20, step_rejected=True, trials_attempted=20
        )

    monkeypatch.setattr(
        irls_direct_qp, "_solve_constrained_qp_with_cold_retry", feasible_qp_proposal
    )
    monkeypatch.setattr(irls_direct_search, "_select_irls_trial", reject_every_trial)
    result, _ = irls_direct.fit_irls_direct(
        design,
        y,
        np.ones(x.size),
        Poisson(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        beta_init=np.array([-1.0]),
        intercept_init=3.0,
        max_iter=1,
        weight_semantics="frequency",
    )

    assert result.beta[0] == 10.0
    eta = linear_predictor(design, result, None)
    reach = abs(result.intercept) + np.abs(x) * abs(result.beta[0])
    assert abs(float(np.min(eta)) - y.mean()) <= 4 * np.finfo(float).eps * float(np.max(reach))
    assert result.termination_reason == "constraint_kkt_incomplete"


@pytest.mark.parametrize(("n", "n_knots"), [(100, 6), (200, 6), (200, 8)])
def test_an_identity_link_poisson_fit_at_the_zero_edge_is_returned_unconverged(
    n: int, n_knots: int
) -> None:
    """Two events in the last tenth: the increasing mean of most rows sits at zero.

    At that edge the Fisher weights ``1 / mu`` can leave the next working QP
    unbounded, and it raised ``H is rank-deficient``. The fit returns its
    last accepted coefficients, finite, shape-feasible and not converged.
    Mutation: without the catch all three raise on the Haswell and Sandy
    Bridge kernels, and the last two on SkylakeX.
    """
    from superglm import PSpline
    from superglm.diagnostics.convergence import ConvergenceWarning

    x = np.linspace(0.0, 1.0, n)
    y = np.zeros(n)
    y[[90 * n // 100, 96 * n // 100]] = 1.0
    model = SuperGLM(
        family="poisson",
        link="identity",
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=n_knots, constraint=Constraint.fit.increasing)},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(pd.DataFrame({"x": x}), y)

    assert not model.result.converged
    assert np.all(np.isfinite(model.predict(pd.DataFrame({"x": x}))))
    _assert_shape_certified(model, "x")
    for item in caught:
        if issubclass(item.category, ConvergenceWarning):
            assert "log link" in str(item.message)


@pytest.mark.parametrize("link", ["identity", "log"])
def test_an_unbounded_working_qp_is_disclosed_only_under_a_zero_bounded_link(
    monkeypatch: pytest.MonkeyPatch, link: str
) -> None:
    """A ``ValueError`` from the working QP ends an identity-link fit at its committed state.

    The fit returns unconverged, with a ``ConvergenceWarning`` naming the
    edge and the log link and carrying the QP's own message; under the log link, whose domain has no edge,
    the error is raised. Mutation: catching it under every link fails the
    log case, and dropping the warning fails the identity case.
    """
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm import PSpline
    from superglm.diagnostics.convergence import ConvergenceWarning

    real_qp = irls_direct_qp._solve_constrained_qp_with_cold_retry
    calls: list[int] = []

    def unbounded_second_solve(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("solve_constrained_qp: H is rank-deficient")
        return real_qp(*args, **kwargs)

    monkeypatch.setattr(
        irls_direct_qp, "_solve_constrained_qp_with_cold_retry", unbounded_second_solve
    )
    x = np.linspace(0.0, 1.0, 80)
    y = np.random.default_rng(0).poisson(1.0 + 2.0 * x).astype(float)
    model = SuperGLM(
        family="poisson",
        link=link,
        selection_penalty=0.0,
        features={"x": PSpline(n_knots=6, constraint=Constraint.fit.increasing)},
    )
    if link == "log":
        with pytest.raises(ValueError, match="rank-deficient"):
            model.fit(pd.DataFrame({"x": x}), y)
        return
    with pytest.warns(ConvergenceWarning, match="log link.*H is rank-deficient"):
        model.fit(pd.DataFrame({"x": x}), y)

    assert len(calls) == 2
    assert not model.result.converged
    assert np.all(np.isfinite(model.result.beta))
    _assert_shape_certified(model, "x")


def test_an_unbounded_working_qp_at_an_infeasible_start_restarts_at_the_zero_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``ValueError`` from the QP at an infeasible start leads to phase one, not a stop.

    Poisson/identity, with every working QP raising: the fit commits the
    violated block's feasible zero, then stops there unconverged with the
    zero-edge warning. Mutation: stopping at the infeasible start ends the
    fit ``constraint_infeasible``, which the REML layers refuse.
    """
    import superglm.solvers.irls_direct as irls_direct
    import superglm.solvers.irls_direct._qp as irls_direct_qp
    from superglm.diagnostics.convergence import ConvergenceWarning
    from superglm.model.base import model_build_design_matrix

    def unbounded_qp(*args, **kwargs):
        raise ValueError("solve_constrained_qp: H is rank-deficient")

    monkeypatch.setattr(irls_direct_qp, "_solve_constrained_qp_with_cold_retry", unbounded_qp)
    x = np.linspace(0.0, 1.0, 80)
    y = np.random.default_rng(0).poisson(1.0 + 2.0 * x).astype(float)
    model = SuperGLM(
        family="poisson",
        link="identity",
        selection_penalty=0.0,
        features={"x": BSplineSmooth(n_knots=8, constraint=Constraint.fit.increasing)},
    )
    y_fit, weights, offset = model_build_design_matrix(
        model, pd.DataFrame({"x": x}), y, np.ones_like(y), None
    )
    group = model._groups[0]
    assert group.constraints is not None
    infeasible_beta = np.zeros(model._dm.p)
    infeasible_beta[group.sl] = -0.1 * group.constraints.A[0]
    with pytest.warns(ConvergenceWarning, match="log link.*H is rank-deficient"):
        result = irls_direct.fit_irls_direct(
            model._dm,
            y_fit,
            weights,
            model._distribution,
            model._link,
            model._groups,
            lambda2={"x": 1.0},
            offset=offset,
            beta_init=infeasible_beta,
            intercept_init=float(np.mean(y)),
            max_iter=50,
            tol=1e-8,
            weight_semantics="frequency",
        )[0]

    assert not result.converged
    assert result.termination_reason != "constraint_infeasible"
    np.testing.assert_array_equal(result.beta[group.sl], 0.0)
    assert np.isfinite(result.intercept)
    eta = result.intercept + model._dm.matvec(result.beta)
    assert np.all(eta > 0.0)
