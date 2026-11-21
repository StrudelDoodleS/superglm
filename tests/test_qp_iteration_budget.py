"""Limits on one active-set attempt and on a sequence of scalar IRLS proposals."""

import numpy as np
import pytest

from superglm.solvers.constrained_qp import QPResult, solve_constrained_qp


def test_default_budget_can_identify_201_binding_rows():
    # Every multiplier must equal one at the unique nonnegative minimizer.
    p = 201
    result = solve_constrained_qp(
        np.eye(p), -np.ones(p), np.eye(p), np.zeros(p), beta_start=np.zeros(p)
    )
    assert result.converged
    np.testing.assert_array_equal(result.beta, np.zeros(p))
    assert sorted(result.active_set) == list(range(p))


@pytest.mark.parametrize("both_end_early", [False, True])
def test_budget_hits_count_exhausted_loops_only(monkeypatch, both_end_early):
    from superglm.solvers import constrained_qp as qp

    run = qp._solve_qp

    def replay(*args, conservative, departed):
        # Force a regular/conservative replay. Both spend their final allowed
        # iteration; a zero direction with a failed certificate exits early.
        if not conservative:
            departed.append(True)
        if conservative or both_end_early:
            with monkeypatch.context() as patch:
                patch.setattr(
                    qp, "_stationarity_certificate", lambda *a, **k: (False, None, np.zeros(2))
                )
                return run(
                    *args[:4], [0, 1], *args[5:], conservative=conservative, departed=departed
                )
        return run(*args, conservative=conservative, departed=departed)

    monkeypatch.setattr(qp, "_solve_qp", replay)
    result = solve_constrained_qp(np.eye(2), -np.ones(2), np.eye(2), np.zeros(2), max_iter=1)
    assert not result.converged
    assert result.n_iter == 2
    assert result._budget_hits == (0 if both_end_early else 1)


def _partial_gaussian_fit(monkeypatch, *, persistent=False, max_iter=6):
    from superglm.distributions import Gaussian
    from superglm.links import IdentityLink
    from superglm.solvers.irls_direct import _qp, _search, fit_irls_direct
    from superglm.solvers.irls_state import _IRLSStepDecision
    from superglm.types import GroupSlice, LinearConstraintSet

    # Orthogonal centred columns: independently project each least-squares
    # coefficient onto [0, infinity). Duplicate rows make the size cap >200.
    x = np.vstack([np.eye(3), -np.eye(3)])
    y = 4.0 + x @ [-1.0, -2.0, 3.0]
    rows = np.tile(np.eye(3), (20, 1))
    constraint = LinearConstraintSet(A=rows, b=np.zeros(len(rows)))
    groups = [GroupSlice("x", 0, 3, constraints=constraint, monotone_engine="qp")]
    solve, select = _qp.solve_constrained_qp, _search._select_irls_trial
    calls, trials = [], []

    def force_partial(H, g, A, b, *, max_iter, active_set_init, beta_start, **kwargs):
        calls.append((max_iter, active_set_init, beta_start))
        if len(calls) == 2:
            # The warm result ends early. Its cold retry alone exhausts;
            # the wrapper retains this candidate, which the search rejects.
            assert active_set_init
            return QPResult(beta=np.zeros(3), active_set=[2], n_iter=1, converged=False)
        if len(calls) == 1:
            beta_start = np.array([1.0, 3.0, 1.0])
        budget = 1 if max_iter == 200 or persistent else max_iter
        return solve(
            H,
            g,
            A,
            b,
            max_iter=budget,
            active_set_init=active_set_init,
            beta_start=beta_start,
            **kwargs,
        )

    def choose_trial(**kwargs):
        trials.append(kwargs["proposal"].beta.copy())
        if len(trials) == 2 or (persistent and len(trials) > 2) or max_iter == 1:
            return _IRLSStepDecision(alpha=0.0, step_halvings=1, step_rejected=True)
        return select(**kwargs)

    monkeypatch.setattr(_qp, "solve_constrained_qp", force_partial)
    monkeypatch.setattr(_search, "_select_irls_trial", choose_trial)
    result, _ = fit_irls_direct(
        x,
        y,
        np.ones(len(x)),
        Gaussian(),
        IdentityLink(),
        groups,
        lambda2=0.0,
        max_iter=max_iter,
        record_diagnostics=True,
        weight_semantics="frequency",
    )
    return x, result, calls, trials


@pytest.mark.parametrize("persistent", [False, True])
def test_rejected_budget_limited_step_restarts_at_committed_point(monkeypatch, persistent):
    x, result, calls, trials = _partial_gaussian_fit(monkeypatch, persistent=persistent)
    assert [budget for budget, _, _ in calls[:4]] == [200, 200, 200, 252]
    assert calls[3][1] is None
    np.testing.assert_array_equal(calls[3][2], trials[0])
    assert np.all(trials[0] >= 0) and not np.array_equal(trials[0], trials[1])
    assert result.iteration_log[1].step_rejected
    assert result.iteration_log[1].termination_reason == "continue"
    if persistent:
        assert len(calls) == 4
        assert not result.converged
        assert result.termination_reason == "constraint_kkt_incomplete"
        np.testing.assert_array_equal(result.beta, trials[0])
    else:
        assert result.converged and len(calls) == 5
        assert calls[-1][0] == 252
        np.testing.assert_array_equal(calls[-1][2], trials[2])
        np.testing.assert_allclose(
            result.intercept + x @ result.beta, 4.0 + 3.0 * x[:, 2], atol=1e-12, rtol=0
        )


def test_unrepaired_infeasible_start_with_no_blocker_uses_remaining_budget(monkeypatch):
    from superglm.solvers import constrained_qp as qp

    # The first bound binds. The face step moves y from -4 to -3, so the
    # second row is violated along the entire segment and cannot block it.
    monkeypatch.setattr(qp, "_project_feasible", lambda *a, **kw: np.array([-1.0, -4.0]))
    result = solve_constrained_qp(
        np.eye(2), -np.array([2.0, 3.0]), np.eye(2), -np.ones(2), [0], max_iter=12
    )
    assert result.converged
    np.testing.assert_array_equal(result.beta, [-1.0, -1.0])


def test_coordinate_recovery_reconstructs_small_working_equations(monkeypatch):
    from superglm.solvers import constrained_qp as qp

    # Force the normwise QR error seen on the mixed-block fit. An order-one
    # bound contaminates the reconstruction of an independent small bound.
    solve = qp._solve_triangular

    def rounded(triangle, rhs, **kwargs):
        x = solve(triangle, rhs, **kwargs)
        if rhs.ndim == 1 and np.max(np.abs(rhs)) >= 1:
            x[0] += np.finfo(float).eps
        return x

    monkeypatch.setattr(qp, "_solve_triangular", rounded)
    bounds = np.array([2.0**-40, 1.0])
    x = qp._coordinate_face_minimum(np.eye(3), np.ones(3), np.eye(2, 3), bounds)
    np.testing.assert_allclose(x, [*bounds, 1.0], rtol=1e-12, atol=0)


@pytest.mark.pins_behaviour
def test_last_outer_iteration_cannot_restart_a_rejected_step(monkeypatch):
    """Pin the caller's outer iteration limit even when the QP exhausts."""
    _, result, calls, _ = _partial_gaussian_fit(monkeypatch, max_iter=1)
    assert len(calls) == result.n_iter == 1
    assert not result.converged and result.termination_reason == "constraint_kkt_incomplete"
