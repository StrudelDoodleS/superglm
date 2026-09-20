"""The discrete REML tensor step is a descent direction, and a dead search is named.

The shared-tensor branch of ``optimize_discrete_reml_cached_w`` used to
post-process its modified-Newton step by clipping every coordinate to
``base_cap`` and then re-solving each shared tensor pair in sum/difference
coordinates ``(u, v)``, clipping ``u`` and ``v`` INDEPENDENTLY. Clipping one
coordinate of a direction harder than its partner rotates the direction and
can flip a coordinate's sign; a line search needs ``g . d < 0`` (Nocedal and
Wright, *Numerical Optimization*, 2nd ed., Springer 2006, ch. 3), and once
that is gone the quadratic surrogate rejects every step length, no true
objective is evaluated, ``rho`` never moves and the loop runs to
``max_reml_iter`` publishing a fit above a reachable point.

The repair imposes the same trust region by DAMPING the step
(Levenberg-Marquardt: ``delta(mu) = -(H_pd + mu I)^-1 g`` is a descent
direction for every ``mu >= 0``; Nocedal and Wright ch. 4; More and
Sorensen, SIAM J. Sci. Stat. Comput. 4 (1983) 553-572), lifts the surrogate
backtrack floor, and lets a dead tensor line search name its exit through
the exact engine's ``classify_dead_feasible_exit``.

Every pinned number below was measured on the UNFIXED engine (v0.34.0) with
the fixture builders in this file; the descent-direction tests fail there
for the reasons their docstrings give.
"""

import numpy as np
import pandas as pd
import pytest

from superglm import Spline, SuperGLM
from superglm.reml import discrete as discrete_reml

# ── Pinned measurements from the UNFIXED engine ──────────────────────────
#
# Stall fixture: max_reml_iter=12 and 40 both stop at max_reml_iter with
# exactly this objective, i.e. rho is frozen from the first dead search
# (iteration 3) onward; only two true objectives are evaluated in 40 outer
# iterations.
UNFIXED_STALL_OBJECTIVE = 1011.706149461513
# Same fixture with the per-pair (u, v) clip removed (the workflow-1
# counterfactual): converges in 15 iterations at this objective, so the
# published stall sits at least 13.19 REML units above a reachable point.
# Provenance for STALL_OBJECTIVE_MARGIN below; no assertion reads this value.
UNFIXED_STALL_REACHABLE_OBJECTIVE = 998.5159505701776
# Three quarters of that measured gap is the margin the repair must clear.
STALL_OBJECTIVE_MARGIN = 10.0
# Additive discrete Poisson: never enters the tensor branch.
UNFIXED_ADDITIVE_OBJECTIVE = 860.770270858582
UNFIXED_ADDITIVE_LAMBDAS = {"x": 0.45392953968926786, "z": 4110319.212586605}
# The same additive fit stopped at max_reml_iter=3.
UNFIXED_ADDITIVE_MAXITER3_OBJECTIVE = 860.8097629812323
UNFIXED_ADDITIVE_MAXITER3_LAMBDAS = {"x": 0.4539935346095709, "z": 3648.391372339347}
# Gamma with a tensor interaction: unknown scale gates the surrogate off.
UNFIXED_GAMMA_OBJECTIVE = 1935.8168444276907
UNFIXED_GAMMA_LAMBDAS = {
    "x2": 0.3611971859690182,
    "x3": 653156.0860129567,
    "x2:x3:margin_x2": 0.11123849119865857,
    "x2:x3:margin_x3": 131574.57853927236,
}
# A one-pair fit that already converges on the surrogate path, chosen so
# that every lambda is DETERMINED (all four between 2.8 and 7.1, none
# parked against a bound where a relative comparison means nothing).
UNFIXED_MILD_OBJECTIVE = 3285.951078170296
UNFIXED_MILD_LAMBDAS = {
    "x2": 3.7617589041505957,
    "x3": 2.794799975612845,
    "x2:x3:margin_x2": 7.059626672120411,
    "x2:x3:margin_x3": 6.928358650537662,
}


# ── Fixtures ─────────────────────────────────────────────────────────────


def _stall_frame():
    """Two P-spline margins whose pair solve wants ``|v|`` far beyond ``cap_v``.

    A wiggly ``x2`` margin multiplied by a nearly linear ``x3`` margin: the
    two tensor margins want log-lambdas five units apart, so the pair solve
    asks for ``v = -5.2`` against ``cap_v = 1.0``. Measured on the unfixed
    engine (reconstruction agreeing with the engine's own ``tensor_uv``
    record to every digit): steps accepted at iterations 1-2, then from
    iteration 3 ``g . d`` is ``-24.28`` after the base_cap clip and
    ``+7.67`` after the per-pair clip, all five surrogate halvings reject,
    no true objective is evaluated, and the loop runs to the budget.
    """
    rng = np.random.default_rng(7)
    n = 2000
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    eta = -0.3 + 1.2 * np.sin(3.1 * x2) + 0.2 * x3 + 1.5 * np.sin(3.0 * x2) * x3
    y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x2": x2, "x3": x3}), y


def _mild_frame():
    """A one-pair tensor fit the unfixed engine already converges."""
    rng = np.random.default_rng(303)
    n = 6000
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    eta = (
        0.6
        + 0.8 * np.sin(2.2 * x2)
        + 0.8 * np.sin(2.2 * x3)
        + 0.7 * np.sin(1.6 * x2) * np.sin(1.6 * x3)
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    return pd.DataFrame({"x2": x2, "x3": x3}), y


def _additive_frame():
    rng = np.random.default_rng(404)
    n = 1500
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.2 + np.sin(2.0 * np.pi * x) + 0.6 * z)).astype(float)
    return pd.DataFrame({"x": x, "z": z}), y


def _gamma_frame():
    rng = np.random.default_rng(909)
    n = 2000
    x2 = rng.uniform(-1.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.5 + 0.8 * np.sin(2.4 * x2) + 0.3 * x3 + 0.6 * np.sin(2.0 * x2) * x3)
    y = rng.gamma(shape=6.0, scale=mu / 6.0)
    return pd.DataFrame({"x2": x2, "x3": x3}), y


def _tensor_model(family="poisson"):
    return SuperGLM(
        family=family,
        discrete=True,
        n_bins=48,
        features={"x2": Spline(kind="ps", k=8), "x3": Spline(kind="ps", k=8)},
        interactions=[("x2", "x3")],
    )


def _additive_model():
    return SuperGLM(
        family="poisson",
        discrete=True,
        n_bins=48,
        features={"x": Spline(kind="ps", k=8), "z": Spline(kind="ps", k=8)},
    )


def _reject_every_move():
    """A ``reml_laml_objective`` stand-in that rejects every lambda move."""
    evaluated: dict[str, float] = {}

    def reject(*args, **kwargs):
        candidate = args[6]
        if not evaluated:
            evaluated.update(candidate)
            return 0.0
        unchanged = all(
            candidate[name] == pytest.approx(value) for name, value in evaluated.items()
        )
        return 0.0 if unchanged else 1.0

    return reject, evaluated


# ── The repair ───────────────────────────────────────────────────────────


class TestDiscreteTensorStepIsADescentDirection:
    @pytest.mark.parametrize("base_cap,cap_v", [(1.0, 0.25), (2.5, 1.0)])
    def test_damping_preserves_descent_and_the_coordinate_and_ratio_bounds(self, base_cap, cap_v):
        gradient = np.array([4.0, -3.0, 2.0])
        eigenvectors, _ = np.linalg.qr(np.random.default_rng(29).normal(size=(3, 3)))
        delta, mu, _ = discrete_reml._damped_tensor_newton_step(
            eigenvectors,
            np.array([0.01, 0.4, 2.0]),
            gradient,
            np.arange(3),
            3,
            ["x", "z", "w"],
            [("x:z", (0, 1)), ("z:w", (1, 2))],
            np.zeros(3, dtype=bool),
            base_cap=base_cap,
            cap_v=cap_v,
        )
        slack = 64 * np.finfo(float).eps
        assert np.max(np.abs(delta)) <= base_cap * (1 + slack)
        assert abs(delta[0] - delta[1]) / 2 <= cap_v * (1 + slack)
        assert abs(delta[1] - delta[2]) / 2 <= cap_v * (1 + slack)
        assert gradient @ delta < 0
        assert mu > 0

    def test_every_tensor_step_is_a_descent_direction(self):
        """Fails unfixed: iterations 3-12 build a step with ``g . d = +6.98``.

        The unfixed engine records no directional derivative at all, and
        the quantity the fixed engine records is positive for ten of the
        twelve iterations there, because the per-pair clip rotated the
        step into an ascent direction. A null step has ``g . d == 0``, so
        the bound is ``<= 0`` with at least one strict descent.
        """
        X, y = _stall_frame()
        model = _tensor_model()
        model.fit_reml(X, y, max_reml_iter=12, runtime_validation="skip")

        stats = model.reml_diagnostics()["profile"]["reml_outer_step_stats"]
        assert stats, "the surrogate branch must have run"
        gdots = [entry["gdot_damped"] for entry in stats]
        assert all(g <= 0.0 for g in gdots), gdots
        assert any(g < 0.0 for g in gdots), gdots
        # Damping is what makes it so: the trust region binds on this
        # geometry and mu is driven off zero on at least one iteration.
        mus = [entry["trust_mu"] for entry in stats]
        assert any(mu > 0.0 for mu in mus), mus
        assert any(entry["trust_binding"] for entry in stats)

    def test_stalled_tensor_fit_reaches_a_lower_objective(self):
        """Fails unfixed: the published objective is pinned at 1011.7061.

        The unfixed fit freezes rho at iteration 3 and publishes the same
        objective at max_reml_iter=12 and at 40. Removing the ascent step
        reaches at least 10 REML units lower (measured headroom 13.19).
        """
        X, y = _stall_frame()
        model = _tensor_model()
        model.fit_reml(X, y, max_reml_iter=40, runtime_validation="skip")

        result = model._reml_result
        assert result.objective <= UNFIXED_STALL_OBJECTIVE - STALL_OBJECTIVE_MARGIN
        assert result.n_reml_iter < 40

    def test_dead_tensor_line_search_names_its_exit(self, monkeypatch):
        """Fails unfixed: reports ``max_reml_iter`` after the full budget.

        With every lambda move rejected by the true objective, rho never
        moves, the candidate's own working-model step settles at
        iteration 2 (measured: PIRLS ``converged`` per iteration is
        ``[False, True, True, ...]``) and every later search is a genuine
        dead search with evaluated, rejected trials. The unfixed loop has
        no exit for it and burns all twelve iterations.
        """
        reject, evaluated = _reject_every_move()
        monkeypatch.setattr(discrete_reml, "reml_laml_objective", reject)

        X, y = _stall_frame()
        model = _tensor_model()
        model.fit_reml(X, y, max_reml_iter=12, runtime_validation="skip")

        result = model._reml_result
        assert result.termination_reason == "line_search_failed"
        assert not result.converged
        assert 2 <= result.n_reml_iter < 12
        # The evaluated-lambdas contract of the rejected-search path holds.
        assert result.lambdas == pytest.approx(evaluated)
        record = model.reml_diagnostics()["profile"]["reml_dead_line_search"]
        assert record["candidate_mode_stationary"] is True
        assert record["evaluated_trial"] is True
        assert record["active_gradient_norm"] > record["bar"]

    def test_surrogate_backtrack_reaches_below_one_thirty_second(self, monkeypatch):
        """Fails unfixed: the surrogate stops at halving 5, above ``s = 1/32``.

        ``local_max_halving = 5`` floored the backtrack at ``s = 1/16``,
        so a direction whose model minimiser sits below ``1/32`` died with
        every trial predicting an increase and no true objective evaluated
        (the workflow-1 counterfactual arm: ``s* = 0.0149``). Such a state
        cannot arise from the damped step itself (its minimiser is at or
        beyond ``s = 1``), so the step is injected through the damping
        seam -- the unfixed engine has no such seam and no such schedule.
        The injected direction is 100x the modified-Newton step, whose
        model minimiser is therefore ``s* = 1/100``.
        """
        real = discrete_reml._damped_tensor_newton_step

        def hundredfold_newton(eigvecs, eigvals_pd, grad_sub, active_idx, m, *args, **kwargs):
            delta = np.zeros(m)
            delta[active_idx] = -(eigvecs * (1.0 / eigvals_pd)) @ (eigvecs.T @ grad_sub)
            return 100.0 * delta, 0.0, []

        assert callable(real)
        monkeypatch.setattr(discrete_reml, "_damped_tensor_newton_step", hundredfold_newton)

        X, y = _mild_frame()
        model = _tensor_model()
        model.fit_reml(X, y, max_reml_iter=1, runtime_validation="skip")

        profile = model.reml_diagnostics()["profile"]
        entry = profile["reml_outer_step_stats"][0]
        s_star = -entry["quad_grad"] / entry["quad_curv"]
        assert entry["quad_grad"] < 0.0 and s_star < 1.0 / 32.0, entry
        # The surrogate vetoed every length down to and including 1/32 ...
        assert entry["halvings"] >= 5, entry
        # ... and the true objective was consulted below it.
        assert entry["first_full_eval_step"] is not None
        assert entry["first_full_eval_step"] < 1.0 / 32.0, entry
        assert profile["reml_n_linesearch_full_evals"] >= 1


# ── Regression guards: pass on the unfixed engine by design ──────────────


class TestUntouchedByTheTensorRepair:
    def test_additive_discrete_fit_is_bit_identical(self):
        """Regression guard: an additive fit never enters the tensor branch."""
        X, y = _additive_frame()
        model = _additive_model()
        model.fit_reml(X, y, runtime_validation="skip")

        result = model._reml_result
        assert result.converged
        assert result.termination_reason == "score_objective_tolerance"
        assert result.objective == UNFIXED_ADDITIVE_OBJECTIVE
        assert result.lambdas == UNFIXED_ADDITIVE_LAMBDAS
        assert "reml_outer_step_stats" not in model.reml_diagnostics()["profile"]

    def test_additive_discrete_fit_keeps_its_iteration_limit(self):
        """Regression guard: a budget-limited additive fit stays unconverged.

        The C3 contract for this change: an additive discrete fit that
        terminates at ``max_reml_iter`` today must keep ``converged=False``
        and ``max_reml_iter`` -- it must not acquire ``converged_at_precision``
        or ``line_search_failed`` -- and publish the same numbers.
        """
        X, y = _additive_frame()
        model = _additive_model()
        model.fit_reml(X, y, max_reml_iter=3, runtime_validation="skip")

        result = model._reml_result
        assert not result.converged
        assert result.termination_reason == "max_reml_iter"
        assert result.n_reml_iter == 3
        assert result.objective == UNFIXED_ADDITIVE_MAXITER3_OBJECTIVE
        assert result.lambdas == UNFIXED_ADDITIVE_MAXITER3_LAMBDAS

    def test_additive_dead_search_never_becomes_converged(self, monkeypatch):
        """C3 on the non-surrogate path: a dead search there keeps iterating.

        The gate measured a real additive binomial fit that accepts a step
        immediately after every one of its dead searches (one working-model
        update per outer iteration moves the gradient at unchanged rho), so
        the dead-search exit is restricted to the surrogate path. This fit
        runs the 25-halving true-objective path with every move rejected:
        it keeps ``converged=False`` and ``max_reml_iter`` and never sees
        the dead-search record.
        """
        reject, evaluated = _reject_every_move()
        monkeypatch.setattr(discrete_reml, "reml_laml_objective", reject)

        X, y = _additive_frame()
        model = _additive_model()
        model.fit_reml(X, y, max_reml_iter=4, runtime_validation="skip")

        result = model._reml_result
        assert not result.converged
        assert result.termination_reason == "max_reml_iter"
        assert result.n_reml_iter == 4
        assert result.lambdas == pytest.approx(evaluated)
        assert "reml_dead_line_search" not in model.reml_diagnostics()["profile"]

    def test_gamma_interaction_fit_is_bit_identical(self):
        """Regression guard: unknown scale gates the surrogate branch off."""
        X, y = _gamma_frame()
        model = _tensor_model(family="gamma")
        model.fit_reml(X, y, max_reml_iter=12, runtime_validation="skip")

        result = model._reml_result
        assert result.converged
        assert result.termination_reason == "score_objective_tolerance"
        assert result.objective == UNFIXED_GAMMA_OBJECTIVE
        assert result.lambdas == UNFIXED_GAMMA_LAMBDAS
        assert "reml_outer_step_stats" not in model.reml_diagnostics()["profile"]

    def test_converging_one_pair_fit_finds_the_same_optimum(self):
        """Regression guard: a converging surrogate-path fit keeps its optimum.

        The step changed, so this fit need not be bit-identical, but it
        must reach the same optimum and still exit on the objective
        criterion.
        """
        X, y = _mild_frame()
        model = _tensor_model()
        model.fit_reml(X, y, max_reml_iter=20, runtime_validation="skip")

        result = model._reml_result
        assert result.converged
        assert result.termination_reason == "score_objective_tolerance"
        assert result.objective == pytest.approx(UNFIXED_MILD_OBJECTIVE, rel=1e-6)
        for name, value in UNFIXED_MILD_LAMBDAS.items():
            assert result.lambdas[name] == pytest.approx(value, rel=1e-4)
