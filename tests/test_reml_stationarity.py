"""Stopping evidence on flat tails and on free log smoothing parameters."""

from dataclasses import replace

import numpy as np
import pytest

from superglm.reml.convergence import evaluate_reml_candidate, freeze_flat_directions

from .test_distributional_newton_practical_handoff import _two_smooth_gaussian


def test_tail_shoulder_and_free_score_survive_an_objective_shift():
    # V = c + a exp(-rho). The two signs have opposite descent directions.
    # Changing c leaves both derivatives and their scientific units intact.
    g = np.array([-5e-5, 5e-5, -3e-4, 1.0])
    h = np.diag([5e-5, -5e-5, 3e-4, 2.0])
    for objective in (10.0, 1e8):
        verdict = freeze_flat_directions(
            g,
            h,
            np.ones(4),
            np.ones(4, dtype=bool),
            objective=objective,
            tolerance=1e-9,
            step_bound=5.0,
        )
        np.testing.assert_array_equal(verdict.frozen, [True, False, False, False])
        np.testing.assert_array_equal(verdict.shoulder, [False, True, False, False])
        assert not evaluate_reml_candidate(
            iteration=1,
            objective=objective,
            previous_objective=objective,
            projected_gradient=g[2:3],
            tolerance=1e-9,
        ).converged


def test_tighter_fidelity_releases_a_previously_flat_term():
    g = np.array([-6e-5])
    h = np.diag(-g)

    def held(tolerance):
        return freeze_flat_directions(
            g,
            h,
            np.ones(1),
            np.ones(1, dtype=bool),
            objective=1e5,
            tolerance=tolerance,
            step_bound=5.0,
        ).frozen[0]

    assert held(1e-9)
    assert not held(1e-10)


def test_a_dead_search_cannot_hide_an_unresolved_score_in_a_large_objective():
    from superglm.reml.convergence import classify_dead_feasible_exit

    assert (
        classify_dead_feasible_exit(
            0.004,
            objective=1e6,
            tolerance=1e-9,
            evaluated_trial=True,
        )
        == "line_search_failed"
    )


def test_a_tail_trial_reaches_the_score_bar_only_in_the_settled_working_model():
    from superglm.reml.convergence import flat_tail_trial

    # One normal coefficient, unit data precision, zero fitted signal.
    # V = log(1 + 1/lambda)/2, -2V' = EDF = 1/(1 + lambda).
    lam = 99.0
    score = np.array([-0.5 / (1.0 + lam), 1.0])
    h = np.diag([lam / (2.0 * (1.0 + lam) ** 2), 1.0])
    verdict = freeze_flat_directions(
        score,
        h,
        np.ones(2),
        np.ones(2, dtype=bool),
        objective=1e5,
        tolerance=1e-9,
    )

    def trial(*, score_at_fit=score, fit=0.0, isolated=True, partner=0.1, step_cap=5.0):
        return flat_tail_trial(
            np.array([1.0, partner]),
            score_at_fit,
            score,
            np.array([fit, 0.0]),
            np.array([isolated, True]),
            verdict,
            bar=1e-4,
            step_cap=step_cap,
        )

    proposal = trial()
    assert proposal is not None
    assert 0.5 / (1.0 + lam * np.exp(proposal[0])) < 1e-4
    assert proposal[1] == 0.1
    assert trial(step_cap=2.0)[0] == 2.0
    assert trial(isolated=False) is None
    assert trial(partner=2.0) is None
    assert trial(score_at_fit=-score) is None
    assert trial(fit=0.5) is None


def test_a_contracting_scalar_step_rechecks_the_curvature_of_held_terms(monkeypatch):
    from superglm.reml import direct

    from .test_reml_tol_determination import _compound_test_never_accepts, _fit_informative_fixture

    _compound_test_never_accepts(monkeypatch)
    # Force a small accepted Newton step on x1. At the new fit only x2's
    # curvature changes. The old Hessian would certify the x1-only step.
    reads = 0
    gradient = direct.reml_direct_gradient

    def scores(*args, **kwargs):
        nonlocal reads
        reads += 1
        g = gradient(*args, **kwargs)
        return np.full_like(g, -4e-5) if reads == 1 else np.full_like(g, -1e-5)

    def curvature(*args, **kwargs):
        n = len(kwargs["gradient"])
        diag = np.ones(n)
        if reads == 1:
            diag[-1] = 1e-5
        return np.diag(diag)

    objective = direct.reml_laml_objective

    def downhill(*args, **kwargs):
        ev = objective(*args, **kwargs)
        return replace(ev, value=ev.value - 100.0 * reads)

    monkeypatch.setattr(direct, "reml_direct_gradient", scores)
    monkeypatch.setattr(direct, "reml_direct_hessian", curvature)
    monkeypatch.setattr(direct, "reml_w_correction", lambda *a, **kw: None)
    monkeypatch.setattr(direct, "reml_laml_objective", downhill)
    model = _fit_informative_fixture(discrete=False, max_reml_iter=2)
    assert reads == 2
    assert not model.reml_diagnostics()["converged"]
    assert not any(model._reml_profile["reml_freeze_decision"]["frozen"])


@pytest.mark.parametrize("unavailable", [False, True])
def test_stationary_finish_publishes_a_gradient_at_the_returned_fit(monkeypatch, unavailable):
    from superglm.distributional.smoothing import newton
    from superglm.distributional.smoothing.derivatives import LamlDerivativeError

    if unavailable:

        def refuse(*args, **kwargs):
            raise LamlDerivativeError("forced unavailable derivative")

        monkeypatch.setattr(newton, "laml_derivatives", refuse)
    model, frame, y = _two_smooth_gaussian()
    model.fit_reml(frame, y, outer="efs+stationary")
    sm = model._require_fitted().smoothing
    assert sm.converged == (not unavailable)
    assert sm.convergence_reason == ("derivative_unavailable" if unavailable else "stationary")
    assert np.isfinite(model.predict(frame)).all().all()
    if not unavailable:
        assert sm.terminal_evidence_fresh
        assert max(map(abs, sm.terminal_gradient.values())) <= 1e-4


@pytest.mark.parametrize("penalty_sign", [-1.0, 1.0], ids=["visible", "likelihood-curvature"])
def test_a_stationary_finish_cannot_accept_a_shoulder_on_a_dead_search(monkeypatch, penalty_sign):
    from superglm.distributional.smoothing import newton

    derivatives = newton.laml_derivatives

    def shoulder(*args, **kwargs):
        result = derivatives(*args, **kwargs)
        g = np.full_like(result.gradient, 5e-5)
        h = -np.diag(g)
        return replace(
            result,
            gradient=g,
            gradient_certificate=np.zeros_like(g),
            penalty_hessian=penalty_sign * np.diag(g),
            hessian=h if kwargs["want_hessian"] else None,
        )

    monkeypatch.setattr(newton, "laml_derivatives", shoulder)
    model, frame, y = _two_smooth_gaussian()
    monkeypatch.setattr(
        newton._TrialFitter, "fit", lambda self, *a, **kw: (kw["reuse_source"], None)
    )
    model.fit_reml(frame, y, outer="efs+stationary")
    sm = model._require_fitted().smoothing
    assert not sm.converged
    assert sm.convergence_reason == "objective_rejected"


def test_a_finish_with_only_held_terms_uses_one_gradient_pass(monkeypatch):
    from superglm.distributional.smoothing import newton

    derivatives = newton.laml_derivatives
    passes = []

    def tail(*args, **kwargs):
        passes.append(kwargs["want_hessian"])
        result = derivatives(*args, **kwargs)
        g = np.full_like(result.gradient, -5e-5)
        return replace(result, gradient=g, gradient_certificate=0 * g, penalty_hessian=-np.diag(g))

    monkeypatch.setattr(newton, "laml_derivatives", tail)
    model, frame, y = _two_smooth_gaussian()
    model.fit_reml(frame, y, outer="efs+stationary")
    sm = model._require_fitted().smoothing
    assert sm.converged and sm.terminal_evidence_fresh
    assert sm.terminal_projected_gradient_norm == 0
    assert passes == [False]


@pytest.mark.parametrize(
    ("replay", "reason"),
    [
        ("cap_pressure", "lambda_cap_unresolved"),
        ("max_iterations", "gradient_unresolved"),
        ("derivative_unavailable", "derivative_unavailable"),
    ],
)
def test_a_face_refit_must_supply_its_own_stationarity(replay, reason, monkeypatch):
    from .test_distributional_newton_practical_handoff import _forced_face_plateau_fit

    fits = []

    def verdict(result):
        fits.append(result.state.fit)
        if len(fits) == 1:
            return replace(result, kind="stationary")
        caps = tuple(result.derivatives.names[:1]) if replay == "cap_pressure" else ()
        return replace(result, kind=replay, cap_pressure=caps)

    calls, sm = _forced_face_plateau_fit(monkeypatch, verdict, outer="efs+stationary")
    assert len(calls) == 2
    assert calls[1]["budget"] == 0
    assert fits[1] is sm.terminal_fit and fits[0] is not fits[1]
    assert not sm.converged and sm.convergence_reason == reason
    assert bool(sm.unresolved_upper_bound) == (replay == "cap_pressure")


def test_a_validated_face_with_no_free_terms_has_an_empty_projected_score():
    from superglm import GaussianLS, SuperLSS, re

    from .test_distributional_practical_stop import _preempt_fixture

    frame, y = _preempt_fixture()
    family = GaussianLS(scale_floor=1e-4)
    model = SuperLSS(family, family.location(re("effect")), family.scale())
    model.fit_reml(
        frame,
        y,
        lambdas={"location:effect#wiggle": 1000.0},
        max_lambda=1002.5,
        max_log_step=1e-3,
        max_reml_iter=10,
        reml_tol=1e-8,
        inner_tol=1e-10,
        outer="efs+stationary",
    )
    sm = model._require_fitted().smoothing
    assert sm.converged and sm.convergence_reason == "stationary"
    assert tuple(sm.terminal_endpoint_directions) == ("location:effect#wiggle",)
    assert sm.terminal_gradient == {} and sm.terminal_projected_gradient_norm == 0
    assert sm.terminal_evidence_fresh


def test_outward_and_inward_scores_at_the_same_cap_are_distinguished(monkeypatch):
    from superglm.distributional.smoothing import loop, newton

    real_derivatives = newton.laml_derivatives
    endgame = loop.run_newton_endgame
    verdicts = []

    def compare(*args, **kwargs):
        state = kwargs["state"]
        for signed_score in (-5e-5, 5e-5):

            def scores(*a, **kw):
                result = real_derivatives(*a, **kw)
                g = np.full_like(result.gradient, -5e-5)
                g[0] = signed_score
                hessian = np.diag(np.full_like(g, 5e-5))
                hessian[0, 0] = 1.0
                return replace(
                    result, gradient=g, gradient_certificate=0 * g, penalty_hessian=hessian
                )

            cap = next(iter(state.lambdas))
            with monkeypatch.context() as m:
                m.setattr(newton, "laml_derivatives", scores)
                result = endgame(
                    *args, **{**kwargs, "upper_bounds": {cap: state.lambdas[cap]}, "budget": 0}
                )
            verdicts.append((result.kind, result.projected_gradient_norm))
        return endgame(*args, **kwargs)

    monkeypatch.setattr(loop, "run_newton_endgame", compare)
    model, frame, y = _two_smooth_gaussian()
    model.fit_reml(frame, y, outer="efs+stationary")
    assert verdicts == [("cap_pressure", 0.0), ("max_iterations", 5e-5)]


@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
def test_a_refused_tail_leaves_the_full_newton_trial_available(monkeypatch, discrete):
    from superglm.reml import direct
    from superglm.reml import discrete as cached

    from .test_reml_tol_determination import _fit_informative_fixture

    engine = cached if discrete else direct
    objective = engine.reml_laml_objective
    refit = False
    trials = []

    def propose(delta, *args, **kwargs):
        nonlocal refit
        refit = True
        return 0.5 * delta

    def force_one_refusal(*args, **kwargs):
        value = objective(*args, **kwargs)
        if refit and len(trials) < 2:
            trials.append(np.log(list(args[6].values())))
            if len(trials) == 1:
                return replace(value, value=np.inf) if hasattr(value, "value") else np.inf
        return value

    original_gradient = engine.reml_direct_gradient
    current = []

    def score(*args, **kwargs):
        if not refit:
            current.append(np.log(list(args[3].values())))
        return original_gradient(*args, **kwargs)

    monkeypatch.setattr(engine, "flat_tail_trial", propose)
    monkeypatch.setattr(engine, "reml_laml_objective", force_one_refusal)
    monkeypatch.setattr(engine, "reml_direct_gradient", score)
    _fit_informative_fixture(discrete=discrete, max_reml_iter=4)
    half, full = trials
    np.testing.assert_allclose(full - current[-1], 2 * (half - current[-1]), atol=1e-12)
    assert np.max(np.abs(full - half)) > 0.01


@pytest.mark.parametrize("scored", [True, pytest.param(False, marks=pytest.mark.pins_behaviour)])
def test_a_binned_rejection_supplies_precision_evidence_only_if_scored(monkeypatch, scored):
    """Pin refusal without a finite trial; a scored rejection can resolve a small PD step."""
    from superglm.reml import discrete

    from .test_reml_tol_determination import _compound_test_never_accepts, _fit_informative_fixture

    _compound_test_never_accepts(monkeypatch)
    objective = discrete.reml_laml_objective
    fit = discrete.fit_irls_direct

    def fixed_objective(*args, **kwargs):
        value = objective(*args, **kwargs)
        if kwargs.get("return_evaluation"):
            return replace(value, value=100.0)
        return 101.0 if scored else np.inf

    def certified(*args, **kwargs):
        result, *rest = fit(*args, **kwargs)
        return replace(result, converged=True), *rest

    monkeypatch.setattr(discrete, "fit_irls_direct", certified)
    monkeypatch.setattr(discrete, "reml_laml_objective", fixed_objective)
    monkeypatch.setattr(discrete, "reml_direct_gradient", lambda *a, **kw: np.full(len(a[3]), 1e-7))
    monkeypatch.setattr(
        discrete, "reml_direct_hessian", lambda *a, **kw: np.eye(len(kw["gradient"]))
    )
    model = _fit_informative_fixture(discrete=True, max_reml_iter=3)
    diag = model.reml_diagnostics()
    if scored:
        assert diag["termination_reason"] == "converged_at_precision"
        assert model._reml_profile["reml_dead_line_search"]["evaluated_trial"]
    else:
        assert not diag["converged"]
        assert "reml_newton_step_stop" not in model._reml_profile
