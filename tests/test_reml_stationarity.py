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


def test_a_term_whose_curvature_places_a_minimum_is_not_held_at_the_tail_bar():
    """The absolute bar is a tail's: curvature beyond a tail's keeps the relative resolution.

    At V = -81 that resolution is 8.2e-6. A score of 1.4e-5 on a curvature of
    2.5e-3 has its minimum 0.6% away in lambda and stays free, while the same
    score on a tail (curvature equal to the score) is held, as the first is
    once its score is under the resolution. On a book the absolute bar is the
    smaller of the two and the curvature changes nothing, and at a coefficient
    fit the guard stopped the curvature places nothing.
    """
    g = np.array([-1.4e-5, -1.4e-5, -5e-6])
    h = np.diag([2.5e-3, 1.4e-5, 2.5e-3])

    def held(objective, **kwargs):
        return freeze_flat_directions(
            g,
            h,
            np.full(3, 10.0),
            np.ones(3, dtype=bool),
            objective=objective,
            tolerance=1e-9,
            step_bound=5.0,
            **kwargs,
        ).frozen

    np.testing.assert_array_equal(held(-81.3), [False, True, True])
    np.testing.assert_array_equal(held(1e5), [True, True, True])
    np.testing.assert_array_equal(held(-81.3, guard_stopped=True), [True, True, True])


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


@pytest.mark.parametrize(
    "half_step", [pytest.param("halved", marks=pytest.mark.pins_behaviour), "tail-trial"]
)
def test_a_step_stop_reads_contraction_against_the_newton_step_proposed(monkeypatch, half_step):
    """Pin the contraction read after a halved step; an accepted tail trial gives none.

    Newton proposes 4e-5 on every term and half of it is taken, which leaves
    2e-5 on the same Hessian. Against the proposal that is a contraction of
    1/2 and a distance left of 4e-5, under the step bar: the simplified Newton
    correction over the ordinary one (Deuflhard 2011, ch. 3, the natural
    monotonicity test). Against the half taken it would read as no
    contraction. Where the half was a tail trial's, the held Hessian's step
    was not the one taken and confirms nothing.
    """
    from superglm.reml import direct

    from .test_reml_tol_determination import _compound_test_never_accepts, _fit_informative_fixture

    _compound_test_never_accepts(monkeypatch)
    reads = trials = 0
    gradient = direct.reml_direct_gradient
    objective = direct.reml_laml_objective

    def scores(*args, **kwargs):
        nonlocal reads
        reads += 1
        return np.full_like(gradient(*args, **kwargs), -4e-5 if reads == 1 else -2e-5)

    def downhill(*args, **kwargs):
        nonlocal trials
        ev = objective(*args, **kwargs)
        trials += reads == 1
        if half_step == "halved" and reads == 1 and trials == 1:
            return replace(ev, value=np.inf)
        return replace(ev, value=ev.value - 100.0 * reads)

    monkeypatch.setattr(direct, "reml_direct_gradient", scores)
    monkeypatch.setattr(direct, "reml_direct_hessian", lambda *a, **kw: np.eye(len(kw["gradient"])))
    monkeypatch.setattr(direct, "reml_w_correction", lambda *a, **kw: None)
    monkeypatch.setattr(direct, "reml_laml_objective", downhill)
    if half_step == "tail-trial":
        monkeypatch.setattr(direct, "flat_tail_trial", lambda delta, *a, **kw: 0.5 * delta)
    model = _fit_informative_fixture(discrete=False, max_reml_iter=2)
    stop = model._reml_profile.get("reml_newton_step_stop")
    if half_step == "tail-trial":
        assert stop is None
        assert not model.reml_diagnostics()["converged"]
    else:
        assert model.reml_diagnostics()["termination_reason"] == "newton_step_tolerance"
        assert stop["taken_step"] == pytest.approx(4e-5)
        assert stop["step_left"] == pytest.approx(2e-5)


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


def test_a_revalidated_face_publishes_the_fit_its_stationarity_was_judged_on(monkeypatch):
    """A stationary stop on an exact face publishes the fit the exact gradient judged.

    The face revalidation judges that fit and does not fit the face again, so
    the stop needs no second verdict: the endgame runs once, and the fit it
    called stationary is the terminal fit. Mutation check: a revalidation that
    fits the face again publishes another fit, whose stationarity the endgame
    never judged.
    """
    from .test_distributional_newton_practical_handoff import _forced_face_plateau_fit

    fits = []

    def verdict(result):
        fits.append(result.state.fit)
        return replace(result, kind="stationary")

    calls, sm = _forced_face_plateau_fit(monkeypatch, verdict, outer="efs+stationary")
    assert len(calls) == 1
    assert fits[0] is sm.terminal_fit
    assert sm.converged and sm.convergence_reason == "stationary"
    assert sm.history[-1].revalidated_face_components
    assert sm.unresolved_upper_bound == ()


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


def test_the_flat_bar_is_proportional_to_the_tolerance():
    from superglm.reml.convergence import flat_score_bar

    # A candidate that only ranks Tweedie powers runs at 1e-6: with |V| above
    # 1e5 its stop reads the absolute bar, a thousand times the default's.
    def stops(score, tolerance):
        return evaluate_reml_candidate(
            iteration=1,
            objective=3e5,
            previous_objective=3e5,
            projected_gradient=np.array([score]),
            tolerance=tolerance,
        ).converged

    assert stops(0.05, 1e-6)
    assert not stops(0.2, 1e-6)
    assert not stops(0.05, 1e-7)
    assert [flat_score_bar(t) for t in (1e-12, 1e-9, 1e-6)] == pytest.approx([1e-7, 1e-4, 0.1])


def test_a_coarse_fit_holds_at_the_relative_resolution_of_its_score():
    from superglm.reml.convergence import hold_score_bar

    # Default fidelity and tighter: the flat-tail bar, whatever the objective.
    assert [hold_score_bar(v, 1e-9) for v in (10.0, 3e3, 1e7)] == [1e-4] * 3
    assert hold_score_bar(3e3, 1e-12) == pytest.approx(1e-7)
    # reml_tol=1e-6: 1e-7 (1 + |V|), between the default's bar and its own.
    assert hold_score_bar(3e3, 1e-6) == pytest.approx(3.001e-4)
    assert hold_score_bar(10.0, 1e-6) == 1e-4
    assert hold_score_bar(1e7, 1e-6) == pytest.approx(0.1)


@pytest.mark.pins_behaviour
def test_a_coarse_fit_does_not_stop_on_a_hold_looser_than_its_resolution():
    """Pin the hold a fit at reml_tol=1e-6 reads, in the step and in the stop's mask.

    At V = 3e3 it is 1e-7 (1 + |V|) = 3e-4: a held score of 2e-4 stays held
    and out of the stop's evidence, and one of 0.05 does neither.
    """
    from superglm.reml.convergence import mask_frozen_stop_gradient

    score = np.array([0.05, 2e-4])
    held = np.array([True, True])
    masked = mask_frozen_stop_gradient(score, held, objective=3e3, tolerance=1e-6)
    np.testing.assert_array_equal(masked, [0.05, 0.0])
    verdict = freeze_flat_directions(
        score, np.diag([1e-5, 1e-5]), np.ones(2), held, objective=3e3, tolerance=1e-6
    )
    np.testing.assert_array_equal(verdict.frozen, [False, True])


def test_a_guard_stopped_coefficient_fit_loosens_the_hold_and_never_tightens_it():
    from superglm.reml.convergence import hold_score_bar

    # 1e-7 (1 + |V|) is below the flat bar on a small objective.
    assert hold_score_bar(10.0, 1e-9, guard_stopped=True) == 1e-4
    assert hold_score_bar(1e5, 1e-9, guard_stopped=True) == pytest.approx(1e-2, rel=1e-4)
    assert hold_score_bar(1e5, 1e-9, guard_stopped=False) == 1e-4


def _script_scalar_search(monkeypatch, engine, *, scores, curvature, coefficient_fit):
    """Fix the objective at 1e5 and script the score, Hessian diagonal and coefficient fit.

    ``scores`` and ``curvature`` give (every other term, last term).
    ``coefficient_fit`` is "converged", "unconverged" (a step short of its
    tolerance), "guard-stopped" (unconverged, with the solver's finding that
    the link's overflow guard stopped the walk) or "guard-converged" (that
    finding on a fit that met its tolerance).
    """
    objective = engine.reml_laml_objective
    gradient = engine.reml_direct_gradient
    fit = engine.fit_irls_direct
    state = {
        "converged": dict(converged=True),
        "guard-converged": dict(converged=True, separation_message="scripted"),
        "unconverged": dict(converged=False, termination_reason="max_iter"),
        "guard-stopped": dict(
            converged=False, termination_reason="max_iter", separation_message="scripted"
        ),
    }[coefficient_fit]

    def fixed_objective(*args, **kwargs):
        value = objective(*args, **kwargs)
        return replace(value, value=1e5) if hasattr(value, "value") else 1e5

    def scripted(values, n):
        out = np.full(n, values[0])
        out[-1] = values[1]
        return out

    def coefficients(*args, **kwargs):
        result, *rest = fit(*args, **kwargs)
        return replace(result, **state), *rest

    monkeypatch.setattr(engine, "fit_irls_direct", coefficients)
    monkeypatch.setattr(engine, "reml_laml_objective", fixed_objective)
    monkeypatch.setattr(
        engine, "reml_direct_gradient", lambda *a, **kw: scripted(scores, len(gradient(*a, **kw)))
    )
    monkeypatch.setattr(
        engine,
        "reml_direct_hessian",
        lambda *a, **kw: np.diag(scripted(curvature, len(kw["gradient"]))),
    )
    if hasattr(engine, "reml_w_correction"):
        monkeypatch.setattr(engine, "reml_w_correction", lambda *a, **kw: None)


@pytest.mark.filterwarnings("ignore::superglm.SeparationWarning")
@pytest.mark.parametrize("discrete", [False, True], ids=["exact", "binned"])
@pytest.mark.parametrize(
    "coefficient_fit", ["guard-stopped", "guard-converged", "unconverged", "converged"]
)
def test_a_score_between_the_bars_is_held_only_at_a_guard_stopped_fit(
    monkeypatch, discrete, coefficient_fit
):
    """The hold bar on a flat term with a score of 1e-3, at V = 1e5.

    A coefficient fit the link's guard stopped resolves the score to
    1e-7 (1 + |V|) = 1e-2, and the term is held for that round. Any other fit
    reads 1e-4 whatever the objective's size and leaves it free: a binned
    round one step short of its tolerance is the engine's ordinary state.
    """
    from superglm.reml import direct
    from superglm.reml import discrete as cached

    from .test_reml_tol_determination import _fit_informative_fixture

    _script_scalar_search(
        monkeypatch,
        cached if discrete else direct,
        scores=(-1.0, -1e-3),
        curvature=(1.0, 1e-5),
        coefficient_fit=coefficient_fit,
    )
    model = _fit_informative_fixture(discrete=discrete, max_reml_iter=1)
    frozen = model._reml_profile["reml_freeze_decision"]["frozen"]
    assert not any(frozen[:-1])
    assert frozen[-1] is (coefficient_fit == "guard-stopped")


@pytest.mark.filterwarnings("ignore::superglm.SeparationWarning")
@pytest.mark.parametrize(("score", "converged"), [(-1e-3, False), (-1e-5, True)])
def test_a_binned_fit_cannot_stop_on_terms_held_at_the_looser_bar(monkeypatch, score, converged):
    """Every term is flat and held at a guard-stopped coefficient fit.

    With scores of 1e-3 the holds rest on the looser bar and the fit is not
    converged; with scores inside the flat bar it stops as before.
    """
    from superglm.reml import discrete

    from .test_reml_tol_determination import _fit_informative_fixture

    _script_scalar_search(
        monkeypatch,
        discrete,
        scores=(score, score),
        curvature=(1e-5, 1e-5),
        coefficient_fit="guard-stopped",
    )
    model = _fit_informative_fixture(discrete=True, max_reml_iter=4)
    assert all(model._reml_profile["reml_freeze_decision"]["frozen"])
    result = model._reml_result
    assert result.converged is converged
    assert converged or result.termination_reason == "max_reml_iter"
