"""Fit-time shape rules on SuperLSS terms (``solver/shape.py``).

Smoothing is chosen without the rules and the coefficients are refitted under them at that
smoothing, by sequential QP. Each fit is checked against an independent optimizer on the same
penalized likelihood and rows, and its covariance against the null-space formula.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from superglm import (
    Constraint,
    GammaLS,
    GaussianLS,
    NegativeBinomialLS,
    SuperLSS,
    TweedieLSS,
    s,
)
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning
from superglm.distributional.solver.assembly import (
    _evaluate_predictors_from_matrices,
    dense_predictor_matrices,
)
from superglm.distributional.solver.shape import shape_rows
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

INCREASING = Constraint.fit.increasing
# Pointwise coverage the seeded test demands of a nominal 95% band where a rule holds a fit flat.
COVERAGE_FLOOR = 0.92
_GRID = pd.DataFrame({"x": np.linspace(0.02, 9.98, 500)})


def _dip(x):
    return 0.4 + 0.18 * x - 0.8 * np.exp(-(((x - 6.0) / 0.9) ** 2))


def _rtweedie(mu, phi, power, rng):
    count = rng.poisson(mu ** (2 - power) / (phi * (2 - power)))
    shape = (2 - power) / (power - 1)
    scale = phi * (power - 1) * mu ** (power - 1)
    return np.where(count > 0, rng.gamma(np.maximum(count, 1) * shape, scale), 0.0)


def _data(kind: str, *, n: int = 400, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 10.0, n)
    mu = np.exp(_dip(x))
    if kind == "gamma":
        y = rng.gamma(6.0, mu / 6.0)
    elif kind == "tweedie":
        y = _rtweedie(mu, 1.2, 1.5, rng)
    elif kind == "nb":
        y = rng.negative_binomial(4.0, 4.0 / (4.0 + mu)).astype(float)
    else:  # gamma with a dip in the scale
        y = rng.gamma(1.0 / np.exp(_dip(x) - 1.0), mu * np.exp(_dip(x) - 1.0))
    return pd.DataFrame({"x": x}), y


def _model(kind: str, constraint=None, *, k: int = 8, basis: str = "cr") -> SuperLSS:
    rule = {} if constraint is None else {"constraint": constraint}
    term = s("x", kind=basis, k=k, **rule)
    if kind == "gamma":
        f = GammaLS()
        return SuperLSS(f, f.mean(term), f.scale())
    if kind == "tweedie":
        f = TweedieLSS()
        return SuperLSS(f, f.mu(term), f.phi(), f.p())
    if kind == "nb":
        f = NegativeBinomialLS()
        return SuperLSS(f, f.mean(term), f.theta())
    f = GammaLS()
    return SuperLSS(f, f.mean(), f.scale(term))


def _curve(model: SuperLSS, parameter: str) -> np.ndarray:
    return model.predict_link(_GRID)[parameter].to_numpy()


def _state(model: SuperLSS):
    return model._model.fit_state


def _oracle(model: SuperLSS, y: np.ndarray) -> tuple[float, np.ndarray]:
    """The penalized negative log-likelihood's minimum under the model's own rows, by trust-constr.

    The objective is the family's log-likelihood at the linear predictors the layout's design
    gives, less half the fitted penalty quadratic; the rows are the compiled shape rows.
    """
    state = _state(model)
    layout, family = state.layout, model.family
    penalty = np.asarray(state.solver_result.penalty)
    rows = shape_rows(layout, state.compiled_predictors)
    matrices = dense_predictor_matrices(layout)
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = [predictor.link for predictor in layout.predictors]

    def objective(beta: np.ndarray) -> float:
        eta = _evaluate_predictors_from_matrices(layout, beta, matrices)
        theta = np.column_stack([link.inverse(eta[:, j]) for j, link in enumerate(links)])
        natural = family.evaluate_natural(y, theta, plan, derivative_order=0)
        loglik = float(np.sum(natural.optimizing_log_likelihood))
        return -(loglik - 0.5 * float(beta @ penalty @ beta))

    start = np.zeros(layout.n_coefficients)
    for predictor in layout.predictors:
        start[predictor.intercept_index] = 0.0
    start[layout.predictors[0].intercept_index] = float(np.log(max(np.mean(y), 1e-3)))
    result = optimize.minimize(
        objective,
        start,
        method="trust-constr",
        constraints=[optimize.LinearConstraint(rows.A, rows.b, np.inf)],
        options={"maxiter": 3000, "gtol": 1e-10, "xtol": 1e-12},
    )
    return float(result.fun), result.x


def _objective(model: SuperLSS) -> float:
    return -float(_state(model).solver_result.penalized_optimizing_log_likelihood)


@pytest.mark.parametrize(
    ("kind", "parameter"),
    [("gamma", "mean"), ("tweedie", "mean"), ("nb", "mean"), ("gamma_scale", "scale")],
)
@pytest.mark.parametrize("basis", ["cr", "ps"])
def test_a_rule_fit_matches_an_independent_optimizer_and_is_monotone(kind, parameter, basis):
    X, y = _data(kind)
    free = _model(kind, basis=basis).fit_reml(X, y)
    assert np.min(np.diff(_curve(free, parameter))) < -1e-3, "the data must dip for this to test"

    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        held = _model(kind, INCREASING, basis=basis).fit_reml(X, y)
    shape = _state(held).shape_fit
    assert shape.applied and shape.converged and shape.n_binding > 0
    assert held.result_.converged

    # the same smoothing, chosen without the rules
    assert held.smoothing_parameters_ == pytest.approx(free.smoothing_parameters_, rel=0, abs=0)
    # monotone to rounding on a dense grid
    assert np.min(np.diff(_curve(held, parameter))) >= -1e-9

    oracle_value, oracle_beta = _oracle(held, y)
    ours = _objective(held)
    assert ours <= oracle_value + 1e-6 * (1.0 + abs(oracle_value))
    assert ours >= oracle_value - 1e-5 * (1.0 + abs(oracle_value))
    state = _state(held)
    spread = np.max(np.abs(state.solver_result.coefficients)) + 1.0
    np.testing.assert_allclose(
        state.solver_result.coefficients, oracle_beta, rtol=0, atol=2e-3 * spread
    )
    # the constrained fit pays for the rule: a worse penalized likelihood than the free one
    assert ours > _objective(free)


@pytest.mark.parametrize("kind", ["gamma", "tweedie", "nb"])
def test_the_kkt_conditions_hold_at_the_constrained_fit(kind):
    """Stationarity on the face, nonnegative multipliers on the binding rows, feasibility."""
    X, y = _data(kind)
    held = _model(kind, INCREASING).fit_reml(X, y)
    state = _state(held)
    rows = shape_rows(state.layout, state.compiled_predictors)
    beta = np.asarray(state.solver_result.coefficients)
    score = np.asarray(state.solver_result.terminal_score)
    slack = (rows.A @ beta - rows.b) / (np.abs(rows.A) @ np.abs(beta))
    assert slack.min() >= -1e-12
    binding = np.flatnonzero(slack <= 1e-9)
    assert binding.size
    # the penalized score is a nonnegative combination of the binding rows
    multipliers = optimize.nnls(rows.A[binding].T, -score)[0]
    residual = score + rows.A[binding].T @ multipliers
    assert np.linalg.norm(residual) <= 1e-6 * (1.0 + np.linalg.norm(score)) + 1e-6 * np.linalg.norm(
        rows.A[binding].T @ multipliers
    )
    free_direction = np.linalg.svd(rows.A[binding])[2][binding.size :].T
    assert np.max(np.abs(free_direction.T @ score)) <= 1e-5 * (1.0 + np.abs(score).max())


def test_an_already_monotone_fit_is_the_unconstrained_fit():
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 10.0, 400)
    y = rng.gamma(6.0, np.exp(0.3 + 0.2 * x) / 6.0)
    X = pd.DataFrame({"x": x})
    free = _model("gamma").fit_reml(X, y)
    held = _model("gamma", INCREASING).fit_reml(X, y)
    shape = _state(held).shape_fit
    assert not shape.applied and shape.n_binding == 0 and shape.converged
    assert np.array_equal(held.result_.coefficients, free.result_.coefficients)
    assert np.array_equal(held.result_.covariance, free.result_.covariance)
    assert held.result_.total_effective_df == free.result_.total_effective_df
    # no row binds, so there is nothing for the sentence to say but that
    assert "did not bind" in held.diagnose().render(detail="full")


def test_neither_a_fit_time_nor_a_postfit_rule_warns_and_only_the_fit_time_one_refits():
    X, y = _data("gamma")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        _model("gamma", INCREASING).fit_reml(X, y)
        post = _model("gamma", Constraint.postfit.increasing).fit_reml(X, y)
    assert _state(post).shape_fit is None


def _face_covariance(model: SuperLSS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Z (Z' G Z)^+ Z'`` from the rows that bind, formed here from the penalized curvature."""
    state = _state(model)
    rows = shape_rows(state.layout, state.compiled_predictors)
    solver = state.solver_result
    beta = np.asarray(solver.coefficients)
    slack = (rows.A @ beta - rows.b) / (np.abs(rows.A) @ np.abs(beta))
    held = rows.A[slack <= 1e-9]
    unit = held / np.linalg.norm(held, axis=1)[:, None]
    _, singular, vt = np.linalg.svd(unit)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    basis = vt[rank:].T
    curvature = np.asarray(solver.terminal_penalized_curvature)
    covariance = basis @ np.linalg.pinv(basis.T @ curvature @ basis) @ basis.T
    return covariance, curvature, held


@pytest.mark.parametrize("kind", ["gamma", "tweedie"])
def test_a_binding_rule_reports_the_covariance_of_the_fit_without_it(kind):
    """Which rows bind is estimated, so the covariance is not the one conditional on them."""
    X, y = _data(kind)
    free = _model(kind).fit_reml(X, y)
    held = _model(kind, INCREASING).fit_reml(X, y)
    assert _state(held).shape_fit.n_binding > 0
    reported = np.asarray(held.result_.covariance)
    np.testing.assert_allclose(reported, np.asarray(free.result_.covariance), rtol=1e-10, atol=0.0)
    # a binding row keeps its variance: the face's covariance gives it none
    face, _, binding = _face_covariance(held)
    scale = np.abs(reported).max() * np.max(np.abs(binding)) ** 2
    assert np.max(np.abs(binding @ face @ binding.T)) <= 1e-9 * scale
    assert np.min(np.diag(binding @ reported @ binding.T)) > 1e-6 * scale
    # so some standard error is materially larger than the face's
    se, face_se = np.sqrt(np.diag(reported)), np.sqrt(np.maximum(np.diag(face), 0.0))
    assert np.max(se / np.maximum(face_se, 1e-300)) > 1.05


@pytest.mark.slow
def test_bands_cover_a_rising_curve_where_the_rule_holds_the_fit_flat():
    """Seeded coverage of the pointwise 95% band at the points an increasing rule holds flat.

    The truth rises everywhere, so every flat stretch of a fit is the rule binding by chance.
    The smoothing is fixed and light, which makes that happen in every fit. A band conditional
    on the binding rows covers the truth at 80% of those points; the band has to cover it about
    as often as its level says, and the floor leaves room for a few points either way.
    """
    from superglm import LambdaPolicy

    family = GaussianLS()
    hits = points = binding_fits = 0
    for replicate in range(40):
        rng = np.random.default_rng([616, replicate])
        x = rng.uniform(0.0, 10.0, 200)
        y = 1.0 + 0.06 * x + 0.3 * rng.standard_normal(200)
        term = s("x", kind="cr", k=10, constraint=INCREASING, lambda_policy=LambdaPolicy.fixed(1.0))
        model = SuperLSS(family, family.location(term), family.scale()).fit(
            pd.DataFrame({"x": x}), y
        )
        if _state(model).shape_fit.n_binding == 0:
            continue
        binding_fits += 1
        effect = model.term_inference("location", "x", n_points=200, simultaneous=False)
        truth = 0.06 * effect.x - float(np.mean(0.06 * x))
        step = np.abs(np.diff(effect.effect)) <= 1e-9 * np.ptp(effect.effect)
        held = np.zeros(effect.x.size, dtype=bool)
        held[:-1] |= step
        held[1:] |= step
        hits += int(np.sum(((truth >= effect.lower) & (truth <= effect.upper))[held]))
        points += int(np.sum(held))
    assert binding_fits >= 30, "the rule must bind in most fits for this to test anything"
    assert points >= 150, "the fits must hold enough of the grid flat"
    assert hits / points >= COVERAGE_FLOOR


def test_effective_degrees_of_freedom_are_those_of_the_face():
    X, y = _data("gamma")
    held = _model("gamma", INCREASING).fit_reml(X, y)
    state = _state(held)
    solver = state.solver_result
    covariance, curvature, _ = _face_covariance(held)
    influence = covariance @ np.asarray(solver.terminal_data_curvature)
    np.testing.assert_allclose(
        np.diag(influence), state.inference.coefficient_edf, rtol=1e-6, atol=1e-8
    )
    assert held.result_.total_effective_df == pytest.approx(np.trace(influence), rel=1e-8)
    free = _model("gamma").fit_reml(X, y)
    assert held.result_.total_effective_df < free.result_.total_effective_df


def test_diagnose_says_what_a_binding_rule_did_in_one_sentence_per_term():
    X, y = _data("gamma")
    held = _model("gamma", INCREASING).fit_reml(X, y)
    text = held.diagnose().render(detail="full")
    assert "The rule 'increasing' on mean:x binds at" in text
    assert "bands are those of the fit without the rule" in text


def test_diagnose_describes_the_covariance_a_saved_model_holds():
    """A record saved by 0.43.0 has no ``ShapeFit.covariance`` and holds the face's covariance,
    which it keeps until refit; Diagnose says so, and that a refit gives the new bands. Made
    here by deleting the stamp before saving. Against ``d4d43fed5`` the legacy record read
    "bands are those of the fit without the rule", the bands it does not have.
    """
    import pickle

    X, y = _data("gamma")
    for legacy in (False, True):
        held = _model("gamma", INCREASING).fit_reml(X, y)
        assert _state(held).shape_fit.binds
        if legacy:
            vars(_state(held).shape_fit).pop("covariance", None)
        text = pickle.loads(pickle.dumps(held)).diagnose().render(detail="full")
        assert "The rule 'increasing' on mean:x binds at" in text
        new = "bands are those of the fit without the rule" in text
        old = "standard errors treat those checkpoints as fixed" in text
        assert (new, old) == ((False, True) if legacy else (True, False))
        assert ("refit it to get the bands of the fit without the rule" in text) is legacy


def test_a_direction_and_a_curvature_rule_hold_together_and_match_the_optimizer():
    X, y = _data("gamma")
    rules = Constraint.fit.increasing & Constraint.fit.concave
    held = _model("gamma", rules).fit_reml(X, y)
    eta = _curve(held, "mean")
    assert np.min(np.diff(eta)) >= -1e-9
    assert np.max(np.diff(eta, 2)) <= 1e-7
    oracle_value, _ = _oracle(held, y)
    assert _objective(held) <= oracle_value + 1e-6 * (1.0 + abs(oracle_value))


def test_the_face_of_binding_rows_sits_inside_an_exact_penalty_face():
    """Binding rows hold a face inside the exact-infinity penalty face, which keeps its names."""
    from superglm.distributional.smoothing.penalty_face import build_penalty_face
    from superglm.distributional.solver.shape import _face_of

    X, y = _data("gamma")
    state = _state(_model("gamma", INCREASING).fit_reml(X, y))
    layout = state.layout
    base = build_penalty_face(layout, [layout.penalty_names[0]])
    rows = shape_rows(layout, state.compiled_predictors)
    anchor = np.asarray(state.solver_result.coefficients)
    held = rows.A[:2]
    face = _face_of(held, anchor, base, layout.coefficient_names)
    assert face.component_names == base.component_names
    basis = face.null_basis
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-12)
    assert np.max(np.abs(held @ basis)) <= 1e-12 * np.abs(held).max()
    assert np.max(np.abs(base.constraint_matrix @ basis)) <= 1e-12
    assert face.reduced_width < base.reduced_width
    # a step off the face is projected back through the anchor, not through the origin
    np.testing.assert_allclose(face.project(anchor), anchor, atol=1e-12)


def test_a_damped_qp_step_is_the_damped_newton_step_from_the_current_point(monkeypatch):
    """Damping shifts the QP target with the curvature, so the step is not pulled toward zero."""
    from superglm.distributional.results.solver import DenseSolverConfig
    from superglm.distributional.solver import shape
    from superglm.solvers.constrained_qp import _constraint_structure

    curvature = np.array([[2.0, 0.3], [0.3, 1.0]])
    anchor = np.array([1.5, -2.0])
    score = np.array([0.4, -0.1])
    rows, bound = np.array([[1.0, 0.0]]), np.array([-1.0e6])  # never binds
    solve = shape._solve_constrained_qp_with_cold_retry
    systems = []

    def damped_only(system, target, *args, **kwargs):
        # The undamped solve is refused, as an indefinite curvature's is, to force the damping.
        systems.append(system)
        if len(systems) == 1:
            raise np.linalg.LinAlgError("indefinite")
        return solve(system, target, *args, **kwargs)

    monkeypatch.setattr(shape, "_solve_constrained_qp_with_cold_retry", damped_only)
    found = shape._qp_step(
        curvature,
        curvature @ anchor + score,
        anchor,
        rows,
        bound,
        None,
        _constraint_structure(rows),
        DenseSolverConfig(initial_levenberg_shift=0.5),
    )
    damped = systems[1]
    assert not np.allclose(damped, curvature)
    np.testing.assert_allclose(damped @ (found.beta - anchor), score, rtol=0.0, atol=1e-10)


def test_a_refit_with_no_feasible_start_is_disclosed_as_the_fit_without_the_rules(monkeypatch):
    """The fit keeps no rule then: it must not report converged, nor call its curve the refit's."""
    from superglm.distributional.solver import shape

    monkeypatch.setattr(shape, "_MAX_PROJECTIONS", 0)  # no projection: the start stays infeasible
    X, y = _data("gamma")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held = _model("gamma", INCREASING).fit_reml(X, y)
    record = _state(held).shape_fit
    assert record.applied and not record.converged and record.unruled
    assert record.reason == "no_feasible_start"
    assert not held.result_.converged
    assert np.min(np.diff(_curve(held, "mean"))) < 0.0
    said = " ".join(str(w.message) for w in caught)
    assert "the fit without the rules" in said and "the last the refit reached" not in said
    diagnosed = held.diagnose().render(detail="full")
    assert "could not be applied" in diagnosed
    assert "held the fit in place nowhere" not in diagnosed and "refitted under it" not in diagnosed


def test_a_refinement_that_crosses_a_rule_steps_back_and_holds_it(monkeypatch):
    """Refining on the binding face must keep the rows that do not bind: one it crosses is held
    from where the move meets it, and the refinement runs again."""
    from superglm.distributional.solver import shape

    class FirstRefinementHoldsNoRow:
        # The first refinement runs off the face, as one that crosses a row would.
        def __init__(self, solver):
            self.solver, self.calls = solver, 0

        def __getattr__(self, name):
            return getattr(self.solver, name)

        def fit_dense_fixed_lambda(self, *args, coefficient_face=None, **kwargs):
            self.calls += 1
            face = None if self.calls == 1 else coefficient_face
            return self.solver.fit_dense_fixed_lambda(*args, coefficient_face=face, **kwargs)

    proxy = FirstRefinementHoldsNoRow(shape._solver)
    monkeypatch.setattr(shape, "_solver", proxy)
    X, y = _data("gamma")
    held = _model("gamma", INCREASING).fit_reml(X, y)
    record = _state(held).shape_fit
    assert proxy.calls >= 2
    assert record.applied and record.converged
    assert record.max_violation <= 1e-9
    assert np.min(np.diff(_curve(held, "mean"))) >= -1e-9


def test_the_feasible_start_reaches_the_rules_whatever_the_response_units():
    """An identity-link curve far from increasing is projected onto the rules in one move; the
    solver's cap on a predictor step (a few units) must not leave it unconstrained."""
    from superglm import GaussianLS, LambdaPolicy

    x = np.linspace(0.0, 1.0, 100)
    y = 100.0 * (2.0 - 4.0 * x + 0.1 * np.sin(np.arange(100)))
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(
            s("x", kind="cr", k=6, constraint=INCREASING, lambda_policy=LambdaPolicy.fixed(0.1))
        ),
        family.scale(),
    ).fit(pd.DataFrame({"x": x}), y)
    record = _state(model).shape_fit
    assert record.applied and record.converged and not record.unruled
    assert model.result_.converged
    location = model.predict_parameters(pd.DataFrame({"x": np.linspace(0, 1, 200)}))["location"]
    assert np.min(np.diff(location.to_numpy())) >= -1e-9 * np.abs(location).max()


def test_a_cap_on_a_link_that_adds_is_disclosed_and_dropped_keeping_the_shape():
    """A cap bounds a relativity; on an identity link a term adds, so the cap would bind a sum."""
    from superglm import GaussianLS, LambdaPolicy

    x = np.linspace(0.0, 1.0, 100)
    y = 2.0 + 4.0 * x + 0.1 * np.sin(np.arange(100))
    family = GaussianLS()
    with pytest.warns(ShapeConstraintIgnoredWarning, match=r"cap\(0\.9, 1\.1\) on location:x"):
        model = SuperLSS(
            family,
            family.location(
                s(
                    "x",
                    kind="cr",
                    k=6,
                    constraint=INCREASING & Constraint.fit.cap(0.9, 1.1),
                    lambda_policy=LambdaPolicy.fixed(0.1),
                )
            ),
            family.scale(),
        ).fit(pd.DataFrame({"x": x}), y)
    location = model.predict_parameters(pd.DataFrame({"x": x}))["location"].to_numpy()
    # The cap is gone (log(1.1) - log(0.9) is 0.2), the increasing rule kept.
    assert np.ptp(location) > 3.0
    assert np.min(np.diff(location)) >= -1e-9 * np.abs(location).max()


def test_the_rows_reported_as_binding_are_the_rows_the_face_holds():
    """Far from zero a row's own slack reads nothing as binding; the face's reading of the whole
    fit holds the flattened spline, and the record and Diagnose must say so."""
    from superglm import GaussianLS, LambdaPolicy

    x = np.linspace(0.0, 1.0, 100)
    y = 1e6 * (1000.0 + 2.0 - 4.0 * x + 0.1 * np.sin(np.arange(100)))
    family = GaussianLS()
    model = SuperLSS(
        family,
        family.location(
            s("x", kind="cr", k=6, constraint=INCREASING, lambda_policy=LambdaPolicy.fixed(0.1))
        ),
        family.scale(),
    ).fit(pd.DataFrame({"x": x}), y)
    record = _state(model).shape_fit
    assert record.applied and record.converged
    assert record.n_binding > 0 and record.binds
    assert "did not bind" not in model.diagnose().render(detail="full")
