"""Fit-time shape rules on SuperLSS terms (``solver/shape.py``).

The coefficients are refitted under the rules by sequential QP, and the smoothing is re-chosen
with the rows that bind held (``smoothing/shape_efs.py``). Each fit is checked against an
independent optimizer on the same penalized likelihood and rows, its smoothing against the
Laplace marginal likelihood on the rules' face, and its covariance against the null-space formula.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
from scipy import optimize

from superglm import Constraint, GammaLS, NegativeBinomialLS, SuperLSS, TweedieLSS, s
from superglm.distributional.family import COMPLETE_OBSERVATION
from superglm.distributional.predictor import ShapeConstraintIgnoredWarning
from superglm.distributional.solver.assembly import (
    _evaluate_predictors_from_matrices,
    dense_predictor_matrices,
)
from superglm.distributional.solver.shape import shape_rows
from superglm.distributional.weights import WeightContract, resolve_likelihood_weights

INCREASING = Constraint.fit.increasing
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

    # the smoothing was re-chosen with the rules held, starting from the one chosen without them
    assert shape.smoothing is not None and shape.smoothing.converged
    assert dict(shape.smoothing.lambdas_without_rules) == _state(free).lambdas
    assert held.smoothing_parameters_ != pytest.approx(free.smoothing_parameters_, rel=0.05)
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
    # the solver's stationarity test reads the score against the objective's scale
    scale = 1.0 + abs(float(state.solver_result.penalized_optimizing_log_likelihood))
    assert np.linalg.norm(residual) <= 1e-7 * scale
    free_direction = np.linalg.svd(rows.A[binding])[2][binding.size :].T
    assert np.max(np.abs(free_direction.T @ score)) <= 1e-7 * scale


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
    assert shape.smoothing is None and held.smoothing_parameters_ == free.smoothing_parameters_
    # no row binds, so there is nothing for the sentence to say but that
    assert "did not bind" in held.diagnose().render(detail="full")


def test_a_fit_time_rule_no_longer_warns_and_a_postfit_rule_still_does():
    X, y = _data("gamma")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShapeConstraintIgnoredWarning)
        _model("gamma", INCREASING).fit_reml(X, y)
    with pytest.warns(ShapeConstraintIgnoredWarning, match="postfit"):
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
def test_standard_errors_treat_the_binding_rows_as_fixed(kind):
    X, y = _data(kind)
    held = _model(kind, INCREASING).fit_reml(X, y)
    covariance, curvature, binding = _face_covariance(held)
    reported = np.asarray(held.result_.covariance)
    np.testing.assert_allclose(
        reported, covariance, rtol=1e-6, atol=1e-9 * np.abs(covariance).max()
    )
    # a binding row is a fixed combination of coefficients: it has no variance
    assert np.max(np.abs(binding @ reported @ binding.T)) <= 1e-9 * np.abs(reported).max() * (
        np.max(np.abs(binding)) ** 2
    )
    # and it is not the unconstrained standard error the term would have had
    free_se = np.sqrt(np.diag(np.linalg.pinv(curvature)))
    se = np.sqrt(np.maximum(np.diag(reported), 0.0))
    assert np.max(np.abs(se - free_se) / free_se) > 0.05


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
    assert "standard errors treat those checkpoints as fixed" in text


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


def test_the_rows_reported_as_binding_are_the_rows_the_standard_errors_hold():
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


def _face_laplace(model: SuperLSS, y: np.ndarray, log_lambda: float) -> float:
    """The Laplace marginal likelihood on the face of the binding rows at one smoothing, formed
    here: the constrained optimum by trust-constr, the rows its multipliers hold as equalities,
    a finite-difference Hessian, and the penalty projected onto the face.

    ``-f(b) + log|Z' S Z|_+ / 2 - log|Z' H Z| / 2 + (m - r) log(2 pi) / 2`` for ``f`` the
    penalized negative log-likelihood, ``Z`` (``m`` columns) the binding rows' null space and
    ``r`` the rank of ``Z' S Z``. The model must carry one smoothing parameter.
    """
    state = _state(model)
    layout, family = state.layout, model.family
    ((_, fitted),) = state.lambdas.items()
    unit = np.asarray(state.solver_result.penalty) / fitted
    rows = shape_rows(layout, state.compiled_predictors)
    matrices = dense_predictor_matrices(layout)
    weights = resolve_likelihood_weights(
        None, n_observations=len(y), contract=WeightContract("prior")
    )
    plan = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    links = [predictor.link for predictor in layout.predictors]
    penalty = np.exp(log_lambda) * unit

    def objective(beta: np.ndarray) -> float:
        eta = _evaluate_predictors_from_matrices(layout, beta, matrices)
        theta = np.column_stack([link.inverse(eta[:, j]) for j, link in enumerate(links)])
        natural = family.evaluate_natural(y, theta, plan, derivative_order=0)
        return -float(np.sum(natural.optimizing_log_likelihood)) + 0.5 * beta @ penalty @ beta

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        found = optimize.minimize(
            objective,
            np.asarray(state.solver_result.coefficients),
            method="trust-constr",
            constraints=[optimize.LinearConstraint(rows.A, rows.b, np.inf)],
            options={"maxiter": 3000, "gtol": 1e-11, "xtol": 1e-13},
        )
    beta = found.x
    # trust-constr is an interior method, so its active rows sit just inside; its multipliers say
    # which rows hold the optimum
    multipliers = np.abs(np.asarray(found.v[0]))
    held = rows.A[multipliers > 1e-4 * (1.0 + multipliers.max())]
    basis = np.eye(beta.size)
    if held.shape[0]:
        _, singular, vt = np.linalg.svd(held)
        basis = vt[int(np.sum(singular > 1e-10 * singular[0])) :].T
    step = 1e-4
    width = beta.size
    hessian = np.empty((width, width))
    for i in range(width):
        for j in range(i, width):
            di, dj = np.eye(width)[i] * step, np.eye(width)[j] * step
            hessian[i, j] = hessian[j, i] = (
                objective(beta + di + dj)
                - objective(beta + di - dj)
                - objective(beta - di + dj)
                + objective(beta - di - dj)
            ) / (4.0 * step * step)
    projected = np.linalg.eigvalsh(basis.T @ penalty @ basis)
    positive = projected[projected > 1e-9 * projected.max()]
    return float(
        -found.fun
        + 0.5 * np.sum(np.log(positive))
        - 0.5 * np.linalg.slogdet(basis.T @ hessian @ basis)[1]
        + 0.5 * (basis.shape[1] - positive.size) * np.log(2.0 * np.pi)
    )


def test_the_smoothing_chosen_with_the_rules_held_beats_the_one_chosen_without():
    """The rules' smoothing is the face's Laplace optimum, by an independent evaluation of it.

    Fellner--Schall neglects the curvature's change with the smoothing, so its fixed point is
    the optimum to that approximation: higher than the smoothing chosen without the rules and
    higher than a step either side of it. The same rows bind across this range; where the
    rows that bind change with the smoothing, the face's likelihood jumps there, and the update
    is stationary on the face it ends on.
    """
    X, y = _data("gamma")
    free = _model("gamma").fit_reml(X, y)
    held = _model("gamma", INCREASING).fit_reml(X, y)
    smoothing = _state(held).shape_fit.smoothing
    assert smoothing is not None and smoothing.converged and smoothing.reason == "stationary"
    assert smoothing.residual <= _state(free).smoothing.config.tolerance
    ((_, without),) = _state(free).lambdas.items()
    ((_, chosen),) = _state(held).lambdas.items()
    rho = np.log(chosen)
    at = _face_laplace(held, y, rho)
    assert at > _face_laplace(held, y, np.log(without)) + 0.1
    # a quarter of a log unit either side costs about 0.05 here; the evaluation is good to ~1e-4
    for side in (rho - 0.25, rho + 0.25):
        assert at >= _face_laplace(held, y, side) + 0.01


def test_only_the_ruled_terms_smoothing_moves():
    """A rule changes the model only in its own term's block; every other penalty keeps the
    smoothing the search chose, to the bit."""
    X, y = _data("gamma")
    z = np.random.default_rng(11).uniform(0.0, 1.0, len(y))
    X = X.assign(z=z)
    y = y * np.exp(0.3 * np.sin(6.0 * z))
    family = GammaLS()

    def model(rule):
        extra = {} if rule is None else {"constraint": rule}
        return SuperLSS(
            family, family.mean(s("x", k=8, **extra), s("z", k=6)), family.scale(s("z", k=5))
        )

    free = model(None).fit_reml(X, y)
    held = model(INCREASING).fit_reml(X, y)
    smoothing = _state(held).shape_fit.smoothing
    assert smoothing is not None and smoothing.converged
    assert smoothing.estimated == ("mean:x#wiggle",)
    for name, value in _state(free).lambdas.items():
        if name == "mean:x#wiggle":
            assert _state(held).lambdas[name] != pytest.approx(value, rel=0.05)
        else:
            assert _state(held).lambdas[name] == value


def test_a_cut_block_has_the_rank_of_its_penalty_on_the_face():
    """On the face, a single-penalty block's effective rank is the rank of the penalty there:
    the face's width in the block less the penalty-free directions the face keeps."""
    from superglm.distributional.smoothing.shape_efs import face_component_states

    X, y = _data("gamma")
    held = _model("gamma", INCREASING).fit_reml(X, y)
    state = _state(held)
    face = state.solver_result.coefficient_face
    (component,) = state.layout.penalties
    block = component.group_sl
    (cut,) = face_component_states(state.layout, state.lambdas, face, None)
    local = face.null_basis[block]
    left, singular, _ = np.linalg.svd(local, full_matrices=False)
    within = left[:, singular > 0.5]
    penalty = np.asarray(cut.penalty)
    expected = np.linalg.matrix_rank(within.T @ penalty @ within)
    assert cut.rank == pytest.approx(expected, abs=1e-9)
    # the penalty-free directions the face keeps: the null space of the penalty within it
    null = scipy.linalg.null_space(penalty)
    kept = within.shape[1] + null.shape[1] - np.linalg.matrix_rank(np.column_stack((within, null)))
    assert expected == within.shape[1] - kept < np.linalg.matrix_rank(penalty)


def test_a_fixed_smoothing_fit_keeps_its_smoothing():
    X, y = _data("gamma")
    free = _model("gamma").fit_reml(X, y)
    given = dict(_state(free).lambdas)
    held = _model("gamma", INCREASING).fit(X, y, lambdas=given)
    shape = _state(held).shape_fit
    assert shape.applied and shape.converged and shape.binds and shape.smoothing is None
    assert dict(_state(held).lambdas) == given


def test_a_smoothing_search_under_the_rules_that_stops_short_is_disclosed(monkeypatch):
    """The fit is returned, meets its rules, and says the smoothing did not settle."""
    from superglm.distributional.smoothing import shape_efs

    calls = []
    refit = shape_efs.refit_with_shape_rules

    def stalls_after_the_first(*args, **kwargs):
        result, record = refit(*args, **kwargs)
        calls.append(1)
        if len(calls) > 1:
            record = replace(record, converged=False, reason="max_iterations")
        return result, record

    monkeypatch.setattr(shape_efs, "refit_with_shape_rules", stalls_after_the_first)
    X, y = _data("gamma")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held = _model("gamma", INCREASING).fit_reml(X, y)
    smoothing = _state(held).shape_fit.smoothing
    assert smoothing is not None and not smoothing.converged
    assert smoothing.reason == "refit_not_converged" and smoothing.iterations >= 2
    assert np.min(np.diff(_curve(held, "mean"))) >= -1e-9
    said = " ".join(str(w.message) for w in caught)
    assert "did not settle the smoothing with the shape rules held" in said
    assert "did not settle" in held.diagnose().render(detail="full")


def test_a_smoothing_where_a_rule_starts_to_bind_is_bracketed_not_cycled():
    """Held, the row pulls the smoothing up; free, the search's update pulls it down. The
    smoothing is the boundary between them, reported on the side where the row binds."""
    from superglm import LomaxLSS

    rng = np.random.default_rng(6)
    x = rng.uniform(0.0, 10.0, 400)
    y = np.exp(_dip(x)) * rng.pareto(4.0, 400) * 3.0 + 1e-6
    X = pd.DataFrame({"x": x})
    family = LomaxLSS()

    def model():
        return SuperLSS(family, family.scale(s("x", k=8, constraint=INCREASING)), family.shape())

    held = model().fit_reml(X, y)
    shape = _state(held).shape_fit
    smoothing = shape.smoothing
    assert smoothing.converged and smoothing.reason == "face_boundary"
    assert smoothing.iterations < _state(held).smoothing.config.max_iterations
    assert shape.binds and held.result_.converged
    assert "where a checkpoint starts to bind" in held.diagnose().render(detail="full")
    # a little more smoothing and the rule is met without holding any row
    ((name, chosen),) = _state(held).lambdas.items()
    above = model().fit(X, y, lambdas={name: chosen * 1.02})
    assert not _state(above).shape_fit.binds
