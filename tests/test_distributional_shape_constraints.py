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
