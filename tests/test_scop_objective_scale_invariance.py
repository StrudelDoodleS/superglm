"""SCOP admission uses represented quadratic actions, not absolute units."""

import functools
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Categorical, Constraint, ConvergenceWarning, PSpline, Spline, SuperGLM
from superglm._lambda_box import LAMBDA_BOX, LAMBDA_CEILING, LAMBDA_FLOOR, scaled_lambda_box
from superglm._roundoff import TINY
from superglm.diagnostics.convergence import _REML_CONVERGED_REASONS
from superglm.distributions import Binomial, Gamma, Gaussian, Poisson, Tweedie
from superglm.group_matrix import DesignMatrix
from superglm.links import IdentityLink, LogLink
from superglm.reml.scop_efs import (
    _SCOP_EFS_PLATEAU_REMAINING_MOVEMENT,
    _SCOP_LINE_SEARCH_RELATIVE_TOLERANCE,
    _SCOP_SUPPRESSION_EDF,
    _scop_lambda_scale,
)
from superglm.solvers.centered_system import build_centered_system
from superglm.solvers.irls_state import _irls_objective_relative_change
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import SCOPGroupSpec, SCOPJointNewton
from superglm.types import GroupSlice


@pytest.mark.parametrize("scale", [1e-24, 1e-20, 1.0, 1e20, 1e24])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("penalty", [0.0, 0.5])
def test_scop_step_is_invariant_to_the_objective_scale(scale, groups, discrete, penalty):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    design = np.kron(np.eye(groups), np.array([[-1.0], [1.0]]))
    response = np.tile([-10.0, 10.0], groups)
    weights = np.full(len(response), scale)
    n = len(response)
    bin_idx = np.arange(n) if discrete else None
    specs = [
        SCOPGroupSpec(
            group_index=i,
            group=GroupSlice(str(i), i, i + 1),
            reparam=reparam,
            B_scop=design[:, i : i + 1],
            S_scop=np.eye(1),
            bin_idx=bin_idx,
        )
        for i in range(groups)
    ]
    empty = DesignMatrix([], n=n, p=0)
    engine = SCOPJointNewton(specs, scale * penalty, empty, n)
    thetas = [np.zeros(1)] * groups
    reduced = build_centered_system(
        dm=empty, W=weights, z_off=response - engine.predictor(thetas), penalty=np.zeros((0, 0))
    )
    step = engine.step(
        thetas,
        W=weights,
        z=response,
        eta=engine.predictor(thetas),
        reduced=reduced,
        solve=lambda column: column,
        ordinary_change=np.zeros(0),
    )

    # The map curvature makes the block indefinite, and the Gauss-Newton
    # direction is 18 / (2 + penalty) in every coordinate at every scale.
    beta = np.concatenate([step.results[i].beta_new for i in range(groups)])
    assert all(result.used_fisher_fallback for result in step.results.values())
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(beta, np.full(groups, 18.0 / (2.0 + penalty)), rtol=allowance)


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
def test_actual_scop_outer_boundary_uses_weighted_objective_units(scale):
    """The Newton proposal (beta_eff 9) overshoots, and the outer search halves it twice."""
    import superglm.solvers.irls_direct as irls_direct

    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    result, _, states = irls_direct.fit_irls_direct(
        np.array([[-1.0], [1.0]]),
        np.array([-10.0, 10.0]),
        np.full(2, scale),
        Gaussian(),
        IdentityLink(),
        [GroupSlice("shape", 0, 1, monotone_engine="scop", scop_reparameterization=reparam)],
        lambda2=0.0,
        beta_init=np.ones(1),
        intercept_init=0.0,
        max_iter=1,
        S_override=np.zeros((1, 1)),
        return_scop_state=True,
        scop_state_init={0: {"beta_eff": np.zeros(1), "S_scop": np.zeros((1, 1))}},
        record_diagnostics=True,
        _compute_scop_postfit_inference=False,
        weight_semantics="frequency",
    )
    assert not result.converged
    assert result.n_iter == 1
    assert result.deviance / scale < 162.0
    expected_change = abs(result.deviance - 162.0 * scale) / (162.0 * scale)
    assert result.iteration_log[0].convergence_value == pytest.approx(
        expected_change, rel=16 * np.finfo(float).eps, abs=0.0
    )
    np.testing.assert_allclose(states[0]["beta_eff"], [2.25], rtol=128 * np.finfo(float).eps)
    assert result.iteration_log[0].step_halvings == 2


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_objective_change_ratio_preserves_finite_extreme_units(scale):
    ratio = _irls_objective_relative_change(
        objective=0.9 * scale, previous=scale, objective_scale=scale
    )
    assert ratio == pytest.approx(0.05, rel=16 * np.finfo(float).eps, abs=0.0)


def test_objective_change_ratio_has_explicit_zero_action_semantics():
    assert _irls_objective_relative_change(objective=0.0, previous=0.0, objective_scale=0.0) == 0.0
    assert np.isinf(
        _irls_objective_relative_change(objective=1.0, previous=0.0, objective_scale=0.0)
    )


@pytest.mark.parametrize("scale", [1e-100, 1e-3, 1.0, 1e5, 1e100])
def test_scaled_lambda_box_widens_by_the_scale(scale):
    """Each end of the box follows the scale past one; the box never narrows."""
    floor, ceiling = scaled_lambda_box(scale)
    # Each end is one correctly rounded product.
    assert floor == LAMBDA_FLOOR * min(scale, 1.0)
    assert ceiling == LAMBDA_CEILING * max(scale, 1.0)


def test_scaled_lambda_box_stays_in_the_positive_normal_range():
    for scale in (0.0, -1.0, np.nan, np.inf):
        assert scaled_lambda_box(scale) == LAMBDA_BOX
    assert scaled_lambda_box(1e300) == (LAMBDA_FLOOR, np.finfo(np.float64).max)
    assert scaled_lambda_box(1e-310) == (TINY, LAMBDA_CEILING)


@pytest.mark.parametrize(
    ("family", "power"), [(Gaussian(), 0.0), (Tweedie(1.5), 1.5), (Gamma(), 2.0)]
)
@pytest.mark.parametrize("weight_semantics", ["prior", "frequency"])
def test_scop_box_scale_carries_the_dispersions_units(family, power, weight_semantics):
    """Under the identity link y -> c y moves the scale by c**(2 - p), as it moves the dispersion.

    Each scale is a few sums of n positive terms and one power, so each is
    within (n + 4) u of its exact value and their ratio within twice that.
    """
    rng = np.random.default_rng(530)
    n = 200
    y, weights = rng.gamma(2.0, 1.5, n), rng.uniform(0.5, 2.0, n)

    def scale(response):
        return _scop_lambda_scale(
            family, IdentityLink(), response, weights, weight_semantics=weight_semantics
        )

    reference = scale(y)
    assert np.isfinite(reference) and reference > 0.0
    for units in (1e-3, 1e4):
        expected = units ** (2.0 - power)
        assert scale(units * y) / reference == pytest.approx(
            expected, rel=4 * (n + 4) * np.finfo(float).eps, abs=0.0
        )


@pytest.mark.parametrize(
    ("family", "link"),
    [(Poisson(), IdentityLink()), (Binomial(), IdentityLink()), (Gaussian(), LogLink())],
)
def test_scop_box_scale_is_one_without_units_to_follow(family, link):
    """A fixed dispersion has no units; under a log link every term's lambda carries them."""
    y = np.random.default_rng(530).uniform(0.5, 2.0, 50)
    assert _scop_lambda_scale(family, link, 1e4 * y, np.ones(50), weight_semantics="prior") == 1.0


def _fit(units, truth, lambda2_init=None):
    """A Gaussian/identity fit of ``units * y`` whose monotone x2 truth is linear or exp.

    The fit must converge: no ``ConvergenceWarning`` and a converged reason.
    """
    rng = np.random.default_rng(20261024)
    n = 3000
    frame = pd.DataFrame({"x1": rng.uniform(0.0, 1.0, n), "x2": rng.uniform(0.0, 1.0, n)})
    x2 = frame["x2"].to_numpy()
    monotone = 0.8 * x2 if truth == "linear" else 2.0 * np.exp(2.0 * x2)
    y = np.sin(2 * np.pi * frame["x1"].to_numpy()) + monotone + 0.3 * rng.normal(size=n)
    model = SuperGLM(
        family="gaussian",
        selection_penalty=0.0,
        features={
            "x1": Spline(kind="ps", k=10),
            "x2": Spline(kind="ps", k=10, constraint=Constraint.fit.increasing),
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit_reml(frame, units * y, lambda2_init=lambda2_init)
    assert not [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    diagnostics = model.reml_diagnostics()
    assert diagnostics["converged"], diagnostics["termination_reason"]
    assert diagnostics["termination_reason"] in _REML_CONVERGED_REASONS
    return model


_fit_once = functools.cache(_fit)


def _x2_curve(model, units):
    """The x2 term on the link scale and its pointwise standard error, divided by ``units``."""
    with np.errstate(over="ignore"):  # an identity link's relativity is exp of the curve
        inference = model.term_inference("x2")
    curve = np.asarray(inference.log_relativity, dtype=float)
    return curve / units, np.asarray(inference.se_log_relativity, dtype=float) / units


@pytest.mark.parametrize("units", [1e5, 1e9])
def test_scop_reml_fit_at_large_units_matches_the_unit_fit(units):
    """Rescaling y rescales the fit; the SCOP smoothing parameter scales with it.

    With a linear truth the x2 penalty climbs until its residual EDF falls
    under ``_SCOP_SUPPRESSION_EDF``, where the increase is held, so each fit
    stops within that EDF of the infinite-lambda limit and the two agree to
    it.  To first order the curve moves by at most ``sqrt(rEDF)`` standard
    errors per unit of log lambda; integrated from the bar to infinite
    lambda that is ``2 sqrt(bar)`` per fit.  The unit box's 1e10 ceiling
    held the 1e5 fit at 1e-3 of its optimum (EDF 12.07 against 9.71, the
    curve 3.8 standard errors away), and the 1e9 fit further from it.  With
    the ceiling lifted, a penalty quadratic formed as the difference of two
    quadratics near 6e31 came out negative at 1e9 and the fit raised.
    """
    reference = _fit_once(1.0, "linear")
    scaled = _fit_once(units, "linear")
    assert scaled.reml_diagnostics()["lambdas"]["x2"] > LAMBDA_CEILING
    edf = reference.result.effective_df
    assert abs(scaled.result.effective_df - edf) <= _SCOP_SUPPRESSION_EDF
    curve, _ = _x2_curve(scaled, units)
    reference_curve, standard_error = _x2_curve(reference, 1.0)
    shift = np.max(np.abs(curve - reference_curve) / standard_error)
    assert shift <= 4.0 * math.sqrt(_SCOP_SUPPRESSION_EDF)


def test_scop_reml_fit_at_small_units_matches_the_unit_fit():
    """The SCOP smoothing parameter falls with the squared units, below the unit box's floor.

    The truth has curvature, so the optimum is interior: lambda 1.09 at unit
    scale and 1.09e-8 at y * 1e-4, where the unit box's 1e-6 floor held it
    (EDF 12.45 against 16.01, the curve 4.3 standard errors away).  A
    converged fit stops within ``_SCOP_EFS_PLATEAU_REMAINING_MOVEMENT`` of
    its fixed point in log lambda: the plateau exit's bound on the movement
    left, the loosest converged exit (a full Newton step under ``reml_tol``
    leaves less than ``reml_tol``), so the two fits agree to twice that
    whichever exits they take.  To first order EDF then
    moves by at most EDF times that, and the curve by
    ``sum_j sqrt(rEDF_j) |d log lambda_j|`` standard errors, at most
    ``sqrt(n_lambda p)`` times it.
    """
    units = 1e-4
    reference = _fit_once(1.0, "exp")
    scaled = _fit_once(units, "exp")
    tolerance = 2.0 * _SCOP_EFS_PLATEAU_REMAINING_MOVEMENT
    lambdas = scaled.reml_diagnostics()["lambdas"]
    reference_lambdas = reference.reml_diagnostics()["lambdas"]
    assert lambdas["x2"] < LAMBDA_FLOOR
    assert abs(math.log(lambdas["x2"] / units**2 / reference_lambdas["x2"])) <= tolerance
    assert abs(math.log(lambdas["x1"] / reference_lambdas["x1"])) <= tolerance
    edf = reference.result.effective_df
    assert abs(scaled.result.effective_df - edf) <= edf * tolerance
    curve, _ = _x2_curve(scaled, units)
    reference_curve, standard_error = _x2_curve(reference, 1.0)
    shift = np.max(np.abs(curve - reference_curve) / standard_error)
    assert shift <= math.sqrt(len(lambdas) * reference.result.beta.size) * tolerance


@pytest.mark.parametrize("units", [1e5, 1e9])
def test_scop_warm_start_keeps_a_lambda_above_the_unit_ceiling(units):
    """A refit from a large-units fit's estimates starts there and returns that fit.

    It starts at the fit's own smoothing parameters, not at the unit box's
    ceiling, so the two stop near one fixed point and agree to the bars the
    small-units test derives.  At 1e9 the warm x2 lambda, applied unbounded
    to the mapped coordinates of the QP initialization, put a data direction
    of its Hessian under the rank cut and the refit raised.
    """
    reference = _fit_once(units, "linear")
    warm = reference.reml_diagnostics()["lambdas"]
    refit = _fit(units, "linear", lambda2_init=warm)
    assert warm["x2"] > LAMBDA_CEILING
    assert refit.reml_diagnostics()["lambda_history"][0] == warm
    tolerance = 2.0 * _SCOP_EFS_PLATEAU_REMAINING_MOVEMENT
    lambdas = refit.reml_diagnostics()["lambdas"]
    assert max(abs(math.log(lambdas[name] / warm[name])) for name in warm) <= tolerance
    edf = reference.result.effective_df
    assert abs(refit.result.effective_df - edf) <= edf * tolerance
    curve, _ = _x2_curve(refit, units)
    reference_curve, standard_error = _x2_curve(reference, units)
    shift = np.max(np.abs(curve - reference_curve) / standard_error)
    assert shift <= math.sqrt(len(lambdas) * reference.result.beta.size) * tolerance


@functools.cache
def _sparse_binomial_fit(start):
    """The release differential's sparse log-link binomial SCOP case, from ``start``.

    ``"cold"``; ``"warm"``, at x1 lambda 1 and x2 lambda 3000; or
    ``"refit"``, from the cold fit's own estimates.
    """
    rng = np.random.default_rng(1190)
    n, total = 4000, 6000
    x1, x2 = rng.uniform(0, 10, total), rng.uniform(-2, 2, total)
    levels = [f"a{i}" for i in range(6)]
    c1 = rng.choice(levels, total, p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05])
    c2 = rng.choice([f"g{i:02d}" for i in range(25)], total)
    lin = rng.normal(0, 1, total)
    rng.uniform(0, 1, (2, total))  # two columns this model does not use
    c1_effect = dict(zip(levels, [0, 0.2, -0.15, 0.3, -0.1, 0.25], strict=True))
    c2_effect = dict(zip(sorted(set(c2)), rng.normal(0, 0.15, 25), strict=False))
    eta = (
        0.06 * x1
        + 0.3 * np.sin(1.3 * x2)
        + np.array([c1_effect[v] for v in c1])
        + np.array([c2_effect[v] for v in c2])
        + 0.1 * lin
        + 0.04 * x1 * np.tanh(x2)
    )
    probability = np.clip(0.15 * np.exp(0.5 * (eta - eta.mean()) - 1.5), 0, 0.9)
    y = (rng.uniform(size=total) < probability).astype(float)
    starts = {
        "cold": lambda: None,
        "warm": lambda: {"x1": 1.0, "x2": 3000.0},
        "refit": lambda: _sparse_binomial_fit("cold").reml_diagnostics()["lambdas"],
    }
    model = SuperGLM(
        family="binomial",
        link="log",
        features={
            "x1": PSpline(n_knots=8, constraint=Constraint.fit.increasing),
            "x2": Spline(kind="cr", k=8),
            "c1": Categorical(),
        },
    )
    frame = pd.DataFrame({"x1": x1, "x2": x2, "c1": c1})
    model.fit_reml(frame.iloc[:n], y[:n], lambda2_init=starts[start]())
    return model


@pytest.mark.parametrize("start", ["cold", "warm", "refit"])
def test_a_flat_scop_term_publishes_one_fit_from_every_start(start):
    """A log-link SCOP block keeps the unit box, and every start publishes the same fit.

    The data leave the increasing x1 term flat: every increment is held at
    its log-space boundary, so x1 adds nothing to the fit and its lambda
    moves no fitted value.  From a cold start, a warm one and a refit from
    its own estimates, the fit converges, publishes a criterion no higher
    than the lowest it accepted (up to the line search's acceptance slack),
    and gives the cold fit's EDF within the plateau's remaining movement.
    x1's lambda only has to stay inside the unit box.  A box read from the
    increments followed them down (1e-6 times 7e-3 cold, 3e-24 warm): below
    about 1e-8 a Newton line search failed, the run restarted from the
    bootstrap on EFS steps and published a criterion as much as 9.6 above
    the lowest it had accepted.
    """
    model = _sparse_binomial_fit(start)
    diagnostics = model.reml_diagnostics()
    result = model._reml_result
    assert diagnostics["termination_reason"] in _REML_CONVERGED_REASONS
    history = result.objective_history
    slack = _SCOP_LINE_SEARCH_RELATIVE_TOLERANCE * max(abs(min(history)), 1.0)
    assert diagnostics["objective"] <= min(history) + slack
    assert LAMBDA_FLOOR <= diagnostics["lambdas"]["x1"] <= LAMBDA_CEILING
    tolerance = 2.0 * _SCOP_EFS_PLATEAU_REMAINING_MOVEMENT
    edf = _sparse_binomial_fit("cold").result.effective_df
    assert abs(model.result.effective_df - edf) <= edf * tolerance
