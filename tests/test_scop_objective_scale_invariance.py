"""SCOP admission uses represented quadratic actions, not absolute units."""

import functools
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import Constraint, ConvergenceWarning, Spline, SuperGLM
from superglm._lambda_box import LAMBDA_BOX, LAMBDA_CEILING, LAMBDA_FLOOR, latent_lambda_box
from superglm._roundoff import TINY
from superglm.diagnostics.convergence import _REML_CONVERGED_REASONS
from superglm.distributions import Gaussian
from superglm.links import IdentityLink
from superglm.model.reml_execute import STEP_REML_TOL_DEFAULT
from superglm.reml.scop_efs import _SCOP_SUPPRESSION_EDF
from superglm.solvers.irls_state import _irls_objective_relative_change
from superglm.solvers.scop import build_scop_solver_reparam
from superglm.solvers.scop_newton import (
    SCOPNewtonResult,
    scop_joint_newton_step,
    scop_newton_step,
)
from superglm.types import GroupSlice


@pytest.mark.parametrize("scale", [1e-24, 1e-20, 1.0, 1e20, 1e24])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("penalty", [0.0, 0.5])
def test_scop_step_decreases_the_scaled_quadratic(scale, joint, discrete, penalty):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    group_count = 2 if joint else 1
    design = np.kron(np.eye(group_count), np.array([[-1.0], [1.0]]))
    response = np.tile([-10.0, 10.0], group_count)
    weights = np.full(len(response), scale)
    bin_idx = np.arange(len(response)) if discrete else None
    if joint:
        groups = [GroupSlice(str(i), i, i + 1) for i in range(group_count)]
        states = {
            i: {
                "B_scop": design[:, i : i + 1],
                "S_scop": np.eye(1),
                "beta_scop": np.zeros(1),
                "reparam": reparam,
                "bin_idx": bin_idx,
            }
            for i in range(group_count)
        }
        results = scop_joint_newton_step(states, weights, response, scale * penalty, groups)
        beta = np.concatenate([results[i].beta_new for i in range(group_count)])
        reported = results[0]
    else:
        reported = scop_newton_step(
            design,
            weights,
            response,
            np.zeros(1),
            reparam,
            np.eye(1),
            scale * penalty,
            bin_idx=bin_idx,
        )
        beta = reported.beta_new

    # The well-conditioned scalar Fisher direction is 18/(2+penalty).
    # Two halvings provide strict descent for both penalty choices.
    expected = np.full(group_count, 4.5 / (2.0 + penalty))
    allowance = 128 * np.finfo(float).eps
    np.testing.assert_allclose(beta, expected, rtol=allowance, atol=allowance)
    normalized_objective = np.sum((10.0 - np.exp(beta)) ** 2 + 0.5 * penalty * beta**2)
    before = 81.0 * group_count
    assert normalized_objective < before
    assert reported.objective_after / scale == pytest.approx(
        normalized_objective, rel=allowance, abs=allowance * before
    )


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
@pytest.mark.parametrize("joint", [False, True])
def test_exhausted_scop_line_search_retains_the_committed_state(scale, joint):
    reparam = build_scop_solver_reparam(q_raw=2, kind="increasing")
    design = np.array([[-1.0], [1.0]])
    if joint:
        result = scop_joint_newton_step(
            {
                0: {
                    "B_scop": design,
                    "S_scop": np.zeros((1, 1)),
                    "beta_scop": np.zeros(1),
                    "reparam": reparam,
                    "bin_idx": None,
                }
            },
            np.full(2, scale),
            np.array([-10.0, 10.0]),
            0.0,
            [GroupSlice("shape", 0, 1)],
            max_halving=0,
        )[0]
    else:
        result = scop_newton_step(
            design,
            np.full(2, scale),
            np.array([-10.0, 10.0]),
            np.zeros(1),
            reparam,
            np.zeros((1, 1)),
            0.0,
            max_halving=0,
        )
    np.testing.assert_array_equal(result.beta_new, np.zeros(1))
    assert result.step_norm == 0.0
    assert result.objective_after == result.objective_before


@pytest.mark.parametrize("scale", [1e-24, 1.0, 1e24])
@pytest.mark.parametrize("force_uphill_inner", [False, True])
def test_actual_scop_outer_boundary_uses_weighted_objective_units(
    monkeypatch, scale, force_uphill_inner
):
    import superglm.solvers.irls_direct as irls_direct

    if force_uphill_inner:

        def propose_uphill(**kwargs):
            return SCOPNewtonResult(
                beta_new=np.array([9.0]),
                objective_before=81.0 * scale,
                objective_after=scale * (10.0 - np.exp(9.0)) ** 2,
                step_norm=9.0,
                used_fisher_fallback=True,
            )

        monkeypatch.setattr(irls_direct, "scop_newton_step", propose_uphill)
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
        _scop_joint=False,
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
    assert result.iteration_log[0].step_halvings == (2 if force_uphill_inner else 0)


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


@pytest.mark.parametrize("units", [1e-100, 1e-3, 1.0, 1e5, 1e100])
def test_latent_lambda_box_widens_by_the_squared_increments(units):
    """Each end of the box follows the squared increments past one; the box never narrows."""
    latent = np.log(np.array([0.5, 2.0, 3.0]))
    floor, ceiling = latent_lambda_box(latent + math.log(units))
    mean_square = units**2 * float(np.mean(np.exp(2.0 * latent)))
    # exp of a log-domain sum of a few terms: its error is u times their magnitude.
    terms = math.log(LAMBDA_CEILING) + 2.0 * abs(math.log(units)) + 2.0
    rel = 8 * np.finfo(float).eps * terms
    assert floor == pytest.approx(LAMBDA_FLOOR * min(mean_square, 1.0), rel=rel, abs=0.0)
    assert ceiling == pytest.approx(LAMBDA_CEILING * max(mean_square, 1.0), rel=rel, abs=0.0)


def test_latent_lambda_box_stays_in_the_positive_normal_range():
    assert latent_lambda_box(np.zeros(3)) == LAMBDA_BOX
    assert latent_lambda_box(np.array([0.0, np.nan])) == LAMBDA_BOX
    assert latent_lambda_box(np.full(3, 400.0)) == (LAMBDA_FLOOR, np.finfo(np.float64).max)
    assert latent_lambda_box(np.full(3, -400.0)) == (TINY, LAMBDA_CEILING)


def _fit(units, truth):
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
        model.fit_reml(frame, units * y)
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
    curve 3.8 standard errors away), and at 1e9 the penalty quadratic, the
    difference of two quadratics near 6e31, came out negative and the fit
    raised.
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
    (EDF 12.45 against 16.01, the curve 4.3 standard errors away).  Each fit
    stops once its last step moves log lambda by less than ``reml_tol``, so
    the two agree to ``2 reml_tol`` in log lambda.  To first order EDF then
    moves by at most EDF times that, and the curve by
    ``sum_j sqrt(rEDF_j) |d log lambda_j|`` standard errors, at most
    ``sqrt(n_lambda p)`` times it.
    """
    units = 1e-4
    reference = _fit_once(1.0, "exp")
    scaled = _fit_once(units, "exp")
    tolerance = 2.0 * STEP_REML_TOL_DEFAULT
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

