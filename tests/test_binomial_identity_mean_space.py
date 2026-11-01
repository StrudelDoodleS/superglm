"""Identity-binomial means must remain in the fitted affine probability model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, minimize
from scipy.special import xlogy

from superglm import Numeric, SuperGLM
from superglm.distributions import Binomial
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.irls_state import (
    mean_space_log_likelihood_rows,
    mean_space_newton_rows,
    mean_space_score_rows,
)
from superglm.solvers.mode_score import (
    MODE_CERTIFICATION_BAR,
    linear_predictor,
    truncated_direction_ratio,
)
from superglm.types import GroupSlice, LinearConstraintSet


def _assert_prediction_matches_canceling_affine_state(model, X, offset, means):
    # With nearly aliased columns the large individual contributions cancel.
    # Prediction and fitted-state accumulation may order their sums differently;
    # compare at the contributions' float64 scale, plus the probability clip band.
    magnitude = np.abs(model.result.centred_intercept) + np.abs(offset)
    magnitude += np.sum(
        np.abs((X.to_numpy() - model.result.state_center) * model.result.beta), axis=1
    )
    tolerance = 1e-7 + 8 * np.finfo(float).eps * magnitude
    prediction = model.predict(X, offset=offset)
    assert np.all(np.abs(prediction - means) <= tolerance), (prediction, means, tolerance)


def _reference(X, y, weights, offset, *, start=None):
    """Independent concave likelihood optimization with linear probability bounds."""
    design = np.column_stack([np.ones(len(X)), np.asarray(X)])
    margin = 1e-10

    def objective(beta):
        mu = design @ beta + offset
        if np.any((mu <= 0.0) | (mu >= 1.0)):
            return np.inf
        return -np.sum(weights * (xlogy(y, mu) + xlogy(1 - y, 1 - mu)))

    def gradient(beta):
        mu = design @ beta + offset
        return design.T @ (weights * ((1 - y) / (1 - mu) - y / mu))

    if start is None:
        start = np.linalg.lstsq(design, 0.5 - offset, rcond=None)[0]
    result = minimize(
        objective,
        start,
        jac=gradient,
        constraints=LinearConstraint(design, margin - offset, 1 - margin - offset),
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200},
    )
    assert result.success, result.message
    return design @ result.x + offset, result.fun


@pytest.mark.parametrize("direct_solve", ["auto", "qr"])
def test_boundary_fit_preserves_affine_means_and_discloses_boundary(direct_solve):
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 4)})
    y = np.array([0.0, 0.0, 1.0, 1.0])
    expected, expected_nll = _reference(X, y, np.ones(4), np.zeros(4))
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=0,
        direct_solve=direct_solve,
        features={"x": Numeric()},
    )
    model.fit(X, y)
    raw = model._dm.matvec(model.result.beta) + model.result.intercept
    predicted = model.predict(X)
    assert np.all((raw > 0.0) & (raw < 1.0)), raw
    np.testing.assert_allclose(predicted, expected, atol=3e-7, rtol=0.0)
    # A boundary clip may move a row by its probability floor, never by 0.1.
    np.testing.assert_allclose(predicted, raw, atol=1e-7, rtol=0.0)
    nll = -np.sum(xlogy(y, raw) + xlogy(1 - y, 1 - raw))
    assert abs(nll - expected_nll) < 1e-6
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"
    assert model.result.mean_space_boundary_rows == 2


@pytest.mark.parametrize(
    "offset_shift, offset_slope", [(0.0, 0.0), (3.0, 0.0), (-3.0, 0.0), (3.0, 2.0)]
)
def test_interior_identity_fit_matches_weighted_offset_likelihood(offset_shift, offset_slope):
    rng = np.random.default_rng(431)
    X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 120)})
    offset = offset_shift + offset_slope * X.x.to_numpy() + 0.04 * np.sin(3.0 * X.x.to_numpy())
    y = rng.binomial(1, 0.45 + 0.15 * X.x.to_numpy()).astype(float)
    weights = rng.integers(1, 5, len(y)).astype(float)
    expected, expected_nll = _reference(X, y, weights, offset)
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=0,
        weight_semantics="frequency",
        features={"x": Numeric()},
    ).fit(X, y, sample_weight=weights, offset=offset, tol=1e-10)
    raw = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    assert model.result.converged
    np.testing.assert_allclose(raw, expected, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(model.predict(X, offset=offset), raw, rtol=0.0, atol=2e-15)
    nll = -np.sum(weights * (xlogy(y, raw) + xlogy(1 - y, 1 - raw)))
    assert abs(nll - expected_nll) < 1e-8


@pytest.mark.parametrize("direct_solve", ["auto", "qr"])
def test_identity_start_restores_feasibility_when_least_squares_projection_does_not(direct_solve):
    X = pd.DataFrame({"x": np.arange(5) / 4})
    feasible = np.array([0.01, 0.01, 0.01, 0.99, 0.01])
    offset = 3.0 + 2.0 * X.x.to_numpy() + feasible
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    expected, expected_nll = _reference(X, y, np.ones(5), offset, start=[-3.0, -2.0])
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=0,
        direct_solve=direct_solve,
        features={"x": Numeric()},
    ).fit(X, y, offset=offset, tol=1e-10, max_iter=40)
    raw = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    assert np.all((raw >= 0.0) & (raw <= 1.0)), raw
    # The fitted coefficients must describe probabilities without clipping
    # an invalid affine predictor into the probability interval.
    np.testing.assert_allclose(model.predict(X, offset=offset), raw, atol=1e-7, rtol=0.0)
    nll = -np.sum(xlogy(y, raw) + xlogy(1 - y, 1 - raw))
    feasible_nll = -np.sum(xlogy(y, feasible) + xlogy(1 - y, 1 - feasible))
    assert expected_nll - 1e-7 <= nll <= feasible_nll + 1e-7
    if model.result.converged:
        np.testing.assert_allclose(raw, expected, atol=2e-6, rtol=0.0)
    else:
        assert model.result.termination_reason in {"mean_space_boundary", "max_iter"}


def test_identity_start_ignores_zero_weight_rows_when_restoring_offset_feasibility():
    X = pd.DataFrame({"x": np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.0])})
    offset = 3.0 + 2.0 * X.x.to_numpy() + np.array([0.01, 0.01, 0.01, 0.99, 0.01, 100.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    ).fit(X, np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]), sample_weight=weights, offset=offset)
    raw = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    assert np.all((raw[:5] >= 0.0) & (raw[:5] <= 1.0)), raw


@pytest.mark.parametrize(
    "translation, spread", [(1e12, 1.0), (1e16, 8.0), (0.0, 1e-12), (0.0, 1e12)]
)
def test_identity_start_restoration_uses_the_retained_centered_coordinates(translation, spread):
    x = np.arange(5) / 4
    offset = 3.0 + 2.0 * x + np.array([0.01, 0.01, 0.01, 0.99, 0.01])
    X = pd.DataFrame({"x": translation + spread * x})
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    ).fit(X, np.array([0.0, 0.0, 0.0, 0.0, 1.0]), offset=offset, max_iter=40)
    # Raw intercept coordinates lose the small probability residual next to
    # the translated column. The fit and prediction use the centered state.
    means = linear_predictor(model._dm, model.result, offset)
    assert np.all((means >= 0.0) & (means <= 1.0)), means
    np.testing.assert_allclose(model.predict(X, offset=offset), means, atol=1e-7, rtol=0.0)
    _, expected_nll = _reference(
        pd.DataFrame({"x": x}),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        np.ones(5),
        offset,
        start=[-3.0, -2.0],
    )
    nll = -np.sum(xlogy(np.array([0.0, 0.0, 0.0, 0.0, 1.0]), means))
    nll -= np.sum(xlogy(np.array([1.0, 1.0, 1.0, 1.0, 0.0]), 1.0 - means))
    feasible_nll = -3 * np.log(0.99) - 2 * np.log(0.01)
    assert expected_nll - 1e-7 <= nll <= feasible_nll + 1e-7
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"


def test_identity_closed_only_feasible_start_is_retained_with_boundary_disclosure():
    X = pd.DataFrame({"x": np.zeros(3)})
    offset = np.array([0.0, 1.0, 0.5])
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    ).fit(X, np.array([0.0, 1.0, 0.0]), offset=offset)
    raw = model._dm.matvec(model.result.beta) + model.result.intercept + offset
    np.testing.assert_array_equal(raw, offset)
    np.testing.assert_allclose(model.predict(X, offset=offset), raw, atol=1e-7, rtol=0.0)
    assert not model.result.converged
    assert model.result.termination_reason == "mean_space_boundary"


def test_identity_closed_only_intercept_model_retains_feasible_offset_means():
    offset = np.array([0.0, 1.0, 0.5])
    result, _ = fit_irls_direct(
        X=np.empty((3, 0)),
        y=np.array([0.0, 1.0, 0.0]),
        weights=np.ones(3),
        family=Binomial(),
        link=IdentityLink(),
        groups=[],
        lambda2=0.0,
        offset=offset,
        weight_semantics="frequency",
    )
    np.testing.assert_array_equal(result.intercept + offset, offset)
    assert not result.converged
    assert result.termination_reason == "mean_space_boundary"


def test_identity_start_rejects_infeasible_offset_design_instead_of_clipping_it():
    X = pd.DataFrame({"x": np.zeros(3)})
    with pytest.raises(ValueError, match="(?i)binomial.*identity.*infeasible"):
        SuperGLM(
            family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
        ).fit(X, np.array([0.0, 1.0, 0.0]), offset=np.array([0.0, 2.0, 0.5]))


def test_identity_start_does_not_mistake_near_aliased_columns_for_infeasible_offsets():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, 50)
    z = rng.uniform(0.0, 1.0, 50)
    X = pd.DataFrame({"x": x, "alias": x + 1e-11 * z})
    feasible = np.full(50, 0.01)
    feasible[::5] = 0.99
    # These same float64 design products cancel at the stated known start.
    offset = 3.0 + X.to_numpy() @ np.array([1e11, -1e11]) + feasible
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=0,
        features={"x": Numeric(), "alias": Numeric()},
    ).fit(X, np.resize(np.array([0.0, 1.0]), len(X)), offset=offset, max_iter=10)
    means = linear_predictor(model._dm, model.result, offset)
    assert np.all((means >= 0.0) & (means <= 1.0)), means
    _assert_prediction_matches_canceling_affine_state(model, X, offset, means)


@pytest.mark.parametrize("seed", [10, 31, 95])
@pytest.mark.parametrize(
    "translation, scales, alias",
    [
        (0.0, (1.0, 1.0), 1.0),
        (1e8, (1.0, 1e-6), 1.0),
        (0.0, (1e-12, 1e12), 1e-8),
        (0.0, (1.0, 1.0), 1e-11),
    ],
)
def test_identity_start_feasibility_stress_grid(seed, translation, scales, alias):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(0.0, 1.0, (2, 50))
    X = pd.DataFrame(
        {"x": scales[0] * (translation + x), "alias": scales[1] * (translation + x + alias * z)}
    )
    known_beta = np.array([1.0 / (scales[0] * alias), -1.0 / (scales[1] * alias)])
    centered = X.to_numpy() - X.to_numpy().mean(axis=0)
    contribution = centered[:, 0] * known_beta[0] + centered[:, 1] * known_beta[1]
    feasible = np.full(len(X), 0.01)
    feasible[::5] = 0.99
    offset = 3.0 + contribution + feasible
    # Validate the known start on the delivered binary64 offset, after all
    # construction roundings; ideal real-valued feasibility would not suffice.
    known_means = (-3.0 - contribution) + offset
    assert np.all((known_means > 0.0) & (known_means < 1.0)), known_means
    model = SuperGLM(
        family="binomial",
        link="identity",
        selection_penalty=0,
        features={"x": Numeric(), "alias": Numeric()},
    ).fit(X, rng.binomial(1, 0.5, len(X)), offset=offset, max_iter=10)
    means = linear_predictor(model._dm, model.result, offset)
    assert np.all((means >= 0.0) & (means <= 1.0)), means
    _assert_prediction_matches_canceling_affine_state(model, X, offset, means)


def test_identity_qp_recovery_cannot_override_probability_feasibility():
    x = np.arange(5) / 4
    offset = 3.0 + 2.0 * x + np.array([0.01, 0.01, 0.01, 0.99, 0.01])
    result, _ = fit_irls_direct(
        X=x[:, None],
        y=np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        weights=np.ones(5),
        family=Binomial(),
        link=IdentityLink(),
        groups=[
            GroupSlice("x", 0, 1, constraints=LinearConstraintSet(A=np.ones((1, 1)), b=np.zeros(1)))
        ],
        lambda2=0.0,
        offset=offset,
        max_iter=5,
        weight_semantics="frequency",
    )
    means = x * result.beta[0] + result.intercept + offset
    assert np.all((means >= 0.0) & (means <= 1.0)), means
    assert not result.converged
    assert result.termination_reason == "constraint_infeasible"


def test_invalid_identity_warm_start_restores_a_feasible_coefficient_state():
    rng = np.random.default_rng(17)
    X = pd.DataFrame({"x": rng.uniform(-1.0, 1.0, 120)})
    y = rng.binomial(1, 0.45 + 0.1 * X.x.to_numpy()).astype(float)
    model = SuperGLM(
        family="binomial", link="identity", selection_penalty=0, features={"x": Numeric()}
    ).fit(X, y, tol=1e-10)
    result, _ = fit_irls_direct(
        X=model._dm,
        y=y,
        weights=np.ones(len(y)),
        family=Binomial(),
        link=IdentityLink(),
        groups=model._groups,
        lambda2=0,
        beta_init=np.array([5.0]),
        intercept_init=-3.0,
        tol=1e-10,
        weight_semantics="frequency",
    )
    raw = model._dm.matvec(result.beta) + result.intercept
    assert result.converged
    np.testing.assert_allclose(raw, model.predict(X), rtol=0.0, atol=2e-6)


def test_feasible_centered_identity_warm_start_is_not_read_in_raw_coordinates():
    spread = np.repeat(np.array([0.0, 2.0, 4.0, 6.0]), 2)
    dm = DesignMatrix([DenseGroupMatrix((1e16 + spread)[:, None])], n=8, p=1)
    # The success fractions are exactly the affine probabilities below.
    # Their feasible centered start has raw eta[0] == 0 from cancellation.
    result, _ = fit_irls_direct(
        X=dm,
        y=np.tile([0.0, 1.0], 4),
        weights=np.array([19.0, 1.0, 15.0, 5.0, 11.0, 9.0, 7.0, 13.0]),
        family=Binomial(),
        link=IdentityLink(),
        groups=[GroupSlice("x", 0, 1, penalized=False)],
        lambda2=0,
        beta_init=np.array([0.1]),
        intercept_init=-1e15,
        _centred_init=(0.05, np.array([1e16])),
        max_iter=1,
        weight_semantics="frequency",
    )
    np.testing.assert_allclose(
        linear_predictor(dm, result, np.zeros(8)), 0.05 + 0.1 * spread, atol=1e-14, rtol=0.0
    )
    assert result.converged


@pytest.mark.parametrize("direct_solve", ["auto", "qr"])
@pytest.mark.parametrize("max_iter", [1, 5])
def test_raw_identity_warm_start_is_checked_in_its_retained_center(direct_solve, max_iter):
    x = 1e16 + np.array([0.0, 2.0, 4.0, 6.0])
    dm = DesignMatrix([DenseGroupMatrix(x[:, None])], n=4, p=1)
    raw_start = x * 0.1 - 1e15
    offset = 0.99 - raw_start
    # Raw multiplication happens to give a feasible start. The fit's
    # retained centered evaluation differs materially after cancellation.
    np.testing.assert_array_equal(raw_start + offset, np.full(4, 0.99))
    assert np.all((offset - 0.1775 > 0.0) & (offset - 0.1775 < 1.0))
    result, _ = fit_irls_direct(
        X=dm,
        y=np.array([0.0, 1.0, 0.0, 1.0]),
        weights=np.ones(4),
        family=Binomial(),
        link=IdentityLink(),
        groups=[GroupSlice("x", 0, 1, penalized=False)],
        lambda2=0,
        beta_init=np.array([0.1]),
        intercept_init=-1e15,
        offset=offset,
        direct_solve=direct_solve,
        max_iter=max_iter,
        weight_semantics="frequency",
    )
    means = linear_predictor(dm, result, offset)
    assert np.all((means >= 0.0) & (means <= 1.0)), means


def test_identity_true_rows_use_identity_derivatives_and_ignore_zero_weight_rows():
    eta = np.array([0.25, 0.75, 1e-15, 1.0 - 1e-15, 5.0])
    y = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    weights = np.array([2.0, 3.0, 4.0, 5.0, 0.0])
    score, fisher = mean_space_score_rows(y, weights, eta, IdentityLink())
    curvature, newton_score = mean_space_newton_rows(y, weights, eta, IdentityLink())
    expected_score = np.array([8.0, -12.0, -4.0 / (1.0 - eta[2]), 5.0 / eta[3], 0.0])
    expected_curvature = np.array([32.0, 48.0, 4.0 / (1.0 - eta[2]) ** 2, 5.0 / eta[3] ** 2, 0.0])
    tolerance = 8 * np.finfo(float).eps
    np.testing.assert_allclose(score, expected_score, rtol=tolerance)
    np.testing.assert_allclose(newton_score, expected_score, rtol=tolerance)
    np.testing.assert_allclose(curvature, expected_curvature, rtol=tolerance)
    np.testing.assert_allclose(fisher[:4], weights[:4] / (eta[:4] * (1 - eta[:4])), rtol=tolerance)
    assert fisher[4] == 0.0
    expected_likelihood = np.array(
        [2 * np.log(0.25), 3 * np.log(0.25), 4 * np.log1p(-eta[2]), 5 * np.log(eta[3]), 0.0]
    )
    np.testing.assert_allclose(
        mean_space_log_likelihood_rows(y, weights, eta, IdentityLink()),
        expected_likelihood,
        rtol=tolerance,
    )


@pytest.mark.parametrize("response", [0.0, 1.0])
@pytest.mark.parametrize("penalty, boundary", [(0.0, True), (1e-6, False)])
def test_identity_truncated_direction_uses_both_probability_bounds(response, penalty, boundary):
    # The light row is invisible beside the heavy rows' score scale. Its
    # unpenalized likelihood has a finite boundary at 0 or 1; with the stated
    # penalty its Fisher step stays interior. Neither verdict uses log eta=0.
    dm = DesignMatrix([DenseGroupMatrix(np.array([[0.0], [0.0], [1.0]]))], n=3, p=1)
    y = np.array([0.0, 1.0, response])
    eta = np.array([0.5, 0.5, 0.8 if response == 0 else 0.2])
    weights = np.array([1e8, 1e8, 1e-8])
    score = weights * (y / eta - (1 - y) / (1 - eta))
    fisher = weights / (eta * (1 - eta))
    ratio, found = truncated_direction_ratio(
        dm=dm,
        null_basis=np.ones((1, 1)),
        angle=1e-12,
        mean_x=np.zeros(1),
        row_score=score,
        fisher_weights=fisher,
        response=y,
        positive_prior=np.ones(3, dtype=bool),
        penalty_gradient=np.zeros(1),
        penalty_size=np.zeros(1),
        penalty_apply=lambda v: penalty * v,
        penalty_size_apply=lambda v: penalty * np.abs(v),
        bar=MODE_CERTIFICATION_BAR,
        underflow=0.0,
        eta=eta,
        eta_bounds=(0.0, 1.0),
    )
    assert ratio > 1.0
    (record,) = found
    assert record.rows == (2,)
    assert not record.at_maximum
    assert record.boundary is boundary
