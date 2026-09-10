"""Analytic scalar-fit oracles under changes of objective and coefficient units."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from superglm.distributions import Gaussian, Poisson
from superglm.links import IdentityLink, LogLink
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.solvers.irls_state import _evaluate_irls_state
from superglm.solvers.pirls import (
    _composite_kkt_violation,
    _radial_block_eigensystems,
    _solve_radial_block,
    _wrap_dense_X,
    fit_pirls,
)
from superglm.types import GroupSlice

_EPS = np.finfo(float).eps
_TOL = 1e-10


def _fit(X, y, penalty, groups=None, **kwargs):
    if groups is None:
        groups = [GroupSlice(f"x{j}", j, j + 1, weight=1.0) for j in range(X.shape[1])]
    return fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        penalty,
        tol=_TOL,
        weight_semantics="frequency",
        **kwargs,
    )


def _correlated_problem(correlation=0.5):
    H = np.array([[1.0, correlation], [correlation, 1.0]])
    Q = 0.5 * np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    root = np.linalg.cholesky(H)
    X = Q @ root.T
    target = np.array([0.6, 0.8])
    y = Q @ np.linalg.solve(root, H @ target + 0.1)
    return X, y, H, target


@pytest.mark.parametrize("c", [1.0, 1e-8, 1e8])
@pytest.mark.parametrize("base_lambda, expected", [(0.0, 1.0), (0.5, 0.75), (3.0, 0.0)])
def test_scalar_lasso_preserves_whole_objective_scale(c, base_lambda, expected):
    """An absolute PSD cutoff must not erase positive condition-one curvature."""
    X = c * np.array([[-1.0], [1.0]])
    result = _fit(X, X[:, 0].copy(), GroupLasso(lambda1=base_lambda * c**2))
    assert result.converged
    np.testing.assert_allclose(result.beta, [expected], rtol=0.0, atol=_TOL + 8 * _EPS)


@pytest.mark.parametrize("c", [1.0, 1e-8, 1e8])
def test_scalar_elastic_net_scales_its_ridge_with_the_quadratic(c):
    X = c * np.array([[-1.0], [1.0]])
    result = _fit(X, X[:, 0].copy(), GroupElasticNet(lambda1=c**2, alpha=0.5))
    assert result.converged
    np.testing.assert_allclose(result.beta, [0.6], rtol=0.0, atol=_TOL + 8 * _EPS)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("elastic_net", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
def test_radial_quadratic_has_scale_invariant_kkt_solution(scale, elastic_net, inactive):
    """A root tolerance in raw Hessian units leaves a visible radial KKT error."""
    H = np.array([[2.0, 1.0], [1.0, 3.0]])
    A = H + (0.4 * np.eye(2) if elastic_net else 0.0)
    expected = np.array([0.6, 0.8])
    threshold = 0.5
    rhs = np.array([0.1, 0.2]) if inactive else A @ expected + threshold * expected
    group = GroupSlice("pair", 0, 2, weight=1.0)
    penalty = (
        GroupElasticNet(lambda1=0.9 * scale, alpha=5.0 / 9.0)
        if elastic_net
        else GroupLasso(lambda1=threshold * scale)
    )
    _, systems = _radial_block_eigensystems([scale * H], [group], penalty)
    beta = _solve_radial_block(systems[0], scale * rhs, scale * threshold)
    if inactive:
        np.testing.assert_array_equal(beta, np.zeros(2))
        assert scipy.linalg.norm(rhs) < threshold
    else:
        # Conditioned forward error and the original normalized KKT equations.
        bound = 64 * _EPS * len(beta) * np.linalg.cond(A)
        np.testing.assert_allclose(beta, expected, rtol=0.0, atol=bound)
        residual = A @ beta - rhs + threshold * beta / scipy.linalg.norm(beta)
        assert scipy.linalg.norm(residual) <= bound * scipy.linalg.norm(rhs)


@pytest.mark.parametrize("coefficient_scale", [1e-200, 1e200])
def test_radial_root_does_not_square_raw_rhs_coordinates(coefficient_scale):
    H = np.array([[2.0, 1.0], [1.0, 3.0]])
    target = np.array([0.6, 0.8])
    rhs = coefficient_scale * (H @ target + 0.5 * target)
    _, systems = _radial_block_eigensystems(
        [H],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    beta = _solve_radial_block(systems[0], rhs, 0.5 * coefficient_scale)
    normalized = beta / coefficient_scale
    residual = H @ normalized - rhs / coefficient_scale
    residual += 0.5 * normalized / scipy.linalg.norm(normalized)
    bound = 64 * _EPS * len(beta) * np.linalg.cond(H)
    np.testing.assert_allclose(normalized, target, rtol=0.0, atol=bound)
    assert scipy.linalg.norm(residual) <= bound * scipy.linalg.norm(rhs / coefficient_scale)


class _IdentityProxLasso(GroupLasso):
    """The inherited lambda does not describe this custom penalty's semantics."""

    def prox_group(self, bg, group, step):
        return bg

    def eval(self, beta, groups):
        return 0.0


def test_tiny_positive_generic_curvature_keeps_custom_prox_semantics():
    X = 1e-8 * np.array([[-1.0], [1.0]])
    result = _fit(X, X[:, 0].copy(), _IdentityProxLasso(lambda1=1e-16))
    assert result.converged
    np.testing.assert_allclose(result.beta, [1.0], rtol=0.0, atol=_TOL + 8 * _EPS)


@pytest.mark.parametrize("response_scale", [1.0, 1e-13])
def test_small_nonzero_groups_remain_active(response_scale):
    X, y, H, target = _correlated_problem(0.8)
    fits = [
        _fit(
            X,
            response_scale * y,
            GroupLasso(lambda1=0.1 * response_scale),
            active_set=active,
        )
        for active in (False, True)
    ]
    bound = 2 * _TOL * np.linalg.cond(H) * scipy.linalg.norm(target)
    for result in fits:
        assert result.converged
        normalized = result.beta / response_scale
        np.testing.assert_allclose(normalized, target, rtol=0.0, atol=bound)
        kkt = X.T @ (X @ normalized + result.intercept / response_scale - y) + 0.1
        assert scipy.linalg.norm(kkt) <= bound * np.linalg.norm(H, 2)
    np.testing.assert_allclose(
        (X @ fits[0].beta + fits[0].intercept) / response_scale,
        (X @ fits[1].beta + fits[1].intercept) / response_scale,
        rtol=0.0,
        atol=2 * bound * np.linalg.norm(X, 2),
    )


def test_active_set_does_not_freeze_small_nonzero_blocks_within_a_sweep_budget():
    X, y, H, target = _correlated_problem(0.8)
    groups = [GroupSlice(f"x{j}", j, j + 1, weight=1.0) for j in range(2)]
    # A small requested tolerance keeps the unchanged absolute inner heuristic
    # from ending this single outer step before the active-set branch runs.
    fits = [
        fit_pirls(
            X,
            1e-13 * y,
            np.ones(len(y)),
            Gaussian(),
            IdentityLink(),
            groups,
            GroupLasso(lambda1=1e-14),
            active_set=active,
            max_iter_outer=1,
            max_iter_inner=80,
            tol=1e-28,
            weight_semantics="frequency",
        )
        for active in (False, True)
    ]
    bound = 128 * _EPS * np.linalg.cond(H)
    for result in fits:
        normalized = result.beta / 1e-13
        np.testing.assert_allclose(normalized, target, rtol=0.0, atol=bound)
        kkt = X.T @ (X @ normalized + result.intercept / 1e-13 - y) + 0.1
        assert scipy.linalg.norm(kkt) <= bound * np.linalg.norm(H, 2)


@pytest.mark.parametrize("design_scale, response_scale", [(1.0, 1.0), (1e13, 1.0), (1.0, 1e-13)])
def test_final_stationarity_is_homogeneous_in_coefficient_units(design_scale, response_scale):
    """Objective stagnation must not accept an unfinished small coefficient."""
    X, y, H, target = _correlated_problem()
    result = _fit(
        design_scale * X,
        response_scale * y,
        GroupLasso(lambda1=0.1 * design_scale * response_scale),
    )
    assert result.converged
    normalized = result.beta * design_scale / response_scale
    bound = 2 * _TOL * np.linalg.cond(H) * scipy.linalg.norm(target)
    np.testing.assert_allclose(normalized, target, rtol=0.0, atol=bound)
    kkt = X.T @ (X @ normalized + result.intercept / response_scale - y) + 0.1
    assert scipy.linalg.norm(kkt) <= bound * np.linalg.norm(H, 2)


@pytest.mark.parametrize("signal", [0.0, 1e-14, 1e-10])
def test_cancellation_uses_absolute_score_error_not_relative_zero_accuracy(signal):
    """An orthogonal response must not exhaust the budget chasing assembly error."""
    X, _, H, target = _correlated_problem()
    y = np.array([1.0, -1.0, -1.0, 1.0]) + signal * (X @ target)
    result = _fit(X, y, GroupLasso(lambda1=0.0), max_iter_outer=40)

    # Independent extended-precision normal equations for the represented rows.
    x_long = X.astype(np.longdouble)
    y_long = y.astype(np.longdouble)
    x_long -= x_long.mean(axis=0)
    y_long -= y_long.mean()
    gram = x_long.T @ x_long
    rhs = x_long.T @ y_long
    determinant = gram[0, 0] * gram[1, 1] - gram[0, 1] ** 2
    expected = (
        np.array(
            [gram[1, 1] * rhs[0] - gram[0, 1] * rhs[1], gram[0, 0] * rhs[1] - gram[0, 1] * rhs[0]],
            dtype=np.longdouble,
        )
        / determinant
    )
    eta = X @ result.beta + result.intercept
    n, p = X.shape
    gamma = (n + p + 4) * _EPS / (1 - (n + p + 4) * _EPS)
    score_error = gamma * scipy.linalg.norm(abs(X).T @ (abs(y) + abs(eta)))
    bound = 4 * score_error / np.linalg.eigvalsh(H)[0]
    assert result.converged
    assert result.n_iter < 40
    assert np.linalg.norm(result.beta.astype(np.longdouble) - expected) <= bound


@pytest.mark.parametrize("c", [1.0, 1e-8])
def test_intercept_has_an_independent_scale_relative_score(c):
    X = c * np.array([[1.0], [2.0]])
    result = _fit(X, X[:, 0].copy(), GroupLasso(lambda1=0.0), max_iter_outer=400)
    assert result.converged
    # The augmented, normalized design is well conditioned; account for its
    # normal-equation condition number rather than an absolute response bound.
    condition = np.linalg.cond(np.column_stack((np.ones(2), X[:, 0] / c))) ** 2
    bound = 2 * _TOL * condition
    np.testing.assert_allclose(result.beta, [1.0], rtol=0.0, atol=bound)
    assert abs(result.intercept / c) <= bound


@pytest.mark.parametrize("c", [1.0, 1e-8])
def test_stalled_radial_update_cannot_report_convergence(monkeypatch, c):
    """Mutation: a zero feature update must not be hidden by a fitted intercept."""
    import superglm.solvers.pirls as pirls

    monkeypatch.setattr(
        pirls, "_solve_radial_block", lambda system, rhs, threshold: np.zeros_like(rhs)
    )
    X = c * np.array([[1.0], [2.0]])
    result = _fit(X, X[:, 0].copy(), GroupLasso(lambda1=0.0), max_iter_outer=4)
    assert not result.converged
    assert result.termination_reason == "max_iter"


def test_final_check_recomputes_curvature_when_working_weights_change():
    """Old weights may define a proximal step but cannot bound current score error."""
    X = np.array([[-1.0], [1.0]])
    groups = [GroupSlice("x", 0, 1, weight=1.0)]
    dm = _wrap_dense_X(X, groups)
    y = np.exp(np.array([-0.2, 0.2]))
    weights = np.ones(2)
    offset = np.zeros(2)
    state = _evaluate_irls_state(
        dm,
        y,
        weights,
        Poisson(),
        LogLink(),
        offset,
        np.array([0.2 - 1e-5]),
        0.0,
    )
    kwargs = dict(
        dm=dm,
        state=state,
        y=y,
        weights=weights,
        family=Poisson(),
        link=LogLink(),
        offset=offset,
        groups=groups,
        penalty=GroupLasso(lambda1=0.0),
        S=None,
        has_smooth_penalty=False,
    )
    expected = _composite_kkt_violation(**kwargs)
    stale = _composite_kkt_violation(**kwargs, L_groups=[1e20], curvature_weights=weights)
    assert expected > _TOL
    assert stale == pytest.approx(expected, rel=64 * _EPS, abs=0.0)


@pytest.mark.parametrize("c, intercept", [(1e-160, 0.0), (0.0, 1e308)])
def test_nonfinite_stationarity_arithmetic_cannot_certify_a_zero_residual(c, intercept):
    """A nonfinite proximal step or allowance invalidates an apparent fixed point."""
    X = c * np.array([[-1.0], [1.0]])
    groups = [GroupSlice("x", 0, 1, weight=1.0)]
    dm = _wrap_dense_X(X, groups)
    y = X[:, 0] + intercept
    weights, offset = np.ones(2), np.zeros(2)
    state = _evaluate_irls_state(
        dm,
        y,
        weights,
        Gaussian(),
        IdentityLink(),
        offset,
        np.ones(1),
        intercept,
        deviance=0.0,
    )
    violation = _composite_kkt_violation(
        dm=dm,
        state=state,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        offset=offset,
        groups=groups,
        penalty=_IdentityProxLasso(lambda1=0.0),
        S=None,
        has_smooth_penalty=False,
        tol=_TOL,
    )
    assert np.isinf(violation)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_radial_refuses_materially_indefinite_quadratic_at_any_scale(scale):
    with pytest.raises(np.linalg.LinAlgError, match="positive semidefinite"):
        _radial_block_eigensystems(
            [scale * np.diag([1.0, -0.25])],
            [GroupSlice("pair", 0, 2, weight=1.0)],
            GroupLasso(lambda1=0.0),
        )


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_radial_refuses_unbounded_zero_quadratic(threshold):
    _, systems = _radial_block_eigensystems(
        [np.zeros((1, 1))],
        [GroupSlice("x", 0, 1, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    with pytest.raises(np.linalg.LinAlgError, match="unbounded"):
        _solve_radial_block(systems[0], np.ones(1), threshold)


def test_radial_zero_quadratic_can_return_a_certified_zero():
    _, systems = _radial_block_eigensystems(
        [np.zeros((2, 2))],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    np.testing.assert_array_equal(_solve_radial_block(systems[0], np.array([0.1, 0.2]), 0.5), 0.0)


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("c", [1e-8, 1.0, 1e8])
def test_aliased_unpenalized_block_preserves_predictions_and_score(width, c):
    """An eigensystem's null-projection roundoff is not an incompatible score."""
    X = c * np.tile(np.array([[-1.0], [1.0]]), (1, width))
    y = X[:, 0].copy()
    result = _fit(
        X,
        y,
        GroupLasso(lambda1=0.0),
        groups=[GroupSlice("aliases", 0, width, weight=1.0)],
    )
    assert result.converged
    normalized_residual = (X @ result.beta + result.intercept - y) / c
    bound = 16 * _EPS * (len(y) + width) + _TOL
    assert scipy.linalg.norm(normalized_residual) <= bound * scipy.linalg.norm(y / c)
    normalized_score = (X / c).T @ normalized_residual
    assert scipy.linalg.norm(normalized_score) <= bound * scipy.linalg.norm(X / c) ** 2


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_radial_singular_quadratic_preserves_incompatible_null_score(scale, threshold):
    """Discarding every null projection would hide an unbounded or unattained mode."""
    _, systems = _radial_block_eigensystems(
        [scale * np.diag([0.0, 1.0])],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    message = "no finite minimizer" if threshold == 1.0 else "unbounded"
    with pytest.raises(np.linalg.LinAlgError, match=message):
        _solve_radial_block(systems[0], scale * np.ones(2), scale * threshold)


@pytest.mark.parametrize("null_score", [1e-16, 1e-100])
def test_small_resolved_null_score_is_not_hidden_by_a_large_range_score(null_score):
    _, systems = _radial_block_eigensystems(
        [np.diag([0.0, 1.0])],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    with pytest.raises(np.linalg.LinAlgError, match="unbounded"):
        _solve_radial_block(systems[0], np.array([null_score, 1.0]), 0.0)


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_radial_bounded_singular_quadratic_retains_a_resolved_null_penalty_force(scale):
    H = np.diag([0.0, 2.0])
    target = np.array([0.6, 0.8])
    rhs = H @ target + 0.5 * target
    _, systems = _radial_block_eigensystems(
        [scale * H],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    beta = _solve_radial_block(systems[0], scale * rhs, scale * 0.5)
    # The nonzero selection force resolves this null direction; certify KKT,
    # without using the arbitrary representative of an unpenalized null space.
    residual = H @ beta - rhs + 0.5 * beta / scipy.linalg.norm(beta)
    bound = 128 * _EPS * len(beta) * scipy.linalg.norm(rhs)
    assert scipy.linalg.norm(residual) <= bound


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("rhs", [np.array([1.0, 1.0]), np.array([1.0, 2.0]), np.array([0.6, 0.8])])
def test_nearly_inactive_radial_block_has_a_certified_finite_solution(scale, rhs):
    """Losing an ulp in the radial norm must not turn a bounded SPD fit into refusal."""
    H = np.array([[2.0, 1.0], [1.0, 3.0]])
    threshold = np.nextafter(scipy.linalg.norm(rhs), 0.0)
    _, systems = _radial_block_eigensystems(
        [scale * H],
        [GroupSlice("pair", 0, 2, weight=1.0)],
        GroupLasso(lambda1=0.0),
    )
    beta = _solve_radial_block(systems[0], scale * rhs, scale * threshold)
    assert np.all(np.isfinite(beta))
    # At this boundary a backward-certified zero and an accurate nonzero
    # solution are both valid.  Do not assert an eigenvalue-roundoff sign.
    beta_norm = scipy.linalg.norm(beta)
    if beta_norm == 0.0:
        residual = max(scipy.linalg.norm(rhs) - threshold, 0.0)
    else:
        residual = scipy.linalg.norm(H @ beta - rhs + threshold * beta / beta_norm)
    dimension = len(beta)
    gamma = (2 * dimension + 8) * _EPS / (1 - (2 * dimension + 8) * _EPS)
    bound = 2 * gamma * (scipy.linalg.norm(H @ beta) + scipy.linalg.norm(rhs) + threshold)
    assert residual <= bound


def test_public_nearly_inactive_group_finishes_with_a_certified_score():
    X = np.array(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ]
    )
    y = np.array([0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    threshold = np.nextafter(np.sqrt(2.0), 0.0)
    result = _fit(
        X,
        y,
        GroupLasso(lambda1=threshold),
        groups=[GroupSlice("pair", 0, 2, weight=1.0)],
    )
    assert result.converged
    score = X.T @ (X @ result.beta + result.intercept - y)
    beta_norm = scipy.linalg.norm(result.beta)
    if beta_norm == 0.0:
        violation = max(scipy.linalg.norm(score) - threshold, 0.0)
    else:
        violation = scipy.linalg.norm(score + threshold * result.beta / beta_norm)
    bound = 32 * _EPS * (len(y) + X.shape[1]) * threshold
    assert violation <= bound
