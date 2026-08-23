"""Regression oracles for the PIRLS composite-penalty optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.distributions import Gaussian
from superglm.features.spline import PSpline, Spline
from superglm.links import IdentityLink
from superglm.penalties.base import penalty_targets_group
from superglm.penalties.group_elastic_net import GroupElasticNet
from superglm.penalties.group_lasso import GroupLasso
from superglm.penalties.ridge import Ridge
from superglm.reml.penalty_algebra import build_penalty_matrix
from superglm.solvers.pirls import fit_pirls
from superglm.types import GroupSlice


def _correlated_gaussian_problem() -> tuple[np.ndarray, np.ndarray, list[GroupSlice]]:
    rng = np.random.default_rng(123)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = 0.8 * x1 + 0.2 * rng.standard_normal(n)
    X = np.column_stack((x1, x2))
    X -= X.mean(axis=0)
    y = X @ np.array([2.0, -1.0])
    groups = [GroupSlice("pair", 0, 2, weight=np.sqrt(2.0))]
    return X, y, groups


class _IdentityProxRidgeSubclass(Ridge):
    """Custom protocol implementation whose inherited class is not its semantics."""

    def prox_group(self, bg, group, step):
        return bg

    def prox(self, beta, groups, step):
        return beta.copy()

    def eval(self, beta, groups):
        return 0.0


@pytest.mark.parametrize("lambda2", [0.0, 0.1])
def test_selection_only_fit_skips_dense_smooth_penalty_assembly(
    monkeypatch, lambda2: float
) -> None:
    """Absent smoothing structure must retain the no-p-by-p-allocation path."""
    import superglm.reml.penalty_algebra as penalty_algebra

    monkeypatch.setattr(
        penalty_algebra,
        "build_penalty_matrix",
        lambda *args, **kwargs: pytest.fail("ordinary groups must not build dense S"),
    )
    rng = np.random.default_rng(18)
    X = rng.standard_normal((80, 4))
    X -= X.mean(axis=0)
    y = X @ np.array([0.7, -0.3, 0.0, 0.2])
    groups = [GroupSlice(f"x{j}", j, j + 1) for j in range(X.shape[1])]

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        GroupLasso(lambda1=0.1),
        lambda2=lambda2,
        weight_semantics="frequency",
    )

    assert result.converged


def test_correlated_ridge_matches_exact_gaussian_oracle() -> None:
    """A Ridge block update must solve its quadratic, not mix two metrics."""
    X, y, groups = _correlated_gaussian_problem()
    lam = 10.0

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        Ridge(lambda1=lam),
        tol=1e-8,
        weight_semantics="frequency",
    )

    expected = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    expected_inverse = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
    expected_edf = np.diag(expected_inverse @ (X.T @ X))
    assert result.converged
    np.testing.assert_allclose(result.intercept, 0.0, atol=1e-10)
    np.testing.assert_allclose(result.beta, expected, rtol=1e-8, atol=1e-8)
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        expected_inverse,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        result.rank_info.coefficient.pseudo_inverse(),
        expected_inverse,
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(result.rank_info.feature_edf, expected_edf)
    assert result.effective_df == pytest.approx(1.0 + float(np.sum(expected_edf)))


def test_penalty_subclass_uses_its_authoritative_custom_prox() -> None:
    """Exact built-in accelerators must not capture custom Protocol semantics."""
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((120, 2))
    raw -= raw.mean(axis=0)
    X = np.linalg.qr(raw)[0] * np.sqrt(len(raw))
    y = X @ np.array([2.0, -1.0])
    groups = [GroupSlice("pair", 0, 2, weight=np.sqrt(2.0))]
    penalty = _IdentityProxRidgeSubclass(lambda1=10.0)

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        penalty,
        tol=1e-8,
        weight_semantics="frequency",
    )

    expected = np.linalg.solve(X.T @ X, X.T @ y)
    assert result.converged
    np.testing.assert_allclose(result.beta, expected, rtol=1e-8, atol=1e-8)
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        np.linalg.inv(X.T @ X),
        rtol=2e-10,
        atol=2e-10,
    )


def test_correlated_group_lasso_default_convergence_satisfies_kkt() -> None:
    """Penalized-deviance stagnation alone must not stop before composite KKT."""
    X, y, groups = _correlated_gaussian_problem()
    lam = 10.0
    penalty = GroupLasso(lambda1=lam)

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        penalty,
        tol=1e-8,
        weight_semantics="frequency",
    )

    residual = y - result.intercept - X @ result.beta
    gradient = -X.T @ residual
    expected_subgradient = lam * groups[0].weight * result.beta / np.linalg.norm(result.beta)
    assert result.converged
    np.testing.assert_allclose(gradient + expected_subgradient, 0.0, atol=1e-5)
    unit = result.beta / np.linalg.norm(result.beta)
    local_penalty_hessian = (lam * groups[0].weight / np.linalg.norm(result.beta)) * (
        np.eye(len(unit)) - np.outer(unit, unit)
    )
    expected_inverse = np.linalg.inv(X.T @ X + local_penalty_hessian)
    expected_edf = np.diag(expected_inverse @ (X.T @ X))
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        expected_inverse,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(result.rank_info.feature_edf, expected_edf)
    assert result.effective_df == pytest.approx(1.0 + float(np.sum(expected_edf)))


def test_correlated_group_elastic_net_satisfies_composite_kkt() -> None:
    X, y, groups = _correlated_gaussian_problem()
    lam = 10.0
    alpha = 0.35
    penalty = GroupElasticNet(lambda1=lam, alpha=alpha)

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        penalty,
        tol=1e-8,
        weight_semantics="frequency",
    )

    residual = y - result.intercept - X @ result.beta
    gradient = -X.T @ residual
    gradient += lam * (1.0 - alpha) * result.beta
    gradient += lam * alpha * groups[0].weight * result.beta / np.linalg.norm(result.beta)
    assert result.converged
    np.testing.assert_allclose(gradient, 0.0, atol=1e-8)
    unit = result.beta / np.linalg.norm(result.beta)
    local_penalty_hessian = lam * (1.0 - alpha) * np.eye(len(unit))
    local_penalty_hessian += (lam * alpha * groups[0].weight / np.linalg.norm(result.beta)) * (
        np.eye(len(unit)) - np.outer(unit, unit)
    )
    expected_inverse = np.linalg.inv(X.T @ X + local_penalty_hessian)
    expected_edf = np.diag(expected_inverse @ (X.T @ X))
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        expected_inverse,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(result.rank_info.feature_edf, expected_edf)
    assert result.effective_df == pytest.approx(1.0 + float(np.sum(expected_edf)))


def test_ridge_with_dense_s_override_matches_exact_oracle() -> None:
    X, y, groups = _correlated_gaussian_problem()
    lam = 4.0
    S = np.array([[5.0, 1.25], [1.25, 3.0]])

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        Ridge(lambda1=lam),
        S_override=S,
        tol=1e-8,
        weight_semantics="frequency",
    )

    expected = np.linalg.solve(X.T @ X + S + lam * np.eye(X.shape[1]), X.T @ y)
    expected_inverse = np.linalg.inv(X.T @ X + S + lam * np.eye(X.shape[1]))
    assert result.converged
    np.testing.assert_allclose(result.beta, expected, rtol=1e-8, atol=1e-8)
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        expected_inverse,
        rtol=2e-9,
        atol=2e-9,
    )


def test_group_lasso_with_dense_s_override_satisfies_composite_kkt() -> None:
    X, y, groups = _correlated_gaussian_problem()
    lam = 5.0
    S = np.array([[4.0, 1.0], [1.0, 3.0]])

    result = fit_pirls(
        X,
        y,
        np.ones(len(y)),
        Gaussian(),
        IdentityLink(),
        groups,
        GroupLasso(lambda1=lam),
        S_override=S,
        tol=1e-8,
        weight_semantics="frequency",
    )

    residual = y - result.intercept - X @ result.beta
    gradient = -X.T @ residual + S @ result.beta
    gradient += lam * groups[0].weight * result.beta / np.linalg.norm(result.beta)
    assert result.converged
    np.testing.assert_allclose(gradient, 0.0, atol=1e-8)
    unit = result.beta / np.linalg.norm(result.beta)
    local_penalty_hessian = (lam * groups[0].weight / np.linalg.norm(result.beta)) * (
        np.eye(len(unit)) - np.outer(unit, unit)
    )
    expected_inverse = np.linalg.inv(X.T @ X + S + local_penalty_hessian)
    assert result.rank_info is not None
    np.testing.assert_allclose(
        result.rank_info.augmented.pseudo_inverse(),
        expected_inverse,
        rtol=2e-9,
        atol=2e-9,
    )


def test_public_ridge_spline_fit_matches_dense_composite_oracle() -> None:
    """Ordinary public PIRLS must fit selection and spline penalties together."""
    x = np.linspace(-2.0, 2.0, 160)
    y = 0.4 + np.sin(1.7 * x) + 0.15 * x
    frame = pd.DataFrame({"x": x})
    ridge_lambda = 0.75
    spline_lambda = 8.0
    model = SuperGLM(
        family="gaussian",
        penalty="ridge",
        selection_penalty=ridge_lambda,
        spline_penalty=spline_lambda,
        features={"x": Spline(n_knots=9, penalty="ssp")},
        tol=1e-9,
    )

    model.fit(frame, y)

    X = model._dm.toarray()
    S = build_penalty_matrix(
        model._dm.group_matrices,
        model._groups,
        spline_lambda,
        X.shape[1],
    )
    ridge = np.zeros_like(S)
    fitted_penalty = model.penalty
    for group in model._groups:
        if penalty_targets_group(fitted_penalty, group):
            ridge[group.sl, group.sl] += ridge_lambda * np.eye(group.size)

    augmented = np.block(
        [
            [np.array([[len(y)]], dtype=float), np.sum(X, axis=0, keepdims=True)],
            [np.sum(X, axis=0, keepdims=True).T, X.T @ X + S + ridge],
        ]
    )
    rhs = np.concatenate(([np.sum(y)], X.T @ y))
    expected = np.linalg.solve(augmented, rhs)

    assert model.result.converged
    np.testing.assert_allclose(model.result.intercept, expected[0], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(model.result.beta, expected[1:], rtol=1e-7, atol=1e-7)
    covariance, _ = model._coef_covariance
    np.testing.assert_allclose(
        covariance / model.result.phi,
        np.linalg.inv(augmented)[1:, 1:],
        rtol=2e-8,
        atol=2e-8,
    )


def test_pirls_ill_conditioned_smooth_basis_has_no_spurious_step_rejection() -> None:
    """Audit S3's open 'Verify:' — pirls with a large S must not reject terminal steps.

    ``lambda2`` is 2.0e14 because that is where the defect is observable on this
    fixture.  At outer iteration 3 the raw penalized-deviance comparison reports
    ``+3.858892e-04`` for a step whose exact (rational-arithmetic) merit change is
    ``-1.051594e-04``: the cancellation between two ~1e13 penalty quadratics
    reverses the sign, all 21 trials are rejected, and the fit stops unconverged
    at deviance 1023.5299755551212.  The polarized delta reports ``-1.051421e-04``,
    accepts the full step, and converges at 1023.5298702885818.  Smaller lambdas
    do not reach the cancellation; much larger ones (>~1e15) make the merit itself
    too noisy for either comparison to be meaningful.

    Deliberately no threshold on the converged deviance.  Perturbing ``x`` by one
    ulp moves it by up to 2.5e-04 -- an order of magnitude more than the gap
    between the two arms -- and one such perturbation lands *worse* than the
    pre-change value, so no threshold separates them.  A different BLAS reduction
    order does the same thing, deterministically, which is why fixing the thread
    count does not make such a bound safe.  The structural assertions below are
    the property; they survived every one of those perturbations.
    """
    rng = np.random.default_rng(11)
    n = 600
    x = np.sort(rng.uniform(0.0, 1.0, n))
    mu = np.exp(0.5 + np.sin(4.0 * np.pi * x))
    y = rng.poisson(mu).astype(float)
    df = pd.DataFrame({"x": x})

    model = SuperGLM(
        features={"x": PSpline(n_knots=25)},
        family="poisson",
        selection_penalty=0.01,
    )
    model.lambda2 = {"x": 2.0e14}
    model.fit(df, y, record_diagnostics=True)

    iteration_log = model.result.iteration_log
    assert iteration_log
    assert not any(row.step_rejected for row in iteration_log), (
        "pirls rejected a terminal step in an ill-conditioned smooth basis"
    )
    # Stricter than the property under test -- a halving is ordinarily a benign
    # safeguard -- but on this fixture exact rational arithmetic says alpha=1 is
    # the safe step at every outer iteration (merit changes -5.72e+01, -6.86e-01,
    # -1.0516e-04), so any halving here is the same mis-signed comparison caught
    # one severity short of a rejection.  Kept as an early-warning tripwire.
    assert not any(row.step_halvings for row in iteration_log)
    # The rejection above is what stops the fit converging, so assert the consequence.
    assert model.result.converged


def test_pirls_line_search_merit_delta_reproduces_its_penalized_deviance(monkeypatch) -> None:
    """The wired delta must be the difference of exactly what pirls calls its merit.

    ``_irls_trial_is_unsafe`` scales its roundoff tolerance from ``_state_merit``
    magnitudes, so a delta on any other scale makes the line search incoherent
    rather than merely different.  On a well-conditioned basis the raw difference
    is trustworthy, which makes it a usable oracle for the three term groups:
    drop the quadratic, or the factor of 2 on ``penalty.eval``, and this fails.
    """
    import superglm.solvers.pirls as pirls_module
    from superglm.solvers.irls_state import _select_irls_trial as real_select

    observed: list[tuple[float, float]] = []

    def spy(*, committed, proposal, evaluate_state, merit_delta=None, **kwargs):
        assert merit_delta is not None, "pirls must pass a merit delta to the line search"

        def record(state) -> None:
            assert state.penalized_deviance is not None
            assert committed.penalized_deviance is not None
            observed.append(
                (
                    float(merit_delta(state, committed)),
                    state.penalized_deviance - committed.penalized_deviance,
                )
            )

        def wrapped(alpha: float):
            candidate = evaluate_state(alpha)
            record(candidate)
            return candidate

        record(proposal)
        return real_select(
            committed=committed,
            proposal=proposal,
            evaluate_state=wrapped,
            merit_delta=merit_delta,
            **kwargs,
        )

    monkeypatch.setattr(pirls_module, "_select_irls_trial", spy)

    rng = np.random.default_rng(4)
    n = 300
    x = np.sort(rng.uniform(0.0, 1.0, n))
    y = rng.poisson(np.exp(0.3 + np.sin(2.0 * np.pi * x))).astype(float)
    model = SuperGLM(
        # A selection penalty is required: without it ``penalty.eval`` is zero and
        # the factor of 2 on the non-quadratic term would be unobservable.
        features={"x": PSpline(n_knots=8)},
        family="poisson",
        selection_penalty=0.05,
        spline_penalty=1.5,
    )
    model.fit(pd.DataFrame({"x": x}), y)

    assert observed, "the line search never ran"
    for delta, raw in observed:
        assert delta == pytest.approx(raw, rel=1e-9, abs=1e-9)
