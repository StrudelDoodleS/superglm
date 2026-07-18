"""Authoritative REML/LAML regressions derived from Wood (2011, 2016)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from superglm.distributions import Gamma, Poisson
from superglm.links import IdentityLink, LogLink
from superglm.reml.objective import reml_laml_objective
from superglm.reml.penalty_algebra import compute_penalty_nullity
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice, PenaltyComponent

from ._wood_reml_oracles import (
    full_logdet,
    gaussian_profiled_reml_reduced,
    poisson_laml_reduced,
    profile_edf_scale_term,
    solve_gaussian_state,
    solve_poisson_log_state,
)


class _NoMatvecDesign:
    group_matrices: list = []

    def matvec(self, beta):
        del beta
        raise AssertionError("cached objective path must not expand the design")


def _pirls_result(state) -> PIRLSResult:
    return PIRLSResult(
        beta=state.beta.copy(),
        intercept=state.intercept,
        n_iter=1,
        deviance=state.deviance,
        converged=True,
        phi=1.0,
        effective_df=0.0,
    )


def _cached_objective(
    *,
    distribution,
    link,
    y: np.ndarray,
    result: PIRLSResult,
    slope_xtwx: np.ndarray,
    slope_penalty: np.ndarray,
    xtw1: np.ndarray | None = None,
    sum_w: float | None = None,
) -> float:
    return reml_laml_objective(
        _NoMatvecDesign(),
        distribution,
        link,
        [],
        y,
        result,
        {},
        np.ones_like(y),
        np.zeros_like(y),
        XtWX=slope_xtwx,
        XtW1=xtw1,
        sum_W=sum_w,
        S_override=slope_penalty,
    )


def _gaussian_fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.4, 1.2, 11)
    X = np.column_stack((x, x**2 - np.mean(x**2), np.sin(1.7 * x)))
    y = 0.8 + 1.1 * x - 0.6 * x**2 + 0.25 * np.cos(2.3 * x)
    return X, y


def test_gaussian_reml_uses_full_augmented_hessian_and_penalty_nullity() -> None:
    """Wood (2011) Eq. (4) uses full H and Mp=nullity(S), not rank(S)."""
    X, y = _gaussian_fixture()
    slope_penalty = np.diag([2.5, 0.0, 0.0])
    state = solve_gaussian_state(X, y, slope_penalty)

    assert state.penalty_nullity == 3  # intercept plus two unpenalized slope directions
    expected = gaussian_profiled_reml_reduced(state, len(y))
    actual = _cached_objective(
        distribution=SimpleNamespace(scale_known=False),
        link=IdentityLink(),
        y=y,
        result=_pirls_result(state),
        slope_xtwx=state.slope_xtwx,
        slope_penalty=slope_penalty,
        xtw1=state.full_hessian[1:, 0],
        sum_w=float(state.full_hessian[0, 0]),
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_gaussian_reml_is_invariant_to_translating_penalized_columns() -> None:
    """An intercept absorbs X-column translations, so the REML score cannot change."""
    X, y = _gaussian_fixture()
    slope_penalty = np.diag([0.7, 1.3, 2.1])
    shifted_X = X + np.array([8.0, -5.0, 2.5])
    state = solve_gaussian_state(X, y, slope_penalty)
    shifted_state = solve_gaussian_state(shifted_X, y, slope_penalty)

    expected = gaussian_profiled_reml_reduced(state, len(y))
    shifted_expected = gaussian_profiled_reml_reduced(shifted_state, len(y))
    assert shifted_expected == pytest.approx(expected, rel=1e-11, abs=1e-11)

    actual = _cached_objective(
        distribution=SimpleNamespace(scale_known=False),
        link=IdentityLink(),
        y=y,
        result=_pirls_result(state),
        slope_xtwx=state.slope_xtwx,
        slope_penalty=slope_penalty,
        xtw1=state.full_hessian[1:, 0],
        sum_w=float(state.full_hessian[0, 0]),
    )
    shifted_actual = _cached_objective(
        distribution=SimpleNamespace(scale_known=False),
        link=IdentityLink(),
        y=y,
        result=_pirls_result(shifted_state),
        slope_xtwx=shifted_state.slope_xtwx,
        slope_penalty=slope_penalty,
        xtw1=shifted_state.full_hessian[1:, 0],
        sum_w=float(shifted_state.full_hessian[0, 0]),
    )
    assert shifted_actual == pytest.approx(actual, rel=1e-11, abs=1e-11)


def test_gaussian_reml_counts_nullity_only_in_the_identified_subspace() -> None:
    """An unpenalized exact alias is not an additional fixed-effect dimension."""
    x = np.linspace(-1.0, 1.0, 12)
    X = np.column_stack((x, x))
    y = 0.7 + 1.2 * x + 0.15 * np.cos(2.0 * x)
    augmented_design = np.column_stack((np.ones(len(y)), X))
    coefficients = np.linalg.lstsq(augmented_design, y, rcond=None)[0]
    residual = y - augmented_design @ coefficients
    deviance = float(residual @ residual)
    result = PIRLSResult(
        beta=coefficients[1:],
        intercept=float(coefficients[0]),
        n_iter=1,
        deviance=deviance,
        converged=True,
        phi=1.0,
        effective_df=0.0,
    )
    slope_penalty = np.zeros((2, 2))
    centered = X - np.mean(X, axis=0)
    centered_eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    positive = centered_eigenvalues[centered_eigenvalues > 1e-12]
    assert positive.size == 1
    logdet_identified_hessian = float(np.log(len(y)) + np.log(positive[0]))
    expected = 0.5 * (len(y) - 2) * np.log(deviance) + 0.5 * logdet_identified_hessian

    actual = _cached_objective(
        distribution=SimpleNamespace(scale_known=False),
        link=IdentityLink(),
        y=y,
        result=result,
        slope_xtwx=X.T @ X,
        slope_penalty=slope_penalty,
        xtw1=np.sum(X, axis=0),
        sum_w=float(len(y)),
    )

    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-11)


def test_penalty_nullity_ignores_rotated_psd_roundoff_eigenvalues() -> None:
    """Numerical dust in an exactly rank-deficient S must not reduce M_p."""
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    exact_eigenvalues = np.concatenate((np.geomspace(1.0, 1e-4, 6), np.zeros(2)))
    penalty = (basis * exact_eigenvalues) @ basis.T

    assert compute_penalty_nullity(penalty, hessian_rank=9) == 3.0


def test_structural_penalty_nullity_is_invariant_to_positive_lambda_ratios() -> None:
    """Positive smoothing ratios do not change null(S); exact zeros do."""
    components = [
        PenaltyComponent(
            name="left",
            group_name="shared",
            group_index=0,
            group_sl=slice(0, 2),
            omega_raw=np.diag([1.0, 0.0]),
            omega_ssp=np.diag([1.0, 0.0]),
            rank=1.0,
        ),
        PenaltyComponent(
            name="right",
            group_name="shared",
            group_index=0,
            group_sl=slice(0, 2),
            omega_raw=np.diag([0.0, 1.0]),
            omega_ssp=np.diag([0.0, 1.0]),
            rank=1.0,
        ),
    ]

    extreme = {"left": 1e10, "right": 1e-6}
    extreme_matrix = np.diag([extreme["left"], extreme["right"]])
    assert (
        compute_penalty_nullity(
            extreme_matrix,
            hessian_rank=3,
            penalties=components,
            lambdas=extreme,
        )
        == 1.0
    )
    assert (
        compute_penalty_nullity(
            np.diag([1e-6, 1e10]),
            hessian_rank=3,
            penalties=components,
            lambdas={"left": 1e-6, "right": 1e10},
        )
        == 1.0
    )
    assert (
        compute_penalty_nullity(
            np.diag([1e10, 0.0]),
            hessian_rank=3,
            penalties=components,
            lambdas={"left": 1e10, "right": 0.0},
        )
        == 2.0
    )


def test_canonical_poisson_laml_uses_observed_full_hessian() -> None:
    """For canonical Poisson, Wood's observed H is the full augmented IRLS H."""
    x = np.linspace(-1.2, 1.5, 13)
    X = np.column_stack((x + 0.8, np.cos(1.3 * x) + 0.4))
    y = np.array([0, 1, 0, 2, 1, 3, 2, 4, 3, 5, 7, 6, 9], dtype=np.float64)
    slope_penalty = np.diag([0.9, 1.7])
    state = solve_poisson_log_state(X, y, slope_penalty)

    expected = poisson_laml_reduced(state)
    actual = _cached_objective(
        distribution=Poisson(),
        link=LogLink(),
        y=y,
        result=_pirls_result(state),
        slope_xtwx=state.slope_xtwx,
        slope_penalty=slope_penalty,
        xtw1=state.full_hessian[1:, 0],
        sum_w=float(state.full_hessian[0, 0]),
    )

    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-11)


def test_dense_poisson_oracle_handles_roundoff_limited_newton_decrease() -> None:
    """A converged Newton step may be smaller than objective-value resolution."""
    rng = np.random.default_rng(20260718)
    n_obs = int(rng.integers(6, 30))
    n_features = int(rng.integers(1, 5))
    X = rng.normal(size=(n_obs, n_features)) * 10 ** rng.uniform(-1.0, 1.0, size=n_features)
    coefficients = rng.normal(scale=0.5, size=n_features)
    eta = np.clip(rng.normal() + X @ coefficients, -3.0, 3.0)
    y = rng.poisson(np.exp(eta)).astype(np.float64)
    slope_penalty = np.diag(10 ** rng.uniform(-2.0, 2.0, size=n_features))

    state = solve_poisson_log_state(X, y, slope_penalty)
    augmented_design = np.column_stack((np.ones(n_obs), X))
    fitted_coefficients = np.concatenate(([state.intercept], state.beta))
    mu = np.exp(augmented_design @ fitted_coefficients)
    score = augmented_design.T @ (mu - y)
    score[1:] += slope_penalty @ state.beta

    assert np.max(np.abs(score)) < 1e-9


def test_gamma_scale_profile_retains_saturated_likelihood() -> None:
    """Wood Eq. (4) does not permit the Gaussian scale shortcut for Gamma."""
    distribution = Gamma()
    y = np.array([0.35, 0.7, 1.1, 1.8, 2.6, 3.2], dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(y))
    X = x[:, None]
    slope_penalty = np.array([[1.4]])
    slope_xtwx = X.T @ X
    results: list[PIRLSResult] = []
    penalized_deviances: list[float] = []
    production_scale_terms: list[float] = []

    for beta_value in (0.15, 0.85):
        beta = np.array([beta_value])
        intercept = 0.25
        mu = np.exp(intercept + X @ beta)
        deviance = float(np.sum(distribution.deviance_unit(y, mu)))
        result = PIRLSResult(
            beta=beta,
            intercept=intercept,
            n_iter=1,
            deviance=deviance,
            converged=True,
            phi=1.0,
            effective_df=0.0,
        )
        results.append(result)
        penalized_deviance = deviance + float(beta @ slope_penalty @ beta)
        penalized_deviances.append(penalized_deviance)
        objective = _cached_objective(
            distribution=distribution,
            link=LogLink(),
            y=y,
            result=result,
            slope_xtwx=slope_xtwx,
            slope_penalty=slope_penalty,
        )
        slope_k = 0.5 * (full_logdet(slope_xtwx + slope_penalty) - full_logdet(slope_penalty))
        production_scale_terms.append(objective - slope_k)

    del results
    expected_scale_terms = [
        profile_edf_scale_term(distribution, y, dp, penalty_nullity=1)[0]
        for dp in penalized_deviances
    ]
    expected_change = expected_scale_terms[1] - expected_scale_terms[0]
    production_change = production_scale_terms[1] - production_scale_terms[0]
    assert production_change == pytest.approx(expected_change, rel=2e-8, abs=2e-8)


def test_direct_reml_retains_an_accepted_trial_on_the_final_outer_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A line-search state already evaluated as better must be eligible for return."""
    import superglm.reml.direct as direct

    group = GroupSlice(name="smooth", start=0, end=1)
    omega = np.ones((1, 1))
    penalty = PenaltyComponent(
        name="smooth",
        group_name="smooth",
        group_index=0,
        group_sl=slice(0, 1),
        omega_raw=omega,
        omega_ssp=omega,
        rank=1.0,
        log_det_omega_plus=0.0,
        eigvals_omega=np.ones(1),
    )
    group_matrix = SimpleNamespace(R_inv=np.eye(1), omega=omega)
    dm = SimpleNamespace(group_matrices=[group_matrix], p=1)
    fit_lambdas: list[float] = []

    def fake_fit_irls_direct(*args, lambda2, **kwargs):
        del args, kwargs
        lam = float(lambda2["smooth"])
        fit_lambdas.append(lam)
        beta = np.zeros(1) if len(fit_lambdas) == 1 else np.array([lam])
        result = PIRLSResult(
            beta=beta,
            intercept=lam,
            n_iter=1,
            deviance=1.0,
            converged=True,
            phi=1.0,
            effective_df=0.0,
            log_det_H=0.0,
        )
        return result, np.eye(1), np.eye(1)

    def fake_objective(*args, **kwargs):
        del kwargs
        candidate_lambdas = args[6]
        return float(10.0 + np.log(candidate_lambdas["smooth"]))

    monkeypatch.setattr(direct, "fit_irls_direct", fake_fit_irls_direct)
    monkeypatch.setattr(direct, "reml_laml_objective", fake_objective)
    monkeypatch.setattr(
        direct,
        "build_penalty_matrix",
        lambda *args, **kwargs: (
            np.array([[float(args[2]["smooth"])]])
            if len(args) > 2
            else np.array([[float(kwargs["lambdas"]["smooth"])]])
        ),
    )
    monkeypatch.setattr(
        direct,
        "reml_direct_gradient",
        lambda *args, **kwargs: np.ones(1),
    )
    monkeypatch.setattr(
        direct,
        "reml_direct_hessian",
        lambda *args, **kwargs: np.ones((1, 1)),
    )
    monkeypatch.setattr(direct, "reml_w_correction", lambda *args, **kwargs: None)

    result = direct.optimize_direct_reml(
        dm,
        SimpleNamespace(scale_known=True),
        SimpleNamespace(),
        [group],
        False,
        np.ones(5),
        np.ones(5),
        np.zeros(5),
        [(0, group)],
        {"smooth": 1.0},
        {"smooth": 1.0},
        max_reml_iter=1,
        reml_tol=1e-8,
        verbose=False,
        reml_penalties=[penalty],
    )

    assert len(fit_lambdas) == 3
    accepted_lambda = fit_lambdas[-1]
    accepted_objective = 10.0 + np.log(accepted_lambda)
    assert accepted_lambda < fit_lambdas[-2]
    assert result.lambdas["smooth"] == pytest.approx(accepted_lambda)
    assert result.pirls_result.beta[0] == pytest.approx(accepted_lambda)
    assert result.objective == pytest.approx(accepted_objective)
