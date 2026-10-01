"""Profiled-intercept geometry for cached discrete REML trials."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from superglm.distributions import Gaussian
from superglm.group_matrix import DenseGroupMatrix, DesignMatrix
from superglm.links import IdentityLink
from superglm.reml.discrete import _solve_cached_profiled_system
from superglm.reml.objective import reml_laml_objective
from superglm.solvers.irls_direct import fit_irls_direct
from superglm.solvers.mode_score import centred_matvec, prior_weighted_centre
from superglm.solvers.pirls import PIRLSResult
from superglm.types import GroupSlice


def _translated_gaussian_fixture() -> tuple[DesignMatrix, np.ndarray, np.ndarray, GroupSlice]:
    rng = np.random.default_rng(20260718)
    n = 80
    centered = rng.normal(size=(n, 3))
    centered -= np.mean(centered, axis=0)
    X = centered + np.array([1.0e10, -3.0e10, 7.0e9])
    beta = np.array([0.7, -0.25, 0.4])
    y = 1.3 + centered @ beta + rng.normal(scale=0.05, size=n)
    weights = rng.uniform(0.4, 1.7, size=n)
    dm = DesignMatrix([DenseGroupMatrix(X)], n=n, p=X.shape[1])
    return dm, y, weights, GroupSlice(name="x", start=0, end=X.shape[1])


def test_direct_working_cache_retains_stable_profiled_system() -> None:
    dm, y, weights, group = _translated_gaussian_fixture()
    cache: dict[str, object] = {}
    penalty = np.diag([0.2, 0.4, 0.8])

    fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=Gaussian(),
        link=IdentityLink(),
        groups=[group],
        lambda2=1.0,
        max_iter=2,
        tol=1.0e-12,
        cache_out=cache,
        S_override=penalty,
        compute_rank_info=False,
        _return_working_system=True,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    assert set(cache) >= {
        "centered_XtWX",
        "centered_rhs",
        "mean_x",
        "mean_z",
        "sum_W",
    }
    assert np.all(np.isfinite(cache["centered_XtWX"]))
    assert np.all(np.isfinite(cache["centered_rhs"]))


@pytest.mark.parametrize("lambda_value", [1.0e-4, 0.7, 1.0e4])
def test_cached_trial_matches_full_profiled_objective_after_large_translation(
    lambda_value: float,
) -> None:
    dm, y, weights, group = _translated_gaussian_fixture()
    family = Gaussian()
    link = IdentityLink()
    base_penalty = np.diag([0.3, 0.9, 1.7])
    trial_penalty = lambda_value * base_penalty
    cache: dict[str, object] = {}

    fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=[group],
        lambda2=1.0,
        max_iter=2,
        tol=1.0e-12,
        cache_out=cache,
        S_override=base_penalty,
        compute_rank_info=False,
        _return_working_system=True,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    beta_cached, intercept_cached, log_det_cached, hessian_rank_cached = (
        _solve_cached_profiled_system(
            cache["centered_XtWX"],
            trial_penalty,
            cache["centered_rhs"],
            cache["mean_x"],
            cache["sum_W"],
            cache["mean_z"],
        )
    )
    full_result, _, full_xtwx = fit_irls_direct(
        X=dm,
        y=y,
        weights=weights,
        family=family,
        link=link,
        groups=[group],
        lambda2=lambda_value,
        max_iter=3,
        tol=1.0e-12,
        return_xtwx=True,
        S_override=trial_penalty,
        compute_rank_info=False,
        _compute_fit_statistics=False,
        weight_semantics="frequency",
    )

    np.testing.assert_allclose(beta_cached, full_result.beta, rtol=2.0e-11, atol=2.0e-11)
    assert intercept_cached == pytest.approx(full_result.intercept, rel=2.0e-11, abs=2.0e-4)
    assert log_det_cached == pytest.approx(full_result.log_det_H, rel=2.0e-12, abs=2.0e-12)
    assert hessian_rank_cached == full_result.reml_hessian_rank

    # the linear predictor about the fixed prior-weighted centre, as the cached
    # trials (``reml.discrete``) and every PIRLS state form it: at a 1e10
    # translation X beta against the raw intercept errs by ~1e-6 per row
    centre = prior_weighted_centre(dm, weights)
    alpha_cached = intercept_cached + math.fsum(centre * beta_cached)
    eta_cached = alpha_cached + centred_matvec(dm, beta_cached, centre)
    mu_cached = link.inverse(eta_cached)
    cached_result = PIRLSResult(
        beta=beta_cached,
        intercept=intercept_cached,
        n_iter=0,
        deviance=float(np.sum(weights * family.deviance_unit(y, mu_cached))),
        converged=True,
        phi=1.0,
        effective_df=0.0,
        log_det_H=log_det_cached,
        reml_hessian_rank=hessian_rank_cached,
        centred_intercept=alpha_cached,
        state_center=centre,
    )
    cached_objective = reml_laml_objective(
        dm,
        family,
        link,
        [group],
        y,
        cached_result,
        {},
        weights,
        np.zeros_like(y),
        XtWX=cache["XtWX"],
        log_det_H=log_det_cached,
        S_override=trial_penalty,
        weight_semantics="frequency",
    )
    full_objective = reml_laml_objective(
        dm,
        family,
        link,
        [group],
        y,
        full_result,
        {},
        weights,
        np.zeros_like(y),
        XtWX=full_xtwx,
        log_det_H=full_result.log_det_H,
        S_override=trial_penalty,
        weight_semantics="frequency",
    )
    assert cached_objective == pytest.approx(full_objective, rel=2.0e-11, abs=2.0e-11)


def test_cached_profiled_solve_keeps_well_conditioned_trials_on_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.discrete as discrete_reml

    def fail_spectral_fallback(*args, **kwargs):
        raise AssertionError("well-conditioned cached trial used the rank fallback")

    monkeypatch.setattr(discrete_reml, "decompose_gram", fail_spectral_fallback)
    centered_gram = np.diag([2.0, 3.0, 5.0])
    penalty = np.diag([0.2, 0.4, 0.8])
    rhs = np.array([1.0, -2.0, 0.5])
    beta, intercept, log_det_h, rank = discrete_reml._solve_cached_profiled_system(
        centered_gram,
        penalty,
        rhs,
        np.array([0.3, -0.5, 0.1]),
        12.0,
        -0.7,
    )

    np.testing.assert_allclose(beta, np.linalg.solve(centered_gram + penalty, rhs))
    assert np.isfinite(intercept)
    assert np.isfinite(log_det_h)
    assert rank == 4


def test_a_discrete_trial_forms_eta_about_the_prior_weighted_centre(monkeypatch) -> None:
    """A cached discrete REML trial scores the deviance at ``alpha + X~ beta`` (design §3.8).

    With a column translated by 1e10, ``X beta + intercept`` cancels about
    ``1e10 |beta|`` per row: an eta error near ``1e10 u |beta| ~ 1e-6``, which
    moves the Poisson deviance the line search compares by ``~ sum |mu - y|
    1e-6``.  Every recorded trial's deviance must instead match the deviance
    at the centred predictor within the rounding both evaluations carry:
    ``gamma_{p+2} m_r`` in each row's eta (``m_r = |alpha| + sum |x~||beta| +
    |o|``, Higham 2002 §3.1), through ``|d'| = 2 (mu e^delta + y)``, plus 16 u
    of each unit deviance and ``gamma_n`` of the sum.  Mutation: the trial's
    eta formed raw (``intercept + X beta``).
    """
    import pandas as pd

    import superglm.reml.discrete as discrete_module
    from superglm import Numeric, Spline, SuperGLM

    recorded = []
    original = discrete_module.reml_laml_objective

    def record(dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kwargs):
        if pirls.n_iter == 0 and pirls.state_center is not None:
            recorded.append(
                (
                    dm,
                    pirls.beta.copy(),
                    float(pirls.centred_intercept),
                    pirls.state_center.copy(),
                    float(pirls.deviance),
                    np.asarray(weights),
                    np.asarray(offset),
                )
            )
        return original(
            dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kwargs
        )

    monkeypatch.setattr(discrete_module, "reml_laml_objective", record)
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.uniform(size=n)
    z = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + np.sin(2 * np.pi * x) + 0.4 * z)).astype(float)
    model = SuperGLM(
        family="poisson",
        features={"x": Spline(n_knots=12), "z": Numeric()},
        selection_penalty=0,
        discrete=True,
    )
    with warnings.catch_warnings():
        # the translated fit's terminal mode is disclosed as not certified
        warnings.simplefilter("ignore")
        model.fit_reml(pd.DataFrame({"x": x, "z": z + 1.0e10}), y)
    assert recorded

    u = np.finfo(float).eps / 2
    for dm, beta, alpha, centre, deviance, weights, offset in recorded:
        X = dm.toarray()
        p = X.shape[1]
        centred = X - centre
        eta = alpha + centred @ beta + offset
        mu = np.exp(eta)
        units = 2.0 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu))
        reference = float(np.sum(weights * units))
        magnitude = abs(alpha) + np.abs(centred) @ np.abs(beta) + np.abs(offset)
        delta = 2.0 * (p + 2) * u / (1.0 - (p + 2) * u) * magnitude
        slope = 2.0 * (mu * np.exp(delta) + y)
        per_row = slope * delta + 16.0 * u * (np.abs(units) + y + mu)
        bound = float(np.sum(weights * per_row)) + 2.0 * n * u / (1.0 - n * u) * float(
            np.sum(weights * np.abs(units))
        )
        assert abs(deviance - reference) <= bound, (deviance - reference, bound)
