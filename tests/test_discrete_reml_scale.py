"""Discrete REML integration checks for family-correct scale profiling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperGLM
from superglm.features import Categorical, Spline
from superglm.reml.scale import (
    prepare_gamma_reml_scale_data,
    profile_gamma_reml_scale,
)


def _gamma_data(n: int = 240) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4821)
    x = rng.uniform(-1.0, 1.0, n)
    mu = np.exp(0.35 + 0.7 * np.sin(2.2 * x))
    y = rng.gamma(shape=4.5, scale=mu / 4.5)
    weights = rng.integers(1, 4, size=n).astype(np.float64)
    return pd.DataFrame({"x": x}), y, weights


def _penalized_deviance(result, group_matrices, lambdas, penalties) -> float:
    penalty_quad = 0.0
    for component in penalties:
        omega = component.omega_ssp
        if omega is None:
            gm = group_matrices[component.group_index]
            omega = gm.R_inv.T @ gm.omega @ gm.R_inv
        beta = result.beta[component.group_sl]
        penalty_quad += float(lambdas[component.name]) * float(beta @ omega @ beta)
    return float(result.deviance + penalty_quad)


def test_discrete_gamma_prepares_once_and_forwards_fd_profile_curvature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import superglm.reml.discrete as discrete

    X, y, weights = _gamma_data()
    profile_data = prepare_gamma_reml_scale_data(y, weights, weight_semantics="prior")
    prepare_calls = 0
    hessian_checks = 0
    original_prepare = discrete.prepare_reml_scale_data
    original_hessian = discrete.reml_direct_hessian

    def counted_prepare(distribution, actual_y, actual_weights, *, weight_semantics):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(
            distribution,
            actual_y,
            actual_weights,
            weight_semantics=weight_semantics,
        )

    def checked_hessian(*args, **kwargs):
        nonlocal hessian_checks
        group_matrices = args[0]
        lambdas = args[3]
        result = kwargs["pirls_result"]
        penalties = kwargs["reml_penalties"]
        penalty_nullity = kwargs["penalty_nullity"]
        expected = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties),
            penalty_nullity,
        )
        assert kwargs["inverse_phi"] == pytest.approx(expected.inverse_phi, rel=2.0e-13)
        assert kwargs["d_inverse_phi_d_penalized_deviance"] == pytest.approx(
            expected.d_inverse_phi_d_penalized_deviance,
            rel=2.0e-13,
        )

        step = max(1.0e-6, 1.0e-6 * _penalized_deviance(result, group_matrices, lambdas, penalties))
        lower = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties) - step,
            penalty_nullity,
        )
        upper = profile_gamma_reml_scale(
            profile_data,
            _penalized_deviance(result, group_matrices, lambdas, penalties) + step,
            penalty_nullity,
        )
        finite_difference = (upper.inverse_phi - lower.inverse_phi) / (2.0 * step)
        assert kwargs["d_inverse_phi_d_penalized_deviance"] == pytest.approx(
            finite_difference,
            rel=3.0e-7,
            abs=2.0e-12,
        )
        hessian_checks += 1
        return original_hessian(*args, **kwargs)

    monkeypatch.setattr(discrete, "prepare_reml_scale_data", counted_prepare)
    monkeypatch.setattr(discrete, "reml_direct_hessian", checked_hessian)

    model = SuperGLM(
        family="gamma",
        selection_penalty=0.0,
        discrete=True,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )
    model.fit_reml(
        X,
        y,
        sample_weight=weights,
        max_reml_iter=6,
        runtime_validation="skip",
    )

    assert prepare_calls == 1
    assert hessian_checks >= 2


def test_discrete_line_search_refuses_a_trial_outside_the_inverse_links_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A smoothing-parameter trial that carries a row to eta <= 0 under the inverse link is
    refused, as a rejected trial, instead of being scored on clipped means of about 1e12: no
    trial the line search scores has a positive-weight row outside the domain."""
    from superglm.reml import discrete
    from superglm.solvers.mode_score import centred_matvec

    scored_outside = []
    real_objective = discrete.reml_laml_objective

    def checked_objective(dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kw):
        if pirls.n_iter == 0 and pirls.state_center is not None:  # a line-search trial
            eta = pirls.centred_intercept + centred_matvec(dm, pirls.beta, pirls.state_center)
            scored_outside.append(bool(np.any((eta + offset <= 0.0) & (weights > 0.0))))
        return real_objective(
            dm, distribution, link, groups, y, pirls, lambdas, weights, offset, **kw
        )

    monkeypatch.setattr(discrete, "reml_laml_objective", checked_objective)

    real = discrete._solve_cached_profiled_system
    crossed = {"n": 0}

    def crossing_first(*args, **kwargs):
        beta, intercept, log_det_h, rank = real(*args, **kwargs)
        if crossed["n"] == 0:
            crossed["n"] += 1
            beta = beta * 1e3  # far enough to put rows on both sides of eta = 0
        return beta, intercept, log_det_h, rank

    monkeypatch.setattr(discrete, "_solve_cached_profiled_system", crossing_first)
    X, y, weights = _gamma_data()
    model = SuperGLM(
        family="gamma",
        link="inverse",
        selection_penalty=0.0,
        discrete=True,
        features={"x": Spline(n_knots=6, penalty="ssp")},
    )
    model.fit_reml(X, y, sample_weight=weights, max_reml_iter=6, runtime_validation="skip")

    assert crossed["n"] == 1
    assert scored_outside and not any(scored_outside)
    assert model._reml_profile["reml_n_domain_refused_trials"] >= 1
    mu = model.predict(X)
    assert np.all(np.isfinite(mu)) and mu.max() < 1e3 * y.max()


def _curvature_pair_fixture(name):
    """A 4,000-row fit whose binning is lossless (40 x values, 4 levels), for one family and link."""
    from superglm.distributions import NegativeBinomial, Tweedie

    rng = np.random.default_rng(3)
    n = 4000
    x = rng.integers(0, 40, n) / 40.0
    level = rng.integers(0, 4, n).astype(str)
    f = np.sin(5.0 * x) + 0.2 * (level == "1")
    if name == "gamma/log":
        family, link, y = "gamma", "log", rng.gamma(0.7, np.exp(1.0 + f) / 0.7)
    elif name == "tweedie/log":
        mu = np.exp(1.0 + f)
        counts = rng.poisson(mu**0.5 / 1.5)
        y = np.array([rng.gamma(2.0, 1.5 * m**0.5 / 2.0, k).sum() for k, m in zip(counts, mu)])
        family, link = Tweedie(p=1.5), "log"
    elif name in ("binomial/probit", "binomial/cloglog"):
        family, link = "binomial", name.split("/")[1]
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-f))).astype(float)
    else:
        family, link = NegativeBinomial(theta=1.0), "log"
        y = rng.negative_binomial(1.0, 1.0 / (1.0 + np.exp(f))).astype(float)
    return pd.DataFrame({"x": x, "c": level}), y, family, link


def _binned_curvature_cases():
    from superglm.distributions import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson, Tweedie
    from superglm.links import (
        CauchitLink,
        CloglogLink,
        IdentityLink,
        InverseLink,
        LogitLink,
        LogLink,
        NegativeBinomialLink,
        ProbitLink,
        SqrtLink,
    )

    class _GammaSubclass(Gamma):
        pass

    return [
        (Gamma(), LogLink(), "observed"),
        (Tweedie(p=1.5), LogLink(), "observed"),
        (NegativeBinomial(theta=1.0), LogLink(), "observed"),
        (Binomial(), ProbitLink(), "observed"),
        (Binomial(), CloglogLink(), "observed"),
        # canonical: Fisher is the observed information
        (Poisson(), LogLink(), "fisher"),
        (Binomial(), LogitLink(), "fisher"),
        (Gaussian(), IdentityLink(), "fisher"),
        (Gamma(), InverseLink(), "fisher"),
        (NegativeBinomial(theta=1.0), NegativeBinomialLink(theta=1.0), "fisher"),
        # observed on the exact route, not declared on the binned one:
        # binomial/log for its mean-space boundary (the clip, and no
        # curvature on event rows), Poisson/sqrt not yet; Gaussian/log and
        # cauchit have observed rows that go negative
        (Binomial(), LogLink(), "fisher"),
        (Gaussian(), LogLink(), "fisher"),
        (Binomial(), CauchitLink(), "fisher"),
        (Poisson(), SqrtLink(), "fisher"),
        # a subclass does not inherit the claim
        (_GammaSubclass(), LogLink(), "fisher"),
    ]


def test_binned_reml_curvature_is_a_function_of_family_and_link():
    """The binned route's curvature, pair by pair (``classify_binned_reml_curvature``).

    Observed where the exact route's is and the observed rows can never go
    negative (a log-likelihood concave in eta), Fisher everywhere else.
    Mutation check: the rule does not exist on the base (task A published
    Fisher for every pair).
    """
    from superglm.reml.observed_geometry import (
        classify_binned_reml_curvature,
        classify_reml_curvature,
    )

    for family, link, expected in _binned_curvature_cases():
        assert classify_binned_reml_curvature(family, link) == expected, (family, link)
        if expected == "observed":
            assert classify_reml_curvature(family, link) == "observed"


@pytest.mark.parametrize(
    "name", ["gamma/log", "tweedie/log", "nb2/log", "binomial/probit", "binomial/cloglog"]
)
def test_a_binned_observed_pair_converges_on_the_exact_routes_criterion(name, monkeypatch):
    """Each pair the rule makes observed is observed throughout on the binned route.

    Every performance-iteration solve (bootstrap and candidates) asks for
    observed working rows and takes them, the fit converges, and it publishes
    the exact route's criterion: the exact route with its smoothing
    parameters held at the binned fit's publishes the same objective.  This
    fixture's binning loses nothing, so the two objectives are the same
    function at the same smoothing parameters, evaluated at two certified
    modes; the mode certificate's bar is set so a certified mode moves the
    objective by less than the REML stop's resolution, ``reml_tol (1 + |V|)``
    (``mode_score.mode_certification_bar``).  The two optima are not
    compared: performance iteration leaves out the ``dW/drho`` term, so its
    stationary point differs from the exact route's.  Mutation checks: on the
    base the binned route was Fisher throughout and its objective differed
    from the exact route's at the same smoothing parameters by 1e-3 to
    2e-2; without ``_working_newton`` on the performance-iteration solves
    their observed counters stay at zero.
    """
    import superglm.reml.discrete as discrete
    from superglm.types import LambdaPolicy

    real = discrete.fit_irls_direct
    iteration_solves: list[tuple[bool, int]] = []

    def spy(*args, **kwargs):
        profile = kwargs.get("profile")
        before = 0 if profile is None else profile.get("irls_observed_newton_iters", 0)
        out = real(*args, **kwargs)
        if kwargs.get("trace_purpose") in {"reml_bootstrap", "reml_candidate"}:
            after = 0 if profile is None else profile.get("irls_observed_newton_iters", 0)
            iteration_solves.append((bool(kwargs.get("_working_newton")), after - before))
        return out

    monkeypatch.setattr(discrete, "fit_irls_direct", spy)
    X, y, family, link = _curvature_pair_fixture(name)
    binned = SuperGLM(
        family=family,
        link=link,
        features={"x": Spline(n_knots=8), "c": Categorical()},
        discrete=True,
    )
    binned.fit_reml(X, y)
    assert binned.reml_diagnostics()["converged"]
    assert binned._reml_result.curvature_source == "observed"
    assert len(iteration_solves) >= 2
    assert all(asked and observed > 0 for asked, observed in iteration_solves)

    (lam,) = binned._reml_result.lambdas.values()
    held = SuperGLM(
        family=family,
        link=link,
        features={
            "x": Spline(n_knots=8, lambda_policy=LambdaPolicy.fixed(lam)),
            "c": Categorical(),
        },
    )
    held.fit_reml(X, y)
    assert held._reml_result.termination_reason == "fixed_lambdas"
    assert held._reml_result.curvature_source == "observed"
    objective = held._reml_result.objective
    resolution = binned._reml_profile["reml_tol_resolved"] * (1.0 + abs(objective))
    assert abs(binned._reml_result.objective - objective) <= resolution


def test_probit_and_cloglog_observed_rows_stay_non_negative_past_the_mean_clip():
    """Probit and cloglog working rows are formed in eta, so the clip cannot make them negative.

    ``clip_mu`` holds a binomial mean inside ``[1e-7, 1 - 1e-7]`` while the
    probit link lets eta reach +-20.  The generic observed row reads that
    clipped mean beside derivatives of the unclipped eta: probit eta = +-6
    against the opposite response gives -0.361 where the curvature is
    +0.976, and cloglog eta = -17 with an event -0.243 where it is 2.07e-8.
    Inside the band the closed forms agree with the generic row.  Mutation
    check: taking the generic row for these links (as the first version of
    this branch did) fails the sign assertions.
    """
    from scipy.special import ndtr

    from superglm.distributions import Binomial, clip_mu
    from superglm.links import CloglogLink, ProbitLink
    from superglm.reml.observed_geometry import compute_observed_information_weights
    from superglm.solvers.working_rows import coefficient_working_rows

    def rows(link, inverse, eta, y):
        eta, y = np.asarray(eta, dtype=float), np.asarray(y, dtype=float)
        mu = clip_mu(inverse(eta), Binomial())
        out = coefficient_working_rows(
            distribution=Binomial(),
            link=link,
            y=y,
            mu=mu,
            eta=eta,
            sample_weight=np.ones(eta.size),
            prefer_observed=True,
        )
        generic = compute_observed_information_weights(
            Binomial(), link, y, mu, eta, np.ones(eta.size)
        )
        return out, generic

    def cloglog_inverse(eta):
        return -np.expm1(-np.exp(eta))

    probit, _ = rows(ProbitLink(), ndtr, [6.0, -6.0], [0.0, 1.0])
    assert probit.rejection_reason is None
    assert np.all(probit.weights > 0.0)
    np.testing.assert_allclose(probit.weights, 0.976012, rtol=1e-6)
    cloglog, _ = rows(CloglogLink(), cloglog_inverse, [-17.0, -6.0, 2.0], [1.0, 1.0, 0.0])
    assert cloglog.rejection_reason is None
    assert np.all(cloglog.weights > 0.0)
    np.testing.assert_allclose(cloglog.weights[0], 0.5 * np.exp(-17.0), rtol=1e-6)
    for link, inverse in ((ProbitLink(), ndtr), (CloglogLink(), cloglog_inverse)):
        eta = np.linspace(-1.5, 1.5, 7)
        inside, generic = rows(link, inverse, eta, np.linspace(0.0, 1.0, 7))
        np.testing.assert_allclose(inside.weights, generic, rtol=1e-12)


def test_binned_binomial_log_at_the_boundary_returns_and_discloses_it():
    """A binned log-binomial whose maximum holds probabilities of 1 returns, as before.

    Binomial/log stays Fisher on the binned route: its maximum can sit on
    the mean-space boundary, where the clip holds rows and event rows carry
    no curvature.  The fit ends unconverged with its final refit on the
    mean-space boundary, and the statement says so.  Mutation check: with
    the published curvature taken from the exact route's rule (commit
    a80d264c) the observed terminal refit raised
    ``ObservedModeNotConvergedError`` here.
    """
    import warnings

    from superglm.diagnostics.convergence import ConvergenceWarning

    rng = np.random.default_rng(0)
    n = 3000
    x = rng.integers(0, 50, n) / 50.0
    level = rng.integers(0, 3, n).astype(str)
    p = np.minimum(1.0, np.exp(-1.2 + 1.6 * x + 0.1 * (level == "1")))
    y = (rng.uniform(size=n) < p).astype(float)
    model = SuperGLM(
        family="binomial",
        link="log",
        features={"x": Spline(n_knots=8), "c": Categorical()},
        discrete=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit_reml(pd.DataFrame({"x": x, "c": level}), y)
    result = model._reml_result
    assert result.curvature_source == "fisher"
    assert not model.reml_diagnostics()["converged"]
    assert result.terminal_refit_termination == "mean_space_boundary"
    messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    assert any("no interior optimum" in message for message in messages)
