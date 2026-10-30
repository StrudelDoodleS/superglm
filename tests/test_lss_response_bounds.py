"""Truncated and censored rows in SuperLSS fits (issue 544).

Fits are checked against direct maximum likelihood in scipy on the truncated
or censored likelihood, an independent reference; the complete-row fit of the
same rows is the unfixed behaviour each fit test also rejects.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

from superglm import GammaLS, GaussianLS, GeneralizedParetoLSS, LogNormalLS, SuperLSS, s
from superglm.distributional.families.gaussian import LowerBoundedLogLink
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.kernels._response_tails import (
    inverse_mills,
    normal_window_channels,
)
from superglm.distributional.kernels.generalized_pareto import (
    GeneralizedParetoInitializationWarning,
    window_rows,
)
from superglm.distributional.kernels.log_normal import mean_window
from superglm.distributional.observation import ResponseBounds, resolve_response_bounds
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink


def _intercepts(n: int) -> pd.DataFrame:
    return pd.DataFrame({"one": np.ones(n)})


def _fit(family, y, *, weight_semantics="prior", **kwargs) -> dict[str, float]:
    model = SuperLSS(
        family,
        *[getattr(family, parameter.name)() for parameter in family.parameters],
        weight_semantics=weight_semantics,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(y)), y, **kwargs)
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    return {name: float(value) for name, value in row.items()}


def _mle(negative_log_likelihood, start) -> np.ndarray:
    result = optimize.minimize(
        negative_log_likelihood,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
    )
    assert result.success
    return result.x


# -- kernels ------------------------------------------------------------------


def test_mills_gap_is_continuous_across_the_continued_fraction_switch():
    # The differenced LAML derivatives difference this Hessian, so the two
    # evaluation branches must meet without a step.
    x = np.array([-3.0 - 1e-12, -3.0, -3.0 + 1e-12])
    _, ratio, gap = inverse_mills(x)
    assert np.all(np.abs(np.diff(gap)) <= 1e-11 * np.abs(gap[1]))
    assert np.all(np.abs(np.diff(ratio)) <= 1e-11 * np.abs(ratio[1]))


def test_mills_gap_keeps_relative_accuracy_in_the_far_lower_tail():
    t = np.array([40.0, 1.0e4])
    _, ratio, gap = inverse_mills(-t)
    # x + r = 1/t - 2/t^3 + 10/t^5 - 74/t^7 + 706/t^9 - ...; at t = 40 the
    # first omitted term is below 1e-12 of the value.
    asymptotic = 1.0 / t - 2.0 / t**3 + 10.0 / t**5 - 74.0 / t**7 + 706.0 / t**9
    np.testing.assert_allclose(gap, asymptotic, rtol=1e-12)
    np.testing.assert_allclose(ratio, t + gap, rtol=1e-15)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(-np.inf, 0.7), (-np.inf, -4.5), (1.5, np.inf), (-0.8, 2.2), (2.0, 2.4)],
)
def test_normal_window_derivatives_match_differences(lower, upper):
    mu, sigma, root = np.array([0.3]), np.array([1.4]), np.array([1.7])

    def channels(m, s_):
        return normal_window_channels(
            np.array([lower]), np.array([upper]), np.array([m]), np.array([s_]), root,
            derivative_order=2,
        )  # fmt: skip

    value, score, hessian = channels(mu[0], sigma[0])
    reference = stats.norm(mu[0], sigma[0] / root[0])
    np.testing.assert_allclose(
        value[0], np.log(reference.cdf(upper) - reference.cdf(lower)), rtol=1e-12
    )
    h = 1e-5
    for column, (dm, ds) in enumerate(((h, 0.0), (0.0, h))):
        plus = channels(mu[0] + dm, sigma[0] + ds)
        minus = channels(mu[0] - dm, sigma[0] - ds)
        np.testing.assert_allclose(
            score[0, column], (plus[0][0] - minus[0][0]) / (2 * h), rtol=1e-7, atol=1e-9
        )
        second = (plus[1][0] - minus[1][0]) / (2 * h)
        packed = (0, 1) if column == 0 else (1, 2)
        np.testing.assert_allclose(hessian[0, list(packed)], second, rtol=1e-6, atol=1e-8)


def test_generalized_pareto_window_matches_scipy_and_its_exponential_limit():
    lower, upper = np.array([0.0, 50.0, 0.0]), np.array([np.inf, 400.0, 300.0])
    scale, shape = np.full(3, 120.0), np.array([0.35, 0.2, 1e-12])
    value, _, _ = window_rows(lower, upper, scale, shape, derivative_order=2)
    law = [stats.genpareto(c=c, scale=120.0) for c in shape]
    expected = [np.log(d.sf(lo) - d.sf(hi)) for d, lo, hi in zip(law, lower, upper, strict=True)]
    np.testing.assert_allclose(value, expected, rtol=1e-12)


# -- fits against direct maximum likelihood ---------------------------------


def test_truncated_log_normal_body_recovers_the_parent_law():
    rng = np.random.default_rng(544)
    mu, sigma = 8.0, 1.2
    y = np.exp(rng.normal(mu, sigma, 20_000))
    u = float(np.exp(mu + sigma * stats.norm.ppf(0.9)))
    body = y[y <= u]
    family = LogNormalLS(parametrisation="location")

    fitted = _fit(family, body, truncation=(None, u))
    reference = _mle(
        lambda t: (
            -(
                stats.norm.logpdf(np.log(body), t[0], np.exp(t[1]))
                - stats.norm.logcdf(np.log(u), t[0], np.exp(t[1]))
            ).sum()
        ),
        [7.5, 0.0],
    )
    assert fitted["location"] == pytest.approx(reference[0], rel=1e-6)
    assert fitted["scale"] == pytest.approx(np.exp(reference[1]), rel=1e-5)
    # The complete-row reading of the same rows is the bias this fixes.
    complete = _fit(family, body)
    assert complete["scale"] < 0.9 * fitted["scale"]

    mean_form = _fit(LogNormalLS(), body, truncation=(None, u))
    implied = np.log(mean_form["mean"]) - 0.5 * mean_form["scale"] ** 2
    assert implied == pytest.approx(fitted["location"], rel=1e-6)


def test_censored_log_normal_at_a_policy_limit_recovers_the_parent_law():
    rng = np.random.default_rng(545)
    y = np.exp(rng.normal(8.0, 1.2, 20_000))
    limit = float(np.quantile(y, 0.85))
    paid = np.minimum(y, limit)
    hit = paid == limit
    family = LogNormalLS(parametrisation="location")

    fitted = _fit(family, paid, censoring=(None, limit))
    reference = _mle(
        lambda t: (
            -(
                stats.norm.logpdf(np.log(paid[~hit]), t[0], np.exp(t[1])).sum()
                + hit.sum() * stats.norm.logsf(np.log(limit), t[0], np.exp(t[1]))
            )
        ),
        [7.5, 0.0],
    )
    assert fitted["location"] == pytest.approx(reference[0], rel=1e-6)
    assert fitted["scale"] == pytest.approx(np.exp(reference[1]), rel=1e-5)
    assert _fit(family, paid)["scale"] < 0.9 * fitted["scale"]


def test_gaussian_two_sided_truncation_under_prior_weights():
    rng = np.random.default_rng(546)
    weights = rng.choice([0.5, 1.0, 2.0], 30_000)
    y = rng.normal(1.0, 2.0 / np.sqrt(weights))
    keep = (y > -1.0) & (y <= 4.0)
    y, weights = y[keep], weights[keep]

    fitted = _fit(GaussianLS(), y, sample_weight=weights, truncation=(-1.0, 4.0))

    def negative(t):
        scale = np.exp(t[1]) / np.sqrt(weights)
        window = stats.norm.cdf(4.0, t[0], scale) - stats.norm.cdf(-1.0, t[0], scale)
        return -(stats.norm.logpdf(y, t[0], scale) - np.log(window)).sum()

    reference = _mle(negative, [1.0, 0.5])
    assert fitted["location"] == pytest.approx(reference[0], rel=1e-5)
    assert fitted["scale"] == pytest.approx(np.exp(reference[1]), rel=1e-5)


def test_left_censored_gaussian_is_tobit():
    rng = np.random.default_rng(547)
    latent = rng.normal(1.0, 2.0, 20_000)
    y = np.maximum(latent, 0.0)
    low = y == 0.0
    fitted = _fit(GaussianLS(), y, censoring=(0.0, None))
    reference = _mle(
        lambda t: (
            -(
                stats.norm.logpdf(y[~low], t[0], np.exp(t[1])).sum()
                + low.sum() * stats.norm.logcdf(0.0, t[0], np.exp(t[1]))
            )
        ),
        [1.0, 0.5],
    )
    assert fitted["location"] == pytest.approx(reference[0], rel=1e-6)
    assert fitted["scale"] == pytest.approx(np.exp(reference[1]), rel=1e-6)


def test_generalized_pareto_censored_at_a_limit_keeps_its_tail_shape():
    rng = np.random.default_rng(548)
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(20_000, random_state=rng)
    limit = 4000.0
    paid = np.minimum(excess, limit)
    hit = paid == limit
    fitted = _fit(GeneralizedParetoLSS(), paid, censoring=(None, limit))
    reference = _mle(
        lambda t: (
            -(
                stats.genpareto.logpdf(paid[~hit], t[1], scale=np.exp(t[0])).sum()
                + hit.sum() * stats.genpareto.logsf(limit, t[1], scale=np.exp(t[0]))
            )
        ),
        [7.0, 0.2],
    )
    assert fitted["scale"] == pytest.approx(np.exp(reference[0]), rel=1e-5)
    assert fitted["shape"] == pytest.approx(reference[1], rel=1e-4)
    assert _fit(GeneralizedParetoLSS(), paid)["shape"] < 0.5 * fitted["shape"]


def test_frequency_weights_replicate_bounded_rows():
    rng = np.random.default_rng(549)
    y = np.exp(rng.normal(0.0, 1.0, 3_000))
    y = y[y > 0.05]
    limit = float(np.quantile(y, 0.8))
    paid = np.minimum(y, limit)
    counts = rng.integers(1, 4, len(paid)).astype(float)
    family = LogNormalLS(parametrisation="location")
    weighted = _fit(
        family, paid, weight_semantics="frequency", sample_weight=counts,
        censoring=(None, limit), truncation=(0.05, None),
    )  # fmt: skip
    expanded = np.repeat(paid, counts.astype(int))
    repeated = _fit(family, expanded, censoring=(None, limit), truncation=(0.05, None))
    for name in ("location", "scale"):
        assert weighted[name] == pytest.approx(repeated[name], rel=1e-7)


def test_per_row_bounds_and_zero_weight_rows_follow_the_retained_rows():
    rng = np.random.default_rng(550)
    y = np.exp(rng.normal(1.0, 0.8, 4_000))
    limits = np.where(np.arange(len(y)) % 2 == 0, 3.0, 6.0)
    paid = np.minimum(y, limits)
    weights = np.ones(len(y))
    weights[:10] = 0.0
    # Dropped rows may carry any bounds; their values are never read.
    limits_with_junk = limits.copy()
    limits_with_junk[:10] = 0.5
    family = LogNormalLS(parametrisation="location")
    dropped = _fit(family, paid, sample_weight=weights, censoring=(None, limits_with_junk))
    reference = _fit(family, paid[10:], censoring=(None, limits[10:]))
    assert dropped == pytest.approx(reference, rel=1e-10)


# -- smoothing, inference and persistence ------------------------------------


def _smooth_problem(n=6_000, seed=551):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.exp(rng.normal(1.0 + np.sin(5.0 * x), 0.6))
    limit = float(np.quantile(y, 0.9))
    return pd.DataFrame({"x": x}), np.minimum(y, limit), limit


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
def test_fit_reml_with_censored_rows_differences_the_laml_derivatives(outer):
    X, paid, limit = _smooth_problem()
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censoring=(None, limit), outer=outer)
    state = model._require_fitted().fit_state
    assert state.censored_row_count == int(np.count_nonzero(paid == limit))
    assert state.truncated_row_count == 0
    # The analytic third-derivative hook is complete-row only and refuses.
    rows = state.retained_rows
    plan = family.bind_likelihood(rows.response, rows.likelihood_weights, rows.observation)
    links = (IdentityLink(), LowerBoundedLogLink(family.scale_floor))
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.predictor_curvature_directional_derivative(
            rows.response, rows.fitted_eta, rows.fitted_eta, links, plan
        )
    complete = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complete.fit_reml(X, paid, outer=outer)
    grid = pd.DataFrame({"x": np.linspace(0.05, 0.95, 7)})
    # Censoring lifts the fitted curve where the limit bites.
    assert np.all(
        model.predict_parameters(grid)["scale"].to_numpy()
        > complete.predict_parameters(grid)["scale"].to_numpy()
    )


def test_bounded_fit_survives_pickling_and_replays_its_corrected_covariance():
    X, paid, limit = _smooth_problem(n=3_000, seed=552)
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censoring=(None, limit))
        restored = pickle.loads(pickle.dumps(model))
        draws = restored.posterior_draws(50, covariance="corrected")
    assert draws.covariance_kind == "corrected"
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))


def test_discrete_gaussian_binds_bounded_rows_eagerly():
    rng = np.random.default_rng(553)
    x = rng.uniform(0.0, 1.0, 6_000)
    y = rng.normal(np.sin(4.0 * x), 0.5)
    keep = y > -0.5
    X, y = pd.DataFrame({"x": x[keep]}), y[keep]
    family = GaussianLS()
    fits = []
    for discrete in (False, True):
        model = SuperLSS(family, family.location(s("x")), family.scale(), discrete=discrete)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(X, y, truncation=(-0.5, None))
        fits.append(model.predict(X))
    np.testing.assert_allclose(fits[0], fits[1], rtol=0.0, atol=2e-2)


# -- contract and input validation ------------------------------------------


def _bounds(lower, upper, side):
    return ResponseBounds.from_arrays(
        np.asarray(lower, float), np.asarray(upper, float), np.asarray(side)
    )


def test_bounded_plans_hash_their_bounds_and_slice_them():
    family = LogNormalLS()
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = resolve_likelihood_weights(None, n_observations=4, contract=WeightContract("prior"))
    bounds = _bounds([-np.inf] * 4, [5.0, 5.0, 3.0, 6.0], [0, 0, 1, 0])
    plan = family.bind_likelihood(y, weights, bounded_observation(bounds))
    complete = family.bind_likelihood(y, weights, COMPLETE_OBSERVATION)
    other = family.bind_likelihood(
        y, weights, bounded_observation(_bounds([-np.inf] * 4, [5.0, 5.0, 3.0, 7.0], [0, 0, 1, 0]))
    )
    assert len({plan.plan_identifier, complete.plan_identifier, other.plan_identifier}) == 3
    # A censored row has no density, so no parameter-free carrier either.
    assert plan.parameter_independent_carrier[2] == 0.0
    assert complete.parameter_independent_carrier[2] != 0.0
    child = plan.take(np.array([2, 3]))
    assert child.observation.bounds == _bounds([-np.inf] * 2, [3.0, 6.0], [1, 0])
    with pytest.raises(UnsupportedLikelihoodContractError):
        family.expected_information_natural(np.ones((4, 2)), plan)


def test_families_without_bounds_refuse_them_plainly():
    with pytest.raises(UnsupportedLikelihoodContractError, match="does not fit truncated"):
        _fit(GammaLS(), np.array([1.0, 2.0, 3.0, 4.0]), truncation=(None, 5.0))


def test_fisher_coefficient_curvature_is_refused_for_bounded_rows():
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale(), coefficient_curvature="fisher")
    y = np.linspace(-1.0, 1.0, 50)
    with pytest.raises(ValueError, match="truncated or censored rows"):
        model.fit(_intercepts(50), y, truncation=(-2.0, 2.0))


def test_bounds_that_change_no_row_leave_the_fit_complete():
    rng = np.random.default_rng(554)
    y = rng.normal(0.0, 1.0, 500)
    family = GaussianLS()
    open_fit = SuperLSS(family, family.location(), family.scale()).fit(
        _intercepts(500), y, censoring=(None, 99.0), truncation=(None, None)
    )
    plain = SuperLSS(family, family.location(), family.scale()).fit(_intercepts(500), y)
    assert (
        open_fit._require_fitted().fit_state.family_likelihood_plan_identifier
        == plain._require_fitted().fit_state.family_likelihood_plan_identifier
    )


@pytest.mark.parametrize(
    ("truncation", "censoring", "message"),
    [
        ((None, 1.5), None, "outside the truncation bounds"),
        ((2.0, 1.0), None, "lower bound at or above"),
        (None, (None, 1.5), "beyond a censoring bound"),
        ((None, np.nan), None, "missing"),
        ((None, 3.0), (None, 3.0), "right-censored at or above"),
        ((None, [3.0, 3.0]), None, "one number per row"),
        ((1.0,), None, "pair"),
    ],
)
def test_invalid_bounds_name_the_problem(truncation, censoring, message):
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises((ValueError, TypeError), match=message):
        _fit(GaussianLS(), y, truncation=truncation, censoring=censoring)


def test_resolution_reads_a_response_at_its_bound_as_censored():
    y = np.array([1.0, 2.0, 5.0, 0.0])
    arrays = (
        np.full(4, -np.inf),
        np.full(4, np.inf),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([5.0, 5.0, 5.0, 5.0]),
    )
    bounds = resolve_response_bounds(y, arrays, np.arange(4))
    assert bounds is not None
    np.testing.assert_array_equal(bounds.side, [0, 0, 1, -1])


# -- diagnostics read bounded rows ------------------------------------------


def _bounded_model(y, **bounds) -> SuperLSS:
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(y)), y, **bounds)
    return model


def _truncated_body(n: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(8.0, 1.2, n))
    u = float(np.exp(8.0 + 1.2 * stats.norm.ppf(0.9)))
    return y[y <= u], u


def test_truncated_pit_is_uniform_only_when_the_bounds_are_passed():
    body, u = _truncated_body(5_600, seed=560)
    model = _bounded_model(body, truncation=(None, u))
    X = _intercepts(len(body))

    read = model.residuals(X, body, kind="pit", truncation=(None, u))
    assert stats.kstest(read, "uniform").pvalue > 1e-3
    with pytest.warns(UserWarning, match="reads every row it is given as complete"):
        unread = model.residuals(X, body, kind="pit")
    # Read as complete, a body cut at the 90th percentile stops short of 0.9.
    assert stats.kstest(unread, "uniform").pvalue < 1e-6


def test_censored_rows_are_randomised_within_their_interval():
    rng = np.random.default_rng(561)
    y = np.exp(rng.normal(8.0, 1.2, 4_000))
    limit = float(np.quantile(y, 0.8))
    paid = np.minimum(y, limit)
    hit = paid == limit
    model = _bounded_model(paid, censoring=(None, limit))
    X = _intercepts(len(paid))

    payload = model.residual_set(X, paid, censoring=(None, limit))
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    at_limit = stats.norm.cdf(np.log(limit), row["location"], row["scale"])
    assert payload.randomised_rows == hit.sum()
    assert np.all(payload.pit[hit] >= at_limit - 1e-12)
    assert np.all(payload.pit[~hit] == payload.pit_upper[~hit])
    assert stats.kstest(payload.pit, "uniform").pvalue > 1e-3
    # The Q-Q envelope validates the same bounded evaluation.
    qq = model.plot_data("qq", X=X, y=paid, censoring=(None, limit), n_sim=20)
    assert qq["n_rows"] == len(paid)


def test_bounded_log_score_is_the_bounded_log_likelihood():
    body, u = _truncated_body(3_000, seed=562)
    cap = float(np.quantile(body, 0.7))
    paid = np.minimum(body, cap)
    capped = paid == cap
    bounds = {"truncation": (None, u), "censoring": (None, cap)}
    model = _bounded_model(paid, **bounds)
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    mu, sigma = row["location"], row["scale"]

    scores = model.scores(_intercepts(len(paid)), paid, which=("log",), **bounds)["log"]
    window = stats.norm.logcdf(np.log(u), mu, sigma)
    observed = stats.norm.logpdf(np.log(paid), mu, sigma) - np.log(paid) - window
    beyond = np.log(stats.norm.cdf(np.log(u), mu, sigma) - stats.norm.cdf(np.log(cap), mu, sigma))
    expected = -np.where(capped, beyond - window, observed)
    np.testing.assert_allclose(scores.to_numpy(), expected, rtol=1e-10, atol=1e-10)


def test_crps_refuses_bounded_rows_and_names_the_log_score():
    body, u = _truncated_body(500, seed=563)
    model = _bounded_model(body, truncation=(None, u))
    X = _intercepts(len(body))
    with pytest.raises(NotImplementedError, match="log score supports"):
        model.scores(X, body, truncation=(None, u))
    with pytest.raises(NotImplementedError, match="log score supports"):
        model.scores(X, body, which=("log",), thresholds=(1.0,), truncation=(None, u))
    assert np.all(np.isfinite(model.scores(X, body, which=("log",), truncation=(None, u))["log"]))


def test_diagnostics_warn_once_when_a_bounded_fit_gets_no_bounds():
    body, u = _truncated_body(800, seed=564)
    X = _intercepts(len(body)).assign(x=np.linspace(0.0, 1.0, len(body)))
    bounded = _bounded_model(body, truncation=(None, u))
    complete = _bounded_model(body)

    with pytest.warns(UserWarning, match="fitted with .* truncated and 0 censored rows") as seen:
        bounded.check(X, body, "x", truncation=None)
    assert len(seen) == 1
    with pytest.warns(UserWarning, match="actual_expected reads every row"):
        bounded.actual_expected(X, body, "x")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bounded.check(X, body, "x", truncation=(None, u))
        complete.check(X, body, "x")
        complete.residuals(X, body)


# -- review regressions -------------------------------------------------------


def _intercept_gaussian(n=2_000, seed=560):
    rng = np.random.default_rng(seed)
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale())
    model.fit(_intercepts(n), rng.normal(0.0, 1.0, n))
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    return model, float(row["location"]), float(row["scale"])


@pytest.mark.parametrize(("start", "position"), [(9.0, 9.01), (9.0, 9.1), (8.0, 8.1), (5.0, 5.1)])
def test_pit_of_a_row_truncated_far_in_the_upper_tail_keeps_its_window(start, position):
    # The window's mass is a difference of CDF values that both round to one
    # beyond about 8 sd; the PIT must come from the survival side instead.
    model, mu, sigma = _intercept_gaussian()
    lower, y = mu + start * sigma, mu + position * sigma
    residuals = model.residual_set(_intercepts(1), np.array([y]), truncation=(lower, None))
    law = stats.norm(mu, sigma)
    expected = 1.0 - law.sf(y) / law.sf(lower)
    assert residuals.pit[0] == pytest.approx(expected, rel=1e-9)


def test_bounds_with_no_effect_on_the_support_leave_the_fit_complete():
    y = np.exp(np.linspace(-1.0, 1.0, 30))
    family = LogNormalLS(parametrisation="location")

    def fitted(**kwargs):
        model = SuperLSS(family, family.location(), family.scale(), coefficient_curvature="fisher")
        return model.fit(_intercepts(30), y, **kwargs)

    no_op = fitted(truncation=(0.0, None))
    plain = fitted()
    state = no_op._require_fitted().fit_state
    assert state.truncated_row_count == 0
    assert (
        state.family_likelihood_plan_identifier
        == plain._require_fitted().fit_state.family_likelihood_plan_identifier
    )
    # Scoring reads the bounds on the support too, so the CRPS stays available.
    table = no_op.scores(_intercepts(30), y, truncation=(0.0, None))
    assert np.all(np.isfinite(table["crps"].to_numpy()))


def test_rows_with_zero_probability_on_the_support_are_refused():
    excess = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    with pytest.raises(ValueError, match="zero probability"):
        _fit(GeneralizedParetoLSS(), excess, censoring=(0.0, None))


def test_parameter_uncertainty_envelope_refuses_bounded_rows():
    X, paid, limit = _smooth_problem(n=1_500, seed=561)
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censoring=(None, limit))
    with pytest.raises(NotImplementedError, match="parameter_uncertainty=True"):
        model.plot_data("qq", X=X, y=paid, censoring=(None, limit), parameter_uncertainty=True)
    payload = model.plot_data("qq", X=X, y=paid, censoring=(None, limit))
    assert payload is not None


def test_mills_gap_agrees_with_the_direct_ratio_between_three_and_six():
    # Below -3 the gap comes from the continued fraction; down to -6 the direct
    # phi/Phi ratio cancels by under two digits, so it is an independent check.
    x = -np.linspace(3.0 + 1e-9, 6.0, 25)
    _, _, gap = inverse_mills(x)
    direct = x + np.exp(-0.5 * x * x - 0.5 * np.log(2.0 * np.pi) - special.log_ndtr(x))
    np.testing.assert_allclose(gap, direct, rtol=1e-12)


@pytest.mark.parametrize(
    ("lower", "upper", "shape"),
    [(50.0, np.inf, 0.2), (900.0, np.inf, 0.35), (50.0, 400.0, 0.2), (0.0, 700.0, 0.6)],
)
def test_generalized_pareto_window_derivatives_match_differences(lower, upper, shape):
    # (50, 0.2) stays inside the |z| < 0.5 series; (900, 0.35) and the upper
    # ends reach the positive-shape tail branch.
    def channels(scale, xi):
        return window_rows(
            np.array([lower]), np.array([upper]), np.array([scale]), np.array([xi]),
            derivative_order=2,
        )  # fmt: skip

    scale = 120.0
    _, score, hessian = channels(scale, shape)
    for column, (ds, dx) in enumerate(((1e-4 * scale, 0.0), (0.0, 1e-6))):
        step = ds + dx
        plus, minus = channels(scale + ds, shape + dx), channels(scale - ds, shape - dx)
        np.testing.assert_allclose(
            score[0, column], (plus[0][0] - minus[0][0]) / (2 * step), rtol=1e-7, atol=1e-12
        )
        packed = (0, 1) if column == 0 else (1, 2)
        np.testing.assert_allclose(
            hessian[0, list(packed)], (plus[1][0] - minus[1][0]) / (2 * step), rtol=1e-6,
            atol=1e-12,
        )  # fmt: skip


def test_censored_fit_standard_errors_are_the_inverse_observed_information():
    rng = np.random.default_rng(562)
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(8_000, random_state=rng)
    limit = 4000.0
    paid = np.minimum(excess, limit)
    hit = paid == limit
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(paid)), paid, censoring=(None, limit))
    links = [parameter.default_link for parameter in family.parameters]
    eta = np.asarray(model.predict_link(_intercepts(1)), dtype=np.float64)[0]

    def negative(point):
        scale, xi = (float(link.inverse(np.array([value]))[0]) for link, value in zip(links, point))
        return -(
            stats.genpareto.logpdf(paid[~hit], xi, scale=scale).sum()
            + hit.sum() * stats.genpareto.logsf(limit, xi, scale=scale)
        )

    h = 1e-4
    hessian = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * h, np.eye(2)[j] * h
            hessian[i, j] = (
                negative(eta + ei + ej) - negative(eta + ei - ej)
                - negative(eta - ei + ej) + negative(eta - ei - ej)
            ) / (4 * h * h)  # fmt: skip
    np.testing.assert_allclose(model.covariance_, np.linalg.inv(hessian), rtol=2e-3)


def test_pit_of_left_censored_and_prior_weighted_bounded_rows():
    rng = np.random.default_rng(563)
    weights = rng.choice([0.5, 2.0], 3_000)
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale())
    model.fit(_intercepts(3_000), rng.normal(0.0, 1.0 / np.sqrt(weights)), sample_weight=weights)
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    mu, sigma = float(row["location"]), float(row["scale"])
    y = np.array([-0.5, 0.2, 1.1])
    w = np.array([2.0, 0.5, 2.0])
    residuals = model.residual_set(
        _intercepts(3), y, sample_weight=w, censoring=(-0.5, None), truncation=(-2.0, 1.5)
    )
    law = [stats.norm(mu, sigma / np.sqrt(value)) for value in w]
    mass = [d.cdf(1.5) - d.cdf(-2.0) for d in law]
    edges = [(d.cdf(v) - d.cdf(-2.0)) / m for d, v, m in zip(law, y, mass, strict=True)]
    # Row 0 is left-censored at -0.5: its PIT is drawn on (0, edge).
    assert residuals.pit_lower[0] == 0.0
    assert residuals.pit_upper[0] == pytest.approx(edges[0], rel=1e-10)
    assert 0.0 < residuals.pit[0] < edges[0]
    np.testing.assert_allclose(residuals.pit[1:], edges[1:], rtol=1e-10)


@pytest.mark.parametrize(("lower", "upper"), [(-np.inf, 900.0), (2.0, np.inf), (300.0, 1500.0)])
def test_log_normal_mean_form_window_derivatives_match_differences(lower, upper):
    def channels(mean, sigma):
        return mean_window(
            np.array([lower]), np.array([upper]), np.array([mean]), np.array([sigma]),
            derivative_order=2,
        )  # fmt: skip

    mean, sigma = 700.0, 0.8
    _, score, hessian = channels(mean, sigma)
    for column, (dm, ds) in enumerate(((1e-4 * mean, 0.0), (0.0, 1e-5))):
        step = dm + ds
        plus, minus = channels(mean + dm, sigma + ds), channels(mean - dm, sigma - ds)
        np.testing.assert_allclose(
            score[0, column], (plus[0][0] - minus[0][0]) / (2 * step), rtol=1e-7, atol=1e-12
        )
        packed = (0, 1) if column == 0 else (1, 2)
        np.testing.assert_allclose(
            hessian[0, list(packed)], (plus[1][0] - minus[1][0]) / (2 * step), rtol=1e-6,
            atol=1e-12,
        )  # fmt: skip


def test_censored_generalized_pareto_fit_starts_without_a_warning():
    rng = np.random.default_rng(564)
    paid = np.minimum(stats.genpareto(c=0.3, scale=1000.0).rvs(2_000, random_state=rng), 1500.0)
    family = GeneralizedParetoLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeneralizedParetoInitializationWarning)
        SuperLSS(family, family.scale(), family.shape()).fit(
            _intercepts(len(paid)), paid, censoring=(None, 1500.0)
        )
