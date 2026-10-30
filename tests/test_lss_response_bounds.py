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
from scipy import optimize, stats

from superglm import GammaLS, GaussianLS, GeneralizedParetoLSS, LogNormalLS, SuperLSS, s
from superglm.distributional.families.gaussian import LowerBoundedLogLink
from superglm.distributional.family import COMPLETE_OBSERVATION, bounded_observation
from superglm.distributional.kernels._response_tails import (
    inverse_mills,
    normal_window_channels,
)
from superglm.distributional.kernels.generalized_pareto import window_rows
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
