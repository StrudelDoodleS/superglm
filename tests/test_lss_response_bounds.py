"""Truncated and censored rows in SuperLSS fits (issue 544).

Fits are checked against direct maximum likelihood in scipy on the truncated
or censored likelihood, an independent reference; the complete-row fit of the
same rows is the unfixed behaviour each fit test also rejects.
"""

from __future__ import annotations

import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

from superglm import (
    GammaLS,
    GaussianLS,
    GeneralizedParetoLSS,
    LogNormalLS,
    SeparationError,
    SeparationWarning,
    SuperLSS,
    cat,
    distributional,
    s,
)
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
from superglm.distributional.kernels.log_normal import location_window, mean_window
from superglm.distributional.model import refit_dense_distributional
from superglm.distributional.observation import (
    ResponseBounds,
    resolve_response_bounds,
    unvalidated_bound_arrays,
)
from superglm.distributional.weights import (
    UnsupportedLikelihoodContractError,
    WeightContract,
    resolve_likelihood_weights,
)
from superglm.links import IdentityLink

_U = 2.0**-53


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
    hit = y >= limit
    family = LogNormalLS(parametrisation="location")

    fitted = _fit(family, paid, censored=hit)
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
    low = latent <= 0.0
    fitted = _fit(GaussianLS(), y, censored=np.where(low, "left", "none"))
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
    hit = excess >= limit
    fitted = _fit(GeneralizedParetoLSS(), paid, censored=hit)
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


def test_the_flag_decides_which_rows_are_censored_not_equality():
    # A capped payment computed as paid / share can land a rounding short of
    # its limit, and a complete claim can sit exactly on a round limit.
    rng = np.random.default_rng(572)
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(6_000, random_state=rng)
    limit = 4000.0
    capped = excess >= limit
    paid = np.minimum(excess, limit)
    short = np.flatnonzero(capped)[::2]
    paid[short] = np.nextafter(limit, 0.0)
    at_limit = np.flatnonzero(~capped)[:5]
    paid[at_limit] = limit
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(paid)), paid, censored=capped)
    assert model._require_fitted().fit_state.censored_row_count == int(capped.sum())
    reference = _mle(
        lambda t: (
            -(
                stats.genpareto.logpdf(paid[~capped], t[1], scale=np.exp(t[0])).sum()
                + stats.genpareto.logsf(paid[capped], t[1], scale=np.exp(t[0])).sum()
            )
        ),
        [7.0, 0.2],
    )
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    assert row["scale"] == pytest.approx(np.exp(reference[0]), rel=1e-5)
    assert row["shape"] == pytest.approx(reference[1], rel=1e-4)


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
        censored=y >= limit, truncation=(0.05, None),
    )  # fmt: skip
    expanded = np.repeat(paid, counts.astype(int))
    expanded_flags = np.repeat(y >= limit, counts.astype(int))
    repeated = _fit(family, expanded, censored=expanded_flags, truncation=(0.05, None))
    for name in ("location", "scale"):
        assert weighted[name] == pytest.approx(repeated[name], rel=1e-7)


def test_per_row_bounds_and_zero_weight_rows_follow_the_retained_rows():
    rng = np.random.default_rng(550)
    y = np.exp(rng.normal(1.0, 0.8, 4_000))
    limits = np.where(np.arange(len(y)) % 2 == 0, 3.0, 6.0)
    paid = np.minimum(y, limits)
    weights = np.ones(len(y))
    weights[:10] = 0.0
    censored = np.where(y >= limits, "right", "none").astype(object)
    # Dropped rows may carry any flags and bounds; their values are never read.
    flags_with_junk = censored.copy()
    flags_with_junk[:10] = [np.nan, None, pd.NA, "Right", 2.5, "x", 1, np.nan, None, "?"]
    lower_with_junk = np.where(np.arange(len(y)) < 10, np.nan, -np.inf)
    family = LogNormalLS(parametrisation="location")
    dropped = _fit(
        family, paid, sample_weight=weights, censored=flags_with_junk,
        truncation=(lower_with_junk, None),
    )  # fmt: skip
    reference = _fit(family, paid[10:], censored=censored[10:])
    assert dropped == pytest.approx(reference, rel=1e-10)


def test_flags_on_rows_the_fit_drops_are_never_read():
    # Rows 0 and 5 carry zero weight and invalid flags; the fit, its scores
    # and its residuals are those of the retained rows alone.
    rng = np.random.default_rng(573)
    y = rng.normal(0.0, 1.0, 400)
    paid = np.minimum(y, 1.0)
    flags = (y >= 1.0).astype(object)
    weights = np.ones(len(y))
    weights[[0, 5]] = 0.0
    keep = weights > 0
    junk = flags.copy()
    junk[[0, 5]] = [np.nan, "Right"]
    family = GaussianLS()
    dropped = _fit(family, paid, sample_weight=weights, censored=junk)
    assert dropped == pytest.approx(_fit(family, paid[keep], censored=flags[keep]), rel=1e-10)

    model = SuperLSS(family, family.location(), family.scale())
    model.fit(_intercepts(int(keep.sum())), paid[keep], censored=flags[keep])
    scores = model.scores(
        _intercepts(len(y)), paid, sample_weight=weights, censored=junk, which=("log",)
    )
    retained = model.scores(_intercepts(int(keep.sum())), paid[keep], censored=flags[keep],
                            which=("log",))  # fmt: skip
    np.testing.assert_allclose(scores["log"].to_numpy()[keep], retained["log"].to_numpy())
    residuals = model.residuals(_intercepts(len(y)), paid, sample_weight=weights, censored=junk)
    reference = model.residuals(_intercepts(int(keep.sum())), paid[keep], censored=flags[keep])
    np.testing.assert_allclose(residuals, reference)


# -- smoothing, inference and persistence ------------------------------------


def _smooth_problem(n=6_000, seed=551):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = np.exp(rng.normal(1.0 + np.sin(5.0 * x), 0.6))
    limit = float(np.quantile(y, 0.9))
    return pd.DataFrame({"x": x}), np.minimum(y, limit), y >= limit


@pytest.mark.parametrize("outer", ["efs", "efs+newton"])
def test_fit_reml_with_censored_rows_differences_the_laml_derivatives(outer):
    X, paid, censored = _smooth_problem()
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censored=censored, outer=outer)
    state = model._require_fitted().fit_state
    assert state.censored_row_count == int(np.count_nonzero(censored))
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


def test_a_censored_pareto_tail_with_a_smooth_scale_recovers_the_law_under_reml():
    rng = np.random.default_rng(544)
    n = 8_000
    x = rng.uniform(0.0, 3.0, n)

    def true_scale(values):
        return 1000.0 * np.exp(0.5 * np.sin(2.0 * values))

    excess = stats.genpareto(c=0.3, scale=true_scale(x)).rvs(n, random_state=rng)
    paid = np.minimum(excess, 6000.0)
    censored = excess > 6000.0
    X, grid = pd.DataFrame({"x": x}), pd.DataFrame({"x": np.linspace(0.1, 2.9, 15)})

    def fitted(flags):
        family = GeneralizedParetoLSS()
        model = SuperLSS(family, family.scale(s("x")), family.shape())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit_reml(X, paid, censored=flags)
        row = model.predict_parameters(grid)
        error = np.abs(row["scale"].to_numpy() / true_scale(grid["x"].to_numpy()) - 1.0)
        return float(row["shape"].iloc[0]), error

    shape, error = fitted(censored)
    # The shape's standard error here is about 0.017; 0.07 is about four of
    # them.  The smooth's pointwise sampling error is a few percent.
    assert abs(shape - 0.3) < 0.07
    assert error.max() < 0.2 and error.mean() < 0.07
    # Reading the capped rows as complete thins the tail well beyond that.
    complete_shape, complete_error = fitted(None)
    assert abs(complete_shape - 0.3) > 0.14
    assert complete_error.mean() > 2.0 * error.mean()


def test_bounded_fit_survives_pickling_and_replays_its_corrected_covariance():
    X, paid, censored = _smooth_problem(n=3_000, seed=552)
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censored=censored)
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
        _intercepts(500), y, censored=np.zeros(500, dtype=bool), truncation=(None, None)
    )
    plain = SuperLSS(family, family.location(), family.scale()).fit(_intercepts(500), y)
    assert (
        open_fit._require_fitted().fit_state.family_likelihood_plan_identifier
        == plain._require_fitted().fit_state.family_likelihood_plan_identifier
    )


@pytest.mark.parametrize(
    ("truncation", "censored", "message"),
    [
        ((None, 1.5), None, "outside the truncation bounds"),
        ((2.0, 1.0), None, "lower bound at or above"),
        ((None, np.nan), None, "missing"),
        ((None, 3.0), [False, False, True], "right-censored at or above"),
        ((None, [3.0, 3.0]), None, "one number per row"),
        ((1.0,), None, "pair"),
    ],
)
def test_invalid_bounds_name_the_problem(truncation, censored, message):
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises((ValueError, TypeError), match=message):
        _fit(GaussianLS(), y, truncation=truncation, censored=censored)


def test_resolution_takes_each_side_from_the_flag():
    y = np.array([1.0, 2.0, 5.0, 0.0])
    arrays = unvalidated_bound_arrays(None, ["none", False, "right", "left"], 4)
    bounds = resolve_response_bounds(y, arrays, np.arange(4))
    assert bounds is not None
    np.testing.assert_array_equal(bounds.side, [0, 0, 1, -1])
    np.testing.assert_array_equal(bounds.lower, np.full(4, -np.inf))
    np.testing.assert_array_equal(bounds.upper, np.full(4, np.inf))


@pytest.mark.parametrize(
    ("censored", "error", "message"),
    [
        ([True, False], ValueError, r"shape \(2,\) for 3 rows; pass one flag per row"),
        (["right", "Right", "none"], ValueError, r"unknown flag 'Right' .*first at input row 1"),
        ([True, "False", False], ValueError, r"unknown flag 'False' .*first at input row 1"),
        (np.array([1.0, np.nan, 0.0]), ValueError, r"missing \(NaN, NA or None\) .*input row 1"),
        (pd.Series([True, None, False], dtype="boolean"), ValueError, r"missing .*input row 1"),
        # A nullable column's missing values arrive as None: never "not censored".
        (pd.Series([True, None, False]), ValueError, r"missing .*first at input row 1"),
        (["right", None, "none"], ValueError, r"missing .*first at input row 1"),
        (np.array([1, 0, 0]), TypeError, "values, which do not say which rows are censored"),
        (np.array([1.0, 0.0, 0.0]), TypeError, "for example censored=\\(flag == 1\\)"),
        # A numeric list stays numeric, so it gets the hint numeric arrays get.
        ([0, 1, 0], TypeError, "for example censored=\\(flag == 1\\)"),
        ([True, 0.5, False], TypeError, "for example censored=\\(flag == 1\\)"),
        ([True, np.nan, False], ValueError, r"missing \(NaN, NA or None\) .*input row 1"),
    ],
)
def test_invalid_censored_flags_name_the_problem(censored, error, message):
    with pytest.raises(error, match=message):
        _fit(GaussianLS(), np.array([1.0, 2.0, 3.0]), censored=censored)


def test_a_mixed_list_of_flags_is_read_entry_by_entry():
    # NumPy would promote [False, "left"] to the strings ['False', 'left'].
    rng = np.random.default_rng(574)
    y = rng.normal(0.0, 1.0, 200)
    mixed = [False, "left", True, "none"] * 50
    labels = np.array(["none", "left", "right", "none"] * 50)
    family = GaussianLS()
    assert _fit(family, y, censored=mixed) == pytest.approx(
        _fit(family, y, censored=labels), rel=1e-12
    )
    assert _fit(family, y, censored=tuple(mixed)) == pytest.approx(
        _fit(family, y, censored=labels), rel=1e-12
    )


@pytest.mark.parametrize(
    ("y", "truncation", "censored", "message"),
    [
        ([1.0, 2.0, 3.0], (1.0, None), ["left", "none", "none"], "are left-censored at or below"),
        ([1.0, 2.0, 3.0], (None, 3.0), [False, False, True], "are right-censored at or above"),
        (
            [1.0, 2.0, np.inf],
            None,
            [False, False, True],
            "are censored but have no finite response",
        ),
    ],
)
def test_censored_rows_that_observed_nothing_are_refused(y, truncation, censored, message):
    with pytest.raises(ValueError, match=rf"1 row\(s\) {message}"):
        _fit(GaussianLS(), np.array(y), truncation=truncation, censored=censored)


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
    hit = y >= limit
    model = _bounded_model(paid, censored=hit)
    X = _intercepts(len(paid))

    payload = model.residual_set(X, paid, censored=hit)
    row = model.predict_parameters(_intercepts(1)).iloc[0]
    at_limit = stats.norm.cdf(np.log(limit), row["location"], row["scale"])
    assert payload.randomised_rows == hit.sum()
    assert np.all(payload.pit[hit] >= at_limit - 1e-12)
    assert np.all(payload.pit[~hit] == payload.pit_upper[~hit])
    assert stats.kstest(payload.pit, "uniform").pvalue > 1e-3
    # The Q-Q envelope validates the same bounded evaluation.
    qq = model.plot_data("qq", X=X, y=paid, censored=hit, n_sim=20)
    assert qq["n_rows"] == len(paid)


def test_bounded_log_score_is_the_bounded_log_likelihood():
    body, u = _truncated_body(3_000, seed=562)
    cap = float(np.quantile(body, 0.7))
    paid = np.minimum(body, cap)
    capped = body >= cap
    bounds = {"truncation": (None, u), "censored": capped}
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
        _fit(GeneralizedParetoLSS(), excess, censored=["left"] + ["none"] * 5)


def test_parameter_uncertainty_envelope_refuses_bounded_rows():
    X, paid, censored = _smooth_problem(n=1_500, seed=561)
    family = LogNormalLS(parametrisation="location")
    model = SuperLSS(family, family.location(s("x")), family.scale())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit_reml(X, paid, censored=censored)
    with pytest.raises(NotImplementedError, match="parameter_uncertainty=True"):
        model.plot_data("qq", X=X, y=paid, censored=censored, parameter_uncertainty=True)
    payload = model.plot_data("qq", X=X, y=paid, censored=censored)
    assert payload is not None


def test_mills_gap_agrees_with_an_erfcx_reference_between_three_and_forty():
    # Below -3 the gap comes from the continued fraction.  The reference forms
    # r = sqrt(2/pi)/erfcx(t/sqrt(2)) to a few u, and its gap r - t cancels by
    # a factor of t^2, so it is good to a small multiple of u t^2.
    t = np.linspace(3.0 + 1e-9, 40.0, 200)
    _, ratio, gap = inverse_mills(-t)
    reference = np.sqrt(2.0 / np.pi) / special.erfcx(t / np.sqrt(2.0))
    np.testing.assert_allclose(ratio, reference, rtol=16 * _U)
    assert np.all(np.abs(gap - (reference - t)) <= 16 * _U * t * t * np.abs(reference - t))


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
    hit = excess >= limit
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(len(paid)), paid, censored=hit)
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
        _intercepts(3),
        y,
        sample_weight=w,
        censored=["left", "none", "none"],
        truncation=(-2.0, 1.5),
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
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(2_000, random_state=rng)
    paid = np.minimum(excess, 1500.0)
    family = GeneralizedParetoLSS()
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeneralizedParetoInitializationWarning)
        SuperLSS(family, family.scale(), family.shape()).fit(
            _intercepts(len(paid)), paid, censored=excess >= 1500.0
        )


# -- review regressions, round 2 ----------------------------------------------


def _normal_window_reference(lower: float, h: float) -> float:
    # The mass of (m - h/2, m + h/2] is h phi(m) sum_j He_2j(m) (h/2)^2j/(2j+1)!,
    # since phi^(n) = (-1)^n He_n phi; the series is entire and converges fast
    # for h max(1, |m|) <= 1.  The width is passed, not differenced.
    m = lower + 0.5 * h
    series = math.fsum(
        special.eval_hermitenorm(2 * j, m) * (0.5 * h) ** (2 * j) / math.factorial(2 * j + 1)
        for j in range(25)
    )
    return math.log(h) - 0.5 * m * m - 0.5 * math.log(2.0 * math.pi) + math.log(series)


@pytest.mark.parametrize("centre", [0.0, 2.5, 12.0, 30.0])
def test_narrow_normal_windows_keep_their_mass_on_both_sides_of_the_switch(centre):
    # Widths times max(1, |centre|): a few float spacings, then either side of
    # the 0.05 switch to quadrature.
    reaches = (1e-16, 1e-9, 1e-3, 0.049, 0.051, 0.5)
    lower = np.full(len(reaches), centre)
    upper = np.array(
        [max(centre + r / max(1.0, centre), np.nextafter(centre, np.inf)) for r in reaches]
    )
    ones = np.ones(len(reaches))
    value, _, _ = normal_window_channels(lower, upper, 0.0 * ones, ones, ones, derivative_order=0)
    expected = [_normal_window_reference(lo, hi - lo) for lo, hi in zip(lower, upper, strict=True)]
    # Above the switch the difference of the two log tails keeps the mass to
    # about 10 u centre^2; below it the quadrature's error is under 2e-13.
    np.testing.assert_allclose(value, expected, rtol=0.0, atol=2e-13 + 40 * _U * centre**2)


@pytest.mark.parametrize("relative_width", [1e-14, 1e-10, 1e-6])
def test_narrow_log_normal_windows_keep_their_width(relative_width):
    # log(upper) - log(lower) would lose the width to the rounding of log(lower).
    lower = math.exp(5.0)
    upper = lower * (1.0 + relative_width)
    mu, sigma = 4.0, 0.7
    value, _, _ = location_window(
        np.array([lower]), np.array([upper]), np.array([mu]), np.array([sigma]),
        derivative_order=0,
    )  # fmt: skip
    z_lower = (math.log(lower) - mu) / sigma
    z_width = math.log1p((upper - lower) / lower) / sigma
    expected = _normal_window_reference(z_lower, z_width)
    assert value[0] == pytest.approx(expected, rel=1e-13)


def _pareto_window_reference(lower: float, upper: float, scale: float, shape: float) -> float:
    # f^(n)(y) = (-1)^n f(y) prod_{k<=n} (1 + k xi)/(scale + xi y)^n, so the mass
    # is w f(m) sum_j prod_{k<=2j} (1 + k xi) (rho/2)^2j/(2j+1)!, rho = w/(scale + xi m).
    w = upper - lower
    m = lower + 0.5 * w
    rho = w / (scale + shape * m)
    terms = [
        math.prod(1.0 + k * shape for k in range(1, 2 * j + 1))
        * (0.5 * rho) ** (2 * j)
        / math.factorial(2 * j + 1)
        for j in range(20)
    ]
    log_density = -math.log(scale) - (1.0 / shape + 1.0) * math.log1p(shape * m / scale)
    return math.log(w) + log_density + math.log(math.fsum(terms))


@pytest.mark.parametrize("shape", [0.05, 0.3, 0.9])
def test_narrow_generalized_pareto_windows_keep_their_mass(shape):
    scale = 100.0
    rows = [(lo, reach) for lo in (0.0, 10.0, 1e5) for reach in (1e-12, 1e-6, 0.04, 0.06, 0.2)]
    lower = np.array([lo for lo, _ in rows])
    upper = np.array([lo + reach * (scale + shape * lo) / (1.0 + shape) for lo, reach in rows])
    n = len(rows)
    value, _, _ = window_rows(
        lower, upper, np.full(n, scale), np.full(n, shape), derivative_order=0
    )
    expected = [
        _pareto_window_reference(lo, hi, scale, shape) for lo, hi in zip(lower, upper, strict=True)
    ]
    # The two log tails' difference keeps about u |log S|/0.05 here, the
    # quadrature about 1e-13.
    np.testing.assert_allclose(value, expected, rtol=0.0, atol=5e-13)


@pytest.mark.parametrize(
    ("kernel", "lower", "upper"),
    [
        ("normal", 2.0, 2.0 + 1e-3),
        ("normal", -0.8, -0.8 + 1e-9),
        ("normal", 2.0, 2.016),
        ("pareto", 50.0, 50.0 + 1e-3),
        ("pareto", 900.0, 900.0 + 1e-8),
        ("pareto", 50.0, 54.0),
    ],
)
def test_narrow_window_derivatives_match_differences(kernel, lower, upper):
    # (2.0, 2.016) and (50, 54) sit at about 0.04 of the 0.05 switch, where the
    # spread of the node scores is about 3e-5 of the Hessian.
    root = np.array([1.7])

    def channels(first, second):
        if kernel == "normal":
            return normal_window_channels(
                np.array([lower]), np.array([upper]), np.array([first]), np.array([second]),
                root, derivative_order=2,
            )  # fmt: skip
        return window_rows(
            np.array([lower]), np.array([upper]), np.array([first]), np.array([second]),
            derivative_order=2,
        )  # fmt: skip

    point = (0.3, 1.4) if kernel == "normal" else (120.0, 0.35)
    steps = (1e-5, 1e-5) if kernel == "normal" else (1e-4 * point[0], 1e-6)
    _, score, hessian = channels(*point)
    for column in range(2):
        step = np.eye(2)[column] * steps[column]
        plus, minus = channels(*(point + step)), channels(*(point - step))
        np.testing.assert_allclose(
            score[0, column], (plus[0][0] - minus[0][0]) / (2 * steps[column]), rtol=1e-7,
            atol=1e-12,
        )  # fmt: skip
        packed = (0, 1) if column == 0 else (1, 2)
        np.testing.assert_allclose(
            hessian[0, list(packed)], (plus[1][0] - minus[1][0]) / (2 * steps[column]),
            rtol=1e-6, atol=1e-12,
        )  # fmt: skip


def test_a_window_narrower_than_float_spacing_keeps_its_mass():
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale())
    model.fit(_intercepts(2), np.array([-1.0, 1.0]))
    window = (-1e-16, 1e-16)
    table = model.scores(_intercepts(1), np.array([0.0]), truncation=window, which=("log",))
    # The row's density cancels against the window's mass, which is the width
    # times that density to second order in the width.
    assert table["log"].iloc[0] == pytest.approx(np.log(2e-16), rel=1e-12)
    residuals = model.residual_set(_intercepts(1), np.array([0.0]), truncation=window)
    assert residuals.pit[0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (-np.inf, 1e200),
        (-np.inf, np.finfo(np.float64).max),
        (-np.finfo(np.float64).max, np.inf),
        (-1e200, 1e200),
    ],
)
def test_huge_finite_bounds_read_as_open_ends(lower, upper):
    value, score, hessian = normal_window_channels(
        np.array([lower]), np.array([upper]), np.array([0.3]), np.array([0.7]), np.ones(1),
        derivative_order=2,
    )  # fmt: skip
    np.testing.assert_array_equal(value, [0.0])
    np.testing.assert_array_equal(score, [[0.0, 0.0]])
    np.testing.assert_array_equal(hessian, [[0.0, 0.0, 0.0]])


@pytest.mark.parametrize("lower", [-1e100, -1e150, -5e153])
def test_a_far_end_with_a_negligible_tail_is_the_open_end(lower):
    # The far end's log tail is finite but its derivatives overflow; it
    # carries no weight, so the window is the one-sided one exactly.
    def channels(lo):
        return normal_window_channels(
            np.array([lo]), np.array([1.0]), np.array([0.3]), np.array([0.5]), np.ones(1),
            derivative_order=2,
        )  # fmt: skip

    for bounded, open_ended in zip(channels(lower), channels(-np.inf), strict=True):
        np.testing.assert_array_equal(bounded, open_ended)


@pytest.mark.parametrize("upper", [1e200, np.finfo(np.float64).max])
def test_fits_with_a_huge_finite_upper_bound_are_the_unbounded_fits(upper):
    y = np.linspace(-1.0, 1.0, 30)
    assert _fit(GaussianLS(), y, truncation=(None, upper)) == pytest.approx(
        _fit(GaussianLS(), y), rel=1e-8, abs=1e-12
    )
    excess = stats.genpareto(c=0.3, scale=100.0).rvs(500, random_state=np.random.default_rng(5))
    assert _fit(GeneralizedParetoLSS(), excess, truncation=(None, upper)) == pytest.approx(
        _fit(GeneralizedParetoLSS(), excess), rel=1e-6
    )


def test_zero_probability_rows_are_refused_by_input_row_on_either_branch():
    excess = np.array([0.5, 1.0, 0.0, 2.0, 0.0, 4.0])
    upper = np.array([np.inf, np.inf, np.inf, np.inf, 0.0, np.inf])
    weights = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    # Row 4 is truncated at zero from above; row 1 carries no weight, so the
    # bound rows are not the input rows.
    with pytest.raises(ValueError, match=r"zero probability \(first at input row 4\)"):
        _fit(
            GeneralizedParetoLSS(), excess, weight_semantics="frequency", sample_weight=weights,
            truncation=(None, upper),
        )  # fmt: skip
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_intercepts(4), np.array([0.5, 1.0, 2.0, 4.0]))
    for bounds in ({"truncation": (None, 0.0)}, {"censored": "left"}):
        with pytest.raises(ValueError, match="zero probability"):
            model.scores(_intercepts(1), np.array([0.0]), which=("log",), **bounds)


def test_a_bounded_row_whose_window_has_no_mass_scores_inf(monkeypatch):
    # No float64 input reaches a massless window here: the row's own density
    # overflows first.  The event is forced on the window kernel instead.
    from superglm.distributional.families import gaussian as gaussian_family

    real = gaussian_family.gaussian_window

    def massless(*args, **kwargs):
        value, score, hessian = real(*args, **kwargs)
        return np.full_like(value, -np.inf), score, hessian

    model, _, _ = _intercept_gaussian()
    monkeypatch.setattr(gaussian_family, "gaussian_window", massless)
    table = model.scores(
        _intercepts(2), np.array([0.0, 3.0]), censored=[False, True], which=("log",)
    )
    assert np.isfinite(table["log"].iloc[0])
    assert table["log"].iloc[1] == np.inf


def _all_censored_level(n=1_500, seed=567, keep_one=False):
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b", "c"], n)
    limit = 5000.0
    loss = np.exp(rng.normal(8.0, 1.0, n))
    censored = (loss >= limit) | (group == "c")
    paid = np.where(censored, limit, loss)
    if keep_one:
        first = np.flatnonzero(group == "c")[0]
        paid[first], censored[first] = 0.5 * limit, False
    return pd.DataFrame({"g": group}), paid, limit, censored


def test_a_level_censored_on_one_side_only_is_disclosed_as_separated():
    X, paid, limit, censored = _all_censored_level()
    family = LogNormalLS(parametrisation="location")
    with pytest.warns(
        SeparationWarning, match=r"'location:g': 1 of 3 occupied levels have only right-censored"
    ):
        SuperLSS(family, family.location(cat("g")), family.scale()).fit(X, paid, censored=censored)
    strict = SuperLSS(family, family.location(cat("g")), family.scale(), separation="error")
    with pytest.raises(SeparationError, match="Censoring separation"):
        strict.fit(X, paid, censored=censored)
    # Tobit: a level whose rows are all at the lower limit.
    gaussian = GaussianLS()
    with pytest.warns(SeparationWarning, match="only left-censored rows: 'c'"):
        SuperLSS(gaussian, gaussian.location(cat("g")), gaussian.scale()).fit(
            X, limit - paid, censored=np.where(censored, "left", "none")
        )
    X, paid, _, censored = _all_censored_level(keep_one=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        SuperLSS(family, family.location(cat("g")), family.scale()).fit(X, paid, censored=censored)


def test_a_level_censored_on_one_side_of_a_scale_predictor_keeps_its_estimate():
    # Group b's two rows are right-censored at -10 and 1 about a location set
    # by group a, so P(Y > c) peaks at a finite group-b scale: no separation.
    complete = np.tile([-1.0, 1.0], 10)
    limits = np.array([-10.0, 1.0])
    y = np.concatenate([complete, limits])
    censored = np.arange(len(y)) >= len(complete)
    X = pd.DataFrame({"g": ["a"] * len(complete) + ["b"] * len(limits)})
    family = GaussianLS()
    model = SuperLSS(family, family.location(), family.scale(cat("g")), separation="error")
    model.fit(X, y, censored=censored)

    def negative(point):
        mu, log_a, log_b = point
        return -(
            stats.norm.logpdf(complete, mu, np.exp(log_a)).sum()
            + stats.norm.logsf(limits, mu, np.exp(log_b)).sum()
        )

    mu, log_a, log_b = _mle(negative, np.array([0.0, 0.0, 1.0]))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    np.testing.assert_allclose(fitted["location"], [mu, mu], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(fitted["scale"], np.exp([log_a, log_b]), rtol=1e-5)


def _pareto_shape_levels(truncation_side):
    # Group a: 1,000 midpoint quantiles of GPD(0.3, 1); group b: 20 rows at 1,
    # right-censored there and truncated at 3 above or 0.5 below.
    group_a = stats.genpareto.ppf((np.arange(1_000) + 0.5) / 1_000, 0.3)
    y = np.concatenate([group_a, np.ones(20)])
    in_b = np.arange(len(y)) >= len(group_a)
    censored = in_b
    if truncation_side == "upper":
        truncation = (None, np.where(in_b, 3.0, np.inf))
    else:
        truncation = (np.where(in_b, 0.5, -np.inf), None)
    X = pd.DataFrame({"g": np.where(in_b, "b", "a")})
    return X, y, group_a, censored, truncation


def test_upper_truncation_anchors_a_censored_pareto_level():
    # (S(1) - S(3))/(1 - S(3)) peaks at an interior shape, so the level has an
    # estimate; with only a lower truncation bound it walks to the wall.
    X, y, group_a, censored, truncation = _pareto_shape_levels("upper")
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape(cat("g")), separation="error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralizedParetoInitializationWarning)
        model.fit(X, y, censored=censored, truncation=truncation)

    def negative(point):
        scale, xi_a, xi_b = np.exp(point[0]), point[1], point[2]
        sf = stats.genpareto(c=xi_b, scale=scale).sf
        return -(
            stats.genpareto.logpdf(group_a, xi_a, scale=scale).sum()
            + 20 * np.log((sf(1.0) - sf(3.0)) / (1.0 - sf(3.0)))
        )

    log_scale, xi_a, xi_b = _mle(negative, np.array([0.0, 0.3, 0.3]))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    np.testing.assert_allclose(fitted["shape"], [xi_a, xi_b], rtol=1e-4)
    np.testing.assert_allclose(fitted["scale"], np.exp([log_scale] * 2), rtol=1e-5)

    # Along the scale the bound does not anchor the level: P(Y > 1 | Y <= 3)
    # rises with psi towards 2/3.
    strict = SuperLSS(family, family.scale(cat("g")), family.shape(), separation="error")
    with pytest.raises(SeparationError, match=r"'scale:g': 1 of 2 occupied levels"):
        strict.fit(X, y, censored=censored, truncation=truncation)

    X, y, _, censored, truncation = _pareto_shape_levels("lower")
    with pytest.warns(SeparationWarning, match=r"'shape:g': 1 of 2 occupied levels"):
        SuperLSS(family, family.scale(), family.shape(cat("g"))).fit(
            X, y, censored=censored, truncation=truncation
        )


def test_a_left_censored_mean_form_scale_level_is_separated():
    # P(Y <= c) = Phi(sigma/2 - log(m/c)/sigma) nears one only as sigma grows,
    # whatever the mean, so the level's scale has no finite estimate.
    rng = np.random.default_rng(570)
    group = rng.choice(["a", "b", "c"], 900)
    y = np.exp(rng.normal(8.0, 1.0, 900))
    y[group == "c"] = 400.0
    family = LogNormalLS()
    with pytest.warns(
        SeparationWarning, match=r"'scale:g': 1 of 3 occupied levels have only left-censored"
    ):
        SuperLSS(family, family.mean(), family.scale(cat("g"))).fit(
            pd.DataFrame({"g": group}), y, censored=np.where(group == "c", "left", "none")
        )


def test_a_right_censored_mean_form_scale_level_below_its_limit_keeps_its_estimate():
    # P(Y > c) = Phi(log(m/c)/sigma - sigma/2) peaks at sigma = sqrt(2 log(c/m))
    # while the mean is below the limit, so the level's scale has an estimate.
    y_a = np.exp(8.0 + stats.norm.ppf((np.arange(200) + 0.5) / 200))
    limit = 15_000.0
    y = np.concatenate([y_a, np.full(20, limit)])
    group = np.array(["a"] * 200 + ["b"] * 20)
    family = LogNormalLS()
    model = SuperLSS(family, family.mean(), family.scale(cat("g")), separation="error")
    model.fit(pd.DataFrame({"g": group}), y, censored=group == "b")

    def negative(point):
        mean, scale_a, scale_b = np.exp(point)
        return -(
            stats.norm.logpdf(np.log(y_a), np.log(mean) - 0.5 * scale_a**2, scale_a).sum()
            + 20 * stats.norm.logsf(np.log(limit), np.log(mean) - 0.5 * scale_b**2, scale_b)
        )

    mean, scale_a, scale_b = np.exp(_mle(negative, np.array([8.5, 0.0, 0.3])))
    assert mean < limit
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    np.testing.assert_allclose(fitted["mean"], [mean, mean], rtol=1e-5)
    np.testing.assert_allclose(fitted["scale"], [scale_a, scale_b], rtol=1e-5)


def _deductible_scale_levels(anchored, side="right"):
    # Group a: complete rows at -1 and 1.  Group b: five rows at 1, right-censored
    # above a lower truncation bound 0, so P(Y > 1 | Y > 0) nears one only as the
    # scale grows (mirrored: rows at -1 left-censored below an upper bound 0).
    # Anchored, right: rows at -10 and 1 censored inside (-11, -9] and (0, 2],
    # whose probabilities pull the scale in opposite directions.  Anchored, left:
    # rows at -10 and 1 left-censored above lower bounds -11 and 0 only, whose
    # P(a < Y <= c | Y > a) tends to zero as the scale grows.
    complete = np.tile([-1.0, 1.0], 10)
    if anchored and side == "right":
        limits, lower, upper = np.array([-10.0, 1.0]), [-11.0, 0.0], [-9.0, 2.0]
    elif anchored:
        limits, lower, upper = np.array([-10.0, 1.0]), [-11.0, 0.0], [np.inf] * 2
    elif side == "right":
        limits, lower, upper = np.ones(5), [0.0] * 5, [np.inf] * 5
    else:
        limits, lower, upper = -np.ones(5), [-np.inf] * 5, [0.0] * 5
    n = len(complete)
    y = np.concatenate([complete, limits])
    truncation = (np.concatenate([np.full(n, -np.inf), lower]),
                  np.concatenate([np.full(n, np.inf), upper]))  # fmt: skip
    X = pd.DataFrame({"g": ["a"] * n + ["b"] * len(limits)})
    censored = np.where(np.arange(len(y)) >= n, side, "none")
    return X, y, censored, truncation, complete, limits, np.array(lower), np.array(upper)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("family", [GaussianLS(), LogNormalLS(parametrisation="location")])
def test_a_scale_level_censored_beyond_its_only_truncation_bound_is_separated(family, side):
    log_scale = isinstance(family, LogNormalLS)
    lift = np.exp if log_scale else (lambda values: values)
    X, y, censored, truncation, *_ = _deductible_scale_levels(anchored=False, side=side)
    with pytest.warns(
        SeparationWarning, match=rf"'scale:g': 1 of 2 occupied levels have only {side}-censored"
    ):
        SuperLSS(family, family.location(), family.scale(cat("g"))).fit(
            X, lift(y), censored=censored, truncation=tuple(lift(b) for b in truncation)
        )

    # Right: with both truncation bounds the probabilities tend to
    # (b - c)/(b - a) < 1.  Left: above a lower bound only they tend to zero.
    # Either way these two rows give the level a finite scale.
    X, y, censored, truncation, complete, limits, lower, upper = _deductible_scale_levels(
        True, side
    )
    model = SuperLSS(family, family.location(), family.scale(cat("g")), separation="error")
    model.fit(X, lift(y), censored=censored, truncation=tuple(lift(b) for b in truncation))

    def negative(point):
        mu, log_a, log_b = point
        window = stats.norm(mu, np.exp(log_b)).cdf
        inside = (
            window(upper) - window(limits) if side == "right" else window(limits) - window(lower)
        )
        return -(
            stats.norm.logpdf(complete, mu, np.exp(log_a)).sum()
            + np.log(inside / (window(upper) - window(lower))).sum()
        )

    mu, log_a, log_b = _mle(negative, np.array([0.0, 0.0, 1.0]))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    np.testing.assert_allclose(fitted["location"], [mu, mu], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(fitted["scale"], np.exp([log_a, log_b]), rtol=1e-5)


def test_a_bounded_scale_link_keeps_a_censored_level_its_estimate():
    # Group b's rows are right-censored above their lower truncation bounds, so
    # their probabilities near one only as the scale grows; a logit scale link
    # stops at one, and inside (0, 1) the level's likelihood peaks at a finite
    # scale.  The default log link reaches infinity, so there it is separated.
    group_a = np.tile([-0.05, 0.05], 500)
    limits, lower = np.array([-0.5, 0.001]), np.array([-1.0, 0.0])
    n = len(group_a)
    y = np.concatenate([group_a, limits])
    X = pd.DataFrame({"g": ["a"] * n + ["b"] * len(limits)})
    censored = np.arange(len(y)) >= n
    truncation = (np.concatenate([np.full(n, -np.inf), lower]), None)
    family = GaussianLS(scale_floor=0.0)
    model = SuperLSS(
        family, family.location(), family.scale(cat("g"), link="logit"), separation="error"
    )
    model.fit(X, y, censored=censored, truncation=truncation, inner_tol=1e-10)

    def negative(point):
        mu, scale_a, scale_b = point[0], *special.expit(point[1:])
        return -(
            stats.norm.logpdf(group_a, mu, scale_a).sum()
            + (stats.norm.logsf(limits, mu, scale_b) - stats.norm.logsf(lower, mu, scale_b)).sum()
        )

    mu, logit_a, logit_b = _mle(negative, np.array([0.0, -2.0, -1.0]))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    assert special.expit(logit_b) < 0.5
    np.testing.assert_allclose(fitted["location"], [mu, mu], atol=1e-8)
    np.testing.assert_allclose(fitted["scale"], special.expit([logit_a, logit_b]), rtol=1e-5)

    with pytest.warns(
        SeparationWarning, match=r"'scale:g': 1 of 2 occupied levels have only right-censored"
    ):
        SuperLSS(family, family.location(), family.scale(cat("g"))).fit(
            X, y, censored=censored, truncation=truncation
        )


def test_a_left_censored_mean_form_scale_level_below_an_upper_bound_is_separated():
    # P(Y <= c | Y <= b) also nears one only as sigma grows, whatever the mean.
    y_a = np.exp(8.0 + stats.norm.ppf((np.arange(200) + 0.5) / 200))
    y = np.concatenate([y_a, np.full(20, 400.0)])
    group = np.array(["a"] * 200 + ["b"] * 20)
    upper = np.where(group == "b", 1000.0, np.inf)
    family = LogNormalLS()
    with pytest.warns(
        SeparationWarning, match=r"'scale:g': 1 of 2 occupied levels have only left-censored"
    ):
        SuperLSS(family, family.mean(), family.scale(cat("g"))).fit(
            pd.DataFrame({"g": group}), y, censored=np.where(group == "b", "left", "none"),
            truncation=(None, upper),
        )  # fmt: skip
    # A lower bound keeps the probability below one as sigma grows, so the
    # level is not scanned; these limits, either side of the mean, anchor it.
    limits = np.repeat([20_000.0, 4_000.0], 10)
    y = np.concatenate([y_a, limits])
    in_b = group == "b"
    model = SuperLSS(family, family.mean(), family.scale(cat("g")), separation="error")
    model.fit(
        pd.DataFrame({"g": group}), y, censored=np.where(in_b, "left", "none"),
        truncation=(np.where(in_b, 1_000.0, -np.inf), np.where(in_b, 40_000.0, np.inf)),
    )  # fmt: skip

    def negative(point):
        mean, scale_a, scale_b = np.exp(point)
        cdf = stats.norm(np.log(mean) - 0.5 * scale_b**2, scale_b).cdf
        window = cdf(np.log(40_000.0)) - cdf(np.log(1_000.0))
        return -(
            stats.norm.logpdf(np.log(y_a), np.log(mean) - 0.5 * scale_a**2, scale_a).sum()
            + np.log((cdf(np.log(limits)) - cdf(np.log(1_000.0))) / window).sum()
        )

    mean, scale_a, scale_b = np.exp(_mle(negative, np.array([8.5, 0.0, 0.0])))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    np.testing.assert_allclose(fitted["mean"], [mean, mean], rtol=1e-5)
    np.testing.assert_allclose(fitted["scale"], [scale_a, scale_b], rtol=1e-5)


def test_every_row_censored_on_one_side_separates_the_intercept():
    # An intercept-only location has no categorical level to scan; its
    # intercept is the level holding every row.
    y = np.linspace(-1.0, 1.0, 40)
    family = GaussianLS()
    with pytest.warns(SeparationWarning, match=r"'location': every row is right-censored"):
        SuperLSS(family, family.location(), family.scale()).fit(_intercepts(40), y, censored=True)
    strict = SuperLSS(family, family.location(), family.scale(), separation="error")
    with pytest.raises(SeparationError, match="intercept"):
        strict.fit(_intercepts(40), y, censored=np.ones(40, dtype=bool))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        SuperLSS(family, family.location(), family.scale()).fit(
            _intercepts(40), y, censored=np.arange(40) > 0
        )


def test_a_reference_level_on_a_predictor_without_an_intercept_is_not_separated():
    # Level b is the reference of a scale predictor with no intercept, so its
    # scale is pinned and cannot run off, however its rows are censored.
    y_a = np.exp(np.linspace(-1.0, 1.0, 20))
    y = np.concatenate([y_a, np.full(5, 0.5)])
    group = np.array(["a"] * 20 + ["b"] * 5)
    family = LogNormalLS()
    model = SuperLSS(
        family,
        family.mean(),
        family.scale(cat("g", base="b"), intercept=False),
        separation="error",
    )
    model.fit(pd.DataFrame({"g": group}), y, censored=np.where(group == "b", "left", "none"))
    fitted = model.predict_parameters(pd.DataFrame({"g": ["a", "b"]}))
    pinned = float(fitted["scale"].iloc[1])

    def negative(point):
        mean, scale = np.exp(point)
        location = np.log(mean) - 0.5 * scale**2
        return -(
            stats.norm.logpdf(np.log(y_a), location, scale).sum()
            - np.log(y_a).sum()
            + 5 * stats.norm.logcdf(np.log(0.5), np.log(mean) - 0.5 * pinned**2, pinned)
        )

    mean, scale = np.exp(_mle(negative, np.array([0.0, np.log(0.5)])))
    np.testing.assert_allclose(fitted["mean"], [mean, mean], rtol=1e-5)
    assert float(fitted["scale"].iloc[0]) == pytest.approx(scale, rel=1e-5)

    # A non-reference level keeps its own column, so it still runs off.
    y = np.concatenate([y, np.full(5, 0.5)])
    group = np.concatenate([group, ["c"] * 5])
    with pytest.raises(SeparationError, match=r"only left-censored rows: 'c'") as raised:
        model.fit(pd.DataFrame({"g": group}), y, censored=np.where(group == "a", "none", "left"))
    assert "'b'" not in str(raised.value)


def test_an_intercept_free_interaction_pins_only_its_reference_cell():
    from superglm.features import Categorical
    from superglm.features.interaction import CategoricalInteraction
    from tests.bound_predictor_fixtures import model_from_templates

    rng = np.random.default_rng(571)
    n = 800
    X = pd.DataFrame({"g": rng.choice(["a", "b"], n), "h": rng.choice(["x", "y"], n)})
    limit = 5000.0
    loss = np.exp(rng.normal(8.0, 1.0, n))

    def fit(cell):
        g, h = cell
        censored = ((X.g == g).to_numpy() & (X.h == h).to_numpy()) | (loss >= limit)
        paid = np.where(censored, limit, loss)
        model_from_templates(
            family=LogNormalLS(parametrisation="location"),
            predictors=(
                distributional.Predictor(
                    "location",
                    {"g": Categorical(base="a"), "h": Categorical(base="x")},
                    interaction_specs={"g:h": CategoricalInteraction("g", "h")},
                    intercept=False,
                ),
                distributional.Predictor("scale", {}),
            ),
            separation="error",
        ).fit(X, paid, censored=censored)

    # Without an intercept only the (a, x) cell is pinned at the offset.
    fit(("a", "x"))
    with pytest.raises(SeparationError, match=r"'location:g:h': 1 of 4 occupied cells") as raised:
        fit(("a", "y"))
    assert "('a' x 'y')" in str(raised.value)


def test_a_wide_log_window_far_from_overflow_keeps_its_mass():
    # log1p((upper - lower)/lower) overflows here; the window is 9e-18 sd wide
    # in log space, and both log tails round to log(1/2).
    family = LogNormalLS(parametrisation="location")
    value = family.log_window_probability(1e-200, 1e200, np.array([[0.0, 1e20]]))
    z = math.log(1e200) / 1e20
    assert value[0] == pytest.approx(math.log(math.erf(z / math.sqrt(2.0))), rel=1e-12)


def test_a_crossed_cell_censored_on_one_side_is_separated():
    from superglm.features import Categorical
    from superglm.features.interaction import CategoricalInteraction
    from tests.bound_predictor_fixtures import model_from_templates

    rng = np.random.default_rng(569)
    n = 1_600
    X = pd.DataFrame({"g": rng.choice(["a", "b"], n), "h": rng.choice(["x", "y"], n)})
    limit = 5000.0
    loss = np.exp(rng.normal(8.0, 1.0, n))
    censored = ((X.g == "b").to_numpy() & (X.h == "y").to_numpy()) | (loss >= limit)
    paid = np.where(censored, limit, loss)
    model = model_from_templates(
        family=LogNormalLS(parametrisation="location"),
        predictors=(
            distributional.Predictor(
                "location",
                {"g": Categorical(), "h": Categorical()},
                interaction_specs={"g:h": CategoricalInteraction("g", "h")},
            ),
            distributional.Predictor("scale", {}),
        ),
    )
    # Only the (b, y) cell is all at the limit; its parents' levels are not.
    with pytest.warns(SeparationWarning, match=r"'location:g:h': 1 of 4 occupied cells") as seen:
        model.fit(X, paid, censored=censored)
    message = str(seen[0].message)
    assert "('b' x 'y')" in message and "'location:g'" not in message


def test_a_level_censored_on_one_side_of_a_pareto_shape_is_separated():
    # S(c) rises with the shape, so the level's shape walks to its wall.
    rng = np.random.default_rng(568)
    group = rng.choice(["a", "b"], 1_200)
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(1_200, random_state=rng)
    censored = (excess >= 3000.0) | (group == "b")
    paid = np.where(censored, 3000.0, excess)
    family = GeneralizedParetoLSS()
    model = SuperLSS(family, family.scale(), family.shape(cat("g")))
    with pytest.warns(SeparationWarning, match=r"'shape:g': 1 of 2 occupied levels"):
        model.fit(pd.DataFrame({"g": group}), paid, censored=censored)


def test_window_probabilities_at_or_below_the_support_are_empty():
    # Empty on an infinite end, or so far out that both log tails underflow.
    gaussian_lower = np.array([np.inf, -np.inf, 1e300, -np.inf])
    gaussian_upper = np.array([np.inf, -np.inf, np.inf, -1e300])
    np.testing.assert_array_equal(
        GaussianLS().log_window_probability(
            gaussian_lower, gaussian_upper, np.tile((0.0, 1.0), (4, 1))
        ),
        -np.inf,
    )
    windows = [(-2.0, -1.0), (-np.inf, 0.0), (0.0, 0.0), (3.0, 3.0), (np.inf, np.inf)]
    lower = np.array([lo for lo, _ in windows])
    upper = np.array([hi for _, hi in windows])
    for family, row in (
        (LogNormalLS(), (700.0, 0.8)),
        (LogNormalLS(parametrisation="location"), (6.2, 0.8)),
        (GeneralizedParetoLSS(), (120.0, 0.3)),
    ):
        theta = np.tile(row, (len(windows), 1))
        np.testing.assert_array_equal(
            family.log_window_probability(lower, upper, theta), np.full(len(windows), -np.inf)
        )
        whole = family.log_window_probability(np.array([-5.0, 0.0]), np.inf, theta[:2])
        np.testing.assert_array_equal(whole, [0.0, 0.0])
        below = family.log_window_probability(np.array([-5.0, 0.0]), 40.0, theta[:2])
        assert below[0] == below[1] < 0.0


def test_narrow_normal_windows_do_not_depend_on_the_origin():
    # A window and location sharing a large offset are the same law as both
    # at zero; nodes formed before the offset cancels round it away.
    def channels(origin):
        return normal_window_channels(
            np.array([origin]), np.array([origin + 2.0]), np.array([origin]), np.array([100.0]),
            np.ones(1), derivative_order=2,
        )  # fmt: skip

    for shifted, centred in zip(channels(1e16), channels(0.0), strict=True):
        np.testing.assert_allclose(shifted, centred, rtol=1e-13)


def test_window_probabilities_match_scipy_for_every_family_form():
    lower = np.array([-np.inf, 2.0, 300.0, 0.0])
    upper = np.array([900.0, np.inf, 1500.0, 40.0])
    sigma = 0.8
    mean = 700.0
    location = np.log(mean) - 0.5 * sigma**2
    law = stats.lognorm(s=sigma, scale=np.exp(location))
    expected = np.log(law.cdf(upper) - law.cdf(lower))
    # 1 - cdf(2) loses digits to a mass of 2e-12 below 2; logsf keeps them.
    expected[1] = law.logsf(lower[1])
    for family, row in (
        (LogNormalLS(), (mean, sigma)),
        (LogNormalLS(parametrisation="location"), (location, sigma)),
    ):
        value = family.log_window_probability(lower, upper, np.tile(row, (4, 1)))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
    pareto = stats.genpareto(c=0.3, scale=120.0)
    value = GeneralizedParetoLSS().log_window_probability(
        lower, upper, np.tile((120.0, 0.3), (4, 1))
    )
    np.testing.assert_allclose(
        value, np.log(pareto.sf(np.maximum(lower, 0.0)) - pareto.sf(upper)), rtol=1e-12
    )


def test_window_probability_takes_a_number_or_one_per_row():
    theta = np.tile((0.0, 1.0), (3, 1))
    family = GaussianLS()
    value = family.log_window_probability(-1.0, 1.0, theta)
    np.testing.assert_allclose(value, np.full(3, np.log(2.0 * stats.norm.cdf(1.0) - 1.0)))
    assert np.all(family.log_window_probability(0.5, 0.5, theta) == -np.inf)
    with pytest.raises(ValueError, match="one number per row"):
        family.log_window_probability(np.zeros(2), np.ones(2), theta)
    with pytest.raises(ValueError, match="lower <= upper"):
        family.log_window_probability(1.0, -1.0, theta)
    with pytest.raises(ValueError, match="must not be NaN"):
        family.log_window_probability(np.nan, 1.0, theta)


def test_bounds_contracts_are_public():
    names = {"CensoredSeparationFamily", "ResponseBounds", "ResponseWindowFamily"}
    assert names <= set(distributional.__all__)
    for family in (GaussianLS(), LogNormalLS(), GeneralizedParetoLSS()):
        assert isinstance(family, distributional.ResponseWindowFamily)
        assert isinstance(family, distributional.CensoredSeparationFamily)


def test_generalized_pareto_above_a_deductible_recovers_the_parent_law():
    excess = stats.genpareto(c=0.3, scale=1000.0).rvs(6_000, random_state=np.random.default_rng(8))
    deductible = 500.0
    kept = excess[excess > deductible]
    fitted = _fit(GeneralizedParetoLSS(), kept, truncation=(deductible, None))

    # By threshold stability the excess over the deductible is GPD with scale
    # scale + xi d, which the reference fits directly.
    def negative(point):
        return -stats.genpareto.logpdf(kept - deductible, point[1], scale=np.exp(point[0])).sum()

    log_excess_scale, xi = _mle(negative, np.array([np.log(1150.0), 0.3]))
    scale = np.exp(log_excess_scale) - xi * deductible
    assert fitted["scale"] == pytest.approx(scale, rel=1e-5)
    assert fitted["shape"] == pytest.approx(xi, rel=1e-4)
    complete = _fit(GeneralizedParetoLSS(), kept)
    assert abs(complete["scale"] - scale) > 0.1 * scale


def test_refit_takes_the_bounds_of_the_rows_it_fits():
    body, u = _truncated_body(800, seed=566)
    X = _intercepts(len(body))
    dense = _bounded_model(body, truncation=(None, u))._require_fitted()
    refit_dense_distributional(dense, X, body, truncation=(None, u))
    assert dense.fit_state.truncated_row_count == len(body)
    refit_dense_distributional(dense, X, body)
    assert dense.fit_state.truncated_row_count == 0


def test_compare_warns_when_either_model_is_bounded():
    body, u = _truncated_body(800, seed=565)
    X = _intercepts(len(body))
    bounded = _bounded_model(body, truncation=(None, u))
    complete = _bounded_model(body)
    for other in (bounded, bounded._require_fitted()):
        with pytest.warns(UserWarning, match="other was fitted with") as seen:
            complete.compare(other, X, body)
        assert len(seen) == 1


class _HiddenObservationPlan:
    """A bound Gaussian plan that does not expose its observation contract."""

    def __init__(self, plan):
        self._plan = plan

    @property
    def weights(self):
        return self._plan.weights

    @property
    def plan_identifier(self):
        return self._plan.plan_identifier

    def take(self, indices):
        return _HiddenObservationPlan(self._plan.take(indices))


class _HiddenObservationGaussian(GaussianLS):
    def bind_likelihood(self, y, weights, observation):
        return _HiddenObservationPlan(super().bind_likelihood(y, weights, observation))


def test_a_bounded_plan_must_expose_its_observation():
    family = _HiddenObservationGaussian()
    model = SuperLSS(family, family.location(), family.scale())
    with pytest.raises(UnsupportedLikelihoodContractError, match="plan.observation"):
        model.fit(_intercepts(30), np.linspace(-1.0, 1.0, 30), truncation=(None, 2.0))


def test_a_monotone_recession_flag_holds_on_a_non_default_link():
    # A censored row's bound probability is monotone in the location, so a level
    # whose rows are all right-censored runs off along any link onto the whole
    # location range; only the normal scale's flags depend on the link.
    X, paid, _, censored = _all_censored_level()
    family = GaussianLS()
    strict = SuperLSS(
        family, family.location(cat("g"), link="log"), family.scale(), separation="error"
    )
    with pytest.raises(SeparationError, match=r"'location:g': 1 of 3 occupied levels"):
        strict.fit(X, paid, censored=censored)


@pytest.mark.parametrize("names", ["scale", ("sigma",)])
def test_a_link_sensitive_declaration_names_the_familys_parameters(names):
    # A bare string or an unknown name would silently read every link's flags.
    X, paid, _, censored = _all_censored_level()

    class Declared(GaussianLS):
        censored_recession_link_sensitive = names

    family = Declared()
    model = SuperLSS(family, family.location(cat("g")), family.scale())
    with pytest.raises(ValueError, match="censored_recession_link_sensitive must be a tuple"):
        model.fit(X, paid, censored=censored)
