"""Contract tests for the LSS Q-Q, worm and PIT checking payloads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from superglm import Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional.checks import qq as qq_module
from superglm.distributional.checks.pit import PITPayload, pit_payload
from superglm.distributional.checks.qq import QQPayload, order_statistic_grid, qq_payload
from superglm.distributional.checks.worm import (
    WormPanel,
    WormPayload,
    q_statistics,
    worm_payload,
)
from superglm.distributional.families.gamma import GammaLS
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.families.tweedie import TweedieLSS
from superglm.distributional.model import fit_dense_distributional
from superglm.distributional.residuals import ResidualSet, compute_residuals
from superglm.distributional.result import DenseSolverConfig
from superglm.distributional.weights import WeightContract


def _order_statistic_grid(n: int) -> np.ndarray:
    return special.ndtri((np.arange(n) + 0.5) / n)


@pytest.fixture(scope="module")
def gaussian_case():
    """The true data-generating process: a smooth location and a smooth scale."""
    rng = np.random.default_rng(20260903)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    X = pd.DataFrame({"x": x})
    y = 0.6 * np.sin(2.4 * x) + scale * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8)}),
            Predictor("scale", {"x": Spline("cr", k=6)}),
        ],
    ).fit_reml(X, y)
    fitted = model._require_fitted()
    return fitted, X, y, compute_residuals(fitted, X, y)


@pytest.fixture(scope="module")
def gamma_case():
    """A gamma mean-cv fit on gamma responses, for a non-Gaussian PIT."""
    rng = np.random.default_rng(11)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    mean = np.exp(0.8 + 0.5 * np.sin(2.0 * x))
    cv2 = 0.36
    X = pd.DataFrame({"x": x})
    y = rng.gamma(1.0 / cv2, mean * cv2)
    model = SuperLSS(
        family=GammaLS(),
        predictors=[
            Predictor("mean", {"x": Spline("cr", k=8)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(X, y)
    fitted = model._require_fitted()
    return fitted, X, y, compute_residuals(fitted, X, y)


@pytest.fixture(scope="module")
def misspecified_case():
    """A Gaussian with a constant scale fitted to log-normal responses."""
    rng = np.random.default_rng(4242)
    n = 1500
    x = rng.uniform(-1.0, 1.0, n)
    X = pd.DataFrame({"x": x})
    y = np.exp(0.5 * x + 0.8 * rng.standard_normal(n))
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor("location", {"x": Spline("cr", k=8)}),
            Predictor("scale", {}),
        ],
    ).fit_reml(X, y)
    fitted = model._require_fitted()
    return fitted, X, y, compute_residuals(fitted, X, y)


@pytest.fixture(scope="module")
def frequency_case():
    """A small fit declaring replication weights rather than prior weights."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.uniform(-1.0, 1.0, n)
    X = pd.DataFrame({"x": x})
    y = 0.4 * x + 0.5 * rng.standard_normal(n)
    counts = rng.integers(1, 4, n).astype(np.float64)
    model = SuperLSS(
        family=GaussianLS(),
        weight_semantics="frequency",
        predictors=[Predictor("location", {"x": Spline("cr", k=4)}), Predictor("scale", {})],
    ).fit_reml(X, y, sample_weight=counts)
    fitted = model._require_fitted()
    residuals = compute_residuals(fitted, X, y, sample_weight=counts)
    return fitted, X, y, residuals, counts


@pytest.fixture(scope="module")
def burn_cost_case():
    """A Tweedie burn-cost fit whose prior weight is the policy's exposure.

    The rows are compound Poisson-gamma draws made without a production Tweedie
    evaluator: a Poisson claim count at the weighted rate and a gamma total
    over the claims, both scaled by the row's own exposure.
    """
    rng = np.random.default_rng(20260903)
    n = 600
    first = rng.uniform(-1.0, 1.0, n)
    second = rng.uniform(-1.0, 1.0, n)
    weights = rng.uniform(0.2, 1.0, n)
    mean = np.exp(0.4 + 0.5 * np.sin(np.pi * first))
    dispersion = np.exp(-0.2 + 0.3 * second)
    power = np.full(n, 1.5)
    tail_index, jump_index = 2.0 - power, power - 1.0
    rate = weights * mean**tail_index / (dispersion * tail_index)
    jump_scale = dispersion * jump_index * mean**jump_index / weights
    counts = rng.poisson(rate)
    y = np.zeros(n, dtype=np.float64)
    claimed = counts > 0
    y[claimed] = rng.gamma(
        (tail_index / jump_index)[claimed] * counts[claimed], jump_scale[claimed]
    )

    X = pd.DataFrame({"first": first, "second": second})
    fitted = fit_dense_distributional(
        X,
        y,
        family=TweedieLSS(),
        predictors=(
            Predictor("mean", {"first": Spline(kind="cr", n_knots=6)}),
            Predictor("dispersion", {"second": Spline(kind="cr", n_knots=5)}),
            Predictor("power", {}),
        ),
        weight_contract=WeightContract("prior"),
        sample_weight=weights,
        config=DenseSolverConfig(coefficient_curvature="observed", tolerance=1.0e-8),
        lambdas={"mean:first#wiggle": 1.0, "dispersion:second#wiggle": 1.0},
        discrete=False,
        chunk_size=None,
    )
    return fitted, X, y, compute_residuals(fitted, X, y, sample_weight=weights), weights


def _envelope_coverage(payload) -> float:
    inside = (payload.observed >= payload.envelope_lower) & (
        payload.observed <= payload.envelope_upper
    )
    return float(inside.mean())


def _stub_residuals(n: int = 8) -> ResidualSet:
    rng = np.random.default_rng(0)
    pit = rng.uniform(0.1, 0.9, n)
    return ResidualSet(
        pit=pit,
        quantile=special.ndtri(pit),
        theta=np.column_stack([np.zeros(n), np.ones(n)]),
        eta=np.column_stack([np.zeros(n), np.zeros(n)]),
        y=rng.standard_normal(n),
        weights=np.ones(n),
        prior_weights=np.ones(n),
        clipped_rows=0,
        randomised_rows=0,
        weight_semantics="prior",
    )


# --------------------------------------------------------------------------- Q-Q


def test_qq_payload_envelope_covers_the_true_model(gaussian_case):
    fitted, X, _, residuals = gaussian_case
    payload = qq_payload(fitted, residuals, n_sim=100, X=X)

    assert isinstance(payload, QQPayload)
    assert payload.kind == "qq" and payload.schema_version == 1
    assert payload.n_rows == 1500 and payload.n_sim == 100 and payload.seed == 42
    assert payload.subsampled is False
    assert np.allclose(payload.theoretical, _order_statistic_grid(1500), rtol=0, atol=1e-12)
    assert np.array_equal(payload.observed, np.sort(residuals.quantile))
    assert np.all(np.diff(payload.observed) >= 0.0)
    assert np.all(payload.envelope_lower <= payload.envelope_upper)
    inside = (payload.observed >= payload.envelope_lower) & (
        payload.observed <= payload.envelope_upper
    )
    assert inside.mean() >= 0.90
    for name in ("theoretical", "observed", "envelope_lower", "envelope_upper"):
        assert not getattr(payload, name).flags.writeable


def test_qq_payload_subsamples_the_envelope_grid_above_max_points(gaussian_case):
    fitted, X, _, residuals = gaussian_case
    payload = qq_payload(fitted, residuals, n_sim=40, max_points=200, X=X)

    assert payload.subsampled is True and payload.n_rows == 1500
    for name in ("theoretical", "observed", "envelope_lower", "envelope_upper"):
        assert len(getattr(payload, name)) == 200
    assert np.allclose(payload.theoretical, _order_statistic_grid(200), rtol=0, atol=1e-12)
    expected = np.interp(
        payload.theoretical, _order_statistic_grid(1500), np.sort(residuals.quantile)
    )
    assert np.allclose(payload.observed, expected, rtol=0, atol=1e-12)
    inside = (payload.observed >= payload.envelope_lower) & (
        payload.observed <= payload.envelope_upper
    )
    assert inside.mean() >= 0.85


def test_qq_payload_parameter_uncertainty_widens_the_envelope(gaussian_case):
    fitted, X, _, residuals = gaussian_case
    plug_in = qq_payload(fitted, residuals, n_sim=60, max_points=400, X=X)
    posterior = qq_payload(
        fitted, residuals, n_sim=60, max_points=400, X=X, parameter_uncertainty=True
    )
    plug_width = float(np.mean(plug_in.envelope_upper - plug_in.envelope_lower))
    posterior_width = float(np.mean(posterior.envelope_upper - posterior.envelope_lower))
    assert posterior_width > plug_width


def test_qq_payload_is_reproducible_and_json_round_trips(gaussian_case):
    fitted, X, _, residuals = gaussian_case
    first = qq_payload(fitted, residuals, n_sim=20, max_points=300, X=X)
    second = qq_payload(fitted, residuals, n_sim=20, max_points=300, X=X)
    assert np.array_equal(first.envelope_lower, second.envelope_lower)
    assert np.array_equal(first.envelope_upper, second.envelope_upper)

    payload = json.loads(json.dumps(first.to_json()))
    assert payload["kind"] == "qq" and payload["schema_version"] == 1
    assert payload["n_rows"] == 1500 and payload["subsampled"] is True
    assert np.allclose(payload["observed"], first.observed, rtol=0, atol=1e-12)
    assert np.allclose(payload["envelope_upper"], first.envelope_upper, rtol=0, atol=1e-12)


def test_qq_payload_validates_its_arguments(gaussian_case):
    fitted, X, _, residuals = gaussian_case
    with pytest.raises(TypeError, match="ResidualSet"):
        qq_payload(fitted, object(), X=X)
    with pytest.raises(ValueError, match="design frame"):
        qq_payload(fitted, residuals)
    with pytest.raises(ValueError, match="same rows"):
        qq_payload(fitted, residuals, X=X.iloc[:10])
    with pytest.raises(ValueError, match="n_sim"):
        qq_payload(fitted, residuals, n_sim=0, X=X)
    with pytest.raises(ValueError, match="max_points"):
        qq_payload(fitted, residuals, max_points=0, X=X)


def test_qq_payload_refuses_a_family_without_a_distribution_function():
    fitted = SimpleNamespace(family=object())
    with pytest.raises(NotImplementedError, match="distribution function"):
        qq_payload(fitted, _stub_residuals(), X=pd.DataFrame({"x": np.zeros(8)}))


def test_qq_payload_covers_a_gamma_fit(gamma_case):
    fitted, X, _, residuals = gamma_case
    payload = qq_payload(fitted, residuals, n_sim=60, X=X)
    inside = (payload.observed >= payload.envelope_lower) & (
        payload.observed <= payload.envelope_upper
    )
    assert inside.mean() >= 0.90


def test_qq_payload_replicates_frequency_weights(frequency_case):
    fitted, X, _, residuals, counts = frequency_case
    payload = qq_payload(fitted, residuals, n_sim=40, X=X)
    assert payload.n_rows == int(counts.sum())
    assert len(payload.observed) == int(counts.sum())


def test_qq_envelope_simulates_each_row_on_its_prior_weighted_law(burn_cost_case, monkeypatch):
    fitted, X, _, residuals, weights = burn_cost_case
    assert np.any(residuals.prior_weights != 1.0)
    assert np.array_equal(residuals.prior_weights, weights)
    # The exposure-weighted Tweedie has a point mass at zero, so the observed
    # transform is the randomised one and the envelope has to invert the same
    # event on the same law to be a band for it.
    assert residuals.randomised_rows > 0

    payload = qq_payload(fitted, residuals, n_sim=40, X=X)
    coverage = _envelope_coverage(payload)
    assert payload.n_rows == len(X)
    assert coverage >= 0.90

    # Simulating the responses on the unit-weight law -- a full year's exposure
    # for every policy -- while the observed residuals read the row's own law
    # bands the wrong model.
    unit_law_simulation = qq_module.posterior_predictive

    def _drop_the_prior_weight(*args, weights=None, **kwargs):
        return unit_law_simulation(*args, **kwargs)

    monkeypatch.setattr(qq_module, "posterior_predictive", _drop_the_prior_weight)
    unit = qq_payload(fitted, residuals, n_sim=40, X=X)
    assert _envelope_coverage(unit) < coverage


# -------------------------------------------------------------------------- worm


def test_worm_payload_bands_cover_the_true_model(gaussian_case):
    _, _, _, residuals = gaussian_case
    payload = worm_payload(residuals)

    assert isinstance(payload, WormPayload)
    assert payload.kind == "worm" and payload.schema_version == 1
    assert payload.covariate is None and len(payload.panels) == 1
    panel = payload.panels[0]
    assert isinstance(panel, WormPanel)
    assert panel.label == "all" and panel.n == 1500 and panel.interval is None
    assert np.allclose(panel.deviation, np.sort(residuals.quantile) - panel.z, atol=1e-12)
    assert len(panel.band_z) == 200 and len(panel.band) == 200
    assert panel.band_z[0] == pytest.approx(-3.0) and panel.band_z[-1] == pytest.approx(3.0)
    assert np.all(panel.band > 0.0)
    assert float(panel.band.min()) == pytest.approx(float(panel.band[len(panel.band) // 2]))
    probability = stats.norm.cdf(panel.band_z)
    van_buuren = (
        1.96 * np.sqrt(probability * (1.0 - probability) / panel.n) / stats.norm.pdf(panel.band_z)
    )
    assert np.allclose(panel.band, van_buuren, rtol=0, atol=1e-12)
    half_width = np.interp(panel.z, panel.band_z, panel.band)
    assert np.mean(np.abs(panel.deviation) <= half_width) >= 0.90
    assert payload.q_statistics is not None and len(payload.q_statistics) == 1
    assert payload.q_statistics["group"].tolist() == ["all"]


def test_worm_payload_cuts_a_numeric_covariate_into_equal_count_intervals(gaussian_case):
    _, X, _, residuals = gaussian_case
    covariate = X["x"].to_numpy()
    payload = worm_payload(residuals, covariate=covariate, covariate_name="x", n_intervals=4)

    assert payload.covariate == "x" and len(payload.panels) == 4
    counts = [panel.n for panel in payload.panels]
    assert max(counts) - min(counts) <= 1 and sum(counts) == 1500
    lows = [panel.interval[0] for panel in payload.panels]
    highs = [panel.interval[1] for panel in payload.panels]
    assert lows == sorted(lows) and highs == sorted(highs)
    assert all(low < high for low, high in zip(lows, highs, strict=True))
    assert lows[1:] == highs[:-1]
    for panel in payload.panels:
        half_width = np.interp(panel.z, panel.band_z, panel.band)
        assert np.mean(np.abs(panel.deviation) <= half_width) >= 0.90
    table = payload.q_statistics
    assert table["group"].tolist() == [panel.label for panel in payload.panels] + ["all"]
    assert int(table["n"].iloc[-1]) == 1500


def test_worm_payload_makes_one_panel_per_level_of_a_categorical_covariate(gaussian_case):
    _, _, _, residuals = gaussian_case
    rng = np.random.default_rng(5)
    levels = rng.choice(np.array(["a", "b", "c"], dtype=object), size=residuals.n_rows)
    payload = worm_payload(residuals, covariate=levels, covariate_name="group")
    assert [panel.label for panel in payload.panels] == ["a", "b", "c"]
    assert all(panel.interval is None for panel in payload.panels)
    assert sum(panel.n for panel in payload.panels) == 1500

    flags = rng.random(residuals.n_rows) < 0.5
    boolean = worm_payload(residuals, covariate=flags)
    assert [panel.label for panel in boolean.panels] == ["False", "True"]
    assert boolean.covariate is None


def test_worm_payload_json_round_trips(gaussian_case):
    _, X, _, residuals = gaussian_case
    payload = worm_payload(residuals, covariate=X["x"].to_numpy(), covariate_name="x", n_points=32)
    encoded = json.loads(json.dumps(payload.to_json()))
    assert encoded["kind"] == "worm" and encoded["covariate"] == "x"
    assert len(encoded["panels"]) == 4
    first = encoded["panels"][0]
    assert len(first["band"]) == 32 and len(first["z"]) == first["n"]
    assert first["interval"] == list(payload.panels[0].interval)
    assert encoded["q_statistics"][-1]["group"] == "all"
    assert isinstance(encoded["q_statistics"][0]["flagged"], bool)


def test_worm_payload_validates_its_arguments(gaussian_case):
    _, _, _, residuals = gaussian_case
    with pytest.raises(TypeError, match="ResidualSet"):
        worm_payload(object())
    with pytest.raises(ValueError, match="one value per row"):
        worm_payload(residuals, covariate=np.zeros(3))
    with pytest.raises(ValueError, match="n_intervals"):
        worm_payload(residuals, covariate=np.arange(residuals.n_rows), n_intervals=0)
    with pytest.raises(ValueError, match="n_points"):
        worm_payload(residuals, n_points=1)
    unusable = np.arange(residuals.n_rows, dtype=np.float64)
    unusable[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        worm_payload(residuals, covariate=unusable)
    with pytest.raises(ValueError, match="too many ties"):
        worm_payload(residuals, covariate=np.zeros(residuals.n_rows))


def test_worm_payload_replicates_frequency_weights(frequency_case):
    _, _, _, residuals, counts = frequency_case
    payload = worm_payload(residuals)
    assert payload.panels[0].n == int(counts.sum())


# ------------------------------------------------------------------ Q-statistics


def _standardised(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std(ddof=1)


def test_q_statistics_are_quiet_on_standard_normal_residuals():
    draws = np.random.default_rng(1).standard_normal(4000)
    table = q_statistics(draws, np.zeros(4000, dtype=int))
    assert list(table.columns) == [
        "group",
        "n",
        "mean_z",
        "variance_z",
        "skewness_z",
        "kurtosis_z",
        "flagged",
    ]
    assert len(table) == 1 and int(table["n"].iloc[0]) == 4000
    row = table.iloc[0]
    for column in ("mean_z", "variance_z", "skewness_z", "kurtosis_z"):
        assert abs(float(row[column])) < 3.0
    assert bool(row["flagged"]) is False

    # The four constants are the asymptotic null standard deviations of the
    # sample moments, so pin them against the moments computed from scratch.
    n = len(draws)
    centred = draws - draws.mean()
    m2 = float(np.mean(centred**2))
    assert float(row["mean_z"]) == pytest.approx(np.sqrt(n) * draws.mean(), rel=1e-12)
    assert float(row["variance_z"]) == pytest.approx(
        np.sqrt(n / 2.0) * (float(centred @ centred) / (n - 1) - 1.0), rel=1e-12
    )
    assert float(row["skewness_z"]) == pytest.approx(
        np.sqrt(n / 6.0) * float(np.mean(centred**3)) / m2**1.5, rel=1e-12
    )
    assert float(row["kurtosis_z"]) == pytest.approx(
        np.sqrt(n / 24.0) * (float(np.mean(centred**4)) / m2**2 - 3.0), rel=1e-12
    )


def test_q_statistics_flag_the_moment_that_is_wrong():
    rng = np.random.default_rng(2)
    groups = np.zeros(4000, dtype=int)

    inflated = q_statistics(1.5 * rng.standard_normal(4000), groups).iloc[0]
    assert float(inflated["variance_z"]) > 2.0 and abs(float(inflated["mean_z"])) < 3.0
    assert bool(inflated["flagged"]) is True

    shifted = q_statistics(rng.standard_normal(4000) + 0.1, groups).iloc[0]
    assert float(shifted["mean_z"]) > 2.0

    skewed = _standardised(stats.skewnorm.rvs(4.0, size=4000, random_state=3))
    assert float(q_statistics(skewed, groups).iloc[0]["skewness_z"]) > 2.0

    heavy = _standardised(stats.t.rvs(5.0, size=4000, random_state=4))
    assert float(q_statistics(heavy, groups).iloc[0]["kurtosis_z"]) > 2.0


def test_q_statistics_add_an_overall_row_only_when_groups_differ():
    rng = np.random.default_rng(6)
    values = np.concatenate([rng.standard_normal(600), 1.5 * rng.standard_normal(600)])
    groups = np.array(["low"] * 600 + ["high"] * 600, dtype=object)
    table = q_statistics(values, groups)
    assert table["group"].tolist() == ["low", "high", "all"]
    assert int(table["n"].iloc[-1]) == 1200
    assert float(table.set_index("group").loc["high", "variance_z"]) > 2.0
    assert table["flagged"].dtype == bool


def test_q_statistics_report_nan_where_a_group_is_too_thin():
    values = np.array([0.3, 1.0, 1.0, 1.0])
    groups = np.array([0, 1, 1, 1])
    table = q_statistics(values, groups).set_index("group")
    thin = table.loc[0]
    assert np.isfinite(float(thin["mean_z"]))
    assert np.isnan(float(thin["variance_z"]))
    assert np.isnan(float(thin["skewness_z"])) and np.isnan(float(thin["kurtosis_z"]))
    constant = table.loc[1]
    assert np.isfinite(float(constant["variance_z"]))
    assert np.isnan(float(constant["skewness_z"])) and np.isnan(float(constant["kurtosis_z"]))
    assert bool(thin["flagged"]) is False

    with pytest.raises(ValueError, match="one group label per residual"):
        q_statistics(values, groups[:2])
    with pytest.raises(ValueError, match="one-dimensional"):
        q_statistics(values.reshape(2, 2), groups.reshape(2, 2))


# --------------------------------------------------------------------------- PIT


def test_pit_payload_histogram_matches_the_uniform_band(gaussian_case):
    _, _, _, residuals = gaussian_case
    payload = pit_payload(residuals)

    assert isinstance(payload, PITPayload)
    assert payload.kind == "pit" and payload.schema_version == 1
    assert payload.n_bins == 20 and payload.n_rows == 1500
    assert len(payload.edges) == 21 and len(payload.counts) == 20
    assert payload.edges[0] == 0.0 and payload.edges[-1] == 1.0
    assert int(payload.counts.sum()) == 1500
    assert payload.expected == pytest.approx(75.0)
    band = stats.binom.ppf([0.025, 0.975], 1500, 1.0 / 20.0)
    assert (payload.band_lower, payload.band_upper) == (float(band[0]), float(band[1]))
    assert payload.band_lower < payload.expected < payload.band_upper
    inside = (payload.counts >= payload.band_lower) & (payload.counts <= payload.band_upper)
    assert int(inside.sum()) >= 18

    encoded = json.loads(json.dumps(payload.to_json()))
    assert encoded["counts"] == payload.counts.tolist()
    assert encoded["n_rows"] == 1500 and encoded["kind"] == "pit"


def test_pit_payload_shows_a_misspecified_model(misspecified_case, gamma_case):
    _, _, _, wrong = misspecified_case
    payload = pit_payload(wrong, n_bins=10)
    inside = (payload.counts >= payload.band_lower) & (payload.counts <= payload.band_upper)
    assert int(inside.sum()) <= 5

    _, _, _, right = gamma_case
    good = pit_payload(right, n_bins=10, alpha=0.01)
    inside_good = (good.counts >= good.band_lower) & (good.counts <= good.band_upper)
    assert int(inside_good.sum()) >= 9


def test_pit_payload_validates_its_arguments(gaussian_case):
    _, _, _, residuals = gaussian_case
    with pytest.raises(TypeError, match="ResidualSet"):
        pit_payload(object())
    with pytest.raises(ValueError, match="n_bins"):
        pit_payload(residuals, n_bins=0)
    with pytest.raises(ValueError, match="alpha"):
        pit_payload(residuals, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        pit_payload(residuals, alpha=1.0)


def test_pit_payload_replicates_frequency_weights(frequency_case):
    _, _, _, residuals, counts = frequency_case
    payload = pit_payload(residuals, n_bins=10)
    assert payload.n_rows == int(counts.sum())
    assert int(payload.counts.sum()) == int(counts.sum())
    assert payload.expected == pytest.approx(counts.sum() / 10.0)


# ---------------------------------------------------------------- payload guards


def _panel(n: int = 4, points: int = 5) -> WormPanel:
    return WormPanel(
        label="all",
        z=np.zeros(n),
        deviation=np.zeros(n),
        band_z=np.zeros(points),
        band=np.ones(points),
        n=n,
        interval=None,
    )


def test_payloads_guard_their_own_shapes():
    with pytest.raises(ValueError, match="at least one point"):
        order_statistic_grid(0)

    grid = np.linspace(-2.0, 2.0, 5)
    with pytest.raises(ValueError, match="one value per theoretical grid point"):
        QQPayload(
            theoretical=grid,
            observed=grid[:4],
            envelope_lower=grid,
            envelope_upper=grid,
            n_sim=2,
            n_rows=5,
            subsampled=False,
            seed=1,
        )
    with pytest.raises(ValueError, match="non-empty one-dimensional grid"):
        QQPayload(
            theoretical=np.zeros(0),
            observed=np.zeros(0),
            envelope_lower=np.zeros(0),
            envelope_upper=np.zeros(0),
            n_sim=2,
            n_rows=0,
            subsampled=False,
            seed=1,
        )

    with pytest.raises(ValueError, match="one value per order statistic"):
        WormPanel(
            label="all",
            z=np.zeros(4),
            deviation=np.zeros(3),
            band_z=np.zeros(5),
            band=np.zeros(5),
            n=4,
            interval=None,
        )
    with pytest.raises(ValueError, match="one half-width per band grid point"):
        WormPanel(
            label="all",
            z=np.zeros(4),
            deviation=np.zeros(4),
            band_z=np.zeros(5),
            band=np.zeros(4),
            n=4,
            interval=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="at least one WormPanel"):
        WormPayload(panels=(), covariate=None, q_statistics=None)

    without_table = WormPayload(panels=(_panel(),), covariate="x", q_statistics=None)
    assert without_table.to_json()["q_statistics"] is None
    assert without_table.panels[0].interval is None
    assert _panel().to_json()["interval"] is None

    with pytest.raises(ValueError, match="n_bins"):
        PITPayload(
            edges=np.linspace(0.0, 1.0, 3),
            counts=np.zeros(2, dtype=np.int64),
            expected=1.0,
            band_lower=0.0,
            band_upper=2.0,
            n_bins=0,
            n_rows=2,
        )
    with pytest.raises(ValueError, match="one count per bin"):
        PITPayload(
            edges=np.linspace(0.0, 1.0, 3),
            counts=np.zeros(3, dtype=np.int64),
            expected=1.0,
            band_lower=0.0,
            band_upper=2.0,
            n_bins=2,
            n_rows=2,
        )
