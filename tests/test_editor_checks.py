"""The editor's four SuperLSS checks on one split, against SciPy laws and exact counts."""

from __future__ import annotations

import copy
import dataclasses
import json
import tracemalloc
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from superglm import (
    Categorical,
    GammaLS,
    GaussianLS,
    GeneralizedParetoLSS,
    Spline,
    SuperGLM,
    SuperLSS,
    TweedieLSS,
    cat,
    s,
    ti,
)
from superglm.distributional.kernels.generalized_pareto import (
    GeneralizedParetoInitializationWarning,
)
from superglm.distributional.posterior import PosteriorDraws, posterior_parameters
from superglm.distributional.residuals import compute_residuals, replication_sample
from superglm.editor import EditorSession, checks
from superglm.editor import widget as widget_module
from superglm.editor.checks import (
    CHECK_REPORTS,
    QUANTILE_GRID,
    check_body,
    check_inputs,
    check_report_payload,
    null_replicates,
    offered_checks,
    qq_from_residuals,
)
from superglm.editor.cv import InForce
from superglm.editor.evaluation import EvaluationDataset
from superglm.model.base import NoPredictedMeanError
from superglm.stats import (
    UniformBand,
    band_at_level,
    binomial_cell_level,
    bonferroni_binomial_band,
    equal_local_levels_band,
    monte_carlo_level,
    order_statistic_level,
)

SEED = 20261127
#: ``.fit`` holds each smooth's penalty fixed, so each smooth names its own.
LAMBDA = 1.0
GROUPS = np.array(["a", "b", "c"])
CV = {"a": 0.4, "b": 0.6, "c": 0.8}


class Case(NamedTuple):
    lss: Any
    validation: EvaluationDataset
    train: EvaluationDataset | None = None


def _rows(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, n), "g": rng.choice(GROUPS, n)})
    mean = np.exp(1.0 + 0.5 * np.sin(2.0 * np.pi * X["x"]) + 0.3 * (X["g"] == "b"))
    return X, mean.to_numpy(), X["g"].map(CV).to_numpy()


def _gamma_spec(**kwargs) -> SuperLSS:
    family = GammaLS()
    return SuperLSS(family, family.mean(s("x", k=6), cat("g")), family.scale(cat("g")), **kwargs)


def _validation(X, y, sample_weight=None) -> EvaluationDataset:
    return EvaluationDataset(
        name="validation", label="Validation", X=X, y=y, sample_weight=sample_weight
    )


def _fitted_case(y_of, *, n_validation: int = 200, **kwargs) -> Case:
    rng = np.random.default_rng(SEED)
    X, mean, cv = _rows(600 + n_validation, rng)
    y = y_of(rng, mean, cv)
    lss = _gamma_spec(**kwargs).fit(X.iloc[:600], y[:600], lambdas={"mean:x#wiggle": LAMBDA})
    train = EvaluationDataset(
        name="train", label="Train", X=X.iloc[:600], y=y[:600], source="retained_fit_data"
    )
    return Case(lss, _validation(X.iloc[600:].reset_index(drop=True), y[600:]), train)


def _calm_case() -> Case:
    return _fitted_case(lambda rng, mean, cv: rng.gamma(1.0 / cv**2, mean * cv**2))


def _wrong_case() -> Case:
    """Log-normal amounts under a Gamma fit, read on 1,000 rows.

    At 200 rows the PIT histogram's Bonferroni band over 20 bins holds this misfit
    (no bin outside, measured); at 1,000 four bins leave it.
    """
    sigma = 1.4
    return _fitted_case(
        lambda rng, mean, cv: mean * np.exp(sigma * rng.standard_normal(len(mean)) - sigma**2 / 2),
        n_validation=1_000,
    )


@pytest.fixture(scope="module")
def calm() -> Case:
    return _calm_case()


@pytest.fixture(scope="module")
def wrong() -> Case:
    return _wrong_case()


@pytest.fixture(scope="module")
def heavy_gpd() -> Case:
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 600)})
    y = stats.genpareto.rvs(0.7, scale=np.exp(0.5 * X["x"].to_numpy()), random_state=rng)
    family = GeneralizedParetoLSS(shape_lower=0.55)
    # A method-of-moments start needs a finite variance, which shape 0.7 lacks.
    with pytest.warns(GeneralizedParetoInitializationWarning):
        lss = SuperLSS(family, family.scale(s("x", k=5)), family.shape()).fit(
            X.iloc[:400], y[:400], lambdas={"scale:x#wiggle": LAMBDA}
        )
    return Case(lss, _validation(X.iloc[400:].reset_index(drop=True), y[400:]))


@pytest.fixture(scope="module")
def prior_weighted() -> Case:
    """Gamma amounts whose prior weight divides each row's squared CV."""
    rng = np.random.default_rng(SEED)
    X, mean, cv = _rows(800, rng)
    w = rng.uniform(0.5, 3.0, 800)
    y = rng.gamma(w / cv**2, mean * cv**2 / w)
    lss = _gamma_spec().fit(
        X.iloc[:600], y[:600], sample_weight=w[:600], lambdas={"mean:x#wiggle": LAMBDA}
    )
    held = X.iloc[600:].reset_index(drop=True)
    return Case(lss, _validation(held, y[600:], sample_weight=w[600:]))


@pytest.fixture(scope="module")
def with_offset() -> Case:
    """Gamma amounts whose Mean carries a log exposure the model reads as an offset column."""
    rng = np.random.default_rng(SEED)
    X, mean, cv = _rows(800, rng)
    X["log_exposure"] = np.log(rng.uniform(0.5, 2.0, 800))
    y = rng.gamma(1 / cv**2, mean * np.exp(X["log_exposure"].to_numpy()) * cv**2)
    family = GammaLS()
    spec = SuperLSS(
        family,
        family.mean(s("x", k=6), cat("g"), offset="log_exposure"),
        family.scale(cat("g")),
    )
    lss = spec.fit(X.iloc[:600], y[:600], lambdas={"mean:x#wiggle": LAMBDA})
    return Case(lss, _validation(X.iloc[600:].reset_index(drop=True), y[600:]))


@pytest.fixture(scope="module")
def zeros() -> Case:
    """Compound Poisson-gamma amounts, 16% zero: a family with an atom, randomised at zero."""
    rng = np.random.default_rng(SEED)
    X, mean, _ = _rows(600, rng)
    events = rng.poisson(1.2 * np.sqrt(mean))
    y = np.zeros(600)
    y[events > 0] = rng.gamma(events[events > 0], np.sqrt(mean[events > 0]) / 1.2)
    family = TweedieLSS()
    lss = SuperLSS(family, family.mu(s("x", k=5), cat("g")), family.phi(), family.p()).fit(
        X.iloc[:400], y[:400], lambdas={"mean:x#wiggle": LAMBDA}
    )
    return Case(lss, _validation(X.iloc[400:].reset_index(drop=True), y[400:]))


@pytest.fixture(scope="module")
def zeros_frequency() -> Case:
    """ZEROS's compound Poisson-gamma amounts under frequency weights of 1 to 3."""
    rng = np.random.default_rng(SEED)
    X, mean, _ = _rows(600, rng)
    events = rng.poisson(1.2 * np.sqrt(mean))
    y = np.zeros(600)
    y[events > 0] = rng.gamma(events[events > 0], np.sqrt(mean[events > 0]) / 1.2)
    counts = rng.integers(1, 4, 600).astype(np.float64)
    family = TweedieLSS()
    spec = SuperLSS(
        family,
        family.mu(s("x", k=5), cat("g")),
        family.phi(),
        family.p(),
        weight_semantics="frequency",
    )
    lss = spec.fit(
        X.iloc[:400], y[:400], sample_weight=counts[:400], lambdas={"mean:x#wiggle": LAMBDA}
    )
    held = X.iloc[400:].reset_index(drop=True)
    return Case(lss, _validation(held, y[400:], counts[400:]))


@pytest.fixture(scope="module")
def calm_frequency() -> tuple[Case, Case]:
    rng = np.random.default_rng(SEED)
    X, mean, cv = _rows(800, rng)
    y = rng.gamma(1.0 / cv**2, mean * cv**2)
    counts = rng.integers(1, 4, 800).astype(np.float64)
    lss = _gamma_spec(weight_semantics="frequency").fit(
        X.iloc[:600], y[:600], sample_weight=counts[:600], lambdas={"mean:x#wiggle": LAMBDA}
    )
    held, held_y, held_counts = X.iloc[600:].reset_index(drop=True), y[600:], counts[600:]
    repeat = np.repeat(np.arange(200), held_counts.astype(int))
    weighted = Case(lss, _validation(held, held_y, held_counts))
    duplicated = Case(lss, _validation(held.iloc[repeat].reset_index(drop=True), held_y[repeat]))
    return weighted, duplicated


@pytest.fixture(scope="module")
def poisson_glm() -> SuperGLM:
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 300), "g": rng.choice(GROUPS, 300)})
    y = rng.poisson(np.exp(0.5 + np.sin(2.0 * np.pi * X["x"]))).astype(np.float64)
    features = {"x": Spline(n_knots=6), "g": Categorical(base="first")}
    return SuperGLM(family="poisson", selection_penalty=0.0, features=features).fit(X, y)


def _reference_u(lss, X, y):
    theta = lss.predict_parameters(X)
    cv, mean = theta["scale"].to_numpy(), theta["mean"].to_numpy()
    return stats.gamma.cdf(y, a=1 / cv**2, scale=mean * cv**2), mean, cv


def test_the_qq_body_is_the_sorted_residuals_against_the_simultaneous_band(calm):
    body = check_body("check_qq", calm.lss, calm.validation, inputs=["x", "g"])
    u, _, _ = _reference_u(calm.lss, calm.validation.X, calm.validation.y)
    z = np.sort(special.ndtri(u))
    n = len(z)
    band = band_at_level(n, body["local_level"])
    k = np.asarray(body["index"])
    scale = np.max(np.abs(z))
    np.testing.assert_allclose(body["observed"], z[k], rtol=0, atol=64 * 2.0**-53 * scale)
    np.testing.assert_allclose(body["band_lower"], special.ndtri(band.lower[k]), rtol=1e-12)
    np.testing.assert_allclose(body["theoretical"], special.ndtri(band.median[k]), rtol=1e-12)
    assert body["outside"] == int(
        np.sum((z < special.ndtri(band.lower)) | (z > special.ndtri(band.upper)))
    )
    assert body["n"] == n == body["cells"] and body["flagged"] == (body["outside"] > 0)


def test_the_pit_body_counts_each_bin_against_the_band_for_all_twenty(calm):
    body = check_body("check_pit", calm.lss, calm.validation, inputs=[])
    u, _, _ = _reference_u(calm.lss, calm.validation.X, calm.validation.y)
    assert body["counts"] == np.histogram(u, bins=20, range=(0, 1))[0].tolist()
    (lo,), (hi,) = bonferroni_binomial_band([len(u)], [1 / 20], cut=body["local_level"])
    assert (body["band_lower"], body["band_upper"]) == (lo, hi)
    counts = np.asarray(body["counts"])
    assert body["outside"] == int(np.sum((counts < lo) | (counts > hi)))


def test_the_calibration_body_reads_each_quantile_against_its_binomial_band(calm):
    body = check_body("check_calibration", calm.lss, calm.validation, inputs=[])
    _, mean, cv = _reference_u(calm.lss, calm.validation.X, calm.validation.y)
    y, n = calm.validation.y, len(calm.validation.y)
    for i, p in enumerate(QUANTILE_GRID):
        share = np.mean(y > stats.gamma.ppf(p, a=1 / cv**2, scale=mean * cv**2))
        assert body["quantiles"]["realised_exceedance"][i] == pytest.approx(share, abs=1e-12)
    q = 1 - np.asarray(QUANTILE_GRID)
    lower, upper = bonferroni_binomial_band(np.full(len(q), n), q, cut=body["local_level"])
    np.testing.assert_array_equal(body["band_lower"], lower / n)
    np.testing.assert_array_equal(body["band_upper"], upper / n)
    shares = np.asarray(body["quantiles"]["realised_exceedance"])
    assert body["outside"] == int(
        np.sum((shares < body["band_lower"]) | (shares > body["band_upper"]))
    )


def test_the_ae_body_sums_to_the_split_and_reads_its_interval_from_the_gamma_law(calm):
    body = check_body("check_ae", calm.lss, calm.validation, inputs=["x", "g"])
    _, mean, cv = _reference_u(calm.lss, calm.validation.X, calm.validation.y)
    y, g = calm.validation.y, np.asarray(calm.validation.X["g"])
    band, by_x, by_g = body["tables"]
    assert [t["by"] for t in body["tables"]] == ["band", "input", "input"]
    assert sum(band["actual"]) == pytest.approx(y.sum())
    assert sum(band["expected"]) == pytest.approx(mean.sum())
    assert body["overall"]["ratio"] == pytest.approx(y.sum() / mean.sum(), rel=1e-12)
    for level, actual, expected, se in zip(
        by_g["levels"], by_g["actual"], by_g["expected"], by_g["ratio_se"], strict=True
    ):
        rows = g == level
        assert actual == pytest.approx(y[rows].sum()) and expected == pytest.approx(
            mean[rows].sum()
        )
        assert se == pytest.approx(
            np.sqrt(np.sum((mean[rows] * cv[rows]) ** 2)) / mean[rows].sum(), rel=1e-9
        )


@pytest.mark.parametrize("name", ["calm", "wrong"])
def test_the_ae_flag_counts_the_intervals_that_exclude_one(name, request):
    """Mutation check: ``outside`` counts the intervals ratio -/+ z se that leave out 1, either side.

    WRONG's intervals leave 1 out on both sides, so a count of one side alone fails.
    """
    case = request.getfixturevalue(name)
    body = check_body("check_ae", case.lss, case.validation, inputs=["x", "g"])
    z = body["z"]
    expected = 0
    for table in body["tables"]:
        for ratio, se, lower, upper in zip(
            table["ratio"], table["ratio_se"], table["lower"], table["upper"], strict=True
        ):
            assert lower == pytest.approx(ratio - z * se) and upper == pytest.approx(ratio + z * se)
            expected += abs(ratio - 1.0) > z * se
    assert body["outside"] == expected and body["flagged"] == (expected > 0)
    assert body["cells"] == sum(len(t["ratio_se"]) for t in body["tables"])


def test_a_wrong_family_is_flagged_on_distribution_and_calibration(wrong):
    flags = {
        r: check_body(r, wrong.lss, wrong.validation, inputs=["x", "g"])["flagged"]
        for r in CHECK_REPORTS
    }
    assert flags["check_qq"] and flags["check_pit"] and flags["check_calibration"]


def test_a_band_that_lets_everything_in_flags_nothing(wrong, monkeypatch):
    """Mutation check: the Q-Q flag is read from the band, not from the residuals alone."""

    def open_band(n, alpha=0.05):
        return UniformBand(
            lower=np.zeros(n),
            upper=np.ones(n),
            median=np.full(n, 0.5),
            local_level=0.05,
            alpha=0.05,
        )

    assert check_body("check_qq", wrong.lss, wrong.validation, inputs=[])["flagged"]
    monkeypatch.setattr(checks, "equal_local_levels_band", open_band)
    monkeypatch.setattr(checks, "band_at_level", lambda n, local_level, alpha=0.05: open_band(n))
    body = check_body("check_qq", wrong.lss, wrong.validation, inputs=[])
    assert not body["flagged"]
    # The open band's edges are infinite on the normal scale, and JSON has no infinity.
    assert body["band_lower"][0] is None and body["band_upper"][-1] is None
    json.dumps(body, allow_nan=False)


def test_a_law_without_finite_variance_has_no_ae_interval(heavy_gpd):
    # GeneralizedParetoLSS(shape_lower=0.55): every fitted shape is above 1/2.
    ae = check_body("check_ae", heavy_gpd.lss, heavy_gpd.validation, inputs=["x"])
    assert ae == {
        "available": False,
        "flagged": False,
        "outside": 0,
        "cells": 0,
        "reason": "The predicted distribution has no finite variance at these parameters, "
        "so actual over expected has no interval.",
    }
    replicates = null_replicates(heavy_gpd.lss, heavy_gpd.validation, inputs=["x"])
    assert replicates.levels["check_ae"] is None
    for report in ("check_qq", "check_pit", "check_calibration"):
        body = check_body(
            report, heavy_gpd.lss, heavy_gpd.validation, inputs=["x"], replicates=replicates
        )
        assert body["available"] and body["level_source"] == "simulated"
        assert body["n_draws"] == 99 and body["local_level"] == replicates.levels[report]


@pytest.mark.parametrize(
    ("name", "source"), [("train", "retained_fit_data"), ("validation", "supplied")]
)
def test_a_zero_expected_total_has_no_overall_ratio(name, source):
    """An intercept-only Gaussian fit to alternating -1 and +1: its Mean is 0 on every row.

    Actual over expected of the whole split is then undefined, NaN as the library's grouped
    ratios are, which the body sends as ``null``; the card still answers, on the rows the
    model was fitted on and on a held-out split.
    """
    family = GaussianLS()
    X = pd.DataFrame({"x": np.arange(40.0)})
    y = np.tile([-1.0, 1.0], 20)
    lss = SuperLSS(family, family.location(), family.scale()).fit(X, y)
    split = EvaluationDataset(name=name, label=name.title(), X=X, y=y, source=source)
    body = check_body("check_ae", lss, split, inputs=["x"])
    assert body["available"] and body["overall"]["expected"] == 0.0
    assert body["overall"]["ratio"] is None
    json.dumps(body, allow_nan=False)


def test_an_error_that_is_not_about_variance_still_raises(calm, monkeypatch):
    """Mutation check: only the finite-variance refusal becomes a reason; other faults raise."""

    def broken(*args, **kwargs):
        raise ValueError("covariate must give one value per row of X")

    monkeypatch.setattr(checks, "actual_expected_check", broken)
    with pytest.raises(ValueError, match="one value per row"):
        check_body("check_ae", calm.lss, calm.validation, inputs=[])


def test_a_split_of_five_rows_draws_every_check(calm):
    five = dataclasses.replace(
        calm.validation, X=calm.validation.X.iloc[:5], y=calm.validation.y[:5]
    )
    bodies = {r: check_body(r, calm.lss, five, inputs=["x", "g"]) for r in CHECK_REPORTS}
    assert all(body["level_source"] == "simulated" for body in bodies.values())
    assert bodies["check_qq"]["n"] == 5 and sum(bodies["check_pit"]["counts"]) == 5
    assert all(len(t["ratio"]) <= 5 for t in bodies["check_ae"]["tables"])
    json.dumps(bodies, allow_nan=False)


def test_frequency_weights_read_as_repeated_rows(calm_frequency):
    # The same fit; a split with integer weights, and its rows repeated.
    weighted, duplicated = calm_frequency
    a = check_body("check_pit", weighted.lss, weighted.validation, inputs=[])
    b = check_body("check_pit", duplicated.lss, duplicated.validation, inputs=[])
    assert a["counts"] == b["counts"] and a["n_rows"] == b["n_rows"]
    qq_a = check_body("check_qq", weighted.lss, weighted.validation, inputs=[])
    qq_b = check_body("check_qq", duplicated.lss, duplicated.validation, inputs=[])
    assert qq_a["n"] == qq_b["n"] == int(np.sum(weighted.validation.sample_weight))


def test_frequency_weights_and_repeated_rows_give_the_same_levels_and_flags(calm_frequency):
    """Each occurrence of a weighted row is simulated as a repeated row is.

    Actual over expected keeps the library's cells, the ones its card shows. A categorical
    input's cells are its levels in both splits, and each level's replicate totals are the
    same. The band table of the predicted Mean is left out: its equal-count cells are cut over
    physical rows, so a weighted split and its repeated rows fall into different cells. Mutation
    check: one response per weighted row instead of one per occurrence fails it.
    """
    weighted, duplicated = calm_frequency
    for report in ("check_qq", "check_pit", "check_calibration"):
        a = check_body(report, weighted.lss, weighted.validation, inputs=[])
        b = check_body(report, duplicated.lss, duplicated.validation, inputs=[])
        assert a["level_source"] == b["level_source"] == "simulated"
        assert a["local_level"] == pytest.approx(b["local_level"], rel=1e-9)
        assert (a["outside"], a["flagged"]) == (b["outside"], b["flagged"])
    totals = [
        checks._scored_replicates(case.lss, case.validation, inputs=["g"]).totals["g"]
        for case in (weighted, duplicated)
    ]
    assert totals[0].shape == (99, len(GROUPS))
    np.testing.assert_array_equal(*totals)


_UNSIMULATED_ROWS = (
    "The band treats the fitted parameters as exact, because these rows could not be simulated."
)


def test_rows_no_replicate_can_draw_keep_the_known_level_and_say_so(calm, monkeypatch):
    """A draw outside the links' domain leaves the rows no law for a replicate to draw from.

    The draw scales every coefficient a millionfold, so the log links overflow. Mutation check:
    naming the fit's uncertainty, as a refused draw does, fails it.
    """
    fitted = calm.lss._require_fitted()

    def beyond(model, n_draws, *, covariance, seed):
        beta = np.tile(fitted.prediction_coefficients * 1e6, (n_draws, 1))
        names = tuple(fitted.layout.coefficient_names)
        return PosteriorDraws(beta, covariance_kind="fixed", seed=seed, coefficient_names=names)

    monkeypatch.setattr(checks, "posterior_draws", beyond)
    replicates = null_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    assert (replicates.n_draws, replicates.note) == (0, _UNSIMULATED_ROWS)
    for report in CHECK_REPORTS:
        body = check_body(report, calm.lss, calm.validation, inputs=["x", "g"])
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "known",
            0,
            _UNSIMULATED_ROWS,
        )


#: The residual sampler's cap: the Q-Q plot and the PIT histogram read at most this many.
_OCCURRENCE_CAP = 100_000


def test_a_split_beyond_the_occurrence_cap_draws_the_sample_the_checks_read(calm_frequency):
    """A hundred million occurrences: the replicates draw the 100,000 the Q-Q plot and PIT
    histogram read, as the library selects them, and never hold an index of every occurrence.

    The split's own responses at that sample, scored as a replicate, give the PIT histogram
    bin by bin. Quantile calibration of the response and actual over expected read every
    occurrence, which no replicate draws, so they keep the known level and say why. Mutation
    check: expanding every occurrence before taking the sample allocates an 800 MB index and
    breaks the envelope.
    """
    weighted, _ = calm_frequency
    counts = weighted.validation.sample_weight * 250_000
    split = dataclasses.replace(weighted.validation, sample_weight=counts)
    assert np.sum(counts) > 9.0e7
    tracemalloc.start()
    try:
        replicates = null_replicates(weighted.lss, split, inputs=["x", "g"], n_draws=19)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # An index of every occurrence alone is 8 bytes each, 800 MB.
    assert peak < 200e6
    bodies = {
        r: check_body(r, weighted.lss, split, inputs=["x", "g"], replicates=replicates)
        for r in CHECK_REPORTS
    }
    for report in ("check_qq", "check_pit"):
        body = bodies[report]
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "simulated",
            19,
            None,
        )
    for report in ("check_calibration", "check_ae"):
        body = bodies[report]
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "known",
            0,
            _UNSIMULATED_ROWS,
        )
    fitted = weighted.lss._require_fitted()
    residuals = compute_residuals(fitted, split.X, split.y, sample_weight=counts, seed=42)
    sample = replication_sample(residuals, seed=42)
    own = np.asarray(split.y, dtype=np.float64)[sample]
    scored = checks._scored_replicates(weighted.lss, split, inputs=[], responses=own[None, :])
    pit = bodies["check_pit"]
    assert pit["n_rows"] == len(own) == _OCCURRENCE_CAP
    assert scored.counts[0].tolist() == pit["counts"]


def test_a_split_beyond_the_cap_whose_total_overflows_still_simulates_qq_and_pit(
    calm_frequency, monkeypatch
):
    """The review's probe: among 104 million occurrences one response of 1e305 on 750,000 of
    them overflows actual over expected's weighted total.

    Beyond the cap actual over expected keeps the known level, so the replicates build none of
    its tables, and the Q-Q plot and the PIT histogram are simulated. Mutation check: building
    the tables before the cap is tested builds them for the replicates to discard.
    """
    weighted, _ = calm_frequency
    counts = weighted.validation.sample_weight * 250_000
    y = np.array(weighted.validation.y, dtype=np.float64)
    y[np.flatnonzero(counts == 750_000)[0]] = 1e305
    split = dataclasses.replace(weighted.validation, y=y, sample_weight=counts)
    assert np.sum(counts) == 104_000_000
    built = []
    library = checks.actual_expected_check

    def counted(*args, **kwargs):
        built.append(kwargs["name"])
        return library(*args, **kwargs)

    monkeypatch.setattr(checks, "actual_expected_check", counted)
    replicates = null_replicates(weighted.lss, split, inputs=["x", "g"], n_draws=19)
    assert built == []
    for report in ("check_qq", "check_pit"):
        body = check_body(report, weighted.lss, split, inputs=["x", "g"], replicates=replicates)
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "simulated",
            19,
            None,
        )


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_a_total_the_rows_refuse_leaves_the_other_checks_simulated(calm_frequency):
    """One response of 1e306 on 1,000 occurrences overflows actual over expected's weighted
    total on a split whose occurrences are all simulated.

    That check's replicate preparation stands alone: it has no level, and the Q-Q plot, the PIT
    histogram and quantile calibration are simulated. Mutation check: letting the refusal out
    of the preparation raises it out of all three bodies.
    """
    weighted, _ = calm_frequency
    counts = np.array(weighted.validation.sample_weight, dtype=np.float64)
    y = np.array(weighted.validation.y, dtype=np.float64)
    counts[0], y[0] = 1_000.0, 1e306
    split = dataclasses.replace(weighted.validation, y=y, sample_weight=counts)
    replicates = null_replicates(weighted.lss, split, inputs=["x", "g"])
    assert (replicates.n_draws, replicates.note, replicates.levels["check_ae"]) == (99, None, None)
    for report in ("check_qq", "check_pit", "check_calibration"):
        body = check_body(report, weighted.lss, split, inputs=["x", "g"], replicates=replicates)
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "simulated",
            99,
            None,
        )


def test_a_level_that_cannot_be_formed_keeps_that_check_alone_at_the_known_level(calm, monkeypatch):
    """An undefined replicate statistic of actual over expected, forced: one NaN among its 99.

    Each check's level is selected on its own, so actual over expected keeps the known level
    and says why, and the Q-Q plot, the PIT histogram and quantile calibration are simulated.
    Mutation check: selecting every level in one step raises the refusal out of all four bodies.
    """
    library = checks._ae_statistics

    def undefined(*args, **kwargs):
        values = np.array(library(*args, **kwargs), dtype=np.float64)
        values[0] = np.nan
        return values

    monkeypatch.setattr(checks, "_ae_statistics", undefined)
    replicates = null_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    assert (replicates.n_draws, replicates.levels["check_ae"]) == (99, None)
    for report in CHECK_REPORTS:
        body = check_body(report, calm.lss, calm.validation, inputs=["x", "g"])
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            ("known", 0, _UNSIMULATED_ROWS) if report == "check_ae" else ("simulated", 99, None)
        )


def test_a_large_split_sends_at_most_800_points_and_flags_on_all():
    z = np.random.default_rng(1).standard_normal(30_000) * 1.02
    body = qq_from_residuals(z)
    k = np.asarray(body["index"])
    assert np.all(np.diff(k) > 0) and k[0] == 0 and k[-1] == 29_999
    assert len(k) <= 800 + 20 + body["outside"]
    assert len(json.dumps(body)) < 200_000 and body["cells"] == 30_000


def test_the_qq_body_keeps_every_point_outside_the_band():
    """Mutation check: every order statistic outside the band is among the points sent."""
    z = np.random.default_rng(2).standard_normal(5_000)
    z[:40] -= 3.0  # forty low points far outside the band
    body = qq_from_residuals(z)
    band = equal_local_levels_band(5_000)
    order = np.sort(z)
    outside = np.flatnonzero(
        (order < special.ndtri(band.lower)) | (order > special.ndtri(band.upper))
    )
    assert body["outside"] == len(outside) >= 40
    assert set(outside.tolist()) <= set(body["index"])


def test_offered_checks_and_inputs(calm, poisson_glm):
    assert offered_checks(calm.lss) == list(CHECK_REPORTS) and offered_checks(poisson_glm) == []
    assert check_inputs(calm.lss, ["mean:x", "mean:g", "scale:g"]) == ["x", "g"]


def test_an_interaction_is_not_an_input():
    """Mutation check: a term that reads two columns gives no A/E table of its own."""
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({"x": rng.uniform(0.0, 1.0, 300), "w": rng.uniform(0.0, 1.0, 300)})
    y = rng.gamma(4.0, np.exp(X["x"] + X["w"]).to_numpy() / 4.0)
    family = GammaLS()
    lss = SuperLSS(
        family, family.mean(s("x", k=5), s("w", k=5), ti("x", "w")), family.scale()
    ).fit_reml(X, y)
    assert check_inputs(lss, ["mean:x", "mean:x:w", "mean:w"]) == ["x", "w"]


def test_a_superglm_and_a_split_the_model_cannot_score_are_unavailable(calm, poisson_glm):
    glm_split = _validation(calm.validation.X, np.ones(len(calm.validation.y)))
    assert check_body("check_pit", poisson_glm, glm_split, inputs=[]) == {
        "available": False,
        "reason": "This check is available for SuperLSS models.",
        "flagged": False,
        "outside": 0,
        "cells": 0,
    }
    negative = dataclasses.replace(calm.validation, y=-np.asarray(calm.validation.y))
    for report in CHECK_REPORTS:
        body = check_body(report, calm.lss, negative, inputs=["x"])
        assert not body["available"]
        assert body["reason"] == (
            "The Validation split cannot be scored: its responses include values the model's "
            "family does not take. Remove those rows from the split to score it."
        )


def test_rows_without_a_parameter_value_are_named_among_the_split(calm, monkeypatch):
    """Mutation check: a refused row is the page's sentence, its position among the split's rows."""
    refusal = NoPredictedMeanError(
        "no value",
        positions=np.array([2]),  # the third retained row
        n_rows=3,
        link="PowerLink",
        what="the scores",
        advice="",
        value="value of 'mean'",
    )

    def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr(checks, "compute_residuals", refuse)
    weights = np.array([1.0, 0.0, 1.0, 1.0])  # so the third retained row is the split's fourth
    split = _validation(
        calm.validation.X.iloc[:4], np.asarray(calm.validation.y[:4]), sample_weight=weights
    )
    body = check_body("check_pit", calm.lss, split, inputs=[])
    assert not body["available"]
    assert body["reason"].startswith("1 of 4 rows have no quote under this model (row positions 3)")


def test_the_report_payload_carries_both_bodies(calm):
    edited = check_body("check_pit", calm.lss, calm.validation, inputs=[])
    payload = check_report_payload(
        "check_pit", dataset=calm.validation, compares_edit=False, edited=edited, original=None
    )
    assert payload == {
        "report": "check_pit",
        "available": True,
        "reason": None,
        "split": {"name": "validation", "label": "Validation", "n_obs": 200},
        "compares_edit": False,
        "models": {"edited": edited, "original": None},
    }


def test_a_payload_without_rows_names_no_split(poisson_glm):
    body = check_body("check_qq", poisson_glm, None, inputs=[])
    payload = check_report_payload(
        "check_qq", dataset=None, compares_edit=False, edited=body, original=None
    )
    assert payload["split"] is None and payload["reason"] == body["reason"]


def _ae_statistic(body: dict[str, Any]) -> float:
    """The largest ``|ratio - 1| / ratio_se`` over every cell of every table of an A/E body."""
    ratio = np.concatenate([np.asarray(t["ratio"], dtype=np.float64) for t in body["tables"]])
    se = np.concatenate([np.asarray(t["ratio_se"], dtype=np.float64) for t in body["tables"]])
    cells = np.isfinite(se) & (se > 0)
    return float(np.max(np.abs(ratio[cells] - 1.0) / se[cells]))


def _statistics(bodies: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Each check's statistic, read from its body alone."""
    qq, pit, cal = bodies["check_qq"], bodies["check_pit"], bodies["check_calibration"]
    assert qq["index"] == list(range(qq["n"]))  # a small split sends every point
    trials = np.rint(cal["quantiles"]["weight"])
    exceeded = np.rint(np.asarray(cal["quantiles"]["realised_exceedance"]) * trials)
    return {
        "check_qq": order_statistic_level(special.ndtr(qq["observed"])),
        "check_pit": binomial_cell_level(pit["counts"], pit["n_rows"], 1 / 20),
        "check_calibration": binomial_cell_level(exceeded, trials, 1 - np.asarray(QUANTILE_GRID)),
        "check_ae": _ae_statistic(bodies["check_ae"]),
    }


def _direct_qq_levels(transforms: np.ndarray) -> np.ndarray:
    """Each row's least two-sided tail, every order statistic read from SciPy's Beta law."""
    n = transforms.shape[1]
    i = np.arange(1, n + 1)
    lower, upper = stats.beta.cdf(transforms, i, n + 1 - i), stats.beta.sf(transforms, i, n + 1 - i)
    return np.min(2 * np.minimum(lower, upper), axis=1)


def _replicate_statistics(scored: Any) -> dict[str, np.ndarray]:
    """Each check's statistic on every replicate ``_scored_replicates`` scores, Q-Q's direct."""
    return {**scored.statistics, "check_qq": _direct_qq_levels(scored.transforms)}


def test_a_held_out_split_reads_each_level_from_99_replicates(calm):
    """Barnard's rule at 99 replicates: the 5th smallest level, or the 5th largest ``|z|``.

    The replicates keep each check's level alone, one float, and it is the statistic of the
    replicate at that rank: for Q-Q, read here from every point's Beta tail. Mutation check: a
    Q-Q level read from a screen that resolves only the least replicate fails it.
    """
    replicates = null_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    assert (replicates.n_draws, replicates.seed, replicates.note) == (99, 42, None)
    fields = [field.name for field in dataclasses.fields(replicates)]
    assert fields == ["levels", "n_draws", "seed", "note"]
    assert all(type(replicates.levels[report]) is float for report in CHECK_REPORTS)
    statistics = _replicate_statistics(
        checks._scored_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    )
    bodies = {
        r: check_body(r, calm.lss, calm.validation, inputs=["x", "g"], replicates=replicates)
        for r in CHECK_REPORTS
    }
    for report, body in bodies.items():
        assert (body["level_source"], body["n_draws"], body["level_note"]) == (
            "simulated",
            99,
            None,
        )
        assert body == check_body(report, calm.lss, calm.validation, inputs=["x", "g"])
    qq = bodies["check_qq"]
    eta = monte_carlo_level(statistics["check_qq"], alpha=0.05)
    assert qq["local_level"] == replicates.levels["check_qq"] == eta
    band = band_at_level(qq["n"], eta)
    k = np.asarray(qq["index"])
    np.testing.assert_array_equal(qq["band_lower"], special.ndtri(band.lower[k]))
    np.testing.assert_array_equal(qq["band_upper"], special.ndtri(band.upper[k]))
    for report in ("check_pit", "check_calibration"):
        level = monte_carlo_level(statistics[report], alpha=0.05)
        assert bodies[report]["local_level"] == replicates.levels[report] == level
    ae = bodies["check_ae"]
    assert ae["z"] == replicates.levels["check_ae"] == np.sort(statistics["check_ae"])[-5]
    for table in ae["tables"]:
        ratio, se = np.asarray(table["ratio"]), np.asarray(table["ratio_se"])
        np.testing.assert_array_equal(table["lower"], ratio - ae["z"] * se)
        np.testing.assert_array_equal(table["upper"], ratio + ae["z"] * se)


def test_a_supplied_training_split_keeps_the_known_parameter_bands(calm):
    """The session stores a user's ``train_data`` as a supplied split named ``"train"``.

    Those are the rows the model was fitted on, so they keep the known-parameter bands and
    nothing is simulated for them. Mutation check: deciding by the split's ``source`` instead
    simulates 99 replicates on them.
    """
    train = _session(calm)._evaluation_data["train"]
    assert (train.name, train.source) == ("train", "supplied")
    assert null_replicates(calm.lss, train, inputs=["x", "g"]).n_draws == 0
    for report in CHECK_REPORTS:
        body = check_body(report, calm.lss, train, inputs=["x", "g"])
        assert (body["level_source"], body["n_draws"], body["level_note"]) == ("known", 0, None)


def test_the_training_rows_keep_the_known_parameter_bands(calm):
    bodies = {r: check_body(r, calm.lss, calm.train, inputs=["x", "g"]) for r in CHECK_REPORTS}
    for body in bodies.values():
        assert (body["level_source"], body["n_draws"], body["level_note"]) == ("known", 0, None)
    n = calm.train.n_obs
    assert bodies["check_qq"]["local_level"] == equal_local_levels_band(n).local_level
    pit = bodies["check_pit"]
    lo, hi = stats.binom.ppf([0.025 / 20, 1 - 0.025 / 20], n, 1 / 20)
    assert (pit["band_lower"], pit["band_upper"], pit["local_level"]) == (lo, hi, 0.05 / 40)
    cal = bodies["check_calibration"]
    q = 1 - np.asarray(QUANTILE_GRID)
    lower, _ = bonferroni_binomial_band(np.full(len(q), n), q, alpha=0.05)
    np.testing.assert_array_equal(cal["band_lower"], lower / n)
    assert cal["local_level"] == 0.05 / 22
    ae = bodies["check_ae"]
    assert ae["z"] == pytest.approx(stats.norm.isf(0.025 / ae["cells"]), rel=1e-12)


def _assert_same_statistics(scored: dict[str, Any], j: int, observed: dict[str, float]) -> None:
    """Replicate ``j``'s statistics are ``observed``: the counts' exactly, Q-Q's and A/E's to
    the rounding of a normal round trip and of a sum in another order."""
    for report in ("check_pit", "check_calibration"):
        assert scored[report][j] == observed[report]
    for report in ("check_qq", "check_ae"):
        assert scored[report][j] == pytest.approx(observed[report], rel=1e-9)


@pytest.mark.parametrize(
    "name", ["calm", "prior_weighted", "with_offset", "zeros", "zeros_frequency"]
)
def test_a_replicate_is_scored_as_the_split_is(name, request):
    """The split's own responses, scored as a replicate, give each body's statistic.

    Prior weights enter each law and each total, an offset each draw's parameters, and a
    response at an atom is randomised as the split's is: from one stream across the rows, or
    under frequency weights per occurrence from the occurrences' own. Where each row has one
    response, a replicate is also a split of its own: the one whose Q-Q statistic is the Q-Q
    level gives its bodies' statistics, and every replicate gives its PIT histogram's. A
    frequency-weighted row's occurrences share one response in a split but not
    in a replicate, so there the first check stands alone. Mutation checks: scoring each
    replicate under the draw it was simulated from, totals without the prior weights, and atoms
    randomised from another stream, or weighted occurrences' atoms from the rows' stream, each
    fail it.
    """
    case = request.getfixturevalue(name)
    replicates = null_replicates(case.lss, case.validation, inputs=["x", "g"])
    own = np.asarray(case.validation.y, dtype=np.float64)
    if name == "zeros_frequency":
        own = np.repeat(own, case.validation.sample_weight.astype(np.int64))
    scored = checks._scored_replicates(
        case.lss, case.validation, inputs=["x", "g"], responses=own[None, :]
    )
    bodies = {
        r: check_body(r, case.lss, case.validation, inputs=["x", "g"], replicates=replicates)
        for r in CHECK_REPORTS
    }
    _assert_same_statistics(_replicate_statistics(scored), 0, _statistics(bodies))
    # Bin by bin, so a randomised atom drawn otherwise shows wherever it lands.
    assert scored.counts[0].tolist() == bodies["check_pit"]["counts"]
    if name == "zeros_frequency":
        return
    responses = checks._replicate_responses(case.lss, case.validation)
    assert responses.shape == (99, case.validation.n_obs)
    statistics = _replicate_statistics(
        checks._scored_replicates(case.lss, case.validation, inputs=["x", "g"])
    )
    j = int(np.argsort(statistics["check_qq"])[4])
    assert statistics["check_qq"][j] == replicates.levels["check_qq"]
    copy = dataclasses.replace(case.validation, y=responses[j])
    bodies = {
        r: check_body(r, case.lss, copy, inputs=["x", "g"], replicates=replicates)
        for r in CHECK_REPORTS
    }
    _assert_same_statistics(statistics, j, _statistics(bodies))
    # The PIT histogram reads every transform, so every replicate's is its own split's.
    for i, y in enumerate(responses):
        split = dataclasses.replace(case.validation, y=y)
        pit = check_body("check_pit", case.lss, split, inputs=[], replicates=replicates)
        level = binomial_cell_level(pit["counts"], pit["n_rows"], 1 / 20)
        assert statistics["check_pit"][i] == level


def test_the_replicates_follow_the_fitted_law_with_its_offset(with_offset):
    """Over the replicates, the rows of high and of low exposure each average their fitted Mean.

    Monte Carlo error is under 1% here. Mutation check: drawing the responses without the
    offset leaves the low-exposure rows about 40% high and the high-exposure ones 30% low.
    """
    case = with_offset
    responses = checks._replicate_responses(case.lss, case.validation)
    predicted = case.lss.predict_parameters(case.validation.X)["mean"].to_numpy()
    high = case.validation.X["log_exposure"].to_numpy() > 0
    for rows in (high, ~high):
        assert responses[:, rows].mean() / predicted[rows].mean() == pytest.approx(1.0, abs=0.05)


_REFUSAL = (
    "the posterior covariance is not positive semidefinite at arithmetic resolution; "
    "its square root reconstruction is unresolved"
)


def test_a_refused_draw_falls_back_to_the_known_parameter_band(calm, monkeypatch):
    """Mutation check: without the fallback the refusal raises out of every body."""

    def refuse(*args, **kwargs):
        raise ValueError(_REFUSAL)

    monkeypatch.setattr(checks, "posterior_draws", refuse)
    replicates = null_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    assert replicates.n_draws == 0 and all(v is None for v in replicates.levels.values())
    for report in CHECK_REPORTS:
        body = check_body(report, calm.lss, calm.validation, inputs=["x", "g"])
        assert (body["level_source"], body["n_draws"]) == ("known", 0)
        assert body["level_note"] == (
            "The band treats the fitted parameters as exact, because the fit's uncertainty "
            "could not be sampled."
        )
        json.dumps(body, allow_nan=False)
    qq = check_body("check_qq", calm.lss, calm.validation, inputs=[])
    assert qq["local_level"] == equal_local_levels_band(calm.validation.n_obs).local_level


#: Held-out rows of the calibration check: twice CALM's 600 fitted rows, so the fitting noise
#: is large against the split's own and the known-parameter bands flag a correct model often.
_HELD_OUT = 1_200
#: Independent held-out splits, each with its own replicates.
_SPLITS = 40
#: Replicates per split: Barnard's rule at 5% is exact at 19 (the smallest) as at 99.
_DRAWS = 19


def _posterior_splits(case: Case) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """``_SPLITS`` response vectors on one set of new rows, each from its own coefficient draw.

    The coefficients are drawn with NumPy from ``N(beta_hat, V)`` and the responses from
    ``scipy.stats.gamma``: the holdout predictive check's own null, built without its sampler.
    """
    fitted = case.lss._require_fitted()
    rng = np.random.default_rng(SEED + 1)
    X, _, _ = _rows(_HELD_OUT, rng)
    values, vectors = np.linalg.eigh(fitted.inference.covariance)
    root = vectors * np.sqrt(np.clip(values, 0.0, None))
    beta = fitted.prediction_coefficients + rng.standard_normal((_SPLITS, len(values))) @ root.T
    draws = PosteriorDraws(
        coefficients=beta,
        covariance_kind="fixed",
        seed=0,
        coefficient_names=tuple(fitted.layout.coefficient_names),
    )
    theta = np.concatenate(
        [block for _, block in posterior_parameters(fitted, X, draws, name_levels=False)], axis=1
    )
    mean, cv = theta[..., 0], theta[..., 1]
    y = stats.gamma.rvs(a=1 / cv**2, scale=mean * cv**2, random_state=rng)
    return X, list(y)


def test_a_correct_model_is_flagged_at_its_level_on_a_held_out_split(calm):
    """Over independent splits each check flags about 5% of the time.

    Each split's responses come from a fresh posterior draw, so its rows and the replicates
    share one law, and each flag holds its level exactly: the count of flags is
    ``Binomial(40, 0.05)`` or, for the discrete PIT and quantile statistics, a little below.
    The limits hold 99.8% of that law. Mutation check: replicates simulated at the fitted
    coefficients alone, as the known-parameter bands assume, flag more than the limit allows.
    """
    X, responses = _posterior_splits(calm)
    flags = {report: 0 for report in CHECK_REPORTS}
    for seed, y in enumerate(responses):
        split = _validation(X, y)
        replicates = null_replicates(calm.lss, split, inputs=["x", "g"], n_draws=_DRAWS, seed=seed)
        for report in CHECK_REPORTS:
            body = check_body(report, calm.lss, split, inputs=["x", "g"], replicates=replicates)
            flags[report] += body["flagged"]
    low, high = stats.binom.ppf([0.001, 0.999], _SPLITS, 0.05)
    assert all(low <= count <= high for count in flags.values()), (flags, low, high)


# The /report route: each check for the model as edited and, after an edit, as fitted.

FIXTURE = Path(__file__).parent / "editor_frontend" / "fixtures" / "assess_checks.json"
#: The validation report's payload keys, read on master before the checks joined /report.
VALIDATION_KEYS = {
    "available",
    "can_run_cv",
    "compares_edit",
    "cv_report",
    "in_force",
    "metric_labels",
    "model_revision",
    "note",
    "report",
    "request_sequence",
    "splits",
    "title",
}
#: What the widget adds to ``check_report_payload``; the fixture holds the payload alone.
_WIDGET_FIELDS = ("model_revision", "request_sequence", "in_force")
_U = 2.0**-53


def _session(case: Case) -> EditorSession:
    return EditorSession.from_model(
        case.lss,
        train_data=(case.train.X, case.train.y),
        validation_data=(case.validation.X, case.validation.y),
    )


def _close_widgets(session: EditorSession) -> None:
    for widget in [w for w in widget_module._LIVE_WIDGETS if w.session is session]:
        widget.close()


@pytest.fixture
def calm_session(calm):
    session = _session(calm)
    yield session
    _close_widgets(session)


@pytest.fixture
def glm_session(poisson_glm):
    session = EditorSession.from_model(poisson_glm)
    yield session
    _close_widgets(session)


def _ask(widget, report, **extra):
    with widget._lock:
        rev, token = widget.session.model_revision, InForce.of(widget.session).token
    return widget._report(report, model_revision=rev, in_force=token, request_sequence=5, **extra)


def _average(session: EditorSession, count: int) -> None:
    session.select_indices("mean:x", list(range(count)))
    session.weighted_average("mean:x")


def test_each_check_report_answers_with_the_model_as_edited(calm_session):
    widget = calm_session.widget()
    assert widget._state()["checks"] == list(CHECK_REPORTS)
    for report in CHECK_REPORTS:
        payload = _ask(widget, report)
        assert payload["report"] == report and payload["request_sequence"] == 5
        assert payload["split"] == {"name": "validation", "label": "Validation", "n_obs": 200}
        assert payload["compares_edit"] is False and payload["models"]["original"] is None
        assert payload["in_force"] == InForce.of(calm_session).token
        assert payload["models"]["edited"] == check_body(
            report,
            calm_session.model,
            calm_session._evaluation_data["validation"],
            inputs=["x", "g"],
        )


def test_after_an_edit_the_report_carries_the_model_as_fitted(calm_session):
    widget = calm_session.widget()
    before = _ask(widget, "check_ae")["models"]["edited"]
    _average(calm_session, 6)
    after = _ask(widget, "check_ae")
    assert after["compares_edit"] is True and after["models"]["original"] == before
    assert after["models"]["edited"] != before


def test_the_fitted_half_is_computed_once_across_edits(calm_session, monkeypatch):
    calls = []
    real = checks.check_body
    monkeypatch.setattr(
        widget_module, "check_body", lambda *a, **k: calls.append(a[1]) or real(*a, **k)
    )
    widget = calm_session.widget()
    for k in (2, 4):
        _average(calm_session, k)
        _ask(widget, "check_pit")
    reference = calm_session.reference_model
    assert sum(model is reference for model in calls) == 1 and len(calls) == 3


def test_the_four_checks_share_one_simulation(calm_session, monkeypatch):
    """Counted wherever a body may draw them: the widget, or a body given none.

    Mutation checks: drawing the replicates for every body, or giving a body none, makes four
    calls per model.
    """
    drawn = []
    real = checks.null_replicates

    def counted(*a, **k):
        drawn.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(widget_module, "null_replicates", counted)
    monkeypatch.setattr(checks, "null_replicates", counted)
    widget = calm_session.widget()
    for report in CHECK_REPORTS:
        _ask(widget, report)
    assert len(drawn) == 1
    for k in (2, 4):
        _average(calm_session, k)
        for report in CHECK_REPORTS:
            assert _ask(widget, report)["compares_edit"] is True
    # One for each edited model; the fitted model's simulation from before the edits serves.
    assert len(drawn) == 3 and sum(model is calm_session.reference_model for model in drawn) == 1
    calm_session.undo()
    calm_session.undo()
    for report in CHECK_REPORTS:
        _ask(widget, report)
    assert len(drawn) == 3


def test_a_split_replaced_under_its_name_is_read_again(calm_session, calm):
    """Mutation checks: a body or a simulation kept by the split's name alone reads old rows."""
    widget = calm_session.widget()
    _average(calm_session, 6)
    _ask(widget, "check_pit")
    rows = calm.validation
    half = _validation(rows.X.iloc[:100].reset_index(drop=True), rows.y[:100])
    calm_session._evaluation_data["validation"] = half
    payload = _ask(widget, "check_pit")
    assert payload["split"]["n_obs"] == 100
    reference = calm_session.reference_model
    assert payload["models"]["original"] == check_body(
        "check_pit", reference, half, inputs=["x", "g"]
    )
    edited, _ = widget._current_model_for_evidence(statistics=False)
    assert payload["models"]["edited"] == check_body("check_pit", edited, half, inputs=["x", "g"])


def test_undo_returns_to_no_comparison(calm_session):
    widget = calm_session.widget()
    _average(calm_session, 2)
    assert _ask(widget, "check_qq")["compares_edit"] is True
    calm_session.undo()
    payload = _ask(widget, "check_qq")
    assert payload["compares_edit"] is False and payload["models"]["original"] is None


@pytest.mark.pins_behaviour
def test_a_stale_request_is_superseded(calm_session):
    """Pins that a check named for a model no longer in force is answered superseded."""
    widget = calm_session.widget()
    answer = widget._report("check_qq", model_revision=-1, in_force="gone", request_sequence=1)
    assert answer["status"] == "superseded"


def test_an_edit_while_the_check_is_drawn_supersedes_its_answer(calm_session, monkeypatch):
    """Mutation check: without the closing test the answer names a model no longer in force."""
    real = checks.check_body

    def edit_then_draw(*a, **k):
        if calm_session.model_revision == 0:
            _average(calm_session, 3)
        return real(*a, **k)

    monkeypatch.setattr(widget_module, "check_body", edit_then_draw)
    widget = calm_session.widget()
    asked = InForce.of(calm_session).token
    answer = _ask(widget, "check_pit")
    assert answer["status"] == "superseded" and answer["in_force"] == asked


def _with_covariance(lss, scale: float):
    """``lss``'s fit with its coefficients and ``scale`` times its covariance."""
    fitted = lss._require_fitted()
    state = fitted.fit_state
    covariance = scale * state.inference.covariance
    state = dataclasses.replace(
        state,
        inference=dataclasses.replace(state.inference, covariance=covariance),
        result=dataclasses.replace(state.result, covariance=covariance),
    )
    return dataclasses.replace(fitted, _fit_state=state)


def test_a_refit_that_moves_only_the_covariance_is_drawn_again(calm):
    """Both caches warm, then the fit replaced in place by one with the same coefficients and
    another covariance: each check answers as a fresh widget does. Mutation check: keyed by the
    scoring signature alone, the caches serve the old covariance's levels."""
    lss = copy.deepcopy(calm.lss)
    session = _session(calm._replace(lss=lss))
    try:
        widget = session.widget()
        for report in CHECK_REPORTS:
            _ask(widget, report)
        _average(session, 6)
        before = {report: _ask(widget, report) for report in CHECK_REPORTS}
        lss._model = _with_covariance(lss, 9.0)
        coefficients = lss._require_fitted().coefficients
        assert np.array_equal(coefficients, calm.lss._require_fitted().coefficients)
        fresh = session.widget()
        for report in CHECK_REPORTS:
            answer = _ask(widget, report)
            assert answer == _ask(fresh, report)
            assert answer["compares_edit"] is True
        assert answer["models"]["original"] != before[report]["models"]["original"]
    finally:
        _close_widgets(session)


def test_a_training_only_session_keeps_the_known_parameter_bands(calm):
    """A session opened with train_data alone reads its checks on the rows the model was fitted
    on. Mutation check: deciding held out by ``source == "retained_fit_data"`` simulates them."""
    session = EditorSession.from_model(calm.lss, train_data=(calm.train.X, calm.train.y))
    try:
        widget = session.widget()
        for report in CHECK_REPORTS:
            payload = _ask(widget, report)
            assert payload["split"]["name"] == "train"
            body = payload["models"]["edited"]
            assert body["available"] is True
            assert body["level_source"] == "known" and body["n_draws"] == 0
    finally:
        _close_widgets(session)


def test_a_superglm_session_offers_no_checks_and_says_why(glm_session):
    widget = glm_session.widget()
    assert widget._state()["checks"] == []
    payload = _ask(widget, "check_qq")
    assert payload["available"] is False
    assert payload["reason"] == "This check is available for SuperLSS models."


@pytest.mark.pins_behaviour
def test_the_four_reports_answer_as_before(calm_session):
    """Pins that the report names validation, cv, final and diagnose keep their payload keys."""
    widget = calm_session.widget()
    assert set(_ask(widget, "validation")) == VALIDATION_KEYS
    for report in ("cv", "final", "diagnose"):
        assert _ask(widget, report)["report"] == report


def _reports(session: EditorSession) -> dict[str, Any]:
    widget = session.widget()
    try:
        answers = {report: _ask(widget, report) for report in CHECK_REPORTS}
    finally:
        widget.close()
    return {
        report: {key: value for key, value in answer.items() if key not in _WIDGET_FIELDS}
        for report, answer in answers.items()
    }


def _assess_checks_fixture(calm: Case, wrong: Case) -> dict[str, Any]:
    """The four check payloads the editor's Assess graphs draw, by case.

    ``calm`` and ``wrong`` read the validation split (simulated levels); ``edited`` is the calm
    fit after a weighted average over ``mean:x``'s first six points, with the fitted model
    beside it; ``train`` is the calm fit on the rows it was fitted on (known levels).
    """
    edited = _session(calm)
    _average(edited, 6)
    train = {
        report: check_report_payload(
            report,
            dataset=calm.train,
            compares_edit=False,
            edited=check_body(report, calm.lss, calm.train, inputs=["x", "g"]),
            original=None,
        )
        for report in CHECK_REPORTS
    }
    cases = {
        "calm": _reports(_session(calm)),
        "wrong": _reports(_session(wrong)),
        "edited": _reports(edited),
        "train": train,
    }
    return json.loads(json.dumps(cases, allow_nan=False))


def _assert_close_json(stored: Any, fresh: Any, path: str = "$", scale: float | None = None):
    """Numbers within ``64 u M``, ``M`` the largest magnitude of their list; the rest equal."""
    if isinstance(stored, dict):
        assert isinstance(fresh, dict) and stored.keys() == fresh.keys(), path
        for key in stored:
            _assert_close_json(stored[key], fresh[key], f"{path}.{key}")
    elif isinstance(stored, list):
        assert isinstance(fresh, list) and len(stored) == len(fresh), path
        numbers = [abs(x) for x in stored if type(x) is float]
        largest = max(numbers, default=0.0)
        for i, (a, b) in enumerate(zip(stored, fresh, strict=True)):
            _assert_close_json(a, b, f"{path}[{i}]", largest)
    elif type(stored) is float and type(fresh) is float:
        bound = 64 * _U * (abs(stored) if scale is None else max(scale, abs(stored)))
        assert abs(stored - fresh) <= bound, (path, stored, fresh)
    else:
        assert type(stored) is type(fresh) and stored == fresh, (path, stored, fresh)


def test_the_assess_checks_fixture_is_current(calm, wrong):
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fresh = _assess_checks_fixture(calm, wrong)
    assert {case["check_qq"]["models"]["edited"]["level_source"] for case in fresh.values()} == {
        "simulated",
        "known",
    }
    _assert_close_json(stored, fresh)


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _assess_checks_fixture(_calm_case(), _wrong_case()),
        indent=1,
        sort_keys=True,
        allow_nan=False,
    )
    FIXTURE.write_text(text + "\n", encoding="utf-8")
