"""The editor's four SuperLSS checks on one split, against SciPy laws and exact counts."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

from superglm import (
    Categorical,
    GammaLS,
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
from superglm.editor import checks
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


@pytest.fixture(scope="module")
def calm() -> Case:
    return _fitted_case(lambda rng, mean, cv: rng.gamma(1.0 / cv**2, mean * cv**2))


@pytest.fixture(scope="module")
def wrong() -> Case:
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
    assert replicates.statistics["check_ae"] is None
    for report in ("check_qq", "check_pit", "check_calibration"):
        body = check_body(
            report, heavy_gpd.lss, heavy_gpd.validation, inputs=["x"], replicates=replicates
        )
        assert body["available"] and body["level_source"] == "simulated"
        assert body["n_draws"] == 99 and len(replicates.statistics[report]) == 99


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

    Actual over expected is left out: its cells are equal-count over rows, so a weighted split
    and its repeated rows are cut into different cells (Task 2's tables, unchanged here).
    """
    weighted, duplicated = calm_frequency
    for report in ("check_qq", "check_pit", "check_calibration"):
        a = check_body(report, weighted.lss, weighted.validation, inputs=[])
        b = check_body(report, duplicated.lss, duplicated.validation, inputs=[])
        assert a["level_source"] == b["level_source"] == "simulated"
        assert a["local_level"] == pytest.approx(b["local_level"], rel=1e-9)
        assert (a["outside"], a["flagged"]) == (b["outside"], b["flagged"])


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


def test_a_held_out_split_reads_each_level_from_99_replicates(calm):
    """Barnard's rule at 99 replicates: the 5th smallest level, or the 5th largest ``|z|``."""
    replicates = null_replicates(calm.lss, calm.validation, inputs=["x", "g"])
    assert (replicates.n_draws, replicates.seed, replicates.note) == (99, 42, None)
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
    eta = monte_carlo_level(replicates.statistics["check_qq"], alpha=0.05)
    assert qq["local_level"] == eta
    band = band_at_level(qq["n"], eta)
    k = np.asarray(qq["index"])
    np.testing.assert_array_equal(qq["band_lower"], special.ndtri(band.lower[k]))
    np.testing.assert_array_equal(qq["band_upper"], special.ndtri(band.upper[k]))
    for report in ("check_pit", "check_calibration"):
        level = monte_carlo_level(replicates.statistics[report], alpha=0.05)
        assert bodies[report]["local_level"] == level
    ae = bodies["check_ae"]
    assert ae["z"] == np.sort(replicates.statistics["check_ae"])[-5]
    for table in ae["tables"]:
        ratio, se = np.asarray(table["ratio"]), np.asarray(table["ratio_se"])
        np.testing.assert_array_equal(table["lower"], ratio - ae["z"] * se)
        np.testing.assert_array_equal(table["upper"], ratio + ae["z"] * se)


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


@pytest.mark.parametrize("name", ["calm", "prior_weighted", "zeros"])
def test_a_replicate_is_scored_as_the_split_is(name, request):
    """The statistics of one replicate are those its responses give as a split of their own.

    The replicate is the one that sets the Q-Q level, whose Q-Q statistic is exact; every
    replicate's PIT statistic is checked too. Prior weights enter each law and each total, and
    a response at the atom is randomised from the split's own stream. Mutation checks: scoring
    each replicate under the draw it was simulated from, totals without the prior weights, and
    atoms randomised from another stream each fail it.
    """
    case = request.getfixturevalue(name)
    replicates = null_replicates(case.lss, case.validation, inputs=["x", "g"])
    responses = checks._replicate_responses(case.lss, case.validation)
    assert responses.shape == (99, case.validation.n_obs)
    j = int(np.argsort(replicates.statistics["check_qq"])[4])
    copy = dataclasses.replace(case.validation, y=responses[j])
    bodies = {
        r: check_body(r, case.lss, copy, inputs=["x", "g"], replicates=replicates)
        for r in CHECK_REPORTS
    }
    observed = _statistics(bodies)
    for report in ("check_pit", "check_calibration"):
        assert replicates.statistics[report][j] == observed[report]
    for report in ("check_qq", "check_ae"):
        assert replicates.statistics[report][j] == pytest.approx(observed[report], rel=1e-9)
    # The PIT histogram reads every transform, so every replicate's is its own split's.
    for i, y in enumerate(responses):
        split = dataclasses.replace(case.validation, y=y)
        pit = check_body("check_pit", case.lss, split, inputs=[], replicates=replicates)
        level = binomial_cell_level(pit["counts"], pit["n_rows"], 1 / 20)
        assert replicates.statistics["check_pit"][i] == level


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
    assert replicates.n_draws == 0 and all(v is None for v in replicates.statistics.values())
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
