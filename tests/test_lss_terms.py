"""Contract tests for per-parameter term inference on a distributional fit."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from superglm import Categorical, Numeric, Spline, SuperLSS
from superglm.distributional import Predictor
from superglm.distributional import terms as terms_module
from superglm.distributional.families.gaussian import GaussianLS
from superglm.distributional.model import DenseDistributionalModel
from superglm.distributional.terms import (
    ParameterTermEffect,
    TermTest,
    summary_table,
    term_effect,
    term_test,
)
from superglm.features.factor_smooth import FactorSmooth
from superglm.features.interaction import TensorInteraction
from superglm.features.ordered_categorical import OrderedCategorical

SUMMARY_COLUMNS = (
    "parameter",
    "term",
    "edf",
    "lambda",
    "statistic",
    "rank",
    "p_value",
    "estimate",
    "se",
    "note",
)


def _simulated(n: int = 1200, seed: int = 20260903) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    w = rng.uniform(-2.0, 2.0, n)
    location = (
        0.9 * np.sin(2.4 * x) + np.where(g == "a", 0.5, np.where(g == "b", -0.4, 0.0)) + 0.35 * w
    )
    scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
    frame = pd.DataFrame({"x": x, "g": g, "w": w})
    return frame, location + scale * rng.standard_normal(n)


def _fit(frame: pd.DataFrame, y: np.ndarray) -> DenseDistributionalModel:
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location",
                {"x": Spline("cr", k=8), "g": Categorical(), "w": Numeric()},
            ),
            Predictor("scale", {"x": Spline("cr", k=6)}),
        ],
    ).fit_reml(frame, y, outer="efs+newton")
    return model._require_fitted()


@pytest.fixture(scope="module")
def fit_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    frame, y = _simulated()
    return _fit(frame, y), frame


@pytest.fixture(scope="module")
def interaction_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    rng = np.random.default_rng(4242)
    n = 600
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    y = 0.8 * x1 * x2 + 0.4 * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location",
                {"x1": Spline("cr", k=5), "x2": Spline("cr", k=5)},
                interaction_specs={"x1:x2": TensorInteraction("x1", "x2", n_knots=(4, 4))},
            ),
            Predictor("scale", {"x1": Numeric()}, intercept=False),
        ],
    ).fit_reml(frame, y)
    return model._require_fitted(), frame


_SPECIAL_LEVELS = ("l1", "l2", "l3", "l4", "l5")


@pytest.fixture(scope="module")
def specials_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    """An ordered categorical whose ``MISSING`` level sits beside the smooth."""
    rng = np.random.default_rng(20260903)
    n = 900
    draw = rng.choice(_SPECIAL_LEVELS, n)
    band = np.where(rng.uniform(size=n) < 0.12, "MISSING", draw)
    position = np.array(
        [_SPECIAL_LEVELS.index(value) if value in _SPECIAL_LEVELS else 0 for value in band],
        dtype=float,
    )
    x = rng.uniform(-1.0, 1.0, n)
    frame = pd.DataFrame({"band": band, "x": x})
    effect = np.where(band == "MISSING", 1.4, 0.35 * position)
    y = effect + 0.5 * x + 0.3 * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location",
                {
                    "band": OrderedCategorical(
                        order=list(_SPECIAL_LEVELS),
                        basis=Spline("cr", k=3),
                        specials=["MISSING"],
                    ),
                    "x": Numeric(),
                },
            ),
            Predictor("scale", {}),
        ],
    ).fit_reml(frame, y)
    return model._require_fitted(), frame


@pytest.fixture(scope="module")
def absorbed_case() -> tuple[DenseDistributionalModel, pd.DataFrame]:
    """A level term the factor smooth's own main effect leaves unidentified.

    A constrained factor smooth on ``g`` spans the level indicators exactly, so
    the unpenalized ``g`` block has no estimable direction of its own: its EDF
    and its covariance block are zero while the min-norm solve still writes a
    non-zero number into its coefficients.
    """
    rng = np.random.default_rng(1)
    n = 900
    x = rng.uniform(-1.0, 1.0, n)
    g = rng.choice(["a", "b", "c"], n)
    level = pd.Series(g).map({"a": 0.4, "b": -0.3, "c": 0.0}).to_numpy()
    frame = pd.DataFrame({"x": x, "g": g})
    y = 0.7 * np.sin(2.2 * x) + level + level * x + 0.3 * rng.standard_normal(n)
    model = SuperLSS(
        family=GaussianLS(),
        predictors=[
            Predictor(
                "location",
                {"x": Spline("cr", k=6), "g": Categorical()},
                interaction_specs={"x:g:sz": FactorSmooth("x", group="g", basis="sz", k=6)},
            ),
            Predictor("scale", {}),
        ],
    ).fit(
        frame,
        y,
        lambdas={"location:x#wiggle": 1.0, "location:x:g:sz#wiggle": 1.0},
    )
    return model._require_fitted(), frame


def test_spline_effect_spans_the_training_range(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "x")

    assert isinstance(effect, ParameterTermEffect)
    assert effect.parameter == "location"
    assert effect.term == "x"
    assert effect.kind == "spline"
    assert effect.link == "IdentityLink"
    assert effect.covariance_kind == "fixed"
    assert effect.alpha == 0.05
    assert effect.levels is None
    assert effect.x is not None
    assert effect.x.shape == (200,)
    assert effect.x[0] == pytest.approx(frame["x"].min())
    assert effect.x[-1] == pytest.approx(frame["x"].max())
    assert np.all(np.diff(effect.x) > 0.0)
    assert effect.effect.shape == effect.se.shape == (200,)
    assert np.all(effect.se > 0.0)
    assert np.all(effect.lower < effect.effect)
    assert np.all(effect.effect < effect.upper)
    assert effect.edf == pytest.approx(fitted.inference.term_edf["location:x"])
    assert dict(effect.lambdas) == {"location:x#wiggle": fitted.lambdas["location:x#wiggle"]}
    # Identity link: no multiplicative reading of a link-scale effect.
    assert effect.multiplier is None


def test_pointwise_band_is_the_normal_interval(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "x", alpha=0.1, n_points=40)

    critical = stats.norm.ppf(0.95)
    assert effect.alpha == 0.1
    assert effect.lower == pytest.approx(effect.effect - critical * effect.se)
    assert effect.upper == pytest.approx(effect.effect + critical * effect.se)


def test_simultaneous_band_is_wider_than_pointwise(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "x", n_points=60, n_sim=4000, seed=3)

    assert effect.critical_value is not None
    assert effect.critical_value >= stats.norm.ppf(0.975)
    assert effect.lower_simultaneous is not None
    assert effect.upper_simultaneous is not None
    assert np.all(effect.lower_simultaneous < effect.lower)
    assert np.all(effect.upper_simultaneous > effect.upper)


def test_simultaneous_band_can_be_switched_off(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "x", n_points=20, simultaneous=False)

    assert effect.critical_value is None
    assert effect.lower_simultaneous is None
    assert effect.upper_simultaneous is None


def test_log_link_term_reports_a_multiplier(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "scale", "x", n_points=50)

    assert effect.link == "LowerBoundedLogLink"
    assert effect.multiplier is not None
    assert effect.multiplier == pytest.approx(np.exp(effect.effect))


def test_categorical_effect_is_one_row_per_level_against_the_reference(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "g")
    spec = fitted.compiled_predictors[0].compiled.specs["g"]

    assert effect.kind == "categorical"
    assert effect.x is None
    assert effect.levels == tuple(str(level) for level in spec._levels)
    assert effect.lambdas == {}

    base = list(spec._levels).index(spec._base_level)
    assert effect.effect[base] == 0.0
    assert effect.se[base] == 0.0
    assert effect.lower[base] == 0.0
    assert effect.upper[base] == 0.0

    beta = np.asarray(fitted.coefficients)[fitted.layout.term_slices["location:g"]]
    for column, level in enumerate(spec._non_base):
        row = list(spec._levels).index(level)
        assert effect.effect[row] == pytest.approx(beta[column])
        assert effect.se[row] > 0.0
    assert np.all(effect.lower <= effect.effect)
    assert np.all(effect.effect <= effect.upper)


def test_single_column_numeric_term_is_reported_as_a_line(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "w", n_points=30)

    assert effect.kind == "numeric"
    slope = float(np.asarray(fitted.coefficients)[fitted.layout.term_slices["location:w"]][0])
    assert effect.effect == pytest.approx(slope * effect.x)


def test_effect_serialises_to_plain_json(fit_case) -> None:
    fitted, frame = fit_case
    payload = term_effect(fitted, frame, "location", "x", n_points=12).to_json()
    restored = json.loads(json.dumps(payload))

    assert restored == payload
    assert restored["schema_version"] == 1
    assert restored["kind"] == "spline"
    assert restored["multiplier"] is None
    assert len(restored["effect"]) == 12
    for key, value in restored.items():
        if key == "lambdas":
            assert all(isinstance(item, float) for item in value.values())
            continue
        assert isinstance(value, list | float | int | str | type(None))


def test_categorical_effect_serialises_its_levels(fit_case) -> None:
    fitted, frame = fit_case
    payload = term_effect(fitted, frame, "location", "g", simultaneous=False).to_json()

    assert payload["x"] is None
    assert payload["levels"] == ["a", "b", "c"]
    assert payload["critical_value"] is None
    assert json.loads(json.dumps(payload)) == payload


def test_wood_test_rejects_a_real_smooth(fit_case) -> None:
    fitted, frame = fit_case
    outcome = term_test(fitted, frame, "location", "x")

    assert isinstance(outcome, TermTest)
    assert outcome.parameter == "location"
    assert outcome.term == "x"
    assert outcome.statistic > 0.0
    assert outcome.rank > 1.0
    assert outcome.p_value < 1e-6
    assert outcome.edf == pytest.approx(fitted.inference.term_edf["location:x"])
    assert json.loads(json.dumps(outcome.to_json())) == outcome.to_json()


def test_categorical_term_test_is_a_block_wald_test(fit_case, monkeypatch) -> None:
    from superglm.distributional import terms as module

    fitted, frame = fit_case

    def refuse(*args: object) -> None:
        raise AssertionError("a level block must not take the smooth-test path")

    monkeypatch.setattr(module, "wood_test_smooth", refuse)
    outcome = term_test(fitted, frame, "location", "g")

    term_slice = fitted.layout.term_slices["location:g"]
    beta = np.asarray(fitted.coefficients)[term_slice]
    block = np.asarray(fitted.inference.covariance)[term_slice, term_slice]
    expected = float(beta @ np.linalg.solve(block, beta))

    assert outcome.rank == pytest.approx(2.0)
    assert outcome.statistic == pytest.approx(expected, rel=1e-8)
    assert outcome.p_value == pytest.approx(stats.chi2.sf(expected, 2.0), rel=1e-8)
    assert outcome.p_value < 1e-6


def test_wood_test_keeps_null_p_values_calibrated() -> None:
    """A pure-noise smooth must not be declared significant seed after seed."""
    accepted = 0
    for seed in range(10):
        rng = np.random.default_rng(1000 + seed)
        n = 600
        x = rng.uniform(-1.0, 1.0, n)
        z = rng.uniform(-1.0, 1.0, n)
        frame = pd.DataFrame({"x": x, "z": z})
        scale = np.exp(-1.0 + 0.5 * np.cos(1.8 * x))
        y = 0.9 * np.sin(2.4 * x) + scale * rng.standard_normal(n)
        model = SuperLSS(
            family=GaussianLS(),
            predictors=[
                Predictor("location", {"x": Spline("cr", k=8), "z": Spline("cr", k=6)}),
                Predictor("scale", {"x": Spline("cr", k=6)}),
            ],
        ).fit_reml(frame, y)
        fitted = model._require_fitted()
        outcome = term_test(fitted, frame, "location", "z")
        assert 0.0 <= outcome.p_value <= 1.0
        accepted += int(outcome.p_value > 0.01)
    assert accepted >= 8


def test_summary_table_covers_every_term_and_intercept(fit_case) -> None:
    fitted, frame = fit_case
    table = summary_table(fitted, frame)

    assert tuple(table.columns) == SUMMARY_COLUMNS
    intercepts = [state for state in fitted.layout.predictors if state.intercept_index is not None]
    assert len(table) == len(fitted.layout.term_slices) + len(intercepts)
    assert list(table["parameter"]) == sorted(table["parameter"], key=["location", "scale"].index)

    terms = table[table["term"] != "(intercept)"]
    assert set(terms["parameter"] + ":" + terms["term"]) == set(fitted.layout.term_slices)
    assert np.all((terms["p_value"] >= 0.0) & (terms["p_value"] <= 1.0))
    assert np.all(terms["rank"] > 0.0)

    smooth = table[(table["parameter"] == "location") & (table["term"] == "x")].iloc[0]
    assert smooth["edf"] == pytest.approx(fitted.inference.term_edf["location:x"])
    assert smooth["lambda"] == pytest.approx(fitted.lambdas["location:x#wiggle"])
    assert math.isnan(smooth["estimate"])
    assert math.isnan(smooth["se"])

    line = table[(table["parameter"] == "location") & (table["term"] == "w")].iloc[0]
    beta = np.asarray(fitted.coefficients)
    assert line["estimate"] == pytest.approx(beta[fitted.layout.term_slices["location:w"]][0])
    assert math.isnan(line["lambda"])
    # The scale is modelled here, so the reference is chi-square and never an F.
    assert line["rank"] == pytest.approx(1.0)
    assert line["p_value"] == pytest.approx(
        stats.chi2.sf(line["statistic"], line["rank"]), rel=1e-9
    )

    row = table[(table["parameter"] == "scale") & (table["term"] == "(intercept)")].iloc[0]
    index = fitted.layout.predictor("scale").intercept_index
    covariance = np.asarray(fitted.inference.covariance)
    assert row["estimate"] == pytest.approx(beta[index])
    assert row["se"] == pytest.approx(math.sqrt(covariance[index, index]))
    assert row["rank"] == pytest.approx(1.0)
    assert row["p_value"] == pytest.approx(
        stats.chi2.sf((beta[index] / row["se"]) ** 2, 1.0), rel=1e-8
    )
    assert row["edf"] == pytest.approx(fitted.inference.intercept_edf["scale:(intercept)"])


def test_summary_prepares_a_later_missing_term_column_before_covariance(
    fit_case,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, frame = fit_case
    slices = fitted.layout.term_slices
    reordered = {
        "location:g": slices["location:g"],
        "location:w": slices["location:w"],
        **{
            name: block
            for name, block in slices.items()
            if name not in {"location:g", "location:w"}
        },
    }
    layout = dataclasses.replace(fitted.layout, term_slices=reordered)
    state = dataclasses.replace(fitted.fit_state, layout=layout)
    ordered = DenseDistributionalModel(family=fitted.family, _fit_state=state)
    calls: list[str] = []

    def covariance_spy(model, *, kind="fixed") -> np.ndarray:
        calls.append(kind)
        return np.asarray(model.inference.covariance)

    monkeypatch.setattr(terms_module, "posterior_covariance", covariance_spy)
    with pytest.raises(KeyError, match="w"):
        summary_table(ordered, frame.drop(columns="w"), covariance="corrected")

    assert calls == []


def test_interaction_term_is_tested_on_the_training_design(interaction_case) -> None:
    fitted, frame = interaction_case
    outcome = term_test(fitted, frame, "location", "x1:x2")

    assert 0.0 <= outcome.p_value <= 1.0
    assert outcome.statistic >= 0.0

    table = summary_table(fitted, frame)
    labels = table["parameter"].str.cat(table["term"], sep=":").tolist()
    assert "location:x1:x2" in labels
    # The scale predictor carries no intercept, so the table gives it no row.
    assert "scale:(intercept)" not in labels

    with pytest.raises(NotImplementedError, match="one-dimensional"):
        term_effect(fitted, frame, "location", "x1:x2")


def test_test_rank_is_woods_alternative_edf(fit_case) -> None:
    """Wood (2013) section 2.4: the rank is ``sum diag(2F - FF)``, not ``tr F``."""
    fitted, frame = fit_case
    influence = np.asarray(fitted.inference.influence)
    term_slice = fitted.layout.term_slices["location:x"]
    edf = fitted.inference.term_edf["location:x"]
    alternative = 2.0 * edf - float(np.trace(influence[term_slice, :] @ influence[:, term_slice]))

    outcome = term_test(fitted, frame, "location", "x")
    # The two candidate rules must be far enough apart here that the assertion
    # below is about the rule and not about floating point.
    assert alternative - edf > 0.1
    assert outcome.rank == pytest.approx(alternative, rel=1e-9)
    assert outcome.edf == pytest.approx(edf)


def test_the_smooth_test_is_told_the_scale_is_modelled(fit_case, monkeypatch) -> None:
    """A modelled scale means a chi-square reference, never an F reference."""
    from superglm.distributional import terms as module

    fitted, frame = fit_case
    seen: dict[str, float] = {}
    real = module.wood_test_smooth

    def spy(beta, design, block, edf1, res_df):
        seen["res_df"] = float(res_df)
        return real(beta, design, block, edf1, res_df)

    monkeypatch.setattr(module, "wood_test_smooth", spy)
    term_test(fitted, frame, "location", "x")
    assert seen["res_df"] <= 0.0


def test_helper_guards_hold_their_edges(fit_case, absorbed_case, specials_case) -> None:
    from superglm.distributional import terms as module

    fitted, _ = fit_case
    absorbed, _ = absorbed_case
    with_specials, _ = specials_case

    class _Unlinked:
        pass

    class _WrongShape:
        def deriv_inverse(self, eta):
            return np.ones(1)

    assert module._term_kind(_Unlinked()) == "other"
    assert module._is_shifted_log_link(_Unlinked()) is False
    assert module._is_shifted_log_link(_WrongShape()) is False
    assert module._level_domain(_Unlinked()) is None

    class _Ordered:
        _ordered_levels = ("low", "high")

    assert module._level_domain(_Ordered()) == ("low", "high")

    with pytest.raises(KeyError, match="unknown predictor"):
        module._compiled_spec(fitted, "nope", "x")
    with pytest.raises(ValueError, match="materially negative"):
        module._band_variance(np.eye(2), -np.eye(2))
    assert module._block_wald(np.zeros(2), np.zeros((2, 2)), 1e-6) == (0.0, 0.0, 1.0)
    assert module._json_number(float("nan")) is None
    assert module._json_array(None) is None
    assert module._special_domain(_Unlinked()) == ()
    assert module._summary_label("band") == "band"

    # The absorption note is a conjunction: an estimable term, a term with no
    # levels, and a level term no interaction is built on each read as silence.
    ones = np.ones(1)
    assert module._absorption_note(fitted, "location", "g", 1.0, ones) == ""
    assert module._absorption_note(fitted, "location", "g", 0.0, np.zeros(1)) == ""
    assert module._absorption_note(fitted, "location", "x", 0.0, ones) == ""
    assert module._absorption_note(with_specials, "location", "band", 0.0, ones) == ""
    assert module._interaction_on(absorbed, "location", "unrelated") is None


def test_unknown_parameter_and_term_are_refused(fit_case) -> None:
    fitted, frame = fit_case

    with pytest.raises(KeyError, match="nope"):
        term_effect(fitted, frame, "nope", "x")
    with pytest.raises(KeyError, match="location:nope"):
        term_test(fitted, frame, "location", "nope")
    with pytest.raises(ValueError, match="n_points"):
        term_effect(fitted, frame, "location", "x", n_points=1)
    with pytest.raises(ValueError, match="n_points"):
        term_test(fitted, frame, "location", "x", n_points=1)
    with pytest.raises(ValueError, match="alpha"):
        term_effect(fitted, frame, "location", "x", alpha=1.5)


# --------------------------------------------------------------------------- #
# Special levels and absorbed level terms
# --------------------------------------------------------------------------- #


def test_special_levels_carry_their_own_effect_and_error(specials_case) -> None:
    """A free special level reads its own coefficient, not the smooth's zero."""
    fitted, frame = specials_case
    effect = term_effect(fitted, frame, "location", "band")

    assert effect.kind == "categorical"
    assert effect.levels == (*_SPECIAL_LEVELS, "MISSING")
    assert effect.special == (False, False, False, False, False, True)

    special_slice = fitted.layout.term_slices["location:band:special"]
    coefficient = float(np.asarray(fitted.coefficients)[special_slice][0])
    assert abs(coefficient) > 0.1
    assert float(effect.effect[-1]) == pytest.approx(coefficient)
    assert float(effect.se[-1]) > 0.0
    assert float(effect.lower[-1]) < coefficient < float(effect.upper[-1])
    assert np.all(effect.se[:-1] >= 0.0)

    assert effect.edf == pytest.approx(
        fitted.inference.term_edf["location:band"]
        + fitted.inference.term_edf["location:band:special"]
    )


def test_special_flags_serialise_beside_the_levels(specials_case) -> None:
    fitted, frame = specials_case
    payload = term_effect(fitted, frame, "location", "band", simultaneous=False).to_json()

    assert payload["special"] == [False, False, False, False, False, True]
    assert payload["levels"][-1] == "MISSING"
    assert json.loads(json.dumps(payload)) == payload


def test_a_term_without_special_levels_reports_none(fit_case) -> None:
    fitted, frame = fit_case
    effect = term_effect(fitted, frame, "location", "g", simultaneous=False)

    assert effect.special is None
    assert effect.to_json()["special"] is None
    assert effect.edf == pytest.approx(fitted.inference.term_edf["location:g"])


def test_summary_table_labels_the_special_block(specials_case) -> None:
    fitted, frame = specials_case
    table = summary_table(fitted, frame)

    assert "band:special" not in set(table["term"])
    row = table[table["term"] == "band (special level)"]
    assert len(row) == 1
    special_slice = fitted.layout.term_slices["location:band:special"]
    assert float(row["estimate"].iloc[0]) == pytest.approx(
        float(np.asarray(fitted.coefficients)[special_slice][0])
    )
    assert float(row["se"].iloc[0]) > 0.0
    assert set(table["note"]) == {""}


def test_summary_table_notes_a_level_term_absorbed_by_its_interaction(absorbed_case) -> None:
    fitted, frame = absorbed_case
    table = summary_table(fitted, frame)

    term_slice = fitted.layout.term_slices["location:g"]
    coefficients = np.asarray(fitted.coefficients)[term_slice]
    assert fitted.inference.term_edf["location:g"] == pytest.approx(0.0, abs=1e-12)
    assert np.any(coefficients != 0.0)

    absorbed = table[(table["parameter"] == "location") & (table["term"] == "g")].iloc[0]
    assert absorbed["note"] == "absorbed by x:g:sz"
    assert float(absorbed["statistic"]) == 0.0
    assert float(absorbed["p_value"]) == 1.0

    others = table[table["term"] != "g"]
    assert set(others["note"]) == {""}


def test_corrected_covariance_refuses_without_retained_rows(fit_case) -> None:
    fitted, frame = fit_case
    compact_state = dataclasses.replace(fitted.fit_state, retained_rows=None)
    compact = DenseDistributionalModel(family=fitted.family, _fit_state=compact_state)

    with pytest.raises(RuntimeError, match="retained training rows"):
        term_effect(compact, frame, "location", "x", covariance="corrected", n_points=8)
