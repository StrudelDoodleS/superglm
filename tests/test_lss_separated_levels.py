"""Inference and disclosure for categorical separation in SuperLSS."""

import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from superglm import CategoricalInteraction, Constraint, SuperLSS, cat, collapse_levels, s
from superglm.diagnostics.separation import SeparationError, SeparationWarning
from superglm.distributional import NegativeBinomialLS
from superglm.terms import interaction


def _model(term=None, *, smooth=False, intercept=True, family=None, **kwargs):
    family = NegativeBinomialLS() if family is None else family
    terms = (cat("lev"),) if term is None else (term,)
    if smooth:
        terms += (s("x", n_knots=6),)
    return SuperLSS(
        family,
        family.mean(*terms, offset="log_expo" if smooth else None, intercept=intercept),
        family.theta(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def separated_book():
    rng = np.random.default_rng(0)
    n = 20_000
    lev = rng.choice(list("ABCDE"), n, p=[0.4, 0.3, 0.2, 0.098, 0.002])
    x = rng.uniform(0, 1, n)
    exposure = rng.uniform(0.2, 1, n)
    y = rng.poisson(exposure * 0.08 * np.exp(0.5 * x) * np.where(lev == "B", 1.3, 1.0))
    y[lev == "E"] = 0
    X = pd.DataFrame({"lev": lev, "x": x, "log_expo": np.log(exposure)})
    return X, y.astype(float)


@pytest.fixture(scope="module")
def separated_fit(separated_book):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        return _model(smooth=True).fit_reml(*separated_book, retain_rows=False)


def test_joint_wald_uses_the_healthy_marginal_covariance(separated_fit, separated_book):
    model = separated_fit
    X, y = separated_book
    keep = X.lev != "E"
    without = _model(smooth=True).fit_reml(X.loc[keep], y[keep])
    # A is the base, so B, C and D are the first three mean contrasts after the intercept.
    healthy = [1, 2, 3]
    beta = model.coef_by_predictor_["mean"][healthy]
    covariance = model.covariance_[np.ix_(healthy, healthy)]
    expected = beta @ np.linalg.solve(covariance, beta)
    term = model.term_test("mean", "lev")
    row = model.summary().query("parameter == 'mean' and term == 'lev'").iloc[0]
    assert term.rank == row["rank"] == 3
    for report in (term, row):
        assert report.statistic == pytest.approx(expected, rel=1e-12)
        assert report.p_value == pytest.approx(chi2.sf(expected, 3), rel=1e-12, abs=0)
        assert "SEP" in report.note and "E" in report.note and "excluded" in report.note
    reference = without.term_test("mean", "lev")
    assert term.statistic == pytest.approx(reference.statistic, rel=2e-3)
    assert term.p_value == pytest.approx(reference.p_value, rel=0.03, abs=0)
    with_rows = model.term_inference("mean", "lev", simultaneous=False)
    without_rows = without.term_inference("mean", "lev", simultaneous=False)
    np.testing.assert_allclose(with_rows.effect[1:4], without_rows.effect[1:4], rtol=2e-3)
    np.testing.assert_allclose(with_rows.se[1:4], without_rows.se[1:4], rtol=2e-3)
    with pytest.warns(UserWarning, match="conditional|fixed"):
        corrected = model.term_test("mean", "lev", covariance="corrected")
    assert corrected.statistic == term.statistic
    assert corrected.rank == 3
    assert (
        "SEP" in corrected.note
        and "corrected covariance unavailable: fixed shown" in corrected.note
    )


def test_separated_reference_leaves_the_test_and_intercept_blank(separated_book):
    with pytest.warns(SeparationWarning):
        model = _model(cat("lev", base="E", unseen="base"), smooth=True).fit_reml(*separated_book)
    term = model.term_test("mean", "lev")
    assert np.isnan([term.statistic, term.rank, term.p_value]).all()
    assert "SEP" in term.note and "'E'" in term.note and 'base="most_exposed"' in term.note
    rows = model.summary().query("parameter == 'mean'").set_index("term")
    assert (
        rows.loc[["(intercept)", "lev"], ["se", "statistic", "rank", "p_value"]].isna().all().all()
    )
    assert "SEP" in rows.loc["(intercept)", "note"]
    effect = model.term_inference("mean", "lev")
    assert all(effect.separated)
    assert np.isnan(effect.se).all()
    assert np.isnan(effect.lower_simultaneous).all()
    score = pd.DataFrame({"lev": ["new"], "x": [0.5], "log_expo": [0.0]})
    with pytest.warns(SeparationWarning, match="mean:lev.*E"):
        model.predict(score)


@pytest.fixture
def two_levels():
    X = pd.DataFrame({"lev": np.repeat(["A", "E"], 125)})
    X.loc[124, "lev"] = "E"
    y = np.where(X.lev == "A", np.arange(len(X)) % 6, 0).astype(float)
    return X, y


def test_healthy_base_with_no_identified_contrasts_has_rank_zero(two_levels):
    with pytest.warns(SeparationWarning):
        model = _model(cat("lev", base="A")).fit(*two_levels)
    term = model.term_test("mean", "lev")
    assert term.rank == 0
    assert np.isnan([term.statistic, term.p_value]).all()
    assert "E" in term.note and "no identified contrasts" in term.note


@pytest.mark.parametrize("method", ["fit", "fit_reml"])
def test_fit_discloses_after_terminal_preparation_and_before_installation(
    two_levels, monkeypatch, method
):
    from superglm.distributional import model as dense_model

    X, y = two_levels
    healthy_y = np.arange(len(X), dtype=float) % 6
    model = _model(cat("lev", base="A"))
    fit = getattr(model, method)
    fit(X, healthy_y)
    previous = model._require_fitted().fit_state
    prediction = model.predict(X)
    prepared = []
    prepare = dense_model.prepare_distributional_fit_state

    def observe_terminal(*args, **kwargs):
        state = prepare(*args, **kwargs)
        prepared.append(state.revision)
        return state

    monkeypatch.setattr(dense_model, "prepare_distributional_fit_state", observe_terminal)
    with warnings.catch_warnings(), pytest.raises(SeparationWarning, match="mean:lev"):
        warnings.simplefilter("error", SeparationWarning)
        fit(X, y)
    assert prepared == [previous.revision + 1]
    assert model._require_fitted().fit_state is previous
    np.testing.assert_array_equal(model.predict(X), prediction)
    with pytest.warns(SeparationWarning) as caught:
        fit(X, y)
    assert len(caught) == 1 and "where the fit stopped" in str(caught[0].message)
    assert "SEP" in model.term_test("mean", "lev").note
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        fit(X, healthy_y)
        model.predict(X)
    assert "SEP" not in model.term_test("mean", "lev").note


def test_fit_discloses_a_separated_level_and_its_crossed_cells_once():
    rng = np.random.default_rng(42)
    a = rng.choice(["A", "B", "E"], 4000, p=[0.48, 0.48, 0.04])
    b = rng.choice(["X", "Y"], len(a))
    y = rng.poisson(
        0.5 * np.exp(0.3 * (a == "B") + 0.2 * (b == "Y") + 0.8 * ((a == "B") & (b == "Y")))
    )
    y[a == "E"] = 0
    family = NegativeBinomialLS()
    model = SuperLSS(
        family,
        family.mean(
            cat("a", base="A"), cat("b", base="X"), interaction(CategoricalInteraction("a", "b"))
        ),
        family.theta(),
    )
    with pytest.warns(SeparationWarning) as caught:
        model.fit(pd.DataFrame({"a": a, "b": b}), y)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "'mean:a' level 'E'" in message
    assert "'mean:a:b' cell 'E' x 'X'" in message
    assert "'mean:a:b' cell 'E' x 'Y'" in message
    assert "where the fit stopped" in message


@pytest.mark.parametrize("method", ["predict", "predict_parameters"])
def test_prediction_warns_only_on_affected_rows(separated_fit, method):
    score = pd.DataFrame({"lev": ["A", "E"], "x": 0.5, "log_expo": 0.0})
    baseline = getattr(separated_fit._require_fitted(), method)(score)
    predict = getattr(separated_fit, method)
    with pytest.warns(SeparationWarning, match="mean:lev.*E") as caught:
        result = predict(score)
    assert len(caught) == 1
    np.testing.assert_array_equal(result, baseline)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        predict(score.iloc[:1])


def test_grouped_prediction_and_inference_use_each_predictors_boundary(two_levels):
    class MeanOnlyBoundary(NegativeBinomialLS):
        def response_boundaries(self, links):
            return super().response_boundaries(links)[0], ()

    X, y = two_levels
    X = X.replace({"lev": {"A": 1, "E": 2}})
    X.iloc[124:127, 0] = 1
    y[124:127] = 1
    X.iloc[-1, 0] = 3
    family = MeanOnlyBoundary()
    grouping = collapse_levels([1, 2, 3], groups={"group": [2, 3]})
    term = cat("lev", grouping=grouping)
    with pytest.warns(SeparationWarning):
        model = SuperLSS(family, family.mean(term), family.theta(term)).fit(X, y)
    for method in (model.predict, model.predict_parameters):
        with pytest.warns(SeparationWarning, match="mean:lev.*group") as caught:
            method(X.iloc[-1:])
        assert len(caught) == 1 and "theta:lev" not in str(caught[0].message)
    assert model.term_inference("mean", "lev", simultaneous=False).separated == (False, True)
    assert model.term_inference("theta", "lev", simultaneous=False).separated is None


def test_prediction_aggregates_hits_from_multiple_predictors(two_levels):
    X, y = two_levels
    family = NegativeBinomialLS()
    with pytest.warns(SeparationWarning):
        model = SuperLSS(
            family, family.mean(cat("lev", base="A")), family.theta(cat("lev", base="A"))
        ).fit(X, y)
    with pytest.warns(SeparationWarning) as caught:
        model.predict_parameters(X.iloc[-1:])
    assert len(caught) == 1
    assert "mean:lev" in str(caught[0].message) and "theta:lev" in str(caught[0].message)


def test_default_mean_prediction_does_not_warn_for_a_theta_only_level():
    rng = np.random.default_rng(7)
    lev = rng.choice(["A", "B", "E"], 1500, p=[0.45, 0.45, 0.10])
    x = rng.uniform(0, 1, len(lev))
    y = rng.poisson(0.5 * np.exp(0.4 * x))
    y[lev == "E"] = 0
    X = pd.DataFrame({"lev": lev, "x": x})
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(s("x", n_knots=5)), family.theta(cat("lev", base="A")))
    with pytest.warns(SeparationWarning, match="theta:lev.*E"):
        model.fit(X, y, lambdas={"mean:x#wiggle": 1.0})
    affected = X.loc[X.lev == "E"].iloc[:2]
    with pytest.warns(SeparationWarning, match="theta:lev.*E") as caught:
        parameters = model.predict_parameters(affected)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        np.testing.assert_array_equal(
            model.predict(affected), model.predict(affected.assign(lev="A"))
        )
        np.testing.assert_array_equal(model.predict(affected), parameters["mean"])


def test_level_inference_and_workbook_keep_the_fitted_factor(
    separated_fit, separated_book, tmp_path
):
    from openpyxl import load_workbook

    from superglm.distributional.posterior import _posterior_draws_from_covariance

    model = separated_fit
    X, _ = separated_book
    effect = model.term_inference("mean", "lev", n_sim=512)
    j = effect.levels.index("E")
    assert effect.separated == (False, False, False, False, True)
    for name in ("se", "lower", "upper", "lower_simultaneous", "upper_simultaneous"):
        assert np.isnan(getattr(effect, name)[j])
        assert effect.to_json()[name][j] is None
    assert effect.to_json()["separated"][j] is True
    healthy = [1, 2, 3]
    expected_se = np.sqrt(np.diag(model.covariance_)[healthy])
    np.testing.assert_array_equal(effect.se[healthy], expected_se)
    draws = _posterior_draws_from_covariance(
        model._require_fitted(), model.covariance_, 512, covariance="fixed", seed=42
    )
    beta = model.coef_by_predictor_["mean"]
    z = (draws.coefficients[:, healthy] - beta[healthy]) / expected_se
    critical = max(np.quantile(np.max(abs(z), axis=1), 0.95), 1.959963984540054)
    assert effect.critical_value == pytest.approx(critical)
    assert effect.lower_simultaneous[1:4] == pytest.approx(
        effect.effect[1:4] - critical * expected_se
    )
    block = next(
        b for b in model.rating_table_payload(X).parameters[0].main_effects if b.name == "lev"
    )
    factor = np.exp(beta[4])
    assert effect.effect[j] == beta[4] and effect.multiplier[j] == factor
    assert tuple(block.table.loc[block.table.lev == "E"].iloc[0]) == ("E", factor, 35)
    assert "E" in block.note and "where the fit stopped" in block.note
    path = model.export_rating_tables(tmp_path / "rating.xlsx", X)
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook["Mean"].values)
        values = next(row[row.index("E") : row.index("E") + 3] for row in rows if "E" in row)
        assert values == ("E", pytest.approx(factor, rel=1e-12, abs=0), 35)
        assert any(block.note in row for row in rows)
    finally:
        workbook.close()


def test_weighted_levels_use_effective_rows(two_levels):
    X, y = two_levels
    term = cat("lev", base="A", levels=["A", "E", "unused"])
    y[-1] = 1
    weight = np.ones(len(y))
    weight[-1] = 0
    with pytest.warns(SeparationWarning, match="mean:lev.*E") as caught:
        model = _model(term).fit(X, y, sample_weight=weight)
    assert "unused" not in str(next(w.message for w in caught if w.category is SeparationWarning))
    effect = model.term_inference("mean", "lev", simultaneous=False)
    assert dict(zip(effect.levels, effect.separated, strict=True)) == {
        "A": False,
        "E": True,
        "unused": False,
    }
    weight[-1] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(X, y, sample_weight=weight)
    assert model.term_inference("mean", "lev", simultaneous=False).separated is None


def test_a_capped_level_still_separates_in_the_rule_free_covariance(separated_book):
    model = _model(cat("lev", base="A", constraint=Constraint.fit.cap(0.5, 2)), smooth=True)
    with pytest.warns(SeparationWarning, match="mean:lev.*E") as caught:
        model.fit_reml(*separated_book)
    assert len(caught) == 1
    assert "Constraint.fit.cap" not in str(caught[0].message)
    term = model.term_test("mean", "lev")
    row = model.summary().query("parameter == 'mean' and term == 'lev'").iloc[0]
    healthy = [1, 2, 3]  # B, C and D against A
    beta = model.coef_by_predictor_["mean"][healthy]
    covariance = model.covariance_[np.ix_(healthy, healthy)]
    expected = beta @ np.linalg.solve(covariance, beta)
    assert term.rank == row["rank"] == 3
    for report in (term, row):
        assert report.statistic == pytest.approx(expected)
        assert "SEP" in report.note and "'E'" in report.note
    effect = model.term_inference("mean", "lev", simultaneous=False)
    j = effect.levels.index("E")
    assert effect.multiplier[j] == pytest.approx(0.5)
    assert effect.separated[j] and np.isnan(effect.se[j])


@pytest.mark.pins_behaviour
def test_an_intercept_free_reference_cannot_separate(two_levels):
    """Pin the absence of a separation warning or flag for a fixed reference."""
    model = _model(cat("lev", base="E"), intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(*two_levels)
    assert model.term_inference("mean", "lev", simultaneous=False).separated is None


@pytest.fixture
def bounded_separation_book():
    rng = np.random.default_rng(7)
    lev = rng.choice(["A", "B", "E"], 1500, p=[0.45, 0.45, 0.10])
    x = rng.uniform(0, 1, len(lev))
    y = rng.poisson(0.5 * np.exp(0.5 * (lev == "B") + 0.4 * x)).astype(float)
    y[lev == "E"] = 0.0
    return pd.DataFrame({"lev": lev}), y


def test_right_truncation_does_not_free_a_reference_without_an_intercept(
    bounded_separation_book,
):
    model = _model(cat("lev", base="E"), intercept=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(*bounded_separation_book, truncation=(None, 20.0))
    assert model.term_inference("mean", "lev", simultaneous=False).separated is None
    assert model.term_test("mean", "lev").rank == 2


@pytest.mark.parametrize("censor", [False, True], ids=["truncated", "censored"])
def test_bounded_fits_exclude_an_exact_claim_free_level(bounded_separation_book, censor):
    X, y = bounded_separation_book
    keep = X.lev != "E"
    censored = y >= 3 if censor else None
    truncation = (None, 20.0)
    model = _model(cat("lev", base="A"))
    with pytest.warns(SeparationWarning, match="mean:lev.*E"):
        model.fit(X, y, truncation=truncation, censored=censored)
    reference = _model(cat("lev", base="A")).fit(
        X.loc[keep],
        y[keep],
        truncation=truncation,
        censored=censored[keep] if censored is not None else None,
    )
    term = model.term_test("mean", "lev")
    reference_term = reference.term_test("mean", "lev")
    assert term.rank == reference_term.rank == 1
    assert term.statistic == pytest.approx(reference_term.statistic, rel=2e-3)
    assert "SEP" in term.note and "'E'" in term.note
    effect = model.term_inference("mean", "lev", simultaneous=False)
    j = effect.levels.index("E")
    assert effect.separated[j] and np.isnan(effect.se[j])


def test_a_right_censoring_limit_of_zero_is_not_an_observed_zero(bounded_separation_book):
    X, y = bounded_separation_book
    model = _model(cat("lev", base="A"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model.fit(X, y, censored=(X.lev == "E").to_numpy())
    assert model.term_inference("mean", "lev", simultaneous=False).separated is None


def test_a_left_censoring_limit_of_zero_is_an_observed_zero(bounded_separation_book):
    """``Y <= 0`` on counts is ``Y = 0``: a level of exact zeros and rows left-censored
    at zero is still separated, warned about and refused, as on master."""
    X, y = bounded_separation_book
    rows = np.flatnonzero((X.lev == "E").to_numpy())
    censored = np.array(["none"] * len(y), dtype=object)
    censored[rows[::2]] = "left"
    with pytest.warns(SeparationWarning, match="mean:lev.*E"):
        _model(cat("lev", base="A")).fit(X, y, censored=list(censored))
    with pytest.raises(SeparationError):
        _model(cat("lev", base="A"), separation="error").fit(X, y, censored=list(censored))


@pytest.mark.pins_behaviour
def test_ignore_leaves_inference_available_and_error_refuses_the_build(two_levels, monkeypatch):
    """Pin the silent unrestricted fit under ignore and build-time refusal under error."""
    from superglm.distributional import model as dense_model

    X, y = two_levels
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        model = _model(separation="ignore").fit(X, y)
        model.predict(X)
    assert np.isfinite(model.term_inference("mean", "lev", simultaneous=False).se).all()

    def should_not_solve(*args, **kwargs):
        pytest.fail("strict mode reached the solver")

    monkeypatch.setattr(dense_model, "fit_dense_fixed_lambda", should_not_solve)
    with pytest.raises(SeparationError, match="mean:lev"):
        _model(separation="error").fit(X, y)


@pytest.mark.parametrize("codec", ["pickle", "joblib"])
def test_saved_model_retains_separation_without_training_rows(
    separated_fit, separated_book, tmp_path, codec
):
    X, _ = separated_book
    if codec == "pickle":
        restored = pickle.loads(pickle.dumps(separated_fit))
    else:
        path = tmp_path / "model.joblib"
        joblib.dump(separated_fit, path)
        restored = joblib.load(path)
    affected = X.loc[X.lev == "E"].iloc[:1]
    with pytest.warns(SeparationWarning, match="mean:lev.*E"):
        predicted = restored.predict(affected)
    np.testing.assert_array_equal(predicted, separated_fit._require_fitted().predict(affected))
    effect = restored.term_inference("mean", "lev", X_train=X, simultaneous=False)
    assert effect.separated[effect.levels.index("E")]


@pytest.mark.pins_behaviour
def test_legacy_model_without_separation_record_still_scores(two_levels):
    """Pin pre-record artifacts' predictions and inference on a supplied frame."""
    X, y = two_levels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeparationWarning)
        model = _model(cat("lev", base="A")).fit(X, y, retain_rows=False)
    spec = model._require_fitted().compiled_predictors[0].compiled.specs["lev"]
    spec.__dict__.pop("_separated_levels_at_fit", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeparationWarning)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    effect = restored.term_inference("mean", "lev", X_train=X, simultaneous=False)
    assert np.isfinite(effect.se).all()
