"""``cross_validate`` on a SuperLSS: folds, scores and what it refuses."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from superglm import (
    Categorical,
    GammaLS,
    GaussianLS,
    NegativeBinomialLS,
    Spline,
    SuperGLM,
    SuperLSS,
    cat,
    cross_validate,
    families,
    s,
)
from superglm.distributional.cross_validation import _POOLED_PARTS, LSSFolds
from superglm.model_selection import _data_fingerprint
from superglm.validation import _normalized_gini

N = 900
LAMBDAS = {"mean:age#wiggle": 1.0, "scale:age#wiggle": 1.0}
FOLDS = KFold(3, shuffle=True, random_state=2)


def _rows(n: int = N) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(11)
    X = pd.DataFrame(
        {
            "age": rng.uniform(18.0, 80.0, n),
            "region": rng.choice(list("abc"), n),
            "unread": rng.normal(size=n),
        }
    )
    mean = np.exp(5.0 + 0.015 * (X["age"].to_numpy() - 50.0) + 0.3 * (X["region"] == "b"))
    cv = np.exp(-0.4 + 0.005 * (X["age"].to_numpy() - 50.0))
    return X, rng.gamma(1.0 / cv**2, mean * cv**2)


def _gamma(**kw) -> SuperLSS:
    family = GammaLS()
    return SuperLSS(
        family,
        family.mean(s("age", kind="cr", k=6), cat("region")),
        family.scale(s("age", kind="cr", k=5)),
        **kw,
    )


def _fixed_cv(X, y, **kw):
    """Fixed smoothing, so a fold repeats exactly when fitted by hand."""
    return cross_validate(
        kw.pop("model", _gamma()),
        X,
        y,
        cv=FOLDS,
        fit_kwargs={"lambdas": LAMBDAS},
        error_score="raise",
        **kw,
    )


def test_every_score_is_the_one_a_by_hand_fold_gets():
    X, y = _rows()
    result = _fixed_cv(X, y)
    assert result.builtin_scores == ("nll", "crps", "gini", "actual_expected")
    for fold, (train, test) in enumerate(result.fold_indices):
        model = _gamma().fit(X.iloc[train], y[train], lambdas=LAMBDAS)
        scores = model.scores(X.iloc[test], y[test], which=("log", "crps"))
        mean = model.predict(X.iloc[test])
        row = result.fold_scores.loc[fold]
        tolerance = 1e-9
        assert row["nll"] == pytest.approx(scores["log"].mean(), rel=tolerance)
        assert row["crps"] == pytest.approx(scores["crps"].mean(), rel=tolerance)
        assert row["actual_expected"] == pytest.approx(y[test].sum() / mean.sum(), rel=tolerance)
        assert row["gini"] == pytest.approx(_normalized_gini(y[test], mean, None), rel=tolerance)
        assert row["converged"]
        assert row["effective_df"] == pytest.approx(model.result_.total_effective_df)


def test_pooled_scores_are_ratios_of_sums_across_unequal_folds():
    X, y = _rows()
    uneven = [(np.arange(300, N), np.arange(300)), (np.arange(600), np.arange(600, N))]
    result = cross_validate(
        _gamma(), X, y, cv=_Stored(uneven), fit_kwargs={"lambdas": LAMBDAS}, error_score="raise"
    )
    sizes = result.fold_scores["n_test"].to_numpy(dtype=float)
    expected = np.sum(result.fold_scores["nll"].to_numpy() * sizes) / sizes.sum()
    assert result.pooled_scores["nll"] == pytest.approx(expected, rel=1e-12)
    actual = [y[test].sum() for _train, test in uneven]
    predicted = [
        actual_total / ratio
        for actual_total, ratio in zip(actual, result.fold_scores["actual_expected"], strict=True)
    ]
    assert result.pooled_scores["actual_expected"] == pytest.approx(
        sum(actual) / sum(predicted), rel=1e-12
    )
    assert "gini" not in result.pooled_scores


class _Stored:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X=None, y=None, groups=None):
        yield from self.folds


def test_integer_frequency_weights_score_like_repeated_rows():
    X, y = _rows(300)
    weights = np.random.default_rng(5).integers(1, 4, 300).astype(np.float64)
    fold = [(np.arange(100, 300), np.arange(100))]
    weighted = cross_validate(
        _gamma(weight_semantics="frequency"),
        X,
        y,
        cv=_Stored(fold),
        sample_weight=weights,
        fit_kwargs={"lambdas": LAMBDAS},
        error_score="raise",
    )
    repeated_index = np.repeat(np.arange(300), weights.astype(int))
    X_repeated = X.iloc[repeated_index].reset_index(drop=True)
    test = np.flatnonzero(repeated_index < 100)
    train = np.flatnonzero(repeated_index >= 100)
    repeated = cross_validate(
        _gamma(),
        X_repeated,
        y[repeated_index],
        cv=_Stored([(train, test)]),
        fit_kwargs={"lambdas": LAMBDAS},
        error_score="raise",
    )
    for name in ("nll", "crps", "actual_expected"):
        assert weighted.mean_scores[name] == pytest.approx(repeated.mean_scores[name], rel=1e-6)


def test_prior_weighted_rows_divide_by_scored_rows_and_skip_zero_weights():
    X, y = _rows(600)
    weights = np.random.default_rng(6).uniform(0.5, 2.0, 600)
    weights[:40] = 0.0
    fold = [(np.arange(200, 600), np.arange(200))]
    result = cross_validate(
        _gamma(),
        X,
        y,
        cv=_Stored(fold),
        sample_weight=weights,
        fit_kwargs={"lambdas": LAMBDAS},
        error_score="raise",
    )
    model = _gamma().fit(X.iloc[200:], y[200:], weights[200:], lambdas=LAMBDAS)
    log = model.scores(X.iloc[:200], y[:200], which=("log",), sample_weight=weights[:200])["log"]
    assert np.isnan(log.to_numpy()[:40]).all()
    assert result.mean_scores["nll"] == pytest.approx(np.nansum(log) / 160.0, rel=1e-9)
    kept = weights[40:200]
    mean = model.predict(X.iloc[40:200])
    assert result.mean_scores["actual_expected"] == pytest.approx(
        np.sum(kept * y[40:200]) / np.sum(kept * mean), rel=1e-9
    )


def test_nll_compares_with_a_superglm_on_the_same_prior_weighted_folds():
    X, y = _rows()
    weights = np.random.default_rng(8).uniform(0.5, 2.0, N)
    family = GammaLS()
    lss = SuperLSS(family, family.mean(s("age", kind="cr", k=6), cat("region")), family.scale())
    glm = SuperGLM(
        family=families.gamma(),
        selection_penalty=0.0,
        features={"age": Spline(kind="cr", k=6), "region": Categorical()},
    )
    kwargs = {"cv": FOLDS, "sample_weight": weights, "scoring": ("nll",), "error_score": "raise"}
    glm_nll = cross_validate(glm, X, y, fit_mode="fit_reml", **kwargs).mean_scores["nll"]
    lss_nll = cross_validate(lss, X, y, fit_mode="fit_reml", **kwargs).mean_scores["nll"]
    # The same law and rows, so a different denominator would read as a large gap.
    assert lss_nll == pytest.approx(glm_nll, abs=0.02)


def test_a_level_only_in_a_held_out_fold_is_scored_and_the_model_is_left_alone():
    X, y = _rows()
    X.loc[7, "region"] = "rare"
    model = _gamma()
    result = _fixed_cv(X, y, model=model, return_oof=True, return_estimators=True)
    assert result.fold_scores["nll"].notna().all()
    assert np.isfinite(result.oof_predictions).all()
    assert model._model is None
    assert model._predictors[0].features["region"]._declared_levels is None
    for fold, (train, _test) in zip(result.estimators, result.fold_indices, strict=True):
        spec = fold._require_fitted().compiled_predictors[0].compiled.specs["region"]
        assert list(spec._declared_levels) == ["a", "b", "c", "rare"]
        assert ("rare" in spec._pinned_levels) == (7 not in train)


def _fold_levels(result):
    """Each fold's level universe and base for ``region``, as its fit was bound to them."""
    specs = (
        fold._require_fitted().compiled_predictors[0].compiled.specs["region"]
        for fold in result.estimators
    )
    return [
        (None if spec._declared_levels is None else list(spec._declared_levels), spec._pinned_base)
        for spec in specs
    ]


@pytest.mark.parametrize("label", [None, "zzz"], ids=["a missing label", "a label seen only there"])
def test_levels_are_bound_on_the_rows_the_fit_keeps(label):
    # A fit drops a row of weight zero before it reads anything, so a missing label on one must
    # not switch the shared binding off (each fold would take its own levels and base), and a
    # label seen only on one must not join the universe.
    X, y = _rows()
    weights = np.random.default_rng(6).uniform(0.5, 2.0, N)
    padding = pd.DataFrame({"age": [40.0, 50.0], "region": [label, label], "unread": [0.0, 0.0]})
    padded = _fixed_cv(
        pd.concat([X, padding], ignore_index=True),
        np.r_[y, 1.0, 1.0],
        sample_weight=np.r_[weights, 0.0, 0.0],
        return_estimators=True,
    )
    clean = _fixed_cv(X, y, sample_weight=weights, return_estimators=True)

    base = X["region"].value_counts().idxmax()
    assert _fold_levels(clean) == [(["a", "b", "c"], base)] * 3
    assert _fold_levels(padded) == _fold_levels(clean)


@pytest.mark.parametrize("label", [None, "zzz"], ids=["a missing label", "a label seen only there"])
def test_a_label_on_a_dropped_row_cannot_fail_its_fold(label):
    # Mutation check: calling ``folds.predict`` directly fails this, under ``error_score="raise"``
    # here and with the fold's scores NaN under the default.
    X, y = _rows()
    weights = np.random.default_rng(6).uniform(0.5, 2.0, N)
    padding = pd.DataFrame({"age": [40.0, 50.0], "region": [label, label], "unread": [0.0, 0.0]})
    frame = pd.concat([X, padding], ignore_index=True)
    result = _fixed_cv(
        frame,
        np.r_[y, 1.0, 1.0],
        sample_weight=np.r_[weights, 0.0, 0.0],
        return_oof=True,
        return_estimators=True,
    )
    assert np.isnan(result.oof_predictions[N:]).all()
    assert result.fold_scores["nll"].notna().all()
    for fold, (_train, test) in zip(result.estimators, result.fold_indices, strict=True):
        kept = test[test < N]
        np.testing.assert_allclose(
            result.oof_predictions[kept], fold.predict(frame.iloc[kept]), rtol=1e-12
        )


def test_a_dropped_row_that_cannot_be_priced_leaves_a_priceable_one_in_its_fold_alone():
    # Mutation check: leaving every dropped row of a failing fold NaN fails this.
    X, y = _rows()
    weights = np.random.default_rng(6).uniform(0.5, 2.0, N)
    padding = pd.DataFrame(
        {"age": [40.0] * 8, "region": ["a", "zzz"] * 4, "unread": [0.0] * 8},
    )
    frame = pd.concat([X, padding], ignore_index=True)
    result = _fixed_cv(
        frame,
        np.r_[y, np.ones(8)],
        sample_weight=np.r_[weights, np.zeros(8)],
        return_oof=True,
        return_estimators=True,
    )
    known, unseen = np.arange(N, N + 8, 2), np.arange(N + 1, N + 8, 2)
    assert np.isnan(result.oof_predictions[unseen]).all()
    shared = 0
    for fold, (_train, test) in zip(result.estimators, result.fold_indices, strict=True):
        priced = np.intersect1d(test, known)
        shared += priced.size > 0 and np.intersect1d(test, unseen).size > 0
        np.testing.assert_allclose(
            result.oof_predictions[priced], fold.predict(frame.iloc[priced]), rtol=1e-12
        )
    assert shared  # some fold holds both, or the fallback was not reached


def test_offsets_by_parameter_are_sliced_for_fits_and_scores():
    rng = np.random.default_rng(3)
    X, _ = _rows()
    exposure = rng.uniform(0.2, 1.0, N)
    mean = exposure * np.exp(1.0 + 0.01 * (X["age"].to_numpy() - 50.0))
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mean)).astype(np.float64)
    family = NegativeBinomialLS()
    model = SuperLSS(family, family.mean(s("age", kind="cr", k=5)), family.theta())
    offsets = {"mean": np.log(exposure)}
    lambdas = {"mean:age#wiggle": 1.0}
    result = cross_validate(
        model,
        X,
        y,
        cv=FOLDS,
        offset=offsets,
        fit_kwargs={"lambdas": lambdas},
        scoring=("nll", "actual_expected"),
        return_oof=True,
        error_score="raise",
    )
    train, test = result.fold_indices[0]
    by_hand = SuperLSS(family, family.mean(s("age", kind="cr", k=5)), family.theta()).fit(
        X.iloc[train], y[train], offsets={"mean": offsets["mean"][train]}, lambdas=lambdas
    )
    held = {"mean": offsets["mean"][test]}
    log = by_hand.scores(X.iloc[test], y[test], which=("log",), offsets=held)["log"]
    assert result.fold_scores.loc[0, "nll"] == pytest.approx(log.mean(), rel=1e-9)
    assert result.oof_predictions[test] == pytest.approx(
        by_hand.predict(X.iloc[test], offsets=held)
    )
    # A mean below by the offset would read far from 1 on any fold.
    assert result.mean_scores["actual_expected"] == pytest.approx(1.0, abs=0.1)


def test_the_fingerprint_reads_the_model_columns_and_the_offsets():
    X, y = _rows()
    result = _fixed_cv(X, y)
    assert result.fingerprint_columns == ("age", "region")
    columns = result.fingerprint_columns
    changed = X.assign(unread=0.0)
    assert _data_fingerprint(changed, y, None, None, columns) == result.data_fingerprint
    moved = X.assign(age=X["age"] + 1.0)
    assert _data_fingerprint(moved, y, None, None, columns) != result.data_fingerprint

    zeros = {"mean": np.zeros(N)}
    assert _data_fingerprint(X, y, None, zeros, columns) == result.data_fingerprint
    first = _data_fingerprint(X, y, None, {"mean": np.ones(N)}, columns)
    assert first != result.data_fingerprint
    assert first != _data_fingerprint(X, y, None, {"scale": np.ones(N)}, columns)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset": np.zeros(N)}, "dict keyed by parameter name"),
        ({"offset": {"nope": np.zeros(N)}}, "Unknown offset parameter 'nope'"),
        ({"offset": {"mean": np.zeros(3)}}, "offset 'mean' length 3"),
        ({"scoring": ("deviance",)}, "Built-in scorers"),
        ({"fit_kwargs": {"offsets": {}}}, "fit_kwargs cannot set"),
        ({"fit_kwargs": {"truncation": (None, 1.0)}}, "fits and scores complete rows"),
    ],
)
def test_what_cannot_be_folded_is_refused_with_its_reason(kwargs, message):
    X, y = _rows()
    with pytest.raises(ValueError, match=message):
        cross_validate(_gamma(), X, y, cv=FOLDS, **kwargs)


def test_a_fold_that_fails_scores_nan_and_the_others_are_kept():
    X, y = _rows()
    calls = []

    def scorer(model, X, y, *, sample_weight=None, offset=None):
        calls.append(len(calls))
        if len(calls) == 2:
            raise ValueError("this fold cannot be scored")
        return float(np.mean(model.predict(X, offsets=offset)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cross_validate(
            _gamma(), X, y, cv=FOLDS, scoring=scorer, fit_kwargs={"lambdas": LAMBDAS}
        )
    assert result.fold_scores["scorer"].isna().tolist() == [False, True, False]
    assert result.builtin_scores == ()


def test_a_custom_scorer_of_a_builtin_name_is_not_recorded_as_built_in():
    X, y = _rows()

    def nll(model, X, y, *, sample_weight=None, offset=None):
        return 1.0

    result = _fixed_cv(X, y, scoring=(nll, "crps"))
    assert result.mean_scores["nll"] == 1.0
    assert result.builtin_scores == ("crps",)


def test_reml_folds_start_cold_unless_asked_and_disclose_their_iterations():
    X, y = _rows()
    kwargs = {"cv": FOLDS, "fit_mode": "fit_reml", "scoring": ("nll",), "error_score": "raise"}
    cold = cross_validate(_gamma(), X, y, **kwargs)
    assert not cold.fold_scores["warm_started"].any()
    assert (cold.fold_scores["n_reml_iter"] > 0).all()
    assert cold.fit_mode == "fit_reml"
    assert cold.warm_start is False
    warm = cross_validate(_gamma(), X, y, warm_start=True, **kwargs)
    assert warm.fold_scores["warm_started"].tolist() == [False, True, True]
    assert warm.warm_start is True
    assert np.isfinite(warm.mean_scores["nll"])


def test_a_warm_fold_starts_from_the_first_folds_smoothing(monkeypatch):
    X, y = _rows()
    seen = []
    fit_reml = SuperLSS.fit_reml

    def recording(self, *args, **kwargs):
        seen.append(kwargs.get("lambdas"))
        fitted = fit_reml(self, *args, **kwargs)
        first.append(fitted.smoothing_parameters_)
        return fitted

    first = []
    monkeypatch.setattr(SuperLSS, "fit_reml", recording)
    kwargs = {"cv": FOLDS, "fit_mode": "fit_reml", "scoring": ("nll",), "error_score": "raise"}
    cross_validate(_gamma(), X, y, warm_start=True, **kwargs)
    assert seen[0] is None
    assert seen[1] == dict(first[0]) == seen[2]

    seen.clear()
    cross_validate(_gamma(), X, y, **kwargs)
    assert seen == [None, None, None]

    seen.clear()
    given = dict(first[0])
    cross_validate(_gamma(), X, y, warm_start=True, fit_kwargs={"lambdas": given}, **kwargs)
    assert seen == [given, given, given]


class _Fit:
    """What the warm start reads of a fitted fold."""

    def __init__(self, faces=(), capped=None):
        self.exact_face_components_ = faces
        self.smoothing_unresolved_upper_bound_ = capped
        self.smoothing_parameters_ = {
            "mean:age#wiggle": 2.0,
            "mean:age#null": 3.0,
            "mean:power#wiggle": 4.0,
            "scale:age#wiggle": 5.0,
        }


def test_a_term_stopped_at_its_face_or_cap_starts_cold_with_its_other_penalties():
    folds = LSSFolds(_gamma())
    assert len(folds.live_lambdas(_Fit())) == 4
    assert folds.live_lambdas(_Fit(faces=("mean:age#null",))) == {
        "mean:power#wiggle": 4.0,
        "scale:age#wiggle": 5.0,
    }
    assert folds.live_lambdas(_Fit(capped=("scale:age#wiggle",))) == {
        "mean:age#wiggle": 2.0,
        "mean:age#null": 3.0,
        "mean:power#wiggle": 4.0,
    }


def test_a_family_with_no_distribution_function_has_no_crps(monkeypatch):
    X, y = _rows()

    def no_law(*args, **kwargs):
        raise NotImplementedError("no distribution function")

    monkeypatch.setattr("superglm.distributional.cross_validation._retained_crps", no_law)
    result = _fixed_cv(X, y)
    assert np.isnan(result.mean_scores["crps"])
    assert np.isfinite(result.mean_scores["nll"])


def test_a_fold_failing_a_later_scorer_stays_out_of_the_pooled_totals():
    X, y = _rows()
    scored = []

    def fails_on_the_second_fold(model, X, y, *, sample_weight=None, offset=None):
        scored.append(len(y))
        if len(scored) == 2:
            raise ValueError("this fold has no such score")
        return 1.0

    result = cross_validate(
        _gamma(),
        X,
        y,
        cv=FOLDS,
        fit_kwargs={"lambdas": LAMBDAS},
        scoring=("nll", fails_on_the_second_fold),
    )
    scores = result.fold_scores
    assert np.isnan(scores.loc[1, "nll"])
    kept = scores.loc[[0, 2]]
    expected = np.sum(kept["nll"] * kept["n_test"]) / kept["n_test"].sum()
    assert result.pooled_scores["nll"] == pytest.approx(expected, rel=1e-12)


def test_an_undefined_actual_expected_keeps_the_folds_other_scores(monkeypatch):
    X, y = _rows()
    monkeypatch.setitem(_POOLED_PARTS, "actual_expected", lambda *args, **kwargs: (1.0, 0.0))
    result = _fixed_cv(X, y, scoring=("nll", "actual_expected"))
    assert result.fold_scores["actual_expected"].isna().all()
    assert np.isfinite(result.fold_scores["nll"]).all()
    assert result.fold_scores["converged"].all()
    assert "actual_expected" not in result.pooled_scores


def test_fold_curves_are_left_to_the_editor():
    X, y = _rows()
    result = _fixed_cv(X, y, return_estimators=True)
    assert result.curve_similarity is None
    with pytest.raises(ValueError, match="Cross-validation tab"):
        result.plot_terms_by_fold(X)


def test_the_default_call_estimates_each_fold_s_smoothing():
    X, y = _rows()
    result = cross_validate(_gamma(), X, y, cv=KFold(3, shuffle=True, random_state=2))
    assert result.fit_mode == "fit_reml"
    assert np.isfinite(result.fold_scores["nll"]).all()


def test_given_smoothing_is_held_fixed_by_default():
    X, y = _rows()
    fitted = _gamma().fit_reml(X, y)
    lambdas = dict(fitted._require_fitted().lambdas)
    result = cross_validate(
        fitted, X, y, cv=KFold(3, shuffle=True, random_state=2), fit_kwargs={"lambdas": lambdas}
    )
    assert result.fit_mode == "fit"
    assert np.isfinite(result.fold_scores["nll"]).all()


def test_fixed_smoothing_with_none_given_is_refused_before_any_fold():
    X, y = _rows()
    fitted = _gamma().fit_reml(X, y)
    with pytest.raises(ValueError, match="fit_mode='fit_reml'"):
        cross_validate(fitted, X, y, cv=KFold(3), fit_mode="fit")


def test_a_fit_on_truncated_rows_is_refused_before_any_fold():
    X, y = _rows()
    fitted = _gamma().fit_reml(X, y, truncation=(None, float(y.max()) * 2.0))
    with pytest.raises(ValueError, match="truncated or censored rows"):
        cross_validate(fitted, X, y, cv=KFold(3), fit_mode="fit_reml")


def test_a_crps_that_cannot_be_resolved_costs_only_the_crps(monkeypatch, caplog):
    import superglm.distributional.cross_validation as module

    def unresolved(*args, **kwargs):
        raise ValueError("numeric CRPS tail is unresolved")

    monkeypatch.setattr(module, "_retained_crps", unresolved)
    X, y = _rows()
    logger = "superglm.distributional.cross_validation"
    with caplog.at_level(logging.WARNING, logger=logger):
        result = cross_validate(_gamma(), X, y, cv=KFold(3, shuffle=True, random_state=2))
    assert np.isnan(result.fold_scores["crps"]).all()
    for name in ("nll", "gini", "actual_expected"):
        assert np.isfinite(result.fold_scores[name]).all()
    assert caplog.text.count("numeric CRPS tail is unresolved") == 3


def test_a_prediction_that_fails_is_not_read_as_an_unresolved_crps(monkeypatch):
    X, y = _rows()
    model = _gamma().fit(X, y, lambdas=LAMBDAS)

    def refuses(*args, **kwargs):
        raise ValueError("this level cannot be read")

    monkeypatch.setattr(type(model._require_fitted()), "predict_parameters", refuses)
    with pytest.raises(ValueError, match="cannot be read"):
        _POOLED_PARTS["crps"](model, X, y)


def test_a_negative_total_has_no_gini_and_its_ratio_still_pools(caplog):
    X, y = _rows()
    family = GaussianLS()
    with caplog.at_level(logging.WARNING, logger="superglm.distributional.cross_validation"):
        result = cross_validate(
            SuperLSS(family, family.location(cat("region")), family.scale()),
            X,
            y - y.max() - 1.0,
            cv=FOLDS,
            scoring=("gini", "actual_expected"),
            error_score="raise",
        )
    assert result.fold_scores["gini"].isna().all()
    assert caplog.text.count("has no Gini") == 3
    assert result.fold_scores["actual_expected"].notna().all()
    assert result.pooled_scores["actual_expected"] == pytest.approx(1.0, abs=0.05)
