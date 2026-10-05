"""Tests for fold curve similarity helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from superglm import Categorical, Spline, SuperGLM

_U = np.finfo(np.float64).eps / 2


def test_pairwise_similarity_matrices_have_expected_diagonals():
    from superglm.plotting.curve_similarity import _pairwise_curve_similarity

    labels = ["fold_0", "fold_1", "fold_2"]
    curves = {
        "fold_0": np.array([1.0, 2.0, 3.0]),
        "fold_1": np.array([1.0, 2.0, 3.0]),
        "fold_2": np.array([2.0, 3.0, 4.0]),
    }
    weights = np.array([1.0, 2.0, 1.0])

    result = _pairwise_curve_similarity(curves, weights, labels=labels)

    np.testing.assert_allclose(np.diag(result["rmse"]), 0.0)
    np.testing.assert_allclose(np.diag(result["max_abs_diff"]), 0.0)
    np.testing.assert_allclose(np.diag(result["correlation"]), 1.0)


def test_weighting_changes_rmse_in_expected_direction():
    from superglm.plotting.curve_similarity import _pairwise_curve_similarity

    curves = {
        "fold_0": np.array([0.0, 0.0, 10.0]),
        "fold_1": np.array([0.0, 0.0, 0.0]),
    }
    low_tail = np.array([10.0, 10.0, 1.0])
    high_tail = np.array([1.0, 1.0, 10.0])

    low_tail_rmse = _pairwise_curve_similarity(curves, low_tail, labels=["fold_0", "fold_1"])[
        "rmse"
    ]
    high_tail_rmse = _pairwise_curve_similarity(curves, high_tail, labels=["fold_0", "fold_1"])[
        "rmse"
    ]

    assert high_tail_rmse.loc["fold_0", "fold_1"] > low_tail_rmse.loc["fold_0", "fold_1"]


def test_fold_mean_distances_are_reported():
    from superglm.plotting.curve_similarity import _summarize_against_fold_mean

    curves = {
        "fold_0": np.array([1.0, 2.0, 3.0]),
        "fold_1": np.array([1.0, 2.0, 3.0]),
        "fold_2": np.array([2.0, 3.0, 4.0]),
    }
    weights = np.array([1.0, 1.0, 1.0])

    summary = _summarize_against_fold_mean(curves, weights)

    assert list(summary.columns) == ["rmse_to_mean", "max_abs_diff_to_mean", "correlation_to_mean"]
    assert summary.index.tolist() == ["fold_0", "fold_1", "fold_2"]


def test_build_cv_curve_similarity_returns_both_scales_for_all_comparable_terms():
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(7)
    n = 160
    x = rng.uniform(0, 10, n)
    band = rng.choice(["A", "B", "C", "D"], n)
    w = rng.uniform(0.5, 1.2, n)
    eta = -1.3 + 0.2 * np.sin(x) + 0.15 * (band == "C")
    y = rng.poisson(np.exp(eta) * w).astype(float)
    X = pd.DataFrame({"x": x, "band": band})

    models = []
    for seed in [1, 2, 3]:
        idx = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=False)
        model = SuperGLM(
            features={
                "x": Spline(n_knots=6),
                "band": Categorical(base="first"),
            }
        )
        model.fit(X.iloc[idx], y[idx], sample_weight=w[idx])
        models.append(model)

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w, n_points=51)

    assert set(similarity) == {"x", "band"}
    assert "response" in similarity["x"]["pairwise"]
    assert "link" in similarity["x"]["pairwise"]
    assert len(similarity["x"]["domain"]["x"]) == 51


def test_build_cv_curve_similarity_accepts_polars_without_converting_fold_models():
    from superglm._frame import as_eager_frame
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(8)
    n = 120
    x = rng.uniform(0, 10, n)
    band = rng.choice(["A", "B", "C"], n)
    w = rng.uniform(0.5, 1.2, n)
    y = rng.poisson(np.exp(-1.0 + 0.15 * np.sin(x) + 0.1 * (band == "C")) * w).astype(float)
    X = pl.DataFrame({"x": x, "band": band})
    frame = as_eager_frame(X)

    models = []
    for seed in [1, 2, 3]:
        idx = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=False)
        model = SuperGLM(
            features={
                "x": Spline(n_knots=6),
                "band": Categorical(base="first"),
            }
        )
        model.fit(frame.take_rows(idx), y[idx], sample_weight=w[idx])
        models.append(model)

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w, n_points=41)

    assert set(similarity) == {"x", "band"}
    assert len(similarity["x"]["domain"]["x"]) == 41
    # Model order, not the order the rows first show each level.
    assert similarity["band"]["domain"]["levels"] == ["A", "B", "C"]
    assert all(isinstance(model._fit_X_ref, pl.DataFrame) for model in models)


def test_build_cv_curve_similarity_scores_integer_coded_levels_in_model_order():
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(9)
    n = 150
    code = np.tile([10, 1, 2], n // 3)
    w = rng.uniform(0.5, 1.2, n)
    y = rng.poisson(np.exp(-1.0 + 0.2 * (code == 2)) * w).astype(float)
    X = pd.DataFrame({"code": code})

    models = []
    for seed in [1, 2, 3]:
        idx = np.random.default_rng(seed).choice(n, size=int(0.8 * n), replace=False)
        model = SuperGLM(features={"code": Categorical(base="first")})
        model.fit(X.iloc[idx], y[idx], sample_weight=w[idx])
        models.append(model)

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w, n_points=41)

    assert similarity["code"]["domain"]["levels"] == ["1", "2", "10"]
    for label, model in zip(["fold_0", "fold_1", "fold_2"], models, strict=True):
        inference = model.term_inference("code", with_se=False)
        np.testing.assert_array_equal(
            similarity["code"]["curves"]["link"][label], inference.log_relativity
        )


def test_a_fold_that_never_saw_a_level_has_a_gap_there_not_a_crash(monkeypatch):
    """A level only one fold's test rows hold is unknown to that fold's model.

    That model has no value at the level: its curve has a gap there, the
    similarity summary reads it on the levels it has, and nothing raises.
    """
    from superglm.model_selection import CrossValidationResult
    from superglm.plotting import comparison_plotly
    from superglm.plotting.comparison import _build_term_comparison_data, _feature_beta
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(11)
    n = 300
    band = rng.choice(["A", "B", "C"], n)
    band[[7, 19, 42]] = "D"  # only in the first fold's test rows, 0-99
    x = rng.uniform(0, 10, n)
    w = rng.uniform(0.5, 1.2, n)
    y = rng.poisson(np.exp(-1.0 + 0.1 * np.sin(x) + 0.2 * (band == "C")) * w).astype(float)
    X = pd.DataFrame({"x": x, "band": band})
    folds = [(np.setdiff1d(np.arange(n), test), test) for test in np.array_split(np.arange(n), 3)]
    models = [
        SuperGLM(
            selection_penalty=0.0,
            features={"x": Spline(n_knots=5), "band": Categorical(base="first")},
        ).fit(X.iloc[train], y[train], sample_weight=w[train])
        for train, _test in folds
    ]
    assert [list(model._specs["band"]._levels) for model in models] == [
        ["A", "B", "C"],
        ["A", "B", "C", "D"],
        ["A", "B", "C", "D"],
    ]
    labeled = {f"fold_{i}": model for i, model in enumerate(models)}

    payload = _build_term_comparison_data(models=labeled, terms=["band"], X=X, sample_weight=w)

    [term] = payload["terms"]
    assert term["domain"]["levels"] == ["A", "B", "C", "D"]
    levels = np.asarray(["A", "B", "C", "D"], dtype=object)
    gap = np.array([False, False, False, True])
    for label, model in labeled.items():
        link = term["series"][label]["link"]
        known = ~gap if label == "fold_0" else np.ones(4, dtype=bool)
        expected = model._specs["band"].score(levels[known], _feature_beta(model, "band"))
        np.testing.assert_array_equal(link[known], expected)
        assert np.isnan(link[~known]).all()
        assert np.isnan(term["series"][label]["response"][~known]).all()

    similarity = build_cv_curve_similarity(models=models, X=X, sample_weight=w)

    link = similarity["band"]["curves"]["link"]
    weights = np.asarray(similarity["band"]["support"]["density"])
    stacked = np.vstack(list(link.values()))
    mean_curve = np.array(
        [np.mean(stacked[~np.isnan(stacked[:, j]), j]) for j in range(stacked.shape[1])]
    )
    vs_mean = similarity["band"]["vs_mean"]["link"]
    assert np.isfinite(vs_mean.to_numpy()).all()
    # The first fold is read against the fold mean on the three levels it has;
    # the others on all four. Each is one weighted mean of at most four
    # squares, accurate to a few units of rounding (Higham, sec. 3.1).
    for label, curve in link.items():
        has = ~np.isnan(curve)
        expected = np.sqrt(np.average((curve[has] - mean_curve[has]) ** 2, weights=weights[has]))
        assert abs(vs_mean.loc[label, "rmse_to_mean"] - expected) <= 16 * _U * expected
    assert np.isfinite(similarity["band"]["pairwise"]["link"]["rmse"].to_numpy()).all()

    # plot_terms_by_fold reads the same payload; the renderer is stubbed so
    # the check runs without plotly installed.
    monkeypatch.setattr(comparison_plotly, "plot_term_comparison_plotly", lambda data, **_: data)
    result = CrossValidationResult(
        fold_scores=pd.DataFrame(),
        mean_scores={},
        pooled_scores={},
        std_scores={},
        fold_indices=folds,
        estimators=models,
    )
    plotted = result.plot_terms_by_fold(X, sample_weight=w, terms="band")
    assert np.isnan(plotted["terms"][0]["series"]["fold_0"]["link"][3])


def test_a_fold_that_holds_a_level_pinned_has_a_gap_there_on_the_cross_validate_path(
    monkeypatch,
):
    """cross_validate gives every fold the frame's levels, so a fold whose
    training rows lack D holds D pinned to its base.

    Its score at D is the base's, not an estimate, so its curve has a gap
    there in the similarity diagnostics and in plot_terms_by_fold.
    """
    from sklearn.model_selection import KFold

    from superglm import cross_validate
    from superglm.plotting import comparison_plotly

    X = pd.DataFrame({"band": ["D", *["A", "B"] * 30]})
    y = np.array([3.0, *[1.0, 2.0] * 30])
    model = SuperGLM(
        family="gaussian", selection_penalty=0.0, features={"band": Categorical(base="A")}
    )
    with pytest.warns(UserWarning, match="pinned to base"):
        result = cross_validate(model, X, y, cv=KFold(3), return_estimators=True)
    assert [list(fold._specs["band"]._pinned_levels) for fold in result.estimators] == [
        ["D"],
        [],
        [],
    ]

    similarity = result.curve_similarity["band"]
    d = similarity["domain"]["levels"].index("D")
    for scale in ("link", "response"):
        curves = similarity["curves"][scale]
        assert np.isnan(curves["fold_0"][d])
        assert np.isfinite(np.delete(curves["fold_0"], d)).all()
        assert np.isfinite(curves["fold_1"]).all() and np.isfinite(curves["fold_2"]).all()
    monkeypatch.setattr(comparison_plotly, "plot_term_comparison_plotly", lambda data, **_: data)
    plotted = result.plot_terms_by_fold(X, terms="band")
    assert np.isnan(plotted["terms"][0]["series"]["fold_0"]["link"][d])


@pytest.mark.parametrize("unseen", ["base", "group"])
def test_a_level_a_fold_never_saw_is_a_gap_whatever_its_unseen_policy(unseen):
    """A fold reads a level outside its universe by its unseen policy, without refusing.

    cross_validate's folds share the CV rows' levels, and the data the curves
    are read on also holds D, which the CV rows do not. Under unseen="base" a
    fold would read D at its base, relativity 1, and under a group policy at
    that group's value, as if it had estimated D. D is a gap in every fold's
    curve instead: in the comparison payload, the similarity diagnostics and
    the editor's fold curves.
    """
    from sklearn.model_selection import KFold

    from superglm import collapse_levels, cross_validate
    from superglm.editor import EditorSession
    from superglm.editor.cv import fold_log_curves
    from superglm.plotting.comparison import _build_term_comparison_data
    from superglm.plotting.curve_similarity import build_cv_curve_similarity

    rng = np.random.default_rng(20261005)
    band = rng.choice(["A", "B", "C", "D"], 400, p=[0.3, 0.3, 0.3, 0.1])
    X_all = pd.DataFrame({"band": band})
    y_all = 1.0 + 0.2 * (band == "B") + 0.4 * (band == "C") + rng.normal(0.0, 0.1, 400)
    rows = band != "D"

    def declared(levels):
        if unseen == "base":
            return Categorical(base="A", unseen="base")
        grouping = collapse_levels(pd.Series(levels), groups={"BC": ["B", "C"]})
        return Categorical(base="A", grouping=grouping, unseen="BC")

    def model(levels):
        return SuperGLM(
            family="gaussian", selection_penalty=0.0, features={"band": declared(levels)}
        )

    result = cross_validate(
        model(["A", "B", "C"]), X_all[rows], y_all[rows], cv=KFold(3), return_estimators=True
    )
    labeled = {f"fold_{i}": fold for i, fold in enumerate(result.estimators)}

    [term] = _build_term_comparison_data(models=labeled, terms=["band"], X=X_all)["terms"]
    d = term["domain"]["levels"].index("D")
    similarity = build_cv_curve_similarity(models=result.estimators, X=X_all)["band"]
    assert similarity["domain"]["levels"].index("D") == d
    in_force = model(["A", "B", "C", "D"]).fit(X_all, y_all)
    session = EditorSession.from_model(in_force, terms=["band"])
    editor_d = session.terms["band"].levels.index("D")
    for label, fold in labeled.items():
        for curve, at in (
            (term["series"][label]["link"], d),
            (similarity["curves"]["link"][label], d),
            (fold_log_curves(fold, session.terms)["band"], editor_d),
        ):
            assert np.isnan(curve[at])
            assert np.isfinite(np.delete(curve, at)).all()
    assert np.isfinite(similarity["vs_mean"]["link"].to_numpy()).all()
