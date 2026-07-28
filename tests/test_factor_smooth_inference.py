"""Prediction and compact reporting for fitted factor smooths."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from superglm import (
    FactorSmooth,
    FactorSmoothResult,
    LambdaPolicy,
    Numeric,
    Spline,
    SuperGLM,
)
from superglm.solvers.structured import FactorSmoothLevelSupport


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(1103)
    n_levels = 6
    repeats = 34
    codes = np.repeat(np.arange(n_levels), repeats)
    rng.shuffle(codes)
    x = rng.uniform(-1.0, 1.0, size=len(codes))
    z = rng.normal(size=len(codes))
    amplitudes = np.array([0.7, -0.45, 0.3, -0.6, 0.2, 0.5])
    y = (
        0.8
        + 0.16 * z
        + amplitudes[codes] * (x + 0.35 * x**2)
        + rng.normal(scale=0.13, size=len(codes))
    )
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "segment": np.array([f"segment-{code}" for code in codes], dtype=object),
        }
    )
    return X, y


def _model(
    *,
    discrete: bool,
    retain_fit_state: bool = True,
    unseen: str = "population",
    direct_solve: str = "structured",
    lambda_value: float | None = None,
) -> SuperGLM:
    policies = (
        {
            "wiggle": LambdaPolicy.fixed(lambda_value),
            "null_0": LambdaPolicy.fixed(lambda_value),
            "null_1": LambdaPolicy.fixed(lambda_value),
        }
        if lambda_value is not None
        else {
            "wiggle": LambdaPolicy.fixed(1.3),
            "null_0": LambdaPolicy.fixed(0.7),
            "null_1": LambdaPolicy.fixed(0.9),
        }
    )
    return SuperGLM(
        family="gaussian",
        features={"z": Numeric()},
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                k=6,
                unseen=unseen,
                lambda_policy=policies,
            )
        ],
        selection_penalty=0.0,
        discrete=discrete,
        direct_solve=direct_solve,
        retain_fit_state=retain_fit_state,
    )


@pytest.mark.parametrize("direct_solve", ["gram", "auto"])
def test_released_fs_report_is_backend_neutral(direct_solve: str):
    X, y = _data()
    keep = X["segment"].isin(["segment-0", "segment-1", "segment-2", "segment-3"])
    X = X.loc[keep].reset_index(drop=True)
    y = y[keep.to_numpy()]
    model = _model(
        discrete=False,
        retain_fit_state=False,
        direct_solve=direct_solve,
    ).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )

    assert model.result.direct_backend == "gram"
    report = model.factor_smooth("x:segment:fs", grid=9)
    restored = pickle.loads(pickle.dumps(model)).factor_smooth(
        "x:segment:fs",
        grid=9,
    )
    pd.testing.assert_frame_equal(restored.table, report.table)
    pd.testing.assert_frame_equal(restored.curves, report.curves)


@pytest.mark.parametrize("basis", ["fs", "sz"])
def test_factor_smooth_evaluation_metrics_expand_structured_penalties(
    basis: str,
) -> None:
    X, y = _data()
    policies = (
        {
            "wiggle": LambdaPolicy.fixed(1.3),
            "null_0": LambdaPolicy.fixed(0.7),
            "null_1": LambdaPolicy.fixed(0.9),
        }
        if basis == "fs"
        else {"wiggle": LambdaPolicy.fixed(1.3)}
    )
    model = SuperGLM(
        family="gaussian",
        features={
            "z": Numeric(),
            **(
                {"x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.2))}
                if basis == "sz"
                else {}
            ),
        },
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                basis=basis,
                k=6,
                lambda_policy=policies,
            )
        ],
        selection_penalty=0.0,
        direct_solve="gram",
    ).fit_reml(
        X,
        y,
        max_reml_iter=1,
        runtime_validation="skip",
    )

    leverage = model.metrics(X.copy(), y.copy()).leverage

    assert leverage.shape == y.shape
    assert np.all(np.isfinite(leverage))


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_conditional_and_population_prediction(discrete: bool) -> None:
    X, y = _data()
    model = _model(discrete=discrete).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    spec = model._interaction_specs["x:segment:fs"]
    group = next(group for group in model._groups if group.name == "x:segment:fs")

    conditional_eta = model._predict_eta_exact(X, random_effects="conditional")
    population_eta = model._predict_eta_exact(X, random_effects="population")
    expected_deviation = spec.score(
        X["x"].to_numpy(),
        X["segment"].to_numpy(),
        model.result.beta[group.sl],
    )
    np.testing.assert_allclose(conditional_eta - population_eta, expected_deviation)

    unseen = X.iloc[:5].copy()
    unseen["segment"] = "new-segment"
    unseen_conditional = model._predict_eta_exact(unseen, random_effects="conditional")
    unseen_population = model._predict_eta_exact(unseen, random_effects="population")
    np.testing.assert_allclose(unseen_conditional, unseen_population)


@pytest.mark.parametrize("random_effects", ["conditional", "population"])
def test_factor_smooth_prediction_policies_validate_unseen_and_missing(
    random_effects: str,
) -> None:
    X, y = _data()
    model = _model(discrete=False, unseen="error").fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    unseen = X.iloc[:4].copy()
    unseen["segment"] = "never-seen"
    if random_effects == "conditional":
        with pytest.raises(ValueError, match="unseen FactorSmooth"):
            model.predict(unseen, random_effects=random_effects)
    else:
        np.testing.assert_allclose(
            model.predict(unseen, random_effects=random_effects),
            model.predict(X.iloc[:4], random_effects=random_effects),
        )

    missing_group = X.iloc[:4].copy()
    missing_group.loc[missing_group.index[0], "segment"] = None
    with pytest.raises(ValueError, match="missing"):
        model.predict(missing_group, random_effects=random_effects)

    missing_x = X.iloc[:4].copy()
    missing_x.loc[missing_x.index[0], "x"] = np.nan
    with pytest.raises(ValueError, match="missing|non-finite"):
        model.predict(missing_x, random_effects=random_effects)


@pytest.mark.parametrize(
    "predictor_name",
    ["_predict_eta_exact", "_predict_eta_fast_discrete"],
)
def test_factor_smooth_population_prediction_ignores_unseen_policy(
    predictor_name: str,
) -> None:
    X, y = _data()
    model = _model(discrete=True, unseen="error").fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    known = X.iloc[:4].copy()
    unseen = known.copy()
    unseen["segment"] = "never-seen"
    predictor = getattr(model, predictor_name)

    np.testing.assert_allclose(
        predictor(unseen, random_effects="population"),
        predictor(known, random_effects="population"),
    )
    with pytest.raises(ValueError, match="unseen FactorSmooth"):
        predictor(unseen, random_effects="conditional")


@pytest.mark.parametrize("discrete", [False, True])
def test_factor_smooth_prediction_survives_released_state_and_pickle(
    discrete: bool,
) -> None:
    X, y = _data()
    model = _model(discrete=discrete, retain_fit_state=False).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    rows = X.iloc[:12]
    expected = model.predict(rows)

    assert model._dm is None
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(rows), expected)
    np.testing.assert_allclose(
        restored.predict(rows, random_effects="population"),
        model.predict(rows, random_effects="population"),
    )
    report = model.factor_smooth("x:segment:fs", grid=7)
    restored_report = restored.factor_smooth("x:segment:fs", grid=7)
    pd.testing.assert_frame_equal(restored_report.table, report.table)
    pd.testing.assert_frame_equal(restored_report.curves, report.curves)


def test_factor_smooth_report_matches_dense_covariance_and_edf() -> None:
    X, y = _data()
    dense = _model(discrete=False, direct_solve="gram").fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    structured = _model(discrete=False).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )
    grid = np.linspace(-0.9, 0.9, 11)
    dense_report = dense.factor_smooth("x:segment:fs", grid=grid)
    report = structured.factor_smooth("x:segment:fs", grid=grid)

    assert isinstance(report, FactorSmoothResult)
    assert report.name == "x:segment:fs"
    assert report.variable == "x"
    assert report.grouping_variable == "segment"
    assert report.smoothing_lambdas == {
        "wiggle": 1.3,
        "null_0": 0.7,
        "null_1": 0.9,
    }
    assert report.variance_components == pytest.approx(
        {name: report.phi / value for name, value in report.lambdas.items()}
    )
    assert report.effective_df == pytest.approx(
        structured._group_edf["x:segment:fs"],
        abs=2e-10,
    )
    np.testing.assert_allclose(
        report.table["effective_df"],
        dense_report.table["effective_df"],
        atol=3e-9,
    )
    np.testing.assert_allclose(
        report.curves["effect"],
        dense_report.curves["effect"],
        atol=3e-9,
    )
    np.testing.assert_allclose(
        report.curves["posterior_se"],
        dense_report.curves["posterior_se"],
        atol=3e-9,
    )

    state = structured._linear_system_state
    support = state.support_totals["x:segment:fs"]
    assert isinstance(support, FactorSmoothLevelSupport)
    local_penalty = state.penalized_operator.D - state.system.operator.D
    expected_credibility = np.array(
        [
            np.trace(
                np.eye(state.system.operator.block_size)
                - np.linalg.solve(state.penalized_operator.D[level], local_penalty[level])
            )
            / state.system.operator.block_size
            for level in range(state.system.operator.n_levels)
        ]
    )
    np.testing.assert_allclose(
        report.table["credibility"],
        expected_credibility,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        report.table["shrinkage"],
        1.0 - expected_credibility,
    )


def test_factor_smooth_report_selects_levels_and_flags_insufficient_support() -> None:
    X, y = _data()
    rare_rows = X["segment"] == "segment-5"
    keep_rare = np.flatnonzero(rare_rows.to_numpy())[:3]
    keep_common = np.flatnonzero(~rare_rows.to_numpy())
    keep = np.concatenate((keep_common, keep_rare))
    model = _model(discrete=False).fit_reml(
        X.iloc[keep].reset_index(drop=True),
        y[keep],
        max_reml_iter=2,
        runtime_validation="skip",
    )

    report = model.factor_smooth(
        "x:segment:fs",
        grid=5,
        levels=["segment-5", "segment-1"],
    )

    assert report.curves["level"].drop_duplicates().tolist() == [
        "segment-5",
        "segment-1",
    ]
    rare = report.table.set_index("level").loc["segment-5"]
    assert rare["count"] == 3
    assert not rare["sufficient_support"]
    assert report.diagnostics["n_levels_with_insufficient_support"] >= 1
    with pytest.raises(KeyError, match="Unknown FactorSmooth levels"):
        model.factor_smooth("x:segment:fs", levels=["not-fitted"])


def test_factor_smooth_zero_weight_level_has_no_information() -> None:
    X, y = _data()
    sample_weight = np.ones(len(X))
    zero_weight_level = X["segment"] == "segment-5"
    sample_weight[zero_weight_level.to_numpy()] = 0.0
    model = _model(discrete=False).fit_reml(
        X,
        y,
        sample_weight=sample_weight,
        max_reml_iter=2,
        runtime_validation="skip",
    )

    report = model.factor_smooth("x:segment:fs", grid=5)
    level = report.table.set_index("level").loc["segment-5"]

    assert level["count"] > 0
    assert level["fit_weight"] == 0.0
    assert level["information_trace"] == 0.0
    assert not level["has_information"]
    assert report.diagnostics["n_levels_without_information"] >= 1


def test_unpenalized_factor_smooth_reports_infinite_variance_components() -> None:
    X, y = _data()
    policies = {
        "wiggle": LambdaPolicy.off(),
        "null_0": LambdaPolicy.off(),
        "null_1": LambdaPolicy.off(),
    }
    model = SuperGLM(
        family="gaussian",
        features={"z": Numeric()},
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                k=6,
                lambda_policy=policies,
            )
        ],
        selection_penalty=0.0,
        direct_solve="gram",
    ).fit_reml(
        X,
        y,
        runtime_validation="skip",
    )

    report = model.factor_smooth("x:segment:fs", grid=5)

    assert set(report.lambdas.values()) == {0.0}
    assert all(np.isinf(value) for value in report.variance_components.values())


def test_factor_smooth_upper_boundary_reports_collapse() -> None:
    X, y = _data()
    model = _model(
        discrete=False,
        lambda_value=1.0e10,
    ).fit_reml(
        X,
        y,
        max_reml_iter=2,
        runtime_validation="skip",
    )

    with pytest.warns(UserWarning, match="collapsed"):
        report = model.factor_smooth("x:segment:fs", grid=5)

    assert report.collapsed
    assert all(report.at_upper_boundary.values())
    assert report.table["collapsed"].all()
