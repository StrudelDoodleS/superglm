"""Prediction and basis-aware reporting for sum-to-zero factor smooths."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, LambdaPolicy, Spline, SuperGLM


def _data(
    *,
    labels: tuple[object, ...] = ("alpha", "beta", "delta", "gamma", "omega"),
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(8173)
    counts = np.array([72, 56, 41, 25, 9])
    codes = np.repeat(np.arange(len(labels)), counts)
    order = rng.permutation(len(codes))
    codes = codes[order]
    x = rng.uniform(-1.0, 1.0, size=len(codes))
    z = rng.uniform(-0.9, 0.9, size=len(codes))
    raw_coefficients = np.array(
        [
            [0.48, -0.14, 0.16],
            [-0.36, 0.22, -0.08],
            [0.29, 0.11, 0.05],
            [-0.18, -0.17, 0.02],
            [-0.23, -0.02, -0.15],
        ]
    )
    raw_coefficients -= raw_coefficients.mean(axis=0)
    deviation = (
        raw_coefficients[codes, 0]
        + raw_coefficients[codes, 1] * x
        + raw_coefficients[codes, 2] * x**2
    )
    y = (
        0.65
        + 0.42 * np.sin(2.2 * x)
        - 0.24 * np.cos(1.7 * z)
        + deviation
        + rng.normal(scale=0.11, size=len(codes))
    )
    X = pd.DataFrame(
        {
            "x": x,
            "z": z,
            "segment": np.asarray(labels, dtype=object)[codes],
        }
    )
    return X, y


def _model(
    *,
    discrete: bool = False,
    direct_solve: str = "structured",
    unseen: str = "population",
    retain_fit_state: bool = True,
    lambda_value: float = 1.7,
) -> SuperGLM:
    return SuperGLM(
        family="gaussian",
        features={
            "x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.2)),
            "z": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(0.9)),
        },
        interactions=[
            FactorSmooth(
                "x",
                group="segment",
                basis="sz",
                k=6,
                unseen=unseen,
                lambda_policy={"wiggle": LambdaPolicy.fixed(lambda_value)},
            )
        ],
        selection_penalty=0.0,
        discrete=discrete,
        n_bins=160,
        direct_solve=direct_solve,
        retain_fit_state=retain_fit_state,
    )


def _fit(model: SuperGLM, X: pd.DataFrame, y: np.ndarray) -> SuperGLM:
    return model.fit_reml(
        X,
        y,
        max_reml_iter=2,
        pirls_tol=1e-10,
        runtime_validation="skip",
    )


@pytest.mark.parametrize("direct_solve", ["gram", "auto"])
def test_released_sz_report_is_backend_neutral(direct_solve: str):
    X, y = _data()
    keep = X["segment"].isin(["alpha", "beta"])
    X = X.loc[keep].reset_index(drop=True)
    y = y[keep.to_numpy()]
    model = _fit(
        _model(
            direct_solve=direct_solve,
            retain_fit_state=False,
        ),
        X,
        y,
    )

    assert model.result.direct_backend == "gram"
    report = model.factor_smooth("x:segment:sz", grid=9)
    restored = pickle.loads(pickle.dumps(model)).factor_smooth(
        "x:segment:sz",
        grid=9,
    )
    pd.testing.assert_frame_equal(restored.table, report.table)
    pd.testing.assert_frame_equal(restored.curves, report.curves)


@pytest.mark.parametrize("discrete", [False, True])
def test_sz_report_has_deviation_semantics_and_exact_zero_sum(discrete: bool) -> None:
    X, y = _data()
    model = _fit(_model(discrete=discrete), X, y)
    grid = np.linspace(-0.85, 0.85, 13)

    report = model.factor_smooth("x:segment:sz", grid=grid)

    assert report.basis == "sz"
    assert report.collapsed is None
    assert report.lambdas == {"wiggle": 1.7}
    assert {"credibility", "shrinkage", "collapsed"}.isdisjoint(report.table)
    assert report.table["level"].tolist() == list(model._interaction_specs["x:segment:sz"]._levels)
    assert np.all(np.isfinite(report.table["effective_df"]))
    assert report.table["effective_df"].sum() == pytest.approx(
        model._group_edf["x:segment:sz"],
        abs=2e-9,
    )
    assert report.effective_df == pytest.approx(model._group_edf["x:segment:sz"], abs=2e-9)
    assert np.all(np.isfinite(report.curves[["effect", "posterior_se", "lower", "upper"]]))
    pivot = report.curves.pivot(index="x", columns="level", values="effect")
    np.testing.assert_allclose(pivot.sum(axis=1), 0.0, atol=2e-12)
    assert report.diagnostics["max_abs_level_effect_sum"] < 2e-12


def test_sz_report_dense_and_structured_covariance_match_including_final_level() -> None:
    X, y = _data()
    grid = np.linspace(-0.8, 0.8, 9)
    dense = _fit(_model(direct_solve="gram"), X, y)
    structured = _fit(_model(direct_solve="structured"), X, y)

    dense_report = dense.factor_smooth("x:segment:sz", grid=grid)
    report = structured.factor_smooth("x:segment:sz", grid=grid)

    np.testing.assert_allclose(
        report.table["effective_df"],
        dense_report.table["effective_df"],
        atol=4e-9,
    )
    np.testing.assert_allclose(report.curves["effect"], dense_report.curves["effect"], atol=4e-9)
    np.testing.assert_allclose(
        report.curves["posterior_se"],
        dense_report.curves["posterior_se"],
        atol=4e-9,
    )
    final_level = report.table["level"].iloc[-1]
    final_rows = report.curves["level"] == final_level
    assert np.all(np.isfinite(report.curves.loc[final_rows, "posterior_se"]))


def test_sz_prediction_population_and_unseen_policies() -> None:
    X, y = _data()
    model = _fit(_model(), X, y)
    term = model._interaction_specs["x:segment:sz"]
    group = next(group for group in model._groups if group.name == "x:segment:sz")

    conditional = model._predict_eta_exact(X, random_effects="conditional")
    population = model._predict_eta_exact(X, random_effects="population")
    expected_deviation = term.score(
        X["x"].to_numpy(),
        X["segment"].to_numpy(),
        model.result.beta[group.sl],
    )
    np.testing.assert_allclose(conditional - population, expected_deviation, atol=2e-12)
    assert model.predict(X.iloc[[0]]).shape == (1,)
    assert model.predict(X.iloc[[0]], random_effects="population").shape == (1,)

    unseen = X.iloc[:4].copy()
    unseen["segment"] = "not-fitted"
    np.testing.assert_allclose(
        model.predict(unseen),
        model.predict(unseen, random_effects="population"),
    )

    error_model = _fit(_model(unseen="error"), X, y)
    with pytest.raises(ValueError, match="not-fitted"):
        error_model.predict(unseen)
    np.testing.assert_allclose(
        error_model.predict(unseen, random_effects="population"),
        error_model.predict(X.iloc[:4], random_effects="population"),
    )


def test_sz_numeric_zero_and_negative_zero_group_keys_predict_identically() -> None:
    X, y = _data(labels=(0.0, 1.0, 2.0, 3.0, 4.0))
    model = _fit(_model(), X, y)
    row = X.iloc[[0]].copy()
    row["segment"] = 0.0
    negative_zero = row.copy()
    negative_zero["segment"] = -0.0

    np.testing.assert_allclose(model.predict(row), model.predict(negative_zero), atol=0.0)


def test_sz_large_wiggle_lambda_does_not_claim_collapse_and_reports_without_design() -> None:
    X, y = _data()
    dense = _fit(
        _model(
            direct_solve="gram",
            lambda_value=1.0e10,
        ),
        X,
        y,
    )
    model = _fit(
        _model(
            lambda_value=1.0e10,
            retain_fit_state=False,
        ),
        X,
        y,
    )

    assert model._dm is None
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        report = model.factor_smooth("x:segment:sz", grid=11)

    assert report.collapsed is None
    assert report.at_upper_boundary == {"wiggle": True}
    assert np.all(np.isfinite(report.curves["effect"]))
    assert np.linalg.norm(report.curves["effect"]) > 1e-5
    pivot = report.curves.pivot(index="x", columns="level", values="effect")
    np.testing.assert_allclose(pivot.sum(axis=1), 0.0, atol=2e-12)
    np.testing.assert_allclose(model.predict(X), dense.predict(X), atol=5e-5)
    assert model.result.deviance == pytest.approx(dense.result.deviance, abs=1e-6)


def test_sz_level_edf_is_invariant_to_level_relabeling() -> None:
    labels = ("alpha", "beta", "delta", "gamma", "omega")
    relabeled = ("zeta", "upsilon", "tau", "sigma", "rho")
    X, y = _data(labels=labels)
    renamed = X.copy()
    mapping = dict(zip(labels, relabeled, strict=True))
    renamed["segment"] = renamed["segment"].map(mapping)

    original = _fit(_model(), X, y).factor_smooth("x:segment:sz", grid=5)
    changed = _fit(_model(), renamed, y).factor_smooth("x:segment:sz", grid=5)
    original_edf = original.table.set_index("level")["effective_df"]
    changed_edf = changed.table.set_index("level")["effective_df"]

    for old, new in mapping.items():
        assert changed_edf[new] == pytest.approx(original_edf[old], abs=4e-9)


def test_sz_sparse_levels_keep_other_global_smooth_in_summary_and_reconstruction() -> None:
    X, y = _data()
    model = _fit(_model(), X, y)

    summary_text = str(model.summary())
    reconstructed = model.reconstruct_feature("z")

    assert "z" in summary_text
    assert np.all(np.isfinite(reconstructed["x"]))
    assert np.all(np.isfinite(reconstructed["log_relativity"]))


def test_sz_structured_estimability_uses_centered_data_geometry():
    X, y = _data()
    rare = X["segment"] == "omega"
    keep = ~rare
    keep[np.flatnonzero(rare)[:2]] = True
    X = X.loc[keep].reset_index(drop=True)
    y = y[keep.to_numpy()]

    def model(direct_solve: str) -> SuperGLM:
        return SuperGLM(
            family="gaussian",
            features={
                "x": Spline(n_knots=5, lambda_policy=LambdaPolicy.fixed(1.2)),
            },
            interactions=[
                FactorSmooth(
                    "x",
                    group="segment",
                    basis="sz",
                    k=6,
                    lambda_policy={"wiggle": LambdaPolicy.fixed(1.7)},
                )
            ],
            selection_penalty=0.0,
            direct_solve=direct_solve,
        ).fit_reml(X, y, runtime_validation="skip")

    dense = model("gram")
    structured = model("structured")

    np.testing.assert_array_equal(
        structured._fit_inference_info["coefficient_estimable"],
        dense._fit_inference_info["coefficient_estimable"],
    )
    assert not np.any(structured._fit_inference_info["coefficient_estimable"])


def test_sz_summary_uses_one_structured_group_row():
    X, y = _data()
    model = _fit(_model(), X, y)

    row = next(row for row in model.summary()._coef_rows if row.name == "x:segment:sz")

    assert row.coef is None
    assert row.structured_kind == "factor_smooth_sz"
    assert row.n_levels == len(model._interaction_specs["x:segment:sz"]._levels)
    assert row.n_params == (row.n_levels - 1) * 6
    assert row.smoothing_lambdas == (("wiggle", 1.7),)
    assert "factor smooth (sz)" in str(model.summary()).lower()
