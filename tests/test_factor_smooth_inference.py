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
from superglm.solvers._structured.state import (
    FactorSmoothLevelSupport,
)
from tests._identifiability_parity import assert_same_identifiability, coefficient_ses


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


def test_factor_smooth_population_prediction_ignores_unseen_policy() -> None:
    X, y = _data()
    model = _model(discrete=True, unseen="error").fit_reml(
        X,
        y,
        max_reml_iter=2,
    )
    known = X.iloc[:4].copy()
    unseen = known.copy()
    unseen["segment"] = "never-seen"
    predictor = model._predict_eta_exact

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
    )
    structured = _model(discrete=False).fit_reml(
        X,
        y,
        max_reml_iter=2,
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
    )

    report = model.factor_smooth("x:segment:fs", grid=5)
    level = report.table.set_index("level").loc["segment-5"]

    assert level["count"] > 0
    assert level["fit_weight"] == 0.0
    assert level["information_trace"] == 0.0
    assert not level["has_information"]
    assert report.diagnostics["n_levels_without_information"] >= 1


@pytest.mark.pins_behaviour
def test_unpenalized_factor_smooth_reports_infinite_variance_components() -> None:
    """Control: a component with its smoothing parameter off reports an infinite variance.

    Master reported it for every component; null_0 now stays penalized, since a zero
    null_0 beside the intercept is refused, and the off components still report it.
    """
    X, y = _data()
    # A zero null_0 aliases the levels' constants with the intercept and is refused.
    policies = {
        "wiggle": LambdaPolicy.off(),
        "null_0": LambdaPolicy.fixed(1.0),
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
    )

    report = model.factor_smooth("x:segment:fs", grid=5)

    unpenalized = ("wiggle", "null_1")
    assert {name: report.lambdas[name] for name in unpenalized} == dict.fromkeys(unpenalized, 0.0)
    assert all(np.isinf(report.variance_components[name]) for name in unpenalized)
    assert np.isfinite(report.variance_components["null_0"])


def _alias_rows() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(203)
    n = 240
    group = np.repeat(np.array(["a", "b", "c"], dtype=object), n // 3)
    x = rng.uniform(-1.0, 1.0, n)
    y = 6.0 + np.sin(2.0 * x) + np.repeat([1.0, 2.0, 4.0], n // 3) + 0.1 * rng.normal(size=n)
    return pd.DataFrame({"x": x, "group": group}), y


@pytest.mark.parametrize("null_0", [0.0, 1.0e-14])
def test_null_0_at_zero_is_refused_below_the_floor_disclosed_and_above_it_prices_in_any_order(
    null_0,
) -> None:
    """A null_0 below the REML floor leaves the levels' constants free against the intercept.

    The rank treatment kept all but the last-listed level's constant, so listing the
    same levels in another order moved the unseen-level price (9.893 against 6.988 on
    this data, at zero and at 1e-14 alike) and the plotted standard errors.  The fit
    refuses zero and warns below the floor; above the floor the penalty identifies the
    constants, and the declared order does not matter.
    """
    from superglm.plotting.interactions import _factor_smooth_curves

    X, y = _alias_rows()
    unseen = pd.DataFrame({"x": [0.0], "group": ["unseen"]})
    policy = {"wiggle": LambdaPolicy.fixed(2.0), "null_1": LambdaPolicy.fixed(30.0)}
    prices, errors = [], []
    for levels in (["a", "b", "c"], ["c", "b", "a"]):

        def model(value, levels=levels):
            smooth = FactorSmooth(
                "x",
                group="group",
                levels=levels,
                lambda_policy={**policy, "null_0": LambdaPolicy.fixed(value)},
            )
            return SuperGLM(
                family="gaussian", interactions=[smooth], selection_penalty=0.0, tol=1e-10
            )

        if null_0 == 0.0:
            with pytest.raises(ValueError, match="aliased with the fitted intercept"):
                model(null_0).fit_reml(X, y)
        else:
            with pytest.warns(UserWarning, match="may follow the order the levels"):
                model(null_0).fit_reml(X, y)
        fitted = model(1.0e-2).fit_reml(X, y)
        prices.append(fitted.predict(unseen)[0])
        curves = _factor_smooth_curves(fitted, "x:group:fs", n_points=7)["per_level"]
        errors.append(np.concatenate([curves[level]["se_log_relativity"] for level in "abc"]))

    assert prices[1] == pytest.approx(prices[0], rel=1e-9)
    np.testing.assert_allclose(errors[1], errors[0], rtol=1e-9)


def test_superlss_refuses_a_zero_null_0_beside_an_intercept_and_fits_it_without_one() -> None:
    """SuperLSS: a zero null_0 is refused through ``fit(lambdas=...)`` and a declaration
    beside the intercept, and fitted without one, where the level constants are identified.

    Through ``fit(lambdas=...)`` the unseen-level price followed the level order
    (9.943 against 6.962); without an intercept both orders give the same fit.
    """
    from superglm import GaussianLS, SuperLSS, interaction

    X, y = _alias_rows()
    family = GaussianLS()
    lambdas = {"wiggle": 2.0, "null_0": 0.0, "null_1": 30.0}
    by_order = []
    for levels in (["a", "b", "c"], ["c", "b", "a"]):

        def model(intercept, declared, levels=levels):
            policy = {name: LambdaPolicy.fixed(value) for name, value in lambdas.items()}
            smooth = FactorSmooth(
                "x", group="group", levels=levels, lambda_policy=policy if declared else None
            )
            return SuperLSS(
                family, family.location(interaction(smooth), intercept=intercept), family.scale()
            )

        fixed = {f"location:x:group:fs#{name}": value for name, value in lambdas.items()}
        with pytest.raises(ValueError, match="aliased with the fitted intercept"):
            model(True, False).fit(X, y, lambdas=fixed)
        with pytest.raises(ValueError, match="aliased with the fitted intercept"):
            model(True, True).fit(X, y)
        with pytest.raises(ValueError, match="aliased with the fitted intercept"):
            model(True, True).fit_reml(X, y)
        by_order.append(model(False, True).fit(X, y).predict(X))

    np.testing.assert_allclose(by_order[1], by_order[0], rtol=1e-12)


def test_factor_smooth_upper_boundary_reports_collapse() -> None:
    X, y = _data()
    model = _model(
        discrete=False,
        lambda_value=1.0e10,
    ).fit_reml(
        X,
        y,
        max_reml_iter=2,
    )

    with pytest.warns(UserWarning, match="collapsed"):
        report = model.factor_smooth("x:segment:fs", grid=5)

    assert report.collapsed
    assert all(report.at_upper_boundary.values())
    assert report.table["collapsed"].all()


def test_factor_smooth_standard_errors_survive_a_near_singular_level() -> None:
    """Forty levels of 10-23 rows with k=5: one level's data block is singular
    to ~1e-13 (equilibrated).  The structured backend once reported NaN for
    every coefficient here, the numeric column ``z`` included.  The fs penalty
    curves every direction of a level, its line included, so the fit identifies
    every coefficient (``solvers.identifiability``): both backends report every
    standard error, the same values and footnotes.  Mutation: stage 1's
    structured backend blanked the data's null set.
    """
    rng = np.random.default_rng(5102)
    n, n_levels = 600, 40
    codes = rng.integers(0, n_levels, n)
    x = rng.uniform(size=n)
    z = rng.normal(size=n)
    y = 0.2 * z + 0.4 * np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n_levels)[codes]
    y = y + rng.normal(0, 0.4, n)
    X = pd.DataFrame({"x": x, "z": z, "g": np.array([f"g{c:02d}" for c in codes], dtype=object)})
    metrics = {}
    for direct_solve in ("gram", "auto"):
        model = SuperGLM(
            family="gaussian",
            features={"z": Numeric()},
            interactions=[FactorSmooth("x", group="g", basis="fs", k=5)],
            selection_penalty=0.0,
            direct_solve=direct_solve,
        )
        model.fit_reml(X, y)
        metrics[direct_solve] = model.metrics(X, y)
    assert metrics["auto"]._model._reml_profile["direct_backend"] == "structured"
    assert not np.any(np.isnan(coefficient_ses(metrics["auto"])))
    assert_same_identifiability(metrics["auto"], metrics["gram"], rtol=1e-5)
