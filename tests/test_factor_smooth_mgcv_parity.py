"""Pinned predictive parity against mgcv's ``bs="fs"`` factor smooth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import FactorSmooth, Spline, SuperGLM

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "factor_smooth_mgcv_reference.json"
_CASE_NAMES = (
    "gaussian",
    "poisson",
    "poisson_global",
    "poisson_global_discrete",
)


@pytest.fixture(scope="module")
def mgcv_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def fitted_cases(mgcv_fixture: dict) -> dict[str, SuperGLM]:
    fitted: dict[str, SuperGLM] = {}
    for name in _CASE_NAMES:
        case = mgcv_fixture[name]
        data = case["data"]
        X = pd.DataFrame({"x": data["x"], "f": data["f"]})
        y = np.asarray(data["y"], dtype=np.float64)
        offset = (
            None
            if "exposure" not in data
            else np.log(np.asarray(data["exposure"], dtype=np.float64))
        )
        has_global = "global" in name
        model = SuperGLM(
            family="gaussian" if name == "gaussian" else "poisson",
            features={"x": Spline(kind="ps", k=7, m=2)} if has_global else {},
            interactions=[FactorSmooth("x", group="f", k=6, m=2)],
            selection_penalty=0.0,
            direct_solve="structured",
            discrete=name.endswith("_discrete"),
            n_bins=512,
            tol=1.0e-10,
            max_iter=200,
        )
        model.fit_reml(
            X,
            y,
            offset=offset,
            max_reml_iter=50,
            reml_tol=1.0e-9,
            pirls_tol=1.0e-10,
            max_pirls_iter=200,
            runtime_validation="skip",
        )
        fitted[name] = model
    return fitted


def _prediction_frame(case: dict) -> tuple[pd.DataFrame, np.ndarray | None]:
    values = case["prediction_data"]
    frame = pd.DataFrame({"x": values["x"], "f": values["f"]})
    offset = (
        None
        if "exposure" not in values
        else np.log(np.asarray(values["exposure"], dtype=np.float64))
    )
    return frame, offset


def _unseen_frame(case: dict) -> tuple[pd.DataFrame, np.ndarray | None]:
    values = case["unseen_data"]
    frame = pd.DataFrame(
        {
            "x": values["x"],
            "f": [f"unseen-{index}" for index in range(len(values["x"]))],
        }
    )
    offset = (
        None
        if "exposure" not in values
        else np.log(np.asarray(values["exposure"], dtype=np.float64))
    )
    return frame, offset


def test_reference_fixture_is_pinned_to_documented_mgcv_release(mgcv_fixture: dict) -> None:
    assert mgcv_fixture["metadata"]["r_version"] == "R version 4.5.3 (2026-03-11)"
    assert mgcv_fixture["metadata"]["mgcv_version"] == "1.9.4"
    assert mgcv_fixture["metadata"]["factor_smooth"] == (
        's(x, f, bs="fs", k=6, xt=list(bs="ps"), m=2)'
    )


@pytest.mark.parametrize(
    ("name", "prediction_rtol", "deviance_rtol", "edf_atol"),
    [
        ("gaussian", 7.0e-4, 2.0e-6, 2.0e-3),
        ("poisson", 8.0e-3, 4.0e-4, 8.0e-2),
        ("poisson_global", 6.0e-3, 2.0e-4, 5.0e-2),
        ("poisson_global_discrete", 9.0e-3, 5.0e-4, 9.0e-2),
    ],
)
def test_factor_smooth_fit_matches_mgcv_reml_and_freml(
    mgcv_fixture: dict,
    fitted_cases: dict[str, SuperGLM],
    name: str,
    prediction_rtol: float,
    deviance_rtol: float,
    edf_atol: float,
) -> None:
    case = mgcv_fixture[name]
    reference = case["reference"]
    model = fitted_cases[name]
    report = model.factor_smooth("x:f:fs", grid=np.asarray(case["curve_grid"]))
    prediction_frame, offset = _prediction_frame(case)

    assert model.result.deviance == pytest.approx(
        reference["deviance"],
        rel=deviance_rtol,
        abs=2.0e-7,
    )
    assert model.result.effective_df == pytest.approx(reference["total_edf"], abs=edf_atol)
    assert report.effective_df == pytest.approx(reference["factor_smooth_edf"], abs=edf_atol)
    assert model.result.phi == pytest.approx(reference["scale"], rel=3.0e-4, abs=2.0e-8)

    # mgcv rescales every smooth penalty before optimization.  The fixture
    # divides its reported sp by S.scale so these values multiply the same
    # unscaled natural penalties that SuperGLM retains.
    actual_wiggle = report.lambdas["wiggle"]
    expected_wiggle = reference["unscaled_lambdas"]["wiggle"]
    assert actual_wiggle == pytest.approx(expected_wiggle, rel=2.5e-2)
    np.testing.assert_allclose(
        np.sort([report.lambdas["null_0"], report.lambdas["null_1"]]),
        np.sort(
            [
                reference["unscaled_lambdas"]["null_0"],
                reference["unscaled_lambdas"]["null_1"],
            ]
        ),
        rtol=0.12,
    )
    np.testing.assert_allclose(
        np.sort(
            [
                report.variance_components["null_0"],
                report.variance_components["null_1"],
            ]
        ),
        np.sort(
            [
                reference["variance_components"]["null_0"],
                reference["variance_components"]["null_1"],
            ]
        ),
        rtol=0.12,
    )

    np.testing.assert_allclose(
        model.predict(prediction_frame, offset=offset),
        reference["conditional_prediction"],
        rtol=prediction_rtol,
        atol=3.0e-4,
    )
    np.testing.assert_allclose(
        model.predict(
            prediction_frame,
            offset=offset,
            random_effects="population",
        ),
        reference["population_prediction"],
        rtol=prediction_rtol,
        atol=3.0e-4,
    )
    actual_deviation = model._predict_eta_exact(
        prediction_frame,
        random_effects="conditional",
    ) - model._predict_eta_exact(
        prediction_frame,
        random_effects="population",
    )
    np.testing.assert_allclose(
        actual_deviation,
        reference["factor_smooth_link"],
        rtol=prediction_rtol,
        atol=2.5e-3,
    )
    np.testing.assert_allclose(
        report.curves["effect"],
        reference["factor_smooth_link"],
        rtol=prediction_rtol,
        atol=2.5e-3,
    )

    if "global_unscaled_lambda" in reference:
        assert model._reml_lambdas["x"] == pytest.approx(
            reference["global_unscaled_lambda"],
            rel=5.0e-2,
        )


@pytest.mark.parametrize("name", _CASE_NAMES)
def test_unseen_factor_levels_use_mgcv_population_prediction(
    mgcv_fixture: dict,
    fitted_cases: dict[str, SuperGLM],
    name: str,
) -> None:
    case = mgcv_fixture[name]
    model = fitted_cases[name]
    frame, offset = _unseen_frame(case)
    conditional = model.predict(frame, offset=offset, random_effects="conditional")
    population = model.predict(frame, offset=offset, random_effects="population")

    np.testing.assert_allclose(conditional, population, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        population,
        case["reference"]["unseen_population_prediction"],
        rtol=9.0e-3,
        atol=3.0e-4,
    )
