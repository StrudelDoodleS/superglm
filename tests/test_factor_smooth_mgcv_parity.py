"""Pinned parity against mgcv fitting the ``bs="fs"`` factor smooth's penalties."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import superglm.features.factor_smooth as factor_smooth_module
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
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


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
    # oracles/factor_smooth_mgcv_reference.R: mgcv fits fs's penalties through paraPen,
    # as its bs="fs" names its null columns in an eigensolver's order.
    assert mgcv_fixture["metadata"]["r_version"] == "R version 4.5.0 (2025-04-11)"
    assert mgcv_fixture["metadata"]["mgcv_version"] == "1.9.3"
    assert mgcv_fixture["metadata"]["factor_smooth"].startswith(
        'paraPen on s(x, bs="ps", k=6, m=2) per level of f'
    )


def test_superglm_source_does_not_name_private_mgcv_symbols() -> None:
    source_root = Path(factor_smooth_module.__file__).parents[1]
    private_namespace = "mgcv" + ":" * 3
    offenders = [
        str(path.relative_to(source_root))
        for path in source_root.rglob("*.py")
        if private_namespace in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


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

    # paraPen penalties are not rescaled, so mgcv's sp multiply the same
    # penalties SuperGLM retains, each null_j by name: the level's constant and
    # its line.
    for component in ("wiggle", "null_0", "null_1"):
        assert report.lambdas[component] == pytest.approx(
            reference["unscaled_lambdas"][component], rel=2.5e-2
        )
        assert report.variance_components[component] == pytest.approx(
            reference["variance_components"][component], rel=2.5e-2
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

    # Exact: an unseen level contributes exactly zero to the conditional prediction, and
    # adding zero changes no bit of the population prediction.
    np.testing.assert_allclose(conditional, population, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        population,
        case["reference"]["unseen_population_prediction"],
        rtol=9.0e-3,
        atol=3.0e-4,
    )
