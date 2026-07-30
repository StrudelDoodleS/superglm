"""Pinned parity against mgcv's ``bs="re"`` random-effect smooth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superglm import Numeric, RandomEffect, SuperGLM

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "random_effect_mgcv_reference.json"


@pytest.fixture(scope="module")
def mgcv_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def fitted_cases(mgcv_fixture: dict) -> dict[str, tuple[SuperGLM, pd.DataFrame, np.ndarray]]:
    fitted = {}
    for name in ("gaussian", "poisson", "poisson_discrete"):
        case = mgcv_fixture[name]
        data = case["data"]
        X = pd.DataFrame({"x": data["x"], "level": data["level"]})
        y = np.asarray(data["y"], dtype=np.float64)
        offset = (
            None
            if "exposure" not in data
            else np.log(np.asarray(data["exposure"], dtype=np.float64))
        )
        model = SuperGLM(
            family="gaussian" if name == "gaussian" else "poisson",
            features={"x": Numeric(), "level": RandomEffect()},
            selection_penalty=0,
            direct_solve="structured",
            discrete=name == "poisson_discrete",
            tol=1e-10,
            max_iter=200,
        )
        model.fit_reml(
            X,
            y,
            offset=offset,
            max_reml_iter=50,
            reml_tol=1e-10,
            pirls_tol=1e-10,
            max_pirls_iter=200,
            runtime_validation="skip",
        )
        fitted[name] = (model, X, offset)
    return fitted


def test_reference_fixture_is_pinned_to_documented_mgcv_release(mgcv_fixture: dict):
    assert mgcv_fixture["metadata"]["r_version"] == "R version 4.5.3 (2026-03-11)"
    assert mgcv_fixture["metadata"]["mgcv_version"] == "1.9.4"


@pytest.mark.parametrize(
    ("name", "relative_tolerance", "prediction_tolerance"),
    [
        ("gaussian", 2e-5, 2e-7),
        ("poisson", 2e-6, 2e-7),
        ("poisson_discrete", 3e-6, 3e-7),
    ],
)
def test_random_effect_fit_matches_mgcv_reml_and_freml(
    mgcv_fixture: dict,
    fitted_cases: dict[str, tuple[SuperGLM, pd.DataFrame, np.ndarray]],
    name: str,
    relative_tolerance: float,
    prediction_tolerance: float,
):
    model, X, offset = fitted_cases[name]
    reference = mgcv_fixture[name]["reference"]
    report = model.random_effects("level")

    actual_scalars = {
        "lambda": report.lambda_value,
        "scale": model.result.phi,
        "variance_component": report.variance_component,
        "smooth_edf": report.effective_df,
        "total_edf": model.result.effective_df,
        "deviance": model.result.deviance,
        "intercept": model.result.intercept,
        "slope": model.result.beta[0],
    }
    for field, actual in actual_scalars.items():
        assert actual == pytest.approx(
            reference[field],
            rel=relative_tolerance,
            abs=2e-8,
        ), field

    np.testing.assert_allclose(
        report.table["effect"],
        reference["random_effects"],
        rtol=relative_tolerance,
        atol=prediction_tolerance,
    )
    np.testing.assert_allclose(
        model.predict(X, offset=offset),
        reference["conditional_prediction"],
        rtol=relative_tolerance,
        atol=prediction_tolerance,
    )
    np.testing.assert_allclose(
        model.predict(X, offset=offset, random_effects="population"),
        reference["population_prediction"],
        rtol=relative_tolerance,
        atol=prediction_tolerance,
    )


@pytest.mark.parametrize("name", ["gaussian", "poisson", "poisson_discrete"])
def test_unseen_random_effect_levels_score_at_mgcv_population_prediction(
    mgcv_fixture: dict,
    fitted_cases: dict[str, tuple[SuperGLM, pd.DataFrame, np.ndarray]],
    name: str,
):
    model, _, _ = fitted_cases[name]
    reference = mgcv_fixture[name]["reference"]
    x = np.array([-0.7, 0.15, 1.1])
    X = pd.DataFrame({"x": x, "level": ["unseen-a", "unseen-b", "unseen-c"]})
    if name == "gaussian":
        offset = None
        expected = reference["intercept"] + reference["slope"] * x
    else:
        exposure = np.array([0.6, 1.3, 2.1])
        offset = np.log(exposure)
        expected = exposure * np.exp(reference["intercept"] + reference["slope"] * x)

    population = model.predict(X, offset=offset, random_effects="population")
    conditional = model.predict(X, offset=offset, random_effects="conditional")

    np.testing.assert_allclose(population, expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(conditional, population, rtol=0.0, atol=0.0)
