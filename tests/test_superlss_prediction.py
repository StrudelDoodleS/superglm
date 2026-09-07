from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.features import Numeric


@pytest.mark.parametrize("shift", [-1000.0, 1000.0])
def test_review_prediction_refuses_excluded_scale_floor_and_overflow_without_warning(shift):
    import warnings

    frame, response, _ = _fixture()
    model = SuperLSS(family=GaussianLS(scale_floor=0.03), predictors=_predictors()).fit(
        frame, response
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="scale.*support"):
            model.predict_parameters(frame, offsets={"scale": np.full(len(frame), shift)})


@pytest.mark.parametrize("family_name", ["GaussianLS", "GammaLS"])
def test_review_public_quantile_preserves_nonfinite_refusal(family_name):
    from superglm import distributional

    family = getattr(distributional, family_name)()
    frame, response, _ = _fixture()
    response = response if family_name == "GaussianLS" else np.exp(response)
    predictors = (Predictor(family.parameters[0].name, {"x": Numeric()}), Predictor("scale", {}))
    model = SuperLSS(family=family, predictors=predictors).fit(frame, response)
    with pytest.raises(ValueError):
        model.predict_quantile(frame, np.nan)


def test_review_plot_terms_resolve_across_selected_predictors():
    import matplotlib.pyplot as plt

    frame, response, _ = _fixture()
    model = SuperLSS(family=GaussianLS(), predictors=_predictors()).fit(frame, response)
    figures = model.plot(terms=["x", "z"], n_sim=32)
    assert set(figures) == {"location", "scale"}
    for figure in figures.values():
        plt.close(figure)
    for parameter, terms, missing in [
        (None, ["x", "typo"], "typo"),
        ("location", ["x", "z"], "z"),
        (None, ["typo"], "typo"),
    ]:
        with pytest.raises(ValueError, match=f"unknown.*{missing}"):
            model.plot(parameter=parameter, terms=terms, n_sim=32)


def test_review_explicit_plot_interaction_still_refuses():
    from superglm import NumericInteraction

    frame, response, _ = _fixture()
    model = SuperLSS(
        family=GaussianLS(),
        predictors=(
            Predictor(
                "location",
                {"x": Numeric(), "z": Numeric()},
                interaction_specs={"x:z": NumericInteraction("x", "z")},
            ),
            Predictor("scale", {}),
        ),
    ).fit(frame, response)
    with pytest.raises(NotImplementedError, match="interaction.*one-dimensional"):
        model.plot(terms=["x", "x:z"], n_sim=32)


def _fixture(n: int = 84) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(1701)
    x = np.linspace(-1.0, 1.0, n)
    z = np.mod(0.25 + 1.6 * x, 1.0)
    sigma = 0.2 + np.exp(-1.5 + 0.2 * np.cos(2.0 * np.pi * z))
    response = 0.4 + 0.8 * np.sin(np.pi * x) + rng.normal(scale=sigma)
    frame = pd.DataFrame(
        {"x": x, "z": z},
        index=pd.Index(np.arange(n) + 500, name="policy"),
    )
    offsets = {
        "location": np.linspace(-0.1, 0.12, n),
        "scale": 0.04 * np.cos(np.linspace(0.0, 2.0 * np.pi, n)),
    }
    return frame, response, offsets


def _predictors() -> tuple[Predictor, Predictor]:
    return (
        Predictor("location", {"x": Numeric()}),
        Predictor("scale", {"z": Numeric()}),
    )


def test_prediction_exposes_both_parameter_and_link_columns_with_input_index() -> None:
    frame, response, offsets = _fixture()
    model = SuperLSS(
        family=GaussianLS(scale_floor=0.03),
        predictors=_predictors(),
    ).fit(frame, response, offsets=offsets)

    link = model.predict_link(frame, offsets=offsets)
    parameters = model.predict_parameters(frame, offsets=offsets)
    mean = model.predict(frame, offsets=offsets)

    assert isinstance(link, pd.DataFrame)
    assert isinstance(parameters, pd.DataFrame)
    assert tuple(link.columns) == ("location", "scale")
    assert tuple(parameters.columns) == ("location", "scale")
    assert link.index.equals(frame.index)
    assert parameters.index.equals(frame.index)
    assert np.all(parameters["scale"].to_numpy() > 0.03)
    np.testing.assert_allclose(mean, parameters["location"].to_numpy())
    np.testing.assert_allclose(parameters["location"], link["location"])
    np.testing.assert_allclose(
        parameters["scale"],
        0.03 + np.exp(link["scale"]),
    )


def test_prediction_offsets_are_predictor_keyed_and_apply_on_link_scale() -> None:
    frame, response, _ = _fixture()
    model = SuperLSS(family=GaussianLS(), predictors=_predictors()).fit(frame, response)
    base = model.predict_link(frame)
    shifts = {
        "location": np.full(len(frame), 0.2),
        "scale": np.full(len(frame), -0.15),
    }
    shifted = model.predict_link(frame, offsets=shifts)

    np.testing.assert_allclose(shifted["location"] - base["location"], 0.2)
    np.testing.assert_allclose(shifted["scale"] - base["scale"], -0.15)
    with pytest.raises(ValueError, match="unknown offset.*shape"):
        model.predict_link(frame, offsets={"shape": np.zeros(len(frame))})
    with pytest.raises(TypeError, match="offset"):
        model.predict_link(  # type: ignore[call-arg]
            frame,
            offset=np.zeros(len(frame)),
        )


def test_prediction_and_fitted_views_require_a_successful_fit() -> None:
    frame, _, _ = _fixture()
    model = SuperLSS(family=GaussianLS(), predictors=_predictors())

    for operation in (
        lambda: model.predict_link(frame),
        lambda: model.predict_parameters(frame),
        lambda: model.predict(frame),
        lambda: model.result_,
        lambda: model.training_telemetry(),
    ):
        with pytest.raises(RuntimeError, match="not fitted"):
            operation()
