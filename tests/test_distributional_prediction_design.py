from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional import GaussianLS, Predictor
from superglm.features import Categorical, Numeric, Spline, SplineCategorical
from superglm.features.interaction import TensorInteraction


@pytest.fixture(scope="module")
def fitted_joint_model() -> tuple[pd.DataFrame, SuperLSS]:
    rng = np.random.default_rng(20_260_724)
    n = 120
    x1 = np.linspace(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    z = rng.normal(0.0, 1.0, n)
    region = np.array(["north", "south", "urban"], dtype=object)[np.arange(n) % 3]
    frame = pd.DataFrame({"x1": x1, "x2": x2, "z": z, "region": region})
    mu = 0.4 + 0.7 * np.sin(2.0 * np.pi * x1) + 0.15 * z + 0.2 * (region == "urban")
    sigma = 0.08 + np.exp(-1.6 + 0.25 * np.cos(2.0 * np.pi * x1))
    response = rng.normal(mu, sigma)
    predictors = (
        Predictor(
            "location",
            {
                "z": Numeric(),
                "x1": Spline(kind="cr", n_knots=5),
                "x2": Spline(kind="cr", n_knots=5),
                "region": Categorical(base="north"),
            },
            interaction_specs={
                "x1:x2": TensorInteraction("x1", "x2", n_knots=(4, 4)),
            },
        ),
        Predictor(
            "scale",
            {
                "x1": Spline(kind="cr", n_knots=5),
                "region": Categorical(base="north"),
            },
            interaction_specs={
                "x1:region": SplineCategorical("x1", "region"),
            },
        ),
    )
    penalty_names = (
        "location:x1#wiggle",
        "location:x2#wiggle",
        "location:x1:x2#margin_x1",
        "location:x1:x2#margin_x2",
        "scale:x1#wiggle",
        "scale:x1:region[south]#wiggle",
        "scale:x1:region[urban]#wiggle",
    )
    model = SuperLSS(family=GaussianLS(), predictors=predictors).fit(
        frame,
        response,
        lambdas={name: 0.5 for name in penalty_names},
    )
    return frame, model


def test_joint_prediction_design_matches_every_fitted_term_path(
    fitted_joint_model: tuple[pd.DataFrame, SuperLSS],
) -> None:
    from superglm.distributional.prediction_design import (
        build_joint_prediction_design,
        link_standard_errors,
    )

    frame, model = fitted_joint_model
    fitted = model._require_fitted()
    design = build_joint_prediction_design(
        frame,
        fitted.compiled_predictors,
        fitted.layout,
    )
    eta = model.predict_link(frame).to_numpy()

    assert design.parameter_names == ("location", "scale")
    assert isinstance(design.local, Mapping)
    for parameter_index, state in enumerate(fitted.layout.predictors):
        local = model.coef_by_predictor_[state.name]
        np.testing.assert_allclose(
            design.local[state.name] @ local,
            eta[:, parameter_index],
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        assert not design.local[state.name].flags.writeable

    standard_errors = link_standard_errors(
        design,
        model.covariance_,
        fitted.layout,
    )
    assert tuple(standard_errors) == ("location", "scale")
    assert all(np.all(values >= 0.0) for values in standard_errors.values())
    assert all(not values.flags.writeable for values in standard_errors.values())


def test_offsets_shift_eta_without_changing_coefficient_design(
    fitted_joint_model: tuple[pd.DataFrame, SuperLSS],
) -> None:
    from superglm.distributional.prediction_design import build_joint_prediction_design

    frame, model = fitted_joint_model
    fitted = model._require_fitted()
    before = build_joint_prediction_design(frame, fitted.compiled_predictors, fitted.layout)
    offsets = {
        "location": np.linspace(-0.2, 0.2, len(frame)),
        "scale": np.full(len(frame), 0.15),
    }

    base = model.predict_link(frame)
    shifted = model.predict_link(frame, offsets=offsets)
    after = build_joint_prediction_design(frame, fitted.compiled_predictors, fitted.layout)

    np.testing.assert_allclose(shifted["location"] - base["location"], offsets["location"])
    np.testing.assert_allclose(shifted["scale"] - base["scale"], offsets["scale"])
    for name in before.parameter_names:
        np.testing.assert_array_equal(before.local[name], after.local[name])


def test_materially_negative_prediction_variance_is_rejected(
    fitted_joint_model: tuple[pd.DataFrame, SuperLSS],
) -> None:
    from superglm.distributional.prediction_design import (
        build_joint_prediction_design,
        link_standard_errors,
    )

    frame, model = fitted_joint_model
    fitted = model._require_fitted()
    design = build_joint_prediction_design(
        frame.iloc[:3], fitted.compiled_predictors, fitted.layout
    )
    covariance = np.zeros_like(model.covariance_)
    covariance[0, 0] = -1.0

    with pytest.raises(ValueError, match="materially negative"):
        link_standard_errors(design, covariance, fitted.layout)
