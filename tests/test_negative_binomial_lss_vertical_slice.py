"""One public fixed-smoothing NB2 fit and fitted-artifact round trip."""

from __future__ import annotations

import numpy as np
import pandas as pd

from superglm import SuperLSS
from superglm.distributional import NegativeBinomialLS, Predictor
from superglm.features import Numeric

_INNER_TOLERANCE = 1.0e-8


def _fixture(
    n_rows: int = 240,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Draw N ~ NB2(exposure * mean, exposure * theta)."""

    rng = np.random.default_rng(2026083115)
    x_mean = rng.permutation(np.linspace(-1.0, 1.0, n_rows))
    x_theta = rng.permutation(np.linspace(-1.0, 1.0, n_rows))
    exposure = np.resize(np.array([0.5, 1.0, 1.5, 2.0]), n_rows)
    mean_offset = 0.08 * np.sin(np.pi * x_mean)
    theta_offset = -0.06 * np.cos(np.pi * x_theta)
    mean = np.exp(0.55 + 0.35 * x_mean + mean_offset)
    theta = np.exp(0.20 - 0.25 * x_theta + theta_offset)
    count = rng.negative_binomial(
        exposure * theta,
        theta / (mean + theta),
    ).astype(np.float64)
    return (
        pd.DataFrame({"x_mean": x_mean, "x_theta": x_theta}),
        count / exposure,
        exposure,
        {"mean": mean_offset, "theta": theta_offset},
    )


def _model() -> SuperLSS:
    return SuperLSS(
        family=NegativeBinomialLS(),
        predictors=(
            Predictor("mean", {"x_mean": Numeric()}),
            Predictor("theta", {"x_theta": Numeric()}),
        ),
    )


def test_public_fixed_fit_predicts_and_round_trips_complete_state() -> None:
    frame, response, exposure, offsets = _fixture()
    model = _model()

    assert (
        model.fit(
            frame,
            response,
            sample_weight=exposure,
            offsets=offsets,
            inner_tol=_INNER_TOLERANCE,
        )
        is model
    )

    scoring = frame.iloc[:17].reset_index(drop=True)
    log_exposure = np.log(np.linspace(0.75, 2.25, len(scoring)))
    scoring_offsets = {"mean": log_exposure, "theta": log_exposure}
    links = model.predict_link(scoring, offsets=scoring_offsets)
    parameters = model.predict_parameters(scoring, offsets=scoring_offsets)

    assert model.parameter_names_ == ("mean", "theta")
    assert tuple(parameters.columns) == ("mean", "theta")
    np.testing.assert_allclose(
        parameters.to_numpy(),
        np.exp(links.to_numpy()),
        rtol=8.0 * np.finfo(np.float64).eps,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        model.predict(scoring, offsets=scoring_offsets),
        parameters["mean"],
    )
    assert model.result_.coefficient_converged is True
    assert model.training_telemetry().curvature_policy == "observed"
    assert model.result_.curvature_telemetry.actual_source == "observed"

    restored = SuperLSS.from_bytes(model.to_bytes())

    assert type(restored.family_) is NegativeBinomialLS
    assert restored.family_.to_config() == {
        "type": "NegativeBinomialLS",
        "parameterization": "nb2_mean_theta",
    }
    np.testing.assert_array_equal(restored.covariance_, model.covariance_)
    np.testing.assert_array_equal(
        restored.predict_parameters(scoring, offsets=scoring_offsets),
        parameters,
    )
    np.testing.assert_array_equal(
        restored.predict(scoring, offsets=scoring_offsets),
        model.predict(scoring, offsets=scoring_offsets),
    )
    assert restored.result_.log_likelihood == model.result_.log_likelihood
    assert restored.result_.total_effective_df == model.result_.total_effective_df


def _fit_exposure_formulations():
    frame, rate, exposure, _ = _fixture()
    count = np.rint(exposure * rate)
    np.testing.assert_array_equal(count / exposure, rate)
    log_exposure = np.log(exposure)
    unit = np.ones(len(rate))
    scenarios = {
        "rate": (rate, exposure, None),
        "both": (
            count,
            unit,
            {"mean": log_exposure, "theta": log_exposure},
        ),
        "mean_only": (count, unit, {"mean": log_exposure}),
        "theta_only": (count, unit, {"theta": log_exposure}),
    }
    models = {
        name: _model().fit(
            frame,
            response,
            sample_weight=weights,
            offsets=offsets,
            inner_tol=_INNER_TOLERANCE,
            retain_rows=False,
        )
        for name, (response, weights, offsets) in scenarios.items()
    }
    return frame, rate, exposure, log_exposure, models


def test_public_fit_requires_each_exposure_offset_and_reconstructs_rate_parameters() -> None:
    frame, _, exposure, log_exposure, models = _fit_exposure_formulations()
    assert all(model.result_.coefficient_converged for model in models.values())

    rate_coefficients = np.fromiter(models["rate"].coef_.values(), dtype=np.float64)
    both_coefficients = np.fromiter(models["both"].coef_.values(), dtype=np.float64)
    coefficient_scale = max(
        1.0,
        float(np.linalg.norm(rate_coefficients, ord=np.inf)),
        float(np.linalg.norm(both_coefficients, ord=np.inf)),
    )
    coefficient_bound = 64.0 * len(rate_coefficients) * _INNER_TOLERANCE * coefficient_scale
    assert np.linalg.norm(rate_coefficients - both_coefficients, ord=np.inf) <= coefficient_bound

    rate_parameters = models["rate"].predict_parameters(frame).to_numpy()
    both_parameters = (
        models["both"]
        .predict_parameters(
            frame,
            offsets={"mean": log_exposure, "theta": log_exposure},
        )
        .to_numpy()
        / exposure[:, None]
    )
    parameter_scale = max(1.0, float(np.linalg.norm(rate_parameters, ord=np.inf)))
    parameter_bound = 128.0 * len(rate_coefficients) * _INNER_TOLERANCE * parameter_scale
    np.testing.assert_allclose(
        both_parameters,
        rate_parameters,
        rtol=0.0,
        atol=parameter_bound,
    )

    mean_only = (
        models["mean_only"]
        .predict_parameters(
            frame,
            offsets={"mean": log_exposure},
        )
        .to_numpy()
        / exposure[:, None]
    )
    theta_only = (
        models["theta_only"]
        .predict_parameters(
            frame,
            offsets={"theta": log_exposure},
        )
        .to_numpy()
        / exposure[:, None]
    )
    theta_gap = np.linalg.norm(mean_only[:, 1] - rate_parameters[:, 1], ord=np.inf)
    mean_gap = np.linalg.norm(theta_only[:, 0] - rate_parameters[:, 0], ord=np.inf)
    assert theta_gap > 100.0 * parameter_bound
    assert mean_gap > 100.0 * parameter_bound
